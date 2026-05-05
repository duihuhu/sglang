# v1版本 — Tier 1 动态监控 + Bug修复

## 概述

本版本在 v0 基础上引入了 **Tier 1 联合 ILP 资源规划 + 动态监控系统**，并修复了多个 AFD 和 PD+AF 模式下的运行时 bug。

---

## 一、Tier 1 动态监控系统（新功能）

### 1.1 架构

Tier 1 是一个运行在 PA（Prefill-Attention）调度器上的闭环控制系统：

```
WorkloadMonitor ← WorkloadMetricsCollector ← PA Scheduler
       │                      ↑
       ▼                      │
  Tier1Solver ──→ 新配置      │
                        PA/DA/PF/DF (重新规划)
```

### 1.2 新增模块（`python/sglang/srt/energy/`）

| 文件 | 用途 |
|---|---|
| `af_launcher.py` | 多模块批启动器，按 DF→DA→PF→PA→router 顺序启动，支持 `--start-with-workload` 预求解 |
| `af_launch_config.json` | 完整启动配置（4 模块 + 路由 + DVFS + Tier1） |
| `af_launch_config_minimal.json` | 最小启动配置（无 DVFS，无 Tier1） |
| `profile_table.py` | ProfileTable：加载 prefill/decode 测试数据 + 能量模型 |
| `tier1_solver.py` | Tier1Solver：ILP 整数线性规划求解（含 SLO 约束） |
| `workload_collector.py` | WorkloadMetricsCollector：逐请求收集 TTFT/TPOT + GPU 利用率 |
| `workload_monitor.py` | WorkloadMonitor：滑动窗口检测 SLO 违规 + A/F/负载分布漂移 |

### 1.3 新增 CLI 参数（14 个）

全部以 `--tier1-*` 前缀，见 `server_args.py:5490-5550`：

- `--enable-tier1-pa`：启用 PA 上的动态监控
- `--tier1-initial-solution`：预求解 JSON 路径（由 af_launcher.py `--start-with-workload` 写入）
- `--tier1-gpu-count`：GPU 总预算（默认 16）
- `--tier1-lambda-prefill`：prefill 请求到达率（默认 10.0 req/s）
- `--tier1-n-active-decode`：稳态 decode 请求数（默认 32）
- `--tier1-il-rep-p/d`：代表性的 prefill/decode 输入长度
- `--tier1-bs-avg-p/d`：平均 prefill/decode batch 大小
- `--tier1-monitor-window-s`：监控窗口大小（默认 30s）
- `--tier1-prefill/decode-data-path`：profile 数据路径
- `--tier1-stats-path`：DA→PA 共享的 decode 计时文件

### 1.4 启动流程

```
af_launcher.py [--config config.json] [--start-with-workload]

  1. [可选] 预求解: ProfileTable + Tier1Solver.solve() → tier1_initial_solution.json
  2. 分配 GPU (支持异构 TP)
  3. 锁定 GPU 频率 (pynvml SetGpuLockedClocks)
  4. 依次启动: DF → DA → PF → PA → router
      - 每个模块通过环境变量传递: CUDA_VISIBLE_DEVICES, AFD_UCX_*, AFD_NVML_DEVICE_INDEX
      - PA 额外接收: --enable-tier1-pa, GPU→pool 映射 (AFD_ATTN/FFN_GPU_INDICES)
  5. 等待各模块端口就绪，启动 router
```

启动日志输出到 `af_launch_logs/{PA,PF,DA,DF,router}.log`。

### 1.5 动态重规划流程

```
PA event_loop: 每 batch 结束后
  → _tier1_record_batch() → WorkloadMetricsCollector 记录
  → _tier1_monitor_check() 每 window_s 秒:
     → 构建 MonitoringWindow (TTFT/TPOT/util/distribution)
     → WorkloadMonitor.should_replan() 检测是否需重规划
     → 是: _replan_tier1()
        → _ensure_tier1_solver() (延迟初始化)
        → Tier1Solver.solve() → 新 tp/f/k 配置
        → 更新 _tier1_solution
        → [TODO] drain-then-switch 迁移
```

---

## 二、PD+AF 解码侧支持增强

### 2.1 FFN 端请求生命周期管理

**`scheduler.py`:**

- `_afd_ensure_reqs_from_afdreq()`：FFN decode 端，当 DA 发送的 AFDReqInput 中没有对应 Req 对象时，从 `input_ids_per_req` / `max_new_tokens_per_req` 创建最小 Req（用于 PD+AF decode 模式）
- `_afd_ffn_cleanup_stale()`：当 Attn 端不再跟踪某个 req_id 时，清理 FFN 端对应的 KV cache 并标记完成
- `_afd_ffn_cleanup_all()`：空闲时强制清理所有残留请求，防止 `self_check_during_idle` 误报内存泄漏

**`io_struct.py`:** AFDReqInput 新增 `input_ids_per_req`、`max_new_tokens_per_req` 字段。

**`scheduler_afd_mixin.py`:** `afd_send_batch_info` 携带 `input_ids_per_req` 和 `max_new_tokens_per_req`。

### 2.2 KV cache 分配修复（DF 端）

**`decode_schedule_batch_mixin.py`:** FFN 路径设置 `kv_committed_len` 和 `kv_allocated_len` 为完整 `fill_ids` 长度，防止 `alloc_for_extend` 分配的 token 永久泄漏。

### 2.3 输出 token 守卫（PD+AF decode）

- `decode_schedule_batch_mixin.py`: 当请求刚迁移还没有 output token 时，fallback 到 `origin_input_ids[-1]`。
- `schedule_batch.py merge_batch`: 当 `self.output_ids` 为 None 时从 reqs 重建。

### 2.4 KV 传输跳过

- `prefill.py`: FFN 侧跳过 `send_kv_chunk()` + 增加 `disagg_kv_sender is None` 守卫。

---

## 三、Bug 修复

### 3.1 微批次分割空张量崩溃（afd_overlap.py）

**根因：** 当 `num_seqs == m` 时，分割点可能等于 `num_seqs`，产生空子 batch。`AfdForwardBatchPreparer.prepare` 仍用 `m` 构建 boundaries 导致越界。

**修复：** 使用 `actual_m = len(seq_indices) + 1` 替代 `m`；增加空值防御（`skip empty children`）；同步 `scheduler_afd_mixin.py` 和 `scheduler.py` 中 `batch_size < m` 时设置 `afd_split_seq_index = None`。

### 3.2 DF 端 KV cache 内存泄漏

**根因：** `kv_committed_len` / `kv_allocated_len` 未初始化导致 `release_kv_cache` 只释放部分 token。

**修复：** `decode_schedule_batch_mixin.py:54-55` 设置两个值为 `len(req.fill_ids)`。

### 3.3 output_ids 为空导致的 CUDA 崩溃

**根因：** PD+AF decode 中请求刚迁移时无 output_ids，`prepare_for_decode` → `self.input_ids = self.output_ids` 为 None → forward pass 崩溃。

**修复：** 多处添加 output_ids 空值守卫 + 从 reqs 重建逻辑。

### 3.4 AFD pipeline 空 hidden_states 检查

**`afd.py`:** 增加 `hidden_states.numel() == 0` 检测 + `torch.cuda.synchronize()` 同步捕获上游 kernel 错误。

### 3.5 位置张量越界（forward_batch_info.py）

- `extend_num_tokens` 替换为 `num_tokens`
- `positions` 截断匹配 `input_ids` 形状
- `compute_position_torch` 接收 `extend_seq_lens_sum` 参数并截断

### 3.6 AFD 通信时序修复

- `tbo_backend.py`: 移除 `zip(strict=True)` 允许 children 数动态变化
- `afd.py`: 增加 pipeline 阶段名称日志

---

## 四、优化

### 4.1 FFN 侧跳过 embedding 和 logits（llama.py）

- `embed_tokens`：FFN 侧用 `torch.zeros` 替代，因为 UCX `recv_wait` 会覆盖 hidden_states
- `lm_head`：FFN 侧返回 dummy logits，跳过 `logits_processor`

### 4.2 TTFT 计时传播（req_time_stats.py）

`APIServerReqTimeStats` / `DPControllerReqTimeStats` 的 `__getstate__` 序列化 `api_server_dispatch_time`，供 scheduler 进程上的 collector 计算 TTFT。

### 4.3 DVFS 初始化改进

`scheduler.py _init_afd_dvfs` 使用物理 NVML GPU index（`AFD_NVML_DEVICE_INDEX` 环境变量），而非 CUDA index。

### 4.4 构建文档更新（BUILD.md）

添加 maturin、rustup、sgl-model-gateway、AFD-UCX、Mooncake libibverbs 安装说明。

---

## 文件变更统计

```
 15 files changed, 804 insertions(+), 51 deletions(-)

 修改:
   BUILD.md                                          |  14 +
   benchmark/test_motivation/energy_model.py          |   4 +-
   python/sglang/srt/batch_overlap/afd_overlap.py     |  43 +-
   .../disaggregation/decode_schedule_batch_mixin.py  |  14 +-
   python/sglang/srt/disaggregation/prefill.py        |  20 +-
   python/sglang/srt/layers/afd.py                    |  22 +
   python/sglang/srt/layers/attention/tbo_backend.py  |   2 +-
   python/sglang/srt/managers/io_struct.py            |   4 +
   python/sglang/srt/managers/schedule_batch.py       |  13 +-
   python/sglang/srt/managers/scheduler.py            | 525 +++++++++++
   python/sglang/srt/managers/scheduler_afd_mixin.py  |   4 +
   .../srt/model_executor/forward_batch_info.py       |  14 +-
   python/sglang/srt/models/llama.py                  |  18 +-
   python/sglang/srt/observability/req_time_stats.py  |  25 +-
   python/sglang/srt/server_args.py                   | 133 +++++

 新增:
   python/sglang/srt/energy/af_launcher.py            | 496 ++++++++++
   python/sglang/srt/energy/af_launch_config.json     |  89 ++
   python/sglang/srt/energy/af_launch_config_minimal.json | 40 +
   python/sglang/srt/energy/profile_table.py          | 独立模块
   python/sglang/srt/energy/tier1_solver.py           | 独立模块
   python/sglang/srt/energy/workload_collector.py     | 独立模块
   python/sglang/srt/energy/workload_monitor.py       | 独立模块
```
