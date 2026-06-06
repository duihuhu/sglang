# v3 版本 — Async Recv 流水线优化 + TPOT 分解计时 + 关键 Bug 修复

## 概述

本版本的核心目标是**消除 PD+AF decode 流水线中的 CPU 阻塞等待**，最大化 GPU 利用率以提升输出总吞吐。通过将 UCX recv 异步化（后台线程）并在 Attn 端激进预发起，使 DA 的 A-stage 计算与 DF→DA 的网络传输完全重叠，从根本上减少 212ms 的 F-stage 等待气泡。同时新增了 CUDA Event 级别的 TPOT 分解计时系统和 per-step wall-clock 时间线，用于精确定位流水线瓶颈。

---

## 一、Async Recv 流水线优化（核心特性）

### 1.1 问题背景

在 v2 的 PD+AF M=3 流水线中，实测数据显示：

| 节点 | A-stage | F-stage | F-stage 中等待占比 |
|------|---------|---------|-------------------|
| DA (Attn) | 156.6ms (attn=113.8 + send=30.4 + recv=12.5) | 216.0ms (wait DF=212.4 + proxy MLP=3.6) | **98.3%** |
| DF (FFN) | 225.6ms (wait DA=219.4 + proxy attn=6.2) | 150.1ms (FFN=129.9 + send=20.2) | **97.2%** |

**根因：** DA 的 `recv_start()` 在 A-stage 完成后阻塞 CPU 等待 DF 的 UCX 回传。由于 DA 使用 layer-major 调度（先跑完 3 个 micro-batch 的 A-stage 再进入 F-stage），CPU 被阻塞时无法继续发射下一层的计算任务，导致 GPU 空转。

### 1.2 架构改造

#### AsyncTensorCommunicator — 3BO Ring Buffer

```
改造前 (v2):                          改造后 (v3):
┌─────────────┐                       ┌──────────────────────┐
│ recv_start()│ blocks CPU             │ recv_start()         │ returns immediately
│   └─ UCX recv (block)               │   └─ thread.start()  │ → background
│      ...wait...                     │      while thread:   │
│ recv_wait() │ gets tensor           │        UCX recv(block)│
└─────────────┘                       │        tensor → ring  │
                                      │ recv_wait()          │
                                      │   └─ thread.join()   │
                                      │   └─ ev.sync()       │
                                      └──────────────────────┘
```

**关键改动：**
- 新增 `_RING_SIZE = 3` 环形缓冲区，支持最多 3 个并发 recv
- `recv_start()`: 将 `inner.recv_tensor()`（UCX 阻塞调用）移到后台 daemon 线程执行，立即返回
- `recv_wait()`: 先 `thread.join()` 等待后台线程完成，再取 tensor
- `drain_recvs()`, `drain_sends()`: 错误恢复时的线程清理
- `_pending_recv` 属性: 后台线程未完成时返回 thread 作为 truthy sentinel，保证向后兼容

#### 激进预发起策略

```python
# v2: 仅在 A→F 和 F→F 转换时预发起（每层仅 1 次）
should_preissue = (stage == A and next == F) or (stage == F and next == F)

# v3: 每次 A-stage 后都预发起（每层 M 次）
should_preissue = (stage == A) or (stage == F and next == F)
```

**效果：** DA 在执行 A(0,0), A(0,1), A(0,2) 时，3 个后台线程同时等待 DF 的 F-stage 回传。当 DA 到达 F(0,0) 时，数据可能已就绪，消除了 212ms 的 recv 阻塞。

### 1.3 涉及的代码文件

| 文件 | 改动 |
|------|------|
| `python/sglang/srt/layers/afd.py` | `AsyncTensorCommunicator` 完整重构：3BO ring buffer、后台线程 recv、激进预发起 |

---

## 二、TPOT 分解计时系统

### 2.1 CUDA Event 级别计时

在 `AFDDecoderLayerMixin.forward_afd_A` 和 `forward_afd_F` 中插入 `torch.cuda.Event(enable_timing=True)` 对：

```
forward_afd_A:  ──[prep_attn]──[attn]──[prep_mlp]──
forward_afd_F:  ──[mlp]──[postprocess]──
```

每次 forward pass 结束时，`model_forward_afd()` 调用 `torch.cuda.synchronize()` 后聚合所有 event 的 `elapsed_time()`，以 `[AFD_BREAKDOWN]` 日志输出：

```
[AFD_BREAKDOWN] perspective=attn layers=64 M=3 total=372.6ms
  | A_stage=156.6ms (prep_attn=12.4ms attn=113.8ms prep_mlp=30.4ms)
  | F_stage=216.0ms (mlp=3.6ms postprocess=212.4ms)
  | nA=192 nF=192
```

### 2.2 Per-Step Wall-Clock 时间线

环境变量 `AFD_DETAILED_TIMING=1` 开启后，每个 pipeline step 记录 `time.perf_counter()` 时间戳，以 JSON 格式输出到 `[AFD_TIMELINE]` 日志行，用于绘制 Gantt 图和 schedule 分析。

### 2.3 Per-Token TPOT 记录

`APIServerReqTimeStats` 新增：
- `decode_step_times`: 每个 decode step 的 per-token 耗时（ms）
- `decode_tpot_avg_s`: 平均 decode TPOT
- `decode_tpot_per_token_ms`: per-token 列表（响应 meta_info 中）

`SchedulerReqTimeStats` 新增：
- `model_forward_time` / `model_forward_count`: 调度器端的 forward 耗时累计

### 2.4 涉及的代码文件

| 文件 | 改动 |
|------|------|
| `python/sglang/srt/layers/afd_mixin.py` | CUDA event 记录点插入 A/F stage |
| `python/sglang/srt/layers/afd.py` | `_log_afd_breakdown()` 聚合日志；`[AFD_TIMELINE]` 输出 |
| `python/sglang/srt/model_executor/model_runner.py` | per-forward 耗时日志 `[MODEL_FWD]`；调度器端计时 |
| `python/sglang/srt/managers/scheduler.py` | `add_model_forward_time()` 累计 |
| `python/sglang/srt/managers/tokenizer_manager.py` | `add_decode_step()` 记录 per-token 间隔 |
| `python/sglang/srt/observability/req_time_stats.py` | decode step 计时字段 + TPOT 计算 |

---

## 三、Bug 修复

### 3.1 PD+AF residual batch_size 不匹配（关键）

**现象：** DA 的 F-stage 处理 residual 时报 shape mismatch，导致 decode 中断。

**根因：** PD+AF micro-batch 流水线中，`stage_outputs[A]` 和 `stage_outputs[F]` 队列交错存放不同 micro-batch 的数据。如果 schedule 顺序与 FIFO 队列的 pop 顺序不一致，某个 F-stage 会 pop 到错误 micro-batch 的 residual。

**修复：** `LayerCommunicator.prepare_attn()` 增加 safety guard：当 `residual.shape[0] != hidden_states.shape[0]` 时，re-initialize residual（降级为无 residual 传递，保证正确性优先）。

**文件：** `python/sglang/srt/layers/communicator.py`

### 3.2 FFN 端 req 去重

**现象：** DF 的 `get_next_batch_to_run` 在 `running_batch` 为空时会产生重复 req。

**根因：** `_afd_get_next_batch()` 合并 AFDReqInput 和已有 running_batch 时未检查 rid 重复。

**修复：** 在 DF 端 batch 创建后对 `batch.reqs` 按 `rid` 去重。

**文件：** `python/sglang/srt/managers/scheduler.py`

### 3.3 FFN 端 chunked_req 残留

**现象：** PF 的 chunked_req 机制在 DF 端残留，导致 DF KV pool 大小与 Attn 不一致。

**修复：** DF 端在每次 schedule 循环开始时清除 `self.chunked_req = None`。

**文件：** `python/sglang/srt/managers/scheduler.py`

### 3.4 FFN 端 extend_lens 强制同步

**现象：** UCX 逐层传输固定 `[1, hidden_size]`，但 DF 的 batch 保留了 Attn 端的原始 `extend_lens`（可能 >1），导致 embedding 和 logits 提取维度不匹配。

**修复：** DF 端在 batch 创建后强制设置 `extend_lens = [1] * bs` 和 `extend_input_len = 1`。

**文件：** `python/sglang/srt/managers/scheduler.py`

### 3.5 FFN 端跳过 sampling

**现象：** DF 产生的 logits 是占位符（dummy），但 `tp_worker.py` 仍调用 `model_runner.sample()`，产生错误的 next_token。

**修复：** DF 端跳过采样，直接返回全零 `next_token_ids`。真正的采样在 DA 端执行。

**文件：** `python/sglang/srt/managers/tp_worker.py`

### 3.6 Qwen3 _run_mlp 签名兼容

**现象：** `AFDDecoderLayerMixin.forward_afd_F()` 调用 `self._run_mlp(hidden_states, forward_batch)`，但 Qwen3 的 `Qwen2MLP.forward(self, x)` 不接受 `forward_batch`。

**修复：** `Qwen3DecoderLayer` 重写 `_run_mlp()`，丢掉 `forward_batch` 参数。

**文件：** `python/sglang/srt/models/qwen3.py`

### 3.7 原生模式 TTFT 回退

**现象：** 原生（非 PD）模式下 `prefill_finished_time` 可能为 0，导致 TTFT 缺失。

**修复：** 增加 fallback：当 `cached_ttft_processing` 也为 0 时，使用 `first_token_time - api_server_dispatch_time` 估算。

**文件：** `python/sglang/srt/observability/req_time_stats.py`

### 3.8 af_launcher.py DB 登录失败

**现象：** 启动时尝试连接 MySQL 数据库（硬编码凭据），连接失败导致启动中断。

**修复：** 移除硬编码的数据库连接代码。

**文件：** `python/sglang/srt/energy/af_launcher.py`

---

## 四、Pipeline 分析工具

| 文件 | 用途 |
|------|------|
| `benchmark/.../pipeline_analysis/test_m3_timeline.py` | 启动 PD+AF M=3 服务器 → 发送并发请求 → 收集 `[AFD_TIMELINE]` 日志 |
| `benchmark/.../pipeline_analysis/draw_multi_pipeline.py` | 解析时间线 JSON → 绘制 DA/DF 并排 Gantt 图 + 放大视图 + 指标汇总 |
| `benchmark/.../pipeline_analysis/results/parsed_timeline_m3.json` | 解析后的 384-step 时间线数据 |
| `benchmark/.../pipeline_analysis/results/breakdown_stats_m3.json` | CUDA event 聚合的 TPOT 分解数据 |

**时钟对齐修复：** DA 和 DF 运行在同一台机器的不同进程上，`time.perf_counter()` 存在跨进程 jitter（~0.33ms）。`draw_multi_pipeline.py` 使用 DA 的第一个事件作为共同 t0，并通过数据依赖约束（DA 发送后才能被 DF 接收）计算经验修正量。

---

## 五、配置变更

| 变更项 | 旧值 | 新值 | 原因 |
|--------|------|------|------|
| 模型 | llama3.1-8B | Qwen3-32B | 32B 模型更真实地体现通信开销 |
| GPU | [0,1,2,7] | [0,1,2,3] | NV8 switch 拓扑下 0,1,2,3 直连 |
| mem_fraction | 0.7 | 0.93 | 32B 模型需更大显存 |
| DF/DA UCX port | 25100 | 25200 | 与 PA/PF (25100) 错开，避免端口冲突 |
| DVFS | enabled | disabled | 基准测试阶段暂不启用 |
| Tier1 | enabled | disabled | 架构对比测试期间关闭 |
| extra_cli_args | (none) | per-module | `--skip-server-warmup`, `--max-running-requests 16` |

---

## 六、文件变更

```
python/sglang/srt/layers/afd.py                    | AsyncTensorCommunicator 3BO重构 + 激进预发起 + 分解日志
python/sglang/srt/layers/afd_mixin.py              | CUDA event 计时 + Qwen3 _run_mlp override
python/sglang/srt/layers/communicator.py           | residual batch_size safety guard
python/sglang/srt/managers/scheduler.py            | FFN batch修复(x4) + forward计时
python/sglang/srt/managers/tokenizer_manager.py    | decode_step TPOT 记录
python/sglang/srt/managers/tp_worker.py            | FFN 跳过采样
python/sglang/srt/model_executor/model_runner.py   | per-forward 耗时日志
python/sglang/srt/models/qwen3.py                  | _run_mlp 签名适配
python/sglang/srt/observability/req_time_stats.py  | TPOT 分解 + TTFT fallback
python/sglang/srt/energy/af_launch_config.json     | Qwen3-32B 配置 + per-module args
python/sglang/srt/energy/af_launch_config_minimal.json | 同步更新
python/sglang/srt/energy/af_launcher.py            | DB 移除 + extra_cli_args
benchmark/.../pipeline_analysis/test_m3_timeline.py       | 新增: pipeline 时间线采集
benchmark/.../pipeline_analysis/draw_multi_pipeline.py    | 新增: Gantt 图可视化
benchmark/.../pipeline_analysis/results/*                  | 新增: 分析数据
python/sglang/srt/energy/versions/version3.md              | 本文件
```

---

## 七、下一步规划

### 7.1 继续完善 PD+AF 流水线功能，提升输出吞吐

当前 async recv 优化消除了 CPU 阻塞，但仍有优化空间：
- **Per-step Python overhead** (~1ms × 384 steps ≈ 384ms per forward pass)：考虑用 CUDA Graph capture 或 C++ 调度器取代 Python pipeline loop
- **M 值自适应调优**：探索 M=4/5/6 时的吞吐上限（ring buffer depth 需同步扩展）
- **FFN 端对称优化**：DF 也可采用激进预发起，使 DA→DF 的传输更早开始
- **Send/Recv 延迟合并**：利用 UCX multi-rail（多 NIC）提升带宽利用率

### 7.2 测试不同参数下的性能对比

- M 值 sweep (1/2/3/4/6) × batch_size (1-16) 全矩阵测试
- 不同 TP 配置对比：tp=1 per-module vs tp=2 vs tp=4
- M 值对 TPOT 的影响：TPOT 会随 M 增大而升高（批量化通信导致），需找到吞吐/延迟的帕累托最优
- 不同 sequence length (ol=128/256/512) 下的吞吐

### 7.3 测试 Tier1/Tier2 能耗-性能对比

- **Tier1 (ILP 资源规划 + 动态监控)**：在 PA 端启用 workload-driven 的 GPU 频率调整
- **Tier2 (慢节点重调度)**：基于 latency SLO 的超时检测与请求迁移
- 对比四种组合：Tier1 ON/OFF × Tier2 ON/OFF
- 关键指标：总能耗 (J)、输出吞吐 (tok/s)、TTFT、TPOT、GPU 利用率

---

## 八、待解决问题

1. **M=3 以上未系统测试** — 当前仅验证 M=1/3，M=4-6 可能触达 UCX ring buffer 深度限制
2. **Python pipeline loop overhead** (~384ms per forward pass) — 是最大残余瓶颈，需 C++/CUDA 级别优化
3. **multi-pipeline reference pattern 未完全达成** — 当前 DA 侧仍有 layer 0 的 3 个 A-stage 全算完后才进入 F-stage 的结构性气泡
4. **TPOT 退化** — 分离架构的 TPOT 天然高于原生（逐层网络通信），需接受这一 tradeoff，优化目标聚焦于总吞吐
5. **prefill M>1 未启用** — 当前仅 decode 端使用 M=3，prefill 端仍是 M=1
