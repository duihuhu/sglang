# v2版本 — 架构对比基准测试 + Tier1/Tier2 调试 + Bug修复

## 概述

本版本在 v1 基础上新增了**四种 serving 架构的对比测试框架**，完成了 **Tier1/Tier2 四种组合的能耗-性能测试**，并修复了 AFD FFN 端 detokenizer 崩溃等关键 bug。

---

## 一、架构对比测试框架（新功能）

### 1.1 四种架构

| 架构 | 缩写 | 说明 | GPU 分配 |
|------|------|------|----------|
| 原生 sglang | Native | 单服务器 tp=4，无任何分离 | GPU 0,1,2,7 |
| PD 分离 | PD | Prefill tp=2(GPU2,7) + Decode tp=2(GPU0,1) + Router，Mooncake 传输 | GPU 0,1,2,7 |
| AF 分离 | AF | Attn tp=2(GPU1,7) + FFN tp=2(GPU0,2)，UCX 逐层通信 | GPU 0,1,2,7 |
| PD+AF 分离 | PD+AF | DF(GPU0) + DA(GPU1) + PF(GPU2) + PA(GPU7) + Router | GPU 0,1,2,7 |

### 1.2 新增文件

| 文件 | 用途 |
|---|---|
| `benchmark/test_motivation/AzurePublicDataset/run_4arch.py` | 四种架构一键测试脚本：启动 → 压测 → 采集 → 对比 |
| `benchmark/test_motivation/AzurePublicDataset/benchmark_replay.py` | 基于 AzureLLMInferenceTrace 的 trace 回放压测工具（支持能量监控） |

测试输出：
- `af_launch_logs/results_{native,pd,af,pdaf}.json` — 各架构原始结果
- `af_launch_logs/arch_comparison.json` — 聚合对比数据
- `af_launch_logs/arch_comparison.md` — 对比报告
- `af_launch_logs/arch_comparison_bar.png` — 柱状图
- `af_launch_logs/arch_comparison_normalized.png` — 归一化对比图

### 1.3 对比测试结果

**条件：CUDA Graph 全部关闭，全部使用 4 张 GPU（公平对比）**

| Metric | Native | PD | AF | PD+AF |
|--------|:------:|:----:|:----:|:-------:|
| Wall Duration (s) | **16.3** | 18.8 | 163.5 | 104.9 |
| Mean TTFT (ms) | 1086.1 | **466.8** | 12570.4 | 754.7 |
| Mean TPOT (ms) | **6.0** | 126.9 | 71.4 | 251.5 |
| Output Throughput (tok/s) | **643.1** | 557.0 | 64.1 | 100.0 |
| Total Energy (J) | **8647** | 9222 | 57183 | 32867 |

**分析：**

1. **原生 tp=4 仍是吞吐之王** — CUDA Graph 关闭后 TPOT 仅 6ms，吞吐 643 tok/s，能耗最低
2. **PD 分离 TTFT 最优** — 466.8ms vs 原生 1086ms，但 TPOT 退化到 126.9ms（原生 6ms）—— CUDA Graph 不可用对 decode 影响严重
3. **AF 分离大幅倒退** — tp=2 未带来收益，TTFT 超 12s，UCX 逐层通信 + CUDA Graph 不可用的双重瓶颈
4. **PD+AF 最差 TPOT** — 两级通信叠加（Mooncake KV transfer + UCX tensor transfer）

---

## 二、Tier1/Tier2 测试（新功能）

### 2.1 四种场景对比

通过 `run_af_scenarios.py`（已废弃，合并入 run_4arch.py）测试了 Tier1+Tier2、Tier1-only、Tier2-only、Neither 四种组合：

| Metric | Tier1+Tier2 | Tier1-only | Tier2-only | Neither |
|--------|:-----------:|:-----------:|:-----------:|:-------:|
| Wall Duration (s) | 251.5 | **104.1** | 254.1 | 105.5 |
| Mean TPOT (ms) | 489.0 | **239.1** | 488.9 | 244.9 |
| Total Energy (J) | 67749.4 | **40428.7** | 67728.8 | 33228.8 |

### 2.2 Tier2 根因分析

**结论：Tier2 使性能倒退约 2x，能耗增加约 2x。**

**根因：** DVFS 频率预测模型未计入 UCX/RDMA 通信开销，实际 decode 延迟比预测高约 2.3x。即使将 TPOT SLO 从 800ms 降到 300ms，因地步频率与通信损耗的叠加效应，模型仍然选择过低频率。

### 2.3 配置更新

`af_launch_config.json` 的 Tier1 参数从默认/占位值改为实际 workload 驱动的分析值：

| 参数 | v1 值 | v2 值（workload-driven） |
|------|-------|------------------------|
| `lambda_prefill` | 10.0 | 25.2 |
| `n_active_decode` | 32 | 58 |
| `il_rep_p` | 1024 | 1609 |
| `il_rep_d` | 512 | 1 |
| `ol_rep_d` | 256 | 116 |
| `bs_avg_d` | 16 | 58 |
| `tpot_slo_us` | 50000 | 300000 |

---

## 三、Bug 修复

### 3.1 AF分离 — FFN 端 Detokenizer 崩溃（关键修复）

**根因：** AFD FFN 视角产生 dummy logits（`torch.zeros`），但仍通过 `stream_output_generation()` 将垃圾 token ID 发送到 detokenizer。detokenizer 因找不到对应 request ID 抛出 `KeyError`，导致 FFN 进程崩溃 → UCX 连接断开 → Attn 端连锁崩溃。

**修复：** `scheduler_output_processor_mixin.py:stream_output_generation()` 增加 FFN 视角提前返回：
```python
from sglang.srt.layers.afd import afd_is_ffn
if afd_is_ffn():
    return
```

### 3.2 原生/PD/AF 服务器缺少 `--enable-metrics`

TTFT 数据因 `enable_metrics` 默认为 False 而缺失。已在所有架构的启动命令中补充 `--enable-metrics`。

### 3.3 PD+AF decode 模式 TTFT 传播中断

**根因：** PD+AF 中 prefill（PA）和 decode（DA）运行在不同进程，PA 计算的 processing TTFT 无法通过 `scheduler_time_stats` 传给 DA。

**修复：**
- `disaggregation/utils.py:MetadataBuffers` 新增 `prefill_ttft_processing` 张量字段，随 KV metadata 一同传输
- `disaggregation/decode.py` 中 `_commit_transfer_to_req` 将 `output_prefill_ttft` 写入 `req.time_stats.cached_ttft_processing`
- `req_time_stats.py` 中 `SchedulerReqTimeStats.__getstate__` 序列化 `cached_ttft_processing`，`APIServerReqTimeStats.meta_info` 反序列化并填入 `time_to_first_token_processing`

### 3.4 Tier1 WorkloadCollector 监控改进

- `build_window()`: deduplicate `_decode_steps`（防止 DA 文件未更新时重复采样膨胀窗口）
- `_record_request()`: AFD prefill-only 路径（PA/PF）的请求在 prefill 完成后标记为 finished
- `_poll_decode_stats()`: timestamp 去重
- SLO 日志增加 TTFT_SLO/TPOT_SLO 打印

### 3.5 af_launcher.py metrics & SLO 传播

- 新增 `enable_metrics` 配置项 → 传递 `--enable-metrics --enable-metrics-for-all-schedulers`
- DVFS/非 DVFS 模式下统一加载 energy model dir
- 将 TTFT/TPOT SLO 通过 `--afd-ttft-slo-ms / --afd-tpot-slo-us` 转发到各模块

---

## 四、下一步优化方向

### 4.1 能耗模型的跨模型迁移问题

- **现状：** Energy model（GBDT latency/energy predictors）是在 Qwen 2.5 7B 上采集的 profile 数据（`decode_data_v1.txt`、`prefill_data_v1.txt`）训练的，但实际推理使用 Llama 3.1 8B
- **潜在偏差：**
  1. 不同模型的算子计算量分布不同（Qwen 采用 GQA，Llama 采用 MHA；MLP 中间维度不同）
  2. 同频同 batch 下实际延迟可能偏差 10-30%，导致 DVFS 频率选择偏离最优
  3. energy model 预测的能耗与实测能耗的系统偏差未被校准
- **优化方向：**
  - 在 Llama 3.1 8B 上重新采集 profile 数据，对比 Qwen vs Llama 的 latency/energy 曲线差异
  - 引入跨模型迁移学习或在线校准机制（runtime profiling + 模型参数校正）
  - 评估是否可将 profiling 自动化作为 server 启动的前置步骤

### 4.2 Tier1/Tier2 运行时开销分析

- **Tier1 开销：**
  - ILP 求解耗时未测量（N=4 GPU × F=20 频率档 × K=3 TP 配置，求解复杂度 O(N·F·K)）
  - ILP 求解期间 workload 变化可能导致解已过时
  - 预求解的 TP 切换未落地（`_tier1_solution` 存储了最优配置但未实际切换 GPU 频率和 TP 度）
- **Tier2 开销：**
  - DVFS 频率切换延迟未量化（`nvidia-smi -rgc` 的耗时，预计 10-50ms 级）
  - WorkloadCollector 的 `_poll_decode_stats()` 周期轮询引入额外 scheduler 延迟
  - SLO 违规检查的计算开销
- **优化方向：**
  - 测量 Tier1 ILP 求解时间及 DVFS 切换延迟，评估 Tier1/Tier2 本身的能耗开销
  - 实现 TP 热切换（需要模型重分布，当前框架不支持）
  - 降低 WorkloadCollector 的采样频率或改为 event-driven 触发
  - 评估 Tier2 的收益是否足以覆盖其运行时开销（当前测试显示 Tier2 使能耗增加 ~2x）

### 4.3 AF 分离架构设计合理性评估

- **根本性问题：UCX 逐层通信是否合理？**
  - 当前设计：每层 Transformer 的 FFN 部分通过 UCX 将 activation 从 Attn GPU 发送到 FFN GPU，结果再传回。32 层 × 2 次传输 = 64 次 GPU→CPU→GPU 拷贝
  - 替代方案 1 — **算子融合 + 单次通信**：多个连续 FFN 层的结果合并传输（batch 通信而非逐层通信）
  - 替代方案 2 — **GPU direct RDMA**：利用 NVLink + peer-to-peer 直接访问对方 GPU 显存，绕过 CPU 内存拷贝（示例：`Attn GPU` → `cudaMemcpyPeer` → `FFN GPU`），延迟预计降低 5-10x
  - 替代方案 3 — **放弃 AF 分离，专注 PD 分离**：实验数据显示 PD 分离的收益明确（TTFT -57%），而 AF 分离在任何配置下都比原生差（3.9x-10x），应评估 AF 分离是否有实际价值
- **Microbatch 流水线效率：**
  - M=3 是固定值还是可配置最优值？M=1/2/4/6 的对比测试未做
  - 增大 microbatch 可隐藏通信延迟，但增加显存压力
- **Heterogeneous TP 未探索：**
  - Attn 计算密集（softmax + 矩阵乘），FFN 带宽密集（MLP 扩展/压缩）
  - 理论上 attn_tp > ffn_tp 或 attn_tp < ffn_tp 可能找到更优的资源配置
  - `afd_attn_tp` / `afd_ffn_tp` 参数存在但从未系统测试

### 4.4 CUDA Graph 不可用的架构限制

- **现状：** AFD 由于 UCX 通信调用（`UcxTensorCommunicator`）包含 Python/C 回调，无法被 CUDA Graph capture。每次 decode step 走完整 kernel launch 路径。
- **影响量化：** decoder-only serving 中 CUDA Graph 通常减少 30-70% 的 kernel launch overhead。这是 AF/PD+AF 性能远低于原生的最主要单一因素。
- **探索方向：**
  - **Partial CUDA Graph**：将 attention 内部计算（不含 UCX 通信部分）capture 为 graph，通信部分走 eager
  - **CUDA Graph with external callbacks**：CUDA 12.x 支持 `cudaGraphAddMemcpyNode` 等通信节点，评估是否可用于 UCX
  - **如果无法解决**：AF 分离应考虑上述 4.3 中的替代方案 2 或 3

### 4.5 PD 分离的 TPOT 退化

- **现状：** PD 分离的 TTFT 优秀（466ms vs 原生 1086ms），但 TPOT（127ms vs 原生 6ms）和吞吐（557 tok/s vs 643 tok/s）不如原生
- **根因：**
  1. Decode server tp=2 vs 原生 tp=4：计算能力减半
  2. Mooncake KV transfer 增加 decode 延迟
  3. CUDA Graph 关闭对纯 decode 影响最大（decode kernel 短小密集，launch overhead 占比高）
- **优化方向：**
  - 评估 decode server tp=4（占用全部 4 GPU），prefill server 复用 decode 空闲 GPU
  - Mooncake KV transfer 与 decode compute 重叠（pipeline）
  - 考虑 PD 分离仅在 TTFT SLO 严格且 TPOT SLO 宽松的场景下启用

### 4.6 缺少系统化自动化测试

- 四种架构的对比测试目前依赖手动执行 `run_4arch.py`，无自动化 CI 集成
- Tier1/Tier2 的测试需要实际 workload 触发，无单元测试覆盖能耗模型的正确性
- AFD 的 UCX 通信路径缺少网络故障注入测试
- **优化方向：**
  - 将 `run_4arch.py` 集成到 CI pipeline（nightly 或 PR trigger）
  - 为 energy model 添加离线单元测试（已知输入验证预测输出）
  - 建模仿真环境：不依赖物理 GPU 即可测试 Tier1/Tier2 逻辑正确性

---

## 五、文件变更统计

```
 8 files changed, 114 insertions(+), 19 deletions(-)

 修改:
   python/sglang/srt/disaggregation/decode.py                   |   5 +
   python/sglang/srt/disaggregation/utils.py                    |  18 +
   python/sglang/srt/energy/af_launch_config.json               |  19 +-
   python/sglang/srt/energy/af_launcher.py                      |  10 +
   python/sglang/srt/energy/workload_collector.py               |  31 +-
   python/sglang/srt/managers/scheduler.py                      |   5 +-
   python/sglang/srt/managers/scheduler_output_processor_mixin.py |   6 +
   python/sglang/srt/observability/req_time_stats.py            |  39 +

 新增（未跟踪）:
   benchmark/test_motivation/AzurePublicDataset/run_4arch.py    | 对比测试框架
   benchmark/test_motivation/AzurePublicDataset/benchmark_replay.py | trace 回放压测
   af_launch_logs/arch_comparison.*                             | 对比结果和图表
   af_launch_logs/results_*.json                                | 各架构原始数据
   af_launch_logs/scenario_comparison.*                          | Tier 场景对比
```
