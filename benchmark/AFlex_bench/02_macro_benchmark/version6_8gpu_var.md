# Version 6: TTFT x TPOT SLO 全组合 Sweep

## 实验背景

前序实验（Version 5）只扫描了 TPOT SLO 一个维度（TTFT 固定为 5000ms）。本次实验将 TTFT SLO 和 TPOT SLO 同时作为变量，遍历所有组合，形成完整的二维性能矩阵，以理解两个维度对 DVFS 节能和 SLO 合规的联合影响。

此外，修复了一个关键 bug：Prefill DVFS 在 FFN side 取到的 `extend_input_len` 被 AF 通信同步逻辑强制覆盖为 1，导致 Prefill 预测模型始终以为 input_len=1，预测延迟仅 2-9ms（实际应为 200-1300ms），频率选择完全失效。修复后 Prefill DVFS 能正确感知真实序列长度并动态调频。

## 实验配置

- **GPU**: 8 张（两组 4-GPU 并行：Worker A=GPU 0-3, Worker B=GPU 4-7）
- **模型**: Qwen3-32B, TP=1, AF 分离
- **部署方案**: PDAF DynM + Tier1/DVFS (V2 Pipeline Coupled Model)
- **TTFT SLO**: 5000 / 2000 / 1000 / 500 / 300 / 200 ms（6 个）
- **TPOT SLO**: 300 / 250 / 200 / 150 / 100 / 90 / 80 / 70 ms（8 个）
- **总组合**: 6 × 8 = 48 组
- **对照**: Baseline（满频 1410 MHz，总能耗 99665J，Prefill 44104J，Decode 55561J）
- **Workload**: workload_steady（变长序列，steady QPS=5）

## 核心结果

### 节能性能（Energy Saving vs Baseline）

|  | 300ms | 250ms | 200ms | 150ms | 100ms | 90ms | 80ms | 70ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5000ms** | 32.5% | 33.6% | 30.4% | 30.6% | 31.4% | 31.2% | 28.0% | 20.1% |
| **2000ms** | 30.9% | 31.2% | 30.8% | 26.4% | 31.3% | 28.0% | 28.3% | 19.2% |
| **1000ms** | 29.1% | 28.9% | 29.0% | 28.9% | 28.4% | 25.8% | 25.2% | 17.4% |
| **500ms** | 25.8% | 25.3% | 25.7% | 25.7% | 26.1% | 22.5% | 21.3% | 20.7% |
| **300ms** | 25.0% | 24.8% | 24.9% | 24.5% | 25.4% | 21.4% | 20.9% | 12.8% |
| **200ms** | 23.4% | 22.8% | 22.9% | 23.1% | 23.9% | 24.5% | 21.3% | 10.7% |

（行=TTFT SLO，列=TPOT SLO）

### Per-Token TPOT SLO Violation (%)

|  | 300ms | 250ms | 200ms | 150ms | 100ms | 90ms | 80ms | 70ms |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **5000ms** | 0.1 | 0.1 | 0.2 | 0.1 | 51.7 | 50.1 | 93.7 | 96.4 |
| **2000ms** | 0.1 | 0.1 | 0.1 | 0.0 | 44.9 | 49.0 | 93.7 | 96.3 |
| **1000ms** | 0.1 | 0.2 | 0.2 | 0.2 | 49.2 | 57.6 | 93.5 | 97.2 |
| **500ms** | 0.2 | 0.2 | 0.2 | 0.3 | 47.9 | 49.9 | 93.8 | 96.0 |
| **300ms** | 0.1 | 0.1 | 0.1 | 0.1 | 52.3 | 58.4 | 93.8 | 96.4 |
| **200ms** | 0.1 | 0.1 | 0.2 | 0.2 | 51.8 | 53.8 | 93.4 | 96.4 |

## 关键发现

### 1. DVFS 有效工作区间

- **安全区（SLO 合规 + 高节能）**: TPOT SLO ≥ 150ms，任意 TTFT SLO → 节能 23-34%，per-token 违背率 < 0.3%
- **过渡区**: TPOT SLO = 100-90ms → 违背率 45-58%，节能 21-31%
- **不可行区**: TPOT SLO ≤ 80ms → 违背率 > 93%（硬件物理极限）

### 2. TTFT SLO 对 Prefill 节能的影响

TTFT SLO 从 200ms 放宽到 5000ms，Prefill 能耗节省额外增加约 8-10%。这是因为宽松的 TTFT 约束允许 Prefill DVFS 选择更低频率（平均 869MHz vs 1068MHz）。但由于实际 TTFT 最低也约 700ms（受模型计算+AF通信固有延迟限制），TTFT SLO < 500ms 时 TTFT 违约不可避免。

### 3. TPOT SLO 对 Decode 节能的影响

TPOT SLO 主要影响 Decode 阶段频率。当 TPOT SLO 从 70ms 放宽到 300ms 时，Decode 平均频率从 ~1366MHz 降到 ~1004MHz，节能从 ~3% 提升到 ~30%。100ms 是临界点——实际 TPOT 底噪约 83ms，100ms SLO 仅留 17ms 余量，导致约半数 token 超标。

### 4. Prefill bug 修复效果

修复 `extend_input_len` 覆盖问题后，Prefill DVFS 的频率选择从"始终 930MHz"变为动态调节（869-1068MHz 范围），能正确响应 TTFT SLO 约束。TTFT SLO 越紧，Prefill 频率越高；越宽松，越积极降频省能。

### 5. 两维 SLO 解耦特性

实验揭示了 TTFT SLO 和 TPOT SLO 对系统行为的影响基本正交：
- TTFT SLO 主导 Prefill 频率选择（影响 Prefill 节能 ~8%）
- TPOT SLO 主导 Decode 频率选择（影响 Decode 节能 ~27%）
- 两者可独立配置，互不干扰

## 图表

- `retrain/figures/joint_sweep_slo_violations.png`: TTFT/TPOT/Overall SLO 违背率热力图
- `retrain/figures/joint_sweep_performance.png`: Prefill/Decode/Total 节能热力图
- `retrain/figures/joint_sweep_avg_freq.png`: Prefill/Decode/Overall 平均频率热力图

## 脚本

- `retrain/run_joint_slo_sweep.py`: 双 worker 并行实验调度器
- `retrain/analyze_joint_sweep.py`: 结果分析 + heatmap 生成

---

## 8-GPU 变长部署方案对比（TTFT SLO=2000ms, TPOT SLO=150ms）

### 实验配置

- **GPU**: 8 张 H800, 全部使用
- **模型**: Qwen3-32B
- **Workloads**: 5 个变长 workload（steady/varying/heavy/overload/tier1_demo）
- **SLO**: TTFT ≤ 2000ms, TPOT ≤ 150ms（含排队时间）
- **方案**:
  1. **PDAF DynM** — AF 分离 TP=2 + 动态微批, 锁最高频 1410MHz
  2. **PDAF Tier+DynM** — 同上 + DVFS V2 调频
  3. **Native TP=8** — 单实例 TP=8, 锁最高频
  4. **PD DP=4** — 4x 1P1D PD 分离 (TP=1 per GPU), 锁最高频

### 结果汇总

| 方案 | Workload | Thpt(tok/s) | TTFT(ms) | TPOT(ms) | Energy(J) | SLO(%) |
|------|----------|-------------|----------|----------|-----------|--------|
| PDAF DynM | steady | 688 | 329 | 108.9 | 147,356 | 0.0 |
| PDAF DynM | varying | 180 | 227 | 87.8 | 111,237 | 0.0 |
| PDAF DynM | heavy | 261 | 252 | 93.3 | 124,027 | 0.0 |
| PDAF DynM | overload | 677 | 28,240 | 108.9 | 176,125 | 76.8 |
| PDAF DynM | tier1_demo | 659 | 30,524 | 109.0 | 189,546 | 79.5 |
| PDAF Tier+DynM | steady | 644 | 3,346 | 122.8 | 112,790 | 63.3 |
| PDAF Tier+DynM | varying | 179 | 788 | 92.7 | 85,385 | 0.0 |
| PDAF Tier+DynM | heavy | 259 | 986 | 100.3 | 93,229 | 0.0 |
| PDAF Tier+DynM | overload | 630 | 33,782 | 116.0 | 140,531 | 76.7 |
| PDAF Tier+DynM | tier1_demo | 622 | 35,652 | 113.7 | 149,223 | 79.5 |
| Native TP=8 | steady | 752 | 115 | 56.7 | 158,541 | 0.0 |
| Native TP=8 | varying | 183 | 114 | 51.2 | 128,224 | 0.0 |
| Native TP=8 | heavy | 263 | 117 | 55.7 | 141,803 | 0.0 |
| Native TP=8 | overload | 1,006 | 6,797 | 63.2 | 149,182 | 73.6 |
| Native TP=8 | tier1_demo | 881 | 6,118 | 63.5 | 174,781 | 70.9 |
| PD DP=4 | steady | 752 | 178 | 48.1 | 220,910 | 0.0 |
| PD DP=4 | varying | 183 | 139 | 46.2 | 167,207 | 0.0 |
| PD DP=4 | heavy | 263 | 180 | 47.1 | 186,937 | 0.0 |
| PD DP=4 | overload | 1,004 | 159 | 48.5 | 196,290 | 0.0 |
| PD DP=4 | tier1_demo | 880 | 155 | 48.2 | 226,943 | 0.0 |

### 关键发现

#### 1. 延迟与吞吐

- **PD DP=4 延迟最低**：TTFT 平均仅 155-180ms，TPOT 约 47-48ms。在所有 workload 上均 0% SLO 违背（包括 overload/tier1_demo 高负载）。4 个独立 1P1D 实例提供了充足的并发容量。
- **Native TP=8 延迟次之**：TTFT 114-117ms（轻负载），但高负载下升至 6000-7000ms 导致 70-74% SLO 违背。单实例无法有效应对突发流量。
- **PDAF DynM/Tier+DynM 延迟最高**：AF 分离的 IPC 同步开销导致基础 TPOT 约 90-120ms，TTFT 在高负载下达到 28000-36000ms。

#### 2. 能耗对比

- **PDAF Tier+DynM 能耗最低**：轻中负载（varying/heavy）下仅 85-93kJ，比 DynM 节省 23-25%
- **PDAF DynM 中等能耗**：111-190kJ
- **Native TP=8 较高**：128-175kJ（8 张卡全部参与计算）
- **PD DP=4 能耗最高**：167-227kJ（8 张卡各加载完整模型 TP=1，基础功耗极高）

#### 3. SLO 违背分析

- 所有方案的 SLO 违背**全部来自 TTFT（排队等待）**，TPOT 无违背
- PD DP=4 因容量充足，即使 overload（QPS=10）也不排队 → 0% 违背
- Native TP=8 只有一个实例，高负载排队严重
- PDAF 方案因 IPC 开销导致处理慢，排队更长

#### 4. DVFS 节能效果

PDAF Tier+DynM 相比 PDAF DynM（锁最高频）：
| Workload | DynM Energy | Tier Energy | 节省 |
|----------|------------|------------|------|
| steady | 147,356J | 112,790J | 23.4% |
| varying | 111,237J | 85,385J | 23.2% |
| heavy | 124,027J | 93,229J | 24.8% |
| overload | 176,125J | 140,531J | 20.2% |
| tier1_demo | 189,546J | 149,223J | 21.3% |

平均节能约 **22.6%**，但代价是 steady workload 引入了 63.3% TTFT SLO 违背（因降频导致 prefill 变慢+排队）。

### 图表

- `results/8gpu_var/figures/8gpu_var_comparison.png`: 总对比
- `results/8gpu_var/figures/8gpu_var_energy_breakdown.png`: P/D 能耗分解
- `results/8gpu_var/figures/8gpu_var_slo_violations.png`: SLO 违背率
- `results/8gpu_var/figures/8gpu_var_dvfs_savings.png`: DVFS 节能百分比

---

## 待测试计划（换设备后）

### 问题与改进方向

当前测试存在以下问题需要在新一轮测试中解决：

1. **TTFT SLO 统计包含排队时间**：当前的 SLO 判定使用的是"从请求到达到首 token 返回"的完整 TTFT，包含了 scheduler 队列等待时间。这导致高负载下几乎所有方案都违背 SLO，无法区分"实际 prefill 处理太慢"和"队列排满了"。
   - **已修复**：新增 `ttft_pure_processing` 字段（= `prefill_finished_time - prefill_run_batch_start_time`），仅统计真正的 prefill 计算时间
   - **SLO 判定改为使用纯 processing TTFT**
   - **DVFS 的 slack 计算也改为不含排队时间**

2. **DVFS Prefill 频率选择 slack 包含排队**：之前的 `_compute_prefill_slack` 从 `api_server_dispatch_time` 开始计时（含排队），导致高负载下 slack 为负，DVFS 永远选最高频。
   - **已修复**：改为从 `prefill_run_batch_start_time` 开始计时

### 新一轮 4-GPU 测试计划

**设备**: 4 张 GPU（后四张卡 4-7）

**SLO**: TTFT ≤ 2000ms (纯 processing)，TPOT ≤ 150ms

**方案**（4 种）:
1. **PDAF DynM** — AF 分离 + 动态微批，锁最高频
2. **PDAF Tier+DynM** — AF 分离 + 动态微批 + DVFS V2 调频
3. **PD DP2** — 2x 1P1D PD 分离 (TP=1)
4. **Native DP4** — 4x TP=1 原生完整实例

**Workloads**: 5 个变长 workload（steady/varying/heavy/overload/tier1_demo）

**目标**:
- 对比排除排队时间后的真实 SLO 违背率
- 验证 DVFS 在不含排队干扰下的调频决策是否更合理
- 对比 Native DP4（4x TP1 原生多实例）与 PDAF 方案的能效差异
- 评估 PD 分离 DP2 vs 原生 DP4 的性能/能效权衡

**脚本**: `scripts/bench/run_4gpu_var_v2.sh`

**代码改动**:
- `python/sglang/srt/observability/req_time_stats.py`: 新增 `ttft_pure_processing` 字段输出
- `python/sglang/srt/managers/tokenizer_manager.py`: 无论是否启用 metrics 都输出 `ttft_pure_processing`
- `python/sglang/srt/managers/scheduler.py`: `_compute_prefill_slack` 改用 `prefill_run_batch_start_time`
- `python/sglang/srt/energy/workload_collector.py`: 内部 TTFT 统计改用纯 processing 时间
- `python/sglang/srt/disaggregation/utils.py`: KV transfer 传播的 TTFT 改为纯 processing 时间
- `benchmark/energy_bench/scripts/bench/run_fixed_qps_bench.py`: SLO 判定优先使用 `ttft_pure_processing`
