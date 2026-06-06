# Version 7: TTFT Processing Time 分析（去除排队延迟）

## 实验背景

前序实验（Version 5/6）的 TTFT 指标包含了请求排队时间（queuing delay），导致在高负载下 TTFT 被排队延迟主导，无法准确反映系统实际的 Prefill 处理能力。本版本的核心改进：**分离 TTFT 为 Processing Time（纯计算延迟）和 Queuing Time（排队延迟）**，聚焦分析 DVFS 对 Prefill 处理延迟的真实影响。

指标定义：
- `ttft_avg_ms`：端到端 TTFT = 排队时间 + 处理时间
- `ttft_proc_avg_ms`：纯处理时间 = Prefill 计算延迟（从调度器开始处理到首 token 生成）

---

## Part 1: 8-GPU 多方案对比（去除排队延迟）

### 实验配置

- **GPU**: 8 张（GPU 0-7）
- **模型**: Qwen3-32B
- **四种部署方案**:
  1. **PDAF DynM** (TP=2 per component, P2D2A2F2): AF 分离架构，无 DVFS
  2. **PDAF DynM + Tier** (同上 + DVFS): AF 分离 + V2 耦合模型调频
  3. **PD DP4** (4× 1P1D 实例): PD 分离，DP=4
  4. **Native DP8** (8× TP=1 单实例): 原生数据并行，round-robin 路由
- **SLO**: TTFT=5000ms, TPOT=300ms（宽松约束）
- **Workloads**: steady / varying / heavy / overload / tier1_demo

### 核心结果

#### TTFT Processing Time（不含排队）

| 方案 | steady | varying | heavy | overload | tier1_demo |
|---:|---:|---:|---:|---:|---:|
| **PDAF DynM** | 113.9ms | 102.9ms | 116.7ms | 116.9ms | 116.4ms |
| **PDAF + Tier** | 320.6ms | 177.0ms | 286.4ms | 248.2ms | 250.6ms |
| **PD DP4** | 108.9ms | 85.1ms | 111.4ms | 88.6ms | 86.3ms |
| **Native DP8** | 221.4ms | 132.3ms | 226.2ms | 189.8ms | 189.0ms |

#### 吞吐量 (tok/s)

| 方案 | steady | varying | heavy | overload | tier1_demo |
|---:|---:|---:|---:|---:|---:|
| **PDAF DynM** | 689.7 | 180.0 | 260.4 | 676.0 | 665.3 |
| **PDAF + Tier** | 651.8 | 179.3 | 258.9 | 635.0 | 619.2 |
| **PD DP4** | 751.3 | 182.9 | 262.6 | 1004.5 | 880.5 |
| **Native DP8** | 604.7 | 182.9 | 261.4 | 638.7 | 612.8 |

#### 总能耗 (J) 与 Tier 节能

| 方案 | steady | varying | heavy | overload | tier1_demo |
|---:|---:|---:|---:|---:|---:|
| **PDAF DynM** | 147,991 | 111,089 | 124,618 | 176,504 | 188,070 |
| **PDAF + Tier** | 111,816 | 85,337 | 92,907 | 139,352 | 149,955 |
| **PD DP4** | 221,038 | 167,195 | 187,524 | 195,818 | 227,156 |
| **Native DP8** | 150,582 | 106,535 | 115,558 | 175,668 | 191,758 |
| **Tier 节能** | **24.4%** | **23.2%** | **25.4%** | **21.0%** | **20.3%** |

#### TPOT (ms)

| 方案 | steady | varying | heavy | overload | tier1_demo |
|---:|---:|---:|---:|---:|---:|
| **PDAF DynM** | 108.1 | 87.2 | 93.7 | 109.4 | 107.7 |
| **PDAF + Tier** | 120.4 | 92.2 | 101.8 | 114.8 | 114.5 |
| **PD DP4** | 48.0 | 46.2 | 47.1 | 48.5 | 48.3 |
| **Native DP8** | 84.4 | 69.6 | 86.7 | 79.5 | 81.2 |

### 关键发现

1. **Tier DVFS 使 TTFT proc 增加 2-3 倍**：Tier 方案（320ms）vs 无 DVFS 的 PDAF（114ms），因为 Prefill 阶段的 DVFS 降频直接拉长了处理时间。这是 DVFS 节能的本质代价——用延迟换能耗。

2. **PD DP4 的 TTFT proc 最低**（85-111ms）：因为 4 个独立 PD 实例共享负载，每个实例负载更轻，且 Prefill 不与 Decode 争抢 GPU 时间片。

3. **Native DP8 的 TTFT proc 偏高**（132-226ms）：虽然有 8 个实例，但每个实例是 TP=1（单卡），模型推理本身就更慢（无张量并行加速），导致即使不排队，处理时间也较长。

4. **Tier 在所有 workload 下稳定节能 20-25%**，同时保持 SLO 违背率 0%（因为 SLO 设得宽松 = 5000ms TTFT / 300ms TPOT）。

---

## Part 2: 4-GPU TTFT SLO 敏感度分析（去除排队延迟）

### 实验配置

- **GPU**: 4 张（GPU 0-3）
- **模型**: Qwen3-32B, TP=1, AF 分离
- **部署**: PDAF DynM (Baseline=满频) vs PDAF DynM + Tier (V2 DVFS)
- **TPOT SLO**: 固定 150ms
- **TTFT SLO**: 5000 / 2000 / 1000 / 500 / 300 / 200 ms
- **Workload**: workload_steady

### 核心结果

| TTFT SLO | BL proc | V2 proc | V2 节能 | V2 SLO 违背 | V2 TPOT |
|---:|---:|---:|---:|---:|---:|
| 5000ms | 115ms | 528ms | 30.0% | 0.0% | 84ms |
| 2000ms | 115ms | 528ms | 30.0% | 0.0% | 83ms |
| 1000ms | 115ms | 541ms | 28.6% | 0.0% | 83ms |
| 500ms | 115ms | 412ms | 25.3% | 13.5% | 84ms |
| 300ms | 115ms | 326ms | 23.9% | 69.3% | 83ms |
| 200ms | 115ms | 253ms | 22.6% | 96.5% | 83ms |

### 关键发现

1. **Baseline TTFT proc 恒定 115ms**：不管 SLO 如何变化，baseline 始终以最高频率运行，处理时间不受约束控制。

2. **V2 DVFS 在响应 TTFT SLO 约束**：随着 TTFT SLO 从 5000ms 收紧到 200ms，V2 的 Prefill 处理时间从 528ms 降至 253ms。V2 控制器选择更高的频率来加速 Prefill。

3. **节能与 TTFT SLO 的 trade-off 较平缓**：节能从 30% 仅下降到 22.6%（差 7.4pp），因为**大部分节能来自 Decode 阶段的 DVFS**，Prefill 提频的能耗代价相对较小。

4. **排队延迟是 SLO 违背的主要原因**：
   - TTFT SLO=500ms 时，V2 proc=412ms < 500ms（理论上不应违背）
   - 但端到端 TTFT=970ms（含排队 558ms），SLO 违背 13.5%
   - 说明当前 DVFS 控制器只优化了处理时间，**未考虑排队延迟**

5. **V2 有效控制范围**：TTFT proc 在 SLO≥1000ms 时不变（528ms），说明控制器在约束宽松时总是选最低频；SLO<1000ms 时开始响应——临界点在 TTFT SLO ≈ 1000ms。

---

## Part 3: Joint TTFT × TPOT SLO 敏感度测试（计划中）

### 实验设计

- **GPU**: 4 张（GPU 4-7）
- **模型**: Qwen3-32B, TP=1, AF 分离
- **部署**: PDAF DynM (Baseline) vs PDAF DynM + Tier (V2 DVFS)
- **TTFT SLO**: 2000 / 1000 / 500 / 300 / 200 / 150 ms
- **TPOT SLO**: 200 / 150 / 120 / 100 / 90 / 80 ms
- **总组合**: 6 × 6 × 2 = 72 个测试点
- **Workload**: workload_steady
- **指标**: TTFT proc (ms), TPOT (ms), Energy Saving (%), SLO Violation (%), Per-token TPOT Violation (%)

### 预期热力图

输出格式同 Version 6 的 2D 热力图，但 TTFT 轴使用 **Processing Time（去除排队）**：
1. Energy Saving vs Baseline (%) — 节能随 SLO 收紧如何变化
2. Overall SLO Violation Rate (%) — 哪些 SLO 组合不可行
3. TTFT Processing Time (ms) — 排除排队后的真实 Prefill 延迟
4. Per-Token TPOT Violation (%) — Decode 质量

### 已完成的部分数据（TTFT=2000ms 行）

| TTFT SLO | TPOT SLO | V2 proc | V2 TPOT | V2 Energy | BL Energy | Saving | V2 SLO% |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2000 | 200 | 536ms | 84ms | 69,100J | 99,352J | 30.4% | 0.0% |
| 2000 | 150 | 530ms | 84ms | 69,294J | 99,104J | 30.1% | 0.0% |
| 2000 | 120 | 531ms | 84ms | 69,006J | 99,439J | 30.6% | 0.0% |
| 2000 | 100 | 538ms | 97ms | 69,242J | — | — | 43.3% |

### 运行命令

```bash
cd /workspace/sglang-tier
/workspace/env/sglang-tier/bin/python benchmark/energy_bench/retrain/run_joint_sweep_v2.py
```

脚本: `benchmark/energy_bench/retrain/run_joint_sweep_v2.py`
绘图: `benchmark/energy_bench/retrain/analyze_joint_sweep_v2.py`

---

## 结论与方向

1. **去除排队后的核心发现**：Tier DVFS 对 TTFT Processing Time 的影响是 2-5 倍增加（从 115ms 到 250-530ms），但这个代价换来了 **20-30% 的能耗降低**。在 SLO 宽松时（≥1000ms），这是一个良好的 trade-off。

2. **DVFS 控制器的改进方向**：
   - 当前控制器仅对 TTFT SLO 做出反应（提频），但**未感知排队延迟**
   - 如果能将排队预测纳入决策（例如在高负载时预判排队长，主动提频），可以大幅减少 SLO 违背
   - 建议下一步实现 **queuing-aware DVFS**

3. **部署方案选择建议**：
   - 对延迟敏感场景（TTFT proc < 150ms）：选择 PD DP4（proc 最低 85-111ms）
   - 对能效敏感场景（节能 > 20%）：选择 PDAF + Tier（代价是 proc 增加到 250-530ms）
   - 平衡方案：PDAF DynM 无 DVFS（proc=114ms，能耗中等）
