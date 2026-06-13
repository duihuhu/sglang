# AFlex 实验文档

## 1. Evaluation Setup

### 1.1 硬件环境

| 项目 | 配置 |
|------|------|
| GPU | 8× NVIDIA H800 80GB SXM |
| GPU 频率档位 | 210 / 450 / 690 / 930 / 1170 / 1410 MHz |
| 互联 | NVLink + InfiniBand (mlx5) |
| CPU | 多核 x86_64 |
| 实验分组 | 4 卡实验使用 GPU 4-7；8 卡实验使用 GPU 0-7 |
| 能耗计量 | NVML `nvmlDeviceGetTotalEnergyConsumption` |

### 1.2 软件环境

| 项目 | 版本 |
|------|------|
| Python | ≥ 3.10 |
| CUDA | 12.9 |
| PyTorch | 2.9.1 |
| FlashInfer | 0.6.6 |
| Transformers | 5.3.0 |
| 推理框架 | SGLang (修改版，支持 AFlex) |
| 包管理器 | uv |
| 环境路径 | `/workspace/env/sglang-tier/bin/python` |

### 1.3 模型

| 模型 | 参数量 | 用途 |
|------|--------|------|
| Qwen3-32B | 32B | 主要评测模型 |

### 1.4 系统架构方案

| 方案缩写 | 全称 | 说明 |
|----------|------|------|
| PDAF | PD + AF Disaggregation | Prefill-Decode 分离 + Attention-FFN 算子级分离，TP=1 |
| PDAF+Tier | PDAF + Tier1/Tier2 DVFS | PDAF 基础上加两层动态调频 |
| PD DP2 | PD Disagg + DP2 | Prefill-Decode 分离，2 路数据并行 |
| PD DP2+Tier | PD DP2 + DVFS | PD DP2 + BiScale 风格调频 |
| Native DP4 | Native + DP4 | 原生推理 4 路数据并行，TP=1 |
| Native DP4+Tier | Native DP4 + DVFS | Native DP4 + DynamoLLM 风格调频 |

### 1.5 AFlex 组件

| 层级 | 组件 | 功能 |
|------|------|------|
| Tier1 | 资源规划层 | 全局频率/资源重规划（周期性，秒级） |
| Tier2 | 逐 Batch DVFS | 基于能耗预测模型的实时调频（迭代级，ms 级） |
| DynM | 动态微批 | 根据 batch size 动态选择 M=1 或 M=2 |
| IPC | 通信后端 | CUDA IPC / UCX RDMA 用于 AF 算子间数据传输 |

### 1.6 Baseline 对标方案

| Baseline | 来源论文 | 策略 |
|----------|---------|------|
| DynamoLLM | DynamoLLM (ISCA'24) | Native + 分层 Tier 调频 |
| BiScale | BiScale (2024) | PD + phase-specific DVFS (prefill MPC + decode 逐 batch) |
| Max Freq | — | 锁定最高频率（满频运行） |

### 1.7 SLO 配置

| 指标 | 默认值 | 说明 |
|------|--------|------|
| TTFT SLO | 2000 ms | 首 token 延迟（纯处理时间，去排队） |
| TPOT SLO | 150 ms | 每 token 输出延迟 |

### 1.8 评测指标

- **TTFT** (Time-To-First-Token)：首 token 延迟（ms）
- **TPOT** (Time-Per-Output-Token)：每 token 输出延迟（ms）
- **Throughput**：吞吐量（tok/s）
- **Energy**：总能耗（J），通过 NVML 能量计数器采集
- **SLO Violation Rate**：SLO 违背率（%）
- **Energy Saving**：相对满频的节能百分比（%）

---

## 2. Micro-benchmark

**实验目的**：在固定 input/output length、受控 QPS 条件下，精确刻画各方案的能效特征，消除长度波动干扰。

### 2.1 数据集

定长合成数据集，参数组合：

| 配置 | Input Length | Output Length | QPS |
|------|-------------|--------------|-----|
| 均衡 | 512 | 128 | 1, 2, 4, 6 |
| Prefill-heavy | 2048 | 256 | 1, 2, 4 |
| Decode-heavy | 256 | 512 | 1, 2, 4, 6 |
| 长输入 | 1024 | 128 | 1, 2, 4, 6 |

### 2.2 实验配置

- **4 卡实验**：GPU 4-7，三种频率模式（tier1_freq / max_freq / auto_freq）
- **8 卡实验**：GPU 0-7，四方案对比（PD TP4 / PD DP4 / PDAF DynM / PDAF Tier+DynM）

### 2.3 关键结果

| 发现 | 数据 |
|------|------|
| Tier1 稳定省电 | ~30%（均衡/长输入 26~34%），0% SLO 违背 |
| auto_freq ≈ max_freq | 持续负载下硬件自动 boost，"不干预"≈"锁最高频" |
| Tier 在 prefill-heavy 负载最优 | il2048/il4096 省 35~37% |
| Decode-heavy 负载 Tier 无收益 | memory-bound，对频率不敏感 |

### 2.4 图表

- `01_micro_benchmark/4gpu/fixed_qps/figures/compare_il*.png` — 各配置能耗/吞吐/TTFT/TPOT 对比
- `01_micro_benchmark/8gpu/charts_v4/` — 8 卡全方案 throughput/ttft/tpot/slo/energy vs QPS
- `01_micro_benchmark/8gpu/deploy/figures/deploy_tier_tradeoff.png` — Tier 延迟-能耗权衡

---

## 3. Macro-benchmark

**实验目的**：在变长真实 trace 数据集下验证各方案在动态负载中的能效与 SLO 表现。

### 3.1 数据集

#### 自构建变长 Workload

| Workload | 请求数 | 特点 |
|----------|--------|------|
| steady | 600 | 稳定 QPS=5，混合长度 |
| varying | 280 | 突发模式，低→高→低 |
| heavy | 360 | 长序列，高计算需求 |

#### Azure LLM Inference Trace（真实数据）

| Trace | IL avg/p50 | OL avg/p50 | 特征 |
|-------|-----------|-----------|------|
| Code (代码补全) | 2511/1930 | 23/8 | Prefill-heavy，极短 output |
| Conv (对话) | 1632/928 | 106/41 | 均衡，中等 decode |

负载等级：Light / Medium / Heavy（按 QPS 缩放适配 4 卡容量）。

### 3.2 关键结果 — 三方案对比（变长 Workload）

**节能百分比（Tier vs 满频）：**

| 架构 | steady | varying | heavy |
|------|-------:|--------:|------:|
| PDAF+Tier | 30.2% | 27.7% | 29.3% |
| PD DP2+Tier | 16.0% | 29.0% | 18.2% |
| Native DP4+Tier | 39.2% | 34.4% | 36.2% |

**PDAF+Tier 绝对能耗最低**（即便 Native 节能比例最高，其 Tier 后绝对能耗仍高于 PDAF）：

| 架构 | steady (J) | varying (J) | heavy (J) |
|------|------:|------:|------:|
| PDAF+Tier | 69,260 | 54,555 | 61,069 |
| Native DP4+Tier | 103,066 | 83,288 | 89,797 |
| PD DP2+Tier | 193,081 | 63,907 | 83,776 |

### 3.3 关键结果 — Azure Trace

**Native DP4+Tier 节能：**
- Code trace: Light 28.0% / Medium 38.7% / Heavy 24.0%
- Conv trace: Light 36.1% / Medium 34.8% / Heavy 13.9%

**PDAF 在长 IL trace（Code）下不适用**：TP=1 单卡无法在 SLO 内处理 IL>2000。

### 3.4 关键结论

1. **绝对能耗最低始终是 PDAF**：PDAF+Tier 比 Native DP4+Tier 低 32-35%
2. **Native Tier 节能比例最大（~36%）**：单卡利用率低、slack 大，降频空间充足
3. **负载越重 Tier 空间越小**：高负载 SLO slack 紧张，降频余地有限
4. **DVFS 用延迟换能耗**：PDAF proc TTFT 115→535ms，Native 157→237ms

### 3.5 图表

- `02_macro_benchmark/4gpu/charts_4gpu_3way/4gpu_3way_comparison.png` — 三方案 2×2 对比
- `02_macro_benchmark/4gpu/charts_4gpu_azure/azure_*_comparison.png` — Azure trace 对比
- `02_macro_benchmark/8gpu/results_8gpu_var/figures/8gpu_var_comparison.png` — 8 卡变长对比

---

## 4. Sensitivity

**实验目的**：固定架构，变换关键配置参数（SLO 阈值、微批数 M、QPS），观察系统能效与 SLO 表现的敏感度。

### 4.1 SLO 阈值扫描

**扫描范围（8×8 联合 sweep）：**
- TTFT SLO (ms): 1000 / 500 / 450 / 400 / 350 / 300 / 200 / 100
- TPOT SLO (ms): 200 / 150 / 140 / 130 / 120 / 100 / 90 / 80

**对比策略：** V2 DVFS（Tier 调频）vs Baseline（满频）

**关键发现：**
- V2 调频模型在 SLO≥90ms 仍可用，V1 仅 SLO≥250ms 可用
- TTFT SLO 主导 Prefill 频率，TPOT SLO 主导 Decode 频率，两维基本正交
- TPOT SLO 临界点 ~100ms（系统底噪约 83ms），≤80ms 不可行（>93% 违背）
- 宽松 SLO 下节能 ~28%，极紧 SLO（TPOT=80）下节能降至 ~15%

### 4.2 微批数 M 扫描

| M 值 | 特点 |
|------|------|
| M=1 | 无流水线，低通信开销 |
| M=2 (interleaved) | 大 batch 下 +58% 吞吐，最优 |
| M=3 | 通信次数增加，反而更差 |
| DynM | 动态选择：小 batch M=1，大 batch M=2 |

### 4.3 图表

- `03_sensitivity/slo_sweep/retrain/figures/joint_sweep_slo_violations_noqueue.png` — 8×8 SLO 热力图
- `03_sensitivity/slo_sweep/retrain/figures/joint_sweep_avg_freq.png` — 平均频率热力图
- `03_sensitivity/m_sweep/micro-batch-opt/results/len_sweep_throughput.png` — M 值吞吐对比

---

## 5. Ablation

**实验目的**：以 AFlex 完整方案为基准，逐一去掉组件，验证每个设计的必要性。

### 5.1 消融组件

| 消融目标 | 实验内容 | 关键结论 |
|----------|---------|---------|
| Tier1 资源规划层 | 禁用 Tier1 重规划 | G=4 时 Tier1 求解器 INFEASIBLE，节能实际来自 Tier2；需 G≥8 才体现价值 |
| Tier2 逐 batch DVFS | 对比多种调频策略 | **节能主要来自 Tier2**；不同策略间差异显著 |
| IPC 通信后端 | UCX RDMA vs CUDA IPC | CUDA IPC 小张量延迟更低，是 AF 分离的关键优化 |
| AF 架构本身 | 纯 AF / AF TP2 / Native | AF 通信开销 ~75-95ms/iter，拖低吞吐 40~60% |

### 5.2 Tier2 策略对比

对比不同 DVFS 决策策略的节能效果与 SLO 表现，验证 V2 耦合预测模型的优越性。

### 5.3 图表

- `04_ablation/tier2/tier2_strategies/tier2_strategies_comparison.png` — 策略对比
- `04_ablation/tier2/freq_monitor/freq_trace.png` — 频率轨迹
- `04_ablation/tier2/tier_trace.png` — 调频时间轴

---

## 6. Reshard Breakdown

**实验目的**：对系统环节做细粒度时延拆解，回答"时间花在了哪里"。

### 6.1 D-Stage Pipeline Breakdown

逐层、逐微批的 Attention/FFN 计时，识别 DA 与 DF 调度错配导致的 pipeline bubble。

**核心发现：**
- DA/DF 执行不同调度，~67% forward 时间在等待 UCX 传输
- GPU 利用率仅 ~33%
- M=3 pipeline 因 2 次串行 UCX 传输 + DA/DF 无 overlap，传输延迟主导

### 6.2 DVFS 决策拆解

解析每进程的 DVFS 决策日志，量化预测模型准确性。

**核心发现：**
- 预测器系统性低估 decode 迭代时间 ~28%（未充分计入 pipeline drain / IPC 同步开销）
- 但决策方向正确（按负载选频），SLO 余量充足未造成违背

### 6.3 图表

- `05_breakdown/dstage/01_dstage_breakdown/charts/` — Pipeline Gantt 图、per-layer breakdown、bubble explanation
- `05_breakdown/dvfs_decisions/` — DVFS 预测准确性分析

---

## 实验目录结构

```
AFlex_bench/
├── 01_micro_benchmark/        # 定长数据集 micro-benchmark
│   ├── 4gpu/                  # 4 卡定长 QPS sweep
│   ├── 8gpu/                  # 8 卡全方案对比
│   ├── scripts/               # 运行/绘图脚本
│   └── workloads/             # 定长数据集
├── 02_macro_benchmark/        # 变长真实 trace macro-benchmark
│   ├── 4gpu/                  # 4 卡三方案 + Azure trace
│   ├── 8gpu/                  # 8 卡变长
│   ├── scripts/               # 部署/分析/绘图脚本
│   └── workloads/             # 变长数据集 + 分析
├── 03_sensitivity/            # 敏感度测试
│   ├── slo_sweep/             # SLO 阈值扫描（8×8 联合 sweep）
│   ├── m_sweep/               # 微批数 M 扫描
│   └── qps_sweep/             # QPS 并发扫描
├── 04_ablation/               # 消融实验
│   ├── tier1/                 # Tier1 消融
│   ├── tier2/                 # Tier2 策略对比
│   ├── communication/         # IPC 通信对比
│   └── af_components/         # AF 架构组件消融
├── 05_breakdown/              # 时延拆解
│   ├── dstage/                # D-stage pipeline breakdown
│   ├── dvfs_decisions/        # DVFS 决策分析
│   ├── pipeline_viz/          # 流水线可视化
│   └── throughput/            # 吞吐基准
└── 06_others/                 # 支撑资料
    ├── design_docs/           # 系统设计文档
    ├── baseline_papers/       # 对标论文 (DynamoLLM, BiScale)
    ├── energy_model_profiling/# 能耗模型 profiling 数据
    ├── utils/                 # 共享工具
    └── version_logs/          # 历史版本记录 (v0~v7)
```
