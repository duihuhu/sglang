# AFlex_bench — AFlex 能效评测与分析套件

> AFlex = **PDAF + Tier DVFS**：在 Prefill/Decode（PD）分离的基础上进一步做 Attention/FFN（AF）算子级分离，并叠加两层（Tier1 资源规划 + Tier2 逐 batch DVFS）频率调控，实现 SLO 约束下的能耗最优。

本目录由原 `benchmark/energy_bench/` 与 `benchmark/af_bench/` 两个目录**完整合并重组**而来，按实验目的归为 6 大类。所有代码、原始日志（`*.log`/`*.jsonl`）、结果数据（`*.json`）、图表（`*.png`）与分析文档（`*.md`）均已迁移，原两目录已删除。

---

## 1. 研究背景与方案

在 Qwen3-32B 服务上对比三种架构 + 各自的 DVFS 调频方案，研究 **SLO 约束下的能效（性能 / 能耗 / SLO 违背）权衡**：

| 方案 | 架构 | 调频方案 | 对标工作 |
|------|------|---------|---------|
| **AFlex（本工作）** | PDAF：PD 分离 + AF 算子分离（PA/PF/DA/DF 各占卡） | Tier1（只调频不重规划）+ Tier2 逐 batch DVFS，A/F 频率独立（f_A, f_F） | — |
| **PD + Tier**（baseline） | PD 分离（1P1D，TP=1） | Unified 单旋钮 DVFS（A/F 同卡同频） | BiScale |
| **Native + Tier**（baseline） | 原生 TP/DP 完整实例 | Unified 单旋钮 DVFS（整实例同频） | DynamoLLM |

核心指标：**TTFT**（含 `ttft_proc` 纯处理时间 / 端到端两种口径）、**TPOT**、**吞吐(tok/s)**、**总能耗(J，NVML 计量)**、**P/D 能耗分解**、**SLO 违背率**（TTFT 为 bool，TPOT 逐 token 统计百分比）。

通用配置：Qwen3-32B（64 层，hidden=5120，kv_heads=8），合法频率档 210/450/690/930/1170/1410 MHz，能耗由 NVML `nvmlDeviceGetTotalEnergyConsumption` 计量。

---

## 2. 目录总览

| 分类 | 目录 | 内容 | 体积 |
|------|------|------|------|
| **1. Micro-benchmark** | [`01_micro_benchmark/`](01_micro_benchmark/) | **定长** 数据集下各架构（PDAF/PD/Native）性能与能耗对比（4 卡 + 8 卡） | ~94M |
| **2. Macro-benchmark** | [`02_macro_benchmark/`](02_macro_benchmark/) | **变长** 数据集下的同类对比（4 卡 + 8 卡） | ~299M |
| **3. Sensitivity** | [`03_sensitivity/`](03_sensitivity/) | 变换配置的敏感度：SLO 扫描、M（微批 1/2/DynM）扫描、QPS 扫描 | ~351M |
| **4. Ablation** | [`04_ablation/`](04_ablation/) | 消融实验：去掉 Tier1 / Tier2 / IPC 通信 / AF 各组件后的效果 | ~36M |
| **5. Breakdown** | [`05_breakdown/`](05_breakdown/) | 阶段拆解：流水线 D-stage、pipeline 可视化、压力计时、DVFS 决策拆解 | ~416K |
| **6. Others** | [`06_others/`](06_others/) | 设计文档、baseline 论文、能耗模型 profiling、IPC 调试、工具脚本、历史版本 | ~31M |

每个分类目录下都有独立的 `README.md`，说明其实验目的、脚本、数据、图表与关键结论。

---

## 3. 各分类速览

### 01 Micro-benchmark（定长）
固定 input/output length，只变 QPS。对应历史 **version0**（4 卡 fixed_qps，tier1/max/auto 三模式，省电约 30%）与 **version4**（8 卡 il×ol sweep，PD TP4/DP4/PDAF DynM/Tier 四方案）。含 `disagg_arch_comparison`（PD+AF vs PD TP1/TP2 单请求与并发对比）。

### 02 Macro-benchmark（变长）
从真实 trace 采样的变长序列（steady/varying/heavy/overload/tier1_demo 5 种负载）。对应 **version5_4gpu_var**、**version6**（8 卡变长）、**version7**（去排队 TTFT 处理时间口径），以及 4 卡三方案对比（PDAF / PD DP2 / Native DP4，Tier vs 满频）。

### 03 Sensitivity（敏感度）
- **slo_sweep**：TPOT/TTFT SLO 单维与联合扫描（V1 vs V2 调频模型），含模型重训数据 `retrain/`
- **m_sweep**：微批数 M=1~5 扫描 + interleaved async pipeline（M=2 大 batch +58% 吞吐）
- **qps_sweep**：QPS/并发扫描

### 04 Ablation（消融）
- **tier1 / tier2**：分别去掉资源规划层与逐 batch DVFS 层
- **communication**：UCX RDMA vs CUDA IPC 通信后端对比
- **af_components**：纯 AF（无 PD）、CUDA graph 开关、各 baseline 配置
- **design**：`baseline_dvfs_v1.md`（PD/Native 调频实现设计）
- **early_baseline_vs_dvfs**：最早期 baseline vs dvfs 对比数据与原始日志

### 05 Breakdown（阶段拆解）
- **dstage**：逐层、逐微批的 A/F 计时与流水线 bubble 分析（GPU 利用率仅 ~33% 的根因）
- **pipeline_viz**：Gantt 图与流水线示意
- **stress_timing**：压力测试、序列长度扫描、async 正确性验证、bubble breakdown
- **throughput**：通用吞吐基准、timeline 验证
- **dvfs_decisions**：DVFS 调频决策与预测准确性分析

### 06 Others（其他）
设计文档（`system_detail_v1.md`、`AFlex__ATC_26_.pdf`、`baseline_dvfs_v1.md`）、baseline 论文（`DynamoLLM.pdf`、`BiScale.pdf`）、能耗模型 profiling（GBDT/LinearReg LUT 训练数据）、IPC 调试、共享工具脚本、历史版本文档（version0~7 全集 + plan）。

---

## 4. 运行环境

```bash
# Python 环境
/workspace/env/sglang-tier/bin/python

# 从仓库根目录运行
cd /workspace/sglang-tier
```

各分类 README 中给出了对应的复现命令。脚本中引用的源码改动（`ttft_pure_processing`、`UnifiedDVFSController`、Tier1/Tier2 hook 等）位于主仓 `python/sglang/srt/` 下。

---

## 5. 关键结论汇总

1. **能效最优是 PDAF + Tier**：变长负载轻中区间下比 PDAF DynM（满频）省 23~30%，绝对能耗也低于 PD/Native + Tier。
2. **Native 的 Tier 节能比例最大（~36%）**，但绝对能耗仍高（单卡 TP=1 利用率低、slack 大）；与 DynamoLLM 观察一致。
3. **PD 的 Tier 收益极小（~1%）**：能耗集中在 memory-bound 的 Decode，对频率不敏感；与 BiScale "decode 调频收益远小于 prefill" 一致。
4. **DVFS 用延迟换能耗**：Tier 使 TTFT 处理时间增加 2~5 倍，但在 SLO 宽松（≥1000ms）时是良好的 trade-off。
5. **靠硬件自动 boost 省不了电**：auto_freq ≈ max_freq，必须主动降频。
6. **TTFT 口径很关键**：含排队的 TTFT 在高负载被排队主导，去排队（`ttft_proc`）才能反映真实 prefill 能力；当前 DVFS 仅感知处理时间、未感知排队，是后续 queuing-aware DVFS 的改进方向。

> 各结论的详细数据见对应分类的 README 与版本文档（`06_others/version_logs/`）。
