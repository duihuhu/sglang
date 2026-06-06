# 02 — Macro-benchmark（变长数据集）

**实验目的**：在**变长序列**（从真实 trace 采样的长度分布）、多种负载形态下，对比 PDAF / PD / Native 各架构方案的性能与能耗。变长更贴近真实部署，用于验证各方案在动态负载下的能效与 SLO 表现。

## 变长 Workload

| Workload | 请求数 | 特点 |
|----------|--------|------|
| steady | 600 | 稳定 QPS=5，混合长度 |
| varying | 280 | 突发模式，低→高→低 |
| heavy | 360 | 长序列，高计算需求 |
| overload | 690 | 高 QPS=10，压力测试 |
| tier1_demo | 780 | 混合模式，展示 Tier1 效果 |

数据集与长度分布分析见 `workloads/`（`workload_*.jsonl` + `analysis/`，含 CV、长度分布、QPS-over-time 等图）。

## 涉及的历史版本

| 文档 | 内容 |
|------|------|
| [`version5_4gpu_var.md`](version5_4gpu_var.md) | 4 卡变长全方案（PD TP2 / Native TP2 DP2 / PDAF DynM / PDAF Tier+DynM）；PDAF Tier+DynM 总能耗省 27.3% |
| [`version6_8gpu_var.md`](version6_8gpu_var.md) | 8 卡变长（PDAF DynM/Tier、Native TP8、PD DP4），SLO=2000ms/150ms；附 TTFT×TPOT 联合 sweep |
| [`version7_ttft_proc.md`](version7_ttft_proc.md) | **去排队 TTFT 处理时间口径**重测；4 卡三方案（PDAF/PD DP2/Native DP4）Tier vs 满频对比 |

## 目录结构

| 路径 | 内容 |
|------|------|
| `4gpu/results_4gpu_var/`, `charts_4gpu_var/` | 4 卡变长四方案对比（throughput/ttft/tpot/energy/slo/radar 图） |
| `4gpu/results_4gpu_var_v2/`, `charts_4gpu_var_v2/` | V2 调频模型重测 |
| `4gpu/results_4gpu_3way/`, `_3way_bl/`, `charts_4gpu_3way/` | PDAF / PD DP2 / Native DP4 三方案 Tier vs 满频（去排队口径），含 `README.md` 与 P/D 能耗分解图 |
| `8gpu/results_8gpu_var/` | 8 卡变长四方案，`figures/`（comparison/energy_breakdown/slo_violations/dvfs_savings） |
| `8gpu/results_8gpu_v2/`, `_v2_energy/` | 8 卡 V2 数据与能耗 |
| `workloads/` | 变长数据集 + `analysis/`（特征分析脚本与图） |
| `scripts/` | `run_4gpu_deploy_bench.py`、`run_8gpu_deploy_bench.py`、`analyze_*`、`plot_*`、`run_*.sh` |
| `raw_logs/` | 各 run 原始 server 日志（4gpu_var / 8gpu_var / 4gpu_3way / var_* 等） |

## 复现命令

```bash
cd /workspace/sglang-tier
PY=/workspace/env/sglang-tier/bin/python
# 4 卡三方案（GPU 4-7，去排队 TTFT 口径，SLO TTFT2000/TPOT150）
$PY benchmark/AFlex_bench/02_macro_benchmark/scripts/run_4gpu_deploy_bench.py
$PY benchmark/AFlex_bench/02_macro_benchmark/scripts/plot_4gpu_3way.py
```

## 关键结论

1. **绝对能耗最低始终是 PDAF**：即便 Native 的 Tier 节能比例最高（34~39%），其 Tier 后能耗仍高于 PDAF。
2. **Native 的 Tier 节能比例最大（~36%）**：单卡 TP=1 利用率低、slack 大，降频空间充足。
3. **PD DP2 几乎无 Tier 收益（~1%）**：能耗集中在 memory-bound 的 Decode，对频率不敏感。
4. **去排队 TTFT 口径**：`ttft_proc` 才能反映真实 prefill 能力；高负载下 SLO 违背几乎全部来自 TTFT 排队，而非 TPOT。
5. **DVFS 用延迟换能耗**：PDAF proc TTFT 115→535ms，Native 157→237ms，PD 几乎不变（~49ms）。
