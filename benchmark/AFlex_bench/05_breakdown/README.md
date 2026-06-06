# 05 — Breakdown（阶段拆解测试）

**实验目的**：对系统的具体环节做细粒度拆解分析，回答"时间/能耗花在了哪里"。不同于前四类的端到端对比，本类聚焦**单一环节的内部结构**。

## 子目录

### `dstage/` — 流水线 D-stage 拆解
`01_dstage_breakdown/`：逐层、逐微批的 Attention/FFN 计时，识别 DA 与 DF 调度错配导致的 pipeline bubble。
- 脚本：`analyze_pipeline_breakdown.py`(+v2)、`analyze_schedule.py`、`plot_dstage_pipeline*.py`、`plot_gantt_with_breakdown.py`
- 产物：`charts/`（per-MB breakdown、Gantt、bubble explanation）、`results/`（parsed timeline JSON）
- 详细报告：`breakdown_analysis_report.md`
- **核心发现**：DA/DF 执行不同调度，~67% forward 时间在等 UCX 传输，GPU 利用率仅 ~33%。

### `pipeline_viz/` — 流水线可视化
`03_pipeline_viz/`：Gantt 图、流水线示意图生成脚本。

### `stress_timing/` — 压力与计时
`07_stress_timing/`：长时压力测试、序列长度扫描、async recv 正确性验证、bubble breakdown。

### `throughput/` — 吞吐与 timeline
`08_throughput/`：通用吞吐基准、Gantt bench、快速计时脚本、M=3 timeline 验证。

### `dvfs_decisions/` — DVFS 决策拆解
`analyze_dvfs_decisions.py`、`plot_dvfs_decisions.py`：解析每进程的 DVFS 决策 JSONL，量化预测模型准确性（`obs_iter_us` vs `pred_iter_cur_us`）、频率选择分布。
- **核心发现**：预测器系统性低估 decode 迭代时间 ~28%，但决策方向正确（按负载选频、SLO 不违背）。

## 复现命令

```bash
cd /workspace/sglang-tier
PY=/workspace/env/sglang-tier/bin/python
# D-stage breakdown 分析
$PY benchmark/AFlex_bench/05_breakdown/dstage/01_dstage_breakdown/scripts/analyze_pipeline_breakdown.py
# DVFS 决策准确性分析（决策日志在 03_sensitivity/slo_sweep/retrain/logs_*）
$PY benchmark/AFlex_bench/05_breakdown/dvfs_decisions/analyze_dvfs_decisions.py
```

## 关键结论

1. **M=3 pipeline GPU 利用率仅 ~33%**：每层 2 次串行 UCX 传输 + DA/DF 无 overlap，传输延迟主导。
2. **DVFS 预测模型系统性低估 decode 延迟 ~28%**：疑因未充分计入 pipeline drain / IPC 同步开销，但 SLO 余量充足未造成违背。
