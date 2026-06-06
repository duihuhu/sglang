# 04 — Ablation（消融实验）

**实验目的**：以 AFlex 完整方案（**PDAF + Tier1 + Tier2 + DynM + IPC 通信**）为基准，逐一去掉某个组件，观察其对性能/能耗的贡献。验证每个设计组件的必要性。

## 子目录

### `tier1/` — Tier1 资源规划层消融
Tier1 只做"频率重规划"不触发"模型重加载"（`--tier1-disable-reload`）。在 G=4 约束下 Tier1 求解器结构性 INFEASIBLE，节能实际来自 Tier2，Tier1 仅作监控。
- `scripts/`：`test_tier1_bubble.py`、`test_tier1_overload.py`、`test_tier1_reload.py`、`test_tier1_throughput.py`、`test_tier1_transition.py`、`test_tier1_tier2_coord.py`
- `tier1_overload/`、`tier1_shared/`、`tier1_tier2/`、`results_tier1_shared_bench/`：结果数据

### `tier2/` — Tier2 逐 batch DVFS 层消融
- `scripts/`：`run_tier2_bench.py`、`run_tier2_strategies.py`、`run_freq_monitor_bench.py`、`run_slo_sweep.py`、`test_tier2_standalone.py`
- `tier2_strategies/`：不同 DVFS 策略对比（含 `tier2_strategies_comparison.png`）
- `freq_monitor/`、`freq_monitor_tight/`：频率监控轨迹（`freq_trace.png`）
- `tier_trace.png`：调频时间轴

### `communication/` — IPC 通信消融
`02_communication/`：UCX RDMA vs CUDA IPC 通信后端对比，传输延迟、cycle 级 breakdown、wire time（含 charts + 原始 logs + 脚本）。AF 分离的通信开销是 PDAF 吞吐瓶颈的核心。

### `af_components/` — AF 架构组件消融
`06_af_experiments/`：纯 AF（无 PD）、AF-only TP2、CUDA graph 开关、native baseline（TP4/TP2）、final comparison。

### `design/`
[`baseline_dvfs_v1.md`](design/baseline_dvfs_v1.md)：PD+Tier（BiScale 风格）与 Native+Tier（DynamoLLM 风格）的 DVFS 实现设计，含 `UnifiedDVFSController` 单旋钮算法、scheduler 集成、公平性说明。

### `early_baseline_vs_dvfs/`
最早期 baseline vs dvfs 对比：`baseline_results.json` / `dvfs_results.json` / `comparison*.json`，及原始 server 日志（`server_logs/`、`energy_logs/`）。

## 复现命令

```bash
cd /workspace/sglang-tier
PY=/workspace/env/sglang-tier/bin/python
# Tier2 策略对比
$PY benchmark/AFlex_bench/04_ablation/tier2/scripts/run_tier2_strategies.py
# IPC vs UCX 通信 breakdown
$PY benchmark/AFlex_bench/04_ablation/communication/02_communication/scripts/run_ipc_breakdown.py
# 纯 AF baseline
$PY benchmark/AFlex_bench/04_ablation/af_components/06_af_experiments/run_baselines.py
```

## 关键结论

1. **节能主要来自 Tier2 逐 batch DVFS**；Tier1 在小预算（G=4）下无重规划空间，需 G≥8 才体现资源重规划价值。
2. **CUDA IPC 通信优于 UCX RDMA**：小张量传输 IPC 延迟更低，是 AF 分离的关键优化。
3. **AF 通信开销是 PDAF 吞吐瓶颈**：~75-95ms/iter 的 round-trip 拖低吞吐 40~60%。
