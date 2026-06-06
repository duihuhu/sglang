# 06 — Others（其他）

**实验目的**：存放不属于前五类的支撑性内容——设计文档、baseline 论文、能耗模型 profiling、调试工具、共享脚本与历史版本记录。

## 子目录

### `design_docs/` — 设计文档
| 文件 | 内容 |
|------|------|
| `system_detail_v1.md` | AFlex 系统详细设计（Tier1/Tier2 两层架构） |
| `AFlex__ATC_26_.pdf` | AFlex 论文稿（ATC'26 投稿） |
| `baseline_dvfs_v1.md` | PD/Native baseline 的 DVFS 实现设计（副本，主本在 `04_ablation/design/`） |

### `baseline_papers/` — 对标论文
| 文件 | 代表方案 |
|------|---------|
| `DynamoLLM.pdf` | Native + Tier 调频 baseline（分层控制，Instance Manager 调频） |
| `BiScale.pdf` | PD + Tier 调频 baseline（phase-specific DVFS：prefill MPC + decode 逐 batch） |

### `energy_model_profiling/` — 能耗模型 profiling
`profiling/`：A/F 算子级延迟与能耗 profiling（NVML 能量计数器），训练 GBDT/LinearReg 预测模型的原始数据。含 `PROGRESS.md`、`run_decode_af_bench.py`、`run_decode_af_sweep.py`、`data_decode_af/`、`logs/`。这是 Tier2 DVFS 频率选择的预测器数据来源。

### `ipc_debug/` — IPC 通信调试
`ipc_debug/`：C++ IPC、PDAF concurrent、cpp_ipc 等调试日志与脚本。

### `utils/` — 共享工具
| 路径 | 内容 |
|------|------|
| `energy_bench_utils/` | `gen_workload.py`、`gen_fixed_workload.py`、`monitor_dvfs.py`、`plot_tier_trace.py`、`trace_tier_decisions.py`、`check_env.py`、`demo_tier1_value.py` |
| `af_bench_utils/` | af_bench 共享的 log 解析、画图工具 |

### `misc/` — 杂项
`af_bench_README.md`（原 af_bench 总说明）、`af_bench_paper/`、`af_bench_plan/`、`benchmark_replay.py`、`af_bench_logs/`、`unsorted_bench_logs/`（`_archive_old`、`_sweep`、native_dp4_rerun 等未明确归属的早期日志）、`run_all.sh`。

### `version_logs/` — 历史版本文档全集
`version0.md` ~ `version7.md` + `version5_4gpu_var.md` + `plan.md`。记录了整个研究的演进脉络：
- v0：4 卡定长 Tier1/Tier2 节能评测
- v1~v3：早期迭代
- v4：8 卡全方案定长对比
- v5：4 卡变长 + multi-SLO sweep
- v6：TTFT×TPOT 联合 sweep + 8 卡变长
- v7：去排队 TTFT 处理时间口径分析

## 说明

本目录内容多为**支撑性资产**，不直接产出对比结论，但为前五类实验提供设计依据（design_docs/baseline_papers）、模型数据（profiling）、工具支持（utils）与历史追溯（version_logs）。
