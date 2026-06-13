# 1PA1PF2DA4DF 异构方案 Benchmark 任务

## 目标

在 `8gpu_azure_simplified_comparison.png` 图中补充 `1PA1PF+2DA4DF` 方案（有无 Tier）的测试数据。
该方案使用 V3 异构 TP 能耗模型（tp_a=2, tp_f=4）进行 DVFS 调频。

## 部署配置

- **方案名**: `pdaf_8g_asym_1p6d` / `pdaf_8g_asym_1p6d_tier`
- **GPU 布局**:
  - Prefill: PF(TP=1, GPU0) + PA(TP=1, GPU1) → 2 GPU
  - Decode:  DF(TP=4, GPU2-5) + DA(TP=2, GPU6-7) → 6 GPU
- **Tier 版本使用 V3 能耗模型**: `--afd-energy-model-v3-dir models_v3/`（支持 tp_a=2, tp_f=4 异构预测）

## 工作负载（6个）

| # | 文件 | 说明 |
|---|------|------|
| 1 | `workload_azure_code_light_real.jsonl` | Code 轻负载 |
| 2 | `workload_azure_code_medium_real.jsonl` | Code 中负载 |
| 3 | `workload_azure_code_heavy_real.jsonl` | Code 重负载 |
| 4 | `workload_azure_conv_light_real.jsonl` | Conv 轻负载 |
| 5 | `workload_azure_conv_medium_real.jsonl` | Conv 中负载 |
| 6 | `workload_azure_conv_heavy_real.jsonl` | Conv 重负载 |

## SLO 参数

- TTFT SLO: 4000ms
- TPOT SLO: 200ms
- Max run time: 600s/workload

## 当前进度

### Step 1: 无 Tier (pdaf_8g_asym_1p6d) — 已完成 ✅

**状态**: ✅ 全部完成

**结果**:
| Workload | Thpt (tok/s) | TTFT (ms) | TPOT (ms) | Energy (J) | SLO% |
|----------|:---:|:---:|:---:|:---:|:---:|
| code_light | 44.5 | 3,738 | 79 | 359,901 | 0.0 |
| code_medium | 66.6 | 107,836 | 81 | 632,915 | 0.1 |
| code_heavy | timeout (600s) | - | - | 1,097,623 | - |
| conv_light | 315.3 | 1,602 | 96 | 381,444 | 0.4 |
| conv_medium | 447.2 | 35,176 | 108 | 496,724 | 0.3 |
| conv_heavy | 469.3 | 146,085 | 108 | 774,031 | 0.5 |

### Step 2: 有 Tier (pdaf_8g_asym_1p6d_tier) — 已完成 ✅

**状态**: ✅ 全部完成（使用 V3 异构 TP 能耗模型）

**结果**:
| Workload | Thpt (tok/s) | TTFT (ms) | TPOT (ms) | Energy (J) | SLO% | 节能 vs 无Tier |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|
| code_light | 41.9 | 9,372 | 87 | 291,938 | 0.0 | -18.9% |
| code_medium | 51.1 | 181,243 | 88 | 605,458 | 0.5 | -4.3% |
| code_heavy | timeout (600s) | - | - | 765,699 | - | -30.2% |
| conv_light | 310.9 | 4,867 | 110 | 298,868 | 0.3 | -21.6% |
| conv_medium | 380.5 | 70,907 | 108 | 458,748 | 0.4 | -7.6% |
| conv_heavy | 382.1 | 207,114 | 118 | 720,941 | 0.8 | -6.9% |

### Step 3: 绘图 — 已完成 ✅

绘图脚本: `plot_8gpu_azure_simplified.py`
输出: `charts_8gpu_azure/8gpu_azure_simplified_comparison.png`

图中包含 8 个方案: Native DP8, Native DP8+Tier, PD DP4, PD DP4+Tier, PDAF Sym, PDAF Sym+Tier, PDAF Asym 1P+6D, PDAF Asym 1P+6D+Tier

## 结果输出位置

- JSON 结果: `/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/8gpu/results_8gpu_azure_slo2/json/pdaf_8g_asym_1p6d_*.json`
- 服务器日志: `/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/8gpu/logs_8gpu_azure_asym/`
- 运行日志: `logs_asym_1p6d_run.log` / `logs_asym_1p6d_tier_run.log`

## 检查方法

```bash
# 检查进程是否还在运行
ps -p 3031789

# 检查运行日志
tail -30 /workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/8gpu/logs_asym_1p6d_run.log

# 检查已保存的结果数量
ls /workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/8gpu/results_8gpu_azure_slo2/json/pdaf_8g_asym* 2>/dev/null | wc -l

# 全部6个工作负载跑完后应有6个文件:
# pdaf_8g_asym_1p6d_var_azure_code_heavy_real_results.json
# pdaf_8g_asym_1p6d_var_azure_code_medium_real_results.json
# pdaf_8g_asym_1p6d_var_azure_code_light_real_results.json
# pdaf_8g_asym_1p6d_var_azure_conv_heavy_real_results.json
# pdaf_8g_asym_1p6d_var_azure_conv_medium_real_results.json
# pdaf_8g_asym_1p6d_var_azure_conv_light_real_results.json
```

## 代码修改说明

本次测试涉及的代码修改（已完成）：

1. **`af_profile_predictor.py`** — 新增 V3 异构 TP 模型加载 & `tp_a`/`tp_f` 预测接口
2. **`af_dvfs_controller.py`** — 调用 predict 时传入 `tp_a`/`tp_f`
3. **`server_args.py`** — 新增 `--afd-energy-model-v3-dir` CLI 参数
4. **`scheduler.py`** — Tier2 predictor 实例化传入 v3_model_dir
5. **`run_8gpu_deploy_bench.py`** — `_dvfs_args()` 加入 `--afd-energy-model-v3-dir`
6. **`run_fixed_qps_bench.py`** — 新增 `ENERGY_MODEL_V3_DIR` 常量

## 时间预估

每个工作负载最多 600s + 启动/清理约 120s = ~720s/workload。
6 workloads × 720s ≈ 72min per deploy。
两个 deploy (无Tier + 有Tier) ≈ 2.5h 总计。
