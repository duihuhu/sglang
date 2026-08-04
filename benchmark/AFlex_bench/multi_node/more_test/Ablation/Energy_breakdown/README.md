# Tier1 频率敏感性实验（Energy breakdown）

## 产物

- `data/tier_perf_energy_all.json` — 三组方案的完整 benchmark 数据（含 `request_results` 逐请求记录）
- `charts/tier_perf_energy.pdf` — 输出图

## 数据说明

`tier_perf_energy_all.json` 包含三组方案，每组 8 个点（code/conv × QPS 2/4/8/16）：

| 系列 | 键名 | 含义 |
|------|------|------|
| Vanilla（灰） | `megascale` | 固定 1P+1D TP4，无 DVFS |
| +Scheduler（绿） | `aflex_no_dvfs` | macro e2e AFlex 拓扑，无 DVFS |
| AFlex（红） | `aflex_dvfs` | macro e2e AFlex 拓扑，带 DVFS |

## 脚本

```bash
cd benchmark/AFlex_bench/multi_node/more_test/Ablation/Energy_breakdown/scripts

# 强制清理 node3/node4 端口
python3 force_cleanup_cluster.py

# 单独跑 benchmark（中间结果写入 data/）
python3 run_aflex_e2e_no_dvfs.py --resume      # +Scheduler
python3 run_vanilla_1p1d_tp4_no_dvfs.py --resume  # Vanilla

# 合并 + 绘图
python3 merge_tier_perf_data.py
python3 ../charts/plot_tier_perf.py

# 一键全流程
python3 run_all.py --resume
```
