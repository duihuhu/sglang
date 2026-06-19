# Version 14: 能耗模型训练数据整理 + Git 跟踪

## 概述

将 `retrain/` 下的能耗模型 profile 数据整理为两套**自包含**数据集，纳入 Git 跟踪；同步更新 V1/V2/V3 训练脚本，使拟合时只读取对应文件夹内的数据，不交叉引用。另新增 Dense vs MoE Prefill/Decode 特性对比分析脚本。

**暂存统计：** 17 files, +23662 / -59 lines

---

## 一、训练数据目录重组

原 `data/` 根目录下 7 个散乱 TXT 文件，整理为两个独立文件夹：

### `data/v1_layer_profile/` — V1 完整数据集

| 文件 | 行数 | 说明 |
|------|------|------|
| `prefill_data_v1.txt` | 2204 | Prefill 逐层 profile（Qwen3-32B dense） |
| `decode_data_v1.txt` | 7532 | Decode 逐层 profile，同构 TP 1/2/4/8 |
| `README.md` | — | 数据集说明与用法 |

**用途：** 仅本文件夹即可拟合 V1 模型（Prefill/Decode 独立层 LUT + LinearReg + GBDT）。

### `data/v2_pipeline_profile/` — V2/V3 完整数据集

| 文件 | 行数 | 说明 |
|------|------|------|
| `prefill_data_v1.txt` | 2204 | Prefill 逐层 profile（与 V1 同源，副本） |
| `decode_pipeline_v1.txt` | 1761 | TP=1 pipeline profile，M=1/2 |
| `decode_pipeline_tp2.txt` | 1865 | TP=2 同构 pipeline |
| `decode_pipeline_hetero_2a4f.txt` | 1721 | 异构 tp_a=2, tp_f=4 |
| `decode_pipeline_merged.txt` | 5344 | 合并集（1×1, 2×2, 2×4） |
| `README.md` | — | 数据集说明与用法 |

**用途：** 仅本文件夹即可拟合 V2/V3 模型（Prefill 逐层 + Decode pipeline GBDT/LUT）。

**删除：** `decode_pipeline_merged_v2_backup.txt`（冗余备份，内容已含于 merged）。

---

## 二、训练脚本更新

### `energy_model_v1.py`

- 新增 `--data-dir`，默认 `data/v1_layer_profile`
- `--prefill` / `--decode` 可选覆盖；未指定时从 `data-dir` 自动解析

### `energy_model_v2.py`

- 新增 `--data-dir`，默认 `data/v2_pipeline_profile`
- 新增 `--decode-data`，默认 `decode_pipeline_v1.txt`
- 新增 `train_prefill_models()`：从同目录 `prefill_data_v1.txt` 训练 Prefill_A/F 能量与延迟模型
- V2 拟合流程：**Prefill 逐层 + Decode pipeline（TP=1）**，全部数据来自同一 `data-dir`

### `energy_model_v3.py`

- 同 V2 的 `--data-dir` 机制
- `--decode-data` 默认 `decode_pipeline_merged.txt`（异构 TP）
- 调用 V2 的 `train_prefill_models()`，再训练含 `tp_a/tp_f` 的 Decode_iter 模型

### `ENERGY_MODELS_README.md`

- 重写为两套自包含数据集的说明与训练命令

---

## 三、Git 跟踪配置

### `benchmark/AFlex_bench/.gitignore`

- 保留 `**/retrain/data/*.txt`（忽略根目录散落 TXT）
- 子目录 `v1_layer_profile/`、`v2_pipeline_profile/` 下的 TXT 不受该规则匹配，可正常跟踪
- 添加注释标明两个受管数据集路径

---

## 四、新增分析脚本

路径：`06_others/more_model/energy_model/analysis/`

| 文件 | 说明 |
|------|------|
| `compare_dense_vs_moe_prefill.py` | Dense (Llama3.1-8B) vs MoE (Qwen3-30B-A3B) Prefill 延迟/能耗/频率敏感性对比图 |
| `plot_moe_prefill_characteristics.py` | MoE Prefill 特性绘图；Dense 基准数据指向 `v1_layer_profile/prefill_data_v1.txt` |
| `plot_moe_decode_characteristics.py` | MoE Decode 特性绘图；Dense 基准数据指向 `v2_pipeline_profile/decode_pipeline_v1.txt` |

---

## 五、拟合命令

```bash
cd benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain

# V1 — 仅用 v1_layer_profile/
python energy_model_v1.py --data-dir data/v1_layer_profile --output-dir models_v1

# V2 — 仅用 v2_pipeline_profile/
python energy_model_v2.py --data-dir data/v2_pipeline_profile --output-dir models_v2

# V3 — 仅用 v2_pipeline_profile/（decode 用 merged）
python energy_model_v3.py --data-dir data/v2_pipeline_profile --output-dir models_v3
```

---

## 六、建议 Commit Message

```
Organize energy model training data into two self-contained datasets.

Split retrain profile data into v1_layer_profile (layer-level prefill/decode)
and v2_pipeline_profile (prefill + pipeline decode including hetero TP).
Track both datasets in git; update V1/V2/V3 training scripts to fit from a
single --data-dir without cross-folder references. Add Dense vs MoE analysis
scripts with updated data paths.
```

---

## 七、本次暂存文件清单

```
M  benchmark/AFlex_bench/.gitignore
M  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/ENERGY_MODELS_README.md
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v1_layer_profile/README.md
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v1_layer_profile/decode_data_v1.txt
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v1_layer_profile/prefill_data_v1.txt
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v2_pipeline_profile/README.md
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v2_pipeline_profile/decode_pipeline_hetero_2a4f.txt
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v2_pipeline_profile/decode_pipeline_merged.txt
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v2_pipeline_profile/decode_pipeline_tp2.txt
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v2_pipeline_profile/decode_pipeline_v1.txt
A  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/data/v2_pipeline_profile/prefill_data_v1.txt
M  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/energy_model_v1.py
M  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/energy_model_v2.py
M  benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/energy_model_v3.py
A  benchmark/AFlex_bench/06_others/more_model/energy_model/analysis/compare_dense_vs_moe_prefill.py
A  benchmark/AFlex_bench/06_others/more_model/energy_model/analysis/plot_moe_decode_characteristics.py
A  benchmark/AFlex_bench/06_others/more_model/energy_model/analysis/plot_moe_prefill_characteristics.py
```

**注意：** 工作区中另有未暂存修改（`scheduler.py`、`af_dvfs_controller.py` 等）和未跟踪文件，**不在本次 commit 范围内**。
