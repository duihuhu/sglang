# Energy Models & Training Data

两套**自包含**训练数据，拟合时只使用对应文件夹内的文件，不交叉引用。

## 目录结构

```
retrain/
├── data/
│   ├── v1_layer_profile/          # V1 完整数据集
│   │   ├── prefill_data_v1.txt
│   │   └── decode_data_v1.txt
│   └── v2_pipeline_profile/       # V2/V3 完整数据集
│       ├── prefill_data_v1.txt
│       ├── decode_pipeline_v1.txt
│       ├── decode_pipeline_tp2.txt
│       ├── decode_pipeline_hetero_2a4f.txt
│       └── decode_pipeline_merged.txt
│
├── models_v1/                     # V1 拟合产出
├── models_v2/                     # V2 拟合产出（当前 benchmark 默认）
├── models_v3/                     # V3 拟合产出（异构 TP decode）
│
├── energy_model_v1.py
├── energy_model_v2.py
└── energy_model_v3.py
```

## 数据与模型对应

| 数据集文件夹 | 内容 | 训练脚本 | 产出 |
|-------------|------|---------|------|
| `v1_layer_profile/` | Prefill 逐层 + Decode 逐层 | `energy_model_v1.py` | Prefill/Decode 独立层 LUT/LR/GBDT |
| `v2_pipeline_profile/` | Prefill 逐层 + Decode pipeline（含异构） | `energy_model_v2.py` / `energy_model_v3.py` | Prefill 逐层 + Decode_iter GBDT/LUT |

## 使用

```bash
# V1（仅 v1_layer_profile/）
python energy_model_v1.py --data-dir data/v1_layer_profile --output-dir models_v1

# V2（仅 v2_pipeline_profile/，decode 默认 decode_pipeline_v1.txt）
python energy_model_v2.py --data-dir data/v2_pipeline_profile --output-dir models_v2

# V3（仅 v2_pipeline_profile/，decode 默认 decode_pipeline_merged.txt）
python energy_model_v3.py --data-dir data/v2_pipeline_profile --output-dir models_v3
```

## 注意

- 所有数据基于 **Qwen3-32B (dense)** 模型采集
- V2 模型是当前 benchmark 默认版本（`ENERGY_MODEL_DIR` → `models_v2`）
- V3 decode 特征含 `tp_a, tp_f`；运行时需要 `af_profile_predictor.py` 支持异构 TP
