# Energy Models & Training Data

统一存放能耗预测模型和训练数据。

## 目录结构

```
retrain/
├── data/                          # 训练数据
│   ├── prefill_data_v1.txt        # Prefill 阶段 profile (Qwen3-32B, TP=1/2/4/8)
│   ├── decode_data_v1.txt         # Decode 阶段逐层 profile (单频率, TP=1/2/4/8)
│   ├── decode_pipeline_v1.txt     # Decode pipeline profile (TP=1, M=1/2)
│   ├── decode_pipeline_tp2.txt    # Decode pipeline profile (TP=2, M=1/2)
│   ├── decode_pipeline_hetero_2a4f.txt  # Decode pipeline 异构 (tp_a=2, tp_f=4)
│   └── decode_pipeline_merged.txt # 合并版 (tp_a/tp_f=1/1, 2/2, 2/4), V3 训练用
│
├── models_v1/                     # V1 模型 (原始版本)
│   ├── Prefill_{A,F}_{LUT,LinearReg}.pkl
│   ├── Prefill_{A,F}_lat_{LUT,LinearReg}.pkl
│   ├── Decode_{A,F}_{LUT,LinearReg}.pkl
│   ├── Decode_{A,F}_lat_{LUT,LinearReg}.pkl
│   └── Decode_iter_{lat,energy_A,energy_F}_GBDT.pkl  (TP=1 only)
│
├── models_v2/                     # V2 模型 (当前生产使用)
│   ├── (同 V1 结构, Prefill + Decode 独立层模型)
│   └── Decode_iter_{lat,energy_A,energy_F}_GBDT.pkl  (TP=1 only)
│
├── models_v3/                     # V3 模型 (支持异构 TP)
│   └── Decode_iter_{lat,energy_A,energy_F}_{GBDT,LUT}.pkl
│       特征: (tp_a, tp_f, M, f_A, f_F, input_len, batch_size)
│       数据: TP组合 1×1, 2×2, 2×4
│
├── energy_model_v1.py             # V1 训练脚本
├── energy_model_v2.py             # V2 训练脚本 (decode_pipeline_v1.txt)
└── energy_model_v3.py             # V3 训练脚本 (decode_pipeline_merged.txt)
```

## 模型版本差异

| 版本 | Decode 特征 | TP 支持 | 训练数据 |
|------|------------|---------|---------|
| V1 | (tp, gpu_clock, bs, il) | 1/2/4/8 同构 | decode_data_v1.txt |
| V2 | (tp, M, f_A, f_F, il, bs) | 1 only | decode_pipeline_v1.txt |
| V3 | (tp_a, tp_f, M, f_A, f_F, il, bs) | 1×1, 2×2, 2×4 | decode_pipeline_merged.txt |

## 使用

```bash
# 重训 V2
python energy_model_v2.py --data data/decode_pipeline_v1.txt --output-dir models_v2

# 重训 V3
python energy_model_v3.py --data data/decode_pipeline_merged.txt --output-dir models_v3
```

## 注意

- 所有数据基于 **Qwen3-32B (dense)** 模型采集，对 MoE 模型预测精度有偏差
- V2 模型是当前 benchmark 默认使用的版本 (`ENERGY_MODEL_DIR` 指向 `models_v2`)
- V3 模型需要修改 `af_profile_predictor.py` 支持 `tp_a/tp_f` 双参数后才能在运行时使用
