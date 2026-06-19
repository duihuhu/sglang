# V2/V3 完整能耗模型数据

**仅使用本文件夹内数据**即可拟合 V2/V3 能耗模型，无需引用 `v1_layer_profile/` 或其他目录。

模型：**Qwen3-32B (dense)**

## Prefill（与 V1 相同的逐层 profile）

| 文件 | 说明 |
|------|------|
| `prefill_data_v1.txt` | Prefill 逐层 profile，用于 Prefill_A/F 能量与延迟模型 |

## Decode（pipeline 级耦合 profile，含异构）

| 文件 | 说明 |
|------|------|
| `decode_pipeline_v1.txt` | TP=1，M=1/2，独立 f_A/f_F → **V2 默认 decode 训练数据** |
| `decode_pipeline_tp2.txt` | TP=2 同构 |
| `decode_pipeline_hetero_2a4f.txt` | 异构 tp_a=2, tp_f=4 |
| `decode_pipeline_merged.txt` | 合并 (1×1, 2×2, 2×4) → **V3 默认 decode 训练数据** |

## 拟合

```bash
# V2：Prefill 逐层 + Decode pipeline (TP=1)
python energy_model_v2.py --data-dir data/v2_pipeline_profile --output-dir models_v2

# V3：Prefill 逐层 + Decode pipeline 异构合并
python energy_model_v3.py --data-dir data/v2_pipeline_profile --output-dir models_v3
```
