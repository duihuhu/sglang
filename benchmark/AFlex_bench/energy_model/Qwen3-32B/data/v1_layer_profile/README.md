# V1 完整能耗模型数据

**仅使用本文件夹内数据**即可拟合 V1 能耗模型（`energy_model_v1.py`），无需引用其他目录。

模型：**Qwen3-32B (dense)**

| 文件 | 说明 |
|------|------|
| `prefill_data_v1.txt` | Prefill 逐层 profile：tp × input_len × gpu_clock × batch_size |
| `decode_data_v1.txt` | Decode 逐层 profile：tp × input_len × output_len × gpu_clock × batch_size |

拟合产出：Prefill/Decode 独立层模型（LUT + LinearReg + GBDT），特征为同构 TP、A/F 共用单一 `gpu_clock`。

```bash
python energy_model_v1.py --data-dir data/v1_layer_profile --output-dir models_v1
```
