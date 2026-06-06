# Decode AF Profiling 进展

## 当前正在运行的测试（2026-05-29 05:11）

### Decode (freq_a, freq_f, bs) Sweep

**脚本**: `run_decode_af_sweep.py`  
**PID**: 672318  
**日志**: `/tmp/decode_af_sweep.log`  
**输出**: `data_decode_af/decode_af_sweep.tsv`

**命令**:
```bash
cd /workspace/sglang-tier/benchmark/energy_bench/profiling
/workspace/env/sglang-tier/bin/python run_decode_af_sweep.py \
    --freq 450 690 930 1170 1410 \
    --bs 1 2 4 8 16 32 48 64 96 128 \
    --il 32 --ol 512
```

**配置**:
- 频率组合: 5×5 = 25 种 (freq_a, freq_f)
- Batch sizes: 1, 2, 4, 8, 16, 32, 48, 64, 96, 128
- Input len: 32（短，加速 prefill）
- Output len: 512（长，确保 decode 有足够 steady-state iterations）
- 总配置数: 250
- 预计时间: 30-50 分钟

**输出格式**:
```
tp  freq_a  freq_f  batch_size  input_len  output_len  TPOT_ms  total_iter_ms  pipeline_ms  drain_ms  A_energy_per_iter_mj  F_energy_per_iter_mj  peak_running_req  n_steady_iters
```

**字段说明**:
- `TPOT_ms`: 用户感知的 per-token 延迟 = e2e_latency / output_len（训练 y 值）
- `total_iter_ms`: 单次 decode iteration 时间 = pipeline + drain
- `pipeline_ms`: CPU 异步 dispatch 时间（~34ms 固定）
- `drain_ms`: GPU 真实计算 + 同步时间（随 freq 变化）
- `A_energy_per_iter_mj`: DA 卡（Attention）每次 iteration 的能耗
- `F_energy_per_iter_mj`: DF 卡（FFN）每次 iteration 的能耗
- `peak_running_req`: 测量期间 decode batch 中的最大并发请求数
- `n_steady_iters`: 用于统计的 steady-state iteration 数量

---

## 测试完成后的下一步

### 1. 训练新 Predictor

用 `data_decode_af/decode_af_sweep.tsv` 训练：
- **X 特征**: `(tp, freq_a, freq_f, batch_size)`
- **y 目标（延迟）**: `TPOT_ms`
- **y 目标（能耗）**: `A_energy_per_iter_mj + F_energy_per_iter_mj`
- **模型**: LUT 优先 + GBDT fallback

Prefill 阶段直接用原始数据（`benchmark/test_motivation/hucc/paper/prefill_data_v1.txt`），因为 Prefill 没有 micro-batch pipeline。

### 2. 修改 Predictor 接口

当前接口：
```python
predictor.predict_latency(phase, "A"/"F", tp, freq, bs, il, ol)  # 分别预测 A 和 F
```

新接口：
```python
predictor.predict_tpot(phase, tp, freq_a, freq_f, bs, il, ol)    # 直接预测 TPOT
predictor.predict_energy(phase, tp, freq_a, freq_f, bs, il, ol)  # 直接预测总能耗
```

### 3. 修改 Tier2 DVFS Controller

`af_dvfs_controller.py` 的 `select_freq_decode()`:
```python
# 旧逻辑：
lat_A = predictor.predict_latency("decode", "A", tp, f_a, bs, il, ol)
lat_F = predictor.predict_latency("decode", "F", tp, f_f, bs, il, ol)
t_iter = max(lat_A, lat_F) * num_layers + t_drain

# 新逻辑：
tpot = predictor.predict_tpot("decode", tp, f_a, f_f, bs, il, ol)
if tpot > slo_tpot_ms:
    continue  # 不满足 SLO
```

### 4. 验证

重新跑 Tier2 策略对比测试，验证新 predictor 的调频效果。

---

## 关键发现

1. **Decode 的 `pipeline`（~34ms）是 CPU dispatch 时间，不是 GPU 执行时间**
2. **`drain` 才是 GPU 真实计算时间**，随 freq 变化（450MHz: 56ms → 1410MHz: 10ms）
3. **`total_iter_ms` 不是 TPOT**——它是单次 forward 时间，不包含排队延迟
4. **TPOT 随 bs 显著增长**（bs=1: 50ms → bs=128: 200ms），因为排队效应
5. **A 和 F 在 pipeline 上互相影响**，不能独立建模，需要测 36 种 (freq_a, freq_f) 组合
6. **Prefill 不需要重测**——没有 micro-batch pipeline，原始数据可直接使用

---

## 文件结构

```
benchmark/energy_bench/profiling/
├── run_decode_af_sweep.py    # 当前测试脚本
├── data_decode_af/           # 输出数据
│   └── decode_af_sweep.tsv   # 测试结果
└── logs/                     # 服务器日志（da.log, df.log, pa.log, pf.log, router.log）
```
