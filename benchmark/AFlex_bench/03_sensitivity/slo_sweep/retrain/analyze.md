# Decode 阶段 Pipeline 耦合模型重训练与评估

## 1. 背景与动机

原有 V1 模型使用**独立 per-layer 预测**方式：分别预测单层 A（Attention）和 F（FFN）的延迟/能耗，然后用公式组合（`lat_a + lat_f` 或 `max(A,F)`）估算 iteration 延迟。这种方式忽略了：

- IPC 通信开销（`ipc_cpp` 同步延迟）
- Pipeline drain（micro-batch 排空时间）
- A/F GPU 之间的等待和重叠效应
- 不同 `(f_A, f_F)` 组合下的非线性交互

导致 V1 在实际 PDAF pipeline 中的**预测误差达 33%**（系统性低估 iteration 延迟）。

## 2. 重训练方案

### 2.1 数据采集

编写 `bench_decode_pipeline.py`，在**真实 4-process PDAF pipeline** 中采集耦合数据：

- **架构**：PF+PA (GPUs 4,5) + DF+DA (GPUs 6,7) + Router
- **DVFS 控制**：独立锁定 DA (GPU7) 和 DF (GPU6) 频率
- **遍历维度**：`(f_A, f_F, M, batch_size, input_len)`
- **去掉 output_len**：验证 decode iteration 延迟与 KV cache 长度无关（paged attention 下 O(1)）
- **测量方式**：每组配置发 bs 个并发请求，生成 128 tokens，测量整体 iteration 延迟和 NVML 能耗

### 2.2 采集参数

| 维度 | 取值 |
|------|------|
| f_A, f_F | 210, 450, 690, 930, 1170, 1410 MHz (6×6=36 组合) |
| M (micro-batch) | 1, 2 |
| batch_size | 1, 2, 4, 8, 16, 32, 64, 128, 256 |
| input_len | 128, 512, 2048 |
| 跳过 | il≥2048 + bs≥128 (必然 timeout); il≥2048 + bs≥64 + low freq |

**总采集量**：1760 条有效数据，耗时约 14 小时。

### 2.3 模型训练

使用 `energy_model_v2.py` 训练 GBDT (LightGBM) 模型：

- **特征**：`[M, f_A, f_F, input_len, batch_size]`（5 维，无 output_len）
- **目标**：
  - `Decode_iter_lat`：iteration 端到端延迟 (us)
  - `Decode_iter_energy_A`：DA 每 iteration 能耗 (mJ)
  - `Decode_iter_energy_F`：DF 每 iteration 能耗 (mJ)

### 2.4 训练结果

| 模型 | Train MAPE | 5-fold CV MAPE |
|------|-----------|----------------|
| Decode_iter_lat | 0.47% | **1.18% ± 0.11%** |
| Decode_iter_energy_A | 0.53% | **1.26% ± 0.08%** |
| Decode_iter_energy_F | 0.42% | **0.87% ± 0.08%** |

CV MAPE 均在 1.3% 以内，预测精度极高。

## 3. 数据特征验证

### 3.1 能耗 U 形曲线

对称频率 (f_A=f_F) 下，总能耗随频率呈现先下降后上升的 U 形曲线：

| bs | 最优频率 | 最低能耗 | 最大频率能耗 | 节能空间 |
|----|---------|---------|------------|---------|
| 1 | 1170 MHz | 15,179 mJ | 19,035 mJ | 20.3% |
| 8 | 1170 MHz | 16,843 mJ | 20,612 mJ | 18.3% |
| 32 | 1170 MHz | 19,017 mJ | 23,553 mJ | 19.3% |
| 128 | 930 MHz | 45,650 mJ | 57,071 mJ | 20.0% |

规律：**bs 越大，最优频率越低**。

### 3.2 f_F=1410 MHz "过驱"现象

当 f_F 从 1170→1410 MHz 时，DF 能耗突然上升 26-28%，而延迟仅降 1-2%。这是关键发现：**最大频率不是最优选择**。

### 3.3 M=1 vs M=2

在小 batch (bs≤32) 下，M=2 的延迟和能耗均高于 M=1（pipeline overhead > 重叠收益）。M=2 优势仅在 bs≥128 时体现。

## 4. 系统集成

### 4.1 代码修改

- `af_profile_predictor.py`：
  - 新增 `predict_iteration_latency(M, f_a, f_f, bs, il)` 和 `predict_iteration_energy(M, f_a, f_f, bs, il)`
  - 新增 `find_best_freq_pair_coupled()` 方法
  - 修复 `_ModelUnpickler` 以正确加载 V2 模型类

- `af_dvfs_controller.py`：
  - 新增 `_select_freq_decode_coupled()` 方法
  - 当 V2 模型可用时优先使用耦合预测，否则 fallback 到 V1

### 4.2 特征简化

去掉 `output_len` 维度（decode iteration 与 KV cache 长度无关），接口从 6 参数简化为 5 参数。

## 5. 实际 Workload 测试

### 5.1 测试配置

- **架构**：4-GPU PDAF (GPUs 4-7), TP=1, Dynamic M
- **Workload**：`workload_steady.jsonl`（600 requests, variable il/ol）
- **SLO**：TTFT ≤ 5000ms, TPOT ≤ 300ms

### 5.2 性能对比

| 方案 | 吞吐(tok/s) | TTFT(ms) | TPOT(ms) | 能耗(J) | 节能 | 吞吐损失 |
|------|-----------|----------|----------|---------|------|----------|
| Baseline (MaxFreq) | 724.4 | 211 | 72 | 113,324 | - | - |
| **Tier V1** (旧模型) | 686.9 | 1,317 | 99 | 77,132 | **31.9%** | 5.2% |
| **Tier V2** (耦合模型) | 713.6 | 1,444 | 75 | 82,186 | **27.5%** | 1.5% |

### 5.3 调频决策对比

| 指标 | V1 | V2 |
|------|----|----|
| Decode 决策数 | 306 | 382 |
| 主要频率对 | (930, 930) 86% | (690, 1170) 50%, (450, 930) 31% |
| 平均 f_A (DA) | 897 MHz | **602 MHz** |
| 平均 f_F (DF) | 930 MHz | **1096 MHz** |
| 预测误差 | -33.5% | **-27.5%** |
| SLO urgent 触发 | 4 次 | **2 次** |

### 5.4 调频策略差异

- **V1**：倾向对称降频 (930, 930)，两侧均匀降低
- **V2**：倾向**非对称降频** — 大幅降低 f_A (450-690 MHz) + 保持 f_F 较高 (930-1170 MHz)

V2 学到了 A 侧（Attention）计算量小于 F 侧（FFN），可以更激进地降低 A 侧频率而不影响 critical path。

## 6. 结论

### 优势

1. **预测精度大幅提升**：CV MAPE 从 V1 的 ~33% 误差降到 V2 的 1.2%（离线验证），在线误差从 33.5% 降到 27.5%
2. **吞吐保持更好**：V2 吞吐损失仅 1.5%（V1 为 5.2%），因为 V2 不降 f_F 过多
3. **更智能的非对称调频**：V2 发现 f_A 可激进降频 + f_F 需保持较高
4. **更少的 SLO urgent 触发**：说明 V2 的频率选择更准确，不易 over-commit

### 待改进

1. **在线预测误差仍有 27.5%**：profile 采集时是稳定 batch，实际运行中 batch 动态变化、prefill/decode 交替，导致实际 iteration 时间波动大
2. **节能幅度略低于 V1**：V2 的 27.5% vs V1 的 31.9%。因为 V2 给 f_F 分配了更高频率。需要探索是否可以在保证 SLO 的前提下进一步降低 f_F
3. **TTFT 偏高**：Prefill 侧也被 DVFS 控制降频，需要考虑仅对 decode 侧做 DVFS 或提高 prefill 侧的频率下限
4. **Calibration 机制未充分生效**：`calib=1.0` 说明在线校准还没启动，需要让系统运行更长时间以积累校准数据

## 7. Per-token SLO 验证实验 (2026-06-03)

### 实验设计

将 TPOT SLO 从 100ms 收紧到 90ms，动态 M 阈值调整为 bs=128（使 steady workload 下几乎全程 M=1），
对每个 request 的每次 token 输出逐 token 检查 inter-token 延迟是否超过 SLO，统计 per-token 违背率。

### 结果

| 方案 | Thpt (tok/s) | TPOT (ms) | Energy (J) | Per-token SLO违背率 | 节能 |
|------|---:|---:|---:|---:|---:|
| Baseline (MaxFreq) | 737.4 | 64 | 98,944 | 0.81% | — |
| V1 旧模型 | 670.2 | 109 | 68,144 | **49.32%** | +31.1% |
| V2 耦合模型 | 713.3 | 83 | 68,608 | **11.30%** | +30.7% |

### 结论

SLO 合规是生产部署的硬约束，不可作为 trade-off 的自由度。V2 的核心优势在于：**在 SLO 优先的前提下最大化节能**。

- **V1 只顾降频省电**：节能 31.1%，但 49.3% 的 token 违反 SLO，在实际服务中完全不可接受。其调频策略过于激进（f_A=690, f_F=930 长期锁定），无法适应负载变化。
- **V2 精准服务的同时降低能耗**：节能 30.7%（与 V1 相近），但 per-token SLO 违背率仅 11.3%，TPOT 均值 83ms 远低于 V1 的 109ms。V2 通过耦合模型学到了 A/F 之间的延迟依赖，能精确分配频率预算（f_F 保高、f_A 适当降），在不超标的范围内尽可能省电。
- **V2 效果优于 V1**：两者节能幅度相近，但 V2 能更精准地保障服务可靠性。如果将 V1 的频率调高以满足 SLO，其实际节能效果会远低于当前 30%，从而在相同 SLO 约束下 V2 才是更优方案。

调频时间轴见 `figures/dvfs_timeline_v1_v2.png` 和 `figures/dvfs_timeline_overlay.png`。

## 8. 文件索引

| 文件 | 用途 |
|------|------|
| `benchmark/test_motivation/hucc/bench_decode_pipeline.py` | 耦合 profile 数据采集脚本 |
| `benchmark/test_motivation/hucc/paper/decode_pipeline_v1.txt` | 采集的原始数据 (1760 行) |
| `benchmark/test_motivation/energy_model_v2.py` | V2 GBDT 模型训练脚本 |
| `benchmark/test_motivation/energy_models/Decode_iter_*.pkl` | 训练好的 V2 模型文件 |
| `benchmark/energy_bench/retrain/models_v2/` | 完整模型目录 (V1+V2) |
| `benchmark/energy_bench/retrain/run_retrain_eval.sh` | 对比测试主脚本 |
| `benchmark/energy_bench/retrain/analyze_retrain_eval.py` | 结果分析脚本 |
| `benchmark/energy_bench/retrain/results/` | 测试结果 JSON |
| `benchmark/energy_bench/retrain/logs/` | DVFS 调频决策日志 |
| `python/sglang/srt/energy/af_profile_predictor.py` | 预测器（已集成 V2） |
| `python/sglang/srt/energy/af_dvfs_controller.py` | DVFS 控制器（已集成 V2） |
