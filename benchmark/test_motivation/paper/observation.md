# Characterization: Energy-Performance Trade-offs in Disaggregated LLM Inference

## Overview

We conduct a systematic characterization of the energy-performance trade-offs in disaggregated LLM inference, where each transformer layer is split into an Attention operator (A) and an FFN operator (F) that can be independently frequency-scaled. Our profiling covers **7,530 Decode** and **906 Prefill** configurations on NVIDIA A800-80GB SXM GPUs, sweeping across 6 GPU frequencies (210–1410 MHz), 4 tensor parallelism degrees (TP=1/2/4/8), up to 9 batch sizes (BS=1–256), 6 input lengths (IL=128–4096 for Decode, 10 values up to 40000 for Prefill), and 7 output lengths (OL=64–4096 for Decode). We report five observations that progressively build the case for operator-level, phase-aware, runtime-adaptive frequency control.

**逻辑链条：**
```
Obs 1 (现象)  →  Obs 2 (调制)  →  Obs 3 (后果)  →  Obs 4 (收益)  →  Obs 5 (复杂性)
  A/F 异构       参数空间全貌      能耗最优频率错位    量化 saving       需要 runtime-adaptive
```

**数据来源：**
- `decode_data_v1.txt`：7,530 条 Decode 记录，字段包括 tp, input_len, output_len, gpu_clock, batch_size, A (latency), F (latency), TPOT_ms, (A+F)*64_ms, A_energy_mj, F_energy_mj
  - il: {128,256,512,1024,2048,4096}, ol: {64,128,256,512,1024,2048,4096}, bs: {1..256} (tp=1 最大 128)
- `prefill_data_v1.txt`：906 条 Prefill 记录，字段包括 tp, input_len, gpu_clock, batch_size, A, F, TTFT_ms, (A+F)*64_ms, A_energy_mj, F_energy_mj
  - il: {128,256,512,1024,2048,4096,8192,16384,32000,40000} (tp=1 无 40000), bs: {1..128}
- 模型：Qwen3-32B，硬件：NVIDIA A800-80GB SXM

---

## Obs 1: A and F exhibit distinct frequency sensitivity

**论点：** Within a single inference step, the Attention operator (A) and the FFN operator (F) respond fundamentally differently to GPU frequency scaling. This intra-step heterogeneity is the foundational observation that enables operator-level energy optimization.

**图：Fig. 1 — Normalized Latency vs GPU Frequency**

| 属性 | 说明 |
|------|------|
| 布局 | 2 个子图：(a) Decode, (b) Prefill |
| 线条 | 每子图 4 条：A-tp=1, A-tp=4, F-tp=1, F-tp=4 |
| X 轴 | GPU frequency (210–1410 MHz) |
| Y 轴 | Normalized latency（相对 210 MHz 归一化） |
| 固定配置 | Decode: il=1024, ol=64, bs=16；Prefill: il=1024, bs=1 |

**数据处理方式：** 对每条线，取该配置在各频率下的延迟值，除以 210 MHz 下的延迟进行归一化。ratio 接近 1.0 表示频率不敏感（memory-bound），ratio 远小于 1.0 表示频率敏感（compute-bound）。

**关键发现：**
- Decode 子图中，A-tp=4 几乎水平（ratio ≈ 1.0），表明 A 完全 memory-bound——提高频率不能加速 A，但会增加功耗。F 线陡降至 0.15–0.17，表明 F 始终 compute-bound。
- Prefill 子图中，A 和 F 都陡降（都 compute-bound），异构性较弱。
- 这种差异源于 A 和 F 的计算特征不同：A 的 KV cache 访问是 memory-bound 的，而 F 的矩阵乘法是 compute-bound 的。

**⋆ Takeaway:** A and F have fundamentally different frequency-performance profiles within the same inference step. A unified frequency applied to both operators is inherently suboptimal—it either over-provisions compute for A (wasting energy) or under-provisions for F (degrading performance).

**Motivates:** Per-operator frequency selection（为 A 和 F 分别选择频率）。

---

## Obs 2: The heterogeneity is modulated by serving parameters

**论点：** The A/F sensitivity gap is not fixed—it is strongly modulated by tensor parallelism (TP), batch size (BS), and input length (IL). This means the optimal frequency pair (fA, fF) shifts across the serving parameter space.

**图：Fig. 2 — Frequency Sensitivity Ratio Heatmap**

| 属性 | 说明 |
|------|------|
| 布局 | 2×2 heatmap：上排 Decode (a) A ratio, (b) F ratio；下排 Prefill (c) A ratio, (d) F ratio |
| X 轴 | TP = {1, 2, 4, 8} |
| Y 轴 | Batch Size = {1, 4, 16, 64, 128, 256} |
| 颜色 | ratio = lat@1410 / lat@210，红色 = 敏感（compute-bound），蓝色 = 不敏感（memory-bound） |
| 固定配置 | Decode: il=1024, ol=64；Prefill: il=128 |

**数据处理方式：** 对每个 (TP, BS) 组合，计算 `ratio = latency@1410MHz / latency@210MHz`。ratio 越小表示频率越敏感（compute-bound），越接近 1.0 表示越不敏感（memory-bound）。使用 RdBu_r colormap，红色 = 敏感，蓝色 = 不敏感。

**关键发现：**
- **(a) Decode A ratio：** 颜色变化剧烈。右下角（tp=8, bs=1）深蓝（ratio ≈ 1.0，完全 memory-bound），左上角（tp=1, bs=128）深红（ratio ≈ 0.17，完全 compute-bound）。TP 是最强驱动因子：TP↑ → A 更 memory-bound；BS↑ → A 变回 compute-bound。
- **(b) Decode F ratio：** 全红（ratio 0.15–0.19），F 在所有配置下始终 compute-bound，不受 TP/BS 调制。
- **(c) Prefill A ratio：** 大部分红色，只有右下角（tp=8, bs=1）变蓝。Prefill 中 A 的异构性只在高 TP + 小 batch 的窄窗口内出现。
- **(d) Prefill F ratio：** 全红，F 始终 compute-bound。

**⋆ Takeaway:** The A/F heterogeneity is not a binary property but a continuous spectrum modulated by serving parameters. A's sensitivity spans the full range from fully memory-bound to fully compute-bound, while F remains consistently compute-bound. Any frequency selection strategy must account for this parameter-dependent behavior.

**Motivates:** Parameter-aware frequency selection（频率选择必须感知当前的 TP、BS 等运行时参数）。

**与 Obs 1 的关系：** Obs 1 用一个代表配置建立直觉（A/F 不同），Obs 2 将其扩展到全参数空间，展示这种差异的普遍性和参数依赖性。

---

## Obs 3: The heterogeneity leads to misaligned energy-optimal frequencies

**论点：** Because A and F have different frequency-performance profiles, their energy-optimal frequencies are different. A unified frequency is always a compromise—it cannot simultaneously minimize energy for both operators.

**图：Fig. 3 — Energy vs GPU Frequency**

| 属性 | 说明 |
|------|------|
| 布局 | 2 个子图：(a) tp=8, il=128, ol=64, bs=16；(b) tp=2, il=2048, ol=1024, bs=4 |
| 线条 | 每子图 2 条：A energy (mJ), F energy (mJ) |
| X 轴 | GPU frequency (210–1410 MHz) |
| Y 轴 | Energy per step (mJ) |
| 标注 | 箭头标出 A opt 和 F opt 的最优频率点 |

**数据处理方式：** 直接读取每个频率下的 A_energy_mj 和 F_energy_mj，绘制 energy vs frequency 曲线。标注 argmin 点。

**关键发现：**
- **(a) tp=8：** A energy 曲线相对平坦（178→250 mJ），最优频率 fA*=210 MHz（最低频）。F energy 呈 U 型（353→221→296 mJ），最优频率 fF*=930 MHz。最优频率错位 720 MHz。
  - 物理解释：A 是 memory-bound，降频不增加延迟但降低功率 → energy 单调递增 → 最低频最省电。F 是 compute-bound，降频增加延迟，功率×时间的乘积呈 U 型 → 中间频率最省电。
- **(b) tp=2：** A energy 也呈 U 型（89→69→94 mJ），fA*=450 MHz。F energy 呈 U 型，fF*=1170 MHz。最优频率仍然错位 720 MHz。
  - 关键信息：即使 A 不是完全 memory-bound（U 型而非单调），A 和 F 的最优频率仍然不同。A 降频不能无脑选最低——过低反而费电。

**⋆ Takeaway:** The energy-optimal frequency for A and F are misaligned across all operating regimes. A unified frequency forces a compromise: either waste energy on A (frequency too high) or on F (frequency too low). Disaggregated frequency control can eliminate this compromise.

**Motivates:** Disaggregated frequency scaling（A 和 F 分别设置频率，而非统一频率）。

**与 Obs 1/2 的关系：** Obs 1/2 建立了 A/F 的 bound 属性差异（延迟维度），Obs 3 推导出这种差异在能耗维度的后果——最优频率错位。这是从 "现象" 到 "可利用的机会" 的关键跳跃。

---

## Obs 4: Disaggregated frequency scaling yields significant energy savings

**论点：** Quantifying the energy savings from disaggregated frequency scaling (AFlex) over unified frequency scaling, and characterizing the end-to-end impact across different serving scenarios.

**图：Fig. 4 — Energy Saving and End-to-End Breakdown**

| 属性 | 说明 |
|------|------|
| 布局 | 3 个子图：(a) Decode saving vs SLO, (b) Prefill saving vs SLO, (c) Energy breakdown |
| (a)(b) X 轴 | SLO Multiplier = {1.0, 1.05, 1.1, 1.2, 1.5, 2.0} |
| (a)(b) Y 轴 | Max Energy Saving (%) |
| (a)(b) 线条 | 4 条线：tp=1/2/4/8 |
| (c) 类型 | Stacked bar chart，4 个场景 |
| (c) 颜色 | 青绿 = Decode，珊瑚红 = Prefill |

**数据处理方式：**
- **(a)(b) Saving 计算：** 对每个配置 (tp, il, ol, bs)，定义 SLO = max(A_lat, F_lat) @ 1410MHz × multiplier。在 SLO 约束下，分别搜索：(1) 最优统一频率 f*（min energy s.t. max(A_lat@f, F_lat@f) ≤ SLO），(2) 最优频率对 (fA*, fF*)（min A_energy@fA + F_energy@fF s.t. max(A_lat@fA, F_lat@fF) ≤ SLO）。Saving = (unified - AFlex) / unified × 100%。展示甜点位（max saving across all configs per TP）。
- **(c) Energy breakdown：** 对每个场景，Prefill energy = A_energy + F_energy（1 步），Decode energy = (A_energy + F_energy) × output_len。

**关键发现：**
- **(a) Decode：** SLO×1.0（无松弛）时 max saving 达 25–28%（tp=2/4/8）。SLO 一旦松弛（×1.05+），saving 骤降至 ~9%。原因：严格 SLO 下 unified 被迫选高频，而 AFlex 可以让 memory-bound 的 A 降频而不影响延迟；SLO 松弛后 unified 也能降频，优势消失。
- **(b) Prefill：** Saving 整体较小（max ~14% for tp=8），因为 Prefill 中 A/F 异构性弱（Obs 2）。
- **(c) Energy breakdown：** Chatbot（il=128, ol=1024）Decode 占 99.9%；RAG（il=2048, ol=64）Prefill 占 15.7%；Summary（il=4096, ol=64）Prefill 占 26.4%。长 prompt 场景下 Prefill 能耗不可忽略。

**⋆ Takeaway:** Disaggregated frequency scaling achieves up to 28% energy savings in Decode under strict SLO constraints. While Prefill savings are smaller per-step, Prefill energy constitutes 15–26% of end-to-end energy in RAG and summarization workloads, making it non-negligible.

**Motivates:** SLO-aware energy optimization（在 SLO 约束下最大化节能）+ Phase-aware optimization（Prefill 和 Decode 分别优化）。

**与 Obs 3 的关系：** Obs 3 展示了 "为什么能省"（energy 曲线形状不同），Obs 4 回答 "能省多少"（量化 saving）。Obs 3 是定性的，Obs 4 是定量的。

---

## Obs 5: The optimal A/F configuration shifts with operating regime, requiring runtime-adaptive control

**论点：** The optimal frequency pair and the required A/F GPU ratio both shift dynamically with serving parameters and load intensity. No single static configuration is universally optimal—runtime-adaptive control is necessary.

**图：Fig. 5 — Runtime Variability**

| 属性 | 说明 |
|------|------|
| 布局 | 2 个子图：(a) F/A ratio box plot, (b) F/A ratio vs load intensity |
| (a) 类型 | Box plot，X 轴 = TP，每个 TP 两个 box（Prefill 红色，Decode 青色） |
| (a) Y 轴 | F/A Latency Ratio |
| (a) 参考线 | F=A (ratio=1.0) |
| (b) 类型 | 折线图，X 轴 = Batch Size（负载强度代理） |
| (b) Y 轴 | F/A Latency Ratio |
| (b) 线条 | 每个 TP 两条线：实线 = Prefill，虚线 = Decode |

**数据处理方式：**
- **(a) Box plot：** 对每个 TP，收集所有配置（遍历 il, ol, bs）在 1410 MHz 下的 F/A latency ratio = F_lat / A_lat。Prefill 和 Decode 分别画 box。
- **(b) 折线图：** 固定 il=1024，对每个 (TP, BS) 组合，计算 Prefill 和 Decode 的 F/A ratio @ 1410 MHz。F/A ratio 直接决定了 disaggregated serving 中 A:F 的 GPU 配比需求：ratio=3 意味着需要 3 个 F GPU per 1 个 A GPU 来平衡流水线。

**关键发现：**
- **(a) Box plot：**
  - Prefill（红色）：几乎全在 F=A 线上方（ratio 1.7–5.3），F 是瓶颈，需要更多 F GPU。但 tp=4/8 时有 outlier 伸到线下方 → 少数配置 A > F。
  - Decode（青色）：tp=1 全在线上方（F > A），tp≥2 大部分在线下方（A > F）→ TP 驱动了瓶颈方向的反转。
  - 关键信息：**两个阶段内部都存在 A>F 和 F>A 的情况**，取决于配置。
- **(b) 折线图：**
  - Prefill 实线全在上方（ratio 2.4–4.9），且随 BS 增大 ratio 上升（tp=8: 2.4→4.9）。
  - Decode 虚线全在下方（ratio 0.6–0.9），相对稳定。
  - 关键信息：**随着负载强度变化，所需的 A:F GPU 配比在动态变化**。低负载时 Prefill 需要 ~2.4 个 F per A，高负载时需要 ~4.9 个 F per A。

**⋆ Takeaway:** The bottleneck direction (A vs F) and the required GPU ratio both shift across phases (Prefill vs Decode), across TP configurations, and across load intensities. A static A/F allocation is suboptimal under varying workloads. Runtime-adaptive control that dynamically adjusts both frequency and resource allocation is essential.

**Motivates:** Runtime-adaptive control（运行时动态调整频率和资源配比）+ Dynamic A/F resource provisioning（动态 GPU 配比）。

**与 Obs 1–4 的关系：** Obs 1–4 证明了 disaggregated frequency scaling 能省电，Obs 5 回答 "能不能用一个固定策略搞定"——答案是不能。这是从 observation 过渡到 system design 的桥梁。

---

## Summary: From Observations to Design Requirements

| Observation | Key Finding | Design Requirement |
|-------------|-------------|-------------------|
| Obs 1 | A/F 频率敏感性不同 | Per-operator frequency selection |
| Obs 2 | 异构性受 TP/BS/IL 调制 | Parameter-aware frequency selection |
| Obs 3 | Energy-optimal frequency 错位 | Disaggregated frequency scaling |
| Obs 4 | 甜点位 saving 25–28%，Prefill 占比 15–26% | SLO-aware + Phase-aware optimization |
| Obs 5 | 最优配置随负载动态变化 | Runtime-adaptive control |

These five observations collectively motivate a system that performs **per-operator, phase-aware, SLO-constrained, runtime-adaptive GPU frequency scaling** for energy-efficient LLM inference serving.
