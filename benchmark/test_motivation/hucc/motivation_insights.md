# AF 分离 Motivation 数据分析报告

## 数据概况

| 维度 | Prefill | Decode |
|------|---------|--------|
| **列** | tp, input_len, gpu_clock, batch_size, P_Attention, P_FFN | tp, input_len, output_len, gpu_clock, batch_size, D_Attention, D_FFN |
| **TP** | 1, 2, 4, 8 | 1, 2, 4, 8 |
| **Input Length** | 128, 512, 4096 | 128, 512, 4096 |
| **Output Length** | — | 64, 256, 512 |
| **GPU Clock (MHz)** | 210, 540, 870, 1200 | 210, 540, 870, 1200 |
| **Batch Size** | 1, 16, 256 (tp=8) | 1, 16, 256 (tp=8) |

---

## 角度 0（核心论证）：为什么单纯 PD 分离不够？为什么静态 AF 配比不行？

### 0.1 为什么 PD 分离不够 — 频率困境（分 P/D 分析）

#### 总体原理

![PD Frequency Dilemma](figures/fig7_1_pd_only_frequency_dilemma.png)

PD 分离后，每个 P/D 节点内 A 和 F 仍在同一 GPU 上顺序执行，共享同一频率。但 A 和 F 的频率响应截然不同 → 形成"频率困境"。

#### 按 Phase × Workload 量化困境严重程度

![PD Dilemma by Zone](figures/fig7_4_pd_dilemma_by_zone.png)

**Prefill 的频率困境**（左图）：

| 配置 | PF 效率 | PA 效率 | 效率差 (gap) | 含义 |
|------|---------|---------|-------------|------|
| in=128, bs=1 | ~26% | ~16% | ~10% | 两者都访存密集，困境较轻 |
| in=128, bs=256 | ~44% | ~31% | ~13% | 中等差距 |
| in=4096, bs=1 | ~38% | ~25% | ~13% | PF 更计算密集 |
| in=4096, bs=256 | ~53% | ~42% | ~11% | 大 workload 两者趋同 |

- **PF 的频率效率始终高于 PA** → PD 模式下为了满足 PF 性能需求必须高频，但 PA 在高频下浪费更多
- Gap 在各配置间相对稳定（10-13%）→ **Prefill 的频率困境是普遍性的，不依赖特定 workload**
- AF 分离后：PF GPU 拉满频率（高 ROI），PA GPU 可独立降频

**Decode 的频率困境**（右图）— **关键区分：困境严重度取决于 batch size**：

| Zone | 条件 | DA 效率 | DF 效率 | 效率差 | 分析 |
|------|------|---------|---------|-------|------|
| **Zone 1** | 小 batch (bs=1) | ~24% | ~27% | **~3%** | 两者都访存密集，差距极小 → **PD-only 在此区间足够** |
| **Zone 2** | 中 batch (bs=16) | ~25% | ~29% | **~4-8%** | DF 开始转计算密集 → **困境开始显现** |
| **Zone 3** | 大 batch (bs=256) | ~30-42% | ~35-44% | **~3-8%** | 两者都转计算密集 → 困境程度中等 |

> **Prefill vs Decode 的关键区别**：
> - **Prefill**：频率困境在几乎所有 workload 下都存在且稳定（PF 始终比 PA 更计算密集）
> - **Decode**：频率困境**主要集中在 Zone 2（中等 batch）**，这恰好是 continuous batching 的典型工作区间
> - Decode 小 batch (Zone 1) 时 DA 和 DF 行为类似，PD-only 的频率损失很小 → AF 分离的频率收益主要来自 Zone 2

#### PD-only 的本质局限

PD 分离只解决了 Prefill vs Decode 的资源隔离，但忽略了**每个阶段内部 A 和 F 的特征差异**：
- **Prefill 内部**：PF（计算密集）和 PA（混合型）频率需求天然不同 → 同频是浪费
- **Decode 内部**：在实际 serving 的 Zone 2 工作区间，DF 和 DA 频率需求出现分化 → 同频开始低效

### 0.2 为什么静态 AF 配比不行 — 分 P/D 分析

#### 流水线利用率分布

![Pipeline Utilization](figures/fig7_2_static_af_pipeline_util.png)

AF 分离后 A 和 F 形成流水线，利用率 = min(time_A, time_F) / max(time_A, time_F)。

#### 最优 A/(A+F) 比例在各 workload 间的变化

![Static AF Optimal Fraction](figures/fig7_5_static_af_optimal_fraction.png)

**Prefill 的静态 AF 问题**（上图仅展示 tp=8 数据，低估了问题严重性）：

**全 TP 范围数据**：在 96 个 Prefill 配置中，有 **24 个（25%）** PA > PF，Attention 成为瓶颈！

| TP | PA > PF 占比 | 跨越条件 |
|----|-------------|---------|
| tp=1 | **58% (7/12)** | 所有短序列 + 中序列高频 |
| tp=2 | **38% (9/24)** | 小 batch 短/中序列 + bs=16 短序列高频 |
| tp=4 | **33% (8/24)** | 类似 tp=2 模式 |
| tp=8 | **0% (0/36)** | 从不跨越（通信开销人为抬高 PF） |

- tp=8 不跨越的原因是 **TP 通信开销膨胀了 PF**（tp=1 时 PF=721μs，tp=8 时 PF=2826μs），而非 FFN 天然更慢
- **SLA 加剧跨越**：SLA 越紧 → 需要更高频率 → PF 缩短更多 → PA/PF 上升 → 更容易跨越 1.0 → Attention 在 SLA 压力下更易成为瓶颈
- 极端案例：tp=1, input=128, 1200MHz → PA/PF = **4.16**（Attention 耗时是 FFN 的 4 倍）
- **因此 Prefill 也存在瓶颈反转**，"偏 F"的静态方案在低 TP、短序列、高频场景下方向完全错误

**Decode 的静态 AF 问题**（下图）— **比 Prefill 严重得多**：

- A/(A+F) 最优比例范围约 **0.37~0.60**
- **比例跨越了 0.5 的平衡线！** → 发生了瓶颈反转
  - 短 KV + 小 batch：DA/DF < 1，FFN 是瓶颈 → 应该给 F 更多资源
  - 长 KV + 大 batch：DA/DF > 1，Attention 是瓶颈 → 应该给 A 更多资源
- **任何固定的 A:F 比例都会在一半 workload 上做出相反的错误分配**
- 更严重的是：在同一请求的生成过程中，KV cache 不断增长 → **比例在单请求内就会漂移甚至穿越反转点**

> **Prefill vs Decode 静态 AF 问题的比较**：
> - **Prefill**：在 tp=8 时问题较轻（FFN 通常是瓶颈），但 **跨 TP、频率、序列长度后，25% 的配置发生瓶颈反转**（PA > PF），尤其在 SLA 压力（高频）下更严重
> - **Decode**：瓶颈反转更频繁且由 batch_size/KV 长度驱动（实际服务中持续漂移）
> - **两个阶段都需要动态 AF 调度**，只是驱动因素不同：Prefill 由 TP/频率/序列长度驱动，Decode 由 batch size/KV cache 长度驱动

### 0.3 三种方案的量化对比

![Three-way Comparison](figures/fig7_3_three_way_comparison.png)

在 **相同延迟预算** 下比较三种方案的能量消耗：

| 方案 | 说明 | 能效 |
|------|------|------|
| **PD-only** | A 和 F 同频 1200MHz | 100%（基准） |
| **AF-static** | A 固定 870MHz, F 固定 1200MHz | 省 5~15% 能量 |
| **AF-dynamic** | 按 workload 为 A/F 各自选择最优频率 | **省 10~30% 能量** |

**分 P/D 的关键观察**：
- **Prefill**：AF-static 已有显著改善（PF 始终需要高频，PA 降到 870MHz 几乎无损）→ 即使是"粗糙的"AF 分离也有价值
- **Decode**：AF-static 的改善不稳定——在某些 workload 下效果好，另一些 workload 下 AF-static 选择的固定频率反而不合适 → **动态调频的增量收益在 Decode 中更显著**
- AF-dynamic 在两个阶段都一致优于 AF-static → **动态感知 workload 是必要的**

**SLA 视角（与 Fig 8 同一故事线）**：在**给定延迟上限（SLA）**下，若 **Attention 与 FFN 不能独立设频**（无 AF 分离、二者绑在同一频率域），为满足 SLA 往往只能**整体提高频率**——**所有子算子**一起被拉到高档，**非瓶颈那一段**也被迫高频运转，能耗上是**连带抬升**，属于结构性浪费。  
**AF 分离**后，为满足同一 SLA，只需**提高真正卡住延迟的那条路径**的频率（只动 A 或只动 F），**不必**把整条流水上「已经够快」的子阶段一并拉高。Fig 8 在**归一化的延迟–能耗平面**上对比：无 AF 时只有「整条统一提频」的**少数工作点与陡峭轨迹**（Fig 8.1）；有 AF 时在 **(f_A, f_F)** 网格上得到**一大片可达集**，其 **Pareto 前沿**相对统一轨迹更靠近**左下角**（Fig 8.2）——即**相同延迟预算下可付更少能耗代理**，或**相同能耗下可压更低延迟**。若系统连 **PD 也未分离**，则往往只剩**更粗的一档全局频率**，上述浪费会被进一步放大。

### Fig 8.1：Prefill / Decode 各一子图 — **无 AF 分离**，给定 SLA 只能统一提频 → 能效差

![PD unified energy vs freq](figures/fig8_1_pd_unified_energy_vs_freq.png)

**坐标**（与 Fig 8.2 一致）：横轴 **L / L₀**（L₀ = 该 workload 在统一 210MHz 下的总延迟 PA+PF 或 DA+DF），纵轴 **E / E₀**（E₀ = f·L 能耗代理基线）。**左下**＝更快且更省。

**画法**：每个 workload（6 种颜色）一条**折线**，连接统一频率 **210→540→870→1200 MHz** 的 **4 个工作点**（散点颜色表频率）。图中**橙色竖虚线** = SLA 阈值（基于 75th percentile of L(870)/L(210)），其**左侧浅色区域**为「满足 SLA 的可行域」；每个 workload 在可行域内**能耗最低的点**用 **★** 标出，右下角文本框列出对应**最低需频率与 E/E₀**。

**解读（核心论点：统一提频 → 结构性浪费）**：
- **仅 4 个工作点**——要满足 SLA 只能向折线更高频端跳：**A 与 F 同时被拉到高档**，即使非瓶颈段（如 memory-bound 的 Attention）提频收益很小，也无法单独保持低频。
- ★ 通常落在 **870 或 1200 MHz**，对应**较高 E/E₀**——说明统一 DVFS 在 SLA 约束下**为延迟付出过多能耗**。

### Fig 8.2：Prefill / Decode 各一子图 — **AF 分离**后 **(f_A,f_F)** 搜索空间扩大，Pareto 更省能耗

![AF decoupled energy vs freq](figures/fig8_2_af_decoupled_energy_vs_freq.png)

**设定**：L = T_A(f_A)+T_F(f_F)，E = (f_A·T_A + f_F·T_F)/10⁶；**f_A, f_F** 各自在 {210,540,870,1200} 上独立取值 → 每个 workload **4²=16 个组合**。归一化分母与 Fig 8.1 **完全相同**（仍用统一 210MHz 基线），便于一对一比较。

**画法**：
- **浅色散点**：16 个组合的 (L/L₀, E/E₀)；
- **同色实线**：这些点的 **Pareto 前沿**（L 与 E 都尽量小时的非支配集）；
- **同色淡虚线**：Fig 8.1 的**统一频率轨迹**，原样叠入作参照；
- 同一 **SLA 竖线**；在可行域内的 AF **最低能耗点** 标 ★，**箭头**从统一最优点指向 AF 最优点，注释标出 **能耗降幅** 和 **可行点数变化**（如 `2→11 feas.pts`）。

**与 Fig 8.1 的对比叙事**：
1. **搜索空间扩大**：统一频率只有 4 个可行点；AF 分离后在同一 SLA 下可行点数增加到 5-12，选择自由度大幅提高。
2. **Pareto 前沿更靠左下**：在**同一 SLA（同一竖线）**上，AF 可行集中的最低 E/E₀ 常常低于统一最优点——对应叙述即「**不必同时拉高所有算子频率，只需提高瓶颈那一路**」。
3. 箭头直观量化 energy saving；在 4 档粗粒度频率下已可见 **1-12%** 的节省，更细粒度硬件频率阶梯下该收益会进一步放大。

---

## 角度 1：Roofline — A/F 子阶段算子特征天然不同

### Fig 1.1：DVFS 加速比对比

![DVFS Speedup](figures/fig1_1_dvfs_speedup_bar.png)

**核心发现**：FFN 的 DVFS 加速比在所有配置下均显著高于 Attention，证明：
- **FFN 是计算密集型算子**：频率提升直接转化为性能提升
- **Attention 是访存密集型算子**：频率提升的收益受限于内存带宽

| 阶段 | 210→1200 MHz 加速比 | 属性 | 备注 |
|------|---------------------|------|------|
| PF | ~2.2x (tp=8) / ~5.5x (tp=1) | 计算密集 | 全配置下均为计算密集 |
| PA | ~1.4x (tp=8,bs=1) / ~3.1x (tp=1 或大batch) | **混合型（随配置变化）** | 长序列+低TP/大batch 时偏计算，短序列+高TP 时偏访存 |
| DF | ~1.5x | 访存/计算混合 | 大 batch 时转向计算密集 |
| DA | ~1.4x | 访存密集 | 几乎全配置下均为访存密集（仅大batch+长KV时偏计算） |

> **PA vs DA 的关键区别**：Prefill Attention 对全序列做 Q·K^T 计算（O(n²d) FLOPs），长序列时有显著计算量；Decode Attention 每次仅用 1 个 token 的 query 查询 KV cache，计算量极小，几乎始终是访存密集型。因此 PA 的计算/访存特征是随 workload **动态变化**的，而 DA 基本固定为访存密集。

### Fig 1.3：PA 特征漂移 — PA 随配置在计算/访存之间切换

![PA Characteristic Drift](figures/fig1_3_pa_characteristic_drift.png)

**核心发现**：
- **左图 Prefill**：PA 的 DVFS 加速比从 ~1.4x（tp=8,bs=1，访存密集）到 ~3.1x（tp=1,bs=1，偏计算密集）变化剧烈；而 PF 在所有配置下都保持高加速比（计算密集）
- **右图 Decode**：DA 的加速比在所有配置下都保持低水平（~1.3x~1.5x），始终是访存密集型
- **PA vs DA 的本质区别**：PA 对全序列做 O(n²d) 的 Q·K^T 计算，长序列/低TP/大batch 时有足够的计算量饱和 GPU；DA 每步仅用 1 个 token 查询 KV cache，计算量始终不足
- 这意味着 PA 的 DVFS 策略不能"一刀切"，需要根据当前 workload 动态选择频率

### Fig 1.2：归一化执行时间 vs GPU 频率

![Normalized Time](figures/fig1_2_normalized_time_vs_freq.png)

**核心发现**：
- PF 的曲线几乎与理想线性缩放重合（计算密集型的标志）
- PA/DA 的曲线远离理想线，频率提升的效果大幅衰减
- 这直接反映了 Roofline 模型中两类算子位于不同的区域

---

## 角度 2：端到端瓶颈切换 — PA/PF 与 DA/DF 之间切换

### Fig 2.1：Attention 占比 vs Input Length

![Attn Fraction vs InputLen](figures/fig2_1_attn_fraction_vs_inputlen.png)

**核心发现**：
- **Prefill**：随着 input_len 增大，Attention 占比下降（FFN 以 O(n) 增长，占比逐渐主导）
- **Decode**：随着 input_len 增大（KV cache 变长），Attention 占比上升
- 在不同 input_len 下，瓶颈从一个子阶段切换到另一个

### Fig 2.2：Attention 占比 vs Batch Size

![Attn Fraction vs BS](figures/fig2_2_attn_fraction_vs_batchsize.png)

**核心发现**：
- **Prefill**：batch 增大 → FFN 占比上升（GEMM 效率提升，计算量 ∝ batch）
- **Decode**：
  - 短 KV (input=128)：batch 增大 → FFN 仍占主导
  - **长 KV (input=4096)：batch 增大 → Attention 占比突破 50%，成为瓶颈**
- 这是"瓶颈反转"现象：大 batch + 长 KV cache 下，Decode Attention 反超 FFN

### Fig 2.3：Attention 占比 vs GPU 频率

![Attn Fraction vs Freq](figures/fig2_3_attn_fraction_vs_freq.png)

**核心发现**：
- 频率升高 → FFN 加速更多 → Attention 的相对占比被动上升
- 这意味着**提高 GPU 频率本身就会改变瓶颈位置**
- 在高频条件下，Attention 更容易成为瓶颈

### Fig 2.4：四阶段时间分解

![Stacked Bar](figures/fig2_4_stacked_bar_breakdown.png)

**核心发现**：
- 在不同 workload 配置下，PA/PF 和 DA/DF 的比例完全不同
- 短序列小 batch：A ≈ F（接近均衡）
- 长序列大 batch：Prefill 中 FFN 占绝对主导；Decode 中 Attention 成为瓶颈
- **没有一个固定的 A/F 比例能适用于所有场景**

---

## 角度 3：动态配比 — A/F 相对需求随 workload/时间漂移

### Fig 3.1/3.2：A/F Ratio 热力图

![Heatmap](figures/fig3_1_3_2_af_ratio_heatmap.png)

**核心发现**：
- **Prefill PA/PF Ratio**：从 0.27（大 batch 短 seq）到 0.74（小 batch 长 seq）
- **Decode DA/DF Ratio**：从 0.83（小 batch 短 seq）到 **1.47**（大 batch 长 seq）
- Decode 中比值跨越了 1.0 的平衡线 → 瓶颈实际发生了反转
- batch_size 和 input_len 的组合效应造成了巨大的 ratio 变化范围

### Fig 3.3：A/F Ratio 箱线图

![Boxplot](figures/fig3_3_af_ratio_boxplot.png)

**核心发现**：
- **Prefill PA/PF Ratio**：min ≈ 0.24, max ≈ 1.33，范围 ~1.09
- **Decode DA/DF Ratio**：min ≈ 0.46, max ≈ 1.47，范围 ~1.01
- 两个阶段的 A/F 比都有超过 1.0 的变化幅度
- **这种巨大的变化范围使得任何静态 A/F 资源分配方案都不可能最优**

### Fig 3.4：变量敏感度分析

![Sensitivity](figures/fig3_4_sensitivity_tornado.png)

**核心发现**：
- **Prefill**：batch_size 对 A/F 比的影响最大，其次是 input_len，gpu_clock 影响最小
- **Decode**：batch_size 同样是最大影响因子
- 在实际 LLM 服务中，batch_size 随请求到达率动态变化 → A/F 比持续漂移

---

## 角度 4：四阶段 DVFS / 功耗敏感度差异

### Fig 4.1：绝对执行时间 vs GPU 频率

![Time vs Freq](figures/fig4_1_time_vs_freq_absolute.png)

**核心发现**：
- PF 的时间-频率曲线最陡（收益最大）
- DA 的曲线最平（收益最小）
- 四阶段的曲线形状各不相同 → 统一 DVFS 策略必然不是最优的

### Fig 4.2：边际加速比

![Marginal Speedup](figures/fig4_2_marginal_speedup.png)

**核心发现**：
- **PF 在每个频率区间都保持 >20% 的边际加速** → 值得全程拉满频率
- **DA 在 870→1200 区间边际加速趋近于零** → 超过 870MHz 的频率对 DA 几乎无效
- 这种差异化的边际收益是 per-stage DVFS 的直接动机

---

## 深入 Insight A：频率缩放效率

### Fig 5.1：缩放效率对比

![Scaling Efficiency](figures/fig5_1_scaling_efficiency.png)

**定义**：缩放效率 = 实际加速比 / 理论线性加速比

| 阶段 | @1200MHz 缩放效率 (input=4096) | 含义 |
|------|-------------------------------|------|
| **PF** | **>95%** | 几乎完美的计算密集型，频率投资 ROI 极高 |
| **PA** | ~55% | 近一半的频率提升被浪费在等待内存上 |
| **DF** | ~30-50% | 中间状态，取决于 batch size |
| **DA** | ~24-40% | 访存瓶颈最严重，频率投资 ROI 最低 |

**结论**：对不同阶段统一调频意味着 PA/DA 上浪费了大量功耗预算。

---

## 深入 Insight B：频率膝点（Knee Point）

### Fig 5.2：DA 的频率膝点分析

![Knee Point](figures/fig5_2_freq_kneepoint.png)

**核心发现**：
- DA (bs=1, in=128) 在 870→1200MHz 区间边际加速仅 **0.3%** — 频率膝点为 ~870MHz
- DA (bs=256, in=4096) 在同区间仍有 **~10%** 加速 — 大 batch 下特征改变
- 频率膝点取决于 workload 配置，不是固定的

**设计启示**：
- AF 分离后，Attention 节点在小 batch 时可将频率上限设为 870MHz
- 节省 ~27% 频率（和更多的功耗），而性能损失 < 1%

---

## 深入 Insight C/D：DVFS 再平衡的适用边界

### Fig 5.3：不同 workload 规模下 A/F Ratio 随频率变化

![AF Ratio vs Freq Workloads](figures/fig5_3_af_ratio_vs_freq_workloads.png)

**核心发现**：

| workload 规模 | PA/PF Ratio 变化幅度 (210→1200) | DVFS 再平衡效果 |
|---------------|-------------------------------|-----------------|
| 小 (in=128, bs=1) | 0.72 → 0.93 (+29%) | ✅ 有效 |
| 中 (in=128, bs=16) | 0.47 → 1.16 (+147%) | ✅ 非常有效 |
| 大 (in=4096, bs=256) | **0.31 → 0.31 (+0%)** | ❌ 完全无效 |

**关键结论**：
- **小-中 workload**：DVFS 可以有效调整 A/F 平衡（低延迟、零开销的再平衡手段）
- **大 workload**：A/F ratio 对频率完全不敏感 → DVFS 无法再平衡，必须通过 GPU 数量/资源配比调整
- **动态调度系统需要同时具备两种再平衡机制**

---

## 深入 Insight E：Decode 瓶颈反转临界点

从 Fig 3.1/3.2 的热力图和 Fig 2.2 可以推导：

**Decode DA/DF Ratio（tp=8, bs=256, clock=870）**：

| input_len | DA/DF Ratio | 瓶颈 |
|-----------|-------------|------|
| 128 | 0.77~0.83 | FFN |
| 512 | 0.83~0.85 | FFN |
| 4096 | **1.34~1.47** | **Attention** |

在 input_len ≈ **1500-2000** 处存在 DA=DF 的临界反转点。

**动态影响**：
- 在实际服务中，随着请求持续生成 token → KV cache 不断增长
- 系统会**不可避免地穿越临界反转点**
- 瓶颈从 FFN → Attention 实时切换
- 需要动态感知这一切换并重新分配资源

---

## 深入 Insight F：能效分析

### Fig 5.5：能量代理 (Time × Freq)

![Energy Proxy](figures/fig5_5_energy_proxy.png)

**定义**：Energy Proxy = Time × Frequency（越低表示越节能）

| 阶段 | 能量代理趋势 | 能效最优频率 | 策略 |
|------|-------------|-------------|------|
| **PA** | 随频率单调递增 | **最低频率 (210MHz)** | 性能目标允许时尽量降频 |
| **PF** | 几乎恒定 | **任意频率均可** | 直接拉满追求性能 |
| **DA (小batch)** | 随频率单调递增（陡峭） | **最低频率** | 超过膝点的频率纯浪费 |
| **DF (小batch)** | 随频率递增 | **较低频率** | 降频节能效果好 |

**结论**：各阶段的能效最优频率天然不同。AF 分离后可以为每个子阶段独立设置频率，实现全局最优能效。

---

## 深入 Insight G：Batch 改变算子特征（"特征漂移"）

### Fig 5.6：按 Batch Size 分区的计算/访存特征

![Zone Classification](figures/fig5_6_zone_classification.png)

**核心发现**：

| Batch Size | DA 特征 | DF 特征 | A/F 差异 | AF 分离收益 |
|------------|---------|---------|----------|------------|
| bs=1 | 访存密集 | 访存密集 | 小 | 低 |
| bs=16 | 访存密集 | **计算密集** | **大** | **高** |
| bs=256 | 转计算密集 | 计算密集 | 中 | 中 |

**三区间划分**：

- **Zone 1 (低收益)**：小 batch — A/F 都访存密集，行为相似
- **Zone 2 (高收益)**：中 batch — A 访存密集 vs F 计算密集，差异最大，AF 分离收益最高
- **Zone 3 (中收益)**：大 batch — 都计算密集但比例不同

**实际 LLM 服务大部分时间处于 Zone 2**（continuous batching 的典型 batch 大小），这正是 AF 分离收益最大的区域。

---

## 深入 Insight H：再平衡的分层策略 — DVFS 优先，资源调整兜底

### 再平衡手段的适用边界

AF 分离后 A/F 形成流水线，流水线不平衡时需要再平衡。有两种手段：

| 手段 | 优势 | 劣势 | 适用场景 |
|------|------|------|---------|
| **per-stage DVFS** | 零开销、实时、动态 | 对大 workload / 访存瓶颈无效 | 中小 workload（A/F 比可被频率改变） |
| **资源配比调整** | 可处理 DVFS 无法解决的场景 | 开销大、粒度粗 | 大 workload（A/F 比恒定）或 DA 访存瓶颈 |

**DVFS 有效区间**（来自 Fig 5.3）：
- 中小 workload (in=128, bs=16)：PA/PF ratio 可从 0.47 调到 1.16 → ✅ **同构 TP + DVFS 足以满足 SLA**
- 大 workload (in=4096, bs=256)：PA/PF ratio 恒定 0.31 → ❌ **DVFS 完全失效，需要资源层面调整**

**DVFS 无法解决的典型场景**：
- Decode 大 batch + 长 KV cache：DA 成为瓶颈 (DA/DF=1.47)，DA 是访存密集型 → 调频对 DA 几乎无效（膝点 870MHz 后仅 0.3% 加速）
- 此时**只有增加 Attention 节点的资源（更高带宽、更多 GPU）才能缓解**

### Fig 5.7：TP 缩放效率对比 — 资源调整时的决策依据

![TP Scaling](figures/fig5_7_tp_scaling.png)

**TP 缩放效率数据的意义不是"必须异构 TP"，而是回答：当确实需要资源调整时，给哪一侧加 GPU 性价比更高？**

| 阶段 | TP=1→8 加速比 (bs=1) | 每 GPU 效率 | 谁缩放更好 |
|------|---------------------|------------|-----------|
| PA | **1.48x** | 18.5% | ← PA > PF |
| PF | 1.42x | 17.8% | |
| DA | 1.25x | 15.6% | |
| DF | **1.53x** | **19.1%** | ← DF > DA |

Prefill 和 Decode 的 TP 缩放特征相反：
- **Decode**：DF (19.1%) > DA (15.6%)，FFN 更受益于多 GPU
- **Prefill**：PA (18.5%) ≥ PF (17.8%)，PF 在高 TP 低 batch 时甚至退化

**资源调整决策表**（仅在 DVFS 无法再平衡时启用）：

| 场景 | 瓶颈 | 调整方向 |
|------|------|---------|
| Decode 大 batch + 长 KV | DA（访存密集，DVFS 无效） | 增加 Attention 资源 |
| Prefill 大 batch + 长 seq | PF（计算密集，但 DVFS 可调） | 通常 DVFS 足够，极端情况增加 FFN 资源 |

> **对系统设计的核心结论**：动态调度系统应以 **per-stage DVFS 为主要的、实时的再平衡机制**（覆盖大部分场景），以**资源配比调整为兜底策略**（处理 DVFS 失效的极端场景）。两者结合实现全 workload 范围的 A/F 平衡。

---

## 补充：Output Length 的影响

### Fig 6.1：不同 Batch Size 下 Output Length 对 DA/DF 的影响

![Output Length](figures/fig6_1_output_len_impact.png)

**核心发现**：
- **bs=1, bs=16**：output_len 对 DA/DF 几乎无影响（变化 < 3%）
- **bs=256**：DA 随 output_len 增长而增长（最高 ~17.5%），DF 保持不变
- 原因：output_len 增大 → KV cache 增长 → 大 batch 下 Attention 的访存量放大

**对 AF 分离的意义**：在大 batch 场景下，随着生成进行，DA/DF ratio **在同一请求的服务过程中持续漂移**——不仅跨请求需要动态调度，单请求内也需要。

---

## Insight 总结表

| # | Insight | 对应图 | 设计决策 |
|---|---------|--------|---------|
| 基础 | A/F 计算/访存特征天然不同 | Fig 1.1, 1.2 | AF 分离的基本前提 |
| 基础 | 瓶颈在 A/F 间切换 | Fig 2.1~2.4 | 需要动态调度 |
| 基础 | A/F 比例随 workload 漂移 | Fig 3.1~3.4 | 不能静态拆分 |
| 基础 | 四阶段 DVFS 响应不同 | Fig 4.1, 4.2 | per-stage DVFS |
| A | PF 缩放效率 95%+ vs PA 仅 55% | Fig 5.1 | PF 值得最高频，PA 可降频 |
| B | DA 存在频率膝点（870MHz 后 0.3% 加速） | Fig 5.2 | DA 可大幅降频几乎无损 |
| C | Batch 改变算子的计算/访存属性 | Fig 5.6 | 调度策略需感知 batch 大小 |
| D | DVFS 再平衡仅对中小 workload 有效 | Fig 5.3 | 需要 DVFS + 资源配比两种手段 |
| E | Decode 存在瓶颈反转临界点 | Fig 3.1/3.2, 2.2 | 需要实时监测并切换策略 |
| F | 各阶段能效最优频率天然不同 | Fig 5.5 | per-stage DVFS 参数设定 |
| G | DVFS 优先 + 资源调整兜底：DVFS 适用于中小 workload，大 workload 或访存瓶颈时需资源层调整 | Fig 5.3, 5.7 | 分层再平衡策略设计 |
| H | output_len 在大 batch 下使 DA 漂移 | Fig 6.1 | 单请求内也需动态调整 |

---

## 论证链条

```
角度 0：为什么需要 AF 分离？（核心论证，分 P/D 分析）
├── PD-only 频率困境
│   ├── Prefill：PF/PA 效率差 ~10-13%，普遍存在
│   └── Decode：Zone 2（中 batch）时困境最显著，Zone 1 时 PD-only 尚可
├── 静态 AF 失败
│   ├── Prefill：跨全 TP/频率/序列长度后 25% 配置 PA>PF（低TP+短seq+高频），也存在瓶颈反转
│   └── Decode：比例跨越 0.5（瓶颈反转），由 batch/KV 长度驱动，更频繁更动态
└── 三方案对比：AF-dynamic 在能效上显著优于 PD-only 和 AF-static（尤其 Decode）

角度 1：Roofline — A/F 天然特征差异
├── FFN 计算密集（高频率缩放效率）
├── Attention 访存密集 / 混合型（低频率效率 + 随 workload 漂移）
└── 提供 AF 分离的理论基础

角度 2：瓶颈切换
├── 随 input_len, batch_size, gpu_clock 变化，瓶颈在 A/F 间切换
└── 证明需要动态感知和调整

角度 3：动态配比
├── A/F 比从 0.24 到 1.47，跨越平衡点
├── batch_size 是最大影响因子（实际服务中动态变化）
└── 证明静态配比不可行，需要动态调度

角度 4：DVFS 调频策略
├── 各阶段 DVFS 响应不同 → per-stage DVFS 有理论基础
├── DA 存在频率膝点 → 可降频省电
├── DVFS 适用边界：中小 workload 有效，大 workload 需资源调整
└── 分层策略：DVFS 优先 + 资源调整兜底
```

---

## 文件结构

```
benchmark/test_motivation/
├── prefill_data.txt          # 原始 Prefill 数据
├── decode_data.txt           # 原始 Decode 数据
├── plot_motivation.py        # 绘图脚本
├── motivation_insights.md    # 本文档
└── figures/
    ├── fig7_1_pd_only_frequency_dilemma.png
    ├── fig7_2_static_af_pipeline_util.png
    ├── fig7_3_three_way_comparison.png
    ├── fig7_4_pd_dilemma_by_zone.png
    ├── fig7_5_static_af_optimal_fraction.png
    ├── fig8_1_pd_unified_energy_vs_freq.png
    ├── fig8_2_af_decoupled_energy_vs_freq.png
    ├── fig1_1_dvfs_speedup_bar.png
    ├── fig1_3_pa_characteristic_drift.png
    ├── fig1_2_normalized_time_vs_freq.png
    ├── fig2_1_attn_fraction_vs_inputlen.png
    ├── fig2_2_attn_fraction_vs_batchsize.png
    ├── fig2_3_attn_fraction_vs_freq.png
    ├── fig2_4_stacked_bar_breakdown.png
    ├── fig3_1_3_2_af_ratio_heatmap.png
    ├── fig3_3_af_ratio_boxplot.png
    ├── fig3_4_sensitivity_tornado.png
    ├── fig4_1_time_vs_freq_absolute.png
    ├── fig4_2_marginal_speedup.png
    ├── fig5_1_scaling_efficiency.png
    ├── fig5_2_freq_kneepoint.png
    ├── fig5_3_af_ratio_vs_freq_workloads.png
    ├── fig5_5_energy_proxy.png
    ├── fig5_6_zone_classification.png
    ├── fig5_7_tp_scaling.png
    └── fig6_1_output_len_impact.png
```
