# Dense vs MoE Prefill 特征对比分析

## 实验设置

| 项目 | Dense 模型 | MoE 模型 |
|------|-----------|----------|
| 模型 | Llama3.1-8B | Qwen3-30B-A3B |
| 参数量 | 8B (全激活) | 30B (激活 3B) |
| 架构 | 标准 FFN | 128 experts, top-8, 48 layers |
| 硬件 | A800 单卡 (TP=1) | A800 单卡 (TP=1) |
| 频率范围 | 210-1410 MHz | 210-1410 MHz |
| Batch Size | 1-128 | 1-32 |
| Input Length | 128-32000 | 128-16384 |
| 数据量 | 546 rows | 198 rows |

**注意**：两个模型参数量差异大（8B vs 30B），但 MoE 模型每次推理只激活 ~3B 参数（top-8 out of 128 experts），因此实际计算量远小于 30B dense 模型。

## 核心发现

### 1. MoE 的绝对延迟和能耗远低于 Dense

在 TP=1、freq=1410MHz 条件下（33 个重叠数据点）：

| 指标 | MoE/Dense 比值 (avg) | MoE/Dense 比值 (median) | 范围 |
|------|---------------------|------------------------|------|
| A (Attention) 延迟 | 0.39x | 0.32x | [0.26, 0.94] |
| F (FFN) 延迟 | 0.22x | 0.17x | [0.15, 0.84] |
| A 能耗 | 0.30x | 0.28x | [0.23, 0.54] |
| F 能耗 | 0.20x | 0.18x | [0.14, 0.57] |

**解释**：MoE 模型虽然总参数 30B，但每个 token 只路由到 8/128 个 expert，实际 FFN 计算量约为 `8/128 × FFN_size`，远小于 8B dense 模型的完整 FFN。Attention 部分差距较小（因为 MoE 的 attention 层大小与模型 hidden_size 相关，而 Qwen3-30B 的 hidden_size=2048 接近 Llama-8B 的 4096 的一半）。

### 2. F/A 计算比例差异显著

| 模型 | F/A 延迟比 (avg) | F/A 延迟比 (range) |
|------|-----------------|-------------------|
| Dense | 3.10x | [1.52, 3.69] |
| MoE | 1.75x | [0.56, 2.30] |

**关键洞察**：
- Dense 模型 FFN 占比 ~76%（F/A ≈ 3.1），是典型的 FFN-dominant 负载
- MoE 模型 FFN 占比 ~64%（F/A ≈ 1.75），A 和 F 更加平衡
- 这意味着 MoE 模型在 AF 分离部署时，A 和 F 节点的负载更均衡，**更适合 AF 对称部署**（而 Dense 模型需要给 F 分配更多资源）

![F/A Ratio](fig3_fa_ratio.png)

### 3. MoE 的 DVFS 节能空间更大

| 模型 | 平均节能 | 中位节能 | 最大节能 |
|------|---------|---------|---------|
| Dense | 16.1% | 15.8% | 21.3% |
| MoE | 26.8% | 25.6% | 42.7% |

**解释**：MoE 模型有更大的 DVFS 节能空间，原因：
1. **稀疏激活导致 GPU 利用率低**：仅 8/128 expert 被激活，大量 SM 闲置，此时降频对延迟影响小但能大幅降低空闲功耗
2. **Memory-bound 比例更高**：MoE 层的 expert 参数需要从 HBM 加载，计算强度 (FLOP/byte) 较低，降频后计算时间增加有限
3. **最大节能 42.7% 出现在小 batch + 小 input 场景**：此时 GPU 严重 under-utilized，降频几乎不影响延迟

![DVFS Saving](fig6_dvfs_saving_potential.png)

### 4. 频率敏感性对比 (BS=8, IL=1024)

![Frequency Sensitivity](fig4_freq_sensitivity.png)

关键观察：
- **Dense F 算子**：频率敏感性高，从 1410→210 MHz 延迟增加 ~6x
- **MoE F 算子**：频率敏感性中等，从 1410→210 MHz 延迟增加 ~3.5x
- **MoE A 算子**：频率几乎不敏感（210→1410 MHz 延迟变化 <30%），完全 memory-bound
- **Dense A 算子**：有一定频率敏感性

MoE Attention 的频率不敏感性意味着在 AF 分离部署中，**Attention 节点可以大幅降频而几乎不影响延迟**。

### 5. 能耗效率 (mJ/token)

![Energy Per Token](fig5_energy_per_token.png)

- MoE 的 per-token 能耗在所有配置下都低于 Dense（约 1/3-1/5）
- 随着 input_len 增大，两个模型的 per-token 能耗都趋于稳定（amortized fixed overhead）
- 大 batch 时 MoE 的 per-token 能耗优势更明显

### 6. 延迟与能耗随 Input Length 的缩放

![Latency vs Input Length](fig1_latency_vs_input_len.png)
![Energy vs Input Length](fig2_energy_vs_input_len.png)

两个模型的延迟和能耗都近似线性随 input_len 增长，但：
- Dense 的斜率更陡（FFN 线性增长快）
- MoE 在大 input_len 时增长更平缓（稀疏激活的优势在大序列上更明显）

## 对 DVFS/Tier 策略的启示

| 维度 | Dense 模型建议 | MoE 模型建议 |
|------|--------------|-------------|
| AF 分离资源分配 | F 需要 2-3x A 的资源 | A/F 接近 1:1.75 |
| Attention 频率 | 可适度降低 (10-20% saving) | **可大幅降低**（几乎免费节能） |
| FFN 频率 | 保守降低（延迟敏感） | 可更激进降低（memory-bound） |
| DVFS 总节能预期 | 15-20% | **25-40%** |
| 最优频率选择 | 通常 690-930 MHz | 通常 450-690 MHz |
| Expert Load 特征 | 不适用 | 需纳入（max_expert_tokens, ELS） |

## 结论

1. MoE 模型由于稀疏激活特性，其 prefill 延迟和能耗绝对值远低于同等规模的 Dense 模型
2. MoE 的 F/A 比更均衡（1.75 vs 3.10），更适合对称的 AF 分离部署
3. MoE 有更大的 DVFS 节能空间（avg 26.8% vs 16.1%），因为 GPU 利用率低、memory-bound 程度高
4. MoE Attention 算子几乎完全 memory-bound，降频免费
5. 针对 MoE 模型的 DVFS 策略应比 Dense 更激进，特别是在 Attention 侧

## 产出文件

| 文件 | 说明 |
|------|------|
| `compare_dense_vs_moe_prefill.py` | 分析脚本 |
| `fig1_latency_vs_input_len.png` | 延迟 vs 输入长度 |
| `fig2_energy_vs_input_len.png` | 能耗 vs 输入长度 |
| `fig3_fa_ratio.png` | F/A 比值对比 |
| `fig4_freq_sensitivity.png` | 频率敏感性 |
| `fig5_energy_per_token.png` | 每 token 能耗效率 |
| `fig6_dvfs_saving_potential.png` | DVFS 节能潜力 |
| `READ.md` | 本报告 |
