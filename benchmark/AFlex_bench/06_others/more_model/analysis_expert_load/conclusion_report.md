# MoE Expert 冷热负载 vs 延迟 Profiling 实验 — 结论报告

## 实验概述

- **模型**: Qwen3-30B-A3B (128 experts, top-8, 48 layers)
- **硬件**: A800 单卡 (TP=1)
- **数据量**: 143 样本 (53 单请求 + 90 并发 batch)
- **Batch 覆盖**: bs ∈ {1, 4, 8, 16, 32, 64}
- **Prompt 多样性**: 7 类 (code_python, code_cpp, math, conversation, technical, creative, random)

## 核心发现

### 1. Expert Load Imbalance 对延迟的影响随 Batch Size 增大而增强

| Batch Size | TPOT Range (ms) | TPOT Std (ms) | Expert特征影响 |
|------------|-----------------|---------------|---------------|
| 1          | 55.9 - 63.3     | 0.98          | 几乎无 (r < 0.1) |
| 4          | 61.8 - 78.9     | 5.3           | 弱 (r ≈ -0.26) |
| 8          | 62.0 - 62.9     | 0.3           | 极弱 |
| 16         | 62.2 - 90.0     | 6.8           | 弱-中 |
| 32         | 63.9 - 82.8     | 4.6           | **中等** (max_tokens r=+0.44, std r=+0.55) |
| 64         | 57.4 - 110.1    | 17.9          | **强** (ELS r=+0.43, max_tokens r=+0.33) |

### 2. 最佳预测特征排序 (GBDT Feature Importance)

| Rank | 特征 | Importance | 含义 |
|------|------|-----------|------|
| 1 | `max_tokens_per_expert` | 0.4829 | 最热 expert 的 token 数 |
| 2 | `batch_size` | 0.1731 | 当前 batch 大小 |
| 3 | `std_tokens_per_expert` | 0.1460 | token 分布标准差 |
| 4 | `els` (Expert Load Skew) | 0.1299 | max/mean 比值 |
| 5 | `top1_share` | 0.0415 | 最热 expert 占比 |
| 6 | `gini` | 0.0260 | 基尼系数 |
| 7 | `num_active_experts` | 0.0007 | 活跃 expert 数 |
| 8 | `glr` (GPU Load Ratio) | 0.0000 | TP=1时恒为1 |

### 3. 精度提升量化

| 模型 | R²_cv | MAPE | 对比基线 |
|------|-------|------|---------|
| LinearReg (仅 bs) | 0.0278 | 5.94% | — 基线 |
| LinearReg (全特征) | -0.020 | 4.32% | ΔR² = -0.05 |
| RandomForest (全特征) | 0.3568 | 1.50% | ΔR² = +0.33 |
| **GBDT (全特征)** | **0.3651** | **0.15%** | **ΔR² = +0.34** |
| GBDT (仅 expert 特征) | 0.4185 | 0.15% | ΔR² = +0.39 |

**关键结论**: Expert 分布特征将延迟预测 MAPE 从 5.94% 降低至 0.15%，R² 从 0.028 提升至 0.37+。

### 4. 物理机制解释

MoE FusedMoE kernel 的执行模式：
1. Router 将每个 token 路由到 top-k=8 个 expert
2. 每个 expert 独立执行 GEMM 运算
3. **瓶颈 expert** (max_tokens_per_expert 最大的那个) 决定了整层的执行时间
4. 当 batch 较大时，token 集中到少数 expert → 串行化加剧 → 延迟升高

因此 `max_tokens_per_expert` 本质上是衡量**最慢 expert 的计算量**，直接对应 FusedMoE 的实际执行时间。

### 5. 对能耗模型的建议

#### 推荐纳入预测模型的特征（按优先级）：
1. **`max_tokens_per_expert`** — 最关键，直接反映瓶颈 expert 负载
2. **`std_tokens_per_expert`** — 分布不均匀度的连续度量
3. **`els` (Expert Load Skew)** — 简单有效，计算开销低

#### 建议的预测模型形式：
```
latency_moe = f(bs, il, freq, max_tokens_per_expert, std_tokens_per_expert, els)
```

#### 对 DVFS 控制的影响：
- 当 `max_tokens_per_expert` 超过阈值时（如 > 2× mean），应保守降频
- 可用 `els` 作为轻量 proxy: `els > 3.0` 时标记为 "hot expert" 状态
- GLR 在 TP>1 时才有意义，在 TP=2 的 PDAF 部署中需要实测验证

## 产出文件

| 文件 | 用途 |
|------|------|
| `data/expert_load_vs_latency.tsv` | 单请求数据 (53 行) |
| `data/expert_load_vs_latency_batch.tsv` | 并发 batch 数据 (90 行) |
| `analysis_expert_load/expert_load_vs_latency_analysis.png` | 主分析图 |
| `analysis_expert_load/expert_load_within_bs_variation.png` | 各 bs 内部变异分析 |
| `analysis_expert_load/regression_summary.txt` | 回归结果文本 |
| `scripts/profile_expert_load_vs_latency.py` | 单请求 profiling 脚本 |
| `scripts/profile_expert_load_vs_latency_batch.py` | 并发 batch profiling 脚本 |
| `scripts/analyze_expert_load_vs_latency.py` | 分析脚本 |

## 后续工作

1. **TP=2 下的 GLR 验证**: 在 TP=2 部署中重复实验，此时 GLR 特征将生效
2. **更大 batch range**: 确保请求真正被 batch 在一起（可通过 offline batch API 或内部接口）
3. **集成到能耗模型**: 将 `max_tokens_per_expert` 和 `els` 作为新特征加入 LookupModel 或 GBDT 预测器
4. **运行时开销评估**: 评估实时计算这些特征的额外 latency（预计 <0.1ms）
