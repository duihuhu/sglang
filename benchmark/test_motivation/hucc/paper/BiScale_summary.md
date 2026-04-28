# BiScale: Energy-Efficient Disaggregated LLM Serving via Phase-Aware Placement and DVFS

> **来源**: arXiv:2602.18755v2, Feb 2026  
> **作者**: Omar Basit*, Yunzhao Liu*, Z. Jonny Kong, Y. Charlie Hu  
> **机构**: Purdue University

---

## 一、研究动机

Prefill/Decode 分离 (disaggregation) 已成为 LLM serving 的主流趋势 (DistServe, Splitwise, NVIDIA Dynamo, vLLM 等均支持)。然而，现有能效优化工作 (DynamoLLM, throttLL'eM) 均基于非分离架构，未能解决分离架构下的能效优化问题。

### 核心矛盾

- **LLM 推理能耗巨大**: 前沿模型每请求 0.34-4.32 Wh (H100)，10 亿请求/天约 0.8-1.8 GWh/天
- **Autoscaling 粒度太粗**: 重配置开销 (模型加载等) 需数分钟，无法跟踪细粒度负载波动
- **DVFS 在分离架构下更复杂**: Prefill 和 Decode 具有不同的性能瓶颈和 SLO 敏感性

## 二、核心挑战

### Challenge 1: 分离架构下配置空间的组合爆炸

分离架构需要联合配置 prefill 和 decode 两个池:
- 每个池的实例数量、TP 度、基线频率、路由权重
- 两个池紧密耦合: prefill 不足→TTFT 恶化；decode 不足→post-prefill 积压→TPOT 膨胀
- 输入/输出长度分布随时间变化，负载压力在两阶段间迁移

### Challenge 2: 阶段不对称性使 DVFS 控制复杂化

| 维度 | Prefill | Decode |
|------|---------|--------|
| **计算瓶颈** | 计算密集型 (compute-bound) | 内存带宽密集型 (memory-bound) |
| **频率敏感性** | 高 (GPU 频率直接影响 TTFT) | 低 (频率变化对 TPOT 影响较小) |
| **内存压力** | 低 (不保留大量 KV cache) | 高 (需跨迭代维护 KV cache) |
| **负载动态** | 突发性强，到达驱动 | 平滑变化，请求持续多次迭代 |
| **SLO 指标** | TTFT | TPOT |

## 三、系统设计: BiScale

### 总体架构: 两层控制

```
Tier 1: 粗粒度 Provisioning (每 5 分钟)
├── 确定 prefill/decode 实例数量和 TP 度
├── 设置基线 GPU 频率
├── 计算请求路由权重
└── 保证 SLO 可行性

Tier 2: 细粒度 DVFS 控制 (每次迭代)
├── Prefill: MPC (模型预测控制)
│   └── 多批次 horizon 优化，考虑队列演化
└── Decode: 轻量 per-batch 频率选择
    └── Slack-aware 简单策略
```

### Tier 1: 阶段感知的粗粒度 Provisioning

#### ILP 优化公式

**目标**: 最小化总能耗 Σ(n_c × E_c × R_c)

**约束**:
1. GPU 总量不超过集群容量: Σ(n_c × G_c) ≤ G
2. Prefill 容量满足需求: Σ(n_c × R_c) ≥ (1+α)R (c ∈ prefill)
3. Decode 容量满足需求: Σ(n_c × R_c) ≥ (1+α)R (c ∈ decode)

#### 配置表构建

利用**频率感知、内存感知的迭代级推理模拟器**:
- 输入: 请求 trace、实例类型、TP 度、GPU 频率
- 输出: 逐迭代的 TTFT/TPOT、功耗、KV cache 使用量
- 通过二分搜索确定每个配置的最大可行 goodput R_c
- 使用下采样 (非时间膨胀) 生成缩放 trace 以保留真实到达模式

### Tier 2: 阶段特定的细粒度 DVFS

#### Prefill: MPC 控制

选择 MPC 的原因: 频率决策不仅影响当前批次，还影响后续队列演化和 TTFT。

**三步流程**:
1. **批次投射**: 基于当前等待/运行请求，模拟未来 K=8 个批次的调度
2. **频率评估**: 对每种频率分配方案，预测批次完成时间和 TTFT
3. **可行能耗最小化**: 在满足 TTFT SLO 的方案中选择最小能耗

**高效搜索 (Algorithm 1)**:
- 从全部使用最高频率开始
- 贪心逐步引入更低频率
- 枚举所有可能的突变组合，选择满足 SLO 且功耗最低的方案
- 将搜索复杂度从 K^N 降至 O(K × 3^N)
- 使用 N=7 个候选频率，K=8 个 horizon 批次
- 平均运行时间约 4ms

#### Decode: Per-Batch 频率选择

选择简单策略的原因: Decode 负载平滑，频率对 TPOT 影响小，无需 horizon 优化。

- 用 TBT (time between tokens) 作为 TPOT 的保守代理
- 按升序评估候选频率，选择满足 TBT 约束的最低频率
- KV cache 利用率超阈值时临时切换最高频率以加速请求完成

### 建模基础设施

| 模型 | 方法 | 精度 |
|------|------|------|
| Prefill 延迟 | Histogram Gradient Boosting Trees | MAPE 2.9% |
| Decode 延迟 | Histogram Gradient Boosting Trees | MAPE 2.7% |
| Prefill 功耗 | 3D 查找表 + 线性插值 | MAPE 4.1% |
| Decode 功耗 | 回归模型 (频率维度单调约束) | MAPE 1.0% |

特征: 批次请求数、输入长度统计量 (和/均值/标准差)、TP 度、GPU SM 频率。

## 四、实验评估

### 实验环境

- **硬件**: 2 节点 × 8 NVIDIA H100 (共 16 GPU)，InfiniBand 连接
- **模型**: Llama 3.3 70B
- **Trace**: 受控负载 (Gamma 分布) + Azure 生产 trace (ShareGPT 请求)
- **SLO**: TTFT P99 ≤ 600ms, TPOT P99 ≤ 100ms

### Baseline

- **DistServe**: 最大吞吐配置，所有 GPU 最高频率
- **PlaceOnly**: 仅 Tier 1 (能效 placement，固定频率)
- **BiScale**: Tier 1 + Tier 2 (placement + DVFS)

### 主要结果

#### 受控负载 (10-85 RPS)

| 指标 | PlaceOnly vs DistServe | BiScale vs DistServe |
|------|----------------------|---------------------|
| Prefill 能耗 | -20% ~ -31% | **-27% ~ -36%** |
| Decode 能耗 | 可比 | 可比 |
| SLO 满足 | 是 | 是 |

#### 生产负载 (67%/85% 容量)

| 负载 | 指标 | PlaceOnly vs DistServe | BiScale vs DistServe |
|------|------|----------------------|---------------------|
| 67% | Prefill 能耗 | -16% ~ -29% | **-28% ~ -39%** |
| 67% | Decode 能耗 | -37% ~ -45% | **-44% ~ -48%** |
| 85% | 趋势 | 类似 | 类似 |

### 能耗节省分析

#### Tier 1 (PlaceOnly) 的贡献
- Prefill: 平均 -20% (范围 -11% ~ -29%)
- Decode: 平均 -33% (范围 -16% ~ -45%)
- 主要来源: 降低基线频率 + 优化 TP 配置

#### Tier 2 (DVFS) 的增量贡献
- Prefill: 平均 +15% 额外节省 (范围 9% ~ 29%)
- Decode: 平均 +6% 额外节省 (范围 -4% ~ 20%)
- **Prefill DVFS 贡献约为 Decode 的 2.5×**

#### 为什么 Prefill 从 DVFS 获益更多?

1. Prefill 负载突发性强 → 更多低负载间隙可降频
2. Prefill 对频率更敏感 → 降频节能效果更显著
3. Decode 负载平滑 → PlaceOnly 的固定频率已接近最优

#### DVFS 作为预测误差的纠正机制

- 受控 trace 下: DVFS 增量收益小 (PlaceOnly 基于已知 trace，接近最优)
- 生产 trace 下: DVFS 增量收益大 (预测误差导致过度/不足配置，DVFS 在线纠正)

## 五、关键贡献

1. **首个**面向 P/D 分离架构的能效 LLM 推理系统
2. 揭示了阶段不对称性和工作负载动态如何联合影响能效-SLO 权衡
3. 设计了两层控制架构: 粗粒度 placement + 细粒度 DVFS
4. 针对 prefill/decode 不同特征采用不同的 DVFS 策略 (MPC vs per-batch)
5. 在真实多 GPU 集群上验证了显著的能效提升

## 六、与 DynamoLLM 的对比

| 维度 | DynamoLLM | BiScale |
|------|-----------|---------|
| **架构** | 非分离 (collocated) | P/D 分离 (disaggregated) |
| **控制层次** | 三层 (实例/池/集群) | 两层 (placement/DVFS) |
| **频率控制** | 全池统一频率 | 逐实例、逐迭代细粒度调频 |
| **DVFS 策略** | 统一 per-epoch 调频 | 阶段感知 (Prefill: MPC, Decode: per-batch) |
| **优化方法** | MILP | ILP + MPC + 贪心搜索 |
| **负载预测** | 模板方法 | 前 5min 窗口作为预测 |
| **评估规模** | 大规模集群 (40 节点) | 中等规模 (16 GPU) |
| **能耗节省** | 52% (vs 统一配置) | 39% prefill, 48% decode (vs DistServe) |

## 七、局限性

- 每 5 分钟的 provisioning 窗口使用前一窗口负载作为预测，预测方法较简单
- 配置切换时需要拆除旧实例并启动新实例，简化处理为独立运行每个 5min trace
- 评估规模相对较小 (16 GPU)，未验证大规模集群效果
- 仅评估了单一模型 (Llama 3.3 70B)
- 未考虑混合并行策略 (如 PP + TP) 或更复杂的路由策略
