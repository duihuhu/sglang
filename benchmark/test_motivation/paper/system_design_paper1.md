# Paper 1: AF 分离下的算子级动态调频能效优化系统设计

> **目标**: 在 PD 分离的基础上，进一步做 Attention/FFN (AF) 算子级分离，通过动态调整 A/F 实例配比和各自频率，在满足 SLO 的前提下实现系统能耗最低。

---

## 一、三篇相关工作分析

### 1.1 DynamoLLM (HPCA 2025)

- **架构**: 非分离 (collocated)
- **系统设计**: 三层分层控制器——集群管理器 (~30min, Scale in/out 实例数量)、池管理器 (~5min, 调整 TP 并行度)、实例管理器 (~5s, 调整 GPU 频率)。将请求按输入/输出长度分成 9 类，分配到不同配置池 (MultiPool)。
- **调频设计**: 全池统一频率，以 5 秒为周期调整。基于离线 Profiling 构建配置表，使用 MILP 求解最小能耗配置。频率调整粒度较粗 (per-epoch)。
- **核心特点**: 请求异构感知 + 多维配置空间联合优化 + 低开销重配置技术。
- **能耗节省**: ~52% (vs 统一配置)

### 1.2 BiScale (arXiv 2026)

- **架构**: PD 分离 (disaggregated)
- **系统设计**: 两层控制——Tier 1 粗粒度 Provisioning (每 5min, 确定 P/D 实例数量、TP 度、基线频率) + Tier 2 细粒度 DVFS (每次迭代级)。核心在于**阶段感知**: 针对 Prefill 和 Decode 的不同计算特征采用不同控制策略。
- **调频设计**:
  - Prefill: MPC (模型预测控制)，考虑未来 K=8 个批次的队列演化，运行时间约 4ms
  - Decode: 轻量 per-batch 频率选择，利用 TBT slack 选择满足约束的最低频率
- **核心特点**: 首个面向 PD 分离的能效系统，揭示阶段不对称性。
- **能耗节省**: ~39% prefill, ~48% decode (vs DistServe)

### 1.3 throttLL'eM (arXiv 2025)

- **架构**: 非分离 (collocated)
- **系统设计**: 核心是**预测 + 控制**闭环。组件包括: 生成长度预测器、Scoreboard + KV/Batch 投影、准入控制 (虚拟调度 + SLO 检验)、GBDT 性能预测模型、二分频率搜索。Autoscaling 以 ~10s 周期调整 TP 引擎规模。
- **调频设计**: 迭代级 (毫秒级) 调频。在频率可行域上**二分搜索满足 SLO 的最低频率**，全实例 GPU 统一频率。
- **核心特点**: 最细粒度的预测式控制，准入控制换取降频空间，KV 作为性能代理特征。
- **能耗节省**: ~24.7-43.8% (vs Triton)

### 1.4 三篇工作的共同点

| 维度 | 共性 |
|------|------|
| **目标** | 在满足 SLO (延迟约束) 的前提下最小化能耗 |
| **核心旋钮** | GPU DVFS (频率调整) 是核心节能手段 |
| **建模基础** | 均依赖离线 Profiling 构建性能/功耗模型 (GBDT/查找表/回归等) |
| **频率策略** | 均遵循"选择满足 SLO 的最低频率"这一基本原则 |
| **多旋钮协同** | 均将 DVFS 与资源配置 (实例数/TP度) 联合优化 |
| **SLO 安全机制** | 均有降级/回退机制——违反时升频、请求迁移或排队等 |
| **负载感知** | 均感知负载动态变化，按需调整配置 |

### 1.5 三篇工作的差异

| 维度 | DynamoLLM | BiScale | throttLL'eM |
|------|-----------|---------|-------------|
| **架构假设** | 非分离 | PD 分离 | 非分离 |
| **控制层次** | 三层 (集群/池/实例) | 两层 (placement/DVFS) | 两层 (Autoscaling/DVFS) |
| **频率调整粒度** | ~5s, per-epoch | 每次迭代, per-batch | 每次迭代, per-batch |
| **频率调整范围** | 全池统一 | 逐实例、阶段差异化 | 全实例统一 |
| **阶段感知** | 无 | 有 (P: MPC, D: per-batch) | 无 (P 阶段未优化) |
| **搜索算法** | MILP | ILP + MPC + 贪心 | 二分搜索 |
| **请求异构处理** | 9 类请求池 + 长度预测 | 依赖 trace 统计 | 生成长度预测 + Scoreboard |
| **准入控制** | 无显式 | 无显式 | 有 (虚拟调度 + SLO 检验) |
| **评估规模** | 大 (40 节点 H100) | 中 (16 GPU H100) | 小 (8 GPU A100) |

---

## 二、核心洞察：算子级频率异构性

### 2.1 Profiling 数据揭示的关键发现

**Prefill 阶段** (tp=1, bs=1):

| input_len | P_A@210 (us) | P_A@1200 (us) | A 加速比 | P_F@210 (us) | P_F@1200 (us) | F 加速比 | F/A@1200 |
|-----------|-------------|--------------|---------|-------------|--------------|---------|---------|
| 128 | 3,798 | 2,912 | 1.3x | 2,196 | 734 | 3.0x | 0.25 |
| 1024 | 6,423 | 3,598 | 1.8x | 10,586 | 2,195 | 4.8x | 0.61 |
| 8192 | 39,585 | 9,665 | 4.1x | 77,085 | 14,036 | 5.5x | 1.45 |
| 32000 | 322,876 | 59,444 | 5.4x | 297,617 | 53,065 | 5.6x | 0.89 |

**关键发现 1**: FFN 对频率的敏感性远高于 Attention (F 加速比 > A 加速比)。

**关键发现 2**: F/A 延迟比随 input_len 变化——短输入时 A 是瓶颈，长输入时 F 是瓶颈。

**Decode 阶段** (tp=8, 1200MHz):

| bs | input_len | D_A (us) | D_F (us) | F/A |
|----|-----------|---------|---------|-----|
| 1 | 128 | 381 | 432 | 1.13 |
| 16 | 128 | 413 | 507 | 1.23 |
| 256 | 128 | 441 | 537 | 1.22 |
| 256 | 4096 | 767 | 551 | **0.72** |

**关键发现 3**: Decode 侧大 batch + 长上下文时 Attention 反而成为瓶颈 (F/A < 1)。

### 2.2 AF 分离的 Pareto 优势

统一调频 (同频 f 给 A 和 F) 只能在 1D 曲线上选点。AF 差异化调频 (f_A, f_F 独立) 扩展到 2D 空间，其中存在 **Pareto 前沿优于统一调频** 的配置点。

原理: 当 Attention 是 memory-bound 时，降低 f_A 几乎不增加 t_A，但大幅降低功耗 P_A (功耗 ∝ f^α)。同时 f_F 保持高频以保证 t_F 不成为瓶颈。总能耗 = P_A(f_A)·t_A + P_F(f_F)·t_F 可低于统一频率方案。

---

## 三、系统设计

### 3.1 问题定义

**前提假设**: PD 分离已完成 (Prefill 池和 Decode 池独立部署)，在此基础上进一步做 AF 分离。

**四个算子池**:

| 池 | 计算特征 | 频率敏感性 |
|---|---------|-----------|
| **PA** (Prefill-Attn) | 偏 memory-bound | 中等 |
| **PF** (Prefill-FFN) | compute-bound | **高** |
| **DA** (Decode-Attn) | 强 memory-bound | **低** |
| **DF** (Decode-FFN) | 偏 memory-bound | 中等 |

**优化变量**:
1. A/F 实例配比: n_A 和 n_F 的 GPU 数量分配 (Prefill 内和 Decode 内各自独立)
2. 算子级频率: f_A 和 f_F 独立设置 (每个 iteration 可调)

**目标**: 满足 SLO (TTFT_P99, TPOT_P99) 的前提下最小化总能耗

**系统参数**: 频率切换开销 P50 ~4.5ms, avg ~6ms, P99 ~13ms (A800-80GB SXM 实测, SetGpuLockedClocks)

### 3.2 整体架构: 两层控制

```
┌─────────────────────────────────────────────────────────────────────────┐
│                     AF-Disaggregated Serving Cluster                    │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Tier 1: Joint Resource Provisioner (每 T₁ = 几分钟)               │  │
│  │                                                                   │  │
│  │  输入: 近期负载统计 (到达率 λ, 输入/输出长度分布, 活跃 decode 数)   │  │
│  │  联合决策 (一个 ILP 同时输出):                                     │  │
│  │    • P/D 资源划分: n_P = n_PA + n_PF,  n_D = n_DA + n_DF          │  │
│  │    • A/F 实例配比: n_PA vs n_PF,  n_DA vs n_DF                    │  │
│  │    • TP 并行度:    tp_PA, tp_PF, tp_DA, tp_DF                     │  │
│  │    • 基线频率:     f̄_PA, f̄_PF, f̄_DA, f̄_DF                        │  │
│  │  约束: n_PA + n_PF + n_DA + n_DF ≤ G_total                        │  │
│  │  方法: ILP (Profile 表驱动)                                        │  │
│  │                                                                   │  │
│  │  关键: P/D 分配与 A/F 配比存在耦合，联合优化可发现分层方案无法         │  │
│  │        达到的全局最优配置 (如: 减少 PA 给 DF 以换取更低频率)          │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                          │                                              │
│          ┌───────────────┴───────────────┐                              │
│          ▼                               ▼                              │
│  ┌──────────────────────┐  ┌──────────────────────────┐                 │
│  │ Prefill AF Pipeline   │  │ Decode AF Pipeline        │                │
│  │                       │  │                           │                │
│  │  Tier 2 Controller    │  │  Tier 2 Controller        │                │
│  │  (每次 iteration)     │  │  (每次 iteration)         │                │
│  │                       │  │                           │                │
│  │  ┌─────┐   ┌─────┐   │  │  ┌─────┐    ┌─────┐      │                │
│  │  │ PA  │──→│ PF  │   │  │  │ DA  │──→ │ DF  │      │                │
│  │  │f_PA │   │f_PF │   │  │  │f_DA │    │f_DF │      │                │
│  │  └─────┘   └─────┘   │  │  └─────┘    └─────┘      │                │
│  │                       │  │                           │                │
│  │  SLO: TTFT            │  │  SLO: TPOT                │                │
│  └──────────────────────┘  └──────────────────────────┘                 │
│                                                                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │  Monitoring & Feedback (~10-30s)                                   │  │
│  │  • SLO 违反率 → 触发 Tier 1 重规划                                  │  │
│  │  • P/D 吞吐失衡检测 → 触发 P/D 资源重分配                           │  │
│  │  • A/F 负载不均衡检测 → 触发 A/F 配比调整                            │  │
│  │  • 负载模式突变检测 (请求长度分布变化) → 触发全局重规划               │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.3 Tier 1: P/D + A/F 联合资源规划

#### 3.3.1 设计选择: 联合优化 vs 分层优化

一种自然的做法是**分层优化**: 先决定 P/D 资源划分 (复用 BiScale/DistServe)，再在各自内部优化 A/F 配比。但我们选择**联合优化**，原因如下:

**P/D 分配与 A/F 配比存在耦合**，分层优化会丢失全局最优:

```
例: 总共 16 GPU

分层优化:
  Level 0: 先按吞吐匹配决定 n_P=8, n_D=8
  Level 1: 在 n_P=8 内优化 → PA=3, PF=5, DA=4, DF=4

联合优化可能发现:
  PA=2, PF=4, DA=4, DF=6 (即 n_P=6, n_D=10)
  → Decode-FFN 在低频下延迟增加大，多给它 GPU 可用更低频率
  → 虽然 n_P=6 不是分层方案的最优 P/D 比，但整体能耗更低
  → 这种跨层优化只有联合方案能发现
```

**联合优化的可行性**: 搜索空间虽大于分层，但实际可控:
- TP 度候选有限: {1, 2, 4, 8}
- 频率候选有限: {210, 540, 870, 1200}
- 副本数受 GPU 总量约束: k ≤ G / tp
- 大量组合因违反 SLO 或超出资源可提前剪枝
- ILP 求解器 (Gurobi/CPLEX/PuLP) 在此规模下通常秒级可解

#### 3.3.2 四池联合 ILP 优化公式

**目标函数**: 最小化四个算子池的总能耗

```
minimize:
  Σ_{c ∈ C} k_c × E_c(tp_c, f̄_c, workload_c)

  其中 C = {PA, PF, DA, DF}
  E_c = 池 c 中单个实例在给定配置和负载下的能耗 (来自 Profile 表)
```

**约束集合**:

```
// ===== 资源约束 =====
(1)  n_PA + n_PF + n_DA + n_DF ≤ G                  // GPU 总量上限
     其中 n_c = k_c × tp_c

// ===== TP 整数约束 =====
(2)  tp_c ∈ {1, 2, 4, 8}                            // TP 度候选集
(3)  k_c ∈ Z⁺                                        // 正整数副本数

// ===== 频率选择 =====
(4)  f̄_c ∈ {210, 540, 870, 1200}                     // 离散频率候选

// ===== 延迟 SLO 约束 (P/D 和 A/F 耦合的关键) =====
(5)  ∀ 请求类型 r ∈ R_prefill:
     t_PA(r, tp_PA, f̄_PA) + t_PF(r, tp_PF, f̄_PF) + t_AF_comm ≤ TTFT_SLO / L
     // L = 模型层数, 单层延迟约束

(6)  ∀ 请求类型 r ∈ R_decode:
     t_DA(r, tp_DA, f̄_DA) + t_DF(r, tp_DF, f̄_DF) + t_AF_comm ≤ TPOT_SLO

// ===== 吞吐容量约束 (P/D 耦合) =====
(7)  k_PA × Throughput_PA(tp_PA, f̄_PA) ≥ (1+α) × λ
(8)  k_PF × Throughput_PF(tp_PF, f̄_PF) ≥ (1+α) × λ
     // PA 和 PF 都需要匹配 prefill 到达率 λ

(9)  k_DA × Throughput_DA(tp_DA, f̄_DA) ≥ (1+α) × N_active
(10) k_DF × Throughput_DF(tp_DF, f̄_DF) ≥ (1+α) × N_active
     // DA 和 DF 都需要匹配活跃 decode 请求数 N_active

// ===== A/F 流水线平衡约束 =====
(11) |k_PA × Thpt_PA - k_PF × Thpt_PF| ≤ ε × max(k_PA × Thpt_PA, k_PF × Thpt_PF)
(12) |k_DA × Thpt_DA - k_DF × Thpt_DF| ≤ ε × max(k_DA × Thpt_DA, k_DF × Thpt_DF)
     // A/F 吞吐差距不超过 ε (如 10%), 避免严重的流水线气泡
```

**变量汇总**:

| 变量 | 含义 | 类型 | 范围 |
|------|------|------|------|
| k_PA, k_PF, k_DA, k_DF | 各池实例副本数 | 正整数 | [1, G/tp_min] |
| tp_PA, tp_PF, tp_DA, tp_DF | 各池 TP 并行度 | 离散 | {1, 2, 4, 8} |
| f̄_PA, f̄_PF, f̄_DA, f̄_DF | 各池基线频率 | 离散 | {210, 540, 870, 1200} |

**输入参数**:

| 参数 | 含义 | 来源 |
|------|------|------|
| G | GPU 总量 | 集群配置 |
| λ | Prefill 请求到达率 (req/s) | 负载监控窗口统计 |
| N_active | 活跃 Decode 请求数 | 负载监控窗口统计 |
| R_prefill, R_decode | 请求类型集合 (按长度分桶) | 负载监控窗口统计 |
| TTFT_SLO, TPOT_SLO | 延迟 SLO 目标 | 用户配置 |
| α | 容量裕度 | 超参 (如 0.1-0.2) |
| ε | A/F 平衡松弛度 | 超参 (如 0.1) |
| t_AF_comm | A→F 中间激活传输延迟 | Profiling 测量 |

#### 3.3.3 联合优化为什么 P/D 和 A/F 耦合

从约束可见，P/D 和 A/F 之间存在多层耦合:

1. **资源竞争**: n_PA + n_PF + n_DA + n_DF ≤ G，给 PF 多一个 GPU，DA 就少一个
2. **延迟耦合**: PA 和 PF 的频率/TP 联合决定能否满足 TTFT SLO (约束 5)
3. **吞吐耦合**: PA 和 PF 的副本数各自需匹配 λ (约束 7-8)，但用不同 TP/频率时所需副本数不同
4. **能耗-频率权衡的跨池传递**: 降低 f̄_DA 省能耗但可能需要更多 DA 副本 → 挤占 PF 的 GPU → PF 需要更高频率 → 能耗在 DA 和 PF 之间转移

**联合优化捕捉这些耦合，找到全局最优的资源-频率分配。**

#### 3.3.4 ILP 求解加速

为控制求解时间，采用以下策略:

```
1. Profile 表预计算
   - 离线穷举所有 (tp, freq, workload_bin) 组合
   - 每个组合记录 (latency, power, throughput)
   - ILP 中直接查表，无需在线推理

2. 对称性与支配性剪枝
   - 若配置 A Pareto 支配配置 B (延迟更低且能耗更低)，剪掉 B
   - 每个池的候选配置从 |TP| × |F| = 16 种降至 Pareto 前沿上的少数几种

3. Warm Start
   - 用上一个规划窗口的解作为初始可行解
   - ILP 求解器可在此基础上快速改进

4. 负载分桶
   - 将连续的请求长度分布离散化为若干代表性桶
   - 约束 (5)(6) 只针对桶的代表值，而非每个请求
```

#### 3.3.5 A/F 负载不对称性驱动配比

F/A 延迟比随配置变化，是联合优化发现差异化配比的关键数据基础:

| 配置 | P_A (us) | P_F (us) | F/A 比 | 联合优化的含义 |
|------|---------|---------|--------|-------------|
| tp=1, in=128, 1200MHz | 2,912 | 734 | 0.25 | A 是瓶颈 → 多给 PA GPU |
| tp=1, in=8192, 1200MHz | 9,665 | 14,036 | 1.45 | F 是瓶颈 → 多给 PF GPU |
| tp=1, in=32000, 1200MHz | 59,444 | 53,065 | 0.89 | 接近均衡 → 均分 |
| tp=8, in=128, 1200MHz | 2,881 | 3,234 | 1.12 | 接近均衡 |
| tp=8, in=8192, 1200MHz | 3,984 | 6,313 | 1.58 | F 是瓶颈 → 多给 PF GPU |

**当负载以长请求为主时，ILP 倾向给 PF 分配更多 GPU (降低 f̄_PF 的同时保持吞吐); 短请求为主时反之。**

#### 3.3.6 重规划触发与开销控制

**触发条件** (Monitoring 模块, 每 10-30s 采样):

```
metric_1: SLO_violation_rate > threshold        // SLO 违反率过高
metric_2: |A_util - F_util| > δ                  // A/F 利用率失衡
metric_3: |P_util - D_util| > δ                  // P/D 利用率失衡
metric_4: workload_distribution_shift detected   // 请求长度分布突变

任一指标在连续多个窗口超过阈值 → 触发 Tier 1 ILP 重规划
```

**开销控制**:
- 重规划涉及 GPU 在四个池之间的角色切换 (加载不同权重)
- 预缓存 PA/PF/DA/DF 四种角色的权重在本地磁盘
- Shadow instancing: 后台预创建新配置的实例，ready 后原子切换
- 控制重规划间隔不低于 T₁_min (如 2-5 分钟)

#### 3.3.7 Profile 表构建

离线对每个池独立 profiling:

```
PA_Profile[tp, input_len, batch_size, freq] → (latency_PA, power_PA, throughput_PA)
PF_Profile[tp, input_len, batch_size, freq] → (latency_PF, power_PF, throughput_PF)
DA_Profile[tp, input_len, output_len, batch_size, freq] → (latency_DA, power_DA, throughput_DA)
DF_Profile[tp, input_len, output_len, batch_size, freq] → (latency_DF, power_DF, throughput_DF)
```

#### 3.3.8 Ablation 设计: 联合 vs 分层

在 Evaluation 中通过以下对比验证联合优化的优越性:

| Baseline | 描述 | 预期结果 |
|----------|------|---------|
| **Hierarchical** | 先按吞吐匹配决定 n_P/n_D，再各自内部优化 A/F | 能耗高于联合方案 |
| **Fixed-Ratio** | 固定 A/F=1:1，仅优化频率 | 流水线气泡导致资源浪费 |
| **PD-Only** | 仅 P/D 分离，无 AF 分离 (如 BiScale) | 缺少算子级频率自由度 |
| **Joint (本文)** | 联合优化 P/D+A/F 配比+频率 | **最优** |

这组 ablation 可以量化两个维度的增益:
- **AF 分离的增量收益**: Joint vs PD-Only
- **联合优化的增量收益**: Joint vs Hierarchical

### 3.4 Tier 2: Per-Iteration 算子级 DVFS

每次 iteration，独立选择 f_A 和 f_F，目标是找到满足 SLO 的最低能耗频率组合。

#### 3.4.1 Prefill 侧 DVFS

Prefill 突发性强，延迟敏感 (TTFT)，采用基于 SLO slack 的联合频率搜索:

```
算法: Prefill AF-DVFS (每次 iteration)

输入:
  当前 batch B_PA (已调度到 PA 的请求集合)
  每个请求 r 的 TTFT deadline: d_r
  PA/PF Profile 模型 M_PA, M_PF

步骤:
  1. 对当前 batch, 计算 TTFT slack:
     slack = min_{r ∈ B} (d_r - elapsed_r) - t_AF_comm

  2. 搜索 (f_PA, f_PF) 组合:
     candidates = {}
     for f_PA in [210, 540, 870, 1200]:
       for f_PF in [210, 540, 870, 1200]:
         t_PA = M_PA.predict(batch, f_PA)
         t_PF = M_PF.predict(batch, f_PF)
         if (t_PA + t_PF + t_AF_comm) × remaining_layers ≤ slack:
           e = power_PA(f_PA) × t_PA + power_PF(f_PF) × t_PF
           candidates.add((f_PA, f_PF, e))

  3. 选择能耗最低的可行组合:
     (f_PA*, f_PF*) = argmin_{(f_PA, f_PF) ∈ candidates} e

  4. 设置频率并执行

复杂度: O(|F|²) = O(16) — 常数时间，可忽略
```

**设计选择说明 (为什么不用 MPC)**:
- BiScale 的 MPC 是因为 Prefill 只有一个频率旋钮 (f_P)，需要在时间维度上优化未来 K 个 batch 的频率序列
- AF 分离提供了空间维度的额外自由度 (f_A vs f_F)，使得单步的 (f_A, f_F) 联合搜索就已经有足够大的节能空间
- 如需进一步优化，可扩展为 MPC

#### 3.4.2 Decode 侧 DVFS

Decode 负载平滑，延迟不那么敏感，采用 per-batch slack-aware 策略:

```
算法: Decode AF-DVFS (每次 iteration)

输入:
  当前 batch B_DA (活跃 decode 请求)
  TPOT budget: SLO_TPOT
  DA/DF Profile 模型 M_DA, M_DF

步骤:
  1. 获取当前 batch 特征: batch_size, avg_kv_blocks

  2. 按能耗升序搜索可行 (f_DA, f_DF):
     for (f_DA, f_DF) in sorted_by_energy(all_combos):
       t_DA = M_DA.predict(batch, f_DA)
       t_DF = M_DF.predict(batch, f_DF)
       if t_DA + t_DF + t_AF_comm ≤ SLO_TPOT:
         return (f_DA, f_DF)

     // 若无可行组合, 使用最高频率
     return (1200, 1200)

  注: 由于 DA 对频率非常不敏感 (强 memory-bound),
      很可能最优解总是 f_DA = 最低频, f_DF 按 slack 选择
```

#### 3.4.3 频率切换开销处理 (实测数据)

> **实验平台**: A800-80GB SXM × 8, NVIDIA Driver, SetGpuLockedClocks API
>
> **实验结论** (benchmark/test_motivation/bench_dvfs_overhead.py):
>
> | 指标 | SetApplicationsClocks | SetGpuLockedClocks |
> |------|----------------------|-------------------|
> | API latency P50 | ~2.7 ms | ~4.5 ms |
> | API latency avg | ~3.4 ms | ~6.0 ms |
> | API latency P99 | ~10 ms | ~13 ms |
> | 降频生效? | **不生效** (仅升频有效) | **立即生效** (升降均有效) |
> | Settle 100%? | 仅降到最低频 | **全部 100%** |
> | Settle avg | N/A (超时) | ~0.9 ms |
>
> **必须使用 SetGpuLockedClocks**: SetApplicationsClocks 是 "建议" 频率, 降频不生效, 不适合精确调频场景。
>
> **多 GPU 切频**: 受 NVIDIA 内核驱动全局锁限制, N 卡切频被串行化, 总开销 ~N × 6ms。
> 多线程/多进程均无法绕过 (锁在 nvidia.ko 内核模块中)。
> 8 卡串行: ~46ms, 8 卡多线程: ~46ms, 8 卡多进程: ~56ms (额外 IPC 开销)。
>
> **对系统设计的影响**:
> - 单 GPU 切频 ~6ms, 在 AF 分离架构中每个池独立调频, 单次涉及 1-2 卡, 开销 6-12ms
> - Tier 2 调频粒度应为 per-batch 而非 per-iteration (Decode iteration ~1-2ms, 切频开销远大于 iteration)
> - 惰性切频策略为必选项, 不能频繁切频

**AF 分离架构下**: A 和 F 在不同 GPU 上，各自维持各自的频率，**不存在切频开销**。切频开销仅在以下场景相关:

- **Tier 1 重规划**: 调整某个池的基线频率 → ~6ms 完全可接受 (T₁ 是分钟级)
- **Tier 2 相邻 batch 间切换频率**: 每个池内部切频 ~6ms (per-batch, 非 per-iteration)

**对 Decode 的影响**:
- Decode iteration 典型耗时 ~1ms (bs=1) 到 ~2ms (bs=256)
- ~6ms 切频 + 1ms iteration = 7ms → per-iteration 切频开销过大, 不可行
- 应采用 per-batch 调频: batch 间隔 ~50-200ms, 6ms 开销占 3-12%, 可接受

**Decode 惰性切频策略**:
```
if |f_new - f_current| ≥ threshold AND estimated_savings > switching_cost:
  switch_freq(f_new)      // 承担 ~6ms 开销, 需确保后续运行足够多 iteration 回本
else:
  keep f_current           // 维持当前频率, 避免开销
```

**对 Prefill 的影响**:
- Prefill iteration 通常 ≥10ms (长序列可达数十到数百 ms)
- ~6ms 切频开销比例较小 (<60%)，per-batch 调频可行

### 3.5 能耗模型

```
单次 iteration 能耗:
  E_iter = P_A(f_A, batch, ctx) × t_A(f_A, batch, ctx)   // Attention 能耗
         + P_F(f_F, batch, ctx) × t_F(f_F, batch, ctx)   // FFN 能耗
         + E_comm                                         // AF 通信能耗
         + E_idle                                         // 流水线空闲能耗

AF 分离的 Pareto 优势:
  E_AF(f_A, f_F) ≤ E_unified(f)  当 f 使得延迟相同时

  原理: 当 A 是 memory-bound 时, 降 f_A 几乎不增加 t_A,
        但大幅降低 P_A (功耗 ∝ f^α)。
        同时 f_F 可以保持高频以保证 t_F 不成为瓶颈。
```

---

## 四、与已有工作的定位对比

| 维度 | DynamoLLM | throttLL'eM | BiScale | **本文 (Paper 1)** |
|------|-----------|------------|---------|-------------------|
| 分离粒度 | 无 | 无 | PD 两阶段 | **PD + AF 四算子** |
| 频率自由度 | 1 | 1 | 2 (f_P, f_D) | **4 (f_PA, f_PF, f_DA, f_DF)** |
| 核心 insight | 请求异构 | 预测式控制 | 阶段不对称 | **算子级频率异构** |
| 资源配比 | 固定 TP 池 | 动态 TP | P/D 实例数 | **A/F 实例配比** |
| 切频开销 | 50-80ms | ~200ms | 未讨论 | **~6ms (P50 4.5ms)** |

---

## 五、核心贡献点

1. **首次揭示 Attention 和 FFN 的频率敏感性异构**——并用 Profiling 数据量化 (Pareto 图)
2. **AF 分离 + 差异化调频**——比 PD 级 DVFS 多出两个频率自由度，Pareto 前沿更优
3. **P/D + A/F 联合资源规划**——首次将 P/D 分配和 A/F 配比视为一个联合优化问题，揭示并利用跨层耦合效应，发现分层分解无法达到的全局最优配置
4. **动态 A/F 配比**——首次将 A/F 负载不对称性作为资源分配的依据
5. **低切频开销 (P50 ~4.5ms, avg ~6ms) 使 per-batch 级调频成为可能**——比 throttLL'eM (200ms) 低一个数量级; 实测验证 SetGpuLockedClocks 升降频均立即生效, 多 GPU 受内核驱动全局锁限制需串行化但在 AF 独立进程架构下影响有限

---

## 六、论文结构建议

```
1. Introduction
   - LLM 推理能耗问题
   - AF 分离作为新的能效优化维度

2. Background & Motivation
   - PD 分离与 AF 分离的背景
   - 算子级频率异构性观察 (Pareto 图作为 motivation)
     → 统一调频 vs AF 差异化调频的 Pareto 前沿对比
   - A/F 负载不对称性随工作负载变化 (F/A ratio 数据)

3. System Design
   - 两层控制架构
   - Tier 1: P/D + A/F 联合 ILP 资源规划 (四池联合优化)
   - Tier 2: Per-batch (f_A, f_F) 联合搜索 (实测表明 per-iteration 不可行)
   - 切频开销处理 (~6ms overhead analysis, SetGpuLockedClocks 实测)
   - A/F 流水线平衡约束
   - 联合优化 vs 分层优化的设计选择论证

4. Implementation
   - 基于 SGLang 的 AF 分离实现
   - Profile 表构建
   - 频率控制接口 (NVML)

5. Evaluation
   - vs 统一调频 (同一频率给 A 和 F)
   - vs BiScale (PD 分离但无 AF 分离)
   - vs DynamoLLM (无分离)
   - Ablation: Tier 1 alone, Tier 2 alone, combined
   - Ablation: 联合优化 vs 分层优化 (量化跨层耦合的收益)
   - 不同负载模式下的 P/D+A/F 配比自适应
   - 切频开销分析

6. Related Work
7. Conclusion
```

---

## 七、前置工作清单

### Phase 1: Profiling 数据采集 (详细执行方案)

#### 现有基础设施

| 组件 | 路径 | 状态 |
|------|------|------|
| DVFSController (Python) | `python/sglang/srt/layers/dvfs.py` | ✅ 可用，支持 `lock_sm_clock()` + `get_power()` |
| libdvfs_ctrl.so (C++ NVML) | `benchmark/test_motivation/dvfs/` | ✅ 已编译，需 `cd dvfs && make` |
| DVFS 开销 benchmark | `benchmark/test_motivation/bench_dvfs_overhead.py` | ✅ 已完成 |
| AFD 实现 (Attention/FFN 分离) | `python/sglang/srt/layers/afd_mixin.py` | ✅ 可独立调用 self_attn / mlp |
| A/F Profiling 脚本 | — | ❌ 不存在，需新建 |
| NVML 能耗计数器接口 | `dvfs_ctrl.cpp` | ❌ 需新增 `dvfs_get_energy_mj()` |

#### 能耗采集方案: NVML 硬件能耗计数器

NVML 提供 `nvmlDeviceGetTotalEnergyConsumption()` API，返回 GPU 自驱动加载以来的
**累积能耗** (单位 mJ)。这是硬件级计数器，连续累积，精度远高于功耗采样。

```
直接测能耗 (推荐):
  e0 = nvmlDeviceGetTotalEnergyConsumption()   // mJ
  执行 Attention × N 次
  e1 = nvmlDeviceGetTotalEnergyConsumption()   // mJ
  energy_A = (e1 - e0) / N                    // mJ/次 (精确)

vs 间接算能耗 (旧方案, 已弃用):
  power_samples = [后台线程采样功耗 W]          // 50-100ms 间隔, 可能漏掉瞬态
  energy_A ≈ mean(power_samples) × latency     // 近似值
```

**优势**: 无需后台采样线程，无采样对齐问题，几行代码即可完成。

**需要新增的接口** (一次性开发):

```cpp
// dvfs_ctrl.cpp 新增
int dvfs_get_energy_mj(int gpu_idx) {
    nvmlDevice_t dev = get_device(gpu_idx);
    if (!dev) return -1;
    unsigned long long energy = 0;
    nvmlReturn_t ret = nvmlDeviceGetTotalEnergyConsumption(dev, &energy);
    if (ret != NVML_SUCCESS) return -1;
    return (long long)energy;  // millijoules
}
```

```python
# dvfs.py 新增
def get_energy_mj(self) -> int:
    """Cumulative energy consumption in millijoules since driver load."""
    return self._lib.dvfs_get_energy_mj(self.device_index)
```

#### A/F 独立 Profiling 方案

**核心思路**: 在同一块 GPU 上，**时间上隔离** Attention 和 FFN，独立测量各自的延迟和能耗。
不需要 AF 分离基础设施，不需要多 GPU，不需要 AFD 通信。

```
同一块 GPU，时间上隔离:

  ┌── Attention × N 次 ──┐    ┌── FFN × N 次 ──┐
  │  e0            e1    │    │  e2         e3  │
  └──────────────────────┘    └─────────────────┘
  
  energy_A = (e1 - e0) / N    energy_F = (e3 - e2) / N
  latency_A = wall_time / N    latency_F = wall_time / N
  power_A = energy_A / latency_A  (反推平均功率, 可选)
```

SGLang `LlamaDecoderLayer` 的两个子模块可直接独立调用:
```python
layer.self_attn(positions, hidden_states, forward_batch)  # Attention
layer.mlp(hidden_states)                                   # FFN
```

---

#### [P0-1] Prefill A/F 延迟 + 功耗 Profiling

**现有数据对比**:

| | prefill_data.txt (旧) | prefill_data_v0.txt (旧) | prefill_data_v1.txt (新) |
|--|----------------------|------------------------|------------------------|
| batch_size | {1, 16, 256(tp=8)} | {1} | **{1, 2, 4, 8, 16, 32}** |
| input_len | {128, 512, 4096} | {128..40000, 10 档} | **{128..40000, 10 档}** |
| tp | {1, 2, 4, 8} | {1, 2, 4, 8} | {1, 2, 4, 8} |
| gpu_clock | {210, 540, 870, 1200} | {210, 540, 870, 1200} | **{210, 540, 870, 1200} + 可选更细** |
| 功耗 | ❌ | ❌ | **✅ P_A_power, P_F_power** |

**采集矩阵**:

```
tp         ∈ {1, 2, 4, 8}                                              → 4 值
input_len  ∈ {128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32000}     → 9 值
batch_size ∈ {1, 2, 4, 8, 16, 32}                                      → 6 值
gpu_clock  ∈ {210, 450, 690, 930, 1170, 1410}                          → 6 值

总配置数 = 4 × 9 × 6 × 6 = 1296 个组合
需排除不可行组合 (OOM 等), 预计实际 ~900-1000 个

注意:
  - tp=1 时 input_len=32000 + bs=32 可能 OOM → 自动跳过
  - tp=8 时可额外测 bs=64, 128, 256
  - 频率值需 snap_to_supported() 对齐到 GPU 实际支持的频率点
```

**每个配置的执行流程**:

```
对每个 (tp, input_len, batch_size, gpu_clock):
  1. 设置 TP 并行度 (需启动对应的模型实例)
  2. lock_sm_clock(gpu_clock)
  3. 构造输入: input_ids = random tokens, shape = (batch_size, input_len)
  4. 预热 10 次 forward (单层)
  5. 启动功耗采样线程
  6. 执行 50 次 forward (单层), 用 CUDA events 分别记录 t_A, t_F
  7. 停止功耗采样
  8. 记录:
     - P_A_lat = median(t_A_list) (us)
     - P_F_lat = median(t_F_list) (us)
     - P_A_power = mean(power_samples_during_A) (W)  -- 若无法对齐则用总 power
     - P_F_power = mean(power_samples_during_F) (W)
     - P_total_power = mean(all_power_samples) (W)
  9. unlock_sm_clock()
```

**A/F 功耗分离: 采用方案 B (独立执行 + 独立功耗采集)**

在 AF 分离架构下，A 和 F 在不同 GPU 上独立运行、独立设频，能耗模型为:
```
E = P_A(f_A) × t_A(f_A) + P_F(f_F) × t_F(f_F)
```
P_A(f) 和 P_F(f) 是两个独立的函数——Attention (memory-bound) 和 FFN (compute-bound)
在同一频率下的功耗不同，功耗-频率关系也不同。因此**必须分别测量**。

**SGLang 已有 AFD 支持，可直接独立调用 Attention / FFN**:

```python
# LlamaDecoderLayer 的两个子模块可独立调用:
#   self.self_attn(positions, hidden_states, forward_batch)  → Attention
#   self.mlp(hidden_states)                                  → FFN
# 
# 或使用 AFD mixin 的接口:
#   layer.forward_afd_A(positions, hidden_states, forward_batch, residual)
#   layer.forward_afd_F(hidden_states, forward_batch, residual)
#
# 关键文件:
#   python/sglang/srt/layers/afd_mixin.py  → _run_attn(), _run_mlp()
#   python/sglang/srt/models/llama.py      → LlamaDecoderLayer
```

**方案 B 执行流程 (每个配置点)**:

```
Phase A: 纯 Attention 功耗采集
  1. lock_sm_clock(target_freq)
  2. 构造输入 hidden_states, positions, forward_batch
  3. 预热: 执行 self.self_attn(...) × N_warmup 次
  4. 启动功耗采样线程
  5. 连续执行 self.self_attn(...) × N_repeat 次, CUDA events 计时
  6. 停止采样 → power_A(freq), t_A(freq)

Phase F: 纯 FFN 功耗采集
  7. 构造 FFN 输入 (Attention 的输出 shape)
  8. 预热: 执行 self.mlp(...) × N_warmup 次
  9. 启动功耗采样线程
  10. 连续执行 self.mlp(...) × N_repeat 次, CUDA events 计时
  11. 停止采样 → power_F(freq), t_F(freq)
  12. unlock_sm_clock()

N_repeat ≥ 50, 确保采样线程 (~20ms 间隔) 采到 ≥10 个功耗样本
```

**输出格式**: `prefill_data_v1.txt` (Tab 分隔)

```
tp  input_len  gpu_clock  batch_size  P_A_lat  P_F_lat  P_A_power  P_F_power  idle_power_W
```

**时间估算**:

```
每个配置: ~10s (预热 + 50次执行 + 功耗采样)
每个 TP 需要重新启动模型: ~2-5 min (模型加载)
总配置: ~700 个 ÷ 4 TP = ~175 个/TP
每个 TP: 175 × 10s ≈ 30 min + 5 min 加载 ≈ 35 min
总时间: 4 × 35 min ≈ 2.5 小时
加上 OOM 重试和间歇: ~3-4 小时
```

---

#### [P0-2] Decode A/F 延迟 + 功耗 Profiling

**采集矩阵**:

```
tp         ∈ {1, 2, 4, 8}                                → 4 值
input_len  ∈ {128, 512, 1024, 2048, 4096, 8192}           → 6 值
output_len ∈ {64, 256, 512, 4096}                         → 4 值
batch_size ∈ {1, 4, 8, 16, 32, 64, 128, 256}              → 8 值
gpu_clock  ∈ {210, 450, 690, 930, 1170, 1410}              → 6 值

总配置数 = 4 × 6 × 4 × 8 × 6 = 4608 个组合
去除 OOM 等: 预计实际 ~1200-1400 个
```

**Decode 特殊处理 (方案 B: 独立 A/F 功耗)**:

```
对每个 (tp, input_len, output_len, batch_size, gpu_clock):

  准备阶段:
    1. 先执行 Prefill (input_len tokens) 填充 KV cache
    2. 执行 output_len 步 Decode 生成 KV cache 到目标长度
       (此时 KV cache 包含 input_len + output_len 个 token)

  数据含义: 测量的是生成第 output_len 个 output token 时的单步 Decode 延迟和功耗。
            此时 KV cache 中有 (input_len + output_len - 1) 个 token 的 KV 条目
            (input_len 个来自 Prefill, output_len - 1 个来自前序 Decode 步)。
            output_len 越大 → KV cache 越大 → Attention 访存量越大 → 延迟越高。

  Phase A: 纯 Decode-Attention 功耗
    3. lock_sm_clock(gpu_clock)
    4. 连续执行 self.self_attn(...) × N_repeat 次 (batch_size 个请求同时 decode)
    5. 后台采样功耗 → D_A_power(freq)
    6. CUDA events → D_A_lat(freq)

  Phase F: 纯 Decode-FFN 功耗
    7. 连续执行 self.mlp(...) × N_repeat 次
    8. 后台采样功耗 → D_F_power(freq)
    9. CUDA events → D_F_lat(freq)
    10. unlock_sm_clock()
```

**输出格式**: `decode_data_v1.txt` (Tab 分隔)

```
tp  input_len  output_len  gpu_clock  batch_size  D_A_lat  D_F_lat  D_A_power  D_F_power  idle_power_W
```

**时间估算**: ~2-3 小时

---

#### [P0-3] AF 通信开销测量 (t_AF_comm)

**目的**: 测量 Attention GPU → FFN GPU 之间中间激活传输延迟。

**测量方法**:

```python
# 伪代码
import torch
import torch.distributed as dist

hidden_size = 8192  # Llama 70B

for bs in [1, 4, 16, 64, 256]:
    for seq_len in [1, 128, 1024, 8192]:
        tensor = torch.randn(bs, seq_len, hidden_size, dtype=torch.bfloat16, device='cuda:0')
        
        # 预热
        for _ in range(10):
            dist.send(tensor, dst=1)  # 或 NCCL P2P
        
        # 计时
        times = []
        for _ in range(100):
            torch.cuda.synchronize()
            t0 = time.perf_counter_ns()
            dist.send(tensor, dst=1)
            torch.cuda.synchronize()
            t1 = time.perf_counter_ns()
            times.append((t1 - t0) / 1000)  # us
        
        data_bytes = bs * seq_len * hidden_size * 2  # bf16 = 2 bytes
        latency_us = median(times)
        bw_gbps = data_bytes / (latency_us * 1e-6) / 1e9
```

**测量维度**:

```
传输方式:
  - NVLink 直连 (同节点, GPU 0 → GPU 1)
  - NCCL send/recv
  - torch.distributed P2P

batch_size ∈ {1, 4, 16, 64, 256}
seq_len    ∈ {1, 128, 1024, 8192}
  注: Decode 时 seq_len=1 (仅传 1 token 的 hidden)
      Prefill 时 seq_len = input_len
```

**输出格式**: `af_comm_overhead.txt`

```
transport  batch_size  seq_len  data_bytes  latency_us  bandwidth_gbps
```

**时间估算**: ~30 分钟

---

#### [P0-4] 频率切换开销 (已完成)

> ✅ 已完成。脚本: `bench_dvfs_overhead.py`
> 结论: SetGpuLockedClocks P50 ~4.5ms, avg ~6ms, P99 ~13ms。
> 多 GPU 受内核驱动全局锁限制串行化。

---

#### [P0-5] 空闲功耗基线

**目的**: 测量各频率下 GPU 空闲功耗，用于计算净功耗。

```
对每个 gpu_clock ∈ {210, 450, 690, 930, 1170, 1410}:
  1. lock_sm_clock(gpu_clock)
  2. 无任何 CUDA kernel 运行
  3. 采样 30s 功耗
  4. 记录 idle_power_W

输出: idle_power.txt
  gpu_clock  idle_power_W  idle_power_std_W
```

**时间估算**: ~5 分钟

---

#### 基于现有 AFD 代码的 Profiling 实现方案

SGLang 已有完整的 AFD 实现，可以直接利用:

```
关键代码文件:
  python/sglang/srt/layers/afd_mixin.py    → forward_afd_A(), forward_afd_F()
  python/sglang/srt/layers/afd.py          → model_forward_afd(), AFDProxyAttention/MLP
  python/sglang/srt/models/llama.py        → LlamaDecoderLayer (已集成 AFD mixin)
  python/sglang/srt/layers/dvfs.py         → DVFSController (锁频 + 功耗读取)
```

**两种 Profiling 方式**:

```
方式 1 (推荐, 更简单): 单 GPU 直接调用子模块
  → 加载完整 LlamaDecoderLayer 到单 GPU
  → 独立调用 layer.self_attn(...) 采集 Attention 延迟+功耗
  → 独立调用 layer.mlp(...) 采集 FFN 延迟+功耗
  → 优点: 无需多 GPU, 无需启动 AFD 通信, 简单快速
  → 缺点: 不含 LayerNorm/residual 开销 (很小, 可忽略)

方式 2 (完整, 更精确): 启动两个 AFD 进程
  → 用 --afd-perspective attn 启动 Attention 节点
  → 用 --afd-perspective ffn 启动 FFN 节点
  → 各自锁频, 各自采集功耗
  → 优点: 包含完整 AFD 通信 + norm 开销
  → 缺点: 需要多 GPU + 配置 AFD 通信后端, 复杂度高
```

**方式 1 的核心代码框架**:

```python
import torch
from sglang.srt.layers.dvfs import DVFSController

# 加载单层 LlamaDecoderLayer (而非整个模型, 节省显存)
layer = load_single_decoder_layer(model_path, layer_id=0, tp=tp, device='cuda:0')
ctrl = DVFSController(device_index=0)

FREQS = [210, 450, 690, 930, 1170, 1410]

for freq in FREQS:
    freq = ctrl.snap_to_supported(freq)
    ctrl.lock_sm_clock(freq)
    
    # --- Attention Profiling ---
    hidden = torch.randn(batch_size, input_len, hidden_size, device='cuda:0', dtype=torch.bfloat16)
    # warmup
    for _ in range(10):
        _ = layer.self_attn(positions=pos, hidden_states=hidden, forward_batch=fb)
    # measure
    power_sampler.start()
    attn_times = []
    for _ in range(N_REPEAT):
        start_event.record()
        _ = layer.self_attn(positions=pos, hidden_states=hidden, forward_batch=fb)
        end_event.record()
        torch.cuda.synchronize()
        attn_times.append(start_event.elapsed_time(end_event) * 1000)  # us
    power_A = power_sampler.stop_and_mean()
    t_A = median(attn_times)
    
    # --- FFN Profiling ---
    ffn_input = torch.randn(batch_size, input_len, hidden_size, device='cuda:0', dtype=torch.bfloat16)
    for _ in range(10):
        _ = layer.mlp(ffn_input)
    power_sampler.start()
    ffn_times = []
    for _ in range(N_REPEAT):
        start_event.record()
        _ = layer.mlp(ffn_input)
        end_event.record()
        torch.cuda.synchronize()
        ffn_times.append(start_event.elapsed_time(end_event) * 1000)
    power_F = power_sampler.stop_and_mean()
    t_F = median(ffn_times)
    
    ctrl.unlock_sm_clock()
    # 写入结果: tp, input_len, freq, batch_size, t_A, t_F, power_A, power_F
```

**关键挑战: 如何加载单层**

```
问题: SGLang 模型加载是整体的 (所有层 + embedding + lm_head)
      Llama 70B 整体加载需要大量显存

解决方案 (按推荐度排序):

1. 只加载 1 层的权重 (最省显存)
   → 用 safetensors.torch.load_file() 仅加载 layer.0 的权重
   → 手动构造 LlamaDecoderLayer + LlamaAttention + LlamaMLP
   → 需要模型 config (hidden_size, num_heads, num_kv_heads 等)

2. 加载完整模型但只 profile 第 0 层 (最简单)
   → 正常启动 SGLang server / model_runner
   → 在 LlamaDecoderLayer.forward 中插入 profiling hooks
   → 利用已有的 forward_batch 构造机制

3. 使用 HuggingFace transformers 加载单层 (中等复杂度)
   → from transformers import LlamaForCausalLM
   → model.model.layers[0].self_attn / .mlp
   → 但 kernel 实现不同 (无 FlashAttention 等优化)
   → 仅适合初步验证, 不代表 SGLang 真实性能

推荐先用方案 2 (加载完整模型), 开发效率最高。
对于 tp=1/2/4, 70B 模型可以放下。
```

**Decode Profiling 的特殊要求**:

```
Decode 阶段 Attention 需要 KV cache:
  → 必须先 Prefill 填充 KV cache 到目标长度
  → 然后在目标 KV cache 大小下反复执行 Decode Attention
  → Decode-FFN 与 KV cache 大小无关 (仅 batch_size 影响)

实现:
  1. forward_batch 设置为 extend/prefill 模式, 执行 Prefill 填充 KV cache
  2. forward_batch 切换为 decode 模式 (seq_len=1)
  3. 在 decode 模式下反复执行 layer.self_attn() 采集延迟+功耗
  4. 在 decode 模式下反复执行 layer.mlp() 采集延迟+功耗
```

---

#### 需新建的脚本清单

| 脚本 | 功能 | 依赖 |
|------|------|------|
| `bench_prefill_af.py` | Prefill A/F 延迟+功耗 profiling (方案 B) | DVFSController, 模型加载, CUDA events |
| `bench_decode_af.py` | Decode A/F 延迟+功耗 profiling (方案 B) | 同上 + KV cache 管理 |
| `bench_af_comm.py` | AF 通信开销测量 | torch.distributed, NCCL |
| `bench_idle_power.py` | 空闲功耗基线 | DVFSController |
| `utils_profiling.py` | 公共工具 (功耗采样线程, CUDA 计时, 结果输出) | DVFSController |

**公共工具 `utils_profiling.py` 包含**:

```python
class PowerSampler:
    """后台线程持续采样 GPU 功耗 (via DVFSController.get_power())"""
    def __init__(self, ctrl: DVFSController, interval_ms=20): ...
    def start(self): ...
    def stop_and_mean(self) -> float: ...  # 返回平均功耗 W
    def stop_and_samples(self) -> list[float]: ...  # 返回所有样本

class CUDATimer:
    """CUDA events 计时器, 自动管理 start/end events"""
    def __init__(self): ...
    def start(self): ...
    def stop(self) -> float: ...  # 返回 us

def load_single_decoder_layer(model_path, layer_id, tp, device):
    """加载单个 LlamaDecoderLayer 的权重到指定设备"""
    ...

def run_profiling_sweep(layer, configs, output_file, phase='prefill'):
    """通用 profiling 扫描循环"""
    ...
```

---

#### 执行顺序与依赖

```
Step 1: 编写 utils_profiling.py (公共工具)
  └── 功耗采样线程, CUDA 计时, 结果输出格式

Step 2: 编写 bench_idle_power.py → 跑一次 [P0-5]    (~5 min)
  └── 验证功耗采集链路正确

Step 3: 编写 bench_prefill_af.py → 跑一次 [P0-1]    (~3-4 hours)
  └── 需要: 模型加载 (Llama 70B), A/F 分离计时方案确认

Step 4: 编写 bench_decode_af.py → 跑一次 [P0-2]     (~2-3 hours)
  └── 可与 Step 3 合并为一个脚本 (prefill+decode 一起跑)

Step 5: 编写 bench_af_comm.py → 跑一次 [P0-3]       (~30 min)
  └── 需要多 GPU 环境

Step 6: 数据验证
  └── 与旧数据 (prefill_data_v0.txt, decode_data.txt) 的延迟列对比
  └── 验证趋势一致性
  └── 更新 Pareto 图 (用真实功耗替代 f^α 代理)

总计: ~1 天 (脚本编写) + ~1 天 (数据采集) + ~半天 (验证)
```

### Phase 2: 模型构建与验证

#### [P1-1] 能耗模型验证与拟合

```
任务:
  1. 用 [P0-1/P0-2] 的真实功耗数据拟合:
     - power(f) = a × f^α + b，分别为 A 和 F 拟合 α_A, α_F
     - 评估查找表 vs 线性回归 vs f^α 公式的 MAPE
  2. 确定 Tier 2 使用哪种功耗预测方法:
     - 查找表 + 线性插值 (BiScale 方案, Prefill 功耗 MAPE 4.1%)
     - GBDT (throttLL'eM 方案, IPS MAPE 2.8-5.8%)
     - 简单回归 (BiScale Decode 功耗 MAPE 1.0%)
  3. 验证 Pareto 图结论在真实功耗下是否仍然成立
     (即 AF grid 是否仍然 Pareto 优于 unified DVFS)
```

#### [P1-2] 吞吐量 Profiling (ILP Throughput 参数)

```
任务:
  对每个 (池类型, tp, freq) 配置:
    - 用递增的 RPS 压力发送请求
    - 找到满足 SLO 的最大吞吐 → Throughput_c(tp_c, f̄_c)
    - 记录 goodput vs RPS 曲线
  
  或使用 BiScale 的方法:
    - 构建频率感知的迭代级模拟器
    - 二分搜索确定每个配置的最大可行 goodput
```

### Phase 3: 系统实现

#### [P2-1] SGLang AF 分离推理引擎

```
任务:
  - Attention 和 FFN 在不同 GPU 组上执行
  - A→F 中间激活传输 (hidden states via NCCL/P2P)
  - KV cache 管理 (Attention GPU 上)
  - AF 流水线调度器
```

#### [P2-2] NVML 频率控制接口集成

```
任务:
  - 封装 SetGpuLockedClocks 为 Python 可调用接口
  - 每个 GPU 独立设频, 不影响同节点其他 GPU
  - 验证推理中动态切频不导致 CUDA 错误或精度问题
  - 实现惰性切频策略 (Decode 侧)
```

#### [P2-3] Tier 1 ILP 求解器

```
任务:
  - 基于 PuLP/Gurobi 实现四池联合 ILP
  - Profile 表加载与查询
  - Pareto 剪枝 + Warm Start
  - 负载分桶
```

#### [P2-4] Tier 2 DVFS 控制器

```
任务:
  - Prefill: (f_A, f_F) 联合搜索 (SLO slack-aware)
  - Decode: per-batch slack-aware + 惰性切频
  - 功耗预测模型在线推理
  - SLO 违反回退机制 (升频到最高)
```

### Phase 4: 评估

#### [P3-1] Trace 准备

```
任务:
  - 获取 Azure LLM trace 或 ShareGPT/LMSYS 公开 trace
  - 提取: 到达率 λ(t)、输入长度分布、输出长度分布
```

#### [P3-2] Baseline 实现

```
需要准备:
  - Max-Freq: 所有池最高频率 (DistServe 风格)
  - Unified DVFS: 同频给 A 和 F
  - PD-Only: PD 分离但无 AF 分离 (BiScale 方案)
  - Hierarchical: 分层优化 (先 P/D 再 A/F)
  - Fixed-Ratio: A/F=1:1 固定, 仅优化频率
  - Joint (本文): 联合优化 P/D+A/F 配比+频率
```

### 全部工作项 (按执行顺序, 含具体产出和时间)

```
═══════════════════════════════════════════════════════════════════
 Phase 0: 基础设施准备 (~1 天)                       ← 阻塞 Phase 1
═══════════════════════════════════════════════════════════════════

 [T0-1] dvfs_ctrl.cpp 新增能耗计数器接口              ⏱ 0.5h
   输入: 无
   产出: dvfs_get_energy_mj() 函数 (封装 nvmlDeviceGetTotalEnergyConsumption)
   文件: benchmark/test_motivation/dvfs/dvfs_ctrl.cpp
   验证: 编译 → 调用 → 确认返回递增 mJ 值

 [T0-2] dvfs.py 新增 get_energy_mj() Python 接口      ⏱ 0.5h
   输入: [T0-1] 完成
   产出: DVFSController.get_energy_mj() 方法
   文件: python/sglang/srt/layers/dvfs.py
   验证: Python 调用 → 确认返回值与 nvidia-smi 功耗一致

 [T0-3] 编写 utils_profiling.py 公共工具               ⏱ 2h
   产出:
     - EnergyMeter 类 (start/stop, 基于能耗计数器)
     - load_single_decoder_layer() (加载单层权重)
     - run_profiling_sweep() (通用扫描循环)
     - 结果写入 TSV 的工具函数
   文件: benchmark/test_motivation/utils_profiling.py

 [T0-4] 验证单层独立调用可行性                         ⏱ 2h
   任务: 加载 LlamaDecoderLayer, 确认:
     - layer.self_attn(...) 可独立调用 (需构造 forward_batch)
     - layer.mlp(...) 可独立调用
     - Decode 模式下 KV cache 可正确填充和使用
     - 不同 TP 下均可工作
   产出: 验证通过的最小可运行示例

═══════════════════════════════════════════════════════════════════
 Phase 1: Profiling 数据采集 (~1 周)                  ← 阻塞 Phase 2-4
═══════════════════════════════════════════════════════════════════

 [T1-1] 编写 bench_prefill_af.py                      ⏱ 3h
   输入: [T0-3] [T0-4] 完成
   产出: Prefill A/F 延迟+能耗 profiling 脚本
   参数: --model, --tp, --freqs, --input-lens, --batch-sizes, --output
   文件: benchmark/test_motivation/bench_prefill_af.py

 [T1-2] 运行 Prefill Profiling                        ⏱ 2-3h (GPU 时间)
   输入: [T1-1] 完成
   命令: 依次以 tp=1,2,4,8 启动, 扫描所有 (input_len, bs, freq) 组合
   产出: prefill_data_v1.txt (~1000 行)
     tp  input_len  gpu_clock  batch_size  P_A_lat  P_F_lat  P_A_energy  P_F_energy

 [T1-3] 编写 bench_decode_af.py                       ⏱ 3h
   输入: [T0-3] [T0-4] 完成
   产出: Decode A/F 延迟+能耗 profiling 脚本 (含 KV cache 填充逻辑)
   文件: benchmark/test_motivation/bench_decode_af.py

 [T1-4] 运行 Decode Profiling                         ⏱ 2-3h (GPU 时间)
   输入: [T1-3] 完成
   产出: decode_data_v1.txt (~900 行)
     tp  input_len  output_len  gpu_clock  batch_size  D_A_lat  D_F_lat  D_A_energy  D_F_energy

 [T1-5] 编写+运行 bench_af_comm.py                    ⏱ 2h (开发) + 0.5h (运行)
   输入: 多 GPU 环境
   产出: af_comm_overhead.txt
     transport  batch_size  seq_len  data_bytes  latency_us  bandwidth_gbps

 [T1-6] 编写+运行 bench_idle_power.py                 ⏱ 0.5h
   输入: [T0-2] 完成
   产出: idle_power.txt (各频率下的空闲功耗基线)

 [T1-7] 数据验证 + 更新 Pareto 图                     ⏱ 2h
   任务:
     - 与旧数据 (prefill_data_v0.txt, decode_data.txt) 延迟列对比, 验证趋势一致
     - 用真实能耗数据替代 f^α 代理, 重新生成 Pareto 图
     - 确认 AF grid 在真实能耗下仍 Pareto 优于 unified DVFS
   产出: 更新后的 plot_prefill_pareto.py, plot_decode_pareto.py 和 figures/

═══════════════════════════════════════════════════════════════════
 Phase 2: 模型构建与分析 (~1 周)
═══════════════════════════════════════════════════════════════════

 [T2-1] 能耗模型拟合                                  ⏱ 3h
   输入: [T1-2] [T1-4] 数据
   任务:
     - 拟合 energy_A(f, input_len, bs) 和 energy_F(f, input_len, bs)
     - 对比查找表 vs 回归 vs GBDT 的 MAPE
     - 分别为 Prefill/Decode × Attention/FFN 拟合
   产出: energy_model.py (模型代码) + 精度报告

 [T2-2] 吞吐量 Profiling                              ⏱ 1 天
   输入: [T1-2] [T1-4] 数据 + 模型实例
   任务: 测量每个 (tp, freq) 配置的最大可行 goodput
   方法: 递增 RPS 压测 或 模拟器二分搜索
   产出: throughput_table.txt (ILP 输入)

 [T2-3] Motivation 论证撰写                           ⏱ 1 天
   输入: [T1-7] 更新后的 Pareto 图
   任务:
     - 量化 AF 差异化调频的能耗节省百分比 (vs unified DVFS)
     - 量化 A/F 负载不对称性随 workload 的变化
     - 撰写 paper Section 2 (Background & Motivation)
   产出: motivation 文本 + 论文图表

═══════════════════════════════════════════════════════════════════
 Phase 3: 系统实现 (~2-3 周)
═══════════════════════════════════════════════════════════════════

 [T3-1] Tier 1: ILP 联合求解器                        ⏱ 3-4 天
   输入: [T2-1] [T2-2] 能耗模型 + 吞吐表
   任务:
     - 基于 PuLP 实现四池联合 ILP (PA/PF/DA/DF)
     - Profile 表加载与查询
     - Pareto 剪枝 + Warm Start + 负载分桶
     - 输入: 负载参数 (λ, N_active, 长度分布, SLO)
     - 输出: (k_PA, k_PF, k_DA, k_DF, tp_*, f̄_*)
   产出: tier1_solver.py
   验证: 用 [T1-2][T1-4] 的 Profile 数据, 验证 ILP 有可行解

 [T3-2] Tier 2: DVFS 控制器                           ⏱ 2-3 天
   输入: [T2-1] 能耗模型
   任务:
     - Prefill: (f_A, f_F) 联合搜索 (O(36) 全枚举, SLO slack-aware)
     - Decode: per-batch slack-aware + 惰性切频
     - SLO 违反回退机制 (升频到最高)
   产出: tier2_dvfs.py

 [T3-3] AFD + DVFS 集成                               ⏱ 3-4 天
   输入: [T3-2], SGLang AFD 代码 (已有)
   任务:
     - 在 AFD pipeline (model_forward_afd) 中集成 DVFS 控制
     - forward_afd_A 前后调用 lock_sm_clock(f_A)
     - forward_afd_F 前后调用 lock_sm_clock(f_F)
     - 惰性切频逻辑 (Decode 侧避免频繁切换)
   产出: 修改 afd_mixin.py / afd.py + 新增 dvfs_scheduler.py
   注意: SGLang AFD 引擎已存在 (afd.py + afd_mixin.py),
         不需要从零实现 AF 分离, 只需集成 DVFS 控制

 [T3-4] Tier 1 + Tier 2 + Monitoring 联调             ⏱ 2-3 天
   任务:
     - Tier 1 输出配置 → 启动 AFD 实例 → Tier 2 运行时调频
     - Monitoring 模块: SLO 违反率、A/F 利用率、负载变化检测
     - 触发 Tier 1 重规划的完整闭环
   产出: 端到端可运行的系统原型

═══════════════════════════════════════════════════════════════════
 Phase 4: 评估 (~1-2 周)
═══════════════════════════════════════════════════════════════════

 [T4-1] Trace 准备                                    ⏱ 1 天
   任务:
     - 获取 Azure LLM trace 或 ShareGPT/LMSYS 公开 trace
     - 提取: 到达率 λ(t)、输入/输出长度分布
     - 构造不同负载模式 (高/中/低, 长请求/短请求为主)

 [T4-2] Baseline 实现                                 ⏱ 2 天
   需要实现的对比系统:
     ① Max-Freq: 所有 GPU 最高频率 (DistServe 风格)
     ② Unified-DVFS: 同频给 A 和 F (统一调频)
     ③ PD-Only: PD 分离但无 AF 分离 (BiScale 方案)
     ④ Hierarchical: 分层优化 (先 P/D 再 A/F)
     ⑤ Fixed-Ratio: A/F=1:1 固定, 仅优化频率
     ⑥ Joint (本文): 联合优化 P/D+A/F 配比+频率

 [T4-3] 端到端实验                                    ⏱ 3-4 天
   实验矩阵:
     - 6 个系统 × 3+ 种负载模式 × 2 种 SLO 设置
   采集指标:
     - 总能耗 (mJ), 能效 (tokens/J)
     - TTFT P99, TPOT P99, SLO 满足率
     - GPU 频率分布, A/F 利用率
   产出: 评估结果表格 + 论文图表

 [T4-4] Ablation 实验                                 ⏱ 2 天
   实验:
     - Tier 1 alone vs Tier 2 alone vs Combined
     - 联合优化 vs 分层优化 (量化跨层耦合收益)
     - AF 分离的增量收益 (Joint vs PD-Only)
     - 不同频率档数的影响 (4档 vs 6档)
   产出: ablation 结果 + 分析

 [T4-5] 论文撰写                                     ⏱ 1-2 周
   基于上述所有产出撰写完整论文

═══════════════════════════════════════════════════════════════════
 总计: ~6-8 周
═══════════════════════════════════════════════════════════════════
```

### 依赖关系图

```
T0-1 → T0-2 → T0-3 ─┬→ T1-1 → T1-2 ──┐
                      │                 │
            T0-4 ─────┼→ T1-3 → T1-4 ──┼→ T1-7 → T2-1 → T3-1 ─┐
                      │                 │         T2-2 ─────────┤
                      └→ T1-5           │         T2-3          │
                         T1-6           │                       │
                                        └→ T2-1 → T3-2 → T3-3 → T3-4
                                                                   │
                                           T4-1 ──────────────────┤
                                           T4-2 ──────────────────┤
                                                                   ↓
                                                          T4-3 → T4-4 → T4-5
```

### 关键路径

```
T0-1 → T0-2 → T0-3 → T0-4 → T1-1 → T1-2 → T2-1 → T3-1 → T3-3 → T3-4 → T4-3 → T4-5
(基础设施)        (Profiling)    (建模)   (系统实现)              (评估)    (论文)
  1天                1周          1周        2-3周                1-2周     1-2周
```

### 立即可开始的工作

```
可并行启动:
  [T0-1] dvfs_ctrl.cpp 新增 dvfs_get_energy_mj()     ← 现在就可以做
  [T0-4] 验证单层独立调用可行性                        ← 现在就可以做
  [T4-1] Trace 准备 (下载 + 分析)                     ← 现在就可以做
```

---

## 八、保留工作

### 保留 1: 动态部署模式切换 (Paper 2)

> 在不同负载下动态选择部署模式 (全 AF 分离 / 仅 PD 分离 / 仅 AF 分离 / 全共置)，将部署模式本身作为能效优化的决策变量。不同负载场景适合不同模式:
>
> | 负载场景 | 最优部署模式 |
> |----------|------------|
> | 高负载、A/F 频率差异大 | 全分离 (PA+PF+DA+DF) |
> | 中负载、AF 通信开销大 | PD 分离 (P 池 + D 池) |
> | P/D 负载均衡 | AF 分离 (A 池 + F 池) |
> | 低负载 | 全共置 (统一池) |

### 保留 2: 分层优化方案 (Tier 1 替代设计)

> 将 Tier 1 的联合优化替换为分层优化，作为对比方案或简化实现:
>
> ```
> Level 0: P/D 资源划分 (~10-30 min)
>   • 依据: 长期负载趋势，匹配 Prefill/Decode 吞吐到工作负载
>   • 决策: n_P, n_D
>   • 方法: 复用 BiScale/DistServe 的 provisioning 方案
>
> Level 1: A/F 配比 + 基线频率 (~几分钟)
>   • 输入: n_P (来自 Level 0)
>   • 决策: n_PA, n_PF, f̄_PA, f̄_PF (使得 n_PA + n_PF = n_P)
>   • 同理: n_DA, n_DF, f̄_DA, f̄_DF (使得 n_DA + n_DF = n_D)
>   • 目标: 在给定 P/D 资源下最小化能耗
>
> Level 2: Per-iteration DVFS (同 Tier 2, 不变)
> ```
>
> **优点**: 分解降低求解复杂度，P/D 层可直接复用已有方案，层次清晰易理解。
>
> **缺点**: P/D 分配与 A/F 配比存在耦合 (资源竞争、能耗跨池转移)，分层分解可能丢失全局最优解。
>
> **用途**: 在 Evaluation 中作为 ablation baseline (Hierarchical)，量化联合优化相对于分层优化的增量收益。