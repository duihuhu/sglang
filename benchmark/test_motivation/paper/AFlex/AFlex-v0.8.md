# AFlex: Operator-Level DVFS for Energy-Efficient LLM Serving under Attention–FFN Disaggregation

## 前三章写作大纲（OSDI 风格修订版 v5.1）

本文档面向论文 **AFlex** 的前三章写作，目标是形成一份更接近 **OSDI/NSDI/EuroSys** 风格、并与当前论文真实贡献边界严格对齐的写作蓝图。  
相比 v5，本版的关键修正是进一步明确 **Lex 与 AFlex 的包含关系**：

- **Prior work**：DynamoLLM、BiScale 等已有 DVFS 系统
- **Lex**：本文提出的 **optimized unified-frequency frequency-selection algorithm**
- **AFlex**：本文提出的 **operator-aware disaggregated serving framework**
- **AFlex builds on and subsumes Lex**：
  - AFlex 继承 Lex 的优化频率选择逻辑
  - 并在此基础上通过 **Attention–FFN disaggregation** 扩展控制空间
  - 再辅以必要的系统机制，将新增自由度转化为端到端节能收益

因而，本文的核心主张应表述为：

> **Even after optimized unified-frequency control with Lex, the single-frequency-domain abstraction remains structurally limiting; AFlex inherits Lex and breaks this limit through operator-level disaggregation.**

这一定义带来三个叙事上的重要约束：

1. **不要把 Lex 和 AFlex 写成两个平行系统**
  - Lex 是算法贡献
  - AFlex 是建立在 Lex 之上的系统贡献
2. **AFlex 与 Lex 的比较应被解释为“abstraction gap”**
  - 因为 AFlex 继承了 Lex 的核心频率选择逻辑
  - 二者差异主要来自是否暴露 A/F 独立控制自由度
3. **Motivation 与 Design 必须显式体现“Lex inside AFlex”**
  - 先证明为什么需要 Lex
  - 再证明即使使用 Lex，仍需要 AFlex

---

# 1. Introduction

> 正式论文中建议采用 **五段式引言**。  
> 其中第 4 段引入 **Lex** 作为 unified-frequency control foundation，  
> 第 5 段引入 **AFlex** 作为在 Lex 基础上的 operator-disaggregated extension。

---

## 1.1 Why is energy-efficient LLM serving an urgent systems problem, and why is DVFS the right lever?

### 本段回答的核心问题

为什么 LLM serving 的能耗问题重要且紧迫？为什么本文以 DVFS 作为系统切入点？

### Topic sentence 建议

> Large-scale LLM serving has become a major GPU workload in modern datacenters, making energy efficiency a first-class systems concern under strict user-facing latency objectives.

### 应讲内容

- LLM inference 已成为现代数据中心中的重要 GPU 负载
- 在线 serving 长期运行、规模大、能耗高
- 成本与碳排放压力显著
- 在线服务不同于离线作业：
  - 受 TTFT / TPOT 等严格 SLO 约束
  - 本质是 **SLO-constrained energy minimization**
- 现代 GPU 已提供 DVFS 等硬件功耗调控能力
- 关键问题不是有没有 DVFS，而是：
  - serving 系统是否暴露了足够细粒度、足够有意义的控制对象
- 过渡：
  - 如果控制粒度过粗，DVFS 的硬件能力难以变成真实的系统收益

### 推荐加入的关键句

> Modern GPUs already expose DVFS knobs for trading performance for energy efficiency, but whether current LLM-serving stacks expose sufficient control granularity to fully exploit these knobs remains an open systems question.

### 与下一段的逻辑衔接

> 因而接下来的关键问题是：现有 LLM-serving 系统究竟向 DVFS 暴露了哪些粒度的异构性？

---

## 1.2 Why does prior DVFS remain limited by the single-frequency-domain abstraction?

### 本段回答的核心问题

既然已有 DynamoLLM、BiScale 等工作，为什么问题仍未被解决？

### Topic sentence 建议

> Prior LLM-serving systems have progressively refined DVFS control, yet they still treat each GPU as a single frequency domain and therefore miss finer-grained heterogeneity inside a serving phase.

### 应讲内容

- 归纳 prior work 的粒度演进：
  - DynamoLLM：粗粒度
  - throttLL’eM：iteration 级
  - BiScale：phase 级
- 肯定其意义：
  - phase asymmetry 已被证明对 DVFS 有价值
- 但共同前提仍是：
  - **single GPU = single frequency domain**
- 点出隐含假设：
  - 阶段内主要算子可共享同一个 operating point
- 抽象为：
  - **Intra-phase Operator Homogeneity**
- 过渡：
  - 如果这一假设不成立，则 unified per-GPU DVFS 可能带来结构性浪费

### 推荐加入的关键句

> Existing systems have improved DVFS granularity from coarse model-level control to phase-level tuning, but they still implicitly assume intra-phase operator homogeneity by treating each GPU as a single frequency domain.

### 与下一段的逻辑衔接

> 这就引出了本文的核心经验问题：同一阶段内部的主要算子，是否真的足够相似到可以共享统一频率？

---

## 1.3 Why do Attention and FFN break this assumption?

### 本段回答的核心问题

Attention 和 FFN 是否真的不同到足以打破统一频域抽象？

### Topic sentence 建议

> Our key observation is that Attention and FFN respond fundamentally differently to frequency scaling, making a unified operating point an inherently suboptimal compromise.

### 应讲内容

- 高层 profiling 观察：
  - Decode 中，Attention 更快转向 memory-bound
  - FFN 在更宽范围内保持 compute-sensitive
- Intro 中轻量点出根因：
  - Attention 受 KV cache 访问与 arithmetic intensity 下降影响更大
  - FFN 由 dense GEMM 主导，通常保持更高计算强度
- 说明统一频域的结构性折中：
  - FFN 为满足 TPOT 需要较高频率
  - Attention 因而被迫高频运行，造成浪费
- 明确强调：
  - 这不是简单 heuristic 不够好
  - 而是 **single-frequency-domain abstraction** 的限制
- 过渡：
  - 在打破该抽象之前，首先要问 unified-frequency control 自身是否已经被充分优化

### 推荐加入的关键句

> This heterogeneity stems from distinct bottleneck regimes: attention increasingly becomes constrained by memory traffic—especially KV-cache accesses and reduced arithmetic intensity under parallel execution—whereas FFN remains dominated by dense matrix multiplications and therefore retains stronger sensitivity to compute frequency.

### 与下一段的逻辑衔接

> 因而本文首先提出 Lex，在统一频域约束下尽量强化频率选择本身。

---

## 1.4 How far can unified-frequency control go with Lex?

### 本段回答的核心问题

Lex 是什么？它在本文中的角色是什么？为什么必须先有 Lex？

### Topic sentence 建议

> We first develop Lex, an optimized unified-frequency frequency-selection algorithm that strengthens the best achievable efficiency under a single-frequency-domain constraint.

### 应讲内容

- 明确定义 Lex：
  - optimized unified-frequency frequency-selection algorithm
- 指出 Lex 的三重角色：
  1. 它本身是一个算法贡献
  2. 它是 strongest unified baseline
  3. 它是 **AFlex 继承的 control foundation**
- 强调如果没有 Lex，就无法公平区分：
  - unified heuristic 不够强
  - unified abstraction 本身有上限
- 说明 Lex 相比 prior phase-level heuristics 的意义：
  - 更接近 single-frequency-domain constraint 下的 best achievable point
- 末尾过渡到 AFlex：
  - 即使 unified-frequency control 被 Lex 强化，仍存在由单一频域引起的剩余结构性 gap

### 推荐加入的关键句

> Lex is both a practical contribution in its own right and the optimized control foundation inherited by AFlex, allowing us to separate the limitation of unified-frequency abstraction from the weakness of prior heuristics.

### 与下一段的逻辑衔接

> Building on Lex, AFlex extends the control space through Attention–FFN disaggregation and turns the remaining abstraction gap into practical energy savings.

---

## 1.5 What does AFlex add beyond Lex, and what does the paper achieve?

### 本段回答的核心问题

AFlex 在继承 Lex 的基础上增加了什么？论文最终取得了哪些结果？

### Topic sentence 建议

> Building on Lex, AFlex extends optimized frequency control with Attention–FFN disaggregation and delivers substantial additional energy savings in practical serving regimes.

### 应讲内容

- 明确定义 AFlex：
  - operator-aware, SLO-aware serving framework
  - built on top of Lex
  - enabled by Attention–FFN disaggregation
- 点出其核心机制：
  - low-overhead cross-operator state transmission path
  - fast switching-aware frequency scheduler
- 明确比较逻辑：
  - AFlex 与 Lex 共享频率选择基础
  - 因而 AFlex vs Lex 的差距可解释为 **breaking the single-frequency-domain abstraction** 的收益
- 提醒代价：
  - A/F 分离引入通信与控制开销
- 但在 practical regimes 下收益通常覆盖成本
- 放 headline results：
  - Compared with **Lex**, AFlex achieves **5%–14%** additional energy reduction
  - Up to **39%** end-to-end energy savings
  - Median clock-switching overhead around **4.5 ms**
- 给出 contributions，建议改为：
  1. **We identify and characterize** operator-level frequency heterogeneity between Attention and FFN in LLM serving.
  2. **We develop Lex**, an optimized unified-frequency frequency-selection algorithm that strengthens prior phase-level DVFS, serves as a strong unified baseline, and forms the control foundation of AFlex.
  3. **We design and implement AFlex**, an operator-aware, SLO-aware serving framework that extends Lex with Attention–FFN disaggregation, a low-overhead state transmission path, and a fast switching-aware scheduler.
  4. **We evaluate** AFlex on Qwen3-32B/A800, showing that unified DVFS remains structurally suboptimal even under optimized single-frequency control, and that AFlex delivers substantial gains across practical serving regimes.
- 自然引出 Background

### 与下一章的逻辑衔接

> To understand why this opportunity arises and why existing serving architectures cannot capture it, we next review the necessary background on LLM serving, GPU DVFS, and phase-disaggregated serving.

---

# 2. Background

> 本章保留 **2.1 / 2.2 / 2.3** 三节。  
> 原则不变：**中性、事实性、为后文服务**。  
> 但结尾过渡需更自然地承接“Lex inside AFlex”的后续叙事。

---

## 2.1 LLM Serving Pipeline and Latency Objectives

### 本节回答的核心问题

LLM serving 的执行流程与性能目标是什么？这些目标如何约束节能优化空间？

### Topic sentence 建议

> LLM serving consists of multiple execution stages and user-facing latency objectives, both of which shape the design space of energy-aware serving systems.

### 段落组织建议

- 与 v5 基本一致：
  - Prefill / Decode
  - TTFT / TPOT
  - SLO 对 DVFS 的张力
  - Attention / FFN 作为主要算子
- 结论要自然落在：
  - energy optimization 必须是 selective 的
  - 这为更细粒度、算子感知的控制奠定背景

### 建议加入的关键句

> Strict TPOT constraints make energy optimization selective rather than global: the system cannot simply reduce frequency everywhere, and must instead exploit slack on non-critical or frequency-insensitive execution paths.

### 与下一节的逻辑衔接

> 这意味着控制策略的有效性，最终取决于系统是否能够识别并利用不同执行对象对频率的不同敏感性。

---

## 2.2 GPU DVFS Fundamentals

### 本节回答的核心问题

GPU DVFS 如何影响性能与能耗？其收益为何依赖 bottleneck 类型？现实中切频有哪些成本？

### Topic sentence 建议

> GPU DVFS trades execution speed for power efficiency, but its effectiveness depends on workload sensitivity and the practical cost of clock control.

### 段落组织建议

- 与 v5 基本一致：
  - 能耗–频率非单调关系
  - compute-bound / memory-bound 的差异
  - NVIDIA 切频接口与现实开销
- 结论应更明确服务后文：
  - 如果不同执行对象的频率敏感性显著不同，就需要更精细的控制对象
  - 同时算法本身也必须避免落入 U 型能耗陷阱

### 与下一节的逻辑衔接

> 因而问题不仅在于是否做细粒度控制，也在于 unified-frequency control 本身是否被足够好地优化。

---

## 2.3 Phase-Disaggregated LLM Serving

### 本节回答的核心问题

为什么现代系统做 prefill–decode 分离？它暴露了什么异构性，又遗漏了什么？

### Topic sentence 建议

> Phase-disaggregated serving exposes asymmetry between prefill and decode, but still leaves finer-grained heterogeneity hidden inside each phase.

### 段落组织建议

#### 第 1 段：P/D 分离的价值

- DistServe、Splitwise 等通过 P/D 分离降低干扰并改善 SLO
- BiScale 表明 phase-level DVFS 有意义

#### 第 2 段：P/D 分离的边界

- A/F 在阶段内仍通常共享同一 GPU 与同一频率域
- phase asymmetry 被暴露
- operator asymmetry 仍被隐藏

#### 第 3 段：与 Lex / AFlex 的关系

- 提炼为：
  - **Intra-phase Operator Homogeneity**
- 在这一前提下：
  - **Lex** 尽量优化 unified-frequency control
  - **AFlex** 在 Lex 基础上进一步打破该前提
- Motivation 将连续回答：
  1. A/F 差异是否真实存在？
  2. 现有 unified heuristics 为什么不足？
  3. Lex 能把 unified route 推到多远？
  4. 即使如此，AFlex 还能释放多少剩余空间？

### 建议加入的关键句

> Existing phase-disaggregated serving exposes phase asymmetry, but still assumes intra-phase operator homogeneity; Lex strengthens control within this abstraction, while AFlex extends beyond it.

### 与下一章的逻辑衔接

> 这引出了本文 Motivation 的两层必要性：为什么 unified-frequency control 需要先被 Lex 做强，以及为什么即使如此仍需要 AFlex。

---

# 3. Motivation

> 本章任务是建立 **Lex 的必要性** 与 **AFlex 的必要性**，并显式说明：
>
> - **Lex is inside AFlex**
> - AFlex 继承了 Lex 的频率选择逻辑
> - 因而 AFlex vs Lex 的差距能够隔离出 **single-frequency-domain abstraction** 的剩余限制
>
> 相比 v5，本章的重点不是新增结构，而是强化这一“包含关系”。

---

## 3.1 Are Attention and FFN frequency behaviors actually different enough to matter?

### 本节回答的核心问题

A/F 的频率行为差异是否真实、显著，并且足以影响系统设计？

### Topic sentence 建议

> Operator-level profiling reveals that Attention and FFN respond to frequency scaling very differently, and this gap is large enough to materially affect energy-optimal operation.

### 段落组织建议

- 与 v5 基本一致：
  - 实验设置
  - Decode 中 A/F 异构性观察
  - Roofline / AI 解释
  - Prefill 中的补充现象
- 结论：
  - A/F 的确构成值得暴露给 DVFS 的 operator-level heterogeneity

### 图表优先级

- **必须图 1**：A/F sensitivity / Pareto / roofline annotation

### 与下一节的逻辑衔接

> 既然 A/F 的频率敏感性存在系统性差异，下一个问题就是：现有 unified-frequency heuristics 为何仍不足以利用这部分空间？

---

## 3.2 Why do existing unified-frequency heuristics and phase-level policies still fall short?

### 本节回答的核心问题

DynamoLLM、BiScale 以及 lowest-feasible 等 unified-frequency 策略为什么仍不够？

### Topic sentence 建议

> Existing unified-frequency policies improve over coarse-grained control, but they still fall short because heuristic clock selection does not always match the true energy-optimal operating point.

### 段落组织建议

- 与 v5 基本一致：
  - 现有方法边界
  - heuristic 不足
  - U 型能耗陷阱
- 但结尾更明确引出 Lex：
  - 如果不先改进 unified-frequency 频率选择本身，就无法公平分析 abstraction gap

### 图表优先级

- **Figure 2a（前半）**：heuristic / phase-level vs Lex

### 与下一节的逻辑衔接

> 这就引出 Lex：一个更强的 unified-frequency frequency-selection algorithm，既用于改进 unified route，也作为 AFlex 继承的控制核心。

---

## 3.3 How far can unified-frequency control be pushed with Lex, and why does AFlex inherit it?

### 本节回答的核心问题

Lex 到底有多强？为什么 AFlex 要建立在 Lex 之上，而不是另起一套控制逻辑？

### Topic sentence 建议

> Lex substantially strengthens unified-frequency control and serves as the optimized control core inherited by AFlex.

### 段落组织建议

#### 第 1 段：Lex 的角色

- Lex 是：
  - optimized unified-frequency frequency-selection algorithm
  - stronger than prior heuristics
  - strongest unified baseline
  - control core reused by AFlex

#### 第 2 段：Lex 的经验效果

- 与 BiScale / heuristic 比较：
  - 节能更优
  - 更少落入 U 型低频高能耗区域
  - 在不同 SLO 下更稳健

#### 第 3 段：为什么 AFlex 继承 Lex

- AFlex 不是另一套完全不同的频率选择逻辑
- 而是：
  - **在 Lex 的控制基础上**
  - 将 unified operating point 扩展为 operator-specific operating points
- 这一点对于后续公平比较至关重要

### 图表优先级

- **Figure 2a（完整）**
- Caption 应明确：
  - Lex closes much of the algorithmic gap within unified-frequency control and forms the control foundation of AFlex.

### 与下一节的逻辑衔接

> 那么，即使 AFlex 与 Lex 共享同一控制基础，单一频域抽象本身是否仍然造成不可消除的能耗浪费？

---

## 3.4 Why does unified-frequency control remain structurally suboptimal even when AFlex inherits Lex?

### 本节回答的核心问题

如果 AFlex 已经继承了 Lex 的优化控制逻辑，为什么 unified-frequency control 仍然存在结构性上限？

### Topic sentence 建议

> Even when built on the same optimized frequency-selection logic as Lex, AFlex achieves lower energy because it breaks the single-frequency-domain abstraction.

### 段落组织建议

#### 第 1 段：核心结果

- 直接给出：
  - Compared with Lex, AFlex still achieves additional energy savings
- 强调：
  - 这不是 algorithm 不同导致的
  - 因为 AFlex 继承了 Lex 的控制核心

#### 第 2 段：差距来源解释

- 最优配置中常呈现：
  - Attention 降频
  - FFN 保高频
- FFN 决定性能下界
- Attention 提供独立降频收益
- unified-frequency control 无法表达该 asymmetric optimum

#### 第 3 段：限定分析范围

- 主分析对象仍为 Decode
- Prefill 作为补充证据
- 结论是：
  - 差距来源于 abstraction，而非 frequency-selection logic

### 图表优先级

- **Figure 2b（必须）**：AFlex vs Lex
- Caption 应明确：
  - Because AFlex inherits Lex’s control logic, the remaining gap isolates the benefit of breaking the single-frequency-domain abstraction.

### 与下一节的逻辑衔接

> 一个更现实的问题是：这部分由 abstraction gap 带来的收益，是否出现在实际有价值的部署区间？

---

## 3.5 Do AFlex’s gains over Lex appear in practical deployment regimes, and where do they fade?

### 本节回答的核心问题

AFlex 相对 Lex 的收益是否出现在真实有价值的部署区间？何时衰减？

### Topic sentence 建议

> The benefits of AFlex over Lex are not uniform, but they concentrate in high-value deployment regimes and diminish in predictable boundary conditions.

### 段落组织建议

- 延续 v5 的 operating-regime 分析
- 但全节比较对象明确固定为：
  - **AFlex vs Lex**
- 收益集中区域：
  - strict SLO
  - medium TP
  - medium / large BS
  - short / medium IL
- 用 traces / 统计 / 文献证明：
  - 这些区间是实际生产区间
- 说明何时衰减：
  - TP 很高
  - IL 很长
  - SLO 宽松
- 结论：
  - abstraction gap 在高价值区间最显著

### 图表优先级

- **Figure 3（必须）**：AFlex gain over Lex in operating regimes
- 建议标出：
  - **Typical Production Workload Zone**

### 与下一节的逻辑衔接

> 既然 AFlex 相对 Lex 的收益在重要部署区间中真实存在，那么最后需要回答的是：A/F separation 为系统设计具体带来了哪些机会与挑战？

---

## 3.6 What opportunities and challenges does AF separation create beyond Lex?

### 本节回答的核心问题

在 Lex 已经优化 unified-frequency control 的基础上，AF separation 额外带来哪些机会与挑战？这些挑战如何映射到 Chapter 4？

### Topic sentence 建议

> Beyond Lex, AF separation creates new operator-level energy-saving opportunities, but also introduces communication, provisioning, and control-overhead challenges that require coordinated system design.

### 段落组织建议

#### 第 1 段：机会总结

- 相对 Lex，AFlex 的额外收益来自：
  - operator-level control freedom
  - asymmetric operating points
  - 避免 unified setting 中的 forced overprovisioning

#### 第 2 段：挑战总结

- **Challenge 1: Disaggregation overhead**
  - A/F 物理分离带来状态传输开销
- **Challenge 2: Coupled provisioning**
  - A/F 配比与 P/D 资源分配耦合
- **Challenge 3: Fast and switching-aware control**
  - 在继承 Lex 的基础上，还需支持更快且切频感知的在线控制

#### 第 3 段：与 Chapter 4 建立闭环

- 建议显式写出：
  - AFlex addresses these challenges by extending Lex with three coordinated system mechanisms...

### 与下一章的逻辑衔接

> These opportunities and challenges together define AFlex’s design space. We next present the design of Lex and show how AFlex extends it into an operator-aware, SLO-aware serving framework under A/F disaggregation.

---

# 前三章整体逻辑链条（v5.1）

## Chapter 1: Introduction

**为什么问题重要，为什么是 DVFS，为什么 unified-frequency control 即使被优化也不够？**  
→ LLM serving 能耗严峻且受 SLO 约束  
→ prior work 暴露了 phase asymmetry，但仍受 single-frequency-domain abstraction 限制  
→ A/F 异构性打破阶段内算子同构假设  
→ 先提出 Lex 强化 unified-frequency 频率选择  
→ 再提出 AFlex，在继承 Lex 的基础上通过 A/F disaggregation 打破 abstraction 上限  
→ 给出 headline results 与贡献

## Chapter 2: Background

**理解论文需要哪些事实性背景？**  
→ LLM serving 的阶段与 SLO  
→ GPU DVFS 的性能–能耗逻辑  
→ phase-disaggregated serving 暴露了什么、遗漏了什么  
→ 为“Lex inside AFlex”的后续叙事铺垫

## Chapter 3: Motivation

**为什么同时需要 Lex 和 AFlex，且为什么 AFlex 必须建立在 Lex 之上？**  
→ A/F 异构性确实存在  
→ 现有 unified heuristics 不够  
→ Lex 将 unified-frequency route 做强，并作为 AFlex 继承的控制核心  
→ 即使共享 Lex，unified abstraction 仍有结构性上限  
→ AFlex 相对 Lex 的收益出现在高价值生产区间  
→ AF separation 值得做，但带来新的系统挑战

---

# 图表优先级总结（v5.1）

## 必须图

1. **Fig. 1：A/F frequency sensitivity + Pareto frontier + roofline annotation**
  - 目标：证明 operator-level heterogeneity exists
2. **Fig. 2a：heuristic / BiScale vs Lex**
  - 目标：证明 Lex 是强 unified baseline
  - 并说明 Lex 构成 AFlex 继承的 control core
3. **Fig. 2b：AFlex vs Lex**
  - 目标：证明 AFlex 关闭的是 abstraction gap
  - 且该 gap 在共享 Lex 控制逻辑的前提下依然存在
4. **Fig. 3：operating-regime heatmap of AFlex over Lex**
  - 目标：证明 AFlex 相对 Lex 的收益集中在高价值生产区间

## 可选图

- LLM serving timeline
- Transformer layer decomposition
- Conceptual energy–frequency curve
- P/D disaggregation sketch
- Opportunities vs challenges summary diagram

---

# 写作原则总结（v5.1）

1. **全文 thesis 必须体现“Lex inside AFlex”**
  - Lex closes the algorithmic gap within unified-frequency control
  - AFlex inherits Lex and closes the remaining abstraction gap
2. **不要把 Lex 和 AFlex 写成平行系统**
  - Lex 是算法核心
  - AFlex 是基于 Lex 的系统扩展
3. **AFlex vs Lex 的比较必须明确解释为 abstraction-gap isolation**
  - 因为二者共享频率选择基础
4. **第三章先证明为什么需要 Lex，再证明为什么即使有 Lex 仍需要 AFlex**
  - 先解决 algorithmic insufficiency
  - 再解决 abstraction limitation
5. **AFlex 的收益表达应尽量围绕“在继承 Lex 后仍然额外获得的收益”**
  - 这样最有说服力
6. **Decode 仍是端到端主要收益战场**
  - Prefill 作为跨阶段异构性的补充证据保留
7. **Chapter 4 开头最好先交代 Lex，再展开 AFlex**
  - 先给出 control core
  - 再给出 disaggregation extension
8. **图 2 的双重任务必须写清**
  - Fig. 2a：Lex 把 unified route 做强
  - Fig. 2b：AFlex 在共享 Lex 基础上仍超越 unified abstraction
9. **Introduction 只放 headline numbers，不提前展开过多细节**
  - 特别是 Lex 与 AFlex 的算法和系统拆解，留给后文

