# Hybrid 调度：机制分析、最小反例与条件化收益空间

日期：2026-09-08。对应[主 motivation 文档](hybrid_state_aware_admission_motivation.md)的 M1/M2/M3。

本轮交付是机制命题、证明/反例、可执行穷举及收益边界，不是新 scheduler，也没有运行新的 GPU 实验。下文数字除第 7 节的历史实验外，全部来自明确假设下的有限模型，单位不是秒。命题在模型内成立不等于已经在 Kimi-Linear 上确认，更不等于论文新颖性成立。

## 1. 结论先行

| Motivation | 本轮得到的机制结论 | 最小反例 | 收益结论 |
|---|---|---|---|
| M1：token 计量的资源盲区 | 相同 token 量不能决定 state 可行性；相同累计 state-time 也不能决定准入阻塞 | 相同租户 P/O 总量、两种请求组织：新租户完成时间 5 对 1 | 改变 workload 不是合法优化收益；已满且不可抢占时，queue-only 收益为 0 |
| M2：容量准入与服务后果 | 对 token-service 的约束不能单独固定 wall-clock 时间；合批可能是必要权衡，也可能产生可避免损失 | 两个单步请求，比较合批与分批的全部 3 个调度 | 当 tau(2)>2*tau(1)、允许一 token 服务差及延迟准入时，最差完成时间可改善；反条件下收益为 0 |
| M3：静态校准的边界 | 请求 shape 相同而环境不同，固定执行时间预测必有误差；同一候选集合的较好准入顺序可随邻居反转 | X/Y 等候，Z 在跑，只改变 Z 上下文；全部 2 个 work-conserving 调度可枚举 | 某例平均完成时间改善 14.51%，makespan 不变；但最差 slowdown 变差，不能称为公平/隔离收益 |

目前建立了“有条件的问题存在性”，而没有证明“当前真实 workload 的全部损失可被修复”。收益为零的例子也是结论，不以寻找正收益替代验证。

## 2. 对象、代码事实与分析边界

### 2.1 实际检查的实现

本地 HEAD 为 `0077f84d37a013213fb59700564e1637987a3f79`，工作树包含已有修改。不能用该 commit 单独代表实验版本；本轮结果另记录实际 `schedule_policy.py` SHA-256。

| 代码位置 | 本次读到的行为 | 分析含义 |
|---|---|---|
| [schedule_policy.py](../../../python/sglang/srt/managers/schedule_policy.py#L475)，`_sort_by_tenant_vtc` | 先选 shadow counter 最小租户，同值按首次队列顺序；选中后才增加 request/input cost | 它不是按候选请求预测 JCT 排序的 shortest-job-first |
| 同文件 `_vtc_admission_cost` / `_vtc_output_cost`，第 500/508 行 | request/input/P² 在 admission 记账，output/PO/O² 在 decode 更新 | 固定成本数值和未来输出记账时机是不同因素 |
| [scheduler.py](../../../python/sglang/srt/managers/scheduler.py#L3307)，第 3307--3449 行 | 先排等待队列，再由 PrefillAdder、容量与并发检查选出 can-run 集合 | 排序和可行性检查是两步；队列分数不等于运行时间控制 |
| [kv_cache_configurator.py](../../../python/sglang/srt/mem_cache/kv_cache_configurator.py#L1813)，`_calculate_mamba_ratio` / `resolve_max_num_reqs` | 关闭 radix 的 planning ratio 为 1；其他路径考虑额外 slot，max-running 受 state cap/ratio 限制 | 需要分离 active state 与缓存/缓冲预留，不能将所有 slots 视为 active requests |
| [schedule_policy.py](../../../python/sglang/srt/managers/schedule_policy.py#L977)，joint budget 逻辑 | unified pool 已将新增 state 对共享空间的需求纳入预算 | 不声称 SGLang 忽略 state 安全，也不假定所有配置均为独立物理池 |

验证程序从当前文件 AST 提取未改写的 queue/cost 方法执行，避免导入 CUDA。它验证的是这些方法，不是完整 scheduler、counter-lift 的可达状态或远端 GPU 执行。

### 2.2 与 VTC 理论的关系

[VTC v2](https://arxiv.org/html/2401.00588v2) 已讨论 variable capacity、不同 cost、输出预记账及内存限制的权衡。其 Def.4.1/Theorem 4.4 的 backlog 指等待队列持续非空；Theorem 4.11 还给特定低载请求的 dispatch-delay bound，依赖服务速率下界。不能笼统说 VTC 没有任何延迟分析。这里不反驳这些定理，也不把“执行成本随 batch 变化”当作其未认识的现象。

本轮区分三种成本：token service 是服务计量，state/KV 是容量占用，execution time 是共享设备上请求实际经历的时间。三个量不应互相替代。下文有限窗口的服务预算不是原论文持续 backlog 定理的复现。

### 2.3 有限模型的共同假设

- 单实例；不复用 prefix；每个运行请求占一个有效 active-state slot；KV 可按 P+O 预留，避免增长超额。
- 到分析起点所有请求已可用；部分请求已准入，记为初始 running set。它们之前的 prefill 已发生，分析的是起点之后的剩余完成时间，不是端到端 JCT。
- 在 decode iteration 边界准入；所有 running 请求每轮生成一个 token，完成后释放；不暂停、不抢占、不丢请求、不改变请求内容。
- 基本模型新增 prefill 时长为零；M3 另分析串行 prefill 时长 kappa*P，其间原 running 请求暂停 decode。两者都只是可检验的执行近似。
- 输入在 admission 计入服务，输出在 iteration 结束计入，权重均为 1。同一 batch 的输出原子更新；模型权重不同于历史实验的 1/2。
- 时间函数为显式假设，不由历史 queue/active 均值反推；不模拟 MoE routing、CUDA graph、chunking、TP 通信或真实 KDA kernel。
- “穷举最优”只指列明的有限动作集合；不包含连续时间任意暂停、未枚举的缓存操作或在线算法泛化。

## 3. M1：token 相同为何仍会出现不同准入后果

### 3.1 命题 A：token-only 无法判定 state 可行性

设每个 active request 占固定 state 字节 S，当前 resident tokens 为 T。资源快照近似为：

\[
M=SN+KT.
\]

独立有效容量约束写为 N<=C_s、T<=C_k。对于 N_1<=C_s<N_2、T 相同且不超过 C_k 的两个状态，任何仅观察 T 的判定器都给出相同答案，而正确可行性不同。这构成信息不足的反例。

最小例子：C_s=1、C_k=4，`[4]` 和 `[2,2]` 都有 4 resident tokens，但前者可行、后者不可行。负对照：C_s=2 时二者都可行；S=0 时没有 fixed-state 字节差。

该命题只否定 token-only 的充分性。`S*N+K*T` 本身就是 scalar，统一字节池还可能适合这种表达；不能据此证明所有 scalar 无效或向量公平理论必需。

### 3.2 最小后果例子：资源差异实际阻塞另一个租户

设 C_s=2、C_k=64，每轮时间恒为 1，不随 batch 变化，以排除计算干扰。租户 A 已被准入、还未开始本轮 decode，新租户 B 在分析起点到达，B=(P=1,O=1)。

比较 A 的两种形态，aggregate input=2、aggregate output=8 完全相同：

| A 的初始 running set | 占用 slots | B 的行为 | B 剩余完成时间 |
|---|---:|---|---:|
| packed：一个 (P=2,O=8) | 1 | 立即准入，下一轮完成 | 1 |
| split：两个 (P=1,O=4) | 2 | 等待 4 轮，A 完成后准入 | 5 |
| split，C_s 改为 3 | 2 | 立即准入 | 1 |

KV 不构成限制；A 的已计输入服务量、resident input 总量和未来 output 总量相同。仅仅 active-holder 数不同就足以改变 B 的准入延迟。

一般将每个 split 请求的剩余 output 设为 q，B 的等待差为 q。固定请求上限内这是有限差异，不能用 q 无界宣称真实系统等待无界。

**具体问题：token 服务/总量不能说明一个租户已经占用了多少不可立即释放的请求级容量。** 即使 B 的 counter 更低，也没有可用 slot 兑现准入优先级。

### 3.3 必须保留的负结论：peak state 不等于 state-time

在这个 flat-step 模型中，A 的 state-time 分别为 S*1*8 和 S*2*4，完全相同。更一般地，将 O=f*q 拆成 f 个 O=q 请求并同时执行：

\[
\frac{A_{state}^{split}}{A_{state}^{packed}}
=\frac{S f q\tau(f)}{S f q\tau(1)}
=\frac{\tau(f)}{\tau(1)}.
\]

若 tau(f)=tau(1)，累计 state-time 相同，但峰值 slots、释放时刻和邻居准入不同。prefill/reservation/排队额外驻留不在这条公式内。

**Insight：不但 token 总量不足，单一累计 resource-time 也可能掩盖何时占满、何时释放的时序差异。** 这说明应测时间轨迹，不证明需要预测 lifetime 或按 resource-time 收费。

### 3.4 收益空间：此时 queue-only 为零

在 split 已占满两个 slots、不可抢占、四轮前无人完成的同一初始状态下，任何只改变 waiting order 的策略都不能提前准入 B。完整枚举也只得到 B 最早完成于 5。

因此，packed 对 split 的 5→1 不是算法收益：它改变了 workload；C_s=3 对 2 也改变了资源条件。要讨论此前少准入一个 A 的收益，必须另外考虑当时 B 尚未到达、是否允许闲置、对 A 的影响及到达不确定性。本轮不把它假设成免费的修复。

## 4. M2：何时是必要权衡，何时有可避免损失

### 4.1 命题 B：计量服务相同不固定 wall-clock 时间

固定一条离散事件轨迹，将事件时间全部乘 kappa>0，同时保持到达/准入/完成事件顺序不变。每个对应事件窗口的 token-service 差不变，完成时间和 resource-time 则乘 kappa。

所以仅有服务差界不能导出不依赖执行速率的绝对时间界。这不是 VTC 的反例：其 dispatch bound 显式依赖容量速率下界，该下界也会随时间缩放改变。

对运行请求 i，一次加入候选 j 的单轮外部影响可写为：

\[
\Delta\tau(j\mid B)=\tau(B\cup\{j\})-\tau(B).
\]

重叠多少轮以及各轮 batch 如何变化还影响总干扰，不能用一次 delta 直接代表完整请求成本。此处 externality 是时间后果，不是已经确定应向谁收费的资源债务。

### 4.2 最小二请求模型与全部调度

A/B 各有一个剩余 decode token；共同或已完成的 prefill 从例子中剥离。硬容量允许两者同跑，tau(1)=1、tau(2)=r>0。所有请求最终均完成。

| 调度 | 完成时间 (A,B) | Mean | Worst / makespan | 最大 token-service 差 |
|---|---|---:|---:|---:|
| 合批 | (r,r) | r | r | 0 |
| A 后 B | (1,2) | 1.5 | 2 | 1 |
| B 后 A | (2,1) | 1.5 | 2 | 1 |

这些就是不抢占、只在边界准入且不空转设备的全部 3 个调度。分批保持 GPU 有工作，但主动留下可容纳请求的 slot，**不满足严格 admission work-conservation**。

若允许服务差预算 delta>=1 和延迟准入，则相对合批的最差完成时间收益上限在这个动作集合内精确为：

\[
H_{worst}=\max(0,1-2/r).
\]

若 delta<1，或要求只要有空位就必须准入，则这里只有合批合法，收益为 0。

- r=1：合批同时更快且服务差更小，不应限流。
- r=1.5：合批 (1.5,1.5)，分批 (1,2)；保护 A 会拖慢 B，makespan 变差。是权衡，不是免费收益。
- r=2：分批能降低 mean 但不能降低 worst/makespan；收益取决于目标。
- r=2.5：分批 (1,2) 对比合批 (2.5,2.5)，两请求都更早，worst/makespan 改善 20%，mean 改善 40%；前提是上述服务预算和动作允许。

如限定 makespan 相对合批最多增加 epsilon，分批还必须满足 `2 <= (1+epsilon)*r`。平均完成时间的条件化收益为 `max(0,1-1.5/r)`，不满足约束则为零。不能只报一个不带约束的百分比。

### 4.3 机制结论与适用边界

**具体问题：容量可行并不能判定 batch 是否有利于指定的服务目标。** 合批发生超线性代价时可能留下可避免损失；亚线性或近线性区间则可能只是不同租户/目标之间的权衡。

该反例不依赖 KDA，普通 Transformer 也可能满足时间条件。Hybrid 的潜在额外作用是改变哪些 batch 容量可行、何时形成，而非创造了 batching 干扰本身。历史 peak batch 和平均 interval 不能用来直接估计这里的 r。

在所有候选都被一次准入、且 prefill 次序成本也不变的情形，仅修改等待队列分数无法改变这个运行集合。Request tax、输出预测或更复杂的 counter 并不自动产生新的控制空间。

## 5. M3：静态成本的误差是否真的影响决策

### 5.1 命题 C：固定预测的误差下界

若同一个请求在两个环境下的 execution time 为 c_1<c_2，而估计器只读取固定 shape，因此必须输出同一个常数 c_hat，则：

\[
\min_{\hat c}\max\left(
\frac{|\hat c-c_1|}{c_1},\frac{|\hat c-c_2|}{c_2}
\right)=\frac{c_2-c_1}{c_2+c_1}.
\]

证明：误差不超过 e 要求两个区间 `[(1-e)c_1,(1+e)c_1]` 和 `[(1-e)c_2,(1+e)c_2]` 相交，得到下界；取 `c_hat=2*c_1*c_2/(c_1+c_2)` 达到它。

这不是所有 scalar 的不可能性：允许读取环境的 scalar 可以不同；若 cost 是服务权益定义而非 execution-time 预测，也不要求拟合这两个时间。更关键的是，预测有误差还不代表调度会选错，需下一例闭合。

### 5.2 三请求、两个可枚举的准入顺序

初始 running 中只有 Z=(P=h,O=2)。等待请求 X=(8,1)、Y=(1,2)，分属不同租户，候选 counter 相同，队列首次顺序 Y/X。C_s=2、C_k=512，两者单独加入都可行，但不能同时加入。

使用一个不含任意 pair penalty 的时间函数：

\[
\tau(B)=1+\sum_{i\in B}(P_i+g_i),
\]

其中 g_i 为该请求已生成的 output 数。它只表示一个可检验的线性 context-cost 假设，不是实测 KDA/MLA kernel 模型。

选择 X 后或 Y 后，其余调度由不抢占和填满可用容量唯一确定：

| 首选 | iteration 序列 | 完成时间 (X,Y,Z) | 完成时间之和 |
|---|---|---|---:|
| X | {Z,X}，{Z,Y}，{Y} | (h+9, 2h+15, 2h+12) | 5h+36 |
| Y | {Z,Y}，{Z,Y}，{X} | (2h+15, 2h+6, 2h+6) | 6h+27 |

两种 makespan 始终同为 2h+15。`sum_X - sum_Y = 9-h`：h<9 时 Y-first 较好，h>9 时 X-first 较好，h=9 相同。

| 背景 h | X-first mean | Y-first mean | 较好顺序 | 相对另一顺序 mean 改善 |
|---:|---:|---:|---|---:|
| 1 | 13.667 | 11 | Y | 19.51% |
| 9 | 27 | 27 | 相同 | 0 |
| 100 | 178.667 | 209 | X | 14.51% |

对于这两个初始环境，候选 shape、候选 counter、FIFO 顺序和剩余 slot 数完全相同，只有 running context 不同。任何仅依赖这些候选静态信息且看不到 h 的确定性规则都会在两环境做相同首选，因此不能在两者都达到最小 mean。这是**受限信息集的反例**，不是所有 VTC 变体或所有静态算法的下界。

### 5.3 与真实 queue 实现的对应

从实际代码提取的方法验证：在 `counter_X=counter_Y=0`、队列 `[Y,X]` 下，token、request-tax 和另一组 quadratic 权重都得到 `[Y,X]`。其原因是先根据 counter/tie 选租户，然后才加该请求的成本；不是“profile 预测 Y 更短所以选 Y”。

本例因此暴露了一个具体待测机制：**公平分数相同的请求，准入 tie 的选择也可能具有 batch-dependent 性能后果，而成本校准不一定改变这个 tie。** 只改 tie 可能就处理这个例子，不能因此声称需要完整新调度框架。

输出预测并没有在本轮提取测试中实现。它可能改变历史 counter、后续选择和真实场景，不能用这个人工固定 counter 的例子排除其整体收益。

### 5.4 收益空间与反收益检查

该例的服务定义是 input+output，Z 初始 input 已计入，不重新收费。X/Y 最终服务分别 9/3，所有请求和总工作完全不变。X-first 的最大 X/Y 服务差为 9，Y-first 为 6。

若允许共同有限窗口预算 delta=9，两种顺序均合法。h=100 时，Y-first 到 X-first 的 mean 收益为 `91/627=14.51%`，makespan 都为 215。这个动作集合中它是精确最优，不是对任意调度的全局上界。

但若要求不扩大现有的实际 gap，或冻结 delta=6，则 X-first 不合法，收益为零。这个预算只核查 X/Y 的有限波次差；Z 的完成时间被计入性能，但没有建立三租户稳态 token 公平保证。仅最终服务相同不足以证明公平不变。

更重要的是，h=100 的孤立剩余执行时间分别为 X=9、Y=5、Z=203：

- X-first worst slowdown ratio = max(109/9,215/5,212/203) = 43。
- Y-first worst slowdown ratio = max(215/9,206/5,206/203) = 41.2。

**Mean 的改善伴随 worst slowdown 恶化约 4.37%，worst completion 没有改善。** 因此本例没有证明当前主线所关心的 worst-tenant isolation 有正收益，更不能用它替代 p95 实验。

若 h 增大，在 delta>=9、同样的非抢占填满策略集合中，相对 Y-first 的 mean 收益为 `max(0,(h-9)/(6h+27))`，趋近 1/6。不能把 16.67% 当真实系统上限。

### 5.5 非零 prefill 和其他负对照

加入新增 prefill 时长 kappa*P，期间 decode 等待；初始 Z 的 prefill 是已付成本。则：

\[
\Sigma C_X=5h+36+26\kappa,\qquad
\Sigma C_Y=6h+27+11\kappa.
\]

两者 makespan 仍同为 `2h+15+9*kappa`，顺序反转条件变为 `h>9+15*kappa`。

在 h=100 时，kappa=1 仍有 `76/638=11.91%` mean 收益；kappa=7 时这个收益消失。这说明反例不只依赖零 prefill，但也不对任意 prefill 代价成立。若把 C_s 放宽为 3，两候选都被准入，枚举只有一个 running-set 选择，queue-order 收益为零。

单独 X 的执行时间为 9，h=100 且 X 首选时为 109。代入命题 C，任何固定 X 时间估计的两环境最坏相对误差至少为 `50/59=84.75%`。这些是假设时间函数下的数字，不能写成已测 profiled VTC 的误差。

## 6. 从反例得到的问题、观察与 insight

| 编号 | 已在模型内证明的问题 | Insight | 尚不能宣称 |
|---|---|---|---|
| M1 | token 相同可对应不同的 active-slot 占用及邻居准入延迟；积分相同仍可能释放时刻不同 | 不可即时释放的并发占用与计量服务量不同 | 固定 state 必然增加 state-time；排队重排一定能修复 |
| M2 | 同一容量可行集合中，合批可能产生执行外部影响；存在可避免损失的充分条件，也存在零收益区间 | 必须把容量、时间、服务预算分开分析 | 所有大 batch 都低效；实测 9→96 已满足二请求反例条件 |
| M3 | 静态候选信息无法区分某些较好顺序相反的环境，单独校准可能不改变决策 | 执行环境与实际决策的联系比预测误差大小更关键 | 有 mean 收益就有 fairness/worst-slowdown 收益；新 ledger 必需 |

最值得保留的逻辑线是：**token 计量不足以描述运行承诺；运行承诺与 batch 共同影响后续服务；但能否改善还受动作空间和服务目标限制。** “问题存在”与“给定策略可修复”必须分别下结论。

## 7. 与已有实测的对应，不混合证据等级

| 理论环节 | 已有证据 | 还需要真实验证的预测 |
|---|---|---|
| M1 state occupancy → admission | equal-token state/KV 差异及 cap/并发实验 | 控制 radix 与 max-running，记录 simultaneous footprint、拒绝原因和下一次 state 释放；确认类似 slot wall 真实发生 |
| M2 admission → execution | 9→96 并发干预使 queue 下降、active/interval 上升、packed p95 恶化；makespan 同时改善 20.29% | 对固定请求/背景测 batch 时间函数，区分必要权衡与可避免区间；不能以平均 interval 比值充当 tau(2)/tau(1) |
| M3 execution environment → decision loss | 已测 profile 改善有限，instant-share 效果随并发改变 | 相同候选/计量状态下改变背景组成，观察首选改变能否改善冻结的目标；分别核查 prefill、输出预记账、tie 和候选全准入 |

真实收益应相对同一 workload、同一硬件容量、同一 token-service 要求和动作限制计算。确认例子出现后再讨论扩展，不从 toy model 宣称已获得真实吞吐或公平收益。

本轮没有将旧 O2/O3 或原 M3 gates 改写为新的实测 PASS。也没有推翻之前 lifetime/ranking held-out 的负结果。

## 8. 可复核产物与完成检查

- [精确模型与源码方法验证程序](verify_mechanism_counterexamples.py)：使用 Fraction，穷举有限 admission 选择，输出代码指纹与有理数结果。
- [测试](test_mechanism_counterexamples.py)：12 项，包含独立闭式公式、容量安全、服务预算、负收益、零收益、非零 prefill、时间缩放和真实方法执行。
- [结果快照](results/mechanism_analysis_20260908.json)：仅模型结果，无 GPU 实验指标。

运行：

```sh
python3 -m unittest discover -s benchmark/test_motivation/Ledger -p test_mechanism_counterexamples.py -v
python3 benchmark/test_motivation/Ledger/verify_mechanism_counterexamples.py
```

完成标准核查：三个 motivation 均有对象/假设、命题/推导、最小例子、反例或负对照、收益或零收益条件；数字由枚举与独立公式交叉验证；真实收益和 hybrid 特异性仍明确属于下一阶段，不伪装成已证事实。
