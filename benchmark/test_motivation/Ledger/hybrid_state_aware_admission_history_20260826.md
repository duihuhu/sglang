# 历史归档：Hybrid-State-Aware Scheduling（截至 2026-08-26）

> 本文件保留早期研究假设、实验和被否定的机制主张，不代表当前设计计划。
> 当前故事线及 M1/M2/M3 见 [主文档](hybrid_state_aware_admission_motivation.md)。
> 以下内容按原记录保留，其中旧 M1--M5 与当前三个 motivation 的编号不同。

> 文献与代码审视日期：2026-08-20  
> 目标模型：`/models/Kimi-Linear-48B-A3B`（20 层 KDA + 7 层 MLA）  
> 本文只讨论两条研究主线：**状态生命周期驱动的请求成本收益选择**，以及**混合主导资源时间上的多租户公平**。

## 0. 执行摘要

Hybrid recurrent-attention 模型改变了 serving 的资源结构：一个 active request 不仅占用随 token 增长的 attention KV，还占用近似固定大小的 recurrent state。因此，系统至少同时受 request/state-slot 数、resident KV token 数和计算资源约束。

当前 SGLang 已经能回答大部分“这个请求能否安全放下”的问题：

- 分离内存池下，分别检查 Mamba/KDA state slot 和 KV token 预算；
- unified memory 下，把 recurrent state 折算进共同内存预算；
- 预算不足时，支持 prefix cache eviction 和 running request retraction。

但当前机制尚未系统回答两个不同的问题：

1. **方向 A：当多个请求都能放下时，先放哪个请求的成本收益更高？**
2. **方向 B：当请求来自多个用户时，如何按 state、KV 和 compute 的实际占用时间做公平分配？**

这两个方向不能只用“短请求更贵”概括。短请求的固定状态摊销可能更高，但其状态释放也可能更快：

\[
\text{instantaneous bytes/token}=\frac{S_i}{L_i}+K_i,
\qquad
\text{state holding cost}=S_iT_i.
\]

第一项随长度变短而增大，第二项却可能随完成时间缩短而减小。调度真正需要比较的是**未来资源时间、完成收益和 SLO 风险**，而不是只比较当前长度或瞬时 bytes/token。

文献审视后的结论是：

- 方向 A 与 output-length-aware scheduling、future memory prediction 和 memory-constrained online scheduling 有明显重叠；不能把“预测剩余长度”本身作为新意。
- 方向 B 与 VTC、DLPM 等多租户公平调度直接冲突；不能声称首次研究 LLM serving fairness。
- 可守住的差异化在于：**固定 recurrent state 与增长型 KV 共同形成的异构资源约束，以及 state-slot-time / KV-byte-time 对选择、retraction 和公平债务的影响。**

建议先分别验证两个方向的关键反例，再决定做一篇统一工作还是拆成两篇。若关键反例不成立，不应继续扩大系统实现。

截至 2026-08-26 的最终实验结论是：**资源表征成立，标准 VTC 的 token currency 在 equal-token partition workload 下不充分，但“state-capacity wall 单独导致 fairness gap”的归因失败。** `fixed state tax`、equal-token resource mismatch 和 token-VTC cost transfer 都可重复；profiled/request-tax scalar VTC 不能稳定修复，而 instantaneous state/KV share 在新矩阵中显著缩小 gap。不过 request-ordering oracle 没有足够效率 headroom，固定 lifecycle ledger 也未稳定优于 instantaneous share。因此当前材料仍不足以支撑原计划的统一系统论文；新的可守故事必须围绕 **partition-aware hybrid resource arbitration**，并显式处理 capacity scarcity 与 batch interference，而不是只给每个请求增加固定 tax。

### 0.1 Preliminary motivation validation status

第一轮可证伪实验已经完成，实验协议见 [motivation validation plan](motivation_validation_plan.md)，逐请求 replay 工具见 [validation harness](validate_hybrid_motivation.py)，三次重复的自动汇总见 [preliminary report](results/preliminary_20260820/preliminary_report.md)。

| 方向 | 初步 gate | 当前证据 | 决策 |
|---|---|---|---|
| A：lifetime request selection | FAIL | `short-first` 相对 `long-first` 的 short p95 JCT 平均改善 11.1%，paired bootstrap 95% CI 为 -14.9% 至 24.3%；10s SLO goodput 无提升 | 暂不实现在线 predictor；先做 state-cap sensitivity 和 resource-time-vs-SJF cross case |
| B：hybrid tenant fairness | RESOURCE PREMISE PASS；VECTOR CLAIM FAIL | exact accounting 的异构资源前提成立；但 9-cell capacity x scalar sweep 中 per-cell oracle scalar 的 mean relative gap 为 0.544，vector 为 0.579（差 6.4%），vector 仅在一个 state-bound cell 改善超过 10% | 暂停 GPU-time/splitting 扩展；若继续，先做同一 run 内动态 bottleneck-switch kill test |

因此，当前证据不支持把两个方向直接拆成两篇论文。方向 A 仍是待验证候选；方向 B 的资源测量 premise 成立，但当前 vector scheduler claim 已被调优 scalar baseline 否定，不能再解释为 VTC/DRF 的一般性失败。

2026-08-21 开始执行下一道 gate：`state-cap x KV-cap x scalar-weight` 交叉实验。这里要验证的不是“某个手选 scalar 是否输给 vector”，而是：(1) 固定 scalar 能否跨 bottleneck 保持稳健；(2) 每个容量点重新调优后的 oracle scalar 是否仍不能匹配 vector。只有第二项在多个 bottleneck regime 成立，才支持 vector resource debt 的不可约性。完整协议见 [validation plan section 4.6](motivation_validation_plan.md#46-state-cap-x-kv-cap-x-scalar-weight-%E4%BA%A4%E5%8F%89%E5%AE%9E%E9%AA%8C2026-08-21)。

### 0.3 Capacity x scalar-weight 结果（2026-08-21）

完整报告见 [capacity-scalar report](results/capacity_scalar_20260821/capacity_scalar_report.md)，原始目录包含 89 个 policy artifact；每个 artifact 都是相同的 204-request sustained trace。

- 3 x 3 容量网格覆盖 6 个 state-bound、2 个 dual-bound、1 个 KV-bound cell；FCFS 校准全部 204/204 完成、0 retraction；
- 基础扫描为 FCFS、7 个 scalar weights（0 至 32,768）和 vector；对一个边界最优 cell 再扩展 8 个 weights，直到 262,144/524,288 附近出现平台、1,048,576 开始回升；
- 89/89 artifacts 的实际 scheduler policy 与标签一致，所有请求完成；只有 KV-bound cell 的 scalar 2,048/4,096/8,192 各出现 1 次 retraction；
- throughput-bounded per-cell oracle scalar 的 mean relative dominant-debt gap 为 0.544，vector 为 0.579，vector 相对 oracle **退化 6.4%**；
- vector 在 3/9 cell 数值更小，但只有 `S=480,K=786432` 一个 state-bound cell 的改善超过 10%（23.2%）；dual-bound 与 KV-bound 的改善只有 1.5% 和 0.8%，单次 screening 下按噪声处理；
- 固定 scalar weight 0 的 mean gap 为 0.623，vector 仅改善约 7.1%。这可以说明 vector 比完全不调参稍稳健，但达不到预设 10% gate，更不能支持 scalar representational insufficiency；
- per-cell oracle weight 从 0、1,024、4,096 到 262,144，说明 scalar 的确有容量相关调参敏感性；但“需要调参”与“无法表达”是不同 claim，当前证据只支持前者。

因此，Topic B 的 **vector-necessity kill test 为 FAIL**。不应继续把 request splitting、GPU-time accounting 或生产 tenant identity 当作当前机制的补全项，因为它们不能修复核心 baseline 已追平/反超的问题。若仍推进，下一道最小实验应改为：固定容量下在同一 run 内切换 state-heavy/KV-heavy arrival mix，比较一个全程固定 scalar 与无需重调的 vector；只有 vector 在动态 bottleneck 切换下稳定改善，才有“自适应而非不可表达”的剩余论文空间。

### 0.4 动态请求成本、batch 外部性与 cache lifetime 结果（2026-08-21）

完整报告见 [dynamic request-cost report](results/dynamic_cost_20260821/dynamic_cost_report.md)。目录包含 108 个正式 artifact 和 3 个 smoke artifact；成本面与排序筛选各重复 3 次，达到 gate 的 batch 外部性和 eviction 结果扩展到 5 次。实验入口为 [dynamic cost harness](validate_dynamic_cost_motivation.py)、[matrix runner](run_dynamic_cost_experiments.sh)、[eviction runner](run_eviction_experiments.sh) 和 [phase runner](run_phase_switch_experiments.sh)。

这组实验把“哪个请求成本收益更高”拆成四个可证伪命题：

| 命题 | 结果 | 关键证据 |
|---|---|---|
| 原始请求长度能否代表成本 | **PASS：不能** | 单条 `P=1K,O=512` 的 JCT 为 2.608s，`P=16K,O=32` 为 1.218s；短 prompt 并不等于便宜请求 |
| prefix reuse 是否改变增量成本 | **PASS** | 8 条并发 `P=16K,O=32` 在无复用时 p95 JCT 为 8.789s，90% 复用时为 2.077s；state-slot-time 与 KV-token-time 都下降约 68% |
| admission candidate 是否产生不同 batch 外部性 | **PASS** | 插入 8 条 long-unique 后，background p95 JCT 增加 70.4%，paired bootstrap 95% CI 为 69.1% 至 71.5%，5/5 次 background 20s SLO goodput 都从 64 降到 0；long-reused 只增加 13.2% |
| cache state lifetime 是否改变未来重算成本 | **PASS** | 在 `KV cap=196,608` 下，KV-heavy pressure 5/5 次把 16K prefix 从全命中变成 0 命中，probe TTFT 放大 25.2x，95% CI 为 24.9x 至 25.5x；state-heavy pressure 5/5 次保留全部 prefix |

但两个机制 gate 没有通过：

- **简单 request order rule FAIL**：三个 cross pair 中，最佳 lifetime order 对 overall p95 JCT 只改善 5.9%，低于 15% gate；state/KV pair 更小；在 reuse pair 中优先 long-reused 反而更差。原因是 prefix hit 降低 prefill 计算，却不消除长序列运行时的 resident KV footprint。不能由“增量 token 少”直接推出“应先调度”。
- **dynamic scalar/vector fairness gate FAIL**：固定 `state=480, KV=196,608`，分别对高拆分 state phase 和低拆分 KV phase 扫描 scalar weight `0..262,144`。最优标签虽有变化，但 phase latency gap 的扫描振幅小于 4%，没有任何固定 weight 产生预注册的 15% regret，因此不触发 switch confirm。它不能挽救 0.3 已失败的 vector-policy claim。

截至 2026-08-21，最准确的结论是：**动态增量成本与 cache lifetime 作为 admission/eviction motivation 已成立，但现有数据尚未证明某个 request-ranking scheduler 能获得足够收益；多租户 vector fairness 方向仍为 FAIL。** `state lifetime` 成为核心，不是指必须预测 output length，而是调度器若给 cached state 估值，就必须同时估计它在未来压力下的存活概率、驱逐后的重算代价，以及 admission 对当前 decode batch 的外部性。该段提出的下一阶段 oracle 已在 0.5 执行，其结果没有通过机制 gate。

### 0.5 五个 lifecycle-cost gate 的 A800 结果（2026-08-25）

预注册协议见 [lifecycle gate protocol](lifecycle_gate_experiment_protocol.md)，完整结果见 [lifecycle gate report](results/lifecycle_gates_20260825/lifecycle_gate_report.md)。实验全部在 `182.43.57.115` 的 `sglang` 容器内运行，只使用物理 A800 GPU 6/7 和 `/models/Kimi-Linear-48B-A3B`。本轮产生 120 个正式 artifact、2,898 个请求；请求成功率为 2,898/2,898，恢复后的服务 health 为 200。

| gate | 结果 | A800 证据 | 当前解释 |
|---|---|---|---|
| M1：token/length ranking | **PASS** | 324 个可比较 class pair 中有 44 个（13.6%）出现超过 5% 的 raw-token 排序反转；最大稳定反转为 10.06x | “短 prompt/较少 raw tokens 不一定便宜”是稳定 characterization |
| M2：state-dependent ranking | **FAIL** | state-bound 与 KV-bound 的最佳顺序确实反转，但最佳固定顺序的 worst-regime regret 只有 2.0% | 有方向变化，但没有足够的动态调度收益 |
| M3：admission/batch/eviction coupling | **FAIL** | FCFS reuse-first p95 只改善 3.3%，多保留 1,824 cached tokens；相对 pressure-first LPM 的剩余收益为 4.6% | cache lifetime 现象没有转化成可观的联合控制收益 |
| M4：equal-token cost transfer | **PASS** | 两租户各请求 33,792 tokens，per-token 资源成本最大相差 6.60x；token VTC 下 normalized slowdown gap 为 101.5 个百分点，3/3 次为 100.6--103.2 个百分点 | token fairness 会造成稳定、可见的跨租户成本转移 |
| M5：empirical oracle headroom | **FAIL** | 24/24 class-block order 完成；受限 oracle p95 为 20.226s，最强预注册 future-memory order 为 20.754s，只改善 2.5% | 当前 workload 下没有实现 lifecycle request-ranking scheduler 的空间 |

因此，这五个 gate **不支持**原先的“lifecycle-cost admission + eviction + fairness”统一系统论文。M1 证明成本模型存在失真，M4 证明这种失真会形成多租户公平伤害；但是 M2、M3 和尤其 M5 表明，靠 request ordering 或联合 cache/admission 控制无法获得足够的效率收益。不能用两个通过的 characterization gate 覆盖三个失败的机制 gate。

仍然成立的窄方向是：**cost-denominated tenant fairness**。其问题陈述应改为“相同 token service 为什么对应不同 physical resource-time 和不同 normalized slowdown，以及如何按实际生命周期成本记账”，而不是继续主张 vector debt 必要，也不应把 admission/eviction efficiency 作为并列贡献。下一道机制 gate 应比较 token VTC、tuned scalar VTC 和基于 realized lifecycle cost 的 fairness oracle；只有公平 gap 明显缩小、makespan 损失不超过 5%，才值得实现在线 cost ledger policy。

### 0.6 Token VTC、tuned scalar VTC 与 realized-cost oracle（2026-08-25）

0.5 提出的窄 fairness gate 已完成。预注册协议见 [fairness oracle protocol](lifecycle_fairness_oracle_protocol.md)，完整结果见 [fairness oracle report](results/lifecycle_fairness_oracle_20260825/lifecycle_fairness_oracle_report.md)。本轮在 A800 GPU 6/7、`state=48, KV=196608` 下运行 8-wave sustained workload；每个 wave 中，split tenant 的 `16 x (1K+32)` 与 packed tenant 的 `1 x (16K+512)` 都恰好提供 16,896 个 input-plus-output tokens。

校准与确认严格分离：

- scalar VTC 扫描 request weight `0, 1K, 4K, 8K, 12K, 16K, 32K, 64K`；最优点为 **0**，即 tuned scalar 退化为 token VTC，所有正 request charge 都扩大 slowdown gap；
- realized-cost policy 按 `w_s * state-slot-time/state-cap + w_k * KV-token-time/KV-cap` 的精确累计债务排序；扫描后冻结为 `w_s=0.25,w_k=1`；
- 确认阶段重新运行 3 次 isolation、token VTC、冻结 scalar VTC 和冻结 realized-cost policy；33 个正式 artifact 共 3,944 个请求，失败与 retraction 均为 0，33/33 有 exact resource trace，15/15 confirmation artifact 的 resolved policy 与标签一致。

| policy | split / packed p95 JCT | split / packed normalized slowdown | slowdown gap | makespan | 30s SLO goodput |
|---|---|---|---:|---:|---:|
| token VTC | 11.595s / 16.322s | 958.6% / 25.3% | 933.3% | 25.010s | 136/136 |
| tuned scalar VTC (`r=0`) | 11.431s / 16.216s | 943.6% / 24.5% | 919.1% | 24.833s | 136/136 |
| realized cost (`s=0.25,k=1`) | 3.489s / 20.296s | 218.6% / 55.8% | 162.8% | 24.229s | 136/136 |

realized-cost oracle 相对 token VTC 将 slowdown gap 降低 **82.6%**，相对 tuned scalar VTC 降低 **82.3%**；三次 paired run 方向全部一致，gap 的 bootstrap 95% CI 为 `[143.7%,192.5%]`，makespan 相对 token VTC 下降 3.1%。它通过了预注册的 15% fairness improvement 与 5% makespan gate。

这个 PASS 改变了 0.5 的后续判断，但不复活已失败的 admission/eviction 方向。当前可以继续的论文主张是：**token/request-count scalar accounting 无法表达 Hybrid LLM 的实际服务成本，而按 realized state/KV lifecycle cost 记账可以显著减少租户间 normalized slowdown 不平衡。** 它不是 Pareto latency improvement：packed tenant 的 slowdown 从 25.3% 上升到 55.8%，换取 split tenant 从 958.6% 降到 218.6%，因此论文必须把目标明确写成 fairness balancing，而不是“所有租户延迟都改善”。

当前机制仍是 workload-specific price oracle：`0.25:1` 由该 calibration mix 的结果选择，尚未证明跨 request mix、capacity 和 arrival rate 稳健。下一道 gate 应冻结该权重并在未参与调参的 workload/capacity 上测试，同时比较 per-workload oracle；若固定价格 regret 超过 15%，才有实现 online scarcity-price adaptation 的必要，否则固定 realized-cost ledger 已足够。

### 0.7 Held-out 泛化与 strongest-baseline kill test（2026-08-25）

0.6 的下一道 gate 已全部完成。协议见 [generalization protocol](lifecycle_fairness_generalization_protocol.md)，正式结果见 [generalization report](results/lifecycle_fairness_generalization_20260825/lifecycle_fairness_generalization_report.md)，逐 artifact 完整性检查见 [audit](results/lifecycle_fairness_generalization_20260825/audit.md)。所有正式运行均使用物理 A800 GPU 6/7、TP=2 和 `/models/Kimi-Linear-48B-A3B`；没有用 smoke 结果做选择或判定。

首先保留“短请求不便宜”的正确含义。外部 SGLang K3 serving figures 给出约 54 MB/request 的固定 KDA state 和 27 KB/token 的 MLA KV；按这些 topology-specific 数值估算，在相同 128K context tokens 下：

- `128 x 1K` 约为 6.75 GiB KDA state + 3.38 GiB MLA KV；
- `4 x 32K` 约为 0.21 GiB KDA state + 3.38 GiB MLA KV；
- 总 footprint 粗略相差 2.8x，差值约 6.5 GiB。

这说明 `total cached tokens` 不是 hybrid server memory pressure 的充分统计量；系统至少需要 `(active requests/state slots, resident KV tokens)` 二维描述。这里的 54 MB/27 KB 是外部 K3、特定 TP/DCP/cache dtype/topology 下的数值，不是当前 Kimi-Linear TP=2 实验的直接显存测量。本文的当前模型证据应引用 exact `state-slot-time` 与 `KV-token-time`，不能把外部 footprint 数字当成本机校准结果。

泛化矩阵包含 10 个 cell，正交改变 split ratio `4/16/32`、prompt/output shape、arrival interval `1/1.5/3s`、state cap `48/96` 和 KV cap `196608/393216`。每个 wave 中 split/packed tenant 的 input tokens 和 output tokens 分别严格相等。baseline 层级为：

| baseline | 回答的问题 |
|---|---|
| FCFS | 当前系统不做 tenant cost arbitration 时的结果 |
| request VTC / token VTC | 按请求数或 token 数定义 service 是否足够 |
| tuned static request tax | 一个固定 per-request charge 能否吸收 state tax |
| instantaneous share | 只按当前 normalized dominant state/KV share 仲裁，不累计 lifetime debt |
| vector debt | 累计 normalized state/KV resource-time，但不使用固定 scalar price |
| fixed lifecycle `0.25:1` | 0.6 校准后冻结、可部署的 realized-cost ledger |
| per-cell lifecycle oracle | 探索 workload-specific 价格的上界，不作为可部署方案 |

正式执行分为 200 个 screening、276 个 confirmation 和 150 个独立 kill-confirm artifacts，共 **626 artifacts、64,152 requests**。审计结果为 0 failure、0 retraction、626/626 exact traces、626/626 resolved policy/weight 精确匹配，486/486 mixed artifacts 的两个 tenant 均保持 equal input/output tokens。

第一轮 confirmation 暴露出 `instantaneous share` 才是 10/10 cell 的最强 confirmed static baseline。为避免在同一批数据上 post-hoc 选择 baseline，随后冻结比较对象并执行 150 个全新 kill-confirm artifacts，不再扫描任何权重。最终结果为：

| gate | 结果 |
|---|---:|
| fixed lifecycle 相对 instantaneous share 的 mean slowdown-gap improvement | 4.0% |
| cell bootstrap 95% CI | -24.7% 至 27.3% |
| paired direction | 7/10 cells |
| 相对 token VTC 的 mean makespan change | +0.4% |

独立 gate 因此为 **FAIL**。失败不是吞吐开销造成，而是 fairness 收益不稳定：`arrival1` 和 `arrival3` 分别退化 99.9% 和 30.6%，而 `base`、`relaxed` 又分别改善 46.2% 和 58.2%。固定 state/KV price 无法跨 arrival dynamics 与 capacity 稳定工作；同时，当前数据也没有证明复杂 ledger 比即时资源 share 有足够增益。

所以最终边界是：

1. **PASS**：fixed state tax 使 equal-token/equal-cached-token workload 产生不同资源压力，token-only accounting 不充分；
2. **PASS**：token VTC 在受控 workload 下会形成显著 normalized slowdown cost transfer；
3. **FAIL**：lifecycle request ranking、admission/eviction 联合控制没有足够 oracle headroom；
4. **FAIL**：冻结的 realized lifecycle price 相对 `instantaneous share` 不具备 held-out 泛化优势；
5. **当前论文判断**：只能形成扎实的 resource/fairness characterization，尚不能声称已有一个可独立成篇的新 scheduler mechanism。

### 0.8 Equal-token partition、VTC scalar 与 state-cap 干预（2026-08-26）

为直接回答“短请求固定 state tax 能否融入 VTC 问题”，新增了无 prefix reuse 的预注册实验。协议见 [VTC state-tax protocol](vtc_state_tax_observation_protocol.md)，正式报告见 [VTC state-tax report](results/vtc_state_tax_20260826/vtc_state_tax_observation_report.md)。每个 wave 中两个 tenant 都完成 16,384 input 和 512 output tokens；split tenant 把它们拆为 `f` 个请求，packed tenant 保持一个请求。每个请求使用唯一 `cache_salt`，因此该实验不涉及 DLPM 或跨请求 prefix locality。

“短请求更贵”在这里严格指：每个 active request 都要支付固定 recurrent-state allocation，所以 request partition 不是资源中性的。它**不**表示单个 1K 请求的总显存或计算大于单个 16K 请求。实验分别报告瞬时峰值与 resource-time，避免把固定 state tax、生命周期和总计算混成同一概念。

正式矩阵包含 27 个 profile、117 个 screening 和 243 个 confirmation artifacts，共 387 个 artifacts、34,107 个请求。全部请求成功，cache hit 和 retraction 均为 0；387/387 exact resource traces、216/216 VTC traces、387/387 resolved policy/weights 均通过，288/288 mixed artifacts 保持 equal input/output tokens。

三个预注册 gate 的结果是：

| gate | 结果 | 关键证据 |
|---|---|---|
| O1：VTC abstraction mismatch | **PASS** | f16/f32 标准 VTC 的 token service 相等且 counter gap 有界，但 max per-token resource ratio 为 17.50x/33.69x，peak state/KV composition ratio 为 15.81x/31.62x，normalized slowdown gap 为 913.2%/924.6% |
| O2：state-capacity specificity | **FAIL** | state cap 从 48 放宽到 480 后，f16 gap 从 913.2% 变为 988.0%，没有下降；KV 放宽后的 gap 为 915.0%；f1 负对照仅 1.2% |
| O3：scalar repair insufficiency | **PASS** | 相对 throughput-bounded strongest scalar，instant-share mean gap improvement 43.8%，cell bootstrap 95% CI `[21.7%,65.5%]`，8/9 cells 同方向，mean makespan change -1.2% |

![Equal-token resource vectors and VTC policy gaps](results/vtc_state_tax_20260826/vtc_state_tax_observation.png)

fixed state tax 的直接资源证据仍然成立。在 state-relaxed f16 cell 中，标准 VTC 下 split/packed 的 peak physical state slots 为 `270/24`，而 peak active KV tokens 为 `94,845/135,086`：相同 token service 产生完全不同的资源向量。base cell 中 state cap 又把 split peak 限制到 24 slots，并使其 VTC queue p95 达到 10.69s；放宽 state 后 queue p95 降到 2.68s，说明 fixed tax 确实改变 admission 与排队。

但 O2 的失败阻止了更强归因。放宽 state 后，token-VTC mixed p95 中 split 从 11.31s 降到 9.84s，packed 却从 15.79s 升到 18.91s；更多请求被同时 admission 后，batch/compute interference 发生变化。与此同时 split-only isolation 从 1.09s 降到 0.87s，因此 isolation-normalized split slowdown 反而更大。结论应写成：

> Request partitioning changes fixed-state demand, admission concurrency, KV residency, and batch interference even at equal token service; token-only VTC cannot represent this joint effect.

不能写成“VTC 的 fairness gap 主要由 state pool 太小造成”。简单 fixed request tax 也没有解决问题：screening 选出的 global tax 是 `r=0`，7/9 per-cell request-tax oracle 同样退化为标准 token VTC。隔离 JCT 拟合的六项 profiled VTC 虽通过 0.2% median/1.6% max fit error gate，在 f16-base 也只把 gap 从 913.2% 降到 877.4%，明显弱于 instant-share 的 290.2%。这说明已知请求级 scalar cost 仍不足以替代运行时资源状态。

该结果强化了 motivation，却还没有给出新机制。instant-share 已是必须比较的 simple baseline，而且在 state-relaxed/full cells 相对 strongest scalar 只改善 2.4%/2.6%。若继续，新的机制 gate 应要求一个 **VTC-bounded、scarcity- and interference-aware** policy 在独立 traces 上稳定优于 instant-share；只增加 request tax、固定 lifecycle price 或 state-only admission priority 不再值得实现。

### 0.2 第二阶段精确计量与策略 kill test

第二阶段结果见 [exact resource report](results/exact_20260820/exact_resource_report.md) 和 [policy report](results/policy_20260820/policy_report.md)。实验实现包括 env-gated scheduler-side tenant tracker、标准 token VTC、带固定 request cost 的 tuned scalar VTC，以及 normalized state/KV vector debt。

本节是 2026-08-20 的单容量历史结果；其 PARTIAL PASS 已被 0.3 的 capacity x dense-scalar 实验推翻。

关键结果：

- exact active accounting 下，state-heavy 峰值为 288 个物理 state slots，KV-heavy 为 18 个；两组 completed tokens 完全相同；
- 9/9 个 co-run 中，state-heavy 的累计 debt 由 state 主导，KV-heavy 由 KV 主导；equal-token service 对应的 dominant debt 平均相差 2.79x；
- sustained workload 下，FCFS、token VTC、VTC-request-8076、VTC-request-13460、vector debt 各运行 3 次；
- 最好的 tested scalar baseline 是 request cost 8,076，其 dominant-debt gap 为 1.41；vector debt 为 0.82，进一步降低 41.6%；
- vector makespan 相对 FCFS 改善 4.4%，20s goodput 增加 1 个请求，但最大 token-service lag 明显增大。

最后一点不是矛盾：token VTC 优化 token fairness，vector policy 优化 normalized physical-resource fairness。若论文声称两者同时改善就是错误的；应报告二者的 Pareto 权衡，并说明高 state-cost tenant 为什么在等权资源公平下得到更少 token service。

---

## 1. 资源模型与当前 SGLang 边界

### 1.1 分离内存池是当前默认模型

对请求 \(i\)，记：

- \(s_i\)：需要的 recurrent state slots；
- \(S_i\)：这些 slots 对应的字节数；
- \(l_i(t)\)：时刻 \(t\) 的 resident attention token 数；
- \(K_i\)：每个 attention token 的 KV 字节数；
- \(g_i(t)\)：GPU compute 使用量。

当前默认的 separate pools 不是一个简单的可加总内存不等式，而是至少两个约束：

\[
\sum_i s_i \le C_{\text{state}},
\qquad
\sum_i l_i(t) \le C_{\text{KV}}.
\]

只有在 unified memory 模式下，才适合近似写成共同字节预算：

\[
\sum_i S_i+\sum_i K_i l_i(t)\le M_{\text{joint}}.
\]

因此论文应把“heterogeneous resource constraints”作为基础，不能默认所有资源都能无损换算成一种 token。

### 1.2 Kimi-Linear-48B-A3B 的量化校准

当前两卡 TP=2 启动日志给出：

| 项目 | 当前值 |
|---|---:|
| recurrent state slots | 485 |
| `max_running_requests` | 97 |
| 容量规划 | 每个 request 预留 5 个 state slots |
| active request 实际常驻 | 3 slots（1 active + 2 ping-pong） |
| MLA KV token capacity | 约 1.458M tokens（第二阶段为 1,458,422） |
| unified memory | 关闭 |

按模型配置计算，每个 TP rank 上：

| state dtype | 单 state slot | 折合 MLA KV tokens | 5 slots/request 的折合量 |
|---|---:|---:|---:|
| 当前默认：SSM FP32 | 20.70 MiB | 2,692 | 13,460 |
| 显式 SSM BF16 | 10.70 MiB | 1,392 | 6,959 |

这里有两个容易产生的误判。第一，当前日志中的 SSM state 默认是 FP32，不是 BF16；只按模型权重 dtype 估算会把 state footprint 低估约一半。第二，`max_running_requests=485/5=97` 使用的是包含 radix lock/COW headroom 的容量规划因子，而当前 `extra_buffer` 活动请求通常只物理分配 3 个 slots。前者用于 feasibility，后者用于 active state-slot-time；不能把 5-slot proxy 称为物理占用。

### 1.3 当前 SGLang 已解决和未解决的部分

| 问题 | 当前状态 | 含义 |
|---|---|---|
| 请求是否满足 state/KV 预算 | 已有 | `PrefillAdder` 会检查 state gap、Mamba slots 和 token budget |
| unified memory 下联合可行性 | 已有 | state bytes 可折算进共同预算 |
| admissible 请求之间的价值排序 | 缺失 | 默认 FCFS；其他策略主要按 prefix、长度或静态优先级排序 |
| 基于未来 state lifetime 的选择 | 缺失 | 没有显式比较 \(S_iT_i\) 或完成后的资源释放收益 |
| state-aware retraction victim | 缺失 | 默认启发式主要看已生成长度和输入长度 |
| tenant 身份贯穿调度器 | 缺失 | OpenAI `user` 字段没有成为内部 request 的公平调度身份 |
| 多租户资源公平 | 缺失 | priority 不是公平债务；没有 per-tenant resource accounting |
| inactive prefix 的价值驱逐 | 部分但非本文核心 | state 与 KV 有各自 LRU 类策略；未按本文的 active lifetime utility 选择 |

所以准确的研究陈述应是：

> SGLang 已具备 hybrid state/KV 的 admission feasibility accounting，但尚未用未来资源时间和多租户资源债务决定“选择谁、保留谁、撤回谁”。

---

## 2. 方向 A：状态生命周期驱动的成本收益选择

### 2.1 研究问题与可检验 thesis

研究问题：

> 当等待队列中有多个可行请求时，能否利用 recurrent state lifetime、未来 KV 增长、完成收益和 SLO 风险，获得比 FCFS、长度优先和纯 KV-memory-aware 策略更高的 goodput？

建议把请求 \(i\) 的未来成本拆成：

\[
C_i^{\text{state}}=S_i\hat T_i,
\qquad
C_i^{\text{KV}}=K_i\int_0^{\hat T_i}\hat l_i(t)\,dt,
\qquad
C_i^{\text{compute}}=\hat G_i.
\]

收益不是“生成 token 数”，而应与任务目标一致，例如：

\[
U_i=w_i\Pr(\text{meet SLO}_i)-P_i^{\text{recompute}}-P_i^{\text{starvation}}.
\]

调度分数可以探索 \(U_i/C_i\)、边际释放资源、或受两个池约束的 knapsack/dual-price 形式。论文贡献不应绑定某一个手写比例，而应强调**hybrid resource-time estimator 如何改变 admission 与 retraction 结果**。

### 2.2 必须回答的反例

需要同时构造并解释三类请求：

| 类型 | 瞬时 state bytes/token | state lifetime | 可能的合理决策 |
|---|---:|---:|---|
| 短 prompt、短 output | 高 | 短 | 常应尽快完成并释放 slot |
| 短 prompt、长 output | 初始高 | 长 | 可能是最昂贵的 state holder |
| 长 prompt、短 output | 低 | 短，但 prefill 重 | 取决于 prefill/recompute 与完成收益 |

如果只证明第一列，就只能得到 memory characterization；只有第二、三列导致不同调度选择，才构成 scheduler motivation。

### 2.3 实验 A0：state/KV/compute 校准

**目的**：验证成本模型，得到后续策略使用的真实资源单位。

**设置**：

- 固定一个请求，分别改变 prompt length、decode length、state dtype、TP 和 radix cache 配置；
- 记录 logical state slots、state bytes、KV pages/tokens、prefill/decode GPU time；
- 同时测试 separate pools 和 unified memory；
- 加一个 KV-only Transformer 作为 control。

**对照**：解析公式、allocator 计数、`nvidia-smi`。后者只能作为辅助，不能作为 logical occupancy 真值。

**指标**：估计误差、非零 state intercept、每层/每 rank 字节数、吞吐和显存池容量。

**通过标准**：allocator 观测与模型误差小于 5%，并且 hybrid 模型的 per-request state intercept 显著且可重复；否则先修正模型，不进入调度实验。

### 2.4 实验 A1：证明“可行性相同，价值不同”

**目的**：把现有 admission accounting 与新选择问题分开。

**设置**：构造一组在当前时刻都能安全 admission、且初始 state/KV footprint 相同的请求，只改变真实剩余 output length 或 deadline。

**对照**：FCFS、random、shortest prompt、longest prefix match、静态 priority。

**指标**：完成请求数、SLO goodput、mean/p95 JCT、state-slot-seconds、KV-byte-seconds、retraction 次数。

**通过标准**：在完全相同的可行性判断下，oracle lifetime 策略相对最强现有排序显著改善 goodput，且不靠降低 offered load 获胜。这是方向 A 的首个 go/no-go 实验。

### 2.5 实验 A2：短请求“摊销贵”与“释放快”的解耦

**目的**：直接回答“短请求可能更贵，是否改变调度”的问题。

**设置**：保持总 token arrival rate 接近，独立扫描 request arrival rate 和 output length：

- many-short：高 request rate、短 output；
- few-long：低 request rate、长 output；
- short-prompt/long-output；
- long-prompt/short-output。

**控制变量**：相同模型、硬件、总输入/输出 token rate、prefix reuse、deadline 分布和预热状态。

**指标**：瞬时 bytes/token、state-slot-seconds/request、完成后释放的 dominant resource、goodput、尾延迟。

**通过标准**：证明 \(S/L\) 不能单独预测最优选择，而 \(S\hat T+K\int l(t)dt\) 或其在线近似能稳定解释选择差异。

### 2.6 实验 A3：离线 oracle 与选择问题上界

**目的**：先判断问题是否值得做，再开发在线 predictor。

**设置**：使用真实 output length、实际 profile 的 prefill/decode 时间以及精确 state/KV 轨迹，求小规模窗口内的最优或近最优 admission subset。两个池应作为独立约束；unified memory 另做一组。

**对照**：FCFS、SJF/shortest remaining output、最小当前 memory、最大 freed-resource rate、现有 SGLang policies。

**指标**：相对 oracle gap、admitted-set 差异、goodput、JCT、资源利用率。

**通过标准**：oracle 相对最强 baseline 至少在多个负载区间有稳定收益。若只有单个极端点有效，论文价值不足。

### 2.7 实验 A4：在线 lifetime 估计与不确定性

**目的**：检验 oracle 收益能否转化成在线收益。

**设置**：分别使用：

- 无预测；
- mean output length；
- 分位数或置信上界；
- 从历史同类请求得到的分布；
- 人为注入 0%、10%、25%、50%、100% 相对误差。

**对照**：SJF、uncertainty-aware SJF、Past-Future 风格 future-memory estimator、保守 KV reservation。

**指标**：oracle gap、错误 admission、额外 eviction/retraction、SLO violation、starvation。

**通过标准**：在现实误差范围内仍优于不用预测的资源感知 baseline；必须报告收益随误差退化的曲线，不能只报告准确预测点。

### 2.8 实验 A5：state-lifetime-aware retraction

**目的**：验证内存压力出现后，“撤回谁”与“先放谁”同样受 state lifetime 影响。

**候选 victim cost**：

\[
V_i=
\frac{\text{freed state/KV dominant share}}
{\text{recompute cost}+\text{SLO penalty}+\text{near-completion penalty}}.
\]

**设置**：用突发到达和输出长度低估主动触发 retraction；分别制造 state-slot exhaustion 和 KV exhaustion。

**对照**：当前 SGLang heuristic、largest-memory-first、youngest-first、longest-remaining-first、random。

**指标**：每次 retraction 释放量、重复 prefill tokens、被撤回次数/request、尾延迟、goodput。

**通过标准**：同时降低重复计算和 SLO violation；若只减少 retraction 次数但降低吞吐，不算成功。

### 2.9 实验 A6：真实 workload 与模型泛化

**目的**：排除只对合成双峰 workload 有效的可能。

**设置**：

- 至少一个对话 trace、一个 agent/long-generation trace、一个 bursty API trace；
- Kimi-Linear 为主，至少一个其他 hybrid 模型；
- KV-only Transformer control；
- arrival rate 从 30% capacity 扫到 overload，重点报告 knee 附近。

**指标**：SLO goodput、p50/p95/p99 TTFT/TPOT/JCT、吞吐、资源时间、starvation、调度开销。

**通过标准**：收益主要出现在 hybrid state pressure 下；在 Transformer control 上不应出现无法解释的巨大收益或严重回退。

### 2.10 方向 A 的文章冲突与安全边界

| 工作 | 已覆盖内容 | 与本文冲突 | 应如何区分/比较 |
|---|---|---|---|
| [FastServe](https://arxiv.org/abs/2305.05920) | 未知长度下的 preemptive iteration-level scheduling、MLFQ、JCT 优化 | “短作业优先/剩余时间优先”不是新意 | 作为 latency/preemption baseline；强调 hybrid state 与 KV 双约束 |
| [Past-Future Scheduler](https://arxiv.org/abs/2507.10150) | 根据 output distribution 预测未来 memory occupancy，平衡排队与 eviction | 与 lifetime/future-memory 动机高度重叠 | 必须实现或忠实复现；差异应落在固定 state lifetime、双资源池和 state-aware retraction |
| [Uncertainty-Aware Output Length Scheduling](https://arxiv.org/abs/2604.00499) | 用输出长度分布做 uncertainty-aware SJF | output predictor 和稳健 SJF 不能作为核心贡献 | 比较相同 predictor 下的 token-time 与 hybrid resource-time 决策 |
| [Variable Prefill and Decode Length Scheduling](https://arxiv.org/abs/2508.06133) | memory-constrained request selection、近似/竞争算法 | 泛化的 subset selection claim 有冲突 | 限定为 hybrid heterogeneous state/KV 约束，并用其选择算法作 baseline |
| [Fluid-Guided Online Scheduling](https://arxiv.org/abs/2504.11320) | 已知/未知 output length 的在线 memory-constrained scheduling | “在线内存约束调度”本身已有工作 | 不宣称首次；比较等待策略与 state-lifetime 决策 |
| [Robust KV Cache Management](https://arxiv.org/abs/2607.16892) | 输出长度不确定下的稳健 KV reservation/routing | 与稳健 admission 有部分冲突 | 说明其资源为增长型 KV，本文研究固定 state + KV 的组合 |
| [Marconi](https://arxiv.org/abs/2411.19379) | hybrid prefix cache admission/eviction，按复用、计算收益和 footprint 估值 | 若把 prefix cache value eviction 作为主贡献会直接冲突 | MVP 聚焦 active request admission/retraction；inactive prefix eviction 只做正交扩展 |
| [Jenga](https://arxiv.org/abs/2503.18292) | heterogeneous embedding memory allocator、统一/两级分配与 eviction | “首次异构/统一内存管理”不可成立 | 把 allocator 当底座，只研究其上的 resource-time scheduling |

**方向 A 可守住的 claim**：

> 在 hybrid recurrent-attention serving 中，用固定 state 的持有时间与增长型 KV 的未来占用共同评价 admissible requests，并把同一价值模型用于 admission 和 active-request retraction。

**方向 A 不应使用的 claim**：首次基于 output length 调度、首次预测未来显存、首次 memory-aware admission、首次 hybrid cache eviction。

---

## 3. 方向 B：混合主导资源时间上的多租户公平

### 3.1 为什么 token fairness 不一定是 resource fairness

两个 tenant 即使获得相同输入/输出 token 服务，也可能产生不同资源压力：

- tenant A 提交大量短并发请求，占据更多 fixed state slots；
- tenant B 提交少量长上下文请求，占据更多 KV；
- tenant C 的请求 prefill 或 decode compute 更重。

若只累计 token，A 的 state-slot-time 和 B 的 KV-byte-time 都不会被准确表达。公平对象应从 token service 扩展为一段时间内的资源使用向量：

\[
\mathbf r_u(t)=
\left(
\frac{n_u^{\text{state}}(t)}{C_{\text{state}}},
\frac{b_u^{\text{KV}}(t)}{C_{\text{KV}}},
\frac{g_u(t)}{C_{\text{compute}}}
\right).
\]

瞬时 dominant share：

\[
d_u(t)=\max_k r_{u,k}(t).
\]

长期公平不能只看瞬时 share，而应维护加权资源债务：

\[
D_{u,k}(t)=\int_0^t\frac{r_{u,k}(\tau)}{w_u}\,d\tau,
\qquad
D_u^{\max}(t)=\max_kD_{u,k}(t).
\]

调度器外层优先服务 normalized dominant debt 较低的 tenant，空闲份额允许 work-conserving borrowing；内层再用方向 A 的 utility 选择该 tenant 的请求。

### 3.2 研究问题与可检验 thesis

> 相比按 token 计费或按请求轮转，显式累计 state-slot-time、KV-byte-time 和 GPU-time，能否在不显著牺牲总 goodput 的情况下，改善 hybrid workload 下的 tenant isolation、service lag 和 SLO fairness？

这里的核心不是“更平均”，而是三个性质：

1. **strategy resistance**：把一个任务拆成许多短请求，不能免费占用更多 state slots；
2. **isolation**：state-heavy 或 KV-heavy noisy neighbor 不能无限拖累其他 tenant；
3. **work conservation**：其他 tenant 空闲时，活跃 tenant 可以借用资源，不因静态 quota 浪费 GPU。

### 3.3 实验 B0：tenant identity 与资源记账验证

**目的**：建立后续公平实验的可信观测面。

**实现要求**：把认证主体或显式 tenant ID 从 API 层传入内部 request；不能直接信任任意用户填写的 OpenAI `user` 字段。记录每 tenant 的 state-slot-seconds、KV-byte-seconds、prefill/decode GPU time、tokens 和完成请求。

**设置**：单 tenant 可控 replay，逐项改变并发数、context length、output length。

**通过标准**：per-tenant 计数求和与全局 allocator/compute 计数误差小于 2%，请求结束、retract、cache hit 和取消时无泄漏或重复计费。

### 3.4 实验 B1：equal tokens 不等于 equal resource share

**目的**：形成方向 B 最关键的 motivation figure。

**设置**：两个相同权重 tenant，给它们相同 token arrival rate 和长期完成 token 数：

- tenant A：many-short/high-concurrency，state-dominant；
- tenant B：few-long/long-context，KV-dominant。

再交换 workload 身份，排除 tenant 顺序和连接差异。

**对照**：global FCFS、per-tenant round-robin、token-WFQ/VTC 风格计数。

**指标**：各 tenant 的 token service、state-slot-seconds、KV-byte-seconds、dominant share、SLO goodput 和 service lag。

**通过标准**：相同 token service 下出现稳定且显著的 dominant-resource 差异，并导致可观测的跨 tenant SLO 干扰。若只有资源计数差而没有服务影响，fair scheduler 的动机较弱。

### 3.5 实验 B2：state-heavy 与 KV-heavy tenant 的权衡

**目的**：证明单一 scalar token cost 无法在不同瓶颈阶段始终正确。

**设置**：系统分别运行在 state-slot 饱和、KV 饱和、compute 饱和和瓶颈切换四个区间；两个 tenant 的 dominant resource 不同。

**对照**：FCFS、request round-robin、输入+输出 token VTC、静态 state/KV quota、scalar byte-time fairness、proposed dominant-resource-time fairness。

**指标**：weighted max-min service gap、最大 normalized debt、Jain index（基于 normalized dominant service）、总 utilization 和 goodput。

**通过标准**：提出的方法在瓶颈切换时无需手工更换 token 权重，且最大 service lag 明显低于 scalar baselines。

### 3.6 实验 B3：noisy neighbor 与低速 tenant 保护

**目的**：验证 isolation 和 starvation bound。

**设置**：

- 一个 tenant 持续饱和提交 many-short 请求；
- 一个 tenant 持续提交 long-context 请求；
- 一个 latency-sensitive tenant 低速、突发到达；
- 扫描权重 1:1、1:2、1:4。

**指标**：受保护 tenant 的 p95/p99 TTFT/JCT、deadline attainment、最长等待时间；攻击 tenant 的额外 state 占用；系统总 goodput。

**通过标准**：低速 tenant 的尾延迟和 SLO 在 overload 下有明确上界或显著改善，同时系统保持 work-conserving。

### 3.7 实验 B4：拆分请求是否能套利

**目的**：证明 state-aware fairness 对 request splitting 具有鲁棒性。

**设置**：同一 tenant 保持总 token 和业务语义近似不变，把 workload 拆为 \(1,2,4,8,16\) 倍请求数；另一 tenant 不变。

**对照**：token fairness、request fairness、静态 `max_num_seqs` quota、dominant-resource-time fairness。

**指标**：拆分 tenant 获得的服务增益、额外 state-slot-seconds、对邻居 SLO 的影响。

**通过标准**：提出的方法下，拆分不能带来超出调度噪声的资源份额增益；同时不应惩罚真实的短请求 workload 到不可用。

### 3.8 实验 B5：公平—效率前沿与 borrowing

**目的**：量化 fairness 不是免费的，并验证空闲份额可借用。

**设置**：扫描公平债务权重/aging 参数；分别测试所有 tenant 饱和、部分 tenant idle、tenant 周期性离开/返回。

**指标**：总 throughput/goodput 对 maximum service lag、Jain index、GPU utilization 的 Pareto curve；tenant 返回后的 debt recovery 时间。

**通过标准**：存在明显优于静态 quota 的 Pareto 区间；单 tenant active 时性能接近无公平调度，tenant 返回后能平滑收回份额而非产生长时间 head-of-line blocking。

### 3.9 实验 B6：fairness-aware retraction

**目的**：当必须撤回 running request 时，不让 victim selection 破坏已经建立的公平性。

**两级选择**：先从资源债务过高的 tenant 中选择，再在 tenant 内按方向 A 的 victim cost 选择请求。

**设置**：让一个 tenant 制造 state pressure，另一个制造 KV pressure，使用 burst 触发 retraction；交换 tenant 角色。

**对照**：全局默认 retraction、largest-memory-first、只按 tenant debt、只按 request utility、两级策略。

**指标**：tenant-level service lag、重复计算、公平债务恢复时间、retraction 集中度、goodput。

**通过标准**：两级策略同时改善公平债务和重算成本；不能通过反复撤回同一个弱势 tenant 获得表面吞吐。

### 3.10 可选实验 B7：共享 prefix 的归属

共享 prefix 的 KV/state 成本应该记给创建者、当前使用者还是按命中者分摊，是独立且复杂的问题。它会把本文直接带入 prefix-aware fair scheduling 和 cache value admission 的竞争范围。

建议 MVP 中：

- active request 的私有 state 和增量 KV 正常计费；
- shared inactive prefix 使用预先声明的简单规则并做 sensitivity analysis；
- 不把共享 prefix ownership 作为核心 claim。

只有在 B1-B6 已经成立后，才加入 B7。

### 3.11 方向 B 的文章冲突与安全边界

| 工作 | 已覆盖内容 | 与本文冲突 | 应如何区分/比较 |
|---|---|---|---|
| [Dominant Resource Fairness (DRF)](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types) | 多资源环境中的 dominant share、weighted max-min fairness 和 strategy-proofness | dominant-resource fairness 这一抽象本身不是新意 | 把 DRF 作为理论起点；新问题是资源占用随 request lifecycle 变化且 admission/retraction 为离散决策 |
| [VTC: Fairness in Serving Large Language Models](https://arxiv.org/abs/2401.00588) | 以 input/output token service cost 定义 client fairness；work-conserving，给出 fairness bound，并允许定制 cost function | 与多租户公平直接冲突；不能声称首次 resource-aware fairness | 必须作为主 baseline；证明简单定制 scalar token cost 在 hybrid 双瓶颈和 request splitting 下不足 |
| [DLPM: Locality-aware Fair Scheduling](https://arxiv.org/abs/2501.14312) | deficit fairness 与 longest-prefix locality 联合，扩展到分布式调度 | 若同时宣称 fairness + prefix locality，会高度重叠 | 主实验控制 prefix reuse；扩展实验再与 DLPM 比较 |
| [Preble](https://arxiv.org/abs/2407.00023) | 分布式 prefix-aware scheduling，并含 fairness/locality 启发式 | 分布式和 prefix locality claim 有冲突 | 本文先限定单实例 hybrid resource fairness；扩展到集群时必须比较 |
| [FairBatching](https://arxiv.org/abs/2510.14392) | prefill/decode task 之间的公平 batching | broad “fair batching” 表述冲突，但公平对象不同 | 明确本文公平对象是 tenants，资源是 state/KV/compute time |
| [DriftSched](https://arxiv.org/abs/2606.02982) | multi-tenant QoS、基于 token drift 的自适应调度 | 与 tenant debt/QoS 调度部分冲突 | 对比 token drift 与 vector resource debt，避免“首次自适应多租户”表述 |
| [Marconi](https://arxiv.org/abs/2411.19379) | prefix cache 的价值 admission/eviction | 若讨论共享 prefix fairness/ownership 会相交 | 将 shared prefix accounting 留作扩展，不作为 MVP |

DRF 已经建立了 dominant-resource allocation 的基本理论，VTC 也允许自定义 cost function。因此，仅把 token cost 换成“state 等价 token”，或把 DRF 名词搬进 SGLang，都不足以形成贡献。必须通过 B2/B4 证明：当 state 与 KV 不可完全替换、占用随 request lifecycle 演化且瓶颈随时间切换时，资源向量及其累计占用时间会产生静态 DRF allocation 和 scalar counter 都无法稳定表达的 admission/retraction 决策差异。

**方向 B 可守住的 claim**：

> 面向 hybrid recurrent-attention serving 的 tenant fairness：用 state-slot-time、KV-byte-time 和 GPU-time 的 normalized dominant debt 做外层 tenant arbitration，并与内层 request utility、retraction 协同。

**方向 B 不应使用的 claim**：首次 dominant-resource fairness、首次 LLM 多租户公平、首次 work-conserving fairness、首次 fairness + locality、首次 deficit scheduler。

---

## 4. 两个方向如何组合

两个方向解决不同层次的问题，不应混成一个总分数：

```text
waiting requests
      |
      v
[outer tenant arbitration]
  least normalized dominant-resource debt
  + weights / aging / work-conserving borrowing
      |
      v
[inner request selection]
  estimated completion utility
  versus state-time + KV-time + compute cost
      |
      v
[existing SGLang feasibility check]
  state slots + KV budget / unified memory
      |
      v
admit and execute
```

内存不足需要 retraction 时反向执行：

1. 判断当前真正的瓶颈是 state、KV 还是 compute；
2. 优先考察 dominant debt 过高的 tenant；
3. 在该 tenant 内选取“释放瓶颈资源收益 / 重算与 SLO 损失”最高的 victim；
4. 加 near-completion 和 starvation 保护，避免抖动。

现有 feasibility check 应保留为硬约束，新的 policy 只决定候选顺序和 victim，不重复实现 allocator。

### 4.1 是否应该合成一篇论文

只有满足以下条件才建议合并：

- A1/A3 证明 resource-time utility 明显改变 admissible set；
- B1/B2 证明 token fairness 确实造成 hybrid resource unfairness；
- 同一套 state/KV accounting 能同时支撑 selection 和 fairness；
- 组合策略相对“VTC + strongest lifetime scheduler”的直接组合仍有增益。

否则更稳妥的拆分方式是：

- 论文 A：hybrid state-lifetime-aware admission and retraction；
- 论文 B：hybrid dominant-resource-time tenant fairness，以 A 的 inner scheduler 作为可替换模块。

**2026-08-26 实验后的实际决策**：不合并，也暂不拆成两篇机制论文。A 的 M2/M3/M5 已失败；B 的新 VTC 实验确认 equal-token partition 会造成强 resource-shape 与 slowdown mismatch，并排除了 request-tax/profiled scalar 这一简单修复，但 state-cap specificity 失败。统一故事仍缺少 inner request-selection 收益；独立 fairness 故事则需要一个在独立 traces 上稳定优于 instantaneous share、同时恢复 VTC 长期服务边界的机制。

---

## 5. 公共实验规范

### 5.1 Baseline 矩阵

所有结果至少包含：

| 类别 | Baselines |
|---|---|
| 当前系统 | SGLang FCFS、LPM、现有 retraction |
| 长度调度 | SJF/shortest remaining output、FastServe 类 MLFQ |
| memory/lifetime | Past-Future 类 future-memory、最小当前 footprint、oracle |
| tenant fairness | per-tenant RR、静态 quota、VTC、DLPM（有 prefix locality 时） |
| ablation | state-only、KV-only、compute-only、无 uncertainty、无 aging、无 fairness-aware retraction |

不能只与 FCFS 比较，也不能用 oracle 直接充当可部署方案。

### 5.2 Workload 维度必须正交控制

- prompt length 与 output length 分开；
- token arrival rate 与 request arrival rate 分开；
- concurrency 与 burstiness 分开；
- prefix reuse 单独开关；
- tenant weight、tenant 数和活跃/空闲周期分开；
- state dtype、TP、radix cache strategy 和 separate/unified memory 明确记录。

### 5.3 指标

**效率**：request throughput、token throughput、SLO goodput、GPU utilization、调度开销。

**延迟**：TTFT、TPOT、JCT 的 p50/p95/p99，deadline attainment。

**资源**：state-slot-seconds、KV-byte-seconds、GPU-ms、pool saturation time、eviction/retraction 和 recompute tokens。

**公平**：per-tenant normalized dominant debt、maximum service lag、weighted Jain index、最长等待时间、noisy-neighbor slowdown。

所有结果报告至少 3 次独立运行和置信区间；负载扫描必须覆盖低负载、capacity knee 和 overload。只报平均 latency 或单个并发点不够。

### 5.4 必需 instrumentation

- request/tenant 的 state slot allocate/free 时间；
- request/tenant 的 KV page allocate/free 时间；
- prefill/decode GPU time 或可靠 proxy；
- admission reject 原因：state、KV、compute/queue；
- eviction/retraction victim、释放资源和重算代价；
- predictor 的估计、真实 output length 和误差；
- scheduler 每次决策耗时。

预分配内存池使 `nvidia-smi` 无法反映 logical occupancy，因此它只能用于发现显存泄漏或 OOM，不能作为主要证据。

---

## 6. 最小推进顺序与停止条件

### 阶段 1：两周内完成的无算法验证

1. A0：校准真实 state/KV footprint；
2. A1：用真实 remaining length 做 oracle 排序；
3. B0：打通 tenant/resource accounting；
4. B1：构造 equal-token/different-resource 反例。

停止条件：A1 对 strongest simple baseline 无稳定收益，且 B1 不产生跨 tenant SLO 干扰。此时该 topic 更适合作为 characterization，而不是完整 scheduler 论文。

### 阶段 2：证明两个核心机制

1. A3/A4：offline upper bound 到 online robust estimator；
2. B2/B4：dominant-resource debt 与 request-splitting resistance；
3. 分别复现 Past-Future 和 VTC 作为主要 baseline。

停止条件：在线收益被预测误差完全抹平，或 vector fairness 不优于调好权重的 VTC scalar cost。

### 阶段 3：系统闭环

1. A5：lifetime-aware retraction；
2. B5/B6：fairness-efficiency frontier 和两级 retraction；
3. A6：真实 trace、多模型和 Transformer control；
4. 最后才考虑 shared prefix accounting 和 distributed routing。

---

## 7. 最终定位

原先推荐的统一题目为：

> **Beyond Feasibility: Lifetime-Aware Request Selection and Dominant-Resource Fairness for Hybrid LLM Serving**

一句话故事：

> Existing hybrid-aware allocators determine whether recurrent state and attention KV fit; this work determines which feasible request is worth admitting and which tenant is owed service, using the time-integrated cost of heterogeneous state, KV, and compute resources.

该题目现在应视为**被 kill tests 否定的候选故事**，不能继续作为投稿定位。当前被数据支持的一句话应更新为：

> In hybrid LLM serving, request partitioning is not cost-neutral: equal token service can induce different fixed-state demand, KV residency, admission concurrency, and slowdown, while token-only or profiled scalar VTC cannot represent their runtime interaction; however, state capacity alone does not explain the fairness gap, and no new policy has yet beaten instantaneous resource share robustly.

最重要的论文边界是：

- 不把 fixed-state observation 本身包装成完整贡献；
- 不把 output length prediction 或 token fairness 重新命名；
- 不把 allocator、inactive prefix eviction、admission、active retraction 和 tenant fairness 全部混为一个问题；
- 用两个关键反例分别证明“选择价值”和“公平债务”确实需要 hybrid state-aware abstraction。

两个反例并未同时转化为机制收益。因此，当前最诚实的产出是 characterization、完整 baseline ladder 和负结果边界；要形成新论文，必须先提出一个能解释并修复 `arrival1/arrival3` 退化、且在独立数据上稳定击败 `instantaneous share` 的新机制，再重新预注册 gate。仅增加更多 lifecycle 权重、predictor 或动态 pricing 项不能替代这个证明。
