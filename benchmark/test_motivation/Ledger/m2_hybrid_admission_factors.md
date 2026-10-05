# M2：Hybrid 请求准入需要区分激活成本、资源瓶颈与占用持续时间

> 2026-09-10；Kimi-Linear-48B-A3B，TP2，物理 GPU 6/7。
> 本文是 M2 的新主线。[旧 cap/batching 扫描](m2_admission_batch_formation.md)保留为辅助机制证据，不修改其原始判据或 held-out 负结果。
> 已完成 132 组正式运行、2119 个请求；另有 28 组完整预热不进入统计。完整性检查通过，测量代码哈希在整个运行期间未变。
> 当前结论：激活、独立资源约束、持续占用三类因素及其准入后果得到支持；另发现具体 request-slot 计数问题。M2 的目标是发现具体问题及局部收益空间，不要求证明现有策略普遍不适用。
> 后续完成：[slot 修正与 batching 对照](m2_slot_reuse_batching_opportunity.md)，另有 100 组/2000 请求。原门限 7 个 holder -> 修正后 8 个；主对照 short mean 改善 8.37%，插值 p95 改善 29.01%，mixed/无 chunk/宽池对照给出收益边界。以下 132 组历史数据仍来自修正前代码。
> 2026-09-15 更新：[自然容量与执行进度](m2_natural_progress_results_20260915.md)，20 组/4316 请求。自然 state 瓶颈得到验证，但“满池时大量 holder 未执行”为主要损失来源未获支持；Mixed 的持续参与没有减少整体 state-time 或 JCT。该报告更新 hybrid 特有性和剩余收益空间的判断，不改写本文历史实验。

## 1. 从 M1 递进到什么问题

M1 说明：相同 token 服务量不能充分表示不同请求形态的 hybrid 资源负担。M2 不再重复“短请求单位 token 更贵”，而追问：**当这些请求被加入运行集合时，现有的候选排序、容量检查与逐轮 token/chunk budget，会把这种资源差异转化成什么准入结果？**

研究对象是 request admission / batch formation 的组合机制，不是 continuous batching 框架本身。必须区分两个动作：

- **首次激活新请求**：在本次无 prefix、无 retraction 配置下，建立一个持续到完成的 recurrent-state holder，并分配当前需要的 KV。
- **继续执行已激活请求**：后续 chunk/decode 继续消耗算力并推进 KV，但不为该请求重复激活一个 state holder。

因此，“本轮再处理多少 token”“再激活多少请求”和“已经激活的请求何时释放资源”不是同一个决策量。

## 2. 理论上有哪些因素，不能直接推出什么

用 state slot 表示固定状态单位，令候选动作 i 的新增激活数为 a_i、实际新增 KV 为 k_i。无复用、无回收时，可行性至少要求：

\[
\sum_i a_i \le S_{free},\qquad
\sum_i k_i \le K_{free}.
\]

这是解释用的简化约束，实际 SGLang 还会检查输出预留、chunk/prefill budget、请求池和并发上限。两个候选动作即使处理相同数量的新 token，a_i 也可能不同：新请求的首 chunk 与已激活请求的后续 chunk 并不等价。

对请求 i，在不抢占、不缓存已完成 state 的设置下，占用为从激活到释放的指示函数；state-time 是这段区间的长度。**state-time 可以度量资源占用，但它不是已经证明合适的公平计费单位，也不是必须预测 lifetime 的依据。**

如果所有 slot 已被不可抢占的请求占满，单纯重排 waiting queue 不能让新请求立即运行。收益空间只有在更早的准入决策存在可替代选择，或允许并评估抢占/资源调整时才可能出现。看到排队并不自动证明已有调度存在可避免损失。

本轮逐步检查三件事：

| 子实验 | 需要观察的因素 | 要定位的现有机制局限 | 不能仅由该实验声称 |
|---|---|---|---|
| E1 激活与继续执行 | 等 input token 可激活不同数量 state；KV 推进不等于新增 state | 仅用逐轮 token 数不能完整表示激活资源需求 | 现有系统不检查 state，或延迟一定更差 |
| E2 相同候选、不同约束 | 实际 binding constraint、候选准入顺序与服务结果 | token 排序的后果依赖独立的 state/KV/并发约束 | 某排序在所有目标下更公平，或全部 SOTA 失效 |
| E3 持续占用与后续请求 | 已激活 holder 持续占用时，下一个请求何时获得 slot | 当前一步预算不直接表示未来的准入机会 | KDA 独有的阻塞机制，或 lifetime predictor 必需 |

## 3. 实验设置与证据身份

冻结协议：[m2_hybrid_factors_protocol_20260910.md](m2_hybrid_factors_protocol_20260910.md)。只新增 opt-in 观测，不改变生产路径的排序、容量判断或准入动作。

| 配置 | State slots | KV tokens | 启动请求上限 / 有效上限 | 运行时 cap | 对照作用 |
|---|---:|---:|---:|---:|---|
| wide | 128 | 131072 | 64 / 64 | 64 | 两个资源池相对宽裕 |
| state-low | 8 | 131072 | 64 / 8 | 8 | state 容量及其派生并发限制 |
| cap-control | 128 | 131072 | 64 / 64 | 8 | 保持宽池，只限制准入数 |
| kv-low | 128 | 16384 | 64 / 64 | 64 | KV 预算受限 |

四组均关闭 radix cache，以及 prefill/decode CUDA graphs；chunk=4096，max-prefill-tokens=16384。这样排除 prefix/checkpoint 和 graph 覆盖变化，但不能把 eager 执行时间直接外推到启用 graph 的生产服务。

每个配置先跑完整 workload/policy 预热，预热不进入正式统计。E1 每点 3 次重复，E2/E3 每点 5 次。配对使用相同输入 token IDs、输出长度、到达偏移；E2 顺序干预只改变两组的到达时刻。配置按照服务器分块运行，因此跨服务器组仍有时间漂移风险；wide 与 cap-control 在同一个服务器中交错运行。

置信区间是给定服务器会话中的 run-level paired bootstrap，不包含多次服务器重启或配置执行顺序带来的不确定性；E2 多个策略对比为探索性，未做同时推断。4 个 long 请求的单次 p95 接近最大值，不能当成生产尾延迟的精确估计。

事件日志记录的是调度器首次观测到的 active ownership，不是精确 kernel 起止或 allocator 调用时刻。下文 queue 为该时刻减客户端实际提交时刻，包含网络/调度可见性开销。拒绝原因是实际走过的代码分支，不把所有 `NO_TOKEN` 都解释为 state 不足；未提交到 allocator 的当轮候选预留也可能影响预算。

### 3.1 对照方法不是同一种公平保证

E2 使用已有的本地 token-VTC 和 frozen-score instant-share。前者按 token 历史服务与 shadow input charge 排队；后者按当前 normalized dominant state/KV share 排序，一轮内不逐 candidate 更新预计占用。它们是机制诊断对照，instant 不是完整 DRF，也不是与 VTC 有相同公平约束的替代实现。

E3 使用 FCFS，以减少公平计账变化对 holder 实验的干扰。本轮没有实现完整 Sarathi-Serve、FairBatching 或 LPRS/APC，不将原生开关等同于这些论文。

## 4. E1：首次激活不是一个普通 token chunk

固定 aggregate input=16384，分别组织为 1x16K、4x4K、16x1K、32x512。每请求 output=128，以便观测 state 驻留；总 output 工作量不同，所以本实验只比较资源结构，不比较形态间 JCT/公平性。

![E1：激活与 chunk 继续执行](figures/m2_hybrid_activation.png)

| Input 形态 | 正式重复 | Peak state slots | Peak KV tokens |
|---|---:|---:|---:|
| 1 x 16384 | 3 | 1 | 16512 |
| 4 x 4096 | 3 | 4 | 16896 |
| 16 x 1024 | 3 | 16 | 18432 |
| 32 x 512 | 3 | 32 | 20480 |

每个形态的 state/KV 峰值在 3 次重复中一致。KV 峰值包含输出增长，不能把这些峰值写成“同一总 resident token 数”的对比。等量约束只针对输入 token。

1x16K 的每次记录均出现 KV 从 4096 到 8192、12288、16384 的推进，而 state 一直为 1，直到完成后释放。左图是请求数决定的激活需求，右图是同一请求继续执行的实际轨迹。

**观察：相同输入工作总量可以激活不同数量的固定 state；继续一个 chunk 不会重复产生同一份激活成本。**

**问题限定：把 token/chunk budget 当作激活资源需求的充分描述会遗漏这一维度。SGLang 的独立容量检查已经弥补了安全层面的遗漏，因此本图不证明现有 admission 不安全。**

**Insight：需要在分析中区分“本轮执行多少工作”与“本轮新增多少持久 holder”，再通过 E2/E3 判断这一区分何时影响实际选择。**

## 5. E2：相同请求，约束改变后准入结果如何变化

每组固定 16 个 P512/O64 短请求与 4 个 P8192/O64 长请求。两组分别在 0 与 0.05s 开始，组内间隔 0.001s；测试 short-first 与 long-first 两种提交顺序。排序决定仍由 VTC/instant 执行，不把提交顺序干预称为一种新 batching 算法。

此处两个租户的总工作量不相同，也不据此计算等工作公平收益。问题是固定候选工作集在不同资源条件下的选择后果。需要同时报告短/长 p95、makespan、实际激活和资源阻塞，而不是只挑更好看的一个租户。

![E2：相同候选的不同准入结果](figures/m2_hybrid_candidates.png)

### 5.1 先确认实际是什么约束

| 配置 | E2 实测 peak active | 主要资源阻塞路径 | 观察 |
|---|---:|---|---|
| wide | 20 | 无 request-limit / NO_TOKEN | 全部请求可进入运行集合；仍有正常 chunk budget 分批 |
| state-low | 8 | 每 run 2 次 request-limit | state 缩小派生并发和 request pool，KV 仍宽裕 |
| cap-control | 8 | 每 run 2 次 request-limit | 宽 state/KV 池，仅运行时 cap 约束；不与 state-low 完全等价 |
| kv-low | 15 | 每 run 4 次 VTC / 5 次 instant 的 NO_TOKEN | peak KV=16320/16384，state 仅 15/128；无 request-limit |

拒绝次数是路径记录数，不是等待时间或全部候选重试次数。在 `kv-low / short-first / VTC / rep10` 的一次拒绝处，日志为 state free=108、KV physical free=3584，而剩余总 token 预留预算只有 252.4，chunk budget=1536。因此是**含预留的 KV/token 预算不允许下一个候选**，不能称为 state 不足，也不能把物理 KV free 误读为可任意使用的空间。

### 5.2 排序效果依赖资源条件与到达顺序

下表是每组 5 次运行的短请求 p95 均值；相对变化是逐对 `1 - instant/VTC` 的均值，正值表示延迟降低。

| 配置 | 提交顺序 | VTC short p95 | Instant short p95 | 相对变化及 95% CI |
|---|---|---:|---:|---:|
| wide | short-first | 5.239s | 5.198s | +0.78% [0.19%, 1.30%] |
| wide | long-first | 5.049s | 5.094s | -0.90% [-2.04%, 0.18%] |
| state-low | short-first | 12.181s | 12.145s | +0.30% [0.09%, 0.55%] |
| state-low | long-first | 12.006s | 8.582s | +28.52% [28.34%, 28.78%] |
| cap-control | short-first | 8.669s | 8.703s | -0.40% [-0.93%, 0.33%] |
| cap-control | long-first | 12.047s | 8.606s | +28.55% [27.78%, 29.44%] |
| kv-low | short-first | 8.078s | 8.047s | +0.38% [-0.09%, 0.85%] |
| kv-low | long-first | 11.781s | 7.919s | +32.78% [32.36%, 33.20%] |

所有 short-first 单元中两种策略的差异均小于 1%；long-first 的受限单元中差异才明显。不是所有现有组合都差，也不是简单“短请求先到就更好”：在 state-low + instant 中 short-first 的短 p95 反而高于 long-first，需结合实际激活与第 5.3 节实现问题分析。

必须保留代价：long-first 的 state-low 中，VTC→instant 的 long p95 为 12.069→12.159s，makespan 为 12.072→12.161s，分别退化约 0.75% / 0.74%；cap-control 对应约 0.76% / 0.75%。kv-low 的 long p95 和 makespan 仅有约 0.39% / 0.35% 的平均改善，CI 均跨 0。完整 16 个单元、所有重复和次指标区间见[结构化分析](results/m2_hybrid_factors_20260910/analysis/analysis.json)与[统计表](results/m2_hybrid_factors_20260910/analysis/report.md)。

实际激活也不同：long-first 的 cap-control/rep10 中，VTC 首批累积为 2 个 long + 6 个 short，instant 为 1 个 long + 7 个 short；后续短请求有不同的准入机会。这是服务轨迹观察，不是提前知道请求输出的 oracle 排序。

**观察：同一候选集、同一模型中，独立约束和已经进入执行的请求共同影响队列排序的结果。**

**具体问题限定：当前 token-VTC + 固定准入预算组合，不能由 token 优先级本身推断请求会何时准入、获得什么尾延迟；已有安全检查并未消除对资源条件和到达顺序的敏感性。** 这是可重复的局部机制问题，不是 VTC 公平保证反例。Instant 改变了公平计量，没有证明上述尾延迟降低满足相同 token-service 约束。

**Insight：资源瓶颈与候选集合必须一起解释 admission 的后果，而不是只比较固定排序或一个总 token cap。** 仍需共同目标下的可避免损失见证，才能将局部敏感性升级为对先进方法的性能缺陷结论。

### 5.3 对照中发现的具体实现问题：继续执行被重复计入新 slot 需求

这项分析是在看到 E2 后追加的探索性诊断，不是预注册结果，也没有改动正式实验中的代码行为。

在正式 `state-low / short-first / VTC / rep10` 的一处 `request_limit` 事件中：running=2，selected=5，request slots free=5，runtime cap=8，剩余 chunk budget=1536。selected 的 5 个请求包含 1 个继续执行的已有 chunk 和 4 个新请求；对应轨迹随后只有 7 个 active holder。相同输入的 cap-control 则 selected=6，随后达到 8 个 holder。

源码的数量门限是：

```text
len(can_run_list) >= min(runtime_cap - running_bs, request_pool.available_size())
```

但 [`ReqToTokenPool.alloc`](../../../python/sglang/srt/mem_cache/memory_pool.py) 的实际新增需求是 `len(reqs) - len(reusing)`，已有 chunk 复用 `req_pool_idx`；[`PrefillAdder.add_chunked_req`](../../../python/sglang/srt/managers/schedule_policy.py) 又确实将继续执行的 chunk 放入 `can_run_list`。因此门限的数量语义与 allocator 的新增需求不一致。

使用实际源码方法、而不是重新手写算法的[只读复核](verify_m2_slot_admission.py)确认：2 个 running + 1 个 existing chunk + 4 个 new 时门限已经拒绝下一个候选，但实际 allocator 可以接受 existing chunk + 5 个 new，只消耗 5 个空闲 slot，最终 active=8。本地与远端方法源码哈希一致；[复核结果](results/m2_hybrid_factors_20260910/slot_admission_witness.json)保留完整输入与边界。

**具体问题：在请求池紧张、存在复用 slot 的继续执行请求时，被测门限可能过早停止准入。** 这恰好说明“本轮参与执行的请求数”与“新增资源持有者数”必须区分。它是请求池计数的实现问题，并非 KDA 独有；可以作为具体 motivation 反例，单点修复本身不等于完整论文贡献。

`state-low` 不只缩小 state，还经初始化缩小有效并发和 request pool；所以它与 `wide + cap8` 不保证所有准入行为相同。短请求尾延迟的额外差异不能据此归因于 physical state tax 或 KDA compute。上述 132 组运行没有修正；后续[最小修正配对实验](m2_slot_reuse_batching_opportunity.md)单独验证其端到端贡献，不将旧组间差异直接当成修复收益。

## 6. E3：已经准入的 holder 会约束后续准入机会

8 个 P512 背景请求，输出长度为 32 或 256；在 t=3s 提交相同的 P512/O16 probe。预先要求 probe 到达时 O32 已全部释放、O256 仍有 8 个 holder，否则该 lifecycle 对照无效。

输出长度干预同时改变计算工作量和占用时间。通过 wide、cap-control、kv-low 对照定位额外排队，不能将它包装成固定总工作量或纯 lifetime 单因素实验。探针时刻由正式实验之前的 smoke 决定，修改记录保留在冻结协议。

![E3：占用持续时间与后续准入](figures/m2_hybrid_holding.png)

| 配置 | O32 probe queue | O256 probe queue | O32 / O256 probe TTFT | O32 / O256 probe JCT |
|---|---:|---:|---:|---:|
| wide | 0.004s | 0.034s | 0.125 / 0.153s | 0.923 / 0.959s |
| state-low | 0.004s | 10.966s | 0.124 / 11.083s | 0.924 / 11.883s |
| cap-control | 0.004s | 10.993s | 0.124 / 11.111s | 0.926 / 11.913s |
| kv-low | 0.004s | 0.034s | 0.125 / 0.153s | 0.934 / 0.965s |

40/40 正式 E3 运行满足预设 holder 条件：O32 在 probe 到达时已经全部释放，O256 均仍有 8 个活跃 holder。state-low 中 probe 在首个背景 holder 释放后平均 1.82ms 获得首次可见激活，cap-control 为 1.49ms；wide/kv-low 则在背景结束前就能激活 probe。

state-low/rep10 的 probe 拒绝记录为 running=8、request/state free=0、KV free=126552，说明此处等待不是 KV 不够。cap-control 的相同长输出对照却在 state 池仍宽裕时重现约 11s 排队，定位到 slot/concurrency 的阻塞机制。

用差分消去 wide 中同样增加背景计算的影响：`(O256 - O32)_regime - (O256 - O32)_wide`，probe queue 的 run-level 配对差分为：

| 相对 wide 的配置 | 额外 queue 效应 | 95% CI |
|---|---:|---:|
| state-low | +10.932s | [10.921s, 10.948s] |
| cap-control | +10.960s | [10.840s, 11.107s] |
| kv-low | +0.0005s | [-0.0169s, 0.0178s] |

**观察：已激活请求持续持有有限 slot，会把后续请求的准入机会推迟到释放时刻；当前待处理 token 数不足以单独说明这种等待。**

**问题限定：只知道“当前能否放下”不表示已经评价“本次激活会占用多久、影响后续谁”。但这组实验还没有证明 FCFS 在当时可用信息、相同目标下做了可避免的错误选择。** probe 到达时所有不可抢占 slot 已被占满，单纯重排 waiting queue 已经没有即时修复空间。

**Insight：需要检查准入决策的跨时刻后果，但不等于必须预测 lifetime。** 本实验同时改变输出计算和 holder 持续时间；cap-control 成功重现说明机制并非 KDA 独有。`kv-low` 在 E2 是 KV-bound，在 E3 却没有耗尽 KV，因此“瓶颈”是配置与 workload 的关系，不是配置名称自带的性质。

## 7. 现有工作已经解决到哪里

本次源码的 `get_num_allocatable_reqs`、`PrefillAdder` 及 hybrid pool 初始化已经存在并发与容量保护。正确叙述是“容量安全之外的选择后果还要检查”，不是“SGLang 忽略 state”。

[Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) 已研究 chunked prefill 和 stall-free batching，因此 prefill/decode 干扰与 token budget 控制不是本研究的新问题。

[FairBatching](https://arxiv.org/html/2510.14392v1) 已用 decode slack 和批执行时间控制 batch formation，其时间模型包含 new tokens 和 context。需要检查 hybrid 激活/占用因素在正确容量支持、本机重校准后是否仍造成剩余损失，不能仅因模型写法不同就宣布该方法失效。

[LPRS/APC](https://arxiv.org/html/2606.09061v1) 的特征已包含 batch request count、context 和 GPU memory，且控制 active prefill。其“请求数 + 时间预算 + 活跃请求控制”与这里有直接重叠，不能称为 token-only baseline。它能否从已有特征学习当前效应，必须靠校准后 held-out 实验判断。

以上为本轮核对的具体机制，不是完整文献穷尽调查；整体 fairness/memory 相关工作仍见[完整审查](topic_review_20260908.md)。

## 8. 从具体缺陷推进到收益与设计空间

源码与轨迹已经支持第 5.3 节的具体缺陷。接下来不是要求“所有先进方法都失败”才承认问题，而是量化最小干预的收益、保留负对照，并确定更强已有方法是否已覆盖该机会。

1. **已完成具体实现因素的独立消融。** [后续 100 组实验](m2_slot_reuse_batching_opportunity.md)保持 VTC 与其他预算不变，确认 slot 修正消除主对照提前拒绝并改善 short mean 8.37%；两类负对照不触发，mixed 收益明显较小。继续比较算法时应使用正确门限，不把已知 bug 留给 baseline。
2. **公平目标相同的可避免损失见证。** 固定 token-service 约束、权重与 SLO，记录持续 backlog 窗口；比较有相同信息和动作权限的候选安排，不能把 instant 的不同公平目标视为免费收益。
3. **更强方法的正确适配与 held-out。** 给 baseline 保留 state/KV 保护、本机 profile、合适参数搜索；至少测试时间预算/active-prefill 控制，检查剩余误差是否随 activation 数与 state headroom 系统性变化。
4. **Hybrid 增量归因。** cap-control 是必要但不充分的对照；还要做合理的 KV-only/算子对照或匹配服务曲线的控制，区分通用并发效应与固定 recurrent state 带来的新增约束。
5. **更真实的 operating point。** 自然形成的资源压力、启用 graph、更多负载与请求长度、重复服务器启动。人工收紧独立池只能说明机制存在，不说明生产默认配置会频繁触发。

旧 f16 配置收益及 f32 held-out 失败继续保留，但仅作为“batch admission 的效果不由一个静态 cap 普遍决定”的辅助证据，不再担任 hybrid 特有缺陷的主证明。

### 8.1 当前适合保留的 M2 问题陈述

**Hybrid 服务中的请求准入既安排本轮 token 工作，也建立持续占用资源的 holder。当前被测排序与预算组合的服务结果依赖首次激活数、独立 state/KV 约束及已有 holder 的释放；安全容量检查并不能替代这些选择后果的评价。**

| 判断 | 本轮状态 |
|---|---|
| 需要区分首次激活与继续执行 | 已有 E1 直接资源轨迹 |
| 独立约束影响实际准入和排序结果 | 已有 E2 多重复、两顺序、拒绝原因与资源对照 |
| 持续占用影响后续准入 | 已有 E3 配对与 cap-only 对照 |
| 被测实现存在具体计数局限及局部服务收益 | 已有源码见证、100 组修正对照；主对照 short mean +8.37%，插值 p95 +29.01%，最大延迟无显著改善 |
| 现有先进方法在 hybrid 上是否还留有设计空间 | 未验证；用于界定后续贡献，不是承认具体缺陷的前置条件 |
| 必须引入向量公平、联合控制或 lifetime predictor | 未证明，也不是本轮验收目标 |

## 9. 复现与数据

- Runner：[run_m2_hybrid_factors.py](run_m2_hybrid_factors.py)。
- 分析：[analyze_m2_hybrid_factors.py](analyze_m2_hybrid_factors.py)；作图：[plot_m2_hybrid_factors.py](plot_m2_hybrid_factors.py)。
- 完整结果：[analysis.json](results/m2_hybrid_factors_20260910/analysis/analysis.json)、[统计报告](results/m2_hybrid_factors_20260910/analysis/report.md)。
- 原始数据留在远端容器 `/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_hybrid_factors_20260910/formal01`，宿主机对应 `/mnt/nvme0/mllm/sglang` 下同一路径。
- 本地同步结构化分析、文字结果和图；每个原始 artifact 的 SHA-256 及源代码/启动命令 manifest 保留在分析文件中。

132/132 正式 artifacts 完整，2119/2119 请求成功且输出长度精确匹配；零 cached token、retraction、trace drop，所有结束后 active 资源归零，所有事件满足 one-state-per-active。客户端最大到达偏差约 3.03ms，小于预设两组 50ms 的提交间隔，但不据此假定服务端每轮都已看到全部候选。实验/分析回归测试 27 项通过；另有原 tracker 的 5 项组件测试通过。实验结束后 GPU 6/7 均为 1MiB、0% utilization，未保留占用 GPU 的实验服务。

远端容器内重新分析，不需要重新跑 GPU：

```bash
/workspace/sglang/.venv/bin/python \
  /workspace/sglang/benchmark/test_motivation/Ledger/analyze_m2_hybrid_factors.py \
  /workspace/sglang/benchmark/test_motivation/Ledger/results/m2_hybrid_factors_20260910/formal01 \
  --output /workspace/sglang/benchmark/test_motivation/Ledger/results/m2_hybrid_factors_20260910/analysis
```

本地从同步后的结构化结果重画三图：

```bash
python3 benchmark/test_motivation/Ledger/plot_m2_hybrid_factors.py \
  benchmark/test_motivation/Ledger/results/m2_hybrid_factors_20260910/analysis/analysis.json \
  benchmark/test_motivation/Ledger/figures
```
