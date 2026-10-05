# M2：容量可行的准入，不等于合适的 batch formation

> 2026-09-10 逻辑定位更新：本文作为历史 cap/batching 干预与配置收益记录保留。M2 的最新主线见[激活成本、资源瓶颈与占用持续时间](m2_hybrid_admission_factors.md)，不再仅靠调 cap 收益论证 hybrid 下的新问题。本文原始判据与负结果保持不变。

实验批次：2026-09-09，held-out 延续至北京时间 2026-09-10。模型：Kimi-Linear-48B-A3B，TP=2，物理 GPU 6、7。

> 实验记录：108 组正式运行、12896 个请求全部完成，完整性检查通过。f16 通过预注册的有限代价收益判据，f32 held-out 不通过。
> 本文研究现有组合的具体问题，不预设新算法；原生机制对照不冒充完整 SOTA 复现。

**当前判断：M2 的 admission/execution 耦合及局部配置收益已经有实验证据；跨 workload 的稳定收益、先进策略仍有缺陷、以及 hybrid 特有归因尚未闭环。** f32 的 FAIL 是固定 cap 调整未获得预注册的可迁移收益，不是“admission 没有影响”。

## 1. 如何从 M1 递进

[M1](m1_token_fairness_request_shape.md) 先说明：相同 token 总量可以对应不同请求数和 hybrid state/KV 负担，标准 token 服务计量不足以描述这些资源差异。它不是说 VTC 完全没有使用请求长度，也不是说 VTC 违反其理论保证。

M2 再追问：即使底层已经检查 state/KV 能否放下，按现有队列顺序和固定预算把请求加入 batch，是否就得到了合适的服务结果？

这里的研究对象是 **request admission / batch formation 的组合机制**：候选请求顺序、并发门限、每轮 prefill 工作量，以及 prefill/decode 的安排。不是否定 continuous batching 这一执行框架。

本地代码的对应位置：`scheduler.py::get_num_allocatable_reqs` 用运行并发门限和 request pool 限制新请求数；`_get_new_batch_prefill_raw` 调用 `calc_priority` 后按队列顺序尝试加入请求；`schedule_policy.py::PrefillAdder` 检查容量和 prefill/chunk 预算。本轮不修改这些生产调度路径，也不声称 SGLang 不存在其他可选调度机制。

## 2. 需要证明的不是普通的吞吐与延迟权衡

必须分开三层问题：

| 层级 | 需要的证据 | 本轮的对应实验 |
|---|---|---|
| Admission 会改变执行和邻居延迟 | 固定内存池、graph、输入，干预准入门限 | 4 个并发门限及 queue/active/decode trace |
| 当前组合存在可改善的运行点 | 新重复中改善一个目标，不以显著牺牲其他目标换取 | 预先冻结 cap96 与 cap48，双租户 p95 和 makespan 联合判定 |
| 已有强方法仍有 hybrid 特有缺陷 | 校准后的强动态 baseline，加 hybrid 因果对照 | 尚未完成，不能由前两层自动推出 |

特别地，fairness gap 变小不一定是改善：拖慢原来较快的租户也能缩小 gap。本文保留两个租户的绝对 p95 和 makespan。相同最终完成 token 数，也不能替代持续积压窗口中的服务公平性检查。

## 3. 实验协议与对照身份

完整协议：[m2_batch_frontier_protocol_20260909.md](m2_batch_frontier_protocol_20260909.md)。确认选择：[m2_confirmation_selection_20260909.json](m2_confirmation_selection_20260909.json)。

- 固定 state pool=480、KV capacity=196608、启动 max-running=96、decode graph coverage=96；关闭 prefill graph。
- 只在空闲时修改运行时 admission cap：9、24、48、96；检查服务端读回值和实际 peak active。
- 原生模式：separate-4096、mixed-4096、mixed-1024；max-prefill-tokens=16384。
- 排序：本地 token-VTC，以及已有的 frozen-score instant-share 诊断对照。后者不是完整 DRF，也不是本文新设计。
- 每组使用同一套 input token IDs、输出长度和到达偏移；重复之间换输入并交替先到租户。唯一 cache salt，要求无 prefix hit、无 retraction、无请求失败。
- 每个配置/重复配套两个 FCFS 单租户 isolation；其输入和到达偏移来自对应混跑请求的原样子集。

每个 workload 有 8 个 wave，间隔 1.5s。每个 wave 两租户各提供 16384 input + 512 output tokens：packed 为 1 个长请求；f16 中 split 为 16 个 1024/32 请求；held-out f32 为 32 个 512/16 请求。因此 f32 改变请求粒度而不改变总 token 工作量，但也同时改变单请求输出长度，不是 KDA 单因素实验。

每轮统计是有限 wave replay 的完成时间，**不是稳态吞吐容量**。不同配置的样本均值前沿也不是最优调度器的前沿。

输入是为控制变量构造的合成 token 序列，不是自然业务请求。两个确认会话的源代码、协议和 runner 哈希完全一致；不同 workload 分会话运行，因此不能将 workload 间绝对延迟之差归因于唯一一个因素。

## 4. 实验 A：原生 batching 与排序扫描

共 3 种模式 × 4 个 cap × 4 类运行（两隔离、两混跑）=48 组正式运行、4896 个请求。所有正式运行完整性检查通过；每个混跑点只有 1 次重复，定位为描述性扫描。

![M2 原生 batch formation 扫描](figures/m2_batch_frontier_screen.png)

### 4.1 并发并不是越大或越小越好

separate-4096 + token-VTC 的扫描结果：

| Cap | Split p95 | Packed p95 | Makespan | Split queue p95 | Split active/request | Split decode interval |
|---:|---:|---:|---:|---:|---:|---:|
| 9 | 13.444s | 18.344s | 26.994s | 12.823s | 0.657s | 16.9ms |
| 24 | 7.247s | 20.280s | 22.230s | 5.608s | 1.525s | 43.3ms |
| 48 | 7.895s | 20.202s | 21.323s | 4.276s | 2.593s | 77.5ms |
| 96 | 10.709s | 20.101s | 20.853s | 3.638s | 4.473s | 138.3ms |

从 48 放宽到 96 时，排队减少，但 active residency 和 decode interval 上升，短请求 p95 反而变差。反向压到 9 又引起明显排队和完成时间代价。p95 queue 与 mean active 不是可直接相加的同一种统计量；这里只用于定位干预伴随的变化。

**观察：准入不是只改变“在外面等多久”，也会改变“进去以后执行多久”。** 本轮固定了 graph coverage，因而排除了上一轮“不同 max-running 改变 graph 捕获范围”的混淆；尚未分解到 KDA/MLA/MoE kernel 的各自贡献。

### 4.2 原生 mixed chunk 不是自动修复

在 cap96 + VTC 下，separate-4096 / mixed-4096 / mixed-1024 的 split p95 分别为 10.709 / 10.910 / 16.846s，makespan 分别为 20.853 / 23.082 / 31.969s。至少在这些已测配置中，开启混合或把 chunk 缩小没有自动解决问题。

不能把这组结果写成“Sarathi-Serve 或 FairBatching 无效”：本轮只用了 SGLang 的原生机制开关，没有复现它们的完整策略和调优流程。mixed-1024 的 cap48/96 实际 peak active 都只有 27，说明并发上限并不总是实际约束。

### 4.3 排序与 batching 相互作用

同为 separate-4096，cap24 时，VTC 换成 instant-share 后 split p95 从 7.247s 降到 3.948s，而 packed p95 为 20.280s/20.151s，makespan 为 22.230s/22.657s。cap96 时两者更接近：split p95 为 10.709s/10.348s。

这提示候选排序效果依赖 admission 工作点。它尚不证明 instant-share 优于 VTC：只有一次扫描，且两种策略的公平语义不同，没有等服务保证的对照。它是机制诊断，不是最终算法比较。

### 4.4 扫描中的质量问题如何处理

mixed-4096 的第一组 split-only/cap48 isolation 存在首 wave 较慢的现象，后续波次恢复，客户端到达偏差不足以解释。其 slowdown 比值不作为主结论。所有原始数据仍保留，不删除不利样本。

因此在查看确认结果之前已修改预热协议：两个 cap 的 split-only、packed-only、mixed 都先跑完整 8-wave 预热，排除全部预热，保留全部正式重复。

## 5. 实验 B：冻结候选后的独立确认

固定 separate-4096 + VTC，比较 cap96 与 cap48；f16 和 held-out f32 分别做 rep10--14 的 5 个配对重复。主指标是 split p95，相对改善的均值至少 10%、95% 配对 bootstrap CI 下界大于 0；packed p95 和 makespan 的相对退化 CI 上界各不超过 5%。两个 workload 单独判定，不合并、不重新选 cap。

此处允许小幅非劣代价，不是要求严格 Pareto 支配；即使通过，也只能称为“在预定义代价约束下有收益”。5 个 run 的 bootstrap CI 仍是小样本证据，不替代更多负载和线上 trace。

![M2 独立确认与 held-out 结果](figures/m2_batch_frontier_confirmation.png)

### 5.1 f16：独立确认通过

30 组正式运行（10 个混跑与 20 个匹配 isolation）、2720 个请求，另有 6 组完整预热被预先排除。正式运行的完整性检查全部通过。

| 指标 | Cap96 | Cap48 | 配对相对改善及 95% CI |
|---|---:|---:|---:|
| Split p95 JCT | 10.635s | 7.825s | +26.42% [26.19%, 26.72%] |
| Packed p95 JCT | 19.994s | 20.116s | -0.61% [-0.76%, -0.41%] |
| Makespan | 20.745s | 21.229s | -2.33% [-2.48%, -2.13%] |

绝对列是 run-level 指标的均值；改善列是每个配对的 `1 - cap48/cap96` 再取均值。负改善表示退化，不把它隐藏成“全面更好”。全部 5 个配对的 split p95 同方向改善。

**结论：相同 VTC、池容量和 graph 下，存在一个比 cap96 更符合预设延迟/代价目标的已测运行点。** 这是经过新重复确认的有限代价权衡，仍不是严格 Pareto 改善，也没有证明 cap96 位于真正的性能前沿之外。它只证明该 workload 的配置收益，不是新算法收益。

对应机制观测：split queue p95 从 3.796s 增到 4.274s，active/request 从 4.388s 降到 2.539s，decode interval 从 135.7ms 降到 75.8ms；mean decode batch 从 15.03 降到 14.17，peak active 从 96 降到 48。也就是说，更早准入的排队收益不足以抵消准入后的执行拉长。

两租户各完成 139264 单位 `input + 2*output` 工作量。这只是固定完成工作的完整性检查，不证明运行过程中的公平保证相同。

### 5.2 f32：held-out 不通过，不能外推固定 cap 收益

同样 30 组正式运行、5280 个请求，另有 6 组预热。所有正式记录通过完整性检查，但收益判据不通过：

| 指标 | Cap96 | Cap48 | 配对相对改善及 95% CI |
|---|---:|---:|---:|
| Split p95 JCT | 7.097s | 7.071s | -0.40% [-8.19%, 7.30%] |
| Packed p95 JCT | 20.140s | 20.197s | -0.28% [-0.47%, -0.11%] |
| Makespan | 20.935s | 21.336s | -1.92% [-2.02%, -1.83%] |

主指标既没有达到 10% 平均改善门槛，CI 也跨过 0。仅 3/5 配对改善，另外 2/5 反向。绝对均值略降、平均相对改善却为负并不矛盾：预注册指标是逐对计算 `1 - candidate/baseline` 后平均，不是对两个绝对均值再求比值；不同重复的 baseline 不同。

| Rep | 预设 wave 内先到租户 | Cap96 split p95 | Cap48 split p95 |
|---:|---|---:|---:|
| 10 | packed | 7.575s | 7.015s |
| 11 | split | 6.404s | 7.139s |
| 12 | packed | 7.568s | 7.008s |
| 13 | split | 6.371s | 7.154s |
| 14 | packed | 7.567s | 7.041s |

这是一个值得追踪的次级观察：本组数据的收益方向与预设到达顺序一致地变化。但每个 rep 也更换了输入 token 内容，实际请求接收还存在时序抖动，故不能把它直接写成“到达顺序导致反转”。需要相同输入、只反转顺序的配对实验，并记录服务端实际接收和首次准入顺序。

执行侧干预仍然成立：cap96→48 时，split active/request 从 2.589s 降到 1.400s，decode interval 从 160.1ms 降到 80.8ms，queue p95 从 4.248s 增到 4.741s。但这些变化没有稳定转换为短请求端到端 p95 收益。这正是为什么不能把 active time 改善本身当作系统收益。

### 5.3 实验、问题和 insight 一一对应

| 实验 | 得到的具体问题/观察 | 可以保留的 insight |
|---|---|---|
| 固定池与 graph，扫描 cap、chunk 和排序 | admission 同时改变排队、执行和其他请求延迟；实际约束随配置变化 | 安全可行的准入量不足以决定合适的运行组合 |
| f16 独立配对确认 | 当前 cap96 组合不是预设延迟/代价目标下更好的已测选择，cap48 有有限代价收益 | 在不改变公平排序的情况下，batch admission 工作点仍有可量化调整空间 |
| f32 原样 held-out | 同一个 cap 调整出现方向反转，整体没有稳定收益 | 不能把一个 workload 上的静态限流收益外推；效果依赖请求形态与运行时条件 |

可用于 M2 的问题陈述：**在当前 token-VTC 与固定 admission/batch budget 的组合下，容量约束满足并不意味着端到端服务目标得到较好满足；准入会内生改变执行代价，而简单静态门限调整的收益并不稳定迁移。** 这里没有声称全部静态配置都不可行、动态控制必需、或先进动态方案仍然失败。

## 6. 现有工作是否已经处理这个问题

| 对照 | 已处理的相关机制 | 本轮状态 |
|---|---|---|
| 本地 token-VTC + SGLang admission | token 服务排序与底层容量/预算约束 | 已实测；不泛化成全部 VTC 实现 |
| 原生 mixed/chunk controls | prefill/decode 混合及 chunk 工作量 | 已扫描；不是完整论文复现 |
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | chunked prefill、stall-free batching | 需忠实适配后比较 |
| [FairBatching](https://arxiv.org/html/2510.14392v1) | 公平感知 batch formation、时间预算、decode 紧迫性 | 本轮未实现，重要直接对照 |
| [LPRS/APC](https://arxiv.org/html/2606.09061v1) | batch latency 预测与 active prefill 控制 | 本轮未实现；[官方代码](https://github.com/Charmstok/Chunked_Prefill_Serving)可供适配与本机重校准 |
| [Equinox](https://arxiv.org/html/2508.16646v1) | 多租户公平与调度的联合考虑 | 整体故事的重要相关工作，不能忽略 |

因此，“token budget 不等价于 batch time”“根据 batch 时间调 admission”本身不能当作新发现或独立技术贡献。尚待证明的是：这些方法在正确支持 hybrid 容量、重新校准之后，是否仍存在与 request-state/KV 组成相关的系统性损失。

尤其 [LPRS/APC 的特征表](https://arxiv.org/html/2606.09061v1#S3.SS2.SSS1) 已包含 batch request count、decode context 统计和 GPU memory 信息，不能把它描述成只看 token 数的弱 baseline。后续必须在 Kimi-Linear 上重新 profiling/训练，并明确各方法的服务约束和公平目标是否一致；不同目标下的排名不能直接用来宣称方法有缺陷。

## 7. 还不能声称什么

- 没有完成强动态 SOTA 的实装、校准、held-out 比较，不能说现有先进 batch formation 普遍失效。
- 没有同架构可控消融或匹配的非 hybrid 对照，不能说干扰是 KDA 独有，或 fixed state tax 单独造成全部差异。
- 没有受控的 per-operator GPU 计时，不能把 CPU launch gap 或端到端 decode interval 当成 kernel 时间。
- 没有持续积压窗口的公平性和预注册业务 SLO goodput 对比，不能以完成 token 相等或更小 latency gap 声称公平保证改善。原始 summary 沿用的 60s 阈值不作为本轮 SLO 结论。
- 没有缓存复用/驱逐实验，本轮不能推导 eviction 设计收益。

## 8. 证据与复现

远端原始记录保留在容器 `/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_batch_frontier_20260909/`，宿主映射为 `/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/m2_batch_frontier_20260909/`。

`screen01` 保存 separate-4096 正式结果；模式切换时端口 TIME_WAIT 检查报错，未影响已落盘结果。修正探测后，`screen02` 保存两个 mixed 模式的完整结果，没有重跑覆盖旧数据。

本地同步分析摘要与图，不复制全部原始 trace：[扫描摘要](results/m2_batch_frontier_20260909/screen_analysis/report.md)、[结构化扫描结果](results/m2_batch_frontier_20260909/screen_analysis/analysis.json)。每条原始记录有输入 fingerprint、完整参数读回、请求记录和资源/batch trace；manifest 保留源代码 hash、GPU UUID 和启动命令。

确认分析：[汇总表](results/m2_batch_frontier_20260909/confirmation_analysis/report.md)、[结构化结果及四个会话 manifest](results/m2_batch_frontier_20260909/confirmation_analysis/analysis.json)、[f16 预注册判定](results/m2_batch_frontier_20260909/confirmation_analysis/gate_f16.json)、[f32 预注册判定](results/m2_batch_frontier_20260909/confirmation_analysis/gate_f32.json)。

实验入口：[run_m2_batch_frontier.py](run_m2_batch_frontier.py)；分析：[analyze_m2_batch_frontier.py](analyze_m2_batch_frontier.py)；确认判定：[confirm_m2_batch_frontier.py](confirm_m2_batch_frontier.py)；绘图：[plot_m2_batch_frontier.py](plot_m2_batch_frontier.py)。

远端 13 项单元检查通过；本地 8 项分析检查通过，runner 检查因本地缺少 aiohttp 跳过，已由远端补齐。本轮未修改生产调度代码。服务由 runner 清理，结束后 GPU 6、7 均为 1 MiB 显存、0% utilization。

## 9. 离完整的问题证明还差什么

1. **定位 f32 反转。** 固定输入内容和工作量，只交换租户到达顺序；同时记录实际 enqueue/admit 序列、VTC counter、batch 组成。用不同 arrival gap 排除毫秒级提交抖动，再决定能否归因到排序与 admission 的耦合。
2. **检查已知方法能否解决。** 在当前模型上忠实适配、校准 FairBatching 或 LPRS/APC 等直接相关机制，与调优的原生配置在相同服务约束下比较。若能消除损失，应收缩问题，不先发明新 predictor。
3. **验证 hybrid 的额外作用和外部有效性。** 固定目标请求，控制邻居请求数、上下文与输出长度；做适当的非 hybrid/算子对照，区分通用 batching 干扰与 fixed-state 作用。再加入不同到达率、真实 trace、合法 backlog 公平窗口和业务 SLO。

这三项未完成前，应将本文称为“受控机制验证与局部收益/负结果”，不能称为“现有 SOTA 在 hybrid 上缺陷的完整证明”。
