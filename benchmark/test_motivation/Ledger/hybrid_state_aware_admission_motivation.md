# Hybrid LLM 多租户调度的问题发现：资源计量、服务机会与隔离

> 故事线整理：2026-09-08；M2 实验更新至 2026-09-17，32组准入筛选/原生对照、12组原生参数校准及16组Mixed near补充筛选均已完成；原Mixed far校准无效，未执行原定32组Mixed全矩阵。
> 范围：Kimi-Linear-48B-A3B，KDA + MLA，单 serving instance，A800，TP=2，无 prefix reuse。早期实验使用GPU6/7；9月16日池管理对照按用户新授权使用空闲GPU0/1，后续准入时机筛选使用空闲GPU4/5，不跨卡组直接配对延迟。
> 当前阶段：验证现有方法的问题，不预设解决方案。2026-09-17重新定位：M1检查token计量盲区；M2回到hybrid如何影响租户之间的服务机会与隔离；M3检查现有成本校准能否解释或修复这些差异。先查明跨租户影响，再在明确公平约束下探索如何更快完成请求。单租户batching收益降为辅助证据，历史数据和gate不变。
> 当前M2入口：[租户服务机会与隔离](m2_tenant_service_opportunity_isolation.md)。这是问题定义与验证顺序的修订，不表示新的双租户实验已完成或旧单租户结果已证明公平收益。
> 新主线的首步已完成：[历史双租户轨迹审计与新图](m2_tenant_service_audit_20260917.md)，60组旧确认数据、8000请求，原始SHA256全部匹配，新增GPU样本0。已重建租户输出时序与资源积分；逐事件counter、租户waiting和拒绝/释放记录缺失，[补测规格](m2_tenant_service_f1_measurement_spec_20260917.md)先补这些证据，不继续单租户加速扫描。
> 2026-09-18：[新观测与固定目标六条件预检](m2_service_preflight_results_20260918.md)完成，12次预检+12次warmup、观测开/关输出检查通过。短目标p95随等token邻居形态变化：长邻居5.059s、短邻居3.596s；无state/KV容量拒绝，且共同waiting窗口短。新增的是过程证据和描述性形态观察，不是hybrid因果确认或同公平约束收益。下一步补共同积压/独立重复，而不是转回单租户加速。
> 本轮依据：[2026-09-08 完整审查](topic_review_20260908.md)。该审查中的方案必要性讨论属于后续阶段；当前以本文的问题发现顺序为准。旧实验 gate 保持原样，新编号不回写历史判定。
> 理论进展：[机制分析、最小反例与收益空间](mechanism_counterexamples_20260908.md)。已完成条件化推导与精确有限模型验证，不替代 GPU 实验。

## 1. 一条主线

Hybrid attention 为每个 active request 引入固定 recurrent state，同时保留随上下文增长的 MLA KV。因此，相同总输入和输出 token 被组织成不同数量、不同长度的请求时，资源需求会变化。

这种差异通过公平排序、request admission和batch formation进入多租户服务：首次激活某租户的请求会建立持久state holder，继续已有chunk不重复支付这份激活成本；一方的state/KV持有、执行与释放可能改变另一方的准入和推进机会。该跨租户链路需要按计量、准入、执行和交付逐步验证，不能仅凭资源峰值或整体JCT推断。

当前首先回答：**请求形态带来的hybrid资源差异，如何在现有公平调度中转换成租户之间的服务机会与隔离差异？哪些损失可以在相同公平约束下避免？** M1中的单位token资源成本不是业务价值或已经确定的公平价格。暂不回答如何设计新scheduler，也不以联合控制、历史账本或向量资源为预设答案。

这是一篇候选的多租户调度论文。早期 lifetime request-ranking 和 cache eviction 方向保留在[历史归档](hybrid_state_aware_admission_history_20260826.md)，不再作为并列贡献。

| Motivation | 检查的现有方法 | 已有观察与具体问题 | 当前结论 |
|---|---|---|---|
| M1：Token 计量遗漏请求数维度 | 按输入/输出 token 计量服务的本地 VTC 风格策略 | 相同 token 总量对应不同 state/KV 需求，且混跑 slowdown 明显不同 | token 计量不能作为 hybrid 资源压力的充分代理；尚未证明该盲区单独导致服务差异 |
| M2：不同token资源成本如何转换成租户服务分配 | 公平计量/排序与state/KV准入、batch formation的组合 | 旧双租户f16中，同VTC改变准入使短p95改善26.42%，但输出50%时刻10.390→10.729s；五个配对的最大/最终counter gap分别相同，实际占用和交付仍不同 | 同计量汇总、尾延迟与实际服务进度不是同一对象；完整counter/waiting/拒绝事件缺失，尚不能判断同公平约束收益或state因果。单租户加速不能替代该链路 |
| M3：静态校准未消除混跑服务差异 | 固定 request-tax / isolated-profiled VTC | 已测 profile 改善有限；instant-share 的相对表现也随并发区间变化 | 当前校准方式没有解决观察到的差异；是否因 batch-dependent cost、记账时机或动作空间仍待区分 |

三者依次检查：**计量是否充分、资源差异如何影响跨租户服务、现有成本校准是否足以解释或修复。** 然后才进入公平约束下的效率优化；整体mean、吞吐或makespan仍用于披露代价，但不再代替M2的租户级结论。M3不是“证明某个新设计必需”。Pulse暂作辅助观察，不凑成另一条已经成立的核心motivation。

### 1.1 什么算一条成立的 motivation

每条采用同一证据链：**被检查的方法与目标 → 可控实验 → 可重复观察 → 具体问题 → insight → 结论边界与剩余验证**。

- 资源形态不同首先是 observation；联系到现有方法遗漏的量及服务后果，才形成问题陈述。
- “缺陷”必须说明相对于什么目标：不能把等 slowdown 或无条件 JCT 隔离当作 VTC 的既有保证，也不能把容量检查未优化延迟写成内存安全实现错误。
- 有统计差异不自动有单因素因果关系；发现某方法在明确场景下有问题，也不要求先发明更好的算法。
- 问题是否存在、解释是否正确、问题是否已有方法解决、新方案是否有效，是四个不同判断。强 baseline 可以限定问题范围，但不是确认每个局部现象存在的统一前置条件。

### 1.2 已有机制分析的证据位置

- M1：最小模型中相同 token 量、甚至相同累计 state-time，也可因 active holders/释放时刻不同而阻塞新租户；但已占满且不可抢占之后，纯队列排序没有修复空间。
- M2辅助理论：推导了二请求模型中存在可避免损失的充分条件及零收益区间。收益依赖batch时间、允许的服务差及能否延迟准入，不能只由容量可行推导，也不能替代新M2的跨租户服务过程验证。
- M3：构造相同候选/计量状态、不同 running context 导致较好顺序反转的例子。在有限模型中可以改善 mean 而不改变 makespan，但 worst slowdown 反而恶化；不能将其写成公平收益。

这些是解释和待验证预测，不是三条新的实测 PASS。完整公式、源码对应、负对照与可复核结果见[机制分析](mechanism_counterexamples_20260908.md)。

## 2. 先固定指标与研究边界

### 2.1 “公平”不是单一指标

- **Token service fairness**：按约定输入/输出权重计量的服务差距。
- **Resource allocation**：state slots、KV tokens 的瞬时份额，以及各自的累计 resource-time。
- **Slowdown balance**：租户混跑时相对其独跑基线的延迟膨胀是否接近。

当前 partition 实验采用：

\[
d_u = \frac{JCT^{mixed}_{u,p95}}{JCT^{isolated}_{u,p95}}-1,
\qquad G=|d_{split}-d_{packed}|.
\]

报告中 gap=913.2% 表示两个 slowdown 相差 913.2 个百分点，不是慢租户的绝对 JCT 比邻居高 913.2%。每个容量配置需要匹配的独跑基线；跨配置比较还必须同时报告绝对延迟，防止分母变化误导解释。

[VTC 原论文](https://arxiv.org/abs/2401.00588) 针对其服务成本定义和 backlogged-client 条件给出服务差距保证。出现 resource mismatch 或 slowdown gap，并不等于 VTC 的理论保证失效。本文研究的是 token service 公平与 hybrid 资源需求、服务隔离之间的关系。

也不能笼统说 VTC 没有延迟保证：其 Theorem 4.11 针对特定低载条件给出 dispatch-delay bound，并依赖服务速率下界。该保证不等于本实验的 per-request p95 JCT 或等 slowdown。

当前问题评价同时报告token-service gap、worst-tenant slowdown/SLO、每租户绝对延迟和整体效率，不把它们合成尚未定义的优化目标。G仅作为辅助指标，因为拖慢较快租户也能缩小G。新M2必须分开记录内部counter、实际处理token和客户端交付，核验共同积压窗口并扣除窗口起始累计值；waiting-backlogged与仅有running请求不能混称。相同排序或最终完成token相同都不等于过程公平相同。公平窗口、口径及对照的非劣/效率代价门槛必须在新的正式干预前冻结，不沿用旧单租户门槛冒充公平约束。

主实验按独立 API 请求解释，不要求不同 partition 具有相同业务语义。若转向 task-parallel application，应另外报告整个 wave 的完成时间，而不是用 request p95 替代 application JCT。

### 2.2 当前系统与实验 baseline 的身份

本次代码版本的 SGLang 已有 state/KV 可行性检查和并发限制。这里研究的是这些可行性约束之上的分配策略。

实验中的 token VTC 是本地集成的 VTC 风格策略；request-tax/profiled VTC 是诊断性的成本变体；instant-share 是本项目实现的简单对照，并非本文声称的新算法或另一篇论文的直接复现。

instant-share 根据租户当前 normalized dominant state/KV share 排序，不累计历史服务。当前实现一轮排序只采样一次 share，并按 frozen score 排列请求，没有逐次更新假想 admission 后的份额。它是必要的简单对照，但不代表所有 instantaneous allocation，更不等同于完整 DRF。需要补充逐 candidate 更新 reservation 的 projected-share。

现有 profiled VTC 在 admission 时只记 request/input cost，output cost 随 decode 累加；它不是原论文已提出的 VTC-predict。未来输出的预记账与完成后校正必须作为强 baseline 纳入。

## 3. M1：Equal Tokens, Unequal Resource Demand

> 独立阅读版：[M1：相同 token 计量，不同 hybrid 资源代价与服务隔离](m1_token_fairness_request_shape.md)，包含完整三图、实验口径、文字分析和与 M2 的衔接。

### 3.1 开场图：相同 token 计量，不同资源负担与服务结果

![M1：相同 token budget、不同固定 state tax 与 slowdown](figures/m1_token_state_fairness.png)

**核心结果：在相同输入、输出 token 总量下，当前 token-VTC 实现中，短请求租户的 p95 延迟相对独跑膨胀至 10.34 倍，而长请求租户为 1.21 倍。混跑带来的延迟代价在两类租户间明显不对称。** 这就是 M1 首先要暴露的现有方法问题：相同 token 计量口径下，不同请求形态获得的服务隔离表现存在巨大差异。

- **(a) 计量相同：** packed 与 split 每 wave 均提供 16,384 input tokens 和 512 output tokens，`input + 2*output` 都是 17,408。
- **(b) 单位 token 固定 state 成本不同：** 相同 active-state 大小 S 下，1K 上下文的 `S/L` 是 16K 上下文的 16 倍。短请求的固定 state 开销分摊到更少的 resident tokens 上。
- **(c) 服务后果不对称：** 8 月 26 日 `f16-base` 实测中，packed/split 的平均 p95 slowdown ratio 分别为 1.21/10.34。更广的 split-factor 对照中，相同请求形态的 f1 gap 仅为 1.2 个百分点，而 f16/f32 达到 913.2/924.6 个百分点。

**Observation：token 计量相同的请求集合，可以同时呈现不同的单位 token 固定 state 负担和显著不对称的混跑 slowdown。** 资源负担与服务后果之间的具体机制，由后续容量和并发干预进一步分析。

*Figure 1 图注：A 是 offered token charge；B 是单请求初始 resident context 上的固定 state 项推导，不代表总显存或累计 state-time 的倍率，也不假设全部短请求同时准入；C 是各自相对匹配独跑基线的 p95 比值在 3 runs 上的均值，当前汇总未提供 CI。三个面板的并列展示尚未完成 fixed state tax 到 slowdown 的单因素因果归因。*

这张图先提出问题，下面两张实测图分别补上资源占用证据和请求形态扫描。M1 不引入候选新策略的胜负比较。

图的[向量版本](figures/m1_token_state_fairness.svg)、[数值与来源记录](figures/m1_token_state_fairness.json)和[绘图程序](plot_m1_opening_observation.py)均保留；它不改变原始实验 gates。

### 3.2 图 2：相同 token 总量对应不同的实测资源占用

![M1 实测 state 与 KV 占用](figures/m1_measured_resource_pressure.png)

**Figure 2. Token 预算相同，资源压力方向不同。** 固定 split factor=16、KV cap=196,608，分别展示 base 和 state-relaxed 配置下两租户的峰值资源占用。

在 state-relaxed 的同一配置中，短请求租户的 peak state 为 270 slots，长请求租户为 24；但 peak KV 则分别约为 94,845 和 135,086 tokens。Base 配置中，短请求租户的 state/KV peak 分别为 24/8,448，长请求租户为 18/100,166。

**Observation：相同 token 预算并没有让两类租户成为相同的资源使用者。短请求集合偏向请求级 state 占用，长请求集合偏向上下文 KV 占用；一个累计 token 数不能充分表征这两个方向。** 这不是“短请求在所有资源上都更贵”，而是公平计量掩盖了不同的资源构成。

*图注：数值为 3 runs 的 peak 均值；state/KV peaks 不一定同时发生，两图分别以各自 pool cap 为分母，不能相加。Slots 按当前缓存/缓冲分配路径统计。State cap 改变也会改变派生并发；此图用于展示资源构成，不将跨配置变化解释成单因素因果效应。*

### 3.3 图 3：请求拆短后，服务隔离不对称明显出现

![M1 请求形态扫描与服务差异](figures/m1_partition_sensitivity.png)

**Figure 3. 同一 token-VTC 策略、相同 token 预算下的请求形态敏感性。** Packed 始终为每 wave 一个 `(16K,512)` 请求，split 依次使用 1/4/16/32 个更短请求。固定 state cap=48、KV cap=196,608、wave interval=1.5s，每个 cell 都有匹配的独跑基线。

| Split 形态 | Packed p95 slowdown ratio | Split p95 slowdown ratio | 平均绝对 slowdown gap |
|---|---:|---:|---:|
| 1 x (16K,512)，相同形态对照 | 1.77 | 1.77 | 1.2 个百分点 |
| 4 x (4K,128) | 1.32 | 8.90 | 757.7 个百分点 |
| 16 x (1K,32) | 1.21 | 10.34 | 913.2 个百分点 |
| 32 x (512,16) | 1.23 | 10.47 | 924.6 个百分点 |

右图给出实际排队时间以补充归一化指标：f16 的短/长请求 queue p95 为 10.69/8.18s，f32 为 11.05/8.28s。短请求仍承担秒级排队，并没有随自身变短而成比例下降；相对其独跑成本，混跑代价因而更加突出。

**Observation：相同形态时两租户 slowdown 接近；token 总量不变、请求拆短后，当前策略下短请求租户承受显著更高的相对延迟膨胀。** 这使 M1 从单个例子推进到请求形态扫描中的具体服务隔离问题。

*图注：P、O 和请求数共同变化，实验识别的是 partition/shape 效应，不是仅输入长度的单因素效应。Slowdown 是各自相对独跑的比值，不是两租户绝对 JCT 之比。表中 gap 为逐 run 绝对差的均值，不是两列均值之差。现有 compact 数据未提供这些点的 CI。*

三图的[统一绘图程序](plot_m1_evidence_series.py)与[数值/来源记录](figures/m1_evidence_series.json)可复核；资源和形态扫描均仅使用现有 token-VTC confirmation 数据，不混入 profiled/instant-share 的方案对比。

### 3.4 实验与机制：为什么请求形态不能被 token 总量替代

每个 wave 中，两租户分别提供相同的 16,384 input tokens 和 512 output tokens：

- packed：1 个 `(16K input, 512 output)` 请求。
- split：16 个 `(1K input, 32 output)` 请求。

进一步扫描 split factor 1/4/16/32，并改变容量与到达间隔。每个请求使用唯一 cache salt，排除 prefix reuse。相同 token 总量是工作量计数的控制，不代表计算过程、模型输出语义或资源需求等价。

对固定模型和 topology，近似资源关系是：

\[
M_{active}(t) \approx N_{active}(t) S_{request}
    + T_{resident}(t) K_{token}.
\]

分离内存池还分别受 state-slot cap 与 KV-token cap 约束，不能只看两者字节之和。

### 3.5 已有证据汇总

[2026-08-26 报告](results/vtc_state_tax_20260826/vtc_state_tax_observation_report.md) 包含 387 artifacts、34,107 requests，零 failure/cache hit/retraction。

| 观察 | 数据 | 能支持的结论 |
|---|---|---|
| 请求形态相同的负对照 | f1 token-VTC slowdown gap 仅 1.2 个百分点 | 在该配置下，租户标签本身没有产生同量级差异 |
| 拆分后的服务差异 | f16/f32 gap 为 913.2/924.6 个百分点 | equal-token workloads 在混跑中受到明显不同的相对延迟影响 |
| 放宽 state 后的资源向量 | f16 split/packed peak state slots 为 270/24；peak KV tokens 为 94,845/135,086 | equal-token service 对应不同的 state/KV 需求形态 |
| 累计资源时间差异 | f16/f32 最大 per-token resource-time ratio 为 17.50×/33.69× | token service 不能还原实际 resource-time；该最大比值不是短请求 state 成本的倍数 |

峰值可能发生在不同时间；混跑 resource-time 还受调度和干扰影响。这些指标需要与固定 state allocation 的结构性证据结合解释。

### 3.6 完整逻辑线与第一个问题

三张图形成以下论证：

1. **当前公平成本如何计量：** 标准 token-VTC 累计 `W_u = sum(P_i + 2*O_i)`。它读取请求长度，但在完整服务相同输入/输出总量后，累计 charge 对 partition 不敏感；实际 admission 顺序仍可能随请求粒度和记账时机变化。
2. **这种计量遗漏什么：** Hybrid active memory 含 `S*N_active + K*T_resident`。短请求的单位 resident token 固定 state tax 更高，且实测长短请求集合呈现不同的 state/KV 占用构成。
3. **现有方法出现什么后果：** 在所测 Kimi-Linear workload 中，token-VTC 的混跑 slowdown 从相同形态时接近，变成拆短之后的明显不对称，短请求租户的相对延迟膨胀约为 8.90--10.47 倍。
4. **M1 的问题结论：** 当前基于 token 累计服务作公平决策的方法，缺少对请求长短所对应 hybrid 固定 state 成本的显式区分；在相同 token 预算下，资源负担和服务隔离表现显著依赖请求形态。

**问题：当前 token-VTC 下，相同 token 总量的不同请求形态在混跑中承受明显不对称的延迟膨胀；仅按 token 计量不能充分刻画它们面对的资源约束与服务隔离表现。**

**Insight：request partition 是服务隔离相关变量，而不只是 token 总量的表示方式。** 固定 state 使请求数具有独立资源含义；因此应同时观察 token 服务、active-state 占用和混跑 slowdown，才能解释不同租户承担的服务代价。

写作上应使用“没有充分考虑请求长短对应的 hybrid 资源代价”，而不是“完全没有考虑请求长短”。后者与输入/输出长度已经进入 VTC counter 的实现不符。M1 的证据范围是当前标准 token-VTC 实现及这些 workload；是否固定 state 单独导致差异由 M2 继续归因，其他已有成本变体能否修复由 M3 检查。

“短请求更贵”的准确含义是固定 state tax 的单位 token 摊销更高。它不表示单个短请求的总显存或总计算一定更高，也不表示短请求的 state-time 一定更高：短请求可能释放得更快。

**可下结论：当前 token 计量存在资源表征盲区，且当前系统出现显著的请求形态相关服务差异。** 两个观察不能直接合成“固定 state tax 单独导致了全部 fairness gap”；需要干预归因。M1 不要求先证明 vector debt 有效，也不推导出应惩罚短请求。

### 3.7 尚需加强

优先补充固定 token 总量、不同请求数的直接 footprint/可行并发曲线，并显式固定 max-running、分别改变 state/KV cap。零 prefix hit 不等于关闭 radix；当前 runner 未指定关闭，因此需要同模型 radix on/off 对照，排除可避免的缓存 slot 开销。[SGLang K3 cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) 已提示无复用流量下的缓存策略选择，但不能直接用 K3 数值替代 Kimi-Linear 测量。

KV-only 对照有助于分离通用请求形态效应；equal tokens 并未控制 MLA prefill 的非线性计算成本。之后再做多模型/topology 校准。外部 54 MB/request、27 KB/token 数字不作为本机测量引用，fixed-state premise 本身也不作为新发现。

## 4. M2：Hybrid 如何影响租户之间的服务机会与隔离

> 当前独立文档：[M2：租户服务机会与隔离](m2_tenant_service_opportunity_isolation.md)。后文实验时间线保留历史事实，其中的旧“下一步”不再作为当前执行顺序；以本节4.1--4.2及独立文档为准。

### 4.1 从M1递进到哪个问题

以M1的单位token资源成本差异为前提，M2应追问：**不同成本的token服务，经现有公平计量、排序和执行后，如何转换成实际资源占用与租户服务分配？这种转换是否留下可避免的隔离损失？** 固定目标租户的请求与权重、仅改变等token邻居的形态，是识别这条链的实验手段，不是从零重新验证“请求长度是否重要”，也不把资源成本直接当作公平价格。

需要对齐“公平计量/候选排序 -> 实际准入 -> batch执行进度 -> 资源释放与输出交付”。分别辨认计量和记账时机、state/KV或普通cap限制、激活后的执行干扰；不能把所有延迟统一解释为固定state。安全可行性、token服务公平和服务隔离不是同一保证。

待验证的insight是：新增某租户的holder不只改变它本轮获得的计算，还可能改变其他租户后续的服务机会。这里不预设应优先短请求、应按state-time收费，或必须实现lifetime predictor。

### 4.2 新验证顺序与主验收

| 阶段 | 最小问题 | 应交付的证据 |
|---|---|---|
| F1 邻居形态 | 固定A，A独跑/A+B-packed/A+B-split；B总输入输出和权重相同，交换标签及到达顺序 | A的准入、实际服务进度和隔离是否随邻居形态变化，另报告B的代价；等token不当作等计算 |
| F2 路径定位 | 同一轨迹对齐counter、候选检查、资源归属、执行和释放；有空间阻塞才做相应资源干预 | 哪一步将邻居形态变化传递成目标服务损失；区分state/KV、普通cap与通用执行干扰 |
| F3 可避免性 | 固定公平计量，校准简单原生控制，在共同服务约束下做局部准入/组批干预 | 是否改善受损租户且不靠牺牲另一方或改变公平定义；记录效率代价，再在held-out检查 |

F1--F3是待冻结的问题序列，不是新增PASS；不修改旧E1--E3编号。旧双租户审计已完成，确认只有counter汇总、全程资源积分和客户端输出流，缺逐事件counter、租户waiting和拒绝/释放。下一步按补测规格增加只读观测并做功能预检，再冻结正式矩阵。双方过程服务、绝对延迟、worst-tenant slowdown及效率必须同时报告。新方案和完整SOTA比较用于后续设计必要性，不能用单租户加速代替当前问题发现。

“如何更快完成一批请求”放到公平目标明确后的效率阶段。当前暂不继续扩大以单租户Decode32 mean收益为主的矩阵；Mixed、统一池和容量审计作为正确基线与归因工具保留。

### 4.3 历史实验进展，保留原判定

> 2026-09-10准入因素、三张图与完整分析：[M2历史机制证据](m2_hybrid_admission_factors.md)。132组正式运行、2119请求，全部通过完整性检查；28组预热排除。验证激活、约束与持续占用，不独立承担跨租户公平结论。
> 最新完成：[具体 slot 缺陷与 batching 收益对照](m2_slot_reuse_batching_opportunity.md)，100 组/2000 请求、三张图。固定共同 VTC，比较三类原生 batching：主对照 short mean +8.37%、插值 p95 +29.01%；mixed 收益缩小，无 chunk/宽池不触发。验证的是具体问题和局部收益，不是普遍不适用。
> 修正基线后的追加筛选：[首次激活节奏实验与两张图](m2_activation_pacing_results_20260910.md)，48 组/3840 请求。固定 per-pass 新请求限额未显示额外收益；这个负结果不推翻 slot 缺陷，但不支持直接进入 lifecycle/pacing 算法设计。一次监测超时后按整会话规则补跑，来源与排除记录完整保留。
> 2026-09-15 自然容量验证：[执行进度与 state 释放报告](m2_natural_progress_results_20260915.md)，20 组/4316 请求。607 个短请求在自动解析的 485-slot 容量下遇到 state/request 瓶颈，KV 峰值仅约 20%--21%。Mixed 消除 decode-ready 未参与代理间隙，但压力点 mean JCT 16.467 -> 22.206s、state-time 增加 35.83%。这说明不能把“持续参与”直接当成“减少驻留”；尚未证明 state 因果性放大了 batching 损失，也未发现强 baseline 之外的新算法收益。
> 2026-09-16 固定工作集干预：[state-cap与batching交互报告](m2_state_mediation_results_20260916.md)，26组/15782请求、两张图。同一宽池cap485→607的mean JCT交互差值为+0.612/+0.781s，但Separate/Mixed自身mean JCT分别增加6.99%/2.07%，排队、p95和makespan改善而state-time增加。支持目标相关的准入–执行权衡，不是“放宽state即可加速”。原生校准存在波动，A/B不能作纯state归因；hybrid独有性和新算法必要性仍未证实，未进入lifetime predictor。
> 2026-09-16 池管理实测完成：[统一池解决了什么，仍没有解决什么](m2_pool_comparison_results_20260916.md)，空闲GPU0/1，24组/14568正式请求，等量完整预热排除。完整idle驻留显存对齐后，原静态/只开统一池均峰值485，重划静态/统一池配合提高有效请求上限均达到607，正确性检查通过。统一池同池cap485→607时，Separate/Mixed的queue分别降低28.14%/11.29%，p95降低22.50%/10.54%，mean JCT却增加7.09%/1.33%，state-time增加25.51%/7.82%。每格仅两次独立会话，运行峰值并非严格等显存。原容量瓶颈能被已有池管理与配置修复，不能再作为“已有方法无法解决”的证据；剩余的是目标相关权衡，尚未证明hybrid独有缺陷或更强baseline外收益。
> 2026-09-17 筛选及原生对照完整完成：[正确统一池之上的准入时机及原生对照](m2_unified_admission_complete_20260917.md)，同GPU4/5，32组正式11776请求，另排除等量完整预热。State-near的诊断decode32在两次独立会话配对中使mean JCT降低54.49%、p95降低27.43%，makespan增加3.05%；新请求mean TTFT增加17.90%。已有512个holder全部释放从候选到达后约25.59s提前到2.23s，state-time降低62.46%；该格没有容量拒绝，因此不能解释为state容量介导。原生Slots64已复现mean改善27.21%、p95改善26.67%，代价是makespan增加3.84%、新请求TTFT增加9.96%；Delayer32全部8组未触发延迟，不能据此否定整个机制。Decode32在State-far无收益、KV-near两次方向不一致。一次迟到不合格会话已整段排除并按冻结规则恢复；Mixed、调优baseline与held-out尚未完成。本轮所有请求属于同一个租户，不直接作为跨租户公平改善证据。
> 2026-09-17 后续原生校准完成：[state归因前的原生控制排除实验](m2_native_counterexample_results_20260917.md)，新增12组正式6912请求及等量完整预热。保持同一统一池，Native607/Cap512在同一会话交错，另测Slots96，两轮新种子rep61/62。State-near的Cap512/Slots96 mean改善54.38%/54.40%，p95改善约27.4%、makespan增加约3.1%，新请求TTFT增加约19.1%；State-far mean改善约53.3%，但makespan增加约6.4%、新请求TTFT增加约62.4%，未通过原约束。两个原生控制均让全部512背景先释放再激活候选，无prefill-NO_TOKEN。说明旧大幅收益量级可由通用控制实现，不是新state抽象必需的证据；新旧rep不直接作性能等效检验。下一步是冻结这些校准基线后的Mixed/held-out缺口检查，而非继续把普通N上限当作state消融。

> 2026-09-17 继续验证进度：[Mixed/held-out准备与旧轨迹容量审计](m2_mixed_holdout_progress_20260917.md)。新32组矩阵已冻结、35项CPU回归通过，但GPU被外部作业占用，启动失败且一次恢复被idle检查拒绝，新增正式性能样本为0。只读复核旧12组原始轨迹全部无prefill_no_token；旧576请求的逻辑state+KV主体上界约19.408GB/rank，仍低于19.481GB共享raw预算，进一步说明旧54%收益不能独立证明容量介导。追加24个探针可使联合上界超预算、总请求600仍低于cap607，但尚未实测，不能当作新正结果。

> 2026-09-17 用户要求继续后的Mixed实测：[近释放窗口结果与两张图](m2_mixed_near_results_20260917.md)。GPU4/5完成4组背景校准；far场景不满足，保留原失败计划，在正式对比前单独冻结near补充协议。16组正式8192请求及等量完整预热完成，38项CPU回归通过。Decode32相对Native607的train512/hold384 mean描述性改善23.70%/18.87%，新TTFT改善12.67%/10.13%，p95和makespan也下降；但部分到达存活数不同，没有完整两重复contrast通过匹配gate。存在单个匹配配对的局部正例。Cap512/Slots96全部未触发，不能视为调优后的强Mixed基线。背景持续参与Mixed decode，仍有准入时机收益线索；16组均无prefill_no_token或request_limit，主体峰值13.605/11.740GB低于19.481GB预算，不支持state/KV容量介导。日志另有可恢复allocation retry，不能声称无workspace压力。服务已退出并释放GPU。

### 4.4 历史机制实验针对什么

针对本地公平排序与 SGLang 请求准入/batch formation 的组合：先排候选，再用 request pool、并发、state/KV 容量和 token/chunk budget 决定本轮加入什么。SGLang 已有容量保护；M2 检查的是这些机制叠加后的选择及服务后果，不否定 continuous batching。

### 4.5 历史E1--E3与slot修复

| 实验 | 从 M1 推进的观察 | 问题与 insight |
|---|---|---|
| E1：等 16K input，1/4/16/32 个请求 | 实测 peak state 为 1/4/16/32；单请求继续 chunk 时 KV 增长而 state 不变 | 首次激活与继续执行不是相同资源动作，不能只用本轮 token 数描述 |
| E2：同一短/长候选集，独立改变 state/KV/cap 与提交顺序 | state/cap 触发 request-limit，KV 组触发 NO_TOKEN；long-first 时 instant 相比 VTC 的短 p95 降低约 28%--33%，short-first 差异小于 1% | 排序结果依赖资源条件与到达顺序；策略公平语义不同，不能称为同保证收益；还需排除继续 chunk 的 slot 计数问题 |
| E3：同一 probe，到达时背景 holder 已释放/仍占用 | state-low probe queue 从 0.004s 升至 10.966s，cap-control 重现 10.993s；wide/KV 对照约 0.034s | 持续占用约束未来的准入机会；这是 slot/concurrency 机制，不是 KDA 独有阻塞 |
| 后续：固定 VTC，最小 slot 修正 x 三类 batching x 两种到达顺序 | separate4K/short-first 提前激活 7 -> 8，short mean 6.874 -> 6.299s；mixed4K 插值 p95 +2.05%，mixed1K 无显著改善；无 chunk/宽池不触发 | batch 参与数不等于新增 holder 数；可行准入机会的价值取决于后续组批与释放，而非拒绝次数本身 |

上述配置均关闭 radix 和 CUDA graphs，仅使用 GPU 6/7。结构观察每点 3 次，其余配对每点 5 次；所有预热排除。前述 E1--E3 未修改准入判断源码；后续单独修正 slot 计数，未改变 VTC 权重和其他资源预算，也没有实现 lifetime predictor。

实际源码复核与后续正式消融确认：继续 chunk 复用已有 request slot，但候选数量门限仍把它与新请求一起计数，可提前拒绝下一个可行候选。主对照修正后减少一个落在末轮的短请求，得到 8.37% 均值收益；29.01% 是 16 个短请求的线性插值 p95 收益，最大延迟没有显著变化。该问题可承担 motivation，但不是 KDA 独有，也不把单点修复等同于完整论文贡献。

### 4.6 旧证据重新归位

[旧双租户cap/batching实验](m2_admission_batch_formation.md)共108组正式运行、12896请求：f16中相同VTC下cap96→48改善split p95 26.4%，以packed p95 / makespan退化0.61% / 2.33%为代价，通过预设5%代价上限；f32 held-out为-0.4%，CI跨0，未通过收益判据。它是最接近当前M2的证据：同一公平排序下，执行控制会改变租户间服务结果；但缺少运行过程中共同积压窗口的公平核验，不能仅凭最终token相同宣布同保证收益，也不证明hybrid特有缺陷或静态调参稳定有效。

[8 月 28 日容量/并发对照](results/admission_batch_20260828/admission_batch_motivation_report.md)仍支持：state cap 的部分影响经派生并发上限传递；排队减少可以伴随执行变长和邻居延迟恶化。旧 O2 单因素解释失败保留，不改写成“没有问题”。

**当前需要区分：资源事实、实现正确性、跨租户服务影响、同一公平约束下的可避免损失、后续新设计必要性。** 前两项和局部服务结果已有证据；新M2重点是第三、四项。强baseline决定后续贡献边界，不是承认每个局部观察的统一前置条件；但观察到等待或单租户mean下降本身都不足以证明公平问题已经解决。

新增池管理对照作为基线排错保留：fixed state tax与原部署容量瓶颈仍成立，但该瓶颈不是统一池/重划静态池无法处理的问题。同一统一池中，更多准入仍可能以平均完成时间和state-time为代价换取排队、tail和整批完成改善。这解释容量与执行的区别，不直接回答租户如何分享服务；后续双租户实验应继承正确的池与请求上限，而非从旧485配置重造问题。

最新原生参数校准已进一步缩小设计空间：同一池上，纯Cap512和Slots96均能避免在背景释放前插入128个新prefill批次，State-near得到约54.4% mean改善。这是通用控制的实际收益，不需要新state记账或lifetime预测。Far也能降低mean，但以超出原门槛的makespan和更高TTFT为代价；不能再把旧Decode32的far负结果概括为“far没有mean收益”。固定state持有随完成提前而下降仍是真实后果，但本轮没有独立改变state成本，不能估计它对性能的额外因果作用。

源码与逐事件核验还表明：本配置每active request对应一份state，因此逻辑state-slot cap与相同N上限等价；启动state参数又同时影响请求上限和统一池raw大小。独立归因不能只给普通并发限制换名字。Mixed/held-out补充筛选有数值线索和单配对证据，但所有请求属于同一租户、完整配对未通过且没有cache容量拒绝；不承担跨租户公平或固定state容量归因。下一步回到4.2的租户级验证，将“谁持有资源、谁因此失去准入/推进机会、谁获益或受损”连接起来，而不是继续以整体mean下降替代M2。

## 5. M3：静态成本校准未消除混跑服务差异

这里检查一个具体解释：前面的服务差异是否只是 token cost 没加 request tax，或隔离成本权重没有校准好？核心证据是原 M3a；原 M3b pulse 保留为辅助观察，不用它直接推出历史账本必需。原 gate 的 PASS 不改变。

### 5.1 实验 A：静态成本修补的边界

比较 token VTC、固定 request tax、隔离 JCT 拟合的 profiled VTC，以及 instant-share。Profile 采集与后续 policy 测试分开，权重拟合后冻结；但下面的拟合误差仍在用于拟合的同一组 9 个 shape 上计算。

8 月 26 日，profile 的 **样本内** median/max residual 为 0.2%/1.6%，不是 held-out 预测误差，也不等于 GPU service cost 测量准确度。该变体在混跑中仍明显弱于 instant-share。8 月 28 日的数据进一步显示：

| 配置 | Profiled VTC 相对 token VTC | Instant-share 相对最强 token/profiled scalar |
|---|---|---:|
| State 48 / running 9 | 改善有限 | gap 改善 61.3% |
| State 480 / running 9 | 改善有限 | gap 改善 82.9% |
| State 480 / running 96 | 改善有限 | gap 改善 -1.3% |

Profiled VTC 在三个 cells 平均改善 4.9%；instant-share 在两个低并发 cells 平均改善 72.1%，高并发 cell 则没有优势。

**问题 A：已测静态 request/input/output 成本校准，并没有消除混跑下观察到的 slowdown 差异。** 不能进一步写成“所有静态成本必然无效”或“运行时资源信息已经被证明不可替代”。

这不证明任意 scalar cost 都不可能有效，也不证明 vector 表达不可约。VTC-predict 和 projected-share 均未被覆盖。高并发若让所有候选都能准入，队列重排可能根本不改变 running set；必须先测控制空间，不能把动作失效解释成信息表达不足。

### 5.2 实验 B：瞬时份额与历史服务的权衡

在 state=480、max-running=4 下，一个 incumbent 提前提交 8 个长请求，pulser 分 8 波提交短请求；两租户总输入、总输出分别相等。

| 指标 | Token VTC | Instant-share |
|---|---:|---:|
| 最大 normalized delivered-service gap | 34.9% | 47.6% |
| 时间平均 normalized service gap | 25.8% | 27.1% |
| Slowdown gap | 1242.3 个百分点 | 533.6 个百分点 |
| Makespan | 31.40s | 32.28s |

Instant-share 缩小 slowdown gap，却使最大 delivered-service gap 增大约 36%，makespan 增加 2.8%。该 gate 由最大 gap 通过，平均 gap 仅增加约 5%，不能写成所有时间维度均显著恶化。

本 probe 将输入成本记在首 token 到达时，输出按 `2 * delivered output tokens` 计量；窗口从最后一波到达开始，到第一个租户完成全部请求结束。它是客户端 delivered-service 指标，不等同于 VTC scheduler counter 或其理论中的 backlogged-window service difference。

**问题 B：按当前资源占用排序并未显式约束历史服务，改善 slowdown balance 时可能扩大另一种服务差距。**

本轮审计发现窗口起点已有明显服务差异，且两策略窗口长度分别约 11.2s 和 19.3s。统一窗口并扣除起点服务后的事后分析，最大增量 gap 仍为 instant-share 约 43.33%、VTC 约 6.58%，但未证明 waiting-backlog 条件。差异方向值得继续检查，不能据此认定 history reset 因果成立或违反 VTC bound。详见[审查 R5](topic_review_20260908.md#r5原-m3b-是时间窗口敏感的-probe不是历史债务定理反例)。

### 5.3 核心 insight 与证据边界

**Insight：当前服务差异没有被一个固定 request tax 或这组隔离拟合权重解释并消除；问题诊断不能停在固定成本校准。** 结合 M2，请求执行成本随运行组合变化是值得检验的机制解释，但 M3 的 policy 排名本身不能证明这一归因。

这些观察提供诊断方向，但没有证明必须增加新 ledger。需要持续 waiting-backlogged、重复 idle-return、交换租户角色，以及同一 runtime rule 加/去预测承诺与历史约束的消融。若预测版 VTC 已修复，就不能主张新的 history 机制。

旧固定 lifecycle ledger 在 held-out 数据上相对 instant-share 仅改善 4.0%，95% CI 为 [-24.7%, 27.3%]。这个失败仍有效，不能用旧 M3a/M3b 的 PASS 重新宣称同一个 ledger 已有优势。

### 5.4 尚需验证：校准不足具体来自哪里

保持目标请求 shape 相同，分别独跑以及与不同 shape/数量的邻居混跑，测校准残差、实际执行时间和调度次序。将隔离 profile 在未参与拟合的 shape 上测试，先排除拟合覆盖不足；对固定输出 workload 比较按实际 decode 记账与真实输出预记账，分离成本数值和收费时机。统计多租户同时 waiting 的时间与排序实际改变 admission 集合的比例，排除队列排序没有动作空间。

这些是区分竞争解释的诊断对照，不是候选新算法。若某个简单对照消除了差异，应据实缩小问题范围：例如归因为未预记输出或 frozen sorting，而不是坚持“资源抽象不够”。

## 6. 当前阶段的补实验清单

目标是补全问题证据，不启动新机制设计。当前优先恢复M2的租户级过程验证；效率指标是共同代价记录，不是替代公平与隔离的主目标。

| 对象 | 最小补充实验 | 需要得到的结论 | 不由此推导的结论 |
|---|---|---|---|
| M1：计量盲区 | 同时刻 state/KV footprint、固定 max-running 的 cap 干预、radix on/off、denial reason | 请求数的独立资源作用，以及它何时实际限制服务 | token VTC 理论失效；必须使用资源向量 |
| M2：租户服务机会与隔离 | 60组旧数据审计完成；先补逐事件counter、租户waiting、候选检查和时钟，再固定A、改变等token邻居B的partition，对齐计量/占用/服务链 | 不同成本的token如何转换成实际服务，作用位于计量/准入/执行哪一步，是否存在同约束下可避免的隔离损失 | 单租户mean等于公平收益；最终token或最大counter gap相同等于过程公平；资源占用不等等于不公平；所有影响均由state单独造成 |
| M3：校准边界 | held-out profile、固定 shape 的混跑残差、输出预记账、waiting/排序机会统计 | 是固定成本遗漏、拟合不足、记账时机，还是执行环境变化导致剩余差异 | 所有 scalar 都无效；历史账本必需 |
| Pulse 辅助观察 | 合法且共同的 backlog 窗口、服务增量、多次 idle-return 和角色互换 | 现有 instant-share 是否稳定积累服务差距及其条件 | 单次累计 gap 就是饥饿或 VTC bound 反例 |

以上实验需要随机化或分块重复、run 级 CI，报告绝对延迟和效率；不把请求数量当作独立系统重复。先在当前模型上把问题与原因写清，再扩到多模型/真实 trace 检查外部有效性。

候选系统设计与联合控制 oracle 暂不作为 motivation 的验收条件。早先的设计讨论留在[审查报告第 6 节](topic_review_20260908.md#6-候选设计的合理范围)，只有问题归因收敛后再使用。

## 7. 对照方法分两层，不混淆验证目的

第一层是当前的问题诊断：确认方法实现、固定条件、排除拟合/记账/队列机制等替代解释。第二层是后续的已有解决能力调查和设计评估：比较完整方法及组合是否解决已确认的问题。后者重要，但无需全部完成才承认前者的局部观察。

新M2先固定本地token-VTC计量、权重与正确容量基线，定位跨租户影响；将成本变体和不同公平语义的策略单列为解释性对照，不把混合目标的排名当作同公平约束收益。下表是后续诊断/评估菜单，不要求F1一次性跑完全部方法。

| Baseline | 排除的简单解释 |
|---|---|
| FCFS、tenant round-robin | 是否只是基础租户隔离带来的收益 |
| Token VTC | 相对 token-service fairness 的变化 |
| Request-tax / profiled VTC | 是否固定请求成本校准已足够 |
| VTC-predict / true-output oracle | 原论文的未来输出预记账是否已修复 admission 承诺问题 |
| Frozen instant-share / projected-share | 当前占用或逐次更新预计占用是否已足够 |
| Deficit round-robin / quantum 约束内重排 | 是否普通 deficit 公平框架已提供所需自由度 |
| 上述策略 + 调优固定 max-running、prefill budget | 是否简单限制并发即可达到同一性能前沿 |
| VTC + Sarathi/FairBatching 风格兼容 batch 控制 | 是否已有公平与动态 batching 的组合已足够；移植规则不冒称完整系统复现 |
| 经过本机重校准的 LPRS/APC | 时延预测与 active-prefill 控制是否已解决问题；其原始特征已含 request count/context，不当作 token-only baseline |
| 后续候选方案的机制消融 | 仅在开始设计后，用于确认设计部分的必要性 |

无 prefix reuse 可以暂不比较 locality 收益，但不能排除 [DLPM](https://arxiv.org/html/2501.14312v1) 的 deficit/quantum 机制。相关工作还必须正面对照 [Equinox](https://arxiv.org/html/2508.16646v1) 的 equal-token 请求形态观察及双 counter/batching，以及 [Justitia](https://arxiv.org/html/2510.17015v2) 的 memory-centric scheduling。公平对象有差别，不等于机制无重叠；不能继续把“token-only 不足，所以加资源与历史”当成核心新意。详细读取范围及比较方式见[审查第 4 节](topic_review_20260908.md#4-相关工作核查)。

当前验收不是“新 policy 超过强组合”，而是每条 motivation 都能写出一句限定明确的问题结论，并由可复核实验支撑。相关工作用于检查问题是否已被认识、现有方法承诺什么、哪些解释已有现成诊断对照；不能把未调研的工作默认成失败。

9 月补查：[LPRS/APC](https://arxiv.org/html/2606.09061v1) 已联合考虑 batch latency 预测和 prefill 并发控制，且有[官方实现](https://github.com/Charmstok/Chunked_Prefill_Serving)。本轮仅测原生机制，未复现该方法，不能将“时延预算 + 并发控制”本身作为新增贡献。

## 8. 当前能讲到哪里

当前故事线应是：**M1提出相同token服务背后的hybrid资源差异；M2解释这些差异如何经公平排序、准入与执行影响其他租户的服务机会和隔离；M3检查现有校准能否解释或修复，之后才研究公平约束下的效率优化。** 其中M2的跨租户过程链路仍待补齐，不能把已完成的单租户加速实验写成该链路已证实。整体是问题发现顺序，不是已经证明某种方案必需的设计链。

8 月 28 日补充实验共 57 artifacts、6,120 requests，全部成功，resource/batch traces 均为 57/57，零 cache hit/retraction/drop；三个子 gate 在预注册口径下通过。但它主要是单模型、合成双租户、每配置三次重复的 motivation 证据，最新汇总未提供各效应的置信区间。

9 月 M2 新增 108 组正式运行、12896 请求，固定池/graph 并以新配对重复提供 CI。它将证据推进到“局部有限代价收益成立”，同时保留 f32 静态 cap 迁移失败这一负结果；两者都不构成先进动态方法普遍失效或新算法必需的证明。

9 月 10 日新主线另完成 132 组、2119 请求，验证了激活、独立约束和持续占用的准入后果，并定位一处 request-slot 计数局限。它缩小了问题范围，也明确保留 cap-only 重现这一归因边界；不把实验中所有不利结果合并为“hybrid 下现有方法无效”。

本轮额外只读检查 57 artifacts 的 policy、weights、KV cap，未发现不一致；数据记录较完整，但问题归因尚未完整。下一步优先完成第 6 节诊断，而不是决定 ledger、vector 或联合控制。发现问题、确认新颖性、设计解法、验证解法分阶段推进；当前不以“是否已经够一篇新 scheduler 论文”替代 motivation 审查。

## 9. 证据索引

- [当前M2主线：租户服务机会与隔离](m2_tenant_service_opportunity_isolation.md)：问题定义、F1--F3验证顺序、历史证据重新归位；新实验尚未冻结或执行。
- [当前M2首步：旧双租户服务审计与图](m2_tenant_service_audit_20260917.md)、[结构化审计](results/m2_tenant_service_20260917/audit01/audit.json)、[F1补测规格](m2_tenant_service_f1_measurement_spec_20260917.md)：60组旧数据、8000请求，只读复核；新增GPU样本0，确认服务轨迹可分析及公平过程证据缺失。
- [历史M2机制：Hybrid准入因素](m2_hybrid_admission_factors.md)、[历史冻结协议](m2_hybrid_factors_protocol_20260910.md)：激活、瓶颈和后续准入三组实验，作为当前归因工具。
- [9月16日统一池/静态池真实对照](m2_pool_comparison_results_20260916.md)、[完整正式汇总](results/m2_pool_comparison_20260916/measure_analysis01/analysis.json)：等idle驻留预算，24组/14568请求；已有配置可解除容量瓶颈，同池cap干预仍有mean与tail的目标权衡。
- [9月17日统一池准入筛选与原生对照](m2_unified_admission_complete_20260917.md)、[32组完整汇总](results/m2_unified_admission_20260916/controls_analysis01/analysis.json)、[恢复记录](m2_unified_admission_recovery_20260916.md)：11776正式请求及等量完整预热；State-near两次配对显示mean/TTFT权衡，原生Slots64覆盖部分收益。旧[12组部分结果](m2_unified_admission_results_20260917.md)保留为历史快照。
- [9月17日原生参数校准与state可识别性](m2_native_counterexample_results_20260917.md)、[冻结协议](m2_native_counterexample_protocol_20260917.md)、[12组完整汇总](results/m2_native_counterexample_20260917/analysis01/analysis.json)：新增6912正式请求及等量预热；Cap512/Slots96获得旧诊断收益量级，far代价越过makespan门槛，收窄新机制必要性而不否定局部问题。
- [9月17日Mixed near补充筛选与两张图](m2_mixed_near_results_20260917.md)、[补充协议](m2_mixed_near_protocol_20260917.md)、[16组汇总](results/m2_mixed_holdout_20260917/near_analysis01/analysis.json)、[容量审计](results/m2_mixed_holdout_20260917/near_capacity01/audit.json)：新增8192正式请求及等量预热；有描述性收益和单配对证据，无完整两重复匹配PASS，无cache容量拒绝；原far校准失败保留。
- [9 月 M2 独立分析与两张图](m2_admission_batch_formation.md)、[协议](m2_batch_frontier_protocol_20260909.md)、[完整统计与 manifest](results/m2_batch_frontier_20260909/confirmation_analysis/analysis.json)：原生扫描、f16 正结果、f32 held-out 负结果。
- [机制分析与最小反例](mechanism_counterexamples_20260908.md)、[精确结果](results/mechanism_analysis_20260908.json)及[验证程序](verify_mechanism_counterexamples.py)：条件化理论结论，不是新 GPU 实测。
- [2026-09-08 完整审查](topic_review_20260908.md)：主要风险、相关工作及 P0--P3 实验优先级。
- [本轮逐 run 审计](results/admission_batch_20260828/review_audit_20260908.json)及[只读脚本](audit_admission_batch_review.py)：makespan、窗口敏感性及 resolved 配置检查。
- [8 月 26 日 equal-token/VTC 实验](results/vtc_state_tax_20260826/vtc_state_tax_observation_report.md)：M1 与静态修补证据。
- [8 月 28 日补充报告](results/admission_batch_20260828/admission_batch_motivation_report.md)及[结构化分析](results/admission_batch_20260828/analysis.json)：M2、M3a、M3b。
- [8 月 28 日预注册协议](admission_batch_motivation_protocol.md)：干预设置、指标和原始 gates。
- [Lifecycle held-out 报告](results/lifecycle_fairness_generalization_20260825/lifecycle_fairness_generalization_report.md)：固定 lifecycle ledger 的负结果。
- [截至 8 月 26 日的完整历史](hybrid_state_aware_admission_history_20260826.md)：旧 Topic A/B、旧 M1--M5、oracle 和相关工作探索；编号与本文不同。

8 月 28 日原始 artifacts 保存在远端宿主机 `/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/admission_batch_20260828`，容器内映射到 `/workspace/sglang` 下同一路径。本地同步报告与结构化分析，供主文档引用。
