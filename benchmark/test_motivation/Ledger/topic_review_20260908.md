# Topic 审查：Hybrid state 约束下的多租户服务隔离

审查日期：2026-09-08。范围：本地代码、8 月 26 日分析产物、远端 8 月 28 日全部 57 个原始 artifacts，以及下列原始论文和官方资料。本轮没有运行新的 GPU 实验，也没有改变 serving 配置。

> 阶段澄清：本文同时审查问题证据与后续方案新颖性，二者不是同一验收条件。当前工作聚焦前者，按[主文档](hybrid_state_aware_admission_motivation.md)的“实验 → 观察 → 具体问题 → insight → 结论边界”推进。下文“联合控制是否必要”的 M3 属于后续机制问题，不再作为当前第三个 motivation；当前 M3 检查已测静态成本校准未消除服务差异的事实与原因。原实验与审计数值不变。

## 1. 审查结论

问题值得继续研究，但上一版“当前三个 motivation 有实验支持”的表述过于整齐。实际证据强度不同：

| 命题 | 审查判断 |
|---|---|
| 固定 recurrent state 使请求数成为独立容量变量 | 结构性前提成立，已有工作也已认识到；可作为起点，不是独立新意 |
| 本配置下改变 max-running 会改变排队、执行延迟和邻居表现 | 数据支持，且有中间对照；尚未证明 hybrid 特有的干扰机制 |
| 已测静态 cost 和 frozen instant-share 未覆盖所有运行区间 | 数据支持，但 baseline 家族和控制维度不完整 |
| 必须引入历史账本才能修复 pulse 行为 | 未完成因果验证；客户端累计 gap 不能代替 backlogged-window 服务指标 |
| 联合 state-aware arbitration 与 admission 优于已有方法的直接组合 | 关键未证命题；应成为新 M3，而不能从上述现象自动推出 |
| 已足够写成一篇新 scheduler 系统论文 | 尚不足；缺主要 baseline、受约束收益上界与泛化证据 |

新主线应是 **Hybrid-State-Constrained Service Isolation**：在既定服务权益下，研究固定 state 和增长型 KV 如何改变可行并发及 batch 选择，并验证是否存在简单公平策略与已知 batching 技术无法消除的效率/隔离损失。

## 2. 按重要性排列的问题

### R1：最接近的相关工作没有覆盖完整

原故事主要围绕 VTC、DLPM、DRF 和生命周期展开，遗漏 Equinox 与 Justitia；对 FairBatching 的排除也过早。详细对照见第 4 节。

尤其是 VTC 自身包含自定义 service cost、输出长度预测、memory cap 与 service bound 的讨论。当前 profiled VTC 在准入时只收 request/input 部分，output 项在实际 decode 时累加；它不等于 VTC-predict 在准入时预记未来输出并在完成后校正。代码见 [schedule_policy.py](../../../python/sglang/srt/managers/schedule_policy.py) 的 `_vtc_admission_cost`、`_vtc_output_cost`。

因此，不能把“输入后才逐步收取 output cost，造成 admission 过量”的问题当成新发现。必须增加 VTC-predict 和真实 output length oracle 对照；当前合成 workload 的目标输出固定，尤其适合这个诊断。

### R2：公平目标尚未构成一致的优化问题

旧分析以两个租户的 slowdown 差 `G` 为核心。若把较快租户拖慢，G 也可能下降；因此 G 单独不是足够的性能目标。建议主要报告 worst-tenant slowdown/SLO attainment，G 作为分配均衡的辅助量，并同时给出全部租户绝对延迟和吞吐。

equal input/output tokens 不等于 equal computation，也不等于相同业务任务。MLA prefill 的 attention work 随序列长度非线性变化；把一个长上下文切成独立请求会改变这部分工作。当前实验适合比较不同请求形态，不能称为语义等价 splitting 攻击或要求 partition-invariant latency。

若最终研究对象是并行任务完成，需要每 wave/application 最后一个请求完成时间；每请求 p95 无法代替 application JCT。若研究对象是独立 API 请求，则保持 request-level SLO，说明租户为何享有给定 token-service 权重，不要混入任务级公平承诺。

### R3：M2 的效率收益不能从叙述中消失

重新读取原始数据得到 token-VTC 的平均 makespan：

| 配置 | Makespan |
|---|---:|
| state 48 / running 9 | 24.764s |
| state 480 / running 9 | 24.732s |
| state 480 / running 96 | 19.714s |

放宽并发使 packed p95 恶化 18.9%，同时 makespan 降低 **20.29%**。反向限流则使 makespan 相对高并发增加约 25.46%。所以结果是实质性的效率/隔离权衡，不能讲成“大 batch 是应消除的错误”。

这些是有限 wave replay 的 makespan，不是持续到达下的稳定吞吐容量。下一步应测整条 frontier；5% throughput budget 是待冻结的约束选择，不能拿不同默认并发的 baseline 分别作为分母来宣称收益。

### R4：M1 尚未把 hybrid 的额外作用分离出来

当前 state cap 同时影响派生 max-running；state-heavy admission 不代表物理 state 池已耗尽。还缺同模型不同缓存策略、明确 max-running 下的容量作用，以及 KV-only control。

特别需要区分 `zero prefix hits` 与 `radix disabled`：唯一 salt 排除了复用收益，却不一定去掉 radix/COW/checkpoint 的 slot 开销。当前 runner 没有指定 `--disable-radix-cache`。官方 K3 cookbook 已提示无复用流量可以关闭 radix；该建议不能直接当作 Kimi-Linear 本机数据，但构成必须验证的配置对照。

若问题主要来自无复用 workload 仍付出的可避免缓存开销，优先改配置；若关闭后固定 active state 仍使调度决策明显变化，才有更强的模型结构性证据。

### R5：原 M3b 是时间窗口敏感的 probe，不是历史债务定理反例

原指标窗口开始于最后一波的 scheduled arrival，但累计服务从 run 开始计数，未减去窗口起点的服务。两策略还分别在自己的第一个租户结束时停止统计。

本轮逐 run 复核：

- 窗口开始于 11.265s；instant-share 此时 pulser 已领先约 4.28%，VTC 则是 incumbent 领先约 28.37%。
- instant-share 原窗口约 11.18--11.27s，VTC 原窗口约 19.22--19.40s；平均 gap 的分母不同。
- 统一为所有 runs 共有的 `[11.265s,22.448s]` 窗口并扣除起点差后，最大增量差仍约为 instant 43.33%、VTC 6.58%。这是事后敏感性分析，保留差异方向，但不提供新的预注册 PASS 或理论验证。
- “已到达且未完成”不能证明“等待队列中持续有请求”。VTC 的 backlog 条件是后者，且 input charge 时点是 admission，不是 first-token delivery。

因此，pulse 可以展示服务目标之间的张力；history reset、饥饿或长期 service bound 仍需 scheduler-side queue/admission/decode 事件，以及多周期、角色交换和 history-on/off 干预。

### R6：Profile 与 instant-share 的实现边界未被充分反映

[fit_vtc_profile.py](fit_vtc_profile.py) 用 9 个 `(P,O)` shape 的 median JCT 拟合 6 个非负系数，然后在同一组 shape 上计算误差。0.2%/1.6% 是 **in-sample residual**，不是 held-out prediction error，也不是准确测得的 GPU service cost。Profile runs 与 policy confirmation 分开，解决了政策测试数据复用问题，但不自动解决拟合泛化问题。

现有 instant-share 每轮取一次 current share，随后把同一租户的请求按该 frozen score 成组排序，并不在每次假想准入后更新其预计 share。这是合法的简单 baseline，但不能代表所有即时资源策略，更不能等同于完整 DRF 实现。需要增加逐个 candidate 更新 state/KV reservation 的 projected-share 对照。

高并发下若候选全部能立即 admission，等待队列排序对最终 running set 可能没有区分度。此时“更好的 counter 没有效果”未必是信息不足，也可能是控制动作已无作用空间。需要统计多租户同时 waiting 的时间、每轮候选数/准入数，以及改变排序实际改变 running set 的比例。

### R7：数据完整性较好，实验覆盖仍不完整

8 月 28 日原审计检查了 resolved metadata 是否存在，但没有逐字段验证 policy/weights，也没有在 `capacities_ok` 中检查 KV cap。本次只读审计补查全部 57 artifacts 的 policy、VTC weights、KV cap，未发现不一致。这是审计覆盖缺口，不能写成已发现实验配置错误。

57 artifacts、6,120 requests 不是 6,120 次独立系统重复；每 cell/policy 只有 3 runs。packed-only/mixed 每 run 仅 8 个 packed 请求，p95 极易受少数请求影响。重复次序固定，多个容量点又按顺序重启，温度/频率/共租环境和启动差异尚未被随机化设计完全控制。

缺少：run/block 级 CI、低载至饱和扫描、多租户/权重、真实 trace、第二个 hybrid 模型、明确 kernel/backend/graph 配置和版本快照。需要分析 instrumentation 开销；不能因为零失败就认定没有观测扰动。

## 3. 三个 motivation 的新组织

| 编号 | 最新命题 | 现有支持 | 尚缺的核心证据 |
|---|---|---|---|
| M1 | Fixed state changes the feasible concurrency of equal-token workloads | 固定 state 前提、allocator/accounting、split-factor 对照 | 排除 radix 开销和通用 Transformer 形态效应，直接画容量曲线 |
| M2 | Feasible concurrency and desirable concurrency differ | 9/96 并发干预、queue/execution/neighbor 差异、20.29% makespan trade-off | 受控 batch 成本分解和整条效率/隔离 frontier |
| M3 | Independent fairness accounting and batch control may leave a hybrid-specific coordination gap | 已测 static/frozen-share 的边界、pulse 的辅助观察 | VTC-predict、projected share、tuned cap、动态 batching 的组合仍有显著残余；联合控制 oracle 能修复 |

M3 是待验证 thesis，不能沿用旧 M3a/M3b PASS 宣称已成立。旧门槛和结果保持原样，新的编号表示论文组织变化，不是回写历史判定。

## 4. 相关工作核查

下面是机制级相关性审查，不是这些系统的完整复现或所有定理的正确性审计。论文是否公开不等于已经正式录用；本表按实际读取版本列出，不统一标成 SOTA。检索覆盖当前主线，但不能据此证明不存在遗漏。

| 工作与读取位置 | 已覆盖内容及对本题的影响 | 比较方式 |
|---|---|---|
| [VTC v2](https://arxiv.org/html/2401.00588v2)，§4.1--4.4、§5、Appendix C | 有自定义 cost、predict/oracle、memory-cap 与 bound/work-conservation 权衡 | 必需主 baseline；预测未来输出不是新技术贡献 |
| [DLPM v1](https://arxiv.org/html/2501.14312v1)，§3--4 | deficit quantum 允许局部排序同时维持服务界 | 无 prefix 时仍是“公平约束内优化排序”的相关机制；不能因 zero hit 全面排除，至少加 DRR/quantum 基线 |
| [Equinox v1](https://arxiv.org/html/2508.16646v1)，Fig.1、§2--4 | 已用 equal-token many-short/few-long 观察及用户/资源双 counter，并涉及 batching | 与原大故事高度重叠；需要机制对照，不能只比较标准 VTC |
| [Justitia v2](https://arxiv.org/html/2510.17015v2)，§3--4 | 面向 task-parallel agents，memory-centric cost 与 virtual-time 完成顺序 | memory-time 不是新抽象；区分 agent completion 与本题 tenant stream isolation，并比较其成本模型扩展 |
| [Sarathi-Serve](https://www.usenix.org/system/files/osdi24-agrawal.pdf)，batching 算法 | chunked prefill 与 stall-free batching 已控制 prefill/decode 干扰 | 将兼容的 batch 规则与 VTC 组合；只扫描 FCFS/max-running 不够 |
| [FairBatching v1](https://arxiv.org/html/2510.14392v1)，§3.2--3.3 | time budget、SLO slack 和动态 batch formation；模型使用 new tokens 与 total context | 虽公平对象是 prefill/decode，控制动作与本题直接重叠；必须测 VTC + 动态 batching |
| [SGLang K3 官方说明](https://www.lmsys.org/blog/2026-07-27-kimi-k3-day0-support)、[cookbook](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3) | 已公开 fixed state/per-token KV、pool/topology 和 cache-strategy 影响 | 是 M1 的已知背景和配置 baseline；外部 K3 数值不能充当本机实测 |
| [Marconi v1](https://arxiv.org/html/2411.19379v1)，§3--4 | hybrid state 的 checkpoint/cache admission 与 eviction | 核心无复用实验可不跑，但恢复 cache-lifetime 主张时必须比较 |
| [DRF](https://www.usenix.org/conference/nsdi11/dominant-resource-fairness-fair-allocation-multiple-resource-types) | 多资源 dominant-share 理论基础 | 不能将 state/KV 向量本身宣称为新公平理论 |
| [DriftSched v2](https://arxiv.org/html/2606.02982v2)，§II | tokenizer/proxy、EMA 校准和网关队列策略 | 是外围 QoS/estimation 相关工作；当前核查不支持将其作为最强 hybrid 机制 baseline |
| [TetriInfer](https://arxiv.org/abs/2401.11181)、[Andes](https://arxiv.org/abs/2404.16283)、[Niyama](https://www.microsoft.com/en-us/research/wp-content/uploads/2025/04/niyama.pdf) | 分别覆盖预测/干扰、流式 QoE、多 QoS 动态 chunking；本轮为范围筛查 | 若扩大到 P/D、用户阅读体验或多 SLA，需深化机制比较；目前不作为已排除的竞争方案 |
| [TOPAS](https://arxiv.org/abs/2608.25523) | 2026-08-26 公开，workflow prefix-state/request 联合调度；本轮读摘要 | 旧 admission/eviction/agent 方向的新近重叠；当前无复用主线暂不扩展过去 |

VTC 和 DLPM 的论述合在一起已经说明：公平服务界、短期局部排序和 batching 不是全新的组合。Equinox 更已覆盖 generic equal-token mismatch 的叙述。差异化必须落在 hybrid 固定 state 对可行集合、运行承诺和调度动作的影响，并由最强组合 baseline 留下的收益支撑。

Justitia 的 KV token-time 使用 decode iteration 累计，文中近似忽略不同 iteration 时长。可以测试把 fixed state 加进该模型后是否仍充分；但仅加入一个 state 常数/项属于直接扩展。要研究 wall-clock interference，必须证明 iteration 近似在目标场景下影响了实际决策。

## 5. 需要补充的关键 motivation 实验

不建议增加更多平行的“短请求很贵”现象图。优先补因果链上缺失的环节：

| 优先级 / 实验 | 干预及必要对照 | 输出与决策 |
|---|---|---|
| P0：公平口径和实现审计 | 标注 queued/admitted/running/completed、charge/counter lift；核对 VTC/predict 的调度序列，固定 chunk/graph 配置 | 确认比较的是声明的算法；按合法 backlog 窗口计算 service increment |
| P1：Hybrid specificity | 同模型 radix on/off、固定 max-running、state/KV cap 独立变化；KV-only control 作辅助 | state slots/bytes 同时刻快照、denial reason/time；若关缓存即可消除主要问题，收缩 topic |
| P1：时域成本与性能前沿 | state 固定，max-running 和 prefill budget 网格；burst/mix 分层 | absolute p95、worst slowdown、SLO、service lag、makespan/稳态吞吐；验证可改进的 Pareto 区域 |
| P1：强组合 baseline 与控制空间 | VTC-predict、output oracle、projected-share、DRR、调优 cap，加 Sarathi/FairBatching 风格兼容 batch 控制 | 统计 waiting 选择机会和准入集合变化；若简单组合消除剩余 gap，则不做复杂联合 scheduler |
| P2：动态必要性 | 同一 run 内改变短/长 mix 和 burst；冻结参数；与 per-phase oracle 比较 | 排除“只需一个固定 cap”；无切换 regret 就无需 adaptive 控制 |
| P2：有约束的联合 oracle | 相同吞吐/service-lag 预算下搜索小窗口 tenant order + admission 数量/预算 | 先证有额外 headroom；旧纯排序 oracle FAIL 不等于新动作集合 PASS |
| P2：历史/承诺归因 | 周期性 idle-return、持续 waiting、角色交换；同一 runtime rule 加/去 service reservation | 分离历史欠账、在跑承诺、队列冻结的作用；若预测版 VTC 已修复，就不主张新的 history 机制 |
| P3：泛化 | 独立真实 trace、3+ tenants、weights、第二 hybrid 模型、默认/优化 cache 路径 | 参数不从测试集选择；给 run/block CI、开销与失败区间 |

建议 P0/P1 先完成，再冻结新 policy 的机制 gate。不能为追求漂亮故事跳过更强 baseline。

## 6. 候选设计的合理范围

**合同层**声明并计量 service entitlement，保留 backlog/counter-lift 语义。

**资源层**观测 active state/KV 与实际 batch 运行状态，把内存 feasibility 与时间预算分开。

**动作层**在公平允许的范围内选择下一批请求并控制 admission；对已经在跑的请求，明确仅能等待自然释放，还是允许暂停/抢占。后者会改变系统成本和理论假设，不默认为免费操作。

可探索的贡献是：在目标模型上发现并定量解释分配与准入间的残余损失；给出能兑现服务合同的联合控制；实现低开销且泛化的反馈。资源 tracker、双 counter、time budget、输出预测分别都已有充分先例，不能仅列为三个新 technique。

若强组合 baseline 已足够，现有证据仍有价值：可形成 hybrid workload/configuration characterization，但不足以支撑新 scheduler 机制论文。

## 7. 可复核产物

- [只读审计脚本](audit_admission_batch_review.py)。
- [逐 run 数值、元数据校验和 pulse 敏感性结果](results/admission_batch_20260828/review_audit_20260908.json)。
- [原始冻结报告](results/admission_batch_20260828/admission_batch_motivation_report.md)，保留原 gate，不回写 PASS/FAIL。
- [更新后的主故事线](hybrid_state_aware_admission_motivation.md)。

审计命令：`python audit_admission_batch_review.py /path/to/admission_batch_20260828`。脚本只读原始 artifacts 并打印严格 JSON。新的敏感性结果是事后分析，不是新实验或独立确认数据。
