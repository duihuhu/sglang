# M2：完整准入筛选与原生机制对照

2026-09-17，Kimi-Linear-48B-A3B，A800 GPU4/5，TP2。

> 后续更新：[12组原生参数校准与state可识别性](m2_native_counterexample_results_20260917.md)已完成。Cap512/Slots96在新配对中取得约54.4%的State-near mean改善，说明本报告的主要收益量级可由已有通用控制实现；State-far的更强等待也改善mean，但makespan/TTFT代价更大。以下32组数值与历史判定保持不变，第6节中的首项校准已由新报告推进，不再把“Slots64只覆盖部分收益”当作已知原生能力的上限。

状态：native/decode32的16组筛选与原生slots64/delayer32的16组对照均完整完成，共32组正式11776请求，另有等量完整预热。原生对照曾因一次77.65ms提交迟到整会话排除，已按冻结计划完成一次整会话恢复。本文接续[12组阶段报告](m2_unified_admission_results_20260917.md)，不覆盖其历史数据与单组图。所有排除与恢复见[执行记录](m2_unified_admission_recovery_20260916.md)。

**总判断：正确统一池之上存在可重复的准入时机收益，但已有Slots64已覆盖部分收益。State-near整体mean JCT从37.88s降至27.58s，诊断Decode32为17.24s；新请求TTFT从11.85s分别增加至13.03s和13.97s。问题是目标相关的准入/执行权衡，尚不是hybrid独有性能缺陷或新算法必要性的证明。**

## 1. 完整筛选结论

**在已经正确配置的统一池上，State-near中暂缓一次新prefill的收益得到两组独立会话配对支持：mean JCT降低54.49%、p95 JCT降低27.43%，makespan增加3.05%。但新请求mean TTFT增加约17.9%，且该场景没有容量拒绝。当前证明的是阶段相关的准入/执行代价，不是state容量因果性放大，也不是新算法必需。**

沿用[冻结协议](m2_unified_admission_protocol_20260916.md)：同一共享raw池19481072000 bytes/rank、实际request/runtime cap607、Separate4K、修正的slot门限、token-VTC input1/output2/request0。无prefix复用，radix/graphs/spec关闭，不改变算子或物理池。State背景为512个P512/O128，KV背景为96个P16384/O128；64个P8192/O64候选在冻结far/near时刻到达。两类背景计算量不同，不作跨类别绝对延迟的纯state归因。

16组正式5888请求和16组完整预热5888请求全部通过请求、trace、连接复用、提交迟到和实际到达场景检查。4组native/rep52在同GPU恢复，前三个完整会话保留；旧不完整会话的160请求预热仍排除。矩阵由有来源记录的会话组装，不是连续一次执行。只有两次独立重复，不报告总体置信区间。

[完整筛选分析](results/m2_unified_admission_20260916/screen_analysis01/analysis.json)与[执行轨迹审计](results/m2_unified_admission_20260916/execution_screen01/audit.json)保留原始路径、文件哈希和每次观测。大型原始文件在远端`/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_unified_admission_20260916`，本地同步汇总和图。

| Decode32相对native | Mean JCT改善 | P95 JCT改善 | Makespan改善 | 两次均满足约束 |
|---|---:|---:|---:|---|
| State / far | -0.90% | -0.30% | -0.30% | 否 |
| State / near | +54.49% | +27.43% | -3.05% | 是，仅此诊断筛选通过 |
| KV / far | -0.90% | -0.15% | -1.32% | 否 |
| KV / near | +4.77% | +6.00% | -0.13% | 否，mean分别-0.91%和+10.46% |

百分比是逐rep变化率的均值，正数表示改善。预注册约束为mean JCT降低且整体p95 JCT、makespan退化不超过5%。TTFT未设该5%约束，但必须披露。诊断筛选通过不等于整个M2或论文新颖性通过。

![完整筛选mean JCT](figures/m2_unified_admission_screen_jct.png)

柱为两次run均值，点为各自观测，不是置信区间。KV-near的波动必须保留；不能用均值+4.77%掩盖方向不一致。

## 2. 问题、观察与Insight

被检查的机制是Separate4K下新请求加入prefill batch的时机，不是VTC公平保证或HBM安全性。

所有请求的`tenant_key`均为`shared`；holder/candidate只是执行阶段和结果分析分组，不是两个租户。因此本轮隔离的是准入/执行时机，不直接测多租户公平改善，不能替代M1的跨租户服务隔离实验。

| State-near指标，两次均值 | Native | Decode32 |
|---|---:|---:|
| 整体mean JCT，s | 37.883 | 17.242 |
| 每次run的p95 JCT，s | 39.392 | 28.588 |
| Makespan，s | 41.155 | 42.410 |
| 已有512请求mean JCT，s | 39.202 | 15.823 |
| 新64请求mean JCT，s | 27.331 | 28.586 |
| 新请求mean TTFT，s | 11.851 | 13.972 |
| State-time，slot-seconds | 19238.72 | 7221.38 |

整体均值按请求计权，因此512个已有请求对结果的影响大于64个新请求。不能把54.49%的整体mean改善写成所有请求获益或公平改善。新请求JCT增加约1.26s、TTFT增加约2.12s；不同服务目标可能做不同选择。

![完整筛选中新请求TTFT代价](figures/m2_unified_admission_screen_ttft.png)

轨迹证据如下，窗口为候选开始到达至最后一个背景holder释放：

| State-near机制 | Native rep51 / rep52 | Decode32 rep51 / rep52 |
|---|---|---|
| 释放前候选prefill批数 | 128 / 128 | 0 / 0 |
| 释放前候选prefill token数 | 524288 / 524288 | 0 / 0 |
| 背景decode调度批数 | 31 / 31 | 31 / 32 |
| 到达后首次背景decode调度，s | 22.993 / 23.026 | 0.009 / 0.021 |
| 背景全部释放，s | 25.570 / 25.611 | 2.187 / 2.283 |

这些是CPU观察的调度时间和提交工作量，不是GPU算子持续时间。两次native都先安排128个候选prefill批次，已有请求的剩余decode和state释放被推迟；诊断动作先推进已有decode。正结果格没有request-limit或prefill-NO_TOKEN拒绝，native可容纳全部576请求。

**具体问题：空间允许的立即准入，可能在Separate组批下推迟大量即将完成请求，放大其执行时间和固定state持有成本。Insight：准入应按端到端服务目标评价，既看新请求排队，也看它对已有请求推进及释放的影响。**

这仍不证明state约束是本次延迟差的原因。资源结果是state-time下降62.46%；性能机制更直接体现为prefill/decode的执行顺序。Hybrid固定state让等待附带额外持有成本，但没有容量介导证据时，不能从额外持有成本跳到hybrid特有性能缺陷。

State-time是共享池内逻辑占用的时间积分。源码中的`UnifiedKVPool`预分配raw buffer，state的`free`归还池内可用空间，不直接销毁这块CUDA分配。因此不能把state-time降低62.46%写成NVML驻留显存降低62.46%；运行峰值显存变化还包括不同batch的workspace和分配器行为。

## 3. 不能忽略的边界

- State-far没有收益：32次诊断扫描不足以完成背景，后续仍插入候选prefill，等待代价没有换来相同释放效果。
- KV-near两次不一致：诊断到达时median remaining分别44和31；后者30次扫描后完成背景，前者没有。Native同rep的remaining分别36和28，说明相同到达时间不保证跨会话完全相同的执行状态；不能把粗粒度near标签当成严格剩余工作匹配。
- State-near的剩余工作匹配较好：rep51双方均32，rep52双方均33；强正结果不依赖上述KV-near不平衡。
- 同rep跨策略也来自不同服务会话，恢复又引入跨时段执行。同物理GPU、相同请求/到达和逐次场景复核可以减少混杂，但不能排除共享宿主机漂移；两次会话观测不是统计显著性或生产稳态证据。
- 四个筛选会话都有每rank一次custom-all-reduce构建失败后的回退警告，条件一致，未为某个策略单独修改后端。
- 四会话CUDA分配重试警告数为2/2/0/2；不是请求失败。恢复会话两条位于State-near预热，随后正式回放通过。其他会话保留日志计数，不在未核对时间窗时声称全部只在预热。实际运行峰值显存随batch变化，不是严格等运行峰值的实验。
- State-near正式GPU负载样本SM时钟范围1365--1410MHz，没有连续高负载低频异常；native峰值NVML约80978--81008MiB，decode32约75594--76138MiB。不能只匹配cache配置就隐去workspace/分配行为。

## 4. 原生对照：已有机制覆盖了多少

原生对照仍使用相同冻结负载、池、cap、Separate4K与完整预热，每个策略两个独立会话，顺序反转：

| 对照 | 改变什么 | 必须检查 |
|---|---|---|
| Slots64 | 原生`--min-free-slots-delay 64` | `native_slot_checks`及`native_slot_delays`；是否在背景推进/候选准入窗口实际触发 |
| Delayer32 | 原生`--enable-prefill-delayer --prefill-delayer-max-delay-passes 32` | 原生reason中`delay`、`no_wait`、`wait_timeout`等；开启不等于触发 |

这两个固定配置不是调好的强基线，更不是两篇SOTA论文的完整复现。Slots64是可用request槽阈值，607-512=95，候选尚未激活时不会立即等待，后续可能触发。PrefillDelayer有slot/可选queue条件和首次跳过等逻辑；scheduler传入各池最大usage，不能声称其只感知KV。

Delayer32两轮共8组正式检查通过，实际延迟计数全部为0。每个State场景记录192次`no_wait`，每个KV场景508次`no_wait`。这说明本次参数组合未触发延迟，不说明整个PrefillDelayer机制无效；仅增加`max-delay-passes`不会让未满足的slot/queue条件自动成立。

### 4.1 State-near完整对比

四策略到达时均有512个已prefill背景，state主体占state+KV主体字节81.58%。每个rep中四策略的median remaining相同，分别为32和33，实际场景匹配。

| 指标，两次均值 | Native | Slots64 | Delayer32 | Decode32诊断 |
|---|---:|---:|---:|---:|
| 整体mean JCT，s | 37.883 | 27.576 | 37.908 | 17.242 |
| 每次run的p95 JCT，s | 39.392 | 28.885 | 39.414 | 28.588 |
| Makespan，s | 41.155 | 42.735 | 41.197 | 42.410 |
| 已有请求mean JCT，s | 39.202 | 27.519 | 39.224 | 15.823 |
| 新请求mean JCT，s | 27.331 | 28.032 | 27.373 | 28.586 |
| 新请求mean TTFT，s | 11.851 | 13.031 | 11.839 | 13.972 |
| 新请求p95 TTFT，s | 22.104 | 24.426 | 22.106 | 24.193 |
| State-time，slot-seconds | 19238.72 | 13239.34 | 19255.76 | 7221.38 |
| 实际延迟次数，rep51 / rep52 | 0 / 0 | 31 / 32 | 0 / 0 | 32 / 32 |

Slots64的mean JCT逐rep改善27.32%/27.09%，均值27.21%；p95改善26.67%，makespan增加3.84%，两次均满足原有受约束收益门槛。其新请求mean JCT增加2.57%、mean TTFT增加9.96%，state-time降低31.18%。因此原生机制已可处理一部分机会，不能把诊断动作相对native的全部54.49%记为新设计空间。

Decode32仍有更低整体mean，但相对Slots64也付出更高新请求mean JCT和mean TTFT；它不是对所有指标都更优。两者都是当前固定参数点，没有证明原生机制调优后的最优能力。

![四策略整体mean JCT](figures/m2_unified_admission_controls_jct.png)

![四策略新请求mean TTFT](figures/m2_unified_admission_controls_ttft.png)

### 4.2 为什么Slots64只覆盖部分收益

| State-near执行证据，rep51 / rep52 | Native | Slots64 | Delayer32 | Decode32 |
|---|---|---|---|---|
| 已有请求全部释放前候选prefill批数 | 128 / 128 | 64 / 64 | 128 / 128 | 0 / 0 |
| 上述候选prefill tokens | 524288 / 524288 | 262144 / 262144 | 524288 / 524288 | 0 / 0 |
| 最长无已有请求decode调度间隔，s | 22.99 / 23.03 | 11.49 / 11.50 | 23.02 / 23.01 | 0.25 / 0.26 |
| 到达后已有请求全部释放，s | 25.57 / 25.61 | 13.88 / 13.98 | 25.60 / 25.66 | 2.19 / 2.28 |
| Peak active | 576 / 576 | 544 / 544 | 576 / 576 | 512 / 512 |

Slots64起初有95个空槽，不触发；新增约32个候选后空槽降到63，才使后续新prefill等待。两次都在背景释放前插入64个候选prefill批次，而native为128。这个执行序列与约一半的mean JCT改善一致，但不是kernel时间分解或完整因果归因。最长无decode调度间隔包含到达与释放边界；只看首次decode可能被边界上的一次调度误导，因此同时保留完整窗口审计。

State-far中Slots64两次均触发65次延迟，mean JCT仅改善0.94%/1.10%，均值1.02%；p95增加0.11%、makespan增加2.33%。它按预注册数值规则筛选通过，但幅度小、仅两次重复，不视为可靠统计显著收益。KV两个场景Slots64均未触发，细小数值波动不作为该机制的收益。

### 4.3 完整性和来源

[32组正式汇总](results/m2_unified_admission_20260916/controls_analysis01/analysis.json)、[逐组与配对表](results/m2_unified_admission_20260916/controls_analysis01/report.md)、[32组执行审计](results/m2_unified_admission_20260916/execution_controls01/audit.json)已同步本地。原始大型轨迹保留远端，汇总保留哈希及恢复来源，不是完整原始数据镜像。

正式32组11776请求与完整预热32组11776请求均通过测量完整性与场景检查；正式最大提交迟到37.05ms，低于冻结50ms门槛，回放全部复用连接。原生失败会话的320正式/320预热请求整会话排除；旧筛选中断的160请求预热也不计入上述数量。没有将预热、校准或不完整会话拼入性能结果。

8个采用会话均有每rank一次custom-all-reduce回退警告；CUDA分配重试共10条，其中screen为6、原生对照为4，不宣称零分配压力。正式GPU高负载采样SM1365--1410MHz，最高66摄氏度，NVML峰值81088MiB；原生对照也不是严格相等运行峰值的实验。42项相关CPU回归测试通过，执行审计4项本地单元测试通过。恢复运行器与服务均已退出，GPU4/5各1MiB、利用率0，未停止其他任务。

## 5. 当前可以与不可以写进Motivation的结论

可以写：正确统一池并不自动决定最佳准入时机；同一可行域、同一排序下，新prefill对已有decode推进和固定state持有时间有可重复的影响，并存在明确的新旧请求代价转移。

还不能写：state容量导致本次54.49%损失；普通attention不会发生；已有动态batching无法处理；必须设计lifetime predictor。原生机制固定参数对照现已完成；后续应先调优原生阈值/cap、校准chunk/Mixed并做held-out，再检查state的额外因果作用。

## 6. 下一轮验证顺序与停止条件

以下是待验证项，不计作本轮已完成实验；开始执行前另冻结完整矩阵、到达时间、重复数和服务约束，不在本轮结果上追加选择性重复。

1. **先排除原生参数没有选到位。** 当前cap607、背景512，初始空槽95。`min-free-slots=96`会在候选首次激活前触发，而64是在新增部分请求后触发；cap512也可能直接实现“背景释放后再准入”。优先将这些机制上可解释的对照加入四场景，而不只测正结果State-near。若它们覆盖诊断收益，则已有机制的配置能力足够，本轮不能声称缺新算法。
2. **再排除Separate4K的特定执行顺序。** 校准Mixed及chunk预算时继续固定模型、raw池、工作集和排序，报告候选加入前的真实执行阶段。不同batching下背景进度可能不同，不能机械复用near标签后声称状态已匹配。原生参数在校准集选定后冻结，再比较未参与调参的请求长度、到达时刻和背景规模。
3. **最后检查state是否介导额外损失。** 需要在保持模型计算路径及工作集不变的干预中，观测“state继续持有 -> 真实可用容量不足 -> 后续请求等待 -> 释放后获得准入”的连续证据，并用宽容量负对照检查该差异是否缩小。人工缩小容量只能证明该约束下的因果作用；要解释自然部署，还须核对完整内存预算与生产配置。不能把state-time下降或跨模型绝对延迟差当成这条证据链。

每一步都同时报告已有/新请求的JCT和TTFT、makespan、实际触发及资源轨迹。当前5%门槛只约束整体p95 JCT和makespan，不能追溯性地当成候选TTFT保证。若收紧候选TTFT约束后收益消失，应保留为目标权衡，而不是改指标寻找正结果。
