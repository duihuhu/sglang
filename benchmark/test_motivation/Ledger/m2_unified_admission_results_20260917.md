# M2：统一池之上的准入时机，部分实验结果

> 后续已补齐16组筛选；本文保留为12组阶段记录。更新结果见[完整筛选与原生机制对照](m2_unified_admission_complete_20260917.md)。

2026-09-17。Kimi-Linear-48B-A3B，A800，TP2，物理GPU4/5。**状态：PENDING，16组正式实验完成12组；每个场景目前只有一组完整策略配对，不判PASS，也不把缺失数据记为FAIL。**

**初步观察：在正确配置的统一池上，state占比较高且已有请求接近完成时，暂缓新prefill、先推进已有decode，可以大幅缩短已有请求JCT及state持有时间，但新请求的排队和TTFT会变差。当前支持“准入时机具有阶段相关的服务代价”，尚未证明“state容量约束导致这些性能损失”，也未证明超出已有组批机制的收益。**

## 1. 相比上一轮，回答什么新问题

[上一轮池管理对照](m2_pool_comparison_results_20260916.md)已经证明：原485请求容量瓶颈可以被重划静态池或统一池配合提高上限解决。不能再重复用这一瓶颈证明已有方法无解。

本轮固定上一轮的C统一池配置、实际请求上限607，问一个更窄的问题：**空间足够时，立即加入新请求是否总是有利？动作价值是否随已有请求的执行阶段变化？**

这是一组新的分阶段到达工作负载，不是上一轮固定607请求轨迹的复跑；不得将两轮延迟直接相减或跨GPU配对。沿用的只是已校验池配置。本轮没有实现lifetime predictor或修改模型、分配器和SGLang服务源码，诊断动作由独立launcher临时加载。

## 2. 控制与完成度

完整规则见[冻结协议](m2_unified_admission_protocol_20260916.md)，场景时间见[校准计划](results/m2_unified_admission_20260916/calibrate03/screen-plan.json)。

| 项目 | 固定内容 |
|---|---|
| 池和容量 | 同一共享raw池19481072000 bytes/rank；request/bootstrap607；KV启动参数781725；runtime cap607 |
| 调度 | Separate4K；本地token-VTC input1/output2/request0；修正继续chunk的slot计数；额外activation limit0 |
| 模型执行 | 无prefix复用，radix/graphs/spec关闭，backend、dtype与TP不变 |
| State主导背景 | 512个P512/O128请求 |
| KV主导背景 | 96个P16384/O128请求 |
| 新候选 | 64个P8192/O64请求；各组请求间隔1ms；同一tenant，holder/candidate仅为分析分组 |
| 阶段 | 背景第16/96个生成token的校准中位时刻定义far/near；先冻结到达，再做性能实验 |
| Native | 正常Separate4K准入 |
| Decode32 | 首次候选等待且无继续chunk时，最多32次合格准入扫描不加prefill，推进已有decode；背景全部释放则提前结束；不重排候选，不使用未来输出长度 |

32次准入扫描不是固定32次GPU forward。该动作是诊断参照，不是已验证的可部署策略；在正式状态中记录实际触发次数。两类背景计算量不同，不能把它们的绝对延迟差视为纯state干预。

背景校准完成4组、1216请求，另有等量完整预热。所有冻结锚点满足背景活跃、prefill完成、剩余输出及资源主体占比检查。正式state场景到达时的state主体占比约81.6%--83.6%，KV场景的KV主体占比约85.9%；它们是按实际live请求/token换算的主体字节代理，不是NVML显存比例。

正式已采用3个完整服务会话，共12组、4416请求；12组完整同形预热、4416请求另行排除。每组完整性、实际到达场景、连接复用及迟到门槛均通过。未完成的native/rep52会话只有160请求预热，整个会话排除。尚缺4组正式1472请求及其完整预热。

原始大文件保留远端`/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_unified_admission_20260916/screen01`；[本地分析JSON](results/m2_unified_admission_20260916/partial_analysis02/analysis.json)保留来源、请求/资源指标、场景检查及采用文件哈希。[恢复记录](m2_unified_admission_recovery_20260916.md)说明所有中断、排除和版本变化。

## 3. Observation：收益取决于阶段，不是越晚准入越好

以下仅是rep51的完整配对，正数表示相对native的改善。预注册主指标为mean JCT，整体p95 JCT与makespan最多退化5%；TTFT和分组代价必须同时披露，但没有预注册TTFT的5%限制。

| 场景 | Mean JCT改善 | P95 JCT改善 | Makespan改善 | 解释 |
|---|---:|---:|---:|---|
| State / far | -0.68% | -0.07% | -0.08% | 暂缓没有显示收益 |
| State / near | +54.63% | +27.68% | -2.78% | 有受约束收益信号，尚缺第二组配对 |
| KV / far | -0.94% | -0.18% | -1.33% | 暂缓没有显示收益 |
| KV / near | -0.91% | -0.34% | -0.31% | 本次32次扫描不足以让背景先完成 |

四格均为PENDING。请求数量不能充当独立系统重复，不给总体置信区间，也不把一次正结果提升为整个M2已成立。

## 4. 具体问题与收益代价：State / near

![单组配对中的请求代价与释放时序](figures/m2_unified_admission_partial.png)

图只展示rep51一组配对，没有误差条。512个已有请求与64个新请求数量不同，整体均值自然更受前者影响，因此必须分别报告。

| 指标 | Native | Decode32 | 变化 |
|---|---:|---:|---|
| 整体mean JCT，s | 37.884 | 17.189 | 降低54.63% |
| 整体p95 JCT，s | 39.392 | 28.487 | 降低27.68% |
| Makespan，s | 41.166 | 42.309 | 增加2.78% |
| 已有512请求mean JCT，s | 39.202 | 15.777 | 明显改善 |
| 新64请求mean JCT，s | 27.342 | 28.485 | 增加1.143s，约4.18% |
| 新请求mean queue，s | 11.401 | 13.471 | 增加2.070s |
| 新请求mean TTFT，s | 11.830 | 13.898 | 增加2.068s，约17.5% |
| 新请求p95 TTFT，s | 22.085 | 24.097 | 增加2.012s |
| 整体p95 TTFT，s | 13.088 | 15.152 | 增加约15.8% |
| State-time，slot-seconds | 19232.98 | 7201.55 | 降低62.56% |
| Peak active requests | 576 | 512 | 候选首次激活前背景已释放 |

Queue为提交到首次state激活，不等于TTFT。该动作满足本次mean-JCT主目标的单次代价门槛，**不是所有指标都获益，更不是公平改善**；若服务目标重视新请求TTFT，这个代价可能不可接受。

释放轨迹解释了结果：native在候选到达后0.068s开始激活新请求，但512个背景holder到25.570s才全部释放；decode32将首次候选激活推迟到2.195s，背景在2.187s已全部释放。已有请求mean queue几乎不变，JCT改善主要发生在激活后的阶段。

**具体问题：对于这个Separate4K配置，立即开始新prefill可能延长大量即将完成请求的执行与持有时间；“现在有空间”并没有回答“此时加入是否符合服务目标”。Insight：准入时机应与已有请求的执行阶段和新旧请求间的代价转移一起评价，不能只看可行容量或queue。**

但这个配对中没有`request_limit`或`prefill_no_token`容量拒绝，native可以同时容纳全部576请求；decode32反而增加新请求queue。因此没有证据说“提前释放state使本来放不下的候选得以加入”。更直接的解释是避免新prefill延迟即将完成的decode，仍需更强已有组批机制对照。

## 5. 第二次干预与负面边界也要保留

Decode32/rep52的State-near mean JCT为17.294s，p95为28.690s；512个holder在候选到达后2.283s释放、首个候选在2.287s激活，形态与rep51相近。**它没有对应native/rep52，不作为第二次配对收益，也不与native/rep51重复配对。**

KV-near的两次干预不同：rep51到达时median remaining为44，等待32次仍未释放全部背景；rep52为31，30次后96个背景全部释放，mean JCT为84.024s。后者仍缺匹配基线。这提示固定32次动作可能对实际剩余工作敏感，不能选择性只保留State-near，或看完结果再扩大等待次数寻找赢家。

## 6. 和Hybrid的关系：已验证什么，尚未验证什么

| 命题 | 状态与证据 |
|---|---|
| Hybrid请求持有固定state，执行延迟会延长这份固定成本的占用 | 已有资源测量成立；本轮观察到state-time随准入时机显著变化 |
| 正确统一池之上仍可能存在阶段相关的服务收益 | 当前单组完整配对提供信号，需补重复与已有机制对照 |
| 该收益由state容量约束因果性介导 | 未证明；正结果格没有容量拒绝，候选queue反而增加 |
| Hybrid使相同组批干扰产生更大的持久state占用后果 | 本模型内可以量化该占用后果；相对其他系统的放大幅度尚未作受控对照 |
| 原生动态机制或更强batching基线也无法处理 | 未测，不能声称 |
| 新scheduler、lifetime predictor或新的公平抽象因此必需 | 未证明，不进入设计 |

不需要证明普通attention系统永远没有类似现象。需要证明的是：**这个模型的固定state或双资源约束，对所关心的服务损失有可观察、可干预的额外作用。** 当前可以将M2写成“hybrid中的准入与持有后果”，不能写成“已证实hybrid特有的剩余性能缺陷”。仅有state-time下降是资源结果，不自动构成性能瓶颈的因果证据。

## 7. 下一步顺序

1. **先补齐本矩阵。** 同GPU4/5、同源码与冻结到达，恢复整个native/rep52会话；不借用不完整会话预热，不跨卡拼接。若等待期间条件已明显变化，则如实标注并重新做完整分块比较，不强行继承旧配对结论。
2. **再检查已有机制是否覆盖信号。** 先完成预注册的`slots64`与`delayer32`触发审计；它们不是调好的强基线。随后另行冻结cap、chunk、Separate/Mixed和原生阈值的校准/held-out划分，不把本轮四格同时用于选参和宣称泛化。
3. **只有仍有收益，才检查state的额外作用。** 使用不改变请求token工作、排序和执行模式的资源预算干预，检查相同动作是否通过可用字节/slot、拒绝原因、释放、后续激活这条链影响服务；完整匹配并披露驻留预算和metadata，不用换模型或长短工作负载的绝对延迟差代替因果对照。

源码审计限定了第2步的解释：`MinFreeSlotsDelayer.should_delay`判断可分配请求数是否低于阈值；607-512=95，阈值64在背景512已激活且候选尚未激活时不会立即等待，但后续可能触发。`PrefillDelayer`有slot条件、可选queue条件、首次跳过与等待上限；scheduler传入的是各池最大usage，不能声称它天然只看KV。应记录真实trigger/reason，调一个更大等待上限并不保证动作会触发。现有代码可处理哪些条件，需要实测，不能由默认关闭推断算法缺失。

截至本次收尾，远端8卡均被其他任务占用，未抢占。剩余GPU实验未运行。恢复计划已经完成哈希校验，分析与恢复单元测试18项通过。未运行的Mixed、原生动态控制、held-out和state因果干预不计入已完成实验。

恢复命令在容器`/workspace/sglang`执行；运行器仍会锁卡并检查空闲：

```bash
.venv/bin/python benchmark/test_motivation/Ledger/recover_m2_unified_admission.py \
  --mode run \
  --plan benchmark/test_motivation/Ledger/results/m2_unified_admission_20260916/recovery-plan02.json \
  --output benchmark/test_motivation/Ledger/results/m2_unified_admission_20260916/recover01
```

仅成功补全后，对`recover01/assembled`运行`analyze_m2_unified_admission.py`；不要给原`screen01`目录手工补完成标记。
