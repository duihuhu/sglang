# M2：Mixed 近释放窗口的收益线索与容量归因边界

2026-09-17。状态：完成16组正式运行、8192请求；另有16组完整同形预热、8192请求，不计入性能样本。GPU4/5上的本轮服务均已退出，结束时两卡均1MiB、0%利用率。

**结论：Mixed下仍观察到准入时机的改善线索，而且部分配对同时改善已有请求完成和新请求TTFT；但两次重复的完整场景匹配门槛未通过，本轮也没有观测到state/KV容量拒绝。因此，不能把它写成“固定state压力导致的剩余调度缺口已证实”。**

## 1. 为什么本轮是16组，而不是原定32组

原[Mixed/held-out协议](m2_mixed_holdout_protocol_20260917.md)计划比较far/near两个阶段。前两次启动受外部作业阻断，历史记录见[恢复记录](m2_mixed_holdout_recovery_20260917.md)。本次用户要求继续后，在空闲GPU4/5完成了`calibrate03`：4组背景校准、1792请求，另有4组等量完整预热。

原far锚点取背景中位第16个输出token时刻，但Mixed在继续部分请求decode时仍在处理其他背景的prefill：

| 背景形态 | far到达，s | 两次均存活 | 两次均已prefill | 原场景检查 |
|---|---:|---:|---:|---|
| train512 | 8.714 | 376/512 | 367/512 | 存活不足80%，已prefill不足95% |
| hold384 | 11.828 | 254/384 | 249/384 | 存活不足80%，已prefill不足95% |

因此原计划的`scenes_valid=false`保持不变，没有给它补根完成标记，也没有执行原32组矩阵。这只是**该固定锚点未形成所需场景**，不是“Mixed没有far状态”或性能策略失败。

near锚点取中位第96个输出token时刻，背景校准两轮均满足原场景条件。在任何正式策略对比之前，单独冻结[near-only补充协议](m2_mixed_near_protocol_20260917.md)：只保留near，其他策略参数、请求、rep81/82及判据不变，不按策略性能选择有利格点。

| 形态 | 冻结near到达，s | 校准存活数，rep71/72 | 剩余输出中位数，rep71/72 |
|---|---:|---:|---:|
| train512 | 17.694 | 488 / 511 | 32 / 35 |
| hold384 | 24.099 | 312 / 320 | 41 / 42 |

新计划保留原far失效证据、原manifest和8份背景原始artifact的SHA256。原协议、原运行器和失败计划均未改写。16组全部完成，没有从正式运行中丢弃不利结果。

## 2. 固定什么，比较什么

模型为Kimi-Linear-48B-A3B，A800 TP2，GPU4/5。固定Mixed4K、max-prefill-tokens=16384、正确的chunk复用slot计数、VTC权重input=1/output=2/request=0；关闭radix、CUDA graphs和speculative decoding。所有请求使用同一个租户`shared`，本轮不测多租户公平。

每rank共享raw预算为19,481,072,000 bytes，请求表607，KV容量参数781725。这里的容量参数不是把两种资源同时放满的承诺。实际state主体为21,708,800 bytes/request，KV主体为8,064 bytes/token；不使用其他拓扑的54MB/27KB数值。

| 形态 | 初始背景holder | 后到候选 | 每组总请求 |
|---|---|---|---:|
| train512 | 512个P512/O128 | 64个P8192/O64 | 576 |
| hold384 | 384个P1024/O128 | 64个P8192/O64 | 448 |

hold384未参与此前Separate阈值调参。它是固定规则迁移检查，不是独立生产trace。

| 策略 | 改变的动作 | 身份 |
|---|---|---|
| Native607 | 运行并发上限607 | 原生参考点 |
| Cap512 | 只将运行时并发上限改成512，不缩池或请求表 | 此前Separate校准的原生控制 |
| Slots96 | 保留607上限，原生min-free-slots阈值96 | 此前Separate校准的原生控制 |
| Decode32 | 候选到达后最多32次推迟新的prefill，继续已有decode | 使用实验标签的诊断动作，不是可部署新算法 |

6个服务会话、每个形态/策略两次独立会话重复。第二轮反转会话和case顺序；Native607和Cap512同会话交错。同rep跨策略的payload、输出长度及到达偏移一致，完整同形预热全部排除。

这不是四篇SOTA算法对比，也不是完成了Mixed全参数调优。Cap512/Slots96是从Separate迁移的固定参数点，不能称为已校准的强Mixed基线。

## 3. 性能观察

下表是两次会话的算术均值，单位秒；整体JCT包含背景和候选，新请求TTFT只含64个候选。

| 形态 | 策略 | Mean JCT | P95 JCT | Makespan | 新请求mean TTFT |
|---|---|---:|---:|---:|---:|
| train512 | Native607 | 29.445 | 40.750 | 61.596 | 22.929 |
| train512 | Cap512 | 29.449 | 40.756 | 61.617 | 22.936 |
| train512 | Slots96 | 29.697 | 40.976 | 61.838 | 23.071 |
| train512 | Decode32 | 22.467 | 36.961 | 57.864 | 20.023 |
| hold384 | Native607 | 36.758 | 51.025 | 69.243 | 23.258 |
| hold384 | Cap512 | 36.071 | 50.093 | 68.743 | 22.960 |
| hold384 | Slots96 | 36.031 | 50.051 | 68.701 | 22.916 |
| hold384 | Decode32 | 29.805 | 39.934 | 65.437 | 20.896 |

![Mixed训练形态：JCT与候选TTFT](figures/m2_mixed_train512.png)

![Mixed held-out形态：JCT与候选TTFT](figures/m2_mixed_hold384.png)

图注：柱是两次会话均值；圆点/方点分别为rep81/82，不是置信区间。图展示描述性结果，部分跨策略配对未满足到达时存活数相等的预设条件。Cap512/Slots96在本轮均未触发。不能仅根据柱高宣称确认性策略收益。矢量PDF见[训练图](figures/m2_mixed_train512.pdf)和[held-out图](figures/m2_mixed_hold384.pdf)。

仅用训练数据选择参考策略：Cap512/Slots96都未触发，不能以其小幅波动选“更优控制”，故冻结选择仍为Native607。Decode32相对这个参考点的描述性变化如下；正数代表改善，百分比是逐rep改善率的均值，不是上表均值的比率。

| 形态 | Mean JCT改善 | P95改善 | Makespan改善 | 新请求mean TTFT改善 | State-time下降 |
|---|---:|---:|---:|---:|---:|
| train512 | 23.70% | 9.30% | 6.06% | 12.67% | 30.13% |
| hold384 | 18.87% | 21.69% | 5.49% | 10.13% | 25.29% |

四个数值配对均同方向改善上述指标，新请求mean JCT分别改善9.96%/9.36%。不同于旧Separate近释放结果，这里没有看到“整体mean改善但新请求TTFT变差”的描述性权衡。不过，下面的匹配边界仍然适用。

## 4. 为什么完整screen仍未通过

16个正式run各自均通过near场景检查：背景prefill足够、仍有足够holder、剩余长度在规定窗口内、state占逻辑活跃主体至少2/3。**一个run属于near，不等于两个run处于同一个near状态。**

原配对条件另要求候选到达时存活背景数相同、剩余输出中位数差不超过8。对Native607与Decode32：

| 形态 / rep | Native607存活 / 剩余 | Decode32存活 / 剩余 | 配对状态 |
|---|---|---|---|
| train512 / 81 | 488 / 32 | 480 / 33 | 存活数不同 |
| train512 / 82 | 488 / 32 | 488 / 33 | 匹配 |
| hold384 / 81 | 336 / 44 | 320 / 42 | 存活数不同 |
| hold384 / 82 | 316 / 41.5 | 320 / 42 | 存活数不同 |

这些差别发生在候选到达之前，不能解释成候选策略的收益。即使请求和计划到达时刻相同，服务执行波动仍会改变该时刻的背景状态。因此不放宽原gate，也不把两次平均值当作严格匹配后的效果估计。

局部有效配对并非没有：train512/rep82匹配且Decode32实际触发，mean JCT改善23.40%、p95改善9.09%、makespan改善5.93%、新TTFT改善12.44%。hold384/rep81中，Decode32与Cap512、Slots96均为320个存活holder、剩余42，mean分别改善17.73%/17.63%，p95、makespan、新TTFT也改善。这些是**单个匹配配对的局部证据**，不是两次独立确认；Cap512/Slots96在这些run中仍然未触发。

所有完整两重复contrast的`screen_positive`均为false。这个false表示未满足冻结的完整筛选判据，不能翻译成“没有改善空间”。也不能反向忽略失配，宣布筛选PASS。

## 5. Mixed下观察到的机制有什么不同

下表窗口是候选到达到最后一个背景holder释放，数值为两次会话均值。Decode launch间隔来自CPU scheduler事件，不是GPU kernel duration。

| 形态 / 策略 | 背景全部释放，s | 释放前候选prefill批数 | 同时含背景decode的批数 | 最大背景decode launch间隔，s |
|---|---:|---:|---:|---:|
| train512 / Native607 | 22.356 | 65.0 | 65.0 | 0.455 |
| train512 / Cap512 | 22.367 | 65.0 | 65.0 | 0.453 |
| train512 / Slots96 | 22.670 | 65.5 | 65.5 | 0.456 |
| train512 / Decode32 | 11.167 | 32.5 | 32.5 | 0.352 |
| hold384 / Native607 | 28.955 | 85.5 | 85.5 | 0.427 |
| hold384 / Cap512 | 28.020 | 83.5 | 83.5 | 0.424 |
| hold384 / Slots96 | 27.979 | 83.5 | 83.5 | 0.424 |
| hold384 / Decode32 | 16.929 | 52.0 | 52.0 | 0.359 |

Mixed下，背景decode参与了窗口内所有包含候选prefill的批次。因此本轮不是再次修复Separate的长时间decode不参与。诊断动作减少重叠的prefill批次，并伴随更早的背景释放；持续参与batch不代表以最有利的速率完成。

Cap512/Slots96在全部8个对应正式run中均未触发，Decode32在全部4个run中都延迟32次。这说明旧固定参数没有覆盖当前状态，**不能**说明重新调优后的并发限制、chunk控制或动态delayer也无法覆盖它。

本轮最值得保留的insight是：**即时加入新prefill，即使不让已有decode完全停顿，也未必使已有请求或新请求更快；组批决策需要关注实际执行与完成速率，而不能只看请求是否参与。** 目前这是当前模型上的局部观察，仍可能由通用的prefill/decode计算干扰或workspace成本解释。

## 6. state/KV容量审计：还没有出现所需因果链

审计对16份正式原始轨迹逐一核对SHA256，在每个资源事件同时计算`state_slots * 21708800 + kv_tokens * 8064`后取峰值，不把不同时间的state和KV峰值相加。

| 形态 | 正式run数 | 逻辑state+KV主体峰值，bytes/rank | Raw减峰值，bytes/rank | Cache拒绝 |
|---|---:|---:|---:|---|
| train512 | 8 | 13,605,230,080 | 5,875,841,920 | 全部0个prefill_no_token、0个request_limit |
| hold384 | 8 | 11,739,525,888 | 7,741,546,112 | 全部0个prefill_no_token、0个request_limit |

这里raw减峰值是**逻辑缓存主体余量**，不是物理空闲HBM，也不是精确allocator可分配空间；不包含全部reserved、碎片、在途及workspace影响。`prefill_other`原样保留，不重命名成“内存不足”。

六个完整服务日志共记录65次可恢复CUDA allocation retry警告、12次custom-all-reduce fallback警告。日志计数含预热和正式阶段，尚未按发生时刻归属性能run。故不能声称整个进程没有显存压力，或已排除workspace/allocator干扰；能说的是**本轮轨迹没有显示state/KV cache容量拒绝介导了服务损失**。

背景更早释放后state-time下降是真实结果，但不能由此推出state空间曾阻塞其他请求，也不能推出预分配CUDA驻留内存同比下降。当前仍缺少“固定state占空间 → 候选因空间不足而等待 → 某批holder释放 → 候选获准 → 服务改善”的事件链。

此前候选三阶段设计的“再加24个P512/O64探针”只让所有请求同时达到完整长度的逻辑上界从19.408GB增加到20.040GB。此次Mixed实测峰值仅13.605GB，额外24个探针的主体上界约0.632GB；**不能再依靠总工作集上界预测实际一定触碰预算**。新探针也可能改变执行重叠，所以该算式不是“不可能产生压力”的证明；必须先测实际同时驻留。

## 7. 完整性、复现和适用范围

16组正式和16组预热全部完成，结束active为0；无请求失败、retraction、prefix hit或trace drop。正式run最大提交迟到3.828ms，低于50ms门槛。高负载SM时钟1380--1410MHz，最高61摄氏度，NVML峰值81036MiB；未改变时钟或功率，也未停止其他用户任务。

运行前38项CPU回归通过；新增适配器、分析器和绘图程序通过Python编译检查。原生产调度源码没有因本轮修改。远端没有matplotlib，使用本地已有matplotlib生成PNG/PDF，两张PNG均已目视检查；没有为绘图安装远端包。

每格仅两次独立会话，不给出稳定总体CI。有限burst、关闭graphs、单租户、单模型和单拓扑的结果不能外推成生产稳态SLO、公平性或全部先进batching方法的结论。

- [正式结构化分析](results/m2_mixed_holdout_20260917/near_analysis01/analysis.json)与[逐run报告](results/m2_mixed_holdout_20260917/near_analysis01/report.md)：含原始artifact路径、SHA256、场景、配对gate、控制触发及日志警告。
- [原校准到达计划](results/m2_mixed_holdout_20260917/calibrate03/arrival-plan.json)保持far无效；[near补充计划](results/m2_mixed_holdout_20260917/near-plan01.json)显式保留原计划来源。
- [正式manifest](results/m2_mixed_holdout_20260917/near_measure01/manifest.json)与[完成标记](results/m2_mixed_holdout_20260917/near_measure01/completed.json)；[逐轨迹容量审计](results/m2_mixed_holdout_20260917/near_capacity01/audit.json)。
- [near运行适配器](run_m2_mixed_near.py)、[分析器](analyze_m2_mixed_near.py)、[回归测试](test_m2_mixed_near.py)、[绘图程序](plot_m2_mixed_holdout.py)。

大型原始请求、stream、资源轨迹和GPU监控仍在远端`/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_mixed_holdout_20260917/near_measure01/`，宿主机对应`/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/m2_mixed_holdout_20260917/near_measure01/`。本地同步小型分析、协议、manifest、完成标记和图，未声称已复制所有原始轨迹。

## 8. 下一步优先级

1. **先使对照可匹配，并校准真正的Mixed原生基线。** 冻结新的场景控制方式或分块重复协议，验证候选到达前状态；在训练形态调cap/chunk/delayer，记录实际触发，再冻结到held-out。不能看完本轮数据后放宽原匹配条件，不能用Slots96未触发代替整个动态控制的负结果。
2. **先验证实际压力可达，再测容量介导收益。** 在同模型、正确统一池、请求上限及完整HBM/workspace审计下，设计让背景state与候选KV增长真正重叠的负载，例如延长或分阶段安排背景decode。首先检查普通cap尚宽裕而共享空间确实不足；不再仅用所有请求最大长度之和判断压力。
3. **补齐拒绝、释放、获准的事件链及宽容量负对照。** 记录被拒请求新增state/KV需求、预留、剩余budget、allocator gap和可压缩空间，区分计算排队与空间阻塞。控制cap不变，不将普通并发上限或人为state padding冒充自然hybrid成本的独立干预。
4. 只有上述链路成立且强原生基线后仍有受约束收益，才补足确认性重复、先进batching策略、多模型和生产稳态；当前不进入lifetime predictor。

本轮将“Mixed和新形态是否还有机会”推进到了可复核的数值线索及单配对证据；**尚未把它推进到“强Mixed基线后仍有可重复缺口”，更未推进到“缺口由hybrid固定state压力额外造成”。** 这不是否定已有M2局部问题，而是明确论文因果主张仍缺哪一段证据。
