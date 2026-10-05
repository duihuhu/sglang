# M2：固定工作集后，state-cap 与 batching 的交互及目标权衡

2026-09-16。Kimi-Linear-48B-A3B，A800 GPU6/7，TP2。完成26组正式运行、15782请求；每组固定607个P512/O128请求，26组同形态完整预热另行排除。中断恢复有独立来源记录，没有实现lifetime predictor或新调度算法。

> 后续测试已完成：[等完整驻留预算的静态/统一池对照](m2_pool_comparison_results_20260916.md)。新增GPU0/1的24组实验，不并入本报告26组：重划静态池和统一池配合提高请求上限均可达到607；同池cap干预仍显示tail改善、mean及state-time增加。原容量瓶颈可由已有配置修复，不能据此声称统一池无解。下文保留原实验与当时的下一步计划。

**核心结论：同一物理池内，cap485→607使Mixed相对Separate的mean JCT差距从5.577s缩到4.881s，两次差值均为正。这是“state派生并发上限影响batching差距”的探索性干预证据，但不是放宽state带来mean JCT收益：Separate和Mixed自身的mean JCT反而分别增加6.99%和2.07%。排队、尾延迟与整批完成时间改善，同时激活后的完成时间与state驻留积分增加。当前更明确的观察是准入与执行的目标权衡，不能升级成hybrid特有缺陷或新算法必要性。**

校准矩阵已完成，不等于校准已收敛：部分配置存在明显运行波动，不能宣布稳定最优cap，也不能将自然池到宽池的全部变化归因于state。

## 1. 要回答的问题与固定条件

上一轮256→607同时改变工作量，不能据此证明state放大。本轮按[冻结协议v3](m2_state_mediation_protocol_20260916.md)，在每个rep内固定实际input IDs、输出长度和计划到达顺序，跨模式/cap/物理池逐项配对。两次独立启动rep11/12，不能当成607个独立系统重复。

共同条件：radix off、prefill/decode graphs off、无spec；mem-fraction-static=0.85，max-prefill-tokens=16384；本地token-VTC input1/output2/request0/二次项0；corrected slot gate=1，额外new-activation limit=0；原生可选delay/prefill-max-requests关闭，统一池未启用。有限burst为1ms计划间隔，共0.606s到达窗口。

| 条件 | State / KV / request物理容量 | Runtime cap | 用途 |
|---|---|---|---|
| A，自然池 | 485 / 1458643 / 485 | 128、256、485 | Separate-4K、Mixed-4K、Mixed-1K校准，18正式runs |
| B，宽池 | 728 / 804471 / 728 | 485 | 与自然cap485比较池布局；与C构成同池干预 |
| C，同一个宽池 | 与B逐字段一致 | 607 | 只放宽runtime cap；B/C共8正式runs |

每次正式运行前都有独立负rep的完整607请求预热、flush和共同策略重置。模式顺序反转，cap按预定seed排列；二值cap顺序并未在每种模式下完全平衡，例如Mixed两次都是485先于607，后续确认需要明确反平衡。

这里比较的是SGLang原生batching配置，不是完整Sarathi-Serve、FairBatching或其他SOTA的复现。工作集全部为同一short组，本轮不检验多租户公平保证、稳态吞吐或SLO goodput。

## 2. 校准：普通控制有影响，但尚无稳定最优配置

![固定607请求的原生batching校准](figures/m2_state_cap_calibration.png)

线为两次均值，点为各次运行。全部配置保留；大幅波动不通过删点或改门限处理。

| 模式 | Cap | Mean JCT rep11 / rep12，s | 两次均值，s | Peak active，两次相同 |
|---|---:|---|---:|---:|
| Separate-4K | 128 | 34.862 / 24.870 | 29.866 | 128 |
| Separate-4K | 256 | 18.135 / 18.198 | 18.166 | 256 |
| Separate-4K | 485 | 18.727 / 16.415 | 17.571 | 485 |
| Mixed-4K | 128 | 40.530 / 25.398 | 32.964 | 128 |
| Mixed-4K | 256 | 20.566 / 20.815 | 20.691 | 256 |
| Mixed-4K | 485 | 29.464 / 22.005 | 25.735 | 485 |
| Mixed-1K | 128 | 33.999 / 38.347 | 36.173 | 128 |
| Mixed-1K | 256 | 46.467 / 32.285 | 39.376 | 229 |
| Mixed-1K | 485 | 34.816 / 43.289 | 39.052 | 229 |

可观察到：

- Mixed-4K的cap256在两次运行中均比cap485更低mean JCT，说明已有并发控制确实具有调优空间；但改善幅度不同，没有held-out验证，不能宣称稳定收益。
- Separate-4K的样本内较低JCT配置从rep11的cap256变成rep12的cap485，不能只取两次均值挑“最优”。
- Mixed-1K在cap256/485下都只达到229 active，均无request-limit记录；四组batch模式计数也相同，均为1 extend、366 mixed、128 decode。名义cap变大不等于实际放宽了正在约束执行的门限。相同rep的两档JCT差异方向还会反转，不能解释成state wall收益。
- 按两次均值，Mixed-1K三档cap的JCT均高于同cap的Separate-4K；并非每个单run都如此，例如cap128/rep11有反例。不能由此否定小chunk在其他目标、负载或backend下的价值。

**Observation：cap与chunk改变服务结果，但是否撞到cap、执行模式及时间波动必须同时检查。Insight：不能用“cap更大/更小”或“chunk更小”直接替代对有效执行与释放的判断。** 这仍是已有控制的实验，不是新算法贡献。

只读事后时序审计发现：26组正式数据均无超过1s的相邻CPU batch-launch间隔，校准最大间隔0.694s，宽池最大0.343s。故“出现单次超长batch间隙”未得到当前审计支持；这些间隔也不是GPU kernel时间。GPU高负载采样时SM为1350--1410MHz、最高63°C，不能据此声称不存在CPU、调度、算子或观测开销波动。根因尚未定位，所有数据仍保留。

## 3. 同池干预：一个有限且依赖指标的交互

![物理池控制和同池cap干预](figures/m2_state_cap_mediation.png)

每根柱为两次均值，黑点为单次结果。A/B跨会话并改变辅助内存；只有B/C在各自同一服务会话内固定全部物理池、改变cap。

预先指定：`D(pool, cap) = meanJCT(Mixed-4K) - meanJCT(Separate-4K)`；`Delta = D(wide,485) - D(wide,607)`。

| Rep | D自然485，s | D宽池485，s | D宽池607，s | Delta，s |
|---|---:|---:|---:|---:|
| 11 | 10.737 | 5.526 | 4.915 | +0.612 |
| 12 | 5.590 | 5.628 | 4.847 | +0.781 |
| 两次均值 | 8.164 | 5.577 | 4.881 | +0.696 |

按冻结方向判据，两次正Delta是对“较紧cap放大当前mean JCT配置差距”的探索性支持。差距相对缩小约12.5%，**不是吞吐提高12.5%，也不是已将总损失的12.5%归因于state**。不能把自然池8.164到宽池4.881的变化整体归给cap。

干预有效性成立：四组C全部达到607 active，超过原485；最大KV占比Mixed为45.24%、Separate为48.29%；全部26组均无prefill-NO_TOKEN、retraction或trace drop，完成时active=0。B/C物理池与源码一致，因此没有在放宽state派生cap时转而碰到KV墙的证据。

但这里改变的是**由state容量派生的并发限制**，没有改变KDA kernel、state dtype或实际单请求state大小。无prefix、同大小state、独立池条件下，这条数量约束与普通并发上限等价；结果不是KDA独有性的证明。只有两次启动、cap顺序未完全反平衡，也尚无确认性置信区间。

## 4. 最重要的服务后果：更早激活不等于更早完成

下表均为宽池两次均值；百分比为均值之比，不是配对改善率均值。B/C差值在每个模式的两次运行中方向一致。

| 指标 | Separate B→C | 变化 | Mixed B→C | 变化 |
|---|---|---:|---|---:|
| Mean queue，s | 5.684 → 4.092 | -28.01% | 7.562 → 6.757 | -10.65% |
| Mean TTFT，s | 5.868 → 4.276 | -27.13% | 7.873 → 7.075 | -10.14% |
| Mean TPOT，ms | 83.493 → 105.100 | +25.88% | 111.622 → 121.495 | +8.84% |
| Mean JCT，s | 16.472 → 17.624 | +6.99% | 22.049 → 22.505 | +2.07% |
| P95 JCT，s | 23.138 → 17.852 | -22.85% | 27.321 → 24.379 | -10.77% |
| Makespan，s | 23.666 → 17.975 | -24.05% | 28.173 → 25.178 | -10.63% |
| State-slot-seconds | 6520.260 → 8171.987 | +25.33% | 8782.034 → 9546.346 | +8.70% |

排队定义为client提交到首次state激活，不是纯scheduler backlog；TPOT为每请求`(JCT-TTFT)/(128-1)`再取平均；p95使用nearest-rank。State积分为active ownership而非NVML预分配显存。

用`mean JCT = mean提交至激活 + mean激活至client完成`分解：

- Separate排队减少1.592s，但激活后到完成从10.788s升到13.532s，增加2.744s，最终mean JCT增加1.152s。
- Mixed排队减少0.805s，但激活后到完成从14.487s升到15.748s，增加1.261s，最终mean JCT增加0.456s。

因此，差距缩小的一部分含义是原本更快的Separate退化更多，而非Mixed得到净mean JCT加速。与此同时，尾部请求更早获得准入，p95和整批结束时间都变好。各指标不能相互替代。

交互符号也依赖指标：mean queue的Mixed-minus-Separate差距由1.879s增至2.666s；makespan差距由4.507s增至7.202s，与mean JCT差距缩小相反。不能泛称“state放大所有性能损失”。

**具体问题：只以可行准入数、排队时长或持续参与衡量组批，无法同时反映平均完成、尾延迟、整批效率和state占用后果。Insight：提前把等待请求变成持久state holder，可以减少池外排队，却增加激活后的完成时间和state-time；它是目标相关的权衡，不是天然的优化方向。** 本轮未做kernel profiling，不能把激活后变慢全部归给KDA或batch变大。

## 5. 池布局与内存预算边界

详见[逐字节内存审计](m2_state_mediation_memory_audit_20260916.md)。本模型TP2每rank每request的state主体为21708800 bytes，MLA KV为8064 bytes/token。不是其他模型/拓扑的54MB与27KB数字。

包含state padding1，自然/宽池主state+KV预算分别为22312973952 / 22312969344 bytes/rank，只差4608 bytes；主缓存确实从KV重新分给state。但request索引表随485→728扩大，额外约0.95GiB/rank，故不能称全部HBM相同。

A/B同cap的mean JCT差：rep11 Separate/Mixed为-2.211/-7.422s，rep12则为+0.013/+0.050s。池布局、辅助内存、跨会话时序和校准波动尚未分离；第二次接近不构成正式等效检验，第一次改善也不构成state重分配收益证明。

## 6. 对M2故事线的更新

| 判断 | 当前状态 |
|---|---|
| Fixed state在自然部署中形成请求数瓶颈 | 保持已验证；自然485满而KV峰值约20%--21% |
| 单纯token/chunk控制不能代表有效驻留并发 | 有具体观察；Mixed-1K峰值229且不触发较大cap |
| 同工作集、同物理池下，cap会改变batching的mean JCT差距 | 新增探索性干预支持；两次Delta +0.612/+0.781s |
| 放宽cap减少排队但可能增加mean JCT与state-time | 新增直接观察；两种模式、两次重复同方向 |
| State是原来Mixed损失的主要原因 | 未证明；A/B不稳定，且指标交互方向不同 |
| Hybrid特有、普通attention不可能发生 | 未证明；本轮未做KV-only对照，cap机制并不独有 |
| 已完成强baseline校准并找到其无法解决的机会 | 未证明；部分校准波动，缺完整先进方法和held-out验证 |
| 应进入lifetime predictor或新的公平调度设计 | 尚无依据 |

与M1衔接的当前表述：

> M1说明相同token计量不对应相同fixed-state负担。M2进一步看到，这种固定成本在自然部署中限制驻留请求数；但更多可行并发不是单向收益，提前激活会改变执行完成与state释放节奏。同样的cap干预对mean、tail和整体完成有不同后果。因此必须区分资源可行性和目标相关的服务权衡；是否存在强已有方法之外的hybrid调度缺陷，仍需确认。

这比“state wall导致所有损失”更符合数据，但还不是一条完成的独立论文算法motivation。原slot计数缺陷和局部修复收益仍成立，本轮不改写其历史判定。

## 7. 下一步，先确认而不是设计

**后续优先级更新：** 根据统一池能否解决问题的讨论，先执行[池管理三方对照预检](m2_pool_comparison_protocol_20260916.md)，将完整内存账本和有效请求上限作为新的前置检查。下列校准、反平衡确认与剩余机会验证仍需要完成，但暂不跳过统一池强基线。新[预检分析](m2_pool_comparison_preflight_20260916.md)不计入本报告的26组性能实验，也不改写原结果。

1. **先定位校准波动。** 在预先固定的重复条件下记录host与device时间、负载和trace成本；单独做观测开销/算子profiling，不与无profiler的延迟结果混合。当前没有长launch-gap证据，不能先指定GC或某个kernel为根因。
2. **反平衡确认B/C。** 两个模式都采用明确的485/607与607/485顺序，增加独立会话并预先规划样本量；冻结mean JCT主指标，同时报告tail、makespan、TPOT和state-time。不能仅以缩小模式差距作为收益目标。
3. **再检验hybrid额外约束及剩余机会。** 核对当前模型/backend对已有统一池的支持，匹配全部HBM而非只匹配mem fraction，比较state/KV可行区间；KV-only参考需匹配cap与工作构成，但更换模型不构成仅state一个变量的消融。使用调优后的正确原生方法及机制匹配的先进baseline，再做held-out负载与目标约束验证。

本轮不自动追加模型实验、SOTA复现或lifetime predictor。若上述确认只留下普通cap/batch调优，应将M2定位为部署/目标权衡insight，不强行包装新算法。

## 8. 完整性、恢复与复现

- 正式26/26、15782/15782请求，完整预热26组/15782请求独立排除。全部输出长度和prefill工作精确匹配；零请求错误、prefix reuse、retraction、NO_TOKEN及batch/resource trace drop；每事件一active对应一state，结束active=0。
- 校准/宽池正式数据最大提交迟到7.620/1.100ms，均低于50ms；高负载SM1350--1410MHz。健康检查只排除其明确覆盖的异常，不保证主机和执行速度恒定。
- v3在计时前预建607条只读`/model_info`连接；正式回放均607次连接复用、零新建，客户端自动GC在回放期间暂停，GPU健康查询使用常驻进程。服务端GC不变。与9月15日的旧客户端条件不同，本轮不跨日期直接配对性能。
- 早期三次首组预热到达失败均保留，未进入正式统计；另有GC/冷连接diagnostic及三版功能预检，详见[测量失败与修正记录](m2_state_mediation_recovery_20260916.md)。计时隔离并不意味着已证明所有先前停顿根因。
- 用户中断令`calibrate04`终止。采用已完整结束的四个会话12正式/12预热；中断会话已保存的1正式/2预热共1821请求排除，另有无完整artifact的被中断运行不臆造完成数。余下两会话整会话重跑至`calibrate05_recovered`。旧目录不补写完成标记，见[冻结恢复规则](m2_state_mediation_interruption_20260916.md)。
- 22项原测量源码/协议在inspect、会话启动与结束间一致；恢复额外记录脚本/规则两个哈希及逐文件采用哈希，分析器核验。没有修改SGLang服务源码、GPU时钟/功率或使用GPU0--5。
- 最终回归：包含新增时序审计的132项M2测试，以及10项tracker/policy组件测试，合计142项通过。组件测试隐藏GPU，有17条配置/弃用/无GPU后端警告，无失败。两张PNG/SVG均已生成并目视检查，无裁切或重叠。
- 本轮runner和模型服务全部退出，wide矩阵完成时间为北京时间11:43:16。收尾时GPU6/7已被其他环境的任务占用，其进程启动于13:31，晚于本轮测量结束；没有终止这些任务，也不将当前GPU状态误报为空闲。

主数据：[analysis.json](results/m2_state_mediation_20260916/analysis01/analysis.json)、[26组逐run表](results/m2_state_mediation_20260916/analysis01/report.md)、[时序审计](results/m2_state_mediation_20260916/analysis01/timing_audit.json)。主数据保留每请求进度、资源时间线、源代码与原始artifact哈希；图为汇总展示，不替代逐run数据。

原始请求、batch与GPU轨迹保留于容器`/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_state_mediation_20260916/`，宿主机为`/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/m2_state_mediation_20260916/`。本地同步分析和图表，未将中断数据删掉。

脚本：[原测量](run_m2_state_mediation.py)、[整会话恢复](recover_m2_state_mediation.py)、[分析](analyze_m2_state_mediation.py)、[绘图](plot_m2_state_mediation.py)、[只读时序审计](audit_m2_state_mediation_timing.py)。以下分析命令需在远端项目执行；原输出目录已经存在，复核时应另指定新目录，不能覆盖既有结果。

```bash
.venv/bin/python benchmark/test_motivation/Ledger/analyze_m2_state_mediation.py \
  benchmark/test_motivation/Ledger/results/m2_state_mediation_20260916/calibrate05_recovered \
  --mediation benchmark/test_motivation/Ledger/results/m2_state_mediation_20260916/mediate01 \
  --output benchmark/test_motivation/Ledger/results/m2_state_mediation_20260916/analysis_verify
```
