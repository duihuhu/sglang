# M2：统一池解决了什么，仍没有解决什么

2026-09-16。Kimi-Linear-48B-A3B，A800-SXM4-80GB，物理GPU0/1，TP2，无prefix reuse。用户已允许使用任意空闲卡；没有停止其他任务、修改时钟/功率或修改服务端实现。

**结论：对于这批607个短请求，原485请求容量瓶颈可以通过已有池管理和配置消除。重新划分静态池、统一池配合提高实际请求上限，都达到607 active；只开启统一池仍停在485。容量扩大改善排队、p95和整批完成时间，却增加平均JCT和state-time。因此，“state满、KV空”不能继续作为已有池管理无法处理的缺陷证据；留下的是容量可行性与服务目标之间的权衡，尚不是hybrid独有调度缺陷或新算法必要性的证明。**

正式24组、14568请求全部完成；24组完整预热、14568请求另外排除。每个条件两个独立服务会话，第二轮反向顺序。以下性能数字仅来自正式数据，不混入诊断、旧GPU6/7实验或预热。两个重复不足以给出可靠总体置信区间。

## 1. 对照与完整内存预算

沿用[三方对照协议](m2_pool_comparison_protocol_20260916.md)，增加[真实GPU执行协议](m2_pool_comparison_live_protocol_20260916.md)。配置先按完整驻留显存校准，再冻结[正式计划](results/m2_pool_comparison_20260916/measure-plan01.json)，不按延迟选择预算。

| 条件 | 管理方式 | Bootstrap state / 实际request容量 | 启动KV参数 | 正式runtime cap |
|---|---|---:|---:|---|
| A，原485-slot布局 | 静态分池 | 485 / 485 | 1187753 | 485 |
| D，只开统一池 | 共享字节池，保留旧请求上限 | 485 / 485 | 1174858 | 485 |
| B，重划静态池 | 增加state、减少KV | 607 / 607 | 791088 | 485、607 |
| C，统一池并提高上限 | 共享字节池，扩大请求表与有效上限 | 607 / 607 | 781725 | 485、607 |

A保留原485-slot布局，但为共同预算和工作空间余量下调了KV，**不是未经调整的默认配置**；初始默认KV为1458643。统一池表中的state/KV是启动配置，不是两个可以同时填满的独立硬池容量，不能把虚拟视图重复计入HBM。

本模型TP2每rank每request state主体21708800 bytes，MLA KV为8064 bytes/token。默认context为1048576，请求索引表从485扩大到607还增加约488MiB/rank。只匹配主cache或`mem-fraction-static`会漏掉这部分。

逐rank审计Torch allocated/reserved/peak、CUDA used/free，并对Python可见GPU tensor storage去重；未归因分配保留为residual。审计hook仅用于独立诊断启动，正式进程不加载它。

| 条件 | Idle Torch allocated，GiB/rank | Idle CUDA used，GiB/rank | 诊断峰值allocated，GiB |
|---|---|---|---:|
| A | 66.587 / 66.587 | 67.898 / 67.898 | 70.691 |
| D | 66.587 / 66.546 | 67.925 / 67.871 | 70.699 |
| B | 66.551 / 66.551 | 68.005 / 68.005 | 71.682 |
| C | 66.581 / 66.541 | 68.043 / 68.003 | 71.724 |

最终跨条件、跨rank的allocated极差47.344MiB，小于冻结64MiB门槛；CUDA used极差176MiB，小于256MiB门槛。初次校准差约83MiB未通过，按字节减少A/D的KV后重新启动验证，没有放宽这两个门槛。**匹配的是idle完整驻留预算，运行时workspace峰值不是严格相等**；峰值及分配重试仍须单独披露。

共同条件：607个P512/O128请求，1ms计划到达间隔，0.606s到达窗口；radix/graphs/spec关闭，相同backend和dtype；本地token-VTC input1/output2/request0，corrected slot gate=1，额外activation limit=0。Separate4K将prefill与decode分开组批，Mixed4K允许同批执行；两者都是SGLang原生配置，不是两篇SOTA算法的完整复现。

每个rep跨池、模式及cap使用同一payload fingerprint。B/C在同一个物理池会话内比较runtime cap485/607，两轮采用相反cap顺序。每次正式回放之前有独立完整607请求预热。主目标预先定为mean JCT，且p95、makespan退化均不得超过5%才称有约束净收益；其他指标的收益另行保留。

## 2. Observation 1：原容量瓶颈可被已有配置消除

A和D的所有正式运行峰值均为485；B/C在runtime cap607时均达到607，runtime cap485时均回到485。没有因扩大state转而出现prefill-NO_TOKEN或retraction，全部请求结束后active归零。

这与源码一致：`KVCacheConfigurator.resolve_max_num_reqs`仍将有效请求上限限制为`max_mamba_cache_size / ratio`，统一池开关没有绕过这条启动约束。仅设置更大的`max-running-requests`而保留旧state bootstrap也不能绕过该上限，离线原生单元测试已覆盖。

**具体问题：共享HBM字节并不自动消除独立请求上限。Insight：判断统一池是否有效，必须同时检查可借用字节、请求表大小与实际生效的准入上限。** 这是一项配置/实现边界，不等于统一池原理上不能解决分区浪费。此工作负载也没有证明动态统一池优于重新划分的静态池。

## 3. Observation 2：容量、平均完成与尾延迟不是同一个目标

![等驻留显存预算的池管理与请求上限对照](figures/m2_pool_comparison.png)

柱为两轮均值，黑点为每轮观测，不是置信区间。下表单位为秒，均为两次均值；p95使用nearest-rank。Queue定义为client提交至首次state激活。

| 模式 | 条件 / cap | Mean JCT | Mean queue | P95 JCT | Makespan |
|---|---|---:|---:|---:|---:|
| Separate4K | A / 485 | 16.474 | 5.684 | 23.145 | 23.674 |
| Separate4K | D / 485 | 16.960 | 5.837 | 23.817 | 24.346 |
| Separate4K | B / 485 | 16.601 | 5.711 | 23.413 | 23.941 |
| Separate4K | B / 607 | 17.666 | 4.100 | 17.889 | 18.029 |
| Separate4K | C / 485 | 16.956 | 5.844 | 23.719 | 24.247 |
| Separate4K | C / 607 | 18.157 | 4.200 | 18.382 | 18.510 |
| Mixed4K | A / 485 | 22.151 | 7.593 | 27.350 | 28.200 |
| Mixed4K | D / 485 | 23.018 | 7.807 | 28.325 | 29.184 |
| Mixed4K | B / 485 | 22.189 | 7.613 | 27.336 | 28.189 |
| Mixed4K | B / 607 | 22.455 | 6.765 | 24.344 | 25.151 |
| Mixed4K | C / 485 | 22.947 | 7.786 | 28.297 | 29.160 |
| Mixed4K | C / 607 | 23.252 | 6.907 | 25.314 | 26.127 |

相对A485，B607的Separate/Mixed平均JCT分别增加7.23%/1.37%，p95降低22.71%/10.99%；C607的平均JCT增加10.21%/4.97%，p95降低20.58%/7.44%。两轮每个方向均一致，但不是统计显著性结论。

更干净的干预是**同池只改runtime cap485→607**，排除请求表与物理池布局变化。下列百分比为逐rep变化率的均值，正号表示数值增加，不沿用分析JSON“正号表示改善”的约定。

| 同池cap干预 | Mean queue | Mean JCT | P95 JCT | Makespan | State-time |
|---|---:|---:|---:|---:|---:|
| B，Separate4K | -28.21% | +6.42% | -23.59% | -24.70% | +24.39% |
| B，Mixed4K | -11.14% | +1.20% | -10.94% | -10.78% | +7.66% |
| C，Separate4K | -28.14% | +7.09% | -22.50% | -23.66% | +25.51% |
| C，Mixed4K | -11.29% | +1.33% | -10.54% | -10.40% | +7.82% |

两轮均出现相同的目标权衡。以统一池Separate为例，排队5.844→4.200s，平均JCT却16.956→18.157s；提交至激活节省约1.645s，激活至client完成增加约2.846s，抵消了排队收益。State-time是实际holder数量的时间积分，不是NVML预分配显存。

**具体问题：以“有空间就尽量加请求”或“排队减少”代替端到端目标，会忽略激活后的执行时间与state持有时间。Insight：统一池改变资源可行域，但不会自动选择mean、tail和整批完成时间之间的取舍。** 该权衡也可能出现在普通attention系统；本实验未把它归因为KDA独有机制，亦未定位kernel级原因。

在预先冻结的mean-JCT主目标下，上述扩容方向均未通过净收益判据；这不表示没有性能收益，tail和makespan的改善是真实观测，只是付出了mean和state-time代价。

## 4. Observation 3：本负载没有显示统一池优于重划静态池

在相同607 cap下，C相对B的平均JCT增加：Separate +2.78%，Mixed +3.55%；两轮分别为[+2.75%, +2.82%]、[+3.29%, +3.81%]。在485 cap下，同样没有观察到统一池更低的mean JCT。

这只能称为本地实现、当前工作集下的端到端差异，不能直接归因于allocator本身，也不能推广为“统一池无用”。全体请求同形，静态池已经按该负载重划；尚未测试短/长构成变化时统一池避免重新划分的价值、碎片化/compaction代价或其他backend。

**Insight：要证明新调度机会，基线应是预算与请求上限均正确配置后的静态/统一池，而不是一个可以通过现有配置修复的485-slot部署。**

## 5. 对M2逻辑线的修订

| 命题 | 当前证据 |
|---|---|
| Hybrid固定state tax使请求数成为独立容量因素 | 保持成立，不因统一池而消失 |
| 原静态485配置会出现state/request满而KV尚宽裕 | 保持已验证，但它是具体布局/上限的结果 |
| 现有池管理无法消除这批607请求的容量瓶颈 | 不成立；B/C已达到607 |
| 只打开统一池就会自动提高有效请求上限 | 不成立；D仍为485 |
| 解除容量瓶颈就会降低平均JCT或state-time | 不成立；同池cap干预方向相反 |
| 存在可避免的尾部排队和整批完成时间损失 | 在这批负载下得到既有配置改善的证据，但有其他指标代价 |
| 已证明hybrid特有损失、强baseline外收益或新算法必要性 | 尚未证明 |

推荐当前故事线：

> M1表明相同token计量不对应相同fixed-state负担。M2进一步表明，这种负担必须进入容量和请求上限配置；已有重划分/统一池可以修复观察到的容量不足，但更早激活并非单向收益。即使固定同一个统一池，扩大准入仍会减少排队和尾延迟、增加平均完成与state-time。因此需要区分“能否放下”和“在指定服务目标下是否该放”，但后者是否有超出现有调优的hybrid特有机会仍待验证。

此前继续chunk重复计数的具体slot缺陷及局部修复收益不被本轮推翻；M1/M3的公平实验也不由这批单组同形请求重新判定。不进入lifetime predictor设计。

后续若继续验证motivation，应先冻结目标和调优基线，测试不同请求形态/到达负载的held-out情况；再在池预算与cap匹配下定位state相关执行开销，区分普通batch并发效应；最后才检验超出静态配置、原生动态控制及适配先进方法的剩余收益。统一池在构成变化下的优势目前是待检验问题，不是本轮负结果可以否定的命题。

## 6. 完整性、正确性与边界

- 正式24/24、请求14568/14568；完整预热另24/14568。分析器核对冻结16会话矩阵、24组配对fingerprint、27项源码/协议哈希、启动与结束的有效容量，以及完整trace。零请求错误、prefix reuse、retraction、prefill-NO_TOKEN和trace drop；结束active=0。
- 正式回放每组复用607条HTTP连接、零新建；最大提交迟到1.126ms，低于50ms门槛。客户端GC仅在回放时暂停，服务端GC不变。GPU高负载SM1350--1410MHz，最高59°C；采样NVML显存峰值81060MiB，并非运行时显存严格相同或完全没有内存压力的证据。
- 串行正确性：10个有效诊断条件中，每个P512/P8192、O16各重复两次，共40请求，greedy token IDs一致。独立并发检查覆盖A/B/C各8请求，共24请求，混合P512/P1024/P8192、O16/O64，两轮之间不flush，跨池token IDs逐项一致。并发检查使用首次预算版本；此后只减少A/D的KV，最终A/D又通过串行复核。未声称验证全部607并发输出的逐token等价或最坏compaction路径。
- 初始4个有效容量诊断、预算4个诊断、微调2个诊断，共10×607请求及等量预热，均不进入性能统计。`live_inspect01`第三次启动在模型加载前因TP可用显存不平衡退出；没有关闭安全检查，失败目录不补写完成标记，独立恢复计划补完其余条件。外部根因未确定。正式`measure01`一次完整结束，无截取/拼接恢复。
- 16个正式服务会话均有custom-all-reduce JIT失败并回退的启动警告，共32条；没有修复或切换通信实现。因此不是所有通信backend的最优SGLang结果。Mixed会话另有48条可恢复显存分配重试警告，包含预热和正式运行；Separate为0。请求均成功，但不能宣称没有allocation pressure，详见[判据澄清](m2_pool_budget_gate_revision_20260916.md)。
- 共享宿主机非独占，其他卡上存在作业，也运行过CPU侧测试；GPU健康监测不能排除CPU/NUMA/系统干扰。每格仅两次独立会话，报告配对点估计与观测范围，不将607个请求当作607个系统重复，不给总体精确CI。没有held-out、多模型、prefix、稳态SLO goodput或多租户fairness验证。
- 工具回归33项通过，16条弃用/无GPU后端警告；Python编译检查通过。PNG/SVG已生成并目视核对，无裁切或文字重叠。运行器和自己的模型服务已退出；收尾确认GPU0/1各1MiB、0%利用率，其他GPU作业保持运行。

## 7. 证据与复现

本地已同步[正式汇总JSON](results/m2_pool_comparison_20260916/measure_analysis01/analysis.json)、[全部对比表](results/m2_pool_comparison_20260916/measure_analysis01/report.md)、[冻结计划与内存证据](results/m2_pool_comparison_20260916/measure-plan01.json)、[完成标记](results/m2_pool_comparison_20260916/measure01/completed.json)和[并发正确性结果](results/m2_pool_comparison_20260916/concurrent_correctness01/completed.json)。原始request/batch/resource/GPU轨迹与日志保留远端，摘要含逐artifact SHA256。

远端容器原始目录：`/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_pool_comparison_20260916/`；宿主机同一目录为`/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/m2_pool_comparison_20260916/`。本地`measure01`仅同步manifest/completed，不是完整原始数据镜像。

脚本：[真实GPU运行](run_m2_pool_comparison.py)、[独立内存审计](launch_m2_pool_audit_server.py)、[按字节校准与冻结](prepare_m2_pool_budget.py)、[并发输出检查](check_m2_pool_concurrent_outputs.py)、[完整性与配对分析](analyze_m2_pool_comparison.py)、[绘图](plot_m2_pool_comparison.py)。在远端项目内复核时使用新输出目录，不覆盖本轮结果：

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python \
  benchmark/test_motivation/Ledger/analyze_m2_pool_comparison.py \
  benchmark/test_motivation/Ledger/results/m2_pool_comparison_20260916/measure01 \
  --out benchmark/test_motivation/Ledger/results/m2_pool_comparison_20260916/measure_analysis_verify
```
