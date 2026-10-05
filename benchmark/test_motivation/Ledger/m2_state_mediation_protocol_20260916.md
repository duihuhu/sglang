# M2：固定 607 请求的 batching 校准与 state-cap 中介检验

2026-09-16，正式数据采集前冻结。v3：三次首组预热到达时序失败及独立客户端诊断后，增加客户端测量隔离及HTTP预连接；此前零正式run，v1/v2源码与协议保存在远端source_v1/source_v2。沿用 2026-09-15 进度观测，但不改变服务源码或实现新调度策略。

## 1. 目标与可证伪对照

不能用 256 -> 607 的请求量变化证明 hybrid 放大。全轮固定每次 607 个 P512/O128 请求、1ms 计划到达间隔；同一 rep 的 payload、输出与计划到达在所有条件下完全配对。新正式 rep=11/12，不与此前 rep1/2 直接作配对性能比较。

先校准普通 batching，再检验 state 派生 cap 的作用：

- A：自然 state 池，历史 readback 为 state485 / KV1458643 / request485；正确 slot 门限。原生 Separate-4K、Mixed-4K、Mixed-1K，各扫 runtime cap128/256/485；两次独立启动，共 18 runs。
- B：更宽 state728，仍 cap485；KV 由同一 mem-fraction-static=0.85 预算自动解析，不能额外增加总显存比例。
- C：与 B 同一个服务进程、相同物理 state/KV/request 池，只将 runtime cap 放至607。B/C 都比较 Separate-4K 与 Mixed-4K，两次独立启动，共 8 runs。

728 由历史自然池 (485+1) 乘1.5后减 padding1 得到。先用 inspect 阶段核验自然池及宽池，每池三个功能请求。不调整 dtype、权重、模型结构、kernel backend、拓扑、graph 模式或 KV 量化。宽池不要求只增加内存：原生 allocator 从 KV 预算中扣除多出的 state；实际 token 容量必须 >= 2 * 607 * (512+128) = 776960。若达不到则停止此协议，不按结果另选容量。

本轮最多 26 正式 runs / 15782 请求；每个正式 cap 对照前用同样607请求、独立负rep输入作完整预热，26 warmup runs / 15782 请求全部排除；inspect6请求排除。模式在第一次启动按固定seed打乱，第二次反序；cap在会话内按固定seed打乱。每个正式run前独立flush并重置共同VTC。预热也通过全部完整性检查。

## 2. 共同条件

Kimi-Linear-48B-A3B，TP2，物理GPU6/7；radix off，prefill/decode CUDA graphs off，spec off；mem-fraction-static0.85，max-prefill-tokens16384；统一池关闭。共同token-VTC input1/output2/request0，二次项0；corrected slot gate1，new-activation limit0。原生延迟及prefill-max-requests控制关闭。

自然池与宽池均不指定 max-total-tokens/max-running-requests。只有宽池显式指定 max-mamba-cache-size728；runtime cap 经已有空闲时接口调整。保留原始 server info 与 resolved capacities。B/C 的物理容量必须逐字段相同；每次启动的容量与 inspect reference 完全一致。

GPU及端口启动前检查；不停止其他任务、不修改时钟或功率。测量源码/协议在inspect、每次启动与结束时核对哈希。保留远端旧helper，不能用本地新增GPU参数版本覆盖冻结远端测量版本。

## 3. 主要结果与中介判据

主结果为 mean JCT，始终同时报告mean queue/TTFT/TPOT、makespan、state-slot-seconds、peak active/KV、request-limit/NO_TOKEN分支及进度代理量。时间线不是GPU kernel profiling。只有两次独立启动，不做确认性CI、最佳参数泛化或公平性保证声称。

对每个rep计算 batching gap：

`D(pool, cap) = JCT(Mixed-4K, pool, cap) - JCT(Separate-4K, pool, cap)`。

1. 校准：展示 A 全部9配置而非只挑赢家；cap/chunk改善可能只是普通batching调优。
2. 池布局控制：比较 A(cap485) 与 B(cap485)，检查相同并发上限下更换物理池划分是否已改变服务。A/B不同会话，漂移和池布局影响不能冒充cap作用；两次近似相等也不构成正式等效检验。
3. 中介干预：B/C在同一物理池中只改cap，报告各模式的 `JCT(B)-JCT(C)`，以及 `Delta = D(wide,485)-D(wide,607)`。Delta为正且两次方向一致才是“较紧cap放大当前batching差距”的探索性支持；为负则约束收紧反而压低差距。没有预设必须正向。
4. 有效性：cap607需要实际peak>485，否则没有放开正在约束执行的上限；KV不能转为binding/retraction，且结束active=0。若这些不满足，只报告干预未形成预期机制，不把负结果当成否定state作用。

状态约束在同大小state、独立池、无prefix条件下退化为并发上限；B/C检验的是这条中介路径，不是KDA专属kernel效应。即使Delta为正，也不能证明普通attention模型不可能存在此现象。Hybrid独有性需进一步比较相同总字节预算下的可行区间及匹配cap/工作构成的控制；不通过直接替换模型来强行单因素归因。

## 4. 完整性与失败处理

每run：607请求全部成功、输出长度精确一致、prefix/retraction=0、batch/resource trace无drop、prefill工作总和及prefix progress完整、一个active对应一个state、结束active=0；实际提交迟到不超过50ms；GPU健康与此前规则一致。

所有失败目录/日志保留，不自动重试或删掉慢run。如果运行中断，停止当前矩阵并先记录原因；后续补跑须另立来源及整会话选择规则，不能静默续跑后伪造原矩阵完成。Inspect字段读取错误可在发送正式请求前修正并重新冻结，明确保留失败预检。

v2客户端隔离：在计时原点建立前启动常驻`nvidia-smi --loop-ms=500`并收到完整双卡样本，避免计时区间中由客户端反复创建查询子进程；回放前主动GC，回放期间仅暂停客户端自动GC，结束后恢复。服务端GC不变。保留全部请求提交时间及相同50ms门限，健康字段与原健康规则不变；不删异常样本。独立诊断数据永不进入M2性能统计，不能因关闭GC后的两次较好结果就声称所有停顿均由GC引起。

v3追加HTTP预连接：在时间原点之前，并发调用与请求数相同的只读`/model_info`，不调用可能触发生成的health接口，不产生模型请求；完整读取响应以保留keep-alive连接。记录aiohttp的公开connection-create/reuse事件，计时回放中新建连接数必须为0，否则此run无效。这样将首次建立607条TCP连接的客户端开销从准入工作负载中分离，不改请求数、时间计划或50ms门限。

v3需重新inspect、再完整运行calibrate/mediate，禁止混用v1/v2/诊断数据。既有A/B/C、模式、cap、rep、请求数和预热规则均不改。若仍失败，继续诊断，不能放宽门限凑矩阵。

本轮到此为止，不自动追加 lifetime predictor、cap搜索或SOTA复现。后续确认实验依据完整结果另行冻结。
