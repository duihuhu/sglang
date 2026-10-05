# M2：先排除分池与请求上限配置，再检查剩余服务损失

状态：预检协议；尚未冻结正式性能矩阵。只使用物理 GPU 6/7，其他任务占用期间不启动模型。不修改旧实验、测量协议或服务端算法。

## 1. 要回答的问题

旧实验已经观察到自动配置下 state/request 达到 485，而 KV 尚有余量。但它们关闭了 unified memory。因此不能把静态分池造成的闲置直接归为 hybrid 无法避免的结构瓶颈，也不能据此断言已有内存管理或准入策略有未解决的缺陷。

本轮先问：**在请求上限正确解析、完整 HBM 预算可比时，已有重新分池/统一池是否消除了这批请求的主要服务损失？** 然后才问是否还有与 hybrid state 约束相关、通过准入/batching 决策可避免的损失。不进入 lifetime predictor。

## 2. 四个候选配置

| 条件 | 配置含义 | 用途 |
|---|---|---|
| A：static-natural | 原自动静态池，历史读回 state/request=485 | 原部署基线；新会话重新测量，不复用旧延迟作为配对样本 |
| B：static-rebalanced | 显式 state=607，按相同预算重新划分 KV | 当前已知 607 请求的针对性静态强基线，不称通用最优 |
| C：unified-raised-cap | 共享 state/KV 字节池，bootstrap state=607 | 已有统一池方案；必须实际读回 request cap>=607 |
| D：unified-old-cap-control | 只开统一池，保留 bootstrap state=485 | 负对照：区分共享字节和残留 request cap 的作用 |

607 来自已冻结工作集大小，不由本轮延迟结果选择。旧 728-state 宽池保留历史身份；本轮不自动沿用其额外容量与辅助开销。离线 KV 估算只用来检查预算量级，不能直接作为已解析的服务器容量。

## 3. G0：离线预检

`preflight_m2_pool_comparison.py` 只在 `CUDA_VISIBLE_DEVICES=''` 下运行，读真实模型 config、历史 inspect 元数据及当前源码。复用原生 KimiLinear state 形状、MLA spec、请求上限 resolver，并在 meta device 构建实际统一池视图，不载入权重、不申请 GPU cache。

检查：TP2 每 rank state 与 KV 单位字节；实际上下文长度；已有 Triton backend 的参数 guard；共享 raw storage 的去重计量；静态重新分配后的 KV 余量；A/B/C/D 的离线 request-cap 解析。记录源码、模型配置和输入记录的 SHA256。

**G0 通过不等于 GPU 启动、allocator 动态分配、算子正确性或性能通过。** Meta 上的虚拟最大 slots 不是可同时用满的两个独立池。现有 resolver 会对 unified 继续施加 bootstrap state cap；单独设置 max-running=607 也可能被旧 state485 再次截断。

## 4. G1：GPU 启动、输出正确性与完整内存账本

GPU 6/7 空闲且端口无冲突后，使用已有所有权锁和进程清理规则，逐条件启动。保持 TP2、BF16 KV、FP32 temporal state、原 Triton backend、radix/graphs/spec off、context=1048576、corrected slot gate=1、activation limit=0。不能为了跑通 unified 暗中换 dtype/backend/context。

先做少量确定性输出、分块 prefill、并发释放再分配检查；比较输出 token IDs 或 logprobs/容差，单纯输出长度一致不是语义正确性证据。同配置自重复先作为数值非确定性参考。失败保留记录，不能把不兼容自动解释为调度缺陷。

逐 rank 保存下列信息，不只用 server-info 中 rounded cache GB：

- 启动配置与实际读回：effective request table size、runtime max-running、bootstrap state/KV、实际共享字节总量、reserved floor、padding、allocator maps、compaction 设置。
- 唯一 tensor storage：weights、request-to-token 表、state/KV、allocator bookkeeping、算子 workspace 等；共享 buffer 与各 views 只计一次。列出未覆盖项，不宣称递归对象遍历能找到所有 CUDA 分配。
- Torch allocated/reserved/current/peak、CUDA free/total 与 NVML used，分别在启动 idle、607-request 完整预热后 idle、单独容量压力回放期间测量；标明统计范围与采样时刻。Torch reserved 包含 allocated，不相加；设备 used 与 Torch 差额保留为未归因/外部分配。
- 有效 state/KV 占用轨迹、峰值 resident tokens/holders、拒绝原因、重定位/压缩、retraction、错误和 trace drops；统一池不能沿用“两个独立容量百分比相加”的判据。

请求表约为 `(effective_requests+1)*(context_len+4)*4` bytes/rank；485->607 增加 511707040 bytes，约 488 MiB。单独缩短 context 可节约该项，但本轮不能只对一个条件这样做。

先报告同 `mem-fraction-static=.85` 的部署结果及账本，再以不高于 A 实测可用预算为上界、只根据字节而非延迟调整主 cache，建立完整预算匹配的确认条件。对 B/C 至少保持相同 request table 容量。冻结总预算允许误差、峰值安全余量、最终 KV/state/bootstrap 参数和辅助开销后，才进入 G2。若某项 workspace 随执行变化，报告独立峰值而不是强行宣称所有时刻字节完全相等。

G1 内存审计属于诊断会话；profiler/详尽 census 不放入正式计时路径。新增审计开销应独立验证。不能从同 .85 或近等 state+KV 主体直接下“严格等 HBM”结论。

## 5. G2：同一工作集的配对服务对照

工作集仍为 607 个 P512/O128 请求，1ms 计划到达间隔、无 prefix reuse，沿用已校准 v3 客户端的 607 连接预建、零计时期新建连接、50ms 到达误差限制、客户端 GC 隔离与持续 GPU 监测。逐请求比较实际 payload/arrival fingerprint；只排除 run_id 的规范化方式应提前写清。新增实验使用独立重复，不和旧客户端/旧池会话合并估计 CI。

保持同一 token-VTC 权重 input1/output2/request0，首先比较原生 Separate4K 与 Mixed4K，不把这两个配置称为两篇 SOTA 算法。B/C 在同一服务会话内做 runtime cap485/607 两档，反平衡顺序；A/D 只能测试其实际支持的 cap485。A->B 含辅助分配及池划分变化，不称纯 state 因果干预；B->C 在相同 request table/预算下检查统一池的价值与开销。

样本量、反平衡顺序和预算容差在 G1 后、看正式延迟前冻结。旧 2 次重复不足以预设稳定最优。主指标 mean JCT，联合报告 queue、p95、TTFT/TPOT、makespan、state-time、租户服务差和资源轨迹；预先声明允许的 tail/吞吐代价约束。缩小 Mixed-Separate 差距不是收益，必须看各自绝对服务结果。

607 同形短请求只用于隔离池/并发机制，不足以验证多租户 fairness 或通用 unified 优势。若进入下一阶段，再固定训练/调参边界，用短密集、长稀疏、交替相位等 held-out 请求形态检验；静态池不得逐测试点事后调优再称可部署策略。

## 6. 判据与故事线调整

- 若 B 已消除主要损失：旧现象主要是静态分池配置问题。M2 保留部署与目标权衡 insight，不证明新 scheduler 必需。
- 若 C 比 B 更稳健，且剩余损失可由已有 cap/batching 调优解释：已有统一池是强解决方案；不把 fixed tax 本身重新包装为未解决问题。
- 若 C 仍有损失：先排除残留 request cap、完整 HBM 差异、layout/compaction 开销、计算饱和与测量波动。统一池未加速本身不是新准入策略的证据。
- 只有在正确配置的统一池上，还能用同预算/同工作集、明确目标约束的干预重复得到更好服务，并证实其随 state 压力而变化，才支持更强的 hybrid-aware admission motivation；仍不能据一个模型称 hybrid 独有。

本轮所要形成的 insight 是条件性的：**固定 state tax 改变请求集合的字节需求；静态分区、请求表上限、实际共享内存和执行干扰是不同限制，必须先排除已有机制能解决的部分，再描述剩余问题。**
