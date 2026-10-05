# M2：固定目标租户的首轮服务机会观测

实验于2026-09-17完成，2026-09-18整理。**功能预检完成；出现邻居形态敏感性的描述性观察，尚未完成F1独立重复、公平约束核验或F2 hybrid因果归因。** 本文不修改历史实验gate，也不引入新调度策略。

关联：[M2问题定义](m2_tenant_service_opportunity_isolation.md)、[执行前协议](m2_service_preflight_protocol_20260917.md)、[只读分析汇总](results/m2_tenant_service_20260917/preflight01-analysis/summary.json)。

## 1. 这次实际完成了什么

旧数据缺少counter、准入和waiting过程。这次新增独立、默认关闭的[实验观测器](m2_service_observer.py)，没有改动生产调度源码；观测开关之外，保持同一token-VTC、corrected slot gate和原生Mixed batching。

- Kimi-Linear-48B-A3B，A800 GPU4/5，TP2；Mixed4096、max-prefill16384，radix/graph/spec关闭。
- 统一池raw预算19,481,072,000 B/rank，请求表/runtime cap607，KV参数781725；没有人为缩cap或state padding。
- 目标A分别为packed或split，邻居B分别为无、packed、split，共六条件。每角色两wave，每wave16384输入+512输出，间隔1.5s；A先到，B延后20ms。packed为1×P16384/O512，split为16×P1024/O32。
- 每条件在新观测off/on两侧各运行一次，每次先完整同形态warmup：12次预检、12次warmup，共680请求。另有8个独立输出正确性请求。
- A的payload/输出预算/到达fingerprint在邻居与观测开关之间一致；B的规范输入token序列及总输出预算一致，仅改变partition。拆分后计算形态与语义并不等价。

31项本地/远端观测与分析测试、10项远端既有VTC/resource tracker测试通过。六个on预检共170请求、12,114条事件；请求收费/处理量、chunk复用、batch匹配、资源激活/释放守恒全部通过，零drop/截断。最大客户端提交迟到3.82ms，小于事先50ms门槛；boot ID/monotonic时钟可对齐。

P512/P8192、O16的独立贪心正确性检查中，off/on实际输出token IDs相同，各自两次重复也一致。**这不是六个混跑工作集逐token相同的证明。** 六条件off/on makespan比为0.997--1.024；只有一次会话/开关，不能据此认定稳定开销小于2.4%。

两会话均有custom allreduce JIT初始化失败，服务继续运行；结果对应该实际通信环境，不能称为最优backend。完成记录确认本轮进程停止、两卡均回到1MiB/0%利用率；之后其他任务的GPU占用不属于本轮。

## 2. 固定同一个租户，改变邻居后看到什么

以下均为observer-on单次功能预检。slowdown使用同一A payload的独跑作分母；packed仅2个请求，其p95尤其不能作为稳定尾分位估计。

| 目标A | 邻居B | A p95 JCT | A相对独跑 | A输出50%时刻 |
|---|---|---:|---:|---:|
| packed | 无 | 31.061s | 1.000× | 17.046s |
| packed | packed | 33.276s | 1.071× | 19.303s |
| packed | split | 31.751s | 1.022× | 17.630s |
| split | 无 | 2.694s | 1.000× | 2.357s |
| split | packed | 5.059s | 1.877× | 4.345s |
| split | split | 3.596s | 1.335× | 3.029s |

![M2固定目标预检：服务差异、容量与记账执行轨迹](figures/m2_service_preflight_20260918.png)

图a/b固定相同短请求A，图c/d取A-split+B-packed，显示最初6秒；完整曲线保存在JSON中。completed work是CPU result路径观察到的prefill处理量及server output，不是GPU kernel时间或客户端交付。活跃缓存字节代理按state21,708,800 B/slot、KV8064 B/token计算，取同一时刻的和；不是完整HBM用量。

**观察1：相同目标、相同邻居总token，服务表现仍随邻居partition变化。** 本轮A-split在长邻居下比短邻居下p95高约40.7%；这是F1需要重复确认的现象，不是统计显著结论。它比旧实验更严格地固定了目标租户，但尚未交换先后顺序或标签。

**观察2：更高固定state成本，不等于更强的实际干扰。** A-split+B-packed的活跃state峰值34 slots、活跃cache主体峰值约占统一池6.51%；A-split+B-split分别为64 slots、9.86%，但前者A更慢。B-packed自身state峰值仅2 slots，B-split为32；反过来，B-packed的KV-token-seconds约1,009,264，B-split约85,390。它们揭示资源维度、持有时间和执行形态不同，**并不证明KV-time单独造成延迟**。

这不否定M1的固定state tax，而是限定其用法：M1是资源成本观察，不能直接拿来给租户的性能干扰排序。M2必须检查这些成本通过什么调度/执行路径转化为他人的服务变化。

## 3. 新过程证据说明了什么

### 3.1 Counter不是“已经执行/交付的token”

本地VTC首次准入时记整个input费用，但长输入随后分多个chunk执行。图c/d直接显示记账曲线领先completed work；继续chunk没有重复input收费，也没有重复建立新的request holder。

此外，counter包含重新进入waiting队列时的lift。这里所谓“空闲返回”只表示该租户此前没有waiting请求，不保证它没有running请求；本地实现的这一队列状态语义必须保留。每角色实际收费最终均为：

`32768 input + 2 * 1024 output = 34816`。

但A-packed+B-packed的最终counter分别是51194/51200，其中lift为16378/16384；A-split+B-packed的最终counter是34816/35840，其中lift为0/1024。**相同总token并不意味着完整counter轨迹、甚至最终counter相同。** 本次每次counter变化均能由input/output charge或lift解释，不是发现记账错误。

由此得到的测量insight是：需要区分虚拟调度位置、真实token收费、已处理工作及客户端交付。把counter gap直接叫作服务隔离差距，会混合不同对象。预记账和lift是通用机制，不是本轮证明的hybrid特有缺陷。

### 3.2 有共同未完成需求，不一定有共同waiting积压

| 混跑条件 | 双方server outstanding时间 | 双方waiting时间 | 最长连续双方waiting |
|---|---:|---:|---:|
| A-packed+B-packed | 33.740s | 0.816s | 0.816s |
| A-packed+B-split | 5.453s | 0.066s | 0.066s |
| A-split+B-packed | 5.433s | 0.348s | 0.347s |
| A-split+B-split | 4.451s | 1.976s | 1.022s |

outstanding从实际插入队列到服务端完成计数；waiting来自实际队列事件，不由客户端未完成数量推测。大部分共同未完成区间中，至少一方已经没有waiting请求，仍有running/chunk工作。尤其不能拿A-packed+B-split的5.45秒全部当作双方持续等待的公平队列窗口。

本轮足以证明必须区分这些状态，**不足以提供可直接用于正式公平判断的共同长窗口**。正式协议还需明确公平定义中的backlogged条件，不能把本表某个时长自行当作VTC定理适用条件。

### 3.3 停止扫描不是容量拒绝

六个on预检一共66次scan-stop，全部是`prefill_other`；其中47次候选已经成功selected后停止，19次候选未selected。没有`request_limit`或`prefill_no_token`记录，也没有retraction。活跃主体峰值最高约占raw pool9.86%。

现有观测记录的是实际方法返回和budget快照，不是`OTHER`的逐内部分支标签；不能进一步把19次未selected全部命名为某一个具体budget原因。资源释放同样是生命周期观察点，不是allocator调用的精确原因。

因此，本轮**没有支持“state容量堵塞导致这些延迟差异”**；但也没有证明hybrid机制毫无影响，或全部差异只来自compute。state算子计算、KV访问、预记账粒度、Mixed执行形态等仍需分离。

## 4. 对M2问题陈述的推进

| 状态 | 当前可以说什么 |
|---|---|
| 已验证的测量能力 | 目标保持一致；计量、候选、准入、工作、输出与资源可连接；收费/释放守恒；停止扫描与拒绝可区分 |
| 新的描述性观察 | 固定短目标在不同等token邻居下服务不同；更高state占用不是更差隔离的单调预测；counter与执行有时间差及lift差异 |
| 尚未验证 | 形态效应在独立重复/镜像下稳定；满足同一公平约束；固定state/KV通过特定约束额外放大干扰；现有控制不能修复 |

更准确的M2候选问题是：**token计量只决定候选的虚拟调度位置，不能单独描述这些服务机会何时兑现，以及兑现过程中形成的持久state/KV占用如何影响其他租户。** 当前结果完成了观测链和一个低容量压力下的形态对照，不等于这个候选问题已经构成hybrid特有缺陷。

## 5. 下一步具体顺序

1. **积压与观测开销预检。** 使用预先给定、目标不随邻居改变的更长固定到达轨迹，只按双方实际积压窗口、资源轨迹完整性选择可用负载，不按收益挑选。开/关顺序平衡、多会话检查开销；明确并核对实际通信backend。保留本轮作为低容量压力对照。
2. **冻结正式F1。** 明确服务口径和窗口规则，再固定独立重复、先后顺序/标签镜像、短/长目标、六条件及全部租户绝对指标。目标是确认邻居形态效应及其发生阶段，不先比较新算法。
3. **F2只干预已定位的路径。** 将计量粒度/lift、实际组批推进与自然触发的state/KV冲突分开。若没有空间拒绝，就不能把池/容量介导当成已知前提；若有，则需同普通cap、同KV预算、完整HBM/workspace核对下检查state约束的额外作用。
4. **最后才做F3与设计。** 检验简单成本校准、计量时机或原生控制在同一公平口径下是否已经足够；必要时再讨论新机制。不进入lifetime predictor，不恢复单租户mean优化作为M2主结论。

## 6. 可复核产物

- 原始回放、完整事件、输出正确性与日志：远端容器`/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_tenant_service_20260917/preflight01/`，宿主对应`/mnt/nvme0/mllm/sglang/...`。
- 本地已同步[manifest](results/m2_tenant_service_20260917/preflight01/manifest.json)、[完成记录](results/m2_tenant_service_20260917/preflight01/completed.json)、[12条件审计及曲线](results/m2_tenant_service_20260917/preflight01-analysis/summary.json)、图与本文；原始大事件文件留在远端。
- [运行器](run_m2_service_preflight.py)、[只读审计](audit_m2_service_preflight.py)、[分析器](analyze_m2_service_preflight.py)、[绘图脚本](plot_m2_service_preflight.py)。汇总为每个原始预检文件保留SHA256；源代码hash冻结在manifest/completed中。
- 本地汇总SHA256：`b288983ee4d753a595f8fec86aea7dc9ae699ca964895ed3a372c7b6cc84947a`。
