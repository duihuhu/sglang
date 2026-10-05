# M2：正确统一池之上的准入决策筛选

状态：分阶段预注册；先完成场景校准，再冻结到达时间与正式顺序。不是新算法或SOTA排名。沿用用户任意空闲卡授权，首选空闲GPU0/1；逐卡锁、两次idle检查，只清理自己的进程，不改时钟/功率。

## 问题与范围

固定Kimi-Linear-48B-A3B、TP2、同一个已验证统一池，检查暂缓一次新请求准入、推进已有请求是否存在受约束服务收益。旧slot缺陷保持修正，固定token-VTC input1/output2/request0，无prefix、graphs、spec；不训练/使用lifetime predictor。

本轮只先使用Separate4K，使背景prefill与decode阶段可验证。Mixed、held-out、论文级baseline是筛选后的工作，不把本轮结果称为已完成这些对照。统一池使用上一轮C配置：bootstrap/request607、KV启动参数781725、mem_fraction_static=.85、默认context、原backend/dtype。运行cap607，背景加候选最多576，不以旧485上限人为制造瓶颈。

## G1：背景校准与场景可实现性

- state-dominant：512个P512/O128背景请求。
- kv-dominant：96个P16384/O128背景请求。
- 全部按1ms间隔提交，同一组排序/计量；分组只用于分析。
- 独立背景回放rep41/42，每次完整同形预热排除。分别计算背景请求生成第16、96个token的中位时间，作为far/near候选到达锚点。两个rep锚点均值固定为后续到达时间，不根据服务收益选取。
- 验证对应时刻至少80%背景仍active，至少95%已完成prefill；far的存活背景median remaining >=80，near在[16,48]。背景state或KV主体占比须至少2/3，否则该场景不成立。校准结果失败就停止并保留记录，不直接进入性能筛选。
- G1只用于场景与容量诊断，不混入性能结果。两类背景不匹配计算量，跨类别的绝对延迟不能作纯state归因。

## G2：最小配对筛选

每条轨迹在冻结far/near时刻加入64个P8192/O64候选，1ms间隔。同一rep跨策略完全相同input IDs、输出长度和到达时间。每次测量之前全轨迹完整预热，初次正式rep51/52；独立服务会话顺序和场景顺序反转。

首先比较native与diagnostic-decode32。后者仅在首次发现候选等待、背景仍在running、没有继续chunk时，最多32次返回“不加入新prefill”，让原生调度推进decode；背景全部释放时提前结束。不改变候选顺序、物理池或模型算子，不空转阻挡继续chunk，不重复触发。它是人为诊断参照，不使用未来输出信息，也不是经held-out验证的可部署策略。记录实际触发次数，不能将32次准入扫描直接等同32次GPU forward。

所有策略使用同一个独立launcher，诊断值0时不干预；仅附加轻量计数及idle读回，原SGLang源码不改。记录native MinFreeSlotsDelayer和PrefillDelayer是否真正拒绝准入。候选进一步对照为原生min-free-slots=64和prefill-delayer max-delay-passes=32。若存在受约束收益，必须检查这些已有机制，之后才称存在简单原生控制之外的机会；单个默认配置不是调好的强基线。

## 测量、解释与门槛

主指标mean JCT，整体p95与makespan均最多退化5%。完整报告背景/候选mean、nearest-rank p95、TTFT、排队/激活后时间、state-time、实际live state/KV主体字节、共享raw总字节与准入拒绝原因。分组指标不得隐藏，单一总均值获益不直接证明公平收益。

使用既有v3客户端：预建全部HTTP连接、回放零新建、客户端GC暂停、GPU持续500ms采样、提交迟到<50ms。阶段间隔可能超过默认连接保活时间，客户端keepalive与服务端SGLANG_TIMEOUT_KEEP_ALIVE共同设为1800秒，所有策略一致；这是测量隔离，不改变请求到达时间。逐请求输出长度、prefill工作量、trace完整性与最终资源释放核验。共享池资源不足导致的正常NO_TOKEN等待不是失败；retraction、请求错误或trace丢失必须报告，不能静默排除。

正式同轨迹每格只有两次重复，只作探索性方向和收益筛选，不给精确总体CI。场景必须以正式运行的真实轨迹复核，不沿用校准标签。对更小的请求cap、不同chunk及更强动态控制尚未完成校准前，不宣称新算法必要。

若所有诊断场景均无满足约束的收益，则保留负结果，不扩大搜索寻找赢家。若只有near有效、far无效，说明动作价值依赖执行阶段，但还不是state独有归因。若原生控制覆盖诊断收益，问题成立但已有机制可处理。仅在收益可重复、原生控制后仍存在、真实释放推动后续准入时，再扩大样本和held-out验证。
