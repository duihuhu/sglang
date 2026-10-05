# M2：先排除原生并发控制解释，再谈state因果归因

状态：执行前冻结，2026-09-17。只做State-far/near的原生机制校准，不是完整参数搜索、held-out或state容量因果实验。

## 源码审查与可识别性

本配置无prefix、无spec、无retraction，每个active request恰好对应一个state holder，既有完整性检查逐事件核对该等式。因此逻辑约束`active_states <= L`与`active_requests <= L`等价；单独加入逻辑state-slot上限不能识别不同于普通并发限制的机制。

`KVCacheConfigurator.resolve_max_num_reqs`还按`max_mamba_cache_size / ratio`裁剪请求上限。`init_unified_mamba_pools`将启动KV字节与state字节相加形成共享raw预算。因此直接改变启动state参数会同时改变请求上限和物理池，不是纯state干预。

统一池实际准入已经包含`mamba_slot_full_token_cost`的共享字节预留及`rem_mamba_slots`检查，不能声称它不感知固定state。真正的联合约束近似为`S*N_active + K*T_resident <= B`，并受独立请求表/运行cap限制。当前问题是资源可行性之上的执行选择，而不是遗漏state内存安全检查。

## 本轮问题

此前State-near的诊断收益，能否由不改变state大小、计算路径或共享池的原生阈值/普通请求上限获得？若可以，说明该组收益不必依赖新的hybrid资源抽象；不等于排除其他场景中的state放大作用。

比较三项：Native607；Slots96（原生min-free-slots=96、runtime cap607）；Cap512（无delayer、runtime cap512）。共享raw固定19481072000 bytes/rank，启动state/request607、KV参数781725；Separate4K、token-VTC input1/output2/request0，修正slot门限，关闭prefix/radix/graphs/spec，单租户。

背景仍为512个P512/O128，候选64个P8192/O64，组内1ms到达。复用已冻结State-far/near到达锚点，不按本轮性能调时刻。请求种子rep61/62，均为新实验；不将旧rep51/52作为新配对样本。此前诊断Decode32只能作历史量级参照。

## 配对与顺序

- rep61：先无delayer会话，far/near各按Native607、Cap512顺序；再Slots96会话，far/near。
- rep62：先Slots96会话，near/far；再无delayer会话，near/far各按Cap512、Native607顺序。
- 每项正式回放前完整同形预热。共4个独立服务会话、12组正式6912请求、12组预热6912请求。
- 相同rep/phase跨策略payload fingerprint必须一致。Native607/Cap512在同一物理池实例内交错，只改变runtime cap；Slots96需要不同启动会话。
- State-near/far继续逐次核对实际背景存活、prefill完成、remaining和state字节占比，不把冻结时间等同严格相同执行状态。

## 测量与判定

使用原HTTP预连接、双端1800s保活、客户端GC暂停、500ms GPU采样、零回放新建连接、提交迟到<50ms、逐请求/trace/释放核验。物理GPU沿用校准GPU4/5，逐卡锁及两次idle检查；只清理自己的服务。源码、协议、输入校准计划哈希在执行前后核对。

主目标与旧筛选一致：mean JCT下降且整体p95 JCT、makespan退化各不超过5%；候选mean/p95 TTFT和JCT必须单列，不能把该门槛写成TTFT保证。另记录state-time、peak active、真实delayer次数、request-limit/NO_TOKEN、候选加入与已有请求释放的执行轨迹。

两次独立会话只作探索性筛选，不给总体精确CI。若scene或完整性失败，保留原始数据并停止；本轮不自动重试，不按结果选择性补跑。只有完整矩阵才能报告完成，预热和失败会话不进入正式统计。

本轮有意先限定State两场景，KV、Mixed、生产graphs与held-out均未覆盖。96和512由607-512=95空槽、512背景数量推导，不是未经负载知识选择的通用参数。即使收益覆盖，也只是校准能力，不是通用最优策略。

## 后续state干预需要什么

仅改变逻辑slot上限不能识别新机制。独立state归因需要至少固定模型/请求、独立请求上限与KV需求，审计共享字节和每请求真实state占用，并观测state持有导致真实可用空间不足、请求等待、释放后准入的链路。

在共享池中只减少总缓存预算，识别的是共享内存压力，不自动识别state特有性。以state dtype改变占用也同时改变数值精度/潜在kernel行为，需输出正确性及宽容量计算开销负对照。任何模拟state padding/扣账只属于人工干预，不能称为自然部署缺陷。本轮不实施这些未充分校验的干预，不将普通cap结果改名为state消融。
