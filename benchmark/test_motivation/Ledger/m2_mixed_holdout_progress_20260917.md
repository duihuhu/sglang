# M2：Mixed 筛选准备与容量证据审计

> 后续更新：用户要求继续后已完成Mixed near补充筛选，16组正式8192请求及等量完整预热。原far校准不满足，保留失败计划，未执行原32组矩阵。最新结果、两张图、匹配边界与容量审计见[Mixed近释放窗口报告](m2_mixed_near_results_20260917.md)。下文是恢复前的历史快照，不再代表最新执行状态。

2026-09-17。状态：新性能实验被外部GPU作业阻断；完成旧12组原始轨迹复核、下一轮冻结协议和运行/分析代码、35项CPU回归。没有新增正式性能样本，不将计划中的32组计入已完成实验。

## 1. 本轮推进了什么

- [冻结协议](m2_mixed_holdout_protocol_20260917.md)：Mixed4K，Native607、Cap512、Slots96、Decode32；train512和此前未参与阈值调参的hold384，far/near、两次独立重复，共32组16384正式请求，完整同形预热另计。
- [运行器](run_m2_mixed_holdout.py)：先做背景单独运行校准，再冻结到达计划；同一raw预算、请求表与生产源码；GPU身份、空闲、逐卡锁、源码哈希、请求/trace/连接/迟到/释放检查。
- [分析器](analyze_m2_mixed_holdout.py)：单独披露候选TTFT/JCT代价、真实触发、场景匹配；原生基线只依据训练形态选择，再原样评价held-out，不在held-out上重选最佳。也报告诊断动作相对全部三个原生点。
- [容量审计](audit_m2_capacity_evidence.py)：只读历史原始轨迹并逐文件验证SHA256；不启动GPU模型，不修改历史结果。

Mixed本轮只迁移已校准参数，并非已完成Mixed全参数搜索。若诊断动作显示剩余机会，仍须给予现有cap/chunk/动态控制进一步校准机会；不能把这四个点称为全部强baseline。

## 2. 为什么旧的54%收益不能直接支持容量介导

本模型TP2每rank的实际state主体为21708800 bytes/request，MLA KV为8064 bytes/token；相当于一份state约2692个KV token。共享raw预算为19481072000 bytes。这里不用其他拓扑的54MB/27KB数值。

旧负载是512个P512/O128背景加64个P8192/O64候选。假设所有请求都同时保留完整input+output，逻辑主体上界为：

```
State = (512 + 64) * 21708800 = 12504268800 bytes
KV    = (512 * 640 + 64 * 8256) * 8064 = 6903300096 bytes
Total = 19407568896 bytes
Raw   = 19481072000 bytes
Raw - Total = 73503104 bytes
```

因此，**旧工作集的逻辑缓存主体上界仍比共享raw预算小约73.5MB，单靠该工作集不能保证制造真实容量短缺。** 这是逻辑上界核算，不是精确allocator可分配性证明：未合并reserved sink、页/索引边界、碎片、候选预留、在途状态和workspace。源码中reserved floor以最大entry为基准，不能把多种共享视图的available_size直接相加。

新审计按同一个资源事件计算`state_slots*S + kv_tokens*K`，没有把不同时间的两个峰值相加。复核[历史12组](m2_native_counterexample_results_20260917.md)得到：

| 历史正式条件 | 逻辑state+KV主体峰值，bytes/rank | raw减该峰值 | 拒绝证据 |
|---|---:|---:|---|
| Native607 / far，两轮一致 | 19209388032 | 271683968 | 无prefill_no_token、无request_limit |
| Native607 / near，rep61 | 19386925056 | 94146944 | 无prefill_no_token、无request_limit |
| Native607 / near，rep62 | 19382796288 | 98275712 | 无prefill_no_token、无request_limit |
| Cap512 / far及near | 13757317120 | 5723754880 | 每run一次普通request_limit，无prefill_no_token |
| Slots96 / far及near | 13757317120 | 5723754880 | 空槽延迟生效，无prefill_no_token、无request_limit |

12组原始轨迹SHA256全部匹配、全部无prefill_no_token，结束active均为0。原始`prefill_other`记录保留，不把它改名为内存不足。当前证据更直接支持执行顺序改变；逻辑state-time下降并不证明state空间曾阻塞后续服务，也不代表预分配CUDA驻留内存同比下降。

当前有效汇总：[容量审计02](results/m2_mixed_holdout_20260917/historical_capacity_audit02/audit.json)。audit01保留为初版；02补充大小写拒绝名称检查和审计脚本哈希，历史轨迹与数值不变。

## 3. 下一步三阶段探针应满足什么

这里先做预算可行性分析，**不是新的实测结果或已冻结的第三阶段性能协议**。

在旧512+64请求后新增24个P512/O64探针：总请求600，小于607请求表/原生cap；KV主体单独约7.015GB，小于raw预算；state+KV同时达到完整长度时约20.040GB，超过raw约0.559GB。

| 形态 | 总请求 | KV主体上界，GB | state+KV主体上界，GB | 是否低于607请求上限 |
|---|---:|---:|---:|---|
| 旧512背景+64候选 | 576 | 6.903 | 19.408 | 是 |
| Held-out384背景+64候选 | 448 | 7.828 | 17.554 | 是 |
| 候选三阶段：再加24探针 | 600 | 7.015 | 20.040 | 是 |

GB均为十进制，每rank；统一raw为19.481GB。上界超过预算不代表实际轨迹会达到它，Mixed可能在候选/探针增长前已经释放背景。故下一步不能只画这张预算表就宣称有缺陷，必须逐次核对：

1. 当探针到达时，已有holder仍活跃，且候选prefill确实改变其推进/释放时刻。
2. 请求表/普通运行cap尚有余量，而实际共享字节或可恢复state槽位不足；不能将Cap512自身的拒绝当state证据。
3. 对被拒候选记录新增state和KV需求、输出预留、`rem_total_tokens/cur_rem_tokens/rem_mamba_slots`、chunk/input预算，以及allocator当前gap和可实现compaction后的可用量。
4. 指出哪批holder释放资源，随后哪个探针获准；同时记录计算排队、TTFT和完成时间。
5. 在同请求、同到达、同cap的宽容量负对照中核验该额外阻塞是否消失。扩大池之前须审计完整驻留和运行workspace余量，不能以CUDA OOM作为压力正例。

这可以识别真实的内存压力介导路径。单凭宽容量对照仍不能定量声称固定state的独立贡献；KV-alone字节核算也不是一个真实KV-only模型的性能实验。修改state dtype/padding或共享预算各有数值、算子和容量混杂，需独立说明。

## 4. 执行阻断与恢复边界

详见[恢复记录](m2_mixed_holdout_recovery_20260917.md)。GPU4/5最初通过两次idle检查，但模型权重加载期间外部TP8占卡，首次启动OOM；未发出任何预热或正式请求。保留`calibrate01/manifest.json`和服务日志，没有完成标记。

外部TP8退出后按独立恢复计划尝试一次；新的外部作业已占用GPU，启动前检查拒绝，因此`calibrate02`没有创建或启动子服务。只清理了自己的第一次服务，没有停止其他任务、改时钟或功率。到本记录为止：新校准0/4、新正式0/32、新预热0组。

本地系统Python缺aiohttp，相关实验回归使用远端现有`.venv`，35项通过；不依赖aiohttp的4项预算审计测试本地也通过。它们验证代码/计划，不代替GPU性能或数值正确性验证。失败日志与审计汇总已同步本地；历史大型原始轨迹仍留远端。

继续需要两张在整个实验期间保持空闲的GPU。重新执行必须使用新输出目录和明确恢复记录，先完成Mixed背景校准及32组固定对比；只有看到有效的剩余机会，才进入三阶段压力/释放验证，最后才扩展确认性重复、完整先进策略和生产稳态。
