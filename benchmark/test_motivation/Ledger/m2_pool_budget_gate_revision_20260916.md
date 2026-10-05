# 正式测量前的内存 gate 澄清

预算确认诊断仍出现 PyTorch caching allocator 的 `memory allocation failed with OOM` 警告，但对应607请求预热/回放全部成功、服务未退出。这类CUDA分配重试可通过回收缓存恢复，不等于Python最终抛出OOM或请求失败。

因此，在任何正式性能样本产生前，移除规划器中“日志中必须零分配重试”的额外门槛。该门槛不是原协议的请求正确性要求，也不适合作为完整显存是否匹配的判据。保留原有分配器配置，不为某个池单独启用 expandable segments、empty-cache 或其他优化。所有条件的重试日志均保留并计数，运行时分配器开销是尚需独立归因的因素。

不变的门槛：零请求错误/长度错误/重算/trace drop、正确输出检查、有效cap读回、逐rank显存账本；steady idle allocated 的跨条件差异不超过64MiB、CUDA used不超过256MiB。不同batch产生的运行时峰值单独报告，不宣称每个时刻的全部字节相等。若请求失败、启动失败或显存预算不匹配，仍不进入正式测量。

共同额外2GiB余量保留；KV仍至少能容纳工作集最大resident tokens的两倍。该修订在预算确认阶段完成，未根据正式JCT胜负选择条件，诊断延迟不进入正式统计。
