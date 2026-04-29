核心原因是：Tier 1 ILP 是静态规划，不是逐请求调度。

如果 r 是每个单独请求，会有两个根本问题：

1. ILP 约束数量爆炸

假设系统每秒来 100 个请求，每个请求的 (seq_len, bs) 都不同。如果 r 是单个请求，约束 (5) 就要为每个请求生成一条：

t_PA(r₁) + t_PF(r₁) + t_comm ≤ TTFT_SLO / L
t_PA(r₂) + t_PF(r₂) + t_comm ≤ TTFT_SLO / L
...
t_PA(r₁₀₀) + t_PF(r₁₀₀) + t_comm ≤ TTFT_SLO / L
但 Tier 1 是提前几分钟做一次的规划，它根本不知道未来具体会来哪些请求。它只有负载的统计分布。

2. 单个请求的延迟差异太大，无法用一个约束覆盖

t_PA(r) 取决于 seq_len 和 bs。seq_len=128 和 seq_len=4096 的 Prefill Attention 延迟可能差 10 倍。如果用最大请求做约束，过度保守（为极端情况配置资源，浪费能耗）；如果用平均请求，又覆盖不了长尾。

所以用分桶 + P90 代表值：

做法是把历史/预测负载按 seq_len 分成几个桶（比如 4 个 Prefill 桶：[128, 512, 1024, 4096]），每个桶取 P90 的 (seq_len, bs) 作为代表值 r。这样：

4 个桶 → 只有 4 条约束 (5)，ILP 规模可控
P90 覆盖了 90% 的请求，比均值保守但不像最大值那么浪费
每个 r 代表的是一类负载的典型最坏情况，不是某个具体请求
打个比方：r 类似于"这条公路上 90% 的卡车都不超过 20 吨"这个统计值，而不是"这辆编号 #12345 的卡车重 18.7 吨"。Tier 1 用它来决定桥的承重设计，而不是逐辆车检查。

总结：r 是负载桶的 P90 代表值，因为 Tier 1 是离线静态规划，需要用少量统计代表值来约束未来一段时间内的所有请求，而不是逐请求优化（那是 Tier 2 的事）。



计算方式：

第一步：对历史/预测负载按请求特征分桶

Prefill：按 input_len 分桶，比如 4 个桶：
[0, 512)、[512, 2048)、[2048, 8192)、[8192, +∞)
Decode：按 (input_len, output_len) 联合分桶（因为 Decode 的 KV cache 大小同时取决于 input 和已生成的 output）
第二步：每个桶内取 P90 代表值

对桶内所有请求的关键特征（seq_len、bs）取 P90 分位数，得到一个代表性的 (seq_len, bs) 组合。这就是 r。

比如 Prefill 桶 [512, 2048) 里有 1000 个历史请求，P90 的 seq_len 是 1536，P90 的 bs（同时在处理的 batch size）是 8，那这个桶的 r = (seq_len=1536, bs=8)。

第三步：用 r 查 Profile 表得到延迟

t_PA(r) 和 t_PF(r) 不是公式算的，是从离线 Profile 表查的：

Profile 表: (tp, freq, seq_len, bs) → (latency_us, energy_mj)
把 r 的 (seq_len, bs) 加上候选的 (tp, freq) 去查表，就得到了约束 (5)(6) 里的具体延迟值。

第四步：生成 ILP 约束

4 个 Prefill 桶 → 4 个 r → 4 条约束 (5)。ILP 求解时，选出的配置必须同时满足所有桶的 SLO 约束。

数据来源是 WorkloadMonitor（滑动窗口统计），它持续收集请求的长度分布，输出 WorkloadStats（包含 λ、N_active、长度分布），Tier 1 Solver 的 workload_bins 参数就是从这里来的。

简单说：历史请求 → 按长度分桶 → 桶内取 P90 → 查 Profile 表 → 变成 ILP 约束里的具体数字。



Decode 分桶确实需要 output_len，但不一定需要一个"预测长度模型"。

有两个层面要区分：

Tier 1 的 Decode 分桶：不需要精确预测
Tier 1 分桶用的是统计分布，不是逐请求预测。数据来源是：

历史 trace 统计：Azure LLM trace 等生产数据本身就有 (input_len, output_len) 的联合分布。Tier 1 只需要知道"大概有多少比例的请求落在哪个桶"，用来生成 P90 代表值。
max_tokens 上界：很多 API 请求自带 max_tokens 参数，这给了 output_len 一个确定的上界。
所以 Tier 1 的 r 可以直接从历史分布统计得到，不需要逐请求预测。

但 Decode 有一个更微妙的问题
Decode 的 output_len 影响的是 KV cache 大小，进而影响 Attention 延迟。Profile 表里 t_DA(r) 的查询键是 (tp, freq, input_len, output_len, bs)，其中 output_len 代表"当前 KV cache 里已经有多少 token"。

对于 Tier 1 的 SLO 约束，取 P90 的 output_len 意味着：保证 90% 的 Decode 步骤都能满足 SLO。生成过程早期（KV cache 小）天然更快，只有接近 P90 output_len 时延迟才接近约束上界。

相关工作怎么做的
DynamoLLM：用 BERT 预测器把请求分成 9 类（SS/SM/SL/.../LL），准确率 81%，误分类走紧急回退
throttLL'eM：用生成长度预测器估计 
∣
r
i
∣
^
∣r 
i
​
 ∣
^
​
 ，驱动 Scoreboard 做 KV/batch 投影，预测误差做保守放大
这两个系统需要逐请求预测，是因为它们在 Tier 2 级别做逐请求的准入控制和频率决策。

对 AFlex 的影响
Tier 1 不需要预测模型——用历史分布的 P90 就够了。

Tier 2 也不需要——因为 Tier 2 的 Decode DVFS 用的是实际观测到的 batch 特征（当前 bs、当前 KV cache 大小），不是预测值。它是在每个 iteration 开始前，根据已知的当前状态选频率，不需要预测未来。

如果未来要做更精细的优化（比如提前预留 KV cache 容量、做准入控制），那确实需要一个生成长度预测模型。但当前 AFlex 的两层设计巧妙地回避了这个需求：Tier 1 用统计分布，Tier 2 用实时观测。