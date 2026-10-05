# M2 下一阶段：修正基线后的首次激活节奏

状态：协议与实验设施准备阶段，尚无本轮 GPU 性能数据。此前 slot 修正的 100 组实验不计入本轮结果。

## 1. 顺序与要回答的问题

1. 所有对照启用 reuse-aware slot 门限，验证原重复计数见证为零。
2. 在每种 batching 内配对比较默认准入和首次激活节奏限制，检查是否存在剩余机会。
3. 若有机会，再在自然容量、state 压力和纯并发上限对照中定位来源。
4. 校准并复现更强的文献级 batching；不能把实现 bug 留给 baseline。

本轮只执行前两项的功能预检和探索筛选；不预先宣称需要新算法、lifetime predictor 或 hybrid 专属机制。

## 2. 单因素干预

共同基线：Kimi-Linear-48B-A3B、TP2、物理 GPU 6/7、token-VTC(input=1, output=2, request=0)、radix off、相同 chunk/token/capacity budget、正确 slot 计数。

`tenant_new_activation_limit=0` 关闭干预；`1` 或 `4` 表示每个 prefill 组批轮次最多接纳这么多个没有 request slot 的候选。继续 chunk 不占这个额度；候选顺序仍由原 VTC 实现决定。达到额度后停止本轮扫描，不跳过队首挑选后面的新请求，不修改 `batch_is_full`，下一轮重新尝试。额度至少允许一个新请求，避免空系统因人为门限永远不能启动。

这不是活跃并发上限，也不是“等到已有请求完成再接纳”，更不保证 separate 模式会给 decode 更多执行机会。实际效果要从执行轨迹判断；若这种干预无效，不能推出所有暂缓策略无效。

干预记录为 `activation_pacing`，与 request/KV/state 拒绝分开。触发时尚未调用候选的 `PrefillAdder.add_one_req`，因此该事件不证明候选已经通过全部预算，也不能叫“错误拒绝”。事件次数不是受影响请求数或累计延迟。

所有开关仅在显式启用 tenant 实验且 scheduler idle 时可更新；默认关闭。验证范围为 monolithic、PP=1、无投机/抢占/会话复用，不将功能测试外推至其他路径。

## 3. Batching 与阶段

原生四个配置：separate-4096、mixed-4096、mixed-1024、unchunked。前两个 mixed 参数属于同一 batching 家族。现有 mixed-1024 不能被省略；不把它称为完整 Sarathi-Serve 或 FairBatching。

源码审计还确认了已有相邻机制：`PrefillAdder.add_one_req` 的 `prefill_max_requests` 限制本轮 prefill 参与数，包含已加入的继续 chunk；`MinFreeSlotsDelayer` 在存在 running 请求且空位不足阈值时暂缓新 prefill，scheduler 在继续 chunk 路径绕过该 delayer。`PrefillDelayer` 还包含 slot/queue 与等待上限控制。因此不能声称“现有系统没有主动暂缓或请求数量控制”。首轮关闭这些可选项以隔离单因素；筛选出机会后必须加入这些简单原生控制，再决定文献级移植是否有价值。

| 阶段 | 配置 | 请求 | 重复 | 结论权限 |
|---|---|---|---|---|
| smoke | state8 / mixed-1024；额度 0/1/4；两种顺序 | 每轮 16 short + 4 long，1 wave | 1 | 仅功能、回收、trace、时钟检查 |
| screen | state8 / 四个 batching；额度 0/1/4；两种顺序 | 每 wave 16 short(P512) + 4 long(P8192)，4 waves，间隔 2s，O64 | 2；各配置独立重启 2 次，每次 1 对 | 48 runs / 3840 requests，探索性剩余机会，不作正式 PASS |
| attribution | 自然池容量、wide-cap8、state8；增加负载及 O256 等输出长度 | 筛选后冻结下一阶段矩阵 | 待定 | 区分持续 state、普通并发和计算干扰 |
| confirmation | 正确且校准的强 baseline、CUDA graphs、较大持续到达负载 | 独立 held-out trace | 待定 | 后续预注册，不能复用筛选数据作独立确认 |

warmup 覆盖本会话所有额度/到达顺序，排除性能分析。会话内随机化 A/B 顺序，跨会话轮换 batching 顺序。相同 rep、请求形状、顺序的 token IDs、输出长度和预设到达完全配对；不同顺序只改到达，不改内容。记录实际提交迟到，默认超过 50ms 则标记无效而非静默纳入。

短/长组工作量不同，不称等工作租户。额度不变更 VTC 代码不代表瞬时服务公平相同。当前 trace 提供 VTC counter gap 和两组分布，尚未记录足以精确复原双方 scheduler 队列均 backlogged 窗口的完整事件；筛选只能提示公平代价，不能确认公平改善。正式公平结论需要补齐队列事件。O64/O256 是 workload 因素，各自比较同一 trace；不能把它们直接当成纯 lifetime 因果控制，控制器不读取输出长度预测。

## 4. 指标和判读

主探索指标：所有请求的平均 JCT。并列报告短/长平均 JCT、nearest-rank p95 TTFT/JCT、最大 JCT、request-average TPOT、makespan、两组排队/活跃时间、state-slot-seconds、KV-token-seconds，以及 VTC max counter gap。后者包含 lift，且不限定双方 backlogged，不能当成公平保证的违反程度。

分位数不再默认使用此前小 cohort 的线性插值。任何吞吐/最终完成时间收益须有对应指标证据；有限 wave 的完成率不是 steady-state goodput。SLO goodput 留到独立校准并冻结按请求类别的 TTFT/TPOT 阈值之后，不根据干预结果挑选阈值。

诊断轨迹包括每次激活/释放、组批形状与 launch 间隔、实际 resource 拒绝和主动 pacing。CPU launch 间隔不是 GPU kernel 时间。state pool 宽窄对照会改变资源配额，额外显存不是算法收益。

分析报告仅给探索性差异与全部对照，不自动选择赢家或宣布 M2 PASS。后续需同时满足：干预确实触发、资源安全不破坏、收益可重复、没有未说明的长请求/公平代价，并且相对调优后的现有 batching 仍有空间。若仅一方变快另一方变慢，只证明权衡。

## 5. 三种可能结论

- 修 bug 并调好原生 batching 后无剩余证据：M2 暂收敛到已证实的实现问题，不等于证明所有策略最优。
- 剩余机会在 cap-only 中也出现：先按一般性 admission/batching 问题解释，不能称 hybrid 独有。
- 剩余机会随 state 压力变化，且轨迹对应激活、持续占用与释放：才进一步验证 hybrid 放大效应及动态 batching 的剩余空间。

## 6. 执行边界

启动前检查 GPU 6/7 与端口，仅停止 runner 自己创建的进程组。卡被其他任务占用即拒绝启动，不恢复时钟、不调整功率、不换其他 GPU。原始 artifacts 保留在远端共享目录，分析报告同步本地。

2026-09-10 本轮开始时 GPU 6 有 PID 3981208 的其他 SGLang scheduler，占用约 60GB；GPU 7 空闲。尚未启动实验服务。
