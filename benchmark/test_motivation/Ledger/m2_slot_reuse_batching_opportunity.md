# M2：Batch formation 混淆执行参与和新增激活，存在可避免的准入损失

> 2026-09-10。延续 [M2 的 E1/E2/E3](m2_hybrid_admission_factors.md)，将一个已定位的问题推进到最小修正与 GPU 对照。
> 验证目标是特定条件下的具体缺陷和收益空间，不是证明现有 batching 普遍不适用，也不预设需要新算法。
> 已完成 100 组正式运行、2000 个请求，另有 20 组预热排除。结论：具体准入缺陷与局部服务收益成立，收益具有明确 batching/到达条件；不推出普遍不适用。
> 后续更新：[修正基线后的首次激活节奏筛选](m2_activation_pacing_results_20260910.md)，48 组/3840 请求、两张图；未发现固定 per-pass 限额的额外收益，不与本报告的 slot 修复收益混用。

## 1. 与 M1 的递进

M1 的问题是计量：相同 token 服务量可以对应不同固定 state 负担，短请求可能有更高的单位 token 资源成本。M2 的问题是动作：即使上层公平排序不变，把候选加入 batch 时，是否正确区分了本轮工作与新增资源需求？

长请求的后续 chunk 仍需参与本轮计算并增加 KV，但已经持有 request slot 和 recurrent state。短请求的首 chunk 则需要建立新的 holder。因此不能把所有 batch 参与者当成同等数量的新增 holder。

这里不是主张一律优先短请求或长请求，而是检查已有门限是否拒绝了本来可行的候选。

## 2. 实验、问题、Insight

| 层次 | 实验与观察 | 得出的具体问题 / Insight |
|---|---|---|
| 资源动作不同 | E1：相同 16K input，分成 1/4/16/32 个请求，peak state 为 1/4/16/32；单请求后续 chunk 增长 KV 而 state 保持 1 | token 工作量、batch 参与者数量、新增 holder 数量不是同一个量 |
| 实际门限错配 | E2 轨迹与实际 allocator 复核：继续 chunk 被计入 free request-slot 门限，但 allocator 复用其已有 slot | 在紧请求池下，安全检查可以过度保守，提前停止准入；不是 OOM 或容量安全失效 |
| 可避免的准入损失 | 同模型、同 VTC、同请求、同 batch budget，只修正 slot 计数；检查提前拒绝和首次释放前的激活数 | 若原先被拒绝的请求确实可以提前运行且资源约束不破坏，就证明存在局部可避免损失 |
| 服务收益及边界 | 对三类 batching、两种到达顺序、紧/宽请求池做配对 GPU 验证，报告短/长延迟和 makespan | 准入动作的改善是否成为端到端收益，依赖后续 batch 执行与释放轨迹；不能仅凭多放一个请求推定更快 |

E3 的持续占用观察用于解释后果：错过一次准入机会后，请求可能要等待已有 holder 释放。但本轮不使用输出长度预测，也不由此直接推出 lifetime predictor 必需。

## 3. 一个具体反例

原始轨迹：running=2，selected=5，其中 1 个继续 chunk、4 个新请求；request pool free=5，runtime cap=8。

旧门限在 `selected >= min(cap - running, free)` 时拒绝下一个候选，于是 `5 >= min(6, 5)`，拒绝。但下一个候选是新请求，加入后：

```text
本轮总参与：2 running + 1 continuation + 5 new = 8 <= runtime cap
新增 slot：5 new = 5 <= free request slots
```

实际 `ReqToTokenPool.alloc` 接受 continuation + 5 new，只有 5 次新分配。修正分别判断参与数与新增 slot 数，其余 token/chunk、KV、state 和并发保护不变。

本地和远端基准提交均为 `0077f84d37a013213fb59700564e1637987a3f79`（2026-08-18）。直接检查该提交中的 scheduler 与 allocator，也存在上述两种不一致的计数语义，并非新加的 tenant 观测代码创造了旧门限。结论针对这一被测代码版本，不推断其他版本是否已有不同实现。

2026-09-10 另行读取上游 main，[scheduler](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/managers/scheduler.py) 仍以全部 selected 数量比较 allocatable request 门限；[allocator](https://raw.githubusercontent.com/sgl-project/sglang/main/python/sglang/srt/mem_cache/memory_pool.py) 仍仅为非复用请求分配新行。上游已增加 beam 等分支，这是非 beam 路径的源码对应观察，不是对上游最新版本做了完整 GPU 复现；本报告所有性能数值仍只属于上述被测提交及其记录的修改。

**问题：把继续执行和首次激活一起计入新增 slot 门限，会产生提前拒绝。**

**Insight：batch formation 应在动作层区分 execution participation 与 incremental allocation；存在独立安全检查，不代表各检查使用了相容的资源语义。**

这是在 hybrid serving 中触发的请求池问题，非 KDA 独有。Hybrid 固定 state 容量可经有效并发和请求池大小使该条件出现；默认部署中的频率仍需测量。这个反例可以承担 motivation，不必先证明所有先进方法失败；单点修复也不等于已完成一篇论文的算法贡献。

### 3.1 理论边界与负对照的对应

令 R 为本轮已选择且复用 slot 的请求数，F 为当前 free request slots，H 为 `runtime_cap - running`。假设 R <= H、候选均为新请求且其他 token/KV/state 预算尚未截断，只比较数量门限允许的新请求数：

```text
旧门限：x_old = max(0, min(H, F) - R)
新门限：x_new = max(0, min(H - R, F))
0 <= x_new - x_old <= R
```

在反例中 H=6、F=5、R=1，因此 x_old=4、x_new=5。R=0 时两者完全相同，对应 unchunked 负对照；F>=H 时也相同，对应宽请求池控制。中间差值只是数量门限留下的准入机会，还要通过后续预算、实际 allocator 与服务执行验证，不能把这个上界直接当成性能收益。

## 4. 到底比较哪些 batching 策略

| 对照 | 真正改变的 batching 机制 | 本轮配置 | 身份 |
|---|---|---|---|
| Unchunked prefill-first | 整段 prefill，prefill/decode 分开执行 | chunk=-1 | SGLang 原生，消除继续 chunk 的负对照 |
| Chunked prefill-first | 切块 prefill，prefill/decode 分开执行 | chunk=4096 | SGLang 原生，最初观察到问题的配置 |
| Mixed prefill/decode | running decode 与 prefill 在同一批执行，预算保留 decode 占用 | chunk=4096、1024 | SGLang 原生；两个参数点是同一类，不是两种算法 |
| Wide-pool control | batching 同第二行，放宽物理池但维持 cap=8 | state128/KV131072 | 区分 free-slot 门限与纯 runtime-cap 约束 |

每个配置比较原门限和 reuse-aware 门限，固定 token-VTC（input=1、output=2、request tax=0）。**VTC 是共同的上层排序，不是这组实验的 batching 主对照。** 之前 E2 的 VTC/instant-share 与 E3 的 FCFS 是诊断性排序对照，不能拿来充当多个先进 batching 策略。

完整 Sarathi-Serve、FairBatching、LPRS/APC 尚未移植和复现；不把原生 mixed 开关包装成这些完整算法。后续更强 baseline 的作用是确定已有动态 batching 能否消除该机会，以及是否还有论文级设计空间，不是否定已发现的具体缺陷。

后续文献级对照应当落在 batching 层，而不是再增加一种 VTC cost：

| 已有方法 | 真正需要复现的机制 | 与本轮原生开关的区别 |
|---|---|---|
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | chunked prefill 与不暂停已有 decode 的 stall-free schedule | 原生 mixed 只覆盖相关机制，不能代表完整调度、预算校准与流水线实现 |
| [FairBatching](https://arxiv.org/html/2510.14392v1) | slack 驱动时间预算；urgent decode、prefill、non-urgent decode 的分阶段组批 | 不是固定 chunk 大小；需按当前模型校准批时间并执行其 SLO 规则 |
| [LPRS/APC](https://arxiv.org/html/2606.09061v1) | latency-aware scheduling 与 active-prefill 控制 | 已有请求数、context、memory 相关特征，不能预设它是 token-only 或必然遗漏本问题 |

这些算法的目标和决策权限不完全相同。比较时需要保留正确的 state/KV 容量支持，说明 SLO、排序目标与可用信息，并报告实际服务差；不能把 slot bug 留给 baseline 后将其损失宣称为新算法优势。

## 5. 测量口径

[冻结协议](m2_slot_batching_protocol_20260910.md)与[执行脚本](run_m2_slot_batching.py)：Kimi-Linear-48B-A3B、TP2、物理 GPU 6/7；关闭 radix 和 CUDA graphs。16 个 P512/O64 短请求 + 4 个 P8192/O64 长请求，short-first/long-first 两种到达顺序。各配对使用相同 token IDs、输出长度和到达偏移。

修正和 GPU 证据针对本次 monolithic 服务路径；未验证 PP、P/D disaggregation、投机执行、prefix/session 复用或抢占配置。P/D prefill 路径的额外请求池门限本轮没有修改，不声称所有路径均已覆盖。

5 个配置 x 2 个门限 x 2 个顺序 x 5 次新重复 = 100 组正式运行、2000 请求。每个配置预热两门限与两顺序；正式重复中随机化门限/顺序，同一配置的门限 A/B 在同一服务器会话内比较。跨 batching 配置的绝对性能仍有会话混杂。

预先指定的局部收益见证是 state8/prefill4096/short-first：短 p95 改善至少 5%，配对 CI 下界大于 0；长 p95 和 makespan 的退化 CI 上界不超过 5%。其余组合全部报告，但作为边界/探索性比较。只有 5 次重复，CI 不包含重启和长期负载变化。

两组请求总工作不同；只能说同一组在 A/B 中完成工作相同，不能称为等工作租户。固定 VTC 实现不等于实现了完全相同的瞬时服务差，也不据此声称改善公平保证。

另一个必须保留的口径是小样本分位数：每 run 只有 16 个短请求，当前线性插值 p95 为排序后第 15 个 JCT 的 75% 加第 16 个的 25%。修正若把最后一轮的短请求从 2 个降到 1 个，会明显降低这个指标，即使最慢请求没有变快。按 nearest-rank 定义，这组的 p95 等于最大值。因此正式结果同时给出短请求均值、最大值和 ECDF；不把线性插值 p95 的相对改善直接外推为生产 p95 收益。这是对预设指标的补充解释，不改变原判据或删除原结果。

## 6. 实验状态

- 源码边界测试通过；不复用时与旧门限等价，复用时只计算新 slot，runtime cap 仍计入所有参与者。
- `smoke01` 共 80 请求，功能检查通过但满负载 GPU 频率异常为 210MHz，计时不用于性能结论；全部保留。
- 用户授权后，仅恢复 GPU 6/7 默认 GPU 时钟，未调整功率上限或其他卡。`smoke02` 负载频率为 1395--1410MHz，80 请求完整性通过。
- 预检 short-first 的首次释放前激活从 7 变 8、slot 提前拒绝见证从 1 变 0。预检短 p95 12.373s -> 8.573s；long-first 12.010s -> 12.031s。这些是单次 pilot，不作为正式收益估计。
- 每个新 run 记录约 0.5s 间隔 GPU 负载/时钟；正式运行共 2333 个采样时点，utilization>=80% 时观测 SM 为 1245--1410MHz，无持续低频 guard 失败。不能因此假定硬件完全无漂移。
- 100/100 正式 artifacts、2000/2000 请求成功，输出长度精确匹配，零 prefix hit/retraction/trace drop；所有 run 结束后 active=0，所有事件满足 one-state-per-active，peak active 不超过 8。实际提交最大迟到 7.77ms，配对的是预设到达偏移而非完全相同的实际提交时刻。
- 12 个测量源码文件的起止 SHA-256 一致，未在运行中修改测量代码；35 项实验/分析回归及 10 项 tracker/policy 组件回归通过。后者在正式测量结束后为 serving venv 补齐 pytest 再执行，未改动模型运行依赖。本次实验于 11:59:35（UTC+8）结束，服务关闭后检查 GPU 6/7 均为 1MiB、0% utilization。最终检查又发现 12:10:16 启动的新 SGLang 进程占用两卡；本次 runner 已退出，未停止这些后启动的进程。

## 7. 已完成的主对照：从缺陷到服务收益

state8 / separate4096 / short-first 的 5 对新重复已经完成，达到预先指定的局部收益判据。此处只比较同一服务器、相同 workload、相同 VTC 和相同预算下的门限修正，不将不同服务器组间差异视为因果收益。

![准入轨迹与服务后果](figures/m2_slot_reuse_mechanism.png)

| 指标 | 原门限 | Reuse-aware 门限 | 解释 |
|---|---:|---:|---|
| 每 run 重复 slot 计数导致的提前拒绝 | 1 | 0 | 去掉的是错误计数，而不是去掉真实容量保护 |
| 首次释放前的 active holder | 7 | 8 | 6 short + 1 long -> 7 short + 1 long |
| Short mean JCT | 6.874s | 6.299s | 改善 8.37%，95% CI [7.29%, 9.57%] |
| Short 线性插值 p95 JCT | 12.311s | 8.738s | 改善 29.01%，95% CI [28.04%, 30.03%] |
| Short max JCT | 12.312s | 12.257s | 未显示显著最大延迟改善 |
| Long p95 JCT | 12.274s | 12.221s | 改善 0.42%，95% CI [-0.96%, 1.85%] |
| Makespan | 12.326s | 12.272s | 改善 0.42%，95% CI [-0.95%, 1.84%] |

所有数值为 run-level 均值；相对变化按每对 `1 - corrected/original` 计算后平均，不是直接用表中均值相除。长 p95 与 makespan 的退化区间均未超过预设 5% 上限，但不能声称吞吐或最终完成时间已显著提升。

第一预设重复 rep20 的轨迹解释了收益来源：原门限的短请求按 6、8、2 个分布在三轮完成中；修正后变为 7、8、1 个。两者仍有最后一轮慢请求，所以不是将整个 workload 少执行了一轮。更准确的描述是**减少落入最后一轮的请求，并改善均值及该有限 cohort 的插值尾指标**。

![小 cohort 分位数与均值、最大值](figures/m2_slot_reuse_distribution.png)

相同配置换成 long-first 后，short p95 只有 +0.06% [-0.35%, 0.66%]，且没有重复 slot 拒绝见证。这个顺序对照符合“具体触发条件”的解释，不否定 short-first 中已验证的缺陷和收益。

### 7.1 其他 batching 与负对照

下表为各配置内门限修正的 short p95 相对变化，全部为 5 对重复及 95% CI。这里比较的是同一 slot 干预在不同 batching 下的后果，不是完整 SOTA 算法排名。

| Batching / 控制 | Short-first | Long-first | Slot 提前拒绝见证，原门限 -> 修正 |
|---|---:|---:|---|
| Separate 4K / state8 | +29.01% [28.04%, 30.03%] | +0.06% [-0.35%, 0.66%] | 每 run 为 1 -> 0 / 0 -> 0 |
| Mixed 4K / state8 | +2.05% [1.45%, 2.68%] | +0.68% [0.46%, 0.93%] | 1 -> 0 / 1 -> 0 |
| Mixed 1K / state8 | +0.05% [-1.17%, 1.25%] | +0.71% [-0.65%, 1.91%] | 10 -> 0 / 3 -> 0 |
| Unchunked / state8 | -0.97% [-2.13%, 0.19%] | -0.03% [-0.59%, 0.58%] | 均 0 -> 0 |
| Separate 4K / wide + cap8 | -0.01% [-0.66%, 0.62%] | +0.41% [0.04%, 0.75%] | 均 0 -> 0 |

![所有 batching 对照及两种提交顺序](figures/m2_slot_reuse_controls.png)

**观察一：门限问题不会等量转化为性能损失。** Mixed 1K 中重复拒绝次数最多，但没有确认短 p95 收益；次数也不等于被影响的独立请求数或累计等待时间。两种 mixed 模式的 short-first 在修正前后，首次释放前均累计激活 7 short + 1 long；不像 separate 4K 那样从 7 个 holder 变成 8 个。应结合拒绝发生位置与后续组批解释，不能用次数直接预测收益。

**观察二：去掉触发条件，结构性反例消失。** 不切块或放宽请求池后没有重复 slot 拒绝，且没有主对照中量级的改善。宽池 long-first 虽有 +0.41% 且该会话 CI 不跨 0，但两门限在这一控制下允许相同的新分配数量，不能将亚百分点时延差异自动归因于 slot 机制。跨重启/时间漂移的重复仍是必要补充。

**Insight：可行准入机会的价值由 batching 执行和 holder 释放共同决定。** 需要同时检查“是否错误排除了可行候选”和“错过这个候选会改变哪一段后续执行”。前者是已定位缺陷；后者解释为何相同修正的收益在 separate、mixed 和不同 chunk 下不同。

## 8. 当前结论的边界

当前已经有“具体门限语义不一致”的源码与真实轨迹证据，也有固定 VTC 下的正式局部服务收益及 batching/宽池负对照。M2 可以写成：**混合模型中的 batch formation 不仅安排 token 工作，还管理持续 holder 的新增与复用；混淆二者会错误排除可行准入机会，其服务损失取决于后续 batching 与释放轨迹。**

已验证的是上述局部问题、最小修正和收益条件。尚需验证：自然形成的 state/request-pool 压力、更大请求集合与多种长短比例、启用 CUDA graph、跨服务器重启，以及正确修正并校准后的动态 batching baseline。没有证明新的公平计费、lifetime predictor、联合控制或资源向量调度必需，也没有证明 steady-state SLO goodput 提升。

M2 不再采用“必须证明先进方法普遍不适用才成立”的判据。应按以下顺序积累证据：具体问题存在 -> 最小干预消除问题 -> 在明确条件下有服务收益 -> 确定现有更强方法与生产配置中的剩余空间。前三项足以形成一个有边界的 motivation，最后一项决定后续设计和论文贡献的范围。

## 9. 数据与复现

- [结构化正式分析](results/m2_slot_batching_20260910/analysis/analysis.json)、[完整统计表](results/m2_slot_batching_20260910/analysis/report.md)、[源码/allocator 见证](results/m2_slot_batching_20260910/slot_correction_witness.json)。
- [分析脚本](analyze_m2_slot_batching.py)、[绘图脚本](plot_m2_slot_batching.py)、[门限回归](test_m2_slot_batching.py)、[分析回归](test_analyze_m2_slot_batching.py)。
- 原始正式数据位于容器 `/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_slot_batching_20260910/formal01`；宿主机是 `/mnt/nvme0/mllm/sglang` 下相同相对路径。`smoke01`、`smoke02` 单独保留，不进入正式统计。
- 统计在远端 serving venv 执行；三图使用本地 Matplotlib 生成后同步回远端。每个正式 artifact 的 SHA-256、启动命令、测量源文件哈希与各重复指标都保留在分析文件中。

```bash
/workspace/sglang/.venv/bin/python \
  /workspace/sglang/benchmark/test_motivation/Ledger/analyze_m2_slot_batching.py \
  /workspace/sglang/benchmark/test_motivation/Ledger/results/m2_slot_batching_20260910/formal01 \
  --output /workspace/sglang/benchmark/test_motivation/Ledger/results/m2_slot_batching_20260910/analysis
```
