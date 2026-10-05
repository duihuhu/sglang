# M2：修正 slot 门限以后，固定首次激活限额未显示额外收益

2026-09-10。最终主分析为 **48 组、3840 请求**，四个原生 batching 配置、两个到达顺序、额度 0/1/4、两次独立服务启动。原运行经历一次 GPU 状态查询超时，按事先记录的整会话规则补跑；不是一次连续完成的矩阵。

**结论：已验证的 slot 计数缺陷及其局部收益仍成立；本轮没有找到固定首次激活限额的可重复正向收益。不能用这个负结果否定所有 admission 策略，也不能把它改写成现有方法的又一个缺陷。**

## 1. 本轮在故事线中的位置

M1 说明：相同 token 计量可以对应不同的 fixed state 负担。M2 进一步检查：把候选加入 batch 时，是否正确区分本轮执行参与和新增 holder？此前已经定位了一个具体问题，并验证最小修正的局部收益。本轮问的是修正之后还有没有额外机会。

| 层次 | 实验 | 观察 | 当前结论 / Insight |
|---|---|---|---|
| 已有具体缺陷 | 原 slot 门限与 reuse-aware 门限配对 | 继续 chunk 已有 slot 却再次占用 free-slot 额度；主对照提前激活 7 -> 8 | 执行参与数不等于新增分配数；存在错误排除可行候选的情况 |
| 该缺陷的服务后果 | 此前 100 组 / 2000 请求 | Separate-4K / short-first 的短请求均值 6.874 -> 6.299s，改善 8.37%；mixed 收益缩小 | 准入机会的价值取决于后续组批与释放，不能从拒绝次数直接推出收益 |
| 修正后的剩余机会 | 本轮 48 组，对正确基线增加每轮首次激活额度 | 14 个真正触发限额的配置/顺序/额度组合，平均 JCT 均变差；另外 2 个组合没有触发 | 仅减少每轮新 holder 数量，并不自动减轻持续占用或改善服务 |

前两行的原始证据、置信区间和小样本 p95 边界见[slot 缺陷与收益报告](m2_slot_reuse_batching_opportunity.md)。第三行是新的探索性负结果，不借用此前修复 bug 的 8.37% 收益，也不是否定前两行。

## 2. 干预到底改变了什么

所有配置启用正确 slot 门限，共用 token-VTC 排序。`k=0` 是关闭额外限额的正确基线；`k=1/4` 限制一次 prefill 组批扫描中新激活的请求数，已有 slot 的继续 chunk 不扣额度。

它不是总并发上限，不要求等待已有 holder 释放，不保证 separate 模式会优先执行 decode，也不使用输出长度或 lifetime predictor。达到额度只结束当前扫描，下一次扫描仍可继续激活新请求。因此“每轮少开几个”与“控制一段时间内的活跃 holder 数”不是同一个动作。

`activation_pacing` 在候选进入剩余 PrefillAdder 预算检查之前记录。事件次数不等于独立受影响请求数，也不证明一个已经满足所有预算的请求被错误拒绝。它不能与旧 slot 缺陷的可行性见证混用。

| 共同设置 | 数值 |
|---|---|
| 模型 / 拓扑 | Kimi-Linear-48B-A3B，TP2，仅物理 GPU 6/7 |
| 资源 | state8，有效 request pool=8，runtime cap=8，KV capacity=131072 |
| 排序 | token-VTC：input=1，output=2，request tax=0，二次项均为 0 |
| Batching | Separate-4K、Mixed-4K、Mixed-1K、Unchunked；两个 mixed 参数属于同一家族 |
| Workload | 每 wave 16 个 P512 短请求 + 4 个 P8192 长请求，均 O64；4 waves，间隔 2s |
| 配对 | 同一 rep 的 token IDs、输出长度和预设到达相同；short-first / long-first 仅改变组到达顺序 |
| 其他 | radix off，CUDA graphs off，max prefill tokens=16384；可选 prefill-size / free-slot-delay / PrefillDelayer 关闭 |
| 重复 | 每个模式两次独立启动；每次覆盖六个额度/顺序对照，正式顺序随机化，独立预热排除 |

本轮是人为紧 state/request pool 下的有限 wave 筛选，不是默认自然容量、持续到达负载或完整 SOTA 排名。协议见[冻结协议](m2_activation_pacing_protocol_20260910.md)。

## 3. 完整结果：没有筛出新的正向干预

![正确基线与首次激活限额的绝对结果](figures/m2_activation_pacing_outcomes.png)

每根柱是两次运行的均值，黑点保留各次结果。因每个模式分别启动服务器，跨模式的绝对数值只作背景，不作严格因果排名。以下相对变化均先对每个 rep 计算 `1 - paced / baseline` 再取均值；**正号表示下降，负号表示变差**。

| Batching | 到达顺序 | k=1：全部 mean JCT | k=4：全部 mean JCT | k=1：makespan | pacing 次数/运行，k=1 / k=4 |
|---|---|---:|---:|---:|---:|
| Separate-4K | short-first | -11.43% | -2.18% | -8.04% | 55 / 9 |
| Separate-4K | long-first | -11.03% | -1.59% | -8.09% | 57 / 9 |
| Mixed-4K | short-first | -3.59% | -1.33% | -1.66% | 33 / 5 |
| Mixed-4K | long-first | -3.72% | -1.06% | -1.86% | 37 / 5 |
| Mixed-1K | short-first | -1.05% | +2.09%* | -0.38% | 33 / 0 |
| Mixed-1K | long-first | -5.39% | -0.96%* | -2.47% | 38 / 0 |
| Unchunked | short-first | -16.23% | -1.41% | -13.65% | 69 / 9 |
| Unchunked | long-first | -12.08% | -1.08% | -9.19% | 69 / 9 |

`*` 两次重复均没有触发额度：Mixed-1K 的原生 token/chunk 条件已先限制了本轮新请求数。它的 k=4 两组中，prefill 批次数与基线也相同。不能把 +2.09% 归因为新的准入决策；两次原始变化分别为 +4.59%、-0.41%，方向也不一致。

![长短请求与最终完成时间的全部配对变化](figures/m2_activation_pacing_tradeoffs.png)

8 个 k=1 组合的两次配对均值都变差，但并非每个单次样本都变差。例如 Mixed-1K / long-first 两次分别为 -11.11%、+0.32%。因此结论是本轮未发现可重复正向机会，不是估计出一个精确、普适的退化比例。只有两次重复，不报告确认性置信区间或自动 PASS/FAIL。

全部单次结果、nearest-rank p95 TTFT/JCT、最大 JCT、TPOT、排队/活跃时间和资源积分保存在[analysis.json](results/m2_activation_pacing_20260910/analysis01/analysis.json)；[完整配对统计表](results/m2_activation_pacing_20260910/analysis01/report.md)不筛选赢家。

## 4. 轨迹解释：少开新请求，并没有省下持续占用

下表均为 short-first 的两次运行均值，比较正确基线与 k=1。

| Batching | Prefill 批次数 | 短请求平均排队，s | State-slot-seconds | 全部 mean JCT，s |
|---|---:|---:|---:|---:|
| Separate-4K | 42 -> 96 | 11.771 -> 13.503 | 310.604 -> 321.980 | 18.262 -> 20.349 |
| Mixed-4K | 48 -> 98 | 11.894 -> 12.392 | 310.902 -> 321.734 | 18.398 -> 19.059 |
| Mixed-1K | 176 -> 194 | 12.309 -> 12.502 | 315.972 -> 316.067 | 18.927 -> 19.124 |
| Unchunked | 15 -> 80 | 11.488 -> 13.682 | 309.648 -> 331.187 | 17.873 -> 20.770 |

全部 48 组最终都达到 8 个 active holder，说明这个 per-pass 限额没有把峰值活跃数从 8 降下来。k=1 的组批更碎、排队更长，而且两次均值下的 state 占用积分并未减少。即使某些 k=4 组合略微减少占用积分，例如 Unchunked / short-first 的 309.648 -> 307.814 slot-s，JCT 仍变差；占用下降也不自动等于服务收益。

这里的 state-slot-seconds 是活跃 holder 数对时间的积分，不是 NVML 保留显存，更不是 GPU kernel 时间。Prefill 批次数也是调度轨迹计数。当前证据与额外组批拆分、排队增长相符，但尚未用 GPU kernel profiling 分解其计算、launch 和同步成本，不能把计数比例当成计算时间比例。

**Insight：必须区分两类问题。第一类是可行性判断错误，已有明确反例与修复收益；第二类是在可行候选之间作时间上的选择，需要证明等待的代价被后续执行或资源释放收益抵消。本轮固定 per-pass 限额没有提供后一种证据。**

## 5. 这轮没有证明什么

- 没有证明所有延迟准入、decode-aware 控制、动态 chunk 或 lifecycle 方法无效。本干预没有实现这些策略。
- 没有发现一个可以新增到论文中的“现有方法必然过度激活”的缺陷。自己加入限额导致退化，不是现有方法存在新缺陷的证据。
- 没有证明 state wall 单独解释全部时延差异；本轮没有执行自然容量、wide-cap8 或 KV-only 归因。
- 没有确认公平改善或违反 VTC 保证。短/长两组总工作不相等，固定 VTC 代码不保证瞬时服务差相同；现有 counter gap 包含 lift，也未限定双方持续 backlogged。
- 没有证明 SLO goodput、生产 p95 或 steady-state throughput 改善。有限 wave 的 makespan 不等于稳态吞吐，SLO 阈值没有事后挑选。
- 没有比较完整 Sarathi-Serve、FairBatching 或 LPRS/APC。不能把原生 mixed 配置当成完整论文复现，也不能从本轮结论推断这些方法在 Kimi-Linear 上的效果。

## 6. 对现有工作的审查更新

已有机制不能笼统描述成“只看 tokens，不看请求数，也不会主动暂缓”。

| 已有机制 | 已核实内容 | 对本 topic 的约束 |
|---|---|---|
| SGLang 原生控制 | `prefill_max_requests` 限制本轮 prefill 参与数；`MinFreeSlotsDelayer` / `PrefillDelayer` 有空位、队列和等待控制，继续 chunk 路径有特殊处理 | 首次激活与复用的语义值得检查，但请求数上限、主动暂缓本身不是新机制 |
| [Sarathi-Serve](https://www.usenix.org/conference/osdi24/presentation/agrawal) | chunked prefill 与避免暂停已有 decode 的 stall-free scheduling | 需对比正确实现和校准后的调度，而不是一个 mixed 开关 |
| [FairBatching](https://arxiv.org/html/2510.14392v1) | slack 驱动的批时间预算；批时间模型包含新 token 和已有 context；分阶段安排 urgent decode、prefill 和 non-urgent decode | 不能称为固定 token-budget baseline；没有显式 holder 项也不自动证明性能缺陷 |
| [LPRS/APC](https://arxiv.org/html/2606.09061v1) | 预测特征包括 batch request count、context 和 memory；APC 使用 sequence headroom、剩余 token budget 与最小 chunk 进度控制 active prefill | “考虑请求数/slot”不足以构成差异；必须明确它仍遗漏的决策因素，并用真实对照验证 |

论文机制的读取和本机复现是两回事。上表是机制审查，不是这些方法在当前混合模型上的运行结果。

## 7. 后续如何推进

按冻结协议，本轮属于“正确基线上的剩余机会筛选”。结果不支持直接进入 lifetime predictor 或把固定激活限额发展为算法；也不值得为了证明它有用而扩大搜索后只保留正结果。

下一步应回到已有具体缺陷的外部有效性，再决定算法研究范围：

1. **自然压力下的发生频率。** 固定模型/拓扑和正确容量口径，记录自然派生的 request/state pool，进行长短比例、输出长度和到达负载变化；检查旧计数的反事实见证，不能仅人为设 state8 后宣称默认部署普遍受影响。
2. **明确区分两种实验。** 衡量 slot 修复收益时可以保留 legacy/corrected 配对；衡量新调度算法时，所有 baseline 必须启用 corrected gate，不能靠保留旧 bug 获胜。
3. **先比较已有简单 batching 控制。** 若在生产相关压力下仍有可定位的损失，校准 `prefill_max_requests`、free-slot-delay、PrefillDelayer 及 mixed/chunk 参数，再验证更强动态 batching 是否仍留下空间。
4. **最后确认服务收益。** 使用独立 held-out 到达轨迹、CUDA graphs、更多独立启动、预先冻结的 SLO 和精确 backlog 事件，同时报告短/长服务代价；再决定是否存在新的资源抽象或联合控制贡献。

其中第 1 项是缺陷的外部有效性验证，不是给本轮没有收益的 pacing 作正向归因。自然压力和完整文献级对照本轮未执行；当前 M2 主论点仍应保持在“新增与复用的语义错配可造成局部可避免损失”。

## 8. 执行、审计与复现

- 主分析：48 个 artifacts、3840 请求；零请求错误、prefix token、retraction、resource/batch trace drop，输出长度精确一致；全部运行结束 active=0，peak active=8，每个事件满足一个 active holder 对应一个 state slot。
- 预检：6 组 smoke / 120 请求，独立排除。原运行与补跑共 54 组预热 / 1080 请求，均不进入性能统计。
- 原始测量：`screen01` 有 39 组，`recovery01` 有 12 组，共 51 个完整 artifacts；按[补跑规则](m2_activation_pacing_recovery_20260910.md)排除被中断会话的 3 组，保留 36+12=48 组。中断的运行无完整 artifact，不补造数据。原目录不改、不删，也未补写原运行成功标记。
- `assembled01` 是单独的数据视图。其 manifest 记录每个选中/排除文件的原始路径和 SHA-256，以及原始、补跑 manifests；完成标记明确指向组合后的矩阵。所有 13 个测量源码哈希在原始启动、补跑启动及结束、合并时一致。
- 保留数据有 3774 个双卡采样时点；1469 个单卡读数满足 utilization>=80%，对应 SM 1350--1410MHz，最高温度 53C。实际提交最大迟到 47.78ms，小于冻结的 50ms 门限。健康检查通过不等于完全没有硬件或主机时序漂移。
- 第一次 Unchunked 启动时权重加载出现约 193s 延迟，服务自行恢复，未找到 Xid/OOM 日志；此段启动不计入请求性能。后来另一次 GPU 状态查询超过 10s，触发上述中断。两次事件的根因未确认，没有通过放宽健康阈值掩盖它们。
- 没有修改 GPU 时钟/功率、换用其他卡或停止其他任务。补跑退出后的 GPU 6/7 均为 1 MiB、0% utilization。
- 本轮保留了本地后来增加的可选 GPU 参数支持，未将其覆盖进冻结的远端测量版本。复现以 manifest 中的远端版本为准，不假定当前本地脚本逐字相同。

原始数据在容器 `/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/`；宿主机对应 `/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/`。本地同步分析与图表，原始请求轨迹留在远端。

脚本：[运行](run_m2_activation_pacing.py)、[分析](analyze_m2_activation_pacing.py)、[整会话合并](assemble_m2_activation_pacing.py)、[绘图](plot_m2_activation_pacing.py)。合并工具不会覆盖原数据或既有结果目录，且拒绝源版本变化、参数变化、不完整补跑和异常矩阵。测量结束后 56 项实验/slot/合并测试与 10 项 tracker/policy 组件测试全部通过；后者在隐藏 CUDA 的进程中执行，有 17 条配置、弃用及无 GPU quantization 警告，无失败。

以下为本次分析命令记录，结果目录已经存在，不能原样覆盖重跑；再次分析应指定新的输出目录。

```bash
.venv/bin/python benchmark/test_motivation/Ledger/analyze_m2_activation_pacing.py \
  benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/assembled01 \
  --output benchmark/test_motivation/Ledger/results/m2_activation_pacing_20260910/analysis01
```
