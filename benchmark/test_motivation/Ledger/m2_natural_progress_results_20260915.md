# M2：自然 state 瓶颈成立，但持续参与不等于更快释放

2026-09-15。Kimi-Linear-48B-A3B，TP2，仅物理 GPU 6/7。完成 20 组正式运行、4316 个请求；20 组预热、540 个请求及 3 个功能预检请求独立排除。

> 后续已完成：[2026-09-16固定607请求的校准与同池cap干预](m2_state_mediation_results_20260916.md)，26组/15782请求。mean JCT的cap交互方向得到探索性支持，但放宽cap后两模式自身mean JCT均变差、tail/makespan变好；不能称为hybrid特有加速机会。下文保留9月15日当时结论与尚未执行的计划，不回写历史数据。

**本轮结论：在正确 slot 门限和自动解析的自然容量下，大量短请求确实遇到 state/request 上限，而 KV 仍很宽裕。但“已激活却未执行”并非当前证据中的主要阻塞解释；让 decode 持续参与的原生 Mixed-4K 反而增加了高压力 workload 的完成时间和 state 驻留积分。M2 应研究准入、组批执行速度与资源释放的联合后果，尚不能声称 hybrid 特有的性能放大已经得到因果证明。**

## 1. 本轮到底检验什么

从 M1 递进的结构性事实是：请求首次激活支付固定 state 成本，后续 chunk/decode 不重复支付。问题不再是系统是否知道 state 放不下，而是已准入请求的执行与释放是否会改变后续请求的服务结果。

本轮采用[数据采集前冻结的协议](m2_progress_probe_protocol_20260915.md)，检验以下候选解释：

> 请求已经持有 state，但在其他请求组批时持续没有被选择；如果此时 state 已满且后续请求等待，那么让已有 holder 更持续地推进，是否可能改善后续准入？

这个说法拆成三项，不能合并判定：

1. 自然容量下是否真的存在 state/request 瓶颈，而非人为缩到 state8 才出现？
2. 已完成 prefill 的 holder 是否在等待者存在时未被选入 batch，尤其是否与 state 满同时发生？
3. 消除这种未参与现象，是否同时减少驻留、排队与 JCT？

第 1 项得到支持；第 2 项的强版本只观察到很小的同时发生区间；第 3 项在当前原生配置对照中没有得到支持。

## 2. 共同条件与 baseline 身份

| 条件 | 本轮设置 |
|---|---|
| 物理资源 | A800-SXM4-80GB，GPU 6/7，TP2，mem-fraction-static=0.85 |
| 自然 readback | state slots=485；KV capacity=1,458,643 tokens；有效 request pool=485；runtime cap=485 |
| 容量设置 | 不手动指定 state、KV 或 max-running 容量；自然容量指上述明确部署选项下的自动解析值 |
| 共同排序 | 本地 token-VTC，input=1、output=2、request tax=0、二次项=0；启动后更新并逐 run 校验 |
| 共同准入 | corrected slot gate=1；额外 new-activation limit=0 |
| Batching 对照 | 原生 Separate-4K 与原生 Mixed-4K；均 chunk=4096、max-prefill-tokens=16384 |
| 隔离条件 | radix off、prefill/decode CUDA graphs off、无 speculative generation；可选延迟与 prefill-max-requests 控制关闭；统一池未启用 |
| 配对与重复 | 每模式两次独立服务启动；同一 shape/rep 的 payload、输出和计划到达完全配对；预热排除 |

这是两个正确原生 batching 配置的机制对照，**不是完整 Sarathi-Serve、FairBatching、LPRS/APC 的复现或 SOTA 排名**。启动命令中的 FCFS 随后由实验运行时接口切换为共同 VTC，原始 readback 与完整命令保存在 manifest/artifacts 中。

| Workload | 请求数及输入 | 总 input tokens | 每请求 output |
|---|---|---:|---:|
| long | 16 x 8192 | 131072 | 128 |
| medium | 64 x 2048 | 131072 | 128 |
| short | 256 x 512 | 131072 | 128 |
| mixed | 128 x 512 + 8 x 8192 | 131072 | 128 |
| state-pressure | 607 x 512 | 310784 | 128 |

压力点数量由协议公式 `min(1024, ceil(1.25 * state_slots))` 得到 607，不按测量结果调参。所有请求在有限 burst 中以 1ms 计划间隔提交。同一形态内比较 batching；不同形态的总 output 工作量不同，不能用跨形态 JCT 差异直接归因固定 state 成本。

## 3. 观察一：自然 state 上限确实先于 KV 耗尽

![自然容量下的 state/KV 压力](figures/m2_natural_pressure.png)

607 个 P512/O128 请求在两个模式、两次重复中均达到 485 个 active holder，即自然 state/request/cap 的上限。全运行的 KV 峰值仅为：

| 模式 | Peak state / state cap | Peak KV / KV cap | State 满且仍有待激活请求的时间，两次均值 |
|---|---:|---:|---:|
| Separate-4K | 485 / 485，100% | 310400 / 1458643，21.28% | 7.984s |
| Mixed-4K | 485 / 485，100% | 294881 / 1458643，20.22% | 4.261s |

每组都有 request-limit 路径记录。资源图是调度器观测到的 active ownership，不是 NVML 预分配显存用量；KV 峰值比例与 state 峰值不是可相加的统一显存比例。实际准入还包括候选预留，不能仅凭 physical free 推断每个候选都满足全部预算。

**具体观察：这种偏向请求数而非 token 容量的瓶颈，在当前自动解析的部署容量下真实发生，不是只存在于此前 state8 的人为设置。**

**Insight：本 workload 中，减少 token chunk 并不会自动增加可同时驻留的请求数，因为第一次激活的 state 成本不随 chunk 缩小。** 不过 SGLang 已经利用 state 容量派生并发上限，因此该观察不证明其容量检查不正确。

无 prefix、单模型同大小 state、独立池条件下，state 的数量可行性可以表示为并发上限。这里的 state/request/runtime cap 同为 485，不能凭本轮区分哪个门限产生了独立于其他门限的额外影响。相比普通请求 cap，hybrid 的不同之处首先是上限有实际固定 state 内存来源，不是阻塞数学机制完全不同。

## 4. 观察二：有参与间隙，但大部分不是“满池时闲置 holder”

新增 default-off 进度日志记录 batch 的请求 ID、prefill/decode 与本批实际工作 token。分析只把最后一次 prefill 结果已记录之后的 holder 作为 decode-ready，排除正常 prefill 执行。相邻 CPU batch launch 时间区间中，没有被前一批选择的活跃 holder 累积为“未参与 slot-seconds 代理量”。

它不是 GPU kernel 空闲时间，也不是可直接回收的时间。Overlap 的在途工作、CPU 观测时刻、正常时间复用都会影响它；待激活区间从 client 提交计起，也不等同于精确 scheduler-only backlog。

| 高压力 workload 的指标，两次均值 | Separate-4K | Mixed-4K |
|---|---:|---:|
| Decode-ready 未参与且仍有待激活请求，slot-s 代理量 | 1735.037 | 0 |
| 再要求 state 同时已满，slot-s 代理量 | 34.498 | 0 |
| 上一行 / 第一行 | 1.99% | 不适用 |

后一项是采集结束前补充的探索性分层分析；没有改变冻结 workload 或测量协议，也没有预设通过阈值。它取 pending intervals 与 state-full intervals 的交集，再积分相应 holder，避免把先后发生的事件误认为同时发生。

**观察：Separate 中确实存在未参与区间，但约 98% 的上述带 pending 代理量发生在 state 尚未满时；不能把全部 1735 slot-s 都解释为满池后挤掉其他请求的无效占用。** 所有非压力 workload 均未满池，但也有类似未参与现象，进一步表明这包含通用 prefill/decode 组批效应。

**Insight：必须同时检查“持有资源、可执行、没有参与、别人等待、资源确实成为约束”这几个条件。单独统计 holder 等待，容易把普通 batching 行为误写成 hybrid 特有缺陷。** 即使条件同时满足，也还需要同目标、同预算干预才能称为可避免损失。

## 5. 观察三：持续参与不保证驻留减少，当前 Mixed 更慢

![参与代理量与服务结果](figures/m2_natural_participation.png)

每根柱是两次运行均值，黑点是各次结果。Mixed 的上述 decode-ready 未参与代理量在全部 10 组均为 0，但全部 workload 的 mean JCT 均高于对应 Separate。

| Workload | Separate mean JCT | Mixed mean JCT | Mixed 的配对 JCT 增幅 |
|---|---:|---:|---:|
| 16 x 8K | 12.583s | 12.970s | +3.07% |
| 64 x 2K | 10.447s | 10.744s | +2.85% |
| 256 x 512 | 10.775s | 11.332s | +5.18% |
| 128 short + 8 long | 11.522s | 12.267s | +6.46% |
| 607 x 512，高压力 | 16.467s | 22.206s | +34.85% |

增幅先按相同 rep 计算 `Mixed / Separate - 1` 再取两次均值；正数表示 Mixed 更慢。只有两个独立启动，未提供确认性 CI，模式执行顺序也不足以完全消除会话漂移；不能将表中比例视为普适效果大小。

| 高压力 workload 的结果，两次均值 | Separate | Mixed | 配对增幅 |
|---|---:|---:|---:|
| Mean JCT | 16.467s | 22.206s | +34.85% |
| Mean queue | 5.680s | 7.593s | +33.69% |
| Mean TTFT | 5.864s | 7.903s | +34.78% |
| Makespan | 23.646s | 28.199s | +19.26% |
| State-slot-seconds | 6521.884 | 8858.517 | +35.83% |

Mixed 虽然把 state 完全满载且有 pending 的区间从约 7.98s 缩至 4.26s，但总体 state-time、排队和 JCT 都增加了。图中它也更晚到达满池，并更缓慢地释放大批 holder。因此“更少满池时间”“更少未参与时间”都不能单独作为优化目标。

**具体问题：以“让已有 decode 持续参与”为局部准则，不能充分预测当前 hybrid 服务的总体完成效率和持久 state 占用。当前两个 batching 配置呈现了真实、方向一致的服务差异，而不是只有资源计数差异。**

**Insight：执行参与不等于有效完成速率；state 是否释放取决于请求何时完成，而非是否进入每个 batch。候选准入会改变后续 batch 的工作结构，batch 执行速度又反过来决定 state 驻留和后续准入。**

这里没有完成 GPU kernel profiling，不能把退化全部归因于 KDA、MLA、MoE，或仅归因于 batch 变大。Mixed 同时改变工作混合、prefill 预算与 chunk 边界；例如 long 的平均 prefill chunk 数从 2 变为 2.9375，pressure 从 1 变为 1.1334。这些共同变化需要分解，不能把消除代理间隙当成唯一干预。

## 6. 对“hybrid 是否放大问题”的正式回答

| 判断 | 当前状态 | 理由 |
|---|---|---|
| Fixed state 对自然容量有实际影响 | 已验证 | 607 短请求触及 485 state/request 上限，而 KV 峰值仅约 20%--21% |
| 正确 batching 仍存在服务结果差异 | 已观察 | 同一 workload 的 Mixed JCT、queue 与 makespan 更高，压力点差异更大 |
| 满池时大量 holder 不执行是主要损失来源 | 当前不支持 | Separate 带 pending 的未参与代理量中，仅约 1.99% 同时处于 state 满的区间；Mixed 消除间隙仍更慢 |
| State wall 因果性放大了 batching 性能差异 | 未验证 | 256 -> 607 同时改变请求数、总计算与压力；尚无固定工作集的 cap/池归因与 KV-only 对照 |
| Hybrid 上现有 SOTA batching 有未解决缺陷 | 未验证 | 当前只是两种原生配置，未校准强 baseline、开启 graphs 或复现完整先进方法 |
| 应实现新的 lifetime/activation 调度算法 | 暂无依据 | 本轮无新算法收益见证；之前固定 per-pass activation 限额也未显示额外收益 |

因此，不能把 `5.18% -> 34.85%` 直接写成“hybrid state 将性能损失放大约 6.7 倍”。它目前只是随请求量和压力升高的观察，需要固定工作量的对照。

适合与 M1 相接的 M2 暂定表述是：

> M1 揭示请求形态改变 fixed-state 负担；M2 进一步观察到，这一负担在自然部署中形成实际并发瓶颈。即使容量检查正确，选择如何混合、推进这些 holder，也会改变完成速度、驻留积分和后续排队。资源安全可行性并不能充分说明服务结果，但要声称新的 hybrid 调度缺陷，仍需排除已有 batching 调优和普通并发效应。

这是一条有实测基础的机制线索，不是已经完成的论文算法 motivation。此前[slot 计数缺陷及修复收益](m2_slot_reuse_batching_opportunity.md)继续成立，单列为通用实现缺陷，不用它填补 hybrid 特有归因的空缺。

## 7. 下一步的优先顺序

不直接追加 lifetime predictor，也不将每轮激活额度换几个数后继续寻找正结果。根据本轮结果，先检验“执行成本和释放节奏”，而不是预设“holder 不参与太多”。以下是建议矩阵，尚未冻结或执行。

1. **固定 607 个请求，校准现有 batching。** 保持物理 state/KV 池与请求工作集不变，扫描少量预先指定的运行时 cap、chunk、原生 prefill 限制；同时报告 JCT、TTFT、TPOT、makespan 与 state-time。目的：确认压力点的损失是否主要能由普通并发/batch 调优消除。这类 cap 扫描不单独算 hybrid 创新。
2. **定位 Mixed 的工作与时间变化。** 单独 profiling 运行，区分新 prefill tokens、decode holders、chunk 数与 KDA/MLA/MoE kernel 时间。比较相同请求数及工作构成的批，检查是执行混合、碎片化、后端开销还是其他路径。Profiling 数据不混入无 profiler 的延迟统计。
3. **再做 hybrid 归因。** 在固定工作集下分开比较普通 cap 中介、state/KV 池划分以及相同总字节预算下的可行性；先核实当前模型对已有统一池的支持。只扩大 state 并额外增加显存不是同预算收益；匹配 cap 的普通模型也不能被当成完全相同的 kernel 对照。
4. **最后检查剩余机会。** 使用调优后的正确原生方法及与机制匹配的先进 batching baseline，在 held-out 到达轨迹和预先明确的服务代价约束下寻找改进。需要精确 scheduler backlog 事件；不能从当前有限 burst 推导公平保证、稳态吞吐或 SLO goodput。

若现有调优或统一池已经处理了主要损失，就将 M2 降为部署/资源瓶颈 insight，不强行包装新算法。若校准后仍存在同目标、同预算下的可避免损失，再冻结有针对性的干预实验。

## 8. 完整性、实现变动与复现

- 正式矩阵 20/20，4316/4316 请求；预热 20 组/540 请求排除。每次 prefill 工作之和精确等于 prompt tokens，prefix progress 连续；零请求错误、prefix reuse、retraction 和 resource/batch trace drop；所有运行结束 active=0，每事件一个 active holder 对应一个 state slot。
- 实际提交最大迟到 43.812ms，未超过冻结的 50ms；每组 GPU 健康检查通过。所有正式请求的已提交 decode 步数均比 `delivered output - 1` 多 1，符合本轮 overlap 路径的记录现象；分析不把额外提交步当作已经交付的服务，也不据此计算公平性。
- 16 个测量源码/协议文件的哈希，在功能预检通过、正式运行开始、各服务会话与最终完成检查中保持一致。独立分析与绘图文件不改变测量策略。
- 唯一新增服务路径是 default-off 的 progress trace；读取 batch 保存的 Python lists，不读取 CUDA tensor、不增加 GPU 同步，也不新增 admission 策略。开 trace 的开销尚未单独做 A/B，本轮共同开启；需要生产外推时补测。
- `inspect01` 启动成功，但自然模式的配置字段 `max_total_tokens=None` 被旧验证器当成无效容量，功能 workload 发送前终止。原日志与失败目录保留。修正 readback：保留原响应，使用 `memory_usage.token_capacity` 校验实际容量；未改变服务器预算。随后 `inspect02` 的 3 个功能请求通过，再执行正式矩阵。
- 测量完成后，远端 103 项 M2 实验/分析回归、10 项 tracker/policy 组件测试全部通过。组件测试隐藏 CUDA，有 17 条配置/弃用/无 GPU 后端警告，无失败。
- 全部本轮服务已退出；收尾 GPU 6/7 均为 1MiB、0% utilization。没有修改 GPU 时钟/功率、使用其他卡或终止其他任务。
- 保留本地已有的 GPU 参数支持，没有覆盖远端冻结的旧 helper 版本。复现以 manifest 的远端源码哈希为准，不假定本地旧 helper 与远端逐字相同。

完整逐 run 数值、每请求参与结果、资源时间线与原 artifact 哈希见 [analysis.json](results/m2_progress_probe_20260915/analysis01/analysis.json)，原始逐 run 主表见 [report.md](results/m2_progress_probe_20260915/analysis01/report.md)。原始请求、进度与 GPU 轨迹保留在容器 `/workspace/sglang/benchmark/test_motivation/Ledger/results/m2_progress_probe_20260915/`，宿主机对应 `/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/m2_progress_probe_20260915/`；本地同步分析与图表。

脚本：[运行](run_m2_progress_probe.py)、[分析](analyze_m2_progress_probe.py)、[绘图](plot_m2_progress_probe.py)。以下命令是已完成的执行记录；输出目录已存在，重分析必须指定新目录，不能覆盖原结果。

```bash
.venv/bin/python benchmark/test_motivation/Ledger/run_m2_progress_probe.py \
  --stage observe \
  --capacity-reference benchmark/test_motivation/Ledger/results/m2_progress_probe_20260915/inspect02/capacity.json \
  --output benchmark/test_motivation/Ledger/results/m2_progress_probe_20260915/observe01

.venv/bin/python benchmark/test_motivation/Ledger/analyze_m2_progress_probe.py \
  benchmark/test_motivation/Ledger/results/m2_progress_probe_20260915/observe01 \
  --output benchmark/test_motivation/Ledger/results/m2_progress_probe_20260915/analysis01
```
