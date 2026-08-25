# MoE Energy Profiling 数据总览

本文档索引 `benchmark/moe-energy/profiling/data/` 下各一级目录，说明测试对象、数据口径、主要文件、关键结论与限制。除非特别说明，模型为 Qwen3-30B-A3B，GPU 为 NVIDIA A800-SXM4-80GB。

## 1. 统一口径与注意事项

### Latency 与 Energy

- `latency_us` 通常是参与 ranks 的最大本地 wall time，单位 µs；具体是否含 barrier 以各目录文档为准。
- `energy_mj` 是测量窗口内所有参与 GPU 的 NVML 能耗增量之和，单位 mJ，不是单卡能耗。
- 不同 GPU 数、不同频率、不同测量边界的数据不能只看绝对 mJ 直接横比。

### 三种主要测量边界

1. `kernel-EP`：仅 `experts.run_moe_core()`，不含 norm/gate/top-k/dispatch/communication。
2. `EP`、`mix-EP-TP`、`hot-EP`：完整 MoE/FFN 段，通常包括 post-attention norm、gate、top-k、dispatch、expert core、combine、reduce，但 A2A 多为 `none`。
3. `Top-k` 的主结果：真实 Attention 先运行并捕获标准 MoE input；Attention 不计时，只回放单层 MoE。

`raw/`、`logs/` 和 manifest 用于复现/诊断；正式分析优先使用各目录标记为 valid/fixed/repeated 的汇总 TSV。

---

## 2. `TP/`：Attention TP 与 FFN TP 基础矩阵

### 内容

TP2/TP4/TP8 下的 Prefill/Decode 组件级 latency/energy 矩阵：

- `PF.txt`：FFN Prefill，1368 行；
- `DF.txt`：FFN Decode，1344 行；
- `PA.txt`：Attention Prefill；
- `DA.txt`：Attention Decode。

主要维度：

- TP size：2/4/8；
- SM 频率：210/450/690/930/1170/1410 MHz；
- length：64–4096；
- batch：1–4096，Decode 含更大 batch 点。

### 结论

- Prefill/大负载 FFN 中，TP2→TP4 能显著降低延迟；TP4→TP8 的额外延迟收益变小，而总能耗常因 GPU 数增加而上升。
- Decode 小负载主要受固定开销控制，TP2/4/8 延迟接近，但总能耗随参与 GPU 数明显增加。
- TP 的最优度取决于 workload 和 SLO；“更多 TP 一定更快/更节能”均不成立。

### 限制

这里是组件 profiling，不是完整服务 TTFT/TPOT；TP collective 和前后层融合口径需结合 runner 脚本确认。

---

## 3. `DeepEP/`：真实/专用 EP 通信路径 Pilot

### 内容

- `PF-ep-balanced.txt`
- `PF-ep-skewed.txt`

覆盖 EP2/4/8、Prefill length 64/256/1024/4096、若干 batch，以及 345/1005/1650/1980 MHz。

### 结论

- 在公共测点上，skewed/balanced 的 latency 中位比约 1.39，energy 中位比约 1.08。
- 真实/专用 dispatch 路径下，路由偏斜对时延影响通常强于总能耗影响；通信、等待和冷 ranks 功率共同决定 energy。

### 限制

矩阵较小且 balanced/skewed 覆盖不完全；只有 Prefill pilot，不应替代 `EP/` 或 `kernel-EP/` 的完整矩阵。

---

## 4. `kernel-EP/`：单层 MoE Kernel-only EP 矩阵

详见 [`data/kernel-EP/doc.md`](data/kernel-EP/doc.md)。

### 内容

- EP2/4/8；
- balanced / middle_rank0 / skewed_rank0；
- 6 档 SM 频率；
- Prefill input length 与 Decode batch 扫描；
- 测量对象仅 `run_moe_core()`。

汇总：

- `PF-{balanced,middle,skewed}.txt`
- `DF-{balanced,middle,skewed}.txt`
- `pic/` 图表。

### 结论

- 负载与延迟/能耗总体随 MoE token 数增长。
- 1410 MHz 通常时延最低，但 930/1170 MHz 更常是能耗甜点。
- EP4/EP8 下，rank0 集中路由显著拉长最慢 rank；偏斜影响随 EP size 增大。
- EP2 skewed 的特殊优势来自 A2A=none、forced routing 和 local filtering，不能解释为真实通信 EP 中 skewed 更优。
- 小 batch 的台阶/非单调与 Triton config、padding、occupancy 和固定开销有关。

### 限制

不含 Attention、norm、gate、top-k、dispatcher、真实 A2A 和完整 FFN；只能用于 kernel 机理分析。

---

## 5. `EP/`：完整 FFN 的纯 EP 矩阵

### 内容

- `PF-balanced.txt` / `PF-skewed.txt`
- `DF-balanced.txt` / `DF-skewed.txt`
- `FFN-baseline.tsv`
- `raw_v2/`、`raw_baseline/`、`pic/`

覆盖 Prefill/Decode、EP2/4/8、balanced/skewed、length/batch 和 210/690/930/1410 MHz。完整 FFN 通常包含 norm、gate、top-k、dispatch、expert core、combine、EP reduce；A2A=none、CUDA Graph关闭。

### 结论

- 完整 FFN 中 fixed gate/dispatch/combine/reduce 会稀释 kernel-only 的差异，小负载下 balanced 与 skewed 延迟更接近。
- 大负载下 skewed 形成 straggler，EP 越大相对损失越明显。
- Prefill 与 Decode 在相同 MoE token 数 M 下定性一致，说明 FFN 主成本由 M 和 routing shape 决定。
- cluster energy 的最坏点不一定是完全 skewed，因为冷 ranks 的等待功率和 active rank 数也参与总能耗。

### 限制

forced routing 是性能压力构造，不代表自然 router 的任务质量；A2A=none，不包含真实跨 rank token 搬运。

---

## 6. `mix-EP-TP/`：混合 EP × MoE-TP

### 内容

`mix-EP-TP.tsv`，744 个汇总点，覆盖：

- TP4/EP2 = EP2×MoE-TP2；
- TP8/EP2 = EP2×MoE-TP4；
- TP8/EP4 = EP4×MoE-TP2；
- P/D、balanced/skewed、length/batch、4 档频率。

### 结论

- 相同 8 GPU 下，balanced 更偏好更多 EP（纯 EP8 或接近纯 EP），因为可将不同 experts 均匀分散。
- 大负载、极端 skewed 时，更多 expert 内 MoE-TP 能缓解单 expert/rank hotspot；EP2×MoE-TP4 优于 EP4×MoE-TP2 和纯 EP8。
- EP4×MoE-TP2 是中间折中，但在 balanced/fully-skewed 两个极端中通常不是最优，可能适用于中度 skew。
- topology 选择必须同时条件化于 GPU budget、M 和 routing skew。

### 限制

A2A=none、Custom AllReduce关闭；结论主要反映本地 FFN和 post-expert reduction，不是完整真实 EP 通信排名。

---

## 7. `hot-EP/`：Rank skew、Expert concentration、Active-rank 与 Selective DVFS

详见 [`data/hot-EP/doc.md`](data/hot-EP/doc.md)。

### 内容

该目录包含多个受控实验：

- `hot-EP.tsv`：EP4 rank-level hot fraction；
- `hot-expert.tsv`：rank 内 active expert 数 32/8/2/1；
- `hot-EP-repeat-summary.tsv`：13 个 hot fractions、4 个 GPU 区组重复；
- `hot-EP-boundary-summary.tsv`：冷 ranks 从极少 assignments 到 0 的边界；
- `hot-EP-active-rank-summary.tsv` / `...ep8...`：active ranks 4→1、8→1；
- `hot-EP-fixed-M-ep8-summary.tsv`：EP8 固定总 M、K=1…8；
- `hot-EP-selective-dvfs*.tsv`：active 930 MHz、inactive降频；
- `pic/`：12 张分析图。

### 结论

- Rank-level skew 与 rank 内 expert concentration 是两个独立维度。
- 中小 M 下，部分偏斜可能比 fully-hot 更耗能：热点仍拉长窗口，而其他 ranks 仍在有效计算。
- 冷 rank 从 `1 assignment` 到 `0` 会有离散节能，但 M越大该差异越小，等待功耗占主导。
- Active-rank 最优数随 M 增大：小 M 少量 ranks 最节能，中 M约2–5 ranks，大 M接近全部 ranks。
- 固定总 M 时，错误 active-rank选择可导致约11%–50%的能耗差异；M≈3584 的最差/最优约2×。
- Selective inactive DVFS 在小/中 M 的甜点约450/690 MHz，可在最优 active-rank基础上额外节能约4%–8%；210 MHz常拖慢collective并使能耗反增。

### 数据卫生

目录保留了 pilot、补测和调试 raw。正式分析优先使用带 `repeat-summary`、`fixed-M`、`selected`、`boundary-summary` 的 TSV。部分早期单次 pilot 精确峰值不稳定，不应引用为正式结论。

---

## 8. `Top-k/`：Router Top-K 的 MoE 质量—性能—能耗

详见 [`data/Top-k/doc.md`](data/Top-k/doc.md)。

### 内容

核心有效数据：

- `moe_stage_quality_performance_fixed.tsv`：真实 Attention 先运行并捕获 MoE input，Attention不计时；固定 Top-K=8/6/4/2/1 的纯 MoE结果；
- `quality_screen.tsv`：完整模型 teacher-forced logits Top-1/KL快速筛选；
- `task_accuracy_proxy.tsv`：离线代理任务准确率；
- `dynamic_topk_*.tsv`：多种动态 Top-K实现 Pilot；
- `raw/attn_inputs/`、`raw/moe_reference_fixed/`、`raw/moe_replay_fixed/`；
- `pic/` 综合图。

### 结论

- 固定 Top-K 降低能显著减少大 M 的 MoE成本；Decode M=4096 中 Top-6 相对 Top-8约节能16.7%，Top-4约33.5%，Top-2约47.8%，Top-1约56.7%。
- 质量误差随 K下降快速增加。Top-6 是唯一较合理的未经训练候选；Top-4及以下局部 MoE输出误差和完整模型分布偏移明显。
- Decode 对 Top-K裁剪比 Prefill更敏感。
- token-level动态Top-K的算法层面可保护重要token，但当前三类复用实现均没有能耗收益：双组执行重复collective；固定宽masked仍按Top-8 buffer；显式packed会展开hidden并scatter-add。
- 真正高效动态Top-K需要 FusedMoE 原生 ragged assignment 接口和专门 kernel。

### 数据卫生

- 早期 `raw/moe_replay/`、`raw/moe_repeats/` 受 FusedMoE in-place输入覆盖影响，文档已标 invalid。
- 正式固定Top-K基线使用 `raw/moe_replay_fixed/` 和 `moe_stage_quality_performance_fixed.tsv`。
- 动态 Pilot 中多个 `pilot*.jsonl` 是调试版本；只引用文档明确标记 valid 的汇总。
- 官方 GSM8K 已缓存；MMLU全量下载未完成，proxy结果不能冒充官方分数。

---

## 9. `router-dvfs/`：路由稳定性、Router Shaping 与 Per-rank DVFS

详见 [`data/router-dvfs/doc.md`](data/router-dvfs/doc.md)。

### 内容

- `raw/trace.json`：12请求×32 Decode iterations×48 layers×Top-8 路由；
- `trace_summary.json`、`prediction.tsv`：跨层/iteration稳定性；
- `cost_model.json`、`dvfs_simulation.tsv`、`slo_sweep.tsv`：频率/SLO控制模拟；
- `bounded_routing.tsv`：有界reroute机制仿真；
- `replica_routing.tsv`：EPLB logical expert副本placement仿真；
- `joint_summary.tsv`、`pic/`。

### 结论

- 小样本单请求trace中，相邻iteration同层load相关约0.40，lag-1/EWMA热点预测仅约34%；跨层热点保持约14%。
- 从all-1410最短时延出发，异构频率可以利用rank-local slack：模型中layer-aware oracle在严格SLO下约有24.6%节能上限；+2% SLO下 iteration-static约14.7%，layer-aware约26%。
- SLO放宽到约5%后，uniform 930 MHz本身已节能约27%，异构额外空间缩小到约3个百分点。
- 有界reroute surrogate移动约12.5% assignments可将max/mean从2.52降至2.16，但缺少真实Top-C logits，不能下质量结论；Python greedy约9 µs/token-layer，在线必须GPU融合。
- EPLB logical expert replication保持模型语义；长期placement仿真中+8 redundant experts（+6.25% expert显存）可显著平滑aggregate rank load，是比任意替换expert更稳妥的方向。

### 限制

DVFS与reroute结果主要是实测数据标定的离线模拟，不是完整在线硬件控制；需与 `expert-hot/` 的大请求聚合trace共同解读。

---

## 10. `expert-hot/`：大规模请求 Expert 热度与跨请求关联校准

详见 [`data/expert-hot/doc.md`](data/expert-hot/doc.md)。

### 内容

- 官方 GSM8K 300 请求；原始顺序与固定shuffle各一遍，共600观测；
- 每请求完整 `[token,48 layers,Top-8]` routed expert IDs；
- 12个压缩NPZ raw分块，约23MB；
- `expert_layout.tsv`、每层expert/rank计数、Decode iteration×layer×rank、request-lag similarity、P/D对比、coactivation；
- 约98万行分析TSV和4张图。

### 结论

- 每层存在稳定hot experts；Prefill expert entropy 0.829、Gini 0.657，Decode entropy 0.849、Gini 0.628。
- 映射到rank后仍有约1.6×平均hot-rank负载，但比expert级平滑很多。
- P/D具体expert分布差异明显（mean JSD≈0.110），rank分布更接近（JSD≈0.010）。
- 相邻请求没有超出随机请求对的额外关联：expert excess similarity≈-0.0027，rank≈-0.00019；高raw similarity来自稳定层级hot experts，不是request adjacency。
- 同一请求在不同顺序下有较高层级重现性（mean similarity≈0.966）。
- 聚合300请求后，每个layer跨Decode iterations高度稳定：相邻iteration同层Pearson均值≈0.90、cosine≈0.984、hot-rank保持77.6%。
- 每层还有长期dominant rank：dominant rank成为hot rank的比例均值72.2%、中位75%；25/48层≥75%，16/48层≥87.5%，4层达到32/32。
- 稳定性主要是“同layer跨iteration”；相邻layer hot-rank保持仅16.8%。

### 含义

不应基于“前一个请求”预测“下一个请求”；更适合按 `semantic class × layer × decode iteration`建模，并用于layer-aware DVFS或慢时间尺度expert placement。

---

## 11. 跨目录总览结论

1. **负载粒度决定最佳并行与频率。** 小M由固定开销/静态功率主导，少active ranks、较低频率可能节能；大M由straggler/makespan主导，应扩大EP/TP并保持瓶颈rank高频。
2. **Routing不是单一skew标量。** 至少包含rank-level skew、rank内expert concentration、active-rank count、logical expert identity与长期placement。
3. **Latency最坏不等于Energy最坏。** Energy=`同步窗口×聚合功率`，部分偏斜可能同时保持较长窗口和较多active GPU。
4. **固定Top-K有清晰质量—成本曲线，动态Top-K仍缺kernel支持。** Top-6是当前最合理边界；真正动态K需要原生ragged FusedMoE。
5. **Per-rank DVFS主要价值在紧SLO。** Uniform无法整体降频时，非瓶颈rank仍有slack；但跨层热点轮换要求layer-aware控制或稳定的expert placement。
6. **大请求聚合揭示长期layer signature。** 请求邻接没有额外关联，但每层hot expert/rank跨iteration稳定，支持layer-aware慢控制。
7. **保语义优化优先于强制reroute。** 首选EPLB副本/placement和physical replica选择；只在Top-C候选内、质量预算约束下做少量reroute。

## 12. 推荐使用顺序

- Kernel机理：`kernel-EP/`
- 完整FFN纯EP：`EP/`
- 混合EP/TP：`mix-EP-TP/`
- 负载倾斜、active ranks、DVFS：`hot-EP/`
- Router Top-K质量/成本：`Top-k/`
- 大规模自然路由统计：`expert-hot/`
- 路由塑形与DVFS控制：`router-dvfs/`
- Attention/FFN TP对照：`TP/`
- 专用/真实A2A Pilot：`DeepEP/`

分析新结果时，先确定测量边界、GPU budget、frequency、phase、M、routing和A2A backend，再选择可比较的数据目录。
