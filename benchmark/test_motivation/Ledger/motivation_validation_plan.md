# Hybrid Scheduling Motivation Validation Plan

> 目标：先用最少机制改动判断两个 topic 是否存在足够强的现象，再决定是否实现 scheduler。
> 当前实验环境：Kimi-Linear-48B-A3B，TP=2，GPU 6/7，separate state/KV pools，`max_running_requests=97`，`max_mamba_cache_size=485`。容量规划为 5 state slots/request，活动请求物理常驻通常为 3 slots/request。

## 1. 本轮只回答两个问题

### M-A：lifetime 是否真的改变请求选择价值

固定同一个 request multiset，只改变到达顺序：

- `short-first`：短生命周期请求先进入 FCFS；
- `long-first`：长生命周期请求先进入 FCFS；
- `interleave`：按两个 group 的相对进度均匀交错。

如果三种顺序的结果几乎相同，说明当前 state capacity 下没有足够大的 request-selection gap，Topic A 暂时不值得实现复杂 predictor。

### M-B：equal-token workload 是否产生不同 hybrid resource pressure

构造两个 input/output token 总量完全相同的 tenant-like groups：

| group | requests | input/request | output/request | total input | total output | state slots |
|---|---:|---:|---:|---:|---:|---:|
| state-heavy | 96 | 1,024 | 32 | 98,304 | 3,072 | 480 |
| KV-heavy | 6 | 16,384 | 512 | 98,304 | 3,072 | 30 |

两组的 token service 和最终 KV token 总量相同，但并发时的 request/state demand 相差 16 倍。总请求数 102 又略高于当前 97-request state cap，因此谁先到达会决定谁先获得 scarce state slots。

如果这种差异既不产生明显资源占用差异，也不产生 group-level TTFT/JCT interference，Topic B 只能作为记账观察，不能支撑公平调度论文。

## 2. 阶段 0：实验前校验

每次运行必须保存 `/get_server_info`，并检查：

- `schedule_policy=fcfs`；
- `enable_unified_memory=false`；
- `effective_max_running_requests_per_dp=97`；
- `max_mamba_cache_size=485`；
- 模型、git version、state dtype 和 radix strategy 未变化。

请求使用唯一 `cache_salt`，避免 prefix reuse 混入结果；`ignore_eos=true`，保证真实 output length 等于指定长度。每个 condition 运行 1 次 warm-up 和至少 3 次正式重复，正式运行前调用 `/flush_cache`。

每次进程运行还必须使用唯一 `run_id` 构造 request ID。不能跨 condition 复用 `rid`，因为 scheduler、detokenizer 和 abort 消息是异步清理的；立即复用可能让新请求受到旧请求终止消息影响。harness 默认自动生成唯一 `run_id`，正式结果建议显式传入 `--run-id <condition>-r<replicate>`。

## 3. 实验 M-A：Lifetime Order Sensitivity

### 3.1 Workload

| group | requests | input/request | output/request | 角色 |
|---|---:|---:|---:|---|
| short | 96 | 1,024 | 32 | 快速完成并释放 state slots |
| long | 24 | 1,024 | 512 | 长时间持有 state slots |

总请求数 120，大于 97-request state cap；总 token 数远低于 1.46M KV capacity。因而首要 admission constraint 应是 state slots，而不是 KV pool。

### 3.2 Conditions

```bash
python benchmark/test_motivation/Ledger/validate_hybrid_motivation.py \
  --mode lifetime-order --order short-first --run-id lifetime-short-first-r1 \
  --output results/lifetime_short_first_r1.json

python benchmark/test_motivation/Ledger/validate_hybrid_motivation.py \
  --mode lifetime-order --order long-first --run-id lifetime-long-first-r1 \
  --output results/lifetime_long_first_r1.json

python benchmark/test_motivation/Ledger/validate_hybrid_motivation.py \
  --mode lifetime-order --order interleave --run-id lifetime-interleave-r1 \
  --output results/lifetime_interleave_r1.json
```

三种 condition 的 request multiset 必须完全一致。`arrival-gap-ms`、`boundary-gap-ms`、SLO 和 server process 在一组重复中保持不变。

### 3.3 Primary metrics

- short group 的 p50/p95 TTFT 和 JCT；
- 固定 deadline 内完成的 short requests；
- 总 SLO goodput；
- `num_retractions`；
- decode state-slot-seconds 和 KV-token-seconds lower-bound proxies。

### 3.4 Motivation gate

Topic A 进入下一阶段需同时满足：

1. `short-first` 相对 `long-first` 的 short-request p95 JCT 至少改善 20%，或固定 SLO goodput 至少改善 15%；
2. 差异在 3 次重复中方向一致，95% bootstrap CI 不跨 0；
3. 总完成 token 数相同，不是通过少做工作取得收益；
4. 结果与 state-cap 饱和时间相关，而不是 cache hit 或请求失败造成。

这一步只能证明“选择顺序有价值”，不能证明 hybrid-specific novelty，因为 SJF 也可能解释结果。

当前 gate 为 **FAIL/inconclusive**，原因不是平均改善为零，而是：三次 paired 改善分别约为 -14.9%、24.3%、24.0%，存在一次方向反转；bootstrap CI 跨 0；20% effect-size gate 和 SLO-goodput gate 都未稳定通过。即便扩大重复后 11.1% 均值变得显著，它仍可能只是已有 SJF 效应，必须再通过 state-cap sensitivity 与 resource-time-vs-SJF cross case 才能形成 hybrid-specific claim。

### 3.5 Topic A 的第二个 kill test

只有 3.4 通过后才做：

1. 把 SSM state 改为 BF16，使 state capacity 近似增大；
2. 或保持 dtype，增大/减小 `--max-mamba-cache-size`；
3. 重复三种 arrival order；
4. 加一个 `shortest-remaining-output` oracle replay，以及考虑 input/prefill、state-time 和 KV-time 的 resource-time oracle replay。

需要观察到：order sensitivity 随 state capacity 放宽而减弱，并且 resource-time oracle 在至少一个 input/output cross case 上优于纯 SJF。否则 Topic A 只是已有短作业优先结论，不足以独立成篇。

## 4. 实验 M-B：Equal Tokens, Unequal State Pressure

### 4.1 Conditions

```bash
python benchmark/test_motivation/Ledger/validate_hybrid_motivation.py \
  --mode fairness-pair --order state-first --run-id fairness-state-first-r1 \
  --output results/fairness_state_first_r1.json

python benchmark/test_motivation/Ledger/validate_hybrid_motivation.py \
  --mode fairness-pair --order kv-first --run-id fairness-kv-first-r1 \
  --output results/fairness_kv_first_r1.json

python benchmark/test_motivation/Ledger/validate_hybrid_motivation.py \
  --mode fairness-pair --order interleave --run-id fairness-interleave-r1 \
  --output results/fairness_interleave_r1.json
```

还需分别单独运行 state-heavy 和 KV-heavy，得到 isolated latency。可以通过把另一组 count 设为 0 完成；harness 已通过真实 zero-count smoke/正式运行验证。

### 4.2 Primary metrics

- 两组完成的 input/output token 数，确认 token service 相同；
- peak decode state slots lower bound；
- decode state-slot-seconds 和 KV-token-seconds lower bounds；
- 各组 p50/p95 TTFT、JCT 和 SLO goodput；
- co-run slowdown：`co-run p95 / isolated p95`；
- arrival order reversal 后，受害 group 是否交换。

注意：黑盒 timeline 不知道 scheduler 内部的精确 admission 时刻。脚本从 first token 到 completion 积分，只是 decode-residency lower bound。若 M-B 通过，下一阶段必须在 allocator allocate/free 路径加入精确 tenant/request timestamps。

### 4.3 Motivation gate

Topic B 进入 scheduler 实现阶段需同时满足：

1. 两组完成 token service 误差小于 1%；
2. state-heavy 的 peak decode state-slot proxy 至少是 KV-heavy 的 8 倍；理论配置值为 16 倍；
3. 至少一个 group 的 co-run p95 TTFT 或 JCT slowdown 大于 1.5x；
4. `state-first` 与 `kv-first` 能显著改变弱势组的 slowdown，证明 scarce-state acquisition 会造成跨组干扰；
5. 三次重复方向一致，且无 request failure、cache reuse 或 retraction anomaly 解释该现象。

这一步只证明 token-equivalent workloads 的 physical pressure 不同。它还不能证明 VTC 失败。

### 4.4 Topic B 的第二个 kill test

只有 4.3 通过后才做：

1. 把 group label 作为可信 tenant ID 传入 scheduler；
2. 加精确 state-slot-time、KV-byte-time 和 GPU-time accounting；
3. 比较 FCFS、per-tenant RR、VTC、调优 scalar-cost VTC 和 vector dominant-resource debt；
4. 扫描 state-cap、KV-cap 和瓶颈切换；
5. 验证 request splitting 是否能从 VTC 获益，以及 vector accounting 是否消除该收益。

只有 tuned VTC 仍出现明显 service lag，而 vector debt 在吞吐损失小于 5% 时显著改善公平，Topic B 才足以独立成篇。

### 4.5 第二阶段当前进度（2026-08-20）

- [x] 受控实验 tenant label 贯穿到 scheduler；当前借用 `extra_key`，且每请求使用唯一 `cache_salt`。生产实现仍须从认证主体产生 tenant ID，不能信任 caller 自报字段。
- [x] scheduler-side active state-slot-time 与 KV-token-time 精确积分；inactive radix cache 明确排除。
- [ ] GPU-time accounting；当前不能把 scheduler wall time 当作精确 GPU time。
- [x] FCFS、标准 token VTC、两个 scalar request-cost VTC、state/KV vector debt 的 sustained-arrival 对照，各 3 次重复。
- [x] 9-cell state-cap/KV-cap sweep、7 个公共 scalar weights，并对边界最优 cell 扩展 8 个 weights 直至 oracle 拐点。
- [ ] per-tenant RR、request splitting、fairness-efficiency frontier；因 vector claim 已过 kill condition，暂不执行。

这一段记录的是当时的 **single-capacity PARTIAL PASS**：vector dominant-debt gap 为 0.82，最优 tested scalar VTC 为 1.41，改善 41.6%；vector makespan 相对 FCFS 没有损失而是改善 4.4%。它已被 4.7 的 dense scalar/capacity sweep 推翻，不能再作为当前结论引用。

### 4.6 state-cap x KV-cap x scalar-weight 交叉实验（2026-08-21）

#### 这组实验主要做什么

保持模型、精度、TP、请求 multiset、arrival trace 和 tenant 权重不变，只改变三个量：

1. `--max-mamba-cache-size`：改变 recurrent-state pool 的物理 slot capacity；
2. `--max-total-tokens`：改变 full-attention KV pool 的 token capacity；
3. `SGLANG_TENANT_VTC_REQUEST_WEIGHT` 的等价运行时实验值：改变 scalar VTC 对每个请求固定 state overhead 的定价。

容量网格为：

| dimension | values | 含义 |
|---|---|---|
| state cap | 240, 480, 720 slots | 约对应 48、96、144 个可并发请求（extra-buffer 规划按 5 slots/request） |
| KV cap | 196,608；393,216；786,432 tokens | 从 KV 紧张到对当前 trace 基本宽松 |
| scalar request weight | 0, 1,024, 2,048, 4,096, 8,192, 16,384, 32,768 | 覆盖 token-only 到强 state-price 的几何扫描 |

每个 `state x KV` cell 都跑 FCFS、7 个 scalar VTC 和 vector debt。screening 先跑一次；只有通过 kill test 的 cell 和相邻 scalar weights 才扩成 3 次重复。这样完整 screening 是 9 个 server capacity 配置、81 个 trace run，而不是在尚未确认瓶颈切换前直接做全部重复。

#### 目的不是只看 vector 是否赢

这组实验区分三个强度不同的结论：

1. **固定 scalar 不稳健**：同一个 scalar weight 无法跨 state-bound、mixed、KV-bound cell 保持公平，但每个 cell 单独重调 weight 可以追平 vector。此时只能声称 vector 减少容量相关调参，不能声称 scalar 表达能力不足。
2. **stationary cell 内也不可约**：即使每个 cell 都从完整 weight sweep 选取 throughput-bounded oracle scalar，vector 仍在至少两种 bottleneck regime 显著更公平。这个结果才支持多维 debt 不能被单一固定价格稳定表达。
3. **没有必要使用 vector**：per-cell oracle scalar 在公平和效率上匹配或优于 vector。此时 Topic B 的 vector-policy claim 应停止，最多保留 hybrid resource accounting/characterization。

#### 指标与判定

- 容量校准先看系统级同时峰值：`peak active requests / effective max running >= 0.9` 表示 state admission 饱和，`peak active KV / KV cap >= 0.9` 表示 KV 饱和；据此标记 state-bound、KV-bound、dual-bound 或 underloaded。另报告 normalized pressure ratio `((total state-slot-time) / state-cap) / ((total KV-token-time) / KV-cap)`，但累计压力较大本身不等于资源已经饱和。
- 公平主指标使用 scale-free relative dominant-debt gap：
  `|d_state - d_KV| / mean(d_state, d_KV)`。原 absolute gap 随容量尺度变化，不能直接跨 cell 比较。
- scalar oracle 必须满足 mean makespan 不超过同 cell FCFS 的 5%，再从候选中选择 relative gap 最小者。
- 同时报告每 tenant p95 JCT、20s SLO goodput、retraction、makespan，以及固定 scalar 相对 per-cell oracle 的 regret。
- screening gate：vector 相对 per-cell oracle scalar 的跨 cell 平均 relative gap 至少改善 10%，并在至少两个不同 bottleneck regime 的至少两个 cell 中分别改善 10%；自身 makespan 仍不得超过 FCFS 5%。通过后才做三次重复与置信区间；小于 10% 的单次差异按噪声处理，不能用于触发重复或论文结论。

实现入口：

- `run_capacity_scalar_sweep.sh`：`calibrate` 或 `screen`；每个容量点只加载一次模型，在 scheduler idle 时重置实验 policy/counter；退出时恢复默认 FCFS 服务；
- `analyze_capacity_scalar_results.py`：校准瓶颈、选择 per-cell oracle、计算跨 cell fixed-weight regret，并执行 kill decision。

### 4.7 交叉实验结果与决策（2026-08-21）

结果见 [capacity-scalar report](results/capacity_scalar_20260821/capacity_scalar_report.md)：

| check | result |
|---|---:|
| capacity cells | 9（6 state-bound、2 dual-bound、1 KV-bound） |
| artifacts | 89；每条 204/204 requests 完成 |
| total retractions | 3；均来自 KV-bound cell 的 scalar policy |
| per-cell oracle mean relative gap | 0.544 |
| vector mean relative gap | 0.579（相对 oracle 退化 6.4%） |
| vector >=10% wins | 1/9 cell，且只有 state-bound 一个 regime |
| best fixed scalar | weight 0，mean gap 0.623 |

最终判定为 **VECTOR CLAIM FAIL**。固定 scalar 的最优 weight 确实随容量点显著变化，说明存在 tuning sensitivity；但允许每个 cell 调优后，scalar 在 6/9 cell 优于 vector，跨 cell 平均也优于 vector。当前数据不能支持“二维 vector debt 是必要表达能力”的 claim。

不继续三次重复：预注册 screening gate 要求 vector 跨 cell 平均至少改善 10%，并在两个 bottleneck regime 分别改善至少 10%；当前不仅未通过，而且总体方向相反。为一个失败 screening 增加重复只会提高置信度，不会补足 effect size。

若保留 Topic B，下一步应另立更窄的 dynamic-adaptation gate：在一个 server capacity 下让 arrival mix 分阶段从 state-heavy 切到 KV-heavy 再切回，禁止 scalar 中途重调。它验证的是 vector 是否避免在线 retuning，而不是 scalar 是否原则上无法表达。该 gate 通过前，不投入 GPU-time、request splitting 或生产化 tenant identity。

### 4.8 动态请求成本与 state-lifetime motivation 实验（2026-08-21）

#### 实验矩阵

统一 driver 为 `validate_dynamic_cost_motivation.py`，每个请求保存 TTFT/JCT、逐 token timeline、实际 `cached_tokens`、retraction，以及 scheduler-side state-slot-time/KV-token-time。

| experiment | conditions | repetitions | 回答的问题 |
|---|---|---:|---|
| cost surface | `P={1K,4K,16K}`、`O={32,512}`、`N={1,8,16}`、reuse `{0,90%}` | 3 | 原始长度、输出 lifetime、并发与 reuse 如何共同决定成本 |
| rank inversion | state-vs-KV、short-total-vs-long-output、short-unique-vs-long-reused；三种 order | 3 | 一个简单排序是否能稳定转化为 JCT/SLO 收益 |
| batch externality | no candidate、short-unique、long-unique、long-reused | 5 | candidate 的增量成本如何扰动已有 decode batch |
| cache lifetime | no pressure、state-heavy pressure、KV-heavy pressure | 5 | 相同 cached state 在不同资源压力下是否被驱逐并产生重算 |
| phase calibration | state phase、KV phase；FCFS、6 个 scalar weights、vector | 1 screening | 最优 scalar 是否产生足够大的 phase-specific migration/regret |

复用实验必须让 warmup 与被测请求使用相同 tenant `extra_key` 和 `cache_salt`，否则 RadixCache key 不同会产生假 miss。warmup 完成后重置 resource tracker，使预热成本不进入正式账本。eviction 实验使用 `state=480, KV=196,608`：80 条 `1K+512` 构成 state pressure，16 条 `16K+32` 构成 KV pressure；两类压力不是 equal-token fairness 对照，而是用于隔离不同 pool 对 cache lifetime 的因果作用。

#### 预注册 gate 与结果

结果见 [dynamic request-cost report](results/dynamic_cost_20260821/dynamic_cost_report.md)：

| gate | threshold | result | decision |
|---|---|---|---|
| non-monotone request cost | 至少一个 raw-length ranking 反转，并由资源账本解释 | `1K+512` JCT 2.608s，`16K+32` 1.218s | PASS |
| reuse-sensitive incremental cost | reused 与 unique 的成本/外部性稳定分离 | 并发 `16K+32` p95 JCT 下降 76.4%；long-reused background penalty 13.2%，long-unique 70.4% | PASS |
| batch externality | background p95 或 SLO 至少退化 15%，5 次方向一致且 CI 不跨 0 | long-unique p95 增加 70.4%，95% CI `[69.1%,71.5%]`；SLO goodput 5/5 为 0 | PASS |
| cache lifetime | 至少一种压力稳定驱逐、另一种不驱逐，并产生明显重算 | KV pressure 5/5 全驱逐，TTFT 25.2x；state pressure 5/5 全保留 | PASS |
| simple order benefit | overall p95 JCT 改善至少 15%，或 SLO goodput 改善 15% | 最佳 lifetime order 仅 5.9%；reuse 的预期方向反转 | FAIL |
| dynamic scalar regret | phase oracle weight 至少相差 4x，且任一固定 weight regret 至少 15% | label 有迁移但全部 latency-gap 差异小于 4% | FAIL；不触发 switch confirm |

#### 决策

motivation 已证明：请求成本必须区分 output lifetime、uncached prefill、resident KV、固定 state、cache state 的未来存活和对当前 batch 的扰动。它没有证明：按某个单一 score 排序就能显著改善系统目标。尤其是 long-reused 请求虽然 prefill 便宜，运行时仍保留完整长序列 KV；因此 reuse-aware 与 lifetime/KV-aware 决策不能合并成“cached tokens 越多优先级越高”。

下一步只允许进入一个窄的机制 kill test：离线 oracle 使用真实 future resource-time 与 eviction/recompute cost，在完全相同的 feasibility/admission budget 下比较 FCFS、SJF/SRPT、LPM 和 future-memory baseline。oracle 若不能在至少两个负载区间把 p95 JCT 或 SLO goodput 提升 15%，则停止 Topic A 的 scheduler 实现。Topic B 的 vector fairness 不与该实验绑定，维持 FAIL。

## 5. 结果解释矩阵

| M-A | M-B | 下一步 |
|---|---|---|
| 强 | 强 | 分别推进第二个 kill test，暂不决定拆稿 |
| 强 | 弱 | 聚焦 lifetime-aware admission/retraction |
| 弱 | 强 | 聚焦 hybrid tenant fairness；内部先用简单 RR/FCFS |
| 弱 | 弱 | 停止 scheduler 实现，把工作降为 resource characterization |

## 6. 当前 harness 的证据边界

`validate_hybrid_motivation.py` 与 `validate_dynamic_cost_motivation.py` 可以证明：

- 相同 request multiset 的 arrival-order sensitivity；
- equal-token groups 的 observable decode residency 差异；
- group-level interference、retraction 和 SLO 差异。
- 原始长度、输出 lifetime、prefix reuse 与并发共同形成的请求成本面；
- candidate 对已有 batch 的 JCT/ITL/SLO 外部性；
- 在受控 pool pressure 下，prefix eviction 导致的 cached-token loss 和重算 TTFT。

它不能直接证明：

- state slot 的精确 allocate/free 时刻；
- 等待队列与 prefill residency 的分离；
- VTC 或 DRF 的失败；
- proposed scheduler 的收益。
- cache state 的线上未来存活概率；当前只观察受控压力后的结果。

这些限制是有意保留的：第一轮只决定 motivation 是否值得继续，不提前实现论文机制。

## 7. 五个 lifecycle-cost gate 的执行结果（2026-08-25）

本轮执行了 [预注册协议](lifecycle_gate_experiment_protocol.md) 中的五个 gate。实验入口为 [gate harness](validate_lifecycle_gate_motivation.py) 和 [A800 runner](run_lifecycle_gate_experiments.sh)，汇总器为 [gate analyzer](analyze_lifecycle_gate_results.py)，完整输出见 [gate report](results/lifecycle_gates_20260825/lifecycle_gate_report.md)。

### 7.1 完整性审计

- M1 54 个 artifacts，M2 18 个，M3 12 个，M4 12 个，M5 24 个，共 120 个；
- 共发出 2,898 个正式请求，失败数为 0；
- M1 使用 `state=480, KV=393216`；M2 分别使用预注册的 state/KV/compute 三种容量；M3/M4/M5 使用 `state=96, KV=196608`；
- 所有 artifact 的模型路径均为 `/models/Kimi-Linear-48B-A3B`；
- 实验在 A800 GPU 6/7 上执行，结束后恢复 full-capacity FCFS server，`/health` 返回 200。

### 7.2 Gate 判定

| gate | decision | measured result |
|---|---|---|
| M1 token/length ranking | **PASS** | 44/324（13.6%）可比较 pair 出现超过 5% 的反转；最大稳定反转 10.06x |
| M2 state-dependent ranking | **FAIL** | 最佳顺序跨 state/KV regime 反转，但最佳固定顺序 worst-regime regret 只有 2.0% |
| M3 cross-layer coupling | **FAIL** | reuse-first p95 gain 3.3%，cached-token delta 1,824，相对 pressure-first LPM 的 residual gain 4.6% |
| M4 equal-token cost transfer | **PASS** | 每租户 33,792 tokens；最大 per-token resource ratio 6.60x；VTC slowdown gap 101.5 个百分点 |
| M5 empirical oracle headroom | **FAIL** | 24-order oracle 相对最强预注册 baseline 的 p95 headroom 2.5%，低于 15% |

### 7.3 Kill decision

M5 是当前统一效率机制的最终 kill test。受限 oracle 都无法达到 15% headroom，因此停止实现 lifecycle request-ranking、动态 scarcity pricing 和 admission/eviction 联合控制器。M2/M3 的独立失败与该决策一致，不应通过扩大 predictor 或增加更多 score 项来回避。

M4 将公平问题从“资源向量不同”推进为“token VTC 下出现稳定的用户 slowdown 差异”，但它不能复活已经失败的 vector-debt policy。后续若继续，只进入 cost-denominated fairness 的独立机制 gate：在完全相同 workload/capacity 下比较 token VTC、每 cell tuned scalar VTC 和 realized lifecycle-cost oracle，并要求公平改善至少 15%、makespan 损失不超过 5%。

## 8. Cost-denominated fairness oracle 结果（2026-08-25）

协议、实现和完整结果分别见 [fairness oracle protocol](lifecycle_fairness_oracle_protocol.md)、[fairness runner](run_lifecycle_fairness_oracle_experiments.sh) 和 [fairness oracle report](results/lifecycle_fairness_oracle_20260825/lifecycle_fairness_oracle_report.md)。

### 8.1 校准选择

- workload 为 8 waves；每 wave split tenant 提交 16 个 `1K+32` 请求，packed tenant 提交一个 `16K+512` 请求，两侧均为 16,896 raw tokens；
- scalar VTC 的合法候选包含 request weight 0。最优 scalar 为 0，说明增加固定 request charge 无法修复该 mix，不能人为排除 0 后声称 tuned scalar 更弱；
- realized lifecycle-cost 扫描 `state weight=0..16, KV weight=1`，在 makespan 不超过 token VTC 5% 的候选中选择 slowdown gap 最小的 `state=0.25,KV=1`；
- 权重在 confirmation 之前冻结，confirmation 不参与选参。

### 8.2 确认结果

| metric | token VTC | tuned scalar VTC | realized-cost oracle |
|---|---:|---:|---:|
| split normalized slowdown | 958.6% | 943.6% | 218.6% |
| packed normalized slowdown | 25.3% | 24.5% | 55.8% |
| slowdown gap | 933.3% | 919.1% | 162.8% |
| makespan | 25.010s | 24.833s | 24.229s |
| 30s SLO goodput | 136 | 136 | 136 |
| retractions | 0 | 0 | 0 |

realized-cost oracle 相对 token/tuned-scalar 的 gap 改善分别为 82.6%/82.3%，三次方向一致，makespan 相对 token VTC 改善 3.1%，因此 fairness mechanism gate 为 **PASS**。确切资源账本也显示 policy 改变了运行组合：packed tenant 的 KV-token-time 从 token VTC 的 1,483,036 降至 1,263,367，而 split tenant 的 state-slot-time 从 271.994 增至 283.038。

### 8.3 决策边界

这项结果支持 cost-denominated fairness，不支持 vector dominant-debt 的必要性，也不支持已被 M5 否定的 lifecycle request-ranking efficiency。realized policy 使用执行中已经发生的精确 resource-time，不需要 output-length predictor；“oracle”仅指 `0.25:1` 的价格由 calibration outcome 选择。

下一步只验证价格泛化：冻结 `0.25:1`，在未参与校准的 state/KV capacity、split ratio、output length 和 arrival rate 上测 fixed-price regret，并与每个 workload 单独调优的 oracle 比较。固定价格若在这些 held-out cells 内均低于 15% regret，就不应增加 online dynamic pricing；只有稳定超过 15% 时，动态价格才成为必要技术。

## 9. Held-out lifecycle fairness 泛化结果（2026-08-25）

第 8 节提出的全部泛化实验已经完成。协议、runner、analyzer 和逐 artifact auditor 分别为 [generalization protocol](lifecycle_fairness_generalization_protocol.md)、[runner](run_lifecycle_fairness_generalization_experiments.sh)、[analyzer](analyze_lifecycle_fairness_generalization.py) 和 [auditor](audit_lifecycle_fairness_generalization.py)。完整分析见 [generalization report](results/lifecycle_fairness_generalization_20260825/lifecycle_fairness_generalization_report.md)，完整性见 [audit](results/lifecycle_fairness_generalization_20260825/audit.md)。原始 626 个 artifacts 保存在远端 `/mnt/nvme0/mllm/sglang/benchmark/test_motivation/Ledger/results/lifecycle_fairness_generalization_20260825`，本地只同步分析产物。

### 9.1 Workload 与 baseline

- 10 个 equal-input/equal-output-token cells；扫描 split ratio、prompt/output shape、arrival interval、state cap 和 KV cap；
- screening：4 waves、1 repetition、200 artifacts；
- confirmation：8 waves、3 repetitions、276 artifacts；
- strongest-baseline kill confirm：8 waves、3 个全新 repetitions、150 artifacts；
- baselines：FCFS、request VTC、token VTC、6 个非零 request-tax 加零 tax、instantaneous dominant share、vector debt、7 个 lifecycle state/KV prices、冻结 `0.25:1` 和 per-cell oracle；
- slowdown 由每个 cell、每个 repetition 的 matching split-only/packed-only isolation 归一化；primary metric 是两个 tenant 的 absolute p95 normalized-slowdown gap。

### 9.2 完整性审计

- 626 artifacts、64,152 requests、0 failure、0 retraction；
- 626/626 exact scheduler traces；
- 626/626 resolved schedule policy 与 request/input/output/lifecycle 权重精确匹配；
- 486/486 mixed artifacts 中 split 与 packed tenant 的 input tokens 和 output tokens 分别严格相等；
- 全部 artifact 记录物理 GPU `6,7`，硬件为两张 NVIDIA A800-SXM4-80GB；
- 每 cell 的 screen/confirm/kill 数量与预注册矩阵完全一致。

### 9.3 Gate 判定

第一轮 confirmation 中，`instantaneous share` 在 10/10 cells 成为 strongest confirmed non-lifecycle baseline。因为这是从 confirmation 数据发现的，不能直接用于同数据判定；因此在查看新结果前固定比较 `token VTC vs instantaneous share vs lifecycle-s0.25-k1`，执行独立 kill confirm。

| metric | independent kill result | gate |
|---|---:|---:|
| mean slowdown-gap improvement over instantaneous share | 4.0% | >=15% |
| cell bootstrap 95% CI | [-24.7%, 27.3%] | supporting |
| paired directions | 7/10 | >=8/10 |
| mean makespan change vs token VTC | +0.4% | <=+5% |

最终 decision 为 **FAIL**。吞吐 gate 通过，但 effect size、方向数和置信区间均不支持固定 lifecycle price 的稳定优势。第 8 节的 82.6% 单 workload PASS 仍是有效测量，却不再能作为机制泛化证据。

### 9.4 Topic 终止边界

当前停止以下实现：lifecycle predictor、request-ranking scheduler、online scarcity price、fairness-aware admission/eviction 和更密的固定权重搜索。原因不是实验不足，而是 strongest simple baseline 已消除大部分可用 headroom。

继续研究需要一个新的、可证伪机制假设，能够在不从 test cells 选参的条件下解释 arrival-rate reversal，并在全新 traces 上稳定优于 instantaneous share。若没有这个新假设，Topic A/B 的合稿与拆稿都只够做 characterization，不够做 scheduler paper。

## 10. VTC state-tax observation 与因果边界（2026-08-26）

预注册协议、runner 和报告分别见 [protocol](vtc_state_tax_observation_protocol.md)、[runner](run_vtc_state_tax_observation_experiments.sh) 和 [report](results/vtc_state_tax_20260826/vtc_state_tax_observation_report.md)。实验禁用跨请求 prefix reuse，只使用物理 A800 GPU 6/7、TP=2 和 `/models/Kimi-Linear-48B-A3B`。

### 10.1 实验与 baseline

- 27 个 isolation profile artifacts 拟合非负六项 cost `r+aP+bP^2+cO+dPO+eO^2`，fit error gate 通过；
- 9 个 cell 正交改变 split factor `1/4/16/32`、state/KV capacity 与 wave interval；
- 每个 mixed wave 的 split/packed tenant 都严格完成 16,384 input 和 512 output tokens；
- screening 比较 FCFS、request VTC、token VTC、6 个 request tax、profiled VTC 和 instant-share；
- confirmation 使用全新 run IDs/salts，比较冻结的 global/per-cell tax、profiled VTC 与 instant-share，每格 3 次并反转 rep2 tie order。

完整性为 387 artifacts、34,107 requests、0 failure、0 cached token、0 retraction；所有 matrix count、模型、容量、GPU metadata、resolved policy/weights 与 exact traces 通过审计。

### 10.2 Gate 结果

| gate | decision | measured result |
|---|---|---|
| O1 VTC abstraction mismatch | **PASS** | f16/f32 equal-token 标准 VTC 的 resource ratio 17.50x/33.69x、composition ratio 15.81x/31.62x、slowdown gap 913.2%/924.6% |
| O2 hybrid state specificity | **FAIL** | state-relaxed gap 未降低而是从 913.2% 到 988.0%；KV-relaxed 为 915.0%；f1 control 1.2% |
| O3 scalar repair insufficiency | **PASS** | instant-share 相对 strongest scalar mean improvement 43.8%，95% CI `[21.7%,65.5%]`，8/9 wins，makespan -1.2% |

### 10.3 结论与下一步 gate

这组实验允许把“短请求更贵”融入 motivation，但只能使用精确定义：更多短请求在相同 aggregate tokens 下支付更多固定 recurrent-state allocations，因而改变 `(active state slots, resident KV tokens)`、admission concurrency 和 batch composition。它不表示单个短请求的总 footprint 大于单个长请求。

state relaxation 把 f16 split queue p95 从 10.69s 降到 2.68s，证明 fixed tax 对 admission/queue 有实际影响；但 normalized fairness gap 没有下降，因为更高并发同时改变 batch interference 与 isolation denominator。因此不再推进“state-only fairness scheduler”，也不把 O1/O3 的 PASS 用来覆盖 O2 的 FAIL。

若继续实现机制，只允许一个新的 kill test：冻结一个保留 VTC long-term service bound、同时读取 instantaneous state/KV scarcity 和 batch-interference proxy 的 policy，与标准 VTC、profiled VTC 和 instant-share 在全新 arrival/capacity cells 上比较。必须相对 instant-share mean slowdown-gap 改善至少 15%、至少 8/10 cells 同方向、makespan 不超过 token VTC 5%；否则 topic 停留在 resource/fairness characterization。
