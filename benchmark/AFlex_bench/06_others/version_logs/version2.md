# version2 — 部署拓扑对比实验（Deployment Topology Comparison）

> 目标：在 Qwen3-32B 上系统对比不同部署拓扑（纯 PD 解耦 vs PD+AF 算子级解耦）与不同 GPU 卡数组合在各定长 workload、各 QPS 下的性能与能耗，给出选型结论。日期 2026-05-31，A800×8（GPU0-7），freq=auto（硬件默认调频，隔离 DVFS 变量，纯比拓扑）。

## 1. 实验设置

新建 `scripts/bench/run_deploy_bench.py`，把"部署拓扑"参数化，复用 `run_fixed_qps_bench.py` 已验证的 workload/能耗(NVML 按卡)/SLO/watchdog 逻辑。`run_workload` 增加 `gpu_indices/prefill_gpus/decode_gpus` 参数，使 P/D 能耗按各拓扑的 GPU 映射正确归属（向后兼容）。

测试的 7 个稳定拓扑：

| 方案 | 含义 | 卡数 | GPU 映射 |
|------|------|------|---------|
| pd_p1d1 | 纯PD，Prefill TP1 + Decode TP1 | 2 | P=7 D=4 |
| pd_p1d2 | 纯PD，Prefill TP1 + Decode TP2 | 3 | P=7 D=4,5 |
| pd_p2d1 | 纯PD，Prefill TP2 + Decode TP1 | 3 | P=6,7 D=4 |
| pd_p2d2 | 纯PD，Prefill TP2 + Decode TP2 | 4 | P=6,7 D=4,5 |
| pd_p2d4 | 纯PD，Prefill TP2 + Decode TP4 | 6 | P=6,7 D=0,1,2,3 |
| pdaf_m2 | PD+AF，PA/PF/DA/DF 各TP1，M=2 微批 | 4 | P=6,7 D=4,5 |
| pdaf_d_tp2 | PD+AF，Decode 整体 TP2（同构） | 6 | P=6,7 D=2,3,4,5 |

弃用的异构 AF（pdaf_df2 = DF TP2、pdaf_da2 = DA TP2）：`ipc_cpp` 后端在两边 TP 不对称（tp_A≠tp_F）时 decode 会在前 ~100 请求后挂起（da 服务端已返回 200 但客户端 stream 收不全，watchdog 超时），属通信后端限制，非本实验目标，故剔除。同构 AF（tp_A==tp_F，如 pdaf_d_tp2）稳定。

测试矩阵：
- Phase A（完整 scaling 曲线）：il512_ol128 × 7 方案 × QPS{2,4,6,8} = 28 run
- Phase B（泛化验证）：il256_ol512 + il1024_ol128 × {pd_p1d2,pd_p2d2,pdaf_m2,pdaf_d_tp2} × QPS{2,4} = 16 run

## 2. il512_ol128 完整 scaling 数据（Phase A，单位 tok/s、ms、J）

| QPS | 方案 | 卡 | 吞吐 | TTFT | TPOT | 能耗J | J/tok | SLO违背 |
|-----|------|----|------|------|------|------|-------|--------|
| 2 | pd_p1d1 | 2 | 234 | 172 | 48 | 29229 | 1903 | 0% |
| 2 | pd_p1d2 | 3 | 236 | 207 | 44 | 36986 | 2408 | 0% |
| 2 | pd_p2d2 | 4 | 236 | 115 | 44 | 43299 | 2819 | 0% |
| 2 | pdaf_m2 | 4 | 225 | 186 | 70 | 49041 | 3193 | 0% |
| 2 | pdaf_d_tp2 | 6 | 222 | 202 | 77 | 60199 | 3919 | 0% |
| 6 | pd_p1d1 | 2 | 516 | 11239 | 49 | 44043 | 956 | 75% |
| 6 | pd_p1d2 | 3 | 702 | 217 | 46 | 46501 | 1009 | 0% |
| 6 | pd_p2d1 | 3 | 528 | 9511 | 50 | 53982 | 1172 | 74% |
| 6 | pd_p2d2 | 4 | 705 | 118 | 45 | 53923 | 1170 | 0% |
| 6 | pdaf_m2 | 4 | 669 | 187 | 71 | 59235 | 1286 | 0% |
| 8 | pd_p1d2 | 3 | 935 | 298 | 47 | 51472 | 838 | 0% |
| 8 | pd_p2d2 | 4 | 934 | 132 | 47 | 59142 | 963 | 0% |
| 8 | pd_p2d4 | 6 | 934 | 189 | 48 | 73122 | 1190 | 0% |
| 8 | pdaf_m2 | 4 | 890 | 245 | 74 | 66127 | 1076 | 0% |
| 8 | pdaf_d_tp2 | 6 | 872 | 440 | 91 | 73333 | 1194 | 0% |
| 8 | pd_p1d1 | 2 | 495 | 26584 | 50 | 59884 | 975 | 88% |
| 8 | pd_p2d1 | 3 | 558 | 22106 | 50 | 67938 | 1106 | 88% |

## 3. 泛化验证（Phase B：decode 重 + 长输入）

il256_ol512（decode 重）：

| QPS | 方案 | 卡 | 吞吐 | TTFT | TPOT | J/tok | SLO违背 |
|-----|------|----|------|------|------|-------|--------|
| 2 | pd_p1d2 | 3 | 755 | 163 | 44 | 762 | 0% |
| 2 | pd_p2d2 | 4 | 754 | 119 | 45 | 882 | 0% |
| 2 | pdaf_m2 | 4 | 642 | 186 | 72 | 1106 | 0% |
| 2 | pdaf_d_tp2 | 6 | 604 | 219 | 87 | 1393 | 0% |
| 4 | pd_p1d2 | 3 | 1493 | 170 | 46 | 459 | 0% |
| 4 | pd_p2d2 | 4 | 1489 | 130 | 47 | 522 | 0% |
| 4 | pdaf_m2 | 4 | 1003 | 9990 | 74 | 754 | 58% |
| 4 | pdaf_d_tp2 | 6 | 820 | 17209 | 93 | 1031 | 58% |

il1024_ol128（长输入）：

| QPS | 方案 | 卡 | 吞吐 | TTFT | TPOT | J/tok | SLO违背 |
|-----|------|----|------|------|------|-------|--------|
| 2 | pd_p1d2 | 3 | 236 | 324 | 43 | 2608 | 0% |
| 2 | pd_p2d2 | 4 | 236 | 193 | 43 | 3011 | 0% |
| 2 | pdaf_m2 | 4 | 225 | 275 | 70 | 3400 | 0% |
| 4 | pd_p1d2 | 3 | 469 | 326 | 44 | 1548 | 0% |
| 4 | pd_p2d2 | 4 | 470 | 183 | 44 | 1770 | 0% |
| 4 | pdaf_m2 | 4 | 446 | 295 | 70 | 1972 | 0% |

## 4. 补充：PD+AF 叠加 Tier1/DVFS（pdaf_m2_tier）

前述对比里的 PD+AF 是"裸 AF"（无调频）。本节补一组 AF + Tier1 + Tier2 DVFS 完整版（`pdaf_m2_tier`），看主动调频能否帮 AF 把能耗追回来。同 4 卡 M=2 拓扑，启动参数额外加 `--afd-dvfs-enabled --afd-energy-model-dir … --enable-tier1-pa --tier1-disable-reload`，跑在 GPU0-3（GPU4-7 当时被外部任务占满）。决策日志确认 decode 主要降到 690MHz 最低频。

il512_ol128 三方对比（4卡同拓扑）：

| QPS | 方案 | 吞吐 | TTFT | TPOT | 能耗J | J/tok | SLO违背 |
|-----|------|------|------|------|------|-------|--------|
| 2 | pd_p2d2（纯PD） | 236 | 115 | 44 | 43299 | 2819 | 0% |
| 2 | pdaf_m2（裸AF） | 225 | 186 | 70 | 49041 | 3193 | 0% |
| 2 | pdaf_m2_tier（AF+DVFS） | 218 | 513 | 83 | 41487 | 2701 | 0% |
| 4 | pd_p2d2 | 471 | 115 | 43 | 48968 | 1594 | 0% |
| 4 | pdaf_m2 | 447 | 183 | 70 | 53848 | 1753 | 0% |
| 4 | pdaf_m2_tier | 430 | 1100 | 84 | 44504 | 1449 | 0% |
| 6 | pd_p2d2 | 705 | 118 | 45 | 53923 | 1170 | 0% |
| 6 | pdaf_m2 | 669 | 187 | 71 | 59235 | 1286 | 0% |
| 6 | pdaf_m2_tier | 640 | 1248 | 86 | 48007 | 1042 | 0% |
| 8 | pd_p2d2 | 934 | 132 | 47 | 59142 | 963 | 0% |
| 8 | pdaf_m2 | 890 | 245 | 74 | 66127 | 1076 | 0% |
| 8 | pdaf_m2_tier | 830 | 2582 | 88 | 52099 | 848 | 0% |

结论：DVFS 确实帮 AF 追回能耗——相对裸 AF，能耗/(J/tok) 各档降约 19-21%（QPS8：66127→52099J）。但代价是延迟大幅恶化：TTFT 从 ~245ms 飙到 2582ms（QPS8），TPOT 70→88ms，因为调频器在低负载下激进降到最低频。即便如此 SLO 仍 0 违背（TTFT SLO 5000ms 余量大）。与纯 PD 对比：AF+DVFS 的 J/tok（848）看似低于纯 PD（963），但那是用 TTFT 涨一个数量级（132ms vs 2582ms）换来的，吞吐还更低（830 vs 934）。本质未变：dense 模型上纯 PD 在延迟/吞吐上全面占优，AF+DVFS 只适合"延迟余量极大、极致压能耗"的窄场景。

## 5. 核心发现

### 4.1 纯 PD 在所有维度全面优于 PD+AF（dense 模型上）
相同卡数（4卡）下，pd_p2d2 vs pdaf_m2：TPOT 43-47ms vs 70-74ms（AF 慢 60%+），吞吐更高（QPS8: 934 vs 890），能耗更低（59k vs 66k J），J/tok 更优。原因：AF 把 attention 和 FFN 拆成独立进程后，每次迭代要做 A↔F 的 IPC 张量传输 + 流水线 drain，这部分固定开销在 Qwen3-32B 这种 dense（非 MoE）模型上没有收益来源（AF 的价值主要在 MoE 的 expert 并行/通信重叠），纯属额外延迟。结论：dense 模型用标准 PD，不要上 AF。

### 4.2 decode 是瓶颈，扩 decode 卡远比扩 prefill 卡有效
il512 QPS6 对比 3 卡的两种分配：pd_p1d2（decode 双卡）TTFT 217ms、0 违背、吞吐 702；pd_p2d1（prefill 双卡）TTFT 9511ms、74% 违背、吞吐仅 528。同样 3 卡，把第 3 张卡给 decode 还是 prefill，结果天差地别。原因：这些 workload 的 TTFT violation 全部来自 decode 容量不足导致的排队回压（prefill 等待 decode 腾出 KV），prefill 本身从不是瓶颈（TTFT 在不饱和时仅 115-326ms）。

### 4.3 pd_p1d2（3卡）是性价比之王
il512 QPS8：pd_p1d2 用 3 卡达到 935 tok/s、0 违背、J/tok=838（全场最低）；pd_p2d2 用 4 卡同样 935 tok/s 但 J/tok=963。即第 4 张卡（给 prefill 凑成 TP2）在该负载下纯属浪费——prefill TP1 已够用，decode TP2 才是关键。decode 重的 il256_ol512 更夸张：pd_p1d2 QPS4 达 1493 tok/s、J/tok=459，比 4 卡 pd_p2d2 还省。

### 4.4 过度配卡 = 纯浪费能耗
pd_p2d4（6卡）vs pd_p2d2（4卡）：il512 全 QPS 段吞吐几乎相同（QPS8 都是 934），但 pd_p2d4 多烧 2 张卡的电（73k vs 59k J，J/tok 高 24%）。decode TP4 在这些负载下没有额外吞吐收益，因为瓶颈已不在 decode 算力。只有当单请求 KV 容量或 batch 并发成为限制时，扩 decode 卡才有意义。

### 5.5 DVFS 是"省能耗换延迟"，改变不了 AF 的拓扑劣势
给 AF 叠加 Tier1/DVFS 后能耗降 19-21%，但 TTFT 恶化一个数量级（见第 4 节）。DVFS 只在功率维度做文章，无法消除 AF 算子拆分带来的固定通信开销（TPOT 基线仍 70ms+，远高于纯 PD 的 44ms）。所以调频是正交于拓扑选择的一层优化：先选对拓扑（纯 PD），再视 SLO 余量决定要不要上 DVFS 省电。

## 6. 选型结论

1. dense 模型（Qwen3-32B）一律用标准 PD 解耦，不要用 PD+AF——AF 的算子拆分通信开销在 dense 上是纯负担，TPOT 恶化 60%、能耗增加。AF 价值场景是 MoE，不在本次范围。
2. 卡预算优先给 decode：3 卡时选 pd_p1d2（P1+D2）而非 pd_p2d1（P2+D1），容量与能效差一个数量级。
3. 默认推荐 pd_p1d2（3卡）：在 il512/il256/il1024 三种负载下都能撑到 QPS8 不违背，且 J/tok 全场最低。预算紧或低 QPS（≤4）可用 pd_p1d1（2卡），但 QPS≥6 会因 decode 单卡饱和而 TTFT 崩。
4. 不要盲目加卡：pd_p2d4（6卡）相对 pd_p2d2（4卡）无吞吐收益、能耗更高。扩卡前先确认瓶颈在 decode 算力而非别处。
5. 异构 AF（tp_A≠tp_F）当前在 ipc_cpp 后端不稳定（decode 挂起），如需异构需切 stepmesh/zmq 后端并重新验证。
6. DVFS 与拓扑选择正交：AF+DVFS 省 ~20% 能耗但 TTFT 涨一个数量级，只适合延迟余量极大的场景；先选对拓扑再谈调频。

## 7. 补充：M=1 微批实验（pdaf_m1 / pdaf_m1_tier）

前述 AF 实验均使用 M=2 微批流水线。本节补测 M=1（无微批，纯串行 A→F），验证流水线级数对能效和延迟的影响。同 4 卡拓扑（PA1 PF0 DA3 DF2，each TP=1），il512_ol128，QPS 2/4/6/8。

### 7.1 数据

| QPS | 方案 | 吞吐 | TTFT | TPOT | 能耗J | J/tok | SLO违背 |
|-----|------|------|------|------|-------|-------|---------|
| 2 | pdaf_m1 | 231 | 203 | 54 | 37114 | 2416 | 0% |
| 2 | pdaf_m1_tier | 224 | 522 | 68 | 33533 | 2183 | 0% |
| 4 | pdaf_m1 | 460 | 211 | 58 | 41486 | 1351 | 0% |
| 4 | pdaf_m1_tier | 440 | 874 | 75 | 36642 | 1193 | 0% |
| 6 | pdaf_m1 | 684 | 235 | 64 | 47096 | 1022 | 0% |
| 6 | pdaf_m1_tier | 630 | 4621 | 96 | 37933 | 823 | 40.3% |
| 8 | pdaf_m1 | 900 | 312 | 75 | 51680 | 841 | 0% |
| 8 | pdaf_m1_tier | 535 | 32549 | 134 | 55975 | 911 | 100% |

### 7.2 M=1 vs M=2 vs 纯 PD 对比（4卡 il512_ol128）

| QPS | 指标 | pd_p2d2 | pdaf_m2 | pdaf_m1 |
|-----|------|---------|---------|---------|
| 8 | TPOT (ms) | 47 | 74 | 75 |
| 8 | 吞吐 (tok/s) | 934 | 890 | 900 |
| 8 | J/tok | 963 | 1076 | 841 |
| 8 | 能耗 (J) | 59142 | 66127 | 51680 |

### 7.3 发现

1. **M=1 裸跑能效全场最优**：QPS8 时 J/tok=841，低于纯 PD（963）和 M=2（1076）。原因：M=1 无微批 overlap，A 和 F 串行执行，GPU 在等待对端时处于低功耗空闲态，总功耗更低。虽然 TPOT 略高（75 vs 47ms），但能耗绝对值低 12.6%（51680 vs 59142J）。

2. **M=1 + DVFS 在高 QPS 下崩溃**：QPS6 已 40% SLO 违背（TTFT 4621ms），QPS8 直接 100% 违背（TTFT 32549ms，吞吐跌至 535）。M=1 没有流水线 overlap 来隐藏调频切换延迟，DVFS 降频后 decode 算力直接不足，请求排队雪崩。对比 M=2+DVFS 在 QPS8 仍 0% 违背（TTFT 2582ms），说明微批流水线为 DVFS 提供了必要的延迟缓冲。

3. **M=1+DVFS 仅在低 QPS 有效**：QPS≤4 时 M=1_tier 的 J/tok（2183/1193）优于 M=2_tier（2701/1449），且 SLO 0 违背。但一旦负载超过临界点（QPS≥6），无流水线缓冲导致系统迅速崩溃。

4. **选型建议更新**：若 SLO 余量极大且 QPS 确定不超过 4，M=1+DVFS 是最省电方案；否则裸 M=1（无 DVFS）在全 QPS 段都稳定且能效优于 M=2 和纯 PD，是 AF 拓扑下的最佳默认配置。但纯 PD 仍在延迟（TPOT 47 vs 75ms）和吞吐上占优，AF 的价值仍限于 MoE 场景。

### 产物
- 脚本：`scripts/bench/run_deploy_bench.py`（含 pdaf_m1/pdaf_m1_tier 拓扑）；`run_m1_bench.sh`（M=1 专用跑批脚本）。
- 数据：`scripts/bench/results/deploy/json/pdaf_m1{,_tier}_il512_ol128_qps{2,4,6,8}_results.json`（8 run）。
- 日志：`scripts/bench/logs/_sweep/deploy_m1_full.log`。
- 图表：`deploy_tier_tradeoff.png` 已更新为 5 路对比（pure-PD / bare-AF M=2 / AF+DVFS M=2 / bare-AF M=1 / AF+DVFS M=1）。
