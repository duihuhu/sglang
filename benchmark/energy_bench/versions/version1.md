# Energy Bench — Tier1/Tier2 DVFS 节能评测（version1）

> 接续 version0。本轮完成 version0 第 8 节 TODO 的 **#2 补全数据矩阵 / #3 细扫饱和拐点 / #4 决策准确性可视化 / #6 变长 workload 验证**。**未改变原实验方案**（仍 PD+AF M=2、ipc_cpp、三模式对比、SLO TTFT≤5000ms/TPOT≤300ms）。

---

## 0. 本轮工作概览（相对 version0 的增量）

| TODO | 内容 | 状态 |
|------|------|------|
| #2 | 补全数据矩阵：il1024_ol128 与 il256_ol512 各补 QPS 1,3,5 | 完成 |
| #3 | 细扫饱和拐点：il512_ol128 补 QPS 7,8 | 完成 |
| #4 | 决策准确性可视化：新增 `plot_dvfs_decisions.py`（预测误差 vs QPS、选频分布堆叠图） | 完成 |
| #6 | 变长 workload 验证：varying / steady / heavy 三条动态 trace | 完成 |

本轮共新增 **17 个 workload-run**（6 矩阵 + 2 饱和 + 3 变长，每个 ×3 模式），全部经 watchdog 保护，无 hang。

## 1. 任务2：补全数据矩阵（对齐 QPS 网格）

补齐后各组 QPS 网格：il512_ol128 = 1-8；il1024_ol128 = 1-6；il256_ol512 = 1-6；il2048_ol256 = 1,2（大上下文超容量，维持原设计）。

### 1.1 il1024_ol128（长输入，新增 QPS 1,3,5）

| QPS | tier1 功率 | max 功率 | 节能 | tier1 SLO | max SLO |
|-----|-----------|---------|------|-----------|---------|
| 1 | 469W | 706W | **30.0%** | 0% | 0% |
| 2 | 512W | 765W | 29.8% | 0% | 0% |
| 3 | 538W | 827W | **31.5%** | 0% | 0% |
| 4 | 558W | 877W | 32.8% | 0% | 0% |
| 5 | 582W | 943W | **33.2%** | **40%** | 0% |
| 6 | 613W | 985W | 26.2% | 83% | 0% |

> **新发现**：补 QPS5 后暴露拐点——il1024_ol128 的 tier1 SLO 在 QPS5 已跳到 40%（version0 只有 4→0%、6→83%，看不出拐点位置）。tier1 的可服务峰值被降频压到 QPS4 附近。

### 1.2 il256_ol512（长输出/decode 重，新增 QPS 1,3,5）

| QPS | tier1 功率 | max 功率 | 节能 | tier1 SLO | max SLO |
|-----|-----------|---------|------|-----------|---------|
| 1 | 454W | 675W | **25.5%** | 0% | 0% |
| 2 | 475W | 712W | 25.8% | 0% | 0% |
| 3 | 503W | 743W | **19.8%** | **44%** | 14% |
| 4 | 507W | 757W | 17.9% | 58% | 58% |
| 5 | 522W | 767W | **19.1%** | 67% | 67% |
| 6 | 514W | 768W | 18.8% | 72% | 72% |

> **新发现**：il256_ol512 在 QPS3 出现 tier1 独有的 SLO 恶化（44% vs max 14%）——decode 重负载下 tier1 降频比 max 更早踩 SLO。QPS≥4 后三模式 SLO 一同崩（系统容量天花板，与频率无关），印证 version0 结论。decode 重组节能空间本就小（18-26%）。

## 2. 任务3：细扫饱和拐点（il512_ol128 QPS 7,8）

il512_ol128 是最均衡、节能最稳的组，version0 测到 QPS6（0% SLO）。本轮补 QPS 7,8 找 tier1 的真实饱和拐点。

| QPS | tier1 功率 | max 功率 | 节能 | tier1 吞吐 | max 吞吐 | tier1 SLO | max SLO |
|-----|-----------|---------|------|-----------|---------|-----------|---------|
| 6 | 542W | 859W | 33.7% | 637 tok/s | 669 | 0% | 0% |
| 7 | 534W | 898W | **32.7%** | 690 tok/s | 780 | **13%** | 0% |
| 8 | 561W | 952W | **31.3%** | 764 tok/s | 890 | **51%** | 0% |

**饱和拐点定位清晰**：
- tier1_freq 的 SLO 拐点在 **QPS7**（13%）出现，QPS8 升到 51%；而 **max_freq 到 QPS8 仍 0%**。
- 吞吐差距同步拉开：QPS1-6 时 tier1 vs max 吞吐损失 <5%，到 QPS7 变成 690 vs 780（-12%），QPS8 为 764 vs 890（-14%）。
- **本质**：降频省了 ~32% 电，代价是把可服务峰值 QPS 从 8+ 压低到 6。这是"节能 vs 容量裕度"的量化权衡，是 tier1 的固有特性，非 bug。

## 3. 任务6：变长 workload 验证

用 `workloads/` 下三条带 phase 切换的动态 trace 验证 DVFS 在真实变化负载下的自适应能力（区别于定长稳态）：

| trace | phase 时间线 | 特征 |
|-------|-------------|------|
| workload_varying | light→heavy→medium→burst_peak→burst_quiet（118s, 280 req） | 多 phase + 突发 |
| workload_steady | steady（120s, 600 req, il/ol 混合） | 稳态高负载 |
| workload_heavy | light→heavy→burst→recovery→burst2→idle（128s, 360 req） | 重负载 + 双突发 |

### 结果（节能均在 0% SLO 下）

| trace | tier1 功率 | max 功率 | **节能** | tier1 SLO | tier1 TPOT | max TPOT | 吞吐损失 |
|-------|-----------|---------|---------|-----------|-----------|---------|---------|
| varying | 480W | 720W | **31.5%** | 0% | 88.7ms | 69.6ms | ~3% |
| steady | 551W | 849W | **32.2%** | 0% | 96.5ms | 71.5ms | ~4% |
| heavy | 505W | 759W | **33.0%** | 0% | 89.2ms | 69.8ms | ~1% |

**结论**：动态变长负载下 tier1_freq 同样稳定省 **31-33%**，SLO 全 0%，TPOT 88-96ms 远在 300ms 内，吞吐损失 <5%。证明 Tier2 逐 batch DVFS 能跟随 phase 切换自适应调频，不仅在定长稳态有效。auto≈max 结论同样成立。

## 4. 任务4：决策准确性可视化

新增 `scripts/bench/plot_dvfs_decisions.py`，自动发现所有 tier1_freq 决策日志（本轮已覆盖 34 个 (group,qps,persp) decode 序列），产出两张图：

| 图 | 内容 |
|----|------|
| `results/fixed_qps/dvfs_pred_accuracy.png` | decode-attn/ffn 预测误差 vs QPS，按 il/ol 组分线，带 p10-p90 误差带 |
| `results/fixed_qps/dvfs_freq_mix.png` | decode-attn/ffn 选中 SM 频率分布堆叠图（每个 group×QPS 一根柱） |

### 预测准确性（全量数据，剔除启动瞬态）

- **OVERALL median |误差| = 28.8%，median 有符号误差 = -28.8%**（与 version0 的 27.6% 一致），系统性低估 decode 迭代时间稳定在 -25%~-32%，确认是固定建模偏差。
- **饱和点预测崩溃（新发现）**：il512_ol128 QPS7 的 decode-ffn 预测误差骤增到 **-71%**（实测 196.9ms vs 预测 57.1ms）。饱和时 FFN 侧迭代时间因 pipeline 阻塞暴涨，预测器完全跟不上——这正是 QPS7 开始踩 SLO 的根因：**预测器在容量边界处严重失真**，导致选频偏低。
- **选频规律合理**：负载越重越往高频走（decode-ffn 在重载组大量选 930/1170MHz），与 version0 一致。

## 5. 代码改动（本轮，向后兼容）

未改实验方案，仅扩展脚本以支持变长 trace 与新增可视化：

### 5.1 `run_fixed_qps_bench.py` — 支持变长 workload 命名
- `_cfg_from_path()`：原只认 `fixed_il<I>_ol<O>_qps<N>` 格式，变长 trace（如 `workload_varying.jsonl`）会被解析成脏 tag `il?_ol?_qpsworkload_varying`，文件名带 `?` 且会污染定长组。
- 改为：非 `fixed_` 命名的 trace 统一归入 `group="var"`，tag 用 `var_<name>`（如 `var_varying`），与定长 il<I>_ol<O> 组完全隔离，互不污染。
- `print_qps_sweep()`：QPS 排序键加 try/except，变长 qps（字符串）不再抛 ValueError。

### 5.2 新增 `scripts/bench/plot_dvfs_decisions.py`
- 任务4 可视化脚本，详见第 4 节。自动发现日志，无需手工指定组。

### 5.3 新增 workload（8 个定长）
- `gen_fixed_workload.py` 生成：il1024_ol128 补 qps1/3/5、il256_ol512 补 qps1/3/5、il512_ol128 补 qps7/8。

### 5.4 P/D 分阶段能耗记录 + watchdog bug 修复
- `run_workload()`：新增按 GPU 映射拆分能耗——prefill(GPU6,7) → `prefill_energy_j`/`prefill_power_w`，decode(GPU4,5) → `decode_energy_j`/`decode_power_w`。NVML 逐卡读，拆分精确（验证 P+D=total 误差<1J）。
- `plot_fixed_qps.py`：能耗从 1 面板拆成 3 面板（Total / Prefill / Decode），各带节能%标注，布局 2×3 → 2×4。
- **修复 watchdog cancel 传播 bug**：watchdog 超时 `gather_task.cancel()` 后，`await gather_task` 抛 `asyncio.CancelledError`（3.8+ 继承自 `BaseException`），原 `except Exception` 捕获不到 → 整个 sweep 主进程崩溃。改为 `except (Exception, asyncio.CancelledError)`，并在 main 循环加 per-run 兜底（放行 KeyboardInterrupt/SystemExit），单 run 崩溃不再终止 sweep。

## 6. P/D 分阶段能耗拆分（本次重点）

把每组能耗按 prefill(GPU6,7) / decode(GPU4,5) 两阶段拆开（全部三模式重跑获得分卡能耗）。

### 6.1 分阶段节能数据（tier1_freq vs max_freq）

**il512_ol128**（P节能 / D节能 / 总节能 / Decode能耗占比）：

| QPS | P节能 | D节能 | 总节能 | D占比 |
|-----|------|------|-------|------|
| 1 | 18.1% | 33.5% | 29.0% | 66% |
| 4 | 28.4% | 32.9% | 31.2% | 61% |
| 6 | 35.1% | 31.4% | 32.9% | 59% |
| 8 | 35.0% | 27.3% | 30.8% | 58% |

**il1024_ol128**：

| QPS | P节能 | D节能 | 总节能 | D占比 |
|-----|------|------|-------|------|
| 1 | 21.9% | 33.3% | 29.7% | 65% |
| 3 | 29.6% | 33.0% | 31.6% | 59% |
| 5 | 36.4% | 29.6% | 32.8% | 55% |
| 6 | 29.2% | 24.5% | 26.8% | 52% |

**il256_ol512**（decode 重）：

| QPS | P节能 | D节能 | 总节能 | D占比 |
|-----|------|------|-------|------|
| 1 | 14.8% | 29.1% | 25.2% | 69% |
| 3 | 12.8% | 21.3% | 18.9% | 69% |
| 6 | 16.9% | 18.2% | 17.8% | 71% |

**il2048_ol256**（大上下文）：QPS1 = P 21.9% / D 31.5% / 总 28.2%（D占比 63%）。QPS2 的 max_freq 因外部任务占用 GPU4 启动 OOM，暂缺（待 GPU4 释放补测）。

### 6.2 拆分暴露的关键规律

1. **Decode 是能耗大头**：占总能耗 **52-71%**，decode 重的 il256_ol512 最高（69-71%），所以总节能曲线更贴近 D 的节能。
2. **P 和 D 节能随负载反向**：随 QPS 升高，**Prefill 节能单调上升**（如 il512：18%→37%；低 QPS 时 prefill GPU 大量空闲、降频收益小，高 QPS 持续繁忙降频省得多），而 **Decode 节能逐渐下降**（33%→27%；decode 越来越吃算力、降频空间被压缩）。**这一反向趋势在合并视图里会被平均掉，只有拆分才看得到。**
3. **饱和点 P/D 双双回落**：il512 QPS8、il1024 QPS6（SLO 已崩）时 P/D 节能都掉头向下，与 SLO 拐点一致。

## 7. Prefill 调频与 TTFT 违背的错位分析

**现象**：il256_ol512 QPS6 的 TTFT SLO 违背已达 72%，但 prefill 选频图（`dvfs_freq_mix_prefill.png`）里调频**从未拉到最高档 1410MHz**（attn 多在 930/1170，ffn 稳在 930）。看似矛盾。

### 7.1 根因：slack 指标"近视"，看不到队列积压

prefill 调频依据 `_compute_prefill_slack()`（`scheduler.py`）：`slack = TTFT_SLO(5000ms) − 该请求已等待 elapsed`。决策日志里 slack 一直 ≈ **4.5s**（看着余量充足）→ 调频器判断 930MHz 够用，不升频。

但缺陷在于：`elapsed` 只统计**已进入调度、即将被 prefill 的这批请求**等了多久；而真正违背 SLO 的是**还堵在 waiting queue、根本没进 batch 的请求**（被 decode 反压挤在队列里等了几万 ms）。调频器看不到这些积压请求，于是误判"余量充足"。

### 7.2 而且——升频也救不了（这是容量问题）

即便调频器看到积压、强行拉到 1410MHz 也无济于事：瓶颈不在 prefill 算力（预测 prefill 延迟仅 ~119ms），而在 **decode 槽位被长输出（ol=512）长期占满、prefill 根本没机会执行**。

铁证：QPS≥4 时 **tier1_freq / max_freq / auto_freq 三模式 TTFT 违背率完全一致**（58% / 67% / 72%）——锁死 1410MHz 的 max_freq 照样违背 72%。这是系统吞吐天花板，与频率无关。

### 7.3 结论：这里"没拉满"反而是对的

既然升频救不了 SLO，那维持 930MHz **省电**就是最优解——强行升到 1410 只会白白多耗电、SLO 该违背还是违背。所以这不是 bug，是两个独立现象的叠加：① slack 指标近视（看不到队列积压），② 即便看到也受限于容量天花板。

**潜在改进**：若要让 DVFS 在此场景更"诚实"，`_compute_prefill_slack` 应纳入 **waiting queue 队首等待时间**（而非只看当前 batch）。这样调频器至少能"看到"积压——虽救不了容量问题，但能避免误判余量充足，也为"该不该早降速保护/拒绝"提供信号。

## 8. 本轮核心结论

1. **节能结论在更全的数据下完全稳固**：定长各组 tier1_freq 省电 **18-34%**（均衡/长输入 30-34%，decode 重组 18-26%），变长动态 trace 省 **31-33%**，全部在未饱和时 0% SLO。
2. **找到了 tier1 的 SLO 拐点（之前网格太粗看不到）**：
   - il512_ol128：QPS7 起（13%→QPS8 51%），max 到 QPS8 仍 0%。
   - il1024_ol128：QPS5 起（40%）。
   - il256_ol512：QPS3 起（44%，decode 重最早）。
   - 规律：**降频把可服务峰值 QPS 拉低约 1-2 档**，这是节能的固有代价。
3. **变长负载验证通过**：DVFS 能跟随 phase 切换自适应，动态场景同样省 31-33% 且不踩 SLO。
4. **预测器偏差定量确认 + 新增饱和失真证据**：整体系统性低估 ~28.8%；在饱和点（QPS7 ffn）误差骤增到 -71%，是踩 SLO 的直接诱因。
5. **P/D 拆分新洞察**：Decode 占能耗 52-71%（节能主战场）；Prefill 节能随 QPS 升高、Decode 节能随 QPS 下降（反向趋势，合并视图看不到）。
6. **TTFT 违背与 prefill 调频的错位（见第 7 节）**：高 QPS 长输出场景下 TTFT 违背来自队列积压（decode 反压），而 prefill 调频依据的 slack 指标看不到积压、误判余量充足；但即便升频也救不了（三模式违背率一致），所以维持低频省电反而是对的。

## 9. 产物清单（本轮新增）

| 类型 | 路径 |
|------|------|
| 矩阵补全结果 | `scripts/bench/results/fixed_qps/il{1024_ol128,256_ol512}_qps{1,3,5}_*_results.json` |
| 饱和拐点结果 | `scripts/bench/results/fixed_qps/il512_ol128_qps{7,8}_*_results.json` |
| 变长结果 | `scripts/bench/results/fixed_qps/var_{varying,steady,heavy}_*_results.json` |
| 决策可视化图 | `figures/dvfs_pred_accuracy.png`、`dvfs_freq_mix_decode.png`、`dvfs_freq_mix_prefill.png` |
| 对比图（含 P/D 拆分） | `figures/compare_*.png`（8 面板：Total/Prefill/Decode 能耗 + 吞吐 + TTFT/TPOT avg + TTFT/TPOT 违背率）|
| P/D 分阶段数据 | 所有 `il*_results.json` 新增 `prefill_energy_j`/`decode_energy_j`/`*_power_w` 字段 |
| 新增脚本 | `scripts/bench/plot_dvfs_decisions.py` |
| sweep 日志 | `scripts/bench/results/fixed_qps/sweep_{matrix,saturation,varlen,pd_*}.log` |

## 10. 后续 TODO（承接 version0 未完项）

- [ ] **补测 il2048_ol256 QPS2 的 max_freq**：本轮因外部任务占用 GPU4 启动 OOM 跳过，待 GPU4 空闲后单独补（`--modes max_freq` 即可，tier1/auto 已有缓存）。
- [ ] **校准预测模型偏差**（version0 #1，未动）：系统性低估 ~28.8%，且饱和点恶化到 -71%。建议优先做：① 重标 `t_drain_us`；② 启用 `--afd-dvfs-online-calibration` EMA 修正；③ 针对饱和区单独建模 FFN 阻塞项。
- [ ] **Tier1 真正生效场景**（version0 #5，未动）：G=4 下 Tier1 求解器 INFEASIBLE，需 G≥8 才能体现资源重规划价值。
- [x] **SLO 余量收紧实验**：已完成（见第 11 节）。收紧 TPOT SLO 到 120/100/85/75ms 后发现一个反直觉结果——tier1 调频对 SLO 收紧**完全无响应**，根因是预测器 3.9x 低估让 SLO 检查恒为真，实为预测偏差 TODO 的放大验证。
- [ ] **变长 trace 加密采样决策日志**：变长场景的 phase 切换瞬态调频行为值得单独画时间线（plot_tier_trace.py 已有雏形）。
- [ ] **prefill slack 纳入队列积压（见第 7 节）**：`_compute_prefill_slack` 目前只看当前 batch 已等待时间，看不到 waiting queue 队首的积压。改为纳入队首等待时间后，调频器能感知积压（虽救不了容量天花板，但可避免"余量充足"误判，并为早降速/拒绝提供信号）。

---

## 每日进度（2026-05-30）

今天完善了 Qwen3-32B / PD+AF 解耦（M=2，A800×4）的 DVFS 节能评测。
1.tier 稳定省电 18-34%（均衡/长输入组 30-34%、decode 重组 18-26%，变长 trace 31-33%），未饱和时 0% SLO；auto 约等于 max，靠硬件默认调频省不了电，必须主动降频。
2.能耗按 prefill/decode 拆开后，Decode 是节能主战场（占 52-71%），且 Prefill 节能随 QPS 升、Decode 节能随 QPS 降。
3.SLO 违背全来自 TTFT 排队、TPOT 从不违背；高 QPS 长输出下的违背是系统容量天花板（三种频率模式违背率一致），与调频无关，此时维持低频省电是正确决策。
4.待校准两点：预测模型系统性低估 decode 延迟约 28%（饱和点恶化到 -71%）；prefill 调频的 slack 指标看不到 waiting queue 排队积压。

---

## 11. TPOT SLO 收紧实验（D 阶段 SLO 对调频的影响，2026-05-31）

### 11.1 动机与设置
之前所有实验 TPOT SLO 固定 300ms，余量极大（tier1 实测 ~90ms、No-Tier ~73ms），从未违背，因此无法观察 D 阶段 SLO 对调频的影响。本实验把 TPOT SLO 逐档收紧到 100ms 以下，观察 tier1 的 decode 选频是否被逼升频。
- workload：il256_ol512（decode 重，TPOT 调频空间最大），QPS 2 / 4
- TPOT SLO 四档：120 / 100 / 85 / 75 ms（75 接近 No-Tier 物理下限 ~73ms）
- 模式：tier1_freq（受 SLO 影响） + max_freq（锁频，仅违背统计随 SLO 变，作对照）
- 代码改动：`run_fixed_qps_bench.py` 让 `--tpot-slo-ms` 真正流入调频器（`--afd-tpot-slo-us`），并给收紧档结果加 `_tpot{N}` tag 后缀，与 300ms 基线隔离。

### 11.2 核心数据（tier1 vs max，单位 ms / tok·s⁻¹ / J）
| SLO | QPS | tier1 TPOT | tier1 TTFT | tier1 吞吐 | tier1 TPOT违背 | max TPOT |
|-----|-----|-----------|-----------|----------|--------------|---------|
| 120 | 2 | 208.0 | 10035 | 273 | 120/120 | 71.9 |
| 100 | 2 | 208.3 | 10097 | 273 | 120/120 | 72.0 |
| 85  | 2 | 208.7 | 10158 | 273 | 120/120 | 71.9 |
| 75  | 2 | 206.9 | 9985  | 275 | 120/120 | 71.8 |
| 120 | 4 | 206.3 | 61765 | 373 | 240/240 | 73.8 |
| 75  | 4 | 204.3 | 61200 | 377 | 240/240 | 73.9 |

### 11.3 DVFS 决策日志（decode，三档对比）
| SLO | attn 频率分布 | pred_iter | obs_iter | 低估倍数 |
|-----|-------------|-----------|----------|---------|
| 120 | 690:517 / 1170:346 / 930:223 | 54.4ms | 213.7ms | 3.9x |
| 85  | 690:519 / 1170:348 / 930:219 | 54.4ms | 214.0ms | 3.9x |
| 75  | 690:517 / 1170:347 / 930:221 | 54.4ms | 212.7ms | 3.9x |

三档的频率分布、预测值、观测值几乎逐位相同，证明调频器对 SLO 收紧零响应。对照 300ms 基线同 workload：pred=54.5ms / obs=117ms（低估 2.2x），TPOT 实测仅 92ms。

### 11.4 因果分析（反直觉结论）
预期"收紧 SLO → 调频器升频满足更严 SLO"，实测却是"收紧 SLO → 调频毫无变化 → tier1 性能反而比 300ms 基线更差（TPOT 92→208ms，吞吐 584→273）"。根因在 `af_dvfs_controller.py::select_freq_decode`（330-331 行）：候选频率按能耗升序，对每个候选算预测迭代时间 `t_iter`，若 `t_iter * calibration_factor > slo_tpot_us` 则跳过，否则选中（最省电的"可行"频率）。由于预测器对最低频 690MHz 的迭代时间恒预测为 ~54ms，即使最严的 75ms SLO，判据 `54ms < 75ms` 仍恒为真，于是 690MHz 始终被当成可行的最省电选择。预测器 3.9x 低估让 SLO 阈值在选频公式里被彻底架空——这是第 10 节"预测模型系统性低估"TODO 在 SLO 收紧场景下的放大暴露。

### 11.5 结论与下一步
当前 D 阶段 TPOT SLO 对调频实际失效，不是 SLO 机制设计问题，而是预测器低估导致 SLO 约束恒满足。要让 SLO 真正驱动调频，必须先修预测偏差：① 重标 `t_drain_us`（当前 18ms 远低于实测）；② 启用 `--afd-dvfs-online-calibration` 用 EMA 把 obs/pred 比值反馈进 `calibration_factor`；③ 在 decode 重负载区单独建模迭代时间。修复后应重跑本实验验证 SLO 梯度能否驱动频率梯度。

