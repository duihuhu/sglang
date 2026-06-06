# Energy Bench — Tier1/Tier2 DVFS 节能评测（version0）

> 记录：Qwen3-32B / PD+AF 解耦 / A800×4（GPU 4-7）上，Tier1+Tier2 频率控制方案的节能评测进展、结论与后续计划。

---

## 1. 实验目标

在 **PD 解耦 + AF 解耦**（Prefill/Decode × Attention/FFN，四进程各占一卡）的 Qwen3-32B 服务上，对比三种 GPU 频率控制策略对**性能（TTFT/TPOT/吞吐）、能耗、SLO 违背率**的影响，验证 DVFS 节能方案的有效性与调频/预测模型的准确性。

### 三种对比方案

| 方案 | 含义 | 频率控制 |
|------|------|---------|
| **tier1_freq** | Tier1（只调频不重规划）+ Tier2 逐 batch DVFS | 动态调频 |
| **max_freq** | 无 Tier，全程锁最高频 1410MHz | 锁死最高频 |
| **auto_freq** | 无 Tier，不锁频，GPU 硬件默认自动 boost | 不干预 |

**关键设计点**：`tier1_freq` 中 Tier1 只做"频率重规划"，不触发"模型重加载"。为此新增了 `--tier1-disable-reload` 开关（详见第 4 节）。在 G=4（每池 1 卡）约束下 Tier1 求解器结构性 INFEASIBLE，因此节能实际**完全来自 Tier2 逐 batch DVFS**；Tier1 仅作监控 + 频率重规划尝试。

---

## 2. 系统配置

- **模型**：Qwen3-32B，每进程 `--tp 1`
- **架构**：PD+AF M=2 微批流水线，`ipc_cpp` C++ IPC 后端，mooncake PD transfer，前置 `sglang_router` mini-lb
- **GPU 映射**：PA=GPU7(attn) / PF=GPU6(ffn) → prefill 对；DA=GPU5(attn) / DF=GPU4(ffn) → decode 对
- **合法频率档**：210 / 450 / 690 / 930 / 1170 / 1410 MHz（NVML SetGpuLockedClocks）
- **SLO**：TTFT ≤ 5000ms，TPOT ≤ 300ms
- **能耗测量**：NVML `nvmlDeviceGetTotalEnergyConsumption`，4 卡能耗差累加
- **能耗模型**：`/workspace/sglang/benchmark/test_motivation/energy_models`（GBDT/LinearReg LUT）

---

## 3. 数据集

定长 workload：固定 input_len / output_len，只改变 QPS（请求到达频率），到达过程为均匀间隔（1/qps）。生成器：`scripts/utils/gen_fixed_workload.py`。

| 组 | il / ol | 特征 | 已测 QPS |
|----|---------|------|---------|
| il512_ol128  | 512 / 128  | 均衡 | 1, 2, 3, 4, 5, 6 |
| il1024_ol128 | 1024 / 128 | 长输入（prefill 重） | 2, 4, 6 |
| il256_ol512  | 256 / 512  | 长输出（decode 重） | 2, 4, 6 |
| il2048_ol256 | 2048 / 256 | 大上下文 | 1, 2 |

> il2048_ol256 的 QPS≥4 超出 4 卡容量（decode KV cache OOM），按设计只测低 QPS。

---

## 4. 代码改动

### 4.1 新增 `--tier1-disable-reload`（核心）
- `python/sglang/srt/server_args.py`：新增布尔参数 `tier1_disable_reload`。
- `python/sglang/srt/managers/scheduler.py` `_apply_tier1_transition()`：当 Tier1 重规划产生 TP/k 变化时，若该开关打开，则**跳过 `_trigger_full_reload`（不重启服务/不重载权重）**，只应用新解的频率（freq-only transition）。
- 作用：把 Tier1 的"资源重规划"与"重调频"解耦，使其退化为纯粹的工作负载感知 re-frequency 控制器。

### 4.2 DVFS 决策日志（评估调频/预测准确性）
- `scheduler.py`：环境变量 `AFD_DVFS_DECISION_LOG` 控制，每进程独立写 JSONL（按 perspective/disagg/gpu 命名，无并发写冲突）。
- 记录字段：
  - **prefill**：bs, il, slack_us, 选中 f_a/f_f, 预测延迟/能耗
  - **decode**：bs, il, ol, 当前 f_a/f_f, 选中 f_a/f_f, switched, **实测迭代时间 obs_iter_us**, **当前频率下的预测迭代时间 pred_iter_cur_us**, **预测误差 pred_iter_err_pct**, 预测能耗, 校准因子
- `obs_iter_us` vs `pred_iter_cur_us` 直接量化预测模型准确性。

### 4.3 Bench 脚本（`scripts/bench/run_fixed_qps_bench.py`）
- 三模式（tier1_freq / max_freq / auto_freq），max_freq 锁 1410，auto_freq 解锁。
- **SLO 违背率统计**：逐请求判定 TTFT/TPOT 是否超限，失败请求也计入违背。
- **多 il/ol × 多 QPS sweep**，结果按 `il<I>_ol<O>_qps<N>_<mode>` 命名缓存。
- **日志分离**：server log + 决策日志写到 `logs/<tag>/<mode>/`，不与脚本/数据集混放。
- **显存释放关卡** `wait_gpu_mem_free()`：每轮 teardown 后等 VRAM 真正释放再启动下一轮（修复跑测顺序污染，见 6.1）。
- **崩溃保护 watchdog**：单个 server 进程崩溃（OOM）或运行超 `--max-run-s` 时，自动中止该 run、清理、继续，不再 hang 死整个 sweep；崩溃 run 不写缓存（见 6.2）。

---

## 5. 结果汇总

> 全部产物在 `scripts/bench/results/fixed_qps/`（JSON + `compare_*.png`），决策日志在 `scripts/bench/logs/`。
> 注：早期 `qps<N>_*_results.json`（无 il/ol 前缀）是 il512_ol128 的初版数据，已迁移为新命名，二者一致。

### 5.1 性能/能耗（tier1_freq vs max_freq，节能均在 0% SLO 下）

| 组 | QPS | tier1 功率(W) | max 功率(W) | tier1 能耗(J) | max 能耗(J) | **节能** | tier1 SLO |
|----|-----|--------------|-------------|--------------|-------------|---------|-----------|
| il512_ol128 | 1 | 458 | 679 | 32596 | 45902 | **29.0%** | 0% |
| il512_ol128 | 2 | 488 | 714 | 34879 | 48811 | **28.5%** | 0% |
| il512_ol128 | 3 | 502 | 750 | 36011 | 51514 | **30.1%** | 0% |
| il512_ol128 | 4 | 516 | 783 | 37128 | 53822 | **31.0%** | 0% |
| il512_ol128 | 5 | 527 | 818 | 38072 | 56319 | **32.4%** | 0% |
| il512_ol128 | 6 | 542 | 859 | 39236 | 59146 | **33.7%** | 0% |
| il1024_ol128 | 2 | 512 | 765 | 36763 | 52335 | **29.8%** | 0% |
| il1024_ol128 | 4 | 558 | 877 | 40605 | 60421 | **32.8%** | 0% |
| il1024_ol128 | 6 | 613 | 985 | 50282 | 68101 | **26.2%** | 83%（饱和）|
| il256_ol512 | 2 | 475 | 712 | 50523 | 68064 | **25.8%** | 0% |
| il256_ol512 | 4 | 507 | 757 | 75937 | 92544 | **17.9%** | 58%（饱和）|
| il256_ol512 | 6 | 514 | 768 | 100816 | 124121 | **18.8%** | 72%（饱和）|
| il2048_ol256 | 1 | 498 | 748 | 41047 | 57332 | **28.4%** | 0% |
| il2048_ol256 | 2 | 548 | 840 | 46130 | 65304 | **29.4%** | 0% |

### 5.2 核心结论

1. **tier1_freq 稳定省电约 30%**（均衡/长输入/大上下文组 26-34%；decode 重的 il256_ol512 约 18-26%，因 decode 本就吃算力、降频空间小），**且 SLO 不违背**（未饱和时）。
2. **代价是延迟升高但在 SLO 内**：TTFT 升到 0.5-2s（max_freq ~0.2-0.5s），TPOT 升约 25-30%（~90ms vs ~70ms），均远在 SLO（TTFT 5s / TPOT 300ms）内。吞吐损失 <5%。
3. **auto_freq ≈ max_freq**（所有组）：持续负载下硬件自动 boost 到接近最高频，"不干预"几乎等于"锁最高频"，省电可忽略（±0.5%）。**说明靠硬件默认调频省不了能耗，必须主动降频。**
4. **饱和区的 SLO 崩溃是系统容量问题，非调频导致**：il256_ol512 QPS≥4 三模式 SLO 同样违背（系统吞吐天花板），且此时 tier1_freq 仍省 ~18%。

### 5.3 调频决策 & 预测模型准确性（分析脚本 `analyze_dvfs_decisions.py`，已剔除启动瞬态）

**预测模型存在系统性偏差**：
```
OVERALL: median |误差| = 27.6%,  median 有符号误差 = -27.6%
（实测 decode 迭代 ~76ms vs 预测 ~55ms）
```
- 预测器**系统性低估** decode 迭代时间约 28%，在所有 il/ol 组、所有 QPS 下高度一致（-25% ~ -30%）→ 是**固定建模偏差**，非随机噪声。
- 怀疑原因：预测模型未充分计入 pipeline drain / IPC 同步开销，或 `t_drain_us=18000` 常数估偏。

**调频决策方向正确**：
- 尽管预测偏低 28%，实测 TPOT（~90ms）仍远小于 SLO（300ms），余量充足，低估未造成 SLO 违背。
- 频率选择规律合理：decode-attn 偏好 690MHz、decode-ffn 偏好 930MHz；**负载越重越往高频走**（如 il256_ol512 decode 重，QPS=6 时 ffn 大量选 1170MHz；il1024_ol128 prefill 重则 decode 侧稳定低频）。

**结论**：调频**决策方向准确**（按负载选频、SLO 不违背、省 ~30%），但**预测模型偏乐观**（系统性低估 decode 延迟 ~28%）。若未来 SLO 余量收紧，该低估可能导致选频偏低踩 SLO。

---

## 6. 过程中发现并修复的问题

### 6.1 跑测顺序污染（已修复）
- **现象**：QPS≥5 时，每轮 sweep 的第 2、3 个模式（如 max_freq）出现"降频反而更快/最高频反而崩"的反常，SLO 崩溃。
- **根因**：sglang worker 进程在端口释放后仍占着显存，下一个模式在 VRAM 没腾干净时启动 → KV cache 偏小 → 高负载下崩。
- **验证**：反序重跑（max_freq 先跑）后 max_freq 完全正常（0% SLO），证明是顺序污染而非频率效应。
- **修复**：`wait_gpu_mem_free()`，每轮 teardown 后等 VRAM 降到阈值以下再启动下一轮。修复后 QPS 1-6 全部自洽。

### 6.2 单 server 崩溃导致整体 hang（已修复）
- **现象**：il2048_ol256 QPS=4 时 DF（decode-ffn, GPU4）OOM 崩溃，但 PA/PF/DA 还活着 → 流水线等不到 FFN 响应 → 整个服务 hang，bench 主进程卡死。
- **根因**：大上下文（2048+256）高 QPS 下 decode KV cache 撑爆显存。
- **修复**：watchdog 检测进程存活 + `--max-run-s` 超时，崩溃/超时自动中止该 run 并继续；崩溃 run 不缓存。
- **决策**：il2048_ol256 大上下文只测低 QPS（1,2），高 QPS 超容量不测。

---

## 7. 产物清单

| 类型 | 路径 |
|------|------|
| 结果 JSON | `scripts/bench/results/fixed_qps/<组>_qps<N>_<mode>_results.json` |
| 汇总 | `scripts/bench/results/fixed_qps/summary.json` |
| 对比图 | `scripts/bench/results/fixed_qps/compare_<组>.png`（4 张：6 面板 = 能耗/功率/吞吐/TTFT/TPOT/SLO vs QPS）|
| DVFS 决策日志 | `scripts/bench/logs/<组>_qps<N>/tier1_freq/*dvfs_decisions_<persp>_<disagg>_gpu<N>.jsonl` |
| server 日志 | `scripts/bench/logs/<组>_qps<N>/<mode>/*pdaf_*.log` |
| bench 主脚本 | `scripts/bench/run_fixed_qps_bench.py` |
| 数据集生成 | `scripts/utils/gen_fixed_workload.py` |
| 画图 | `scripts/bench/plot_fixed_qps.py` |
| 决策分析 | `scripts/bench/analyze_dvfs_decisions.py` |

### 复现命令
```bash
# 生成定长数据集
python scripts/utils/gen_fixed_workload.py --output-dir workloads --qps 1,2,4,6 --input-len 512 --output-len 128
# 跑 sweep（三模式，日志分离，崩溃保护）
python scripts/bench/run_fixed_qps_bench.py \
    --workload-glob 'workloads/fixed_il512_ol128_qps*.jsonl' \
    --modes tier1_freq,max_freq,auto_freq \
    --output-dir results/fixed_qps --log-dir logs --max-run-s 240
# 画图 + 决策准确性分析
python scripts/bench/plot_fixed_qps.py
python scripts/bench/analyze_dvfs_decisions.py
```

---

## 8. 后续 TODO

- [ ] **校准预测模型偏差**：decode 迭代时间被系统性低估 ~28%。两条路径：① 重新标定 `t_drain_us`（当前 18000us 常数）；② 启用已有的 `--afd-dvfs-online-calibration`（在线用实测/预测比值 EMA 修正），重测一组验证能否把误差压到 <10%。
- [ ] **补全数据矩阵**：il1024_ol128 补 QPS 1,3,5；il256_ol512 补 QPS 1；让各组 QPS 网格对齐，便于横向对比。
- [ ] **找真实饱和点**：il512_ol128 在 QPS 6-7 附近接近吞吐天花板，可细扫 QPS 7,8 标出 SLO 崩溃的拐点。
- [ ] **决策准确性可视化**：把预测误差 vs QPS、选中频率分布堆叠图画出来（目前只有文本分析）。
- [ ] **Tier1 真正生效场景**：当前 G=4 下 Tier1 求解器 INFEASIBLE（无重规划空间）。若要体现 Tier1 资源重规划价值，需在 G≥8（多卡预算）下测，让 TP/k 有调整空间。
- [ ] **变长 workload 验证**：当前是定长，可用 workloads/ 下的变长/多 phase trace 验证动态负载下 DVFS 的自适应能力。
