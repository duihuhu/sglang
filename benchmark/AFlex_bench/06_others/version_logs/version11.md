# Version 11: MoE Conv Light 能耗优化 — 空闲 GPU 降频

## 目标

将 MoE (Qwen3-30B-A3B) Conv Light 数据集的 PDAF Asym (3P+5D)+Tier 能耗从 203kJ 降低到 172kJ，实现 vs Native+Tier 30%+ 的节能（当前仅 17.5%）。

## 问题分析

Dense 模型在 Conv Light 上 PDAF+Tier vs Native+Tier 节能 42.8%，而 MoE 仅 17.5%。分析发现：

1. **Dense 的节能来自架构分离**：AF 分离降低每 GPU 计算密度，功耗从 184W/GPU 降到 99W/GPU
2. **MoE 本身已高效**：memory-bound + sparse activation（只激活 3B/30B），原生功耗就只有 94W/GPU
3. **Prefill GPU 空闲浪费**：Conv Light 中 Prefill GPU (GPU0,1,2) 有 73% 时间空闲，却维持在 690-930MHz

关键数据：
- Conv Light: 979 请求 / 336s，Prefill 利用率仅 27%
- TPOT 仅 76.5ms vs SLO 250ms，headroom 高达 174ms
- DF 已经 61% 时间在 210MHz

## 优化方案

### Strategy 1: Prefill GPU 空闲降频锁定（主要杠杆）

在 AFD 事件循环中，当 Prefill 侧 scheduler 检测到没有待处理的 batch 时，立即将 GPU 频率锁定到 210MHz。新 batch 到达时恢复到 DVFS 目标频率。

实现位置：
- `python/sglang/srt/managers/scheduler.py`：`event_loop_afd` 的 FFN idle 等待分支和 batch=None 分支
- `python/sglang/srt/disaggregation/prefill.py`：`event_loop_afd_disagg_prefill` 的 PA/PF 空闲分支
- 恢复时机：`_afd_dvfs_before_batch` 开头

### Strategy 2: Headroom-Aware Aggressive DA 降频（次要杠杆）

当 TPOT headroom > 60% 时，尝试将 DA (Attention) 频率从 930MHz 降到更低值。

实现位置：`python/sglang/srt/energy/af_dvfs_controller.py`

### Strategy 3: 新增启动参数

- `--afd-dvfs-idle-lock`：启用空闲降频机制
- `--afd-dvfs-idle-lock-freq`：空闲时锁定频率（默认 210MHz）

## 实验结果

### Idle Freq Lock (仅 Prefill 侧)

| 指标 | 优化前 | 优化后 |
|------|--------|--------|
| 能耗 | 203.0 kJ | 200.5 kJ |
| TPOT avg | 76.5 ms | 76.7 ms |
| 吞吐量 | 322 tok/s | 322 tok/s |
| SLO 违反 | 0% | 0.1% |
| vs Native+Tier | -17.5% | **-18.5%** |

节省 ~2.5 kJ（1.2%），远低于预期的 25.8 kJ。

### Headroom-Aggressive DA 降频（已回退）

将 DA 从 930MHz 强制降到 210MHz 后：
- TPOT 从 76.5ms 飙升到 104.5ms（+37%）
- 能耗从 203kJ 增加到 216kJ（+6.4%）
- SLO 违反 1.6%

原因：DA 是 pipeline 关键路径，降频增加 iteration latency → DF idle 时间增加 → 总能耗反而更高。已禁用此功能（`headroom_aggressive_threshold=0.0`）。

## 关键发现：为什么 30% 目标不可达

Plan 中预期 Prefill idle lock 可节省 25.8 kJ 的假设存在重大误差：

### 假设 vs 现实

| 项目 | Plan 假设 | 实际测量 |
|------|-----------|----------|
| 930MHz idle 功耗 | ~45W/GPU | ~55-60W/GPU（含 HBM） |
| 210MHz idle 功耗 | ~10W/GPU | ~50-55W/GPU（含 HBM） |
| SM clock 对 idle 功耗的影响 | 35W 差异 | **仅 5-10W 差异** |
| 3 GPU × 246s 节省 | 25.8 kJ | ~2-5 kJ |

### 根因

1. **GPU idle 功耗由 HBM 主导**：A100 的 HBM2e 在 idle 状态仍消耗 ~25-30W，这部分不受 SM clock 频率控制
2. **SM clock 仅影响计算单元 leakage**：从 930→210MHz 只减少了 GPU die 上 SM 模块的漏电流，对总板级功耗影响有限（~5-10W）
3. **固定电路功耗**：GPU 的 PCIe 接口、NVLink、电源管理电路等在 idle 时持续消耗 ~15-20W
4. **切频开销**：每次 lock/unlock 需要 2-5ms，频繁切换（每 ~10ms 一次 poll）有微小的额外能量开销

### MoE vs Dense 节能差距的根本原因

| 因素 | Dense (Llama-3-70B) | MoE (Qwen3-30B-A3B) |
|------|--------------------|--------------------|
| 原生功耗 | 184W/GPU（compute-bound） | 94W/GPU（memory-bound） |
| PDAF 架构降功耗 | → 99W/GPU（-46%） | → 75W/GPU（-20%） |
| DVFS 频率调节空间 | 大（930→210 对计算密集型有显著影响） | 小（memory-bound 对频率不敏感） |
| 总节能 | 42.8%（架构+DVFS 双重收益） | 18.5%（几乎全靠 DVFS） |

## 结论

1. MoE 模型 PDAF+Tier 在 Conv Light 上的节能已接近上限（~18-19% vs Native+Tier）
2. 要达到 30% 需要超出 SM clock DVFS 的能力范围（如 GPU power capping、动态关闭 GPU、或 HBM power management）
3. Idle freq lock 机制本身是正确的设计，在更高功耗的 GPU（如 H100/H200）或 compute-bound 模型上可能有更大收益
4. MoE 的最大节能贡献来自 DVFS 对 Decode DF 的降频（61% 时间在 210MHz），这已经是当前架构下的最优策略

## 代码变更清单

- `python/sglang/srt/server_args.py`：新增 `afd_dvfs_idle_lock`, `afd_dvfs_idle_lock_freq` 参数
- `python/sglang/srt/managers/scheduler.py`：idle freq lock 逻辑 + 恢复逻辑 + controller 初始化
- `python/sglang/srt/disaggregation/prefill.py`：PA/PF disagg prefill event loop 的 idle lock
- `python/sglang/srt/energy/af_dvfs_controller.py`：`headroom_aggressive_threshold` 参数（当前禁用）
- `benchmark/AFlex_bench/06_others/more_model/scripts/run_moe_bench.py`：Tier 模式传入 idle-lock 参数
- `benchmark/AFlex_bench/06_others/more_model/charts/moe_idle_lock_comparison.png`：优化前后对比图
