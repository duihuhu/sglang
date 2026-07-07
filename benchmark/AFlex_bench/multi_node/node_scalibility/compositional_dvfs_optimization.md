# Compositional Decode DVFS 模型优化记录

## 背景

PDAF 16 卡部署中，原 V2 pipeline decode DVFS 模型是在 TP=1 数据上训练的，无法准确预测 TP=4 场景。后续如果要支持更多卡数（最多 16 张）和异构 D 阶段 AF 配置，V2 pipeline 模型需要针对每种 (TP_A, TP_F) 组合重新 profiling，成本过高。

**目标**：用 V1 per-layer 模型（已有各 TP 数据）+ 模拟 AF 传输开销的方式替代 V2 pipeline 模型，使 decode DVFS 性能达到 V2 旧的水平，同时不需要为新的 TP 配置重新 profiling。

## 优化过程

### 1. 初始实现：V1 Compositional 路径

**文件**: `python/sglang/srt/energy/af_dvfs_controller.py`

在 `select_freq_decode` 中新增 V1 compositional 路径：
- 使用 V1 per-layer GBDT 模型分别预测 `lat_A(f)` 和 `lat_F(f)`
- 加上可配置的 `t_comm_us`（AF 层间 IPC 通信开销）
- 公式：`t_layer = lat_A + lat_F + t_comm_us`，`t_iter = t_layer × num_layers`

**问题**：初始版本每次 iteration 做 7×7=49 个 freq pair 的完整扫描，每个 pair 需要 2 次 GBDT predict → 总计 98 次 predict ≈ 680ms/iteration → TPOT 从 108ms 飙到 400ms+。

### 2. 优化一：Predict LUT Cache

**改动**：按 `(bs, il, ol)` 缓存 per-freq 的 latency 和 energy 预测值。

- Cache miss 时：做 7+7=14 次 latency predict + 14 次 energy predict = 28 次（而非 98 次）
- Cache hit 时：直接查表，零 predict 开销
- 49 个 freq pair 的 SLO check 变成纯 dict lookup + 算术

**效果**：TPOT 从 400ms+ 降到 ~128ms。

### 3. 优化二：Energy 预计算 Bug 修复

**问题**：energy LUT 的构建代码不在 `else` 分支内，导致即使 cache 命中也会重新构建 `_e_a_lut`（14 次 predict）。

**修复**：将 energy 预计算移入 cache miss 分支，与 latency 一起缓存。

### 4. 优化三：Hold Window 机制

**问题**：V1 路径的 Python 逻辑（dict lookup、candidates 排序、SLO check 遍历）每次 reeval 仍有 ~18ms 开销。频繁 reeval 导致 TPOT 比 baseline 高 15ms。

**改动**：
- 在 `should_reevaluate_decode` 中，hold 期间且频率已是 max 时，直接返回 `REEVAL_NONE`，跳过所有后续逻辑
- Fallback to max freq 后立即设置 `hold_iters_remaining = 500`（约 50 秒）
- Scheduler 层加 fast path：hold 期间只做 `tick_decode_iteration()` + 更新 timestamp，跳过 `compute_window_size`、`_write_afd_decode_stats` 等

**效果**：TPOT 在 QPS=4 时从 128ms 降到 108ms（追平 baseline）。

### 5. 优化四：禁用 Online Calibration

**问题**：`online_calibration` 同时自适应 `calibration_factor` 和 `t_comm_us`，两个参数双重补偿，导致 `t_comm_us` 从 2900 爆涨到 7500+，`calib` 到 2.35，预测 iter=517ms 远超 SLO=300ms → 全程 fallback to max。

**修复**：compositional 模式下不启用 `--afd-dvfs-online-calibration`，使用固定 `t_comm_us=3300`（实测 per-layer IPC overhead ≈ 2900us + scheduler overhead 375us/layer）。

### 6. 优化五：Compositional 频率下限

**问题**：V1 模型在极低频率（690MHz）时预测偏低，不包含 scheduler/memory 等非 compute 开销。选到 690MHz → 实际 TPOT=127ms，比预测的 114ms 高 13ms → GPU 多跑时间 → 能耗反而上升。

**改动**：compositional 模式下设 `freq_floor_v1 = 1050`，限制最低可选频率。

**效果**：TPOT 从 127ms 降到 116ms，E/tok 从 1540 降到 1514（QPS=8）。

## 最终配置

```
--afd-dvfs-decode-compositional
--afd-dvfs-feedback
--afd-dvfs-comm-us 3300.0
# 不启用 --afd-dvfs-online-calibration
# freq_floor=1050 (硬编码在 af_dvfs_controller.py 中)
```

## 最终结果（chatbot_lphd, 16 卡）

| QPS | V2 旧 TPOT | V2 旧 E/tok | Compositional TPOT | Compositional E/tok | 差距 |
|-----|-----------|-------------|-------------------|---------------------|------|
| 4   | 111.1 ms  | 1514 mJ     | 116.4 ms          | 1595 mJ             | +5.4% |
| 8   | 110.7 ms  | 1420 mJ     | 116.5 ms          | 1514 mJ             | +6.6% |
| 16  | 112.6 ms  | 1384 mJ     | 115.7 ms          | 1442 mJ             | +4.2% |

**平均 E/tok 差距：~5.4%**

## 剩余差距分析

~5% 的 E/tok 差距来源：
1. V1 per-layer 模型在中低频率（1050MHz）时仍有 ~5ms/iteration 的预测偏差（不含 scheduler overhead）
2. 这 5ms 让 GPU 多运行 5ms → 多消耗约 5% 能量

这是 compositional 模型用 V1 per-layer 近似 pipeline 行为的 inherent trade-off。

## 涉及文件

| 文件 | 改动 |
|------|------|
| `python/sglang/srt/energy/af_dvfs_controller.py` | V1 compositional 路径、LUT cache、hold window、freq floor、debug logging |
| `python/sglang/srt/managers/scheduler.py` | hold 期间 fast path（跳过非关键计算） |
| `python/sglang/srt/server_args.py` | 新增 `afd_dvfs_decode_compositional`、`afd_dvfs_comm_us` 参数 |
| `benchmark/.../run_compositional_ab.py` | A/B 测试脚本 |

## 后续可优化方向

1. **动态 t_comm_us**：根据实际 TP 配置自动推算（而非固定 3300）
2. **频率下限自适应**：根据 feedback（TPOT 违规次数）动态调整 freq_floor
3. **分离 calibration**：只自适应 t_comm_us，不动 calibration_factor（避免双重补偿）
