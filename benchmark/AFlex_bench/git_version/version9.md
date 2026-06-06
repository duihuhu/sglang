# v9 版本 -- Tier1 频率重规划解耦 + Tier2 DVFS 反馈/在线校准

## 概述

本版本围绕 **DVFS 节能控制（Tier1 频率重规划 + Tier2 逐 batch 调频）** 做了三块改动：①Tier2 调频器新增 SLO 反馈强制升频与在线校准能力；②Tier1 把"资源重规划（TP/k 变化触发整服务重载）"与"纯频率重规划"解耦，新增独立的重载编排进程；③新增 DVFS 决策日志用于量化调频/预测准确性。配套在 `benchmark/energy_bench/` 完成了 Qwen3-32B PD+AF 解耦上的节能评测（结论见该目录 `versions/version1.md`）。

代码改动统计（相对上一版本）：

| 文件 | 变更 | 说明 |
|------|------|------|
| `srt/managers/scheduler.py` | +454 行 | Tier1 重载编排 + freq-only transition + DVFS 决策日志 |
| `srt/energy/af_dvfs_controller.py` | +127 行 | Tier2 反馈升频 + 在线校准 |
| `srt/server_args.py` | +50 行 | 5 个新 DVFS/Tier1 参数 |
| `srt/energy/reload_orchestrator.py` | 新增 301 行 | Tier1 重载编排独立进程 |
| `srt/energy/reload_signal.py` | 新增 72 行 | 重载信号协议（idle/reloading/ready/error）|

## 一、新增参数（`server_args.py`）

| 参数 | 默认 | 说明 |
|------|------|------|
| `--afd-dvfs-feedback` | False | Tier2 反馈：预测器连续多次判定"无需升频"但 SLO 持续紧张时，强制升一档频率 |
| `--afd-dvfs-feedback-threshold` | 3 | 连续 N 次 urgent-no-change 后触发强制升频 |
| `--afd-dvfs-feedback-hold` | 30 | 强制升频后保持 N 个迭代再放开 |
| `--afd-dvfs-online-calibration` | False | Tier2 在线校准：用实测 TPOT 修正预测器偏差 |
| `--afd-dvfs-calibration-ema` | 0.2 | 校准因子 EMA 更新权重 |
| `--tier1-disable-reload` | False | Tier1 重规划不触发整模型重载，退化为纯频率重规划控制器 |


## 二、Tier2 调频器增强（`af_dvfs_controller.py`）

### 2.1 SLO 反馈强制升频
预测器有时系统性低估迭代时间，导致选频偏低却自认为"安全"。新增反馈机制：当 decode 侧连续 `feedback_threshold` 次因 SLO 紧张重评估、却都判定无需升频时，强制调用 `_next_freq_up()` 升一档，并保持 `feedback_hold` 个迭代，避免抖动。

### 2.2 在线校准
新增 `update_calibration(observed_tpot_us, predicted_tpot_us)`：用实测/预测比值按 EMA 更新一个全局 `calibration_factor`（初始 1.0），在选频时对预测延迟乘以该因子（`t_iter * calibration_factor`）做 SLO 判定。这样可在线纠正预测器的系统性偏差（评测中 decode 低估约 28%），无需离线重训模型。


## 三、Tier1 重规划解耦（`scheduler.py` + 新增编排进程）

### 3.1 资源重规划 vs 频率重规划解耦
Tier1 监控工作负载后求解新方案，原本 TP/k 变化会触发整服务重载。新增 `--tier1-disable-reload`：开启后 Tier1 跳过 `_trigger_full_reload`，只应用新解的频率（freq-only transition），不重启服务/不重载权重。用于隔离 Tier1 的"重调频"价值（G=4 下求解器 INFEASIBLE，节能实际全部来自 Tier2）。

### 3.2 重载编排进程（`reload_orchestrator.py` / `reload_signal.py`）
当确实需要 TP 变更时，PA scheduler 通过 `_build_reload_config()` 生成配置并 `_trigger_full_reload()` spawn 独立的 `reload_orchestrator` 进程：写重载信号 → 杀全部 4 个 server + router → 等端口释放、复位频率 → 用新 TP 配置重启 → 健康检查 → 写 ready 信号。`reload_signal.py` 定义信号协议（idle/reloading/ready/error），benchmark 据此在重载期间暂停发压、完成后恢复。新增 `_write_tier1_freq_config()` / `_poll_tier1_freq_config()` 做跨进程频率配置同步。


## 四、DVFS 决策日志（`scheduler.py`）

新增环境变量 `AFD_DVFS_DECISION_LOG` 控制，每进程独立写 JSONL（按 perspective/disagg/gpu 命名，无并发写冲突），用于量化调频决策与预测准确性：

- **prefill**：bs、il、slack_us、选中 f_a/f_f、预测延迟/能耗
- **decode**：bs、il、ol、当前与选中 f_a/f_f、是否切换、实测迭代时间 `obs_iter_us`、当前频率下预测迭代时间 `pred_iter_cur_us`、预测误差 `pred_iter_err_pct`、校准因子

`obs_iter_us` vs `pred_iter_cur_us` 直接量化预测模型准确性，是评测中发现 decode 系统性低估 ~28%、饱和点恶化到 -71% 的数据来源。


## 五、Benchmark 改动

### 5.1 `benchmark/energy_bench/` 重组（新增评测体系）
- **脚本重组**：原散落在顶层的脚本（`bench_real_gpu.py` 等）迁移到 `scripts/{bench,tier1,tier2,utils}/` 子目录分类管理。
- **核心评测脚本**：`scripts/bench/run_fixed_qps_bench.py`（三模式 tier1_freq/max_freq/auto_freq 定长+变长 sweep，含 SLO 违背统计、显存释放关卡、崩溃 watchdog）、`plot_fixed_qps.py`（含 P/D 分阶段能耗 8 面板对比图）、`plot_dvfs_decisions.py`（预测准确性 + decode/prefill 选频分布图）、`analyze_dvfs_decisions.py`。
- **数据/图片归档**：结果数据按 `results/fixed_qps/{json,logs,figures}/` 分目录，新增 `.gitignore` 避免数据/图片污染 git。
- **进展记录**：`versions/version0.md`、`version1.md`。

### 5.2 `benchmark/af_bench/micro-batch-opt/` 新增脚本
新增 `plot_throughput.py`、`run_high_conc_v2.py`、`run_len_sweep.py`、`run_mrr_sweep.py`、`run_pd_tp1_only.py`、`run_supplement.py` 等并发/长度 sweep 脚本（配合 v8 的 M=2 async pipeline 评测）。


## 六、评测结论（详见 `benchmark/energy_bench/versions/version1.md`）

在 Qwen3-32B / PD+AF 解耦（M=2，A800×4）上：

1. tier1 逐 batch 调频稳定省电 **18-34%**（均衡/长输入组 30-34%、decode 重组 18-26%，变长 trace 31-33%），未饱和时 0% SLO 违背；auto 约等于 max，说明靠硬件默认调频省不了电，必须主动降频。
2. 能耗按 prefill/decode 拆分后，**Decode 是节能主战场**（占总能耗 52-71%），且 Prefill 节能随 QPS 升、Decode 节能随 QPS 降。
3. SLO 违背全部来自 **TTFT 排队**、TPOT 从不违背；高 QPS 长输出下的违背是系统容量天花板（三模式违背率一致），与调频无关。
4. 两个待校准点：预测模型系统性低估 decode 延迟约 28%（饱和点 -71%）→ 用 `--afd-dvfs-online-calibration` 修正；prefill 调频的 slack 指标看不到 waiting queue 排队积压。

## 七、使用方式

```bash
# Tier1 freq-only（不重载）+ Tier2 DVFS + 在线校准
sglang serve /models/Qwen/Qwen3-32B/ \
  --afd-perspective attn --afd-comm-backend ipc_cpp \
  --afd-micro-batch 2 --afd-async-pipeline \
  --afd-dvfs-enabled --afd-energy-model-dir <energy_models> \
  --afd-dvfs-online-calibration --afd-dvfs-feedback \
  --enable-tier1-pa --tier1-disable-reload

# DVFS 决策日志
export AFD_DVFS_DECISION_LOG='logs/dvfs_{persp}_{disagg}_gpu{gpu}.jsonl'
```

