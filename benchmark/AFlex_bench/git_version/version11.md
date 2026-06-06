# Version 11: Unified DVFS + TTFT Processing Time + Benchmark Reorganization

## 核心改动概要

本版本主要包含三方面改动：
1. **Unified Single-Knob DVFS 控制器**（支持 PD / Native 架构的调频）
2. **TTFT Processing Time 指标**（去除排队延迟的纯处理时间）
3. **Benchmark 重组**（af_bench → AFlex_bench，版本日志迁移）

---

## 一、新增功能：Unified DVFS 控制器

### 文件
- `python/sglang/srt/energy/unified_dvfs_controller.py`（**新增**）
- `python/sglang/srt/server_args.py`（+60 行）
- `python/sglang/srt/managers/scheduler.py`（+229 行）

### 说明
为 PD（Prefill-Decode 分离）和 Native（原生 DP）架构实现了单频率旋钮的 Tier2 DVFS 控制器，与 PDAF 的 AF 分离双旋钮（f_A/f_F）互补。

- 新增 CLI 参数：`--dvfs-enabled`, `--dvfs-energy-model-dir`, `--dvfs-ttft-slo-ms`, `--dvfs-tpot-slo-us`
- Scheduler 中新增 `_init_unified_dvfs`, `_unified_dvfs_before_batch`, `_apply_freq_single`, `_compute_unified_prefill_slack`, `_unified_kv_util` 等方法
- 在 `event_loop_normal` 和 `event_loop_overlap` 的 batch 执行前调用频率决策
- Decode 阶段：基于窗口化重评估（bs 变化/SLO 紧急/窗口到期），选择满足 TPOT SLO 的最低能耗频率
- Prefill 阶段：基于 TTFT slack 搜索最低能耗频率
- KV-cache 利用率 >85% 时强制最高频率（避免 OOM）

### 已知限制
- PD disagg 的 event loop（`event_loop_normal_disagg_decode/prefill`）中尚未插入 `_unified_dvfs_before_batch` 调用，导致 PD disagg 模式下 DVFS 不生效（全程 1410MHz）
- Native DP 模式已验证生效（降至 930-1170MHz，节能 ~37%）

---

## 二、TTFT Processing Time（去除排队延迟）

### 文件
- `python/sglang/srt/observability/req_time_stats.py`（+22/-14）
- `python/sglang/srt/managers/tokenizer_manager.py`（+17）
- `python/sglang/srt/disaggregation/utils.py`（+19/-16）

### 说明
引入 `ttft_pure_processing` 指标 = `prefill_finished_time - prefill_run_batch_start_time`，精确衡量 Prefill 纯计算延迟，排除请求排队时间和网络传输时间。

- `req_time_stats.py`：在 metrics 输出中新增 `ttft_pure_processing` 字段；在 metrics 关闭时也传递 `prefill_run_batch_start_time` 和 `prefill_finished_time`
- `tokenizer_manager.py`：在非 metrics 模式下也输出 `ttft_pure_processing` 到 `meta_info`
- `disaggregation/utils.py`：KV Transfer metadata 中的 TTFT 改为纯处理时间（`pft - pbs`），而非原来的端到端（`pft - dispatch_time`）

---

## 三、AFD DVFS 控制器增强

### 文件
- `python/sglang/srt/energy/af_dvfs_controller.py`（+146）
- `python/sglang/srt/energy/af_profile_predictor.py`（+132）
- `python/sglang/srt/managers/scheduler.py`（部分）
- `python/sglang/srt/managers/scheduler_afd_mixin.py`（+15）

### 说明
- **V2 Pipeline Coupled Model**：新增 `_select_freq_decode_coupled` 方法，基于 A/F 延迟耦合关系做非对称调频（f_A 保高保延迟，f_F 降低省能耗）
- **Online Calibration**：新增 `update_calibration` / `get_calibration_factor` 方法，基于观测 TPOT 与预测值的偏差做 EMA 修正
- **Prefill 后调频**：新增 `_afd_dvfs_after_prefill_batch` 在 prefill batch 完成后立即调整频率
- **AFProfilePredictor 扩展**：新增 LUT+LinearReg 混合预测、Decode iter 级别能耗/延迟模型支持

---

## 四、Dynamic Micro-Batch

### 文件
- `python/sglang/srt/server_args.py`（+`afd_dynamic_micro_batch`, `afd_dynamic_mb_threshold`）
- `python/sglang/srt/managers/scheduler_afd_mixin.py`

### 说明
新增 `--afd-dynamic-micro-batch` 和 `--afd-dynamic-mb-threshold` 参数。当 decode batch_size < threshold 时自动切换 M=1（减少 pipeline bubble），高负载时使用配置的 M 值。

---

## 五、Benchmark 重组

### 结构变化
- **删除** `benchmark/af_bench/`（旧目录，178 文件）
- **新增** `benchmark/AFlex_bench/`（新目录，6664 文件），按论文结构重新组织：
  - `01_micro_benchmark/` — 微观性能分析（能耗模型、profiling）
  - `02_macro_benchmark/` — 宏观部署对比（4-GPU/8-GPU，多方案多 workload）
  - `03_sensitivity/` — SLO 敏感度分析（TTFT×TPOT 热力图、M 值 sweep）
  - `04_ablation/` — 消融实验（AF 组件、通信、Tier1/Tier2）
  - `05_breakdown/` — 细粒度分解（pipeline、DVFS 决策、吞吐）
  - `06_others/` — 设计文档、版本日志、工具脚本
  - `git_version/` — Git 提交版本记录
- **版本日志迁移**：`python/sglang/srt/energy/versions/version*.md` → `benchmark/AFlex_bench/06_others/version_logs/`

### 删除的代码内文档
- `python/sglang/srt/energy/versions/` 下所有 md 文件（version0-10，共 ~2000 行）移至 benchmark 目录

---

## 六、其他新增

- `python/sglang/srt/energy/workload_collector.py`（+13）：采集统计增强
- `benchmark/test_motivation/energy_model_v2.py`（新增）：V2 能耗模型训练脚本
- `benchmark/test_motivation/hucc/bench_decode_pipeline.py`（新增）：Decode pipeline profiling

---

## 统计

| 类别 | 文件数 | 行数变化 |
|------|--------|----------|
| Python 代码修改 | 9 | +611/-38 |
| 新增 Python 文件 | 3 | — |
| 删除旧 benchmark | ~178 | -43,000+ |
| 新增 AFlex_bench | ~6,664 | — |
| 总计 | — | +611 / -45,244 |
