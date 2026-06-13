# Version 12: MoE 模型支持 + Expert Load Imbalance (LIF) 感知 DVFS

## 核心改动概要

本版本主要包含以下改动：
1. **MoE 模型 (Qwen3-30B-A3B) 全流程 Benchmark 支持**
2. **Expert Load Imbalance Factor (LIF) 特征设计与实现**
3. **PDAF 跨进程 LIF 通信机制**
4. **能耗模型数据重新采集（batch 覆盖至 128）并重训**
5. **Benchmark 基础设施完善**

---

## 一、MoE Expert Load Imbalance (LIF) 特征

### 背景

MoE 模型（如 Qwen3-30B-A3B，128 experts / 8 active）在 PDAF+Tier DVFS 下 SLO 违背率显著高于 Dense 模型。根因分析发现：
- MoE 的 expert routing 导致**相同 batch_size + seq_len 下延迟方差极大**
- 传统能耗模型仅用 (bs, il, freq) 预测延迟，无法捕捉 expert 负载不均衡
- DVFS 因预测偏低而选择过低频率，导致 SLO 违背

### 新增文件

- `python/sglang/srt/energy/expert_load_metric.py`（**新增**）

### 指标定义

| 指标 | 公式 | 含义 |
|------|------|------|
| ELS (Expert Load Skew) | `max(expert_counts) / mean(expert_counts)` | 单层 expert 热度偏斜 |
| GLR (GPU Load Ratio) | `max(gpu_load) / mean(gpu_load)` | TP 间 GPU 负载比 |
| LIF (Load Imbalance Factor) | `GLR × √ELS` | 综合标量，≥1.0 |

### 实现

- `compute_els(topk_ids, num_experts)` → float
- `compute_glr(topk_ids, num_experts, tp)` → float
- `compute_lif(topk_ids, num_experts, tp)` → float
- `compute_lif_from_counts(expert_counts, num_experts, tp)` → float

---

## 二、AFProfilePredictor LIF Correction

### 文件
- `python/sglang/srt/energy/af_profile_predictor.py`（修改）

### 改动
- `predict_iteration_latency` / `predict_iteration_energy` 新增 `lif: float = 1.0` 参数
- 新增 `_lif_correction(lif)` 静态方法，采用幂律修正：
  ```
  correction = 1.0 + 0.3 × (lif - 1.0)^0.7
  ```
  当 lif > 1 时，预测延迟乘以 correction，使 DVFS 对负载不均衡场景选择更高频率
- `find_best_freq_pair_coupled` 传入 lif 参数

---

## 三、AFDVFSController LIF 传递

### 文件
- `python/sglang/srt/energy/af_dvfs_controller.py`（修改）

### 改动
- `select_freq_decode` / `_select_freq_decode_coupled` 新增 `lif: float = 1.0` 参数
- 在频率搜索循环中将 lif 传递给 predictor 的 latency/energy 预测

---

## 四、Scheduler LIF 计算 + 跨进程通信

### 文件
- `python/sglang/srt/managers/scheduler.py`（修改）

### 改动
- 新增状态变量：`_lif_ema`, `_moe_num_experts`, `_moe_tp_for_lif`
- 新增方法：
  - `_compute_lif_for_dvfs(batch)`: 从 `RoutedExpertsCapturer` 获取 topk_ids → 计算 LIF → EMA 平滑
  - `_write_lif_shared(lif)`: 写入共享文件（FFN 侧写）
  - `_read_lif_shared()`: 从共享文件读取（ATTN 侧读）
- FFN 进程：decode batch 后计算 LIF 并写入 `/tmp/afd_lif_shared.txt`
- ATTN 进程：DVFS 决策前读取 LIF 值

### 跨进程通信方案
PDAF 架构中 expert routing 发生在 FFN (DF) 进程，而 DVFS 控制在 ATTN (DA) 进程。通过环境变量 `AFD_LIF_SHARED_PATH` 指定共享文件路径，实现跨进程 LIF 传递。

---

## 五、MoE Benchmark 基础设施

### 新增文件
- `benchmark/AFlex_bench/06_others/more_model/scripts/run_moe_bench.py`（新增）
- `benchmark/AFlex_bench/06_others/more_model/energy_model/` 目录（模型 pkl + 训练脚本）
- `benchmark/AFlex_bench/06_others/more_model/results/`（结果目录）
- `benchmark/AFlex_bench/06_others/more_model/charts/`（图表）

### run_moe_bench.py 改动
- 支持 6 种部署方案：`native_dp8`, `native_dp8_tier`, `pd_dp4`, `pd_dp4_tier`, `pdaf_tp2`, `pdaf_tp2_tier`
- PDAF 启动时自动传入 `--enable-return-routed-experts` 和 `AFD_LIF_SHARED_PATH`
- 4 个 Azure workload：code_medium, conv_light, conv_medium, conv_heavy

---

## 六、能耗模型数据重采集 & 重训

### 数据采集
- 覆盖 batch_size 1~128（之前仅到 64）
- 数据文件：`benchmark/AFlex_bench/06_others/more_model/energy_model/data/decode_pipeline_moe_tp2.txt`（1142 行）
- 使用 KNN LookupModel 重训，模型保存至 `energy_model/models_lut/`

### .gitignore 更新
能耗模型数据文件（`.txt`）已从 git 跟踪中排除：
```
**/energy_model/data/*.txt
**/retrain/data/*.txt
```

---

## 七、当前 Benchmark 结果（PDAF TP2 + Tier + LIF）

| Workload | Throughput | TPOT avg | TPOT p99 | Energy | SLO Violation |
|----------|-----------|----------|----------|--------|:---:|
| azure_code_medium_real | 105.9 tok/s | 229ms | 963ms | 5506 mJ/tok | 20.2% |
| azure_conv_light_real | 314.1 tok/s | 144ms | 447ms | 1849 mJ/tok | 11.5% |
| azure_conv_medium_real | 560.3 tok/s | 173ms | 847ms | 1072 mJ/tok | 14.3% |
| azure_conv_heavy_real | 623.8 tok/s | 5488ms | 63741ms | 959 mJ/tok | 83.9% |

### 分析
- LIF 机制有效降低了前三个 workload 的 SLO 违背（code_medium: 26% → 20%）
- conv_heavy 的 83.9% 违背源于 **排队效应**（2997 并发请求超出 MoE decode 吞吐上限），非频率问题
- MoE 模型延迟方差大（同 bs+il，不同 expert 分布可导致 2-5x 延迟差异），heuristic LIF correction 改善有限

---

## 八、下一步规划

### Part 1: 异构 AF 的数据集适应

当前 PDAF 使用固定 TP 配置（tp_A = tp_F = 2），所有 workload 共用同一组参数。下一步：

1. **Workload-Aware 异构 TP 策略**
   - 针对不同 workload 特征（code vs conv，light vs heavy），探索最优 tp_A / tp_F 比例
   - 例如 conv_heavy 高并发场景可能受益于 tp_F > tp_A（FFN 计算密集型）
   - 实现 workload 分类器 → 自动选择 TP 配比

2. **Dynamic Micro-Batch (M) 适配**
   - 根据实时 batch_size 动态切换 M=1/2/3
   - 低负载时 M=1 减少 pipeline bubble，高负载时 M=2/3 提高吞吐

3. **Multi-Dataset Profiling**
   - 对不同类型 workload 分别采集 profiling 数据
   - 训练 workload-specific 能耗模型（而非共用一套）

### Part 2: MoE 能耗模型优化

当前 heuristic LIF correction 效果有限，需要更精确的预测机制：

1. **LIF-Aware 能耗模型训练**
   - 将 LIF 作为模型的显式输入特征（7 维：tp_a, tp_f, M, f_a, f_f, il, bs → 加 lif 变为 8 维）
   - 采集大量含 LIF 标注的 profiling 数据（需解决 PDAF 模式下 routed_experts 回传问题）
   - 方案：单 GPU 模式采集 → 映射到 PDAF 配置

2. **Expert Routing 预测器**
   - 基于历史 routing pattern，预测未来几步的 LIF 分布
   - 使用滑动窗口 + 统计模型（EWMA / 线性回归）提前感知负载变化

3. **MoE-Specific DVFS 策略**
   - 针对 MoE 的 memory-bound 特性，设计保守调频策略（频率下限更高）
   - 引入 "batch size cliff" 检测：当 bs 接近临界点时锁定高频
   - conv_heavy 类极端 workload：引入 admission control 或 request throttling

4. **模型精度验证框架**
   - 建立 offline 评估 pipeline：预测 vs 实测 延迟对比
   - 计算 MAPE / R² 等指标，量化 LIF 特征带来的精度提升
   - 目标：将前三个 workload 的 SLO 违背降至 5% 以下

---

## 统计

| 类别 | 文件数 | 说明 |
|------|--------|------|
| 新增 Python 文件 | 1 | expert_load_metric.py |
| 修改 Python 文件 | 4 | af_profile_predictor, af_dvfs_controller, scheduler, run_moe_bench |
| 新增 Benchmark 目录 | 1 | 06_others/more_model/ |
| .gitignore 更新 | 1 | 排除能耗数据 .txt |
