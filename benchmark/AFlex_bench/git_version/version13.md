# Version 13: MoE 异构 PDAF 部署 + 流式 TPOT 测量

## 概述

本版本实现了 MoE 模型 (Qwen3-30B-A3B) 的**异构 PDAF 部署方案** 和**流式 TPOT 计算方法**，解决了对称 PDAF 在重负载下排队严重的问题。

## 主要修改

### 1. `run_moe_bench.py` — MoE Benchmark 脚本

**流式 TPOT 计算 (与 Dense 一致)：**
- `_send_request()` 改为流式 API (`"stream": True`)
- TPOT = `(last_token_time - first_token_time) / (token_count - 1)`，排除排队延迟
- TTFT 分两种：`ttft_ms` (客户端含排队) 和 `ttft_proc_ms` (服务端纯 prefill)
- 添加 `--enable-metrics` 到 PDAF 服务端启动参数

**异构 PDAF 部署 (3P+5D)：**
- 新增 `start_pdaf_asym()` 方法
- 布局: PA(TP=1, 1GPU) + PF(TP=2, 2GPU) + DA(TP=1, 1GPU) + DF(TP=4, 4GPU)
- Prefill CVD="0,1,2", Decode CVD="3,4,5,6,7"
- 给 Decode 侧更多 GPU 增大 KV cache 容量
- 新增配置: `pdaf_asym_1p6d`, `pdaf_asym_1p6d_tier`

### 2. `plot_moe_comparison.py` — 对比图绘制

- 1×3 子图: Energy / TTFT / TPOT
- 四方案对比: Native DP8, PD DP4, PDAF Sym (4P+4D), PDAF Asym (3P+5D)
- 每方案展示 NoTier (实色) 和 +Tier (斜线) 两组柱状图
- 输出到 `charts/moe_simplified_comparison.png`

### 3. `af_dvfs_controller.py` — DVFS 控制器

- 支持 `tp_a != tp_f` 的异构 TP 配置
- `moe_freq_floor` 参数（当前设为 0 禁用）
- online calibration 支持分离 M=1 和 M>1 的校准因子

### 4. `af_profile_predictor.py` — 能耗预测器

- 支持异构 TP 查询（tp_a, tp_f 分别传入）
- KNN 外推机制：当查询 TP=4 但模型只有 TP=1,2 时，自动外推到最近邻

### 5. `scheduler.py` — 调度器

- DVFS 控制器初始化时正确传入 `tp_a` 和 `tp_f`：
  ```python
  tp_a = getattr(server_args, "afd_attn_tp", None) or server_args.tp_size
  tp_f = getattr(server_args, "afd_ffn_tp", None) or server_args.tp_size
  ```

### 6. `run_8gpu_deploy_bench.py` — Dense 8GPU Benchmark

- 新增 `pdaf_8g_asym_1p6d` 和 `pdaf_8g_asym_1p6d_tier` 部署配置
- 异构 PDAF 启动函数 `start_pdaf_asym()`

## 新增文件

| 文件 | 说明 |
|------|------|
| `version_logs/version10.md` | MoE 异构 PDAF 测试分析报告 |
| `train_v4_models.py` | V4 能耗模型训练（含 expert load 特征） |
| `collect_expert_energy_data.py` | Expert 负载相关能耗数据采集 |
| `profile_expert_load_vs_latency.py` | Expert 负载 vs 延迟关系 profiling |
| `bench_decode_pipeline_moe_tp2_210mhz.py` | MoE TP2 210MHz 数据采集 |

## 实验结果

32 组实验 (4方案 × Tier/NoTier × 4 workloads)，GPU 独占环境：

| 方案 | 能耗节省 (Tier) | 重负载吞吐 (Conv Heavy) |
|------|:-:|:-:|
| Native DP8 + Tier | 7.8-9.2% | 943 tok/s (无损) |
| PD DP4 + Tier | 9.2-10.5% | 941 tok/s (无损) |
| PDAF Sym + Tier | 5.9-23.3% | 619 tok/s (-24%) |
| **PDAF Asym + Tier** | **16.8-24.6%** | **789 tok/s (-12%)** |

## 已知限制

- 异构方案 (DA-TP1 + DF-TP4) 的能耗模型是**外推**的（训练数据只有 TP=1,2）
- 需要补充 DA-TP1 / DF-TP4 的实际 profiling 数据
