# 03 — Sensitivity（敏感度测试）

**实验目的**：固定架构，**变换关键配置参数**，观察系统能效与 SLO 表现的敏感度。三条主线：SLO 阈值扫描、微批数 M 扫描、QPS/并发扫描。

## 子目录

### `slo_sweep/` — SLO 阈值敏感度
扫描 TPOT / TTFT SLO，对比三种调频策略（Baseline 满频 / V1 旧模型 / V2 耦合模型）。

| 路径 | 内容 |
|------|------|
| `retrain/` | DVFS 预测模型重训 + 全套 SLO sweep（`run_joint_slo_sweep.py`、`run_ttft_sweep.py`、`analyze_*`、`figures/`、`models_v2/`、各 `logs_*_sweep/`、`results_*_sweep/`） |
| `results_4gpu_tpot_sweep/` | 4 卡 TPOT SLO 单维扫描结果 |
| `results_4gpu_v1/` | V1 模型扫描结果 |
| `results_slo_sweep_early/`, `_heavy_early/` | 最早期 SLO 分档结果（loose/medium/tight 等） |

对应文档 [`version5_multi_slo.md`](version5_multi_slo.md)：V2 在 SLO≥150ms 内保持 ≤0.25% 违背 + ~30% 节能，V1 在 SLO<200ms 时崩溃（96~100% token 超标）。

### `m_sweep/` — 微批数 M 敏感度
| 路径 | 内容 |
|------|------|
| `04_msweep/` | M=1~5 扫描（`run_msweep.py`、`run_m1_vs_m3_sweep.py` 等） |
| `micro-batch-opt/` | Interleaved async pipeline 优化（M=2 大 batch +58% 吞吐），含 `SUMMARY.md`、结果图、脚本 |
| `run_pdaf_m1_vs_m3.py` | PDAF M=1 vs M=3 对比 |

关键发现：M=2 interleaved 在 batch≥200 时 +58% 吞吐；M=3 因通信次数增加反而更差；DynM（动态阈值）在小 batch 走 M=1、大 batch 走 M=2。

### `qps_sweep/` — QPS / 并发敏感度
`05_qps_sweep/`：标准/高 QPS 扫描，含 M=1 / M=3 / native 各配置（`run_qps_sweep*.py`）。

### `raw_logs/`
TPOT 变体（`il256_ol512_qps*_tpot*`）、`4gpu_tpot_sweep`、`4gpu_v1` 等原始日志。

## 复现命令

```bash
cd /workspace/sglang-tier
PY=/workspace/env/sglang-tier/bin/python
# TTFT×TPOT 联合 SLO sweep
$PY benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/run_joint_slo_sweep.py
$PY benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/analyze_joint_sweep.py
# M sweep
$PY benchmark/AFlex_bench/03_sensitivity/m_sweep/04_msweep/run_msweep.py
```

## 关键结论

1. **V2 调频模型有效区间远大于 V1**：V2 在 SLO≥90ms 可用，V1 仅 SLO≥250ms 可用。
2. **TTFT SLO 主导 Prefill 频率，TPOT SLO 主导 Decode 频率**，两维基本正交。
3. **M=2 interleaved 在大 batch 下 +58% 吞吐**，M=3 通信开销过大。
4. **TPOT SLO 临界点 ~100ms**（系统 TPOT 底噪约 83ms），≤80ms 不可行（>93% 违背）。
