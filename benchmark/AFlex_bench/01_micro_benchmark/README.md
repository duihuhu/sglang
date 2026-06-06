# 01 — Micro-benchmark（定长数据集）

**实验目的**：在**固定 input/output length**、只改变 QPS（请求到达率）的受控条件下，对比 PDAF / PD / Native 各架构方案的性能与能耗。定长消除了序列长度波动的干扰，便于精确刻画各方案在特定负载形态下的能效特征。

## 涉及的历史版本

| 文档 | 内容 |
|------|------|
| [`version0_4gpu_fixed.md`](version0_4gpu_fixed.md) | 4 卡（GPU 4-7）PD+AF，tier1_freq / max_freq / auto_freq 三模式对比；定长 il×ol×qps sweep；**tier1 稳定省电 ~30% 且 0% SLO 违背**，auto≈max（硬件自动 boost 省不了电） |
| [`version4_8gpu_fixed.md`](version4_8gpu_fixed.md) | 8 卡 il×ol sweep（il128_ol1024 / il512_ol256 / il2048_ol64 / il4096_ol64），PD TP4 / PD DP4 / PDAF DynM / PDAF Tier+DynM 四方案；Tier 在 prefill-heavy 负载省 35~37% |

## 目录结构

| 路径 | 内容 |
|------|------|
| `4gpu/fixed_qps/` | 4 卡定长结果 JSON + `compare_*.png`（能耗/功率/吞吐/TTFT/TPOT/SLO vs QPS） |
| `4gpu/fixed_qps_old/`, `fixed_qps_verify/` | 早期/校验数据 |
| `8gpu/deploy/` | 8 卡部署对比结果与 `figures/`（deploy_tier_tradeoff 等） |
| `8gpu/charts_v4/` | version4 的全套图表（throughput/ttft/tpot/slo/energy/efficiency vs qps） |
| `8gpu/results_8gpu_fixed/` | 8 卡 il×ol×qps 结果 JSON |
| `disagg_arch_comparison/` | PD+AF vs PD TP1 vs PD TP2 单请求与并发对比（含报告 md + 原始 log + 脚本） |
| `workloads/` | 定长数据集 `fixed_il<I>_ol<O>_qps<N>.jsonl` |
| `scripts/` | `run_fixed_qps_bench.py`、`run_deploy_bench.py`、`plot_*.py`、`bench_real_gpu.py` |
| `raw_logs/` | 各 run 的原始 server 日志（按 `il*_ol*_qps*` / `deploy*` 命名） |

## 复现命令

```bash
cd /workspace/sglang-tier
PY=/workspace/env/sglang-tier/bin/python
# 生成定长数据集
$PY benchmark/AFlex_bench/06_others/utils/energy_bench_utils/gen_fixed_workload.py \
    --output-dir workloads --qps 1,2,4,6 --input-len 512 --output-len 128
# 跑 sweep（三模式，崩溃保护）
$PY benchmark/AFlex_bench/01_micro_benchmark/scripts/run_fixed_qps_bench.py \
    --workload-glob 'workloads/fixed_il512_ol128_qps*.jsonl' \
    --modes tier1_freq,max_freq,auto_freq
# 画图
$PY benchmark/AFlex_bench/01_micro_benchmark/scripts/plot_fixed_qps.py
```

## 关键结论

1. **tier1_freq 稳定省电约 30%**（均衡/长输入组 26~34%；decode 重组 18~26%），未饱和时 0% SLO 违背。
2. **auto_freq ≈ max_freq**：持续负载下硬件自动 boost 到接近最高频，"不干预"≈"锁最高频"。
3. **吞吐**：PD TP4 > PD DP4 >> PDAF DynM > PDAF Tier（AF 通信开销 ~75-95ms/iter 拖低吞吐）。
4. **Tier 节能在 prefill-heavy 负载最大**（il2048/il4096 省 35~37%），decode-heavy 下几乎无收益甚至负优化。
