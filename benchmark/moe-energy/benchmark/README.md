# MoE Energy Benchmark

Qwen3-30B-A3B MoE 在 A800 上的部署与压测工具集。

## 目录结构

```
benchmark/
├── scripts/     # 启动、压测、汇总、batch 监控脚本
├── doc/         # 测试报告与实验记录（Markdown）
└── data/        # 压测结果与导出数据（默认输出目录）
```

## 快速开始

```bash
# 四宫格突发并发
bash benchmark/scripts/run_attn_moe_matrix_parallel.sh

# 四宫格稳态 QPS（含 batch 监控）
bash benchmark/scripts/run_steady_qps_matrix.sh

# 汇总已有结果
python3 benchmark/scripts/summarize_steady_qps_matrix.py benchmark/data/steady_qps_matrix
```

## 脚本索引

| 类别 | 脚本 |
|------|------|
| 四宫格突发 | `run_attn_moe_matrix_parallel.sh`, `run_matrix_config.sh` |
| 稳态 QPS | `run_steady_qps_matrix.sh`, `run_steady_qps_same_config_parallel.sh`, `run_steady_qps_point.sh` |
| TP vs EP | `run_tp_vs_ampere_ep_parallel.sh`, `run_single_backend_sweep.sh` |
| 汇总 | `summarize_attn_moe_matrix.py`, `summarize_steady_qps_matrix.py`, `summarize_tp_vs_ampere.py` |
| 监控 | `monitor_batch.py` |

## 文档

- [四宫格突发并发报告](doc/attn-moe-matrix.md)
- [无 CUDA Graph 对照](doc/no-cuda.md)
- [EP 通信专项](doc/ep-test.md)
