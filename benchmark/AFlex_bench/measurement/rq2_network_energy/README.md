# RQ2：不同网络下的分离架构能效

固定模型 Qwen3-30B-A3B，比较 Native、PD、AF 在 PCIe、NVLink 与 RDMA 拓扑上的性能和能效。目录自包含测试配置、冻结负载、原始结果、分析脚本与文档。

## 目录

- `configs/`：集群与实验矩阵
- `scripts/`：preflight、canary、正式运行、恢复和报告
- `data/workloads/`：六种固定长度负载
- `data/summary/`：按网络/GPU 数整理的数据集
- `results/raw/`：原始不可变实验 artifact
- `analysis/`：统一结果索引与分析
- `docs/`：数据字典、运行手册和迁移审计
- `tests/`：runner/placement/report 回归测试

## 安全原则

脚本默认只执行本地 inventory/dry-run。远程执行仍要求各 runner 的 `--execute`/`--resume`/canary gate。详见 `docs/runbook.md`。
