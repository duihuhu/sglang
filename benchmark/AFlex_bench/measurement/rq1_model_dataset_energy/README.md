# RQ1：模型、数据集与部署架构能效

该目录包含 6 模型 × 3 架构 × 6 定长负载 × 4 档 QPS × 3 次重复的完整测试框架。详细运行方法见 `docs/runbook.md`，指标定义见 `docs/data_dictionary.md`。

- `configs/`：模型、节点、矩阵和 SLA
- `scripts/`：负载生成、canary/formal runner、报告
- `data/workloads/`：冻结请求 trace 和哈希索引
- `results/`：运行状态与原始 artifact
- `analysis/`：RQ1 报告分析入口
- `docs/`：设计、运行和数据说明
- `tests/`：矩阵与恢复语义测试

所有运行命令默认 dry-run；只有 `--execute` 会触发远程/GPU 操作。
