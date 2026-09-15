# RQ1 运行手册

该目录自包含 RQ1 配置、冻结负载、可恢复调度状态和报告脚本；执行时由脚本把相邻旧目录 `benchmark/src` 加入 `sys.path`，不复制核心包，也未修改共享代码。

## 矩阵与部署

- 6 模型 × Native/PD/AF × 6 定长负载 × QPS 2/4/8/16 × 3 repeats = 1296 formal runs。
- Native TP8；PD 为 P TP4 + D TP4；AF 为 A TP4 + F TP4。
- 每个负载/QPS 文件固定 seed、64 请求；`data/workloads/index.json` 冻结 SHA-256。LongContext 为 16384/256。
- 18 个 canary（6 模型 × 3 架构，balanced QPS2）全部 valid 后才允许 formal execute。

## 使用

```bash
python3 scripts/generate_workloads.py
python3 scripts/run_rq1.py --node node1                         # 默认 dry-run
python3 scripts/run_rq1.py --node node1 --phase canary --execute
python3 scripts/run_rq1.py --node node1 --check-gate
python3 scripts/run_rq1.py --node node1 --phase formal --execute
python3 scripts/report_rq1.py --results results/default
```

可重复指定 `--model/--architecture/--workload/--qps/--repeat`，也可传逗号列表。`--execute` 是唯一远程/GPU 执行开关。`scheduler.lock` 防止并发调度，`progress.json` 使用 fsync + replace 原子更新。缺失容器模型路径记录为 `blocked_model_missing`；较低 QPS 的 achieved/target 低于 0.9 后，同一模型/架构/负载/repeat 的更高 QPS 标为 `skipped_saturated`。失败项需 `--retry-failed`。

默认 cluster 仅给出本机占位 `node1`；正式运行前应只在本目录修改 `configs/cluster.json` 的节点 host、容器和 GPU/NIC 映射。SLA 阈值位于 `configs/matrix.json`，正式实验前需按已批准口径确认。
