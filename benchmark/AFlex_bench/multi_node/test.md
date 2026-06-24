# Multi-Node PDAF Benchmark

16 卡 / 双 8 卡节点的 PDAF（4PA4PF4DA4DF）基准。完整说明见 [README.md](./README.md)。

- 部署：Prefill 侧整占 node1 8 卡（PF tp4 + PA tp4），Decode 侧整占 node2 8 卡（DF tp4 + DA tp4）
- 跨节点 KV：mooncake RDMA over RoCE `mlx5_bond_0`；节点内 Attn↔FFN：CUDA IPC（`ipc_cpp`）
- 主脚本（宿主机运行）：`scripts/run_multi_node_bench.py`
- 烟雾测试：`scripts/smoke_pdaf_xnode.sh` / `scripts/smoke_pd_xnode.sh`
- 结果：`results/`，图表：`scripts/plot_results.py` → `charts/`

快速开始：

```bash
cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node
python3 scripts/run_multi_node_bench.py --tp 4 --scenario all --qps 1,2,3,4
```
