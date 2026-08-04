# Multi-Node 16-GPU Benchmark (4 nodes × 8 A800)

## 0. 四节点集群环境状态

> 审计时间：**2026-07-02**（node1 宿主机执行 `ssh` + `docker exec` 采集）

### 0.1 节点清单

| 节点 | IP | 主机名 | uptime | bond0 (RoCE) |
|---|---|---|---|---|
| node1 | `10.252.129.36` | d1n41a11g01 | ~8 天 | Up `10.252.129.36/27` |
| node2 | `10.252.129.35` | d1n41a29g01 | ~8 天 | Up `10.252.129.35/27` |
| node3 | `10.252.129.34` | d1n41a16g02 | ~6 天 | Up `10.252.129.34/27` |
| node4 | `10.252.129.33` | d1n41a16g03 | ~4 小时（近期重启） | Up `10.252.129.33/27` |

### 0.2 Docker / 容器

| 项 | node1 | node2 | node3 | node4 |
|---|---|---|---|---|
| Docker 服务 | active | active | active | active |
| Docker Root | `/mnt/data/docker` | 同左 | 同左 | 同左 |
| 容器名 | `operator_test` | 同左 | 同左 | 同左 |
| 镜像 | `cuda:12.2.0-devel-ubuntu22.04` | `operator_test:from-node1-export` | 同 node2 | 同 node2 |
| 容器 uptime | ~8 天 | ~14 小时 | ~14 小时 | ~25 分钟（重建后） |
| Privileged | ✅ | ✅ | ✅ | ✅ |
| `--ipc host` | ✅ | ✅ | ✅ | ✅ |
| `--network host` | ✅ | ✅ | ✅ | ✅ |
| `--gpus all` | ✅ 8 GPU | ✅ | ✅ | ✅ |
| 挂载 | `/mnt/workspace/lt`→`/workspace`<br>`/mnt/data/models`→`/models` | 同左 | 同左 | 同左 |
