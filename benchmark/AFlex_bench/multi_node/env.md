# Multi-Node 32-GPU Benchmark（4 nodes × 8 GPU）

## 0. 四节点新测试环境状态

> 审计时间：**2026-08-05**（node1 宿主机执行 `ssh` + `docker exec` 采集）
>
> 当前工程已从旧目录 `/mnt/workspace/lt/sglang` 迁移到 `/mnt/workspace/lt/moe-tier`；新容器统一为 `moe-energy`。旧容器 `operator_test` 已停止但未删除。

### 0.1 节点清单

| 节点 | IP | 主机名 | uptime | bond0（RoCE） |
|---|---|---|---|---|
| node1 | `10.252.129.36` | d1n41a11g01 | ~6 周 1 天 | Up `10.252.129.36/27` |
| node2 | `10.252.129.35` | d1n41a29g01 | ~6 周 1 天 | Up `10.252.129.35/27` |
| node3 | `10.252.129.34` | d1n41a16g02 | ~1 周 6 天 | Up `10.252.129.34/27` |
| node4 | `10.252.129.33` | d1n41a16g03 | ~4 周 6 天 | Up `10.252.129.33/27` |

### 0.2 Docker / 容器

| 项 | node1 | node2 | node3 | node4 |
|---|---|---|---|---|
| Docker 服务 | active | active | active | active |
| Docker Root | `/mnt/data/docker` | 同左 | 同左 | 同左 |
| 新容器名 | `moe-energy` | 同左 | 同左 | 同左 |
| 新容器状态 | running | running | running | running |
| 镜像 | `moe-energy:from-operator-test` | 同左 | 同左 | 同左 |
| 工作目录 | `/workspace/moe-tier` | 同左 | 同左 | 同左 |
| Privileged | ✅ | ✅ | ✅ | ✅ |
| `--ipc host` | ✅ | ✅ | ✅ | ✅ |
| `--network host` | ✅ | ✅ | ✅ | ✅ |
| `--gpus all` | ✅ 8 GPU | ✅ 8 GPU | ✅ 8 GPU | ✅ 8 GPU |
| memlock | unlimited | unlimited | unlimited | unlimited |
| 挂载 | `/mnt/workspace/lt`→`/workspace`<br>`/mnt/data/models`→`/models` | 同左 | 同左 | 同左 |
| 旧容器 `operator_test` | stopped（保留） | stopped（保留） | stopped（保留） | stopped（保留） |

### 0.3 工程与 Python 环境

| 项 | 四节点统一配置 |
|---|---|
| 宿主机工程目录 | `/mnt/workspace/lt/moe-tier` |
| 容器内工程目录 | `/workspace/moe-tier` |
| Python | `/usr/bin/python3` |
| PyTorch | `2.9.1+cu128` |
| PyTorch CUDA | `12.8` |
| `sglang` editable | `/workspace/moe-tier/python` |
| `sglang-router` editable | `/workspace/moe-tier/sgl-model-gateway/bindings/python` |
| `sglang` 实际导入 | `/workspace/moe-tier/python/sglang/__init__.py` |
| `sglang-router` 实际导入 | `/workspace/moe-tier/sgl-model-gateway/bindings/python/src/sglang_router/__init__.py` |
| Mooncake | 四节点均可导入 |
| GPU 可见性 | 四节点均可见 8 GPU |

### 0.4 AFlex 运行资产

四节点已同步并验证以下 AFlex 核心运行资产：

- Qwen3-32B macro workload 与 `plan_dense_e2e.json`；
- Qwen3-32B V1 latency/energy predictor；
- `afd_ipc_cpp.so`；
- `sglang_router_rs.abi3.so`；
- `libdvfs_ctrl.so`；
- GPU 锁频与解锁权限；
- Mooncake 与 PDAF 所需 Python 依赖。

当前核心路径：

- Predictor：`/workspace/moe-tier/benchmark/AFlex_bench/energy_model/Qwen3-32B/models_v1`
- DVFS 动态库：`/workspace/moe-tier/benchmark/test_motivation/hucc/dvfs/libdvfs_ctrl.so`
- AFlex 日志：`/workspace/moe-tier/benchmark/AFlex_bench/multi_node/logs`

### 0.5 模型可见性

| 模型 | node1 | node2 | node3 | node4 |
|---|---|---|---|---|
| `/models/Qwen3-32B/` | ✅ | ✅ | ✅ | ✅ |
| `/models/Mixtral-8x7B/` | ✅ | ✅ | ✅ | ❌ 当前缺失 |

> Qwen3-32B 的 PDAF smoke 与 AFlex Code QPS4 已分别在 node1+node2、node3+node4 上验证通过。运行 Mixtral 四节点实验前，需要先补齐 node4 的 `/models/Mixtral-8x7B/`。
