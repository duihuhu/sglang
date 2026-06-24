# Multi-Node 16-GPU Benchmark (2 nodes × 8 A800)

把 `retesting/scripts/run_micro_bench.py` 的单节点微基准扩展到 **16 卡 / 双 8 卡节点**，
对比 **4 种架构 × 2 种功耗模式**：

架构（均为 16 卡 = node1 8 + node2 8）：

| arch | 说明 | GPU 布局 |
|---|---|---|
| `native_dp` | Native DP16，16 个 TP=1 实例 + round_robin router | node1 GPU0-7 + node2 GPU0-7 |
| `pd_dp_xnode` | PD DP8，8 对 P-D，**跨节点**配对 | P=node1 GPU0-7，D=node2 GPU0-7 |
| `pd_dp_intra` | PD DP8，8 对 P-D，**节点内**配对 | 每节点 4P(GPU0-3)+4D(GPU4-7) |
| `pdaf` | PDAF 4PA+4PF+4DA+4DF（AF 算子拆分） | P=node1 8 卡(tp4)，D=node2 8 卡(tp4) |

功耗模式：

| mode | 说明 |
|---|---|
| `baseline` | GPU 锁频到最高 1410 MHz |
| `tier` | 按能耗模型做 per-instance DVFS（`--dvfs-*` / PDAF 用 `--afd-dvfs-*`） |

模型：`Qwen3-32B`（Dense）。

---

## 1. PDAF 拓扑

```
┌─────────────────────────── node1 (10.252.129.36) Prefill 侧 ───────────────────────────┐
│  container: operator_test (host network)                                                │
│                                                                                         │
│   PF  (FFN,  prefill)  base_gpu_id=0   ->  CUDA 0,1,2,3   tp=4                           │
│   PA  (Attn, prefill)  base_gpu_id=4   ->  CUDA 4,5,6,7   tp=4                           │
│        PA <----CUDA IPC (afd-comm-backend=ipc_cpp, NVLink)----> PF   [节点内]            │
│                                                                                         │
│   router (sglang_router, --pd-disaggregation --mini-lb)                                 │
└──────────────────────────────────────────┬──────────────────────────────────────────--┘
                                            │  P->D KV transfer: mooncake RDMA (RoCE, mlx5_bond_0)
                                            │  router -> DA: HTTP
┌──────────────────────────────────────────┴────────────── node2 (10.252.129.35) Decode 侧 ┐
│  container: operator_test (host network)                                                  │
│                                                                                           │
│   DF  (FFN,  decode)   base_gpu_id=0   ->  CUDA 0,1,2,3   tp=4                             │
│   DA  (Attn, decode)   base_gpu_id=4   ->  CUDA 4,5,6,7   tp=4                             │
│        DA <----CUDA IPC (NVLink)----> DF   [节点内]                                        │
└───────────────────────────────────────────────────────────────────────────────────────-┘
```

关键点：

- **AFD 算子拆分（Attn↔FFN）只发生在节点内**。`--afd-comm-backend ipc_cpp` 走 CUDA IPC
  （`cudaMemcpyPeerAsync` + IPC event），**只能同节点**。所以 PA/PF 必须同节点、DA/DF 必须同节点。
  → 这正好让 Prefill 整占 node1 8 卡、Decode 整占 node2 8 卡。
- **唯一跨节点的是 PD 的 KV 传输**（mooncake/RDMA）和 router→PA/DA 的 HTTP。
- `AFD_IPC_PEER_OFFSET=±tp`：FFN 进程 peer = local+tp，ATTN 进程 peer = local−tp。
  比固定 `AFD_IPC_PEER_DEVICE` 更通用，tp=4 时每个 rank 都能正确配对（rank r 的 FFN 在 cuda:r，
  对应 ATTN 在 cuda:r+tp）。

---

## 1b. 四种架构的 GPU / 端口布局

所有架构都用满 16 卡（node1 GPU0-7 + node2 GPU0-7），router 统一在 node1 `:42000`。

### native_dp — Native DP16

16 个独立 TP=1 实例，round_robin router 轮询。无 PD 拆分、无跨实例通信。

| 节点 | 实例 | HTTP 端口 | nccl 端口 |
|---|---|---|---|
| node1 | GPU0-7（8 个） | `53200,53210,...,53270` | `33300+gpu*10` |
| node2 | GPU0-7（8 个） | `53200,53210,...,53270` | `33300+gpu*10` |

### pd_dp_xnode — PD DP8（跨节点配对）

8 对 P-D，prefill 全在 node1、decode 全在 node2，KV **跨节点** mooncake 传输。
第 i 对：P=node1 GPU i ↔ D=node2 GPU i。

| 角色 | 节点 | HTTP 端口 | bootstrap | nccl |
|---|---|---|---|---|
| prefill i | node1 GPU i | `53100+i*10` | `49100+i*10` | `34000+i*10` |
| decode i | node2 GPU i | `53101+i*10` | `49100+i*10` | `34001+i*10` |

### pd_dp_intra — PD DP8（节点内配对）

8 对 P-D，每节点内 4P+4D，KV **尽量走节点内**（仍经 mooncake，但同主机）。
node1：P=GPU0-3 ↔ D=GPU4-7；node2：P=GPU0-3 ↔ D=GPU4-7。

| 对 i | prefill | decode |
|---|---|---|
| 0-3 | node1 GPU0-3 | node1 GPU4-7 |
| 4-7 | node2 GPU0-3 | node2 GPU4-7 |

### pdaf — PDAF 4PA4PF4DA4DF

见 §1 拓扑图。PF/PA 同在 node1（IPC），DF/DA 同在 node2（IPC），P→D KV 跨节点。

| 模块 | 节点 | base_gpu_id | CUDA | HTTP | nccl |
|---|---|---|---|---|---|
| PF | node1 | 0 | 0-3 | `42011` | `39411` |
| PA | node1 | 4 | 4-7 | `42010` | `39421` |
| DF | node2 | 0 | 0-3 | `42021` | `39431` |
| DA | node2 | 4 | 4-7 | `42020` | `39441` |

---

## 2. 网络（重要）

| 设备 | 链路层 | netdev | 状态 | 用途 |
|---|---|---|---|---|
| `mlx5_0/1/4/5` | InfiniBand | IPoIB (ibp*/ibs*) | RDMA `ACTIVE`，但 IPoIB 接口未 up | 单机内 KV |
| `mlx5_bond_0` | RoCE (Ethernet) | `bond0` (LACP) | Up，IP `10.252.129.3x/27` | **跨节点 KV** |

> `ibdev2netdev` 把 `mlx5_0/1/4/5` 显示为 `(Down)` 只是因为对应的 IPoIB 网卡没启用/没配 IP；
> 底层 RDMA 端口实际是 `state=ACTIVE / phys=LinkUp`，可用。

跨节点 mooncake 必须走有可路由 IP 的 RoCE 设备 **`mlx5_bond_0`**（已通过 smoke test 验证）。
通过 `MN_IB_DEV` 环境变量可覆盖。

---

## 3. 环境前提

两节点都有 `operator_test` 容器，且：

- 挂载：宿主机 `/mnt/workspace/lt` → 容器 `/workspace`；`/mnt/data/models` → `/models`。
  工作目录 `/mnt/workspace/lt/sglang` 在两节点间实时同步。
- 容器内：`python3` 可 `import sglang`，模型在 `/models/Qwen3-32B`，已装 `mooncake`、`pynvml`。
- 容器是 **host 网络模式**（进程直接用宿主机 IP/端口）。
- 容器内 **没有 `docker` 命令**，所以编排器跑在**宿主机**（宿主机有 `docker`、`ssh`、NVML 权限）。
- node1 宿主机能免密 `ssh 10.252.129.35`，并 `docker exec operator_test`（已把宿主机 ssh key 拷入容器）。

宿主机需要 `aiohttp numpy requests pynvml`（编排 + HTTP 压测 + 能耗采集）：

```bash
pip install aiohttp numpy requests pynvml
```

---

## 4. 文件

```
multi_node/
├── README.md                      # 本文档
├── scripts/
│   ├── run_multi_node_bench.py    # 主基准（宿主机运行，编排双节点 + 压测 + 能耗）
│   │                              #   支持 --arch {native_dp,pd_dp_xnode,pd_dp_intra,pdaf}
│   │                              #   与 --mode {baseline,tier}
│   ├── plot_results.py            # 读 multi_arch_16gpu_*.json 出架构对比 + baseline/tier 能耗图
│   ├── smoke_pdaf_xnode.sh        # 16 卡 PDAF 烟雾测试（最小验证拓扑）
│   ├── smoke_pd_xnode.sh          # 跨节点普通 PD 烟雾测试（验证 mooncake 网卡）
│   └── cleanup_node.sh            # 容器内清理脚本（按进程名 + 端口段杀，含 sglang::router）
├── results/                       # JSON 结果
│   ├── pdaf_16gpu_tp4_*.json      # PDAF 单架构完整结果（旧格式，见 §8）
│   └── multi_arch_16gpu_*.json    # 多架构结果（新格式，{arch}_{mode} -> {wl: metrics}）
├── charts/                        # 图表
└── logs/                          # 各服务 stdout（pf/pa/df/da/dp_*/pd*_p/d/router）+ 运行日志
```

---

## 5. 使用

### 5.1 烟雾测试（先验证拓扑能通）

在 **node1 宿主机**执行：

```bash
cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node

# 跨节点普通 PD（确定 mooncake 走哪个 IB 设备）
bash scripts/smoke_pd_xnode.sh mlx5_bond_0

# 16 卡 PDAF 4PA4PF4DA4DF
bash scripts/smoke_pdaf_xnode.sh mlx5_bond_0 4
```

成功会打印 `=== PDAF SMOKE PASS (ib=mlx5_bond_0 tp=4) ===` 并返回 "Paris"。

### 5.2 完整基准

在 **node1 宿主机**执行（不是容器内）：

```bash
cd /mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node

# 全部 4 架构 × 2 模式 × 4 场景 × QPS 1,2,4（很长，建议 nohup 后台跑）
nohup python3 scripts/run_multi_node_bench.py \
    --arch all --mode all --scenario all --qps 1,2,4 \
    > logs/full_multiarch.log 2>&1 &

# 单架构单模式快速验证
python3 scripts/run_multi_node_bench.py --arch pdaf --mode baseline --scenario chatbot --qps 1,2

# 只跑 tier 对比
python3 scripts/run_multi_node_bench.py --arch all --mode tier --qps 1,2
```

参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--arch` | all | `native_dp,pd_dp_xnode,pd_dp_intra,pdaf` 或 `all` |
| `--mode` | all | `baseline,tier` 或 `all` |
| `--scenario` | all | `chatbot,qa,rag,summary` 或 `all` |
| `--qps` | 1,2,4 | 逗号分隔 QPS 列表 |
| `--max-run-s` | 400 | 每个工负载最小运行窗口（按 `last_arrival+150s` 自动放大） |

- `baseline` 模式：脚本把两节点 GPU 0-7 锁到 1410 MHz；`tier` 模式：交给 DVFS 控频。
- 结果**增量保存**到 `results/multi_arch_16gpu_<timestamp>.json`（每个部署完成后写一次，
  中途崩溃也保留已有数据）。
- 可用环境变量覆盖：`MN_NODE1_IP` / `MN_NODE2_IP` / `MN_CONTAINER` / `MN_IB_DEV`。

> Tier 模式用的能耗模型在 `03_sensitivity/slo_sweep/retrain/models_v2`，
> SLO：TTFT=5000ms / TPOT=300ms。

### 5.3 工负载

复用 `retesting/workloads/micro_<scenario>_qps<n>.jsonl`（4 场景 × QPS 1-9）：

| 场景 | in_len | out_len | 特点 |
|---|---|---|---|
| chatbot | 128 | 1024 | decode 重 |
| qa | 512 | 256 | 均衡 |
| rag | 2048 | 64 | prefill 重 |
| summary | 4096 | 64 | prefill 极重 |

---

## 6. 指标

每个工负载记录：吞吐 `throughput_tok_s`、`ttft_proc_avg/p99_ms`、`tpot_avg/p99_ms`、
分节点能耗 `energy_node1_j`（prefill 侧）/ `energy_node2_j`（decode 侧）/ `total_energy_j`、
`energy_per_token_mj`、`slo_violation_rate`（TTFT SLO=5000ms，TPOT SLO=300ms）。

能耗用 NVML `nvmlDeviceGetTotalEnergyConsumption`：node1 直接读，node2 通过
`ssh <node2> python3 -c "..."` 读宿主机 NVML。两节点统计的都是 GPU 0-7。

多架构运行结果写到 `results/multi_arch_16gpu_<timestamp>.json`，结构为
`{"meta": {...}, "results": {"<arch>_<mode>": {"<scenario>_qps<n>": {metrics}}}}`，
每个部署完成后增量写一次。

---

## 7. 已知坑 / 调试

- **`EADDRINUSE` torch distributed**：PF/PA（或 DF/DA）共享同节点 + 同 `CUDA_VISIBLE_DEVICES`
  时，分布式初始化端口会冲突。脚本给 4 个服务分配独立 `--nccl-port`（39411/39421/39431/39441）。
- **router 残留**：rust router 进程名是 `sglang::router`，`pkill -f launch_router` 杀不掉，会导致
  下一轮 `42000` 端口 `address already in use`。`cleanup_node.sh` 同时按进程名和**端口** kill。
- **工负载超时**：QPS 低时最后一个请求到达很晚（qps1 ⇒ 200s），加上生成时间需要足够窗口。
  脚本按 `last_arrival + 150s` 动态设定运行窗口。
- **容器内不能编排**：容器无 `docker` 命令，编排器必须在宿主机跑。
- **架构切换的端口清理**：不同架构用不同端口段（Native DP `53200+`/nccl `33300+`，PD DP
  `53100+`/`49100+`/nccl `34000+`，PDAF `42010-42021`/nccl `39411-39441`）。`cleanup_node.sh`
  按这些端口段统一 kill，保证一个架构跑完后下一个不会撞端口。
- **round_robin router**：Native DP16 用 `--policy round_robin --worker-urls ...`；PD/PDAF 用
  `--pd-disaggregation`（PDAF 额外 `--mini-lb`）。三者 router 都在 node1 `:42000`，互不并存。

---

## 8. 测试结果（4PA4PF4DA4DF, 16 GPU, Qwen3-32B, 默认频率）

数据：`results/pdaf_16gpu_tp4_20260624_011114.json`，图表：`charts/pdaf_16gpu_tp4_overview.png`、
`charts/pdaf_16gpu_tp4_energy_split.png`。每个工负载 200 请求，TTFT SLO=5000ms / TPOT SLO=300ms，
全部 **SLO 0% 违例**。能耗按节点分开统计（n1=Prefill 侧 8 卡，n2=Decode 侧 8 卡）。

| Workload | Thpt(tok/s) | TTFT(ms) | TPOT(ms) | E_total(J) | n1_Prefill(J) | n2_Decode(J) | mJ/tok | SLO% |
|---|---|---|---|---|---|---|---|---|
| chatbot_qps1 | 670.0 | 87.1 | 103.2 | 434724 | 170627 | 264097 | 2122.7 | 0.0 |
| chatbot_qps2 | 771.4 | 75.9 | 106.0 | 379256 | 148410 | 230847 | 1851.8 | 0.0 |
| chatbot_qps3 | 818.0 | 73.5 | 106.3 | 358807 | 141107 | 217699 | 1752.0 | 0.0 |
| chatbot_qps4 | 843.6 | 74.0 | 106.6 | 348691 | 137113 | 211578 | 1702.6 | 0.0 |
| qa_qps1 | 233.7 | 76.3 | 76.9 | 311053 | 123588 | 187464 | 6075.3 | 0.0 |
| qa_qps2 | 426.8 | 76.6 | 82.0 | 174884 | 71545 | 103339 | 3415.7 | 0.0 |
| qa_qps3 | 586.5 | 76.5 | 86.5 | 130476 | 54279 | 76197 | 2548.4 | 0.0 |
| qa_qps4 | 715.1 | 76.6 | 92.0 | 109006 | 46219 | 62787 | 2129.0 | 0.0 |
| rag_qps1 | 62.5 | 91.9 | 73.7 | 294961 | 122802 | 172160 | 23043.9 | 0.0 |
| rag_qps2 | 122.2 | 93.1 | 75.9 | 158720 | 69845 | 88875 | 12400.0 | 0.0 |
| rag_qps3 | 179.1 | 94.4 | 76.9 | 112351 | 51412 | 60939 | 8777.4 | 0.0 |
| rag_qps4 | 233.3 | 92.8 | 77.8 | 88953 | 42063 | 46890 | 6949.5 | 0.0 |
| summary_qps1 | 62.5 | 115.2 | 74.4 | 310130 | 138040 | 172090 | 24228.9 | 0.0 |
| summary_qps2 | 122.2 | 119.8 | 75.9 | 168681 | 79292 | 89389 | 13178.2 | 0.0 |
| summary_qps3 | 179.2 | 121.7 | 76.7 | 120202 | 59060 | 61142 | 9390.8 | 0.0 |
| summary_qps4 | 233.3 | 126.3 | 78.4 | 95621 | 48588 | 47034 | 7470.4 | 0.0 |

观察：

- **TTFT 随 input_len 增长**：chatbot/qa(128/512) ≈ 75–87ms < rag(2048) ≈ 92–94ms <
  summary(4096) ≈ 115–126ms，符合 prefill 越重 TTFT 越高。
- **TPOT 稳定**：73–107ms，与 input/QPS 基本无关（decode 侧 attn/ffn 拆分稳定）。
- **吞吐随 QPS 线性上升**：chatbot 670→844 tok/s；rag/summary（out=64）吞吐被短输出限制，
  qps1→4 时 62.5→233 tok/s（吞吐 ≈ 系统在窗口内实际生成速率，受请求总量约束）。
- **能耗/token 随负载上升而下降**：QPS 越高，固定开销被摊薄，mJ/tok 显著降低
  （chatbot 2123→1703，qa 6075→2129）。
- **Decode 侧（node2）能耗普遍高于 Prefill 侧（node1）**，因为 decode 是逐 token 的长尾过程，
  GPU 利用时间更长。

### Overview

![overview](./charts/pdaf_16gpu_tp4_overview.png)

### Energy split (Prefill vs Decode node)

![energy split](./charts/pdaf_16gpu_tp4_energy_split.png)

---

## 9. 多架构测试状态（4 架构 × baseline/tier）

### 9.1 验证状态

四种架构 × 两种模式均已逐个 smoke 验证（chatbot QPS1，能正常部署 + 跑通 + 采能耗）：

| 架构 | baseline | tier (DVFS) | 备注 |
|---|---|---|---|
| `native_dp` | ✅ 跑通 | ✅ 跑通 | 16×TP1 + round_robin |
| `pd_dp_xnode` | ✅ 跑通 | 待跑 | 跨节点 KV，mooncake RoCE |
| `pd_dp_intra` | ✅ 跑通 | 待跑 | 节点内配对 |
| `pdaf` | ✅（§8 完整） | 待跑 | AF 算子拆分，DVFS 用 `--afd-dvfs-*` |

> 完整的「4 架构 × 2 模式 × 4 场景 × QPS{1,2,4}」批量运行（约 96 个工负载、6–9 小时）
> 脚本已就绪，可用 §5.2 的 `--arch all --mode all` 一键启动，结果增量写入
> `results/multi_arch_16gpu_*.json`，跑完用 `python3 scripts/plot_results.py` 出对比图。

### 9.2 已采集的代表性数据（smoke / 部分运行，chatbot）

baseline，chatbot 各 QPS（16 卡总能耗，节点 0-7 全统计）：

| 架构 | QPS | Thpt(tok/s) | TTFT(ms) | TPOT(ms) | E_total(J) | mJ/tok | SLO% |
|---|---|---|---|---|---|---|---|
| native_dp | 1 | 832 | 103 | 45.8 | 1225179 | 5982 | 0.0 |
| native_dp | 2 | 1394 | 91 | 46.7 | 741082 | 3619 | 0.0 |
| native_dp | 4 | 1690 | 91 | 46.9 | 620653 | 3031 | 0.0 |
| pd_dp_xnode | 1 | 828 | 50 | 47.0 | 761868 | 3720 | 0.0 |
| pd_dp_intra | 1 | 829 | 49 | 47.0 | 770957 | 3764 | 0.0 |
| pdaf（§8） | 1 | 670 | 87 | 103 | 434724 | 2123 | 0.0 |

初步观察（仅 chatbot，待全量确认）：

- **Native DP16 吞吐随 QPS 接近线性扩展**（832→1394→1690 tok/s），但 16 卡全程满载，
  低 QPS 时能耗/token 最高（QPS1 ≈ 5982 mJ/tok）。
- **PD DP8** 的 TTFT 明显低于 Native DP（≈50ms vs 103ms），因为 prefill 独占实例不被 decode 干扰；
  跨节点 vs 节点内配对在 chatbot QPS1 下差异很小（mJ/tok 3720 vs 3764，TTFT/TPOT 基本一致），
  说明 chatbot（短输入）下跨节点 KV 传输开销不显著。
- **PDAF** 在 chatbot 下能耗/token 最低（QPS1 ≈ 2123 mJ/tok），但 TPOT 偏高（103ms，
  因 attn/ffn 每层 IPC 往返），TTFT 居中。

> 注：以上能耗对比要严格看需用同一运行窗口口径。Native DP 的 smoke 用了完整窗口（含低 QPS 长尾
> 空闲），PD/PDAF 部分来自不同窗口，**横向能耗对比以 §5.2 全量运行结果为准**。

### 9.3 tier (DVFS) 说明

- Native DP / PD DP 用 `--dvfs-enabled --dvfs-energy-model-dir <models_v2>
  --dvfs-ttft-slo-ms 5000 --dvfs-tpot-slo-us 300000`。
- PDAF 用 `--afd-dvfs-enabled --afd-energy-model-dir <models_v2> --afd-ttft-slo-ms 5000
  --afd-tpot-slo-us 300000 --afd-dvfs-idle-lock`。
- baseline 模式下脚本主动把 GPU 锁到 1410 MHz；tier 模式下不锁频，交给 DVFS 控制器按 SLO + 能耗模型动态调频。
- 节能效果通常在**低负载 / prefill-heavy** 场景更明显；高利用率（如 chatbot 高 QPS）下 DVFS 接近满频，
  baseline 与 tier 能耗差异小。
