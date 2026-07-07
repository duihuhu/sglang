# Version 16: In-Place TP Reshard + HiCache Extent + 跨节点可扩展性 Macro + 能耗模型集中化 + AFD Fused Pipeline

## 概述

本次版本是一次大规模综合提交，核心 runtime 改动约 **+6500 行**（benchmark 数据文件从 git 中剥离约
-53000 行）。主体工作线包括：

1. **In-Place TP Reshard + Graceful Reshard 重构**（`python/sglang/srt/reshard/`）：进程内 live TP 扩缩、
   CUDA IPC 权重继承、AFlex IPC Reconnect；替代 v15 的 `graceful_orchestrator.py`。
2. **HiCache Host KV Extent 在线扩容**（`memory_pool_host.py` + `model_runner_kv_cache_mixin.py`）：多 pinned
   extent 分配器、运行时 grow、fragmentation 可观测。
3. **跨节点 16 卡可扩展性 benchmark 体系**（`multi_node/node_scalibility{,_macro,_moe,_moe_macro}/`）：
   6 方案 × 5+ 数据集 micro/macro，含 Mixtral MoE 扩展与 conv/code Azure trace 优化实验。
4. **Reshard benchmark 目录重组**（`reshard/Baseline/` + `reshard/rebuild/`）：传统/IPC shadow/in-place 链式
   reshard 对比 + 冷启动 breakdown。
5. **能耗/DVFS 增强**：idle/bubble 能耗建模、BiScale MPC 策略、compositional decode DVFS。
6. **AFD IPC Fused Pipeline**（C++ `afd_fused_pipeline.cpp`）：单层 send+recv 融合，可选 comm stream 重叠。
7. **Benchmark 数据 gitignore 整理**：profile txt/json、results/ 等实验产物不再进版本库；数据集中至
   `energy_model/` 本地目录。

> 这是一次"攒了多批改动一起提交"的版本。下面按工作线分别说明；§10 给出完整文件清单与已知缺口。

---

# 一、In-Place TP Reshard + Graceful Reshard 重构（核心 runtime）

v15 的 `energy/graceful_orchestrator.py`（Tier1 drain-then-switch 跨进程编排）在本版本**删除**，功能拆入
`python/sglang/srt/reshard/` 模块，并新增**进程内 live TP 扩缩**路径。

## 1.1 新模块 `python/sglang/srt/reshard/`

| 文件 | 职责 |
|------|------|
| `orchestrator.py` | `GracefulReshardOrchestrator`：shadow rank、drain、IPC 继承、router 切换 |
| `weight_exporter.py` | CUDA IPC 权重 handle 导出 |
| `weight_loader.py` | IPC fast path 加载 + TP 切片 |
| `reshard_controller.py` | 进程内 TP 重建控制 |
| `inplace_reshard_background.py` | 后台 prepare/commit in-place reshard |
| `rank0_subprocess_reload.py` | rank0 子进程冷加载 + IPC 导入（缓解 caching allocator 峰值） |

## 1.2 进程内 In-Place TP Reshard（scheduler + model_runner）

- **`--inplace-reshard-max-tp N`**：预启 N 个 scheduler worker ranks；活跃 TP 仍为 `--tp-size`；standby rank
  跳过 forward，待 `activate_inplace_reshard` 集体切换。
- **状态机**：`pre_draining → draining → preparing → executing → done`；reshard 期间阻塞新请求、等待
  continuous batch drain。
- **`parallel_state.py`**：`SGLANG_INPLACE_RESHARD_MAX_TP` / `ACTIVE_TP` 环境变量；standby rank 独立 TP
  group；`rebuild_inplace_reshard_parallel_state()`。
- **`reshard_weights.py`**（新增）：QKV/MergedColumn 融合权重按 segment 切分（`column_fused`）；NCCL/P2P
  scatter；配合 `vocab_parallel_embedding.reshard_recompute_indices()` 避免 logits all_gather 死锁。
- **HTTP 入口**（`http_server.py`）：
  - `POST /admin/export_weights_ipc`
  - `POST /admin/ipc_reconnect`
  - `POST /reshard_tp`
  - `GET /inplace_reshard_status`

## 1.3 AFlex IPC Reconnect（单组件 PA 扩缩）

允许 PF 不重启、仅 reconnect IPC，实现 PA TP1→TP2→TP4 扩展（5 GPU reshard，非 6 GPU 全重启）。

| 步骤 | 耗时 |
|------|------|
| Export IPC handles | ~0.02s |
| Drain + Kill old PA | ~5s |
| PF IPC Reconnect | **0.33–0.44s** |
| New PA startup | 22–24s |

- `afd_ipc_cpp/communicator.py`：`cleanup()` / `reconnect()`；后台线程 handshake + GIL release。
- `scheduler_update_weights_mixin.py`：`_ipc_reconnect()` 经 ZMQ 管理通道执行。

详见 `benchmark/AFlex_bench/reshard/README.md`。

## 1.4 Reshard Benchmark 结果（`reshard/Baseline/`）

| 方案 | Visible downtime | Weight load |
|------|-----------------|-------------|
| Disk-load whole-instance (TP2→TP4) | **5s** | ~9–10s |
| IPC/NVLink shadow | **1s** | **2.7s** |
| In-place chain TP1→2→4→8（真实 workload D 轮） | 5.5/17.0/20.4s | 无磁盘重载 |

**冷启动 breakdown**（`reshard/rebuild/`，TP=8 A800）：Total **67.43s**，其中 CUDA graph capture **58.25s**
是主要瓶颈（非权重 IO）。

## 1.5 已知缺口

- `scheduler.py` 的 `_trigger_full_reload()` **仍引用已删的** `sglang.srt.energy.graceful_orchestrator`；
  新编排器在 `reshard/orchestrator.py`，尚无 `--config` CLI 入口。Tier1 graceful reload 路径需后续迁移。
- `test/srt/test_graceful_reload.py` 仍 import 旧模块，需更新。
- In-place reshard TP8 阶段 TPOT SLO 仍有瓶颈；rank0 显存 parity 低于冷启动。

---

# 二、HiCache Host KV Extent 在线扩容

## 2.1 `memory_pool_host.py`（+1178 行）

- 新 **`HostKVCacheExtentTable`**：多 pinned extent 分配器；`ACTIVE`/`DRAINING` 状态；全局 index 稳定。
- **`grow_extent_online()`**：运行时在线扩容 host KV，不丢已有 page。
- 传输记录含 **fragmentation 指标**：`page_runs`、`max_run_pages`、`avg_run_pages`。

## 2.2 `model_runner_kv_cache_mixin.py`（+520 行）

- In-place reshard 用 analytic/probe 估算可分配 token。
- `profile_max_num_token` 支持 `distributed`/`empty_cache` 控制。

## 2.3 Scheduler / 测试

- 测试 hook：`SGLANG_TEST_HICACHE_GROW_EXTENT_TOKENS` / `RESERVE_OLD_EXTENT`。
- `/server_info` 暴露 `hicache_extent_debug`。
- 新增 `test/srt/test_host_kv_cache_extent.py`、fragmentation bench/probe 脚本。
- 修改 `test/registered/hicache/test_hicache_storage_file_backend.py`：runtime extent grow 测试。
- 引入 LooGLE 长上下文数据集引用（`benchmark/hicache/LooGLE/`，需本地 clone，data/*.jsonl 不进 git）。

---

# 三、跨节点 16 卡可扩展性 Benchmark（multi_node/ 扩展）

在 v15 四架构双节点基准基础上，新增完整 **6 方案 × 多数据集** 可扩展性测试体系。

## 3.1 六种方案（论文映射）

| 内部名 | 论文名 | 说明 |
|--------|--------|------|
| `native_tp2_baseline` | SGLang | Native TP2 × DP，锁频 1410 MHz |
| `native_tp2_tier` | DynamoLLM | Native + unified DVFS |
| `pd_dp_baseline` | DistServe | PD DP 跨/节点内配对 |
| `pd_dp_tier` | BiScale | PD + BiScale MPC DVFS |
| `pdaf_baseline` | MegaScale | PDAF 4PA+4PF+4DA+4DF |
| `pdaf_tier` | AFlex | PDAF + AF DVFS + idle-lock |

## 3.2 子目录

| 目录 | 内容 |
|------|------|
| `node_scalibility/` | Micro：5 数据集 profile × 4/8/16 卡 × QPS sweep |
| `node_scalibility_macro/` | Macro：Azure trace（conv/code）+ AFlex 优化实验 + `more_trying/` 大规模 sweep |
| `node_scalibility_moe/` | Mixtral-8x7B micro 可扩展性 |
| `node_scalibility_moe_macro/` | Mixtral macro benchmark |

## 3.3 关键结果

**Native DVFS bug fix**（`notify_freq_override`，v16 前已合入）：Native+Tier 节能从 ~2% 提升到 **29–36%**
（8 卡 qa QPS=1）。

**16 卡 micro 重跑**（2026-06-28）：pdaf_tier (AFlex) 30/30 完成；summary_hphd 高 QPS 有 OOM FAIL。

**Compositional DVFS**（V1 vs V2 pipeline）：E/tok 平均差距 **~5.4%**，TPOT ~5ms。

**Conv macro 优化**（`node_scalibility_macro/test.md`）：

| 尝试 | 结论 |
|------|------|
| M=2 / dynamicM | 无收益或退化 |
| V1 compositional DVFS | 单 decode 下最佳 TPOT/能耗 |
| 2Decode（bootstrap port 修复后） | qps16: **596 tok/s**, TPOT 96.8ms |
| 3Decode | qps16: **637.8 tok/s** (+13.5%), TPOT 89.1ms |
| fused/gpu_only AF 通信 | <2% 或无收益 |
| online calibration | 校准后 E/tok +10–12%（conv decode-bound） |

**总体结论**：conv/宽松 SLO/decode-bound 下，引擎层微优化无效；**真正杠杆是 decode 并发拓扑**（2D/3D）。

## 3.4 基础设施更新

- 四节点集群环境审计（node1–4）、Docker 对齐（`multi_node/README.md`）。
- 新增 smoke 脚本：`smoke_pdaf_all_pairs.sh`、`smoke_pdaf_node23/34.sh`、`cleanup_all_nodes.sh` 等。

---

# 四、能耗模型集中化 + DVFS 增强

## 4.1 `benchmark/AFlex_bench/energy_model/`

将分散在 `03_sensitivity/slo_sweep/retrain/data/` 和 `test_motivation/hucc/paper/` 的 profile 数据
**集中管理**：

```
energy_model/
├── Qwen3-32B/data/{v1_layer_profile,v2_pipeline_profile}/
├── Qwen3-32B/models_v{1,2}/
├── Mixtral-8x7B/data/ + profile_tp4_*.py
└── idle_power_vs_freq.json
```

原 git 跟踪的 profile txt 已从索引/features 移除（`.gitignore` 规则），本地 data/ 目录自包含。

## 4.2 新增 `energy/idle_power.py`

- A800 实测空闲功耗表（210–1410 MHz）。
- `layer_bubble_energy_mj()`：AF 流水线 bubble 能耗。
- `scheduler_idle_energy_mj()`：batch 间静态功耗。

## 4.3 `af_dvfs_controller.py` 增强

- 决策分解：`energy_compute_mj` / `energy_bubble_mj` / `energy_idle_mj`。
- `include_idle_energy`、`idle_lock_enabled`、`decode_compositional` 模式。
- Compositional decode：V1 逐层模型 + 在线校准 `t_comm_us`；校准 ratio 上限 3→10。

## 4.4 `unified_dvfs_controller.py` — BiScale 策略

- 新策略 `policy=biscale`：prefill MPC（K=8 horizon）+ decode 升序 min-freq。
- `objective=energy|freq`；TP 感知 safety margin、TPOT 趋势、emergency brake。
- `DECODE_DVFS_DISABLED_TP=8`：高 TP 时跳过 decode 降频。

## 4.5 HUCC Profiling 扩展

- 新增 `bench_prefill_af_tp4.py`、`bench_decode_af_tp4.py`（TP4 profiling + NVML GPU ID 修复）。
- `bench_prefill_af.py`：`MIN_MEASURE_TIME_S` 0.5s → **3s**（TP 下更稳定）。

---

# 五、AFD IPC Fused Pipeline

## 5.1 C++ 实现

- `sgl-kernel/csrc/afd_ipc/afd_fused_pipeline.{h,cpp}`：`FusedPipeline.send_recv()` 单次调用完成
  send+recv；可选独立高优先级 `comm_stream`。
- `afd_ipc_pybind.cpp`：Python 绑定 `get_fused_pipeline()`；修正 send 时误清 recv metadata cache。

## 5.2 Python 集成

- `afd.py`：`AFD_FUSED_PIPELINE=1` 时用 fused 路径驱动逐层 ATTN/FFN IPC 环。
- 可选 `AFD_FUSED_COMM_STREAM=1`、`AFD_GPU_ONLY_IPC=1`。

## 5.3 Benchmark 结论（`fused_pipeline_bench/`）

Qwen3-32B TP4 PDAF decode：non-streaming **无收益**（110.8 vs 111.1 ms）；streaming 约 **-4%**
（噪声范围内）。与 macro conv 实验结论一致。

---

# 六、通信基准扩展（comm/）

| 脚本 | 用途 |
|------|------|
| `bench_mooncake_rdma_write.py` | Mooncake `transfer_sync_write`（匹配 PD KV 传输） |
| `bench_affinity_4nic.py` | 4-NIC GPUDirect RDMA GPU↔mlx5 亲和映射 |
| `plot_comm_comparison.py` | NVLink vs RDMA 综合对比图 |
| `run_comm_bench.sh` | 全套自动化编排 |
| `scripts/bench_pdaf_affinity.py` | PDAF 交错 vs 连续 GPU 布局 A/B |

**关键结果**（`comm/read.md`）：

| 指标 | NVLink | RDMA 单卡 | RDMA 4 卡聚合 |
|------|--------|-----------|---------------|
| 峰值带宽 | **177 GB/s** | 24.6 GB/s | **91.6 GB/s** |
| 小消息延迟 | ~37 μs | **1.84 μs** | — |

## 6.1 RDMA/UCX runtime 优化（`rdma_comm.py`）

- `UCX_MEMTYPE_CACHE=y`：UCX 识别 GPU buffer、走 cuda_copy。
- UCX event loop 线程初始化 CUDA context；`BufferPool.warmup()` 热路径零分配。

---

# 七、Benchmark 数据 gitignore 整理

## 7.1 `.gitignore` 重组

新增「Benchmark & experiment data artifacts」区块：

- `benchmark/**/data/**/*.{txt,json,csv,...}`、`benchmark/**/results/**`
- `benchmark/**/*.txt`、`benchmark/**/*.json`（保留 `benchmark/kernels/`）
- 根目录 `hicache_*.json`、`loogle_*.json`；`core.*` core dump
- 通用格式：`*.csv`、`*.tsv`、`*.jsonl`、`*.pkl`、`*.parquet`、`*.npy`、`*.bak`

## 7.2 删除

- `benchmark/AFlex_bench/.gitignore`（与根目录规则冲突）
- 26 个已跟踪 benchmark profile/trace 数据文件（`git rm --cached`，本地保留）

---

# 八、退役 / 迁移

## 8.1 `retesting/` 脚本退役

单节点 micro/macro 脚本（`run_micro_bench.py`、`run_macro_bench.py` 等 11 个文件）删除；
功能由 `node_scalibility/` + `node_scalibility_macro/` 承接。

## 8.2 `reshard/` 顶层文件重组

| 删除 | 替代 |
|------|------|
| `bench_reshard.py` | `Baseline/scripts/bench_inplace_reshard_real_workload.py` |
| `test_reshard.py` | `Baseline/scripts/test_inplace_chain.py` |
| `plot_reshard.py` | `Baseline/charts/` |
| `graceful_reload_impl.md` | `reshard/README.md` + `reshard_breakdown.md` |
| `rebuild.md` | `rebuild/` 目录 |

---

# 九、建议 Commit Message

```
In-place TP reshard + HiCache extent grow + node scalability macro + energy model consolidation.

- reshard/: GracefulReshardOrchestrator + in-place TP chain (--inplace-reshard-max-tp),
  CUDA IPC weight export/load, AFlex IPC reconnect; remove graceful_orchestrator.py.
- HiCache: HostKVCacheExtentTable, grow_extent_online(), fragmentation metrics + tests.
- multi_node/: 6-scheme micro/macro scalability (dense + Mixtral MoE), conv 3Decode +13.5%.
- energy: idle/bubble power model, BiScale MPC DVFS, compositional decode calibration.
- AFD: C++ fused send_recv pipeline (AFD_FUSED_PIPELINE=1); UCX GPU buffer warmup.
- reshard/Baseline + rebuild/: TP scaling baselines + cold-start breakdown (graph 58s).
- energy_model/: consolidate Qwen3-32B + Mixtral profile data locally; gitignore data artifacts.
```

---

# 十、备注与已知缺口

- `multi_node/logs/`、`benchmark/**/results/`、`benchmark/**/data/**` 等实验产物已被 `.gitignore`
  忽略，不进 commit。
- `node_scalibility/run_node_scalability.py` **缺失**（results 存在，主 runner 需恢复或从备份找回）。
- Tier1 `_trigger_full_reload()` → 已删 `graceful_orchestrator` 的引用需迁移至 `reshard/`。
- PDAF 部署 sweep（`more_trying/`）第 1/2 类布局搜索尚未实现。
- In-place reshard 为实验特性，多节点行为需单独验证。

---

# 十一、本次改动文件清单（摘要）

## 核心源码（M）

```
M python/sglang/srt/managers/scheduler.py              # in-place reshard 状态机 (+1832)
M python/sglang/srt/model_executor/model_runner.py   # prepare/commit/activate reshard (+2264)
M python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py
M python/sglang/srt/mem_cache/memory_pool_host.py    # HostKVCacheExtentTable (+1178)
M python/sglang/srt/managers/scheduler_update_weights_mixin.py  # ipc_reconnect (+553)
M python/sglang/srt/distributed/parallel_state.py   # inplace standby TP groups
M python/sglang/srt/entrypoints/http_server.py       # reshard/admin endpoints
M python/sglang/srt/energy/af_dvfs_controller.py      # idle/bubble/compositional DVFS
M python/sglang/srt/energy/unified_dvfs_controller.py # BiScale MPC
M python/sglang/srt/layers/afd.py                    # fused pipeline path
M python/sglang/srt/layers/afd_ipc_cpp/communicator.py  # reconnect/cleanup
M python/sglang/srt/layers/rdma_comm.py              # UCX GPU buffer + warmup
M python/sglang/srt/server_args.py                   # --inplace-reshard-max-tp, DVFS policy
M sgl-kernel/csrc/afd_ipc/afd_fused_pipeline.cpp     # fused send_recv
M sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp
M .gitignore                                          # benchmark data artifacts
```

## 新增（A）

```
A python/sglang/srt/reshard/                           # 7 files: orchestrator, weight export/load, ...
A python/sglang/srt/layers/reshard_weights.py
A python/sglang/srt/energy/idle_power.py
A sgl-kernel/csrc/afd_ipc/afd_fused_pipeline.{h,cpp}
A build_afd_ipc.py
A benchmark/AFlex_bench/multi_node/node_scalibility/
A benchmark/AFlex_bench/multi_node/node_scalibility_macro/
A benchmark/AFlex_bench/multi_node/node_scalibility_moe/
A benchmark/AFlex_bench/multi_node/node_scalibility_moe_macro/
A benchmark/AFlex_bench/reshard/Baseline/
A benchmark/AFlex_bench/reshard/rebuild/
A benchmark/AFlex_bench/energy_model/
A benchmark/AFlex_bench/fused_pipeline_bench/
A benchmark/AFlex_bench/comm/{bench_*,plot_*,scripts/}
A benchmark/hicache/LooGLE/                           # README + assets (data jsonl ignored)
A test/srt/test_host_kv_cache_extent.py
A test/srt/bench_*extent*.py, probe_hicache_*.py, run_hicache_loogle_ab.py
A benchmark/AFlex_bench/git_version/version16.md
```

## 删除（D）

```
D python/sglang/srt/energy/graceful_orchestrator.py   # → reshard/orchestrator.py
D benchmark/AFlex_bench/retesting/scripts/*            # 11 files, → node_scalibility/
D benchmark/AFlex_bench/reshard/{bench,test,plot}_reshard.py + *.md  # → Baseline/, rebuild/
D benchmark/AFlex_bench/.gitignore
D benchmark/**/data/*.txt, hucc/paper/*.txt, trace_processed/*.json  # git rm --cached, 本地保留
```
