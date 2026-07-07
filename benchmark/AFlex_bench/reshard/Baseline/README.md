# Traditional / Optimized SGLang TP Scaling Baseline

本目录用于复现和量化 SGLang 张量并行（TP）扩展流程，作为 AFlex 组件级 reshard 的 baseline。参考材料放在 `paper/`：`DynamoLLM.pdf` 描述了把模型并行度作为实例级配置进行动态选择，`Breakdown.png` 和 `TP4.png` 展示了传统 TP4 迁移和权重搬迁开销。

## 两种 Baseline

### 1. Disk-load whole-instance baseline

最朴素的做法是启动一个新的 TP 实例，并让它从磁盘重新读取 safetensors：

1. 旧 `TP2` 实例继续服务。
2. 新 `TP4` 实例启动，重新初始化 Python、NCCL、模型权重、KV cache、HTTP server。
3. 新实例 `/health` ready 后，router 或上层调度器停止给旧实例派新请求。
4. 等待 drain window，切流到新 `TP4`。
5. 旧 `TP2` drain 完后退出。

这个流程简单，但新实例的模型加载和初始化时间较长。

### 2. IPC/NVLink shadow baseline

`TP4.png` 对应的优化思路不是从磁盘重载，而是复用旧 GPU 上已经 resident 的权重：

1. 旧 `TP2` 实例继续服务。
2. 旧实例通过 `/admin/export_weights_ipc` 导出 CUDA IPC handles。这个操作只写 metadata，真实权重仍在旧 GPU 显存里。
3. 新 `TP4` shadow 实例异步启动；它使用 `SGLANG_RESHARD_IPC_DIR` 进入 IPC fast path，dummy loader 跳过磁盘 IO。
4. 新 TP rank 打开旧 GPU 的 IPC tensor，按目标 TP 重新切片，并通过 NVLink/P2P 拷贝到目标 GPU。
5. 新实例 ready 后才进入短暂的 visible drain/cutover window。
6. 切流完成后旧实例退出。

这一路径的核心目标是把重活放到旧服务在线期间完成，让用户可见的服务中断只剩最后的连接重置/切流窗口。

## 代码文件

- `scripts/test_traditional_tp_scaling.py`：disk-load whole-instance baseline。
- `scripts/test_optimized_tp_scaling_ipc.py`：IPC/NVLink shadow baseline。
- `charts/plot_traditional_tp_scaling.py`：从 JSONL 生成 breakdown、latency、downtime 对比图。
- `results/`：结果输出目录。
- `logs/`：SGLang server 日志目录。

实现上还修正了 `python/sglang/srt/reshard/weight_loader.py` 的 TP 切片方向：PyTorch linear weight 是 `[out_features, in_features]`，column-parallel 应切 dim 0，row-parallel 应切 dim 1；同时避免新 TP 多 rank 并发加载时第一个 rank 提前删除 IPC handle 文件。

## 运行命令

### Disk-load baseline

```bash
cd /workspace/sglang
python3 benchmark/AFlex_bench/reshard/Baseline/scripts/test_traditional_tp_scaling.py   --model-path /models/Qwen3-32B   --source-tp 2 --source-gpus 0,1   --target-tp 4 --target-gpus 2,3,4,5   --source-port 31000 --target-port 31010   --source-nccl-port 32100 --target-nccl-port 32110   --prompt-len 2048 --output-len 32   --num-requests 4 --concurrency 1   --output benchmark/AFlex_bench/reshard/Baseline/results/node1_tp2_to_tp4_real.jsonl
```

### IPC/NVLink optimized baseline

注意：`--visible-gpus` 必须同时包含 source 和 target GPU，保证 CUDA IPC handle 里的 device index 在 target 进程内仍可见；target 通过 `--base-gpu-id` 实际落到 `--target-gpus`。

```bash
cd /workspace/sglang/benchmark/AFlex_bench/reshard/Baseline/scripts
python3 test_optimized_tp_scaling_ipc.py   --model-path /models/Qwen3-32B   --source-tp 2 --source-gpus 0,1   --target-tp 4 --target-gpus 2,3,4,5   --visible-gpus 0,1,2,3,4,5   --source-port 31200 --target-port 31210   --source-nccl-port 32300 --target-nccl-port 32310   --prompt-len 2048 --output-len 32   --num-requests 4 --concurrency 1   --drain-s 1.0   --ipc-dir /tmp/sglang_reshard_baseline   --output /workspace/sglang/benchmark/AFlex_bench/reshard/Baseline/results/node1_tp2_to_tp4_ipc_real.jsonl
```

Dry-run：

```bash
python3 benchmark/AFlex_bench/reshard/Baseline/scripts/test_optimized_tp_scaling_ipc.py --dry-run
```

## 画图

```bash
python3 benchmark/AFlex_bench/reshard/Baseline/charts/plot_traditional_tp_scaling.py   --input benchmark/AFlex_bench/reshard/Baseline/results/node1_tp2_to_tp4_compare.jsonl   --out-dir benchmark/AFlex_bench/reshard/Baseline/charts
```

输出：

- `charts/traditional_tp_breakdown.png` / `.pdf`：阶段耗时 breakdown。
- `charts/traditional_tp_latency.png` / `.pdf`：扩展前后 TTFT 对比。
- `charts/traditional_tp_downtime.png` / `.pdf`：异步准备时间 vs 用户可见中断时间。

## Node1 Docker 真实测试结果

环境：node1 `operator_test` Docker，8×A800，`/models/Qwen3-32B`，`prompt_len=2048`，`output_len=32`，`num_requests=4`，`concurrency=1`。

| 方案 | Source | Target | Async prepare | Visible downtime | Total | Target weight load |
|---|---|---|---:|---:|---:|---:|
| Disk-load | TP2 GPU0,1 | TP4 GPU2-5 | 31.05s | 5.00s | 83.18s | safetensors 约 9-10s，target ready 31.05s |
| IPC/NVLink shadow | TP2 GPU0,1 | TP4 GPU2-5 | 29.08s | 1.00s | 80.16s | IPC/NVLink 2.72-2.77s |

IPC/NVLink 真实日志关键行：

```text
Reshard IPC fast path: using DUMMY loader (skip disk IO)
FastTPLoader: loaded 515 params, skipped 0 (old_tp=2 → new_tp=4 rank=0)
FastTPLoader: loaded 515 params, skipped 0 (old_tp=2 → new_tp=4 rank=1)
FastTPLoader: loaded 515 params, skipped 0 (old_tp=2 → new_tp=4 rank=2)
FastTPLoader: loaded 515 params, skipped 0 (old_tp=2 → new_tp=4 rank=3)
Load weight end. elapsed=2.72-2.77 s
```

请求指标：

| 方案 | 阶段 | 成功率 | Avg TTFT | P50 TTFT | Avg E2E |
|---|---|---:|---:|---:|---:|
| Disk-load | TP2 before | 4/4 | 142.83ms | 113.04ms | 1280.26ms |
| Disk-load | TP4 after | 4/4 | 171.71ms | 168.16ms | 1357.45ms |
| IPC/NVLink | TP2 before | 4/4 | 144.02ms | 114.52ms | 1293.60ms |
| IPC/NVLink | TP4 after | 4/4 | 168.37ms | 164.70ms | 1319.09ms |

## 结论

IPC/NVLink shadow path 已经把真实权重加载从 safetensors 路径替换成 GPU resident weight transfer：TP4 四个 rank 都在约 2.7s 内完成权重拷贝和重切片。当前总的 target ready 仍约 29s，说明剩余瓶颈主要在 Python 进程启动、distributed/NCCL 初始化、KV cache allocation 和 HTTP ready，而不是模型权重 IO。

对“尽可能缩短服务中断时间”来说，关键指标是 `visible_downtime_s`。本次 optimized baseline 用 1s drain/cutover，服务可见中断从 disk-load baseline 的 5.00s 降到 1.00s；异步准备阶段虽然仍约 29s，但旧 TP2 在这期间持续服务。


## QPS=1 连续流量验证

为了验证服务中断是否真的只发生在 cutover window，新增了连续流量测试脚本：

```bash
cd /workspace/sglang/benchmark/AFlex_bench/reshard/Baseline/scripts
python3 test_tp_scaling_qps_timeline.py   --model-path /models/Qwen3-32B   --source-tp 2 --source-gpus 0,1   --target-tp 4 --target-gpus 2,3,4,5   --visible-gpus 0,1,2,3,4,5   --source-port 31400 --target-port 31410   --source-nccl-port 32500 --target-nccl-port 32510   --qps 1 --workload-duration-s 100   --scale-at-s 55 --cutover-s 2   --prompt-len 2048 --output-len 32   --ipc-dir /tmp/sglang_reshard_qps2   --output /workspace/sglang/benchmark/AFlex_bench/reshard/Baseline/results/node1_tp2_to_tp4_ipc_qps1_timeline.json
```

这次测试从 source ready 后开始按 `QPS=1` 发送 100 个请求，每个请求间隔 1 秒；在实验第 `55.02s` 开始导出 IPC handles 和启动 target shadow，target 在 `84.12s` ready，然后模拟 `2.00s` 的连接重置/cutover window。

真实结果：

| 指标 | 结果 |
|---|---:|
| 总请求数 | 100 |
| 成功请求 | 98 |
| 失败请求 | 2 |
| Source 成功 | 50 |
| Target 成功 | 48 |
| Cutover 失败 | 2 |
| IPC export | 0.058s |
| Target async prepare | 29.04s |
| Visible cutover gap | 2.002s |
| 平均 TTFT（成功请求） | 118.15ms |
| P50 TTFT（成功请求） | 111.75ms |
| 平均 E2E（成功请求） | 1372.94ms |

失败请求均落在 cutover window 内：

| Request ID | Scheduled | Start | Route | Error |
|---:|---:|---:|---|---|
| 50 | 85.055s | 85.090s | cutover | `route_unavailable_during_cutover` |
| 51 | 86.055s | 86.057s | cutover | `route_unavailable_during_cutover` |

时间轴图：

- `charts/qps1_tp_scaling_timeline.png`
- `charts/qps1_tp_scaling_timeline.pdf`

结论：在 QPS=1 的真实连续流量下，异步 IPC/NVLink 准备期间 source 继续服务；可见服务不可用只出现在显式 cutover window，持续 `2.002s`，对应 2 个按秒到达的请求失败。若真实 router 在 cutover 时排队而不是拒绝请求，这两个请求可以表现为额外排队延迟而不是失败。

## In-Place TP Reshard Microbench：TP1→TP2→TP4→TP8

> 重要更正：本节结果不是完整 SGLang 推理服务链路。它只验证了“旧 rank 原地保留、新 rank 追加、GPU tensor 重切片、NCCL/P2P 传输、QPS 时间轴记录”这个控制流和权重传输 microbench。
>
> 它没有加载真实 Qwen3-32B 模型，没有分配真实 SGLang KV cache，也没有让请求经过 `/generate` 的 prefill/decode。因此 GPU 显存占用不会接近真实服务的 80%+，图里的请求 TTFT/E2E 是合成 latency，不应被当作真实推理指标。

新增 `scripts/test_inplace_tp_scaling_qps_timeline.py` 用于验证 in-place TP 扩容的传输控制流。8 个 joinable ranks 在同一个 NCCL world 中预启动，当前 TP 之外的 ranks 处于 standby。扩容时旧 rank 保留原 rank id，只把当前 shard 再切分后发送给追加的新 rank：

- `TP1→TP2`：rank0 保留，rank1 新加入。
- `TP2→TP4`：rank0/rank1 保留，rank2/rank3 新加入。
- `TP4→TP8`：rank0-rank3 保留，rank4-rank7 新加入。

这个 microbench 满足“旧 TP workers 原地保留，只新增缺失 ranks”的 rank 映射约束，因此 `TP4→TP8` 峰值只需要 8 张 GPU，而不是旧 TP4 + 新 TP8 的 12 张 GPU。但它还不能证明完整 SGLang 服务已经支持动态 in-place forward。

在 node1 的 `operator_test` Docker 中运行的命令：

```bash
cd /workspace/sglang/benchmark/AFlex_bench/reshard/Baseline/scripts
python3 test_inplace_tp_scaling_qps_timeline.py \
  --profile bench \
  --qps 1 \
  --workload-duration-s 80 \
  --scale-at-s 10 35 60 \
  --output ../results/inplace_tp1_to_tp8_qps_bench.json

cd /workspace/sglang/benchmark/AFlex_bench/reshard/Baseline/charts
python3 plot_inplace_tp_timeline.py \
  --input ../results/inplace_tp1_to_tp8_qps_bench.json \
  --out-dir . \
  --stem inplace_tp1_to_tp8_qps_timeline
```

Microbench 结果（`profile=bench`，110 个 Qwen-like synthetic TP tensors，真实 CUDA/NCCL P2P 传输）：

| 阶段 | 传输量 | Transfer | 带宽 | synthetic pause window |
|---|---:|---:|---:|---:|
| TP1→TP2 | 3.15 GB | 0.940s | 3.35 GB/s | 0.941s |
| TP2→TP4 | 1.58 GB | 0.130s | 12.16 GB/s | 0.130s |
| TP4→TP8 | 0.79 GB | 0.241s | 3.26 GB/s | 0.242s |

Synthetic QPS=1 请求结果：

| 指标 | 结果 |
|---|---:|
| 总请求数 | 80 |
| 成功请求 | 79 |
| 失败请求 | 1 |
| 失败 Request ID | 60 |
| 总 synthetic pause | 1.313s |
| 合成平均 TTFT | 104.68ms |
| 合成 P50 TTFT | 110.00ms |
| 合成平均 E2E | 259.62ms |

输出文件：

- `results/inplace_tp1_to_tp8_qps_bench.json`
- `charts/inplace_tp1_to_tp8_qps_timeline.png`
- `charts/inplace_tp1_to_tp8_qps_timeline.pdf`

同时保留了一个快速 smoke 结果：

- `results/inplace_tp1_to_tp8_qps_smoke.json`
- `charts/inplace_tp1_to_tp8_qps_smoke_timeline.png`
- `charts/inplace_tp1_to_tp8_qps_smoke_timeline.pdf`

### 完整推理链路还缺什么

真正要证明 in-place reshard 可用于 SGLang serving，需要补齐这些条件：

1. 真实 `sglang.launch_server` 加载 `/models/Qwen3-32B`，显存应主要由权重和 KV cache 占用。
2. 请求必须通过 HTTP `/generate`，完整经过 tokenizer、scheduler、prefill、decode、sampling 和 detokenizer。
3. 每个 TP 阶段要采样 GPU memory/utilization，记录权重 resident、KV cache allocation、reshard 前后 KV cache flush/rebuild。
4. 新 rank 不能只是 receiver 或 synthetic worker，必须成为正式 `Scheduler`/`TpModelWorker` forward path 的参与者。
5. TP group、attention metadata、model runner metadata、KV pool layout 都必须在所有 ranks 同步切换。



### 真实推理显存/KV cache 验证工具

为避免把 synthetic microbench 误认为真实 serving，新增了 `scripts/test_real_inference_memory_timeline.py`。这个脚本不会自己做 reshard；它用于验证任意真实 SGLang serving 路径：持续向真实 HTTP `/generate` 发请求，同时采样 `nvidia-smi` 的 per-GPU memory/utilization。

示例：

```bash
cd /workspace/sglang/benchmark/AFlex_bench/reshard/Baseline/scripts
python3 test_real_inference_memory_timeline.py \
  --url http://127.0.0.1:31400 \
  --qps 1 \
  --duration-s 120 \
  --prompt-len 2048 \
  --output-len 32 \
  --output ../results/real_inference_memory_timeline.json
```

这个结果才适合检查：

- 每张 GPU 是否长期有真实模型权重 resident。
- 显存是否接近真实 serving 的预期占用。
- KV cache allocation 是否存在，以及请求增长/flush 后是否变化。
- QPS 请求是否真实经过 prefill/decode，而不是 synthetic sleep。

注意：核心里的旧 `/reshard_tp?action=live_reshard_tp` 原型之前会只启动 receiver 并把 rank0 权重切成新 TP shard，但不会让新增 rank 进入正式 forward path，后续请求会因为 TP collective/shape 不一致崩溃。因此已默认禁用该危险路径，除非显式设置 `SGLANG_RESHARD_ENABLE_UNSAFE_LIVE_TP=1` 进行调试。

---

## In-Place TP Reshard 真实服务实现进度（handoff，2026-07-06）

本节记录 **真实 SGLang 服务** 上 in-place TP 链式扩容（`TP1→TP2→TP4→TP8`）的实现与验证进度，供换设备后续接。代码在宿主机 `/mnt/workspace/lt/sglang`，容器 `operator_test` 内挂载为 `/workspace/sglang`。

### 目标

- 单节点 dense Qwen3-32B，`pp=1, dp=1, ep=1`，禁用 CUDA graph。
- 启动时 `--tp 1 --inplace-reshard-max-tp 8`：8 个 scheduler 进程同 world，仅 rank0 加载权重；rank1–7 standby。
- 运行时通过 HTTP `POST /reshard_tp` 链式扩容，尽量缩短服务中断。
- 用真实并发 workload 验证：`scripts/bench_inplace_reshard_real_workload.py`。

### 环境与启动

| 项 | 值 |
|---|---|
| 容器 | `operator_test`（8×A800） |
| 模型 | `/models/Qwen3-32B` |
| 代码 | `/workspace/sglang`（宿主机 `/mnt/workspace/lt/sglang`） |
| 端口 | HTTP `31700`，NCCL `32800` |

**启动服务（容器内）：**

```bash
docker exec operator_test bash -c '
cd /workspace/sglang && nohup python3 -m sglang.launch_server \
  --model-path /models/Qwen3-32B \
  --tp 1 \
  --inplace-reshard-max-tp 8 \
  --host 127.0.0.1 --port 31700 --nccl-port 32800 \
  --mem-fraction-static 0.85 \
  --disable-cuda-graph \
  --disable-piecewise-cuda-graph \
  --skip-server-warmup \
  --base-gpu-id 0 \
  --attention-backend triton \
  > /tmp/sglang_bench.log 2>&1 &
'

# 等待 ready（约 20–30s）
docker exec operator_test curl -sf http://127.0.0.1:31700/health

# smoke generate
docker exec operator_test curl -sf -X POST http://127.0.0.1:31700/generate \
  -H "Content-Type: application/json" \
  -d "{\"text\":\"hi\",\"sampling_params\":{\"max_new_tokens\":3,\"temperature\":0}}"
```

**预期启动显存：** GPU0 ~60–70GB（rank0 权重+KV），GPU1–7 ~520MiB（standby）。

**手动单步 reshard：**

```bash
curl -X POST http://127.0.0.1:31700/reshard_tp \
  -H "Content-Type: application/json" -d '{"new_tp_size": 2}'
curl http://127.0.0.1:31700/inplace_reshard_status   # phase=done, active_tp=2
```

**真实 workload 压测（推荐 D 轮配置）：**

```bash
docker exec operator_test bash -lc '
cd /workspace/sglang/benchmark/AFlex_bench/reshard/Baseline
python3 scripts/generate_inplace_reshard_workload.py \
  --seed-workload /tmp/macro_code_qps80.jsonl \
  --output results/inplace_workload_tp_scaled.jsonl
PLAN=$(cat results/inplace_workload_tp_scaled.reshard_plan.json)
SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 PYTHONUNBUFFERED=1 python3 \
  scripts/bench_inplace_reshard_real_workload.py \
  --workload results/inplace_workload_tp_scaled.jsonl \
  --base http://127.0.0.1:31700 \
  --reshard-plan "$PLAN" \
  --reshard-sequential --max-workers 128 \
  --timeout 180 --reshard-status-timeout 400 \
  --out results/inplace_chain_tp1_8_workload_tp_scaled.json
'
```

workload 源：`/tmp/macro_code_qps80.jsonl`。**必须等服务 `/health=200` 后再跑**。
低 QPS 参考结果见 `results/inplace_chain_tp1_8_workload.json`（80/80，pause 9.0/4.5/5.8s）。

### 架构要点

```
启动: --tp 1 --inplace-reshard-max-tp 8
  rank0: 加载 Qwen3-32B + KV，正常 event_loop
  rank1-7: event_loop_inplace_standby，低显存等待 activate

扩容 TP1→TP2:
  rank0: world broadcast → sync parallel groups → IPC 传权重 → narrow → 重建 KV
  rank1: activate（dummy load + 收权重 + 建 KV）→ 加入 active loop
  rank2-7: skip（仅参与 world barrier + rebuild_inplace_reshard_parallel_state）

扩容 TP2→TP4:
  rank0: joinable 编排
  rank1: expand_inplace_reshard_active_rank（旧 active follower 扩 shard）
  rank2,3: activate（standby 加入）
  rank4-7: skip
```

关键环境变量（代码内自动设置）：

- `SGLANG_INPLACE_RESHARD_MAX_TP=8`
- `SGLANG_INPLACE_RESHARD_ACTIVE_TP=<当前 active tp>`

### 已修改的核心文件

| 文件 | 作用 |
|---|---|
| `python/sglang/srt/server_args.py` | 新增 `--inplace-reshard-max-tp` |
| `python/sglang/srt/managers/scheduler.py` | rank0 集中 batching；standby loop；follower world broadcast；sentinel drain |
| `python/sglang/srt/managers/scheduler_update_weights_mixin.py` | `_live_reshard_tp` drain/execute 流程 |
| `python/sglang/srt/model_executor/model_runner.py` | activate/expand/joinable 路径；IPC 权重传输；**rank0 冷加载**；分阶段 barrier |
| `python/sglang/srt/model_executor/model_runner_kv_cache_mixin.py` | `rebuild_memory_pool_after_inplace_reshard()`；analytic KV 画像；probe MIN；headroom 检查 |
| `python/sglang/srt/distributed/parallel_state.py` | **新增** `rebuild_inplace_reshard_parallel_state()` |
| `python/sglang/srt/layers/reshard_weights.py` | `reshard_subshard` / `reshard_shard_for_rank` |
| `benchmark/.../scripts/bench_inplace_reshard_real_workload.py` | 真实 workload + 多步 reshard plan 压测 |

### 已完成的修复

1. **KV 预算重分配**：扩容后 `rebuild_memory_pool_after_inplace_reshard()` 按新 TP 重算 token 上限（否则 KV 严重 under-allocate）。
2. **异步 drain + 集中调度**：reshard 期间 rank0 独占 batching，follower 镜像 batch plan。
3. **World broadcast 协调**：active follower（rank1）必须参与 world broadcast，否则 rank0 `broadcast_pyobj` 死锁。
4. **`rebuild_inplace_reshard_parallel_state`**：全体 8 rank 以相同顺序调用 `new_group()`（active 组 `[0..new_tp-1]` + standby 单例组）；standby 对 ATTN/MOE 子组 alias 到 `_TP`（`inplace_standby` 条件），避免 `_ATTN_CP` 初始化 `cpu_group is None`。
5. **多跳权重映射**：`old_tp>1` 时 rank1 等旧 active rank 应从 source_old 接收 subshard，不能一律 `sub_idx=0` narrow 自身（否则 TP2→TP4 逻辑 shard 错位）。
6. **IPC 传输分阶段**：先 joining rank 收权重 → barrier → 再 old active receiver 收权重 → barrier → 最后 keeper narrow；避免 exporter 被提前覆写。
7. **显存峰值缓解**：old receiver 收权重前 `_drop_inplace_reshard_weight_storage()`；joining/old receiver 收完后 `_release_inplace_reshard_ipc_imports()`（`ipc_collect` + `empty_cache`）。

### 验证结果（截至 2026-07-06，node1 `operator_test`）

以下按**测试类型**分列，避免把不同负载下的 pause 混为一谈。`pause_s` = 客户端从 `POST /reshard_tp` 返回到
`/inplace_reshard_status` 报 `phase=done` 的间隔；`transfer_ms` 来自服务端 `timings`。

#### A. 空闲 smoke（`scripts/test_inplace_chain.py`）

无并发 workload，每步 reshard 后各发 2 条 `/generate`（`max_new_tokens=8`）。

| 步骤 | 结果 | 说明 |
|---|---|---|
| TP1→2→4→8 | **3/3 轮通过** | 每步 `phase=done` 且后续 `generate` HTTP 200 |

#### B. 低 QPS 并发 workload（`results/inplace_chain_tp1_8_workload.json`）

| 项 | 值 |
|---|---|
| Workload | `/tmp/macro_code_qps80.jsonl`，80 请求，到达跨度 ~82s（≈1 QPS） |
| Reshard plan | `@30s→TP2`, `@60s→TP4`, `@75s→TP8`（未加 `--reshard-sequential`） |
| HTTP | **80/80** |
| TP1→2 | trigger 30.01s → done 39.02s，**pause 9.01s**（transfer 4056ms，drain 0.03ms） |
| TP2→4 | trigger 60.00s → done 64.52s，**pause 4.52s**（transfer 3644ms，drain 0.03ms） |
| TP4→8 | trigger 75.08s → done 80.89s，**pause 5.81s**（transfer 4857ms，drain 0.12ms） |
| 结论 | 三步均 `generation=1/2/3, phase=done`；reshard 窗口内 13 条请求 TTFT 中位 **4.05s**（HTTP 仍 200，未按 SLO 统计） |

#### C. 高 QPS 并发 workload（`results/inplace_chain_tp1_8_workload_qps3x.json`）

| 项 | 原 C 轮 | 修复后 rerun（`qps3x_rerun.json`） |
|---|---|---|
| Workload | `/tmp/macro_code_qps240x2.jsonl`，160 请求（3× 加速） | 同上 |
| Reshard plan | `@10s→TP2`, `@32s→TP4`, `@48s→TP8` | 同上 |
| Strict mem check | **默认 ON**（未设 `=0`） | **默认 ON**（验证修复） |
| HTTP | **149/160**（11× ReadTimeout） | **160/160** |
| TP1→2 | pause **18.45s**，`done` | pause **18.50s**，`done` |
| TP2→4 | pause **17.88s**，`done` | pause **17.38s**，`done` |
| TP4→8 | **`phase=draining`，链未走完** | pause **6.91s**，`done` |
| 伴随现象 | `req_to_token_pool memory leak` → scheduler 崩溃 | **无 leak**；`/health` 200 |

**根因（已修复）**：reshard 后 `model_runner` 重建 KV / req pool，但 `RadixCache` 仍持有旧
`req_to_token_pool` / `token_to_kv_pool_allocator` 指针；请求完成时向 stale allocator 归还 slot，
free-list 漂移 → idle `self_check` 在 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=1` 下
`ValueError: req_to_token_pool memory leak detected`。

**修复**：
1. `_reshard_rebuild_tree_cache()`：重绑 scheduler pool 引用 + `tree_cache.reset()`（rank0 execute 与 follower expand 路径）。
2. `_should_skip_memory_check()`：在 `draining` / `executing` / `execute_pending` / `_engine_paused` 期间跳过 idle 内存检查，避免过渡期误报。

> D 轮仍建议设 `SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0` 作为压测保险；修复后 C 轮在 strict=ON 下已可完整走完链。

#### D. 分段递增 QPS workload — 当前主结果（`results/inplace_chain_tp1_8_workload_tp_scaled.json`）

| 项 | 值 |
|---|---|
| Workload | `results/inplace_workload_tp_scaled.jsonl`，**770** 请求，到达跨度 **170.8s** |
| 分段 QPS | TP1 阶段 1.5 / TP2 阶段 3.0 / TP4 阶段 5.5 / TP8 阶段 8.0 |
| Reshard plan | `@45.52s→TP2`, `@85.71s→TP4`, `@125.82s→TP8` |
| 运行参数 | `--reshard-sequential`，`SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0`，`--max-workers 128` |
| HTTP | **770/770** |
| SLO（TTFT≤2000ms 且 TPOT≤100ms） | **500/770 通过**，270 违例 |
| TP1→2 | trigger 45.53s → done 51.05s，**pause 5.52s**（transfer 4180ms，drain 0.03ms） |
| TP2→4 | trigger 85.75s → done 102.76s，**pause 17.02s**（transfer 2980ms，drain 0.03ms） |
| TP4→8 | trigger 125.84s → done 146.29s，**pause 20.44s**（transfer 4367ms，drain 0.04ms） |
| 三步 reshard | 均 `generation=1/2/3, phase=done, accepted=202` |

**SLO 宕机窗口**（连续违例请求，合并重叠区间后 `slo_downtime_total_s=54.563`）：

| 窗口 (issue 时间) | 时长 | 违例请求数 | 与服务事件的对应关系 |
|---|---|---|---|
| 0.0 – 1.6s | 1.6s | 1 | 首条请求：`input_len=4096`，TTFT 1.35s，TPOT 112ms |
| 46.2 – 55.2s | 9.0s | 19 | 覆盖 TP1→2 reshard（45.5–51.0s）及排队中的后续请求 |
| 99.0 – 108.0s | 9.1s | 22+ | 覆盖 TP2→4 reshard 后段（85.7–102.8s） |
| 141.1 – 176.0s | 34.9s | 51+ | 覆盖 TP4→8 reshard（125.8–146.3s）及 **TP8 阶段 8 QPS 恢复长尾** |

**分 TP 延迟中位数（发单时刻的 `tp_at_issue`）**：

| TP | 请求数 | SLO 通过 | TTFT med | TPOT med |
|---|---|---|---|---|
| 1 | 86 | 70/86 | 124 ms | 45 ms |
| 2 | 198 | 170/198 | 73 ms | 53 ms |
| 4 | 291 | 229/291 | 75 ms | 0 ms* |
| 8 | 195 | **31/195** | 237 ms | **864 ms** |

\* TP4 阶段短输出请求多，`output_len` 小导致 TPOT 分母接近 1，中位数偏低；**TP8 阶段 164/195 条违例 TPOT SLO** 是 270 条总违例的主因。

**pause 随负载变化（同一 TP2→4 步骤）**：

| 测试 | 并发强度 | TP2→4 pause |
|---|---|---|
| B（低 QPS） | ~1 QPS，80 req | **4.52s** |
| C（3× 加速） | ~3 QPS，160 req | **17.88s** |
| D（分段，到 TP4 时 ~5.5 QPS） | 770 req 分段 | **17.02s** |

idle smoke 下单步 TP2→4 曾测得 **~5.1s**（`KV re-profile` MIN-probe **32558 tokens/rank**）；与 B/C/D 的差异主要来自 **drain 在途请求**，不是 transfer 本身（transfer 稳定在 ~3–4.9s）。

#### E. GPU 显存 parity 与 rank0 冷加载（2026-07-06）

**目标：** in-place 链式扩容 `TP1→2→4→8` 完成后，各 GPU 显存占用与直接冷启动 `sglang serve --tp N` 一致（各卡均衡 ~85%，KV token 预算对齐）。

**冷启动参考（node1 `operator_test`，Qwen3-32B，`mem_fraction_static=0.85`）：**

| TP | 典型 tokens/rank | 各卡显存占用 |
|---|---:|---|
| TP2 | ~295,434 | ~86% |
| TP4 | ~738k | ~85% |
| TP8 | ~1,926,595 | ~85.7%（~70GB/卡） |

**根因结论（实测，非猜测）：**

| 现象 | 原因 |
|---|---|
| standby 卡 post-xfer 有 **62–70GB** 可用 | **能正常申请** KV |
| rank0 post-xfer 仅 **25–46GB** 可用 | **无法释放** CUDA 分配器中的历史峰值（TP1 全量 KV + in-place narrow 碎片），不是 standby **无法申请** |
| 旧逻辑 TP8：GPU0 ~92%、其他卡 ~60% | rank0 碎片 + 全组被 MIN(token) 拖低 |
| `get_available_gpu_memory` 仅 ~11GB，但 probe 能通过 | PyTorch pool 内仍有大量 reserved-but-unallocated；需用 `_inplace_reshard_effective_avail_gb()` |

典型日志（TP2→4）：

```text
rank 0 post-xfer: weights=15.26GB avail=32.70GB   # rank0
rank 1 post-xfer: weights=15.26GB avail=61.24GB   # standby 正常
rank0 cold reload done: avail=25.14GB (analytic=63.06GB)  # 磁盘重载后仍差 ~38GB
rank 0 KV probe shrank 839460 -> 400319 tokens
```

**已实现机制（`SGLANG_INPLACE_RESHARD_RANK0_COLD_RELOAD=1`，默认开启）：**

| 机制 | 文件 | 作用 |
|---|---|---|
| rank0 **冷加载** | `model_runner.py` | transfer 后删除整个 `model` graph，从磁盘按新 TP `load_model()`，等价冷启动权重布局 |
| **`final_hop` 冷加载策略**（默认） | `model_runner.py` | 中间跳（TP1→2、TP2→4）只做 in-place narrow + finalize；**仅最终跳**（TP4→8）冷加载，避免每跳 30–70s 磁盘 IO |
| **子进程冷加载**（默认开启） | `reshard/rank0_subprocess_reload.py` | 在独立 CUDA 进程中磁盘加载 rank0 shard，经 IPC 导入父进程，缓解 allocator 碎片 |
| **跳过冗余 CPU staging** | `model_runner.py` | 计划冷加载或显存充足时，不再每跳把全量权重搬到 CPU 再搬回（曾导致 transfer 30–70s） |
| 跳过 rank0 **in-place narrow** | `model_runner.py` | 冷加载跳上 rank0 只负责 P2P 导出，不在本地 slice 留峰值 backing buffer |
| cold reload 在 **rebuild collective barrier 之后** | `model_runner_kv_cache_mixin.py` | 与 rank1 activate/expand 路径同步，避免 barrier 死锁 |
| transfer 末尾 **all_reduce 仍参与** | `model_runner.py` | rank0 跳过 finalize 时不能跳过 collective（曾导致 TP1→2 永久卡住） |
| 跳过 transfer 前 metadata 更新 | `model_runner.py` | 避免在完整 TP1 权重上错误减半 `num_heads`（曾导致 attention `view` 崩溃） |
| analytic KV 画像 + 二分 **probe** | `model_runner_kv_cache_mixin.py` | 冷启动公式定目标；probe MIN 防 OOM |
| rebuild 末尾刷新 **LogitsProcessor** | `model_runner_kv_cache_mixin.py` | 修复 TP4+ `logits all_gather` 死锁 |
| KV **headroom** 检查 | `model_runner_kv_cache_mixin.py` | 分配后全组 MIN 空闲 < 6GB 时缩 token 重试 |
| **pre-drain 限流** | `scheduler_update_weights_mixin.py` | `pre_drain_sec` / `SGLANG_INPLACE_RESHARD_PRE_DRAIN_SEC`：reshard 前阻塞新请求，缩短高 QPS drain |
| **后台 prep + 短 commit** | `inplace_reshard_background.py`, `model_runner.py`, `scheduler.py` | drain 期间 standby 预收权重/KV/attn；pause 窗口仅 comm 组切换 + rank0 narrow/KV |
| resume 后 rank0 **后台 compact** | `scheduler_update_weights_mixin.py` | 不阻塞 reshard 主流程，为下一步 hop 做准备 |

**环境变量（新增/常用）：**

| 变量 | 默认 | 含义 |
|---|---|---|
| `SGLANG_INPLACE_RESHARD_RANK0_COLD_RELOAD_POLICY` | `final_hop` | `always` / `final_hop` / `deficit` / `never` |
| `SGLANG_INPLACE_RESHARD_RANK0_SUBPROCESS_RELOAD` | `1` | 子进程磁盘加载 + IPC 导入 |
| `SGLANG_INPLACE_RESHARD_PRE_XFER_MARGIN_GB` | `8` | 跳过 CPU staging 的显存余量阈值 |
| `SGLANG_INPLACE_RESHARD_PRE_DRAIN_SEC` | `0` | 全局默认 pre-drain 秒数（可被 `pre_drain_sec` 覆盖） |
| `SGLANG_INPLACE_RESHARD_BACKGROUND_PREP` | `1` | 后台预准备权重/KV/attn；commit 阶段仅重建 comm 组 |

**验证命令（idle smoke）：**

```bash
docker exec operator_test bash -lc '
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /workspace/sglang && SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 nohup python3 -m sglang.launch_server \
  --model-path /models/Qwen3-32B --tp 1 --inplace-reshard-max-tp 8 \
  --host 127.0.0.1 --port 31700 --nccl-port 32800 \
  --mem-fraction-static 0.85 --disable-cuda-graph --disable-piecewise-cuda-graph \
  --skip-server-warmup --base-gpu-id 0 --attention-backend triton \
  > /tmp/sglang_parity.log 2>&1 &
'
# ready 后
docker exec operator_test bash -lc '
cd /workspace/sglang && python3 benchmark/AFlex_bench/reshard/Baseline/scripts/test_inplace_chain.py \
  --host 127.0.0.1 --port 31700
'
```

**最新 idle 链式结果（2026-07-07，headroom + transfer 优化后）：**

| 步骤 | tokens/rank | 冷启动参考 | rank0 measured_post | generate |
|---|---:|---:|---:|---|
| TP1→2 | **~191k** | ~295k | ~41GB | ✅ |
| TP2→4 | **~331k** | ~738k | ~43GB | ✅ |
| TP4→8 | **~990k** | ~1,927k | ~41–43GB | ✅ |

中间跳采用 `final_hop` 策略后 transfer 恢复 **~3–5s** 量级（不再每跳 CPU staging + 磁盘冷加载）。rank0 子进程 reload 在最终跳进一步改善显存布局；token 预算仍受 rank0 长期进程 allocator 限制，低于冷启动参考。

**历史 idle 结果（headroom 修复前，TP4→8 generate ❌）：** TP8 曾在 forward 时 rank0 ~99% 显存、logits `all_gather` 失败；经 KV headroom 预扣 + measured MIN 后 idle 链已全过。

**TP1→TP2 典型日志（idle / 低负载）：**

```text
live_reshard_tp TP1→TP2 done in ~3800ms
In-place reshard KV re-profile: 24335 -> 222638 tokens
Joining active scheduler loop after in-place reshard
```

### 历史问题（TP2→TP4 曾阻塞）— 已修复项

| 问题 | 现象 | 修复（代码位置） |
|---|---|---|
| Follower 与 rank0 同步顺序错误 | TP2→4 `broadcast_pyobj` 与 rank1 expand 死锁 | `model_runner.py`：先等 world broadcast，再 `sync_groups` |
| Scheduler plan/execute 竞态 | rank0 发 plan 与 execute 差一轮 loop，follower 在 world wait 时 rank0 下一轮 collective 死锁 | `scheduler.py`：plan 与 execute 同轮 `get_next_batch` |
| Multihop phase2 send/recv 死锁 | phase2a send 与 phase2b recv 间 barrier 导致 NCCL P2P 互等 | `model_runner.py`：合并 send/recv 为同一 phase |
| GPU1 OOM / 权重错位 | reshard 中或 TP4 后 `all_gather` OOM / shape 不一致 | IPC 分阶段传输、expand 前释放 KV、KV MIN-probe、`reshard_subshard` 多跳映射 |

### 已知限制（有实测依据，非笼统“待观察”）

1. ~~**高 QPS 下 TP4→8 可能完不成**~~：C 轮 leak 已修复（`qps3x_rerun.json` 160/160）；高负载仍需 `--reshard-sequential`。
2. **SLO 违例 ≠ HTTP 失败**：B 轮 80/80 HTTP 成功，但 reshard 窗口内 TTFT 可达 4–6s；D 轮 270 条 HTTP 200 但 SLO 失败。可用 `pre_drain_sec` 缩短 drain。
3. **TP8 阶段 TPOT 是主要 SLO 瓶颈**：D 轮 TP8 仅 31/195 通过 SLO，TPOT 中位 864ms；合并宕机窗口 34.9s **长于** TP4→8 的 pause 20.4s。
4. ~~**压测后服务可能需手动重启**~~：脚本默认 `inplace_reshard_server_ctl.py` 自动停服。
5. **GPU 显存 parity 仍低于冷启动**：rank0 长期进程 + measured MIN 使 TP8 token ~990k vs 冷启动 ~1.9M；子进程 reload 缓解但未完全对齐。

### 调试命令

压测脚本默认在结束后 **自动停止** 31700 端口的 sglang（`test_inplace_chain.py`、`bench_inplace_reshard_real_workload.py`）；保留服务请加 `--keep-server`。也可手动：

```bash
python3 benchmark/AFlex_bench/reshard/Baseline/scripts/inplace_reshard_server_ctl.py
```

```bash
# 服务状态
docker exec operator_test pgrep -af launch_server
docker exec operator_test curl -s http://127.0.0.1:31700/health
docker exec operator_test nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits

# 日志
docker exec operator_test tail -100 /tmp/sglang_bench.log
docker exec operator_test grep -E "reshard|OOM|Traceback|KV re-profile" /tmp/sglang_bench.log | tail -50

# reshard 状态
curl -s http://127.0.0.1:31700/inplace_reshard_status | python3 -m json.tool
```

### 下一步建议

1. ~~**复现 D 轮主结果**~~：已完成（`inplace_chain_tp1_8_workload_tp_scaled_rerun.json`，770/770）。
2. ~~**调查 C 轮 `req_to_token_pool memory leak`**~~：已修复并验证（`qps3x_rerun.json`，strict=ON，160/160，三步 `done`）。
3. **TP8 SLO**：D 轮 164/195 条 TP8 请求违例 TPOT；使用 workload plan 中的 `pre_drain_sec` 或 `SGLANG_INPLACE_RESHARD_PRE_DRAIN_SEC` 在 reshard 前限流。
4. ~~**GPU 显存 parity**~~：已实现子进程 reload + `final_hop`；待 idle 链重测验证 token 是否接近冷启动。
5. **与传统 baseline 对比**：同一 workload 下对比 in-place pause（D 轮 5.5/17.0/20.4s）与 disk-load / IPC shadow 的 downtime。

### 与上文 Microbench 的区别

| | `test_inplace_tp_scaling_qps_timeline.py` | 真实服务路径（D 轮） |
|---|---|---|
| 模型 | 合成 tensor | Qwen3-32B `/models/Qwen3-32B` |
| 请求 | synthetic sleep | HTTP `/generate` stream |
| KV cache | 无 | 真实分配；reshard 后 `24335→222638` tokens（TP1→2 典型值） |
| Scheduler | 无 | 完整 event loop + inplace standby |
| 链式 TP1→8 | microbench 79/80 合成请求成功 | **770/770 HTTP**；三步 reshard `phase=done`；**500/770 SLO** |

### 相关脚本索引

| 脚本 | 用途 |
|---|---|
| `scripts/inplace_reshard_server_ctl.py` | 测试后停止 31700 端口 sglang（压测脚本默认调用） |
| `scripts/test_inplace_chain.py` | **idle smoke** TP1→2→4→8 链式 reshard + generate |
| `scripts/bench_inplace_reshard_real_workload.py` | **真实服务** workload + 多步 reshard + SLO 宕机窗口分析 |
| `scripts/generate_inplace_reshard_workload.py` | 分段递增 QPS workload 生成 |
| `scripts/bench_inplace_reshard_serial.py` | 串行 pre/post reshard 对比 |
| `scripts/test_inplace_tp_scaling_qps_timeline.py` | 合成 microbench（非真实推理） |
| `scripts/test_real_inference_memory_timeline.py` | 真实服务显存时间线采样 |
| `scripts/test_traditional_tp_scaling.py` | Disk-load baseline |
| `scripts/test_optimized_tp_scaling_ipc.py` | IPC shadow baseline |
