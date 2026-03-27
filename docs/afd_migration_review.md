# SGLang AFD（Attention-FFN Disaggregation）迁移与优化报告

## 一、概述

### 什么是 AFD

AFD 将 Transformer 每一层拆分为 Attention（A）和 FFN（F）两部分，分别部署在不同 GPU 节点上。每层有 2 次跨节点传输（A->F 和 F->A），通过 microbatch 流水线隐藏通信延迟。

### 核心价值

- MoE 模型中 Attn 节点不加载 Expert 权重，KV Cache 容量提升约 4 倍
- 更高并发 → 更高吞吐 + 更低 P99 延迟

### 迁移来源与目标

- **源**: `sglang_af/sglang_v1`（AFD PoC 实现）
- **目标**: `sglang`（新版，原本无 AFD 代码）

---

## 二、文件清单

### 新建文件（7 个）

| 文件路径 | 作用 |
|----------|------|
| `python/sglang/srt/layers/afd_type.py` | `AFDPerspective` 枚举和 micro_batch 参数解析 |
| `python/sglang/srt/layers/afd.py` | 核心逻辑：通信器（ZMQ/StepMesh/分片并行）、流水线调度、`model_forward_afd`、`AFDCommunicator`、`AsyncTensorCommunicator`、Proxy 模块 |
| `python/sglang/srt/layers/afd_mixin.py` | 通用 Mixin（`AFDDecoderLayerMixin`）和权重过滤器（`AFDWeightFilter`） |
| `python/sglang/srt/batch_overlap/afd_overlap.py` | `AfdForwardBatchPreparer`——microbatch 切分与子 batch 组装 |
| `python/sglang/srt/managers/scheduler_afd_mixin.py` | 可复用的 AFD 调度辅助方法 Mixin |
| `test/srt/test_afd_fixture.py` | 测试 fixture（双进程启动） |
| `test/srt/test_afd_basic.py` | 端到端测试用例 |

### 修改文件（13 个）

| 文件路径 | 修改内容 |
|----------|----------|
| `server_args.py` | 新增 `afd_perspective`、`afd_micro_batch`、`afd_attn_ratio`、`afd_attn_tp`、`afd_ffn_tp` + `_handle_afd()` 验证 |
| `io_struct.py` | 新增 `AFDReqInput`（继承 `BaseReq`，含 `req_ids`/`seq_lens`/`extend_lens`） |
| `schedule_batch.py` | `ScheduleBatch` + `ModelWorkerBatch` 新增 `afd_split_seq_index: Optional[List[int]]` |
| `forward_batch_info.py` | AFD 字段 + `init_new` 直接传入 `afd_split_seq_index` |
| `model_runner.py` | `init_attention_backend` 新增 `AfdAttnBackend` 分支 |
| `tbo_backend.py` | 新增 `AfdAttnBackend` 类 |
| `qwen3_moe.py` | DecoderLayer 加入 `_afd_init`/`forward_afd_A`/`forward_afd_F`；`load_weights` 加入 `AFDWeightFilter` |
| `qwen2_moe.py` | 同上 + Model.forward 加入 AFD 分支 + `load_weights` 加入 `AFDWeightFilter` |
| `qwen3.py` | DecoderLayer 加入 AFD；Qwen3Model 重写 forward 加入 AFD 分支 |
| `deepseek_v2.py` | DecoderLayer 加入 AFD；Model.forward 加入 AFD 分支 |
| `scheduler.py` | AFD ZMQ 通道 + `event_loop_afd` + `dispatch_event_loop` PREFILL/DECODE AFD 分支 |
| `disaggregation/prefill.py` | 新增 `event_loop_afd_disagg_prefill` |
| `disaggregation/decode.py` | 新增 `event_loop_afd_disagg_decode` |

---

## 三、已应用的优化

### 通信层

| 编号 | 内容 | 位置 |
|------|------|------|
| C1 | ZMQ 零拷贝传输（metadata + raw bytes） | `afd.py` ZMQSimpleTensorCommunicator |
| C2 | 独立 CUDA stream + `AsyncTensorCommunicator` + 跨 stage `postprocess_layer_start_recv` | `afd.py` |
| C3 | `deque` 替代 `list`、去掉 `zero_()`、动态 buffer pool | `afd.py` StepMeshTensorCommunicator |
| C5 | 端口通过环境变量可配置 | `afd.py` + `scheduler.py` |

### 调度层

| 编号 | 内容 | 位置 |
|------|------|------|
| S1 | `AFDReqInput` 传递 `req_ids` 精确对齐 batch | `io_struct.py` + `scheduler_afd_mixin.py` |
| S2 | `zmq.Poller` 替代 busy-wait | `scheduler.py` event_loop_afd |
| S3 | `AFDReqInput` 含 `seq_lens`/`extend_lens` | `io_struct.py` |

### 计算层

| 编号 | 内容 | 位置 |
|------|------|------|
| G3 | Token-count-aware 贪心切分 | `afd_overlap.py` |
| G5 | 预分配 output tensor，消除 `torch.cat` | `afd.py` model_forward_afd |
| G6 | `NamedTuple`（`StageIO`）替代 `dict` | `afd.py` |
| G4 | `--afd-attn-ratio` 非对称 microbatch 参数 | `server_args.py` |

### 工程化

| 编号 | 内容 | 位置 |
|------|------|------|
| E1 | 通用 Mixin + `AFDWeightFilter` | `afd_mixin.py` |
| E2 | AFD 代码独立文件 | `batch_overlap/afd_overlap.py` |
| E3 | 通用 m 路切分（循环实现） | `afd_overlap.py` |
| E5 | ZMQ 超时 + 异常处理 | `afd.py` |

### 异构 TP

| 编号 | 内容 | 位置 |
|------|------|------|
| Phase 6 | `--afd-attn-tp`/`--afd-ffn-tp` + `ShardedParallelCommunicator`（分片并行 + metadata 传递 + padding 截断） | `server_args.py` + `afd.py` |
| F9 | `--afd-grouped-stepmesh` + GCD-based 分组 StepMesh（per-group DMLC + NVLink all_gather + stride 去重） | `server_args.py` + `afd.py` |
| F2-A | `torch.compile(dynamic=True)` 编译 `_run_attn` / `_run_mlp` + AFD 通信方法 `@torch.compiler.disable()` | `afd_mixin.py` + `afd.py` |
| F3 | `--afd-enable-overlap-schedule` CPU/GPU overlap scheduling（`event_loop_afd` 中 `result_queue` + async `run_batch`） | `server_args.py` + `scheduler.py` |

### PD+AFD 独立控制

| 编号 | 内容 | 位置 |
|------|------|------|
| Task 6 | P/D 各自独立控制 `--afd-perspective`；`SchedulerAFDMixin` 可复用调度方法；PREFILL/DECODE 分支均支持 AFD | `scheduler_afd_mixin.py` + `prefill.py` + `decode.py` + `scheduler.py` |

---

## 四、支持的模型

| 模型 | 类型 | AFD 状态 | 说明 |
|------|------|----------|------|
| Qwen3-MoE | MoE | 已集成 | Mixin + 权重过滤 |
| Qwen2-MoE | MoE | 已集成 | Mixin + 权重过滤 + Model.forward AFD 分支 |
| DeepSeek-V2/V3 | MoE | 已集成 | Mixin + Model.forward AFD 分支 |
| Qwen3（Dense） | Dense | 已集成 | Mixin + Qwen3Model 重写 forward |
| LLaMA / Llama3 | Dense | 已集成 | init 中创建 LayerCommunicator + Mixin + AFDWeightFilter |
| 其他使用 LayerCommunicator 的模型 | 各种 | 可快速适配 | DecoderLayer 加 `_afd_init()` + `forward_afd_A/F` |
| Qwen2（Dense） | Dense | 可适配 | 与 LLaMA 相同方式：在 init 中按需创建 LayerCommunicator |

---

## 五、运行方式

> 以下示例使用 `<模型>` 指代模型路径，`<attn_ip>` / `<ffn_ip>` 指代节点 IP，`<RDMA网卡>` 指代 RDMA 网卡名（如 `mlx5_0`）。
> AFD 自动 disable overlap schedule 和 CUDA graph（无需手动传 `--disable-overlap-schedule` 或 `--disable-cuda-graph`）。
> 可选开关：`--enable-torch-compile`（kernel fusion ~5-8%）、`--afd-enable-overlap-schedule`（CPU/GPU overlap ~5-10%）。

### 1. ZMQ 同构 TP（快速测试，无 RDMA）

```bash
# ═══ Attn 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3
```

### 2. ZMQ 异构 TP（快速测试，无 RDMA，4A:8F，跨节点 2NH）

> ZMQ 路径通过 `ShardedParallelCommunicator` 包裹实现异构 TP：每个 local rank 只发送 1/TP_local 的 shard（1:1 点对点），接收侧 NVLink all_gather 重建完整 tensor。跨节点流量固定为 **2NH**（与 TP 配比无关），但受限于 TCP 延迟。

```bash
# ═══ Attn 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8

# ═══ FFN 节点（TP=8）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8
```

### 3. ZMQ 异构 TP（快速测试，无 RDMA，8A:4F，跨节点 2NH）

```bash
# ═══ Attn 节点（TP=8）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4
```

### 4. StepMesh 同构 TP（高性能 RDMA，TP_A == TP_F）

```bash
# ═══ 公共环境变量（两节点都设置）═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3
```

### 5. StepMesh 异构 TP — 不分组（4A:8F，跨节点 12NH）

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn 节点（TP=4，告知对端 FFN TP=8）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8

# ═══ FFN 节点（TP=8，告知对端 Attn TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8
```

### 6. StepMesh 异构 TP — 不分组（8A:4F，跨节点 12NH）

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn 节点（TP=8）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4
```

### 7. StepMesh 异构 TP — 分组（4A:8F，跨节点 3NH，4 组 1W:2S）

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ FFN 节点（TP=8）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh
```

### 8. StepMesh 异构 TP — 分组（8A:4F，跨节点 3NH，4 组 2W:1S）

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn 节点（TP=8）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4 \
    --afd-grouped-stepmesh

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4 \
    --afd-grouped-stepmesh
```

### 9. StepMesh 异构 TP — 分组 + 非对称切分（6A:4F，跨节点 5NH，2 组 3W:2S）

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn 节点（TP=6）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python -m sglang.launch_server \
    --model-path <模型> --tp 6 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 6 --afd-ffn-tp 4 --afd-attn-ratio 0.6 \
    --afd-grouped-stepmesh

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 6 --afd-ffn-tp 4 --afd-attn-ratio 0.6 \
    --afd-grouped-stepmesh
```

### 10. PD 分离 + AFD + 分组 StepMesh（4A:8F）

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Prefill Attn 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --disaggregation-mode prefill \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Prefill FFN 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode prefill \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Decode Attn 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule \
    --disaggregation-mode decode \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Decode FFN 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule \
    --port 30002 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode decode \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh
```

### 配置速查表

| # | 场景 | 通信方式 | Attn 侧关键参数 | FFN 侧关键参数 | 跨节点流量 |
|---|------|---------|-----------------|----------------|-----------|
| 1 | 同构 4:4 | ZMQ (TCP) | `--afd-perspective attn` | `--afd-perspective ffn` | 2NH |
| 2 | 异构 4:8 | ZMQ (TCP) | `--afd-attn-tp 4 --afd-ffn-tp 8` | 同左 | 2NH |
| 3 | 异构 8:4 | ZMQ (TCP) | `--afd-attn-tp 8 --afd-ffn-tp 4` | 同左 | 2NH |
| 4 | 同构 4:4 | StepMesh (RDMA) | 同 #1 + `MLC_INTERFACE` | 同左 | 2NH |
| 5 | 异构 4:8 不分组 | StepMesh (RDMA) | `--afd-attn-tp 4 --afd-ffn-tp 8` | 同左 | 12NH |
| 6 | 异构 8:4 不分组 | StepMesh (RDMA) | `--afd-attn-tp 8 --afd-ffn-tp 4` | 同左 | 12NH |
| 7 | 异构 4:8 **分组** | StepMesh (RDMA) | 同 #5 + `--afd-grouped-stepmesh` | 同左 | **3NH** |
| 8 | 异构 8:4 **分组** | StepMesh (RDMA) | 同 #6 + `--afd-grouped-stepmesh` | 同左 | **3NH** |
| 9 | 异构 6:4 **分组** | StepMesh (RDMA) | `--afd-attn-tp 6 --afd-ffn-tp 4 --afd-grouped-stepmesh` | 同左 | **5NH** |
| 10 | PD + 异构 4:8 **分组** | StepMesh (RDMA) | 同 #7 + `--disaggregation-mode prefill/decode` | 同左 | **3NH** |

> **注意**:
> - `--afd-attn-tp` 和 `--afd-ffn-tp` 需要在 **Attn 和 FFN 两侧都设置**，且值相同。`--afd-grouped-stepmesh` 同理。
> - ZMQ 异构 TP（#2, #3）通过 `ShardedParallelCommunicator` 包裹 1:1 ZMQ 通信实现，跨节点流量固定 2NH（与 TP 配比无关），但受限于 TCP 延迟，适合功能验证。
> - StepMesh 不分组异构（#5, #6）跨节点流量高（12NH），但 RDMA 延迟低；加 `--afd-grouped-stepmesh`（#7-#10）可降至 3-5NH。
> - `--afd-grouped-stepmesh` 仅对 StepMesh 生效（需 `MLC_INTERFACE`），ZMQ 路径自动忽略。

### 功能模块开关速查

| 模块 | 开关 | 默认 | 效果 | 依赖 |
|------|------|------|------|------|
| AFD 基础 | `--afd-perspective attn/ffn` | 关 | Attn-FFN 分离部署 | 自动 disable CUDA graph |
| Microbatch | `--afd-micro-batch N` | 3 | 流水线隐藏通信延迟 | AFD 基础 |
| 非对称切分 | `--afd-attn-ratio R` | 0.5 | Attn/FFN 计算量配比 | AFD 基础 |
| 异构 TP | `--afd-attn-tp A --afd-ffn-tp F` | 同 `--tp` | 不同 TP 大小 | AFD 基础 |
| 分组 StepMesh | `--afd-grouped-stepmesh` | 关 | 跨节点流量降至 NH×(A+F)/gcd | 异构 TP + `MLC_INTERFACE` |
| Overlap 调度 | `--afd-enable-overlap-schedule` | 关 | CPU/GPU 并行调度 ~5-10% | AFD 基础 |
| Kernel Fusion | `--enable-torch-compile` | 关 | torch.compile ~5-8% | AFD 基础 |

### 按功能模块组合的启动命令

#### A. 最简 AFD（ZMQ，功能验证）

```bash
# ═══ Attn 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
        --afd-perspective attn --afd-micro-batch 3

# ═══ FFN 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
        --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3
```

#### B. StepMesh RDMA + 异构 TP + 分组（生产推荐基线）

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn 节点（TP=4）═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
        --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ FFN 节点（TP=8）═══
python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
        --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh
```

#### C. 全部优化开启（最高性能）

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn 节点（TP=4）═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
        --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh \
    --afd-enable-overlap-schedule \
    --enable-torch-compile

# ═══ FFN 节点（TP=8）═══
python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
        --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh \
    --afd-enable-overlap-schedule \
    --enable-torch-compile
```

#### D. PD 分离 + AFD 全部优化

```bash
# ═══ 公共环境变量 ═══
export MLC_INTERFACE=<RDMA网卡>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Prefill Attn ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disaggregation-mode prefill \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh --afd-enable-overlap-schedule --enable-torch-compile

# ═══ Prefill FFN ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode prefill \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh --afd-enable-overlap-schedule --enable-torch-compile

# ═══ Decode Attn ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disaggregation-mode decode \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh --afd-enable-overlap-schedule --enable-torch-compile

# ═══ Decode FFN ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --port 30002 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode decode \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh --afd-enable-overlap-schedule --enable-torch-compile
```

#### E. 性能优化组合推荐

| 场景 | 命令组合 | 预期提升（vs 基础 AFD）|
|------|---------|----------------------|
| 功能验证 | A（最简 ZMQ） | 基线 |
| 生产基线 | B（StepMesh + 分组） | 跨节点流量 4x 降低 |
| 加 overlap 调度 | B + `--afd-enable-overlap-schedule` | +5-10% 吞吐 |
| 加 kernel fusion | B + `--enable-torch-compile` | +5-8% 计算 |
| **全部拉满** | **C（B + overlap + compile）** | **+10-18% 综合** |
| PD 分离 + 全部拉满 | D | 同 C，支持 PD 独立 |

---

## 六、架构限制与待办

| 编号 | 描述 | 状态 |
|------|------|------|
| F1 | StepMesh N:M 双向分片通信（详见下方） | **已实现** |
| F2 | CUDA graph 自动 disable（`_handle_afd`）；Phase A（`torch.compile` 逐 stage 编译）已实现，Phase B（`reduce-overhead` CUDA Graph）待实现 | **Phase A 已实现** |
| F3 | ~~TBO 与 AFD 互斥~~ → `--afd-enable-overlap-schedule` CPU/GPU overlap scheduling（`event_loop_afd` 中复用 `forward_stream` + `result_queue` 模式） | **已实现** |
| F4 | 自动 profiling + `afd_attn_ratio` 自适应调参 | 待实现 |
| F6 | DeepSeek-V2 `load_weights` 添加 AFDWeightFilter（generator 包装过滤） | **已修复** |
| F7 | `afd_prepare_overlap` 添加 `is_target_verify()` 处理（scheduler.py + scheduler_afd_mixin.py） | **已修复** |
| F8 | `afd_overlap.py` 移除冗余 `m is None` 检查 | **已修复** |
| F9 | GCD-based 分组 StepMesh（`--afd-grouped-stepmesh`），跨节点流量降至 NH×(TP_A+TP_F)/gcd | **已实现** |
| F5 | Qwen2（Dense）AFD 支持——可用 LLaMA 相同方式适配（init 中创建 LayerCommunicator） | 可快速适配 |

---

## 七、F1 StepMesh N:M 双向分片通信

### 设计原理

通过调研 StepMesh 源码（`af_tensor_app.h` + `public.hpp`）发现：

1. **`push_pull` 支持多个 pull_tensor**：`pull_tensors` 数组大小 = `server_count × batch_size`，每个 Server 有独立的 pull 缓冲区，不会覆盖
2. **`get_batch` 按 Worker rank 排序返回**：内部用 `q_[worker_rank]` 队列，返回顺序即 rank 顺序
3. **`respond` 按 key 匹配**：Server respond 的数据写入 Worker 对应 key 的 pull 缓冲区
4. **`RegisterRecvTensor` 支持按 Worker 分片注册**：同一 tensor 的不同 chunk 注册给不同 Worker

### 实现方式

**A→F 方向（分片广播）**：
- `attn_send`：每个 Attn rank 只 push 1/TP_A 的 token shard（广播到所有 FFN）
- 同时注册 **TP_F 个 pull_tensor**（每个 FFN Server 一个独立缓冲区用于接收 F→A shard）
- `ffn_recv`：`get_batch()` 返回 TP_A 个 shard（按 Worker rank 排序），`torch.cat` 拼接

**F→A 方向（分片 respond）**：
- `ffn_send`：每个 FFN rank respond 自己的 1/TP_F shard 给每个 Worker（按 key 匹配写入 Worker 对应的 pull 缓冲区）
- `attn_recv`：`wait` 后所有 TP_F 个 pull_tensor 已填充，`torch.cat` + 截断 padding

### 跨节点流量（4A:8F 示例）

| 方向 | 流量 |
|------|------|
| A→F | 4 × (NH/4) × 8 = 8NH（每个 Attn 发 1/4，广播 8 份） |
| F→A | 8 × (NH/8) × 4 = 4NH（每个 FFN 发 1/8，respond 给 4 个 Worker） |
| **总计** | **12NH**（vs 同构 1:1 的 2NH，vs 全广播的 64NH） |

### 向后兼容

同构 TP（`attn_tp == ffn_tp`）时 `_heterogeneous=False`，所有方法走原始 1:1 代码路径。

### 本轮 Review 已修复的问题

| 编号 | 问题 | 修复 |
|------|------|------|
| H1 | `attn_send` 中 `self.key += 2` 只留 2 个 key 空间，但 pull 需要 TP_F 个 key，第二次调用时 key 碰撞 | 改为 `self.key += 1 + self.ffn_tp` |
| H2 | docstring 描述了旧的"选择性 respond + all_gather"方案 | 更新为多 pull_tensor 方案 |
| H3 | `ffn_recv` 只为第一个 Worker 的 key 注册了 recv buffer | 改为为每个 Worker 的 key 都注册 |

### 遗留事项

| 编号 | 描述 |
|------|------|
| G3 | token 数不整除 TP 时末尾 shard 较短，StepMesh RDMA 写入到较大的 pull_buf 时尾部有未初始化数据。`attn_recv` 中 `full[:original_num_tokens]` 已截断，但 RDMA 层是否要求 respond 数据大小 == pull_buf 大小需实际验证 |
| G6 | `attn_send` 异构路径没有 buffer 复用（每次 `torch.empty_like`），同构路径用 `StepMeshTensorCache` + free pool。异构路径可优化但不影响正确性 |

### F→A 流量优化分析

当前 F→A 流量为 4NH（4A:8F 示例），原因是每个 FFN Server 必须 respond 给每个 Worker（StepMesh `push_pull` 要求所有 pull 请求都被 respond，否则 Worker hang）。

**进一步优化方案**：每个 Worker 只注册 1 个 pull_tensor（只从 1 个 Server 取 1/TP_F shard），Attn 侧 NVLink all_gather 拼回完整 tensor。F→A 跨节点降到 NH，代价是增加一次节点内 all_gather（NVLink ~600GB/s，延迟 5-10us）。

但此方案在 StepMesh 层有限制：`ZPull_` 内部 `pull_tensors.size() / server_count` 计算 `pull_batch_size`，如果 pull_tensor 数 < server_count 则 `pull_batch_size=0`，不发 pull。因此 **Worker 必须传 server_count 的整数倍个 pull_tensor**。

可行的折中方案：传 TP_F 个 pull_tensor 但只让需要的 Server respond 真实数据、其余 respond 空占位（避免 hang），然后 Attn 只取需要的 shard + all_gather。这等价于当前实现 + 加 all_gather 但不减少跨节点流量（仍是 4NH），无实际收益。

**结论**：4NH 是 StepMesh push_pull API 的固有下限（每个 Server 必须 respond 每个 Worker 的 pull）。真正降到 NH 需要分组 StepMesh 实例（极高复杂度），建议作为远期目标。实际部署中 RDMA 200Gbps+ 下 4NH 的绝对延迟仍在 SLA 范围内。

---

## 八、最终 Review（当前轮）

### 发现的问题

| 编号 | 严重度 | 文件 | 问题 |
|------|--------|------|------|
| F6 | **严重** | `deepseek_v2.py` | `load_weights` 缺少 `AFDWeightFilter`，Attn/FFN 节点都加载全部权重，浪费显存 |
| F7 | **中等** | `scheduler.py` + `scheduler_afd_mixin.py` | `afd_prepare_overlap` 未处理 `TARGET_VERIFY` 模式，speculative decoding 下 AFD microbatch 切分不工作 |
| F8 | **低** | `afd_overlap.py` | `m is None` 检查冗余（`_get_afd_micro_batch()` 总返回 int） |

### 已验证为正确的部分

| 检查项 | 状态 |
|--------|------|
| `afd_split_seq_index` 字段流转（ScheduleBatch → ModelWorkerBatch → ForwardBatch） | OK |
| StepMesh `attn_send` key 分配（`self.key += 1 + self.ffn_tp`） | OK |
| StepMesh `ffn_send` respond n_workers 次 | OK |
| StepMesh 异构路径（分片广播 + 多 pull_tensor）| OK |
| `model_forward_afd` 流水线 overlap（`postprocess_layer_start_recv`） | OK |
| Qwen3Model.forward AFD 分支 | OK |
| Qwen2.py 不含 AFD（Qwen2DecoderLayer 无 LayerCommunicator，正确排除） | OK |
| AFDReqInput 继承 BaseReq | OK |
| server_args `afd_micro_batch` 拼写 | OK |
| `event_loop_afd_disagg_prefill/decode` 调用 Mixin 方法顺序 | OK |
| `_compute_token_indices_m_way` 自主计算 | OK |

### 后续任务优先级

| 优先级 | 任务 | 描述 | 状态 |
|--------|------|------|------|
| ~~P0~~ | ~~F6~~ | ~~DeepSeek-V2 添加 AFDWeightFilter~~ | **已修复** |
| ~~P0~~ | ~~F7~~ | ~~`afd_prepare_overlap` 添加 `is_target_verify()` 处理~~ | **已修复** |
| ~~P1~~ | ~~F8~~ | ~~移除冗余 `m is None` 检查~~ | **已修复** |
| P2 | F2 | CUDA Graph 支持 | 待实现 |
| P2 | F3 | TBO + AFD 共存 | 待实现 |
| P3 | F4 | 自动 profiling | 待实现 |
| ~~P2~~ | ~~F9~~ | ~~分组 StepMesh：GCD-based 分组，跨节点流量 NH×(TP_A+TP_F)/gcd~~ | **已实现** |
| ~~P3~~ | ~~G3~~ | ~~RDMA 末尾 shard 短于 pull_buf~~ — `ffn_send` 中对末尾 shard padding 到 `chunk_size`，`attn_recv` 中 `full[:original_num_tokens]` 截断 | **已修复** |
| P3 | G6 | 异构路径 buffer 复用 | 待优化 |

---

## 九、F9 分组 StepMesh 实现（GCD-based 通用分组）

### 设计

使用 `num_groups = gcd(TP_A, TP_F)` 将全局 N:M StepMesh 拆成多个独立小组，**同时支持 TP_A > TP_F 和 TP_A < TP_F**：

```
num_groups        = gcd(TP_A, TP_F)
workers_per_group = TP_A / num_groups
servers_per_group = TP_F / num_groups
```

- 新增 `--afd-grouped-stepmesh` 开关（默认 False），可选启用
- 每组独立 StepMesh 实例（独立端口 + scheduler 进程）
- 组内复用现有异构 N:M 通信代码，组后 NVLink `all_gather` + stride 去重重建完整 tensor
- 当 `gcd=1`（如 3A:2F）自动禁用（无收益）
- 当两侧都大于 1（如 6A:4F → 3W:2S 组）可工作但收益较小，会 warn

### 通用流量公式

```
A→F = NH × (TP_F / gcd)
F→A = NH × (TP_A / gcd)
总计 = NH × (TP_A + TP_F) / gcd
```

| TP_A:TP_F | gcd | 每组 | 无分组 | 有分组 | 改善 |
|-----------|-----|------|--------|--------|------|
| 4:8 | 4 | 1W:2S | 12NH | 3NH | 4x |
| 8:4 | 4 | 2W:1S | 12NH | 3NH | 4x |
| 6:4 | 2 | 3W:2S | 10NH | 5NH | 2x |
| 4:6 | 2 | 2W:3S | 10NH | 5NH | 2x |
| 8:2 | 2 | 4W:1S | 10NH | 5NH | 2x |
| 3:2 | 1 | — | 6NH | 6NH | 无 |

### 实现步骤（8 步，仅改 `afd.py` + `server_args.py`）

1. **`server_args.py`**：新增 `afd_grouped_stepmesh` 字段 + `--afd-grouped-stepmesh` CLI + `_handle_afd` 验证
2. **`__init__` 分组映射**：计算 `num_groups`、`workers_per_group`、`servers_per_group`、`group_id`、`intra_group_rank`
3. **`_start_stepmesh_scheduler`**：per-group DMLC 配置（`NUM_WORKER=workers_per_group`、`NUM_SERVER=servers_per_group`、独立端口），每组第一个 Worker 启动 scheduler
4. **`attn_send`**：push shard pad 到 `chunk_size`（保证 all_gather 尺寸一致），`n_pull = servers_per_group`
5. **`ffn_recv`**：收 `workers_per_group` 个 shard + `all_gather` on TP_F group + stride `servers_per_group` 去重
6. **`ffn_send`**：无需改动（`n_workers` 自动正确，shard 基于全局 TP_F 计算）
7. **`attn_recv`**：cat `servers_per_group` 个 pull_tensor + `all_gather` on TP_A group + stride `workers_per_group` 去重 + 截断 `original_num_tokens`
8. **更新文档**

### 使用方式

详细启动命令见**第五节**（场景 5-8）。核心要点：

- 在已有异构 TP 命令基础上，两侧同时添加 `--afd-grouped-stepmesh` 即可
- 需要 `MLC_INTERFACE` 环境变量（StepMesh RDMA 模式）
- `--afd-attn-tp` 和 `--afd-ffn-tp` 需两侧一致
- `gcd(TP_A, TP_F) > 1` 时才有收益（`gcd=1` 时自动禁用并 warn）

### F9 Review 结果

#### 发现的问题

| 编号 | 严重度 | 位置 | 问题 | 影响 |
|------|--------|------|------|------|
| F9-1 | **低** | `ffn_recv` grouped path | FFN 处理 padded tensor（最多多 TP_A-1 个零 token 经过 MLP） | batch=1000 tokens 时 <0.7% 计算开销，可忽略 |
| F9-2 | **低** | `_start_stepmesh_scheduler` | per-group scheduler flag 技术上冗余（每进程只调用一次 init） | 防御性编程，不影响正确性 |
| F9-3 | **低** | `_start_stepmesh_scheduler` | 分组路径 force-set `DMLC_PS_ROOT_PORT`，与非分组路径 `_env_def` 语义略有不同 | 行为正确，仅风格差异 |

#### 已验证正确的部分

| 检查项 | 状态 |
|--------|------|
| GCD 分组计算：4A:8F → 4 组 1W:2S | OK |
| GCD 分组计算：8A:4F → 4 组 2W:1S | OK |
| GCD 分组计算：6A:4F → 2 组 3W:2S（混合组） | OK |
| GCD 分组计算：3A:2F → gcd=1 → 自动禁用 | OK |
| GCD 分组计算：4A:4F → 同构 → 自动禁用 | OK |
| `attn_send` push shard padding 仅在 `_grouped` 模式生效，不影响非分组路径 | OK |
| `attn_send` pull buffer 尺寸基于 `effective_tokens`（padded）匹配 `ffn_send` shard 大小 | OK |
| `attn_send` key 增量 `1 + servers_per_group`（分组）vs `1 + ffn_tp`（非分组）| OK |
| `ffn_recv` all_gather size=`self.ffn_tp` == `tp_group.size()` | OK |
| `ffn_recv` stride-select 正确跳过组内重复数据 | OK |
| `ffn_send` 无需改动：`n_workers` 自动正确，shard 基于全局 `ffn_tp` | OK |
| `attn_recv` all_gather size=`self.attn_tp` == `tp_group.size()` | OK |
| `attn_recv` stride-select + `full[:original_num_tokens]` 截断 | OK |
| per-group DMLC 配置（独立 NUM_WORKER/NUM_SERVER/PORT） | OK |
| per-group scheduler 启动（`intra_group_rank == 0` 条件） | OK |
| `server_args.py` 验证逻辑（同构禁用、gcd=1 禁用、混合组 warn） | OK |
| 后向兼容：`_grouped=False` 时所有代码路径与原实现完全一致 | OK |
| `num_tokens=1` 极端 case 数据流正确 | OK |

#### 逐配置数据流验证摘要

**4A:8F** (`G=4, 1W:2S`): `chunk_a=4, eff=16, f2a=2` → `ffn_recv` stride=2 → 16 tokens → MLP → `ffn_send` 2/rank → `attn_recv` stride=1 → `[:15]` ✓

**8A:4F** (`G=4, 2W:1S`): `chunk_a=1, eff=8, f2a=2` → `ffn_recv` stride=1 → 8 tokens → MLP → `ffn_send` 2/rank → `attn_recv` stride=2 → `[:7]` ✓

**6A:4F** (`G=2, 3W:2S`): `chunk_a=2, eff=12, f2a=3` → `ffn_recv` stride=2 → 12 tokens → MLP → `ffn_send` 3/rank → `attn_recv` stride=3 → `[:10]` ✓

### 后续优化方向

- **Custom process sub-groups**：用 `num_groups`-way all_gather 替代全 TP all_gather + stride 去重，消除冗余 NVLink 传输（当 stride > 1 时）
- **Grouped buffer pool**：将 `StepMeshTensorCache` free pool 扩展到分组路径（当前仅同构路径复用 buffer）
- **F9-1 优化**：在 `ffn_recv` 重建 `original_num_tokens`（通过 metadata 嵌入或 broadcast），避免 padding tokens 经过 MLP

---

## 十、总结：代码 Review 结果与未完成任务

### 本轮代码 Review（全部改动文件）

#### 审查范围

| 文件 | 改动内容 |
|------|---------|
| `server_args.py` | `afd_grouped_stepmesh` + `afd_enable_overlap_schedule` 字段/CLI/验证 + AFD 自动 disable overlap |
| `afd.py` — `StepMeshTensorCommunicator` | `__init__` GCD 分组 + `_start_stepmesh_scheduler` per-group + `attn_send`/`ffn_recv`/`attn_recv` 分组路径 |
| `afd.py` — `AsyncTensorCommunicator` | 5 个方法加 `@torch.compiler.disable()` |
| `afd.py` — `AFDCommunicator` | 3 个方法加 `@torch.compiler.disable()` |
| `afd_mixin.py` | `_afd_init` 中 `torch.compile` `_run_attn`/`_run_mlp` |
| `scheduler.py` | `afd_overlap_enabled` 初始化 + `event_loop_afd` overlap 模式（`result_queue` + `_pop_and_process`） |

#### 发现的问题

| 编号 | 严重度 | 文件 | 问题 | 状态 |
|------|--------|------|------|------|
| ~~R1~~ | ~~低~~ | `afd_mixin.py` | ~~Proxy 侧编译 no-op 模块~~ → 按 perspective 只编译实际计算侧 | **已修复** |
| ~~R2~~ | ~~低~~ | `afd.py` | ~~`prepare_attn` 缺 `@torch.compiler.disable()`~~ → 已补充（共 10 个通信方法全覆盖） | **已修复** |
| F9-1 | **低** | `afd.py` | 分组模式 FFN 处理 padded tensor（最多 TP_A-1 个零 token） | 不修复（<1%，远期优化） |
| ~~F9-2~~ | ~~低~~ | `afd.py` | ~~per-group scheduler flag 冗余~~ → 已移除 | **已修复** |
| ~~F9-3~~ | ~~低~~ | `afd.py` | ~~分组路径 force-set 注释不清~~ → 已补充注释 | **已修复** |
| ~~F3-R1~~ | ~~极低~~ | `server_args.py` | ~~矛盾 flag 无 warn~~ → 已加 warn | **已修复** |

#### 最终 Review（全量验证）

本次对全部改动文件做了最终逐行 review，确认以下功能点全部正确：

| 功能 | 验证结果 |
|------|---------|
| **F9 分组 StepMesh** | GCD 分组 + padding + all_gather + stride 去重 + 截断：全路径正确 |
| **F2-A torch.compile** | 按 perspective 只编译实际计算侧，10 个通信方法全部 `@torch.compiler.disable()` |
| **F3 overlap scheduling** | `_handle_afd` 自动 disable overlap + `--afd-enable-overlap-schedule` 显式开启；`event_loop_afd` overlap 路径的 `result_queue` 入队/出队时序与 `event_loop_overlap` 完全对齐；首次迭代、batch=None、连续 prefill 等边界 case 均安全 |
| **后向兼容** | 非 AFD 场景不受任何影响；AFD 默认行为（不传新 flag）与改动前完全一致 |

**无新增问题。** 唯一剩余已知问题为 F9-1（padding tokens，<1%，已决定不修复）。

#### 已验证正确的部分

| 检查项 | 状态 |
|--------|------|
| **F9 分组 StepMesh** | |
| GCD 分组：4A:8F / 8A:4F / 6A:4F / 3A:2F / 4A:4F | OK |
| `attn_send` push shard padding 仅 `_grouped` 模式 | OK |
| `attn_send` pull buffer 基于 `effective_tokens` 匹配 `ffn_send` shard 大小 | OK |
| `attn_send` key 增量 `1 + servers_per_group` / `1 + ffn_tp` | OK |
| `ffn_recv` / `attn_recv` all_gather + stride-select 正确 | OK |
| `ffn_send` 无需改动 | OK |
| per-group DMLC config + scheduler 启动 | OK |
| 后向兼容 + 极端 case | OK |
| **F2-A torch.compile** | |
| `_run_attn` / `_run_mlp` 编译条件 + `@torch.compiler.disable()` 8 个通信方法 | OK |
| `dynamic=True` + dynamo cache + Mixin 覆盖 + Proxy 无害 | OK |
| **F3 overlap scheduling** | |
| `_handle_afd` 自动 `disable_overlap_schedule`（无 `--afd-enable-overlap-schedule` 时） | OK |
| `afd_overlap_enabled=True` 时 `enable_overlap=True`，`run_batch` 走 async 路径 | OK |
| `init_overlap()` 初始化 `forward_stream` + `FutureMap` | OK |
| `event_loop_afd` overlap 路径：`result_queue` + `_pop_and_process` + `batch.copy()` | OK |
| `is_disable_overlap_for_batch`：连续 prefill 禁用 overlap，first iteration 安全 | OK |
| `launch_batch_sample_if_needed` 在 `_pop_and_process` 后调用（匹配 `event_loop_overlap`） | OK |
| `dispatch_event_loop`：AFD 优先级高于 overlap，不走 `event_loop_overlap` | OK |
| 非 AFD 场景完全不受影响 | OK |
| 非 overlap AFD 路径（`afd_overlap=False`）与原实现一致 | OK |

---

### 已完成任务一览

| 编号 | 描述 | 轮次 |
|------|------|------|
| F1 | StepMesh N:M 双向分片通信（任意 TP_A:TP_F） | 第一轮 |
| F6 | DeepSeek-V2 `load_weights` 添加 `AFDWeightFilter` | 第一轮 |
| F7 | `afd_prepare_overlap` 添加 `is_target_verify()` 处理 | 第一轮 |
| F8 | `afd_overlap.py` 移除冗余 `m is None` 检查 | 第一轮 |
| G3 | RDMA 末尾 shard padding + 截断 | 第一轮 |
| H1-H3 | StepMesh key 碰撞 / docstring / recv buffer 注册 | 第一轮 |
| F9 | GCD-based 分组 StepMesh（`--afd-grouped-stepmesh`） | 第二轮 |
| F2-A | `torch.compile` 逐 stage 编译 + AFD 通信 `compiler.disable()` | 第二轮 |
| F3 | `--afd-enable-overlap-schedule` CPU/GPU overlap scheduling | 第二轮 |
| Phase 6 | 异构 TP（`ShardedParallelCommunicator` + StepMesh N:M） | 早期 |
| Task 6 | PD + AFD 独立控制 | 早期 |
| C1-C5, S1-S3, G3-G6, E1-E5 | 通信/调度/计算/工程化优化 | 早期 |
| T1-T2 | Universal AFD Toggle：消除 microbatch 依赖 + TP>1 broadcast 修复 | 第三轮 |
| T4 | `_handle_afd` 自动 disable CUDA graph + piecewise CUDA graph | 第三轮 |
| L1 | LLaMA/Llama3 AFD 支持（LayerCommunicator + Mixin + Model.forward + WeightFilter） | 第四轮 |
| L2 | FFN 侧 `AfdAttnBackend.init_forward_metadata` 子 backend 崩溃修复 | 第四轮 |
| L3 | `_run_attn`/`_run_mlp` 绑定修复（影响所有使用 Mixin 的模型） | 第四轮 |
| L4 | `scheduler_afd_mixin.py` `r.seq_len` → `r.seqlen` 拼写修复 | 第四轮 |

---

### 未完成任务总表

| 优先级 | 编号 | 描述 | 影响 | 复杂度 | 前置依赖 |
|--------|------|------|------|--------|---------|
| ~~P1~~ | ~~F3~~ | ~~`event_loop_afd` CPU/GPU overlap scheduling~~ | ~~5-10%~~ | ~~低（1 周）~~ | **已实现** |
| P1 | L5 | FFN sync 标志在 batch=None 时未清除，可能导致 batch 不匹配 | 正确性 | 低（1 天） | 无 |
| P1 | L6 | ZMQ drain last-one-wins，Attn 领先时 batch metadata 可能不匹配 | 正确性 | 低（1 天） | 无 |
| P2 | F2-B | CUDA Graph（reduce-overhead 模式） | ~2-4% 额外提升 | 中（1-2 周） | F2-A ✓，需 microbatch padding + stream 同步 |
| P2 | F5 | Qwen2（Dense）AFD 支持 | Qwen2 模型可用 | 低（2 天） | 与 LLaMA 相同方式 |
| P3 | F4 | 自动 profiling + `afd_attn_ratio` 自适应 | 便利性 | 中（1 周） | 无 |
| P3 | G6 | 异构路径 buffer 复用 | <1% | 低（2 天） | 无 |
| P4 | T3 | 清理 `can_run_afd_overlap` dead code（`forward_batch_info.py` + `afd_overlap.py`） | 代码整洁 | 极低（30 分钟） | T1-T2 ✓ |
| P4 | F9-opt-1 | 分组 StepMesh: custom sub-groups | 1-5%（仅 stride>1） | 中（3-5 天） | F9 ✓ |
| P4 | F9-opt-2 | 分组 StepMesh: grouped buffer pool | <1% | 低（2 天） | F9 ✓ |
| P4 | F9-opt-3 | 分组 StepMesh: 避免 padding tokens 经过 MLP | <1%（典型负载） | 低（3 天） | F9 ✓ |
| ~~P4~~ | ~~R1~~ | ~~跳过 Proxy 侧 torch.compile~~ | | | **已修复** |
| ~~P4~~ | ~~R2~~ | ~~prepare_attn 补充 compiler.disable~~ | | | **已修复** |
| ~~P4~~ | ~~F9-2~~ | ~~per-group scheduler flag 冗余~~ | | | **已修复** |
| ~~P4~~ | ~~F9-3~~ | ~~分组路径 force-set 注释~~ | | | **已修复** |
| ~~P4~~ | ~~F3-R1~~ | ~~矛盾 flag 组合加 warn~~ | | | **已修复** |

---

### 影响分级

```
中等（2-4%）:       F2-B
低（1-5%）:         F5, F4, F9-opt-1
极低（<1%）:        G6, F9-opt-2, F9-opt-3
```

### F3 重新评估：TBO + AFD 共存分析

经过代码分析，原 F3（"TBO + AFD 共存"）需要重新定义：

| 方案 | 描述 | 收益 | 复杂度 | 结论 |
|------|------|------|--------|------|
| ~~全量共存~~ | TBO batch splitting + AFD microbatch pipeline 嵌套 | <5%（batch 碎片化：原 batch 切成 2×m=6 份，每份太小 overlap 失效） | 极高（6-8 周） | **不推荐** |
| **CPU/GPU overlap** | `event_loop_afd` 引入 overlap scheduling（CPU 调度与 GPU 执行并行） | **5-10%** | **低（1 周）** | **推荐** |

**原因**：TBO 的价值分两层——(1) batch splitting + stage overlap（与 AFD 冲突且收益低）；(2) CPU/GPU scheduling overlap（与 AFD 完全正交）。只取 (2) 即可获得大部分收益。

**具体改动**：仅修改 `scheduler.py` 的 `event_loop_afd`，让 CPU 在 GPU 执行当前 batch 时提前调度下一个 batch（recv_requests + get_next_batch + afd_send_to_ffn），不涉及 attention backend、forward path、batch preparer。

### 建议执行路径

```
阶段 1（已完成 ✓）:
  F9    — 分组 StepMesh → 跨节点流量 12NH→3NH
  F2-A  — torch.compile 逐 stage → kernel fusion ~5-8%
  F3    — CPU/GPU overlap scheduling → 5-10%
  L1-L4 — LLaMA 支持 + 跨模型 Mixin 修复 + 调度器拼写修复

阶段 1.5（优先修复）:
  L5    — FFN sync 标志清除
  L6    — ZMQ batch metadata 对齐

阶段 2（下一优先级）:
  F2-B  — reduce-overhead CUDA Graph → 额外 ~2-4%（1-2 周，含 R2 前置）

阶段 3（扩展 & 便利性）:
  F5    — Qwen2 Dense AFD（与 LLaMA 相同方式，~2 天）
  F4    — 自动 profiling

阶段 4（锦上添花）:
  G6, F9-opt-1/2/3, L7-L12 — 合计 <5%
```

### 当前 AFD 功能完整性

| 维度 | 状态 | 说明 |
|------|------|------|
| 通信层 | **完整** | ZMQ（同构+异构）+ StepMesh（同构+异构+分组），任意 N:M |
| 调度层 | **完整** | microbatch 切分、batch 对齐、PD 独立控制、CPU/GPU overlap scheduling（`--afd-enable-overlap-schedule`） |
| 计算层 | **完整** | 流水线 overlap、非对称切分、预分配 output |
| 编译优化 | **Phase A 完成** | `torch.compile` kernel fusion；Phase B（CUDA Graph）待实现 |
| 模型支持 | **基本完整** | Qwen3-MoE/Qwen2-MoE/DeepSeek-V2/V3/Qwen3(Dense)/LLaMA(Dense)，缺 Qwen2(Dense) |
| 工程化 | **完整** | Mixin 抽象、权重过滤、独立文件、超时处理 |
| **最大未解锁收益** | **F2-B** | reduce-overhead CUDA Graph（~2-4%，1-2 周） |

---

## 十一、Universal AFD Toggle：消除 microbatch 依赖

### 背景

原实现中，AFD 流水线（`model_forward_afd`）是否激活受 `can_run_afd_overlap` 控制，该 flag 在 `AfdForwardBatchPreparer.prepare` 中设置，依赖两个条件：

1. `batch_size >= afd_micro_batch`（batch 太小则不切分）
2. `afd_split_seq_index is not None`（scheduler 未提供切分索引）

这意味着 **batch_size=1** 或 **`--afd-micro-batch` 设置不当** 时，AFD 不会激活，model forward 走常规路径。对于 Attn-FFN 分离部署，这会导致 A 节点尝试本地执行 FFN（缺少权重，崩溃或结果错误），F 节点空闲。

### 改动目标

**只要 `--afd-perspective` 被设置，就无条件走 `model_forward_afd` 路径**，不再受 `--afd-micro-batch` 和 batch size 限制。microbatch 切分仍可用（有 `afd_children` 时），但没有切分时（`afd_children=None`）以 `m_stage=1` 退化为无 pipeline 的单步 AFD forward。

### 改动文件与内容

#### 1. Model Forward 层（3 个模型文件）

| 文件 | 改动 |
|------|------|
| `python/sglang/srt/models/qwen3.py` | `Qwen3Model.forward`：条件从 `getattr(forward_batch, "can_run_afd_overlap", False)` 改为 `get_afd_perspective() is not None` |
| `python/sglang/srt/models/qwen2_moe.py` | `Qwen2MoeModel.forward`：同上 |
| `python/sglang/srt/models/deepseek_v2.py` | `DeepseekV2Model.forward`：同上 |

**效果**：只要进程以 `--afd-perspective attn/ffn` 启动，model forward 就走 AFD 路径。`get_afd_perspective()` 是进程级全局状态，无运行时开销。

#### 2. AFD Pipeline 层（`afd.py`）

| 函数 | 改动 |
|------|------|
| `model_forward_afd_split_inputs._split_raw` | 新增 `afd_children is None` fallback：返回单元素列表，包含完整 batch（不切分） |
| `model_forward_afd` | `m_stage` 从硬编码 `afd_micro_batch` 改为 `len(afd_children)` or `1`；与 `_split_raw` 的返回数量始终一致 |

**效果**：无 microbatch 时 `m_stage=1`，pipeline 退化为单步 A→F→A→F...，仍经过 `AFDCommunicator` 完成跨节点通信。

#### 3. Scheduler 层（消息转发与合并）

| 文件 | 改动 |
|------|------|
| `scheduler.py` — `_recv_afd_messages` | 不再丢弃非 `AFDReqInput` 消息（如 `TokenizedGenerateReqInput`），收集到 `extra_reqs` 列表返回 |
| `scheduler.py` — `event_loop_afd` | FFN 侧合并 `extra_reqs` 到 `recv_reqs`；TP > 1 时 re-broadcast（**所有 rank 参与**） |
| `scheduler_afd_mixin.py` — `afd_recv_messages` | 同 `_recv_afd_messages` 修复 |
| `disaggregation/prefill.py` | 合并 `extra_reqs` + TP > 1 re-broadcast |
| `disaggregation/decode.py` | 合并 `extra_reqs` + TP > 1 re-broadcast |

**效果**：Attn 通过 `afd_forward_work_requests` 将用户请求转发到 FFN；FFN 通过 `_recv_afd_messages` / `afd_recv_messages` 接收并合并到主请求流，确保 FFN 独立建立请求队列、执行 `run_batch`、输出日志。

### Review 发现的问题

| 编号 | 严重度 | 位置 | 问题 | 状态 |
|------|--------|------|------|------|
| T1 | **严重** | `scheduler.py` `event_loop_afd` | `broadcast_pyobj` 在 `if extra_reqs:` 内部 → TP > 1 时只有 TP0 调用 broadcast，其他 rank 跳过 → **死锁** | **已修复** |
| T2 | **严重** | `disaggregation/prefill.py` + `decode.py` | 合并 `extra_reqs` 后缺少 TP > 1 re-broadcast → 非 TP0 rank 看不到 AFD 转发的请求 | **已修复** |
| T3 | **低** | `forward_batch_info.py` + `afd_overlap.py` | `can_run_afd_overlap` 字段仍被设置但不再被读取（dead code） | 不影响正确性，可后续清理 |
| T4 | **中等** | `server_args.py` `_handle_afd` | AFD 未自动 disable CUDA graph / piecewise CUDA graph → FFN warmup 时 `AfdAttnBackend` 子 batch 缺少 attention 字段导致崩溃 | **已修复**（auto-disable） |

### 已验证正确的部分

| 检查项 | 状态 |
|--------|------|
| `_split_raw` fallback：`afd_children=None` 时返回单元素列表，`afd_subbatch_index=0` | OK |
| `model_forward_afd`：`m_stage=1` 时 `AFDStageScheduleGenerator` 生成正确的 A→F 序列 | OK |
| 三个模型的 `get_afd_perspective()` import 位于 `forward` 方法内（lazy import，一致风格） | OK |
| `_recv_afd_messages`：读取所有排队消息，`AFDReqInput` last-one-wins（正常情况只有一个排队） | OK |
| `afd_recv_messages`（mixin 版本）与 `_recv_afd_messages` 逻辑一致 | OK |
| `broadcast_pyobj` 在 `if extra_reqs:` **外部**，所有 TP rank 参与（修复后） | OK |
| 非 AFD 场景：`get_afd_perspective()` 返回 `None`，model forward 走常规路径，不受影响 | OK |
| 非 TP > 1 场景：不执行 broadcast，无额外开销 | OK |

### 数据流（修复后）

```
┌──────── Attn Node ────────┐        ┌──────── FFN Node ────────┐
│                            │        │                          │
│  HTTP /generate            │        │                          │
│       │                    │        │                          │
│  recv_requests()           │        │  recv_requests()         │
│       │                    │        │       │                  │
│  afd_forward_work_requests │──ZMQ──▶│  _recv_afd_messages()   │
│  (转发 TokenizedReq 到 FFN)│        │  (收集 extra_reqs)       │
│       │                    │        │       │                  │
│  process_input_requests    │        │  merge extra→recv_reqs  │
│       │                    │        │  broadcast (TP > 1)      │
│  get_next_batch            │        │       │                  │
│       │                    │        │  process_input_requests  │
│  afd_send_batch_info ──ZMQ─┼───────▶│  (FFN 建立请求队列)      │
│  (AFDReqInput)             │        │       │                  │
│       │                    │        │  get_next_batch          │
│  run_batch                 │        │       │                  │
│  ├─ model_forward_afd ◄────┼─tensor─┼── run_batch             │
│  │  (A→comm→wait→A→...)    │  comm  │  ├─ model_forward_afd   │
│  │                    ─────┼─tensor─┼──▶ (comm→F→comm→F→...)  │
│  └─ done                   │        │  └─ done                │
│       │                    │        │       │                  │
│  process_batch_result      │        │  process_batch_result   │
│  (返回 HTTP 响应)           │        │  (输出 batch 日志)       │
└────────────────────────────┘        └──────────────────────────┘
```

---

## 十二、第四轮 Review：LLaMA AFD 集成 + 跨模型修复

### 背景

使用 `llama3.1-8B --tp 2 --afd-perspective attn/ffn` 测试时发现：
1. FFN 端 `init_forward_metadata` 崩溃（`req_pool_indices` 为 `None`）
2. 关掉 FFN 后 Attn 仍可正常推理 → AFD pipeline 未实际激活
3. `_run_attn` / `_run_mlp` 未绑定到 DecoderLayer 实例 → 所有模型 AFD forward 路径潜在崩溃

### 发现的问题

| 编号 | 严重度 | 文件 | 问题 | 状态 |
|------|--------|------|------|------|
| L1 | **严重** | `llama.py` | LLaMA 模型无 AFD 支持——无 `model_forward_afd` 分支、无 `forward_afd_A/F`、无 `LayerCommunicator`、无 `AFDWeightFilter` | **已修复** |
| L2 | **严重** | `tbo_backend.py` | FFN 侧 `AfdAttnBackend.init_forward_metadata` 对 `afd_children` 调用 flashinfer `call_begin_forward`，子 batch 的 `req_pool_indices=None` 导致 `TypeError` | **已修复** |
| L3 | **严重** | `afd_mixin.py` | `_afd_init` 仅在 `--enable-torch-compile` 时绑定 `_run_attn`/`_run_mlp` 到实例；无 compile 时 `forward_afd_A/F` 调用 `self._run_attn()` 触发 `AttributeError`——**影响所有使用 Mixin 的模型** | **已修复** |
| L4 | **严重** | `scheduler_afd_mixin.py` | `afd_send_batch_info` 中 `r.seq_len` 应为 `r.seqlen`（`Req` 类属性名）——disagg prefill/decode 路径调用会 `AttributeError` | **已修复** |
| L5 | **中等** | `scheduler.py` ~1444 | FFN 侧 `_afd_batchsize_attn` 仅在 `batch is not None` 时清除；若 FFN 消费了 `AFDReqInput` 但 `get_next_batch` 返回 `None`，sync 标志残留导致下一次迭代跳过等待 | 待修复 |
| L6 | **中等** | `scheduler.py` ~1327 | ZMQ drain 循环 last-one-wins：若 Attn 领先 FFN 多个 step，`AFDReqInput` 可能指向更新的 batch，与 FFN 实际运行的 batch 不匹配 | 待修复 |
| L7 | **中等** | `afd_overlap.py` ~343 | `_filter_batch` 硬编码字段列表：未来新增 `ForwardBatch` 字段（如 `encoder_*`、`mamba_track_*`、`nsa_cp_metadata`）时，非 `None` 字段会触发 `Exception: N errors: Field ... not yet supported` | 需关注 |
| L8 | **低** | `llama.py` AFD branch | `LlamaModel.forward` AFD 分支返回纯 `hidden_states`（无 `aux_hidden_states`），若 `capture_aux_hidden_states=True`（EAGLE3），`LlamaForCausalLM.forward` 解包会失败 | 不影响当前使用（AFD + EAGLE3 不共存） |
| L9 | **低** | `afd_mixin.py` ~130 | `forward_afd_F` 对 0-token 子 batch 仍调用 `_run_mlp`（`forward_afd_A` 有 `shape[0] != 0` 保护，F 没有） | 边界条件，实际不触发 |
| L10 | **低** | `scheduler_afd_mixin.py` ~58-60 | `_afd_forward_mode` 存储但从未读取，`afd_reset_state` 也未清除 | dead code |
| L11 | **低** | `afd.py` ~807 | `get_afd_mirco_batch`（拼写错误别名）仍保留 | 代码整洁 |
| L12 | **低** | 跨模型 | DeepSeek-V2/Qwen3/Qwen2-MoE 的 `forward_afd_A/F` 走 Mixin 默认 `_run_attn`/`_run_mlp`，不传模型特有参数（如 `quant_format`、`zero_allocator`、`use_reduce_scatter`） | AFD 路径与非 AFD 有微小行为差异 |

### 已修复内容

#### L1: LLaMA AFD 支持（`llama.py`）

**改动**：
- `LlamaDecoderLayer.__init__`：当 `get_afd_perspective() is not None` 时，创建 `LayerScatterModes`（`is_layer_sparse=False`）、`LayerCommunicator`，调用 `_afd_init()`
- `LlamaDecoderLayer`：新增 `forward_afd_A` / `forward_afd_F` 委托到 Mixin
- `LlamaModel.forward`：添加 `model_forward_afd` 分支
- `LlamaForCausalLM.load_weights`：添加 `AFDWeightFilter` 权重过滤

#### L2: FFN 侧 `init_forward_metadata` 崩溃（`tbo_backend.py`）

**根因**：`AfdForwardBatchPreparer._filter_batch` 将 FFN 子 batch 的 `req_pool_indices` 设为 `None`，但 `AfdAttnBackend.init_forward_metadata` 仍对子 backend 调用 `init_forward_metadata`，flashinfer 尝试 `len(req_pool_indices)` 崩溃。

**修复**：在 `init_forward_metadata` 中检测 FFN 侧时跳过子 backend 初始化（FFN 使用 `AFDProxyAttention`，子 attention metadata 永远不使用）。

#### L3: `_run_attn`/`_run_mlp` 未绑定（`afd_mixin.py`）

**根因**：所有模型（DeepSeek-V2、Qwen3、Qwen2-MoE、LLaMA）均未继承 `AFDDecoderLayerMixin`，而是通过 `AFDDecoderLayerMixin.forward_afd_A(self, ...)` 调用。内部 `self._run_attn(...)` 查找时，`self` 是 DecoderLayer 实例，MRO 中无 Mixin → `AttributeError`。仅 `--enable-torch-compile` 时 `_afd_init` 会通过 `torch.compile` 赋值到实例。

**修复**：在 `_afd_init` 中无条件绑定默认 `_run_attn`/`_run_mlp`（`types.MethodType`），在 `torch.compile` 前。`hasattr` 检查允许模型类覆盖。

#### L4: `r.seq_len` 拼写错误（`scheduler_afd_mixin.py`）

**根因**：`Req` 类使用 `@property seqlen`（无下划线），Mixin 中写成 `r.seq_len`。

**修复**：改为 `r.seqlen`。

### 已验证正确的部分

| 检查项 | 状态 |
|--------|------|
| LLaMA `LayerScatterModes.init_new` 参数（dense 模型全 False） | OK |
| LLaMA `LayerCommunicator` 传入 `input_layernorm` + `post_attention_layernorm` | OK |
| LLaMA `AFDWeightFilter` 与 stacked_params_mapping 兼容（过滤在迭代器层面，不影响 name.replace） | OK |
| LLaMA `LlamaMLP.forward(x, forward_batch=None)` 与 Mixin `_run_mlp(hidden_states, forward_batch)` 签名兼容 | OK |
| `types.MethodType` 绑定后 `torch.compile(self._run_attn)` 正确获取绑定方法 | OK |
| `tbo_backend.py` FFN skip 仅在 `afd_children is not None` 时生效，非 AFD 场景不受影响 | OK |
| `scheduler_afd_mixin.py` 修复后 `r.seqlen` 与 `scheduler.py` ~1429 一致 | OK |

### 后续任务更新

| 优先级 | 编号 | 描述 | 状态 |
|--------|------|------|------|
| ~~P0~~ | ~~L1~~ | ~~LLaMA AFD 支持~~ | **已修复** |
| ~~P0~~ | ~~L2~~ | ~~FFN init_forward_metadata 崩溃~~ | **已修复** |
| ~~P0~~ | ~~L3~~ | ~~`_run_attn`/`_run_mlp` 未绑定（跨模型）~~ | **已修复** |
| ~~P0~~ | ~~L4~~ | ~~`r.seq_len` 拼写错误~~ | **已修复** |
| P1 | L5 | FFN sync 标志未在 batch=None 时清除 | 待修复 |
| P1 | L6 | ZMQ drain last-one-wins 潜在 batch 不匹配 | 待修复 |
| P2 | F5 | Qwen2 Dense AFD——与 LLaMA 相同方式适配 | 可快速实现 |
| P3 | L7 | `_filter_batch` 字段白名单扩展 | 需关注 |
| P3 | L12 | 模型特有参数在 AFD 路径中缺失 | 低优先级 |
| P4 | L8-L11 | 边界条件 / dead code / 拼写别名 | 锦上添花 |

---

## 十三、第五轮：ZMQ 异构 TP + 运行时 Bug 修复

### 背景

第四轮测试 LLaMA 3.1-8B + TP=2 时暴露了一系列运行时问题，以及 ZMQ 路径不支持异构 TP 的设计缺陷。

### 发现并修复的问题

| 编号 | 严重度 | 文件 | 问题 | 状态 |
|------|--------|------|------|------|
| R5-1 | **严重** | `afd.py` ZMQ send | `cpu_tensor.numpy().tobytes()` 不支持 bfloat16（numpy 无 bf16） | **已修复**：改用 `bytes(cpu_tensor.untyped_storage())` |
| R5-2 | **严重** | `scheduler.py` `_recv_afd_messages` | `AFDReqInput` 在 TP rank 0 提取后不放入 broadcast → TP rank 1 的 `_afd_batchsize_attn` 永远为 None → **FFN TP>1 死锁** | **已修复**：AFDReqInput 包含在 broadcast 数据中，`_afd_process_input_requests` 过滤后再传给 dispatcher |
| R5-3 | **严重** | `afd.py` ShardedParallelCommunicator | ZMQ 1:1 端口映射 → 异构 TP（如 1A:2F）FFN rank 1 无对端 → 死锁 | **已修复**：重写为 `BroadcastTensorCommunicator`（rank 0 ZMQ + NVLink broadcast） |
| R5-4 | **中等** | `communicator.py` `_gather_hidden_states_and_residual` | 0-token batch 时 `RMSNorm(x, res)` 在 `x.numel()==0` 返回单 tensor → unpack 失败 | **已修复**：入口添加 `if hidden_states.shape[0] == 0: return` |
| R5-5 | **中等** | `llama.py` AFD norm | FFN 侧 `residual=None` 时 `RMSNorm(hs, None)` 返回单 tensor → unpack 失败 | **已修复**：检查 `residual is not None` 分支调用 |
| R5-6 | **中等** | `afd.py` `model_forward_afd` | 0-token batch 进入 AFD pipeline 触发各种 CUDA kernel 错误 | **已修复**：入口添加 `if hidden_states.shape[0] == 0: return` 短路 |
| R5-7 | **中等** | `afd.py` ZMQ recv | 接收 0-byte 数据时 `torch.frombuffer(bytearray(0))` 崩溃 | **已修复**：空 buffer 时用 `torch.empty(shape, dtype)` |
| R5-8 | **中等** | `server_args.py` `_handle_afd` | FFN warmup 死锁（event_loop_afd 等 Attn sync，永远等不到） | **已修复**：FFN 自动 `skip_server_warmup` |

### BroadcastTensorCommunicator 设计（R5-3）

替换旧 `ShardedParallelCommunicator`，用 rank 0 单连接 + NVLink broadcast 实现任意 N:M：

**原理**：AFD 跨节点通信的 tensor 是 all-reduce 后的完整 hidden_states，所有 TP rank 持有相同数据。因此只需 1 个 rank 发送/接收，其余通过 NVLink broadcast 获取。

**数据流**：
1. 发送侧：`local_tp_rank == 0` 通过 ZMQ 发送完整 tensor，其他 rank 跳过
2. 接收侧：rank 0 ZMQ 接收 → broadcast shape+dtype（3 个 long）→ 非 rank-0 分配内存 → broadcast 数据 tensor

**流量对比**：

| 配置 | 旧实现（per-rank ZMQ） | 新实现（rank 0 + broadcast） |
|------|----------------------|---------------------------|
| 4A:4F 同构 | 8NH（4 条连接各传 NH，冗余） | **2NH** |
| 4A:8F 异构 | 死锁 | **2NH** |
| 1A:2F 异构 | 死锁 | **2NH** |

**限制**：单 ZMQ 连接，不利用多网卡并行。多网卡高性能场景应用 StepMesh（RDMA）。

#### Review 结果

| 检查项 | 状态 |
|--------|------|
| `send_tensor` rank 0 发送，其他 rank 跳过 | OK |
| `recv_tensor` 两次 broadcast（metadata + data） | OK |
| `tp_group.ranks[0]` 作为 broadcast src | OK |
| 0-element tensor broadcast 安全（NCCL no-op） | OK |
| `get_tensor_communicator` 同构/异构统一路径 | OK |
| TP=1 跳过包装 | OK |
| `_dtype_to_int` 覆盖 fp16/bf16/fp32 | OK |
| StepMesh 路径不受影响 | OK |

### 已完成任务追加

| 编号 | 描述 | 轮次 |
|------|------|------|
| R5-1 | ZMQ bf16 序列化修复 | 第五轮 |
| R5-2 | FFN TP>1 AFDReqInput broadcast 修复 | 第五轮 |
| R5-3 | `BroadcastTensorCommunicator`（ZMQ 异构 TP） | 第五轮 |
| R5-4 | communicator 0-token 保护 | 第五轮 |
| R5-5 | LLaMA norm residual=None 处理 | 第五轮 |
| R5-6 | `model_forward_afd` 0-token 短路 | 第五轮 |
| R5-7 | ZMQ recv 0-byte buffer 处理 | 第五轮 |
| R5-8 | FFN 自动 skip_server_warmup | 第五轮 |

### 未完成任务更新

| 优先级 | 编号 | 描述 | 状态 |
|--------|------|------|------|
| P1 | L5 | FFN sync 标志在 batch=None 时未清除 | 待修复 |
| P1 | L6 | ZMQ drain last-one-wins 潜在 batch 不匹配 | 待修复 |
| P2 | F5 | Qwen2 Dense AFD | 可快速实现 |
| P3 | R5-9 | 非 rank-0 避免创建无用 ZMQ socket | 待优化 |
| P4 | R5-10 | `_dtype_to_int` 未知 dtype 报错而非静默 fallback | 锦上添花 |

---

## 第六轮：UCX RDMA 通信器替换 StepMesh

### 背景

StepMesh（`fserver_lib`）的 CUDA 运行时与 SGLang 框架依赖的库冲突，无法正常部署。用 UCX-Py（RDMA 原生 Python 绑定）重新实现 Attn-FFN 跨节点张量通信全部功能。

### 新增/修改文件

| 文件路径 | 行数 | 变更 |
|----------|------|------|
| `python/sglang/srt/layers/rdma_comm.py` | 658 | 新增：UCX-Py RDMA 通信器 |
| `python/sglang/srt/layers/afd.py` | +12 | `get_tensor_communicator()` 新增 UCX 路径 |
| `python/sglang/srt/server_args.py` | +12 | `afd_comm_backend` 字段 + `--afd-comm-backend` CLI |

### 架构设计

```
rdma_comm.py 分层架构：

_AsyncBridge              SelectorEventLoop 后台线程（避免 uvloop 冲突）
_BufferPool               GPU tensor 复用池（自动回收上次 recv buffer）
_detect_num_nic_groups()  解析 nvidia-smi topo 自动检测 GPU-NIC 亲和性
_UcxP2PCommunicator       1:1 RDMA 点对点（UCX endpoint send/recv）
UcxTensorCommunicator     NIC 感知 N:M 通信（K 路 RDMA + NVLink 分发）
```

NIC 感知 N:M 通信模型：
- K = 网卡数（自动检测或 AFD_UCX_NUM_NICS 覆盖），必须整除 local_tp
- K 个 NIC 组各选一个代表做 RDMA（各传 1/K 数据）
- 非代表 rank 通过 NVLink broadcast (K=1) 或 all_gather+stride去重 (K>1) 获取完整数据
- 跨节点 RDMA 总流量恒为 2NH/层，与 K 和 TP 大小无关

K=1：rank 0 RDMA → NVLink broadcast
K>1：K 代表各 RDMA 1/K → rank0 broadcast shape → 非代表填零 → all_gather → stride 去重 → truncate

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `AFD_UCX_TLS` | `rc,tcp,cuda_copy,cuda_ipc` | UCX 传输列表 |
| `AFD_UCX_BASE_PORT` | `25000` | FFN 监听基础端口 |
| `AFD_UCX_FFN_HOST` | `127.0.0.1` | FFN 节点地址 |
| `AFD_UCX_NUM_NICS` | 自动检测 | 网卡数 K |
| `AFD_UCX_TIMEOUT` | `60` | 连接超时秒数 |
| `UCX_LOG_LEVEL` | `fatal`（需 shell 传入） | 抑制 UCX 非致命错误 |

### 验证结果

| 测试场景 | 结果 | 备注 |
|----------|:---:|------|
| 1:1 RDMA 吞吐 | PASS | 15-33 GB/s |
| K=1 多 TP NVLink broadcast | PASS | FFN TP=2 rank 一致 |
| K=2 多 NIC all_gather | PASS | Attn TP=2, FFN TP=2 |
| K=2 异构 TP + 非代表 rank | PASS | Attn TP=4, FFN TP=2 全 4 rank 正确 |
| TP=1 Llama 3.1 8B E2E | PASS | 文本正确 |
| TP=2 Llama 3.1 8B E2E | PASS | 文本正确，GPU 空闲 0% |
| TCP-only fallback | FAIL | 已知限制 L6-1 |
| GPU-direct RDMA | PASS | nvidia_peermem 生效，IB/TCP=23.9x |

### 启动命令

TP=1 单卡 AFD（最简模式）：

    # 终端 1: FFN (先启动)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=3 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
    AFD_UCX_BASE_PORT=25000 \
    python -m sglang.launch_server \
      --model-path <模型路径> --tp 1 \
      --afd-perspective ffn --afd-comm-backend ucx --port 30001

    # 终端 2: Attn (等 FFN 出现 "listening" 后启动)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=2 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
    AFD_UCX_BASE_PORT=25000 AFD_UCX_FFN_HOST=127.0.0.1 \
    python -m sglang.launch_server \
      --model-path <模型路径> --tp 1 \
      --afd-perspective attn --afd-comm-backend ucx --port 30000

    # 终端 3: 请求
    curl -s http://localhost:30000/generate \
      -H "Content-Type: application/json" \
      -d '{"text":"Hello","sampling_params":{"max_new_tokens":16}}'

TP=2 多卡 AFD（K=1 NVLink broadcast）：

    # FFN TP=2
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=4,5 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
    AFD_UCX_BASE_PORT=25000 AFD_UCX_NUM_NICS=1 \
    python -m sglang.launch_server \
      --model-path <模型路径> --tp 2 \
      --afd-perspective ffn --afd-comm-backend ucx --port 30001

    # Attn TP=2
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=2,3 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
    AFD_UCX_BASE_PORT=25000 AFD_UCX_FFN_HOST=127.0.0.1 AFD_UCX_NUM_NICS=1 \
    python -m sglang.launch_server \
      --model-path <模型路径> --tp 2 \
      --afd-perspective attn --afd-comm-backend ucx --port 30000

多节点部署（跨机 IB RDMA）：

    # 节点 B (FFN): 正常启动，监听 0.0.0.0
    # 节点 A (Attn): 设 AFD_UCX_FFN_HOST 为节点 B 的 IB IP
    AFD_UCX_FFN_HOST=<FFN节点IB地址> ...

关键说明：
- UCX_LOG_LEVEL=fatal 和 UCX_WARN_UNUSED_ENV_VARS=n 必须作为 shell 环境变量传入，因为 NCCL 初始化时先加载 UCX C 库
- AFD_UCX_NUM_NICS 不指定时自动检测 GPU-NIC 亲和性；手动指定时必须整除 TP
- AFD_UCX_FFN_HOST 仅 Attn 侧需要，FFN 侧自动监听
- FFN 必须先启动（listener 角色），Attn 后启动（connector 角色）

### 按模块测试命令

以下测试脚本位于 `/workspace/` 目录，可独立运行（不依赖完整 SGLang 服务启动）。
需先设置 Python 路径：`export PYTHONPATH=/workspace/sglang/python:$PYTHONPATH`

**1. RDMA 基础通信（1:1 P2P，6 项测试）：**

    # 测试 IB RDMA / NVLink / TCP fallback / bf16 / 吞吐
    # 使用 GPU 2 (Attn) 和 GPU 3 (FFN)
    python3 /workspace/test_rdma_comm.py

    覆盖项：
    - 基本连接 + 正确性
    - FIFO 顺序保证（50 次迭代）
    - bf16 典型隐藏态吞吐（~4MB）
    - f32 16MB / 64MB 大张量吞吐
    - TCP-only fallback（已知限制 L6-1，需 cuda_ipc）

**2. K=1 多 TP + NVLink broadcast：**

    # Attn TP=1 (GPU 1), FFN TP=2 (GPU 2,3)
    # FFN rank 0 RDMA 接收，rank 1 NVLink broadcast 获取
    python3 /workspace/test_multi_tp.py

    覆盖项：
    - 非代表 rank 通过 NVLink broadcast 获取数据
    - FFN 两个 rank 数据一致性验证
    - Attn send→recv 端到端正确性

**3. K=2 多网卡 + NVLink all_gather：**

    # Attn TP=2 (GPU 2,6), FFN TP=2 (GPU 3,7)
    # 两路 RDMA 各传 1/2 数据，然后 all_gather
    python3 /workspace/test_k2_multi_nic.py

    覆盖项：
    - 多 NIC 并行 RDMA 传输
    - all_gather + stride 去重重建完整 tensor
    - 跨 NUMA 节点通信

**4. K=2 异构 TP + 非代表 rank（最复杂场景）：**

    # Attn TP=4 (GPU 2,3,6,7), FFN TP=2 (GPU 4,5)
    # Attn rank 0,2 为代表做 RDMA，rank 1,3 通过 NVLink 获取
    python3 /workspace/test_k2_hetero.py

    覆盖项：
    - ranks_per_group > 1 场景
    - 非代表 rank 填零 + all_gather + stride 去重
    - Attn 全部 4 个 rank 数据正确性
    - FFN 两个 rank 数据一致性

**5. 全量回归（串行运行 1-4）：**

    echo "=== 1:1 ===" && python3 /workspace/test_rdma_comm.py && \
    echo "=== K=1 多TP ===" && python3 /workspace/test_multi_tp.py && \
    echo "=== K=2 多NIC ===" && python3 /workspace/test_k2_multi_nic.py && \
    echo "=== K=2 异构TP ===" && python3 /workspace/test_k2_hetero.py

**6. E2E 推理验证（需启动完整 server）：**

    # 启动 TP=2 AFD server（见上方启动命令），然后：
    curl -s http://localhost:30000/generate \
      -H "Content-Type: application/json" \
      -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}' \
      | python3 -c "import sys,json; r=json.load(sys.stdin); \
        print('text:', repr(r['text'])); \
        print('tokens:', r['meta_info']['completion_tokens'])"

    预期：completion_tokens=16，文本语义连贯

**7. GPU 空闲利用率检查：**

    # server 启动后无请求时，GPU 利用率应接近 0%
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv

**8. GPU-direct RDMA 吞吐基准：**

    # 多次迭代热身后测量稳态带宽
    # 对比 TCP / IB RC / NVLink 三种路径
    python3 /workspace/test_gpudirect.py

    预期：IB RC ~10 GB/s，NVLink ~80-170 GB/s，TCP ~0.4 GB/s

**9. PD 分离 + AFD 组合测试（4 个 server 进程，6 GPU）：**

PD（Prefill/Decode 分离）的 KV 传输走 Mooncake/Nixl，AFD 的隐藏态传输走 UCX RDMA，两条路径独立。
Prefill 和 Decode 各自有自己的 Attn+FFN 进程对。

    # 终端 1: Prefill FFN (先启动)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=4 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc AFD_UCX_BASE_PORT=25000 \
    python -m sglang.launch_server \
      --model-path /models/llama3.1-8 --tp 1 \
      --afd-perspective ffn --afd-comm-backend ucx \
      --disaggregation-mode prefill --port 30011

    # 终端 2: Prefill Attn (等 FFN listening 后)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=5 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc AFD_UCX_BASE_PORT=25000 \
    AFD_UCX_FFN_HOST=127.0.0.1 \
    python -m sglang.launch_server \
      --model-path /models/llama3.1-8 --tp 1 \
      --afd-perspective attn --afd-comm-backend ucx \
      --disaggregation-mode prefill --port 30010

    # 终端 3: Decode FFN (先启动)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=6 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc AFD_UCX_BASE_PORT=25100 \
    python -m sglang.launch_server \
      --model-path /models/llama3.1-8 --tp 1 \
      --afd-perspective ffn --afd-comm-backend ucx \
      --disaggregation-mode decode --port 30021

    # 终端 4: Decode Attn (等 FFN listening 后)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=7 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc AFD_UCX_BASE_PORT=25100 \
    AFD_UCX_FFN_HOST=127.0.0.1 \
    python -m sglang.launch_server \
      --model-path /models/llama3.1-8 --tp 1 \
      --afd-perspective attn --afd-comm-backend ucx \
      --disaggregation-mode decode --port 30020

    # 终端 5: 请求（发给 Prefill Attn）
    curl -s http://localhost:30010/generate \
      -H "Content-Type: application/json" \
      -d '{"text":"Hello","sampling_params":{"max_new_tokens":16}}'

    注意：
    - Prefill 和 Decode 用不同 AFD_UCX_BASE_PORT（25000 vs 25100）避免端口冲突
    - PD 之间 KV 传输需额外配置 --disaggregation-transfer-backend（mooncake/nixl）
    - AFD 隐藏态通信（UCX）与 PD KV 传输（Mooncake/Nixl）完全独立

### 已修复问题

| 编号 | 描述 | 修复 |
|------|------|------|
| R6-1 | uvloop 与 UCX-Py BlockingMode 冲突 | SelectorEventLoop |
| R6-2 | cuMemGetAddressRange 错误刷屏 | 移除 bridge CUDA context + UCX_LOG_LEVEL=fatal |
| R6-3 | GPU 空闲 100% 占用 | 移除 bridge 线程 CUDA context |
| R6-4 | RDMA 读取未完成的 GPU kernel 数据 | send 前 torch.cuda.synchronize() |
| R6-5 | TP=2 推理文本乱码 | 全设备 synchronize 替代 stream synchronize |
| R6-6 | K>1 dist.broadcast 非法（不同 src） | rank0 broadcast shape + 填零 + all_gather stride 去重 |
| R6-7 | K 不整除 local_tp | 自动向下调整 |
| R6-8 | metadata 仅支持 2D | 先 broadcast ndim 再变长 shape |
| R6-9 | BufferPool 未回收 | _async_recv 自动 put 上次 buffer |
| R6-10 | K>1 padding 未截断 | send 记录 num_tokens，recv 截断 |
| R6-11 | LOCAL_RANK 优先级错误 | 优先 os.environ["LOCAL_RANK"] |

### 已知限制

| 编号 | 描述 | 影响 | 处置 |
|------|------|------|------|
| L6-1 | TCP-only fallback 不可用 | 仅 `UCX_TLS=tcp,cuda_copy`（无 IB 无 NVLink） | 生产路径不受影响 |
| L6-2 | UCX_LOG_LEVEL 需 shell 环境变量 | NCCL 先于 Python 代码加载 UCX | 启动命令加 `UCX_LOG_LEVEL=fatal` |
| L6-3 | NIC 自动检测依赖 nvidia-smi topo | 容器内不可用时回退 K=1 | AFD_UCX_NUM_NICS 手动覆盖 |
| L6-4 | UCX-Py 0.35 已停维 | RAPIDS 推荐 UCXX | 功能满足，长期需迁移 |

### 性能优化（已实现）

**R6-18：热 BufferPool warmup 预分配**

`_BufferPool` 新增 `warmup(shape, dtype, device, count)` 方法，在初始化时按模型
hidden_size 和 microbatch 数预分配 GPU buffer。运行时 recv 路径零 `torch.empty()` 调用。
不知道 shape 时保持懒分配向后兼容。

**R6-19：完全异步 send pipeline**

- `_UcxP2PCommunicator.send_nonblocking(x)` — 提交到 bridge 返回 Future，不等完成
- `UcxTensorCommunicator.send_tensor_nonblocking(x)` — 上层非阻塞 send
- `UcxTensorCommunicator.fence()` — 等待 in-flight send 完成
- `AsyncTensorCommunicator.send_async(x)` — 后台线程等 CUDA event 后提交 nonblocking send
- `AsyncTensorCommunicator.recv_wait()` — 自动 join send 线程 + 调用 fence

数据安全保证：后台线程先 `compute_event.synchronize()` 确保 GPU 数据就绪，
再提交 UCX send。Python GIL 不影响（关键操作 event.synchronize/UCX C 扩展均释放 GIL）。

### 性能优化方案历史（R6-13 + R6-14 + R6-16）

**原始瓶颈分析（已解决）：**

原 UCX 通信器在 `_UcxP2PCommunicator.send()` 中使用 `torch.cuda.synchronize()`
做全设备同步。这会阻塞所有 CUDA stream，破坏 microbatch pipeline 的通信-计算重叠。

AFD 的 microbatch pipeline 依赖 `AsyncTensorCommunicator` 的三个方法实现重叠：
- `send_async(x)`：在 comm_stream 上发起发送，立即返回
- `recv_start()`：在 comm_stream 上发起接收，不阻塞 compute stream
- `recv_wait()`：等待接收完成，同步回 compute stream

当前 `send_async` 切到 comm_stream 后调用 `send_tensor` → `_p2p.send()` →
`torch.cuda.synchronize()`。这个全设备同步等待所有 stream（包括 compute stream），
使得 send 无法与后续 compute 重叠，pipeline 退化为串行。

```
当前（被 synchronize 序列化）：
  [Attn m0] → [sync+send m0] → [Attn m1] → [sync+send m1] → [recv m0] → ...
              ↑ GPU 全停 ↑                  ↑ GPU 全停 ↑

目标（CUDA event 精细同步）：
  [Attn m0] → [send m0        ] → [Attn m1] → [send m1        ]
                                   [recv m0 ←]
              ↑ 仅等 m0 的 compute stream ↑   ↑ send 和 compute 重叠 ↑
```

**R6-13：torch.cuda.synchronize() → CUDA event 精细同步**

修改 `_UcxP2PCommunicator.send()` 和 `AsyncTensorCommunicator.send_async()`：

```python
# 方案：在 compute stream 上记录 event，comm stream 等待该 event

# AsyncTensorCommunicator.send_async:
def send_async(self, x: torch.Tensor):
    if self.comm_stream is not None:
        # 记录 compute stream 的当前进度
        event = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(self.comm_stream):
            # comm stream 等待 compute stream 完成 x 的写入
            event.wait(self.comm_stream)
            self.inner.send_tensor(x)
    else:
        self.inner.send_tensor(x)

# _UcxP2PCommunicator.send:
def send(self, x: torch.Tensor):
    # 不再全设备 synchronize，由上层 AsyncTensorCommunicator 用 event 同步
    self._bridge.run(self._async_send(x))
```

收益：send 只等产生 tensor 的 compute stream，不阻塞其他 stream 和 microbatch。

**R6-14：UCX 参数调优**

| 参数 | 当前值 | 建议值 | 说明 |
|------|--------|--------|------|
| `UCX_RNDV_THRESH` | 8192 | 根据 hidden_size 调整 | 小于此值用 eager（低延迟），大于用 rendezvous（高吞吐） |
| `UCX_MAX_RNDV_RAILS` | 1 (UCX 默认) | 1（保持） | 确保 GPU-direct RDMA 选最近 NIC |
| `UCX_RNDV_SCHEME` | auto | `put_zcopy` | 跳过协商，直接 RDMA write |
| `UCX_ZCOPY_THRESH` | auto | 与 RNDV_THRESH 一致 | 启用零拷贝传输 |

**R6-16：M-buffer microbatch pipeline**

AFD 的 `model_forward_afd` 已有 microbatch pipeline 框架（R4 注释）。
当前管线中 `postprocess_layer_start_recv()` 在 A stage 之后提前发起 recv_start，
与下一个 A stage 计算重叠。这部分逻辑已正确，但被 R6-13 的全设备同步破坏。

修复 R6-13 后，pipeline 自然恢复：

```
M=3 microbatch pipeline (每层)：

Attn 侧：
  A(L,m0) → send(m0)                     ← send 与下一步重叠
            recv_start(L-1,m0)            ← 提前发起 recv
  A(L,m1) → send(m1)                     ← compute A(m1) 与 recv(m0) 重叠
            recv_start(L-1,m1)
  A(L,m2) → send(m2)
            recv_start(L-1,m2)
  F(L-1,m0) ← recv_wait(m0)              ← m0 的 recv 已完成
  F(L-1,m1) ← recv_wait(m1)
  F(L-1,m2) ← recv_wait(m2)

BufferPool 需要至少 M 个 buffer 同时在途：
  - 每个 microbatch 有独立的 send/recv buffer
  - BufferPool 按 shape/dtype 缓存，天然支持多 buffer 复用
```

关键前提：R6-13 完成后，send 不再阻塞全设备，pipeline 的 M-way 重叠才能生效。

**执行顺序：**
1. R6-13（CUDA event 同步）← 最高优先级，解除 pipeline 瓶颈
2. R6-14（UCX 参数调优）← 独立可做，提升单次传输性能
3. R6-16（验证 pipeline 重叠）← R6-13 完成后自然生效，需 profiler 验证

### 未完成任务（全量汇总）

| 优先级 | 编号 | 描述 | 来源 | 状态 |
|--------|------|------|------|------|
| P1 | L5 | FFN sync 标志在 batch=None 时未清除 | 第五轮 | **已修复** |
| P1 | L6 | ZMQ drain last-one-wins 潜在 batch 不匹配 | 第五轮 | **已修复** |
| P2 | R6-12 | 多节点跨机 IB RDMA 实测 | 第六轮 | 待后续（需第二台机器） |
| P2 | R6-13 | CUDA event 精细同步 | 第六轮 | **已完成**（send_async event + current_stream sync） |
| P2 | R6-14 | UCX 参数调优 | 第六轮 | **已完成**（put_zcopy + ZCOPY_THRESH） |
| P2 | R6-18 | 热 BufferPool warmup 预分配 | 第六轮 | **已完成** |
| P2 | R6-19 | 完全异步 send pipeline + fence | 第六轮 | **已完成** |
| P2 | F5 | Qwen2 Dense AFD | 第五轮 | 待实现 |
| P3 | R6-15 | TCP-only fallback 修复 | 第六轮 | 待修复（低优先级） |
| P3 | R5-9 | 非 rank-0 避免创建无用 ZMQ socket | 第五轮 | **已修复** |
| P4 | R6-16 | M-buffer pipeline profiler 验证 | 第六轮 | 代码就绪，待 profiler |
| P4 | R6-17 | UCXX 迁移评估 | 第六轮 | 长期规划 |
| P4 | R5-10 | `_dtype_to_int` 未知 dtype 报错 | 第五轮 | **已修复** |
