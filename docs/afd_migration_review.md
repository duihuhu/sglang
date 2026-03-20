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
| 其他使用 LayerCommunicator 的模型 | 各种 | 可快速适配 | DecoderLayer 加 `_afd_init()` + `forward_afd_A/F` |
| Qwen2（Dense） | Dense | 不支持 | DecoderLayer 不使用 LayerCommunicator，无法直接适配 |

---

## 五、运行方式

> 以下示例使用 `<模型>` 指代模型路径，`<attn_ip>` / `<ffn_ip>` 指代节点 IP，`<RDMA网卡>` 指代 RDMA 网卡名（如 `mlx5_0`）。
> 所有 AFD 场景都需要 `--disable-overlap-schedule --disable-cuda-graph`（F2 待解决）。

### 1. ZMQ 同构 TP（快速测试，无 RDMA）

```bash
# ═══ Attn 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8

# ═══ FFN 节点（TP=8）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8

# ═══ FFN 节点（TP=8，告知对端 Attn TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ FFN 节点（TP=8）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4 \
    --afd-grouped-stepmesh

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 6 --afd-ffn-tp 4 --afd-attn-ratio 0.6 \
    --afd-grouped-stepmesh

# ═══ FFN 节点（TP=4）═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule --disable-cuda-graph \
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
    --disable-overlap-schedule --disable-cuda-graph \
    --disaggregation-mode prefill \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Prefill FFN 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule --disable-cuda-graph \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode prefill \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Decode Attn 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 4 \
    --disable-overlap-schedule --disable-cuda-graph \
    --disaggregation-mode decode \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Decode FFN 节点 ═══
python -m sglang.launch_server \
    --model-path <模型> --tp 8 \
    --disable-overlap-schedule --disable-cuda-graph \
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

---

## 六、架构限制与待办

| 编号 | 描述 | 状态 |
|------|------|------|
| F1 | StepMesh N:M 双向分片通信（详见下方） | **已实现** |
| F2 | 需要 `--disable-cuda-graph --disable-overlap-schedule` | 待实现 |
| F3 | TBO 与 AFD 互斥，不能同时启用 | 设计决策（可后续扩展） |
| F4 | 自动 profiling + `afd_attn_ratio` 自适应调参 | 待实现 |
| F6 | DeepSeek-V2 `load_weights` 添加 AFDWeightFilter（generator 包装过滤） | **已修复** |
| F7 | `afd_prepare_overlap` 添加 `is_target_verify()` 处理（scheduler.py + scheduler_afd_mixin.py） | **已修复** |
| F8 | `afd_overlap.py` 移除冗余 `m is None` 检查 | **已修复** |
| F9 | GCD-based 分组 StepMesh（`--afd-grouped-stepmesh`），跨节点流量降至 NH×(TP_A+TP_F)/gcd | **已实现** |
| F5 | Qwen2（不使用 LayerCommunicator 的旧 Dense 模型）不支持 AFD | 需重构 Qwen2DecoderLayer |

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

## 十、总结：未完成任务与影响评估

### 已完成任务一览

| 编号 | 描述 | 完成轮次 |
|------|------|---------|
| F1 | StepMesh N:M 双向分片通信（任意 TP_A:TP_F） | 第一轮 |
| F6 | DeepSeek-V2 `load_weights` 添加 `AFDWeightFilter` | 第一轮 |
| F7 | `afd_prepare_overlap` 添加 `is_target_verify()` 处理 | 第一轮 |
| F8 | `afd_overlap.py` 移除冗余 `m is None` 检查 | 第一轮 |
| G3 | RDMA 末尾 shard padding + 截断 | 第一轮 |
| H1-H3 | StepMesh key 碰撞 / docstring / recv buffer 注册 | 第一轮 |
| F9 | GCD-based 分组 StepMesh（`--afd-grouped-stepmesh`） | 本轮 |
| Phase 6 | 异构 TP（`ShardedParallelCommunicator` + StepMesh N:M） | 早期 |
| Task 6 | PD + AFD 独立控制 | 早期 |
| C1-C5, S1-S3, G3-G6, E1-E5 | 通信/调度/计算/工程化优化 | 早期 |

### 未完成任务总表

| 优先级 | 编号 | 描述 | 影响评估 | 复杂度 |
|--------|------|------|---------|--------|
| **P1** | **F2** | **CUDA Graph + overlap schedule 支持** | **严重**：当前必须 `--disable-cuda-graph --disable-overlap-schedule`，kernel launch overhead 导致 **20-40% 性能损失**，是 AFD 上生产的最大阻碍 | 高 |
| **P2** | **F3** | **TBO + AFD 共存** | **中等**：TBO（Two-Batch Overlap）通过批次间 overlap 提升吞吐。AFD 与 TBO 互斥，导致无法利用 TBO 的 **10-20% 吞吐提升** | 高 |
| P2 | F5 | Qwen2（Dense）AFD 支持 | **低**：Qwen2 DecoderLayer 不使用 `LayerCommunicator`，需要重构。Qwen2 为旧模型，生产中多用 Qwen3/MoE，影响面小 | 中 |
| P3 | F4 | 自动 profiling + `afd_attn_ratio` 自适应 | **低**：手动设置 `--afd-attn-ratio` 可满足需求，自动调参为便利性优化，不影响正确性和峰值性能 | 中 |
| P3 | G6 | 异构路径 buffer 复用 | **极低**：PyTorch caching allocator 已处理重复分配，性能影响 <1%。主要改善长期运行的内存碎片化 | 低 |
| P4 | F9-opt-1 | 分组 StepMesh: custom process sub-groups | **低**：消除 stride>1 时冗余 NVLink 传输，性能影响 1-5%。仅混合组（如 6A:4F）时有意义 | 中 |
| P4 | F9-opt-2 | 分组 StepMesh: grouped buffer pool | **极低**：性能影响 <1%，改善内存友好度 | 低 |
| P4 | F9-opt-3 | 分组 StepMesh: 避免 padding tokens 经过 MLP | **极低**：典型 batch（>100 tokens）下影响 <1%，仅极小 batch（<20）时有意义 | 低 |

### 影响分级说明

```
严重（>20% 性能）: F2
中等（10-20%）:     F3
低（1-5%）:         F5, F4, F9-opt-1
极低（<1%）:        G6, F9-opt-2, F9-opt-3
```

### 建议执行路径

```
阶段 1（上生产必须）:
  F2 — CUDA Graph 支持 → 消除 20-40% kernel launch overhead

阶段 2（提升吞吐）:
  F3 — TBO + AFD 共存 → 解锁 batch overlap 的 10-20% 吞吐提升

阶段 3（扩展模型 & 便利性）:
  F5 — Qwen2 Dense AFD（按需求优先级）
  F4 — 自动 profiling（便利性）

阶段 4（锦上添花，可长期推进）:
  G6, F9-opt-1/2/3 — 性能 <5%，非关键路径
```

### 当前 AFD 功能完整性

| 维度 | 状态 | 说明 |
|------|------|------|
| 通信层 | **完整** | ZMQ（同构+异构）+ StepMesh（同构+异构+分组），支持任意 N:M |
| 调度层 | **完整** | microbatch 切分、batch 对齐、PD 独立控制 |
| 计算层 | **完整** | 流水线 overlap、非对称切分、预分配 output |
| 模型支持 | **基本完整** | Qwen3-MoE/Qwen2-MoE/DeepSeek-V2/V3/Qwen3(Dense)，缺 Qwen2(Dense) |
| 工程化 | **完整** | Mixin 抽象、权重过滤、独立文件、超时处理 |
| **性能** | **受限** | 核心功能正确，但 F2（CUDA Graph）未解决前性能损失 20-40% |
