# SGLang AFD (Attention-FFN Disaggregation) Migration and Optimization Report

## I. Overview

### What Is AFD?

AFD splits each Transformer layer into Attention (A) and FFN (F), deployed on different GPU nodes. Each layer requires two cross-node transfers (A->F and F->A); a microbatch pipeline hides communication latency.

### Core Benefits

- For MoE models, Attn nodes do not load Expert weights, increasing KV Cache capacity by approximately 4x
- Higher concurrency → higher throughput + lower P99 latency

### Migration Source and Target

- **Source**: `sglang_af/sglang_v1` (AFD PoC implementation)
- **Target**: `sglang` (newer version, originally without AFD code)

---

## II. File Inventory

### New Files (7)

| File Path | Purpose |
|----------|------|
| `python/sglang/srt/layers/afd_type.py` | `AFDPerspective` enum and micro_batch argument parsing |
| `python/sglang/srt/layers/afd.py` | Core logic: communicators (ZMQ/StepMesh/sharded parallel), pipeline scheduling, `model_forward_afd`, `AFDCommunicator`, `AsyncTensorCommunicator`, Proxy modules |
| `python/sglang/srt/layers/afd_mixin.py` | Generic Mixin (`AFDDecoderLayerMixin`) and weight filter (`AFDWeightFilter`) |
| `python/sglang/srt/batch_overlap/afd_overlap.py` | `AfdForwardBatchPreparer`—microbatch splitting and sub-batch assembly |
| `python/sglang/srt/managers/scheduler_afd_mixin.py` | Mixin with reusable AFD scheduling helpers |
| `test/srt/test_afd_fixture.py` | Test fixture (dual-process startup) |
| `test/srt/test_afd_basic.py` | End-to-end test cases |

### Modified Files (13)

| File Path | Changes |
|----------|----------|
| `server_args.py` | Added `afd_perspective`, `afd_micro_batch`, `afd_attn_ratio`, `afd_attn_tp`, `afd_ffn_tp` + `_handle_afd()` validation |
| `io_struct.py` | Added `AFDReqInput` (inherits `BaseReq`, includes `req_ids`/`seq_lens`/`extend_lens`) |
| `schedule_batch.py` | Added `afd_split_seq_index: Optional[List[int]]` to `ScheduleBatch` + `ModelWorkerBatch` |
| `forward_batch_info.py` | AFD fields + direct forwarding of `afd_split_seq_index` in `init_new` |
| `model_runner.py` | Added `AfdAttnBackend` branch in `init_attention_backend` |
| `tbo_backend.py` | Added `AfdAttnBackend` class |
| `qwen3_moe.py` | Added `_afd_init`/`forward_afd_A`/`forward_afd_F` to DecoderLayer; added `AFDWeightFilter` to `load_weights` |
| `qwen2_moe.py` | Same as above + AFD branch in Model.forward + `AFDWeightFilter` in `load_weights` |
| `qwen3.py` | Added AFD to DecoderLayer; rewrote Qwen3Model forward with an AFD branch |
| `deepseek_v2.py` | Added AFD to DecoderLayer; added an AFD branch to Model.forward |
| `scheduler.py` | AFD ZMQ channel + `event_loop_afd` + PREFILL/DECODE AFD branches in `dispatch_event_loop` |
| `disaggregation/prefill.py` | Added `event_loop_afd_disagg_prefill` |
| `disaggregation/decode.py` | Added `event_loop_afd_disagg_decode` |

---

## III. Applied Optimizations

### Communication Layer

| ID | Content | Location |
|------|------|------|
| C1 | ZMQ zero-copy transfer (metadata + raw bytes) | `afd.py` ZMQSimpleTensorCommunicator |
| C2 | Dedicated CUDA stream + `AsyncTensorCommunicator` + cross-stage `postprocess_layer_start_recv` | `afd.py` |
| C3 | Replaced `list` with `deque`, removed `zero_()`, dynamic buffer pool | `afd.py` StepMeshTensorCommunicator |
| C5 | Ports configurable through environment variables | `afd.py` + `scheduler.py` |

### Scheduling Layer

| ID | Content | Location |
|------|------|------|
| S1 | `AFDReqInput` carries `req_ids` for precise batch alignment | `io_struct.py` + `scheduler_afd_mixin.py` |
| S2 | Replaced busy-wait with `zmq.Poller` | `scheduler.py` event_loop_afd |
| S3 | `AFDReqInput` includes `seq_lens`/`extend_lens` | `io_struct.py` |

### Compute Layer

| ID | Content | Location |
|------|------|------|
| G3 | Token-count-aware greedy splitting | `afd_overlap.py` |
| G5 | Preallocated output tensor, eliminating `torch.cat` | `afd.py` model_forward_afd |
| G6 | Replaced `dict` with `NamedTuple` (`StageIO`) | `afd.py` |
| G4 | `--afd-attn-ratio` asymmetric microbatch argument | `server_args.py` |

### Engineering

| ID | Content | Location |
|------|------|------|
| E1 | Generic Mixin + `AFDWeightFilter` | `afd_mixin.py` |
| E2 | Separate file for AFD code | `batch_overlap/afd_overlap.py` |
| E3 | Generic m-way splitting (loop implementation) | `afd_overlap.py` |
| E5 | ZMQ timeouts + exception handling | `afd.py` |

### Heterogeneous TP

| ID | Content | Location |
|------|------|------|
| Phase 6 | `--afd-attn-tp`/`--afd-ffn-tp` + `ShardedParallelCommunicator` (sharded parallelism + metadata transfer + padding truncation) | `server_args.py` + `afd.py` |
| F9 | `--afd-grouped-stepmesh` + GCD-based grouped StepMesh (per-group DMLC + NVLink all_gather + stride deduplication) | `server_args.py` + `afd.py` |
| F2-A | `torch.compile(dynamic=True)` compiles `_run_attn` / `_run_mlp` + `@torch.compiler.disable()` on AFD communication methods | `afd_mixin.py` + `afd.py` |
| F3 | `--afd-enable-overlap-schedule` CPU/GPU overlap scheduling (`result_queue` + async `run_batch` in `event_loop_afd`) | `server_args.py` + `scheduler.py` |

### Independent PD+AFD Control

| ID | Content | Location |
|------|------|------|
| Task 6 | Independent `--afd-perspective` control for P/D; reusable scheduling methods in `SchedulerAFDMixin`; both PREFILL/DECODE branches support AFD | `scheduler_afd_mixin.py` + `prefill.py` + `decode.py` + `scheduler.py` |

---

## IV. Supported Models

| Model | Type | AFD Status | Notes |
|------|------|----------|------|
| Qwen3-MoE | MoE | Integrated | Mixin + weight filtering |
| Qwen2-MoE | MoE | Integrated | Mixin + weight filtering + Model.forward AFD branch |
| DeepSeek-V2/V3 | MoE | Integrated | Mixin + Model.forward AFD branch |
| Qwen3 (Dense) | Dense | Integrated | Mixin + rewritten Qwen3Model forward |
| LLaMA / Llama3 | Dense | Integrated | LayerCommunicator created in init + Mixin + AFDWeightFilter |
| Other models using LayerCommunicator | Various | Can be adapted quickly | Add `_afd_init()` + `forward_afd_A/F` to DecoderLayer |
| Qwen2 (Dense) | Dense | Adaptable | Same approach as LLaMA: create LayerCommunicator in init as needed |

---

## V. Running AFD

> The examples below use `<model>` for the model path, `<attn_ip>` / `<ffn_ip>` for node IP addresses, and `<RDMA_nic>` for the RDMA NIC name (e.g., `mlx5_0`).
> AFD automatically disables overlap scheduling and CUDA graph (no need to pass `--disable-overlap-schedule` or `--disable-cuda-graph` manually).
> Optional flags: `--enable-torch-compile` (kernel fusion ~5-8%), `--afd-enable-overlap-schedule` (CPU/GPU overlap ~5-10%).

### 1. ZMQ Homogeneous TP (Quick Test, No RDMA)

```bash
# ═══ Attn node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3

# ═══ FFN node(TP=4)═══
export CUDA_VISIBLE_DEVICES=4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3
```

### 2. ZMQ Heterogeneous TP (Quick Test, No RDMA, 4A:8F, Cross-Node 2NH)

> The ZMQ path supports heterogeneous TP through the `ShardedParallelCommunicator` wrapper: each local rank sends only its 1/TP_local shard (1:1 point-to-point), and NVLink all_gather on the receiving side reconstructs the full tensor. Cross-node traffic is fixed at **2NH** (independent of the TP ratio), but is limited by TCP latency.

```bash
# ═══ Attn node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8

# ═══ FFN node(TP=8)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8
```

### 3. ZMQ Heterogeneous TP (Quick Test, No RDMA, 8A:4F, Cross-Node 2NH)

```bash
# ═══ Attn node(TP=8)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4

# ═══ FFN node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3
export AFD_SCHED_HOST=<ffn_ip>

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4
```

### 4. StepMesh Homogeneous TP (High-Performance RDMA, TP_A == TP_F)

```bash
# ═══ Shared environment variables (set on both nodes)═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3

# ═══ FFN node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3
```

### 5. StepMesh Heterogeneous TP — Ungrouped (4A:8F, Cross-Node 12NH)

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn node(TP=4, inform the peer that FFN TP=8)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8

# ═══ FFN node(TP=8, inform the peer that Attn TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8
```

### 6. StepMesh Heterogeneous TP — Ungrouped (8A:4F, Cross-Node 12NH)

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn node(TP=8)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4

# ═══ FFN node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4
```

### 7. StepMesh Heterogeneous TP — Grouped (4A:8F, Cross-Node 3NH, 4 Groups of 1W:2S)

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ FFN node(TP=8)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh
```

### 8. StepMesh Heterogeneous TP — Grouped (8A:4F, Cross-Node 3NH, 4 Groups of 2W:1S)

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn node(TP=8)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4 \
    --afd-grouped-stepmesh

# ═══ FFN node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 8 --afd-ffn-tp 4 \
    --afd-grouped-stepmesh
```

### 9. StepMesh Heterogeneous TP — Grouped + Asymmetric Splitting (6A:4F, Cross-Node 5NH, 2 Groups of 3W:2S)

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn node(TP=6)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5

python -m sglang.launch_server \
    --model-path <model> --tp 6 \
    --disable-overlap-schedule \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 6 --afd-ffn-tp 4 --afd-attn-ratio 0.6 \
    --afd-grouped-stepmesh

# ═══ FFN node(TP=4)═══
export CUDA_VISIBLE_DEVICES=0,1,2,3

python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 6 --afd-ffn-tp 4 --afd-attn-ratio 0.6 \
    --afd-grouped-stepmesh
```

### 10. PD Disaggregation + AFD + Grouped StepMesh (4A:8F)

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Prefill Attn node ═══
python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --disaggregation-mode prefill \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Prefill FFN node ═══
python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --disable-overlap-schedule \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode prefill \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Decode Attn node ═══
python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disable-overlap-schedule \
    --disaggregation-mode decode \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ Decode FFN node ═══
python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --disable-overlap-schedule \
    --port 30002 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode decode \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh
```

### Configuration Quick Reference

| # | Scenario | Communication | Key Attn-Side Arguments | Key FFN-Side Arguments | Cross-Node Traffic |
|---|------|---------|-----------------|----------------|-----------|
| 1 | Homogeneous 4:4 | ZMQ (TCP) | `--afd-perspective attn` | `--afd-perspective ffn` | 2NH |
| 2 | Heterogeneous 4:8 | ZMQ (TCP) | `--afd-attn-tp 4 --afd-ffn-tp 8` | Same as left | 2NH |
| 3 | Heterogeneous 8:4 | ZMQ (TCP) | `--afd-attn-tp 8 --afd-ffn-tp 4` | Same as left | 2NH |
| 4 | Homogeneous 4:4 | StepMesh (RDMA) | Same as #1 + `MLC_INTERFACE` | Same as left | 2NH |
| 5 | Heterogeneous 4:8 ungrouped | StepMesh (RDMA) | `--afd-attn-tp 4 --afd-ffn-tp 8` | Same as left | 12NH |
| 6 | Heterogeneous 8:4 ungrouped | StepMesh (RDMA) | `--afd-attn-tp 8 --afd-ffn-tp 4` | Same as left | 12NH |
| 7 | Heterogeneous 4:8 **grouped** | StepMesh (RDMA) | Same as #5 + `--afd-grouped-stepmesh` | Same as left | **3NH** |
| 8 | Heterogeneous 8:4 **grouped** | StepMesh (RDMA) | Same as #6 + `--afd-grouped-stepmesh` | Same as left | **3NH** |
| 9 | Heterogeneous 6:4 **grouped** | StepMesh (RDMA) | `--afd-attn-tp 6 --afd-ffn-tp 4 --afd-grouped-stepmesh` | Same as left | **5NH** |
| 10 | PD + heterogeneous 4:8 **grouped** | StepMesh (RDMA) | Same as #7 + `--disaggregation-mode prefill/decode` | Same as left | **3NH** |

> **Notes**:
> - `--afd-attn-tp` and `--afd-ffn-tp` must be set **on both the Attn and FFN sides**, with identical values. The same applies to `--afd-grouped-stepmesh`.
> - ZMQ heterogeneous TP (#2, #3) wraps 1:1 ZMQ communication with `ShardedParallelCommunicator`, fixing cross-node traffic at 2NH (independent of the TP ratio), but is limited by TCP latency and suitable for functional validation.
> - Ungrouped heterogeneous StepMesh (#5, #6) has high cross-node traffic (12NH), but low RDMA latency; adding `--afd-grouped-stepmesh` (#7-#10) reduces traffic to 3-5NH.
> - `--afd-grouped-stepmesh` only applies to StepMesh (requires `MLC_INTERFACE`); the ZMQ path ignores it automatically.

### Feature Flag Quick Reference

| Module | Flag | Default | Effect | Dependencies |
|------|------|------|------|------|
| Basic AFD | `--afd-perspective attn/ffn` | Off | Separate Attn-FFN deployment | Automatically disables CUDA graph |
| Microbatch | `--afd-micro-batch N` | 3 | Pipeline hides communication latency | Basic AFD |
| Asymmetric splitting | `--afd-attn-ratio R` | 0.5 | Attn/FFN compute ratio | Basic AFD |
| Heterogeneous TP | `--afd-attn-tp A --afd-ffn-tp F` | Same as `--tp` | Different TP sizes | Basic AFD |
| Grouped StepMesh | `--afd-grouped-stepmesh` | Off | Reduces cross-node traffic to NH×(A+F)/gcd | Heterogeneous TP + `MLC_INTERFACE` |
| Overlap scheduling | `--afd-enable-overlap-schedule` | Off | CPU/GPU parallel scheduling ~5-10% | Basic AFD |
| Kernel Fusion | `--enable-torch-compile` | Off | torch.compile ~5-8% | Basic AFD |

### Launch Commands by Feature Combination

#### A. Minimal AFD (ZMQ, Functional Validation)

```bash
# ═══ Attn node ═══
python -m sglang.launch_server \
    --model-path <model> --tp 4 \
        --afd-perspective attn --afd-micro-batch 3

# ═══ FFN node ═══
python -m sglang.launch_server \
    --model-path <model> --tp 4 \
        --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3
```

#### B. StepMesh RDMA + Heterogeneous TP + Grouping (Recommended Production Baseline)

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn node(TP=4)═══
python -m sglang.launch_server \
    --model-path <model> --tp 4 \
        --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh

# ═══ FFN node(TP=8)═══
python -m sglang.launch_server \
    --model-path <model> --tp 8 \
        --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh
```

#### C. All Optimizations Enabled (Maximum Performance)

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Attn node(TP=4)═══
python -m sglang.launch_server \
    --model-path <model> --tp 4 \
        --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh \
    --afd-enable-overlap-schedule \
    --enable-torch-compile

# ═══ FFN node(TP=8)═══
python -m sglang.launch_server \
    --model-path <model> --tp 8 \
        --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh \
    --afd-enable-overlap-schedule \
    --enable-torch-compile
```

#### D. PD Disaggregation + All AFD Optimizations

```bash
# ═══ Shared environment variables ═══
export MLC_INTERFACE=<RDMA_nic>
export DMLC_PS_ROOT_URI=<attn_ip>

# ═══ Prefill Attn ═══
python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disaggregation-mode prefill \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh --afd-enable-overlap-schedule --enable-torch-compile

# ═══ Prefill FFN ═══
python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --port 30001 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode prefill \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh --afd-enable-overlap-schedule --enable-torch-compile

# ═══ Decode Attn ═══
python -m sglang.launch_server \
    --model-path <model> --tp 4 \
    --disaggregation-mode decode \
    --afd-perspective attn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh --afd-enable-overlap-schedule --enable-torch-compile

# ═══ Decode FFN ═══
python -m sglang.launch_server \
    --model-path <model> --tp 8 \
    --port 30002 --skip-server-warmup --watchdog-timeout 3600 \
    --disaggregation-mode decode \
    --afd-perspective ffn --afd-micro-batch 3 \
    --afd-attn-tp 4 --afd-ffn-tp 8 \
    --afd-grouped-stepmesh --afd-enable-overlap-schedule --enable-torch-compile
```

#### E. Recommended Performance Optimization Combinations

| Scenario | Command Combination | Expected Improvement (vs Basic AFD) |
|------|---------|----------------------|
| Functional validation | A (minimal ZMQ) | Baseline |
| Production baseline | B (StepMesh + grouping) | 4x reduction in cross-node traffic |
| Add overlap scheduling | B + `--afd-enable-overlap-schedule` | +5-10% throughput |
| Add kernel fusion | B + `--enable-torch-compile` | +5-8% compute |
| **All enabled** | **C (B + overlap + compile)** | **+10-18% overall** |
| PD disaggregation + all enabled | D | Same as C, with independent PD support |

---

## VI. Architectural Limitations and TODOs

| ID | Description | Status |
|------|------|------|
| F1 | StepMesh N:M bidirectional sharded communication (see below) | **Implemented** |
| F2 | CUDA graph automatically disabled (`_handle_afd`); Phase A (`torch.compile` per-stage compilation) implemented, Phase B (`reduce-overhead` CUDA Graph) not yet implemented | **Phase A implemented** |
| F3 | ~~TBO and AFD mutually exclusive~~ → `--afd-enable-overlap-schedule` CPU/GPU overlap scheduling (reuses the `forward_stream` + `result_queue` pattern in `event_loop_afd`) | **Implemented** |
| F4 | Automatic profiling + adaptive `afd_attn_ratio` tuning | Not yet implemented |
| F6 | Add AFDWeightFilter to DeepSeek-V2 `load_weights` (generator-wrapper filtering) | **Fixed** |
| F7 | Add `is_target_verify()` handling to `afd_prepare_overlap` (scheduler.py + scheduler_afd_mixin.py) | **Fixed** |
| F8 | Remove redundant `m is None` check in `afd_overlap.py` | **Fixed** |
| F9 | GCD-based grouped StepMesh (`--afd-grouped-stepmesh`), reducing cross-node traffic to NH×(TP_A+TP_F)/gcd | **Implemented** |
| F5 | Qwen2 (Dense) AFD support—can use the same adaptation as LLaMA (create LayerCommunicator in init) | Can be adapted quickly |

---

## VII. F1 StepMesh N:M Bidirectional Sharded Communication

### Design Principles

Inspection of the StepMesh source (`af_tensor_app.h` + `public.hpp`) revealed:

1. **`push_pull` supports multiple pull_tensor objects**: `pull_tensors` array size = `server_count × batch_size`; each Server has an independent pull buffer, preventing overwrites
2. **`get_batch` returns results sorted by Worker rank**: internally uses the `q_[worker_rank]` queue; return order matches rank order
3. **`respond` matches by key**: Server response data is written to the Worker's pull buffer for the corresponding key
4. **`RegisterRecvTensor` supports per-Worker shard registration**: different chunks of the same tensor are registered for different Workers

### Implementation

**A→F Direction (Sharded Broadcast)**:
- `attn_send`: each Attn rank pushes only its 1/TP_A token shard (broadcast to all FFN ranks)
- Registers **TP_F pull_tensor objects** concurrently (one independent buffer per FFN Server to receive F→A shards)
- `ffn_recv`: `get_batch()` returns TP_A shards (sorted by Worker rank), concatenated with `torch.cat`

**F→A Direction (Sharded Respond)**:
- `ffn_send`: each FFN rank responds with its own 1/TP_F shard to every Worker (key matching writes into the corresponding Worker pull buffer)
- `attn_recv`: after `wait`, all TP_F pull_tensor objects are filled; `torch.cat` + truncate padding

### Cross-Node Traffic (4A:8F Example)

| Direction | Traffic |
|------|------|
| A→F | 4 × (NH/4) × 8 = 8NH (each Attn sends 1/4, broadcasting 8 copies) |
| F→A | 8 × (NH/8) × 4 = 4NH (each FFN sends 1/8, responding to 4 Workers) |
| **Total** | **12NH** (vs 2NH for homogeneous 1:1, vs 64NH for full broadcast) |

### Backward Compatibility

For homogeneous TP (`attn_tp == ffn_tp`), `_heterogeneous=False`, and all methods use the original 1:1 code path.

### Issues Fixed in This Review Round

| ID | Issue | Fix |
|------|------|------|
| H1 | In `attn_send`, `self.key += 2` reserves only 2 keys, but pull needs TP_F keys, causing a key collision on the second call | Changed to `self.key += 1 + self.ffn_tp` |
| H2 | Docstring described the old "selective respond + all_gather" scheme | Updated to the multiple-pull_tensor scheme |
| H3 | `ffn_recv` registered a recv buffer only for the first Worker's key | Changed to register one for every Worker's key |

### Remaining Items

| ID | Description |
|------|------|
| G3 | When the token count is not divisible by TP, the last shard is shorter; StepMesh RDMA writes into a larger pull_buf, leaving uninitialized tail data. `full[:original_num_tokens]` in `attn_recv` already truncates it, but whether the RDMA layer requires response data size == pull_buf size needs practical validation |
| G6 | The heterogeneous `attn_send` path does not reuse buffers (`torch.empty_like` each time); the homogeneous path uses `StepMeshTensorCache` + a free pool. The heterogeneous path can be optimized, but correctness is unaffected |

### F→A Traffic Optimization Analysis

Current F→A traffic is 4NH (4A:8F example), because each FFN Server must respond to every Worker (StepMesh `push_pull` requires all pull requests to receive responses; otherwise the Worker hangs).

**Further optimization proposal**: each Worker registers only 1 pull_tensor (fetching one 1/TP_F shard from only 1 Server); Attn-side NVLink all_gather reconstructs the full tensor. Cross-node F→A traffic drops to NH at the cost of an additional intra-node all_gather (NVLink ~600GB/s, latency 5-10us).

However, StepMesh imposes a limitation: `ZPull_` internally computes `pull_batch_size` as `pull_tensors.size() / server_count`; if the pull_tensor count < server_count, then `pull_batch_size=0` and no pull is issued. Therefore, **Workers must pass an integer multiple of server_count pull_tensor objects**.

A feasible compromise: pass TP_F pull_tensor objects but have only the required Servers respond with real data; the others respond with empty placeholders (avoiding hangs). Attn then takes only the required shards + all_gather. This is equivalent to the current implementation + an all_gather without reducing cross-node traffic (still 4NH), so it offers no practical benefit.

**Conclusion**: 4NH is the inherent lower bound of the StepMesh push_pull API (each Server must respond to every Worker's pull). Reducing it to NH requires grouped StepMesh instances (extremely high complexity), recommended as a long-term goal. In actual deployments with RDMA 200Gbps+, the absolute latency of 4NH still falls within the SLA.

---

## VIII. Final Review (Current Round)

### Issues Found

| ID | Severity | File | Issue |
|------|--------|------|------|
| F6 | **Critical** | `deepseek_v2.py` | `load_weights` lacks `AFDWeightFilter`; both Attn/FFN nodes load all weights, wasting GPU memory |
| F7 | **Medium** | `scheduler.py` + `scheduler_afd_mixin.py` | `afd_prepare_overlap` does not handle `TARGET_VERIFY` mode; AFD microbatch splitting fails under speculative decoding |
| F8 | **Low** | `afd_overlap.py` | Redundant `m is None` check (`_get_afd_micro_batch()` always returns int) |

### Verified Correct Components

| Check | Status |
|--------|------|
| `afd_split_seq_index` field propagation (ScheduleBatch → ModelWorkerBatch → ForwardBatch) | OK |
| StepMesh `attn_send` key allocation (`self.key += 1 + self.ffn_tp`) | OK |
| StepMesh `ffn_send` responds n_workers times | OK |
| StepMesh heterogeneous path (sharded broadcast + multiple pull_tensor objects) | OK |
| `model_forward_afd` pipeline overlap (`postprocess_layer_start_recv`) | OK |
| Qwen3Model.forward AFD branch | OK |
| Qwen2.py contains no AFD (Qwen2DecoderLayer has no LayerCommunicator, correctly excluded) | OK |
| AFDReqInput inherits BaseReq | OK |
| server_args `afd_micro_batch` spelling | OK |
| Order of Mixin method calls in `event_loop_afd_disagg_prefill/decode` | OK |
| `_compute_token_indices_m_way` computes independently | OK |

### Follow-up Task Priorities

| Priority | Task | Description | Status |
|--------|------|------|------|
| ~~P0~~ | ~~F6~~ | ~~Add AFDWeightFilter to DeepSeek-V2~~ | **Fixed** |
| ~~P0~~ | ~~F7~~ | ~~Add `is_target_verify()` handling to `afd_prepare_overlap`~~ | **Fixed** |
| ~~P1~~ | ~~F8~~ | ~~Remove redundant `m is None` check~~ | **Fixed** |
| P2 | F2 | CUDA Graph support | Not yet implemented |
| P2 | F3 | TBO + AFD coexistence | Not yet implemented |
| P3 | F4 | Automatic profiling | Not yet implemented |
| ~~P2~~ | ~~F9~~ | ~~Grouped StepMesh: GCD-based grouping, cross-node traffic NH×(TP_A+TP_F)/gcd~~ | **Implemented** |
| ~~P3~~ | ~~G3~~ | ~~RDMA final shard shorter than pull_buf~~ — pad the final shard to `chunk_size` in `ffn_send`, truncate with `full[:original_num_tokens]` in `attn_recv` | **Fixed** |
| P3 | G6 | Heterogeneous-path buffer reuse | To be optimized |

---

## IX. F9 Grouped StepMesh Implementation (Generic GCD-Based Grouping)

### Design

Use `num_groups = gcd(TP_A, TP_F)` to split global N:M StepMesh into independent small groups, **supporting both TP_A > TP_F and TP_A < TP_F**:

```
num_groups        = gcd(TP_A, TP_F)
workers_per_group = TP_A / num_groups
servers_per_group = TP_F / num_groups
```

- Added optional `--afd-grouped-stepmesh` flag (default False)
- Independent StepMesh instance per group (separate port + scheduler process)
- Reuse existing heterogeneous N:M communication code within groups, then reconstruct the full tensor with NVLink `all_gather` + stride deduplication
- Automatically disabled when `gcd=1` (e.g., 3A:2F), since there is no benefit
- Works when both sides exceed 1 (e.g., 6A:4F → 3W:2S groups), but benefits are smaller and a warning is emitted

### General Traffic Formula

```
A→F = NH × (TP_F / gcd)
F→A = NH × (TP_A / gcd)
Total = NH × (TP_A + TP_F) / gcd
```

| TP_A:TP_F | gcd | Per Group | Ungrouped | Grouped | Improvement |
|-----------|-----|------|--------|--------|------|
| 4:8 | 4 | 1W:2S | 12NH | 3NH | 4x |
| 8:4 | 4 | 2W:1S | 12NH | 3NH | 4x |
| 6:4 | 2 | 3W:2S | 10NH | 5NH | 2x |
| 4:6 | 2 | 2W:3S | 10NH | 5NH | 2x |
| 8:2 | 2 | 4W:1S | 10NH | 5NH | 2x |
| 3:2 | 1 | — | 6NH | 6NH | None |

### Implementation Steps (8 Steps, Only `afd.py` + `server_args.py` Modified)

1. **`server_args.py`**: add `afd_grouped_stepmesh` field + `--afd-grouped-stepmesh` CLI + `_handle_afd` validation
2. **`__init__` group mapping**: compute `num_groups`, `workers_per_group`, `servers_per_group`, `group_id`, `intra_group_rank`
3. **`_start_stepmesh_scheduler`**: per-group DMLC configuration (`NUM_WORKER=workers_per_group`, `NUM_SERVER=servers_per_group`, independent ports); first Worker in each group starts the scheduler
4. **`attn_send`**: pad push shards to `chunk_size` (ensuring equal all_gather sizes), `n_pull = servers_per_group`
5. **`ffn_recv`**: receive `workers_per_group` shards + `all_gather` on TP_F group + stride `servers_per_group` deduplication
6. **`ffn_send`**: no changes needed (`n_workers` is automatically correct, shards computed from global TP_F)
7. **`attn_recv`**: concatenate `servers_per_group` pull_tensor objects + `all_gather` on TP_A group + stride `workers_per_group` deduplication + truncate to `original_num_tokens`
8. **Update documentation**

### Usage

See **Section V** (scenarios 5-8) for detailed launch commands. Key points:

- Add `--afd-grouped-stepmesh` on both sides to existing heterogeneous TP commands
- Requires the `MLC_INTERFACE` environment variable (StepMesh RDMA mode)
- `--afd-attn-tp` and `--afd-ffn-tp` must match on both sides
- Benefits apply only when `gcd(TP_A, TP_F) > 1` (automatically disabled with a warning when `gcd=1`)

### F9 Review Results

#### Issues Found

| ID | Severity | Location | Issue | Impact |
|------|--------|------|------|------|
| F9-1 | **Low** | `ffn_recv` grouped path | FFN processes padded tensors (at most TP_A-1 extra zero tokens through MLP) | <0.7% compute overhead for batch=1000 tokens, negligible |
| F9-2 | **Low** | `_start_stepmesh_scheduler` | Per-group scheduler flag is technically redundant (each process calls init only once) | Defensive programming, no correctness impact |
| F9-3 | **Low** | `_start_stepmesh_scheduler` | Grouped path force-sets `DMLC_PS_ROOT_PORT`, slightly different from `_env_def` semantics in the ungrouped path | Correct behavior, style difference only |

#### Verified Correct Components

| Check | Status |
|--------|------|
| GCD grouping: 4A:8F → 4 groups of 1W:2S | OK |
| GCD grouping: 8A:4F → 4 groups of 2W:1S | OK |
| GCD grouping: 6A:4F → 2 groups of 3W:2S (mixed groups) | OK |
| GCD grouping: 3A:2F → gcd=1 → automatically disabled | OK |
| GCD grouping: 4A:4F → homogeneous → automatically disabled | OK |
| `attn_send` push shard padding applies only in `_grouped` mode, leaving the ungrouped path unaffected | OK |
| `attn_send` pull buffer size uses `effective_tokens` (padded), matching `ffn_send` shard size | OK |
| `attn_send` key increment `1 + servers_per_group` (grouped) vs `1 + ffn_tp` (ungrouped) | OK |
| `ffn_recv` all_gather size=`self.ffn_tp` == `tp_group.size()` | OK |
| `ffn_recv` stride-select correctly skips intra-group duplicate data | OK |
| `ffn_send` needs no changes: `n_workers` is automatically correct, shards use global `ffn_tp` | OK |
| `attn_recv` all_gather size=`self.attn_tp` == `tp_group.size()` | OK |
| `attn_recv` stride-select + `full[:original_num_tokens]` truncation | OK |
| Per-group DMLC configuration (independent NUM_WORKER/NUM_SERVER/PORT) | OK |
| Per-group scheduler startup (`intra_group_rank == 0` condition) | OK |
| `server_args.py` validation (disable for homogeneous TP, disable for gcd=1, warn for mixed groups) | OK |
| Backward compatibility: all code paths match the original implementation when `_grouped=False` | OK |
| Correct data flow for the `num_tokens=1` edge case | OK |

#### Per-Configuration Data-Flow Validation Summary

**4A:8F** (`G=4, 1W:2S`): `chunk_a=4, eff=16, f2a=2` → `ffn_recv` stride=2 → 16 tokens → MLP → `ffn_send` 2/rank → `attn_recv` stride=1 → `[:15]` ✓

**8A:4F** (`G=4, 2W:1S`): `chunk_a=1, eff=8, f2a=2` → `ffn_recv` stride=1 → 8 tokens → MLP → `ffn_send` 2/rank → `attn_recv` stride=2 → `[:7]` ✓

**6A:4F** (`G=2, 3W:2S`): `chunk_a=2, eff=12, f2a=3` → `ffn_recv` stride=2 → 12 tokens → MLP → `ffn_send` 3/rank → `attn_recv` stride=3 → `[:10]` ✓

### Further Optimization Directions

- **Custom process sub-groups**: replace full-TP all_gather + stride deduplication with `num_groups`-way all_gather, eliminating redundant NVLink transfers (when stride > 1)
- **Grouped buffer pool**: extend the `StepMeshTensorCache` free pool to the grouped path (currently only the homogeneous path reuses buffers)
- **F9-1 optimization**: reconstruct `original_num_tokens` in `ffn_recv` (via embedded metadata or broadcast), avoiding MLP execution on padding tokens

---

## X. Summary: Code Review Results and Outstanding Tasks

### Code Review This Round (All Modified Files)

#### Review Scope

| File | Changes |
|------|---------|
| `server_args.py` | `afd_grouped_stepmesh` + `afd_enable_overlap_schedule` fields/CLI/validation + AFD automatically disables overlap |
| `afd.py` — `StepMeshTensorCommunicator` | GCD grouping in `__init__` + per-group `_start_stepmesh_scheduler` + grouped paths in `attn_send`/`ffn_recv`/`attn_recv` |
| `afd.py` — `AsyncTensorCommunicator` | Added `@torch.compiler.disable()` to 5 methods |
| `afd.py` — `AFDCommunicator` | Added `@torch.compiler.disable()` to 3 methods |
| `afd_mixin.py` | `torch.compile` for `_run_attn`/`_run_mlp` in `_afd_init` |
| `scheduler.py` | `afd_overlap_enabled` initialization + `event_loop_afd` overlap mode (`result_queue` + `_pop_and_process`) |

#### Issues Found

| ID | Severity | File | Issue | Status |
|------|--------|------|------|------|
| ~~R1~~ | ~~Low~~ | `afd_mixin.py` | ~~Compiling no-op modules on the Proxy side~~ → compile only the actual compute side according to perspective | **Fixed** |
| ~~R2~~ | ~~Low~~ | `afd.py` | ~~`prepare_attn` missing `@torch.compiler.disable()`~~ → added (all 10 communication methods now covered) | **Fixed** |
| F9-1 | **Low** | `afd.py` | FFN processes padded tensors in grouped mode (at most TP_A-1 zero tokens) | Not fixed (<1%, long-term optimization) |
| ~~F9-2~~ | ~~Low~~ | `afd.py` | ~~Redundant per-group scheduler flag~~ → removed | **Fixed** |
| ~~F9-3~~ | ~~Low~~ | `afd.py` | ~~Unclear force-set comment in grouped path~~ → clarified comment | **Fixed** |
| ~~F3-R1~~ | ~~Very low~~ | `server_args.py` | ~~No warning for conflicting flags~~ → added warning | **Fixed** |

#### Final Review (Full Validation)

A final line-by-line review of all modified files confirmed that the following features are correct:

| Feature | Validation Result |
|------|---------|
| **F9 grouped StepMesh** | GCD grouping + padding + all_gather + stride deduplication + truncation: all paths correct |
| **F2-A torch.compile** | Compiles only the actual compute side according to perspective; all 10 communication methods use `@torch.compiler.disable()` |
| **F3 overlap scheduling** | `_handle_afd` automatically disables overlap + `--afd-enable-overlap-schedule` explicitly enables it; `result_queue` enqueue/dequeue timing in the `event_loop_afd` overlap path exactly matches `event_loop_overlap`; edge cases including first iteration, batch=None, and consecutive prefill are safe |
| **Backward compatibility** | Non-AFD scenarios unaffected; default AFD behavior (without new flags) exactly matches behavior before the changes |

**No new issues.** The only remaining known issue is F9-1 (padding tokens, <1%, deliberately left unfixed).

#### Verified Correct Components

| Check | Status |
|--------|------|
| **F9 grouped StepMesh** | |
| GCD grouping: 4A:8F / 8A:4F / 6A:4F / 3A:2F / 4A:4F | OK |
| `attn_send` push shard padding only in `_grouped` mode | OK |
| `attn_send` pull buffer uses `effective_tokens` to match `ffn_send` shard size | OK |
| `attn_send` key increment `1 + servers_per_group` / `1 + ffn_tp` | OK |
| Correct `ffn_recv` / `attn_recv` all_gather + stride-select | OK |
| `ffn_send` needs no changes | OK |
| Per-group DMLC config + scheduler startup | OK |
| Backward compatibility + edge cases | OK |
| **F2-A torch.compile** | |
| `_run_attn` / `_run_mlp` compilation conditions + `@torch.compiler.disable()` on 8 communication methods | OK |
| `dynamic=True` + dynamo cache + Mixin coverage + harmless Proxy | OK |
| **F3 overlap scheduling** | |
| `_handle_afd` automatically sets `disable_overlap_schedule` (without `--afd-enable-overlap-schedule`) | OK |
| When `afd_overlap_enabled=True`, `enable_overlap=True`, and `run_batch` uses the async path | OK |
| `init_overlap()` initializes `forward_stream` + `FutureMap` | OK |
| `event_loop_afd` overlap path: `result_queue` + `_pop_and_process` + `batch.copy()` | OK |
| `is_disable_overlap_for_batch`: consecutive prefill disables overlap; first iteration is safe | OK |
| `launch_batch_sample_if_needed` called after `_pop_and_process` (matching `event_loop_overlap`) | OK |
| `dispatch_event_loop`: AFD takes precedence over overlap, bypassing `event_loop_overlap` | OK |
| Non-AFD scenarios completely unaffected | OK |
| Non-overlap AFD path (`afd_overlap=False`) matches the original implementation | OK |

---

### Completed Task Overview

| ID | Description | Round |
|------|------|------|
| F1 | StepMesh N:M bidirectional sharded communication (arbitrary TP_A:TP_F) | Round 1 |
| F6 | Added `AFDWeightFilter` to DeepSeek-V2 `load_weights` | Round 1 |
| F7 | Added `is_target_verify()` handling to `afd_prepare_overlap` | Round 1 |
| F8 | Removed redundant `m is None` check in `afd_overlap.py` | Round 1 |
| G3 | RDMA final-shard padding + truncation | Round 1 |
| H1-H3 | StepMesh key collisions / docstring / recv buffer registration | Round 1 |
| F9 | GCD-based grouped StepMesh (`--afd-grouped-stepmesh`) | Round 2 |
| F2-A | `torch.compile` per-stage compilation + AFD communication `compiler.disable()` | Round 2 |
| F3 | `--afd-enable-overlap-schedule` CPU/GPU overlap scheduling | Round 2 |
| Phase 6 | Heterogeneous TP (`ShardedParallelCommunicator` + StepMesh N:M) | Early |
| Task 6 | Independent PD + AFD control | Early |
| C1-C5, S1-S3, G3-G6, E1-E5 | Communication/scheduling/compute/engineering optimizations | Early |
| T1-T2 | Universal AFD Toggle: removed microbatch dependency + TP>1 broadcast fix | Round 3 |
| T4 | `_handle_afd` automatically disables CUDA graph + piecewise CUDA graph | Round 3 |
| L1 | LLaMA/Llama3 AFD support (LayerCommunicator + Mixin + Model.forward + WeightFilter) | Round 4 |
| L2 | Fixed FFN-side sub-backend crash in `AfdAttnBackend.init_forward_metadata` | Round 4 |
| L3 | `_run_attn`/`_run_mlp` binding fix (affects all models using the Mixin) | Round 4 |
| L4 | `r.seq_len` → `r.seqlen` typo fix in `scheduler_afd_mixin.py` | Round 4 |

---

### Complete Outstanding Task List

| Priority | ID | Description | Impact | Complexity | Prerequisites |
|--------|------|------|------|--------|---------|
| ~~P1~~ | ~~F3~~ | ~~`event_loop_afd` CPU/GPU overlap scheduling~~ | ~~5-10%~~ | ~~Low (1 week)~~ | **Implemented** |
| P1 | L5 | FFN sync flag not cleared when batch=None, potentially causing batch mismatch | Correctness | Low (1 day) | None |
| P1 | L6 | ZMQ drain last-one-wins may mismatch batch metadata when Attn runs ahead | Correctness | Low (1 day) | None |
| P2 | F2-B | CUDA Graph (reduce-overhead mode) | ~2-4% additional improvement | Medium (1-2 weeks) | F2-A ✓, requires microbatch padding + stream synchronization |
| P2 | F5 | Qwen2 (Dense) AFD support | Enables Qwen2 models | Low (2 days) | Same approach as LLaMA |
| P3 | F4 | Automatic profiling + adaptive `afd_attn_ratio` | Convenience | Medium (1 week) | None |
| P3 | G6 | Heterogeneous-path buffer reuse | <1% | Low (2 days) | None |
| P4 | T3 | Remove `can_run_afd_overlap` dead code (`forward_batch_info.py` + `afd_overlap.py`) | Code cleanliness | Very low (30 minutes) | T1-T2 ✓ |
| P4 | F9-opt-1 | Grouped StepMesh: custom sub-groups | 1-5% (only stride>1) | Medium (3-5 days) | F9 ✓ |
| P4 | F9-opt-2 | Grouped StepMesh: grouped buffer pool | <1% | Low (2 days) | F9 ✓ |
| P4 | F9-opt-3 | Grouped StepMesh: avoid MLP execution on padding tokens | <1% (typical workloads) | Low (3 days) | F9 ✓ |
| ~~P4~~ | ~~R1~~ | ~~Skip Proxy-side torch.compile~~ | | | **Fixed** |
| ~~P4~~ | ~~R2~~ | ~~Add compiler.disable to prepare_attn~~ | | | **Fixed** |
| ~~P4~~ | ~~F9-2~~ | ~~Redundant per-group scheduler flag~~ | | | **Fixed** |
| ~~P4~~ | ~~F9-3~~ | ~~Force-set comment in grouped path~~ | | | **Fixed** |
| ~~P4~~ | ~~F3-R1~~ | ~~Warn about conflicting flag combinations~~ | | | **Fixed** |

---

### Impact Levels

```
Medium (2-4%):       F2-B
Low (1-5%):          F5, F4, F9-opt-1
Very low (<1%):     G6, F9-opt-2, F9-opt-3
```

### F3 Reassessment: TBO + AFD Coexistence Analysis

Code analysis suggests that the original F3 ("TBO + AFD coexistence") needs to be redefined:

| Approach | Description | Benefit | Complexity | Conclusion |
|------|------|------|--------|------|
| ~~Full coexistence~~ | Nested TBO batch splitting + AFD microbatch pipeline | <5% (batch fragmentation: original batch split into 2×m=6 pieces, each too small for effective overlap) | Very high (6-8 weeks) | **Not recommended** |
| **CPU/GPU overlap** | Introduce overlap scheduling in `event_loop_afd` (CPU scheduling parallel to GPU execution) | **5-10%** | **Low (1 week)** | **Recommended** |

**Reason**: TBO provides value at two levels—(1) batch splitting + stage overlap (conflicts with AFD and offers limited benefit); (2) CPU/GPU scheduling overlap (fully orthogonal to AFD). Taking only (2) captures most of the benefit.

**Specific changes**: modify only `event_loop_afd` in `scheduler.py`, allowing the CPU to schedule the next batch ahead of time while the GPU executes the current batch (recv_requests + get_next_batch + afd_send_to_ffn), without changing the attention backend, forward path, or batch preparer.

### Recommended Execution Path

```
Phase 1 (completed ✓):
  F9    — Grouped StepMesh → cross-node traffic 12NH→3NH
  F2-A  — torch.compile per stage → kernel fusion ~5-8%
  F3    — CPU/GPU overlap scheduling → 5-10%
  L1-L4 — LLaMA support + cross-model Mixin fixes + scheduler typo fix

Phase 1.5 (priority fixes):
  L5    — Clear FFN sync flag
  L6    — ZMQ batch metadata alignment

Phase 2 (next priority):
  F2-B  — reduce-overhead CUDA Graph → additional ~2-4% (1-2 weeks, including R2 prerequisite)

Phase 3 (extensions & convenience):
  F5    — Qwen2 Dense AFD (same approach as LLaMA, ~2 days)
  F4    — Automatic profiling

Phase 4 (nice-to-haves):
  G6, F9-opt-1/2/3, L7-L12 — combined <5%
```

### Current AFD Feature Completeness

| Dimension | Status | Notes |
|------|------|------|
| Communication layer | **Complete** | ZMQ (homogeneous+heterogeneous) + StepMesh (homogeneous+heterogeneous+grouped), arbitrary N:M |
| Scheduling layer | **Complete** | Microbatch splitting, batch alignment, independent PD control, CPU/GPU overlap scheduling (`--afd-enable-overlap-schedule`) |
| Compute layer | **Complete** | Pipeline overlap, asymmetric splitting, preallocated output |
| Compilation optimization | **Phase A complete** | `torch.compile` kernel fusion; Phase B (CUDA Graph) not yet implemented |
| Model support | **Mostly complete** | Qwen3-MoE/Qwen2-MoE/DeepSeek-V2/V3/Qwen3(Dense)/LLaMA(Dense), missing Qwen2(Dense) |
| Engineering | **Complete** | Mixin abstraction, weight filtering, separate files, timeout handling |
| **Largest untapped benefit** | **F2-B** | reduce-overhead CUDA Graph (~2-4%, 1-2 weeks) |

---

## XI. Universal AFD Toggle: Removing the Microbatch Dependency

### Background

In the original implementation, activation of the AFD pipeline (`model_forward_afd`) was controlled by `can_run_afd_overlap`, set in `AfdForwardBatchPreparer.prepare` based on two conditions:

1. `batch_size >= afd_micro_batch` (no splitting if the batch is too small)
2. `afd_split_seq_index is not None` (the scheduler did not provide split indices)

This means that with **batch_size=1** or **an inappropriate `--afd-micro-batch` setting**, AFD is not activated and model forward uses the normal path. In an Attn-FFN disaggregated deployment, the A node then attempts to execute FFN locally (missing weights, causing crashes or incorrect results), while the F node stays idle.

### Change Objective

**Whenever `--afd-perspective` is set, unconditionally use the `model_forward_afd` path**, no longer constrained by `--afd-micro-batch` or batch size. Microbatch splitting remains available (when `afd_children` exists), but without splitting (`afd_children=None`), `m_stage=1` reduces execution to a single-step AFD forward without pipelining.

### Modified Files and Changes

#### 1. Model Forward Layer (3 Model Files)

| File | Changes |
|------|------|
| `python/sglang/srt/models/qwen3.py` | `Qwen3Model.forward`: changed condition from `getattr(forward_batch, "can_run_afd_overlap", False)` to `get_afd_perspective() is not None` |
| `python/sglang/srt/models/qwen2_moe.py` | `Qwen2MoeModel.forward`: same as above |
| `python/sglang/srt/models/deepseek_v2.py` | `DeepseekV2Model.forward`: same as above |

**Effect**: whenever the process starts with `--afd-perspective attn/ffn`, model forward uses the AFD path. `get_afd_perspective()` is process-global state, with no runtime overhead.

#### 2. AFD Pipeline Layer (`afd.py`)

| Function | Changes |
|------|------|
| `model_forward_afd_split_inputs._split_raw` | Added `afd_children is None` fallback: returns a one-element list containing the full batch (no splitting) |
| `model_forward_afd` | Changed `m_stage` from hard-coded `afd_micro_batch` to `len(afd_children)` or `1`; always matches the number of items returned by `_split_raw` |

**Effect**: without microbatches, `m_stage=1` reduces the pipeline to single-step A→F→A→F..., still using `AFDCommunicator` for cross-node communication.

#### 3. Scheduler Layer (Message Forwarding and Merging)

| File | Changes |
|------|------|
| `scheduler.py` — `_recv_afd_messages` | No longer drops non-`AFDReqInput` messages (e.g., `TokenizedGenerateReqInput`); collects them into an `extra_reqs` list and returns it |
| `scheduler.py` — `event_loop_afd` | FFN side merges `extra_reqs` into `recv_reqs`; re-broadcast when TP > 1 (**all ranks participate**) |
| `scheduler_afd_mixin.py` — `afd_recv_messages` | Same fix as `_recv_afd_messages` |
| `disaggregation/prefill.py` | Merge `extra_reqs` + TP > 1 re-broadcast |
| `disaggregation/decode.py` | Merge `extra_reqs` + TP > 1 re-broadcast |

**Effect**: Attn forwards user requests to FFN through `afd_forward_work_requests`; FFN receives and merges them into its main request stream through `_recv_afd_messages` / `afd_recv_messages`, ensuring that FFN independently builds its request queue, executes `run_batch`, and emits logs.

### Issues Found During Review

| ID | Severity | Location | Issue | Status |
|------|--------|------|------|------|
| T1 | **Critical** | `scheduler.py` `event_loop_afd` | `broadcast_pyobj` inside `if extra_reqs:` → only TP0 calls broadcast when TP > 1, other ranks skip it → **deadlock** | **Fixed** |
| T2 | **Critical** | `disaggregation/prefill.py` + `decode.py` | Missing TP > 1 re-broadcast after merging `extra_reqs` → non-TP0 ranks cannot see AFD-forwarded requests | **Fixed** |
| T3 | **Low** | `forward_batch_info.py` + `afd_overlap.py` | `can_run_afd_overlap` still set but no longer read (dead code) | No correctness impact; can be cleaned up later |
| T4 | **Medium** | `server_args.py` `_handle_afd` | AFD does not automatically disable CUDA graph / piecewise CUDA graph → FFN warmup crashes because `AfdAttnBackend` sub-batches lack attention fields | **Fixed** (auto-disable) |

### Verified Correct Components

| Check | Status |
|--------|------|
| `_split_raw` fallback: returns a one-element list when `afd_children=None`, `afd_subbatch_index=0` | OK |
| `model_forward_afd`: `AFDStageScheduleGenerator` generates the correct A→F sequence when `m_stage=1` | OK |
| `get_afd_perspective()` imports in all three models are inside `forward` (lazy imports, consistent style) | OK |
| `_recv_afd_messages`: reads all queued messages; `AFDReqInput` last-one-wins (normally only one queued) | OK |
| `afd_recv_messages` (mixin version) matches `_recv_afd_messages` logic | OK |
| `broadcast_pyobj` **outside** `if extra_reqs:`, all TP ranks participate (after fix) | OK |
| Non-AFD scenarios: `get_afd_perspective()` returns `None`, model forward uses the normal path, unaffected | OK |
| Without TP > 1: no broadcast, no additional overhead | OK |

### Data Flow (After Fix)

```
┌──────── Attn Node ────────┐        ┌──────── FFN Node ────────┐
│                            │        │                          │
│  HTTP /generate            │        │                          │
│       │                    │        │                          │
│  recv_requests()           │        │  recv_requests()         │
│       │                    │        │       │                  │
│  afd_forward_work_requests │──ZMQ──▶│  _recv_afd_messages()   │
│  (forward TokenizedReq to FFN)│     │  (collect extra_reqs)     │
│       │                    │        │       │                  │
│  process_input_requests    │        │  merge extra→recv_reqs  │
│       │                    │        │  broadcast (TP > 1)      │
│  get_next_batch            │        │       │                  │
│       │                    │        │  process_input_requests  │
│  afd_send_batch_info ──ZMQ─┼───────▶│  (FFN builds request queue)│
│  (AFDReqInput)             │        │       │                  │
│       │                    │        │  get_next_batch          │
│  run_batch                 │        │       │                  │
│  ├─ model_forward_afd ◄────┼─tensor─┼── run_batch             │
│  │  (A→comm→wait→A→...)    │  comm  │  ├─ model_forward_afd   │
│  │                    ─────┼─tensor─┼──▶ (comm→F→comm→F→...)  │
│  └─ done                   │        │  └─ done                │
│       │                    │        │       │                  │
│  process_batch_result      │        │  process_batch_result   │
│  (return HTTP response)    │        │  (emit batch logs)        │
└────────────────────────────┘        └──────────────────────────┘
```

---

## XII. Round 4 Review: LLaMA AFD Integration + Cross-Model Fixes

### Background

Testing with `llama3.1-8B --tp 2 --afd-perspective attn/ffn` revealed:
1. FFN-side `init_forward_metadata` crashes (`req_pool_indices` is `None`)
2. Attn continues normal inference after FFN is stopped → AFD pipeline not actually activated
3. `_run_attn` / `_run_mlp` not bound to DecoderLayer instances → potential crashes in every model's AFD forward path

### Issues Found

| ID | Severity | File | Issue | Status |
|------|--------|------|------|------|
| L1 | **Critical** | `llama.py` | LLaMA has no AFD support—no `model_forward_afd` branch, `forward_afd_A/F`, `LayerCommunicator`, or `AFDWeightFilter` | **Fixed** |
| L2 | **Critical** | `tbo_backend.py` | FFN-side `AfdAttnBackend.init_forward_metadata` calls flashinfer `call_begin_forward` on `afd_children`; sub-batch `req_pool_indices=None` causes `TypeError` | **Fixed** |
| L3 | **Critical** | `afd_mixin.py` | `_afd_init` binds `_run_attn`/`_run_mlp` to the instance only with `--enable-torch-compile`; without compile, `forward_afd_A/F` calls `self._run_attn()`, triggering `AttributeError`—**affects all models using the Mixin** | **Fixed** |
| L4 | **Critical** | `scheduler_afd_mixin.py` | `r.seq_len` in `afd_send_batch_info` should be `r.seqlen` (`Req` attribute name)—disagg prefill/decode calls raise `AttributeError` | **Fixed** |
| L5 | **Medium** | `scheduler.py` ~1444 | FFN-side `_afd_batchsize_attn` cleared only when `batch is not None`; if FFN consumes `AFDReqInput` but `get_next_batch` returns `None`, the stale sync flag skips waiting in the next iteration | To be fixed |
| L6 | **Medium** | `scheduler.py` ~1327 | ZMQ drain loop uses last-one-wins: if Attn runs several steps ahead of FFN, `AFDReqInput` may refer to a newer batch, mismatching the batch actually run by FFN | To be fixed |
| L7 | **Medium** | `afd_overlap.py` ~343 | `_filter_batch` hard-codes its field list: future non-`None` `ForwardBatch` fields (e.g., `encoder_*`, `mamba_track_*`, `nsa_cp_metadata`) trigger `Exception: N errors: Field ... not yet supported` | Needs attention |
| L8 | **Low** | `llama.py` AFD branch | `LlamaModel.forward` AFD branch returns only `hidden_states` (no `aux_hidden_states`); with `capture_aux_hidden_states=True` (EAGLE3), unpacking in `LlamaForCausalLM.forward` fails | No impact on current usage (AFD + EAGLE3 do not coexist) |
| L9 | **Low** | `afd_mixin.py` ~130 | `forward_afd_F` still calls `_run_mlp` for 0-token sub-batches (`forward_afd_A` has a `shape[0] != 0` guard, F does not) | Edge case, not triggered in practice |
| L10 | **Low** | `scheduler_afd_mixin.py` ~58-60 | `_afd_forward_mode` stored but never read, and not cleared by `afd_reset_state` | dead code |
| L11 | **Low** | `afd.py` ~807 | `get_afd_mirco_batch` (misspelled alias) still retained | Code cleanliness |
| L12 | **Low** | Cross-model | DeepSeek-V2/Qwen3/Qwen2-MoE `forward_afd_A/F` use default Mixin `_run_attn`/`_run_mlp` without model-specific arguments (e.g., `quant_format`, `zero_allocator`, `use_reduce_scatter`) | Minor behavioral differences between AFD and non-AFD paths |

### Implemented Fixes

#### L1: LLaMA AFD Support (`llama.py`)

**Changes**:
- `LlamaDecoderLayer.__init__`: when `get_afd_perspective() is not None`, create `LayerScatterModes` (`is_layer_sparse=False`), `LayerCommunicator`, and call `_afd_init()`
- `LlamaDecoderLayer`: add `forward_afd_A` / `forward_afd_F` delegating to the Mixin
- `LlamaModel.forward`: add `model_forward_afd` branch
- `LlamaForCausalLM.load_weights`: add `AFDWeightFilter` weight filtering

#### L2: FFN-Side `init_forward_metadata` Crash (`tbo_backend.py`)

**Root cause**: `AfdForwardBatchPreparer._filter_batch` sets FFN sub-batch `req_pool_indices` to `None`, but `AfdAttnBackend.init_forward_metadata` still calls sub-backend `init_forward_metadata`; flashinfer attempts `len(req_pool_indices)` and crashes.

**Fix**: skip sub-backend initialization in `init_forward_metadata` when on the FFN side (FFN uses `AFDProxyAttention`; sub-attention metadata is never used).

#### L3: Unbound `_run_attn`/`_run_mlp` (`afd_mixin.py`)

**Root cause**: none of the models (DeepSeek-V2, Qwen3, Qwen2-MoE, LLaMA) inherit `AFDDecoderLayerMixin`; instead they call `AFDDecoderLayerMixin.forward_afd_A(self, ...)`. During internal lookup of `self._run_attn(...)`, `self` is a DecoderLayer instance whose MRO contains no Mixin → `AttributeError`. Only with `--enable-torch-compile` does `_afd_init` assign it to the instance through `torch.compile`.

**Fix**: unconditionally bind default `_run_attn`/`_run_mlp` (`types.MethodType`) in `_afd_init`, before `torch.compile`. A `hasattr` check allows model classes to override them.

#### L4: `r.seq_len` Typo (`scheduler_afd_mixin.py`)

**Root cause**: `Req` uses `@property seqlen` (no underscore), but the Mixin uses `r.seq_len`.

**Fix**: change to `r.seqlen`.

### Verified Correct Components

| Check | Status |
|--------|------|
| LLaMA `LayerScatterModes.init_new` arguments (all False for dense models) | OK |
| LLaMA `LayerCommunicator` receives `input_layernorm` + `post_attention_layernorm` | OK |
| LLaMA `AFDWeightFilter` compatible with stacked_params_mapping (iterator-level filtering, no effect on name.replace) | OK |
| LLaMA `LlamaMLP.forward(x, forward_batch=None)` signature compatible with Mixin `_run_mlp(hidden_states, forward_batch)` | OK |
| After `types.MethodType` binding, `torch.compile(self._run_attn)` correctly receives the bound method | OK |
| FFN skip in `tbo_backend.py` applies only when `afd_children is not None`; non-AFD scenarios unaffected | OK |
| After fix, `r.seqlen` in `scheduler_afd_mixin.py` matches `scheduler.py` ~1429 | OK |

### Follow-up Task Updates

| Priority | ID | Description | Status |
|--------|------|------|------|
| ~~P0~~ | ~~L1~~ | ~~LLaMA AFD support~~ | **Fixed** |
| ~~P0~~ | ~~L2~~ | ~~FFN init_forward_metadata crash~~ | **Fixed** |
| ~~P0~~ | ~~L3~~ | ~~Unbound `_run_attn`/`_run_mlp` (cross-model)~~ | **Fixed** |
| ~~P0~~ | ~~L4~~ | ~~`r.seq_len` typo~~ | **Fixed** |
| P1 | L5 | FFN sync flag not cleared when batch=None | To be fixed |
| P1 | L6 | Potential ZMQ drain last-one-wins batch mismatch | To be fixed |
| P2 | F5 | Qwen2 Dense AFD—same adaptation approach as LLaMA | Can be implemented quickly |
| P3 | L7 | Extend `_filter_batch` field whitelist | Needs attention |
| P3 | L12 | Missing model-specific arguments in AFD path | Low priority |
| P4 | L8-L11 | Edge cases / dead code / misspelled alias | Nice-to-haves |

---

## XIII. Round 5: ZMQ Heterogeneous TP + Runtime Bug Fixes

### Background

Round 4 testing of LLaMA 3.1-8B + TP=2 exposed a series of runtime issues and the design limitation that the ZMQ path did not support heterogeneous TP.

### Issues Found and Fixed

| ID | Severity | File | Issue | Status |
|------|--------|------|------|------|
| R5-1 | **Critical** | `afd.py` ZMQ send | `cpu_tensor.numpy().tobytes()` does not support bfloat16 (numpy has no bf16) | **Fixed**: use `bytes(cpu_tensor.untyped_storage())` |
| R5-2 | **Critical** | `scheduler.py` `_recv_afd_messages` | `AFDReqInput` extracted on TP rank 0 but excluded from broadcast → TP rank 1 `_afd_batchsize_attn` remains None forever → **FFN TP>1 deadlock** | **Fixed**: include AFDReqInput in broadcast data; `_afd_process_input_requests` filters it before passing to the dispatcher |
| R5-3 | **Critical** | `afd.py` ShardedParallelCommunicator | ZMQ 1:1 port mapping → with heterogeneous TP (e.g., 1A:2F), FFN rank 1 has no peer → deadlock | **Fixed**: rewritten as `BroadcastTensorCommunicator` (rank 0 ZMQ + NVLink broadcast) |
| R5-4 | **Medium** | `communicator.py` `_gather_hidden_states_and_residual` | For 0-token batches, `RMSNorm(x, res)` returns a single tensor when `x.numel()==0` → unpacking fails | **Fixed**: add `if hidden_states.shape[0] == 0: return` at entry |
| R5-5 | **Medium** | `llama.py` AFD norm | FFN-side `RMSNorm(hs, None)` returns a single tensor when `residual=None` → unpacking fails | **Fixed**: branch based on `residual is not None` |
| R5-6 | **Medium** | `afd.py` `model_forward_afd` | 0-token batches entering the AFD pipeline trigger various CUDA kernel errors | **Fixed**: short-circuit with `if hidden_states.shape[0] == 0: return` at entry |
| R5-7 | **Medium** | `afd.py` ZMQ recv | `torch.frombuffer(bytearray(0))` crashes when receiving 0-byte data | **Fixed**: use `torch.empty(shape, dtype)` for empty buffers |
| R5-8 | **Medium** | `server_args.py` `_handle_afd` | FFN warmup deadlocks (event_loop_afd waits for Attn sync that never arrives) | **Fixed**: FFN automatically sets `skip_server_warmup` |

### BroadcastTensorCommunicator Design (R5-3)

Replaces the old `ShardedParallelCommunicator`, supporting arbitrary N:M through a single rank 0 connection + NVLink broadcast:

**Principle**: the tensor used for cross-node AFD communication is the full hidden_states after all-reduce; all TP ranks hold identical data. Only one rank needs to send/receive, with the others obtaining it through NVLink broadcast.

**Data flow**:
1. Sending side: `local_tp_rank == 0` sends the full tensor through ZMQ; other ranks skip
2. Receiving side: rank 0 ZMQ receive → broadcast shape+dtype (3 longs) → non-rank-0 memory allocation → broadcast data tensor

**Traffic comparison**:

| Configuration | Old Implementation (Per-Rank ZMQ) | New Implementation (Rank 0 + Broadcast) |
|------|----------------------|---------------------------|
| 4A:4F homogeneous | 8NH (4 connections each transfer NH, redundant) | **2NH** |
| 4A:8F heterogeneous | Deadlock | **2NH** |
| 1A:2F heterogeneous | Deadlock | **2NH** |

**Limitation**: a single ZMQ connection does not exploit multi-NIC parallelism. High-performance multi-NIC scenarios should use StepMesh (RDMA).

#### Review Results

| Check | Status |
|--------|------|
| `send_tensor` sends on rank 0; other ranks skip | OK |
| `recv_tensor` performs two broadcasts (metadata + data) | OK |
| `tp_group.ranks[0]` used as broadcast src | OK |
| 0-element tensor broadcast is safe (NCCL no-op) | OK |
| Unified homogeneous/heterogeneous path in `get_tensor_communicator` | OK |
| TP=1 skips wrapping | OK |
| `_dtype_to_int` covers fp16/bf16/fp32 | OK |
| StepMesh path unaffected | OK |

### Additional Completed Tasks

| ID | Description | Round |
|------|------|------|
| R5-1 | ZMQ bf16 serialization fix | Round 5 |
| R5-2 | FFN TP>1 AFDReqInput broadcast fix | Round 5 |
| R5-3 | `BroadcastTensorCommunicator` (ZMQ heterogeneous TP) | Round 5 |
| R5-4 | Communicator 0-token guard | Round 5 |
| R5-5 | LLaMA norm residual=None handling | Round 5 |
| R5-6 | `model_forward_afd` 0-token short-circuit | Round 5 |
| R5-7 | ZMQ recv 0-byte buffer handling | Round 5 |
| R5-8 | FFN automatically sets skip_server_warmup | Round 5 |

### Outstanding Task Updates

| Priority | ID | Description | Status |
|--------|------|------|------|
| P1 | L5 | FFN sync flag not cleared when batch=None | To be fixed |
| P1 | L6 | Potential ZMQ drain last-one-wins batch mismatch | To be fixed |
| P2 | F5 | Qwen2 Dense AFD | Can be implemented quickly |
| P3 | R5-9 | Avoid creating unused ZMQ sockets on non-rank-0 processes | To be optimized |
| P4 | R5-10 | `_dtype_to_int` errors on unknown dtype rather than silently falling back | Nice-to-have |

---

## Round 6: UCX RDMA Communicator Replaces StepMesh

### Background

The CUDA runtime in StepMesh (`fserver_lib`) conflicts with libraries required by the SGLang framework, preventing normal deployment. UCX-Py (native Python RDMA bindings) reimplements all cross-node Attn-FFN tensor communication functionality.

### New/Modified Files

| File Path | Lines | Changes |
|----------|------|------|
| `python/sglang/srt/layers/rdma_comm.py` | 658 | New: UCX-Py RDMA communicator |
| `python/sglang/srt/layers/afd.py` | +12 | Added UCX path to `get_tensor_communicator()` |
| `python/sglang/srt/server_args.py` | +12 | `afd_comm_backend` field + `--afd-comm-backend` CLI |

### Architectural Design

```
rdma_comm.py layered architecture:

_AsyncBridge              SelectorEventLoop background thread (avoids uvloop conflicts)
_BufferPool               GPU tensor reuse pool (automatically recycles the previous recv buffer)
_detect_num_nic_groups()  Parses nvidia-smi topo to detect GPU-NIC affinity automatically
_UcxP2PCommunicator       1:1 RDMA point-to-point (UCX endpoint send/recv)
UcxTensorCommunicator     NIC-aware N:M communication (K-way RDMA + NVLink distribution)
```

NIC-aware N:M communication model:
- K = NIC count (automatically detected or overridden by AFD_UCX_NUM_NICS), must divide local_tp
- Each of K NIC groups selects one representative for RDMA (each transfers 1/K of the data)
- Non-representative ranks obtain full data through NVLink broadcast (K=1) or all_gather+stride deduplication (K>1)
- Total cross-node RDMA traffic remains 2NH/layer, independent of K and TP size

K=1: rank 0 RDMA → NVLink broadcast
K>1: K representatives each RDMA 1/K → rank0 broadcasts shape → non-representatives fill with zeros → all_gather → stride deduplication → truncate

### Environment Variables

| Variable | Default | Description |
|------|--------|------|
| `AFD_UCX_TLS` | `rc,tcp,cuda_copy,cuda_ipc` | UCX transport list |
| `AFD_UCX_BASE_PORT` | `25000` | FFN listening base port |
| `AFD_UCX_FFN_HOST` | `127.0.0.1` | FFN node address |
| `AFD_UCX_NUM_NICS` | Auto-detected | NIC count K |
| `AFD_UCX_TIMEOUT` | `60` | Connection timeout in seconds |
| `UCX_LOG_LEVEL` | `fatal` (must be set in shell) | Suppress non-fatal UCX errors |

### Validation Results

| Test Scenario | Result | Notes |
|----------|:---:|------|
| 1:1 RDMA throughput | PASS | 15-33 GB/s |
| K=1 multi-TP NVLink broadcast | PASS | FFN TP=2 ranks agree |
| K=2 multi-NIC all_gather | PASS | Attn TP=2, FFN TP=2 |
| K=2 heterogeneous TP + non-representative ranks | PASS | Attn TP=4, FFN TP=2, all 4 ranks correct |
| TP=1 Llama 3.1 8B E2E | PASS | Correct text |
| TP=2 Llama 3.1 8B E2E | PASS | Correct text, idle GPU utilization 0% |
| TCP-only fallback | FAIL | Known limitation L6-1 |
| GPU-direct RDMA | PASS | nvidia_peermem active, IB/TCP=23.9x |

### Launch Commands

TP=1 single-GPU AFD (minimal mode):

    # Terminal 1: FFN (start first)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=3 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
    AFD_UCX_BASE_PORT=25000 \
    python -m sglang.launch_server \
      --model-path <model_path> --tp 1 \
      --afd-perspective ffn --afd-comm-backend ucx --port 30001

    # Terminal 2: Attn (start after FFN reports "listening")
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=2 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
    AFD_UCX_BASE_PORT=25000 AFD_UCX_FFN_HOST=127.0.0.1 \
    python -m sglang.launch_server \
      --model-path <model_path> --tp 1 \
      --afd-perspective attn --afd-comm-backend ucx --port 30000

    # Terminal 3: Request
    curl -s http://localhost:30000/generate \
      -H "Content-Type: application/json" \
      -d '{"text":"Hello","sampling_params":{"max_new_tokens":16}}'

TP=2 multi-GPU AFD (K=1 NVLink broadcast):

    # FFN TP=2
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=4,5 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
    AFD_UCX_BASE_PORT=25000 AFD_UCX_NUM_NICS=1 \
    python -m sglang.launch_server \
      --model-path <model_path> --tp 2 \
      --afd-perspective ffn --afd-comm-backend ucx --port 30001

    # Attn TP=2
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=2,3 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc \
    AFD_UCX_BASE_PORT=25000 AFD_UCX_FFN_HOST=127.0.0.1 AFD_UCX_NUM_NICS=1 \
    python -m sglang.launch_server \
      --model-path <model_path> --tp 2 \
      --afd-perspective attn --afd-comm-backend ucx --port 30000

Multi-node deployment (cross-machine IB RDMA):

    # Node B (FFN): start normally, listening on 0.0.0.0
    # Node A (Attn): set AFD_UCX_FFN_HOST to node B's IB IP
    AFD_UCX_FFN_HOST=<FFN_node_IB_address> ...

Key notes:
- UCX_LOG_LEVEL=fatal and UCX_WARN_UNUSED_ENV_VARS=n must be passed as shell environment variables, because NCCL initialization loads the UCX C library first
- AFD_UCX_NUM_NICS automatically detects GPU-NIC affinity if unspecified; a manual value must divide TP
- AFD_UCX_FFN_HOST is needed only on the Attn side; FFN listens automatically
- FFN must start first (listener role), followed by Attn (connector role)

### Per-Module Test Commands

The following test scripts are located in `/workspace/` and can run independently (without starting the full SGLang service).
First set the Python path: `export PYTHONPATH=/workspace/sglang/python:$PYTHONPATH`

**1. Basic RDMA Communication (1:1 P2P, 6 Tests):**

    # Test IB RDMA / NVLink / TCP fallback / bf16 / throughput
    # Use GPU 2 (Attn) and GPU 3 (FFN)
    python3 /workspace/test_rdma_comm.py

    Coverage:
    - Basic connection + correctness
    - FIFO ordering guarantees (50 iterations)
    - bf16 typical hidden-state throughput (~4MB)
    - f32 16MB / 64MB large-tensor throughput
    - TCP-only fallback (known limitation L6-1, requires cuda_ipc)

**2. K=1 Multi-TP + NVLink Broadcast:**

    # Attn TP=1 (GPU 1), FFN TP=2 (GPU 2,3)
    # FFN rank 0 receives through RDMA; rank 1 obtains data through NVLink broadcast
    python3 /workspace/test_multi_tp.py

    Coverage:
    - Non-representative ranks obtain data through NVLink broadcast
    - Verify data consistency between both FFN ranks
    - End-to-end correctness of Attn send→recv

**3. K=2 Multiple NICs + NVLink All_gather:**

    # Attn TP=2 (GPU 2,6), FFN TP=2 (GPU 3,7)
    # Two RDMA paths each transfer 1/2 of the data, followed by all_gather
    python3 /workspace/test_k2_multi_nic.py

    Coverage:
    - Multi-NIC parallel RDMA transfer
    - Reconstruct full tensor through all_gather + stride deduplication
    - Cross-NUMA-node communication

**4. K=2 Heterogeneous TP + Non-Representative Ranks (Most Complex Scenario):**

    # Attn TP=4 (GPU 2,3,6,7), FFN TP=2 (GPU 4,5)
    # Attn ranks 0,2 represent groups for RDMA; ranks 1,3 obtain data through NVLink
    python3 /workspace/test_k2_hetero.py

    Coverage:
    - ranks_per_group > 1 scenarios
    - Non-representative ranks fill with zeros + all_gather + stride deduplication
    - Data correctness on all 4 Attn ranks
    - Data consistency between both FFN ranks

**5. Full Regression (Run 1-4 Serially):**

    echo "=== 1:1 ===" && python3 /workspace/test_rdma_comm.py && \
    echo "=== K=1 multi-TP ===" && python3 /workspace/test_multi_tp.py && \
    echo "=== K=2 multi-NIC ===" && python3 /workspace/test_k2_multi_nic.py && \
    echo "=== K=2 heterogeneous TP ===" && python3 /workspace/test_k2_hetero.py

**6. E2E Inference Validation (Requires the Full Server):**

    # Start the TP=2 AFD server (see launch commands above), then:
    curl -s http://localhost:30000/generate \
      -H "Content-Type: application/json" \
      -d '{"text":"The capital of France is","sampling_params":{"max_new_tokens":16,"temperature":0}}' \
      | python3 -c "import sys,json; r=json.load(sys.stdin); \
        print('text:', repr(r['text'])); \
        print('tokens:', r['meta_info']['completion_tokens'])"

    Expected: completion_tokens=16, semantically coherent text

**7. Idle GPU Utilization Check:**

    # After server startup, GPU utilization should be near 0% without requests
    nvidia-smi --query-gpu=index,utilization.gpu --format=csv

**8. GPU-Direct RDMA Throughput Benchmark:**

    # Measure steady-state bandwidth after multiple warmup iterations
    # Compare TCP / IB RC / NVLink paths
    python3 /workspace/test_gpudirect.py

    Expected: IB RC ~10 GB/s, NVLink ~80-170 GB/s, TCP ~0.4 GB/s

**9. Combined PD Disaggregation + AFD Test (4 Server Processes, 6 GPUs):**

PD (Prefill/Decode disaggregation) transfers KV through Mooncake/Nixl, while AFD transfers hidden states through UCX RDMA; the two paths are independent.
Prefill and Decode each have their own Attn+FFN process pair.

    # Terminal 1: Prefill FFN (start first)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=4 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc AFD_UCX_BASE_PORT=25000 \
    python -m sglang.launch_server \
      --model-path /models/llama3.1-8 --tp 1 \
      --afd-perspective ffn --afd-comm-backend ucx \
      --disaggregation-mode prefill --port 30011

    # Terminal 2: Prefill Attn (after FFN is listening)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=5 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc AFD_UCX_BASE_PORT=25000 \
    AFD_UCX_FFN_HOST=127.0.0.1 \
    python -m sglang.launch_server \
      --model-path /models/llama3.1-8 --tp 1 \
      --afd-perspective attn --afd-comm-backend ucx \
      --disaggregation-mode prefill --port 30010

    # Terminal 3: Decode FFN (start first)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=6 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc AFD_UCX_BASE_PORT=25100 \
    python -m sglang.launch_server \
      --model-path /models/llama3.1-8 --tp 1 \
      --afd-perspective ffn --afd-comm-backend ucx \
      --disaggregation-mode decode --port 30021

    # Terminal 4: Decode Attn (after FFN is listening)
    UCX_LOG_LEVEL=fatal UCX_WARN_UNUSED_ENV_VARS=n \
    CUDA_VISIBLE_DEVICES=7 \
    AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc AFD_UCX_BASE_PORT=25100 \
    AFD_UCX_FFN_HOST=127.0.0.1 \
    python -m sglang.launch_server \
      --model-path /models/llama3.1-8 --tp 1 \
      --afd-perspective attn --afd-comm-backend ucx \
      --disaggregation-mode decode --port 30020

    # Terminal 5: Request (send to Prefill Attn)
    curl -s http://localhost:30010/generate \
      -H "Content-Type: application/json" \
      -d '{"text":"Hello","sampling_params":{"max_new_tokens":16}}'

    Notes:
    - Prefill and Decode use different AFD_UCX_BASE_PORT values (25000 vs 25100) to avoid port conflicts
    - PD KV transfer requires additional --disaggregation-transfer-backend configuration (mooncake/nixl)
    - AFD hidden-state communication (UCX) and PD KV transfer (Mooncake/Nixl) are fully independent

### Fixed Issues

| ID | Description | Fix |
|------|------|------|
| R6-1 | uvloop conflicts with UCX-Py BlockingMode | SelectorEventLoop |
| R6-2 | Repeated cuMemGetAddressRange errors | Remove bridge CUDA context + UCX_LOG_LEVEL=fatal |
| R6-3 | GPU utilization 100% while idle | Remove bridge-thread CUDA context |
| R6-4 | RDMA reads unfinished GPU kernel data | torch.cuda.synchronize() before send |
| R6-5 | Garbled TP=2 inference text | Device-wide synchronize replaces stream synchronize |
| R6-6 | Invalid K>1 dist.broadcast (different src) | rank0 broadcasts shape + zero fill + all_gather stride deduplication |
| R6-7 | K does not divide local_tp | Automatically adjust downward |
| R6-8 | Metadata supports only 2D | Broadcast ndim first, then variable-length shape |
| R6-9 | BufferPool not recycled | _async_recv automatically puts the previous buffer back |
| R6-10 | K>1 padding not truncated | send records num_tokens, recv truncates |
| R6-11 | Incorrect LOCAL_RANK priority | Prioritize os.environ["LOCAL_RANK"] |

### Known Limitations

| ID | Description | Impact | Handling |
|------|------|------|------|
| L6-1 | TCP-only fallback unavailable | Only `UCX_TLS=tcp,cuda_copy` (no IB or NVLink) | Production path unaffected |
| L6-2 | UCX_LOG_LEVEL must be a shell environment variable | NCCL loads UCX before Python code | Add `UCX_LOG_LEVEL=fatal` to launch command |
| L6-3 | Automatic NIC detection relies on nvidia-smi topo | Falls back to K=1 if unavailable in containers | Manual AFD_UCX_NUM_NICS override |
| L6-4 | UCX-Py 0.35 no longer maintained | RAPIDS recommends UCXX | Meets functional needs; long-term migration required |

### Performance Optimizations (Implemented)

**R6-18: Warm BufferPool Preallocation**

`_BufferPool` adds a `warmup(shape, dtype, device, count)` method, preallocating GPU buffers during initialization according to the model's
hidden_size and microbatch count. The runtime recv path makes zero `torch.empty()` calls.
Lazy allocation is retained for backward compatibility when the shape is unknown.

**R6-19: Fully Asynchronous Send Pipeline**

- `_UcxP2PCommunicator.send_nonblocking(x)` — submits to the bridge and returns a Future without waiting for completion
- `UcxTensorCommunicator.send_tensor_nonblocking(x)` — upper-level nonblocking send
- `UcxTensorCommunicator.fence()` — waits for in-flight sends to complete
- `AsyncTensorCommunicator.send_async(x)` — background thread waits for the CUDA event, then submits a nonblocking send
- `AsyncTensorCommunicator.recv_wait()` — automatically joins the send thread + calls fence

Data safety guarantee: the background thread first calls `compute_event.synchronize()` to ensure GPU data is ready,
then submits the UCX send. The Python GIL is not an issue (critical operations event.synchronize/UCX C extensions release the GIL).

### Performance Optimization Proposal History (R6-13 + R6-14 + R6-16)

**Original Bottleneck Analysis (Resolved):**

The original UCX communicator used `torch.cuda.synchronize()` in `_UcxP2PCommunicator.send()`
for device-wide synchronization. This blocks all CUDA streams and destroys communication-compute overlap in the microbatch pipeline.

AFD's microbatch pipeline relies on three `AsyncTensorCommunicator` methods for overlap:
- `send_async(x)`: initiates sending on comm_stream and returns immediately
- `recv_start()`: initiates receiving on comm_stream without blocking the compute stream
- `recv_wait()`: waits for reception to complete and synchronizes back to the compute stream

Currently, `send_async` switches to comm_stream and calls `send_tensor` → `_p2p.send()` →
`torch.cuda.synchronize()`. This device-wide synchronization waits for all streams (including the compute stream),
preventing send from overlapping subsequent compute and reducing the pipeline to serial execution.

```
Current (serialized by synchronize):
  [Attn m0] → [sync+send m0] → [Attn m1] → [sync+send m1] → [recv m0] → ...
              ↑ all GPU work stops ↑        ↑ all GPU work stops ↑

Target (fine-grained CUDA event synchronization):
  [Attn m0] → [send m0        ] → [Attn m1] → [send m1        ]
                                   [recv m0 ←]
              ↑ wait only for m0's compute stream ↑   ↑ send overlaps compute ↑
```

**R6-13: torch.cuda.synchronize() → Fine-Grained CUDA Event Synchronization**

Modify `_UcxP2PCommunicator.send()` and `AsyncTensorCommunicator.send_async()`:

```python
# Approach: record an event on the compute stream; the comm stream waits for it

# AsyncTensorCommunicator.send_async:
def send_async(self, x: torch.Tensor):
    if self.comm_stream is not None:
        # Record the compute stream's current progress
        event = torch.cuda.current_stream().record_event()
        with torch.cuda.stream(self.comm_stream):
            # The comm stream waits for the compute stream to finish writing x
            event.wait(self.comm_stream)
            self.inner.send_tensor(x)
    else:
        self.inner.send_tensor(x)

# _UcxP2PCommunicator.send:
def send(self, x: torch.Tensor):
    # No device-wide synchronize; upper-level AsyncTensorCommunicator synchronizes via events
    self._bridge.run(self._async_send(x))
```

Benefit: send waits only for the compute stream producing the tensor, without blocking other streams or microbatches.

**R6-14: UCX Parameter Tuning**

| Parameter | Current Value | Suggested Value | Description |
|------|--------|--------|------|
| `UCX_RNDV_THRESH` | 8192 | Adjust according to hidden_size | Eager below this threshold (low latency), rendezvous above it (high throughput) |
| `UCX_MAX_RNDV_RAILS` | 1 (UCX default) | 1 (keep) | Ensure GPU-direct RDMA selects the nearest NIC |
| `UCX_RNDV_SCHEME` | auto | `put_zcopy` | Skip negotiation, perform RDMA write directly |
| `UCX_ZCOPY_THRESH` | auto | Same as RNDV_THRESH | Enable zero-copy transfer |

**R6-16: M-Buffer Microbatch Pipeline**

AFD's `model_forward_afd` already has a microbatch pipeline framework (R4 comments).
In the current pipeline, `postprocess_layer_start_recv()` initiates recv_start early after the A stage,
overlapping the next A-stage computation. This logic is already correct, but is disrupted by R6-13's device-wide synchronization.

After fixing R6-13, the pipeline naturally resumes:

```
M=3 microbatch pipeline (per layer):

Attn side:
  A(L,m0) → send(m0)                     ← send overlaps the next step
            recv_start(L-1,m0)            ← initiate recv early
  A(L,m1) → send(m1)                     ← compute A(m1) overlaps recv(m0)
            recv_start(L-1,m1)
  A(L,m2) → send(m2)
            recv_start(L-1,m2)
  F(L-1,m0) ← recv_wait(m0)              ← m0 recv already completed
  F(L-1,m1) ← recv_wait(m1)
  F(L-1,m2) ← recv_wait(m2)

BufferPool needs at least M buffers in flight simultaneously:
  - Each microbatch has independent send/recv buffers
  - BufferPool caches by shape/dtype, naturally supporting multi-buffer reuse
```

Key prerequisite: after R6-13 is complete, send no longer blocks the entire device, enabling M-way pipeline overlap.

**Execution Order:**
1. R6-13 (CUDA event synchronization) ← highest priority, removes pipeline bottleneck
2. R6-14 (UCX parameter tuning) ← independent, improves individual transfer performance
3. R6-16 (validate pipeline overlap) ← naturally enabled after R6-13, requires profiler validation

### Outstanding Tasks (Full Summary)

| Priority | ID | Description | Source | Status |
|--------|------|------|------|------|
| P1 | L5 | FFN sync flag not cleared when batch=None | Round 5 | **Fixed** |
| P1 | L6 | Potential ZMQ drain last-one-wins batch mismatch | Round 5 | **Fixed** |
| P2 | R6-12 | Multi-node cross-machine IB RDMA testing | Round 6 | Follow-up needed (requires a second machine) |
| P2 | R6-13 | Fine-grained CUDA event synchronization | Round 6 | **Completed** (send_async event + current_stream sync) |
| P2 | R6-14 | UCX parameter tuning | Round 6 | **Completed** (put_zcopy + ZCOPY_THRESH) |
| P2 | R6-18 | Warm BufferPool preallocation | Round 6 | **Completed** |
| P2 | R6-19 | Fully asynchronous send pipeline + fence | Round 6 | **Completed** |
| P2 | F5 | Qwen2 Dense AFD | Round 5 | Not yet implemented |
| P3 | R6-15 | TCP-only fallback fix | Round 6 | To be fixed (low priority) |
| P3 | R5-9 | Avoid creating unused ZMQ sockets on non-rank-0 processes | Round 5 | **Fixed** |
| P4 | R6-16 | M-buffer pipeline profiler validation | Round 6 | Code ready, awaiting profiler |
| P4 | R6-17 | UCXX migration evaluation | Round 6 | Long-term plan |
| P4 | R5-10 | `_dtype_to_int` errors on unknown dtype | Round 5 | **Fixed** |
