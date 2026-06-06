# v8 版本 -- Async Pipeline (M=2 跨层流水线重叠)

## 概述

本版本在 v7 的 C++ IPC 通信基础上，实现了 **M=2 micro-batch 跨层异步流水线**（`--afd-async-pipeline`），通过多 CUDA stream + event 依赖实现 Attention 与 FFN 计算的跨层重叠，在高并发场景下显著提升吞吐。

核心成果（Qwen3-32B, A800-SXM4-80GB, GPU 4-7, PD+AF 4卡）：

| 并发 | M=1 TPOT | M=2 async TPOT | M=2 吞吐提升 |
|------|----------|----------------|-------------|
| 64 | 63.8 ms | 72.5 ms | -11%（batch 太小，无收益） |
| 128 | 81.3 ms | 78.8 ms | +1.6% |
| 256 | 152.1 ms | 113.9 ms | **+28%** |
| 1024 | 395.2 ms | 241.9 ms | **+63%** |
| 2048 | — | 434.6 ms | 峰值吞吐 3281 tok/s |

---

## 一、新增：AsyncPipelineExecutor (`afd_async_pipeline.py`)

### 1.1 设计思路

M=1 时每层串行执行 `recv → compute → send`，GPU 在等待 IPC 通信时空闲。M=2 async pipeline 将 batch 切为两个 micro-batch，利用多 stream 实现：

```
Layer L:   stream0: Attn(mb0)──────────┐
           stream1:        Attn(mb1)───┤
           comm:   send(mb0) send(mb1) recv(mb0) recv(mb1)
Layer L+1: stream0:                    Attn(mb0)──────────
           stream1:                           Attn(mb1)───
```

FFN 侧同理。关键是 **Layer L 的 send 与 Layer L+1 的 recv 在 comm_stream 上流水线化**，compute stream 通过 CUDA event 等待数据就绪后立即开始计算。

### 1.2 核心文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `python/sglang/srt/layers/afd_async_pipeline.py` | **新增** | AsyncPipelineExecutor 类，多 stream 调度逻辑 |
| `python/sglang/srt/layers/afd.py` | +44 行 | `model_forward_afd` 中检测 `--afd-async-pipeline` 并调用 executor |
| `python/sglang/srt/layers/afd_mixin.py` | +53 行 | 新增 `forward_afd_A_compute` / `forward_afd_F_compute` 纯计算接口 |
| `python/sglang/srt/server_args.py` | +10 行 | 新增 `--afd-async-pipeline` 参数 |

### 1.3 关键实现细节

- **Per-MB CUDA stream**：每个 micro-batch 有独立 compute stream，避免串行等待
- **Event-based 依赖**：`recv_event[mb]` 触发 compute 开始，`compute_done[mb]` 触发 send
- **CPU 不阻塞 GPU**：CPU 只做 SHM flag 轮询（~77μs），GPU 操作全部 stream-ordered
- **兼容 ipc_event 模式**：复用 v7 的 GPU signal/wait kernel 实现零拷贝通信

---

## 二、C++ IPC 扩展：GPU-only send/recv

### 2.1 新增接口

| 方法 | 说明 |
|------|------|
| `send_gpu_only(data_ptr, size, stream)` | 纯 GPU 发送：stream-ordered memcpy + GPU signal kernel，CPU 不阻塞 |
| `recv_gpu_only(out_size, stream)` | 纯 GPU 接收：GPU wait kernel 自旋等待 + stream-ordered memcpy |

### 2.2 MAX_MSG_SIZE 扩容

`MAX_MSG_SIZE` 从 32MB 扩大到 **128MB**，支持大 batch prefill（8192×5120×bf16=80MB）。

### 2.3 Python 绑定扩展

`afd_ipc_pybind.cpp` 新增 `send_gpu_only` / `recv_gpu_only` 绑定（+87 行），以及 `ipc_comm.py` 中的 Python wrapper。

---

## 三、性能分析

### 3.1 收益来源

M=2 async pipeline 的收益来自两个维度的重叠：
1. **层内重叠**：mb0 的 send 与 mb1 的 compute 并行
2. **跨层重叠**：Layer L 的 send/recv 与 Layer L+1 的 compute 流水线化

### 3.2 收益条件

- **batch ≥ 200 时有效**：小 batch 时 2×FFN(N/2) 的 GEMM 效率损失 > overlap 收益
- **拐点在 conc≈64**：此时 decode batch 开始超过 64，M=2 开始追平 M=1
- **conc≥256 时显著**：M=2 吞吐比 M=1 高 28-63%

### 3.3 与 PD TP=2 对比

| 指标 | PD TP=2 | PD+AF M=2 | 差距 |
|------|---------|-----------|------|
| 峰值吞吐 | 3704 tok/s | 3281 tok/s | -11% |
| 达到峰值的并发 | 768 | 2048 | AF 需要更高并发 |
| 单请求 TPOT | 39.5 ms | 45.8 ms | +16% |

AF M=2 在极高并发下接近 TP=2 的吞吐，但单请求延迟仍有 16% 的 IPC 通信开销。

---

## 四、使用方式

```bash
sglang serve /models/Qwen/Qwen3-32B/ \
  --afd-perspective attn \
  --afd-comm-backend ipc_cpp \
  --afd-micro-batch 2 \
  --afd-async-pipeline \
  --afd-disagg-interleave-poll
```

---

## 五、测试数据

完整数据见 `benchmark/af_bench/micro-batch-opt/results/all_concurrency_results.csv`（57 条记录，覆盖 4 种架构、并发 1~2048）。
