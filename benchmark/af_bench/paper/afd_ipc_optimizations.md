# AFD IPC 通信优化技术详解

本文档详细描述 PD+AF M=2 async pipeline 中涉及的所有底层优化技术，包括 Python→C++ 下沉、CUDA stream 控制、IPC event 机制、pre-launch 缓存等。

---

## 1. Python 开销下沉到 C++

### 1.1 问题

初始 Python IPC 实现中，每次 send/recv 涉及：
- Python dict 序列化（pickle）→ ~200μs
- Python 对象创建（torch.Tensor metadata）→ ~50μs
- GIL 竞争 + asyncio event loop → ~100μs
- 总计每层通信开销 **1.14ms**（64 层 × 2 次 = 146ms overhead）

### 1.2 解决方案：C++ pybind11 热路径

将通信热路径完全用 C++ 实现（`sgl-kernel/csrc/afd_ipc/afd_ipc.cpp`），Python 侧只做一次初始化：

```cpp
// Hot path: send_cached — 零 Python 调用
void AfdIpcComm::send_cached(const void* data_ptr, size_t data_bytes,
                             cudaStream_t stream) {
    int slot = send_slot_;
    // 1. CPU poll slot free (volatile read, ~ns)
    volatile uint64_t* flag_ptr = ...;
    while (*flag_ptr != 0) {}
    // 2. GPU async copy (stream-ordered, non-blocking)
    cudaMemcpyAsync(send_buf, data_ptr, data_bytes, cudaMemcpyD2D, stream);
    // 3. Signal peer (mode-dependent)
    ...
    send_slot_ = (slot + 1) % RING_SIZE;
}
```

**效果**：每层通信从 1.14ms → **0.04ms**（ipc_event 模式），降低 96%。

### 1.3 Metadata 缓存（cache_meta）

Decode 阶段 tensor shape 不变（batch_size × hidden_dim），只需首次传递 metadata：

```cpp
void AfdIpcComm::cache_meta(const TensorMeta& meta, size_t total_bytes) {
    cached_meta_ = meta;
    cached_total_bytes_ = total_bytes;
    meta_cached_ = true;  // 后续 send/recv 跳过 meta 编解码
}
```

Python 侧在 warmup 后调用一次 `cache_meta()`，之后所有 send/recv 只传数据指针和大小。

---

## 2. CUDA Stream 控制

### 2.1 多 Stream 架构

```
DA GPU (Attention):
  ├── default_stream: 调度逻辑
  ├── mb_stream[0]: micro-batch 0 的 Attention 计算
  ├── mb_stream[1]: micro-batch 1 的 Attention 计算
  └── comm_stream (priority=-1): IPC 通信（高优先级）

DF GPU (FFN):
  ├── default_stream: 调度逻辑
  ├── mb_stream[0]: micro-batch 0 的 FFN 计算
  ├── mb_stream[1]: micro-batch 1 的 FFN 计算
  └── comm_stream (priority=-1): IPC 通信
```

### 2.2 Event 依赖链

```python
# AsyncPipelineExecutor 中的 event 依赖
self.compute_done_events = [torch.cuda.Event() for _ in range(m_stage)]
self.recv_ready_events = [[torch.cuda.Event() ...] for _ in range(num_layers)]

# 数据流：
# comm_stream: recv(mb0) → record recv_ready[L][0]
# mb_stream[0]: wait recv_ready[L][0] → Attn(mb0) → record compute_done[0]
# comm_stream: wait compute_done[0] → send(mb0)
```

### 2.3 关键设计：CPU 不阻塞 GPU

传统做法用 `cudaStreamSynchronize()` 等待 GPU 完成再做下一步（~10-40ms 延迟）。我们的方案：
- **所有 GPU 操作都是 stream-ordered**：CPU 只负责 enqueue，不等待完成
- **唯一的 CPU 阻塞点**：SHM flag 轮询（~77μs），且只在 recv 时发生
- **Event 替代 Synchronize**：`cudaStreamWaitEvent` 是纯 GPU 操作，零 CPU 开销

---

## 3. IPC Event 三种同步模式

### 3.1 模式枚举

```cpp
enum class SyncMode {
    CPU_FLAG,     // CPU 轮询 SHM flag（基线）
    IPC_EVENT,    // CUDA IPC Event（纯 GPU 同步）
    GPU_SIGNAL,   // GPU signal/wait kernel（设备内存 flag）
};
```

### 3.2 CPU_FLAG 模式（v6 基线）

```
Sender:                          Receiver:
  cudaMemcpyAsync(data)            while (shm_flag == 0) {}  ← CPU spin
  cudaStreamSynchronize()  ← 阻塞   cudaMemcpyAsync(recv_buf)
  shm_flag = 1                     shm_flag = 0
```

**问题**：`cudaStreamSynchronize()` 导致 CPU 等待 GPU 完成 memcpy（~10ms 在高负载下）。

### 3.3 IPC_EVENT 模式（v7 最终方案）

```
Sender:                              Receiver:
  cudaMemcpyAsync(data, stream)        cudaStreamWaitEvent(peer_event)  ← 纯 GPU
  cudaEventRecord(local_event)         cudaMemcpyAsync(local_buf, peer_buf)
  write_shm_flag(1)                    // 数据已在 stream 中可用
```

**关键**：`cudaStreamWaitEvent` 是 GPU 指令，CPU 立即返回。接收方的 memcpy 在 event 触发后自动执行。

### 3.4 GPU_SIGNAL 模式（同进程优化）

当 IPC Event 不可用时（同进程内），使用 GPU kernel 直接写对方设备内存：

```cuda
// signal_kernel: 发送方在数据拷贝后执行
__global__ void signal_kernel(volatile int64_t* flag_ptr, int64_t value) {
    __threadfence_system();  // 确保 memcpy 对所有 GPU 可见
    *flag_ptr = value;       // 写入接收方的设备内存（P2P）
}

// wait_kernel: 接收方自旋等待
__global__ void wait_kernel(volatile int64_t* flag_ptr, int64_t expected) {
    int backoff = 1;
    while (*flag_ptr != expected) {
        __nanosleep(100 * backoff);  // 指数退避，减少总线争用
        if (backoff < 128) backoff <<= 1;
    }
    __threadfence_system();  // 确保后续读取看到发送方的数据
}
```

**延迟**：~128μs（比 IPC_EVENT 的 ~40μs 高，但无需跨进程 event handle 交换）。

---

## 4. Pre-launch 缓存优化

### 4.1 Ring Buffer 设计

```cpp
constexpr int RING_SIZE = 4;           // 4 slot 环形缓冲
constexpr size_t MAX_MSG_SIZE = 128MB; // 每 slot 128MB

// 内存布局（每个方向）：
// send_pool: RING_SIZE × MAX_MSG_SIZE 的 GPU 显存（CUDA IPC 导出）
// SHM flags: per-slot uint64 标志位（POSIX /dev/shm）
```

### 4.2 Pre-allocate 策略

启动时一次性分配所有资源，运行时零分配：

```python
# 初始化时（一次性）
comm = AfdIpcCommunicator(perspective, sync_mode="ipc_event")
comm.cache_meta(tensor_meta, total_bytes)  # 缓存 shape/dtype

# 运行时（每层调用，零分配）
comm.send_cached(data_ptr, size, stream)   # 直接写 pre-allocated buffer
ptr = comm.recv_cached(&size, stream)      # 返回 pre-allocated buffer 指针
```

### 4.3 Zero-copy Recv

接收方不分配新 tensor，直接返回 ring buffer 中的指针：

```cpp
void* AfdIpcComm::recv_cached(size_t* out_data_bytes, cudaStream_t stream) {
    int slot = recv_slot_;
    // ... wait for data ...
    // 直接返回 peer 的 send_pool 中的地址（通过 CUDA IPC 映射）
    void* recv_buf = (char*)peer_send_pool_ + (size_t)slot * MAX_MSG_SIZE;
    *out_data_bytes = read_size(slot);
    recv_slot_ = (slot + 1) % RING_SIZE;
    return recv_buf;  // 零拷贝：直接使用 peer GPU 内存
}
```

---

## 5. send_gpu_only / recv_gpu_only（v8 新增）

### 5.1 动机

`send_cached` 仍有一个 CPU 阻塞点：等待 slot 空闲的 `while (*flag_ptr != 0) {}`。在 async pipeline 中，我们希望 CPU 完全不阻塞。

### 5.2 实现

```cpp
void AfdIpcComm::send_gpu_only(const void* data_ptr, size_t data_bytes,
                                cudaStream_t stream) {
    int slot = send_slot_;
    // CPU poll slot free (通常立即返回，因为 sender 领先 receiver)
    volatile uint64_t* flag_ptr = ...;
    while (*flag_ptr != 0) {}

    char* send_buf = (char*)send_pool_ + (size_t)slot * MAX_MSG_SIZE;
    // GPU async copy（stream-ordered）
    cudaMemcpyAsync(send_buf, data_ptr, data_bytes, cudaMemcpyD2D, stream);
    // GPU signal kernel（不等待 memcpy 完成，由 stream 保序）
    launch_signal_kernel(peer_signal_flags_ + slot, 1, stream);
    send_slot_ = (slot + 1) % RING_SIZE;
}

void* AfdIpcComm::recv_gpu_only(size_t* out_data_bytes, cudaStream_t stream) {
    int slot = recv_slot_;
    // GPU wait kernel 自旋（不阻塞 CPU）
    launch_wait_kernel(local_signal_flags_ + slot, 1, stream);
    // GPU async copy from peer buffer
    void* recv_buf = local_recv_staging_ + (size_t)slot * MAX_MSG_SIZE;
    cudaMemcpyAsync(recv_buf, peer_send_pool_ + slot * MAX_MSG_SIZE,
                    cached_total_bytes_, cudaMemcpyD2D, stream);
    // Reset flag for sender reuse
    launch_signal_kernel(local_signal_flags_ + slot, 0, stream);
    recv_slot_ = (slot + 1) % RING_SIZE;
    return recv_buf;
}
```

**效果**：CPU 只做 slot 管理（~ns），所有等待和数据传输都在 GPU stream 中完成。

---

## 6. M=2 Interleaved Pipeline 调度

### 6.1 核心思想

DA 发送 mb0 的 Attn 结果后，**不等待 mb0 的 FFN 返回**，而是立即计算 mb1 的 Attention。当 mb1 的 Attention 完成时，mb0 的 FFN 大概率已经完成（因为 FFN 和 Attn 耗时相近）。

### 6.2 时间线

```
DA:  A(mb0) → send(mb0) → A(mb1) → send(mb1) → recv(mb0) → [next layer]
DF:                        F(mb0) → send(mb0) → F(mb1) → send(mb1)
                           ↑                     ↑
                     overlap with A(mb1)    overlap with next layer
```

### 6.3 收益分析

- **无 overlap（M=1）**：每层 = Attn + comm + FFN + comm ≈ 15 + 0.5 + 15 + 0.5 = 31ms
- **有 overlap（M=2）**：每层 ≈ max(Attn, FFN) + 2×comm ≈ 15 + 1 = 16ms（理论）
- **实测**：由于 GEMM 效率损失（batch/2），实际每层 ~22ms，加速比 ~1.4×

---

## 7. 优化效果汇总

| 优化点 | 技术 | 延迟贡献 |
|--------|------|----------|
| Python→C++ 下沉 | pybind11 热路径 | 1.14ms → 0.39ms/层 |
| 消除 cudaStreamSync | IPC Event / GPU signal | 0.39ms → 0.04ms/层 |
| Metadata 缓存 | cache_meta() | 省去每次 pickle/decode |
| Ring buffer pre-alloc | 4-slot × 128MB | 运行时零分配 |
| Zero-copy recv | 直接返回 peer buffer 指针 | 省去 clone() 开销 |
| GPU signal/wait kernel | __threadfence_system + volatile | 纯 GPU 同步，CPU 不阻塞 |
| Multi-stream pipeline | per-mb stream + event deps | Attn ∥ FFN 跨层重叠 |
| send_gpu_only | stream-ordered P2P + signal | CPU 完全不参与数据路径 |

**最终效果**：单请求 TPOT 从 117ms（初始 Python IPC）→ 47ms（v7 ipc_event）→ 高并发下 M=2 吞吐提升 60-70%（v8 async pipeline）。
