# v7 版本 -- C++ IPC 通信后端 + 消除 CPU-GPU 同步瓶颈

## 概述

本版本实现了从 **Python IPC → C++ IPC** 通信协议的重写，并进一步消除了 `cudaStreamSynchronize` 和 `clone()` 两大瓶颈，将 PD+AF M=1 decode TPOT 从 **117ms 优化到 47ms**，overhead 从 +160% 降到 **+6%**（vs PD TP=1 baseline 44ms）。

核心成果（Qwen3-32B, A800-SXM4-80GB, GPU 4-7）：

| 阶段 | TPOT | vs Baseline | 每层通信开销 |
|------|------|------------|------------|
| 初始 Python IPC | 117.8 ms | +160% | 1.14 ms |
| Fast-Path | 93.3 ms | +106% | 0.75 ms |
| Pre-launch 缓存 | 80.8 ms | +79% | 0.56 ms |
| **C++ IPC (cpu_flag)** | **68.9 ms** | **+56%** | **0.39 ms** |
| **消除 sync+clone (ipc_event)** | **46.6 ms** | **+6%** | **0.04 ms** |

---

## 一、C++ IPC 通信后端 (`sgl-kernel/csrc/afd_ipc/`)

### 1.1 新增文件 (~1231 行 C++/CUDA)

| 文件 | 行数 | 功能 |
|------|------|------|
| `sgl-kernel/csrc/afd_ipc/afd_ipc.h` | 243 | 头文件: 常量定义、类接口、三种同步模式枚举 |
| `sgl-kernel/csrc/afd_ipc/afd_ipc.cpp` | 675 | 核心实现: ring buffer、SHM flag、三种 sync 模式、pybind11 绑定 |
| `sgl-kernel/csrc/afd_ipc/afd_ipc_kernels.cu` | 89 | CUDA 内核: GPU-side signal/wait kernel (`__threadfence_system` + volatile poll) |
| `sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp` | 224 | Python 绑定: `AfdIpcCommunicator` 类暴露给 Python |

### 1.2 Python 集成层

| 文件 | 功能 |
|------|------|
| `python/sglang/srt/layers/afd_ipc_cpp/communicator.py` | `CppIpcTensorCommunicator`: 封装 C++ pybind11 类，提供 `send_tensor(x)` / `recv_tensor()` 接口 |
| `python/sglang/srt/layers/afd_ipc_cpp/__init__.py` | 包初始化 |

### 1.3 架构

```
Python (sglang)
  └── CppIpcTensorCommunicator.send_tensor(x) / recv_tensor()
        │ pybind11 (零 Python 开销热路径)
        ▼
C++ (libafd_ipc_cpp.so)
  ├── SHM ring buffer (4 slots × 32MB, /dev/shm)
  ├── CUDA IPC handle 交换 (Unix socket handshake)
  ├── cudaMemcpyPeerAsync (NVLink P2P copy)
  └── 三种同步模式 (见下)
```

### 1.4 三种同步模式

| 模式 | 机制 | 延迟 (2.5MB round-trip) | 适用场景 |
|------|------|------------------------|---------|
| `cpu_flag` | CPU poll SHM flag + cudaStreamSynchronize | ~84 us | 最兼容，调试用 |
| `ipc_event` | cudaEventRecord + cudaStreamWaitEvent（纯 GPU 同步） | ~94 us | ★ 推荐，无 CPU 阻塞 |
| `gpu_signal` | device memory + __threadfence_system | — | 实验性，跨进程 L2 coherence 问题 |

### 1.5 关键设计

- **Metadata 走 SHM**（CPU 直接读写，不走 GPU buffer），消除 D2H/H2D copy
- **`recv_tensor()` 返回 buffer view**（`torch::from_blob` 不 clone），消除 ~100us/layer 的 clone 开销
- **IPC_EVENT 模式完全不调用 `cudaStreamSynchronize`**：send 端 `cudaEventRecord` 后直接写 SHM flag（不 sync），recv 端 `cudaStreamWaitEvent` 在 compute stream 上等 peer event，GPU 硬件保证顺序

---

## 二、Python IPC 优化 (`ipc_comm.py`)

### 2.1 Pre-launch 缓存

- 首次 send/recv 时缓存 metadata encode/decode 对象，后续调用跳过 Python 临时对象创建
- `event.query()` busy-wait (~10us) 替代 `event.synchronize()` (~100us OS 调度延迟)

### 2.2 SHM GPU 映射

- `cudaHostRegister` + `cudaHostGetDevicePointer` 将 SHM 映射为 GPU 可写内存
- 为后续 GPU-side flag write 做准备

### 2.3 Peer Access 修复

- 通过 `cudaDeviceEnablePeerAccess` (ctypes 调用 CUDA Runtime) 确保 NVLink P2P 可用
- 修复 `_peer_send_info` 中 device_id 跨进程映射错误

---

## 三、AFD Fast-Path 优化 (`afd.py`)

### 3.1 M=1 Fast-Path

检测到 M=1 + 无需 async recv + 无 timing 时，跳过 pipeline 调度逻辑，直接 `for layer in layers: forward_afd_A → forward_afd_F`：

- 消除 ~36ms Python 开销（128 次迭代 × 0.28ms/次）
- 来源: CUDA event 创建、dict 更新、deque 操作、条件判断

### 3.2 Stream-Ordered 接口

新增 `send_stream_ordered` / `recv_stream_ordered` 方法，在 C++ IPC 的 `ipc_event` 模式下无需 CPU sync。

### 3.3 新后端支持

`get_tensor_communicator()` 支持三种后端:
- `ipc` — 原始 Python IPC
- `ipc_cpp` — C++ pybind11 IPC（新增）
- `nccl_p2p` — NCCL P2P（新增，`nccl_p2p_comm.py`）

启用: `--afd-comm-backend ipc_cpp`

---

## 四、逐层性能剖析 (`qwen3.py`)

### 4.1 `SGLANG_LAYER_PROFILE=2` 模式

在 `Qwen3DecoderLayer.forward()` 中插入 5 次 `torch.cuda.synchronize()`，精确测量每层每个组件的 GPU 时间：

| 组件 | 测量内容 |
|------|---------|
| `prep_attn_ms` | RMSNorm + KV cache 准备 |
| `attn_ms` | QKV + score + output projection |
| `prep_mlp_ms` | RMSNorm + IPC send（AF 模式）或 AllReduce（TP 模式） |
| `mlp_ms` | FFN up/gate/down + IPC recv wait（AF 模式） |
| `postprocess_ms` | residual add + AllReduce |

结果存入 `forward_batch._layer_details`，用于定位每层瓶颈（见 §4.3）。

### 4.2 `SGLANG_LAYER_PROFILE=1` 模式

在 `Qwen3Model.forward()` 中逐层 `cuda.synchronize`，输出每层 wall-clock 时间到日志。

---

## 五、NCCL P2P 通信后端 (`nccl_p2p_comm.py`)

### 5.1 设计

新文件 `python/sglang/srt/layers/nccl_p2p_comm.py`（155 行）：

- 使用 `torch.distributed.ProcessGroupNCCL` 创建专用 2-rank communicator
- 首次调用时通过 NCCL send/recv 协商 tensor shape
- 后续调用: `ncclSend`/`ncclRecv` — 纯 stream-ordered，无 CPU-GPU sync
- 握手: `dist.TCPStore` 在 localhost 交换 NCCL 信息

### 5.2 与 C++ IPC 的关系

- NCCL P2P 是备用方案，优势是同套 API 可扩展到跨节点 RDMA
- 当前 C++ IPC 的 `ipc_event` 模式在单机 NVLink 上延迟更低

---

## 六、辅助文件

| 文件 | 功能 |
|------|------|
| `python/sglang/srt/layers/gpu_flag_kernels.py` | Triton kernel: GPU-side flag spin-poll（实验性，延迟 5.1ms vs CPU 77us） |
| `python/sglang/srt/layers/p2p_signal.py` | P2P signal/wait 封装，同进程测试 ~17us/iter |
| `python/sglang/srt/layers/nccl_p2p_comm.py` | NCCL P2P communicator |

---

## 七、Benchmark 目录整理

### 7.1 `disagg_arch_comparison/` 精简

从 ~90 个文件清理到 3 个:
- `run_pdaf_vs_pd.py` — 自包含的 PD+AF (C++ IPC) vs PD TP=1 对比测试脚本
- `breakdown_comparison.md` — 完整优化历程文档（性能结果、瓶颈分析、各阶段对比）
- `optimization_summary.md` — 精简版结果摘要

### 7.2 `ipc_debug/` 目录

新目录，收录 IPC 调试相关的 5 个测试脚本及对应日志:
- `bench_cpp_ipc.py` — C++ IPC 微基准
- `bench_cpp_vs_python_ipc.py` — C++ vs Python IPC 对比
- `bench_pdaf_concurrent.py` — PDAF 并发测试
- `bench_pdaf_cpp_ipc.py` — PDAF C++ IPC 端到端
- `test_cpp_ipc_e2e.py` — C++ IPC 端到端集成测试

### 7.3 Energy benchmark

新增 `benchmark/energy_bench/`，包含 Tier1 系统的 throughput、transition、bubble 测试及真实 GPU 能耗基准。

---

## 八、测试

| 文件 | 功能 |
|------|------|
| `test/srt/test_afd_ipc_cpp.py` | C++ IPC 单元测试 |
| `test/srt/test_afd_ipc_cpp_e2e.py` | C++ IPC 端到端测试 |
| `test/srt/test_afd_ipc_cpp_subprocess.py` | C++ IPC 子进程测试 |
| `test/srt/bench_afd_ipc_cpp.py` | C++ IPC 性能基准 |

---

## 九、修改文件清单

### 9.1 新增文件

| 文件 | 行数 | 说明 |
|------|------|------|
| `sgl-kernel/csrc/afd_ipc/afd_ipc.h` | 243 | C++ IPC 头文件 |
| `sgl-kernel/csrc/afd_ipc/afd_ipc.cpp` | 675 | C++ IPC 核心实现 |
| `sgl-kernel/csrc/afd_ipc/afd_ipc_kernels.cu` | 89 | GPU-side signal/wait kernel |
| `sgl-kernel/csrc/afd_ipc/afd_ipc_pybind.cpp` | 224 | pybind11 绑定 |
| `python/sglang/srt/layers/afd_ipc_cpp/` | — | Python 集成层 |
| `python/sglang/srt/layers/nccl_p2p_comm.py` | 155 | NCCL P2P communicator |
| `python/sglang/srt/layers/gpu_flag_kernels.py` | — | GPU flag 内核 |
| `python/sglang/srt/layers/p2p_signal.py` | — | P2P signal 封装 |
| `benchmark/af_bench/plan/record.md` | — | 开发记录 |
| `test/srt/test_afd_ipc_cpp*.py` | — | C++ IPC 测试 |
| `benchmark/energy_bench/` | — | 能耗基准测试 |

### 9.2 修改文件

| 文件 | 变更 | 说明 |
|------|------|------|
| `python/sglang/srt/layers/ipc_comm.py` | +425/-0 | Pre-launch 缓存、event.query、SHM GPU 映射、peer access 修复 |
| `python/sglang/srt/layers/afd.py` | +204/-0 | M=1 fast-path、stream-ordered 接口、ipc_cpp/nccl_p2p 后端支持 |
| `python/sglang/srt/layers/afd_mixin.py` | +30/-0 | `_afd_timing_enabled` 默认关闭、skip CUDA event 创建 |
| `python/sglang/srt/models/qwen3.py` | +77/-0 | `SGLANG_LAYER_PROFILE` 逐层计时 |
| `python/sglang/srt/managers/scheduler.py` | +52/-0 | AFD event loop 调整 |
| `python/sglang/srt/server_args.py` | +2/-1 | `--afd-comm-backend ipc_cpp` 选项 |
| `python/sglang/srt/energy/af_dvfs_controller.py` | +32/-0 | DVFS 控制器优化 |
| `python/sglang/srt/energy/tier1_solver.py` | +57/-0 | Tier1 求解器更新 |
| `python/sglang/srt/energy/workload_collector.py` | +8/-0 | 工作负载采集 |
| `python/sglang/srt/layers/dvfs.py` | +11/-0 | DVFS 层更新 |

### 9.3 删除文件

| 文件 | 说明 |
|------|------|
| `benchmark/af_bench/disagg_arch_comparison/README.md` | 旧版 README |
| `benchmark/af_bench/disagg_arch_comparison/run_comparison.py` | 旧版对比脚本 (626行，已被 `run_pdaf_vs_pd.py` 取代) |

---

## 十、性能结果总结

### 10.1 单请求 (concurrency=1, output=128)

| Input | C++ IPC 优化后 TPOT | PD TP=1 TPOT | vs PD TP=1 | vs 优化前(69ms) |
|-------|-------------------|-------------|------------|----------------|
| 128 | **47.3 ms** | 44.8 ms | +5.6% | -32.3% |
| 256 | **46.5 ms** | 44.0 ms | +5.7% | -32.5% |
| 512 | **46.6 ms** | 44.0 ms | +5.9% | -32.4% |
| 1024 | **46.6 ms** | 44.0 ms | +5.9% | -32.4% |
| 2048 | **46.8 ms** | 44.1 ms | +6.1% | -32.2% |

### 10.2 并发扫描 (input=512, output=128)

| Concurrency | C++ IPC TPOT | PD TP=1 TPOT | Overhead | AF TTFT | PD TTFT |
|-------------|-------------|-------------|----------|---------|---------|
| 1 | 46.8 ms | 44.3 ms | +5.6% | 174.9 ms | 163.4 ms |
| 2 | 48.3 ms | 45.7 ms | +5.7% | **187.3 ms** | 284.2 ms |
| 4 | 49.2 ms | 46.2 ms | +6.5% | **250.0 ms** | 400.7 ms |
| 8 | 50.9 ms | 47.4 ms | +7.4% | **417.5 ms** | 657.7 ms |

关键发现:
- **TPOT overhead 稳定在 +6~7%**，对比优化前 +90%（Python IPC）和 +56%（C++ IPC cpu_flag）
- **TTFT 在高并发下 AF 显著优于 PD**（conc≥2: AF 快 34-38%），因为 prefill/decode 使用不同 GPU 组

---

## 十一、后续工作

- [ ] 跨节点 RDMA: 将 `cudaMemcpyPeerAsync` 替换为 `ncclSend`/`ncclRecv`（C++ NCCL API），同类架构扩展到 InfiniBand
- [ ] M>1 pipeline overlap 测试: 当前 M=1 已接近理论极限，评估 M=3 在更大 batch 下的收益
- [ ] 整个 64 层循环下沉到 C++: 一次 Python→C++ 调用完成所有层的通信，消除 pybind11 调用开销 (~20us/layer)
- [ ] 高并发稳定性: conc≥16 时 Mooncake KV transfer timeout 排查
