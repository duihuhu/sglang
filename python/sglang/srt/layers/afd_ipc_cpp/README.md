# afd_ipc_cpp: High-Performance C++ IPC for AF Disaggregation

## Overview

This library replaces the Python-based IPC communication in AF (Attention-FFN) disaggregation
with a C++ implementation that eliminates Python overhead in the hot path.

### Performance Target

| Component | Python IPC | C++ IPC (CPU_FLAG) | C++ IPC (IPC_EVENT) |
|-----------|-----------|-------------------|---------------------|
| Metadata encode/decode | 150 us | 0 us (cached) | 0 us (cached) |
| event.query() busy-wait | 60 us | 0 us | 0 us |
| P2P copy (NVLink) | 34 us | 34 us | 34 us |
| stream.synchronize() | 27 us | 50 us (CPU flag) | 0 us (GPU event) |
| Python function calls | 150 us | 0 us | 0 us |
| **Per-layer total** | **~560 us** | **~100 us** | **~50 us** |
| **64-layer total** | **~35.8 ms** | **~6.4 ms** | **~3.2 ms** |

Expected TPOT improvement: 80.8ms → ~48-52ms (approaching theoretical minimum of 50.4ms).

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Python Layer (sglang)                                          │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │ CppIpcTensorCommunicator (communicator.py)              │    │
│  │   - Drop-in replacement for IpcTensorCommunicator       │    │
│  │   - Calls C++ hot path via pybind11                     │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ pybind11
┌─────────────────────────────────────────────────────────────────┐
│  C++ Layer (afd_ipc_cpp.so)                                     │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ SHM Manager  │  │ Ring Buffer  │  │ CUDA IPC Events      │  │
│  │ (POSIX shm)  │  │ (4 slots)   │  │ (cross-process sync) │  │
│  └──────────────┘  └──────────────┘  └──────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ GPU Kernels: signal_kernel / wait_kernel                  │   │
│  │ (__threadfence_system + volatile poll)                     │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Sync Modes

### 1. `ipc_event` (Recommended)
- Uses `cudaIpcGetEventHandle` / `cudaIpcOpenEventHandle` for cross-process event sharing
- Sender: `cudaEventRecord(event)` after data copy
- Receiver: `cudaStreamWaitEvent(peer_event)` before consuming data
- **Pure GPU synchronization** — CPU never blocks waiting for GPU
- Latency: ~10us per sync point

### 2. `gpu_signal`
- Uses device memory flags + `__threadfence_system()`
- Sender writes flag on receiver's GPU via P2P after data copy
- Receiver's GPU kernel spin-polls local flag (L2 cache, ~30ns per read)
- **Caveat**: Cross-process volatile reads may have L2 coherence issues on some topologies
- Latency: ~30us per sync point (when working correctly)

### 3. `cpu_flag` (Most Compatible)
- CPU polls POSIX shared memory flag (same as Python IPC)
- Still faster than Python due to eliminated metadata overhead
- Latency: ~50us per sync point

## Build

```bash
# JIT compile (automatic on first import)
python3 -c "from sglang.srt.layers.afd_ipc_cpp import get_module; get_module()"

# Or build explicitly
cd python/sglang/srt/layers/afd_ipc_cpp && make build
```

## Usage

```python
# Automatic (uses C++ if available, falls back to Python)
from sglang.srt.layers.afd_ipc_cpp.communicator import create_ipc_communicator
from sglang.srt.layers.afd_type import AFDPerspective

comm = create_ipc_communicator(AFDPerspective.AFD_PERSPECTIVE_ATTN)

# Or force C++ backend
os.environ["AFD_IPC_CPP"] = "1"
os.environ["AFD_IPC_SYNC_MODE"] = "ipc_event"  # or "cpu_flag" or "gpu_signal"
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AFD_IPC_CPP` | `1` | Enable C++ IPC backend |
| `AFD_IPC_SYNC_MODE` | `ipc_event` | Sync mode: `ipc_event`, `gpu_signal`, `cpu_flag` |
| `AFD_IPC_PEER_DEVICE` | auto | Peer GPU device index |
| `AFD_IPC_PEER_OFFSET` | - | Signed offset from local device to peer |

## Benchmark

```bash
# Compare all backends
make bench-compare

# Single benchmark
python3 benchmark/af_bench/bench_cpp_ipc.py --launch --backend cpp --sync-mode ipc_event
```

## Files

```
sgl-kernel/csrc/afd_ipc/
├── afd_ipc.h              # Header: data structures, class declaration
├── afd_ipc.cpp            # Core implementation: SHM, handshake, send/recv
├── afd_ipc_kernels.cu     # GPU kernels: signal, wait, copy_and_signal
└── afd_ipc_pybind.cpp     # pybind11 bindings + PyAfdIpcComm wrapper

python/sglang/srt/layers/afd_ipc_cpp/
├── __init__.py            # JIT compilation loader
├── communicator.py        # Python wrapper (drop-in for IpcTensorCommunicator)
├── CMakeLists.txt         # CMake build (alternative to JIT)
├── setup.py               # setuptools build
└── Makefile               # Convenience targets
```
