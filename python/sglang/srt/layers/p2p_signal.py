"""GPU-side P2P signal/wait kernels for zero-sync IPC communication.

Uses device memory flags + __threadfence_system() to achieve cross-GPU
synchronization without any CPU involvement.

Key idea:
- Flag lives on RECEIVER's GPU (device memory, L2 cache accessible)
- Sender writes data via P2P, then __threadfence_system(), then P2P writes flag=1
- Receiver's kernel spin-polls flag using volatile load (bypasses L1 cache)
- Once flag=1, receiver knows data is visible and proceeds
"""
import torch
import os
from torch.utils.cpp_extension import load_inline

# CUDA source for signal/wait kernels
_CUDA_SRC = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDAStream.h>

// Kernel: write flag value after ensuring all prior writes are visible system-wide
__global__ void signal_kernel(volatile int64_t* flag_ptr, int64_t value) {
    __threadfence_system();
    *flag_ptr = value;
}

// Kernel: spin-poll flag using volatile read (bypasses L1 cache, reads from L2/DRAM)
__global__ void wait_kernel(volatile int64_t* flag_ptr, int64_t expected) {
    while (*flag_ptr != expected) {
        // volatile ensures fresh read from L2/DRAM each iteration
    }
}

void signal_flag(torch::Tensor flag_tensor, int64_t value) {
    TORCH_CHECK(flag_tensor.is_cuda(), "flag must be on CUDA device");
    TORCH_CHECK(flag_tensor.dtype() == torch::kInt64, "flag must be int64");
    
    auto stream = c10::cuda::getCurrentCUDAStream(flag_tensor.device().index());
    signal_kernel<<<1, 1, 0, stream.stream()>>>(
        (volatile int64_t*)flag_tensor.data_ptr<int64_t>(),
        value
    );
}

void wait_flag(torch::Tensor flag_tensor, int64_t expected) {
    TORCH_CHECK(flag_tensor.is_cuda(), "flag must be on CUDA device");
    TORCH_CHECK(flag_tensor.dtype() == torch::kInt64, "flag must be int64");
    
    auto stream = c10::cuda::getCurrentCUDAStream(flag_tensor.device().index());
    wait_kernel<<<1, 1, 0, stream.stream()>>>(
        (volatile int64_t*)flag_tensor.data_ptr<int64_t>(),
        expected
    );
}

void reset_flag(torch::Tensor flag_tensor) {
    TORCH_CHECK(flag_tensor.is_cuda(), "flag must be on CUDA device");
    auto stream = c10::cuda::getCurrentCUDAStream(flag_tensor.device().index());
    signal_kernel<<<1, 1, 0, stream.stream()>>>(
        (volatile int64_t*)flag_tensor.data_ptr<int64_t>(),
        0
    );
}
"""

_CPP_SRC = r"""
#include <torch/extension.h>

void signal_flag(torch::Tensor flag_tensor, int64_t value);
void wait_flag(torch::Tensor flag_tensor, int64_t expected);
void reset_flag(torch::Tensor flag_tensor);
"""

# Lazy-load the extension
_module = None

def _get_module():
    global _module
    if _module is None:
        _module = load_inline(
            name="p2p_signal_wait",
            cpp_sources=[_CPP_SRC],
            cuda_sources=[_CUDA_SRC],
            functions=["signal_flag", "wait_flag", "reset_flag"],
            verbose=False,
            extra_cuda_cflags=["-O3"],
        )
    return _module


def signal(flag_tensor: torch.Tensor, value: int = 1):
    """Signal: __threadfence_system() then write flag=value.
    
    Call on SENDER's stream after data copy. Ensures all prior P2P writes
    are visible to the receiver before the flag is set.
    
    flag_tensor: int64 tensor on RECEIVER's GPU (accessed via P2P).
    """
    _get_module().signal_flag(flag_tensor, value)


def wait(flag_tensor: torch.Tensor, expected: int = 1):
    """Wait: volatile spin-poll until flag==expected.
    
    Call on RECEIVER's stream before consuming data. Blocks the GPU stream
    (not CPU!) until the sender signals.
    
    flag_tensor: int64 tensor on THIS GPU (receiver's device memory).
    """
    _get_module().wait_flag(flag_tensor, expected)


def reset(flag_tensor: torch.Tensor):
    """Reset flag to 0. Call on receiver after consuming data."""
    _get_module().reset_flag(flag_tensor)
