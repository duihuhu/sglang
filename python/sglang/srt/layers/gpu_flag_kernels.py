"""GPU-side spin-poll and flag-write kernels for zero-sync IPC."""
import triton
import triton.language as tl
import torch


@triton.jit
def _spin_wait_flag_kernel(flag_ptr, expected_val: tl.constexpr):
    """GPU kernel that spins until *flag_ptr == expected_val.
    
    flag_ptr: pointer to a int64 value in mapped host memory (SHM).
    The kernel blocks the GPU stream until the flag matches expected_val.
    """
    while tl.load(flag_ptr).to(tl.int64) != expected_val:
        pass


@triton.jit  
def _write_flag_kernel(flag_ptr, val: tl.constexpr):
    """GPU kernel that writes val to *flag_ptr (mapped host memory)."""
    tl.store(flag_ptr, tl.cast(val, tl.int64))


def gpu_spin_wait(flag_tensor: torch.Tensor, expected: int = 1, stream=None):
    """Launch a GPU kernel that spins until flag_tensor[0] == expected.
    
    flag_tensor must be a 1-element int64 tensor on GPU (mapped from SHM).
    The kernel occupies the stream until the condition is met.
    """
    if stream is not None:
        with torch.cuda.stream(stream):
            _spin_wait_flag_kernel[(1,)](flag_tensor, expected)
    else:
        _spin_wait_flag_kernel[(1,)](flag_tensor, expected)


def gpu_write_flag(flag_tensor: torch.Tensor, val: int = 1, stream=None):
    """Launch a GPU kernel that writes val to flag_tensor[0].
    
    flag_tensor must be a 1-element int64 tensor on GPU (mapped from SHM).
    """
    if stream is not None:
        with torch.cuda.stream(stream):
            _write_flag_kernel[(1,)](flag_tensor, val)
    else:
        _write_flag_kernel[(1,)](flag_tensor, val)
