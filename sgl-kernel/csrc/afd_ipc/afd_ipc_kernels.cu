/**
 * GPU-side signal/wait kernels for zero-CPU-sync IPC communication.
 *
 * Strategy: CUDA IPC Event for cross-process GPU synchronization.
 * - Sender: cudaEventRecord(local_event) after data copy
 * - Receiver: cudaStreamWaitEvent(peer_event) before consuming data
 * - No CPU synchronize() needed — pure GPU stream ordering
 *
 * Fallback: __threadfence_system + device memory flag polling
 * - Works within same process or when IPC events unavailable
 * - Higher latency (~128us) but simpler setup
 */

#include <cuda_runtime.h>
#include <cstdint>

namespace afd_ipc {

// Signal kernel: ensures all prior writes are visible system-wide, then writes flag
__global__ void signal_kernel(volatile int64_t* flag_ptr, int64_t value) {
    __threadfence_system();
    *flag_ptr = value;
}

// Wait kernel: spin-polls flag using volatile read (bypasses L1, reads from L2/DRAM)
// Uses backoff to reduce memory bus contention
__global__ void wait_kernel(volatile int64_t* flag_ptr, int64_t expected) {
    int backoff = 1;
    while (*flag_ptr != expected) {
#if __CUDA_ARCH__ >= 700
        // nanosleep available on sm_70+
        __nanosleep(100 * backoff);
#else
        for (int i = 0; i < backoff * 10; ++i) {
            asm volatile("" ::: "memory");
        }
#endif
        if (backoff < 128) backoff <<= 1;
    }
    // Final threadfence to ensure subsequent reads see sender's data
    __threadfence_system();
}

// Combined copy + signal: copies data then signals (for small payloads like flags)
// This avoids launching two separate kernels
__global__ void copy_and_signal_kernel(
    void* __restrict__ dst,
    const void* __restrict__ src,
    size_t bytes,
    volatile int64_t* flag_ptr,
    int64_t value
) {
    // Simple byte copy (for small metadata/flags only, not bulk data)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = blockDim.x * gridDim.x;
    char* d = (char*)dst;
    const char* s = (const char*)src;
    for (size_t i = tid; i < bytes; i += stride) {
        d[i] = s[i];
    }
    __syncthreads();
    if (tid == 0) {
        __threadfence_system();
        *flag_ptr = value;
    }
}

// Host-callable wrappers

void launch_signal_kernel(volatile int64_t* flag_ptr, int64_t value, cudaStream_t stream) {
    signal_kernel<<<1, 1, 0, stream>>>(flag_ptr, value);
}

void launch_wait_kernel(volatile int64_t* flag_ptr, int64_t expected, cudaStream_t stream) {
    wait_kernel<<<1, 1, 0, stream>>>(flag_ptr, expected);
}

void launch_copy_and_signal(
    void* dst, const void* src, size_t bytes,
    volatile int64_t* flag_ptr, int64_t value,
    cudaStream_t stream
) {
    int threads = (bytes < 1024) ? 32 : 256;
    int blocks = 1;
    copy_and_signal_kernel<<<blocks, threads, 0, stream>>>(
        dst, src, bytes, flag_ptr, value);
}

}  // namespace afd_ipc
