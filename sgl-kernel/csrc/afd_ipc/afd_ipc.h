/**
 * libafd_ipc: High-performance C++ IPC communication library for AF disaggregation.
 *
 * Eliminates Python overhead in the AF communication hot path by implementing:
 * 1. Direct SHM ring buffer management (POSIX shm + mmap)
 * 2. CUDA IPC Event cross-process synchronization (pure GPU sync, no CPU round-trip)
 * 3. GPU-side signal/wait kernels (__threadfence_system + volatile poll)
 * 4. Zero-copy send/recv with pre-cached metadata
 *
 * Target: reduce per-layer IPC overhead from ~560us (Python) to ~10-30us (C++ + GPU sync).
 */
#pragma once

#include <cuda_runtime.h>
#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <atomic>

namespace afd_ipc {

// Ring buffer configuration
constexpr int RING_SIZE = 4;
constexpr int HEADER_BYTES = 64;
constexpr size_t MAX_MSG_SIZE = 32 * 1024 * 1024;  // 32 MB (enough for prefill: 2048×5120×bf16=20MB)

// SHM layout: per-slot flags + sizes + metadata for bidirectional communication
// Layout (per direction):
//   flags[RING_SIZE]:    uint64 × 4 = 32 bytes
//   sizes[RING_SIZE]:    uint64 × 4 = 32 bytes
//   metadata[RING_SIZE]: TensorMeta × 4 = 64 × 4 = 256 bytes
// Total per direction: 320 bytes, two directions: 640 bytes
constexpr int SHM_FLAGS_A2F = 0;      // offset   0: flag_a2f[0..3]
constexpr int SHM_FLAGS_F2A = 32;     // offset  32: flag_f2a[0..3]
constexpr int SHM_SIZES_A2F = 64;     // offset  64: size_a2f[0..3]
constexpr int SHM_SIZES_F2A = 96;     // offset  96: size_f2a[0..3]
constexpr int SHM_META_A2F = 128;     // offset 128: meta_a2f[0..3] (64B each)
constexpr int SHM_META_F2A = 384;     // offset 384: meta_f2a[0..3] (64B each)
constexpr int SHM_TOTAL = 640;

// Metadata header layout (64 bytes = 8 × int64)
// Must match Python _encode_meta in rdma_comm.py:
// [ndim, shape[0], shape[1], ..., shape[ndim-1], dtype_code, original_num_tokens, ...]
struct TensorMeta {
    int64_t data[8];  // raw int64 array matching Python layout
};

// Dtype encoding (matches Python _DTYPE_TO_INT in rdma_comm.py)
enum DtypeCode : int64_t {
    DTYPE_FP16 = 0,
    DTYPE_BF16 = 1,
    DTYPE_FP32 = 2,
    DTYPE_FP64 = 3,
    DTYPE_INT32 = 4,
    DTYPE_INT64 = 5,
};

// GPU-side synchronization mode
enum class SyncMode {
    CPU_FLAG,       // CPU polls SHM flag (current baseline)
    IPC_EVENT,      // CUDA IPC Event (cudaStreamWaitEvent, pure GPU sync)
    GPU_SIGNAL,     // GPU-side signal/wait kernel (device memory flag)
};

/**
 * IPC Channel: manages one direction of communication (A→F or F→A).
 * Each channel owns:
 * - A send buffer pool (RING_SIZE × MAX_MSG_SIZE) on local GPU
 * - SHM flags for signaling
 * - CUDA IPC events for GPU-only synchronization
 */
struct IpcChannel {
    // Local GPU send buffer pool (exported via CUDA IPC)
    void* send_pool_ptr;
    size_t send_pool_size;

    // Peer's send buffer (imported via CUDA IPC handle)
    void* peer_send_pool_ptr;

    // SHM mapping
    volatile uint64_t* shm_base;

    // CUDA IPC events for cross-process GPU sync
    cudaEvent_t local_events[RING_SIZE];
    cudaIpcEventHandle_t local_event_handles[RING_SIZE];
    cudaEvent_t peer_events[RING_SIZE];

    // GPU-side signal flags (device memory on receiver's GPU)
    int64_t* signal_flags;  // RING_SIZE int64 values on local device

    // Ring buffer state
    int send_slot;
    int recv_slot;

    // Cached metadata for decode fast-path
    bool meta_cached;
    TensorMeta cached_meta;
    size_t cached_total_bytes;

    // Device info
    int local_device;
    int peer_device;
    cudaStream_t comm_stream;
};

/**
 * Main communicator class.
 * Manages bidirectional IPC between ATTN and FFN processes.
 */
class AfdIpcComm {
public:
    /**
     * @param is_ffn: true if this is the FFN side, false for ATTN
     * @param local_device: local CUDA device index
     * @param peer_device: peer CUDA device index
     * @param rank: rank for SHM/socket naming
     * @param mb_id: microbatch ID (-1 for legacy single comm)
     * @param sync_mode: synchronization strategy
     */
    AfdIpcComm(bool is_ffn, int local_device, int peer_device,
               int rank, int mb_id, SyncMode sync_mode);
    ~AfdIpcComm();

    // Handshake: exchange CUDA IPC handles with peer
    void handshake();

    // Send tensor data (hot path, no Python involvement)
    // data_ptr: pointer to contiguous tensor data on local GPU
    // meta: pre-encoded metadata (shape, dtype)
    // data_bytes: size of tensor data in bytes
    void send(const void* data_ptr, const TensorMeta& meta,
              size_t data_bytes, cudaStream_t stream);

    // Send with pre-cached metadata (skip meta encoding after first call)
    void send_cached(const void* data_ptr, size_t data_bytes,
                     cudaStream_t stream);

    // Receive tensor data (hot path)
    // Returns pointer to received data in recv buffer (valid until next recv on same slot)
    // out_meta: filled with tensor metadata
    // out_data_bytes: filled with data size
    void* recv(TensorMeta* out_meta, size_t* out_data_bytes,
               cudaStream_t stream);

    // Receive with cached metadata (skip meta decoding after first call)
    void* recv_cached(size_t* out_data_bytes, cudaStream_t stream);

    // Cache metadata for subsequent send_cached/recv_cached calls
    void cache_meta(const TensorMeta& meta, size_t total_bytes);

    // Get sync mode
    SyncMode sync_mode() const { return sync_mode_; }

    // Check if handshake is complete
    bool is_ready() const { return ready_; }

    // Accessors
    int local_device() const { return local_device_; }
    int peer_device() const { return peer_device_; }

private:
    void setup_shm();
    void setup_cuda_buffers();
    void setup_ipc_events();
    void setup_gpu_signals();

    void exchange_handles_server();  // FFN side: listen
    void exchange_handles_client();  // ATTN side: connect

    // SHM offset helpers
    inline int flag_offset(int slot, bool is_send) const;
    inline int size_offset(int slot, bool is_send) const;
    inline int meta_offset(int slot, bool is_send) const;

    // SHM flag helpers
    void write_flag(int slot, uint64_t value);
    uint64_t read_flag(int slot);
    void write_size(int slot, uint64_t size);
    uint64_t read_size(int slot);
    void write_meta_shm(int slot, const TensorMeta& meta);
    void read_meta_shm(int slot, TensorMeta* out_meta);

    // Sync primitives
    void signal_peer(int slot, cudaStream_t stream);
    void wait_peer(int slot, cudaStream_t stream);

    bool is_ffn_;
    int local_device_;
    int peer_device_;
    int rank_;
    int mb_id_;
    SyncMode sync_mode_;
    bool ready_;

    // SHM
    int shm_fd_;
    void* shm_ptr_;
    std::string shm_path_;

    // Send pool (local GPU memory, exported via IPC)
    void* send_pool_;
    cudaIpcMemHandle_t send_pool_handle_;

    // Recv buffer (local GPU memory, not shared)
    void* recv_pool_;

    // Peer's send pool (imported via IPC handle)
    void* peer_send_pool_;

    // CUDA IPC events
    cudaEvent_t send_events_[RING_SIZE];
    cudaIpcEventHandle_t send_event_handles_[RING_SIZE];
    cudaEvent_t peer_events_[RING_SIZE];

    // GPU signal flags (device memory)
    int64_t* local_signal_flags_;   // on local device (receiver polls these)
    int64_t* peer_signal_flags_;    // on peer device (sender writes these via P2P)

    // Ring state
    int send_slot_;
    int recv_slot_;

    // Metadata cache
    bool meta_cached_;
    TensorMeta cached_meta_;
    size_t cached_total_bytes_;

    // Communication stream
    cudaStream_t comm_stream_;

    // Socket path for handshake
    std::string socket_path_;
};

// GPU kernel declarations (implemented in afd_ipc_kernels.cu)
void launch_signal_kernel(volatile int64_t* flag_ptr, int64_t value, cudaStream_t stream);
void launch_wait_kernel(volatile int64_t* flag_ptr, int64_t expected, cudaStream_t stream);
void launch_copy_and_signal(void* dst, const void* src, size_t bytes,
                            volatile int64_t* flag_ptr, int64_t value,
                            cudaStream_t stream);

}  // namespace afd_ipc
