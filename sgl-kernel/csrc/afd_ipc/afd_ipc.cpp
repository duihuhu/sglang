/**
 * AfdIpcComm implementation: high-performance IPC for AF disaggregation.
 *
 * Hot path (send_cached / recv_cached) is pure C++ with zero Python calls:
 * 1. Check ring slot availability (volatile SHM read, ~10ns)
 * 2. cudaMemcpyAsync data to send_buf (stream-ordered, non-blocking)
 * 3. Signal peer via CUDA IPC Event or GPU flag write
 *
 * Cold path (first send/recv): caches metadata, sets up IPC events.
 */

#include "afd_ipc.h"

#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <thread>
#include <chrono>

namespace afd_ipc {

#define CUDA_CHECK(call)                                                    \
    do {                                                                     \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            throw std::runtime_error(                                        \
                std::string("CUDA error: ") + cudaGetErrorString(err) +     \
                " at " + __FILE__ + ":" + std::to_string(__LINE__));         \
        }                                                                   \
    } while (0)

AfdIpcComm::AfdIpcComm(bool is_ffn, int local_device, int peer_device,
                       int rank, int mb_id, SyncMode sync_mode)
    : is_ffn_(is_ffn),
      local_device_(local_device),
      peer_device_(peer_device),
      rank_(rank),
      mb_id_(mb_id),
      sync_mode_(sync_mode),
      ready_(false),
      shm_fd_(-1),
      shm_ptr_(nullptr),
      send_pool_(nullptr),
      recv_pool_(nullptr),
      peer_send_pool_(nullptr),
      local_signal_flags_(nullptr),
      peer_signal_flags_(nullptr),
      send_slot_(0),
      recv_slot_(0),
      meta_cached_(false),
      cached_total_bytes_(0),
      comm_stream_(nullptr) {

    memset(send_events_, 0, sizeof(send_events_));
    memset(send_event_handles_, 0, sizeof(send_event_handles_));
    memset(peer_events_, 0, sizeof(peer_events_));

    CUDA_CHECK(cudaSetDevice(local_device_));

    // Create communication stream (high priority for low latency)
    int least_priority, greatest_priority;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    CUDA_CHECK(cudaStreamCreateWithPriority(&comm_stream_, cudaStreamNonBlocking, greatest_priority));

    // Enable peer access
    int can_access = 0;
    CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access, local_device_, peer_device_));
    if (can_access) {
        cudaError_t err = cudaDeviceEnablePeerAccess(peer_device_, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
            // Non-fatal: P2P copy will fall back to staged transfer
        }
    }

    setup_shm();
    setup_cuda_buffers();

    if (sync_mode_ == SyncMode::IPC_EVENT) {
        setup_ipc_events();
    } else if (sync_mode_ == SyncMode::GPU_SIGNAL) {
        setup_gpu_signals();
    }

    // Build socket path for handshake
    if (mb_id_ < 0) {
        socket_path_ = "/tmp/afd_ipc_cpp_" + std::to_string(rank_) + ".sock";
    } else {
        socket_path_ = "/tmp/afd_ipc_cpp_" + std::to_string(rank_) +
                       "_mb" + std::to_string(mb_id_) + ".sock";
    }
}

AfdIpcComm::~AfdIpcComm() {
    if (comm_stream_) {
        cudaStreamDestroy(comm_stream_);
    }
    if (send_pool_) {
        cudaSetDevice(local_device_);
        cudaFree(send_pool_);
    }
    if (recv_pool_) {
        cudaSetDevice(local_device_);
        cudaFree(recv_pool_);
    }
    if (local_signal_flags_) {
        cudaSetDevice(local_device_);
        cudaFree(local_signal_flags_);
    }
    for (int i = 0; i < RING_SIZE; ++i) {
        if (send_events_[i]) cudaEventDestroy(send_events_[i]);
        if (peer_events_[i]) cudaEventDestroy(peer_events_[i]);
    }
    if (shm_ptr_) {
        munmap(shm_ptr_, SHM_TOTAL);
    }
    if (shm_fd_ >= 0) {
        close(shm_fd_);
    }
    if (!shm_path_.empty()) {
        unlink(shm_path_.c_str());
    }
    if (!socket_path_.empty()) {
        unlink(socket_path_.c_str());
    }
}

void AfdIpcComm::setup_shm() {
    if (mb_id_ < 0) {
        shm_path_ = "/dev/shm/afd_ipc_cpp_flags_" + std::to_string(rank_);
    } else {
        shm_path_ = "/dev/shm/afd_ipc_cpp_flags_" + std::to_string(rank_) +
                     "_mb" + std::to_string(mb_id_);
    }

    // Create or open SHM
    shm_fd_ = open(shm_path_.c_str(), O_CREAT | O_RDWR, 0600);
    if (shm_fd_ < 0) {
        throw std::runtime_error("Failed to open SHM: " + shm_path_);
    }
    ftruncate(shm_fd_, SHM_TOTAL);

    shm_ptr_ = mmap(nullptr, SHM_TOTAL, PROT_READ | PROT_WRITE,
                    MAP_SHARED, shm_fd_, 0);
    if (shm_ptr_ == MAP_FAILED) {
        throw std::runtime_error("mmap failed for SHM");
    }

    // Zero all flags
    memset(shm_ptr_, 0, SHM_TOTAL);
}

void AfdIpcComm::setup_cuda_buffers() {
    size_t pool_size = (size_t)RING_SIZE * MAX_MSG_SIZE;

    // Allocate send pool on local device
    CUDA_CHECK(cudaSetDevice(local_device_));
    CUDA_CHECK(cudaMalloc(&send_pool_, pool_size));
    CUDA_CHECK(cudaMemset(send_pool_, 0, pool_size));

    // Get IPC handle for send pool
    CUDA_CHECK(cudaIpcGetMemHandle(&send_pool_handle_, send_pool_));

    // Allocate recv pool on local device
    CUDA_CHECK(cudaMalloc(&recv_pool_, pool_size));
    CUDA_CHECK(cudaMemset(recv_pool_, 0, pool_size));
}

void AfdIpcComm::setup_ipc_events() {
    CUDA_CHECK(cudaSetDevice(local_device_));
    for (int i = 0; i < RING_SIZE; ++i) {
        // Create events with IPC capability + disable timing for lower overhead
        CUDA_CHECK(cudaEventCreateWithFlags(&send_events_[i],
            cudaEventDisableTiming | cudaEventInterprocess));
        CUDA_CHECK(cudaIpcGetEventHandle(&send_event_handles_[i], send_events_[i]));
        peer_events_[i] = nullptr;
    }
}

void AfdIpcComm::setup_gpu_signals() {
    CUDA_CHECK(cudaSetDevice(local_device_));
    // Allocate signal flags on local device (receiver polls these)
    CUDA_CHECK(cudaMalloc(&local_signal_flags_, RING_SIZE * sizeof(int64_t)));
    CUDA_CHECK(cudaMemset(local_signal_flags_, 0, RING_SIZE * sizeof(int64_t)));
    // peer_signal_flags_ will be set during handshake (points to peer's local_signal_flags_)
}

// ── Handshake ──

struct HandshakePayload {
    cudaIpcMemHandle_t mem_handle;
    cudaIpcEventHandle_t event_handles[RING_SIZE];
    cudaIpcMemHandle_t signal_flags_handle;  // for GPU_SIGNAL mode
    int device_id;
    int sync_mode;
};

void AfdIpcComm::handshake() {
    if (is_ffn_) {
        exchange_handles_server();
    } else {
        exchange_handles_client();
    }
    ready_ = true;
}

void AfdIpcComm::exchange_handles_server() {
    // FFN side: listen on Unix socket
    int server_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (server_fd < 0) throw std::runtime_error("socket() failed");

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    unlink(socket_path_.c_str());
    if (bind(server_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        close(server_fd);
        throw std::runtime_error("bind() failed: " + socket_path_);
    }
    listen(server_fd, 1);

    int conn_fd = accept(server_fd, nullptr, nullptr);
    if (conn_fd < 0) {
        close(server_fd);
        throw std::runtime_error("accept() failed");
    }

    // Receive peer's payload
    HandshakePayload peer_payload;
    ssize_t n = ::recv(conn_fd, &peer_payload, sizeof(peer_payload), MSG_WAITALL);
    if (n != sizeof(peer_payload)) {
        close(conn_fd); close(server_fd);
        throw std::runtime_error("recv handshake failed");
    }

    // Send our payload
    HandshakePayload my_payload;
    my_payload.mem_handle = send_pool_handle_;
    my_payload.device_id = local_device_;
    my_payload.sync_mode = static_cast<int>(sync_mode_);
    if (sync_mode_ == SyncMode::IPC_EVENT) {
        memcpy(my_payload.event_handles, send_event_handles_, sizeof(send_event_handles_));
    }
    if (sync_mode_ == SyncMode::GPU_SIGNAL && local_signal_flags_) {
        CUDA_CHECK(cudaIpcGetMemHandle(&my_payload.signal_flags_handle, local_signal_flags_));
    }

    ::send(conn_fd, &my_payload, sizeof(my_payload), 0);

    close(conn_fd);
    close(server_fd);
    unlink(socket_path_.c_str());

    // Import peer's send pool
    CUDA_CHECK(cudaSetDevice(local_device_));
    CUDA_CHECK(cudaIpcOpenMemHandle(&peer_send_pool_, peer_payload.mem_handle,
                                    cudaIpcMemLazyEnablePeerAccess));

    // Import peer's IPC events
    if (sync_mode_ == SyncMode::IPC_EVENT) {
        for (int i = 0; i < RING_SIZE; ++i) {
            CUDA_CHECK(cudaIpcOpenEventHandle(&peer_events_[i],
                                             peer_payload.event_handles[i]));
        }
    }

    // Import peer's signal flags
    if (sync_mode_ == SyncMode::GPU_SIGNAL) {
        CUDA_CHECK(cudaIpcOpenMemHandle((void**)&peer_signal_flags_,
                                       peer_payload.signal_flags_handle,
                                       cudaIpcMemLazyEnablePeerAccess));
    }
}

void AfdIpcComm::exchange_handles_client() {
    // ATTN side: connect to FFN's Unix socket
    int sock_fd = socket(AF_UNIX, SOCK_STREAM, 0);
    if (sock_fd < 0) throw std::runtime_error("socket() failed");

    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);

    // Retry connection (peer may not be ready yet)
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(300);
    bool connected = false;
    while (std::chrono::steady_clock::now() < deadline) {
        if (connect(sock_fd, (struct sockaddr*)&addr, sizeof(addr)) == 0) {
            connected = true;
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!connected) {
        close(sock_fd);
        throw std::runtime_error("connect() timeout: " + socket_path_);
    }

    // Send our payload
    HandshakePayload my_payload;
    my_payload.mem_handle = send_pool_handle_;
    my_payload.device_id = local_device_;
    my_payload.sync_mode = static_cast<int>(sync_mode_);
    if (sync_mode_ == SyncMode::IPC_EVENT) {
        memcpy(my_payload.event_handles, send_event_handles_, sizeof(send_event_handles_));
    }
    if (sync_mode_ == SyncMode::GPU_SIGNAL && local_signal_flags_) {
        CUDA_CHECK(cudaIpcGetMemHandle(&my_payload.signal_flags_handle, local_signal_flags_));
    }

    ::send(sock_fd, &my_payload, sizeof(my_payload), 0);

    // Receive peer's payload
    HandshakePayload peer_payload;
    ssize_t n = ::recv(sock_fd, &peer_payload, sizeof(peer_payload), MSG_WAITALL);
    if (n != sizeof(peer_payload)) {
        close(sock_fd);
        throw std::runtime_error("recv handshake failed");
    }

    close(sock_fd);

    // Import peer's send pool
    CUDA_CHECK(cudaSetDevice(local_device_));
    CUDA_CHECK(cudaIpcOpenMemHandle(&peer_send_pool_, peer_payload.mem_handle,
                                    cudaIpcMemLazyEnablePeerAccess));

    // Import peer's IPC events
    if (sync_mode_ == SyncMode::IPC_EVENT) {
        for (int i = 0; i < RING_SIZE; ++i) {
            CUDA_CHECK(cudaIpcOpenEventHandle(&peer_events_[i],
                                             peer_payload.event_handles[i]));
        }
    }

    // Import peer's signal flags
    if (sync_mode_ == SyncMode::GPU_SIGNAL) {
        CUDA_CHECK(cudaIpcOpenMemHandle((void**)&peer_signal_flags_,
                                       peer_payload.signal_flags_handle,
                                       cudaIpcMemLazyEnablePeerAccess));
    }
}

// ── SHM Flag Helpers ──

inline int AfdIpcComm::flag_offset(int slot, bool is_send) const {
    if (is_send) {
        int base = is_ffn_ ? SHM_FLAGS_F2A : SHM_FLAGS_A2F;
        return base + slot * 8;
    } else {
        int base = is_ffn_ ? SHM_FLAGS_A2F : SHM_FLAGS_F2A;
        return base + slot * 8;
    }
}

inline int AfdIpcComm::size_offset(int slot, bool is_send) const {
    if (is_send) {
        int base = is_ffn_ ? SHM_SIZES_F2A : SHM_SIZES_A2F;
        return base + slot * 8;
    } else {
        int base = is_ffn_ ? SHM_SIZES_A2F : SHM_SIZES_F2A;
        return base + slot * 8;
    }
}

inline int AfdIpcComm::meta_offset(int slot, bool is_send) const {
    if (is_send) {
        int base = is_ffn_ ? SHM_META_F2A : SHM_META_A2F;
        return base + slot * HEADER_BYTES;
    } else {
        int base = is_ffn_ ? SHM_META_A2F : SHM_META_F2A;
        return base + slot * HEADER_BYTES;
    }
}

void AfdIpcComm::write_flag(int slot, uint64_t value) {
    volatile uint64_t* ptr = (volatile uint64_t*)((char*)shm_ptr_ + flag_offset(slot, true));
    *ptr = value;
}

uint64_t AfdIpcComm::read_flag(int slot) {
    volatile uint64_t* ptr = (volatile uint64_t*)((char*)shm_ptr_ + flag_offset(slot, false));
    return *ptr;
}

void AfdIpcComm::write_size(int slot, uint64_t size) {
    volatile uint64_t* ptr = (volatile uint64_t*)((char*)shm_ptr_ + size_offset(slot, true));
    *ptr = size;
}

uint64_t AfdIpcComm::read_size(int slot) {
    volatile uint64_t* ptr = (volatile uint64_t*)((char*)shm_ptr_ + size_offset(slot, false));
    return *ptr;
}

void AfdIpcComm::write_meta_shm(int slot, const TensorMeta& meta) {
    void* ptr = (char*)shm_ptr_ + meta_offset(slot, true);
    memcpy(ptr, &meta, sizeof(TensorMeta));
}

void AfdIpcComm::read_meta_shm(int slot, TensorMeta* out_meta) {
    void* ptr = (char*)shm_ptr_ + meta_offset(slot, false);
    memcpy(out_meta, ptr, sizeof(TensorMeta));
}

// ── Sync Primitives ──

void AfdIpcComm::signal_peer(int slot, cudaStream_t stream) {
    switch (sync_mode_) {
        case SyncMode::CPU_FLAG:
            // Synchronize stream then write SHM flag from CPU
            CUDA_CHECK(cudaStreamSynchronize(stream));
            write_size(slot, cached_total_bytes_);
            write_flag(slot, 1);
            if (cached_total_bytes_ == 0) {
                fprintf(stderr, "[afd_ipc SIGNAL] WARNING: slot=%d cached_total_bytes_=0!\n", slot);
            }
            break;

        case SyncMode::IPC_EVENT:
            // Record event on stream — peer will cudaStreamWaitEvent
            CUDA_CHECK(cudaEventRecord(send_events_[slot], stream));
            // Also write SHM flag so peer knows which slot has data
            // (SHM flag write is after event record, so peer sees flag
            //  only after event is recorded — safe ordering)
            write_size(slot, cached_total_bytes_);
            write_flag(slot, 1);
            break;

        case SyncMode::GPU_SIGNAL:
            // GPU writes flag on peer's device memory via P2P
            // __threadfence_system ensures data copy is visible before flag
            if (peer_signal_flags_) {
                launch_signal_kernel(
                    (volatile int64_t*)(peer_signal_flags_ + slot),
                    1, stream);
            }
            // Also write SHM size for metadata
            write_size(slot, cached_total_bytes_);
            write_flag(slot, 1);
            break;
    }
}

void AfdIpcComm::wait_peer(int slot, cudaStream_t stream) {
    switch (sync_mode_) {
        case SyncMode::CPU_FLAG:
            // CPU spin-polls SHM flag
            while (read_flag(slot) != 1) {
                // Tight loop — flag is in /dev/shm (RAM), ~10ns per read
            }
            break;

        case SyncMode::IPC_EVENT:
            // CPU polls SHM flag to know data is available
            while (read_flag(slot) != 1) {}
            // Then GPU waits on peer's event (pure GPU sync, no CPU block)
            if (peer_events_[slot]) {
                CUDA_CHECK(cudaStreamWaitEvent(stream, peer_events_[slot], 0));
            }
            break;

        case SyncMode::GPU_SIGNAL:
            // GPU spin-polls local signal flag (device memory, L2 cache)
            if (local_signal_flags_) {
                launch_wait_kernel(
                    (volatile int64_t*)(local_signal_flags_ + slot),
                    1, stream);
            }
            break;
    }
}

// ── Send / Recv Hot Path ──

void AfdIpcComm::send(const void* data_ptr, const TensorMeta& meta,
                      size_t data_bytes, cudaStream_t stream) {
    CUDA_CHECK(cudaSetDevice(local_device_));

    int slot = send_slot_;

    // Wait for peer to consume this slot (CPU poll)
    volatile uint64_t* flag_ptr = (volatile uint64_t*)((char*)shm_ptr_ + flag_offset(slot, true));
    while (*flag_ptr != 0) {}

    // Compute send buffer offset for this slot
    char* send_buf = (char*)send_pool_ + (size_t)slot * MAX_MSG_SIZE;

    // Copy tensor data directly to send_buf (no header in GPU buffer)
    CUDA_CHECK(cudaMemcpyAsync(send_buf, data_ptr, data_bytes,
                               cudaMemcpyDeviceToDevice, stream));

    // Record event for GPU-side synchronization (peer can wait on this)
    if (sync_mode_ == SyncMode::IPC_EVENT) {
        CUDA_CHECK(cudaEventRecord(send_events_[slot], stream));
    } else {
        // CPU_FLAG mode: must sync before signaling
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Write metadata to SHM (CPU-side, no GPU involvement)
    write_meta_shm(slot, meta);
    cached_meta_ = meta;
    cached_total_bytes_ = data_bytes;

    // Signal peer: write size then flag
    write_size(slot, data_bytes);
    write_flag(slot, 1);

    send_slot_ = (slot + 1) % RING_SIZE;
}

void AfdIpcComm::send_cached(const void* data_ptr, size_t data_bytes,
                             cudaStream_t stream) {
    CUDA_CHECK(cudaSetDevice(local_device_));

    int slot = send_slot_;

    // Wait for peer to consume this slot
    volatile uint64_t* flag_ptr = (volatile uint64_t*)((char*)shm_ptr_ + flag_offset(slot, true));
    while (*flag_ptr != 0) {}

    char* send_buf = (char*)send_pool_ + (size_t)slot * MAX_MSG_SIZE;

    // Copy tensor data directly (no header in GPU buffer)
    CUDA_CHECK(cudaMemcpyAsync(send_buf, data_ptr, data_bytes,
                               cudaMemcpyDeviceToDevice, stream));

    // Record event for GPU-side synchronization
    if (sync_mode_ == SyncMode::IPC_EVENT) {
        CUDA_CHECK(cudaEventRecord(send_events_[slot], stream));
    } else {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    // Write metadata to SHM (cached meta, same shape)
    write_meta_shm(slot, cached_meta_);
    cached_total_bytes_ = data_bytes;

    // Signal peer
    write_size(slot, data_bytes);
    write_flag(slot, 1);

    send_slot_ = (slot + 1) % RING_SIZE;
}

void* AfdIpcComm::recv(TensorMeta* out_meta, size_t* out_data_bytes,
                       cudaStream_t stream) {
    int slot = recv_slot_;

    CUDA_CHECK(cudaSetDevice(local_device_));

    // Wait for sender's signal (CPU polls SHM flag)
    while (read_flag(slot) != 1) {}

    // Read metadata from SHM (CPU-side, no GPU involvement, no sync needed)
    TensorMeta meta;
    read_meta_shm(slot, &meta);

    // Read data size from SHM
    size_t data_bytes = read_size(slot);

    if (data_bytes == 0) {
        fprintf(stderr, "[afd_ipc RECV] ERROR: slot=%d data_bytes=0!\n", slot);
        if (out_meta) memset(out_meta, 0, sizeof(TensorMeta));
        if (out_data_bytes) *out_data_bytes = 0;
        recv_slot_ = (slot + 1) % RING_SIZE;
        return recv_pool_;
    }

    // For IPC_EVENT mode: GPU waits on peer's send event (pure GPU sync)
    // This ensures P2P copy below sees completed data without CPU blocking
    if (sync_mode_ == SyncMode::IPC_EVENT && peer_events_[slot]) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, peer_events_[slot], 0));
    }

    // Copy data from peer's send_buf to local recv_buf (cross-device P2P)
    char* peer_buf = (char*)peer_send_pool_ + (size_t)slot * MAX_MSG_SIZE;
    char* recv_buf = (char*)recv_pool_ + (size_t)slot * MAX_MSG_SIZE;

    CUDA_CHECK(cudaMemcpyPeerAsync(recv_buf, local_device_, peer_buf, peer_device_,
                                   data_bytes, stream));

    // For CPU_FLAG mode: must sync to ensure data is ready before use
    if (sync_mode_ == SyncMode::CPU_FLAG) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    // For IPC_EVENT mode: no CPU sync needed! The compute stream that uses
    // this data should wait on the same stream (ordering guaranteed).

    if (out_meta) *out_meta = meta;
    if (out_data_bytes) *out_data_bytes = data_bytes;

    // Cache metadata
    cached_meta_ = meta;
    cached_total_bytes_ = data_bytes;
    meta_cached_ = true;

    // Clear flag so sender can reuse slot
    volatile uint64_t* flag_ptr = (volatile uint64_t*)((char*)shm_ptr_ + flag_offset(slot, false));
    *flag_ptr = 0;

    // Reset GPU signal flag if using GPU_SIGNAL mode
    if (sync_mode_ == SyncMode::GPU_SIGNAL && local_signal_flags_) {
        CUDA_CHECK(cudaMemsetAsync(local_signal_flags_ + slot, 0,
                                   sizeof(int64_t), stream));
    }

    recv_slot_ = (slot + 1) % RING_SIZE;

    // Return pointer to data in recv buffer (valid until this slot is reused)
    return recv_buf;
}

void* AfdIpcComm::recv_cached(size_t* out_data_bytes, cudaStream_t stream) {
    int slot = recv_slot_;

    CUDA_CHECK(cudaSetDevice(local_device_));

    // Wait for sender's signal (CPU polls SHM flag)
    while (read_flag(slot) != 1) {}

    size_t data_bytes = cached_total_bytes_;
    if (data_bytes == 0) {
        data_bytes = read_size(slot);
    }

    // For IPC_EVENT mode: GPU waits on peer's send event
    if (sync_mode_ == SyncMode::IPC_EVENT && peer_events_[slot]) {
        CUDA_CHECK(cudaStreamWaitEvent(stream, peer_events_[slot], 0));
    }

    // Copy data from peer's send_buf to local recv_buf
    char* peer_buf = (char*)peer_send_pool_ + (size_t)slot * MAX_MSG_SIZE;
    char* recv_buf = (char*)recv_pool_ + (size_t)slot * MAX_MSG_SIZE;

    CUDA_CHECK(cudaMemcpyPeerAsync(recv_buf, local_device_, peer_buf, peer_device_,
                                   data_bytes, stream));

    // For CPU_FLAG mode: must sync
    if (sync_mode_ == SyncMode::CPU_FLAG) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    if (out_data_bytes) *out_data_bytes = data_bytes;

    // Clear flag
    volatile uint64_t* flag_ptr = (volatile uint64_t*)((char*)shm_ptr_ + flag_offset(slot, false));
    *flag_ptr = 0;

    // Reset GPU signal flag
    if (sync_mode_ == SyncMode::GPU_SIGNAL && local_signal_flags_) {
        CUDA_CHECK(cudaMemsetAsync(local_signal_flags_ + slot, 0,
                                   sizeof(int64_t), stream));
    }

    recv_slot_ = (slot + 1) % RING_SIZE;

    return recv_buf;
}

void AfdIpcComm::cache_meta(const TensorMeta& meta, size_t total_bytes) {
    cached_meta_ = meta;
    cached_total_bytes_ = total_bytes;
    meta_cached_ = true;
}

}  // namespace afd_ipc
