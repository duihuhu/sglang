/**
 * Pipeline Driver: batch send/recv for M micro-batches in a single C++ call.
 *
 * Eliminates Python loop overhead by handling all M send + M recv operations
 * in one function call. Uses GPU signal/wait for zero-CPU-blocking communication.
 *
 * Usage from Python:
 *   driver = PipelineDriver(comm, num_mb)
 *   # Per layer:
 *   driver.send_recv_layer(send_ptrs, send_sizes, recv_ptrs, stream)
 *   # recv_ptrs now point to received data (stream-ordered)
 */
#pragma once

#include "afd_ipc.h"
#include <vector>

namespace afd_ipc {

class PipelineDriver {
public:
    /**
     * @param comm: the underlying IPC communicator (already handshaked)
     * @param num_mb: number of micro-batches (M)
     * @param use_gpu_signal: if true, use GPU signal/wait (no CPU poll)
     */
    PipelineDriver(AfdIpcComm* comm, int num_mb, bool use_gpu_signal = true);

    /**
     * Execute one layer's worth of communication:
     *   - Send M tensors (interleaved with recv for overlap)
     *   - Recv M tensors from peer
     *
     * All operations are enqueued on the given stream. CPU returns immediately
     * when use_gpu_signal=true.
     *
     * @param send_ptrs: array of M GPU pointers to send data
     * @param send_sizes: array of M sizes in bytes
     * @param stream: CUDA stream for all operations
     * @return: array of M GPU pointers to received data (in recv_pool)
     */
    std::vector<void*> send_recv_layer(
        const std::vector<void*>& send_ptrs,
        const std::vector<size_t>& send_sizes,
        cudaStream_t stream
    );

    /**
     * Interleaved send/recv: send(mb_i) then recv(mb_{i-1}).
     * This overlaps DF's FFN compute with DA's next Attn compute.
     *
     * Pattern: send(0) → send(1) → recv(0) → recv(1)
     * (For M=2, DA does Attn(mb0)→send(mb0)→Attn(mb1)→send(mb1)→recv(mb0)→recv(mb1))
     */
    std::vector<void*> send_all_then_recv_all(
        const std::vector<void*>& send_ptrs,
        const std::vector<size_t>& send_sizes,
        cudaStream_t stream
    );

private:
    AfdIpcComm* comm_;
    int num_mb_;
    bool use_gpu_signal_;
};

}  // namespace afd_ipc
