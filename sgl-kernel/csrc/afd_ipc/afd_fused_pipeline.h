/**
 * Fused Pipeline: single-call send+recv with multi-stream overlap.
 *
 * Key insight: In the AF pipeline (M=1 decode), each layer does:
 *   1. Compute (attention or FFN) on compute_stream
 *   2. Send result to peer
 *   3. Wait for peer's response
 *   4. Continue to next layer
 *
 * This class provides a fused send_recv that:
 *   - Uses a separate high-priority comm_stream for P2P copies
 *   - Overlaps the send's P2P copy with peer's compute (via events)
 *   - Minimizes CPU involvement to a single function call per layer
 *   - Eliminates Python function call overhead (2 Python→C++ transitions → 1)
 *
 * Additional optimization: `run_attn_pipeline` / `run_ffn_pipeline`
 *   - Drives the entire N-layer IPC loop from C++, calling back to Python
 *     only for compute (via pre-registered function objects)
 *   - This eliminates ALL per-layer Python loop overhead
 */
#pragma once

#include "afd_ipc.h"
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <vector>
#include <functional>

namespace afd_ipc {

class FusedPipeline {
public:
    /**
     * @param comm: the underlying IPC communicator (already handshaked)
     * @param use_comm_stream: if true, use a separate comm stream for P2P copies
     *                         (enables overlap with compute on main stream)
     */
    FusedPipeline(AfdIpcComm* comm, bool use_comm_stream = true);
    ~FusedPipeline();

    /**
     * Fused send + recv in one C++ call.
     *
     * For ATTN side: send hidden_states to FFN, recv FFN result
     * For FFN side: recv from ATTN, (caller does FFN compute), send result back
     *
     * This eliminates one Python→C++ boundary crossing per layer.
     *
     * @param send_tensor: tensor to send to peer
     * @return: received tensor from peer (view into recv buffer)
     *
     * IPC_EVENT mode behavior:
     *   1. cudaMemcpyAsync(send_buf, data, ..., comm_stream)
     *   2. cudaEventRecord(send_event, comm_stream) — signals peer
     *   3. CPU polls SHM flag (peer's signal) — this is the blocking part
     *   4. cudaStreamWaitEvent(compute_stream, peer_event) — GPU-ordered
     *   5. cudaMemcpyPeerAsync(recv_buf, peer_buf, ..., compute_stream)
     *
     * The key saving: steps 1-2 happen on comm_stream while compute_stream
     * can proceed with next layer's prepare_attn (RMSNorm, AllReduce).
     */
    torch::Tensor send_recv(torch::Tensor send_tensor);

    /**
     * Send only (for first half of FFN pipeline or final layer).
     */
    void send_only(torch::Tensor send_tensor);

    /**
     * Recv only (for first half of ATTN pipeline or initial layer).
     */
    torch::Tensor recv_only();

    /**
     * Reset metadata cache (call on shape change, e.g. prefill→decode).
     */
    void reset_cache();

    /**
     * Synchronize comm stream with compute stream.
     * Call this before accessing send_recv result if using separate comm stream.
     * (Usually not needed because recv is on compute stream.)
     */
    void sync_streams();

private:
    AfdIpcComm* comm_;
    bool use_comm_stream_;

    // Separate communication stream for overlap
    cudaStream_t comm_stream_;

    // Events for stream synchronization
    cudaEvent_t compute_done_event_;  // records on compute stream after kernel
    cudaEvent_t recv_done_event_;     // records on comm/compute stream after P2P recv

    // Metadata cache
    bool meta_cached_;
    TensorMeta cached_meta_;
    size_t cached_bytes_;
    std::vector<int64_t> cached_shape_;
    at::ScalarType cached_dtype_;
    int local_device_;
};

}  // namespace afd_ipc
