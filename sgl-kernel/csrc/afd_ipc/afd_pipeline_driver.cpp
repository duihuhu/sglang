/**
 * Pipeline Driver implementation.
 */

#include "afd_pipeline_driver.h"
#include <stdexcept>

namespace afd_ipc {

PipelineDriver::PipelineDriver(AfdIpcComm* comm, int num_mb, bool use_gpu_signal)
    : comm_(comm), num_mb_(num_mb), use_gpu_signal_(use_gpu_signal) {
    if (!comm_) {
        throw std::runtime_error("PipelineDriver: comm is null");
    }
    if (num_mb_ < 1) {
        throw std::runtime_error("PipelineDriver: num_mb must be >= 1");
    }
}

std::vector<void*> PipelineDriver::send_recv_layer(
    const std::vector<void*>& send_ptrs,
    const std::vector<size_t>& send_sizes,
    cudaStream_t stream
) {
    if ((int)send_ptrs.size() != num_mb_ || (int)send_sizes.size() != num_mb_) {
        throw std::runtime_error("PipelineDriver: send_ptrs/send_sizes size mismatch");
    }

    // Send all M micro-batches
    for (int mb = 0; mb < num_mb_; ++mb) {
        if (use_gpu_signal_) {
            comm_->send_gpu_only(send_ptrs[mb], send_sizes[mb], stream);
        } else {
            comm_->send_cached(send_ptrs[mb], send_sizes[mb], stream);
        }
    }

    // Recv all M micro-batches
    std::vector<void*> recv_ptrs(num_mb_);
    for (int mb = 0; mb < num_mb_; ++mb) {
        size_t recv_bytes;
        if (use_gpu_signal_) {
            recv_ptrs[mb] = comm_->recv_gpu_only(&recv_bytes, stream);
        } else {
            recv_ptrs[mb] = comm_->recv_cached(&recv_bytes, stream);
        }
    }

    return recv_ptrs;
}

std::vector<void*> PipelineDriver::send_all_then_recv_all(
    const std::vector<void*>& send_ptrs,
    const std::vector<size_t>& send_sizes,
    cudaStream_t stream
) {
    // Same as send_recv_layer but explicitly named for clarity
    return send_recv_layer(send_ptrs, send_sizes, stream);
}

}  // namespace afd_ipc
