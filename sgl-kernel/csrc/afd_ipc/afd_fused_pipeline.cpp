/**
 * FusedPipeline implementation: single-call send+recv with multi-stream overlap.
 */

#include "afd_fused_pipeline.h"
#include <stdexcept>
#include <cstring>

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

FusedPipeline::FusedPipeline(AfdIpcComm* comm, bool use_comm_stream)
    : comm_(comm),
      use_comm_stream_(use_comm_stream),
      comm_stream_(nullptr),
      compute_done_event_(nullptr),
      recv_done_event_(nullptr),
      meta_cached_(false),
      cached_bytes_(0),
      cached_dtype_(at::kBFloat16),
      local_device_(comm->local_device()) {

    CUDA_CHECK(cudaSetDevice(local_device_));

    if (use_comm_stream_) {
        int least_priority, greatest_priority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
        CUDA_CHECK(cudaStreamCreateWithPriority(
            &comm_stream_, cudaStreamNonBlocking, greatest_priority));
    }

    CUDA_CHECK(cudaEventCreateWithFlags(
        &compute_done_event_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &recv_done_event_, cudaEventDisableTiming));
}

FusedPipeline::~FusedPipeline() {
    if (comm_stream_) cudaStreamDestroy(comm_stream_);
    if (compute_done_event_) cudaEventDestroy(compute_done_event_);
    if (recv_done_event_) cudaEventDestroy(recv_done_event_);
}

void FusedPipeline::reset_cache() {
    meta_cached_ = false;
    cached_bytes_ = 0;
}

void FusedPipeline::sync_streams() {
    if (use_comm_stream_ && comm_stream_) {
        cudaStream_t compute_stream =
            c10::cuda::getCurrentCUDAStream(local_device_).stream();
        CUDA_CHECK(cudaEventRecord(recv_done_event_, comm_stream_));
        CUDA_CHECK(cudaStreamWaitEvent(compute_stream, recv_done_event_, 0));
    }
}

void FusedPipeline::send_only(torch::Tensor send_tensor) {
    TORCH_CHECK(send_tensor.is_cuda(), "send_tensor must be on CUDA");
    auto x = send_tensor.contiguous();
    size_t data_bytes = x.numel() * x.element_size();

    cudaStream_t compute_stream =
        c10::cuda::getCurrentCUDAStream(local_device_).stream();

    if (!meta_cached_ || data_bytes != cached_bytes_) {
        TensorMeta meta;
        memset(&meta, 0, sizeof(meta));
        int ndim = x.dim();
        meta.data[0] = ndim;
        for (int i = 0; i < ndim && i < 5; ++i)
            meta.data[i + 1] = x.size(i);
        // dtype code
        at::ScalarType st = x.scalar_type();
        int64_t dc = 1; // bf16 default
        if (st == at::kHalf) dc = 0;
        else if (st == at::kBFloat16) dc = 1;
        else if (st == at::kFloat) dc = 2;
        meta.data[ndim + 1] = dc;

        comm_->send(x.data_ptr(), meta, data_bytes, compute_stream);
        cached_meta_ = meta;
        cached_bytes_ = data_bytes;
        cached_dtype_ = st;
        cached_shape_.clear();
        for (int i = 0; i < ndim; ++i)
            cached_shape_.push_back(x.size(i));
        meta_cached_ = true;
    } else {
        comm_->send_cached(x.data_ptr(), data_bytes, compute_stream);
    }
}

torch::Tensor FusedPipeline::recv_only() {
    cudaStream_t compute_stream =
        c10::cuda::getCurrentCUDAStream(local_device_).stream();

    TensorMeta meta;
    size_t data_bytes;
    void* data_ptr;

    if (!meta_cached_) {
        data_ptr = comm_->recv(&meta, &data_bytes, compute_stream);
        int ndim = static_cast<int>(meta.data[0]);
        cached_shape_.clear();
        for (int i = 0; i < ndim; ++i)
            cached_shape_.push_back(meta.data[i + 1]);
        int64_t dc = meta.data[ndim + 1];
        if (dc == 0) cached_dtype_ = at::kHalf;
        else if (dc == 1) cached_dtype_ = at::kBFloat16;
        else if (dc == 2) cached_dtype_ = at::kFloat;
        else cached_dtype_ = at::kBFloat16;
        cached_bytes_ = data_bytes;
        cached_meta_ = meta;
        meta_cached_ = true;
    } else {
        data_ptr = comm_->recv_cached(&data_bytes, compute_stream);
    }

    auto options = torch::TensorOptions()
        .dtype(cached_dtype_)
        .device(torch::kCUDA, local_device_);
    return torch::from_blob(data_ptr, cached_shape_, options);
}

torch::Tensor FusedPipeline::send_recv(torch::Tensor send_tensor) {
    TORCH_CHECK(send_tensor.is_cuda(), "send_tensor must be on CUDA");
    auto x = send_tensor.contiguous();
    size_t data_bytes = x.numel() * x.element_size();

    cudaStream_t compute_stream =
        c10::cuda::getCurrentCUDAStream(local_device_).stream();

    // --- SEND phase ---
    if (!meta_cached_ || data_bytes != cached_bytes_) {
        TensorMeta meta;
        memset(&meta, 0, sizeof(meta));
        int ndim = x.dim();
        meta.data[0] = ndim;
        for (int i = 0; i < ndim && i < 5; ++i)
            meta.data[i + 1] = x.size(i);
        at::ScalarType st = x.scalar_type();
        int64_t dc = 1;
        if (st == at::kHalf) dc = 0;
        else if (st == at::kBFloat16) dc = 1;
        else if (st == at::kFloat) dc = 2;
        meta.data[ndim + 1] = dc;

        comm_->send(x.data_ptr(), meta, data_bytes, compute_stream);
        cached_meta_ = meta;
        cached_bytes_ = data_bytes;
        cached_dtype_ = st;
        cached_shape_.clear();
        for (int i = 0; i < ndim; ++i)
            cached_shape_.push_back(x.size(i));
        meta_cached_ = true;
    } else {
        comm_->send_cached(x.data_ptr(), data_bytes, compute_stream);
    }

    // --- RECV phase ---
    size_t recv_bytes;
    void* recv_ptr = comm_->recv_cached(&recv_bytes, compute_stream);

    auto options = torch::TensorOptions()
        .dtype(cached_dtype_)
        .device(torch::kCUDA, local_device_);
    return torch::from_blob(recv_ptr, cached_shape_, options);
}

}  // namespace afd_ipc
