/**
 * pybind11 bindings for libafd_ipc.
 *
 * Exposes AfdIpcComm to Python with torch.Tensor-aware send/recv methods.
 * The hot path (send_cached/recv_cached) stays entirely in C++ — Python only
 * calls into C++ once per layer, with no intermediate Python object creation.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

#include "afd_ipc.h"
#include "afd_pipeline_driver.h"
#include "afd_fused_pipeline.h"

namespace py = pybind11;

namespace afd_ipc {

// Convert torch dtype to our DtypeCode (matches Python _DTYPE_TO_INT)
static DtypeCode dtype_to_code(at::ScalarType dtype) {
    switch (dtype) {
        case at::kHalf: return DTYPE_FP16;
        case at::kBFloat16: return DTYPE_BF16;
        case at::kFloat: return DTYPE_FP32;
        case at::kDouble: return DTYPE_FP64;
        case at::kInt: return DTYPE_INT32;
        case at::kLong: return DTYPE_INT64;
        default:
            throw std::runtime_error("Unsupported dtype for IPC transfer");
    }
}

// Convert DtypeCode back to torch dtype
static at::ScalarType code_to_dtype(DtypeCode code) {
    switch (code) {
        case DTYPE_FP16: return at::kHalf;
        case DTYPE_BF16: return at::kBFloat16;
        case DTYPE_FP32: return at::kFloat;
        case DTYPE_FP64: return at::kDouble;
        case DTYPE_INT32: return at::kInt;
        case DTYPE_INT64: return at::kLong;
        default:
            throw std::runtime_error("Unknown dtype code: " + std::to_string(static_cast<int64_t>(code)));
    }
}

// Encode tensor metadata matching Python _encode_meta layout:
// [ndim, shape[0], ..., shape[ndim-1], dtype_code, original_num_tokens, ...]
static TensorMeta encode_tensor_meta(const torch::Tensor& t) {
    TensorMeta meta;
    memset(&meta, 0, sizeof(meta));
    int ndim = t.dim();
    meta.data[0] = ndim;
    for (int i = 0; i < ndim && i < 5; ++i) {
        meta.data[i + 1] = t.size(i);
    }
    meta.data[ndim + 1] = static_cast<int64_t>(dtype_to_code(t.scalar_type()));
    meta.data[ndim + 2] = 0;  // original_num_tokens (unused in IPC)
    return meta;
}

/**
 * Python-facing wrapper that accepts torch.Tensor directly.
 */
class PyAfdIpcComm {
public:
    PyAfdIpcComm(bool is_ffn, int local_device, int peer_device,
                 int rank, int mb_id, const std::string& sync_mode_str)
    {
        SyncMode mode = SyncMode::IPC_EVENT;  // default: best performance
        if (sync_mode_str == "cpu_flag") mode = SyncMode::CPU_FLAG;
        else if (sync_mode_str == "ipc_event") mode = SyncMode::IPC_EVENT;
        else if (sync_mode_str == "gpu_signal") mode = SyncMode::GPU_SIGNAL;

        comm_ = std::make_unique<AfdIpcComm>(
            is_ffn, local_device, peer_device, rank, mb_id, mode);
    }

    void handshake() {
        comm_->handshake();
    }

    bool is_ready() const {
        return comm_->is_ready();
    }

    /**
     * Send a tensor. First call encodes metadata; subsequent calls with
     * same shape/dtype use cached path (zero Python overhead in hot loop).
     */
    void send_tensor(torch::Tensor x) {
        TORCH_CHECK(x.is_cuda(), "Tensor must be on CUDA device");
        auto x_cont = x.contiguous();

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
            x_cont.device().index()).stream();

        size_t data_bytes = x_cont.numel() * x_cont.element_size();

        if (!meta_cached_send_ || data_bytes != cached_send_bytes_) {
            // First call or shape changed: encode full metadata
            TensorMeta meta = encode_tensor_meta(x_cont);
            comm_->send(x_cont.data_ptr(), meta, data_bytes, stream);
            cached_send_meta_ = meta;
            cached_send_bytes_ = data_bytes;
            meta_cached_send_ = true;
            // NOTE: do NOT reset the recv cache here. The send/recv metadata
            // caches are independent (different directions/tensors); the recv
            // cache is maintained solely by recv_tensor() and consumed by
            // recv_tensor_gpu(). Resetting it on send breaks the M=1 gpu-only
            // path where the per-layer order is recv -> send (FFN side), which
            // would leave recv_tensor_gpu() without cached shape metadata.
            // Shape changes (e.g. prefill->decode) are handled via reset_cache().
        } else {
            // Hot path: same shape, skip metadata encoding
            comm_->send_cached(x_cont.data_ptr(), data_bytes, stream);
        }
    }

    /**
     * Receive a tensor. First call decodes metadata; subsequent calls
     * use cached shape/dtype (zero Python overhead in hot loop).
     *
     * Returns a view into the pre-allocated recv buffer (no clone!).
     * The buffer is valid until the same ring slot is reused (RING_SIZE=4
     * calls later). In M=1 AF pipeline, each layer consumes the tensor
     * immediately, so this is always safe.
     *
     * In IPC_EVENT mode: no cudaStreamSynchronize! The P2P copy is ordered
     * on the current CUDA stream via cudaStreamWaitEvent. As long as the
     * caller uses the tensor on the same stream, data is guaranteed ready.
     */
    torch::Tensor recv_tensor() {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
            comm_->local_device()).stream();

        TensorMeta meta;
        size_t data_bytes;
        void* data_ptr;
        {
            // Release GIL during blocking busy-poll to allow other threads
            // (e.g. send_tensor from main thread) to proceed concurrently.
            py::gil_scoped_release release;
            data_ptr = comm_->recv(&meta, &data_bytes, stream);
        }
        // GIL re-acquired here for tensor creation

        // Decode metadata (from SHM, already on CPU — no GPU sync needed)
        int ndim = static_cast<int>(meta.data[0]);
        std::vector<int64_t> shape;
        for (int i = 0; i < ndim; ++i) {
            shape.push_back(meta.data[i + 1]);
        }
        int64_t dtype_code = meta.data[ndim + 1];
        auto dtype = code_to_dtype(static_cast<DtypeCode>(dtype_code));

        auto options = torch::TensorOptions()
            .dtype(dtype)
            .device(torch::kCUDA, comm_->local_device());

        // Return view into recv buffer — NO clone!
        // Safe because ring has 4 slots and AF pipeline consumes immediately.
        // Cache shape/dtype for recv_tensor_gpu()
        cached_recv_shape_ = shape;
        cached_recv_dtype_ = dtype;
        cached_recv_bytes_ = data_bytes;
        meta_cached_recv_ = true;
        return torch::from_blob(data_ptr, shape, options);
    }

    /**
     * Reset metadata cache (call when tensor shape changes, e.g. prefill→decode).
     */
    void reset_cache() {
        meta_cached_send_ = false;
        meta_cached_recv_ = false;
    }

    /**
     * GPU-only send: no CPU blocking. Uses GPU signal kernel to notify peer.
     * CPU only enqueues CUDA ops on the current stream and returns immediately.
     */
    void send_tensor_gpu(torch::Tensor x) {
        TORCH_CHECK(x.is_cuda(), "Tensor must be on CUDA device");
        auto x_cont = x.contiguous();

        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
            x_cont.device().index()).stream();

        size_t data_bytes = x_cont.numel() * x_cont.element_size();

        if (!meta_cached_send_ || data_bytes != cached_send_bytes_) {
            TensorMeta meta = encode_tensor_meta(x_cont);
            comm_->cache_meta(meta, data_bytes);
            cached_send_meta_ = meta;
            cached_send_bytes_ = data_bytes;
            meta_cached_send_ = true;
        }

        comm_->send_gpu_only(x_cont.data_ptr(), data_bytes, stream);
    }

    /**
     * GPU-only recv: no CPU blocking. Uses GPU wait kernel to poll device memory.
     * CPU only enqueues wait_kernel + memcpy on the current stream.
     * Returns a view into the recv buffer (stream-ordered, safe to use on same stream).
     */
    torch::Tensor recv_tensor_gpu() {
        cudaStream_t stream = c10::cuda::getCurrentCUDAStream(
            comm_->local_device()).stream();

        size_t data_bytes;
        void* data_ptr = comm_->recv_gpu_only(&data_bytes, stream);

        if (!meta_cached_recv_) {
            // First call: need to get shape from SHM (fallback)
            // After first recv_tensor() call, shape is cached
            throw std::runtime_error(
                "recv_tensor_gpu: metadata not cached. "
                "Call recv_tensor() first to establish shape cache.");
        }

        auto options = torch::TensorOptions()
            .dtype(cached_recv_dtype_)
            .device(torch::kCUDA, comm_->local_device());

        return torch::from_blob(data_ptr, cached_recv_shape_, options);
    }

    std::string sync_mode() const {
        switch (comm_->sync_mode()) {
            case SyncMode::CPU_FLAG: return "cpu_flag";
            case SyncMode::IPC_EVENT: return "ipc_event";
            case SyncMode::GPU_SIGNAL: return "gpu_signal";
            default: return "unknown";
        }
    }

    int local_device() const { return comm_->local_device(); }
    int peer_device() const { return comm_->peer_device(); }

    // Expose raw comm for PipelineDriver
    AfdIpcComm* raw_comm() { return comm_.get(); }

private:
    std::unique_ptr<AfdIpcComm> comm_;

    // Send cache
    bool meta_cached_send_ = false;
    TensorMeta cached_send_meta_;
    size_t cached_send_bytes_ = 0;

    // Recv cache
    bool meta_cached_recv_ = false;
    std::vector<int64_t> cached_recv_shape_;
    at::ScalarType cached_recv_dtype_;
    size_t cached_recv_bytes_ = 0;
};

PYBIND11_MODULE(afd_ipc_cpp, m) {
    m.doc() = "High-performance C++ IPC communication for AF disaggregation";

    py::class_<PyAfdIpcComm>(m, "AfdIpcComm")
        .def(py::init<bool, int, int, int, int, const std::string&>(),
             py::arg("is_ffn"),
             py::arg("local_device"),
             py::arg("peer_device"),
             py::arg("rank"),
             py::arg("mb_id") = -1,
             py::arg("sync_mode") = "ipc_event")
        .def("handshake", &PyAfdIpcComm::handshake,
             py::call_guard<py::gil_scoped_release>(),
             "Exchange IPC handles with peer process")
        .def("is_ready", &PyAfdIpcComm::is_ready,
             "Check if handshake is complete")
        .def("send_tensor", &PyAfdIpcComm::send_tensor,
             py::arg("x"),
             "Send tensor to peer (cached hot path after first call)")
        .def("recv_tensor", &PyAfdIpcComm::recv_tensor,
             "Receive tensor from peer (cached hot path after first call)")
        .def("send_tensor_gpu", &PyAfdIpcComm::send_tensor_gpu,
             py::arg("x"),
             "GPU-only send: no CPU blocking, uses GPU signal kernel")
        .def("recv_tensor_gpu", &PyAfdIpcComm::recv_tensor_gpu,
             "GPU-only recv: no CPU blocking, uses GPU wait kernel")
        .def("reset_cache", &PyAfdIpcComm::reset_cache,
             "Reset metadata cache (call on shape change)")
        .def("sync_mode", &PyAfdIpcComm::sync_mode,
             "Get current synchronization mode")
        .def_property_readonly("local_device", &PyAfdIpcComm::local_device)
        .def_property_readonly("peer_device", &PyAfdIpcComm::peer_device);

    // Expose SyncMode enum
    py::enum_<SyncMode>(m, "SyncMode")
        .value("CPU_FLAG", SyncMode::CPU_FLAG)
        .value("IPC_EVENT", SyncMode::IPC_EVENT)
        .value("GPU_SIGNAL", SyncMode::GPU_SIGNAL);

    // Pipeline Driver: batch send/recv for M micro-batches
    py::class_<PipelineDriver>(m, "PipelineDriver")
        .def(py::init([](PyAfdIpcComm& comm_wrapper, int num_mb, bool use_gpu_signal) {
            return std::make_unique<PipelineDriver>(
                comm_wrapper.raw_comm(), num_mb, use_gpu_signal);
        }), py::arg("comm"), py::arg("num_mb"), py::arg("use_gpu_signal") = true)
        .def("send_recv_layer", [](PipelineDriver& self,
                                    std::vector<int64_t> send_ptrs_int,
                                    std::vector<size_t> send_sizes,
                                    int64_t stream_int) {
            std::vector<void*> send_ptrs;
            for (auto p : send_ptrs_int) send_ptrs.push_back(reinterpret_cast<void*>(p));
            cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_int);
            auto recv_ptrs = self.send_recv_layer(send_ptrs, send_sizes, stream);
            std::vector<int64_t> result;
            for (auto p : recv_ptrs) result.push_back(reinterpret_cast<int64_t>(p));
            return result;
        }, py::arg("send_ptrs"), py::arg("send_sizes"), py::arg("stream"),
           "Batch send M tensors + recv M tensors in one C++ call");

    // FusedPipeline: single-call send+recv with minimal Python overhead
    py::class_<FusedPipeline>(m, "FusedPipeline")
        .def(py::init([](PyAfdIpcComm& comm_wrapper, bool use_comm_stream) {
            return std::make_unique<FusedPipeline>(
                comm_wrapper.raw_comm(), use_comm_stream);
        }), py::arg("comm"), py::arg("use_comm_stream") = true)
        .def("send_recv", &FusedPipeline::send_recv,
             py::arg("send_tensor"),
             "Fused send + recv in one C++ call (one Python→C++ transition)")
        .def("send_only", &FusedPipeline::send_only,
             py::arg("send_tensor"),
             "Send tensor to peer")
        .def("recv_only", &FusedPipeline::recv_only,
             "Receive tensor from peer")
        .def("reset_cache", &FusedPipeline::reset_cache,
             "Reset metadata cache (call on shape change)")
        .def("sync_streams", &FusedPipeline::sync_streams,
             "Synchronize comm stream with compute stream");
}

}  // namespace afd_ipc
