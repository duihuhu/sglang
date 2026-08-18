#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <torch/extension.h>

#include <type_traits>

namespace {

template <typename scalar_t>
__global__ void restore_page_first_mha_to_raw_kernel(
    const scalar_t* host_k,
    const scalar_t* host_v,
    const uint64_t* k_destinations,
    const uint64_t* v_destinations,
    const int64_t* host_indices,
    const int64_t* device_indices,
    int64_t item_count,
    int64_t layer_count,
    int64_t heads,
    int64_t head_dim) {
  const int64_t per_token = heads * head_dim;
  const int64_t total = item_count * layer_count * per_token;
  const int64_t linear = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (linear >= total) {
    return;
  }
  const int64_t item_stride = layer_count * per_token;
  const int64_t item = linear / item_stride;
  const int64_t remainder = linear % item_stride;
  const int64_t layer = remainder / per_token;
  const int64_t feature = remainder % per_token;
  const int64_t host_slot = host_indices[item];
  const int64_t device_slot = device_indices[item];
  const int64_t host_offset = (host_slot * layer_count + layer) * per_token + feature;
  const int64_t device_offset = device_slot * per_token + feature;
  auto* k_destination = reinterpret_cast<scalar_t*>(k_destinations[layer]);
  auto* v_destination = reinterpret_cast<scalar_t*>(v_destinations[layer]);
  k_destination[device_offset] = host_k[host_offset];
  v_destination[device_offset] = host_v[host_offset];
}

}  // namespace

void restore_page_first_mha_to_raw(
    torch::Tensor host_k,
    torch::Tensor host_v,
    torch::Tensor k_destinations,
    torch::Tensor v_destinations,
    torch::Tensor host_indices,
    torch::Tensor device_indices,
    int64_t layer_count,
    int64_t heads,
    int64_t head_dim) {
  TORCH_CHECK(host_k.device().is_cpu() && host_v.device().is_cpu(), "host KV must be CPU memory");
  TORCH_CHECK(host_k.scalar_type() == host_v.scalar_type(), "host K/V dtypes differ");
  TORCH_CHECK(host_k.is_contiguous() && host_v.is_contiguous(), "host KV must be contiguous");
  TORCH_CHECK(k_destinations.is_cuda() && v_destinations.is_cuda(), "destination pointers must be CUDA tensors");
  TORCH_CHECK(host_indices.is_cuda() && device_indices.is_cuda(), "indices must be CUDA tensors");
  TORCH_CHECK(k_destinations.scalar_type() == torch::kUInt64 && v_destinations.scalar_type() == torch::kUInt64,
              "destination pointers must be uint64");
  TORCH_CHECK(host_indices.scalar_type() == torch::kInt64 && device_indices.scalar_type() == torch::kInt64,
              "indices must be int64");
  TORCH_CHECK(host_indices.numel() == device_indices.numel(), "index counts differ");
  TORCH_CHECK(k_destinations.numel() == layer_count && v_destinations.numel() == layer_count,
              "destination layer counts differ");

  constexpr int threads = 256;
  const int64_t total = host_indices.numel() * layer_count * heads * head_dim;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  const auto stream = at::cuda::getCurrentCUDAStream(host_indices.get_device());
  const auto launch = [&](auto* dtype_tag) {
    using scalar_t = std::remove_pointer_t<decltype(dtype_tag)>;
    restore_page_first_mha_to_raw_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        host_k.data_ptr<scalar_t>(), host_v.data_ptr<scalar_t>(),
        k_destinations.data_ptr<uint64_t>(), v_destinations.data_ptr<uint64_t>(),
        host_indices.data_ptr<int64_t>(), device_indices.data_ptr<int64_t>(),
        host_indices.numel(), layer_count, heads, head_dim);
  };
  switch (host_k.scalar_type()) {
    case at::ScalarType::Half:
      launch(static_cast<at::Half*>(nullptr));
      break;
    case at::ScalarType::BFloat16:
      launch(static_cast<at::BFloat16*>(nullptr));
      break;
    default:
      TORCH_CHECK(false, "Central I/O supports fp16 and bf16 HostKV only");
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("restore_page_first_mha_to_raw", &restore_page_first_mha_to_raw);
}
