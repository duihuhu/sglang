import abc
import bisect
import logging
import os
import threading
from collections import defaultdict
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional, Tuple

import psutil
import torch

from sglang.jit_kernel.hicache import (
    can_use_hicache_jit_kernel,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_all_layer as jit_transfer_hicache_all_layer,
)
from sglang.jit_kernel.hicache import (
    transfer_hicache_one_layer as jit_transfer_hicache_one_layer,
)
from sglang.srt.mem_cache.memory_pool import (
    KVCache,
    MHATokenToKVPool,
    MLATokenToKVPool,
    NSATokenToKVPool,
)
from sglang.srt.utils import is_cuda, is_mps, is_npu, is_xpu

_is_cuda = is_cuda()
_is_npu = is_npu()
_is_xpu = is_xpu()
_is_mps = is_mps()
if not (_is_npu or _is_xpu or _is_mps):
    from sgl_kernel.kvcacheio import (
        transfer_kv_all_layer,
        transfer_kv_all_layer_direct_lf_pf,
        transfer_kv_all_layer_lf_pf,
        transfer_kv_all_layer_lf_ph,
        transfer_kv_all_layer_mla,
        transfer_kv_all_layer_mla_lf_pf,
        transfer_kv_direct,
        transfer_kv_per_layer,
        transfer_kv_per_layer_direct_pf_lf,
        transfer_kv_per_layer_mla,
        transfer_kv_per_layer_mla_pf_lf,
        transfer_kv_per_layer_pf_lf,
        transfer_kv_per_layer_ph_lf,
    )
if _is_npu:
    from sgl_kernel_npu.kvcacheio import TransferDirection, transfer_kv_dim_exchange

logger = logging.getLogger(__name__)


@dataclass
class HostKVCacheExtent:
    extent_id: int
    base: int
    size: int
    state: str
    free_slots: torch.Tensor
    kv_buffer: Optional[torch.Tensor] = None


@dataclass
class HostKVCacheExtentGroup:
    extent_id: int
    local_indices: torch.Tensor
    paired_indices: Optional[torch.Tensor] = None


@dataclass
class HostKVCachePageDescriptor:
    extent_id: int
    global_start: int
    local_start: int
    local_page_id: int


class HostKVCacheExtentTable:
    """Track global host KV indices across multiple pinned-memory extents.

    This is the metadata/allocator substrate for a state-preserving
    multi-extent HostKVCache. It intentionally does not own real KV tensors yet;
    the first prototype only proves that global host indices can remain stable
    as extents are added or drained.
    """

    ACTIVE = "active"
    DRAINING = "draining"

    def __init__(self, page_size: int):
        self.page_size = page_size
        self.extents: List[HostKVCacheExtent] = []
        self.extent_bases: List[int] = []
        self.lock = threading.RLock()

    @property
    def size(self) -> int:
        if not self.extents:
            return 0
        last_extent = self.extents[-1]
        return last_extent.base + last_extent.size

    def add_extent(self, num_slots: int, kv_buffer: Optional[torch.Tensor] = None) -> int:
        assert (
            num_slots % self.page_size == 0
        ), "Extent size should be a multiple of the page size."
        with self.lock:
            extent_id = len(self.extents)
            base = self.size
            extent = HostKVCacheExtent(
                extent_id=extent_id,
                base=base,
                size=num_slots,
                state=self.ACTIVE,
                free_slots=torch.arange(base, base + num_slots, dtype=torch.int64),
                kv_buffer=kv_buffer,
            )
            self.extents.append(extent)
            self.extent_bases.append(base)
            return extent_id

    def extent_state(self, extent_id: int) -> str:
        return self.extents[extent_id].state

    def mark_draining(self, extent_id: int) -> None:
        with self.lock:
            self.extents[extent_id].state = self.DRAINING

    def available_size(self) -> int:
        return sum(
            len(extent.free_slots)
            for extent in self.extents
            if extent.state == self.ACTIVE
        )

    def clear(self) -> None:
        with self.lock:
            for extent in self.extents:
                extent.free_slots = torch.arange(
                    extent.base,
                    extent.base + extent.size,
                    dtype=torch.int64,
                    device=extent.free_slots.device,
                )

    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        with self.lock:
            selected_by_extent = []
            remaining = need_size
            for extent in self.extents:
                if extent.state != self.ACTIVE or remaining == 0:
                    continue
                selected = self._select_complete_pages(extent, remaining)
                if len(selected) == 0:
                    continue
                selected_by_extent.append((extent, selected))
                remaining -= len(selected)

            if remaining != 0:
                return None

            selected = []
            for extent, selected_indices in selected_by_extent:
                selected.append(
                    torch.tensor(
                        selected_indices,
                        dtype=extent.free_slots.dtype,
                        device=extent.free_slots.device,
                    )
                )
                self._remove_free_slots(extent, selected_indices)
            return torch.cat(selected)

    def free(self, indices: torch.Tensor) -> int:
        with self.lock:
            released = 0
            for extent in self.extents:
                mask = (indices >= extent.base) & (indices < extent.base + extent.size)
                if not torch.any(mask):
                    continue
                extent.free_slots = torch.cat([indices[mask], extent.free_slots])
                released += int(torch.sum(mask).item())
            return released

    def reserve_from_extent(self, extent_id: int, num_slots: int) -> torch.Tensor:
        assert (
            num_slots % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        with self.lock:
            extent = self.extents[extent_id]
            selected = self._select_complete_pages(extent, num_slots)
            if len(selected) != num_slots:
                raise ValueError(
                    f"Cannot reserve {num_slots} slots from extent {extent_id}; "
                    f"only found {len(selected)} complete-page slots."
                )
            self._remove_free_slots(extent, selected)
            return torch.tensor(
                selected,
                dtype=extent.free_slots.dtype,
                device=extent.free_slots.device,
            )

    def resolve_index(self, index: int) -> Tuple[int, int]:
        extent_pos = bisect.bisect_right(self.extent_bases, index) - 1
        if extent_pos >= 0:
            extent = self.extents[extent_pos]
            if index < extent.base + extent.size:
                return extent.extent_id, index - extent.base
        raise IndexError(f"Host KV index {index} does not belong to any extent.")

    def resolve_indices(self, indices: torch.Tensor) -> List[Tuple[int, int]]:
        return [self.resolve_index(int(index)) for index in indices.tolist()]

    def page_extent_id(self, indices: torch.Tensor) -> int:
        assert len(indices) == self.page_size, "Expected exactly one page of indices."
        extent_id, local_index = self.resolve_index(int(indices[0]))
        assert local_index % self.page_size == 0, "Page should start at a page boundary."
        expected = torch.arange(
            int(indices[0]),
            int(indices[0]) + self.page_size,
            dtype=indices.dtype,
            device=indices.device,
        )
        assert torch.equal(indices, expected), "Page indices should be contiguous."
        for index in indices.tolist()[1:]:
            next_extent_id, _ = self.resolve_index(int(index))
            assert next_extent_id == extent_id, "A page must not cross extents."
        return extent_id

    def group_indices_by_extent(
        self, indices: torch.Tensor, paired_indices: Optional[torch.Tensor] = None
    ) -> List[HostKVCacheExtentGroup]:
        if paired_indices is not None:
            assert len(indices) == len(
                paired_indices
            ), "indices and paired_indices should have the same length."

        grouped_local_indices = defaultdict(list)
        grouped_paired_indices = defaultdict(list)
        for offset, index in enumerate(indices.tolist()):
            extent_id, local_index = self.resolve_index(int(index))
            grouped_local_indices[extent_id].append(local_index)
            if paired_indices is not None:
                grouped_paired_indices[extent_id].append(
                    int(paired_indices[offset].item())
                )

        groups = []
        for extent in self.extents:
            extent_id = extent.extent_id
            if extent_id not in grouped_local_indices:
                continue
            local_tensor = torch.tensor(
                grouped_local_indices[extent_id],
                dtype=indices.dtype,
                device=indices.device,
            )
            paired_tensor = None
            if paired_indices is not None:
                paired_tensor = torch.tensor(
                    grouped_paired_indices[extent_id],
                    dtype=paired_indices.dtype,
                    device=paired_indices.device,
                )
            groups.append(
                HostKVCacheExtentGroup(
                    extent_id=extent_id,
                    local_indices=local_tensor,
                    paired_indices=paired_tensor,
                )
            )
        return groups

    def group_pages_by_extent(
        self, indices: torch.Tensor, paired_indices: Optional[torch.Tensor] = None
    ) -> List[HostKVCacheExtentGroup]:
        assert (
            len(indices) % self.page_size == 0
        ), "Expected indices to contain complete pages."
        if paired_indices is not None:
            assert len(indices) == len(
                paired_indices
            ), "indices and paired_indices should have the same length."

        for offset in range(0, len(indices), self.page_size):
            self.page_extent_id(indices[offset : offset + self.page_size])

        return self.group_indices_by_extent(indices, paired_indices)

    def page_descriptors(self, indices: torch.Tensor) -> List[HostKVCachePageDescriptor]:
        assert (
            len(indices) % self.page_size == 0
        ), "Expected indices to contain complete pages."
        descriptors = []
        for offset in range(0, len(indices), self.page_size):
            page = indices[offset : offset + self.page_size]
            extent_id = self.page_extent_id(page)
            global_start = int(page[0])
            _, local_start = self.resolve_index(global_start)
            descriptors.append(
                HostKVCachePageDescriptor(
                    extent_id=extent_id,
                    global_start=global_start,
                    local_start=local_start,
                    local_page_id=local_start // self.page_size,
                )
            )
        return descriptors

    def get_data_page(self, index: int, layout: str, flat: bool = True) -> torch.Tensor:
        descriptor = self._page_descriptor_for_start(index)
        extent = self.extents[descriptor.extent_id]
        assert extent.kv_buffer is not None, "Extent does not own a KV buffer."
        data_page = self._slice_data_page(
            extent.kv_buffer, descriptor.local_start, descriptor.local_page_id, layout
        )
        if flat:
            return data_page.flatten()
        return data_page

    def set_from_flat_data_page(
        self, index: int, data_page: torch.Tensor, layout: str
    ) -> None:
        descriptor = self._page_descriptor_for_start(index)
        extent = self.extents[descriptor.extent_id]
        assert extent.kv_buffer is not None, "Extent does not own a KV buffer."
        target = self._slice_data_page(
            extent.kv_buffer, descriptor.local_start, descriptor.local_page_id, layout
        )
        target.copy_(data_page.reshape(target.shape))

    def get_page_buffer_meta(
        self,
        indices: torch.Tensor,
        layout: str,
        layer_num: int,
        head_num: int,
        head_dim: int,
    ):
        ptr_list = []
        element_size_list = []
        for descriptor in self.page_descriptors(indices):
            extent = self.extents[descriptor.extent_id]
            assert extent.kv_buffer is not None, "Extent does not own a KV buffer."
            kv_buffer_data_ptr = extent.kv_buffer.data_ptr()
            item_size = extent.kv_buffer.element_size()
            extent_size = extent.size
            v_offset = layer_num * extent_size * head_num * head_dim * item_size

            if layout == "layer_first":
                element_size = item_size * self.page_size * head_num * head_dim
                for layer_id in range(layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + descriptor.local_start * head_num * head_dim * item_size
                        + layer_id * extent_size * head_num * head_dim * item_size
                    )
                    ptr_list.append(k_ptr)
                    ptr_list.append(k_ptr + v_offset)
                    element_size_list.extend([element_size, element_size])
            elif layout in ["page_first", "page_first_direct", "page_head"]:
                element_size = (
                    layer_num * item_size * self.page_size * head_num * head_dim
                )
                k_ptr = (
                    kv_buffer_data_ptr
                    + descriptor.local_start * layer_num * head_num * head_dim * item_size
                )
                ptr_list.append(k_ptr)
                ptr_list.append(k_ptr + v_offset)
                element_size_list.extend([element_size, element_size])
            else:
                raise ValueError(f"Unsupported layout: {layout}")
        return ptr_list, element_size_list

    def get_split_heads_page_buffer_meta(
        self,
        indices: torch.Tensor,
        split_factor: int,
        layer_num: int,
        head_num: int,
        head_dim: int,
    ):
        assert len(indices) % self.page_size == 0
        assert head_num % split_factor == 0
        ptr_list = []
        element_size_list = []
        for descriptor in self.page_descriptors(indices):
            extent = self.extents[descriptor.extent_id]
            assert extent.kv_buffer is not None, "Extent does not own a KV buffer."
            kv_buffer_data_ptr = extent.kv_buffer.data_ptr()
            item_size = extent.kv_buffer.element_size()
            extent_size = extent.size
            v_offset = layer_num * extent_size * head_num * head_dim * item_size
            for head_id in range(0, head_num, head_num // split_factor):
                k_ptr = (
                    kv_buffer_data_ptr
                    + descriptor.local_start
                    * layer_num
                    * head_num
                    * head_dim
                    * item_size
                    + head_id * self.page_size * layer_num * head_dim * item_size
                )
                ptr_list.append(k_ptr)
                ptr_list.append(k_ptr + v_offset)
        element_size = (
            layer_num * item_size * self.page_size * head_num * head_dim // split_factor
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list

    def get_mla_data_page(
        self, index: int, layout: str, flat: bool = True
    ) -> torch.Tensor:
        descriptor = self._page_descriptor_for_start(index)
        extent = self.extents[descriptor.extent_id]
        assert extent.kv_buffer is not None, "Extent does not own a KV buffer."
        data_page = self._slice_mla_data_page(
            extent.kv_buffer, descriptor.local_start, descriptor.local_page_id, layout
        )
        if flat:
            return data_page.flatten()
        return data_page

    def set_mla_from_flat_data_page(
        self, index: int, data_page: torch.Tensor, layout: str
    ) -> None:
        descriptor = self._page_descriptor_for_start(index)
        extent = self.extents[descriptor.extent_id]
        assert extent.kv_buffer is not None, "Extent does not own a KV buffer."
        target = self._slice_mla_data_page(
            extent.kv_buffer, descriptor.local_start, descriptor.local_page_id, layout
        )
        target.copy_(data_page.reshape(target.shape))

    def get_mla_page_buffer_meta(
        self,
        indices: torch.Tensor,
        layout: str,
        layer_num: int,
        kv_cache_dim: int,
    ):
        ptr_list = []
        element_size_list = []
        for descriptor in self.page_descriptors(indices):
            extent = self.extents[descriptor.extent_id]
            assert extent.kv_buffer is not None, "Extent does not own a KV buffer."
            kv_buffer_data_ptr = extent.kv_buffer.data_ptr()
            item_size = extent.kv_buffer.element_size()
            extent_size = extent.size
            if layout == "layer_first":
                element_size = item_size * self.page_size * kv_cache_dim
                for layer_id in range(layer_num):
                    ptr_list.append(
                        kv_buffer_data_ptr
                        + descriptor.local_start * kv_cache_dim * item_size
                        + layer_id * extent_size * kv_cache_dim * item_size
                    )
                    element_size_list.append(element_size)
            elif layout in ["page_first", "page_first_direct"]:
                ptr_list.append(
                    kv_buffer_data_ptr
                    + descriptor.local_start * layer_num * kv_cache_dim * item_size
                )
                element_size_list.append(
                    layer_num * item_size * self.page_size * kv_cache_dim
                )
            else:
                raise ValueError(f"Unsupported MLA layout: {layout}")
        return ptr_list, element_size_list

    def dispatch_transfer_groups(
        self,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        transfer_fn,
    ) -> None:
        for group in self.group_indices_by_extent(
            host_indices, paired_indices=device_indices
        ):
            extent = self.extents[group.extent_id]
            assert extent.kv_buffer is not None, "Extent does not own a KV buffer."
            transfer_fn(
                extent.kv_buffer,
                group.local_indices,
                group.paired_indices,
                group.extent_id,
            )

    def copy_from_device(
        self,
        device_buffer: torch.Tensor,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        layout: str,
    ) -> None:
        def copy_to_extent(kv_buffer, local_host_indices, paired_device_indices, _):
            self._copy_between_buffers(
                src_buffer=device_buffer,
                src_indices=paired_device_indices,
                dst_buffer=kv_buffer,
                dst_indices=local_host_indices,
                layout=layout,
            )

        self.dispatch_transfer_groups(host_indices, device_indices, copy_to_extent)

    def copy_to_device(
        self,
        device_buffer: torch.Tensor,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        layout: str,
    ) -> None:
        def copy_from_extent(kv_buffer, local_host_indices, paired_device_indices, _):
            self._copy_between_buffers(
                src_buffer=kv_buffer,
                src_indices=local_host_indices,
                dst_buffer=device_buffer,
                dst_indices=paired_device_indices,
                layout=layout,
            )

        self.dispatch_transfer_groups(host_indices, device_indices, copy_from_extent)

    def _select_complete_pages(
        self, extent: HostKVCacheExtent, max_slots: int
    ) -> List[int]:
        max_slots = min(max_slots, len(extent.free_slots))
        max_pages = max_slots // self.page_size
        if max_pages == 0:
            return []

        free_values = {int(index) for index in extent.free_slots.tolist()}
        selected = []
        for page_start in range(extent.base, extent.base + extent.size, self.page_size):
            page = list(range(page_start, page_start + self.page_size))
            if all(index in free_values for index in page):
                selected.extend(page)
                if len(selected) == max_pages * self.page_size:
                    break
        return selected

    def _remove_free_slots(
        self, extent: HostKVCacheExtent, selected_indices: List[int]
    ) -> None:
        selected_set = set(selected_indices)
        remaining = [
            int(index)
            for index in extent.free_slots.tolist()
            if int(index) not in selected_set
        ]
        extent.free_slots = torch.tensor(
            remaining, dtype=extent.free_slots.dtype, device=extent.free_slots.device
        )

    def _page_descriptor_for_start(self, index: int) -> HostKVCachePageDescriptor:
        page = torch.arange(index, index + self.page_size, dtype=torch.int64)
        descriptors = self.page_descriptors(page)
        assert len(descriptors) == 1
        return descriptors[0]

    def _slice_data_page(
        self,
        kv_buffer: torch.Tensor,
        local_start: int,
        local_page_id: int,
        layout: str,
    ) -> torch.Tensor:
        if layout == "layer_first":
            return kv_buffer[:, :, local_start : local_start + self.page_size, :, :]
        if layout == "page_first":
            return kv_buffer[:, local_start : local_start + self.page_size, :, :, :]
        if layout in ["page_first_direct", "page_head"]:
            return kv_buffer[:, local_page_id : local_page_id + 1, :, :, :, :]
        raise ValueError(f"Unsupported layout: {layout}")

    def _slice_mla_data_page(
        self,
        kv_buffer: torch.Tensor,
        local_start: int,
        local_page_id: int,
        layout: str,
    ) -> torch.Tensor:
        if layout == "layer_first":
            return kv_buffer[:, local_start : local_start + self.page_size, :, :]
        if layout == "page_first":
            return kv_buffer[local_start : local_start + self.page_size, :, :, :]
        if layout == "page_first_direct":
            return kv_buffer[local_page_id : local_page_id + 1, :, :, :, :]
        raise ValueError(f"Unsupported MLA layout: {layout}")

    def _copy_between_buffers(
        self,
        src_buffer: torch.Tensor,
        src_indices: torch.Tensor,
        dst_buffer: torch.Tensor,
        dst_indices: torch.Tensor,
        layout: str,
    ) -> None:
        if layout == "layer_first":
            dst_buffer[:, :, dst_indices, :, :] = src_buffer[:, :, src_indices, :, :]
        elif layout == "page_first":
            dst_buffer[:, dst_indices, :, :, :] = src_buffer[:, src_indices, :, :, :]
        else:
            raise ValueError(f"Unsupported layout for reference copy: {layout}")


def synchronized(func):
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return func(self, *args, **kwargs)

    return wrapper


class HostTensorAllocator(abc.ABC):
    def __init__(self):
        """Initialize the HostTensorAllocator."""
        self.dtype = None
        self.dims = None

    def allocate(self, dims: tuple, dtype: torch.dtype, device: str) -> torch.Tensor:
        """Allocate a tensor of given dims and dtype on the memory."""
        self.dtype = dtype
        self.dims = dims
        tensor = torch.empty(dims, dtype=dtype, device=device)
        return tensor


def get_allocator_from_storage(allocator_type):
    if allocator_type == "mooncake":
        try:
            from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                MooncakeHostTensorAllocator,
            )

            return MooncakeHostTensorAllocator()
        except ImportError:
            logger.warning(
                "Mooncake's tensor allocator requires mooncake >= 0.3.8.post1. "
                "Please upgrade Mooncake by 'pip install mooncake-transfer-engine --upgrade'. "
                "Fallback to use default allocator."
            )
            return HostTensorAllocator()
    else:
        return HostTensorAllocator()


def alloc_with_host_register(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: HostTensorAllocator,
) -> torch.Tensor:
    """
    Allocate tensor and register host memory with cudaHostRegister.
    CudaHostRegister only applies when pin_memory=True.
    """
    buffer = allocator.allocate(dims, dtype=dtype, device=device)
    if pin_memory:
        torch.cuda.cudart().cudaHostRegister(
            buffer.data_ptr(), buffer.numel() * buffer.element_size(), 0
        )
    return buffer


def alloc_with_pin_memory(
    dims,
    dtype: torch.dtype,
    device: str,
    pin_memory: bool,
    allocator: None,
) -> torch.Tensor:
    """
    Allocate tensor using PyTorch's built-in pin_memory flag.
    """
    buffer = torch.empty(dims, dtype=dtype, device=device, pin_memory=pin_memory)
    return buffer


ALLOC_MEMORY_FUNCS = defaultdict(
    lambda: alloc_with_host_register,
    {
        "npu": alloc_with_pin_memory,
    },
)


class HostKVCache(abc.ABC):

    def __init__(
        self,
        device_pool: KVCache,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool,
        device: str,
        allocator_type: str = "default",
    ):
        self.device_pool = device_pool
        self.page_size = page_size
        self.layout = layout
        self.pin_memory = pin_memory
        self.device = device
        self.allocator = get_allocator_from_storage(allocator_type)

        self.dtype = device_pool.store_dtype
        self.size_per_token = self.get_size_per_token()
        if host_size > 0:
            self.size = int(host_size * 1e9 // self.size_per_token)
        else:
            self.size = int(device_pool.size * host_to_device_ratio)
        # Align up the host memory pool size to the page size
        self.page_num = self.size // self.page_size + 1
        self.size = self.page_num * self.page_size
        self.start_layer = device_pool.start_layer
        self.end_layer = device_pool.end_layer

        assert (
            self.size > device_pool.size
        ), "The host memory should be larger than the device memory with the current protocol"

        # Verify there is enough available host memory.
        host_mem = psutil.virtual_memory()
        requested_bytes = self.size * self.size_per_token
        # preserve at least 10GB for other usage
        ten_gb = 10 * (1024**3)
        available_bytes = host_mem.available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free. Please reduce the "
                f"size of the hierarchical cache."
            )
        else:
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    def _record_extent_transfer_groups(
        self,
        direction: str,
        io_backend: str,
        layout: str,
        groups: List[HostKVCacheExtentGroup],
        layer_id: Optional[int] = None,
    ) -> None:
        if os.environ.get("SGLANG_TEST_HICACHE_TRACE_EXTENT_TRANSFERS", "0") != "1":
            return

        size_per_token = int(getattr(self, "size_per_token", 0))
        layer_num = int(getattr(self, "layer_num", 1))
        if layer_id is not None and layer_num > 0:
            bytes_per_token = size_per_token // layer_num
        else:
            bytes_per_token = size_per_token

        def get_page_run_stats(indices: torch.Tensor) -> Dict[str, float]:
            if len(indices) == 0:
                return {
                    "pages": 0,
                    "page_runs": 0,
                    "max_run_pages": 0,
                    "avg_run_pages": 0.0,
                }

            page_ids = []
            for offset in range(0, len(indices), self.page_size):
                page_ids.append(int(indices[offset].item()) // self.page_size)

            run_count = 0
            current_run_pages = 0
            max_run_pages = 0
            previous_page_id = None
            for page_id in page_ids:
                if previous_page_id is None or page_id != previous_page_id + 1:
                    if current_run_pages > 0:
                        max_run_pages = max(max_run_pages, current_run_pages)
                    run_count += 1
                    current_run_pages = 1
                else:
                    current_run_pages += 1
                previous_page_id = page_id

            max_run_pages = max(max_run_pages, current_run_pages)
            page_count = len(page_ids)
            return {
                "pages": page_count,
                "page_runs": run_count,
                "max_run_pages": max_run_pages,
                "avg_run_pages": float(page_count / run_count) if run_count else 0.0,
            }

        group_records = []
        total_pages = 0
        total_page_runs = 0
        max_run_pages = 0
        for group in groups:
            run_stats = get_page_run_stats(group.local_indices)
            group_tokens = int(len(group.local_indices))
            group_bytes = group_tokens * bytes_per_token
            total_pages += int(run_stats["pages"])
            total_page_runs += int(run_stats["page_runs"])
            max_run_pages = max(max_run_pages, int(run_stats["max_run_pages"]))
            group_records.append(
                {
                    "extent_id": int(group.extent_id),
                    "tokens": group_tokens,
                    "bytes": group_bytes,
                    "pages": int(run_stats["pages"]),
                    "page_runs": int(run_stats["page_runs"]),
                    "max_run_pages": int(run_stats["max_run_pages"]),
                    "avg_run_pages": float(run_stats["avg_run_pages"]),
                }
            )

        records = getattr(self, "_test_extent_transfer_records", [])
        records.append(
            {
                "direction": direction,
                "io_backend": io_backend,
                "layout": layout,
                "layer_id": int(layer_id) if layer_id is not None else None,
                "group_count": len(groups),
                "total_tokens": sum(len(group.local_indices) for group in groups),
                "bytes": sum(group["bytes"] for group in group_records),
                "bytes_per_token": bytes_per_token,
                "pages": total_pages,
                "page_runs": total_page_runs,
                "max_run_pages": max_run_pages,
                "avg_run_pages": (
                    float(total_pages / total_page_runs) if total_page_runs else 0.0
                ),
                "groups": group_records,
            }
        )
        self._test_extent_transfer_records = records[-200:]

    def _record_single_extent_transfer(
        self,
        direction: str,
        io_backend: str,
        layout: str,
        host_indices: torch.Tensor,
        device_indices: torch.Tensor,
        layer_id: Optional[int] = None,
    ) -> None:
        if os.environ.get("SGLANG_TEST_HICACHE_TRACE_EXTENT_TRANSFERS", "0") != "1":
            return
        self._record_extent_transfer_groups(
            direction,
            io_backend,
            layout,
            [
                HostKVCacheExtentGroup(
                    extent_id=0,
                    local_indices=host_indices,
                    paired_indices=device_indices,
                )
            ],
            layer_id=layer_id,
        )

    def _summarize_extent_transfer_records(self) -> Dict[str, object]:
        records = getattr(self, "_test_extent_transfer_records", [])
        directions = {}
        total_bytes = 0
        total_pages = 0
        total_page_runs = 0
        max_run_pages = 0

        for record in records:
            direction = record.get("direction", "unknown")
            direction_summary = directions.setdefault(
                direction,
                {
                    "records": 0,
                    "bytes": 0,
                    "tokens": 0,
                    "pages": 0,
                    "page_runs": 0,
                    "max_run_pages": 0,
                },
            )
            record_bytes = int(record.get("bytes", 0))
            record_tokens = int(record.get("total_tokens", 0))
            record_pages = int(record.get("pages", 0))
            record_page_runs = int(record.get("page_runs", 0))
            record_max_run_pages = int(record.get("max_run_pages", 0))

            total_bytes += record_bytes
            total_pages += record_pages
            total_page_runs += record_page_runs
            max_run_pages = max(max_run_pages, record_max_run_pages)

            direction_summary["records"] += 1
            direction_summary["bytes"] += record_bytes
            direction_summary["tokens"] += record_tokens
            direction_summary["pages"] += record_pages
            direction_summary["page_runs"] += record_page_runs
            direction_summary["max_run_pages"] = max(
                direction_summary["max_run_pages"], record_max_run_pages
            )

        for direction_summary in directions.values():
            page_runs = int(direction_summary["page_runs"])
            pages = int(direction_summary["pages"])
            direction_summary["avg_run_pages"] = (
                float(pages / page_runs) if page_runs else 0.0
            )

        return {
            "record_count": len(records),
            "total_bytes": total_bytes,
            "directions": directions,
            "page_run_records": {
                "records": len(records),
                "pages": total_pages,
                "page_runs": total_page_runs,
                "max_run_pages": max_run_pages,
                "avg_run_pages": (
                    float(total_pages / total_page_runs) if total_page_runs else 0.0
                ),
            },
        }

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
        raise NotImplementedError()

    def init_extent_kv_buffer(self, num_slots: int):
        raise NotImplementedError()

    @abc.abstractmethod
    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        """
        Load KV data from the host memory pool to the device memory pool for a specific layer.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        """
        Backup KV data from the device memory pool to the host memory pool for all layers.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        """
        Get a flat data page from the host memory pool.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def get_dummy_flat_data_page(self) -> torch.Tensor:
        """
        Get a dummy flat data page from the host memory pool.
        This is used for prefetching or initializing empty pages.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        """
        Set a flat data page to the host memory pool.
        """
        raise NotImplementedError()

    @synchronized
    def clear(self):
        # Initialize memory states and tracking structures.
        self.mem_state = torch.zeros(
            (self.size,), dtype=torch.uint8, device=self.device
        )
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            extent_table.clear()
            return
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table.available_size()
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table.alloc(need_size)
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table.free(indices)
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)

    def _ensure_extent_table_from_current_buffer(self) -> HostKVCacheExtentTable:
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table

        extent_table = HostKVCacheExtentTable(page_size=self.page_size)
        extent_table.add_extent(num_slots=self.size, kv_buffer=self.kv_buffer)
        extent_table.extents[0].free_slots = self.free_slots.clone()
        self.extent_table = extent_table
        return extent_table

    @synchronized
    def grow_extent_online(self, num_slots: int) -> int:
        assert num_slots > 0, "The new extent must contain at least one slot."
        grow_slots = ((num_slots + self.page_size - 1) // self.page_size) * self.page_size
        requested_bytes = grow_slots * self.size_per_token
        ten_gb = 10 * (1024**3)
        available_bytes = psutil.virtual_memory().available - ten_gb
        if requested_bytes > available_bytes:
            raise ValueError(
                f"Not enough host memory available. Requesting "
                f"{requested_bytes / 1e9:.2f} GB but only have "
                f"{available_bytes / 1e9:.2f} GB free."
            )

        extent_table = self._ensure_extent_table_from_current_buffer()
        kv_buffer = self.init_extent_kv_buffer(grow_slots)
        extent_id = extent_table.add_extent(num_slots=grow_slots, kv_buffer=kv_buffer)

        self.size += grow_slots
        self.page_num += grow_slots // self.page_size
        self.mem_state = torch.cat(
            [
                self.mem_state,
                torch.zeros((grow_slots,), dtype=torch.uint8, device=self.device),
            ]
        )
        return extent_id


class MHATokenToKVPoolHost(HostKVCache):
    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        self.element_dim = self.device_pool.head_num * self.device_pool.head_dim
        self.can_use_jit = _is_cuda and can_use_hicache_jit_kernel(
            element_size=self.element_dim * self.dtype.itemsize
        )

        self.k_data_refs = [self.k_buffer[i] for i in range(self.layer_num)]
        self.v_data_refs = [self.v_buffer[i] for i in range(self.layer_num)]
        self.k_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.k_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        self.v_data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.v_data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num

        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def get_ksize_per_token(self):
        return self.get_size_per_token() // 2

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (2, self.layer_num, self.size, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, self.size, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                self.page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                self.page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.head_num * self.head_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    def init_extent_kv_buffer(self, num_slots: int):
        page_num = num_slots // self.page_size
        if self.layout == "layer_first":
            dims = (2, self.layer_num, num_slots, self.head_num, self.head_dim)
        elif self.layout == "page_first":
            dims = (2, num_slots, self.layer_num, self.head_num, self.head_dim)
        elif self.layout == "page_first_direct":
            dims = (
                2,
                page_num,
                self.layer_num,
                self.page_size,
                self.head_num,
                self.head_dim,
            )
        elif self.layout == "page_head":
            dims = (
                2,
                page_num,
                self.head_num,
                self.page_size,
                self.layer_num,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

        if self.pin_memory and self.device == "cpu" and self.device_pool.device == "cuda":
            return torch.empty(dims, dtype=self.dtype, device=self.device, pin_memory=True)

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        return alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )

    @property
    def k_buffer(self):
        return self.kv_buffer[0]

    @property
    def v_buffer(self):
        return self.kv_buffer[1]

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            if io_backend == "kernel":
                if self.layout == "page_first":
                    def load_from_extent(
                        kv_buffer, local_host_indices, paired_device_indices, _
                    ):
                        transfer_kv_per_layer_pf_lf(
                            src_k=kv_buffer[0],
                            dst_k=device_pool.k_buffer[layer_id],
                            src_v=kv_buffer[1],
                            dst_v=device_pool.v_buffer[layer_id],
                            src_indices=local_host_indices,
                            dst_indices=paired_device_indices,
                            layer_id=layer_id,
                            item_size=self.token_stride_size,
                            src_layout_dim=self.layout_dim,
                        )

                    extent_table.dispatch_transfer_groups(
                        host_indices, device_indices, load_from_extent
                    )
                    return
                elif self.layout == "page_head":
                    def load_from_extent(
                        kv_buffer, local_host_indices, paired_device_indices, _
                    ):
                        transfer_kv_per_layer_ph_lf(
                            src_k=kv_buffer[0],
                            dst_k=device_pool.k_buffer[layer_id],
                            src_v=kv_buffer[1],
                            dst_v=device_pool.v_buffer[layer_id],
                            src_indices=local_host_indices,
                            dst_indices=paired_device_indices,
                            layer_id=layer_id,
                            item_size=self.token_stride_size,
                            src_layout_dim=self.layout_dim,
                            page_size=self.page_size,
                            head_num=self.head_num,
                        )

                    extent_table.dispatch_transfer_groups(
                        host_indices, device_indices, load_from_extent
                    )
                    return
            elif io_backend == "direct":
                if self.layout == "page_first_direct":
                    def load_from_extent(
                        kv_buffer, local_host_indices, paired_device_indices, _
                    ):
                        transfer_kv_per_layer_direct_pf_lf(
                            src_ptrs=[kv_buffer[0], kv_buffer[1]],
                            dst_ptrs=[
                                device_pool.k_buffer[layer_id],
                                device_pool.v_buffer[layer_id],
                            ],
                            src_indices=local_host_indices,
                            dst_indices=paired_device_indices,
                            layer_id=layer_id,
                            page_size=self.page_size,
                        )

                    groups = extent_table.group_indices_by_extent(
                        host_indices, paired_indices=device_indices
                    )
                    self._record_extent_transfer_groups(
                        "H2D", io_backend, self.layout, groups, layer_id=layer_id
                    )
                    extent_table.dispatch_transfer_groups(
                        host_indices, device_indices, load_from_extent
                    )
                    return
            raise ValueError(
                f"Unsupported multi-extent transfer path: {io_backend=} {self.layout=}"
            )

        if io_backend == "kernel":
            if self.layout == "layer_first":
                self._record_single_extent_transfer(
                    "H2D", io_backend, self.layout, host_indices, device_indices, layer_id
                )
                if self.can_use_jit:
                    jit_transfer_hicache_one_layer(
                        k_cache_dst=device_pool.k_buffer[layer_id],
                        v_cache_dst=device_pool.v_buffer[layer_id],
                        k_cache_src=self.k_buffer[layer_id],
                        v_cache_src=self.v_buffer[layer_id],
                        indices_dst=device_indices,
                        indices_src=host_indices,
                        element_dim=self.element_dim,
                    )
                else:
                    transfer_kv_per_layer(
                        src_k=self.k_buffer[layer_id],
                        dst_k=device_pool.k_buffer[layer_id],
                        src_v=self.v_buffer[layer_id],
                        dst_v=device_pool.v_buffer[layer_id],
                        src_indices=host_indices,
                        dst_indices=device_indices,
                        item_size=self.token_stride_size,
                    )
            elif self.layout == "page_first":
                self._record_single_extent_transfer(
                    "H2D", io_backend, self.layout, host_indices, device_indices, layer_id
                )
                transfer_kv_per_layer_pf_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                )
            elif self.layout == "page_head":
                self._record_single_extent_transfer(
                    "H2D", io_backend, self.layout, host_indices, device_indices, layer_id
                )
                transfer_kv_per_layer_ph_lf(
                    src_k=self.k_buffer,
                    dst_k=device_pool.k_buffer[layer_id],
                    src_v=self.v_buffer,
                    dst_v=device_pool.v_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                self._record_single_extent_transfer(
                    "H2D", io_backend, self.layout, host_indices, device_indices, layer_id
                )
                transfer_kv_direct(
                    src_layers=[self.k_buffer[layer_id], self.v_buffer[layer_id]],
                    dst_layers=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                self._record_single_extent_transfer(
                    "H2D", io_backend, self.layout, host_indices, device_indices, layer_id
                )
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.k_buffer, self.v_buffer],
                    dst_ptrs=[
                        device_pool.k_buffer[layer_id],
                        device_pool.v_buffer[layer_id],
                    ],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            if io_backend == "kernel":
                if self.layout == "page_first":
                    def backup_to_extent(
                        kv_buffer, local_host_indices, paired_device_indices, _
                    ):
                        transfer_kv_all_layer_lf_pf(
                            src_k_layers=device_pool.k_data_ptrs,
                            dst_k=kv_buffer[0],
                            src_v_layers=device_pool.v_data_ptrs,
                            dst_v=kv_buffer[1],
                            src_indices=paired_device_indices,
                            dst_indices=local_host_indices,
                            item_size=self.token_stride_size,
                            dst_layout_dim=self.layout_dim,
                            num_layers=self.layer_num,
                        )

                    extent_table.dispatch_transfer_groups(
                        host_indices, device_indices, backup_to_extent
                    )
                    return
                elif self.layout == "page_head":
                    def backup_to_extent(
                        kv_buffer, local_host_indices, paired_device_indices, _
                    ):
                        transfer_kv_all_layer_lf_ph(
                            src_k_layers=device_pool.k_data_ptrs,
                            dst_k=kv_buffer[0],
                            src_v_layers=device_pool.v_data_ptrs,
                            dst_v=kv_buffer[1],
                            src_indices=paired_device_indices,
                            dst_indices=local_host_indices,
                            item_size=self.token_stride_size,
                            dst_layout_dim=self.layout_dim,
                            num_layers=self.layer_num,
                            page_size=self.page_size,
                            head_num=self.head_num,
                        )

                    extent_table.dispatch_transfer_groups(
                        host_indices, device_indices, backup_to_extent
                    )
                    return
            elif io_backend == "direct":
                if self.layout == "page_first_direct":
                    def backup_to_extent(
                        kv_buffer, local_host_indices, paired_device_indices, _
                    ):
                        transfer_kv_all_layer_direct_lf_pf(
                            src_ptrs=device_pool.k_buffer + device_pool.v_buffer,
                            dst_ptrs=[kv_buffer[0], kv_buffer[1]],
                            src_indices=paired_device_indices,
                            dst_indices=local_host_indices,
                            page_size=self.page_size,
                        )

                    groups = extent_table.group_indices_by_extent(
                        host_indices, paired_indices=device_indices
                    )
                    self._record_extent_transfer_groups(
                        "D2H", io_backend, self.layout, groups
                    )
                    extent_table.dispatch_transfer_groups(
                        host_indices, device_indices, backup_to_extent
                    )
                    return
            raise ValueError(
                f"Unsupported multi-extent transfer path: {io_backend=} {self.layout=}"
            )

        if io_backend == "kernel":
            if self.layout == "layer_first":
                self._record_single_extent_transfer(
                    "D2H", io_backend, self.layout, host_indices, device_indices
                )
                if self.can_use_jit:
                    jit_transfer_hicache_all_layer(
                        k_ptr_dst=self.k_data_ptrs,
                        v_ptr_dst=self.v_data_ptrs,
                        indices_dst=host_indices,
                        k_ptr_src=device_pool.k_data_ptrs,
                        v_ptr_src=device_pool.v_data_ptrs,
                        indices_src=device_indices,
                        kv_cache_dst_stride_bytes=self.token_stride_size,
                        kv_cache_src_stride_bytes=self.token_stride_size,
                        element_size=self.element_dim * self.dtype.itemsize,
                    )
                else:
                    transfer_kv_all_layer(
                        src_k_layers=device_pool.k_data_ptrs,
                        dst_k_layers=self.k_data_ptrs,
                        src_v_layers=device_pool.v_data_ptrs,
                        dst_v_layers=self.v_data_ptrs,
                        src_indices=device_indices,
                        dst_indices=host_indices,
                        item_size=self.token_stride_size,
                        num_layers=self.layer_num,
                    )
            elif self.layout == "page_first":
                self._record_single_extent_transfer(
                    "D2H", io_backend, self.layout, host_indices, device_indices
                )
                transfer_kv_all_layer_lf_pf(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_head":
                self._record_single_extent_transfer(
                    "D2H", io_backend, self.layout, host_indices, device_indices
                )
                transfer_kv_all_layer_lf_ph(
                    src_k_layers=device_pool.k_data_ptrs,
                    dst_k=self.k_buffer,
                    src_v_layers=device_pool.v_data_ptrs,
                    dst_v=self.v_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                    page_size=self.page_size,
                    head_num=self.head_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                self._record_single_extent_transfer(
                    "D2H", io_backend, self.layout, host_indices, device_indices
                )
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,
                    dst_layers=self.k_data_refs + self.v_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                self._record_single_extent_transfer(
                    "D2H", io_backend, self.layout, host_indices, device_indices
                )
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.k_buffer + device_pool.v_buffer,
                    dst_ptrs=[self.k_buffer, self.v_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_direct":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table.get_data_page(index, self.layout, flat)

        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, :, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :, :]
        elif self.layout in ["page_first_direct", "page_head"]:
            real_index = index // self.page_size
            data_page = self.kv_buffer[:, real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (2, self.layer_num, self.page_size, self.head_num, self.head_dim),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            extent_table.set_from_flat_data_page(index, data_page, self.layout)
            return

        if self.layout == "layer_first":
            self.kv_buffer[:, :, index : index + self.page_size, :, :] = (
                data_page.reshape(
                    2,
                    self.layer_num,
                    self.page_size,
                    self.head_num,
                    self.head_dim,
                )
            )
        elif self.layout == "page_first":
            self.kv_buffer[:, index : index + self.page_size, :, :, :] = (
                data_page.reshape(
                    2, self.page_size, self.layer_num, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.layer_num, self.page_size, self.head_num, self.head_dim
                )
            )
        elif self.layout == "page_head":
            real_index = index // self.page_size
            self.kv_buffer[:, real_index : real_index + 1, :, :, :, :] = (
                data_page.reshape(
                    2, 1, self.head_num, self.page_size, self.layer_num, self.head_dim
                )
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_split_heads_page_buffer_meta(
        self, indices: torch.Tensor, split_factor: int
    ):
        """
        get meta data for zero copy of heterogeneous ranks' KVCache
        """
        assert self.layout == "page_head"
        assert len(indices) % self.page_size == 0
        assert self.head_num % split_factor == 0
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table.get_split_heads_page_buffer_meta(
                indices,
                split_factor,
                self.layer_num,
                self.head_num,
                self.head_dim,
            )

        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        for index in range(0, len(indices), self.page_size):
            for head_id in range(0, self.head_num, self.head_num // split_factor):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                    + head_id
                    * self.page_size
                    * self.layer_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
        element_size = (
            self.layer_num
            * self.dtype.itemsize
            * self.page_size
            * self.head_num
            * self.head_dim
            // split_factor
        )
        element_size_list = [element_size] * len(ptr_list)
        return ptr_list, element_size_list

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table.get_page_buffer_meta(
                indices, self.layout, self.layer_num, self.head_num, self.head_dim
            )

        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        v_offset = (
            self.layer_num
            * self.size
            * self.head_num
            * self.head_dim
            * self.dtype.itemsize
        )
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index]
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                        + layer_id
                        * self.size
                        * self.head_num
                        * self.head_dim
                        * self.dtype.itemsize
                    )
                    v_ptr = k_ptr + v_offset
                    ptr_list.append(k_ptr)
                    ptr_list.append(v_ptr)
            element_size = (
                self.dtype.itemsize * self.page_size * self.head_num * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct", "page_head"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.head_num
                    * self.head_dim
                    * self.dtype.itemsize
                )
                v_ptr = k_ptr + v_offset
                ptr_list.append(k_ptr)
                ptr_list.append(v_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.head_num
                * self.head_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class MLATokenToKVPoolHost(HostKVCache):
    device_pool: MLATokenToKVPool

    def __init__(
        self,
        device_pool: MLATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
        override_kv_cache_dim: Optional[int] = None,
    ):
        self.override_kv_cache_dim = override_kv_cache_dim
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
        )
        self.data_refs = [self.kv_buffer[i] for i in range(self.layer_num)]
        self.data_ptrs = torch.tensor(
            [x.data_ptr() for x in self.data_refs],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )

    def get_size_per_token(self):
        self.kv_lora_rank = self.device_pool.kv_lora_rank
        self.qk_rope_head_dim = self.device_pool.qk_rope_head_dim
        self.layer_num = self.device_pool.layer_num
        self.kv_cache_dim = self.override_kv_cache_dim or (
            self.kv_lora_rank + self.qk_rope_head_dim
        )
        return self.kv_cache_dim * self.dtype.itemsize * self.layer_num

    def get_ksize_per_token(self):
        return self.get_size_per_token()

    def init_kv_buffer(self):
        if self.layout == "layer_first":
            dims = (
                self.layer_num,
                self.size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            dims = (
                self.size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        # Ascend-specific: Aligns with NPUMLATokenToKVPool layout
        # Separately allocate k_buffer and v_buffer for easier data transfer.
        elif self.layout == "page_first_kv_split":
            base_dims = (
                self.page_num,
                self.layer_num,
                self.page_size,
                1,
            )
            alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
            self.k_buffer = alloc_func(
                (*base_dims, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.v_buffer = alloc_func(
                (*base_dims, self.qk_rope_head_dim),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
            self.index_k_buffer = None
            if self.device_pool.index_head_dim is not None:
                self.index_k_buffer = alloc_func(
                    (*base_dims, self.device_pool.index_head_dim),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
            # Return k_buffer to preserve original kv_buffer and data_refs init logic,
            # though Ascend doesn't use these parameters.
            return self.k_buffer
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        self.token_stride_size = self.kv_cache_dim * self.dtype.itemsize
        self.layout_dim = self.token_stride_size * self.layer_num

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        buffer = alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )
        return buffer

    def init_extent_kv_buffer(self, num_slots: int):
        page_num = num_slots // self.page_size
        if self.layout == "layer_first":
            dims = (
                self.layer_num,
                num_slots,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            dims = (
                num_slots,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            dims = (
                page_num,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_kv_split":
            base_dims = (
                page_num,
                self.layer_num,
                self.page_size,
                1,
            )
            if (
                self.pin_memory
                and self.device == "cpu"
                and self.device_pool.device == "cuda"
            ):
                return torch.empty(
                    (*base_dims, self.kv_lora_rank),
                    dtype=self.dtype,
                    device=self.device,
                    pin_memory=True,
                )
            alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
            return alloc_func(
                (*base_dims, self.kv_lora_rank),
                dtype=self.dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

        if self.pin_memory and self.device == "cpu" and self.device_pool.device == "cuda":
            return torch.empty(dims, dtype=self.dtype, device=self.device, pin_memory=True)

        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        return alloc_func(
            dims,
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            allocator=self.allocator,
        )

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            if io_backend == "kernel" and self.layout == "page_first":
                def load_from_extent(
                    kv_buffer, local_host_indices, paired_device_indices, _
                ):
                    transfer_kv_per_layer_mla_pf_lf(
                        src=kv_buffer,
                        dst=device_pool.kv_buffer[layer_id],
                        src_indices=local_host_indices,
                        dst_indices=paired_device_indices,
                        layer_id=layer_id,
                        item_size=self.token_stride_size,
                        src_layout_dim=self.layout_dim,
                    )

                extent_table.dispatch_transfer_groups(
                    host_indices, device_indices, load_from_extent
                )
                return
            raise ValueError(
                f"Unsupported MLA multi-extent transfer path: {io_backend=} {self.layout=}"
            )

        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.kv_buffer[layer_id],
                    dst=device_pool.kv_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    item_size=self.token_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.kv_buffer,
                    dst=device_pool.kv_buffer[layer_id],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    item_size=self.token_stride_size,
                    src_layout_dim=self.layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.kv_buffer[layer_id]],
                    dst_layers=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.kv_buffer],
                    dst_ptrs=[device_pool.kv_buffer[layer_id]],
                    src_indices=host_indices,
                    dst_indices=device_indices,
                    layer_id=layer_id,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                # Ascend-specific: transfer KV data for all layers when layer_id == 0
                if layer_id == 0:
                    transfer_kv_dim_exchange(
                        device_indices=device_indices,
                        host_indices=host_indices,
                        device_k=device_pool.k_buffer,
                        host_k=self.k_buffer,
                        device_v=device_pool.v_buffer,
                        host_v=self.v_buffer,
                        device_index_k=device_pool.index_k_buffer,
                        host_index_k=self.index_k_buffer,
                        page_size=self.page_size,
                        direction=TransferDirection.H2D,
                    )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            if io_backend == "kernel" and self.layout == "page_first":
                def backup_to_extent(
                    kv_buffer, local_host_indices, paired_device_indices, _
                ):
                    transfer_kv_all_layer_mla_lf_pf(
                        src_layers=device_pool.data_ptrs,
                        dst=kv_buffer,
                        src_indices=paired_device_indices,
                        dst_indices=local_host_indices,
                        item_size=self.token_stride_size,
                        dst_layout_dim=self.layout_dim,
                        num_layers=self.layer_num,
                    )

                extent_table.dispatch_transfer_groups(
                    host_indices, device_indices, backup_to_extent
                )
                return
            raise ValueError(
                f"Unsupported MLA multi-extent transfer path: {io_backend=} {self.layout=}"
            )

        if io_backend == "kernel":
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=device_pool.data_ptrs,
                    dst_layers=self.data_ptrs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=device_pool.data_ptrs,
                    dst=self.kv_buffer,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=self.token_stride_size,
                    dst_layout_dim=self.layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.kv_buffer,
                    dst_layers=self.data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.kv_buffer,
                    dst_ptrs=[self.kv_buffer],
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "kernel_ascend":
            if self.layout == "page_first_kv_split":
                transfer_kv_dim_exchange(
                    device_indices=device_indices,
                    host_indices=host_indices,
                    device_k=device_pool.k_buffer,
                    host_k=self.k_buffer,
                    device_v=device_pool.v_buffer,
                    host_v=self.v_buffer,
                    device_index_k=device_pool.index_k_buffer,
                    host_index_k=self.index_k_buffer,
                    page_size=self.page_size,
                    direction=TransferDirection.D2H,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table.get_mla_data_page(index, self.layout, flat)

        if self.layout == "layer_first":
            data_page = self.kv_buffer[:, index : index + self.page_size, :, :]
        elif self.layout == "page_first":
            data_page = self.kv_buffer[index : index + self.page_size, :, :, :]
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            data_page = self.kv_buffer[real_index : real_index + 1, :, :, :, :]
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        if flat:
            data_page = data_page.flatten()
        return data_page

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        return torch.zeros(
            (
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            ),
            dtype=self.dtype,
            device=self.device,
            pin_memory=self.pin_memory,
        ).flatten()

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            extent_table.set_mla_from_flat_data_page(index, data_page, self.layout)
            return

        if self.layout == "layer_first":
            self.kv_buffer[:, index : index + self.page_size, :, :] = data_page.reshape(
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first":
            self.kv_buffer[index : index + self.page_size, :, :, :] = data_page.reshape(
                self.page_size,
                self.layer_num,
                1,
                self.kv_cache_dim,
            )
        elif self.layout == "page_first_direct":
            real_index = index // self.page_size
            self.kv_buffer[real_index : real_index + 1, :, :, :, :] = data_page.reshape(
                1,
                self.layer_num,
                self.page_size,
                1,
                self.kv_cache_dim,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def get_page_buffer_meta(self, indices):
        """ "
        meta data for zero copy
        """
        assert len(indices) % self.page_size == 0
        extent_table = getattr(self, "extent_table", None)
        if extent_table is not None:
            return extent_table.get_mla_page_buffer_meta(
                indices, self.layout, self.layer_num, self.kv_cache_dim
            )

        ptr_list = []
        kv_buffer_data_ptr = self.kv_buffer.data_ptr()
        indices = indices.tolist()
        if self.layout == "layer_first":
            for index in range(0, len(indices), self.page_size):
                for layer_id in range(self.layer_num):
                    k_ptr = (
                        kv_buffer_data_ptr
                        + indices[index] * self.kv_cache_dim * self.dtype.itemsize
                        + layer_id * self.size * self.kv_cache_dim * self.dtype.itemsize
                    )
                    ptr_list.append(k_ptr)
            element_size = self.dtype.itemsize * self.page_size * self.kv_cache_dim
            element_size_list = [element_size] * len(ptr_list)
        elif self.layout in ["page_first", "page_first_direct"]:
            for index in range(0, len(indices), self.page_size):
                k_ptr = (
                    kv_buffer_data_ptr
                    + indices[index]
                    * self.layer_num
                    * self.kv_cache_dim
                    * self.dtype.itemsize
                )
                ptr_list.append(k_ptr)
            element_size = (
                self.layer_num
                * self.dtype.itemsize
                * self.page_size
                * self.kv_cache_dim
            )
            element_size_list = [element_size] * len(ptr_list)
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")
        return ptr_list, element_size_list


class NSATokenToKVPoolHost(MLATokenToKVPoolHost):
    device_pool: NSATokenToKVPool

    def __init__(
        self,
        device_pool: NSATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        pin_memory: bool = True,
        device: str = "cpu",
        allocator_type: str = "default",
    ):
        # Initialize indexer metadata before HostKVCache.__init__ calls get_size_per_token.
        self.index_head_dim = device_pool.index_head_dim
        self.indexer_quant_block_size = device_pool.quant_block_size
        self.indexer_dtype = NSATokenToKVPool.index_k_with_scale_buffer_dtype
        self.indexer_size_per_token = (
            self.index_head_dim
            + self.index_head_dim // self.indexer_quant_block_size * 4
        )
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory,
            device,
            allocator_type,
            override_kv_cache_dim=device_pool.kv_cache_dim,
        )
        self.indexer_page_stride_size = (
            self.indexer_size_per_token * self.page_size * self.indexer_dtype.itemsize
        )
        self.indexer_layout_dim = self.indexer_page_stride_size * self.layer_num
        self.indexer_page_num = (self.size + self.page_size + 1) // self.page_size
        self._init_indexer_buffers()
        logger.info(
            f"NSATokenToKVPoolHost initialized with indexer page stride size: {self.indexer_page_stride_size}, page num: {self.indexer_page_num}"
        )

    def get_size_per_token(self):
        base = super().get_size_per_token()
        return (
            base
            + self.indexer_size_per_token * self.layer_num * self.indexer_dtype.itemsize
        )

    def _init_indexer_buffers(self):
        alloc_func = ALLOC_MEMORY_FUNCS[self.device_pool.device]
        self.index_k_device_ptrs = torch.tensor(
            [x.data_ptr() for x in self.device_pool.index_k_with_scale_buffer],
            dtype=torch.uint64,
            device=self.device_pool.device,
        )
        if self.layout == "layer_first":
            self.index_k_with_scale_buffer = [
                alloc_func(
                    (self.indexer_page_num, self.indexer_page_stride_size),
                    dtype=self.indexer_dtype,
                    device=self.device,
                    pin_memory=self.pin_memory,
                    allocator=self.allocator,
                )
                for _ in range(self.layer_num)
            ]
            self.index_k_data_refs = [
                self.index_k_with_scale_buffer[i] for i in range(self.layer_num)
            ]
            self.index_k_data_ptrs = torch.tensor(
                [x.data_ptr() for x in self.index_k_data_refs],
                dtype=torch.uint64,
                device=self.device_pool.device,
            )
        elif self.layout in ["page_first", "page_first_direct"]:
            self.index_k_with_scale_buffer = alloc_func(
                (
                    self.indexer_page_num,
                    self.layer_num,
                    1,
                    self.indexer_page_stride_size,
                ),
                dtype=self.indexer_dtype,
                device=self.device,
                pin_memory=self.pin_memory,
                allocator=self.allocator,
            )
        else:
            raise ValueError(f"Unsupported layout: {self.layout}")

    def _get_indexer_page_indices(self, host_indices, device_indices):
        if host_indices.numel() == 0:
            return host_indices, device_indices
        if host_indices.numel() % self.page_size != 0:
            raise ValueError(
                "Index buffer transfer expects page-aligned indices for NSA."
            )
        host_page_indices = (
            host_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        device_page_indices = (
            device_indices.reshape(-1, self.page_size)[:, 0] // self.page_size
        )
        return host_page_indices, device_page_indices

    def _load_indexer_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_per_layer_mla(
                    src=self.index_k_with_scale_buffer[layer_id],
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    item_size=self.indexer_page_stride_size,
                )
            elif self.layout == "page_first":
                transfer_kv_per_layer_mla_pf_lf(
                    src=self.index_k_with_scale_buffer,
                    dst=device_pool.index_k_with_scale_buffer[layer_id],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    item_size=self.indexer_page_stride_size,
                    src_layout_dim=self.indexer_layout_dim,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=[self.index_k_with_scale_buffer[layer_id]],
                    dst_layers=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_per_layer_direct_pf_lf(
                    src_ptrs=[self.index_k_with_scale_buffer],
                    dst_ptrs=[device_pool.index_k_with_scale_buffer[layer_id]],
                    src_indices=host_page_indices,
                    dst_indices=device_page_indices,
                    layer_id=layer_id,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def _backup_indexer_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        host_page_indices, device_page_indices = self._get_indexer_page_indices(
            host_indices, device_indices
        )
        use_kernel = io_backend == "kernel" and self.indexer_page_stride_size % 8 == 0
        if use_kernel:
            if self.layout == "layer_first":
                transfer_kv_all_layer_mla(
                    src_layers=self.index_k_device_ptrs,
                    dst_layers=self.index_k_data_ptrs,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    num_layers=self.layer_num,
                )
            elif self.layout == "page_first":
                transfer_kv_all_layer_mla_lf_pf(
                    src_layers=self.index_k_device_ptrs,
                    dst=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    item_size=self.indexer_page_stride_size,
                    dst_layout_dim=self.indexer_layout_dim,
                    num_layers=self.layer_num,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        elif io_backend == "direct":
            if self.layout == "layer_first":
                transfer_kv_direct(
                    src_layers=device_pool.index_k_with_scale_buffer,
                    dst_layers=self.index_k_with_scale_buffer,
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            elif self.layout == "page_first_direct":
                transfer_kv_all_layer_direct_lf_pf(
                    src_ptrs=device_pool.index_k_with_scale_buffer,
                    dst_ptrs=[self.index_k_with_scale_buffer],
                    src_indices=device_page_indices,
                    dst_indices=host_page_indices,
                    page_size=1,
                )
            else:
                raise ValueError(f"Unsupported layout: {self.layout}")
        else:
            raise ValueError(f"Unsupported IO backend: {io_backend}")

    def load_to_device_per_layer(
        self,
        device_pool,
        host_indices,
        device_indices,
        layer_id,
        io_backend,
    ):
        super().load_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )
        self._load_indexer_to_device_per_layer(
            device_pool, host_indices, device_indices, layer_id, io_backend
        )

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ):
        super().backup_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
        self._backup_indexer_from_device_all_layer(
            device_pool, host_indices, device_indices, io_backend
        )
