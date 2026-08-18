import abc
import logging
import os
import threading
import time
from collections import defaultdict
from functools import wraps
from typing import Optional

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
from sglang.srt.mem_cache.central_io_pressure import CentralIOPressureLogger
from sglang.srt.mem_cache.local_residency import LocalResidencyState
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


class _FreeSlotRanges:
    """Sparse logical-slot allocator used by the Central I/O range lease path."""

    def __init__(self, capacity: int = 0):
        self.ranges: list[tuple[int, int]] = []
        self.count = 0
        if capacity:
            self.add_range(0, capacity)

    def reset(self, capacity: int) -> None:
        self.ranges = []
        self.count = 0
        if capacity:
            self.add_range(0, capacity)

    def reset_ranges(self, ranges: list[tuple[int, int]]) -> None:
        """Reset from leased logical ranges, which need not be contiguous."""
        self.ranges = []
        self.count = 0
        self.add_ranges(ranges)

    def add_range(self, start: int, count: int) -> None:
        if count <= 0:
            return
        end = start + count
        merged: list[tuple[int, int]] = []
        inserted = False
        for current_start, current_end in self.ranges:
            if current_end < start:
                merged.append((current_start, current_end))
            elif end < current_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((current_start, current_end))
            else:
                if current_start < end and start < current_end:
                    raise RuntimeError("Central I/O free-range overlap")
                start = min(start, current_start)
                end = max(end, current_end)
        if not inserted:
            merged.append((start, end))
        self.ranges = merged
        self.count += count

    def add_ranges(self, ranges: list[tuple[int, int]]) -> None:
        for start, count in sorted(ranges):
            self.add_range(start, count)

    def allocate(self, count: int) -> Optional[torch.Tensor]:
        if count > self.count:
            return None
        pieces: list[torch.Tensor] = []
        remaining = count
        updated: list[tuple[int, int]] = []
        for start, end in self.ranges:
            if not remaining:
                updated.append((start, end))
                continue
            take = min(remaining, end - start)
            pieces.append(torch.arange(start, start + take, dtype=torch.int64))
            if start + take < end:
                updated.append((start + take, end))
            remaining -= take
        self.ranges = updated
        self.count -= count
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def _remove_range(self, start: int, count: int) -> None:
        if count <= 0:
            return
        end = start + count
        removed = 0
        updated: list[tuple[int, int]] = []
        for current_start, current_end in self.ranges:
            overlap_start = max(start, current_start)
            overlap_end = min(end, current_end)
            if overlap_start >= overlap_end:
                updated.append((current_start, current_end))
                continue
            removed += overlap_end - overlap_start
            if current_start < overlap_start:
                updated.append((current_start, overlap_start))
            if overlap_end < current_end:
                updated.append((overlap_end, current_end))
        if removed != count:
            raise RuntimeError("Central I/O attempted to shrink a non-free host range")
        self.ranges = updated
        self.count -= removed

    def remove_ranges(self, ranges: list[tuple[int, int]]) -> None:
        for start, count in ranges:
            self._remove_range(start, count)

    def release(self, indices: torch.Tensor) -> None:
        values = sorted(indices.detach().cpu().tolist())
        if not values:
            return
        run_start = previous = values[0]
        for index in values[1:]:
            if index == previous + 1:
                previous = index
                continue
            self.add_range(run_start, previous - run_start + 1)
            run_start = previous = index
        self.add_range(run_start, previous - run_start + 1)


class _FreePageRanges:
    """Sparse allocator in SGLang KV-page coordinates.

    Central I/O owns quota and liveness at page granularity.  The adapter only
    expands a page allocation to token-slot indices at the unchanged SGLang
    API boundary, so no per-slot Python state crosses the agent RPC boundary.
    """

    def __init__(self, capacity_pages: int = 0):
        self.ranges: list[tuple[int, int]] = []
        self.count = 0
        if capacity_pages:
            self.add_range(0, capacity_pages)

    def reset_ranges(self, ranges: list[tuple[int, int]]) -> None:
        self.ranges = []
        self.count = 0
        self.add_ranges(ranges)

    def add_range(self, start: int, count: int) -> None:
        if count <= 0:
            return
        end = start + count
        merged: list[tuple[int, int]] = []
        inserted = False
        for current_start, current_end in self.ranges:
            if current_end < start:
                merged.append((current_start, current_end))
            elif end < current_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((current_start, current_end))
            else:
                if current_start < end and start < current_end:
                    raise RuntimeError("Central I/O free-page range overlap")
                start = min(start, current_start)
                end = max(end, current_end)
        if not inserted:
            merged.append((start, end))
        self.ranges = merged
        self.count += count

    def add_ranges(self, ranges: list[tuple[int, int]]) -> None:
        for start, count in sorted(ranges):
            self.add_range(start, count)

    def allocate_ranges(self, count: int) -> Optional[list[tuple[int, int]]]:
        if count <= 0:
            raise ValueError("Central I/O page allocation must be positive")
        if count > self.count:
            return None
        selected: list[tuple[int, int]] = []
        updated: list[tuple[int, int]] = []
        remaining = count
        for start, end in self.ranges:
            if not remaining:
                updated.append((start, end))
                continue
            take = min(remaining, end - start)
            selected.append((start, take))
            if start + take < end:
                updated.append((start + take, end))
            remaining -= take
        self.ranges = updated
        self.count -= count
        return selected

    def _remove_range(self, start: int, count: int) -> None:
        if count <= 0:
            return
        end = start + count
        removed = 0
        updated: list[tuple[int, int]] = []
        for current_start, current_end in self.ranges:
            overlap_start = max(start, current_start)
            overlap_end = min(end, current_end)
            if overlap_start >= overlap_end:
                updated.append((current_start, current_end))
                continue
            removed += overlap_end - overlap_start
            if current_start < overlap_start:
                updated.append((current_start, overlap_start))
            if overlap_end < current_end:
                updated.append((overlap_end, current_end))
        if removed != count:
            raise RuntimeError("Central I/O attempted to remove a non-free page range")
        self.ranges = updated
        self.count -= removed

    def remove_ranges(self, ranges: list[tuple[int, int]]) -> None:
        for start, count in ranges:
            self._remove_range(start, count)

    def intersection_ranges(self, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Return only requested page ranges that are currently free."""
        result: list[tuple[int, int]] = []
        for start, count in ranges:
            end = start + count
            for free_start, free_end in self.ranges:
                overlap_start = max(start, free_start)
                overlap_end = min(end, free_end)
                if overlap_start < overlap_end:
                    result.append((overlap_start, overlap_end - overlap_start))
        return result

    def subtract_ranges(self, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Return requested ranges after excluding this allocator's ranges."""
        result: list[tuple[int, int]] = []
        for start, count in ranges:
            pieces = [(start, start + count)]
            for blocked_start, blocked_end in self.ranges:
                updated: list[tuple[int, int]] = []
                for piece_start, piece_end in pieces:
                    if blocked_end <= piece_start or piece_end <= blocked_start:
                        updated.append((piece_start, piece_end))
                        continue
                    if piece_start < blocked_start:
                        updated.append((piece_start, blocked_start))
                    if blocked_end < piece_end:
                        updated.append((blocked_end, piece_end))
                pieces = updated
            result.extend((piece_start, piece_end - piece_start) for piece_start, piece_end in pieces)
        return result


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

        # Central I/O stores the physical host KV only once, in its global
        # pinned pool. This proxy retains a full logical max_size so SGLang
        # can grow later, but it does not allocate that many local bytes.
        if not getattr(self, "_central_io_owns_storage", False):
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
            logger.info(
                f"Allocating {requested_bytes / 1e9:.2f} GB host memory for hierarchical KV cache."
            )

        self.kv_buffer = self.init_kv_buffer()

        # A lock for synchronized operations on memory allocation and state transitions.
        self.lock = threading.RLock()
        self.clear()

    @abc.abstractmethod
    def get_size_per_token(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def init_kv_buffer(self):
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
        self.free_slots = torch.arange(self.size, dtype=torch.int64)

    def available_size(self):
        return len(self.free_slots)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        assert (
            need_size % self.page_size == 0
        ), "The requested size should be a multiple of the page size."
        if need_size > self.available_size():
            return None

        select_index = self.free_slots[:need_size]
        self.free_slots = self.free_slots[need_size:]

        return select_index

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        self.free_slots = torch.cat([self.free_slots, indices])
        return len(indices)


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
        if io_backend == "kernel":
            if self.layout == "layer_first":
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
        if io_backend == "kernel":
            if self.layout == "layer_first":
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
                transfer_kv_direct(
                    src_layers=device_pool.k_buffer + device_pool.v_buffer,
                    dst_layers=self.k_data_refs + self.v_data_refs,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    page_size=self.page_size,
                )
            elif self.layout == "page_first_direct":
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


class CentralIOMHATokenToKVPoolHost(HostKVCache):
    """MHA HostKVCache proxy backed by a persistent Central I/O agent.

    The model process retains only logical host-slot bookkeeping.  The agent
    owns the pinned page-first tensor and accesses this model's ordinary GPU
    KV allocations through persistent CUDA IPC mappings.
    """

    device_pool: MHATokenToKVPool

    def __init__(
        self,
        device_pool: MHATokenToKVPool,
        host_to_device_ratio: float,
        host_size: int,
        page_size: int,
        layout: str,
        socket_path: str,
        model_id: str | None = None,
        initial_capacity: int | None = None,
        segment_tokens: int | None = None,
    ):
        if layout != "page_first":
            raise ValueError("Central I/O V1 requires --hicache-mem-layout page_first")
        self.socket_path = socket_path
        self.model_id = model_id or f"sglang-{os.getpid()}"
        self.client = None
        self._central_io_owns_storage = True
        super().__init__(
            device_pool,
            host_to_device_ratio,
            host_size,
            page_size,
            layout,
            pin_memory=False,
            device="cpu",
            allocator_type="default",
        )
        from sglang.srt.mem_cache.central_io import CentralIOClient, build_registration

        # ``--hicache-size`` is still used by upstream SGLang while it builds
        # the normal runtime.  In Central I/O mode, however, physical host KV
        # lives in the shared agent pool and a model must be able to grow past
        # its initial host-cache size.  Keep a separate logical ceiling for
        # leases so changing that ceiling never changes GPU/HBM allocation.
        logical_max_gib = os.getenv("SGLANG_CENTRAL_IO_LOGICAL_MAX_GIB")
        if logical_max_gib is None:
            self.max_size = self.size
        else:
            logical_max_bytes = int(float(logical_max_gib) * (1024**3))
            logical_max_slots = logical_max_bytes // self.size_per_token
            logical_max_slots -= logical_max_slots % self.page_size
            if logical_max_slots < self.size:
                raise ValueError(
                    "SGLANG_CENTRAL_IO_LOGICAL_MAX_GIB must not be smaller "
                    "than the SGLang host-cache startup capacity"
                )
            self.max_size = logical_max_slots
            self.size = logical_max_slots
            self.page_num = self.size // self.page_size
        requested_initial_capacity = initial_capacity
        requested_initial_bytes = None
        if requested_initial_capacity is None:
            initial_gib = os.getenv("SGLANG_CENTRAL_IO_INITIAL_GIB")
            if initial_gib is None:
                requested_initial_capacity = self.max_size
            else:
                requested_initial_bytes = int(float(initial_gib) * (1024**3))
                requested_initial_capacity = int(
                    requested_initial_bytes // self.size_per_token
                )
        requested_initial_capacity = min(requested_initial_capacity, self.max_size)
        requested_initial_capacity -= requested_initial_capacity % self.page_size
        if requested_initial_bytes is None:
            requested_initial_bytes = requested_initial_capacity * self.size_per_token
        if requested_initial_capacity <= 0:
            raise ValueError("Central I/O initial capacity must contain at least one SGLang KV page")
        if segment_tokens is None:
            segment_tokens = int(
                os.getenv("SGLANG_CENTRAL_IO_SEGMENT_TOKENS", self.page_size * 256)
            )
        segment_tokens -= segment_tokens % self.page_size
        if segment_tokens <= 0:
            raise ValueError("Central I/O segment size must be a positive multiple of page_size")

        registration = build_registration(
            device_pool,
            self.max_size,
            self.model_id,
            initial_capacity=requested_initial_capacity,
            initial_bytes=requested_initial_bytes,
            segment_tokens=segment_tokens,
            page_size=self.page_size,
        )
        self.client = CentralIOClient(self.socket_path, registration)
        if self.client.max_capacity != self.max_size:
            raise RuntimeError(
                f"Central I/O max capacity mismatch: agent={self.client.max_capacity}, "
                f"host cache={self.max_size}"
            )
        self.active_size = self.client.capacity
        self.dynamic_page_leases = self.client.lease_mode == "page"
        self.extent_slots = int(self.client.extent_slots or 0)
        self.arena_extent_bytes = int(self.client.arena_extent_bytes or 0)
        if self.extent_slots <= 0 or self.extent_slots % self.page_size:
            raise RuntimeError("Central I/O agent returned an invalid ArenaExtent slot capacity")
        if self.arena_extent_bytes <= 0:
            raise RuntimeError("Central I/O agent returned an invalid ArenaExtent size")
        # ``active_size`` remains a slot count for SGLang compatibility.  The
        # Central I/O control plane itself uses page ranges: a page is the
        # smallest unit that may become live, free, or leased to another model.
        active_pages = self.active_size // self.page_size
        self._active_page_ranges = _FreePageRanges(active_pages)
        self._free_page_ranges = _FreePageRanges(active_pages)
        # Full page ranges of extents that remain readable by their current
        # owner but are temporarily closed to new host-KV admission.
        self._draining_page_ranges = _FreePageRanges()
        self._draining_segment_ids: set[int] = set()
        # ``HostKVCache.__init__`` invokes ``clear`` before this adapter has
        # connected to the agent, so the Central override intentionally does
        # nothing at that point.  Initialize slot liveness here after the
        # initial page lease is known.  It lets radix-split nodes release
        # independent slot slices while the agent still owns whole pages.
        self.mem_state = torch.zeros((self.size,), dtype=torch.uint8, device=self.device)
        self.element_dim = self.device_pool.head_num * self.device_pool.head_dim
        # One record per second is sufficient to align physical host-KV
        # pressure with replay windows.  The logger is disabled unless the
        # operator explicitly supplies a per-model output path.
        self._pressure_logger = CentralIOPressureLogger(
            os.getenv("SGLANG_CENTRAL_IO_PRESSURE_LOG"),
            model_id=self.model_id,
            page_size=self.page_size,
        )
        self._live_host_pages = 0
        self._local_ready_pages = 0
        self._local_ready_reclaim_score = 0.0
        self._last_residency_report_s = float("-inf")
        self._residency_report_interval_s = float(
            os.getenv("SGLANG_LATTICEKV_RESIDENCY_REPORT_INTERVAL_S", "0.1")
        )
        if self._residency_report_interval_s < 0:
            raise ValueError("SGLANG_LATTICEKV_RESIDENCY_REPORT_INTERVAL_S must be non-negative")
        self._local_residency = LocalResidencyState(
            effective_pages=active_pages,
            floor_pages=min(
                active_pages,
                int(os.getenv("SGLANG_LATTICEKV_LOCAL_FLOOR_PAGES", "0")),
            ),
            large_backup_batch_pages=max(
                1, int(os.getenv("SGLANG_LATTICEKV_BACKUP_BATCH_PAGES", "1"))
            ),
            backup_ingress_pages_per_s=0.0,
            ready_latency_p95_s=float(
                os.getenv("SGLANG_LATTICEKV_READY_LATENCY_P95_S", "0")
            ),
            maintenance_batch_pages=max(
                1,
                int(os.getenv("SGLANG_LATTICEKV_MAINTENANCE_BATCH_PAGES", "128")),
            ),
            maintenance_cycle_s=float(
                os.getenv("SGLANG_LATTICEKV_MAINTENANCE_CYCLE_S", "1")
            ),
            retention_debt_cooldown_s=float(
                os.getenv("SGLANG_LATTICEKV_RETENTION_DEBT_COOLDOWN_S", "60")
            ),
            clean_shortfall_cooldown_s=float(
                os.getenv("SGLANG_LATTICEKV_CLEAN_SHORTFALL_COOLDOWN_S", "5")
            ),
            ingress_window_s=float(
                os.getenv("SGLANG_LATTICEKV_INGRESS_WINDOW_S", "5")
            ),
            backup_batch_window_s=float(
                os.getenv("SGLANG_LATTICEKV_BACKUP_BATCH_WINDOW_S", "60")
            ),
            loss_aware_maintenance=os.getenv(
                "SGLANG_LATTICEKV_LOSS_AWARE_MAINTENANCE", "1"
            ).lower()
            not in {"0", "false", "no"},
        )
        self._refresh_local_residency()

    def _refresh_local_residency(self) -> None:
        """Synchronize local admission facts after an allocator lifecycle event."""
        state = getattr(self, "_local_residency", None)
        if state is None:
            return
        state.observe_pages(
            effective_pages=self.active_size // self.page_size,
            clean_pages=self._free_page_ranges.count,
            ready_pages=self._local_ready_pages,
            live_pages=self._live_host_pages,
        )

    def _publish_effective_quota_change(self) -> None:
        """Publish fresh allocator facts immediately after a lease ack.

        Agent capacity changes at donor/recipient acknowledgement time, not at
        the next request.  Without this push a scheduler tick can temporarily
        compare the new agent capacity to an old ``effective_pages`` report.
        """
        if self.client is not None:
            self.publish_local_residency(force=True)

    def set_local_ready_pages(
        self, ready_pages: int, *, reclaim_score: float = 0.0
    ) -> None:
        """Publish page-complete local reclaim candidates from HiRadix."""
        if ready_pages < 0 or ready_pages > self._live_host_pages:
            raise ValueError("ready page count must be within current live host pages")
        if reclaim_score < 0:
            raise ValueError("ready page reclaim score must be non-negative")
        self._local_ready_pages = ready_pages
        self._local_ready_reclaim_score = reclaim_score
        self._refresh_local_residency()

    def local_residency_status(self) -> dict[str, int | float | str]:
        """Expose instance-local admission state without making a quota decision."""
        state = self._local_residency
        now_s = time.monotonic()
        watermarks = state.watermarks_at(now_s=now_s)
        return {
            "effective_pages": state.effective_pages,
            "floor_pages": state.floor_pages,
            "clean_pages": state.clean_pages,
            "ready_pages": state.ready_pages,
            "ready_reclaim_score": self._local_ready_reclaim_score,
            "live_pages": state.live_pages,
            "backup_ingress_pages_per_s": state.backup_ingress_pages_per_s,
            "ready_latency_p95_s": state.ready_latency_p95_s,
            "feedback_horizon_s": state.feedback_horizon_s(now_s=now_s),
            "retention_debt_tokens": state.active_retention_debt_tokens(
                now_s=now_s
            ),
            "retention_debt_epoch": state.retention_debt_epoch,
            "unresolved_clean_pages": state.active_clean_shortfall_pages(
                now_s=now_s
            ),
            "turnover_deficit_pages": state.turnover_deficit_pages(now_s=now_s),
            "safe_reclaim_pages_per_s": state.safe_reclaim_pages_per_s(now_s=now_s),
            "min_pages": watermarks.min_pages,
            "low_pages": watermarks.low_pages,
            "high_pages": watermarks.high_pages,
            "maintenance_target_pages": state.maintenance_target_pages(now_s=now_s),
            "value_constrained": state.value_constrained(now_s=now_s),
            "action": state.action(now_s=now_s),
        }

    def local_donation_offer(self, requested_pages: int) -> dict[str, int]:
        """Recheck how much quota this model may hand off right now.

        A scheduler decision can be slightly older than the donor's most
        recent HBM backup.  The allocator is therefore the final authority:
        it may offer only clean capacity above its recovery watermark, plus
        page-complete candidates which can be released safely next.  This is
        a capacity offer, not a transfer; effective quota changes later only
        after the usual drain/scrub/recipient-ack path.
        """
        if requested_pages < 0:
            raise ValueError("requested donation pages must be non-negative")
        offer = self._local_residency.offer_donation(
            requested_pages=requested_pages, now_s=time.monotonic()
        )
        return {
            "immediate_pages": offer.immediate_pages,
            "ready_pages": offer.ready_pages,
            "low_loss_pages": offer.low_loss_pages,
            "total_pages": offer.total_pages,
        }

    def record_retention_loss(self, reprefill_tokens: int) -> None:
        """Report that a locally reclaimed prefix had to be rebuilt soon."""
        self._local_residency.record_retention_loss(
            reprefill_tokens=reprefill_tokens, now_s=time.monotonic()
        )
        self._record_pressure("retention_loss_tokens", reprefill_tokens)

    def record_local_reclaim_latency(self, elapsed_s: float) -> None:
        """Feed a completed local reclaim wave back into admission watermarks."""
        self._local_residency.observe_reclaim_latency(elapsed_s=elapsed_s)

    def record_local_reclaim_result(self, pages: int, elapsed_s: float) -> None:
        """Feed completed local page recovery into pressure accounting."""
        self._local_residency.observe_reclaim_result(
            pages=pages, elapsed_s=elapsed_s, now_s=time.monotonic()
        )

    def record_unresolved_clean_shortfall(self, pages: int) -> None:
        """Expose a failed local admission recovery to the global scheduler."""
        self._local_residency.record_clean_shortfall(
            pages=pages, now_s=time.monotonic()
        )
        if pages:
            self._record_pressure("admission_shortfall_pages", pages)

    def reclaim_feedback_horizon_s(self) -> float:
        """Expose the local host-cache turnover horizon to HiRadix's ghosts."""
        return self._local_residency.feedback_horizon_s(now_s=time.monotonic())

    def publish_local_residency(self, *, force: bool = False) -> bool:
        """Rate-limit global health publication; allocator ownership stays local."""
        if self.client is None:
            return False
        now_s = time.monotonic()
        if (
            not force
            and now_s - self._last_residency_report_s < self._residency_report_interval_s
        ):
            return False
        status = self.local_residency_status()
        self.client.report_local_residency(status)
        self._last_residency_report_s = now_s
        return True

    def _record_pressure(self, kind: str, amount: int) -> None:
        self._pressure_logger.record(
            kind,
            amount,
            live_pages=self._live_host_pages,
            active_pages=self.active_size // self.page_size,
            clean_free_pages=self._free_page_ranges.count,
            draining_pages=self._draining_page_ranges.count,
        )

    def record_host_eviction(self, slots: int) -> None:
        """Record discarded host KV separately from an ordinary allocator free."""
        if slots < 0:
            raise ValueError("host eviction slots must be non-negative")
        self._record_pressure("host_evict_slots", slots)

    def record_local_value_reclaim(self, slots: int) -> None:
        """Record an intentional local low-value reclaim, not a fallback."""
        if slots < 0:
            raise ValueError("local reclaim slots must be non-negative")
        self._record_pressure("local_value_reclaim_slots", slots)

    def record_fallback_admission_eviction(self, slots: int) -> None:
        """Record that a backup needed SGLang's generic host eviction."""
        if slots < 0:
            raise ValueError("fallback eviction slots must be non-negative")
        self._record_pressure("fallback_admission_evict_slots", slots)

    def record_donor_drain(self, slots: int) -> None:
        """Record value-aware reclaim done specifically for quota shrink."""
        if slots < 0:
            raise ValueError("donor drain slots must be non-negative")
        self._record_pressure("donor_drain_slots", slots)

    def _align_quota_to_extents(self, target_size: int) -> int:
        """Round a requested slot quota down to complete fixed ArenaExtents.

        The last few token slots of each physical extent may be unusable for a
        model due to page alignment. Returning a partial extent would violate
        Central I/O's ownership invariant, so quota APIs expose the largest
        complete-extent capacity not exceeding the caller's request.
        """
        if self.dynamic_page_leases:
            return target_size - target_size % self.page_size
        requested_bytes = target_size * self.size_per_token
        return self._quota_slots_for_physical_bytes(requested_bytes)

    def _quota_slots_for_physical_bytes(self, target_bytes: int) -> int:
        """Translate a physical-byte quota directly to whole extent slots.

        Public quota APIs originate in GiB. They must not first be rounded to
        token slots and then reconstructed as bytes: that loses enough bytes
        to accidentally drop a 256MiB extent at an otherwise exact 200GiB
        target.
        """
        if self.dynamic_page_leases:
            slots = target_bytes // self.size_per_token
            return slots - slots % self.page_size
        extent_count = target_bytes // self.arena_extent_bytes
        return extent_count * self.extent_slots

    def get_size_per_token(self):
        self.head_num = self.device_pool.head_num
        self.head_dim = self.device_pool.head_dim
        self.layer_num = self.device_pool.layer_num
        return self.head_dim * self.head_num * self.layer_num * self.dtype.itemsize * 2

    def init_kv_buffer(self):
        # Physical page-first storage is held only by the Central I/O agent.
        return torch.empty((0,), dtype=self.dtype, device="cpu")

    @synchronized
    def clear(self):
        """Reset allocator state without re-exposing inactive logical slots."""
        if self.client is not None and hasattr(self, "active_size"):
            self.client.reset_reservations()
            self.mem_state = torch.zeros((self.size,), dtype=torch.uint8, device=self.device)
            self._free_page_ranges.reset_ranges(self._active_page_ranges.ranges)
            self._live_host_pages = 0
            self._local_ready_pages = 0
            self._refresh_local_residency()

    @staticmethod
    def _page_ranges_to_slot_indices(
        page_ranges: list[tuple[int, int]], page_size: int
    ) -> torch.Tensor:
        """Expand pages only at the unchanged SGLang slot-index API boundary."""
        pieces = [
            torch.arange(
                page_start * page_size,
                (page_start + page_count) * page_size,
                dtype=torch.int64,
            )
            for page_start, page_count in page_ranges
        ]
        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    @staticmethod
    def _indices_to_list(indices: torch.Tensor) -> list[int]:
        """Keep the existing transfer-kernel RPC boundary unchanged.

        This helper is intentionally not used by normal alloc/free reservation
        any more; backup/restore still uses SGLang's slot-indexed kernel API.
        """
        return indices.detach().cpu().tolist()

    def _slot_indices_to_page_ranges(
        self, indices: torch.Tensor
    ) -> list[tuple[int, int]]:
        """Compress complete page ids to ranges without expanding page state."""
        page_ids = torch.sort(indices.detach().cpu().to(torch.int64)).values.tolist()
        if not page_ids:
            return []
        ranges: list[tuple[int, int]] = []
        start = previous = page_ids[0]
        for page_id in page_ids[1:]:
            if page_id == previous + 1:
                previous = page_id
                continue
            ranges.append((start, previous - start + 1))
            start = previous = page_id
        ranges.append((start, previous - start + 1))
        return ranges

    def _fully_released_page_ranges(self, indices: torch.Tensor) -> list[tuple[int, int]]:
        """Return pages whose final live slot is contained in ``indices``.

        A radix split may make two host nodes own disjoint slot slices of one
        original SGLang page.  The radix layer can delete either node first,
        but Central I/O may return the physical page only after the last slice
        disappears.  ``mem_state`` is the local slot-liveness authority for
        this small reconciliation step; page-level ranges remain the only
        representation sent to the agent.
        """
        slots = indices.detach().cpu().to(torch.int64).reshape(-1)
        if slots.numel() == 0:
            return []
        if torch.any(slots < 0) or torch.any(slots >= self.size):
            raise ValueError("Central I/O free indices are outside host capacity")
        unique_slots = torch.unique(slots, sorted=True)
        if unique_slots.numel() != slots.numel():
            raise ValueError("Central I/O free contains a duplicate host slot")
        if not torch.all(self.mem_state[unique_slots] == 1):
            raise ValueError("Central I/O free contains an already released host slot")

        # Official SGLang currently defaults to page_size=1.  In that mode a
        # host slot is already the atomic physical page, so a contiguous leaf
        # release must stay a contiguous range rather than re-entering Python
        # once per token.  This is the hot path for normal host eviction.
        if self.page_size == 1:
            return self._slot_indices_to_page_ranges(unique_slots)

        # A radix split can release separate slices of one SGLang page.  Check
        # all touched pages in one tensor operation; do not materialize one
        # scalar Tensor sum per page in Python.
        page_ids = unique_slots // self.page_size
        touched_pages, freed_counts = torch.unique_consecutive(
            page_ids, return_counts=True
        )
        offsets = torch.arange(self.page_size, dtype=torch.int64)
        page_slots = touched_pages[:, None] * self.page_size + offsets
        valid_slots = page_slots < self.size
        safe_slots = page_slots.clamp(max=self.size - 1)
        live_counts = (
            self.mem_state[safe_slots].to(torch.int64) * valid_slots.to(torch.int64)
        ).sum(dim=1)
        released_pages = touched_pages[live_counts == freed_counts]
        return self._slot_indices_to_page_ranges(released_pages)

    @synchronized
    def alloc(self, need_size: int) -> Optional[torch.Tensor]:
        if need_size % self.page_size != 0:
            raise AssertionError("The requested size should be a multiple of the page size.")
        page_ranges = self._free_page_ranges.allocate_ranges(need_size // self.page_size)
        if page_ranges is None:
            return None
        try:
            assert self.client is not None
            self.client.reserve_pages(page_ranges)
        except Exception:
            self._free_page_ranges.add_ranges(page_ranges)
            raise
        indices = self._page_ranges_to_slot_indices(page_ranges, self.page_size)
        if torch.any(self.mem_state[indices] != 0):
            raise RuntimeError("Central I/O allocator selected a non-free host slot")
        self.mem_state[indices] = 1
        allocated_pages = sum(count for _, count in page_ranges)
        self._live_host_pages += allocated_pages
        self._refresh_local_residency()
        self._record_pressure("alloc_pages", allocated_pages)
        return indices

    @synchronized
    def free(self, indices: torch.Tensor) -> int:
        if len(indices) == 0:
            return 0
        assert self.client is not None
        slots = indices.detach().cpu().to(torch.int64).reshape(-1)
        page_ranges = self._fully_released_page_ranges(slots)
        if page_ranges:
            # The agent may have already detached these pages as part of a
            # live drain while HiCache's ordinary eviction queue was pending.
            # That acknowledgement is safe to make idempotent; direct control
            # protocol calls remain strict.
            self.client.release_pages(page_ranges, allow_already_released=True)
        self.mem_state[slots] = 0
        released_pages = sum(count for _, count in page_ranges)
        self._live_host_pages -= released_pages
        self._local_ready_pages = min(self._local_ready_pages, self._live_host_pages)
        self._free_page_ranges.add_ranges(
            self._draining_page_ranges.subtract_ranges(page_ranges)
        )
        self._refresh_local_residency()
        self._record_pressure("release_pages", released_pages)
        return len(indices)

    def available_size(self):
        return self._free_page_ranges.count * self.page_size

    def describe_layout(self) -> dict:
        """Return agent-owned physical layout for a read-only LatticeKV dump."""
        assert self.client is not None
        return self.client.describe_layout()

    @synchronized
    def synchronize_active_size(self) -> int:
        """Reconcile quota capacity with the central agent after async scrub.

        A successful ``commit_live_reclaim`` updates both sides immediately,
        but the scheduler may visit another HiCache event while the agent's
        scrub completion is observed.  Capacity is an agent-owned lease fact,
        so use it as the authority before calculating another drain batch.
        The local page-range maps remain the allocator authority for *which*
        logical pages are usable; this only prevents stale scalar capacity
        from requesting more than the remaining lease can contain.
        """
        assert self.client is not None
        agent_capacity = int(self.client.status()["capacity"])
        if agent_capacity != self.active_size:
            logger.warning(
                "Central I/O capacity reconciliation for %s: local=%d agent=%d "
                "active_page_slots=%d",
                self.model_id,
                self.active_size,
                agent_capacity,
                self._active_page_ranges.count * self.page_size,
            )
            self.active_size = agent_capacity
            self._refresh_local_residency()
        return self.active_size

    def plan_live_reclaim(
        self,
        target_size: int,
        excluded_segment_ids: set[int] | None = None,
        max_segments: int | None = None,
    ) -> dict:
        """Ask the agent for a non-mutating whole-extent drain plan.

        This is intentionally separate from ``resize_quota``.  HiRadix owns
        radix-tree safety and must first evict only safe host leaves; the agent
        only reports which complete physical extents would satisfy the shrink.
        """
        if target_size % self.page_size != 0:
            raise ValueError("Central I/O reclaim target must be page aligned")
        self.synchronize_active_size()
        if target_size < 0 or target_size > self.active_size:
            raise ValueError("Central I/O reclaim target is outside active quota")
        assert self.client is not None
        plan = self.client.plan_live_reclaim(
            target_size, excluded_segment_ids, max_segments
        )
        # The request carries a final target, so the reply makes the agent's
        # current lease capacity authoritative even if a prior async commit
        # crossed a scheduler event boundary.
        self.active_size = int(plan["active_capacity"])
        self._refresh_local_residency()
        return plan

    @synchronized
    def begin_live_page_drain(self, page_ranges: list[tuple[int, int]]) -> dict:
        """Fence selected dynamic-lease pages while HiRadix evicts them.

        The pages remain readable by their present owner.  They are only
        excluded from future HostKV allocation, which prevents a newly
        admitted prefix from defeating an in-flight reclaim.
        """
        if not self.dynamic_page_leases:
            raise RuntimeError("page drain requires dynamic Central I/O leases")
        assert self.client is not None
        response = self.client.begin_live_page_drain(page_ranges)
        ranges = [tuple(item) for item in response["page_ranges"]]
        self._draining_page_ranges.add_ranges(ranges)
        return response

    @synchronized
    def abort_live_page_drain(self, page_ranges: list[tuple[int, int]]) -> dict:
        """Reopen pages whose planned radix drain did not converge."""
        if not self.dynamic_page_leases:
            raise RuntimeError("page drain requires dynamic Central I/O leases")
        assert self.client is not None
        response = self.client.abort_live_page_drain(page_ranges)
        ranges = [tuple(item) for item in response["page_ranges"]]
        self._draining_page_ranges.remove_ranges(ranges)
        reusable: list[tuple[int, int]] = []
        for start, count in ranges:
            for page_id in range(start, start + count):
                slot_start = page_id * self.page_size
                slot_end = slot_start + self.page_size
                if not torch.any(self.mem_state[slot_start:slot_end]):
                    reusable.append((page_id, 1))
        self._free_page_ranges.add_ranges(reusable)
        self._refresh_local_residency()
        return response

    @synchronized
    def commit_live_page_reclaim(self, page_ranges: list[tuple[int, int]]) -> dict:
        """Detach already-drained pages and start their secure async scrub."""
        if not self.dynamic_page_leases:
            raise RuntimeError("page reclaim requires dynamic Central I/O leases")
        assert self.client is not None
        response = self.client.commit_live_page_reclaim(page_ranges)
        ranges = [
            (start // self.page_size, count // self.page_size)
            for start, count in response["ranges"]
        ]
        self._draining_page_ranges.remove_ranges(ranges)
        self._active_page_ranges.remove_ranges(ranges)
        self._free_page_ranges.remove_ranges(
            self._free_page_ranges.intersection_ranges(ranges)
        )
        self.active_size = self.client.capacity
        self._refresh_local_residency()
        self._publish_effective_quota_change()
        return response

    @synchronized
    def begin_live_drain(self, segment_ids: list[int]) -> dict:
        """Fence selected whole extents so new KV cannot prevent convergence."""
        new_ids = [segment_id for segment_id in segment_ids if segment_id not in self._draining_segment_ids]
        if not new_ids:
            return {"segment_ids": [], "segments": []}
        assert self.client is not None
        response = self.client.begin_live_drain(new_ids)
        page_ranges = [
            (item["logical_start"] // self.page_size, item["slot_count"] // self.page_size)
            for item in response["segments"]
        ]
        free_now = self._free_page_ranges.intersection_ranges(page_ranges)
        self._free_page_ranges.remove_ranges(free_now)
        self._draining_page_ranges.add_ranges(page_ranges)
        self._draining_segment_ids.update(response["segment_ids"])
        self._refresh_local_residency()
        return response

    @synchronized
    def abort_live_drain(self, segment_ids: list[int]) -> dict:
        """Reopen an unfinished drain after its bounded recovery interval."""
        active_ids = [segment_id for segment_id in segment_ids if segment_id in self._draining_segment_ids]
        if not active_ids:
            return {"segment_ids": [], "segments": []}
        assert self.client is not None
        response = self.client.abort_live_drain(active_ids)
        all_ranges = [
            (item["logical_start"] // self.page_size, item["slot_count"] // self.page_size)
            for item in response["segments"]
        ]
        live_ranges = [
            (start, count)
            for item in response["segments"]
            for start, count in item["live_page_ranges"]
        ]
        self._draining_page_ranges.remove_ranges(all_ranges)
        self._free_page_ranges.add_ranges(_FreePageRanges().subtract_ranges(all_ranges))
        # Recompute free pages as all range pages excluding currently live KV.
        blocked = _FreePageRanges()
        blocked.add_ranges(live_ranges)
        self._free_page_ranges.remove_ranges(
            self._free_page_ranges.intersection_ranges(live_ranges)
        )
        self._draining_segment_ids.difference_update(response["segment_ids"])
        self._refresh_local_residency()
        return response

    @synchronized
    def commit_live_reclaim(self, segment_ids: list[int]) -> dict:
        """Detach drained whole extents and make their logical pages inactive.

        The agent begins secure scrubbing in the background.  The returned
        physical bytes are intentionally unavailable to another model until
        ``reclaim_status`` becomes ``ready``.
        """
        assert self.client is not None
        response = self.client.commit_live_reclaim(segment_ids)
        page_ranges = [
            (start // self.page_size, count // self.page_size)
            for start, count in response["ranges"]
        ]
        self._free_page_ranges.remove_ranges(
            self._free_page_ranges.intersection_ranges(page_ranges)
        )
        draining = self._draining_page_ranges.intersection_ranges(page_ranges)
        self._draining_page_ranges.remove_ranges(draining)
        self._active_page_ranges.remove_ranges(page_ranges)
        self._draining_segment_ids.difference_update(segment_ids)
        self.active_size = self.client.capacity
        self._refresh_local_residency()
        return response

    def reclaim_status(self, reclaim_id: int) -> dict:
        assert self.client is not None
        return self.client.reclaim_status(reclaim_id)

    def quota_target(self) -> int | None:
        """Return the agent-owned target quota for scheduler-side convergence."""
        assert self.client is not None
        response = self.client.quota_target()
        target = response["target_capacity"]
        raw_set_ns = response.get("quota_target_set_ns")
        self._last_quota_target_set_ns = (
            None if raw_set_ns is None else int(raw_set_ns)
        )
        return None if target is None else int(target)

    @property
    def quota_target_set_ns(self) -> int | None:
        """Return the agent timestamp of the most recently observed target."""
        return getattr(self, "_last_quota_target_set_ns", None)

    def resize_quota(self, target_size: int, *, target_bytes: int | None = None) -> int:
        """Change this instance's active host-KV quota without re-registering memory.

        Shrink only returns completely free Central I/O segments.  Live KV ranges
        are deliberately rejected here; drain/evict belongs to the next phase.
        """
        if target_size % self.page_size != 0:
            raise ValueError("Central I/O quota target must be a multiple of page_size")
        if target_bytes is None and not self.dynamic_page_leases:
            target_size = self._align_quota_to_extents(target_size)
        elif target_bytes is not None:
            target_size = self._quota_slots_for_physical_bytes(target_bytes)
        if target_size <= 0 or target_size > self.max_size:
            raise ValueError("Central I/O quota target is outside this instance's extent-aligned capacity")
        with self.lock:
            assert self.client is not None
            total_started = time.perf_counter()
            old_active_size = self.active_size
            if target_size == self.active_size:
                return self.active_size
            if target_size > self.active_size:
                grow_count = target_size - self.active_size
                if self.max_size - self.active_size < grow_count:
                    raise ValueError("Central I/O has insufficient inactive logical slots to grow")
                prepare_started = time.perf_counter()
                prepare_ms = (time.perf_counter() - prepare_started) * 1000
                rpc_started = time.perf_counter()
                prepared = self.client.prepare_grow_range(grow_count)
                rpc_prepare_ms = (time.perf_counter() - rpc_started) * 1000
                transfer_id = int(prepared["transfer_id"])
                ranges = [tuple(item) for item in prepared["ranges"]]
                local_started = time.perf_counter()
                page_ranges = [
                    (start // self.page_size, count // self.page_size)
                    for start, count in ranges
                ]
                try:
                    self._free_page_ranges.add_ranges(page_ranges)
                    self._active_page_ranges.add_ranges(page_ranges)
                except Exception:
                    self.client.abort_prepared_grow(transfer_id)
                    raise
                local_ms = (time.perf_counter() - local_started) * 1000
                ack_started = time.perf_counter()
                acknowledged = self.client.ack_prepared_grow(transfer_id)
                rpc_ack_ms = (time.perf_counter() - ack_started) * 1000
                rpc_ms = rpc_prepare_ms + rpc_ack_ms
                if int(acknowledged["effective_capacity"]) != target_size:
                    raise RuntimeError("Central I/O grow acknowledgement disagrees with local target")
                direction = "grow"
            else:
                shrink_count = self.active_size - target_size
                prepare_started = time.perf_counter()
                rpc_started = time.perf_counter()
                prepared = self.client.prepare_shrink_free(shrink_count)
                rpc_prepare_ms = (time.perf_counter() - rpc_started) * 1000
                prepare_ms = (time.perf_counter() - prepare_started) * 1000
                transfer_id = int(prepared["transfer_id"])
                page_ranges = [tuple(item) for item in prepared["page_ranges"]]
                local_started = time.perf_counter()
                try:
                    self._free_page_ranges.remove_ranges(page_ranges)
                    self._active_page_ranges.remove_ranges(page_ranges)
                except Exception:
                    self.client.abort_prepared_shrink(transfer_id)
                    raise
                local_ms = (time.perf_counter() - local_started) * 1000
                ack_started = time.perf_counter()
                acknowledged = self.client.ack_prepared_shrink(transfer_id)
                rpc_ack_ms = (time.perf_counter() - ack_started) * 1000
                rpc_ms = rpc_prepare_ms + rpc_ack_ms
                if int(acknowledged["effective_capacity"]) != target_size:
                    raise RuntimeError("Central I/O shrink acknowledgement disagrees with local target")
                direction = "shrink"
            self.active_size = self.client.capacity
            self._refresh_local_residency()
            self._publish_effective_quota_change()
            self._record_pressure(
                "quota_change_pages",
                abs(self.active_size - old_active_size) // self.page_size,
            )
            timing_ms = {
                "model_total": (time.perf_counter() - total_started) * 1000,
                "model_prepare": prepare_ms,
                "rpc_round_trip": rpc_ms,
                "rpc_prepare": rpc_prepare_ms,
                "rpc_ack": rpc_ack_ms,
                "model_post_rpc": local_ms,
                "agent": self.client.last_resize_agent_timing,
            }
            print(
                f"central-io resize model={self.model_id} direction={direction} "
                f"slots={abs(target_size - old_active_size)} timing_ms={timing_ms}",
                flush=True,
            )
            return self.active_size

    def quota_status(self) -> dict[str, int]:
        assert self.client is not None
        return self.client.status()

    def backup_from_device_all_layer(
        self, device_pool, host_indices, device_indices, io_backend
    ) -> None:
        if io_backend != "kernel":
            raise ValueError("Central I/O V1 only supports the kernel backend")
        # Existing controller streams cannot be shared across processes.  V1
        # synchronizes before RPC; the agent synchronizes before replying.
        torch.cuda.synchronize(self.device_pool.device)
        assert self.client is not None
        started = time.perf_counter()
        self.client.backup(
            self._indices_to_list(host_indices), self._indices_to_list(device_indices)
        )
        elapsed_s = time.perf_counter() - started
        backup_pages = (len(host_indices) + self.page_size - 1) // self.page_size
        if backup_pages:
            self._local_residency.observe_backup(
                pages=backup_pages,
                elapsed_s=max(elapsed_s, 1e-9),
                now_s=time.monotonic(),
            )
        self._record_pressure("backup_slots", len(host_indices))

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ) -> None:
        if io_backend != "kernel":
            raise ValueError("Central I/O V1 only supports the kernel backend")
        # The agent restores all MHA layers in one page-first kernel on layer 0.
        # Later per-layer callbacks are retained for controller event semantics.
        if layer_id != 0:
            return
        torch.cuda.synchronize(self.device_pool.device)
        assert self.client is not None
        self.client.restore(
            self._indices_to_list(host_indices), self._indices_to_list(device_indices)
        )
        self._record_pressure("restore_slots", len(host_indices))

    def get_data_page(self, index, flat: bool = True) -> torch.Tensor:
        raise RuntimeError("Central I/O V1 does not support an external storage backend")

    def get_dummy_flat_data_page(self) -> torch.Tensor:
        raise RuntimeError("Central I/O V1 does not support an external storage backend")

    def set_from_flat_data_page(self, index: int, data_page: torch.Tensor) -> None:
        raise RuntimeError("Central I/O V1 does not support an external storage backend")

    def begin_agent_storage_write(
        self, page_keys: list[str], host_indices: torch.Tensor
    ) -> dict | None:
        """Persist only complete Central-I/O host pages through the agent.

        The model process owns logical slot allocation, while the agent owns
        the raw registered bytes.  Passing a tensor to an upstream storage
        backend would therefore be both a second ownership path and unsafe
        for a radix node ending in the middle of a page.  This method reduces
        the node to complete SGLang pages and asks the agent to write exactly
        those page bytes.  ``None`` means that the node has no complete page;
        it remains host-only rather than receiving a misleading durable ack.
        """
        if self.client is None:
            raise RuntimeError("Central I/O client is not connected")
        from sglang.srt.mem_cache.central_io import complete_storage_page_plan

        keys, page_ranges = complete_storage_page_plan(
            page_keys, host_indices, page_size=self.page_size
        )
        if not keys:
            return None
        response = self.client.storage_write_pages(keys, page_ranges)
        page_ids = tuple(
            page_id
            for start, count in page_ranges
            for page_id in range(start, start + count)
        )
        return {
            "operation_id": int(response["operation_id"]),
            "keys": tuple(keys),
            "page_ranges": tuple(page_ranges),
            "page_ids": page_ids,
        }

    def agent_storage_existing_prefix(self, page_keys: list[str]) -> int:
        """Return the contiguous durable prefix available to Central I/O read."""
        if self.client is None:
            raise RuntimeError("Central I/O client is not connected")
        return self.client.storage_existing_prefix(page_keys)

    def begin_agent_storage_read(
        self, page_keys: list[str], host_indices: torch.Tensor
    ) -> dict | None:
        """Restore complete durable pages into already-reserved host slots.

        The caller owns logical allocation and waits for the returned agent
        acknowledgement before publishing any restored radix prefix.
        """
        if self.client is None:
            raise RuntimeError("Central I/O client is not connected")
        from sglang.srt.mem_cache.central_io import complete_storage_page_plan

        keys, page_ranges = complete_storage_page_plan(
            page_keys, host_indices, page_size=self.page_size
        )
        if not keys:
            return None
        response = self.client.storage_read_pages(keys, page_ranges)
        page_ids = tuple(
            page_id
            for start, count in page_ranges
            for page_id in range(start, start + count)
        )
        return {
            "operation_id": int(response["operation_id"]),
            "keys": tuple(keys),
            "page_ranges": tuple(page_ranges),
            "page_ids": page_ids,
        }

    def agent_storage_status(self, operation_id: int) -> dict:
        """Return the agent's durable/read failure acknowledgement."""
        if self.client is None:
            raise RuntimeError("Central I/O client is not connected")
        return self.client.storage_status(operation_id)

    def close(self) -> None:
        self._pressure_logger.flush(
            live_pages=self._live_host_pages,
            active_pages=self.active_size // self.page_size,
            clean_free_pages=self._free_page_ranges.count,
            draining_pages=self._draining_page_ranges.count,
        )
        if self.client is not None:
            self.client.close()
            self.client = None


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

    def load_to_device_per_layer(
        self, device_pool, host_indices, device_indices, layer_id, io_backend
    ):
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
