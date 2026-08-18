"""Synchronous Central I/O backend for SGLang HostKVCache.

The agent owns one mmap-backed, CUDA-registered host pool.  A model process
exports its ordinary SGLang GPU KV allocations once through CUDA IPC; the
agent opens persistent mappings and performs page-first backup/load requests.
V1 deliberately uses synchronous RPC so the existing HiCache event lifecycle
remains correct while the runtime integration is validated.
"""

from __future__ import annotations

import base64
import bisect
import ctypes
import ctypes.util
import mmap
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from multiprocessing.connection import Client, Listener
from pathlib import Path
from typing import Any

from sglang.srt.mem_cache.physical_byte_ranges import FreeByteRanges


IPC_AUTHKEY = b"sglang-central-io-v1"
IPC_LAZY_ENABLE_PEER_ACCESS = 1


class CUDAIpcMemHandle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_char * 64)]


def _load_cudart() -> ctypes.CDLL:
    for candidate in (
        ctypes.util.find_library("cudart"),
        "libcudart.so.12",
        "libcudart.so",
    ):
        if candidate is None:
            continue
        try:
            runtime = ctypes.CDLL(candidate)
            runtime.cudaSetDevice.argtypes = [ctypes.c_int]
            runtime.cudaSetDevice.restype = ctypes.c_int
            runtime.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
            runtime.cudaHostRegister.restype = ctypes.c_int
            runtime.cudaHostUnregister.argtypes = [ctypes.c_void_p]
            runtime.cudaHostUnregister.restype = ctypes.c_int
            runtime.cudaIpcGetMemHandle.argtypes = [ctypes.POINTER(CUDAIpcMemHandle), ctypes.c_void_p]
            runtime.cudaIpcGetMemHandle.restype = ctypes.c_int
            runtime.cudaIpcOpenMemHandle.argtypes = [ctypes.POINTER(ctypes.c_void_p), CUDAIpcMemHandle, ctypes.c_uint]
            runtime.cudaIpcOpenMemHandle.restype = ctypes.c_int
            runtime.cudaIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
            runtime.cudaIpcCloseMemHandle.restype = ctypes.c_int
            runtime.cudaGetErrorString.argtypes = [ctypes.c_int]
            runtime.cudaGetErrorString.restype = ctypes.c_char_p
            return runtime
        except OSError:
            continue
    raise RuntimeError("unable to load libcudart")


def _require(runtime: ctypes.CDLL, code: int, action: str) -> None:
    if code == 0:
        return
    detail = runtime.cudaGetErrorString(code)
    message = detail.decode() if detail else "unknown"
    raise RuntimeError(f"{action} failed: {code} ({message})")


def _decode_handle(encoded: str) -> CUDAIpcMemHandle:
    return CUDAIpcMemHandle.from_buffer_copy(base64.b64decode(encoded.encode()))


def _tensor_descriptors(tensors: list[Any], runtime: ctypes.CDLL) -> list[dict[str, Any]]:
    """Export PyTorch allocator segment handles plus per-tensor offsets."""
    import torch

    segments = [
        (segment["address"], segment["total_size"])
        for segment in torch.cuda.memory_snapshot()
    ]
    exported: dict[int, str] = {}
    result = []
    for tensor in tensors:
        pointer = tensor.data_ptr()
        base = next(
            (address for address, size in segments if address <= pointer < address + size),
            None,
        )
        if base is None:
            raise RuntimeError(f"no allocator segment contains CUDA pointer {pointer:#x}")
        if base not in exported:
            handle = CUDAIpcMemHandle()
            _require(
                runtime,
                runtime.cudaIpcGetMemHandle(ctypes.byref(handle), ctypes.c_void_p(base)),
                "cudaIpcGetMemHandle",
            )
            exported[base] = base64.b64encode(bytes(handle)).decode()
        result.append({"handle": exported[base], "offset": pointer - base})
    return result


class CentralIOClient:
    """Thread-safe synchronous client used by one SGLang model process."""

    def __init__(self, socket_path: str, registration: dict[str, Any]):
        self._connection = Client(socket_path, family="AF_UNIX", authkey=IPC_AUTHKEY)
        self._lock = threading.Lock()
        self._closed = False
        response = self._call({"op": "register", **registration})
        self.model_id = response["model_id"]
        self.capacity = response["capacity"]
        self.max_capacity = response["max_capacity"]
        self.extent_slots = response.get("extent_slots")
        self.arena_extent_bytes = response.get("arena_extent_bytes")
        self.lease_mode = response.get("lease_mode", "extent")
        self.last_resize_agent_timing: dict[str, float] | None = None

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            if self._closed:
                raise RuntimeError("Central I/O client is closed")
            self._connection.send(request)
            response = self._connection.recv()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Central I/O agent failed"))
        return response

    def reserve(self, indices: list[int]) -> None:
        self._call({"op": "reserve", "model_id": self.model_id, "indices": indices})

    def release(self, indices: list[int]) -> None:
        self._call({"op": "release", "model_id": self.model_id, "indices": indices})

    def reserve_pages(self, ranges: list[tuple[int, int]]) -> None:
        """Mark complete SGLang KV pages live in the central host pool.

        ``ranges`` uses page coordinates, not token-slot coordinates.  This is
        deliberately separate from the legacy ``reserve(indices)`` operation,
        which remains available only for the slot-level benchmark comparison.
        """
        self._call(
            {
                "op": "reserve_pages",
                "model_id": self.model_id,
                "ranges": [[start, count] for start, count in ranges],
            }
        )

    def release_pages(
        self, ranges: list[tuple[int, int]], *, allow_already_released: bool = False
    ) -> None:
        """Release complete SGLang KV pages previously reserved by this model.

        ``allow_already_released`` is used only by the normal HostKV free path
        after a concurrent live-drain has already detached the same page.
        """
        self._call(
            {
                "op": "release_pages",
                "model_id": self.model_id,
                "ranges": [[start, count] for start, count in ranges],
                "allow_already_released": allow_already_released,
            }
        )

    def begin_live_page_drain(self, ranges: list[tuple[int, int]]) -> dict[str, Any]:
        return self._call(
            {
                "op": "begin_live_page_drain",
                "model_id": self.model_id,
                "ranges": [[start, count] for start, count in ranges],
            }
        )

    def abort_live_page_drain(self, ranges: list[tuple[int, int]]) -> dict[str, Any]:
        return self._call(
            {
                "op": "abort_live_page_drain",
                "model_id": self.model_id,
                "ranges": [[start, count] for start, count in ranges],
            }
        )

    def commit_live_page_reclaim(self, ranges: list[tuple[int, int]]) -> dict[str, Any]:
        response = self._call(
            {
                "op": "commit_live_page_reclaim",
                "model_id": self.model_id,
                "ranges": [[start, count] for start, count in ranges],
            }
        )
        self.capacity = response["capacity"]
        return response

    def set_quota_target(self, target_capacity: int) -> dict[str, Any]:
        return self._call(
            {
                "op": "set_quota_target",
                "model_id": self.model_id,
                "target_capacity": target_capacity,
            }
        )

    def quota_target(self) -> dict[str, Any]:
        return self._call({"op": "quota_target", "model_id": self.model_id})

    def report_local_residency(self, status: dict[str, Any]) -> dict[str, Any]:
        """Publish model-local watermarks and debt to the central scheduler."""
        return self._call(
            {"op": "report_local_residency", "model_id": self.model_id, **status}
        )

    def describe_layout(self) -> dict[str, Any]:
        """Return read-only page/extent metadata for a runtime snapshot."""
        return self._call({"op": "describe_layout", "model_id": self.model_id})

    def backup(self, host_indices: list[int], device_indices: list[int]) -> None:
        self._call(
            {
                "op": "backup",
                "model_id": self.model_id,
                "host_indices": host_indices,
                "device_indices": device_indices,
            }
        )

    def restore(self, host_indices: list[int], device_indices: list[int]) -> None:
        self._call(
            {
                "op": "restore",
                "model_id": self.model_id,
                "host_indices": host_indices,
                "device_indices": device_indices,
            }
        )

    def grow_range(self, count: int) -> list[tuple[int, int]]:
        response = self._call(
            {"op": "grow_range", "model_id": self.model_id, "count": count}
        )
        self.capacity = response["capacity"]
        self.last_resize_agent_timing = response.get("timing_ms")
        return [tuple(item) for item in response["ranges"]]

    def prepare_grow_range(self, count: int) -> dict[str, Any]:
        """Reserve physical ranges; they remain unusable until ``ack``."""
        return self._call(
            {"op": "prepare_grow", "model_id": self.model_id, "count": count}
        )

    def ack_prepared_grow(self, transfer_id: int) -> dict[str, Any]:
        response = self._call(
            {
                "op": "ack_prepared_grow",
                "model_id": self.model_id,
                "transfer_id": transfer_id,
            }
        )
        self.capacity = int(response["effective_capacity"])
        return response

    def abort_prepared_grow(self, transfer_id: int) -> dict[str, Any]:
        response = self._call(
            {
                "op": "abort_prepared_grow",
                "model_id": self.model_id,
                "transfer_id": transfer_id,
            }
        )
        self.capacity = int(response["effective_capacity"])
        return response

    def prepare_shrink_free(self, count: int) -> dict[str, Any]:
        """Fence clean pages; they remain in the effective lease until ``ack``."""
        return self._call(
            {"op": "prepare_shrink_free", "model_id": self.model_id, "count": count}
        )

    def ack_prepared_shrink(self, transfer_id: int) -> dict[str, Any]:
        response = self._call(
            {
                "op": "ack_prepared_shrink",
                "model_id": self.model_id,
                "transfer_id": transfer_id,
            }
        )
        self.capacity = int(response["effective_capacity"])
        return response

    def abort_prepared_shrink(self, transfer_id: int) -> dict[str, Any]:
        response = self._call(
            {
                "op": "abort_prepared_shrink",
                "model_id": self.model_id,
                "transfer_id": transfer_id,
            }
        )
        self.capacity = int(response["effective_capacity"])
        return response

    def shrink_free_ranges(self, count: int) -> list[tuple[int, int]]:
        response = self._call(
            {"op": "shrink_free", "model_id": self.model_id, "count": count}
        )
        self.capacity = response["capacity"]
        self.last_resize_agent_timing = response.get("timing_ms")
        return [tuple(item) for item in response["ranges"]]

    def plan_live_reclaim(
        self,
        target_capacity: int,
        excluded_segment_ids: set[int] | None = None,
        max_segments: int | None = None,
    ) -> dict[str, Any]:
        """Return a non-mutating whole-extent reclaim plan toward a target.

        The caller must have HiRadix drain the returned live extents before it
        asks the agent to detach them.  Exclusion lets the scheduler retry when
        a candidate contains an in-flight or structurally protected radix node.
        ``target_capacity`` is deliberately sent instead of a precomputed
        reclaim count: only the agent has authoritative knowledge of completed
        asynchronous detachments.
        """
        return self._call(
            {
                "op": "plan_live_reclaim",
                "model_id": self.model_id,
                "target_capacity": target_capacity,
                "excluded_segment_ids": sorted(excluded_segment_ids or set()),
                "max_segments": max_segments,
            }
        )

    def commit_live_reclaim(self, segment_ids: list[int]) -> dict[str, Any]:
        response = self._call(
            {
                "op": "commit_live_reclaim",
                "model_id": self.model_id,
                "segment_ids": segment_ids,
            }
        )
        self.capacity = response["capacity"]
        return response

    def begin_live_drain(self, segment_ids: list[int]) -> dict[str, Any]:
        return self._call(
            {"op": "begin_live_drain", "model_id": self.model_id, "segment_ids": segment_ids}
        )

    def abort_live_drain(self, segment_ids: list[int]) -> dict[str, Any]:
        return self._call(
            {"op": "abort_live_drain", "model_id": self.model_id, "segment_ids": segment_ids}
        )

    def reclaim_status(self, reclaim_id: int) -> dict[str, Any]:
        return self._call(
            {
                "op": "reclaim_status",
                "model_id": self.model_id,
                "reclaim_id": reclaim_id,
            }
        )

    def status(self) -> dict[str, int]:
        return self._call({"op": "status", "model_id": self.model_id})

    def reset_reservations(self) -> None:
        self._call({"op": "reset_reservations", "model_id": self.model_id})

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._call({"op": "unregister", "model_id": self.model_id})
        finally:
            self._connection.close()
            self._closed = True


class CentralIOControlClient:
    """Small scheduler-facing client for setting model quota targets."""

    def __init__(self, socket_path: str):
        self._connection = Client(socket_path, family="AF_UNIX", authkey=IPC_AUTHKEY)

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        self._connection.send(request)
        response = self._connection.recv()
        if not response.get("ok", False):
            raise RuntimeError(response.get("error", "Central I/O control failed"))
        return response

    def status(self, model_id: str) -> dict[str, Any]:
        return self._call({"op": "status", "model_id": model_id})

    def global_status(self) -> dict[str, Any]:
        return self._call({"op": "global_status"})

    def set_quota_target(self, model_id: str, target_capacity: int) -> dict[str, Any]:
        return self._call(
            {
                "op": "set_quota_target",
                "model_id": model_id,
                "target_capacity": target_capacity,
            }
        )

    def set_quota_targets(self, targets: dict[str, int]) -> dict[str, Any]:
        """Publish one scheduler decision for several models atomically."""
        return self._call(
            {
                "op": "set_quota_targets",
                "targets": {model_id: int(target) for model_id, target in targets.items()},
            }
        )

    def set_quota_target_gib(self, model_id: str, target_gib: int) -> dict[str, Any]:
        """Set an integer-GiB scheduling target with explicit page rounding."""
        if isinstance(target_gib, bool) or not isinstance(target_gib, int) or target_gib <= 0:
            raise ValueError("Central I/O quota target must be a positive integer GiB")
        status = self.status(model_id)
        requested_bytes = target_gib * 1024**3
        token_bytes = int(status["token_bytes"])
        page_size = int(status["page_size"])
        target_capacity = requested_bytes // token_bytes
        target_capacity -= target_capacity % page_size
        if target_capacity <= 0:
            raise ValueError("integer-GiB target is smaller than one SGLang KV page")
        response = self.set_quota_target(model_id, target_capacity)
        response.update(
            {
                "requested_gib": target_gib,
                "effective_gib": target_capacity * token_bytes / 1024**3,
                "effective_pages": target_capacity // page_size,
            }
        )
        return response

    def close(self) -> None:
        self._connection.close()


class _PageRangeSet:
    """Disjoint half-open page intervals with validation-oriented updates.

    The agent must know whether a page is live or dirty, but it must not keep
    one Python object per page.  This small interval set is the runtime form
    of that state.  Coordinates are model-local SGLang KV page ids.
    """

    def __init__(self, ranges: list[tuple[int, int]] | None = None) -> None:
        self.ranges: list[tuple[int, int]] = []
        self.count = 0
        if ranges:
            self.add_ranges(ranges)

    def __bool__(self) -> bool:
        return bool(self.ranges)

    def __len__(self) -> int:
        return self.count

    def clear(self) -> None:
        self.ranges.clear()
        self.count = 0

    def copy_ranges(self) -> list[list[int]]:
        return [[start, end - start] for start, end in self.ranges]

    @staticmethod
    def _validate(start: int, count: int) -> tuple[int, int]:
        if start < 0 or count <= 0:
            raise ValueError("Central I/O page range must have non-negative start and positive count")
        return start, start + count

    def overlaps(self, start: int, count: int) -> bool:
        start, end = self._validate(start, count)
        return any(current_start < end and start < current_end for current_start, current_end in self.ranges)

    def contains(self, start: int, count: int) -> bool:
        start, end = self._validate(start, count)
        cursor = start
        for current_start, current_end in self.ranges:
            if current_end <= cursor:
                continue
            if current_start > cursor:
                return False
            cursor = min(end, current_end)
            if cursor == end:
                return True
        return False

    def add_range(self, start: int, count: int, *, require_disjoint: bool = False) -> None:
        start, end = self._validate(start, count)
        if require_disjoint and self.overlaps(start, count):
            raise ValueError("Central I/O page range overlaps existing state")
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
                start = min(start, current_start)
                end = max(end, current_end)
        if not inserted:
            merged.append((start, end))
        self.ranges = merged
        self.count = sum(current_end - current_start for current_start, current_end in merged)

    def add_ranges(
        self, ranges: list[tuple[int, int]], *, require_disjoint: bool = False
    ) -> None:
        for start, count in ranges:
            self.add_range(start, count, require_disjoint=require_disjoint)

    def remove_range(self, start: int, count: int, *, require_present: bool = True) -> None:
        start, end = self._validate(start, count)
        if require_present and not self.contains(start, count):
            raise ValueError("Central I/O page range is not fully present")
        updated: list[tuple[int, int]] = []
        for current_start, current_end in self.ranges:
            if current_end <= start or end <= current_start:
                updated.append((current_start, current_end))
                continue
            if current_start < start:
                updated.append((current_start, start))
            if end < current_end:
                updated.append((end, current_end))
        self.ranges = updated
        self.count = sum(current_end - current_start for current_start, current_end in updated)

    def remove_ranges(
        self, ranges: list[tuple[int, int]], *, require_present: bool = True
    ) -> None:
        # Validate first so a malformed multi-range request remains atomic.
        shadow = _PageRangeSet([(start, end - start) for start, end in self.ranges])
        for start, count in ranges:
            shadow.remove_range(start, count, require_present=require_present)
        self.ranges = shadow.ranges
        self.count = shadow.count

    def intersection_ranges(self, ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
        """Return the page intervals shared with ``ranges`` without mutation."""
        intersections: list[tuple[int, int]] = []
        for requested_start, requested_count in ranges:
            requested_start, requested_end = self._validate(
                requested_start, requested_count
            )
            for current_start, current_end in self.ranges:
                start = max(requested_start, current_start)
                end = min(requested_end, current_end)
                if start < end:
                    intersections.append((start, end - start))
                if current_start >= requested_end:
                    break
        return intersections


@dataclass
class _HostSegment:
    """One model lease of one complete physical ``ArenaExtent``.

    ``slot_count`` can be smaller than ``allocated_bytes / token_bytes`` due
    to page alignment. The unused tail remains part of this lease; it is never
    handed to another model as an interior slice.
    """

    segment_id: int
    byte_offset: int
    allocated_bytes: int
    logical_start: int
    slot_count: int
    host_k: Any
    host_v: Any
    # Legacy slot tracking remains only for the old-path control-plane
    # benchmark.  Central I/O runtime uses ``reserved_pages`` below.
    reserved_slots: set[int] = field(default_factory=set)
    reserved_pages: _PageRangeSet = field(default_factory=_PageRangeSet)
    # Allocator-free pages can still contain former tenant KV bytes. They are
    # not safely transferable to another owner until scrubbed.
    dirty_pages: _PageRangeSet = field(default_factory=_PageRangeSet)
    # A draining extent remains owned by its current model, but receives no
    # new host-KV admission while HiRadix peels safe leaves bottom-up.
    draining: bool = False
    # Dynamic page leases fence only the selected pages.  The rest of the
    # variable physical mapping remains available to its current owner.
    draining_pages: _PageRangeSet = field(default_factory=_PageRangeSet)
    # A grow allocates physical bytes before the recipient model has rebuilt
    # its local allocator views. Pending bytes are owned by neither allocator
    # until the recipient explicitly acknowledges them.
    pending_transfer_id: int | None = None


@dataclass
class _ArenaExtent:
    """Fixed physical ownership unit in the globally pinned host arena."""

    extent_id: int
    byte_offset: int
    byte_size: int
    owner_model_id: str | None = None
    state: str = "free"  # free | leased | scrubbing


@dataclass
class _ModelState:
    model_id: str
    device: int
    max_capacity: int
    segment_tokens: int
    k_ptrs: Any
    v_ptrs: Any
    opened_bases: dict[str, ctypes.c_void_p]
    layer_count: int
    head_count: int
    head_dim: int
    dtype: Any
    reserved: set[int]
    segments: dict[int, _HostSegment]
    segment_starts: list[int]
    segments_by_start: dict[int, int]
    active_capacity: int
    page_size: int = 1
    next_segment_id: int = 0
    # Logical slot intervals not currently leased to this model.  They are
    # separate from the global ArenaExtent free list: a regrown model range
    # may reuse a prior logical interval while receiving different extents.
    inactive_ranges: list[tuple[int, int]] = field(default_factory=list)
    reserved_pages: _PageRangeSet = field(default_factory=_PageRangeSet)
    dynamic_page_leases: bool = False
    quota_target: int | None = None
    quota_target_set_ns: int | None = None
    last_grow_accepted_ns: int | None = None
    first_grow_accepted_ns: int | None = None
    grow_accept_count: int = 0
    last_reclaim_id: int | None = None
    # Last model-process report. It is intentionally separate from agent
    # ownership state: only the model knows local radix value and watermarks.
    residency_report: dict[str, Any] | None = None

    @property
    def capacity(self) -> int:
        return self.active_capacity


@dataclass
class _PendingReclaim:
    """A detached extent set that is being securely scrubbed in the agent."""

    reclaim_id: int
    ranges: list[list[int]]
    byte_count: int
    done: threading.Event = field(default_factory=threading.Event)
    state: str = "scrubbing"
    scrub_ms: float | None = None
    error: str | None = None
    detached_ns: int | None = None
    scrub_started_ns: int | None = None
    scrub_ready_ns: int | None = None


@dataclass(frozen=True)
class _PendingGrow:
    """A recipient-only physical lease awaiting local allocator admission."""

    transfer_id: int
    model_id: str
    segment_ids: tuple[int, ...]
    slot_count: int
    ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _PendingShrink:
    """Donor-only clean-page handoff awaiting local allocator removal."""

    transfer_id: int
    model_id: str
    page_ranges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class _DetachedPageRange:
    """A page-aligned piece removed from one model's physical lease."""

    logical_start: int
    slot_count: int
    byte_offset: int
    byte_count: int


class CentralIOAgent:
    """One globally pinned host pool, shared by registered model endpoints."""

    def __init__(
        self,
        socket_path: str,
        pool_bytes: int,
        arena_extent_bytes: int | None = None,
    ) -> None:
        if pool_bytes <= 0:
            raise ValueError("pool_bytes must be positive")
        if arena_extent_bytes is None:
            arena_extent_bytes = int(
                os.getenv("SGLANG_CENTRAL_IO_ARENA_EXTENT_BYTES", str(256 * 1024**2))
            )
        if arena_extent_bytes <= 0 or arena_extent_bytes % 4096:
            raise ValueError("Central I/O arena extent must be a positive 4KiB multiple")
        self._configured_lease_mode = os.getenv("SGLANG_CENTRAL_IO_LEASE_MODE", "extent")
        if self._configured_lease_mode not in {"extent", "page"}:
            raise ValueError("SGLANG_CENTRAL_IO_LEASE_MODE must be 'extent' or 'page'")
        if self._configured_lease_mode == "extent" and pool_bytes % arena_extent_bytes:
            raise ValueError("Central I/O pool_bytes must be a whole number of arena extents")
        self.socket_path = socket_path
        self.pool_bytes = pool_bytes
        self.arena_extent_bytes = arena_extent_bytes
        self.runtime = _load_cudart()
        self._fd = os.memfd_create("sglang-central-io-host-kv", os.MFD_CLOEXEC)
        os.ftruncate(self._fd, pool_bytes)
        self._mapping = mmap.mmap(self._fd, pool_bytes, access=mmap.ACCESS_WRITE)
        self._address = ctypes.addressof(ctypes.c_char.from_buffer(self._mapping))
        # Registration is independent of the GPU on which a model's IPC
        # transfer kernel later runs.  Keeping it configurable prevents the
        # central pool's driver mapping from competing with a serving model.
        self._registration_device = int(
            os.getenv("SGLANG_CENTRAL_IO_POOL_DEVICE_ID", "0")
        )
        _require(
            self.runtime,
            self.runtime.cudaSetDevice(self._registration_device),
            f"cudaSetDevice({self._registration_device})",
        )
        _require(
            self.runtime,
            self.runtime.cudaHostRegister(ctypes.c_void_p(self._address), pool_bytes, 0),
            "cudaHostRegister(central pool)",
        )
        self._registered = True
        self._models: dict[str, _ModelState] = {}
        self._arena_extents = {
            extent_id: _ArenaExtent(
                extent_id=extent_id,
                byte_offset=extent_id * arena_extent_bytes,
                byte_size=arena_extent_bytes,
            )
            for extent_id in range(pool_bytes // arena_extent_bytes)
        }
        self._free_extent_ids = set(self._arena_extents)
        self._lock = threading.RLock()
        self._reclaims: dict[int, _PendingReclaim] = {}
        self._next_reclaim_id = 0
        self._cuda_lock = threading.Lock()
        self._page_scrub_executor = ThreadPoolExecutor(
            max_workers=self._page_scrub_worker_count(),
            thread_name_prefix="central-io-page-scrub",
        )
        self._listener: Listener | None = None
        self._backup_calls = 0
        self._restore_calls = 0
        self._backup_pages = 0
        self._restore_pages = 0
        self._lease_mode: str | None = None

    @staticmethod
    def _page_scrub_worker_count() -> int:
        """Bound concurrent secure erases so a live handoff cannot stampede DRAM."""
        workers = int(os.getenv("SGLANG_CENTRAL_IO_PAGE_SCRUB_WORKERS", "1"))
        if workers <= 0:
            raise ValueError("SGLANG_CENTRAL_IO_PAGE_SCRUB_WORKERS must be positive")
        return workers

    def _submit_page_scrub(self, task) -> None:
        """Queue page scrub work; later ranges wait instead of spawning threads."""
        executor = getattr(self, "_page_scrub_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(
                max_workers=self._page_scrub_worker_count(),
                thread_name_prefix="central-io-page-scrub",
            )
            self._page_scrub_executor = executor
        executor.submit(task)

    def _init_dynamic_free_bytes(self, pool_bytes: int) -> None:
        """Enable variable page-range leases for an arena with one registration.

        This intentionally coexists with the V1 fixed-extent tables while the
        runtime adapter is migrated.  A dynamic model never consumes the V1
        extent free list: its ownership is represented by its own mapped
        physical ranges, backed by this coalescing byte free list.
        """
        self._free_byte_ranges = FreeByteRanges([(0, pool_bytes)])
        self._next_segment_id = 0

    @staticmethod
    def _page_bytes(state: _ModelState) -> int:
        return state.page_size * CentralIOAgent._token_bytes(state)

    def _insert_dynamic_segment(
        self,
        state: _ModelState,
        *,
        logical_start: int,
        slot_count: int,
        byte_offset: int,
    ) -> _HostSegment:
        """Attach one already-owned physical range to a model logical span."""
        import torch

        if slot_count <= 0 or logical_start % state.page_size or slot_count % state.page_size:
            raise ValueError("Central I/O dynamic segment must be page aligned")
        if logical_start < 0 or logical_start + slot_count > state.max_capacity:
            raise ValueError("Central I/O dynamic segment range outside model capacity")
        byte_count = slot_count * self._token_bytes(state)
        if byte_offset < 0 or byte_offset + byte_count > len(self._mapping):
            raise ValueError("Central I/O dynamic segment range outside physical pool")

        position = bisect.bisect_left(state.segment_starts, logical_start)
        if position and state.segment_starts[position - 1] + state.segments[
            state.segments_by_start[state.segment_starts[position - 1]]
        ].slot_count > logical_start:
            raise ValueError("Central I/O dynamic segment overlaps an active logical range")
        if position < len(state.segment_starts) and logical_start + slot_count > state.segment_starts[position]:
            raise ValueError("Central I/O dynamic segment overlaps an active logical range")

        element_count = byte_count // torch.empty((), dtype=state.dtype).element_size()
        host = torch.frombuffer(
            self._mapping,
            dtype=state.dtype,
            count=element_count,
            offset=byte_offset,
        ).view(
            2,
            slot_count,
            state.layer_count,
            state.head_count,
            state.head_dim,
        )
        segment_id = getattr(self, "_next_segment_id", 0)
        self._next_segment_id = segment_id + 1
        segment = _HostSegment(
            segment_id=segment_id,
            byte_offset=byte_offset,
            allocated_bytes=byte_count,
            logical_start=logical_start,
            slot_count=slot_count,
            host_k=host[0],
            host_v=host[1],
        )
        state.segments[segment_id] = segment
        state.segment_starts.insert(position, logical_start)
        state.segments_by_start[logical_start] = segment_id
        state.active_capacity += slot_count
        return segment

    def _add_dynamic_segment(
        self, state: _ModelState, logical_start: int, slot_count: int
    ) -> _HostSegment:
        """Lease a variable-size page-aligned physical range to one model."""
        if not hasattr(self, "_free_byte_ranges"):
            raise RuntimeError("Central I/O dynamic page leases are not initialized")
        byte_count = slot_count * self._token_bytes(state)
        byte_offset = self._free_byte_ranges.allocate(
            byte_count, alignment=self._page_bytes(state)
        )
        if byte_offset is None:
            raise MemoryError("central host pool has insufficient free page-aligned bytes")
        try:
            return self._insert_dynamic_segment(
                state,
                logical_start=logical_start,
                slot_count=slot_count,
                byte_offset=byte_offset,
            )
        except Exception:
            self._free_byte_ranges.release(byte_offset, byte_count)
            raise

    def _remove_dynamic_segment(self, state: _ModelState, segment: _HostSegment) -> None:
        del state.segments[segment.segment_id]
        position = bisect.bisect_left(state.segment_starts, segment.logical_start)
        if position == len(state.segment_starts) or state.segment_starts[position] != segment.logical_start:
            raise RuntimeError("Central I/O dynamic segment index is inconsistent")
        state.segment_starts.pop(position)
        del state.segments_by_start[segment.logical_start]
        state.active_capacity -= segment.slot_count

    def _detach_dynamic_page_range(
        self, state: _ModelState, page_start: int, page_count: int
    ) -> _DetachedPageRange:
        """Cut one fully clean page range out of its current physical mapping."""
        segment, local_page = self._segment_for_page(state, page_start)
        segment_page_count = segment.slot_count // state.page_size
        if local_page + page_count > segment_page_count:
            raise ValueError("dynamic page reclaim range crosses a segment boundary")
        if state.reserved_pages.overlaps(page_start, page_count):
            raise ValueError("cannot reclaim a Central I/O page that is still live")
        local_start = local_page * state.page_size
        reclaim_slots = page_count * state.page_size
        byte_count = reclaim_slots * self._token_bytes(state)
        detached = _DetachedPageRange(
            logical_start=page_start * state.page_size,
            slot_count=reclaim_slots,
            byte_offset=segment.byte_offset + local_start * self._token_bytes(state),
            byte_count=byte_count,
        )

        def rebased_ranges(
            source: _PageRangeSet, start_page: int, child_pages: int
        ) -> list[tuple[int, int]]:
            """Copy one segment-local page map into a split child segment."""
            end_page = start_page + child_pages
            copied: list[tuple[int, int]] = []
            for source_start, source_end in source.ranges:
                overlap_start = max(source_start, start_page)
                overlap_end = min(source_end, end_page)
                if overlap_start < overlap_end:
                    copied.append((overlap_start - start_page, overlap_end - overlap_start))
            return copied

        def restore_child_metadata(
            child: _HostSegment, start_page: int, child_pages: int
        ) -> None:
            child.reserved_pages = _PageRangeSet(
                rebased_ranges(segment.reserved_pages, start_page, child_pages)
            )
            child.dirty_pages = _PageRangeSet(
                rebased_ranges(segment.dirty_pages, start_page, child_pages)
            )
            child.draining_pages = _PageRangeSet(
                rebased_ranges(segment.draining_pages, start_page, child_pages)
            )
            child.draining = segment.draining

        self._remove_dynamic_segment(state, segment)
        if local_start:
            left = self._insert_dynamic_segment(
                state,
                logical_start=segment.logical_start,
                slot_count=local_start,
                byte_offset=segment.byte_offset,
            )
            restore_child_metadata(left, 0, local_page)
        right_slots = segment.slot_count - local_start - reclaim_slots
        if right_slots:
            right = self._insert_dynamic_segment(
                state,
                logical_start=page_start * state.page_size + reclaim_slots,
                slot_count=right_slots,
                byte_offset=detached.byte_offset + byte_count,
            )
            restore_child_metadata(
                right, local_page + page_count, right_slots // state.page_size
            )
        self._add_logical_range(state.inactive_ranges, detached.logical_start, reclaim_slots)
        return detached

    def _commit_live_page_reclaim(
        self, state: _ModelState, page_ranges: list[tuple[int, int]]
    ) -> dict[str, Any]:
        """Detach clean page ranges and asynchronously scrub their exact bytes."""
        if not page_ranges:
            raise ValueError("Central I/O page reclaim requires at least one page range")
        page_ranges = self._validate_page_ranges(
            page_ranges, state.max_capacity // state.page_size
        )
        locations = self._page_range_locations(state, page_ranges)
        for segment, local_start, count in locations:
            if not segment.draining_pages.contains(local_start, count):
                raise ValueError("cannot commit a Central I/O page that is not draining")
        normalized = sorted(page_ranges, reverse=True)
        detached: list[_DetachedPageRange] = []
        for page_start, page_count in normalized:
            if page_start < 0 or page_count <= 0:
                raise ValueError("Central I/O page reclaim range is invalid")
            remaining = page_count
            cursor = page_start
            # Work from the high logical end so splitting a segment cannot
            # invalidate a later offset in the same original mapping.
            pieces: list[tuple[int, int]] = []
            while remaining:
                segment, local_page = self._segment_for_page(state, cursor)
                take = min(remaining, segment.slot_count // state.page_size - local_page)
                pieces.append((cursor, take))
                cursor += take
                remaining -= take
            for piece_start, piece_count in reversed(pieces):
                detached.append(self._detach_dynamic_page_range(state, piece_start, piece_count))

        reclaim_id = getattr(self, "_next_reclaim_id", 0)
        self._next_reclaim_id = reclaim_id + 1
        ranges = [[item.logical_start, item.slot_count] for item in detached]
        byte_count = sum(item.byte_count for item in detached)
        reclaim = _PendingReclaim(
            reclaim_id=reclaim_id,
            ranges=ranges,
            byte_count=byte_count,
            detached_ns=time.monotonic_ns(),
        )
        self._reclaims[reclaim_id] = reclaim
        state.last_reclaim_id = reclaim_id

        def scrub_then_release() -> None:
            started = time.perf_counter()
            reclaim.scrub_started_ns = time.monotonic_ns()
            try:
                for item in detached:
                    ctypes.memset(self._address + item.byte_offset, 0, item.byte_count)
                with self._lock:
                    for item in detached:
                        self._free_byte_ranges.release(item.byte_offset, item.byte_count)
                    self._clean_epoch = getattr(self, "_clean_epoch", 0) + 1
                    reclaim.scrub_ms = (time.perf_counter() - started) * 1000
                    reclaim.scrub_ready_ns = time.monotonic_ns()
                    reclaim.state = "ready"
                    print(
                        "[CENTRAL-IO] page-reclaim-ready "
                        f"model={state.model_id} id={reclaim_id} "
                        f"bytes={byte_count} scrub_ms={reclaim.scrub_ms:.3f}",
                        flush=True,
                    )
            except Exception as error:
                reclaim.error = repr(error)
                reclaim.state = "failed"
            finally:
                reclaim.done.set()

        self._submit_page_scrub(scrub_then_release)
        print(
            "[CENTRAL-IO] page-reclaim-start "
            f"model={state.model_id} id={reclaim_id} "
            f"page_ranges={page_ranges} bytes={byte_count}",
            flush=True,
        )
        return {
            "capacity": state.capacity,
            "ranges": ranges,
            "reclaim_id": reclaim_id,
            "state": reclaim.state,
            "bytes": byte_count,
            "detached_ns": reclaim.detached_ns,
        }

    def _begin_live_page_drain(
        self, state: _ModelState, page_ranges: list[tuple[int, int]]
    ) -> dict[str, Any]:
        """Fence selected live pages before HiRadix releases their nodes.

        This is intentionally page-granular: a drain must not admission-fence
        the unrelated pages that share the same variable host mapping.
        """
        if not page_ranges:
            raise ValueError("Central I/O page drain requires at least one page range")
        max_pages = state.max_capacity // state.page_size
        ranges = self._validate_page_ranges(page_ranges, max_pages)
        if not all(state.reserved_pages.contains(start, count) for start, count in ranges):
            raise ValueError("Central I/O page drain requires currently live pages")
        locations = self._page_range_locations(state, ranges)
        if any(segment.draining_pages.overlaps(local_start, count) for segment, local_start, count in locations):
            raise ValueError("Central I/O page is already draining")
        for segment, local_start, count in locations:
            segment.draining_pages.add_range(local_start, count, require_disjoint=True)
        return {"page_ranges": [[start, count] for start, count in ranges]}

    def _select_allocator_free_page_ranges(
        self, state: _ModelState, count: int
    ) -> list[tuple[int, int]]:
        """Select page-complete, allocator-free donor capacity without mutation.

        In page-lease mode a model may have a single large physical segment
        containing both live and allocator-free pages.  The legacy
        ``_shrink_free`` path only recognised a *wholly* empty segment, which
        made it report zero free capacity in this common state.  The selector
        deliberately has no side effects so donor shrink can be a true
        prepare -> local-remove -> acknowledge handoff.
        """
        if count <= 0 or count % state.page_size:
            raise ValueError("Central I/O free-page shrink must be page aligned")
        needed_pages = count // state.page_size
        candidates = _PageRangeSet()

        for logical_start in state.segment_starts:
            segment = state.segments[state.segments_by_start[logical_start]]
            page_start = segment.logical_start // state.page_size
            page_count = segment.slot_count // state.page_size
            available = _PageRangeSet([(page_start, page_count)])
            available.remove_ranges(state.reserved_pages.copy_ranges(), require_present=False)
            draining_global = [
                (page_start + local_start, local_count)
                for local_start, local_count in segment.draining_pages.copy_ranges()
            ]
            available.remove_ranges(draining_global, require_present=False)
            candidates.add_ranges(available.copy_ranges(), require_disjoint=True)

        selected = _PageRangeSet()
        remaining = needed_pages
        # Prefer the high logical end. It makes the local allocator's trailing
        # lease compact while preserving the same page-level safety invariant.
        for start, end in reversed(candidates.ranges):
            take = min(remaining, end - start)
            selected.add_range(end - take, take, require_disjoint=True)
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            found = needed_pages - remaining
            raise ValueError(
                f"cannot shrink {count} free slots without touching live KV; "
                f"found {found * state.page_size}"
            )

        ranges = selected.copy_ranges()
        return selected.copy_ranges()

    def _prepare_shrink_free(self, state: _ModelState, count: int) -> dict[str, Any]:
        """Fence clean donor pages without changing its effective quota yet."""
        if not state.dynamic_page_leases:
            raise ValueError("prepared shrink requires dynamic page leases")
        ranges = self._select_allocator_free_page_ranges(state, count)
        locations = self._page_range_locations(state, ranges)
        for segment, local_start, local_count in locations:
            segment.draining_pages.add_range(
                local_start, local_count, require_disjoint=True
            )
        transfer_id = getattr(self, "_next_transfer_id", 0)
        self._next_transfer_id = transfer_id + 1
        transfer = _PendingShrink(
            transfer_id=transfer_id,
            model_id=state.model_id,
            page_ranges=tuple(ranges),
        )
        pending = getattr(self, "_pending_shrinks", None)
        if pending is None:
            pending = {}
            self._pending_shrinks = pending
        pending[transfer_id] = transfer
        return {
            "transfer_id": transfer_id,
            "page_ranges": [list(item) for item in ranges],
            "effective_capacity": state.capacity,
            "pending_capacity": count,
        }

    def _pending_shrink(self, state: _ModelState, transfer_id: int) -> _PendingShrink:
        try:
            transfer = self._pending_shrinks[transfer_id]
        except (AttributeError, KeyError) as error:
            raise ValueError("unknown pending Central I/O shrink") from error
        if transfer.model_id != state.model_id:
            raise ValueError("pending Central I/O shrink belongs to another model")
        return transfer

    def _ack_prepared_shrink(self, state: _ModelState, transfer_id: int) -> dict[str, Any]:
        """Detach only after the donor removed the pages from its allocator."""
        transfer = self._pending_shrink(state, transfer_id)
        reclaim = self._commit_live_page_reclaim(state, list(transfer.page_ranges))
        del self._pending_shrinks[transfer_id]
        return {
            "transfer_id": transfer_id,
            "reclaim_id": reclaim["reclaim_id"],
            "ranges": reclaim["ranges"],
            "state": reclaim["state"],
            "effective_capacity": state.capacity,
        }

    def _abort_prepared_shrink(self, state: _ModelState, transfer_id: int) -> dict[str, Any]:
        """Unfence donor pages if its local allocator rejects the handoff."""
        transfer = self._pending_shrink(state, transfer_id)
        locations = self._page_range_locations(state, list(transfer.page_ranges))
        for segment, local_start, local_count in locations:
            segment.draining_pages.remove_range(
                local_start, local_count, require_present=True
            )
        del self._pending_shrinks[transfer_id]
        return {
            "transfer_id": transfer_id,
            "effective_capacity": state.capacity,
        }

    def _reclaim_allocator_free_pages(
        self, state: _ModelState, count: int
    ) -> dict[str, Any]:
        """Compatibility path for older clients without shrink acknowledgement."""
        ranges = self._select_allocator_free_page_ranges(state, count)
        locations = self._page_range_locations(state, ranges)
        for segment, local_start, local_count in locations:
            segment.draining_pages.add_range(
                local_start, local_count, require_disjoint=True
            )
        return self._commit_live_page_reclaim(state, ranges)

    def _abort_live_page_drain(
        self, state: _ModelState, page_ranges: list[tuple[int, int]]
    ) -> dict[str, Any]:
        if not page_ranges:
            raise ValueError("Central I/O page drain abort requires at least one page range")
        ranges = self._validate_page_ranges(
            page_ranges, state.max_capacity // state.page_size
        )
        locations = self._page_range_locations(state, ranges)
        if not all(
            segment.draining_pages.contains(local_start, count)
            for segment, local_start, count in locations
        ):
            raise ValueError("Central I/O page drain abort references a non-draining page")
        for segment, local_start, count in locations:
            segment.draining_pages.remove_range(local_start, count, require_present=True)
        return {"page_ranges": [[start, count] for start, count in ranges]}

    def _take_extent(self, model_id: str) -> _ArenaExtent:
        """Lease one *entire* fixed ArenaExtent to ``model_id``.

        There is intentionally no ``(offset, length)`` allocator here.  A
        model may own many non-contiguous extents, but an extent is never split
        between models.  This makes owner transfer, secure scrubbing and later
        multi-extent DMA grouping unambiguous.
        """
        if not self._free_extent_ids:
            raise MemoryError("central host pool has no free ArenaExtent")
        extent_id = min(self._free_extent_ids)
        extent = self._arena_extents[extent_id]
        if extent.owner_model_id is not None or extent.state != "free":
            raise RuntimeError("Central I/O free ArenaExtent state is inconsistent")
        self._free_extent_ids.remove(extent_id)
        extent.owner_model_id = model_id
        extent.state = "leased"
        return extent

    def _mark_extent_scrubbing(self, extent_id: int, model_id: str) -> _ArenaExtent:
        extent = self._arena_extents[extent_id]
        if extent.owner_model_id != model_id or extent.state != "leased":
            raise RuntimeError("Central I/O extent owner/state changed during reclaim")
        extent.state = "scrubbing"
        return extent

    def _return_extent(self, extent_id: int, model_id: str) -> None:
        extent = self._arena_extents[extent_id]
        if extent.owner_model_id != model_id or extent.state not in {"leased", "scrubbing"}:
            raise RuntimeError("Central I/O extent owner/state changed before release")
        extent.owner_model_id = None
        extent.state = "free"
        self._free_extent_ids.add(extent_id)

    @staticmethod
    def _add_logical_range(
        ranges: list[tuple[int, int]], start: int, count: int
    ) -> None:
        """Insert a half-open logical interval and coalesce adjacent spans."""
        if count <= 0:
            return
        end = start + count
        merged: list[tuple[int, int]] = []
        inserted = False
        for current_start, current_end in ranges:
            if current_end < start:
                merged.append((current_start, current_end))
            elif end < current_start:
                if not inserted:
                    merged.append((start, end))
                    inserted = True
                merged.append((current_start, current_end))
            else:
                if current_start < end and start < current_end:
                    raise ValueError("Central I/O logical range overlap")
                start = min(start, current_start)
                end = max(end, current_end)
        if not inserted:
            merged.append((start, end))
        ranges[:] = merged

    @classmethod
    def _take_logical_range(cls, state: _ModelState, count: int) -> int:
        """Reserve up to ``count`` contiguous inactive logical slots."""
        for index, (start, end) in enumerate(state.inactive_ranges):
            available = end - start
            if available < count:
                continue
            if available == count:
                state.inactive_ranges.pop(index)
            else:
                state.inactive_ranges[index] = (start + count, end)
            return start
        raise MemoryError(
            f"model {state.model_id} has no inactive logical range of {count} slots"
        )

    @staticmethod
    def _largest_inactive_range(state: _ModelState) -> int:
        return max((end - start for start, end in state.inactive_ranges), default=0)

    @staticmethod
    def _token_bytes(state: _ModelState) -> int:
        import torch

        return (
            2
            * state.layer_count
            * state.head_count
            * state.head_dim
            * torch.empty((), dtype=state.dtype).element_size()
        )

    def _extent_slot_count(self, state: _ModelState) -> int:
        """Usable, page-aligned token slots of one fixed ArenaExtent."""
        slots = self.arena_extent_bytes // self._token_bytes(state)
        slots -= slots % state.page_size
        if slots <= 0:
            raise ValueError("Central I/O ArenaExtent cannot hold one SGLang KV page")
        return slots

    def _add_segment(self, state: _ModelState, logical_start: int, slot_count: int) -> None:
        import torch

        if slot_count <= 0:
            return
        if logical_start % state.page_size or slot_count % state.page_size:
            raise ValueError("Central I/O segments must be SGLang-page aligned")
        if logical_start < 0 or logical_start + slot_count > state.max_capacity:
            raise ValueError("Central I/O segment range outside model logical capacity")
        position = bisect.bisect_left(state.segment_starts, logical_start)
        if position and state.segment_starts[position - 1] + state.segments[
            state.segments_by_start[state.segment_starts[position - 1]]
        ].slot_count > logical_start:
            raise ValueError("Central I/O segment overlaps an active logical range")
        if position < len(state.segment_starts) and logical_start + slot_count > state.segment_starts[position]:
            raise ValueError("Central I/O segment overlaps an active logical range")

        expected_slots = self._extent_slot_count(state)
        if slot_count != expected_slots:
            raise ValueError(
                "Central I/O lease must contain one complete fixed ArenaExtent; "
                f"got {slot_count} slots, expected {expected_slots}"
            )
        byte_size = slot_count * self._token_bytes(state)
        extent = self._take_extent(state.model_id)
        try:
            host = torch.frombuffer(
                self._mapping,
                dtype=state.dtype,
                count=byte_size // torch.empty((), dtype=state.dtype).element_size(),
                offset=extent.byte_offset,
            ).view(
                2,
                slot_count,
                state.layer_count,
                state.head_count,
                state.head_dim,
            )
            segment_id = extent.extent_id
            state.segments[segment_id] = _HostSegment(
                segment_id=segment_id,
                byte_offset=extent.byte_offset,
                allocated_bytes=extent.byte_size,
                logical_start=logical_start,
                slot_count=slot_count,
                host_k=host[0],
                host_v=host[1],
            )
            state.segment_starts.insert(position, logical_start)
            state.segments_by_start[logical_start] = segment_id
            state.active_capacity += slot_count
        except Exception:
            self._return_extent(extent.extent_id, state.model_id)
            raise

    def _detach_segment(self, state: _ModelState, segment_id: int) -> _HostSegment:
        """Remove a clean whole extent from a model without freeing its bytes yet."""
        segment = state.segments[segment_id]
        if segment.reserved_slots or segment.reserved_pages:
            raise ValueError("cannot shrink a Central I/O segment with live KV slots")
        del state.segments[segment_id]
        position = bisect.bisect_left(state.segment_starts, segment.logical_start)
        if position == len(state.segment_starts) or state.segment_starts[position] != segment.logical_start:
            raise RuntimeError("Central I/O segment index is inconsistent")
        state.segment_starts.pop(position)
        del state.segments_by_start[segment.logical_start]
        state.active_capacity -= segment.slot_count
        self._add_logical_range(
            state.inactive_ranges, segment.logical_start, segment.slot_count
        )
        return segment

    def _drop_segment(self, state: _ModelState, segment_id: int) -> list[int]:
        """Synchronously return an already-clean segment to the physical pool."""
        segment = self._detach_segment(state, segment_id)
        self._return_extent(segment.segment_id, state.model_id)
        return [segment.logical_start, segment.slot_count]

    def _commit_live_reclaim(
        self, state: _ModelState, segment_ids: list[int]
    ) -> dict[str, Any]:
        """Detach clean whole extents and asynchronously scrub their old KV bytes.

        The caller is responsible for draining the selected live extent through
        HiRadix first.  This method is deliberately strict: a stale reservation
        is a correctness failure, never a cue to silently hand off a slice.
        """
        if not segment_ids:
            raise ValueError("Central I/O reclaim commit requires at least one extent")
        if len(set(segment_ids)) != len(segment_ids):
            raise ValueError("Central I/O reclaim commit contains a duplicate extent")
        try:
            selected = [state.segments[segment_id] for segment_id in segment_ids]
        except KeyError as error:
            raise ValueError("Central I/O reclaim commit references an unknown extent") from error
        for segment in selected:
            if segment.reserved_slots or segment.reserved_pages:
                raise ValueError("cannot commit live KV extent before HiRadix drain completes")

        detached = [self._detach_segment(state, segment_id) for segment_id in segment_ids]
        return self._scrub_detached_extents(state, detached)

    def _scrub_detached_extents(
        self, state: _ModelState, detached: list[_HostSegment]
    ) -> dict[str, Any]:
        """Securely scrub complete detached extents before global reuse.

        This is shared by live drain and already-free shrink. A host-cache
        page that is allocator-free can still contain an old tenant's KV bytes,
        so "clean" means safe to detach, not safe to reuse without scrubbing.
        """
        if not detached:
            raise ValueError("Central I/O scrub requires at least one detached extent")
        for segment in detached:
            self._mark_extent_scrubbing(segment.segment_id, state.model_id)
        segment_ids = [segment.segment_id for segment in detached]
        ranges = [[segment.logical_start, segment.slot_count] for segment in detached]
        byte_count = sum(segment.allocated_bytes for segment in detached)
        reclaim_id = getattr(self, "_next_reclaim_id", 0)
        self._next_reclaim_id = reclaim_id + 1
        reclaim = _PendingReclaim(
            reclaim_id=reclaim_id,
            ranges=ranges,
            byte_count=byte_count,
        )
        if not hasattr(self, "_reclaims"):
            self._reclaims = {}
        self._reclaims[reclaim_id] = reclaim

        def scrub_then_release() -> None:
            started = time.perf_counter()
            try:
                for segment in detached:
                    ctypes.memset(
                        self._address + segment.byte_offset,
                        0,
                        segment.allocated_bytes,
                    )
                with self._lock:
                    for segment in detached:
                        self._return_extent(segment.segment_id, state.model_id)
                    reclaim.scrub_ms = (time.perf_counter() - started) * 1000
                    reclaim.state = "ready"
                    print(
                        "[CENTRAL-IO] reclaim-ready "
                        f"id={reclaim_id} extents={segment_ids} "
                        f"bytes={byte_count} scrub_ms={reclaim.scrub_ms:.3f}",
                        flush=True,
                    )
            except Exception as error:  # Keep failed bytes unavailable instead of reusing them.
                reclaim.error = repr(error)
                reclaim.state = "failed"
            finally:
                reclaim.done.set()

        threading.Thread(
            target=scrub_then_release,
            name=f"central-io-scrub-{reclaim_id}",
            daemon=True,
        ).start()
        print(
            "[CENTRAL-IO] reclaim-start "
            f"id={reclaim_id} extents={segment_ids} bytes={byte_count}",
            flush=True,
        )
        return {
            "capacity": state.capacity,
            "ranges": ranges,
            "reclaim_id": reclaim_id,
            "state": reclaim.state,
            "bytes": byte_count,
        }

    def _reclaim_status(self, reclaim_id: int) -> dict[str, Any]:
        try:
            reclaim = self._reclaims[reclaim_id]
        except KeyError as error:
            raise ValueError("unknown Central I/O reclaim id") from error
        return {
            "reclaim_id": reclaim.reclaim_id,
            "state": reclaim.state,
            "ranges": reclaim.ranges,
            "bytes": reclaim.byte_count,
            "scrub_ms": reclaim.scrub_ms,
            "error": reclaim.error,
            "detached_ns": reclaim.detached_ns,
            "scrub_started_ns": reclaim.scrub_started_ns,
            "scrub_ready_ns": reclaim.scrub_ready_ns,
        }

    def _wait_for_reclaim(self, reclaim_id: int, timeout_s: float | None = None) -> dict[str, Any]:
        reclaim = self._reclaims[reclaim_id]
        if not reclaim.done.wait(timeout_s):
            raise TimeoutError("Central I/O reclaim scrub did not finish before timeout")
        status = self._reclaim_status(reclaim_id)
        if status["state"] != "ready":
            raise RuntimeError(f"Central I/O reclaim scrub failed: {status['error']}")
        return status

    @staticmethod
    def _segment_for_index(state: _ModelState, logical_index: int) -> tuple[_HostSegment, int]:
        position = bisect.bisect_right(state.segment_starts, logical_index) - 1
        if position < 0:
            raise ValueError("host index outside the model's active Central I/O quota")
        start = state.segment_starts[position]
        segment = state.segments[state.segments_by_start[start]]
        local_index = logical_index - start
        if local_index < 0 or local_index >= segment.slot_count:
            raise ValueError("host index outside the model's active Central I/O quota")
        return segment, local_index

    @classmethod
    def _segment_for_page(cls, state: _ModelState, page_id: int) -> tuple[_HostSegment, int]:
        """Translate a model-local KV page id to one active physical segment.

        A page may never straddle segments: registration and quota resize both
        require segment sizes to be multiples of the model's SGLang page size.
        """
        if page_id < 0 or page_id * state.page_size >= state.max_capacity:
            raise ValueError("host page outside the model's logical capacity")
        slot_index = page_id * state.page_size
        segment, local_slot = cls._segment_for_index(state, slot_index)
        if local_slot % state.page_size or local_slot + state.page_size > segment.slot_count:
            raise RuntimeError("Central I/O page crosses a physical segment boundary")
        return segment, local_slot // state.page_size

    @staticmethod
    def _validate_page_ranges(
        ranges: list[list[int]], page_limit: int
    ) -> list[tuple[int, int]]:
        """Validate disjoint page ranges without expanding them to page ids."""
        validated: list[tuple[int, int]] = []
        seen = _PageRangeSet()
        for encoded in ranges:
            if len(encoded) != 2:
                raise ValueError("Central I/O page range must contain [start, count]")
            start, count = int(encoded[0]), int(encoded[1])
            if start < 0 or count <= 0 or start + count > page_limit:
                raise ValueError("Central I/O page range outside model capacity")
            if seen.overlaps(start, count):
                raise ValueError("Central I/O page range contains duplicate page ids")
            seen.add_range(start, count, require_disjoint=True)
            validated.append((start, count))
        return validated

    @classmethod
    def _page_range_locations(
        cls, state: _ModelState, ranges: list[tuple[int, int]]
    ) -> list[tuple[_HostSegment, int, int]]:
        """Split logical PageRanges at existing host-view boundaries only.

        The resulting local ranges retain page granularity.  A range crossing
        two backing views yields two descriptors, never a list of page ids.
        """
        locations: list[tuple[_HostSegment, int, int]] = []
        for page_start, page_count in ranges:
            remaining = page_count
            page_id = page_start
            while remaining:
                segment, local_page = cls._segment_for_page(state, page_id)
                segment_page_count = segment.slot_count // state.page_size
                available = segment_page_count - local_page
                take = min(remaining, available)
                locations.append((segment, local_page, take))
                page_id += take
                remaining -= take
        return locations

    def _register(self, request: dict[str, Any]) -> dict[str, Any]:
        import torch

        model_id = request["model_id"]
        with self._lock:
            if model_id in self._models:
                raise ValueError(f"model id already registered: {model_id}")
            max_capacity = int(request["capacity"])
            initial_capacity = int(request.get("initial_capacity", max_capacity))
            if initial_capacity <= 0 or initial_capacity > max_capacity:
                raise ValueError("invalid Central I/O initial capacity")
            layer_count = int(request["layer_count"])
            head_count = int(request["head_count"])
            head_dim = int(request["head_dim"])
            device = int(request["device"])
            dtype_name = request["dtype"]
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[dtype_name]
            dtype_bytes = torch.empty((), dtype=dtype).element_size()
            page_size = int(request.get("page_size", 1))
            if page_size <= 0 or max_capacity % page_size:
                raise ValueError("Central I/O capacity must contain whole SGLang pages")
            if initial_capacity % page_size:
                raise ValueError("Central I/O initial capacity must be page aligned")
            lease_mode = str(request.get("lease_mode", "extent"))
            if lease_mode not in {"extent", "page"}:
                raise ValueError("Central I/O lease_mode must be 'extent' or 'page'")
            if lease_mode != self._configured_lease_mode:
                raise ValueError(
                    "Central I/O client lease mode must match the agent process mode"
                )
            if self._lease_mode is not None and self._lease_mode != lease_mode:
                raise ValueError(
                    "Central I/O agent cannot mix fixed-extent and dynamic-page lease modes"
                )

            _require(self.runtime, self.runtime.cudaSetDevice(device), f"cudaSetDevice({device})")
            opened_bases: dict[str, ctypes.c_void_p] = {}
            try:
                pointers = []
                for descriptor in request["k"] + request["v"]:
                    key = descriptor["handle"]
                    base = opened_bases.get(key)
                    if base is None:
                        base = ctypes.c_void_p()
                        _require(
                            self.runtime,
                            self.runtime.cudaIpcOpenMemHandle(
                                ctypes.byref(base), _decode_handle(key), IPC_LAZY_ENABLE_PEER_ACCESS
                            ),
                            "cudaIpcOpenMemHandle",
                        )
                        opened_bases[key] = base
                    pointers.append(base.value + int(descriptor["offset"]))
                k_ptrs = torch.tensor(pointers[:layer_count], dtype=torch.uint64, device=f"cuda:{device}")
                v_ptrs = torch.tensor(pointers[layer_count:], dtype=torch.uint64, device=f"cuda:{device}")
                state = _ModelState(
                    model_id=model_id,
                    device=device,
                    max_capacity=max_capacity,
                    # Retained only for compatibility with older diagnostics.
                    # Physical lease granularity is derived from the global
                    # fixed ArenaExtent and this model's KV geometry below.
                    segment_tokens=0,
                    k_ptrs=k_ptrs,
                    v_ptrs=v_ptrs,
                    opened_bases=opened_bases,
                    layer_count=layer_count,
                    head_count=head_count,
                    head_dim=head_dim,
                    dtype=dtype,
                    reserved=set(),
                    segments={},
                    segment_starts=[],
                    segments_by_start={},
                    active_capacity=0,
                    page_size=page_size,
                    inactive_ranges=[(0, max_capacity)],
                    dynamic_page_leases=lease_mode == "page",
                )
                requested_initial_bytes = int(
                    request.get("initial_bytes", initial_capacity * self._token_bytes(state))
                )
                if state.dynamic_page_leases:
                    if not hasattr(self, "_free_byte_ranges"):
                        self._init_dynamic_free_bytes(self.pool_bytes)
                    initial_slots = min(
                        initial_capacity,
                        requested_initial_bytes // self._token_bytes(state),
                    )
                    initial_slots -= initial_slots % page_size
                    if initial_slots <= 0:
                        raise ValueError(
                            "Central I/O dynamic initial quota must contain one SGLang KV page"
                        )
                    logical_start = self._take_logical_range(state, initial_slots)
                    self._add_dynamic_segment(state, logical_start, initial_slots)
                    state.segment_tokens = initial_slots
                else:
                    extent_slots = self._extent_slot_count(state)
                    state.segment_tokens = extent_slots
                    extent_count = requested_initial_bytes // self.arena_extent_bytes
                    if extent_count <= 0:
                        raise ValueError(
                            "Central I/O initial quota is smaller than one fixed ArenaExtent"
                        )
                    max_extent_count = (
                        max_capacity * self._token_bytes(state) // self.arena_extent_bytes
                    )
                    if extent_count > max_extent_count:
                        raise ValueError("Central I/O initial quota exceeds logical capacity")
                    if len(self._free_extent_ids) < extent_count:
                        raise MemoryError("central host pool has insufficient free ArenaExtents")
                    for _ in range(extent_count):
                        logical_start = self._take_logical_range(state, extent_slots)
                        self._add_segment(state, logical_start, extent_slots)
                self._models[model_id] = state
                self._lease_mode = lease_mode
            except Exception:
                for pointer in opened_bases.values():
                    self.runtime.cudaIpcCloseMemHandle(pointer)
                if "state" in locals():
                    for segment in state.segments.values():
                        if state.dynamic_page_leases:
                            self._free_byte_ranges.release(
                                segment.byte_offset, segment.allocated_bytes
                            )
                        else:
                            self._return_extent(segment.segment_id, state.model_id)
                raise
        print(
            f"central-io register model={model_id} initial_capacity={initial_capacity} "
            f"max_capacity={max_capacity} segments={len(state.segments)}",
            flush=True,
        )
        return {
            "model_id": model_id,
            "capacity": state.capacity,
            "max_capacity": state.max_capacity,
            "extent_slots": state.segment_tokens,
            "arena_extent_bytes": self.arena_extent_bytes,
            "lease_mode": lease_mode,
            "active_ranges": [
                [segment.logical_start, segment.slot_count]
                for segment in sorted(
                    state.segments.values(), key=lambda segment: segment.logical_start
                )
            ],
        }

    def _state(self, request: dict[str, Any]) -> _ModelState:
        try:
            return self._models[request["model_id"]]
        except KeyError as error:
            raise ValueError(f"unknown model id: {request['model_id']}") from error

    def _indices(
        self, request: dict[str, Any], state: _ModelState
    ) -> list[tuple[_HostSegment, Any, Any]]:
        import torch

        host_indices = request["host_indices"]
        device_indices = request["device_indices"]
        if len(host_indices) != len(device_indices):
            raise ValueError("host/device index counts differ")
        groups: dict[int, tuple[list[int], list[int]]] = {}
        for host_index, device_index in zip(host_indices, device_indices):
            segment, local_index = self._segment_for_index(state, host_index)
            segment_id = segment.segment_id
            local_indices, group_device_indices = groups.setdefault(segment_id, ([], []))
            local_indices.append(local_index)
            group_device_indices.append(device_index)
        return [
            (
                state.segments[segment_id],
                torch.tensor(local_indices, dtype=torch.int64, device=f"cuda:{state.device}"),
                torch.tensor(device_indices, dtype=torch.int64, device=f"cuda:{state.device}"),
            )
            for segment_id, (local_indices, device_indices) in groups.items()
        ]

    def _backup(self, request: dict[str, Any]) -> None:
        import torch
        from sgl_kernel.kvcacheio import transfer_kv_all_layer_lf_pf

        state = self._state(request)
        groups = self._indices(request, state)
        with self._cuda_lock:
            torch.cuda.set_device(state.device)
            for segment, host_indices, device_indices in groups:
                transfer_kv_all_layer_lf_pf(
                    src_k_layers=state.k_ptrs,
                    dst_k=segment.host_k,
                    src_v_layers=state.v_ptrs,
                    dst_v=segment.host_v,
                    src_indices=device_indices,
                    dst_indices=host_indices,
                    item_size=state.head_count * state.head_dim * torch.empty((), dtype=state.dtype).element_size(),
                    dst_layout_dim=state.layer_count * state.head_count * state.head_dim * torch.empty((), dtype=state.dtype).element_size(),
                    num_layers=state.layer_count,
                )
            torch.cuda.synchronize(state.device)
        self._backup_calls += 1
        self._backup_pages += len(request["host_indices"])
        print(
            f"central-io backup model={state.model_id} pages={len(request['host_indices'])} "
            f"calls={self._backup_calls} total_pages={self._backup_pages}",
            flush=True,
        )

    def _restore(self, request: dict[str, Any]) -> None:
        import torch

        state = self._state(request)
        groups = self._indices(request, state)
        with self._cuda_lock:
            torch.cuda.set_device(state.device)
            extension = _load_restore_extension()
            for segment, host_indices, device_indices in groups:
                extension.restore_page_first_mha_to_raw(
                    segment.host_k,
                    segment.host_v,
                    state.k_ptrs,
                    state.v_ptrs,
                    host_indices,
                    device_indices,
                    state.layer_count,
                    state.head_count,
                    state.head_dim,
                )
            torch.cuda.synchronize(state.device)
        self._restore_calls += 1
        self._restore_pages += len(request["host_indices"])
        print(
            f"central-io restore model={state.model_id} pages={len(request['host_indices'])} "
            f"calls={self._restore_calls} total_pages={self._restore_pages}",
            flush=True,
        )

    def _unregister(self, model_id: str) -> None:
        with self._lock:
            state = self._models.pop(model_id)
            for pointer in state.opened_bases.values():
                self.runtime.cudaIpcCloseMemHandle(pointer)
            for segment in state.segments.values():
                if state.dynamic_page_leases:
                    self._free_byte_ranges.release(
                        segment.byte_offset, segment.allocated_bytes
                    )
                else:
                    self._return_extent(segment.segment_id, state.model_id)

    @staticmethod
    def _record_grow_accept(state: _ModelState) -> int:
        accepted_ns = time.monotonic_ns()
        state.last_grow_accepted_ns = accepted_ns
        if state.first_grow_accepted_ns is None:
            state.first_grow_accepted_ns = accepted_ns
        state.grow_accept_count += 1
        return accepted_ns

    def _grow_range(self, state: _ModelState, count: int) -> dict[str, Any]:
        """Atomically add whole physical extents to one model's lease.

        A live reclaim releases cold extents asynchronously.  Hot-side grow
        retries can therefore arrive before enough scrubbed extents exist.  A
        failed retry must not leave a prefix of the requested quota attached:
        otherwise the agent and the model-side HostKV allocator disagree about
        capacity.  Plan logical ranges, acquire every physical extent, build
        all tensor views, then publish the batch in one state update.
        """
        import torch

        started = time.perf_counter()
        if count <= 0:
            return {"capacity": state.capacity, "ranges": []}
        if state.dynamic_page_leases:
            if count % state.page_size:
                raise ValueError("Central I/O dynamic grow must be page aligned")
            logical_start = self._take_logical_range(state, count)
            byte_count = count * self._token_bytes(state)
            allocations = self._free_byte_ranges.allocate_many(
                byte_count, alignment=self._page_bytes(state)
            )
            if allocations is None:
                self._add_logical_range(state.inactive_ranges, logical_start, count)
                raise MemoryError("central host pool has insufficient free page-aligned bytes")
            added: list[_HostSegment] = []
            try:
                cursor = logical_start
                for byte_offset, allocation_bytes in allocations:
                    slot_count = allocation_bytes // self._token_bytes(state)
                    added.append(
                        self._insert_dynamic_segment(
                            state,
                            logical_start=cursor,
                            slot_count=slot_count,
                            byte_offset=byte_offset,
                        )
                    )
                    cursor += slot_count
            except Exception:
                for segment in added:
                    self._remove_dynamic_segment(state, segment)
                for byte_offset, allocation_bytes in allocations:
                    self._free_byte_ranges.release(byte_offset, allocation_bytes)
                self._add_logical_range(state.inactive_ranges, logical_start, count)
                raise
            timing_ms = {
                "agent_total": (time.perf_counter() - started) * 1000,
                "agent_add_segments": (time.perf_counter() - started) * 1000,
                "segment_count": len(added),
            }
            grow_accepted_ns = self._record_grow_accept(state)
            return {
                "capacity": state.capacity,
                "ranges": [[segment.logical_start, segment.slot_count] for segment in added],
                "timing_ms": timing_ms,
                "grow_accepted_ns": grow_accepted_ns,
            }
        extent_slots = self._extent_slot_count(state)
        if count % extent_slots:
            raise ValueError(
                "Central I/O grow must be aligned to the model's ArenaExtent capacity"
            )
        validate_started = time.perf_counter()
        extent_count = count // extent_slots
        max_extent_count = (
            state.max_capacity * self._token_bytes(state) // self.arena_extent_bytes
        )
        if len(state.segments) + extent_count > max_extent_count:
            raise ValueError("Central I/O grow range outside logical capacity")
        validate_ms = (time.perf_counter() - validate_started) * 1000
        add_started = time.perf_counter()

        # First reserve only from local copies.  The live state stays exactly
        # unchanged until every requested extent can be materialized.
        inactive_ranges = list(state.inactive_ranges)
        planned: list[tuple[int, int]] = []
        remaining = count
        while remaining:
            largest = max((end - start for start, end in inactive_ranges), default=0)
            if not largest:
                raise MemoryError("Central I/O model has no inactive logical slots to grow")
            slot_count = extent_slots
            if largest < slot_count:
                raise MemoryError("Central I/O model has no extent-aligned logical range to grow")
            for range_index, (range_start, range_end) in enumerate(inactive_ranges):
                if range_end - range_start < slot_count:
                    continue
                logical_start = range_start
                if range_end - range_start == slot_count:
                    inactive_ranges.pop(range_index)
                else:
                    inactive_ranges[range_index] = (range_start + slot_count, range_end)
                break
            else:  # Defensive: ``largest`` above said a suitable range exists.
                raise RuntimeError("Central I/O inactive logical range plan is inconsistent")
            planned.append((logical_start, slot_count))
            remaining -= slot_count

        # Allocate every fixed physical extent before publishing the new
        # logical lease. A partial physical handoff is never legal.
        if len(self._free_extent_ids) < len(planned):
            raise MemoryError("central host pool has insufficient free ArenaExtents")
        added_segment_ids: list[int] = []
        try:
            for logical_start, slot_count in planned:
                self._add_segment(state, logical_start, slot_count)
                added_segment_ids.append(state.segments_by_start[logical_start])
        except Exception:
            for segment_id in added_segment_ids:
                self._drop_segment(state, segment_id)
            raise

        # All failure-prone work has completed. Publish the logical free-list
        # update as one batch; individual extents were already attached above.
        state.inactive_ranges[:] = inactive_ranges
        ranges = [[logical_start, slot_count] for logical_start, slot_count in planned]
        add_ms = (time.perf_counter() - add_started) * 1000
        timing_ms = {
            "agent_total": (time.perf_counter() - started) * 1000,
            "agent_validate": validate_ms,
            "agent_add_segments": add_ms,
            "segment_count": len(ranges),
        }
        print(
            f"central-io grow-range model={state.model_id} slots={count} capacity={state.capacity} "
            f"timing_ms={timing_ms}",
            flush=True,
        )
        grow_accepted_ns = self._record_grow_accept(state)
        return {
            "capacity": state.capacity,
            "ranges": ranges,
            "timing_ms": timing_ms,
            "grow_accepted_ns": grow_accepted_ns,
        }

    def _prepare_grow_range(self, state: _ModelState, count: int) -> dict[str, Any]:
        """Reserve a recipient range without yet increasing effective quota.

        The agent owns physical bytes, while the model process owns its
        allocator page map.  A newly allocated range is therefore pending
        between those two operations.  It cannot be reserved by backup or
        reported as usable capacity until the model acknowledges that its
        local page allocator has admitted the range.
        """
        if not state.dynamic_page_leases:
            raise ValueError("prepared grow requires dynamic page leases")
        if count <= 0 or count % state.page_size:
            raise ValueError("Central I/O prepared grow must be page aligned")
        if state.max_capacity - state.capacity < count:
            raise ValueError("Central I/O has insufficient inactive logical slots to grow")

        logical_start = self._take_logical_range(state, count)
        byte_count = count * self._token_bytes(state)
        allocations = self._free_byte_ranges.allocate_many(
            byte_count, alignment=self._page_bytes(state)
        )
        if allocations is None:
            self._add_logical_range(state.inactive_ranges, logical_start, count)
            raise MemoryError("central host pool has insufficient free page-aligned bytes")

        added: list[_HostSegment] = []
        try:
            cursor = logical_start
            for byte_offset, allocation_bytes in allocations:
                slot_count = allocation_bytes // self._token_bytes(state)
                added.append(
                    self._insert_dynamic_segment(
                        state,
                        logical_start=cursor,
                        slot_count=slot_count,
                        byte_offset=byte_offset,
                    )
                )
                cursor += slot_count
        except Exception:
            for segment in list(added):
                self._remove_dynamic_segment(state, segment)
            for byte_offset, allocation_bytes in allocations:
                self._free_byte_ranges.release(byte_offset, allocation_bytes)
            self._add_logical_range(state.inactive_ranges, logical_start, count)
            raise

        transfer_id = getattr(self, "_next_transfer_id", 0)
        self._next_transfer_id = transfer_id + 1
        for segment in added:
            segment.pending_transfer_id = transfer_id
        # ``_insert_dynamic_segment`` owns the physical range and builds the
        # page-first Tensor views, but effective capacity begins only at ack.
        state.active_capacity -= count
        transfer = _PendingGrow(
            transfer_id=transfer_id,
            model_id=state.model_id,
            segment_ids=tuple(segment.segment_id for segment in added),
            slot_count=count,
            ranges=tuple((segment.logical_start, segment.slot_count) for segment in added),
        )
        pending = getattr(self, "_pending_grows", None)
        if pending is None:
            pending = {}
            self._pending_grows = pending
        pending[transfer_id] = transfer
        return {
            "transfer_id": transfer_id,
            "ranges": [list(item) for item in transfer.ranges],
            "effective_capacity": state.capacity,
            "pending_capacity": count,
        }

    def _pending_grow(self, state: _ModelState, transfer_id: int) -> _PendingGrow:
        try:
            transfer = self._pending_grows[transfer_id]
        except (AttributeError, KeyError) as error:
            raise ValueError("unknown pending Central I/O grow") from error
        if transfer.model_id != state.model_id:
            raise ValueError("pending Central I/O grow belongs to another model")
        return transfer

    def _pending_capacity(self, state: _ModelState) -> int:
        return sum(
            transfer.slot_count
            for transfer in getattr(self, "_pending_grows", {}).values()
            if transfer.model_id == state.model_id
        )

    def _report_local_residency(
        self, state: _ModelState, request: dict[str, Any]
    ) -> dict[str, Any]:
        """Store one validated model-side health report for global scheduling."""
        page_keys = (
            "effective_pages",
            "floor_pages",
            "clean_pages",
            "ready_pages",
            "live_pages",
            "min_pages",
            "low_pages",
            "high_pages",
            "retention_debt_tokens",
            "unresolved_clean_pages",
        )
        try:
            report = {key: int(request[key]) for key in page_keys}
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("invalid Central I/O local residency report") from error
        if min(report.values()) < 0:
            raise ValueError("local residency report has a negative value")
        if report["effective_pages"] != state.capacity // state.page_size:
            raise ValueError("local residency report disagrees with effective quota")
        if report["floor_pages"] > report["effective_pages"]:
            raise ValueError("local residency floor exceeds effective quota")
        if report["clean_pages"] + report["live_pages"] != report["effective_pages"]:
            raise ValueError("local residency clean/live accounting is inconsistent")
        if report["ready_pages"] > report["live_pages"]:
            raise ValueError("local residency ready pages must still be live")
        if not report["min_pages"] <= report["low_pages"] <= report["high_pages"]:
            raise ValueError("local residency watermarks are inconsistent")
        action = str(request.get("action", "idle"))
        if action not in {"idle", "prepare", "emergency"}:
            raise ValueError("local residency action is invalid")
        # Watermarks already encode these measurements for the decision path,
        # but retaining the raw observations in the global snapshot makes the
        # mechanism auditable and lets later policies distinguish an ingress
        # burst from a slow local reclaim path.
        for key in ("backup_ingress_pages_per_s", "ready_latency_p95_s"):
            value = float(request.get(key, 0.0))
            if value < 0:
                raise ValueError(f"local residency {key} must be non-negative")
            report[key] = value
        ready_reclaim_score = float(request.get("ready_reclaim_score", 0.0))
        if ready_reclaim_score < 0:
            raise ValueError("local residency ready_reclaim_score must be non-negative")
        report["ready_reclaim_score"] = ready_reclaim_score
        report["action"] = action
        report["reported_ns"] = time.monotonic_ns()
        state.residency_report = report
        return {"model_id": state.model_id, "reported_ns": report["reported_ns"]}

    def _global_status(self) -> dict[str, Any]:
        """Return effective quotas plus each model's latest local health."""
        if hasattr(self, "_free_byte_ranges"):
            global_free_bytes = sum(
                end - start for start, end in self._free_byte_ranges.ranges
            )
        else:
            global_free_bytes = len(self._free_extent_ids) * self.arena_extent_bytes
        return {
            "global_free_bytes": global_free_bytes,
            "models": {
                state.model_id: {
                    "capacity": state.capacity,
                    "pending_capacity": self._pending_capacity(state),
                    "quota_target": state.quota_target,
                    "page_size": state.page_size,
                    "token_bytes": self._token_bytes(state),
                    "residency": state.residency_report,
                }
                for state in self._models.values()
            },
        }

    def _ack_prepared_grow(self, state: _ModelState, transfer_id: int) -> dict[str, Any]:
        """Publish a prepared grow only after recipient allocator admission."""
        transfer = self._pending_grow(state, transfer_id)
        segments = [state.segments.get(segment_id) for segment_id in transfer.segment_ids]
        if any(segment is None or segment.pending_transfer_id != transfer_id for segment in segments):
            raise RuntimeError("pending Central I/O grow mapping changed before acknowledgement")
        for segment in segments:
            assert segment is not None
            segment.pending_transfer_id = None
        state.active_capacity += transfer.slot_count
        del self._pending_grows[transfer_id]
        accepted_ns = self._record_grow_accept(state)
        return {
            "transfer_id": transfer_id,
            "ranges": [list(item) for item in transfer.ranges],
            "effective_capacity": state.capacity,
            "pending_capacity": 0,
            "grow_accepted_ns": accepted_ns,
        }

    def _abort_prepared_grow(self, state: _ModelState, transfer_id: int) -> dict[str, Any]:
        """Roll back a prepared grow whose recipient failed before admission."""
        transfer = self._pending_grow(state, transfer_id)
        segments = [state.segments.get(segment_id) for segment_id in transfer.segment_ids]
        if any(segment is None or segment.pending_transfer_id != transfer_id for segment in segments):
            raise RuntimeError("pending Central I/O grow mapping changed before abort")
        # `_remove_dynamic_segment` decrements active capacity. Temporarily
        # include the pending slots so rollback leaves effective capacity
        # unchanged.
        state.active_capacity += transfer.slot_count
        for segment in segments:
            assert segment is not None
            self._remove_dynamic_segment(state, segment)
            self._free_byte_ranges.release(segment.byte_offset, segment.allocated_bytes)
        for logical_start, slot_count in transfer.ranges:
            self._add_logical_range(state.inactive_ranges, logical_start, slot_count)
        del self._pending_grows[transfer_id]
        return {
            "transfer_id": transfer_id,
            "released_slots": transfer.slot_count,
            "effective_capacity": state.capacity,
            "pending_capacity": 0,
        }

    def _shrink_free(self, state: _ModelState, count: int) -> dict[str, Any]:
        started = time.perf_counter()
        if count <= 0:
            return {"capacity": state.capacity, "ranges": []}
        if state.dynamic_page_leases:
            reclaim = self._reclaim_allocator_free_pages(state, count)
            reclaim["timing_ms"] = {
                "agent_total": (time.perf_counter() - started) * 1000,
                "agent_select_free_pages": reclaim.get("timing_ms", {}).get(
                    "agent_select_free_pages", 0.0
                ),
            }
            return reclaim
        selected_count = 0
        selected_ranges: list[list[int]] = []
        segment_ids: list[int] = []
        select_started = time.perf_counter()
        for logical_start in reversed(state.segment_starts):
            segment = state.segments[state.segments_by_start[logical_start]]
            if segment.reserved_slots or segment.reserved_pages:
                continue
            selected_count += segment.slot_count
            selected_ranges.append([segment.logical_start, segment.slot_count])
            segment_ids.append(segment.segment_id)
            if selected_count >= count:
                break
        select_ms = (time.perf_counter() - select_started) * 1000
        if selected_count < count:
            raise ValueError(
                f"cannot shrink {count} free slots without touching live KV; found {selected_count}"
            )
        drop_started = time.perf_counter()
        detached = [self._detach_segment(state, segment_id) for segment_id in segment_ids]
        reclaim = self._scrub_detached_extents(state, detached)
        drop_ms = (time.perf_counter() - drop_started) * 1000
        timing_ms = {
            "agent_total": (time.perf_counter() - started) * 1000,
            "agent_select_free_segments": select_ms,
            "agent_drop_segments": drop_ms,
            "segment_count": len(segment_ids),
        }
        print(
            f"central-io shrink-range model={state.model_id} pages={selected_count} capacity={state.capacity} "
            f"timing_ms={timing_ms}",
            flush=True,
        )
        return {
            "capacity": state.capacity,
            "ranges": selected_ranges,
            "reclaim_id": reclaim["reclaim_id"],
            "state": reclaim["state"],
            "timing_ms": timing_ms,
        }

    def _describe_layout(self, state: _ModelState) -> dict[str, Any]:
        """Copy this model's physical lease metadata without exposing bytes."""
        segments = []
        for segment in sorted(state.segments.values(), key=lambda item: item.logical_start):
            segments.append(
                {
                    "extent_id": segment.segment_id,
                    "logical_start": segment.logical_start,
                    "slot_count": segment.slot_count,
                    "byte_offset": segment.byte_offset,
                    "byte_size": segment.allocated_bytes,
                    "draining": segment.draining,
                    # These are local page offsets within this extent. The
                    # snapshot exporter converts them to logical page ids.
                    "reserved_page_ranges": segment.reserved_pages.copy_ranges(),
                    "dirty_page_ranges": segment.dirty_pages.copy_ranges(),
                    "draining_page_ranges": segment.draining_pages.copy_ranges(),
                }
            )
        return {
            "model_id": state.model_id,
            "page_size": state.page_size,
            "segments": segments,
        }

    def _plan_live_reclaim(
        self,
        state: _ModelState,
        count: int,
        excluded_segment_ids: set[int] | None = None,
        max_segments: int | None = None,
    ) -> dict[str, Any]:
        """Choose whole extents for a later scheduler-side live-KV drain.

        This method deliberately does *not* alter a lease.  A Central I/O
        segment is the V1 ownership and tensor-view unit, so handing an
        interior ``PhysicalSlice`` to another model would invalidate the
        existing ``logical_start -> segment`` mapping.  The runtime first
        drains every live page in the chosen complete segment, then commits a
        normal ``_drop_segment``.

        The agent can safely rank only physical facts: already-clean extents
        cost nothing; among live extents, fewer live pages cost less to drain.
        A later HiRadix pass must still reject extents whose pages belong to a
        protected/in-flight or non-leaf host node.  Physical adjacency is a
        tie-breaker so completed handoffs tend to coalesce global free memory.
        """
        if count <= 0:
            return {
                "requested_slots": 0,
                "selected_segment_ids": [],
                "clean_segment_ids": [],
                "drain_segment_ids": [],
                "segments": [],
            }

        # ``count`` is the remaining distance to the final quota target, not
        # the size of this particular drain batch.  Check it against the full
        # active lease first.  The scheduler may temporarily exclude extents
        # with protected radix leaves; treating that filtered candidate set as
        # the active lease caused a false "exceeds active capacity" failure
        # after one or more asynchronous batches had already completed.
        all_segments = list(state.segments.values())
        if sum(segment.slot_count for segment in all_segments) < count:
            raise ValueError("Central I/O reclaim request exceeds active capacity")

        excluded_segment_ids = excluded_segment_ids or set()
        segments = [
            segment
            for segment in all_segments
            if segment.segment_id not in excluded_segment_ids
        ]
        draining = [segment for segment in segments if segment.draining]
        if draining:
            # Once a drain begins, keep its physical candidates stable.  A
            # later plan must not chase a different set of extents while the
            # old set is admission-fenced and being peeled.
            selected = sorted(draining, key=lambda segment: segment.segment_id)
            return self._encode_live_reclaim_plan(state, selected, count)
        if not segments:
            # All extents can be temporarily excluded by the scheduler's
            # radix-safety pass.  This is a normal live-cache state, not a
            # malformed quota request: report it so the scheduler can back
            # off and retry after the working set changes.
            return {
                "requested_slots": count,
                "selected_segment_ids": [],
                "clean_segment_ids": [],
                "drain_segment_ids": [],
                "segments": [],
                "blocked_reason": "no_eligible_extents",
            }
        max_segments = len(segments) if max_segments is None else int(max_segments)
        if max_segments <= 0:
            raise ValueError("Central I/O reclaim batch must contain at least one extent")

        def is_clean(segment: _HostSegment) -> bool:
            return not segment.reserved_slots and not segment.reserved_pages

        def physical_gap(segment: _HostSegment, selected: list[_HostSegment]) -> int:
            if not selected:
                return 0
            start = segment.byte_offset
            end = start + segment.allocated_bytes
            return min(
                max(
                    0,
                    max(start, other.byte_offset)
                    - min(end, other.byte_offset + other.allocated_bytes),
                )
                for other in selected
            )

        selected: list[_HostSegment] = []
        remaining = count

        # Consume clean extents first.  They can be detached immediately and
        # require neither radix work nor KV scrubbing.
        clean = sorted(
            (segment for segment in segments if is_clean(segment)),
            key=lambda segment: (segment.byte_offset, segment.segment_id),
        )
        while remaining > 0 and clean and len(selected) < max_segments:
            segment = min(
                clean,
                key=lambda candidate: (
                    physical_gap(candidate, selected),
                    candidate.byte_offset,
                    candidate.segment_id,
                ),
            )
            clean.remove(segment)
            selected.append(segment)
            remaining -= segment.slot_count

        # A live extent is only a *drain candidate*.  It remains owned by its
        # current model until HiRadix has evicted every safe host leaf in it.
        live = [segment for segment in segments if not is_clean(segment)]
        while remaining > 0 and live and len(selected) < max_segments:
            segment = min(
                live,
                key=lambda candidate: (
                    len(candidate.reserved_pages) + len(candidate.reserved_slots),
                    physical_gap(candidate, selected),
                    candidate.byte_offset,
                    candidate.segment_id,
                ),
            )
            live.remove(segment)
            selected.append(segment)
            remaining -= segment.slot_count

        # A bounded batch is allowed to cover only part of ``count``.  This
        # is essential for live KV: later scheduler iterations can drain the
        # next safe extents without turning a temporary safety exclusion into
        # a quota failure.  Only an empty selection is a true lack of progress.
        if not selected:
            return {
                "requested_slots": count,
                "selected_segment_ids": [],
                "clean_segment_ids": [],
                "drain_segment_ids": [],
                "segments": [],
                "blocked_reason": "no_eligible_extents",
            }

        return self._encode_live_reclaim_plan(state, selected, count)

    def _encode_live_reclaim_plan(
        self, state: _ModelState, selected: list[_HostSegment], count: int
    ) -> dict[str, Any]:
        selected_ids = [segment.segment_id for segment in selected]
        clean_ids = [segment.segment_id for segment in selected if not segment.reserved_slots and not segment.reserved_pages]
        drain_ids = [segment.segment_id for segment in selected if segment.segment_id not in clean_ids]
        return {
            "requested_slots": count,
            "selected_segment_ids": selected_ids,
            "clean_segment_ids": clean_ids,
            "drain_segment_ids": drain_ids,
            "segments": [
                {
                    "segment_id": segment.segment_id,
                    "logical_start": segment.logical_start,
                    "slot_count": segment.slot_count,
                    "live_pages": len(segment.reserved_pages),
                    "live_page_ranges": [
                        [segment.logical_start // state.page_size + start, count]
                        for start, count in segment.reserved_pages.ranges
                    ],
                }
                for segment in selected
            ],
        }

    @staticmethod
    def _drain_segment_descriptor(state: _ModelState, segment: _HostSegment) -> dict[str, Any]:
        return {
            "segment_id": segment.segment_id,
            "logical_start": segment.logical_start,
            "slot_count": segment.slot_count,
            "live_page_ranges": [
                [segment.logical_start // state.page_size + start, count]
                for start, count in segment.reserved_pages.ranges
            ],
        }

    def _begin_live_drain(
        self, state: _ModelState, segment_ids: list[int]
    ) -> dict[str, Any]:
        if not segment_ids or len(set(segment_ids)) != len(segment_ids):
            raise ValueError("Central I/O drain requires unique non-empty extent ids")
        try:
            selected = [state.segments[segment_id] for segment_id in segment_ids]
        except KeyError as error:
            raise ValueError("Central I/O drain references an unknown extent") from error
        for segment in selected:
            segment.draining = True
        return {
            "segment_ids": [segment.segment_id for segment in selected],
            "segments": [self._drain_segment_descriptor(state, segment) for segment in selected],
        }

    def _abort_live_drain(
        self, state: _ModelState, segment_ids: list[int]
    ) -> dict[str, Any]:
        if not segment_ids or len(set(segment_ids)) != len(segment_ids):
            raise ValueError("Central I/O drain abort requires unique non-empty extent ids")
        try:
            selected = [state.segments[segment_id] for segment_id in segment_ids]
        except KeyError as error:
            raise ValueError("Central I/O drain abort references an unknown extent") from error
        for segment in selected:
            segment.draining = False
        return {
            "segment_ids": [segment.segment_id for segment in selected],
            "segments": [self._drain_segment_descriptor(state, segment) for segment in selected],
        }

    def _handle(self, request: dict[str, Any]) -> dict[str, Any]:
        op = request.get("op")
        if op == "register":
            return self._register(request)
        if op == "global_status":
            with self._lock:
                return self._global_status()
        if op == "set_quota_targets":
            raw_targets = request.get("targets")
            if not isinstance(raw_targets, dict) or not raw_targets:
                raise ValueError("Central I/O bulk quota target requires a non-empty targets map")

            # Validate every target before changing any model.  A scheduler
            # decision must be all-or-nothing: publishing the donor first and
            # the recipient later creates an artificial handoff gap.
            prepared: list[tuple[_ModelState, int]] = []
            for model_id, raw_target in raw_targets.items():
                if not isinstance(model_id, str):
                    raise ValueError("Central I/O bulk quota target model id must be a string")
                try:
                    state = self._models[model_id]
                except KeyError as error:
                    raise ValueError(f"Central I/O has no registered model {model_id!r}") from error
                if not state.dynamic_page_leases:
                    raise ValueError("quota target requires dynamic page leases")
                target = int(raw_target)
                if target <= 0 or target > state.max_capacity:
                    raise ValueError("Central I/O quota target is outside model capacity")
                if target % state.page_size:
                    raise ValueError("Central I/O quota target must be page aligned")
                prepared.append((state, target))

            target_set_ns = time.monotonic_ns()
            for state, target in prepared:
                state.quota_target = target
                state.quota_target_set_ns = target_set_ns
            return {
                "target_set_ns": target_set_ns,
                "targets": {
                    state.model_id: {
                        "target_capacity": target,
                        "capacity": state.capacity,
                    }
                    for state, target in prepared
                },
            }
        state = self._state(request)
        if op == "report_local_residency":
            with self._lock:
                return self._report_local_residency(state, request)
        if op == "set_quota_target":
            if not state.dynamic_page_leases:
                raise ValueError("quota target requires dynamic page leases")
            target = int(request["target_capacity"])
            if target <= 0 or target > state.max_capacity:
                raise ValueError("Central I/O quota target is outside model capacity")
            if target % state.page_size:
                raise ValueError("Central I/O quota target must be page aligned")
            state.quota_target = target
            state.quota_target_set_ns = time.monotonic_ns()
            return {
                "target_capacity": target,
                "capacity": state.capacity,
                "quota_target_set_ns": state.quota_target_set_ns,
            }
        if op == "quota_target":
            return {
                "target_capacity": state.quota_target,
                "capacity": state.capacity,
                "quota_target_set_ns": state.quota_target_set_ns,
            }
        if op == "reserve":
            indices = request["indices"]
            if state.reserved.intersection(indices):
                raise ValueError("host slot reserved twice")
            locations = [self._segment_for_index(state, index) for index in indices]
            for index, (segment, local_index) in zip(indices, locations):
                state.reserved.add(index)
                segment.reserved_slots.add(local_index)
            return {}
        if op == "release":
            indices = set(request["indices"])
            if not indices.issubset(state.reserved):
                raise ValueError("releasing an unreserved host slot")
            for index in indices:
                segment, local_index = self._segment_for_index(state, index)
                state.reserved.remove(index)
                segment.reserved_slots.remove(local_index)
            return {}
        if op == "reserve_pages":
            ranges = self._validate_page_ranges(
                request["ranges"], state.max_capacity // state.page_size
            )
            if any(state.reserved_pages.overlaps(start, count) for start, count in ranges):
                raise ValueError("host page reserved twice")
            locations = self._page_range_locations(state, ranges)
            if any(
                segment.draining
                or segment.pending_transfer_id is not None
                or segment.draining_pages.overlaps(local_start, count)
                for segment, local_start, count in locations
            ):
                raise ValueError("cannot reserve host page outside effective Central I/O quota")
            # All validation happens before the first mutation, preserving
            # atomicity for a multi-range allocator request.
            state.reserved_pages.add_ranges(ranges, require_disjoint=True)
            for segment, local_start, count in locations:
                segment.reserved_pages.add_range(local_start, count, require_disjoint=True)
                # Re-admission overwrites stale bytes, so it makes the
                # released range non-dirty from the current owner's view.
                segment.dirty_pages.remove_range(local_start, count, require_present=False)
            return {}
        if op == "release_pages":
            ranges = self._validate_page_ranges(
                request["ranges"], state.max_capacity // state.page_size
            )
            # A live-drain can detach a page while the native HiCache eviction
            # queue still contains its old radix leaf.  The later ordinary
            # ``free`` is therefore an acknowledgement, not a second owner
            # transition.  Keep the protocol strict by default; only the
            # runtime's explicitly marked post-drain free is idempotent.
            present = state.reserved_pages.intersection_ranges(ranges)
            if not present and not request.get("allow_already_released", False):
                raise ValueError("releasing an unreserved host page")
            if not present:
                return {"released_pages": 0, "already_released": True}
            locations = self._page_range_locations(state, present)
            state.reserved_pages.remove_ranges(present, require_present=True)
            for segment, local_start, count in locations:
                segment.reserved_pages.remove_range(local_start, count, require_present=True)
                segment.dirty_pages.add_range(local_start, count)
            return {
                "released_pages": sum(count for _, count in present),
                "already_released": False,
            }
        if op == "describe_layout":
            with self._lock:
                return self._describe_layout(state)
        if op == "backup":
            self._backup(request)
            return {}
        if op == "restore":
            self._restore(request)
            return {}
        if op == "grow_range":
            with self._lock:
                return self._grow_range(state, int(request["count"]))
        if op == "prepare_grow":
            with self._lock:
                return self._prepare_grow_range(state, int(request["count"]))
        if op == "ack_prepared_grow":
            with self._lock:
                return self._ack_prepared_grow(state, int(request["transfer_id"]))
        if op == "abort_prepared_grow":
            with self._lock:
                return self._abort_prepared_grow(state, int(request["transfer_id"]))
        if op == "prepare_shrink_free":
            with self._lock:
                return self._prepare_shrink_free(state, int(request["count"]))
        if op == "ack_prepared_shrink":
            with self._lock:
                return self._ack_prepared_shrink(state, int(request["transfer_id"]))
        if op == "abort_prepared_shrink":
            with self._lock:
                return self._abort_prepared_shrink(state, int(request["transfer_id"]))
        if op == "shrink_free":
            with self._lock:
                return self._shrink_free(state, int(request["count"]))
        if op == "plan_live_reclaim":
            with self._lock:
                # Quota changes are asynchronous: a prior scrub may have
                # detached extents after the model last observed its capacity.
                # Derive the reclaim count here, at the owner of lease state.
                if "target_capacity" in request:
                    target_capacity = int(request["target_capacity"])
                    if target_capacity < 0:
                        raise ValueError("Central I/O reclaim target must be non-negative")
                    remaining_slots = max(0, state.capacity - target_capacity)
                else:
                    # Compatibility with the pre-target protocol for isolated
                    # tests and older model processes.
                    remaining_slots = int(request["count"])
                    target_capacity = state.capacity - remaining_slots
                plan = self._plan_live_reclaim(
                    state,
                    remaining_slots,
                    {int(segment_id) for segment_id in request.get("excluded_segment_ids", [])},
                    request.get("max_segments"),
                )
                plan["active_capacity"] = state.capacity
                plan["target_capacity"] = target_capacity
                plan["remaining_slots"] = remaining_slots
                return plan
        if op == "begin_live_drain":
            with self._lock:
                return self._begin_live_drain(
                    state, [int(segment_id) for segment_id in request["segment_ids"]]
                )
        if op == "begin_live_page_drain":
            with self._lock:
                if not state.dynamic_page_leases:
                    raise ValueError("page drain requires dynamic page leases")
                return self._begin_live_page_drain(
                    state, [tuple(item) for item in request["ranges"]]
                )
        if op == "abort_live_drain":
            with self._lock:
                return self._abort_live_drain(
                    state, [int(segment_id) for segment_id in request["segment_ids"]]
                )
        if op == "abort_live_page_drain":
            with self._lock:
                if not state.dynamic_page_leases:
                    raise ValueError("page drain requires dynamic page leases")
                return self._abort_live_page_drain(
                    state, [tuple(item) for item in request["ranges"]]
                )
        if op == "commit_live_reclaim":
            with self._lock:
                return self._commit_live_reclaim(
                    state, [int(segment_id) for segment_id in request["segment_ids"]]
                )
        if op == "commit_live_page_reclaim":
            with self._lock:
                if not state.dynamic_page_leases:
                    raise ValueError("page reclaim requires dynamic page leases")
                return self._commit_live_page_reclaim(
                    state, [tuple(item) for item in request["ranges"]]
                )
        if op == "reclaim_status":
            return self._reclaim_status(int(request["reclaim_id"]))
        if op == "status":
            return {
                "capacity": state.capacity,
                "pending_capacity": self._pending_capacity(state),
                "max_capacity": state.max_capacity,
                "extent_slots": state.segment_tokens,
                "extent_count": len(state.segments),
                "physical_bytes": len(state.segments) * self.arena_extent_bytes,
                "reserved": len(state.reserved) + len(state.reserved_pages) * state.page_size,
                "reserved_pages": len(state.reserved_pages),
                "segments": len(state.segments),
                "quota_target": state.quota_target,
                "quota_target_set_ns": state.quota_target_set_ns,
                "last_grow_accepted_ns": state.last_grow_accepted_ns,
                "first_grow_accepted_ns": state.first_grow_accepted_ns,
                "grow_accept_count": state.grow_accept_count,
                "last_reclaim_id": state.last_reclaim_id,
                "token_bytes": self._token_bytes(state),
                "page_size": state.page_size,
                "clean_epoch": getattr(self, "_clean_epoch", 0),
            }
        if op == "reset_reservations":
            state.reserved.clear()
            for segment in state.segments.values():
                segment.dirty_pages.add_ranges(segment.reserved_pages.ranges)
            state.reserved_pages.clear()
            for segment in state.segments.values():
                segment.reserved_slots.clear()
                segment.reserved_pages.clear()
            return {}
        if op == "unregister":
            self._unregister(state.model_id)
            return {}
        raise ValueError(f"unsupported Central I/O operation: {op}")

    def _serve_connection(self, connection: Any) -> None:
        trace_rpc = os.getenv("SGLANG_CENTRAL_IO_TRACE_RPC", "0") == "1"
        try:
            while True:
                request = connection.recv()
                started = time.perf_counter()
                op = request.get("op")
                model_id = request.get("model_id", "-")
                if trace_rpc:
                    print(
                        f"[CENTRAL-IO] rpc-start op={op} model={model_id}",
                        flush=True,
                    )
                try:
                    response = self._handle(request)
                    if trace_rpc:
                        elapsed_ms = (time.perf_counter() - started) * 1000
                        print(
                            f"[CENTRAL-IO] rpc-done op={op} model={model_id} "
                            f"elapsed_ms={elapsed_ms:.3f}",
                            flush=True,
                        )
                    connection.send({"ok": True, **response})
                except Exception as error:
                    print(
                        "central-io request failed "
                        f"op={request.get('op')} model={request.get('model_id')} error={error!r}",
                        flush=True,
                    )
                    connection.send({"ok": False, "error": repr(error)})
        except EOFError:
            return
        finally:
            connection.close()

    def serve_forever(self) -> None:
        socket_file = Path(self.socket_path)
        if socket_file.exists():
            socket_file.unlink()
        self._listener = Listener(self.socket_path, family="AF_UNIX", authkey=IPC_AUTHKEY)
        while True:
            connection = self._listener.accept()
            threading.Thread(target=self._serve_connection, args=(connection,), daemon=True).start()

    def close(self) -> None:
        if self._registered:
            self.runtime.cudaHostUnregister(ctypes.c_void_p(self._address))
            self._registered = False
        self._mapping.close()
        os.close(self._fd)


def _load_restore_extension() -> Any:
    from torch.utils.cpp_extension import load

    return load(
        name="sglang_central_io_restore_v1",
        sources=[str(Path(__file__).with_name("central_io_restore.cu"))],
        extra_cuda_cflags=["-O2"],
        verbose=False,
    )


def build_registration(
    device_pool: Any,
    capacity: int,
    model_id: str,
    *,
    initial_capacity: int | None = None,
    initial_bytes: int | None = None,
    segment_tokens: int | None = None,
    page_size: int | None = None,
) -> dict[str, Any]:
    """Construct a persistent CUDA IPC endpoint descriptor for an MHA pool."""
    import torch

    if device_pool.store_dtype == torch.float16:
        dtype = "float16"
    elif device_pool.store_dtype == torch.bfloat16:
        dtype = "bfloat16"
    else:
        raise ValueError(f"Central I/O only supports fp16/bf16 MHA KV, got {device_pool.store_dtype}")
    device = torch.device(device_pool.device)
    # A model process may expose only one GPU through CUDA_VISIBLE_DEVICES,
    # while the node-level agent sees all GPUs.  This value identifies the
    # agent-visible physical device used to open this model's IPC handles.
    agent_device = int(os.getenv("SGLANG_CENTRAL_IO_AGENT_DEVICE_ID", device.index or 0))
    runtime = _load_cudart()
    _require(runtime, runtime.cudaSetDevice(device.index or 0), "cudaSetDevice")
    tensors = device_pool.k_buffer + device_pool.v_buffer
    registration = {
        "model_id": model_id,
        "device": agent_device,
        "capacity": capacity,
        "layer_count": device_pool.layer_num,
        "head_count": device_pool.head_num,
        "head_dim": device_pool.head_dim,
        "dtype": dtype,
        "k": _tensor_descriptors(device_pool.k_buffer, runtime),
        "v": _tensor_descriptors(device_pool.v_buffer, runtime),
        "lease_mode": os.getenv("SGLANG_CENTRAL_IO_LEASE_MODE", "extent"),
    }
    if initial_capacity is not None:
        registration["initial_capacity"] = initial_capacity
    if initial_bytes is not None:
        registration["initial_bytes"] = initial_bytes
    if segment_tokens is not None:
        registration["segment_tokens"] = segment_tokens
    if page_size is not None:
        registration["page_size"] = page_size
    return registration


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="SGLang Central I/O host KV agent")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--pool-bytes", required=True, type=int)
    args = parser.parse_args()
    agent = CentralIOAgent(args.socket, args.pool_bytes)
    try:
        agent.serve_forever()
    finally:
        agent.close()


if __name__ == "__main__":
    main()
