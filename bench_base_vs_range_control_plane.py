"""Compare SGLang's local HostKV allocator with current Central I/O ranges.

No CUDA transfer occurs here.  Both paths start from the same Qwen3-8B-sized
50GiB logical HostKV state.  We measure a minimum page allocation and a full
256MiB physical-extent batch separately.
"""

from __future__ import annotations

import json
import pickle
import statistics
import threading
import time

import torch

from sglang.srt.mem_cache import central_io as CENTRAL
from sglang.srt.mem_cache.memory_pool_host import (
    HostKVCache,
    _FreePageRanges,
    _FreeSlotRanges,
)


GIB = 1024**3
PAGE_TOKENS = 16
LAYERS = 36
HEADS = 8
HEAD_DIM = 128
SLOT_BYTES = 2 * LAYERS * HEADS * HEAD_DIM * 2
EXTENT_PAGES = 113
EXTENT_SLOTS = EXTENT_PAGES * PAGE_TOKENS
TOTAL_GIB = 50
ROUNDS = 15
FULL_POOL_ROUNDS = 5


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return sorted, non-overlapping half-open page ranges."""
    merged: list[tuple[int, int]] = []
    for start, count in sorted(ranges):
        end = start + count
        if not merged or start > merged[-1][0] + merged[-1][1]:
            merged.append((start, count))
            continue
        previous_start, previous_count = merged[-1]
        merged[-1] = (previous_start, max(previous_start + previous_count, end) - previous_start)
    return merged


def _subtract_ranges(
    source: list[tuple[int, int]], removed: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Subtract page ranges without expanding either side to page ids."""
    result = source
    for remove_start, remove_count in removed:
        remove_end = remove_start + remove_count
        updated: list[tuple[int, int]] = []
        for start, count in result:
            end = start + count
            if remove_end <= start or end <= remove_start:
                updated.append((start, count))
                continue
            if start < remove_start:
                updated.append((start, remove_start - start))
            if remove_end < end:
                updated.append((remove_end, end - remove_end))
        result = updated
    return result


class PageRangeLeaseState:
    """Candidate agent metadata: page ownership as compressed ranges, not sets."""

    def __init__(self) -> None:
        self.reserved: list[tuple[int, int]] = []
        self.dirty: list[tuple[int, int]] = []

    def reserve(self, ranges: list[tuple[int, int]]) -> None:
        for start, count in ranges:
            if count <= 0:
                raise ValueError("page range must be non-empty")
            end = start + count
            if any(existing < end and start < existing + length for existing, length in self.reserved):
                raise ValueError("page reserved twice")
        self.reserved = _merge_ranges(self.reserved + ranges)
        self.dirty = _subtract_ranges(self.dirty, ranges)

    def release(self, ranges: list[tuple[int, int]]) -> None:
        self.reserved = _subtract_ranges(self.reserved, ranges)
        self.dirty = _merge_ranges(self.dirty + ranges)


class BaseAllocatorShim(HostKVCache):
    """Concrete shell used solely to call HostKVCache's real alloc/free bodies."""

    def get_size_per_token(self):
        raise NotImplementedError

    def init_kv_buffer(self):
        raise NotImplementedError

    def load_to_device_per_layer(self, *args, **kwargs):
        raise NotImplementedError

    def backup_from_device_all_layer(self, *args, **kwargs):
        raise NotImplementedError

    def get_data_page(self, *args, **kwargs):
        raise NotImplementedError

    def get_dummy_flat_data_page(self, *args, **kwargs):
        raise NotImplementedError

    def set_from_flat_data_page(self, *args, **kwargs):
        raise NotImplementedError


def median_ms(operation, rounds: int = ROUNDS) -> float:
    samples = []
    for _ in range(rounds):
        started = time.perf_counter()
        operation()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def median_with_setup(setup, operation, rounds: int = ROUNDS) -> float:
    """Construct equal state outside the timed operation for each sample."""
    samples = []
    for _ in range(rounds):
        state = setup()
        started = time.perf_counter()
        operation(state)
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def make_base(total_slots: int):
    pool = BaseAllocatorShim.__new__(BaseAllocatorShim)
    pool.page_size = PAGE_TOKENS
    pool.lock = threading.RLock()
    pool.free_slots = torch.arange(total_slots, dtype=torch.int64)
    return pool


def make_agent_state(total_slots: int):
    agent = CENTRAL.CentralIOAgent.__new__(CENTRAL.CentralIOAgent)
    agent._models = {}
    agent._free_ranges = []
    state = CENTRAL._ModelState(
        model_id="cold",
        device=0,
        max_capacity=total_slots,
        segment_tokens=EXTENT_SLOTS,
        k_ptrs=None,
        v_ptrs=None,
        opened_bases={},
        layer_count=LAYERS,
        head_count=HEADS,
        head_dim=HEAD_DIM,
        dtype=torch.float16,
        reserved=set(),
        segments={},
        segment_starts=[],
        segments_by_start={},
        active_capacity=total_slots,
        page_size=PAGE_TOKENS,
    )
    extent_count = total_slots // EXTENT_SLOTS
    for segment_id in range(extent_count):
        start = segment_id * EXTENT_SLOTS
        state.segments[segment_id] = CENTRAL._HostSegment(
            segment_id=segment_id,
            byte_offset=start * SLOT_BYTES,
            allocated_bytes=EXTENT_SLOTS * SLOT_BYTES,
            logical_start=start,
            slot_count=EXTENT_SLOTS,
            host_k=None,
            host_v=None,
        )
        state.segment_starts.append(start)
        state.segments_by_start[start] = segment_id
    state.next_segment_id = extent_count
    agent._models[state.model_id] = state
    return agent, state


def page_ranges_to_slot_indices(
    page_ranges: list[tuple[int, int]], page_size: int
) -> torch.Tensor:
    """Expand only at the legacy SGLang HostKVCache API boundary."""
    pieces = [
        torch.arange(
            page_start * page_size,
            (page_start + page_count) * page_size,
            dtype=torch.int64,
        )
        for page_start, page_count in page_ranges
    ]
    return pieces[0] if len(pieces) == 1 else torch.cat(pieces)


def slot_indices_to_page_ranges(
    indices: torch.Tensor, page_size: int
) -> list[tuple[int, int]]:
    """Validate the legacy slot tensor and compress it back to full pages."""
    if indices.numel() % page_size:
        raise ValueError("page protocol requires a whole number of pages")
    slots = indices.detach().cpu().reshape(-1, page_size)
    expected = slots[:, :1] + torch.arange(page_size, dtype=slots.dtype)
    if not torch.equal(slots, expected):
        raise ValueError("page protocol requires contiguous complete pages")
    page_ids = torch.sort(slots[:, 0] // page_size).values.tolist()
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


def benchmark_request(
    total_slots: int, request_slots: int, *, rounds: int = ROUNDS
) -> dict:
    # Base SGLang: page-aligned slot tensor allocation and local free-list update.
    base_alloc = median_with_setup(
        lambda: make_base(total_slots),
        lambda pool: HostKVCache.alloc(pool, request_slots),
        rounds,
    )

    def base_free_setup():
        pool = make_base(total_slots)
        indices = HostKVCache.alloc(pool, request_slots)
        return pool, indices

    base_free = median_with_setup(
        base_free_setup, lambda item: HostKVCache.free(item[0], item[1])
        , rounds
    )

    # Current Central I/O: the range allocator still expands a page request to
    # slot indices, then sends those slot ids to the agent.
    range_alloc = median_with_setup(
        lambda: _FreeSlotRanges(total_slots),
        lambda allocator: allocator.allocate(request_slots),
        rounds,
    )

    allocator = _FreeSlotRanges(total_slots)
    indices = allocator.allocate(request_slots)
    assert indices is not None
    legacy_reserve_to_list = median_with_setup(
        lambda: _FreeSlotRanges(total_slots).allocate(request_slots),
        lambda slot_indices: slot_indices.tolist(),
        rounds,
    )
    index_list = indices.tolist()
    request = {"op": "reserve", "model_id": "cold", "indices": index_list}
    serialize = median_ms(lambda: pickle.dumps(request), rounds)
    encoded = pickle.dumps(request)
    deserialize = median_ms(lambda: pickle.loads(encoded), rounds)

    agent_reserve = median_with_setup(
        lambda: make_agent_state(total_slots)[0], lambda agent: agent._handle(request), rounds
    )

    def release_setup():
        agent, _state = make_agent_state(total_slots)
        agent._handle(request)
        return agent

    agent_release = median_with_setup(
        release_setup,
        lambda agent: agent._handle(
            {"op": "release", "model_id": "cold", "indices": index_list}
        ),
        rounds,
    )

    def range_free_setup():
        free_ranges = _FreeSlotRanges(total_slots)
        allocated = free_ranges.allocate(request_slots)
        assert allocated is not None
        return free_ranges, allocated

    range_free = median_with_setup(
        range_free_setup, lambda item: item[0].release(item[1])
        , rounds
    )
    legacy_release_to_list = median_with_setup(
        range_free_setup, lambda item: item[1].tolist(), rounds
    )

    # Runtime PageRange protocol: Central control-plane state and RPC carry
    # disjoint page ranges, never 22,713 page ids or the 363,408 token-slot
    # ids exposed by the unchanged SGLang HostKVCache API.
    request_pages = request_slots // PAGE_TOKENS
    page_alloc = median_with_setup(
        lambda: _FreePageRanges(total_slots // PAGE_TOKENS),
        lambda allocator: allocator.allocate_ranges(request_pages),
        rounds,
    )
    page_allocator = _FreePageRanges(total_slots // PAGE_TOKENS)
    page_ranges = page_allocator.allocate_ranges(request_pages)
    assert page_ranges is not None
    page_expand = median_with_setup(
        lambda: _FreePageRanges(total_slots // PAGE_TOKENS),
        lambda allocator: page_ranges_to_slot_indices(
            allocator.allocate_ranges(request_pages), PAGE_TOKENS
        ),
        rounds,
    )
    page_alloc_and_expand = median_with_setup(
        lambda: _FreePageRanges(total_slots // PAGE_TOKENS),
        lambda allocator: page_ranges_to_slot_indices(
            allocator.allocate_ranges(request_pages), PAGE_TOKENS
        ),
        rounds,
    )
    page_request = {"op": "reserve_pages", "model_id": "cold", "ranges": page_ranges}
    page_serialize = median_ms(lambda: pickle.dumps(page_request), rounds)
    page_encoded = pickle.dumps(page_request)
    page_deserialize = median_ms(lambda: pickle.loads(page_encoded), rounds)
    page_agent_reserve = median_with_setup(
        lambda: make_agent_state(total_slots)[0],
        lambda agent: agent._handle(page_request),
        rounds,
    )

    def page_release_setup():
        agent, _state = make_agent_state(total_slots)
        agent._handle(page_request)
        return agent

    page_agent_release = median_with_setup(
        page_release_setup,
        lambda agent: agent._handle(
            {"op": "release_pages", "model_id": "cold", "ranges": page_ranges}
        ),
        rounds,
    )

    range_metadata_reserve = median_with_setup(
        PageRangeLeaseState,
        lambda state: state.reserve(page_ranges),
        rounds,
    )

    def range_metadata_release_setup():
        state = PageRangeLeaseState()
        state.reserve(page_ranges)
        return state

    range_metadata_release = median_with_setup(
        range_metadata_release_setup,
        lambda state: state.release(page_ranges),
        rounds,
    )

    def page_free_setup():
        allocator = _FreePageRanges(total_slots // PAGE_TOKENS)
        allocated = allocator.allocate_ranges(request_pages)
        assert allocated is not None
        return allocator, page_ranges_to_slot_indices(allocated, PAGE_TOKENS)

    page_compress = median_with_setup(
        page_free_setup,
        lambda item: slot_indices_to_page_ranges(item[1], PAGE_TOKENS),
        rounds,
    )
    page_free = median_with_setup(
        page_free_setup,
        lambda item: item[0].add_ranges(
            slot_indices_to_page_ranges(item[1], PAGE_TOKENS)
        ),
        rounds,
    )
    return {
        "pool_gib": total_slots * SLOT_BYTES / GIB,
        "slots": request_slots,
        "pages": request_slots // PAGE_TOKENS,
        "gib": request_slots * SLOT_BYTES / GIB,
        "rounds": rounds,
        "base_sglang_ms": {"alloc": base_alloc, "free": base_free},
        "current_range_ms": {
            "model_allocate_range_and_expand_slots": range_alloc,
            "legacy_slot_tensor_to_list_reserve": legacy_reserve_to_list,
            "reserve_payload_serialize": serialize,
            "reserve_payload_deserialize": deserialize,
            "agent_reserve_slots": agent_reserve,
            "agent_release_slots": agent_release,
            "legacy_slot_tensor_to_list_release": legacy_release_to_list,
            "model_release_range": range_free,
        },
        "reserve_payload_bytes": len(encoded),
        "page_range_ms": {
            "model_allocate_page_ranges": page_alloc,
            "legacy_page_to_slot_expand": page_expand,
            "model_allocate_and_legacy_expand": page_alloc_and_expand,
            "reserve_payload_serialize": page_serialize,
            "reserve_payload_deserialize": page_deserialize,
            "agent_reserve_page_ranges": page_agent_reserve,
            "agent_release_page_ranges": page_agent_release,
            "legacy_slot_to_page_compress": page_compress,
            "model_legacy_compress_and_release_page_ranges": page_free,
            "candidate_range_metadata_reserve": range_metadata_reserve,
            "candidate_range_metadata_release": range_metadata_release,
        },
        "page_reserve_payload_bytes": len(page_encoded),
    }


def main() -> None:
    total_slots = TOTAL_GIB * GIB // SLOT_BYTES
    total_slots -= total_slots % EXTENT_SLOTS
    results = {
        "geometry": {
            "total_gib": total_slots * SLOT_BYTES / GIB,
            "total_slots": total_slots,
            "total_pages": total_slots // PAGE_TOKENS,
            "page_mib": PAGE_TOKENS * SLOT_BYTES / 1024**2,
            "extent_mib": EXTENT_SLOTS * SLOT_BYTES / 1024**2,
            "rounds": ROUNDS,
        },
        "one_page": benchmark_request(total_slots, PAGE_TOKENS),
        "one_extent": benchmark_request(total_slots, EXTENT_SLOTS),
        "whole_50gib_pool": benchmark_request(
            total_slots, total_slots, rounds=FULL_POOL_ROUNDS
        ),
        "50gib_operation_in_100gib_pool": benchmark_request(
            total_slots * 2, total_slots, rounds=FULL_POOL_ROUNDS
        ),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
