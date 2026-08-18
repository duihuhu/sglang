"""Regression tests for arbitrary free-extent reclaim in Central I/O."""

from __future__ import annotations

import importlib.util
import mmap
import os
from pathlib import Path
import sys
import threading
import unittest

import torch

# The staged runtime may be mounted at different depths in a test container.
# Its direct parent is the import root containing ``sglang``.
sys.path.insert(0, str(Path(__file__).parent))
from sglang.srt.mem_cache.memory_pool_host import _FreeSlotRanges


_SPEC = importlib.util.spec_from_file_location(
    "central_io_fragmented_reclaim",
    Path(__file__).with_name("central_io.py"),
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CentralIOAgent = _MODULE.CentralIOAgent
_HostSegment = _MODULE._HostSegment
_ModelState = _MODULE._ModelState
_ArenaExtent = _MODULE._ArenaExtent
_PageRangeSet = _MODULE._PageRangeSet


class FragmentedReclaimTest(unittest.TestCase):
    slot_bytes = 2 * 2 * 1 * 4 * 2
    extent_bytes = slot_bytes * 10

    def setUp(self):
        self.fd = os.memfd_create("central-io-fragmented-reclaim", os.MFD_CLOEXEC)
        os.ftruncate(self.fd, self.extent_bytes * 3)
        self.mapping = mmap.mmap(self.fd, self.extent_bytes * 3, access=mmap.ACCESS_WRITE)
        self.agent = CentralIOAgent.__new__(CentralIOAgent)
        self.agent._mapping = self.mapping
        self.agent.arena_extent_bytes = self.extent_bytes
        self.agent._arena_extents = {
            extent_id: _ArenaExtent(
                extent_id=extent_id,
                byte_offset=extent_id * self.extent_bytes,
                byte_size=self.extent_bytes,
            )
            for extent_id in range(3)
        }
        self.agent._free_extent_ids = set()
        self.agent._address = __import__("ctypes").addressof(
            __import__("ctypes").c_char.from_buffer(self.mapping)
        )
        self.agent._lock = threading.RLock()
        self.agent._reclaims = {}
        self.agent._next_reclaim_id = 0

    def tearDown(self):
        self.mapping.close()
        os.close(self.fd)

    def _state(self, *, all_live: bool = False):
        state = _ModelState(
            model_id="cold",
            device=0,
            max_capacity=30,
            segment_tokens=10,
            k_ptrs=None,
            v_ptrs=None,
            opened_bases={},
            layer_count=2,
            head_count=1,
            head_dim=4,
            dtype=torch.float16,
            reserved=set(),
            segments={},
            segment_starts=[],
            segments_by_start={},
            active_capacity=30,
        )
        for segment_id, start in enumerate((0, 10, 20)):
            segment = _HostSegment(
                segment_id=segment_id,
                byte_offset=segment_id * self.extent_bytes,
                allocated_bytes=self.extent_bytes,
                logical_start=start,
                slot_count=10,
                host_k=None,
                host_v=None,
            )
            # The tail segment remains live.  The desired allocator must be
            # able to return the two earlier, fully free extents anyway.
            if all_live or start == 20:
                segment.reserved_slots.add(0)
                state.reserved.add(start)
            state.segments[segment_id] = segment
            state.segment_starts.append(start)
            state.segments_by_start[start] = segment_id
        state.next_segment_id = 3
        for segment_id in state.segments:
            extent = self.agent._arena_extents[segment_id]
            extent.owner_model_id = state.model_id
            extent.state = "leased"
        return state

    def test_shrink_reclaims_non_tail_free_extents(self):
        state = self._state()

        result = self.agent._shrink_free(state, 20)
        self.agent._wait_for_reclaim(result["reclaim_id"], timeout_s=1)

        self.assertEqual(result["ranges"], [[10, 10], [0, 10]])
        self.assertEqual(state.capacity, 10)
        self.assertEqual(state.segment_starts, [20])
        self.assertEqual(set(state.segments), {2})

    def test_grow_reuses_a_reclaimed_logical_range(self):
        state = self._state()
        reclaimed = self.agent._shrink_free(state, 20)
        self.agent._wait_for_reclaim(reclaimed["reclaim_id"], timeout_s=1)

        result = self.agent._grow_range(state, 10)

        self.assertEqual(result["ranges"], [[0, 10]])
        self.assertEqual(state.capacity, 20)
        self.assertEqual(state.segment_starts, [0, 20])

    def test_failed_multi_extent_grow_is_atomic(self):
        """A hot retry may not keep a prefix when cold scrub is incomplete.

        The model has two inactive logical extents, but the global pool has
        backing bytes for only one.  A failed request for both must leave
        model ownership and global free bytes precisely unchanged.
        """
        state = self._state()
        reclaimed = self.agent._shrink_free(state, 20)
        self.agent._wait_for_reclaim(reclaimed["reclaim_id"], timeout_s=1)
        self.agent._free_extent_ids = {0}
        self.agent._arena_extents[0].owner_model_id = None
        self.agent._arena_extents[0].state = "free"
        before = {
            "capacity": state.capacity,
            "starts": list(state.segment_starts),
            "segments": set(state.segments),
            "inactive": list(state.inactive_ranges),
            "next_id": state.next_segment_id,
            "free_extents": set(self.agent._free_extent_ids),
        }

        with self.assertRaisesRegex(MemoryError, "insufficient free ArenaExtents"):
            self.agent._grow_range(state, 20)

        self.assertEqual(state.capacity, before["capacity"])
        self.assertEqual(state.segment_starts, before["starts"])
        self.assertEqual(set(state.segments), before["segments"])
        self.assertEqual(state.inactive_ranges, before["inactive"])
        self.assertEqual(state.next_segment_id, before["next_id"])
        self.assertEqual(self.agent._free_extent_ids, before["free_extents"])

    def test_cross_model_transfer_leases_one_complete_arena_extent(self):
        """A receiver must never obtain an interior donor-extent slice."""
        cold = self._state()
        self.agent._models = {"cold": cold}

        # Segment 1 is completely clean, so shrink returns exactly that fixed
        # physical extent to the global arena. Segment 0 and the live tail
        # remain owned by cold.
        reclaimed = self.agent._shrink_free(cold, 10)
        self.agent._wait_for_reclaim(reclaimed["reclaim_id"], timeout_s=1)
        released_extent_id = 1
        self.assertIn(released_extent_id, self.agent._free_extent_ids)
        self.assertIsNone(self.agent._arena_extents[released_extent_id].owner_model_id)

        hot = _ModelState(
            model_id="hot",
            device=0,
            max_capacity=30,
            segment_tokens=10,
            k_ptrs=None,
            v_ptrs=None,
            opened_bases={},
            layer_count=2,
            head_count=1,
            head_dim=4,
            dtype=torch.float16,
            reserved=set(),
            segments={},
            segment_starts=[],
            segments_by_start={},
            active_capacity=0,
            page_size=2,
            inactive_ranges=[(0, 30)],
        )
        logical_start = self.agent._take_logical_range(hot, 10)
        self.agent._add_segment(hot, logical_start, 10)
        received = hot.segments[released_extent_id]

        self.assertEqual(received.segment_id, released_extent_id)
        self.assertEqual(received.byte_offset, released_extent_id * self.extent_bytes)
        self.assertEqual(received.allocated_bytes, self.extent_bytes)
        self.assertEqual(
            self.agent._arena_extents[released_extent_id].owner_model_id, "hot"
        )
        self.assertEqual(self.agent._arena_extents[released_extent_id].state, "leased")
        self.assertEqual(
            {segment.segment_id for segment in cold.segments.values()}
            & {segment.segment_id for segment in hot.segments.values()},
            set(),
        )

    def test_lookup_rejects_a_reclaimed_hole(self):
        state = self._state()
        reclaimed = self.agent._shrink_free(state, 20)
        self.agent._wait_for_reclaim(reclaimed["reclaim_id"], timeout_s=1)

        tail, tail_offset = self.agent._segment_for_index(state, 20)

        self.assertEqual((tail.segment_id, tail_offset), (2, 0))
        with self.assertRaisesRegex(ValueError, "outside"):
            self.agent._segment_for_index(state, 0)
        with self.assertRaisesRegex(ValueError, "outside"):
            self.agent._segment_for_index(state, 10)

    def test_failed_shrink_does_not_partially_mutate_state(self):
        state = self._state(all_live=True)
        before = (state.capacity, list(state.segment_starts), set(state.segments))

        with self.assertRaisesRegex(ValueError, "cannot shrink"):
            self.agent._shrink_free(state, 10)

        self.assertEqual((state.capacity, state.segment_starts, set(state.segments)), before)

    def test_live_reclaim_plan_uses_whole_extents_and_prefers_lower_live_cost(self):
        """A drain plan may select an extent, never an interior slice of one.

        Segment 0 is already clean.  Segments 1 and 2 are both live, but
        segment 2 has only one live page while segment 1 has three.  A request
        for two extents must take segment 0 plus segment 2, leaving segment 1
        untouched for the runtime's radix-aware drain decision.
        """
        state = self._state()
        state.page_size = 2
        state.reserved_pages = _PageRangeSet([(5, 3), (10, 1)])
        state.segments[1].reserved_pages = _PageRangeSet([(0, 3)])
        state.segments[2].reserved_pages = _PageRangeSet([(0, 1)])

        plan = self.agent._plan_live_reclaim(state, 20)

        self.assertEqual(plan["selected_segment_ids"], [0, 2])
        self.assertEqual(plan["clean_segment_ids"], [0])
        self.assertEqual(plan["drain_segment_ids"], [2])
        self.assertEqual(
            plan["segments"],
            [
                {
                    "segment_id": 0,
                    "logical_start": 0,
                    "slot_count": 10,
                    "live_pages": 0,
                    "live_page_ranges": [],
                },
                {
                    "segment_id": 2,
                    "logical_start": 20,
                    "slot_count": 10,
                    "live_pages": 1,
                    "live_page_ranges": [[10, 1]],
                },
            ],
        )

    def test_live_reclaim_plan_does_not_mutate_ownership(self):
        """Planning a drain is asynchronous: ownership changes only on commit."""
        state = self._state()
        state.page_size = 2
        state.reserved_pages = _PageRangeSet([(10, 1)])
        state.segments[2].reserved_pages = _PageRangeSet([(0, 1)])
        before = (state.capacity, list(state.segment_starts), set(state.segments))

        self.agent._plan_live_reclaim(state, 20)

        self.assertEqual((state.capacity, state.segment_starts, set(state.segments)), before)

    def test_live_reclaim_plan_limits_each_step_to_whole_extent_batch(self):
        state = self._state()
        state.page_size = 2

        plan = self.agent._plan_live_reclaim(state, 20, max_segments=1)

        self.assertEqual(plan["selected_segment_ids"], [0])
        self.assertEqual(plan["clean_segment_ids"], [0])
        self.assertEqual(plan["drain_segment_ids"], [])
        self.assertEqual(plan["segments"][0]["slot_count"], 10)

    def test_live_reclaim_exclusions_do_not_become_false_capacity_failure(self):
        """A safety exclusion can limit this batch without shrinking the lease.

        The final target still requires 20 slots, but two of three whole
        extents are temporarily excluded by the radix safety pass.  The agent
        must hand back the one eligible 10-slot extent as a partial batch;
        reporting an active-capacity failure would be incorrect because the
        model still owns all 30 slots.
        """
        state = self._state()
        state.page_size = 2

        plan = self.agent._plan_live_reclaim(
            state, 20, excluded_segment_ids={0, 1}, max_segments=32
        )

        self.assertEqual(plan["requested_slots"], 20)
        self.assertEqual(plan["selected_segment_ids"], [2])
        self.assertEqual(plan["clean_segment_ids"], [])
        self.assertEqual(plan["drain_segment_ids"], [2])

    def test_live_reclaim_reports_temporary_safety_block_without_exception(self):
        state = self._state()

        plan = self.agent._plan_live_reclaim(
            state, 20, excluded_segment_ids={0, 1, 2}, max_segments=32
        )

        self.assertEqual(plan["requested_slots"], 20)
        self.assertEqual(plan["selected_segment_ids"], [])
        self.assertEqual(plan["blocked_reason"], "no_eligible_extents")

    def test_target_capacity_protocol_uses_agent_capacity_not_stale_client_count(self):
        """The model sends a final target; the agent derives remaining work."""
        state = self._state()
        self.agent._models = {"cold": state}
        # This emulates two already-completed clean batches.  A client that
        # had cached the old 30-slot quota would incorrectly ask to reclaim
        # 20 more slots, but the agent sees that the 10-slot target is met.
        reclaimed = self.agent._shrink_free(state, 20)
        self.agent._wait_for_reclaim(reclaimed["reclaim_id"], timeout_s=1)

        plan = self.agent._handle(
            {
                "op": "plan_live_reclaim",
                "model_id": "cold",
                "target_capacity": 10,
                "excluded_segment_ids": [],
                "max_segments": 1,
            }
        )

        self.assertEqual(plan["active_capacity"], 10)
        self.assertEqual(plan["target_capacity"], 10)
        self.assertEqual(plan["remaining_slots"], 0)
        self.assertEqual(plan["selected_segment_ids"], [])

    def test_live_drain_fences_an_extent_until_abort(self):
        """A draining extent stays the stable reclaim candidate and can abort safely."""
        state = self._state()
        state.page_size = 2
        state.reserved_pages = _PageRangeSet([(10, 1)])
        state.segments[2].reserved_pages = _PageRangeSet([(0, 1)])

        begin = self.agent._begin_live_drain(state, [2])

        self.assertEqual(begin["segment_ids"], [2])
        self.assertTrue(state.segments[2].draining)
        self.agent._models = {"cold": state}
        with self.assertRaisesRegex(ValueError, "draining"):
            self.agent._handle(
                {"op": "reserve_pages", "model_id": "cold", "ranges": [[11, 1]]}
            )
        # A later plan must continue draining this same extent rather than
        # selecting an unrelated clean extent and losing convergence.
        plan = self.agent._plan_live_reclaim(state, 10)
        self.assertEqual(plan["selected_segment_ids"], [2])

        aborted = self.agent._abort_live_drain(state, [2])

        self.assertEqual(aborted["segment_ids"], [2])
        self.assertFalse(state.segments[2].draining)
        self.assertEqual(aborted["segments"][0]["live_page_ranges"], [[10, 1]])

    def test_describe_layout_separates_live_pages_from_dirty_free_pages(self):
        state = self._state()
        state.page_size = 2
        self.agent._models = {"cold": state}

        self.agent._handle(
            {"op": "reserve_pages", "model_id": "cold", "ranges": [[0, 2]]}
        )
        self.agent._handle(
            {"op": "release_pages", "model_id": "cold", "ranges": [[0, 1]]}
        )

        layout = self.agent._handle({"op": "describe_layout", "model_id": "cold"})
        first = next(segment for segment in layout["segments"] if segment["extent_id"] == 0)

        self.assertEqual(layout["page_size"], 2)
        self.assertEqual(first["reserved_page_ranges"], [[1, 1]])
        self.assertEqual(first["dirty_page_ranges"], [[0, 1]])

    def test_commit_live_reclaim_requires_clean_extent_then_scrubs_before_release(self):
        state = self._state()
        state.page_size = 2
        state.reserved_pages = _PageRangeSet([(10, 1)])
        state.segments[2].reserved_pages = _PageRangeSet([(0, 1)])
        self.mapping[2 * self.extent_bytes : 3 * self.extent_bytes] = b"x" * self.extent_bytes

        with self.assertRaisesRegex(ValueError, "live KV"):
            self.agent._commit_live_reclaim(state, [2])

        state.reserved_pages.clear()
        state.segments[2].reserved_pages.clear()
        state.reserved.clear()
        state.segments[2].reserved_slots.clear()
        result = self.agent._commit_live_reclaim(state, [2])
        self.agent._wait_for_reclaim(result["reclaim_id"], timeout_s=1)

        self.assertEqual(state.capacity, 20)
        self.assertNotIn(2, state.segments)
        self.assertEqual(
            self.mapping[2 * self.extent_bytes : 3 * self.extent_bytes],
            b"\0" * self.extent_bytes,
        )
        self.assertEqual(result["ranges"], [[20, 10]])

    def test_reset_ranges_keeps_a_reclaimed_hole_inactive(self):
        active = _FreeSlotRanges(30)
        active.remove_ranges([(10, 10)])
        free = _FreeSlotRanges()

        free.reset_ranges(active.ranges)
        allocated = free.allocate(20)

        self.assertEqual(allocated.tolist(), list(range(10)) + list(range(20, 30)))


if __name__ == "__main__":
    unittest.main()
