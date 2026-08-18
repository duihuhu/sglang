"""Tests for page-granular cross-model physical lease transfer."""

from __future__ import annotations

import ctypes
import importlib.util
import mmap
import os
from pathlib import Path
import sys
import threading
import time
import unittest

import torch


_SPEC = importlib.util.spec_from_file_location(
    "central_io_dynamic_page_lease", Path(__file__).with_name("central_io.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CentralIOAgent = _MODULE.CentralIOAgent
CentralIOControlClient = _MODULE.CentralIOControlClient
_ModelState = _MODULE._ModelState


class CentralIODynamicPageLeaseTest(unittest.TestCase):
    page_size = 2
    slot_bytes = 2 * 2 * 1 * 4 * 2

    def setUp(self):
        self.fd = os.memfd_create("central-io-dynamic-page-lease", os.MFD_CLOEXEC)
        self.pool_bytes = self.slot_bytes * 8
        os.ftruncate(self.fd, self.pool_bytes)
        self.mapping = mmap.mmap(self.fd, self.pool_bytes, access=mmap.ACCESS_WRITE)
        self.agent = CentralIOAgent.__new__(CentralIOAgent)
        self.agent._mapping = self.mapping
        self.agent._address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        self.agent._lock = threading.RLock()
        self.agent._reclaims = {}
        self.agent._next_reclaim_id = 0
        self.agent._next_segment_id = 0
        self.agent.arena_extent_bytes = self.pool_bytes
        self.agent._init_dynamic_free_bytes(self.pool_bytes)

    def tearDown(self):
        self.mapping.close()
        os.close(self.fd)

    def _state(self, model_id: str) -> _ModelState:
        return _ModelState(
            model_id=model_id,
            device=0,
            max_capacity=8,
            segment_tokens=0,
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
            page_size=self.page_size,
            inactive_ranges=[],
        )

    def test_middle_clean_page_moves_to_hot_without_releasing_cold_neighbours(self):
        cold = self._state("cold")
        hot = self._state("hot")
        cold_segment = self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)
        self.assertEqual(cold_segment.byte_offset, 0)

        # The cold model has made page 1 clean after radix-level drain; page
        # 0 and page 2/3 remain part of its active host mapping.
        cold.reserved_pages.add_range(1, 1)
        cold_segment.reserved_pages.add_range(1, 1)
        self.agent._begin_live_page_drain(cold, [(1, 1)])
        cold.reserved_pages.remove_range(1, 1)
        cold_segment.reserved_pages.remove_range(1, 1)
        reclaim = self.agent._commit_live_page_reclaim(cold, [(1, 1)])
        self.agent._wait_for_reclaim(reclaim["reclaim_id"], timeout_s=1)

        hot_segment = self.agent._add_dynamic_segment(hot, logical_start=0, slot_count=2)
        self.assertEqual(hot_segment.byte_offset, self.page_size * self.slot_bytes)
        self.assertEqual(cold.capacity, 6)
        self.assertEqual(hot.capacity, 2)
        self.assertEqual(
            [(segment.logical_start, segment.slot_count) for segment in cold.segments.values()],
            [(0, 2), (4, 4)],
        )

    def test_drained_page_is_fenced_until_radix_release_then_reassigned(self):
        cold = self._state("cold")
        hot = self._state("hot")
        cold_segment = self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)

        # Page 1 begins as live. The agent must fence it before the radix
        # layer releases it, so a new host-KV allocation cannot race the
        # pending reclaim.
        cold.reserved_pages.add_range(1, 1)
        cold_segment.reserved_pages.add_range(1, 1)
        drain = self.agent._begin_live_page_drain(cold, [(1, 1)])
        self.assertEqual(drain["page_ranges"], [[1, 1]])
        self.assertTrue(cold_segment.draining_pages.contains(1, 1))

        # This is the state after SGLang safely evicts the selected leaf and
        # CentralIOMHATokenToKVPoolHost.free releases the complete page.
        cold.reserved_pages.remove_range(1, 1)
        cold_segment.reserved_pages.remove_range(1, 1)
        reclaim = self.agent._commit_live_page_reclaim(cold, [(1, 1)])
        self.agent._wait_for_reclaim(reclaim["reclaim_id"], timeout_s=1)

        hot_segment = self.agent._add_dynamic_segment(hot, logical_start=0, slot_count=2)
        self.assertEqual(hot_segment.byte_offset, self.page_size * self.slot_bytes)
        self.assertEqual(cold.capacity, 6)
        self.assertEqual(hot.capacity, 2)

    def test_page_handoff_records_detach_scrub_and_grow_timestamps(self):
        cold = self._state("cold")
        hot = self._state("hot")
        cold_segment = self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)
        cold.reserved_pages.add_range(1, 1)
        cold_segment.reserved_pages.add_range(1, 1)
        self.agent._begin_live_page_drain(cold, [(1, 1)])
        cold.reserved_pages.remove_range(1, 1)
        cold_segment.reserved_pages.remove_range(1, 1)

        reclaim = self.agent._commit_live_page_reclaim(cold, [(1, 1)])
        status = self.agent._wait_for_reclaim(reclaim["reclaim_id"], timeout_s=1)
        hot.dynamic_page_leases = True
        hot.inactive_ranges = [(0, 2)]
        grow = self.agent._grow_range(hot, 2)

        self.assertIsNotNone(reclaim["detached_ns"])
        self.assertIsNotNone(status["scrub_started_ns"])
        self.assertIsNotNone(status["scrub_ready_ns"])
        self.assertLessEqual(reclaim["detached_ns"], status["scrub_started_ns"])
        self.assertLessEqual(status["scrub_started_ns"], status["scrub_ready_ns"])
        self.assertGreaterEqual(grow["grow_accepted_ns"], status["scrub_ready_ns"])
        self.assertEqual(hot.first_grow_accepted_ns, grow["grow_accepted_ns"])

    def test_prepared_grow_is_not_effective_until_recipient_acknowledges(self):
        """A physical handoff is not usable until the model allocator admits it."""
        hot = self._state("hot")
        hot.dynamic_page_leases = True
        hot.inactive_ranges = [(0, 4)]

        prepared = self.agent._prepare_grow_range(hot, self.page_size)

        self.assertEqual(hot.capacity, 0)
        self.assertEqual(prepared["effective_capacity"], 0)
        self.assertEqual(prepared["pending_capacity"], self.page_size)

        acknowledged = self.agent._ack_prepared_grow(
            hot, prepared["transfer_id"]
        )

        self.assertEqual(hot.capacity, self.page_size)
        self.assertEqual(acknowledged["effective_capacity"], self.page_size)

    def test_aborted_prepared_grow_returns_capacity_to_central_pool(self):
        hot = self._state("hot")
        hot.dynamic_page_leases = True
        hot.inactive_ranges = [(0, 4)]

        prepared = self.agent._prepare_grow_range(hot, self.page_size)
        aborted = self.agent._abort_prepared_grow(hot, prepared["transfer_id"])

        self.assertEqual(hot.capacity, 0)
        self.assertEqual(aborted["released_slots"], self.page_size)
        self.assertEqual(hot.inactive_ranges, [(0, 4)])

    def test_prepared_shrink_fences_then_changes_capacity_only_at_ack(self):
        cold = self._state("cold")
        cold.dynamic_page_leases = True
        segment = self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)

        # Page 0 is live; pages 1..3 are clean and may be donated.  The
        # prepare phase fences one clean page but must not yet publish a
        # smaller effective quota.
        cold.reserved_pages.add_range(0, 1)
        segment.reserved_pages.add_range(0, 1)
        prepared = self.agent._prepare_shrink_free(cold, self.page_size)

        self.assertEqual(cold.capacity, 8)
        self.assertEqual(prepared["effective_capacity"], 8)
        self.assertEqual(prepared["page_ranges"], [[3, 1]])
        self.assertTrue(segment.draining_pages.contains(3, 1))

        acknowledged = self.agent._ack_prepared_shrink(
            cold, prepared["transfer_id"]
        )
        self.assertEqual(cold.capacity, 6)
        self.assertEqual(acknowledged["effective_capacity"], 6)
        self.agent._wait_for_reclaim(acknowledged["reclaim_id"], timeout_s=1)

    def test_aborted_prepared_shrink_reopens_the_donor_page(self):
        cold = self._state("cold")
        cold.dynamic_page_leases = True
        segment = self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)
        prepared = self.agent._prepare_shrink_free(cold, self.page_size)

        aborted = self.agent._abort_prepared_shrink(cold, prepared["transfer_id"])

        self.assertEqual(cold.capacity, 8)
        self.assertEqual(aborted["effective_capacity"], 8)
        self.assertFalse(segment.draining_pages.contains(3, 1))

    def test_global_status_exposes_pending_donor_release_before_ack(self):
        cold = self._state("cold")
        cold.dynamic_page_leases = True
        self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)
        self.agent._models = {"cold": cold}

        prepared = self.agent._prepare_shrink_free(cold, self.page_size)
        snapshot = self.agent._global_status()

        self.assertEqual(snapshot["models"]["cold"]["capacity"], 8)
        self.assertEqual(snapshot["models"]["cold"]["pending_release_capacity"], 2)
        self.agent._abort_prepared_shrink(cold, prepared["transfer_id"])

    def test_control_protocol_reports_only_acknowledged_capacity(self):
        hot = self._state("hot")
        hot.dynamic_page_leases = True
        hot.inactive_ranges = [(0, 4)]
        self.agent._models = {"hot": hot}

        prepared = self.agent._handle(
            {"op": "prepare_grow", "model_id": "hot", "count": self.page_size}
        )
        before_ack = self.agent._handle({"op": "status", "model_id": "hot"})

        self.assertEqual(prepared["effective_capacity"], 0)
        self.assertEqual(before_ack["capacity"], 0)
        self.assertEqual(before_ack["pending_capacity"], self.page_size)

        acknowledged = self.agent._handle(
            {
                "op": "ack_prepared_grow",
                "model_id": "hot",
                "transfer_id": prepared["transfer_id"],
            }
        )
        after_ack = self.agent._handle({"op": "status", "model_id": "hot"})

        self.assertEqual(acknowledged["effective_capacity"], self.page_size)
        self.assertEqual(after_ack["capacity"], self.page_size)
        self.assertEqual(after_ack["pending_capacity"], 0)

    def test_agent_exposes_one_reported_local_health_snapshot_per_model(self):
        hot = self._state("hot")
        hot.dynamic_page_leases = True
        hot.inactive_ranges = [(0, 4)]
        self.agent._add_dynamic_segment(hot, 0, 4)
        self.agent._models = {"hot": hot}

        self.agent._handle(
            {
                "op": "report_local_residency",
                "model_id": "hot",
                "effective_pages": 2,
                "floor_pages": 1,
                "clean_pages": 1,
                "ready_pages": 0,
                "live_pages": 1,
                "min_pages": 1,
                "low_pages": 1,
                "high_pages": 1,
                "retention_debt_tokens": 64,
                "unresolved_clean_pages": 0,
                "backup_ingress_pages_per_s": 12.5,
                "ready_latency_p95_s": 0.08,
                "feedback_horizon_s": 15.0,
                "action": "quota_deficit",
            }
        )

        snapshot = self.agent._handle({"op": "global_status"})

        self.assertEqual(snapshot["models"]["hot"]["residency"]["clean_pages"], 1)
        self.assertEqual(
            snapshot["models"]["hot"]["residency"]["retention_debt_tokens"], 64
        )
        self.assertEqual(
            snapshot["models"]["hot"]["residency"]["backup_ingress_pages_per_s"],
            12.5,
        )
        self.assertEqual(
            snapshot["models"]["hot"]["residency"]["action"], "quota_deficit"
        )

    def test_page_reclaims_use_one_bounded_scrub_worker(self):
        """A rapid handoff must queue later scrubs instead of spawning threads."""
        cold = self._state("cold")
        segment = self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)
        cold.reserved_pages.add_range(0, 2)
        segment.reserved_pages.add_range(0, 2)
        self.agent._begin_live_page_drain(cold, [(0, 2)])
        cold.reserved_pages.remove_range(0, 2)
        segment.reserved_pages.remove_range(0, 2)

        first_started = threading.Event()
        release_first = threading.Event()
        calls = []
        original_memset = _MODULE.ctypes.memset

        def blocking_memset(*args):
            calls.append(args[2])
            first_started.set()
            release_first.wait(timeout=1)
            return original_memset(*args)

        _MODULE.ctypes.memset = blocking_memset
        try:
            first = self.agent._commit_live_page_reclaim(cold, [(0, 1)])
            self.assertTrue(first_started.wait(timeout=1))
            second = self.agent._commit_live_page_reclaim(cold, [(1, 1)])
            time.sleep(0.02)
            self.assertEqual(len(calls), 1)
            release_first.set()
            self.agent._wait_for_reclaim(first["reclaim_id"], timeout_s=1)
            self.agent._wait_for_reclaim(second["reclaim_id"], timeout_s=1)
        finally:
            _MODULE.ctypes.memset = original_memset

    def test_bulk_target_update_publishes_all_models_at_one_epoch(self):
        cold = self._state("cold")
        hot = self._state("hot")
        cold.dynamic_page_leases = True
        hot.dynamic_page_leases = True
        self.agent._models = {"cold": cold, "hot": hot}

        response = self.agent._handle(
            {
                "op": "set_quota_targets",
                "targets": {"cold": 4, "hot": 6},
            }
        )

        self.assertEqual(cold.quota_target, 4)
        self.assertEqual(hot.quota_target, 6)
        self.assertEqual(cold.quota_target_set_ns, hot.quota_target_set_ns)
        self.assertEqual(response["target_set_ns"], cold.quota_target_set_ns)

    def test_shrink_free_detaches_one_clean_page_from_mixed_segment(self):
        cold = self._state("cold")
        cold.dynamic_page_leases = True
        segment = self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)

        # One physical segment contains four pages. Three still contain KV;
        # page 1 is allocator-free. The shrink must return that single page
        # rather than requiring the entire segment to become empty.
        cold.reserved_pages.add_range(0, 1)
        cold.reserved_pages.add_range(2, 2)
        segment.reserved_pages.add_range(0, 1)
        segment.reserved_pages.add_range(2, 2)

        reclaim = self.agent._shrink_free(cold, self.page_size)
        self.agent._wait_for_reclaim(reclaim["reclaim_id"], timeout_s=1)

        self.assertEqual(cold.capacity, 6)
        self.assertEqual(reclaim["ranges"], [[self.page_size, self.page_size]])
        self.assertEqual(
            sorted((item.logical_start, item.slot_count) for item in cold.segments.values()),
            [(0, 2), (4, 4)],
        )
        left = cold.segments[cold.segments_by_start[0]]
        right = cold.segments[cold.segments_by_start[4]]
        self.assertTrue(left.reserved_pages.contains(0, 1))
        self.assertTrue(right.reserved_pages.contains(0, 2))

    def test_shrink_free_reports_donor_detach_and_scrub_timing(self):
        cold = self._state("cold")
        cold.dynamic_page_leases = True
        self.agent._add_dynamic_segment(cold, logical_start=0, slot_count=8)

        reclaim = self.agent._shrink_free(cold, self.page_size * 2)

        self.assertIn("timing_ms", reclaim)
        self.assertGreaterEqual(reclaim["timing_ms"]["agent_total"], 0.0)
        self.assertGreaterEqual(reclaim["timing_ms"]["agent_select_free_pages"], 0.0)
        self.assertGreaterEqual(reclaim["timing_ms"]["agent_detach_pages"], 0.0)
        self.assertGreaterEqual(reclaim["timing_ms"]["agent_submit_scrub"], 0.0)
        self.assertEqual(reclaim["timing_ms"]["reclaim_pages"], 2)

        self.agent._wait_for_reclaim(reclaim["reclaim_id"], timeout_s=1)
        status = self.agent._reclaim_status(reclaim["reclaim_id"])
        self.assertEqual(status["state"], "ready")
        self.assertIsNotNone(status["detached_ns"])
        self.assertIsNotNone(status["scrub_started_ns"])
        self.assertIsNotNone(status["scrub_ready_ns"])
        self.assertGreaterEqual(status["scrub_ms"], 0.0)

    def test_grow_combines_fragmented_physical_page_runs(self):
        hot = self._state("hot")
        hot.dynamic_page_leases = True
        hot.inactive_ranges = [(0, 4)]
        blocker_a = self._state("blocker-a")
        blocker_b = self._state("blocker-b")
        donor_a = self._state("donor-a")
        donor_b = self._state("donor-b")

        # Allocate four physical pages, then return alternating pages. The
        # two free pages have enough total capacity for hot but are separated
        # by blocker-owned pages, exactly as after fragmented live reclaim.
        donor_segment_a = self.agent._add_dynamic_segment(donor_a, 0, 2)
        self.agent._add_dynamic_segment(blocker_a, 0, 2)
        donor_segment_b = self.agent._add_dynamic_segment(donor_b, 0, 2)
        self.agent._add_dynamic_segment(blocker_b, 0, 2)
        self.agent._remove_dynamic_segment(donor_a, donor_segment_a)
        self.agent._remove_dynamic_segment(donor_b, donor_segment_b)
        self.agent._free_byte_ranges.release(
            donor_segment_a.byte_offset, donor_segment_a.allocated_bytes
        )
        self.agent._free_byte_ranges.release(
            donor_segment_b.byte_offset, donor_segment_b.allocated_bytes
        )

        result = self.agent._grow_range(hot, 4)
        self.assertEqual(hot.capacity, 4)
        self.assertEqual(len(result["ranges"]), 2)
        self.assertEqual(sum(count for _, count in result["ranges"]), 4)

    def test_integer_gib_target_is_converted_to_page_aligned_capacity(self):
        client = CentralIOControlClient.__new__(CentralIOControlClient)
        client.status = lambda _model_id: {"token_bytes": 3, "page_size": 16}
        seen = []
        client.set_quota_target = lambda model_id, slots: seen.append((model_id, slots)) or {
            "target_capacity": slots
        }

        result = client.set_quota_target_gib("hot", 1)

        expected = (1024**3 // 3) // 16 * 16
        self.assertEqual(seen, [("hot", expected)])
        self.assertEqual(result["requested_gib"], 1)
        self.assertEqual(result["effective_pages"], expected // 16)


if __name__ == "__main__":
    unittest.main()
