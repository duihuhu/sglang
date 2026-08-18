"""HiRadix integration tests for page-level live-KV census."""

from __future__ import annotations

import unittest
import os
from queue import Queue
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.mem_cache.hiradix_cache import HiRadixCache
from sglang.srt.mem_cache.live_page_reclaim import (
    PersistentLeafReclaimIndex,
    prefix_fingerprint,
)
from sglang.srt.mem_cache.radix_cache import RadixKey, TreeNode
from sglang.srt.mem_cache.base_prefix_cache import InsertParams


def _node(node_id: int, slots: list[int], hit_count: int = 0) -> TreeNode:
    node = TreeNode(id=node_id)
    node.key = RadixKey([node_id])
    node.host_value = torch.tensor(slots, dtype=torch.int64)
    node.hit_count = hit_count
    return node


class HiRadixLivePageCensusTest(unittest.TestCase):
    def test_partial_storage_ack_never_allows_host_kv_to_be_dropped(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.tp_world_size = 1
        cache.enable_storage_metrics = False
        cache.ongoing_prefetch = {}
        cache.ongoing_backup = {}
        leaf = _node(1, list(range(0, 16)))
        leaf._latticekv_storage_state = "writing"
        leaf.protect_host()
        cache.ongoing_backup[7] = leaf
        cache.cache_controller = SimpleNamespace(
            prefetch_revoke_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            prefetch_tokens_occupied=0,
        )
        cache.cache_controller.ack_backup_queue.put(
            SimpleNamespace(id=7, completed_tokens=15)
        )
        cache._observe_central_reclaim_node = lambda node: None

        cache._drain_storage_control_queues_impl(
            n_revoke=0, n_backup=1, n_release=0, log_metrics=False
        )

        self.assertEqual(leaf._latticekv_storage_state, "retry")
        self.assertEqual(leaf.host_ref_counter, 0)

    def test_storage_ack_marks_host_kv_durable_before_reclaim_can_drop_it(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.tp_world_size = 1
        cache.enable_storage_metrics = False
        cache.ongoing_prefetch = {}
        cache.ongoing_backup = {}
        leaf = _node(1, list(range(0, 16)))
        leaf._latticekv_storage_state = "writing"
        leaf.protect_host()
        cache.ongoing_backup[7] = leaf
        cache.cache_controller = SimpleNamespace(
            prefetch_revoke_queue=Queue(),
            ack_backup_queue=Queue(),
            host_mem_release_queue=Queue(),
            prefetch_tokens_occupied=0,
        )
        cache.cache_controller.ack_backup_queue.put(
            SimpleNamespace(id=7, completed_tokens=16)
        )
        cache._observe_central_reclaim_node = lambda node: None

        cache._drain_storage_control_queues_impl(
            n_revoke=0, n_backup=1, n_release=0, log_metrics=False
        )

        self.assertEqual(leaf._latticekv_storage_state, "durable")
        self.assertEqual(leaf.host_ref_counter, 0)
    def test_reclaim_key_caches_the_full_radix_path_after_first_lookup(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        leaf = TreeNode(id=1)
        leaf.key = RadixKey([10, 11, 12, 13])
        leaf.parent = cache.root_node
        cache.root_node.children[(10,)] = leaf

        with patch(
            "sglang.srt.mem_cache.hiradix_cache.prefix_fingerprint",
            wraps=prefix_fingerprint,
        ) as fingerprint:
            first = cache._central_reclaim_key(leaf)
            second = cache._central_reclaim_key(leaf)

        self.assertEqual(first, second)
        self.assertEqual(fingerprint.call_count, 1)

    def test_reinserted_host_prefix_turns_its_ghost_into_reclaim_feedback(self):
        """A real HiRadix insert must account for a recently reclaimed prefix."""
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        cache.is_eagle = False
        cache.enable_storage = False
        cache.enable_kv_cache_events = False
        cache.evictable_size_ = 0
        cache.get_child_key_fn = lambda key: (key.token_ids[0],)
        cache._update_leaf_status = lambda node: None
        cache._record_store_event = lambda node: None
        cache.cache_controller = SimpleNamespace(write_policy="write_back")
        cache._central_live_reclaim_queue = PersistentLeafReclaimIndex(
            page_size=16, ghost_ttl_s=60.0
        )
        cache._central_live_reclaim_nodes = {}
        reported_losses = []
        cache.token_to_kv_pool_host = SimpleNamespace(
            record_retention_loss=reported_losses.append
        )

        reclaimed_key = prefix_fingerprint([10, 11, 12, 13])
        cache._central_live_reclaim_queue.mark_reclaimed(
            reuse_key=reclaimed_key, now_s=0.0
        )

        with patch(
            "sglang.srt.mem_cache.hiradix_cache.time.monotonic", return_value=10.0
        ):
            cache.insert(
                InsertParams(
                    key=RadixKey([10, 11, 12, 13]),
                    value=torch.arange(4, dtype=torch.int64),
                )
            )

        reinserted = cache.root_node.children[(10,)]
        self.assertEqual(reported_losses, [4])
        reinserted.host_value = torch.arange(16, dtype=torch.int64)
        cheap = _node(2, list(range(16, 32)), hit_count=1)
        cheap.parent = cache.root_node
        cache.root_node.children[(2,)] = cheap
        cache._observe_central_reclaim_node(reinserted)
        cache._observe_central_reclaim_node(cheap)

        action = cache._central_live_reclaim_queue.pop(now_s=11.0)
        self.assertEqual(action.node_id, cheap.id)

    def test_local_low_watermark_reclaims_only_the_gap_to_high(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        calls = []
        cache.token_to_kv_pool_host = SimpleNamespace(
            dynamic_page_leases=True,
            local_residency_status=lambda: {
                "action": "prepare",
                "clean_pages": 3,
                "ready_pages": 2,
                "min_pages": 1,
                "low_pages": 8,
                "high_pages": 12,
            },
        )
        cache.reclaim_live_page_selection = (
            lambda **kwargs: calls.append(kwargs) or {"allocator_free_slots": 112}
        )

        result = cache._maintain_local_residency()

        self.assertEqual(calls[0]["max_pages"], 7)
        self.assertIsNone(calls[0]["protected_hit_count"])
        self.assertIsNone(calls[0]["on_allocator_pages_ready"])
        self.assertEqual(calls[0]["selection_budget_ms"], 2.0)
        self.assertEqual(result["requested_pages"], 7)

    def test_local_emergency_converts_ready_capacity_back_to_clean_pages(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        calls = []
        cache.token_to_kv_pool_host = SimpleNamespace(
            dynamic_page_leases=True,
            local_residency_status=lambda: {
                "action": "emergency",
                "clean_pages": 0,
                "ready_pages": 1,
                "min_pages": 1,
                "low_pages": 1,
                "high_pages": 1,
            },
        )
        cache.reclaim_live_page_selection = (
            lambda **kwargs: calls.append(kwargs) or {"allocator_free_slots": 16}
        )

        result = cache._maintain_local_residency()

        self.assertEqual(calls[0]["max_pages"], 1)
        self.assertEqual(result["requested_pages"], 1)

    def test_host_backup_admission_can_wake_local_reclaim_before_generic_evict(self):
        """A backup-sized clean-page shortfall is a local maintenance trigger."""
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        calls = []
        cache.token_to_kv_pool_host = SimpleNamespace(
            dynamic_page_leases=True,
            local_residency_status=lambda: {
                "action": "idle",
                "clean_pages": 2,
                "ready_pages": 0,
                "min_pages": 1,
                "low_pages": 1,
                "high_pages": 1,
            },
        )
        cache.reclaim_live_page_selection = (
            lambda **kwargs: calls.append(kwargs) or {"allocator_free_slots": 48}
        )

        result = cache._maintain_local_residency(required_clean_pages=5)

        self.assertEqual(calls[0]["max_pages"], 3)
        self.assertEqual(result["local_action"], "admission")
        self.assertEqual(result["required_clean_pages"], 5)

    def test_local_maintenance_uses_persistent_queue_without_cross_model_handoff(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        leaf = _node(1, list(range(0, 16)))
        leaf.parent = cache.root_node
        cache.root_node.children[(1,)] = leaf
        cache.evictable_host_leaves = {leaf}
        cache.get_child_key_fn = lambda key: (key.token_ids[0],)
        cache._record_remove_event = lambda node: None
        live_slots = set(range(16))
        cache.cache_controller = SimpleNamespace(
            evict_host=lambda indices: live_slots.difference_update(indices.tolist())
            or len(indices)
        )
        ready_updates = []
        host_pool = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=16,
            available_size=lambda: 16 - len(live_slots),
            local_residency_status=lambda: {
                "action": "prepare",
                "clean_pages": 0,
                "ready_pages": 0,
                "min_pages": 1,
                "low_pages": 1,
                "high_pages": 1,
            },
        )
        host_pool.set_local_ready_pages = ready_updates.append
        cache.token_to_kv_pool_host = host_pool
        cache._central_live_reclaim_queue = PersistentLeafReclaimIndex(page_size=16)
        cache._central_live_reclaim_nodes = {}
        cache._central_live_reclaim_queue_epoch = None
        cache._update_host_leaf_status(leaf)

        cache._live_host_node_snapshots = lambda: (_ for _ in ()).throw(
            AssertionError("local maintenance rebuilt a full host snapshot")
        )
        result = cache._maintain_local_residency()

        self.assertEqual(result["delete_host_leaf_node_ids"], [leaf.id])
        self.assertEqual(result["allocator_free_slots"], 16)
        self.assertEqual(cache.token_to_kv_pool_host.active_size, 16)
        self.assertNotIn((1,), cache.root_node.children)
        self.assertEqual(ready_updates, [1, 0])

    def test_persistent_index_serves_shrink_without_a_full_host_slot_snapshot(self):
        """Normal node updates prepare the next shrink before it is requested."""
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        parent = _node(1, list(range(0, 8)), hit_count=1)
        child = _node(2, list(range(8, 16)), hit_count=0)
        parent.parent = cache.root_node
        cache.root_node.children[(1,)] = parent
        child.parent = parent
        parent.children[(2,)] = child
        cache.evictable_host_leaves = set()
        cache._central_host_leaf_epoch = 0
        cache._central_live_reclaim_queue = PersistentLeafReclaimIndex(page_size=16)
        cache._central_live_reclaim_nodes = {}
        cache._central_live_reclaim_queue_epoch = None

        cache._update_host_leaf_status(parent)
        cache._update_host_leaf_status(child)

        def snapshot_must_not_run():
            raise AssertionError("shrink rebuilt a full host-slot snapshot")

        cache._live_host_node_snapshots = snapshot_must_not_run
        action = cache.select_next_incremental_live_reclaim_action(
            protected_hit_count=3
        )

        self.assertEqual((action.node_id, action.action_kind), (2, "delete_host_leaf"))
        self.assertEqual(cache._central_live_reclaim_queue.node_updates, 2)

    def test_census_uses_radix_children_to_block_only_mixed_page(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])

        reclaimable_a = _node(1, list(range(0, 16)))
        mixed_leaf_a = _node(2, list(range(16, 27)))
        shared = _node(3, [27, 28], hit_count=27)
        mixed_leaf_b = _node(4, list(range(29, 32)))
        reclaimable_b = _node(5, list(range(32, 48)))
        shared_child = _node(6, [48])
        # The child establishes that node 3 is a shared radix branch.  It is
        # not itself resident in host KV for this census.
        shared_child.host_value = None

        for node in (reclaimable_a, mixed_leaf_a, shared, mixed_leaf_b, reclaimable_b):
            node.parent = cache.root_node
            cache.root_node.children[(node.id,)] = node
        shared_child.parent = shared
        shared.children[(shared_child.id,)] = shared_child

        census = cache.live_page_reclaim_census(protected_hit_count=3)

        self.assertEqual(census.reclaimable_page_ranges, [(0, 1), (2, 1)])
        self.assertEqual(census.page_reasons[1], {"shared_prefix": 1})

    def test_drain_selection_keeps_only_complete_leaf_owned_pages(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])

        complete = _node(1, list(range(0, 16)))
        partial = _node(2, list(range(16, 24)))
        crosses_protected = _node(3, list(range(24, 40)))
        protected = _node(4, list(range(40, 48)), hit_count=30)
        protected_child = _node(5, [])
        protected_child.host_value = None
        protected.children[(5,)] = protected_child
        protected_child.parent = protected

        for node in (complete, partial, crosses_protected, protected):
            node.parent = cache.root_node
            cache.root_node.children[(node.id,)] = node

        selection = cache.select_live_page_drain(protected_hit_count=3)

        self.assertEqual(selection.page_ranges, [(0, 1)])
        self.assertEqual(selection.node_ids, [1])

    def test_live_page_reclaim_iteratively_exposes_parent_in_same_call(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        parent = _node(1, list(range(0, 8)), hit_count=1)
        child = _node(2, list(range(8, 16)), hit_count=0)
        parent.parent = cache.root_node
        cache.root_node.children[(1,)] = parent
        child.parent = parent
        parent.children[(2,)] = child
        cache.evictable_host_leaves = {child}
        cache.get_child_key_fn = lambda key: (key.token_ids[0],)
        cache._record_remove_event = lambda node: None
        cache._update_host_leaf_status = lambda node: None

        live_slots = set(range(16))
        events = []
        def evict_host(indices):
            slots = indices.tolist()
            events.append(slots)
            live_slots.difference_update(slots)
            return len(slots)

        cache.cache_controller = SimpleNamespace(evict_host=evict_host)
        cache.token_to_kv_pool_host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=16,
            available_size=lambda: 16 if not live_slots else 0,
        )

        snapshot_calls = 0
        original_snapshots = cache._live_host_node_snapshots

        def counted_snapshots(**kwargs):
            nonlocal snapshot_calls
            snapshot_calls += 1
            return original_snapshots(**kwargs)

        cache._live_host_node_snapshots = counted_snapshots

        result = cache.reclaim_live_page_selection(protected_hit_count=3)

        self.assertEqual(result["delete_host_leaf_node_ids"], [2, 1])
        self.assertEqual(events, [list(range(8, 16)), list(range(0, 8))])
        self.assertEqual(result["allocator_free_slots"], 16)
        self.assertEqual(result["reuse_summary"]["newly_exposed_parent_deletions"], 1)
        self.assertEqual(result["reuse_summary"]["reclaimed_high_reuse"], 0)
        self.assertEqual(snapshot_calls, 1)
        self.assertGreaterEqual(result["timing_ms"]["total"], 0.0)

    def test_live_page_reclaim_can_drop_host_copy_while_preserving_gpu_leaf(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        leaf = _node(9, list(range(0, 16)))
        leaf.value = torch.tensor(list(range(100, 116)), dtype=torch.int64)
        leaf.parent = cache.root_node
        cache.root_node.children[(9,)] = leaf
        cache.evictable_host_leaves = set()
        cache.get_child_key_fn = lambda key: (key.token_ids[0],)
        removed = []
        cache._record_remove_event = lambda node: removed.append(node.id)
        cache._update_host_leaf_status = lambda node: None

        events = []
        cache.cache_controller = SimpleNamespace(
            evict_host=lambda indices: events.append(indices.tolist()) or len(indices)
        )
        cache.token_to_kv_pool_host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=16,
            available_size=lambda: 16 if leaf.host_value is None else 0,
        )

        result = cache.reclaim_live_page_selection(protected_hit_count=3)

        self.assertEqual(result["drop_host_copy_node_ids"], [9])
        self.assertEqual(events, [list(range(0, 16))])
        self.assertEqual(removed, [])
        self.assertIsNone(leaf.host_value)
        self.assertIsNotNone(leaf.value)
        self.assertIn((9,), cache.root_node.children)

    def test_live_reclaim_streams_each_new_complete_page_without_waiting_for_batch_end(self):
        """A complete page is handed off as soon as its leaf is removed."""
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        first = _node(1, list(range(0, 16)))
        second = _node(2, list(range(16, 32)))
        for node in (first, second):
            node.parent = cache.root_node
            cache.root_node.children[(node.id,)] = node
        cache.evictable_host_leaves = {first, second}
        cache.get_child_key_fn = lambda key: (key.token_ids[0],)
        cache._record_remove_event = lambda node: None
        cache._update_host_leaf_status = lambda node: None

        live_slots = set(range(32))
        cache.cache_controller = SimpleNamespace(
            evict_host=lambda indices: live_slots.difference_update(indices.tolist())
            or len(indices)
        )
        cache.token_to_kv_pool_host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=32,
            available_size=lambda: 32 - len(live_slots),
        )
        streamed = []

        result = cache.reclaim_live_page_selection(
            protected_hit_count=3,
            on_allocator_pages_ready=lambda slots: streamed.append(slots) or slots,
        )

        self.assertEqual(streamed, [16, 16])
        self.assertEqual(result["streamed_slots"], 32)
        self.assertEqual(result["allocator_free_slots"], 0)

    def test_initial_wave_accumulates_a_useful_batch_before_first_stream(self):
        """A one-page leaf must not terminate a large quota handoff wave."""
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache.root_node = TreeNode(id=0)
        cache.root_node.key = RadixKey([])
        first = _node(1, list(range(0, 16)))
        second = _node(2, list(range(16, 32)))
        for node in (first, second):
            node.parent = cache.root_node
            cache.root_node.children[(node.id,)] = node
        cache.evictable_host_leaves = {first, second}
        cache.get_child_key_fn = lambda key: (key.token_ids[0],)
        cache._record_remove_event = lambda node: None
        cache._update_host_leaf_status = lambda node: None

        live_slots = set(range(32))
        cache.cache_controller = SimpleNamespace(
            evict_host=lambda indices: live_slots.difference_update(indices.tolist())
            or len(indices)
        )
        cache.token_to_kv_pool_host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=32,
            available_size=lambda: 32 - len(live_slots),
        )
        streamed = []

        result = cache.reclaim_live_page_selection(
            protected_hit_count=3,
            initial_min_pages=2,
            on_allocator_pages_ready=lambda slots: streamed.append(slots) or slots,
        )

        self.assertEqual(streamed, [32])
        self.assertEqual(result["streamed_slots"], 32)
        self.assertEqual(result["stream_batches"], 1)

    def test_scheduler_target_shrink_uses_exact_page_budget(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        calls = []
        cache.token_to_kv_pool_host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=48,
            quota_target=lambda: 16,
            available_size=lambda: 0,
        )
        cache.reclaim_live_page_selection = (
            lambda **kwargs: calls.append(kwargs) or {"page_ranges": [(0, 2)]}
        )

        result = cache._apply_central_quota_target()

        self.assertIsNone(calls[0]["protected_fraction"])
        self.assertEqual(calls[0]["max_pages"], 2)
        self.assertEqual(calls[0]["selection_budget_ms"], 100.0)
        self.assertEqual(result["direction"], "shrink")

    def test_scheduler_returns_empty_pages_before_draining_live_kv(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        calls = []
        free_slots = [16]
        host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=48,
            quota_target=lambda: 16,
            available_size=lambda: free_slots[0],
        )

        def resize_quota(target):
            calls.append(("free_shrink", target))
            free_slots[0] = max(0, free_slots[0] - (host.active_size - target))
            host.active_size = target
            return target

        host.resize_quota = resize_quota
        cache.token_to_kv_pool_host = host
        cache.reclaim_live_page_selection = (
            lambda **kwargs: calls.append(("live_drain", kwargs))
            or {"page_ranges": [(1, 1)]}
        )

        cache._apply_central_quota_target()

        self.assertEqual(calls[0], ("free_shrink", 32))
        self.assertEqual(calls[1][0], "live_drain")
        self.assertIsNone(calls[1][1]["protected_fraction"])
        self.assertEqual(calls[1][1]["max_pages"], 1)

    def test_scheduler_defers_unchanged_no_complete_page_until_cache_changes(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache._central_host_leaf_epoch = 0
        cache._central_quota_wait_signature = None
        cache._central_quota_next_retry_at = 0.0
        cache.evictable_host_leaves = set()
        host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=48,
            quota_target=lambda: 16,
            available_size=lambda: 0,
        )
        cache.token_to_kv_pool_host = host
        calls = []
        cache.reclaim_live_page_selection = (
            lambda **kwargs: calls.append(kwargs)
            or {"page_ranges": [], "reason": "no_complete_page"}
        )

        first = cache._apply_central_quota_target()
        second = cache._apply_central_quota_target()
        cache._central_host_leaf_epoch += 1
        third = cache._apply_central_quota_target()

        self.assertEqual(first["reason"], "no_complete_page")
        self.assertEqual(second["state"], "waiting_for_page")
        self.assertEqual(third["reason"], "no_complete_page")
        self.assertEqual([call["max_pages"] for call in calls], [2, 2])

    def test_scheduler_throttles_grow_while_donor_is_unavailable(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache._central_grow_wait_target = None
        cache._central_grow_next_retry_at = 0.0
        host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=16,
            quota_target=lambda: 48,
            available_size=lambda: 0,
        )
        calls = []

        def resize_quota(_target):
            calls.append("grow")
            raise MemoryError("central host pool has insufficient free page-aligned bytes")

        host.resize_quota = resize_quota
        cache.token_to_kv_pool_host = host

        first = cache._apply_central_quota_target()
        second = cache._apply_central_quota_target()
        cache._central_grow_next_retry_at = 0.0
        third = cache._apply_central_quota_target()

        self.assertEqual(first["state"], "waiting_for_scrub")
        self.assertEqual(second["state"], "waiting_for_donor")
        self.assertEqual(third["state"], "waiting_for_scrub")
        self.assertEqual(calls, ["grow", "grow"])

    def test_scheduler_grows_one_clean_batch_before_reaching_final_target(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        cache.page_size = 16
        cache._central_grow_wait_target = None
        cache._central_grow_next_retry_at = 0.0
        host = SimpleNamespace(
            dynamic_page_leases=True,
            active_size=16,
            quota_target=lambda: 64,
            available_size=lambda: 0,
        )
        requested_targets = []

        def resize_quota(target):
            requested_targets.append(target)
            host.active_size = target
            return target

        host.resize_quota = resize_quota
        cache.token_to_kv_pool_host = host
        old = os.environ.get("SGLANG_CENTRAL_IO_QUOTA_BATCH_PAGES")
        os.environ["SGLANG_CENTRAL_IO_QUOTA_BATCH_PAGES"] = "1"
        try:
            result = cache._apply_central_quota_target()
        finally:
            if old is None:
                del os.environ["SGLANG_CENTRAL_IO_QUOTA_BATCH_PAGES"]
            else:
                os.environ["SGLANG_CENTRAL_IO_QUOTA_BATCH_PAGES"] = old

        self.assertEqual(requested_targets, [32])
        self.assertEqual(result["state"], "partial")
        self.assertEqual(result["capacity"], 32)

    def test_scheduler_batch_gib_uses_host_pool_geometry_not_cache_page_size(self):
        cache = HiRadixCache.__new__(HiRadixCache)
        # HiRadix and the host allocator deliberately have different page
        # coordinates in this regression: only host_pool.page_size controls
        # the quota handoff alignment.
        cache.page_size = 1
        host = SimpleNamespace(page_size=16, size_per_token=144 * 1024)
        old_gib = os.environ.get("SGLANG_CENTRAL_IO_QUOTA_BATCH_GIB")
        old_pages = os.environ.get("SGLANG_CENTRAL_IO_QUOTA_BATCH_PAGES")
        os.environ["SGLANG_CENTRAL_IO_QUOTA_BATCH_GIB"] = "0.5"
        os.environ.pop("SGLANG_CENTRAL_IO_QUOTA_BATCH_PAGES", None)
        try:
            batch = cache._central_quota_batch_slots(host, remaining_slots=100000)
        finally:
            if old_gib is None:
                del os.environ["SGLANG_CENTRAL_IO_QUOTA_BATCH_GIB"]
            else:
                os.environ["SGLANG_CENTRAL_IO_QUOTA_BATCH_GIB"] = old_gib
            if old_pages is not None:
                os.environ["SGLANG_CENTRAL_IO_QUOTA_BATCH_PAGES"] = old_pages

        expected = int(0.5 * (1024**3) // (144 * 1024))
        expected -= expected % 16
        self.assertEqual(batch, expected)


if __name__ == "__main__":
    unittest.main()
