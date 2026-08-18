"""CPU-only checks for the long-lived live-KV reclaim candidate index."""

from __future__ import annotations

import time
import unittest
from statistics import median

from sglang.srt.mem_cache.live_page_reclaim import (
    HostNodeSnapshot,
    IncrementalLeafReclaimQueue,
    PersistentLeafReclaimIndex,
    prefix_fingerprint,
)


def node(
    node_id,
    slots,
    *,
    hit=0,
    children=0,
    parent=None,
    reuse_key=None,
):
    return HostNodeSnapshot(
        node_id=node_id,
        host_slots=tuple(slots),
        hit_count=hit,
        child_count=children,
        host_ref_counter=0,
        evicted=True,
        parent_id=parent,
        reuse_key=reuse_key,
    )


class PersistentReclaimIndexTest(unittest.TestCase):
    def test_prefix_fingerprint_survives_node_split_but_respects_namespace(self):
        whole = prefix_fingerprint([1, 2, 3, 4], extra_key="adapter-A")
        rebuilt = prefix_fingerprint([1, 2] + [3, 4], extra_key="adapter-A")
        other_namespace = prefix_fingerprint([1, 2, 3, 4], extra_key="adapter-B")

        self.assertEqual(whole, rebuilt)
        self.assertNotEqual(whole, other_namespace)
    def test_child_completion_promotes_parent_without_rebuilding_index(self):
        index = PersistentLeafReclaimIndex(page_size=16)
        parent = node(1, range(0, 8), hit=1, children=1)
        child = node(2, range(8, 16), parent=1)
        index.upsert(parent, [0])
        index.upsert(child, [0])

        first = index.pop(protected_hit_count=3)
        self.assertEqual(first.node_id, 2)
        index.complete_delete(2)

        second = index.pop(protected_hit_count=3)
        self.assertEqual(second.node_id, 1)
        self.assertEqual(index.node_updates, 3)

    def test_ready_pages_are_maintained_from_safe_single_leaf_ownership(self):
        index = PersistentLeafReclaimIndex(page_size=16)
        parent = node(1, range(0, 8), children=1)
        child = node(2, range(8, 16), parent=1)
        index.upsert(parent, [0])
        index.upsert(child, [0])

        self.assertEqual(index.ready_page_count, 0)
        index.complete_delete(child.node_id)

        self.assertEqual(index.ready_page_count, 1)

    def test_high_reuse_node_is_left_in_heap_for_later_policy_relaxation(self):
        index = PersistentLeafReclaimIndex(page_size=16)
        low = node(1, range(0, 16), hit=0)
        high = node(2, range(16, 32), hit=7)
        index.upsert(low, [0])
        index.upsert(high, [1])

        self.assertEqual(index.pop(protected_hit_count=3).node_id, 1)
        index.complete_delete(1)
        self.assertIsNone(index.pop(protected_hit_count=3).node_id)
        self.assertEqual(index.pop(protected_hit_count=None).node_id, 2)

    def test_removing_a_detached_node_invalidates_its_ready_entry(self):
        """A normal cache eviction must not leave a reclaimable ghost node."""
        index = PersistentLeafReclaimIndex(page_size=16)
        detached = node(1, range(0, 16))
        index.upsert(detached, [0])
        index.remove(detached.node_id)

        self.assertIsNone(index.pop(protected_hit_count=3).node_id)

    def test_external_predicted_reuse_score_is_not_a_reclaim_input(self):
        with self.assertRaises(TypeError):
            node(
                1,
                range(0, 16),
                expected_reuse_value=12.0,
            )

    def test_recent_reclaim_loss_protects_same_prefix_without_relaxing_safety(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        recently_hurt = node(1, range(0, 16), hit=0, reuse_key="document-A")
        cheaper = node(2, range(16, 32), hit=3, reuse_key="document-B")
        index.upsert(recently_hurt, [0])
        index.upsert(cheaper, [1])

        index.record_reclaim_loss(
            reuse_key="document-A", reprefill_tokens=8192, now_s=10.0
        )

        self.assertEqual(index.pop(now_s=11.0).node_id, 2)
        self.assertEqual(index.pop(now_s=41.0).node_id, 1)

    def test_ghost_only_becomes_loss_when_the_prefix_is_reprefilled(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        index.mark_reclaimed(reuse_key="document-A", now_s=10.0)
        self.assertFalse(index.record_revisit(reuse_key="document-B", reprefill_tokens=64, now_s=11.0))
        self.assertTrue(index.record_revisit(reuse_key="document-A", reprefill_tokens=64, now_s=11.0))
        hurt = node(1, range(0, 16), hit=0, reuse_key="document-A")
        cheap = node(2, range(16, 32), hit=1, reuse_key="document-B")
        index.upsert(hurt, [0])
        index.upsert(cheap, [1])
        self.assertEqual(index.pop(now_s=12.0).node_id, 2)

    def test_unrevisited_ghost_expires_without_a_loss_record(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        index.mark_reclaimed(reuse_key="document-A", now_s=10.0)

        self.assertFalse(
            index.record_revisit(
                reuse_key="document-A", reprefill_tokens=64, now_s=40.0
            )
        )

    def test_persistent_index_avoids_per_shrink_slot_materialization(self):
        page_size = 16
        slot_count = 728_000
        node_count = 140
        slots_per_node = slot_count // node_count
        snapshots = []
        index = PersistentLeafReclaimIndex(page_size=page_size)
        for node_id in range(node_count):
            start = node_id * slots_per_node
            slots = tuple(range(start, start + slots_per_node))
            snapshot = node(node_id + 1, slots, hit=node_id % 5)
            snapshots.append(snapshot)

        started = time.perf_counter()
        one_shot = IncrementalLeafReclaimQueue(
            page_size=page_size,
            protected_hit_count=3,
            nodes=snapshots,
        )
        one_shot_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        for snapshot in snapshots:
            pages = range(
                snapshot.host_slots[0] // page_size,
                snapshot.host_slots[-1] // page_size + 1,
            )
            index.upsert(snapshot, pages)
        warm_index_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        action = index.pop(protected_hit_count=3)
        shrink_pop_ms = (time.perf_counter() - started) * 1000

        self.assertIsNotNone(action.node_id)
        self.assertGreater(one_shot_ms, shrink_pop_ms)
        print(
            "persistent_reclaim_index_ms "
            f"one_shot_build={one_shot_ms:.3f} "
            f"warm_index_build={warm_index_ms:.3f} "
            f"first_shrink_pop={shrink_pop_ms:.3f}"
        )

    def test_normal_node_update_cost_is_amortized(self):
        """Measure the extra bookkeeping paid on ordinary cache activity."""
        index = PersistentLeafReclaimIndex(page_size=16)
        pages = list(range(512))
        samples_ms = []
        for update_id in range(1_000):
            snapshot = node(
                1,
                (),
                hit=update_id % 8,
                children=0,
            )
            started = time.perf_counter()
            index.upsert(snapshot, pages)
            samples_ms.append((time.perf_counter() - started) * 1000)
        ordered = sorted(samples_ms)
        p99 = ordered[int(len(ordered) * 0.99) - 1]
        print(
            "persistent_reclaim_update_ms "
            f"p50={median(samples_ms):.4f} p99={p99:.4f}"
        )
        self.assertLess(median(samples_ms), 1.0)


if __name__ == "__main__":
    unittest.main()
