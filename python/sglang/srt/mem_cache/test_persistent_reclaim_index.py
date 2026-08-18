"""CPU-only checks for the long-lived live-KV reclaim candidate index."""

from __future__ import annotations

import time
import unittest
from unittest.mock import patch
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
    lock_ref=0,
):
    return HostNodeSnapshot(
        node_id=node_id,
        host_slots=tuple(slots),
        hit_count=hit,
        child_count=children,
        host_ref_counter=0,
        evicted=True,
        lock_ref=lock_ref,
        parent_id=parent,
        reuse_key=reuse_key,
    )


class PersistentReclaimIndexTest(unittest.TestCase):
    def test_inflight_lock_ref_cannot_be_reclaimed_before_host_write_ack(self):
        """A local maintenance wave must not steal a write-through host copy."""
        index = PersistentLeafReclaimIndex(page_size=16)
        in_flight = node(1, range(0, 16), lock_ref=1)
        idle = node(2, range(16, 32))
        index.upsert(in_flight, [0])
        index.upsert(idle, [1])

        self.assertEqual(index.pop().node_id, idle.node_id)
        index.complete_delete(idle.node_id)
        self.assertIsNone(index.pop().node_id)

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

    def test_ready_pages_wait_until_every_fragment_is_a_safe_leaf(self):
        index = PersistentLeafReclaimIndex(page_size=16)
        parent = node(1, range(0, 8), children=1)
        child = node(2, range(8, 16), parent=1)
        index.upsert(parent, [0])
        index.upsert(child, [0])

        self.assertEqual(index.ready_page_count, 0)
        index.complete_delete(child.node_id)

        self.assertEqual(index.ready_page_count, 1)

    def test_ready_page_can_contain_multiple_independently_safe_leaves(self):
        """A radix split must not turn an otherwise reclaimable page into a hole."""
        index = PersistentLeafReclaimIndex(page_size=16)
        left = node(1, range(0, 8), hit=0)
        right = node(2, range(8, 16), hit=1)

        index.upsert(left, [0])
        index.upsert(right, [0])

        self.assertEqual(index.ready_page_count, 1)
        self.assertEqual(index.ready_page_reclaim_score, 0.0)

    def test_ready_page_reclaim_score_clamps_tiny_negative_float_residue(self):
        """Fractional per-page feedback expiration must not publish -0.0 noise."""
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        index._ready_page_reclaim_score = -1e-9

        self.assertEqual(index.ready_page_reclaim_score, 0.0)

    def test_ready_page_fast_path_finishes_its_owners_after_lower_value_pages(self):
        index = PersistentLeafReclaimIndex(page_size=16)
        left = node(1, range(0, 8), hit=1)
        right = node(2, range(8, 16), hit=1)
        unrelated = node(3, range(16, 32), hit=0)
        index.upsert(left, [0])
        index.upsert(right, [0])
        index.upsert(unrelated, [1])

        # Page 1 has lower observed value, so it still goes first. Once that
        # page is removed, the fast path focuses on the owners of page 0
        # rather than falling back to an unrelated heap walk.
        action = index.pop_ready_page_action(now_s=10.0)
        self.assertEqual(action.node_id, unrelated.node_id)
        index.complete_delete(unrelated.node_id)
        action = index.pop_ready_page_action(now_s=10.0)
        self.assertIn(action.node_id, {left.node_id, right.node_id})

    def test_ready_page_pop_uses_its_persistent_ordering_without_resorting_pages(self):
        """A serving-time ready pop must not rebuild an ordering of all pages."""
        index = PersistentLeafReclaimIndex(page_size=16)
        expensive = node(1, range(0, 16), hit=2)
        cheap = node(2, range(16, 32), hit=0)
        index.upsert(expensive, [0])
        index.upsert(cheap, [1])

        # Candidate maintenance may rank pages when their facts change.  The
        # serving-time pop itself must consume that durable ordering rather
        # than calling sorted() across the complete ready-page set again.
        with patch("builtins.sorted", side_effect=AssertionError("resorted ready pages")):
            action = index.pop_ready_page_action(now_s=10.0)

        self.assertEqual(action.node_id, cheap.node_id)

    def test_shared_or_referenced_fragment_keeps_the_whole_page_live(self):
        index = PersistentLeafReclaimIndex(page_size=16)
        safe = node(1, range(0, 8), hit=0)
        shared = node(2, range(8, 16), children=1)

        index.upsert(safe, [0])
        index.upsert(shared, [0])

        self.assertEqual(index.ready_page_count, 0)

    def test_storage_write_is_not_reported_ready_before_durable_ack(self):
        index = PersistentLeafReclaimIndex(
            page_size=16, requires_durable_storage=True
        )
        writing = HostNodeSnapshot(
            node_id=1,
            host_slots=tuple(range(16)),
            hit_count=0,
            child_count=0,
            host_ref_counter=0,
            evicted=True,
            storage_state="writing",
        )
        index.upsert(writing, [0])
        self.assertEqual(index.ready_page_count, 0)

        durable = HostNodeSnapshot(
            **{**writing.__dict__, "storage_state": "durable"}
        )
        index.upsert(durable, [0])
        self.assertEqual(index.ready_page_count, 1)

    def test_central_storage_requires_every_node_page_to_be_durable(self):
        """A partial node tail cannot make its shared host page reclaimable."""
        index = PersistentLeafReclaimIndex(
            page_size=16, requires_durable_storage=True
        )
        partial = HostNodeSnapshot(
            node_id=1,
            host_slots=tuple(range(32)),
            hit_count=0,
            child_count=0,
            host_ref_counter=0,
            evicted=True,
            storage_state="durable",
            durable_page_ids=frozenset({0}),
        )
        index.upsert(partial, [0, 1])
        self.assertEqual(index.ready_page_count, 0)

        complete = HostNodeSnapshot(
            **{**partial.__dict__, "durable_page_ids": frozenset({0, 1})}
        )
        index.upsert(complete, [0, 1])
        self.assertEqual(index.ready_page_count, 2)

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

    def test_recent_host_hit_protects_a_prefix_only_for_the_turnover_window(self):
        """A real host restore is local evidence, not permanent pinning."""
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        just_restored = node(1, range(0, 16), hit=0, reuse_key="document-A")
        older = node(2, range(16, 32), hit=1, reuse_key="document-B")
        index.upsert(just_restored, [0])
        index.upsert(older, [1])

        index.record_host_hit(
            reuse_key="document-A", hit_tokens=16, now_s=10.0
        )

        # Without the host-hit record, A's lower historical hit count would
        # make it the first victim.  A true current host reuse protects A,
        # so B is the lower-loss candidate.
        self.assertEqual(index.pop(now_s=11.0).node_id, older.node_id)
        self.assertEqual(index.pop(now_s=41.0).node_id, just_restored.node_id)

    def test_reclaim_loss_remains_stronger_than_a_recent_host_hit(self):
        """A costly re-prefill is the strongest observed local regret signal."""
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        regret = node(1, range(0, 16), hit=0, reuse_key="regret")
        hit = node(2, range(16, 32), hit=0, reuse_key="host-hit")
        cheap = node(3, range(32, 48), hit=0, reuse_key="cheap")
        index.upsert(regret, [0])
        index.upsert(hit, [1])
        index.upsert(cheap, [2])
        index.record_reclaim_loss(
            reuse_key="regret", reprefill_tokens=8192, now_s=10.0
        )
        index.record_host_hit(
            reuse_key="host-hit", hit_tokens=16, now_s=10.0
        )

        # The candidate with no current reuse evidence is reclaimed first;
        # the host hit is next; the proven reclaim loss remains last.
        self.assertEqual(index.pop(now_s=11.0).node_id, cheap.node_id)
        self.assertEqual(index.pop(now_s=11.0).node_id, hit.node_id)
        self.assertEqual(index.pop(now_s=11.0).node_id, regret.node_id)

    def test_real_reclaim_loss_outranks_immediate_page_yield(self):
        """Fast page yield must not defeat observed reuse loss.

        The local maintainer is allowed to use an immediately reclaimable
        complete page as a tie-breaker.  It must first prefer a leaf that has
        not already demonstrated a costly short-term re-prefill; otherwise
        the watermark worker would repeatedly trade valuable reuse for a
        slightly faster clean page.
        """
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        hurt_but_page_complete = node(
            1, range(0, 16), hit=0, reuse_key="recently-hurt"
        )
        cheap_but_page_mixed = node(
            2, range(16, 24), hit=1, reuse_key="cheap"
        )
        mixed_neighbor = node(3, range(24, 32), hit=1, reuse_key="neighbor")
        index.upsert(hurt_but_page_complete, [0])
        index.upsert(cheap_but_page_mixed, [1])
        index.upsert(mixed_neighbor, [1])
        index.record_reclaim_loss(
            reuse_key="recently-hurt", reprefill_tokens=8192, now_s=10.0
        )

        # Node 1 yields a complete page immediately.  Node 2 does not, but
        # it is the lower-loss candidate and must be selected first.
        self.assertEqual(index.pop(now_s=11.0).node_id, cheap_but_page_mixed.node_id)

    def test_reclaim_loss_is_ranked_per_page_not_added_to_hit_count(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        # ``wide`` had a larger absolute re-prefill loss, but it returns 64
        # pages. Its observed cost per recovered page is lower than ``small``.
        wide = node(1, range(0, 16 * 64), hit=0, reuse_key="wide")
        small = node(2, range(16 * 64, 16 * 65), hit=0, reuse_key="small")
        index.upsert(wide, range(64))
        index.upsert(small, [64])
        index.record_reclaim_loss(
            reuse_key="wide", reprefill_tokens=640, now_s=10.0
        )
        index.record_reclaim_loss(
            reuse_key="small", reprefill_tokens=40, now_s=10.0
        )

        self.assertEqual(index.pop(now_s=11.0).node_id, wide.node_id)

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

    def test_reclaimed_prefix_revisit_matches_later_prompt_extension(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        old_prefix = [10, 11, 12, 13]
        extended_prompt = old_prefix + [14, 15]
        index.mark_reclaimed(
            reuse_key=prefix_fingerprint(old_prefix),
            token_ids=old_prefix,
            now_s=10.0,
        )

        reprefilled = index.record_revisit(
            reuse_key=prefix_fingerprint(extended_prompt),
            token_ids=extended_prompt,
            reprefill_tokens=len(extended_prompt),
            now_s=11.0,
        )

        self.assertEqual(reprefilled, len(old_prefix))
        observed = index.feedback_observability(now_s=12.0)
        self.assertEqual(observed["revisit_matches"], 1)
        self.assertEqual(observed["revisit_misses"], 0)
        self.assertEqual(observed["active_loss_records"], 1)

    def test_exact_revisit_key_uses_reclaimed_prefix_length_not_current_prompt_length(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        old_prefix = [10, 11, 12, 13]
        extended_prompt = old_prefix + [14, 15, 16]
        reuse_key = prefix_fingerprint(old_prefix)
        index.mark_reclaimed(
            reuse_key=reuse_key,
            token_ids=old_prefix,
            now_s=10.0,
        )

        reprefilled = index.record_revisit(
            reuse_key=reuse_key,
            token_ids=extended_prompt,
            reprefill_tokens=len(extended_prompt),
            now_s=11.0,
        )

        self.assertEqual(reprefilled, len(old_prefix))

    def test_overlapping_reclaimed_prefixes_are_not_counted_twice(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        short_prefix = [10, 11]
        long_prefix = short_prefix + [12, 13]
        extended_prompt = long_prefix + [14, 15]
        index.mark_reclaimed(
            reuse_key=prefix_fingerprint(short_prefix),
            token_ids=short_prefix,
            now_s=10.0,
        )
        index.mark_reclaimed(
            reuse_key=prefix_fingerprint(long_prefix),
            token_ids=long_prefix,
            now_s=10.0,
        )

        reprefilled = index.record_revisit(
            reuse_key=prefix_fingerprint(long_prefix),
            token_ids=extended_prompt,
            reprefill_tokens=len(extended_prompt),
            now_s=11.0,
        )
        duplicate = index.record_revisit(
            reuse_key=prefix_fingerprint(short_prefix),
            token_ids=short_prefix,
            reprefill_tokens=len(short_prefix),
            now_s=12.0,
        )

        self.assertEqual(reprefilled, len(long_prefix))
        self.assertEqual(duplicate, 0)
        observed = index.feedback_observability(now_s=13.0)
        self.assertEqual(observed["active_ghosts"], 0)
        self.assertEqual(observed["active_loss_records"], 1)

    def test_feedback_observability_distinguishes_ghost_miss_match_and_candidate(self):
        """Runtime audits must reveal which feedback boundary loses evidence."""
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        index.mark_reclaimed(reuse_key="document-A", now_s=10.0)

        self.assertFalse(
            index.record_revisit(
                reuse_key="document-B", reprefill_tokens=64, now_s=11.0
            )
        )
        self.assertTrue(
            index.record_revisit(
                reuse_key="document-A", reprefill_tokens=64, now_s=11.0
            )
        )
        index.upsert(node(1, range(0, 16), reuse_key="document-A"), [0])

        observed = index.feedback_observability(now_s=12.0)

        self.assertEqual(observed["ghost_marks"], 1)
        self.assertEqual(observed["revisit_matches"], 1)
        self.assertEqual(observed["revisit_misses"], 1)
        self.assertEqual(observed["active_ghosts"], 0)
        self.assertEqual(observed["active_loss_records"], 1)
        self.assertEqual(observed["feedback_candidate_nodes"], 1)

    def test_unrevisited_ghost_expires_without_a_loss_record(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=30.0)
        index.mark_reclaimed(reuse_key="document-A", now_s=10.0)

        self.assertFalse(
            index.record_revisit(
                reuse_key="document-A", reprefill_tokens=64, now_s=40.0
            )
        )

    def test_turnover_update_can_expire_a_ghost_without_rebuilding_the_queue(self):
        index = PersistentLeafReclaimIndex(page_size=16, ghost_ttl_s=60.0)
        index.mark_reclaimed(reuse_key="document-A", now_s=10.0)
        index.set_ghost_ttl_s(5.0, now_s=16.0)

        self.assertFalse(
            index.record_revisit(
                reuse_key="document-A", reprefill_tokens=64, now_s=16.0
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
