"""Unit tests for page-level live-KV reclaim selection.

The policy may use radix-node value information, but a SGLang KV page is the
smallest unit that can be handed to another model.  A protected node fragment
therefore blocks its entire page, not its neighbouring pages.
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.live_page_reclaim import (
    HostNodeSnapshot,
    IncrementalLeafReclaimQueue,
    classify_reclaimable_pages,
    select_next_incremental_leaf_action,
    select_page_reclaim_actions,
    select_page_drain_candidates,
)


class LivePageReclaimTest(unittest.TestCase):
    def test_incremental_queue_uses_relative_protection_not_absolute_hit_cutoff(self):
        """The top reuse fraction stays protected even when its count changes."""
        queue = IncrementalLeafReclaimQueue(
            protected_hit_count=None,
            protected_fraction=1 / 3,
            nodes=[
                HostNodeSnapshot(1, tuple(range(0, 16)), 100, 0, 0, True),
                HostNodeSnapshot(2, tuple(range(16, 32)), 10, 0, 0, True),
                HostNodeSnapshot(3, tuple(range(32, 48)), 1, 0, 0, True),
            ],
        )

        self.assertEqual(queue.pop().node_id, 3)
        self.assertEqual(queue.pop().node_id, 2)
        self.assertIsNone(queue.pop().node_id)

    def test_incremental_queue_prefers_leaf_that_immediately_completes_page(self):
        """A low-value partial leaf must not delay an immediately usable page.

        Node 1 is cheaper to discard, but node 3 keeps page 0 live.  Node 2
        costs slightly more yet solely owns page 1, so deleting it gives the
        recipient one complete KV page right away.
        """
        queue = IncrementalLeafReclaimQueue(
            page_size=16,
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(1, tuple(range(0, 8)), 0, 0, 0, True),
                HostNodeSnapshot(2, tuple(range(16, 32)), 1, 0, 0, True),
                HostNodeSnapshot(3, tuple(range(8, 16)), 100, 1, 0, True),
            ],
        )

        first = queue.pop()

        self.assertEqual(first.node_id, 2)
        self.assertEqual(first.immediate_page_yield, 1)

    def test_incremental_queue_keeps_structural_leaf_after_page_yielding_work(self):
        """A partial leaf remains reclaimable after all immediate yields drain."""
        queue = IncrementalLeafReclaimQueue(
            page_size=16,
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(1, tuple(range(0, 8)), 0, 0, 0, True),
                HostNodeSnapshot(2, tuple(range(16, 32)), 1, 0, 0, True),
                HostNodeSnapshot(3, tuple(range(8, 16)), 100, 1, 0, True),
            ],
        )

        first = queue.pop()
        queue.complete_delete(first.node_id)
        second = queue.pop()

        self.assertEqual(first.node_id, 2)
        self.assertEqual((second.node_id, second.immediate_page_yield), (1, 0))

    def test_incremental_queue_promotes_parent_without_new_tree_scan(self):
        """Completing a child makes only its parent newly eligible."""
        queue = IncrementalLeafReclaimQueue(
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(
                    node_id=70,
                    host_slots=tuple(range(0, 8)),
                    hit_count=1,
                    child_count=1,
                    host_ref_counter=0,
                    evicted=True,
                    parent_id=None,
                ),
                HostNodeSnapshot(
                    node_id=71,
                    host_slots=tuple(range(8, 16)),
                    hit_count=0,
                    child_count=0,
                    host_ref_counter=0,
                    evicted=True,
                    parent_id=70,
                ),
            ],
        )

        first = queue.pop()
        self.assertEqual((first.node_id, first.action_kind), (71, "delete_host_leaf"))
        queue.complete_delete(71)

        second = queue.pop()
        self.assertEqual((second.node_id, second.action_kind), (70, "delete_host_leaf"))
        self.assertEqual(queue.snapshot_builds, 1)

    def test_protected_fragment_blocks_only_its_own_page(self):
        """A shared prefix in page 1 must not poison page 0 or page 2."""
        result = classify_reclaimable_pages(
            page_size=16,
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(
                    node_id=10,
                    host_slots=tuple(range(0, 16)),
                    hit_count=0,
                    child_count=0,
                    host_ref_counter=0,
                    evicted=True,
                ),
                # This exactly mirrors the observed mixed page shape:
                # low-value leaf tail | shared prefix | low-value leaf head.
                HostNodeSnapshot(
                    node_id=11,
                    host_slots=tuple(range(16, 27)),
                    hit_count=0,
                    child_count=0,
                    host_ref_counter=0,
                    evicted=True,
                ),
                HostNodeSnapshot(
                    node_id=12,
                    host_slots=(27, 28),
                    hit_count=27,
                    child_count=2,
                    host_ref_counter=0,
                    evicted=True,
                ),
                HostNodeSnapshot(
                    node_id=13,
                    host_slots=tuple(range(29, 32)),
                    hit_count=1,
                    child_count=0,
                    host_ref_counter=0,
                    evicted=True,
                ),
                HostNodeSnapshot(
                    node_id=14,
                    host_slots=tuple(range(32, 48)),
                    hit_count=1,
                    child_count=0,
                    host_ref_counter=0,
                    evicted=True,
                ),
            ],
        )

        self.assertEqual(result.reclaimable_page_ranges, [(0, 1), (2, 1)])
        self.assertEqual(result.page_reasons[1], {"shared_prefix": 1})
        self.assertEqual(result.reclaimable_pages, 2)
        self.assertEqual(result.blocked_pages, 1)

    def test_in_flight_node_blocks_a_page_even_when_its_hit_count_is_low(self):
        result = classify_reclaimable_pages(
            page_size=16,
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(
                    node_id=20,
                    host_slots=tuple(range(48, 64)),
                    hit_count=0,
                    child_count=0,
                    host_ref_counter=1,
                    evicted=True,
                )
            ],
        )

        self.assertEqual(result.reclaimable_page_ranges, [])
        self.assertEqual(result.page_reasons[3], {"in_flight": 1})

    def test_adjacent_reclaimable_pages_are_coalesced_only_after_selection(self):
        result = classify_reclaimable_pages(
            page_size=16,
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(30, tuple(range(0, 16)), 0, 0, 0, True),
                HostNodeSnapshot(31, tuple(range(16, 32)), 0, 0, 0, True),
                HostNodeSnapshot(32, tuple(range(64, 80)), 0, 0, 0, True),
            ],
        )

        self.assertEqual(result.reclaimable_page_ranges, [(0, 2), (4, 1)])

    def test_drain_selection_requires_full_page_coverage_by_selected_leaves(self):
        nodes = [
            HostNodeSnapshot(1, tuple(range(0, 16)), 0, 0, 0, True),
            # This leaf owns only part of page 1.
            HostNodeSnapshot(2, tuple(range(16, 24)), 0, 0, 0, True),
            # It completes page 1 but also reaches page 2, whose shared
            # prefix makes the whole node ineligible for a targeted drain.
            HostNodeSnapshot(3, tuple(range(24, 40)), 0, 0, 0, True),
            HostNodeSnapshot(4, tuple(range(40, 48)), 99, 1, 0, True),
        ]

        selection = select_page_drain_candidates(
            page_size=16, protected_hit_count=3, nodes=nodes
        )

        self.assertEqual(selection.page_ranges, [(0, 1)])
        self.assertEqual(selection.node_ids, [1])

    def test_drain_selection_honors_page_budget_without_splitting_a_leaf(self):
        nodes = [
            HostNodeSnapshot(1, tuple(range(0, 16)), 0, 0, 0, True),
            HostNodeSnapshot(2, tuple(range(16, 32)), 0, 0, 0, True),
        ]
        selection = select_page_drain_candidates(
            page_size=16, protected_hit_count=3, nodes=nodes, max_pages=1
        )

        self.assertEqual(selection.page_ranges, [(0, 1)])
        self.assertEqual(selection.node_ids, [1])

    def test_action_selection_drops_duplicate_host_copy_without_deleting_gpu_node(self):
        selection = select_page_reclaim_actions(
            page_size=16,
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(
                    node_id=41,
                    host_slots=tuple(range(0, 16)),
                    hit_count=0,
                    child_count=0,
                    host_ref_counter=0,
                    evicted=False,
                )
            ],
        )

        self.assertEqual(selection.page_ranges, [(0, 1)])
        self.assertEqual(selection.drop_host_copy_node_ids, [41])
        self.assertEqual(selection.delete_host_leaf_node_ids, [])

    def test_action_selection_cascades_from_deleted_child_to_new_leaf_parent(self):
        selection = select_page_reclaim_actions(
            page_size=16,
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(
                    node_id=50,
                    host_slots=tuple(range(0, 16)),
                    hit_count=1,
                    child_count=1,
                    host_ref_counter=0,
                    evicted=True,
                    parent_id=None,
                ),
                HostNodeSnapshot(
                    node_id=51,
                    host_slots=tuple(range(16, 32)),
                    hit_count=0,
                    child_count=0,
                    host_ref_counter=0,
                    evicted=True,
                    parent_id=50,
                ),
            ],
        )

        self.assertEqual(selection.page_ranges, [(0, 2)])
        self.assertEqual(selection.drop_host_copy_node_ids, [])
        self.assertEqual(selection.delete_host_leaf_node_ids, [51, 50])

    def test_action_selection_keeps_high_value_parent_after_low_value_child_is_deleted(self):
        selection = select_page_reclaim_actions(
            page_size=16,
            protected_hit_count=3,
            nodes=[
                HostNodeSnapshot(
                    node_id=60,
                    host_slots=tuple(range(0, 16)),
                    hit_count=3,
                    child_count=1,
                    host_ref_counter=0,
                    evicted=True,
                    parent_id=None,
                ),
                HostNodeSnapshot(
                    node_id=61,
                    host_slots=tuple(range(16, 32)),
                    hit_count=0,
                    child_count=0,
                    host_ref_counter=0,
                    evicted=True,
                    parent_id=60,
                ),
            ],
        )

        self.assertEqual(selection.page_ranges, [(1, 1)])
        self.assertEqual(selection.delete_host_leaf_node_ids, [61])

    def test_incremental_selector_exposes_parent_after_child_deletion(self):
        """A child may be worth deleting even before it completes a page."""
        child = HostNodeSnapshot(
            node_id=71,
            host_slots=tuple(range(8, 16)),
            hit_count=0,
            child_count=0,
            host_ref_counter=0,
            evicted=True,
            parent_id=70,
        )
        parent_before = HostNodeSnapshot(
            node_id=70,
            host_slots=tuple(range(0, 8)),
            hit_count=1,
            child_count=1,
            host_ref_counter=0,
            evicted=True,
        )
        first = select_next_incremental_leaf_action(
            protected_hit_count=3, nodes=[parent_before, child]
        )
        self.assertEqual((first.node_id, first.action_kind), (71, "delete_host_leaf"))

        parent_after = HostNodeSnapshot(
            node_id=70,
            host_slots=tuple(range(0, 8)),
            hit_count=1,
            child_count=0,
            host_ref_counter=0,
            evicted=True,
        )
        second = select_next_incremental_leaf_action(
            protected_hit_count=3, nodes=[parent_after]
        )
        self.assertEqual((second.node_id, second.action_kind), (70, "delete_host_leaf"))

    def test_incremental_selector_never_selects_the_radix_root(self):
        root = HostNodeSnapshot(
            node_id=0,
            host_slots=tuple(range(0, 16)),
            hit_count=0,
            child_count=0,
            host_ref_counter=0,
            evicted=True,
            is_root=True,
        )
        result = select_next_incremental_leaf_action(
            protected_hit_count=3, nodes=[root]
        )
        self.assertIsNone(result.node_id)
        self.assertEqual(result.blocker_counts, {"radix_root": 1})


if __name__ == "__main__":
    unittest.main()
