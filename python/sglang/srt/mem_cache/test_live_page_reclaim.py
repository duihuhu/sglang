"""Unit tests for page-level live-KV reclaim selection.

The policy may use radix-node value information, but a SGLang KV page is the
smallest unit that can be handed to another model.  A protected node fragment
therefore blocks its entire page, not its neighbouring pages.
"""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.live_page_reclaim import (
    HostNodeSnapshot,
    classify_reclaimable_pages,
    select_page_drain_candidates,
)


class LivePageReclaimTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
