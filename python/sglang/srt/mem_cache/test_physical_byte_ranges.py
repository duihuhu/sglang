"""Tests for variable-size physical leases in the central pinned pool."""

from __future__ import annotations

import unittest

from sglang.srt.mem_cache.physical_byte_ranges import FreeByteRanges


class FreeByteRangesTest(unittest.TestCase):
    def test_released_middle_range_can_be_reassigned_without_whole_extent_handoff(self):
        free = FreeByteRanges([(0, 4096)])
        cold_left = free.allocate(1024, alignment=256)
        cold_middle = free.allocate(1024, alignment=256)
        cold_right = free.allocate(1024, alignment=256)

        self.assertEqual((cold_left, cold_middle, cold_right), (0, 1024, 2048))
        free.release(cold_middle, 1024)

        # A hot model may receive just B's cleaned middle physical range.  No
        # arbitrary 256MiB owner container needs to be handed over.
        self.assertEqual(free.allocate(1024, alignment=256), 1024)
        self.assertEqual(free.ranges, [(3072, 4096)])

    def test_release_coalesces_neighbouring_physical_ranges(self):
        free = FreeByteRanges()
        free.release(1024, 1024)
        free.release(0, 1024)
        free.release(2048, 1024)

        self.assertEqual(free.ranges, [(0, 3072)])


if __name__ == "__main__":
    unittest.main()
