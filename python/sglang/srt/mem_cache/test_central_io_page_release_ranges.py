"""Regression tests for page-complete Central I/O host releases."""

from __future__ import annotations

import time
import unittest

import torch

from sglang.srt.mem_cache.memory_pool_host import CentralIOMHATokenToKVPoolHost


class CentralIOPageReleaseRangesTest(unittest.TestCase):
    def _pool(self, *, size: int, page_size: int):
        pool = CentralIOMHATokenToKVPoolHost.__new__(CentralIOMHATokenToKVPoolHost)
        pool.size = size
        pool.page_size = page_size
        pool.mem_state = torch.zeros(size, dtype=torch.int32)
        return pool

    def test_page_size_one_large_contiguous_release_is_one_fast_range(self):
        """Default SGLang page_size=1 must not reintroduce per-slot control work."""
        pool = self._pool(size=16_384, page_size=1)
        slots = torch.arange(8_192, dtype=torch.int64)
        pool.mem_state[slots] = 1

        started = time.perf_counter()
        ranges = pool._fully_released_page_ranges(slots)
        elapsed_s = time.perf_counter() - started

        self.assertEqual(ranges, [(0, 8_192)])
        self.assertLess(
            elapsed_s,
            0.20,
            f"page_size=1 release regressed to per-slot work: {elapsed_s:.3f}s",
        )

    def test_partial_page_stays_owned_until_its_last_live_slot_is_released(self):
        pool = self._pool(size=32, page_size=16)
        pool.mem_state[:16] = 1

        self.assertEqual(
            pool._fully_released_page_ranges(torch.arange(0, 8, dtype=torch.int64)),
            [],
        )
        pool.mem_state[:8] = 0
        self.assertEqual(
            pool._fully_released_page_ranges(torch.arange(8, 16, dtype=torch.int64)),
            [(0, 1)],
        )


if __name__ == "__main__":
    unittest.main()
