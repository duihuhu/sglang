"""Regression tests for freeing radix-split host KV in Central I/O."""

from __future__ import annotations

import threading
import unittest

import torch

from sglang.srt.mem_cache.memory_pool_host import (
    CentralIOMHATokenToKVPoolHost,
    _FreePageRanges,
)


class _RecordingClient:
    def __init__(self):
        self.released: list[list[tuple[int, int]]] = []

    def release_pages(
        self, ranges: list[tuple[int, int]], *, allow_already_released: bool = False
    ) -> None:
        self.released.append(ranges)


class CentralIOPartialPageFreeTest(unittest.TestCase):
    def _host(self) -> tuple[CentralIOMHATokenToKVPoolHost, _RecordingClient]:
        host = CentralIOMHATokenToKVPoolHost.__new__(CentralIOMHATokenToKVPoolHost)
        host.page_size = 4
        host.size = 8
        host.lock = threading.RLock()
        host.mem_state = torch.ones(8, dtype=torch.uint8)
        host._free_page_ranges = _FreePageRanges()
        host._draining_page_ranges = _FreePageRanges()
        # This fixture bypasses the production initializer. Mirror the
        # liveness state that ``free`` now updates while keeping pressure
        # logging outside this page-release unit test.
        host._live_host_pages = 2
        host._local_ready_pages = 0
        host._local_residency = None
        host._record_pressure = lambda _kind, _amount: None
        client = _RecordingClient()
        host.client = client
        return host, client

    def test_split_node_frees_page_only_after_its_last_slot_is_released(self):
        host, client = self._host()

        host.free(torch.tensor([0, 1], dtype=torch.int64))
        self.assertEqual(client.released, [])
        self.assertEqual(host.available_size(), 0)

        host.free(torch.tensor([2, 3], dtype=torch.int64))
        self.assertEqual(client.released, [[(0, 1)]])
        self.assertEqual(host.available_size(), 4)
        self.assertTrue(torch.equal(host.mem_state[:4], torch.zeros(4, dtype=torch.uint8)))


if __name__ == "__main__":
    unittest.main()
