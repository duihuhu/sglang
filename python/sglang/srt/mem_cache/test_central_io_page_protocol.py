"""Unit tests for Central I/O's page-granular reservation protocol.

These tests intentionally avoid CUDA.  They exercise the agent-side state
machine that is used by the real SGLang adapter after its slot-index API
boundary has compressed an allocation into full KV pages.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys
import threading
import tempfile
import unittest
from multiprocessing.connection import Client, Listener

import torch


_SPEC = importlib.util.spec_from_file_location(
    "central_io_page_protocol", Path(__file__).with_name("central_io.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CentralIOAgent = _MODULE.CentralIOAgent
IPC_AUTHKEY = _MODULE.IPC_AUTHKEY
_HostSegment = _MODULE._HostSegment
_ModelState = _MODULE._ModelState
_PageRangeSet = _MODULE._PageRangeSet


class CentralIOPageProtocolTest(unittest.TestCase):
    page_size = 16

    def setUp(self):
        self.agent = CentralIOAgent.__new__(CentralIOAgent)
        segment = _HostSegment(
            segment_id=0,
            byte_offset=0,
            allocated_bytes=0,
            logical_start=0,
            slot_count=113 * self.page_size,
            host_k=None,
            host_v=None,
        )
        self.state = _ModelState(
            model_id="model",
            device=0,
            max_capacity=113 * self.page_size,
            segment_tokens=113 * self.page_size,
            k_ptrs=None,
            v_ptrs=None,
            opened_bases={},
            layer_count=2,
            head_count=1,
            head_dim=4,
            dtype=torch.float16,
            reserved=set(),
            segments={0: segment},
            segment_starts=[0],
            segments_by_start={0: 0},
            active_capacity=113 * self.page_size,
            page_size=self.page_size,
        )
        self.agent._models = {"model": self.state}
        self.agent._lock = threading.RLock()
        self.state.dynamic_page_leases = True

    def test_one_extent_records_one_page_range_not_113_page_records(self):
        self.agent._handle(
            {"op": "reserve_pages", "model_id": "model", "ranges": [[0, 113]]}
        )

        segment = self.state.segments[0]
        self.assertEqual(self.state.reserved_pages.copy_ranges(), [[0, 113]])
        self.assertEqual(segment.reserved_pages.copy_ranges(), [[0, 113]])
        self.assertEqual(self.state.reserved, set())
        self.assertEqual(segment.reserved_slots, set())

        self.agent._handle(
            {"op": "release_pages", "model_id": "model", "ranges": [[0, 113]]}
        )
        self.assertEqual(self.state.reserved_pages.copy_ranges(), [])
        self.assertEqual(segment.reserved_pages.copy_ranges(), [])

    def test_release_splits_range_without_expanding_to_page_ids(self):
        self.agent._handle(
            {"op": "reserve_pages", "model_id": "model", "ranges": [[0, 113]]}
        )
        self.agent._handle(
            {"op": "release_pages", "model_id": "model", "ranges": [[20, 40]]}
        )

        segment = self.state.segments[0]
        self.assertEqual(self.state.reserved_pages.copy_ranges(), [[0, 20], [60, 53]])
        self.assertEqual(segment.reserved_pages.copy_ranges(), [[0, 20], [60, 53]])
        self.assertEqual(segment.dirty_pages.copy_ranges(), [[20, 40]])

    def test_one_logical_range_crossing_views_becomes_two_local_ranges(self):
        first = self.state.segments[0]
        second = _HostSegment(
            segment_id=1,
            byte_offset=0,
            allocated_bytes=0,
            logical_start=113 * self.page_size,
            slot_count=113 * self.page_size,
            host_k=None,
            host_v=None,
        )
        self.state.max_capacity = 226 * self.page_size
        self.state.segments[1] = second
        self.state.segment_starts.append(second.logical_start)
        self.state.segments_by_start[second.logical_start] = 1

        self.agent._handle(
            {"op": "reserve_pages", "model_id": "model", "ranges": [[100, 26]]}
        )

        self.assertEqual(self.state.reserved_pages.copy_ranges(), [[100, 26]])
        self.assertEqual(first.reserved_pages.copy_ranges(), [[100, 13]])
        self.assertEqual(second.reserved_pages.copy_ranges(), [[0, 13]])

    def test_unix_socket_keeps_range_payload_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            socket_path = os.path.join(directory, "central.sock")
            try:
                listener = Listener(socket_path, family="AF_UNIX", authkey=IPC_AUTHKEY)
            except PermissionError:
                self.skipTest("the local filesystem sandbox forbids AF_UNIX bind")

            def serve_one_connection():
                self.agent._serve_connection(listener.accept())

            worker = threading.Thread(target=serve_one_connection, daemon=True)
            worker.start()
            connection = Client(socket_path, family="AF_UNIX", authkey=IPC_AUTHKEY)
            connection.send(
                {"op": "reserve_pages", "model_id": "model", "ranges": [[20, 40]]}
            )
            self.assertTrue(connection.recv()["ok"])
            connection.send(
                {"op": "release_pages", "model_id": "model", "ranges": [[20, 40]]}
            )
            self.assertTrue(connection.recv()["ok"])
            connection.close()
            worker.join(timeout=1)
            listener.close()

        self.assertEqual(self.state.reserved_pages.copy_ranges(), [])
        self.assertEqual(self.state.segments[0].dirty_pages.copy_ranges(), [[20, 40]])

    def test_page_protocol_rejects_duplicate_or_unreserved_pages(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.agent._handle(
                {
                    "op": "reserve_pages",
                    "model_id": "model",
                    "ranges": [[0, 2], [1, 1]],
                }
            )
        with self.assertRaisesRegex(ValueError, "unreserved"):
            self.agent._handle(
                {"op": "release_pages", "model_id": "model", "ranges": [[0, 1]]}
            )

    def test_layout_reports_live_and_dirty_pages_separately(self):
        self.agent._handle(
            {"op": "reserve_pages", "model_id": "model", "ranges": [[0, 3]]}
        )
        self.agent._handle(
            {"op": "release_pages", "model_id": "model", "ranges": [[0, 1]]}
        )

        layout = self.agent._handle({"op": "describe_layout", "model_id": "model"})

        self.assertEqual(layout["segments"][0]["reserved_page_ranges"], [[1, 2]])
        self.assertEqual(layout["segments"][0]["dirty_page_ranges"], [[0, 1]])

    def test_quota_target_is_agent_owned_and_page_aligned(self):
        target = 96 * self.page_size
        response = self.agent._handle(
            {"op": "set_quota_target", "model_id": "model", "target_capacity": target}
        )
        self.assertEqual(response["target_capacity"], target)
        current = self.agent._handle({"op": "quota_target", "model_id": "model"})
        self.assertEqual(current["target_capacity"], target)
        with self.assertRaisesRegex(ValueError, "page aligned"):
            self.agent._handle(
                {"op": "set_quota_target", "model_id": "model", "target_capacity": target + 1}
            )


if __name__ == "__main__":
    unittest.main()
