"""CPU-only tests for Central I/O's agent-owned raw page storage proxy."""

from __future__ import annotations

import ctypes
import importlib.util
import mmap
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest import mock

import torch


_SPEC = importlib.util.spec_from_file_location(
    "central_io_storage_proxy", Path(__file__).with_name("central_io.py")
)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
CentralIOAgent = _MODULE.CentralIOAgent
_ModelState = _MODULE._ModelState


class CentralIOStorageProxyTest(unittest.TestCase):
    page_size = 2
    slot_bytes = 2 * 2 * 1 * 4 * 2

    def setUp(self):
        self.pool_bytes = self.slot_bytes * 8
        self._backing_file = None
        if hasattr(os, "memfd_create"):
            self.fd = os.memfd_create("central-io-storage-proxy", os.MFD_CLOEXEC)
            self._close_fd = True
        else:
            # macOS has no memfd, but an anonymous temporary file gives the
            # same writable mmap semantics needed by this CPU-only state test.
            self._backing_file = tempfile.TemporaryFile()
            self.fd = self._backing_file.fileno()
            self._close_fd = False
        os.ftruncate(self.fd, self.pool_bytes)
        self.mapping = mmap.mmap(self.fd, self.pool_bytes, access=mmap.ACCESS_WRITE)
        self.agent = CentralIOAgent.__new__(CentralIOAgent)
        self.agent._mapping = self.mapping
        self.agent._address = ctypes.addressof(ctypes.c_char.from_buffer(self.mapping))
        self.agent._lock = threading.RLock()
        self.agent._next_segment_id = 0
        self.agent._init_dynamic_free_bytes(self.pool_bytes)
        self.storage_dir = tempfile.TemporaryDirectory()
        self.agent._init_storage_proxy(self.storage_dir.name)

    def tearDown(self):
        self.agent._close_storage_proxy()
        self.storage_dir.cleanup()
        self.mapping.close()
        if self._close_fd:
            os.close(self.fd)
        elif self._backing_file is not None:
            self._backing_file.close()

    def _state(self) -> _ModelState:
        state = _ModelState(
            model_id="model-a",
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
            dynamic_page_leases=True,
        )
        self.agent._add_dynamic_segment(state, logical_start=0, slot_count=8)
        return state

    def test_restore_operator_warmup_compiles_before_serving(self):
        self.agent._cuda_lock = threading.Lock()
        self.agent._registration_device = 1
        with (
            mock.patch.object(torch.cuda, "set_device") as set_device,
            mock.patch.object(_MODULE, "_load_restore_extension") as load_extension,
        ):
            self.agent.warmup_restore_operator()

        set_device.assert_called_once_with(1)
        load_extension.assert_called_once_with()

    def test_durable_write_then_read_round_trips_agent_owned_page_bytes(self):
        state = self._state()
        state.reserved_pages.add_range(1, 1)
        segment = next(iter(state.segments.values()))
        segment.reserved_pages.add_range(1, 1)
        page_offset = self.page_size * self.slot_bytes
        original = bytes(range(self.page_size * self.slot_bytes))
        self.mapping[page_offset : page_offset + len(original)] = original

        write = self.agent._storage_write_pages(state, ["prefix-page-1"], [(1, 1)])
        write_status = self.agent._wait_storage_operation(write["operation_id"], 1.0)
        self.assertEqual(write_status["state"], "durable")
        self.assertEqual(write_status["completed_pages"], 1)

        self.mapping[page_offset : page_offset + len(original)] = b"\0" * len(original)
        read = self.agent._storage_read_pages(state, ["prefix-page-1"], [(1, 1)])
        read_status = self.agent._wait_storage_operation(read["operation_id"], 1.0)
        self.assertEqual(read_status["state"], "ready")
        self.assertEqual(bytes(self.mapping[page_offset : page_offset + len(original)]), original)

    def test_write_rejects_a_page_not_live_in_the_model_lease(self):
        state = self._state()

        with self.assertRaisesRegex(ValueError, "live"):
            self.agent._storage_write_pages(state, ["not-live"], [(0, 1)])

    def test_missing_page_read_reports_a_failed_operation_without_modifying_host_bytes(self):
        state = self._state()
        state.reserved_pages.add_range(0, 1)
        segment = next(iter(state.segments.values()))
        segment.reserved_pages.add_range(0, 1)
        original = b"x" * (self.page_size * self.slot_bytes)
        self.mapping[: len(original)] = original

        read = self.agent._storage_read_pages(state, ["missing"], [(0, 1)])
        status = self.agent._wait_storage_operation(read["operation_id"], 1.0)

        self.assertEqual(status["state"], "failed")
        self.assertEqual(bytes(self.mapping[: len(original)]), original)

    def test_storage_probe_returns_only_the_contiguous_durable_prefix(self):
        state = self._state()
        state.reserved_pages.add_range(0, 1)
        segment = next(iter(state.segments.values()))
        segment.reserved_pages.add_range(0, 1)
        write = self.agent._storage_write_pages(state, ["present"], [(0, 1)])
        self.assertEqual(
            self.agent._wait_storage_operation(write["operation_id"], 1.0)["state"],
            "durable",
        )

        self.assertEqual(
            self.agent._storage_existing_prefix(state, ["present", "missing"]), 1
        )

    def test_control_protocol_exposes_only_agent_side_durable_ack(self):
        state = self._state()
        self.agent._models = {state.model_id: state}
        state.reserved_pages.add_range(0, 1)
        segment = next(iter(state.segments.values()))
        segment.reserved_pages.add_range(0, 1)

        write = self.agent._handle(
            {
                "op": "storage_write_pages",
                "model_id": state.model_id,
                "keys": ["control-page"],
                "page_ranges": [[0, 1]],
            }
        )
        status = self.agent._wait_storage_operation(write["operation_id"], 1.0)
        reported = self.agent._handle(
            {
                "op": "storage_status",
                "model_id": state.model_id,
                "operation_id": write["operation_id"],
            }
        )

        self.assertEqual(status["state"], "durable")
        self.assertEqual(reported["state"], "durable")
        self.assertEqual(reported["model_id"], state.model_id)

    def test_storage_plan_excludes_a_partial_tail_page(self):
        indices = torch.tensor(list(range(16)) + list(range(16, 24)))

        keys, ranges = _MODULE.complete_storage_page_plan(
            ["page-0", "partial-page-1"], indices, page_size=16
        )

        self.assertEqual(keys, ["page-0"])
        self.assertEqual(ranges, [(0, 1)])

    def test_storage_plan_keeps_only_complete_noncontiguous_pages_in_key_order(self):
        indices = torch.tensor(list(range(16, 32)) + list(range(64, 80)))

        keys, ranges = _MODULE.complete_storage_page_plan(
            ["page-1", "page-4"], indices, page_size=16
        )

        self.assertEqual(keys, ["page-1", "page-4"])
        self.assertEqual(ranges, [(1, 1), (4, 1)])


if __name__ == "__main__":
    unittest.main()
