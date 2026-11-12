import asyncio
import os
import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.layers.rdma_comm import (
    _BufferPool,
    _UcxP2PCommunicator,
    _as_uint8_numpy_view,
    _cross_node_experimental_enabled,
    _decode_meta,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=5, suite="stage-b-test-small-1-gpu")


class _UnusedBridge:
    _loop = None


class _RecordingEndpoint:
    def __init__(self, incoming=None):
        self.sent = []
        self.incoming = list(incoming or [])
        self.recv_payload = None

    async def send(self, buffer):
        self.sent.append(buffer.copy() if isinstance(buffer, np.ndarray) else buffer)

    async def recv(self, buffer):
        source = self.incoming.pop(0)
        if isinstance(buffer, np.ndarray):
            np.copyto(buffer, source)
        else:
            buffer.copy_(source)
        self.recv_payload = buffer


def _make_p2p(host_staging, pinned_staging=False, experimental=True):
    with patch.dict(
        os.environ,
        {
            "AFD_CROSS_NODE_EXPERIMENTAL": "1" if experimental else "0",
            "AFD_UCX_HOST_STAGING": "1" if host_staging else "0",
            "AFD_UCX_PINNED_STAGING": "1" if pinned_staging else "0",
        },
    ):
        return _UcxP2PCommunicator(
            is_ffn=False,
            local_rank=0,
            peer_ffn_rank=0,
            base_port=25000,
            ffn_host="localhost",
            timeout=1,
            bridge=_UnusedBridge(),
            pool=_BufferPool(),
            device="cuda:0",
        )


class TestUcxHostStaging(CustomTestCase):
    def test_experimental_switch_defaults_off_and_gates_staging(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertFalse(_cross_node_experimental_enabled())
        comm = _make_p2p(host_staging=True, pinned_staging=True, experimental=False)
        self.assertFalse(comm._cross_node_experimental)
        self.assertFalse(comm._host_staging)
        self.assertFalse(comm._pinned_staging)

    def test_default_host_wire_is_pageable_numpy(self):
        comm = _make_p2p(host_staging=True)
        wire = comm._allocate_host_wire(24)
        tensor = comm._wire_tensor(wire, (3, 4), torch.bfloat16)

        self.assertIsInstance(wire, np.ndarray)
        self.assertEqual(wire.dtype, np.uint8)
        self.assertEqual(wire.nbytes, 24)
        self.assertFalse(tensor.is_pinned())
        self.assertEqual(tensor.dtype, torch.bfloat16)
        self.assertEqual(tuple(tensor.shape), (3, 4))
        self.assertEqual(tensor.data_ptr(), wire.ctypes.data)

    def test_bfloat16_wire_round_trip_preserves_raw_bytes(self):
        comm = _make_p2p(host_staging=True)
        source = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        wire = comm._allocate_host_wire(source.numel() * source.element_size())
        np.copyto(wire, _as_uint8_numpy_view(source))
        restored = comm._wire_tensor(wire, tuple(source.shape), source.dtype)

        self.assertTrue(torch.equal(restored, source))
        np.testing.assert_array_equal(wire, _as_uint8_numpy_view(source))

    def test_experimental_send_protocol_and_metadata_cache(self):
        comm = _make_p2p(host_staging=True)
        comm._endpoint = _RecordingEndpoint()
        payload = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        asyncio.run(comm._async_send(comm._prepare_send_buffer(payload), 7))
        self.assertEqual(len(comm._endpoint.sent), 3)
        flag, meta, sent_payload = comm._endpoint.sent
        np.testing.assert_array_equal(flag, np.array([1], dtype=np.uint8))
        self.assertEqual(_decode_meta(meta), ((3, 4), torch.float32, 7))
        np.testing.assert_array_equal(sent_payload, _as_uint8_numpy_view(payload))

        asyncio.run(comm._async_send(comm._prepare_send_buffer(payload), 7))
        second_flag, second_payload = comm._endpoint.sent[3:]
        np.testing.assert_array_equal(second_flag, np.array([0], dtype=np.uint8))
        np.testing.assert_array_equal(second_payload, _as_uint8_numpy_view(payload))

        comm.reset_metadata_cache()
        asyncio.run(comm._async_send(comm._prepare_send_buffer(payload), 7))
        reset_flag, reset_meta, reset_payload = comm._endpoint.sent[5:]
        np.testing.assert_array_equal(reset_flag, np.array([1], dtype=np.uint8))
        self.assertEqual(_decode_meta(reset_meta), ((3, 4), torch.float32, 7))
        np.testing.assert_array_equal(reset_payload, _as_uint8_numpy_view(payload))

    def test_async_send_bfloat16_uses_zero_copy_uint8_payload(self):
        comm = _make_p2p(host_staging=True)
        comm._endpoint = _RecordingEndpoint()
        payload = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)

        staged = comm._prepare_send_buffer(payload)
        asyncio.run(comm._async_send(staged, original_num_tokens=7))
        flag, meta, sent_payload = comm._endpoint.sent

        np.testing.assert_array_equal(flag, np.array([1], dtype=np.uint8))
        self.assertEqual(sent_payload.dtype, np.uint8)
        self.assertEqual(sent_payload.nbytes, payload.numel() * payload.element_size())
        np.testing.assert_array_equal(sent_payload, _as_uint8_numpy_view(payload))
        self.assertEqual(_decode_meta(meta), ((3, 4), torch.bfloat16, 7))

    def test_async_recv_bfloat16_uses_flag_meta_payload_protocol(self):
        comm = _make_p2p(host_staging=True)
        source = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        meta = np.zeros(8, dtype=np.int64)
        meta[:5] = [2, 3, 4, 1, 7]
        comm._endpoint = _RecordingEndpoint(
            [np.array([1], dtype=np.uint8), meta, _as_uint8_numpy_view(source)]
        )

        received, original_tokens = asyncio.run(comm._async_recv())

        self.assertEqual(comm._endpoint.recv_payload.dtype, np.uint8)
        self.assertEqual(received.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(received, source))
        self.assertEqual(original_tokens, 7)

    def test_recv_without_cached_metadata_raises_clear_error(self):
        comm = _make_p2p(host_staging=True)
        comm._endpoint = _RecordingEndpoint([np.array([0], dtype=np.uint8)])

        with self.assertRaisesRegex(RuntimeError, "flag=0.*cached metadata"):
            asyncio.run(comm._async_recv())

    def test_sync_preparation_reuses_wire_for_same_signature(self):
        comm = _make_p2p(host_staging=True)
        first = comm._prepare_send_buffer(
            torch.ones((2, 3), dtype=torch.float32), cache_host_wire=True
        )
        second = comm._prepare_send_buffer(
            torch.zeros((2, 3), dtype=torch.float32), cache_host_wire=True
        )
        self.assertIs(first[0], second[0])
        np.testing.assert_array_equal(
            second[0], _as_uint8_numpy_view(torch.zeros((2, 3)))
        )

        changed_shape = comm._prepare_send_buffer(
            torch.zeros((3, 2), dtype=torch.float32), cache_host_wire=True
        )
        self.assertIsNot(changed_shape[0], second[0])
        changed_dtype = comm._prepare_send_buffer(
            torch.zeros((3, 2), dtype=torch.float16), cache_host_wire=True
        )
        self.assertIsNot(changed_dtype[0], changed_shape[0])

    def test_nonblocking_preparation_does_not_reuse_wire(self):
        comm = _make_p2p(host_staging=True)
        payload = torch.ones((2, 3), dtype=torch.float32)
        first = comm._prepare_send_buffer(payload, cache_host_wire=False)
        second = comm._prepare_send_buffer(payload, cache_host_wire=False)
        self.assertIsNot(first[0], second[0])
        self.assertNotEqual(first[0].ctypes.data, second[0].ctypes.data)

    def test_reset_preserves_send_wire_but_clears_metadata_and_recv_buffers(self):
        comm = _make_p2p(host_staging=True)
        cached = comm._prepare_send_buffer(
            torch.empty((2, 2)), cache_host_wire=True
        )[0]
        comm._send_meta_signature = ((2, 2), torch.float32, 0)
        comm._recv_meta_signature = ((2, 2), torch.float32, 0)
        comm._recv_host_buffer = torch.empty((2, 2))
        comm._recv_buffer = object()

        comm.reset_metadata_cache()

        self.assertIsNone(comm._send_meta_signature)
        self.assertIsNone(comm._recv_meta_signature)
        self.assertIs(comm._send_host_wire, cached)
        self.assertIsNone(comm._recv_host_buffer)
        self.assertIsNone(comm._recv_buffer)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_send_and_receive_stage_through_pageable_numpy(self):
        comm = _make_p2p(host_staging=True)
        source = torch.arange(8, dtype=torch.float32, device="cuda").reshape(2, 4)
        wire, shape, dtype = comm._prepare_send_buffer(source)
        staged = comm._wire_tensor(wire, shape, dtype)
        self.assertIsInstance(wire, np.ndarray)
        self.assertFalse(staged.is_pinned())

        meta = np.zeros(8, dtype=np.int64)
        meta[:5] = [2, 2, 4, 2, 9]
        comm._endpoint = _RecordingEndpoint(
            [np.array([1], dtype=np.uint8), meta, wire.copy()]
        )
        received, original_tokens = asyncio.run(comm._async_recv())
        self.assertEqual(received.device.type, "cpu")
        self.assertFalse(received.is_pinned())
        self.assertEqual(original_tokens, 9)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_pinned_wire_numpy_keeps_owner_and_pinned_storage(self):
        comm = _make_p2p(host_staging=True, pinned_staging=True)
        wire = comm._allocate_host_wire(16)

        self.assertIsNotNone(wire.base)
        self.assertIsInstance(wire.base, torch.Tensor)
        self.assertTrue(wire.base.is_pinned())
        self.assertTrue(torch.from_numpy(wire).is_pinned())


if __name__ == "__main__":
    unittest.main(verbosity=3)
