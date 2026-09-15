import ctypes
import socket
import unittest
from concurrent.futures import ThreadPoolExecutor

import torch
from sglang.srt.layers.afd import FifoTensorCommunicator
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.layers.mooncake_comm import MooncakeTensorCommunicator
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class FakeMooncakeEngine:
    def __init__(self, session_id):
        self.session_id = session_id
        self.registered = {}
        self.deregistered = []
        self.transfers = []

    def get_session_id(self):
        return self.session_id

    def register(self, ptr, length):
        self.registered[ptr] = length
        return 0

    def deregister(self, ptr):
        self.deregistered.append(ptr)
        self.registered.pop(ptr, None)
        return 0

    def transfer_sync(self, session_id, source, destination, length):
        ctypes.memmove(destination, source, length)
        self.transfers.append((session_id, source, destination, length))
        return 0


def _free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_pair(
    *,
    buffer_bytes=16,
    ring_slots=2,
    timeout_ms=2000,
    tp_rank=0,
    tp_size=1,
    attn_port=None,
    ffn_port=None,
    channel=0,
):
    attn_port = attn_port or _free_port()
    ffn_port = ffn_port or _free_port()
    engines = (FakeMooncakeEngine("attn-session"), FakeMooncakeEngine("ffn-session"))
    common = {
        "peer_host": "127.0.0.1",
        "attn_control_port": attn_port,
        "ffn_control_port": ffn_port,
        "buffer_bytes": buffer_bytes,
        "ring_slots": ring_slots,
        "timeout_ms": timeout_ms,
        "tp_rank": tp_rank,
        "tp_size": tp_size,
        "channel": channel,
        "tensor_allocator": lambda size: torch.empty(size, dtype=torch.uint8),
    }
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(
                MooncakeTensorCommunicator,
                AFDPerspective.AFD_PERSPECTIVE_ATTN,
                engine=engines[0],
                **common,
            ),
            pool.submit(
                MooncakeTensorCommunicator,
                AFDPerspective.AFD_PERSPECTIVE_FFN,
                engine=engines[1],
                **common,
            ),
        ]
        comms = tuple(f.result(timeout=5) for f in futures)
    return comms, engines


class TestAFDMooncakeCommunicator(CustomTestCase):
    def test_implements_fifo_interface_and_duplex_loopback(self):
        (attn, ffn), engines = _make_pair()
        self.assertIsInstance(attn, FifoTensorCommunicator)
        try:
            a_to_f = torch.arange(12, dtype=torch.float32).reshape(3, 4)
            f_to_a = torch.arange(7, dtype=torch.int64)
            attn.send_tensor(a_to_f)
            torch.testing.assert_close(ffn.recv_tensor(), a_to_f)
            ffn.send_tensor(f_to_a)
            torch.testing.assert_close(attn.recv_tensor(), f_to_a)
            self.assertEqual(engines[0].transfers[0][0], "ffn-session")
            self.assertEqual(engines[1].transfers[0][0], "attn-session")
        finally:
            attn.close()
            ffn.close()
        self.assertTrue(engines[0].deregistered)
        self.assertTrue(engines[1].deregistered)
        self.assertFalse(engines[0].registered)
        self.assertFalse(engines[1].registered)

    def test_tp2_rank_per_rank_independent_fifo_channels(self):
        attn_base, ffn_base = _free_port(), _free_port()
        rank_pairs = []
        try:
            for rank in range(2):
                pair, engines = _make_pair(
                    tp_rank=rank,
                    tp_size=2,
                    attn_port=attn_base,
                    ffn_port=ffn_base,
                )
                rank_pairs.append((pair, engines))

            for rank, ((attn, ffn), engines) in enumerate(rank_pairs):
                expected = torch.tensor([rank, rank + 10], dtype=torch.int64)
                attn.send_tensor(expected)
                torch.testing.assert_close(ffn.recv_tensor(), expected)
                self.assertEqual(attn.tp_rank, rank)
                self.assertEqual(ffn.tp_rank, rank)
                self.assertEqual(attn.local_control_port, ffn_base + rank)
                self.assertEqual(attn.peer_control_port, attn_base + rank)
                self.assertEqual(ffn.local_control_port, attn_base + rank)
                self.assertEqual(ffn.peer_control_port, ffn_base + rank)
                self.assertEqual(engines[0].transfers[0][0], "ffn-session")
        finally:
            for (attn, ffn), _ in rank_pairs:
                attn.close()
                ffn.close()

    def test_channel_port_mapping_avoids_tp_rank_collisions(self):
        attn_base, ffn_base = _free_port(), _free_port()
        (attn, ffn), _ = _make_pair(
            tp_rank=1,
            tp_size=2,
            channel=1,
            attn_port=attn_base,
            ffn_port=ffn_base,
        )
        try:
            self.assertEqual(attn.local_control_port, ffn_base + 3)
            self.assertEqual(ffn.local_control_port, attn_base + 3)
        finally:
            attn.close()
            ffn.close()

    def test_invalid_tp_rank_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "Invalid Mooncake TP rank/size"):
            MooncakeTensorCommunicator(
                AFDPerspective.AFD_PERSPECTIVE_ATTN,
                engine=FakeMooncakeEngine("invalid"),
                tp_rank=2,
                tp_size=2,
                tensor_allocator=lambda size: torch.empty(size, dtype=torch.uint8),
            )

    def test_server_args_accept_homogeneous_tp2_and_reject_heterogeneous_tp(self):
        args = ServerArgs(
            model_path="dummy",
            tp_size=2,
            afd_perspective="attn",
            afd_comm_backend="mooncake",
            afd_attn_tp=2,
            afd_ffn_tp=2,
        )
        self.assertEqual(args.tp_size, 2)

        with self.assertRaisesRegex(ValueError, "requires homogeneous A/F TP"):
            ServerArgs(
                model_path="dummy",
                tp_size=2,
                afd_perspective="attn",
                afd_comm_backend="mooncake",
                afd_attn_tp=2,
                afd_ffn_tp=1,
            )

    def test_fifo_ring_reuse_and_safe_dynamic_resize(self):
        (attn, ffn), _ = _make_pair(buffer_bytes=4, ring_slots=1)
        try:
            values = [
                torch.tensor([1], dtype=torch.int32),
                torch.arange(32, dtype=torch.float32),
                torch.tensor([9, 10], dtype=torch.int64),
            ]
            for expected in values:
                attn.send_tensor(expected)
                torch.testing.assert_close(ffn.recv_tensor(), expected)
            self.assertGreaterEqual(attn._send_caps[0], values[1].nbytes)
            self.assertGreaterEqual(ffn._recv_caps[0], values[1].nbytes)
            self.assertEqual(attn._send_seq, len(values))
            self.assertEqual(ffn._recv_seq, len(values))
        finally:
            attn.close()
            ffn.close()

    def test_timeout_and_close_errors(self):
        (attn, ffn), _ = _make_pair(timeout_ms=50)
        try:
            with self.assertRaisesRegex(TimeoutError, "READY seq=0"):
                attn.recv_tensor()
            attn.close()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                attn.send_tensor(torch.ones(1))
        finally:
            ffn.close()


if __name__ == "__main__":
    unittest.main(verbosity=3)
