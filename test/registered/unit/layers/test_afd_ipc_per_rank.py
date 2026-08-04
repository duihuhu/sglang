import argparse
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.afd import (
    AFDPerspective,
    BroadcastTensorCommunicator,
    _create_tensor_communicator,
)
from sglang.srt.layers.afd_ipc_cpp.communicator import (
    _resolve_channel_id,
    _resolve_peer_device,
)
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _FakeCppCommunicator:
    def __init__(self, perspective, mb_id=None, channel_rank=None):
        self.perspective = perspective
        self.mb_id = mb_id
        self.channel_rank = channel_rank
        self.channel_id = _resolve_channel_id(channel_rank)


class TestAfdIpcPerRank(CustomTestCase):
    def _create(self, *, rank, per_rank=True, attn_tp=2, ffn_tp=2):
        args = SimpleNamespace(
            afd_perspective=AFDPerspective.AFD_PERSPECTIVE_FFN,
            afd_comm_backend="ipc_cpp",
            afd_ipc_per_rank=per_rank,
            afd_attn_tp=attn_tp,
            afd_ffn_tp=ffn_tp,
            tp_size=2,
        )
        with (
            patch("sglang.srt.layers.afd.get_global_server_args", return_value=args),
            patch("sglang.srt.layers.afd.dist.is_initialized", return_value=True),
            patch("sglang.srt.layers.afd.dist.get_rank", return_value=rank),
            patch(
                "sglang.srt.layers.afd_ipc_cpp.communicator.CppIpcTensorCommunicator",
                _FakeCppCommunicator,
            ),
            patch.dict(os.environ, {"AFD_IPC_CHANNEL_BASE": "400"}, clear=False),
        ):
            return _create_tensor_communicator()

    def test_default_tp2_uses_broadcast_wrapper(self):
        rank0 = self._create(rank=0, per_rank=False)
        rank1 = self._create(rank=1, per_rank=False)
        self.assertIsInstance(rank0, BroadcastTensorCommunicator)
        self.assertIsInstance(rank0.inner_comm, _FakeCppCommunicator)
        self.assertIsInstance(rank1, BroadcastTensorCommunicator)
        self.assertIsNone(rank1.inner_comm)

    def test_per_rank_tp2_uses_direct_unique_channels(self):
        rank0 = self._create(rank=0)
        rank1 = self._create(rank=1)
        self.assertIsInstance(rank0, _FakeCppCommunicator)
        self.assertIsInstance(rank1, _FakeCppCommunicator)
        self.assertEqual((rank0.channel_id, rank1.channel_id), (400, 401))

    def test_per_rank_rejects_heterogeneous_tp(self):
        with self.assertRaisesRegex(ValueError, "requires homogeneous A/F TP"):
            self._create(rank=0, attn_tp=1, ffn_tp=2)

    def test_channel_base_falls_back_to_sched_port(self):
        with patch.dict(os.environ, {"AFD_SCHED_PORT": "60400"}, clear=True):
            self.assertEqual(_resolve_channel_id(1), 401)

    def test_peer_offset_mapping_and_bounds(self):
        with patch.dict(os.environ, {"AFD_IPC_PEER_OFFSET": "2"}, clear=True):
            self.assertEqual(_resolve_peer_device(1, 4), 3)
            with self.assertRaisesRegex(ValueError, "out of bounds"):
                _resolve_peer_device(2, 4)
        with patch.dict(os.environ, {"AFD_IPC_PEER_OFFSET": "-2"}, clear=True):
            self.assertEqual(_resolve_peer_device(3, 4), 1)
            with self.assertRaisesRegex(ValueError, "out of bounds"):
                _resolve_peer_device(1, 4)

    def test_server_args_parses_per_rank_flag(self):
        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)
        args = parser.parse_args(["--model-path", "dummy", "--afd-ipc-per-rank"])
        self.assertTrue(args.afd_ipc_per_rank)


if __name__ == "__main__":
    unittest.main(verbosity=3)
