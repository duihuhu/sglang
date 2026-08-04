import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.layers.afd_multi_peer import (
    AFDPeerChannelPool,
    AFDPeerSpec,
)
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _FakeInner:
    def __init__(self, perspective, *, peer_device, channel_id):
        self.perspective = perspective
        self.peer_device = peer_device
        self.channel_id = channel_id


class _FakeAsync:
    def __init__(self, inner):
        self.inner = inner
        self.drained = []

    def drain_sends(self):
        self.drained.append("send")

    def drain_recvs(self):
        self.drained.append("recv")


class _FakeSocket:
    def __init__(self):
        self.messages = []

    def send_pyobj(self, message):
        self.messages.append(message)


class _FakeBatch:
    def __init__(self, extend_lens):
        self.extend_lens = extend_lens
        self.reqs = []
        for i, extend_len in enumerate(extend_lens):
            self.reqs.append(
                SimpleNamespace(
                    rid=f"r{i}",
                    extend_input_len=extend_len,
                    seqlen=0,
                    origin_input_ids=list(range(extend_len)),
                    output_ids=[],
                    sampling_params=SimpleNamespace(max_new_tokens=8),
                )
            )
        self.forward_mode = SimpleNamespace()
        self.afd_split_seq_index = None
        self.afd_pf_group_ids = None

    def batch_size(self):
        return len(self.reqs)


class TestAFDPeerChannelPool(CustomTestCase):
    def test_builds_independent_channels(self):
        pool = AFDPeerChannelPool(
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            [
                AFDPeerSpec(group_id=0, channel_id=700, peer_device=1),
                AFDPeerSpec(group_id=1, channel_id=716, peer_device=3),
            ],
            comm_factory=_FakeInner,
            async_factory=_FakeAsync,
        )

        self.assertEqual(pool.group_ids, (0, 1))
        self.assertEqual(pool[0].inner.channel_id, 700)
        self.assertEqual(pool[0].inner.peer_device, 1)
        self.assertEqual(pool[1].inner.channel_id, 716)
        self.assertEqual(pool[1].inner.peer_device, 3)
        self.assertIsNot(pool[0].async_comm, pool[1].async_comm)

    def test_rejects_duplicate_group_or_channel(self):
        with self.assertRaisesRegex(ValueError, "group IDs must be unique"):
            AFDPeerChannelPool(
                AFDPerspective.AFD_PERSPECTIVE_ATTN,
                [
                    AFDPeerSpec(0, 700, 1),
                    AFDPeerSpec(0, 716, 3),
                ],
                comm_factory=_FakeInner,
                async_factory=_FakeAsync,
            )
        with self.assertRaisesRegex(ValueError, "channel IDs must be unique"):
            AFDPeerChannelPool(
                AFDPerspective.AFD_PERSPECTIVE_ATTN,
                [
                    AFDPeerSpec(0, 700, 1),
                    AFDPeerSpec(1, 700, 3),
                ],
                comm_factory=_FakeInner,
                async_factory=_FakeAsync,
            )

    def test_drain_all_channels(self):
        pool = AFDPeerChannelPool(
            AFDPerspective.AFD_PERSPECTIVE_FFN,
            [
                AFDPeerSpec(0, 700, 0),
                AFDPeerSpec(1, 716, 0),
            ],
            comm_factory=_FakeInner,
            async_factory=_FakeAsync,
        )

        pool.drain()

        self.assertEqual(pool[0].async_comm.drained, ["send", "recv"])
        self.assertEqual(pool[1].async_comm.drained, ["send", "recv"])

    def test_unknown_group_has_context(self):
        pool = AFDPeerChannelPool(
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            [AFDPeerSpec(2, 732, 5)],
            comm_factory=_FakeInner,
            async_factory=_FakeAsync,
        )
        with self.assertRaisesRegex(KeyError, r"available=\(2,\)"):
            pool[0]

    def test_scheduler_splits_batch_by_token_cost(self):
        sockets = {0: _FakeSocket(), 1: _FakeSocket()}
        scheduler = SimpleNamespace(
            server_args=SimpleNamespace(afd_multi_pf_continuation=True),
            afd_send_to_ffn_groups=sockets,
            _afd_pf_rr_cursor=0,
            _afd_dispatch_id=0,
        )
        batch = _FakeBatch([4096, 128, 128, 128])

        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            SchedulerAFDMixin.afd_send_batch_info(scheduler, batch)

        self.assertEqual(batch.afd_split_seq_index, [1])
        self.assertEqual(batch.afd_pf_group_ids, [0, 1])
        self.assertEqual(sockets[0].messages[0].req_ids, ["r0"])
        self.assertEqual(
            sockets[1].messages[0].req_ids,
            ["r1", "r2", "r3"],
        )
        self.assertEqual(sockets[0].messages[0].dispatch_id, 1)
        self.assertEqual(sockets[1].messages[0].dispatch_id, 1)


if __name__ == "__main__":
    unittest.main(verbosity=3)
