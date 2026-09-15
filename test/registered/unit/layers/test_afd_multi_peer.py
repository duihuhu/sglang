import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import patch

from sglang.srt.layers.afd_multi_peer import (
    AFDCrossNodePeerPool,
    AFDCrossNodePeerSpec,
    AFDPeerChannelPool,
    AFDPeerSpec,
    parse_shared_peer_specs,
)
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode
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
                    finished=lambda: False,
                )
            )
        self.forward_mode = ForwardMode.EXTEND
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

    def test_scheduler_keeps_new_batch_on_one_group(self):
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

        self.assertIsNone(batch.afd_split_seq_index)
        self.assertEqual(batch.afd_pf_group_ids, [0])
        self.assertEqual(
            sockets[0].messages[0].req_ids,
            ["r0", "r1", "r2", "r3"],
        )
        self.assertEqual(sockets[1].messages, [])
        self.assertEqual(sockets[0].messages[0].dispatch_id, 1)


class _CrossInner:
    created: ClassVar[list] = []

    def __init__(self, perspective, **kwargs):
        self.perspective = perspective
        self.kwargs = kwargs
        self.closed = False
        self.created.append(self)

    def close(self):
        self.closed = True


class _Broadcast:
    created: ClassVar[list] = []

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.created.append(self)


class _FakeTPBroadcast:
    value = None

    def __init__(self, rank):
        self.world_size = 2
        self.rank_in_group = rank

    def broadcast_object(self, obj, src=0):
        if self.rank_in_group == src:
            type(self).value = obj
        return type(self).value


class _FakeCoordinatorClient:
    def __init__(self):
        self.acquire_calls = []
        self.begin_calls = []
        self.complete_calls = []
        self.release_calls = []

    def acquire(self, rid, pa_id, *, cost, preferred_pf_instance_id=None):
        self.acquire_calls.append((rid, pa_id, cost, preferred_pf_instance_id))
        return {
            "rid": rid,
            "pf_instance_id": "F1",
            "lease_id": "lease-F1",
            "pair_epoch": 7,
        }

    def begin_dispatch(self, dispatch_id, lease_id, **kwargs):
        self.begin_calls.append((dispatch_id, lease_id, kwargs))

    def complete_dispatch(self, dispatch_id, lease_id):
        self.complete_calls.append((dispatch_id, lease_id))
        return True

    def release(self, rid, pa_id):
        self.release_calls.append((rid, pa_id))
        return True


class TestAFDCrossNodePeerPool(CustomTestCase):
    def setUp(self):
        _CrossInner.created.clear()
        _Broadcast.created.clear()

    @staticmethod
    def specs():
        return parse_shared_peer_specs(
            '[{"peer_id":"A0","peer_host":"a0","ffn_base_port":40000,'
            '"attn_base_port":41000,"ffn_handshake_base_port":42000,'
            '"attn_handshake_base_port":43000},'
            '{"peer_id":"A1","peer_host":"a1","ffn_base_port":44000,'
            '"attn_base_port":45000,"ffn_handshake_base_port":46000,'
            '"attn_handshake_base_port":47000}]'
        )

    def make_pool(self, rank=0):
        return AFDCrossNodePeerPool(
            AFDPerspective.AFD_PERSPECTIVE_FFN,
            self.specs(),
            local_tp_size=8,
            local_tp_rank=rank,
            comm_factory=_CrossInner,
            broadcast_factory=_Broadcast,
            async_factory=lambda comm: SimpleNamespace(inner=comm),
        )

    def test_four_edges_have_unique_ports(self):
        specs = parse_shared_peer_specs(
            "F0@f0:40000:41000:42000:43000,F1@f1:44000:45000:46000:47000,"
            "A0@a0:48000:49000:50000:51000,A1@a1:52000:53000:54000:55000"
        )
        ports = [
            port
            for spec in specs
            for port in (
                spec.ffn_base_port,
                spec.attn_base_port,
                spec.ffn_handshake_base_port,
                spec.attn_handshake_base_port,
            )
        ]
        self.assertEqual(len(ports), len(set(ports)))

    def test_shared_pool_initialization_does_not_select_a_peer(self):
        from sglang.srt.layers import afd

        pool = self.make_pool()
        args = SimpleNamespace(afd_shared_pool=True)
        with (
            patch.object(afd, "get_global_server_args", return_value=args),
            patch.object(afd, "get_afd_cross_node_peer_pool", return_value=pool),
        ):
            self.assertIs(afd.initialize_afd_data_plane(), pool)
            with self.assertRaisesRegex(RuntimeError, "No shared AFD peer"):
                afd.get_async_communicator()

        with pool.select("A1"):
            self.assertIs(pool.get(), pool.get("A1"))

    def test_f_selects_a0_or_a1_and_context_is_isolated(self):
        pool = self.make_pool()
        with pool.select("A0") as a0:
            self.assertIs(pool.get(), a0)
            with pool.select("A1") as a1:
                self.assertIs(pool.get(), a1)
            self.assertIs(pool.get(), a0)
        with self.assertRaisesRegex(RuntimeError, "No shared AFD peer"):
            pool.get()
        with self.assertRaisesRegex(KeyError, "Unknown shared AFD peer"):
            pool.get("stale-A")

    def test_tp_constructs_zmq_only_on_rank_zero(self):
        rank0 = self.make_pool(rank=0)
        self.assertEqual(len(_CrossInner.created), 2)
        self.assertTrue(all(item.inner_comm is not None for item in _Broadcast.created))
        rank0.close()
        _CrossInner.created.clear()
        _Broadcast.created.clear()
        rank7 = self.make_pool(rank=7)
        self.assertEqual(_CrossInner.created, [])
        self.assertTrue(all(item.inner_comm is None for item in _Broadcast.created))
        rank7.close()

    def test_reconnect_and_status(self):
        pool = self.make_pool()
        old = pool.get("A0").inner.inner_comm
        pool.reconnect("A0")
        self.assertTrue(old.closed)
        self.assertEqual(pool.status()["A0"].reconnects, 1)
        pool.close()
        self.assertEqual(pool.status()["A0"].state, "closed")

    def test_tp2_rank_zero_routes_f1_once_and_broadcasts_to_rank_one(self):
        _FakeTPBroadcast.value = None
        client = _FakeCoordinatorClient()
        sockets = {"F0": _FakeSocket(), "F1": _FakeSocket()}
        args = SimpleNamespace(
            afd_shared_pool=True,
            afd_instance_id="A0",
            afd_capacity_wait_timeout=3.0,
            afd_capacity_retry_backoff=0.1,
        )
        rank0 = SimpleNamespace(
            server_args=args,
            tp_group=_FakeTPBroadcast(0),
            afd_send_to_ffn_groups=sockets,
            _afd_shared_client=client,
            _afd_dispatch_id=0,
            afd_global_dispatch_identity=lambda pa_id, epoch, seq: (
                f"{pa_id}:{epoch}:{seq}"
            ),
        )
        rank1 = SimpleNamespace(
            server_args=args,
            tp_group=_FakeTPBroadcast(1),
            afd_send_to_ffn_groups={},
            _afd_dispatch_id=0,
            afd_shared_pool_client=lambda: self.fail(
                "non-authoritative TP rank contacted coordinator"
            ),
        )
        batch0, batch1 = _FakeBatch([16, 8]), _FakeBatch([16, 8])
        for batch in (batch0, batch1):
            for index, req in enumerate(batch.reqs):
                req.afd_pa_instance_id = "A0"
                req.afd_pf_instance_id = "F1"
                req.afd_lease_id = f"lease-F1-{index}"
                req.afd_pair_epoch = 7

        # The tensor transport registry is process-local, but every TP rank
        # constructs it from the same peer specs and can select F1.
        f_specs = parse_shared_peer_specs(
            "F0@f0:40000:41000:42000:43000,F1@f1:44000:45000:46000:47000"
        )
        pool0 = AFDCrossNodePeerPool(
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            f_specs,
            local_tp_size=2,
            local_tp_rank=0,
            comm_factory=_CrossInner,
            broadcast_factory=_Broadcast,
            async_factory=lambda comm: SimpleNamespace(inner=comm),
        )
        pool1 = AFDCrossNodePeerPool(
            AFDPerspective.AFD_PERSPECTIVE_ATTN,
            f_specs,
            local_tp_size=2,
            local_tp_rank=1,
            comm_factory=_CrossInner,
            broadcast_factory=_Broadcast,
            async_factory=lambda comm: SimpleNamespace(inner=comm),
        )
        self.assertEqual(pool0.peer_ids, ("F0", "F1"))
        self.assertEqual(pool1.peer_ids, ("F0", "F1"))
        with pool0.select("F1"), pool1.select("F1"):
            self.assertIsNotNone(pool0.get())
            self.assertIsNotNone(pool1.get())

        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            SchedulerAFDMixin.afd_send_batch_info(rank0, batch0)
            SchedulerAFDMixin.afd_send_batch_info(rank1, batch1)

        self.assertEqual(batch0.afd_peer_id, "F1")
        self.assertEqual(batch1.afd_peer_id, "F1")
        self.assertEqual(batch1.afd_lease_id, "lease-F1-0")
        self.assertEqual(batch1.afd_lease_ids, ["lease-F1-0", "lease-F1-1"])
        self.assertEqual(batch1.afd_pair_epoch, 7)
        self.assertEqual(rank0._afd_dispatch_id, rank1._afd_dispatch_id)
        self.assertEqual(len(client.acquire_calls), 0)
        self.assertEqual(len(client.begin_calls), 1)
        self.assertEqual(len(sockets["F1"].messages), 1)
        self.assertEqual(len(sockets["F0"].messages), 0)

        batch0.reqs[0].finished = lambda: True
        batch1.reqs[0].finished = lambda: True
        SchedulerAFDMixin.afd_complete_shared_dispatch(rank0, batch0)
        SchedulerAFDMixin.afd_complete_shared_dispatch(rank1, batch1)
        self.assertEqual(len(client.complete_calls), 1)
        self.assertEqual(len(client.release_calls), 2)
        pool0.close()
        pool1.close()

    def test_ucx_is_explicitly_blocked(self):
        spec = AFDCrossNodePeerSpec("A0", "a0", 40000, 41000, backend="ucx")
        with self.assertRaisesRegex(NotImplementedError, "UCX"):
            AFDCrossNodePeerPool(
                AFDPerspective.AFD_PERSPECTIVE_FFN,
                [spec],
                comm_factory=_CrossInner,
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
