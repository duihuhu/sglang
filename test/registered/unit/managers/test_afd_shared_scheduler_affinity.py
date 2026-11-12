import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler_afd_mixin import (
    SchedulerAFDMixin,
    afd_shared_route_offset,
    afd_shared_scheduler_peer_ids,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _Client:
    def __init__(self):
        self.seq = 0
        self.preferred = []

    def acquire(self, rid, pa_id, *, cost, preferred_pf_instance_id=None):
        self.seq += 1
        self.preferred.append(preferred_pf_instance_id)
        return {
            "rid": rid,
            "pa_instance_id": pa_id,
            "pf_instance_id": preferred_pf_instance_id,
            "lease_id": f"lease-{self.seq}",
            "pair_epoch": 3,
        }


class _TPBroadcast:
    value = None

    def __init__(self, rank):
        self.world_size = 2
        self.rank_in_group = rank

    def broadcast_object(self, value, src=0):
        if self.rank_in_group == src:
            type(self).value = value
        return type(self).value


class TestAFDSharedSchedulerAffinity(CustomTestCase):
    @staticmethod
    def req(rid, peer=None, lease=None):
        return SimpleNamespace(
            rid=rid,
            afd_pa_instance_id="A0" if peer else None,
            afd_pf_instance_id=peer,
            afd_lease_id=lease,
            afd_pair_epoch=3 if peer else 0,
            finished=lambda: False,
        )

    @staticmethod
    def scheduler(
        reqs, *, rank=0, client=None, sockets=None, specs=None, instance_id="A0"
    ):
        if specs is None:
            specs = "F0@f0:40000:41000:42000:43000,F1@f1:44000:45000:46000:47000"
        return SimpleNamespace(
            server_args=SimpleNamespace(
                afd_shared_pool=True,
                afd_instance_id=instance_id,
                afd_shared_peer_specs=specs,
            ),
            afd_send_to_ffn_groups=(
                {"F0": object(), "F1": object()} if sockets is None else sockets
            ),
            waiting_queue=list(reqs),
            running_batch=ScheduleBatch(reqs=[], batch_is_full=False),
            last_batch=None,
            chunked_req=None,
            _afd_shared_peer_states={},
            _afd_shared_active_peer=None,
            _afd_shared_route_cursor=0,
            _afd_pf_rr_cursor=0,
            _afd_shared_client=client or _Client(),
            tp_group=_TPBroadcast(rank),
        )

    def test_numeric_pa_ids_start_on_distinct_peers_and_wrap(self):
        specs = ",".join(
            f"F{i}@f{i}:{20000 + 4000 * i}:{21000 + 4000 * i}:"
            f"{22000 + 4000 * i}:{23000 + 4000 * i}"
            for i in range(8)
        )
        expected_routes = (("A0", "F0"), ("A1", "F1"), ("A7", "F7"), ("A8", "F0"))
        for pa_id, expected in expected_routes:
            with self.subTest(pa_id=pa_id):
                client = _Client()
                scheduler = self.scheduler(
                    [self.req("request")],
                    client=client,
                    specs=specs,
                    instance_id=pa_id,
                )
                with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
                    SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler)
                self.assertEqual(client.preferred, [expected])

    def test_non_numeric_pa_offset_is_stable(self):
        first = afd_shared_route_offset("attention-west", 8)
        self.assertEqual(first, afd_shared_route_offset("attention-west", 8))
        self.assertGreaterEqual(first, 0)
        self.assertLess(first, 8)

    def test_each_pa_round_robins_from_its_offset(self):
        specs = (
            "F0@f0:40000:41000:42000:43000,"
            "F1@f1:44000:45000:46000:47000,"
            "F2@f2:48000:49000:50000:51000"
        )
        client = _Client()
        scheduler = self.scheduler(
            [self.req(str(i)) for i in range(5)],
            client=client,
            specs=specs,
            instance_id="A1",
        )
        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler)
        self.assertEqual(client.preferred, ["F1", "F2", "F0", "F1", "F2"])
        self.assertEqual(scheduler._afd_shared_route_cursor, 5)

    def test_tp_ranks_share_authority_routing_decisions(self):
        _TPBroadcast.value = None
        specs = "F0@f0:40000:41000:42000:43000," "F1@f1:44000:45000:46000:47000"
        client = _Client()
        rank0 = self.scheduler(
            [self.req("a"), self.req("b")],
            rank=0,
            client=client,
            specs=specs,
            instance_id="A1",
        )
        rank1 = self.scheduler(
            [self.req("a"), self.req("b")],
            rank=1,
            specs=specs,
            instance_id="A1",
        )
        rank1._afd_shared_client = SimpleNamespace(
            acquire=lambda *args, **kwargs: self.fail("non-authority acquired lease")
        )
        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            SchedulerAFDMixin.afd_shared_select_scheduler_lane(rank0)
            SchedulerAFDMixin.afd_shared_select_scheduler_lane(rank1)
        self.assertEqual(client.preferred, ["F1", "F0"])
        self.assertEqual(
            [(req.afd_pf_instance_id, req.afd_lease_id) for req in rank0.waiting_queue],
            [(req.afd_pf_instance_id, req.afd_lease_id) for req in rank1.waiting_queue],
        )
        self.assertEqual(rank0._afd_shared_route_cursor, 2)
        self.assertEqual(rank1._afd_shared_route_cursor, 0)

    def test_mixed_peers_are_exposed_as_distinct_forward_lanes(self):
        reqs = [
            self.req("f0-a", "F0", "l0"),
            self.req("f1-a", "F1", "l1"),
            self.req("f0-b", "F0", "l2"),
        ]
        scheduler = self.scheduler(reqs)
        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            first = SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler)
            self.assertEqual(
                {req.afd_pf_instance_id for req in scheduler.waiting_queue}, {first}
            )
            SchedulerAFDMixin.afd_shared_restore_waiting_queue(scheduler)
            second = SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler)
        self.assertNotEqual(first, second)
        self.assertEqual(
            {req.afd_pf_instance_id for req in scheduler.waiting_queue}, {second}
        )

    def test_same_peer_prefill_and_decode_state_stay_together(self):
        first, second = self.req("a", "F0", "l0"), self.req("b", "F0", "l1")
        scheduler = self.scheduler([first, second])
        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            self.assertEqual(
                SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler), "F0"
            )
            scheduler.running_batch = ScheduleBatch(reqs=[first, second])
            scheduler.waiting_queue = []
            SchedulerAFDMixin.afd_shared_restore_waiting_queue(scheduler)
            self.assertEqual(
                SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler), "F0"
            )
        self.assertEqual([req.rid for req in scheduler.running_batch.reqs], ["a", "b"])

    def test_tp_ranks_use_static_peer_specs_when_non_authority_has_no_sockets(self):
        _TPBroadcast.value = None
        specs = "F0@f0:40000:41000:42000:43000"
        rank0 = self.scheduler(
            [self.req("a"), self.req("b")],
            rank=0,
            sockets={"F0": object()},
            specs=specs,
        )
        rank1 = self.scheduler(
            [self.req("a"), self.req("b")], rank=1, sockets={}, specs=specs
        )
        rank1._afd_shared_client = SimpleNamespace(
            acquire=lambda *args, **kwargs: self.fail("non-authority acquired lease")
        )
        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            peer0 = SchedulerAFDMixin.afd_shared_select_scheduler_lane(rank0)
            peer1 = SchedulerAFDMixin.afd_shared_select_scheduler_lane(rank1)
        self.assertEqual(peer0, "F0")
        self.assertEqual(peer1, "F0")
        self.assertEqual(rank0._afd_shared_peer_ids, ("F0",))
        self.assertEqual(rank1._afd_shared_peer_ids, ("F0",))
        self.assertEqual(
            [(r.afd_pf_instance_id, r.afd_lease_id) for r in rank0.waiting_queue],
            [(r.afd_pf_instance_id, r.afd_lease_id) for r in rank1.waiting_queue],
        )

    def test_static_peer_ids_are_sorted_and_identical_for_multiple_peers(self):
        specs = "F1@f1:44000:45000:46000:47000,F0@f0:40000:41000:42000:43000"
        args0 = SimpleNamespace(afd_shared_peer_specs=specs)
        args1 = SimpleNamespace(afd_shared_peer_specs=specs)
        self.assertEqual(afd_shared_scheduler_peer_ids(args0), ("F0", "F1"))
        self.assertEqual(
            afd_shared_scheduler_peer_ids(args0),
            afd_shared_scheduler_peer_ids(args1),
        )

    def test_static_peer_ids_fail_fast_without_specs(self):
        with self.assertRaisesRegex(ValueError, "peer specs must be non-empty"):
            afd_shared_scheduler_peer_ids(SimpleNamespace(afd_shared_peer_specs=None))


if __name__ == "__main__":
    unittest.main(verbosity=3)
