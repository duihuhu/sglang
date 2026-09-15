import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _Req:
    def __init__(self, rid, peer, *, finished=False, req_pool_idx=None):
        self.rid = rid
        self.afd_pa_instance_id = "A0"
        self.afd_pf_instance_id = peer
        self.afd_lease_id = f"lease-{rid}"
        self.afd_pair_epoch = 1
        self.req_pool_idx = req_pool_idx
        self._finished = finished

    def finished(self):
        return self._finished


class _Batch:
    def __init__(self, reqs=()):
        self.reqs = list(reqs)
        self.batch_is_full = bool(self.reqs)

    def is_empty(self):
        return not self.reqs

    def filter_batch(self, *, keep_indices=None, **_kwargs):
        if keep_indices is None:
            keep_indices = [i for i, req in enumerate(self.reqs) if not req.finished()]
        self.reqs = [self.reqs[i] for i in keep_indices]


def _state(running=None, last=None, chunked=None):
    return {
        "running_batch": running or _Batch(),
        "last_batch": last,
        "chunked_req": chunked,
    }


def _scheduler(peer_ids=("F0", "F1")):
    return SimpleNamespace(
        server_args=SimpleNamespace(
            afd_shared_pool=True,
            afd_instance_id="A0",
            afd_shared_peer_specs=None,
        ),
        waiting_queue=[],
        running_batch=_Batch(),
        last_batch=None,
        chunked_req=None,
        tree_cache=object(),
        tp_group=None,
        _afd_shared_peer_ids=peer_ids,
        _afd_shared_peer_states={},
        _afd_shared_active_peer=None,
        _afd_shared_other_waiting=None,
        _afd_shared_route_cursor=0,
        _afd_shared_route_offset=0,
        _afd_pf_rr_cursor=0,
        _afd_shared_client=SimpleNamespace(
            acquire=lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("all lifecycle-test requests must already be PF-affine")
            )
        ),
    )


class TestAFDSharedLaneLifecycle(CustomTestCase):
    def test_hidden_finished_lane_releases_once_and_filters(self):
        scheduler = _scheduler()
        live = _Req("live", "F0", req_pool_idx=10)
        finished = _Req("done", "F1", finished=True, req_pool_idx=11)
        scheduler._afd_shared_active_peer = "F0"
        scheduler.running_batch = _Batch([live])
        scheduler._afd_shared_peer_states["F1"] = _state(_Batch([finished]))
        released = []

        def release(req, _cache):
            released.append(req.rid)
            req.req_pool_idx = None

        with patch("sglang.srt.mem_cache.common.release_kv_cache", side_effect=release):
            SchedulerAFDMixin.afd_shared_cleanup_finished_lanes(scheduler)
            SchedulerAFDMixin.afd_shared_cleanup_finished_lanes(scheduler)

        self.assertEqual(released, ["done"])
        self.assertEqual(
            scheduler._afd_shared_peer_states["F1"]["running_batch"].reqs, []
        )
        self.assertEqual(scheduler.running_batch.reqs, [live])

    def test_running_last_alias_does_not_double_release(self):
        scheduler = _scheduler()
        finished = _Req("done", "F1", finished=True, req_pool_idx=12)
        alias = _Batch([finished])
        scheduler._afd_shared_peer_states["F1"] = _state(alias, alias)
        released = []

        def release(req, _cache):
            released.append(req.rid)
            req.req_pool_idx = None

        with patch("sglang.srt.mem_cache.common.release_kv_cache", side_effect=release):
            SchedulerAFDMixin.afd_shared_cleanup_finished_lanes(scheduler)

        self.assertEqual(released, ["done"])
        self.assertEqual(alias.reqs, [])

    def test_hidden_live_lane_blocks_idle(self):
        scheduler = _scheduler()
        hidden = _Req("hidden", "F1", req_pool_idx=13)
        scheduler._afd_shared_peer_states["F1"] = _state(_Batch([hidden]))
        self.assertTrue(SchedulerAFDMixin.afd_shared_has_scheduler_work(scheduler))
        hidden._finished = True
        self.assertFalse(SchedulerAFDMixin.afd_shared_has_scheduler_work(scheduler))

    def test_all_finished_restores_allocator_ownership(self):
        scheduler = _scheduler()
        reqs = [
            _Req("f0", "F0", finished=True, req_pool_idx=20),
            _Req("f1", "F1", finished=True, req_pool_idx=21),
        ]
        scheduler._afd_shared_active_peer = "F0"
        scheduler.running_batch = _Batch([reqs[0]])
        scheduler._afd_shared_peer_states["F1"] = _state(_Batch([reqs[1]]))
        available = 98

        def release(req, _cache):
            nonlocal available
            available += 1
            req.req_pool_idx = None

        with patch("sglang.srt.mem_cache.common.release_kv_cache", side_effect=release):
            SchedulerAFDMixin.afd_shared_finish_scheduler_step(scheduler)

        self.assertEqual(available, 100)
        self.assertFalse(SchedulerAFDMixin.afd_shared_has_scheduler_work(scheduler))
        self.assertTrue(
            all(
                not state["running_batch"].reqs
                for state in scheduler._afd_shared_peer_states.values()
            )
        )

    def test_round_robin_uses_full_peer_ring_when_candidates_change(self):
        scheduler = _scheduler(("F0", "F1", "F2"))
        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            scheduler.waiting_queue = [_Req("a", "F0"), _Req("b", "F2")]
            self.assertEqual(
                SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler), "F0"
            )
            SchedulerAFDMixin.afd_shared_restore_waiting_queue(scheduler)

            scheduler.waiting_queue = [_Req("b", "F2")]
            self.assertEqual(
                SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler), "F2"
            )
            SchedulerAFDMixin.afd_shared_restore_waiting_queue(scheduler)

            scheduler.waiting_queue = [_Req("c", "F1"), _Req("b", "F2")]
            self.assertEqual(
                SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler), "F1"
            )

    def test_single_pf_behavior_is_unchanged(self):
        scheduler = _scheduler(("F0",))
        scheduler.waiting_queue = [_Req("a", "F0")]
        with patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            self.assertEqual(
                SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler), "F0"
            )
            SchedulerAFDMixin.afd_shared_restore_waiting_queue(scheduler)
            self.assertEqual(
                SchedulerAFDMixin.afd_shared_select_scheduler_lane(scheduler), "F0"
            )


if __name__ == "__main__":
    unittest.main(verbosity=3)
