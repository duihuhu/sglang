"""Regression tests for prefill bootstrap polling across AFD TP changes."""

import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _Sender:
    def __init__(self, poll=KVPoll.Bootstrapping):
        self.value = poll

    def poll(self):
        return self.value

    def failure_exception(self):
        raise RuntimeError("failed")


def _req(rid, poll=KVPoll.Bootstrapping):
    trace_ctx = SimpleNamespace(abort=mock.Mock())
    return SimpleNamespace(
        rid=rid,
        bootstrap_room=0,
        disagg_kv_sender=_Sender(poll),
        return_logprob=False,
        time_stats=SimpleNamespace(trace_ctx=trace_ctx),
    )


def _queue(rank, rids):
    queue = object.__new__(PrefillBootstrapQueue)
    queue.tp_rank = rank
    queue.tp_size = 2
    queue.collective_rank = rank
    queue.collective_src_rank = 0
    queue.gloo_group = object()
    queue.queue = [_req(rid) for rid in rids]
    queue.scheduler = SimpleNamespace(
        attn_cp_cpu_group=object(),
        enable_metrics=False,
        enable_hicache_storage=False,
        stream_output=mock.Mock(),
    )
    return queue


class TestPrefillBootstrapQueueReshard(CustomTestCase):
    def test_rank0_order_handles_empty_and_partial_followers(self):
        authoritative = [["rid-a", "rid-b"]]
        observed_polls = []

        def reduce_pollers(pollers, cp_group, tp_group):
            observed_polls.append([poller.poll() for poller in pollers])
            # The mocked collective result includes the Failed contribution
            # from followers that do not have rank 0's rid-b.
            return [KVPoll.Bootstrapping, KVPoll.Failed]

        rank0 = _queue(0, ["rid-a", "rid-b"])
        empty_follower = _queue(1, [])
        partial_follower = _queue(1, ["rid-a", "rid-extra"])
        with (
            mock.patch(
                "sglang.srt.utils.common.broadcast_pyobj",
                return_value=authoritative,
            ) as broadcast,
            mock.patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group",
                side_effect=reduce_pollers,
            ),
            mock.patch("sglang.srt.disaggregation.prefill.prepare_abort"),
        ):
            rank0.pop_bootstrapped()
            empty_follower.pop_bootstrapped()
            partial_follower.pop_bootstrapped()

        self.assertEqual([len(polls) for polls in observed_polls], [2, 2, 2])
        self.assertEqual(
            observed_polls,
            [
                [KVPoll.Bootstrapping, KVPoll.Bootstrapping],
                [KVPoll.Failed, KVPoll.Failed],
                [KVPoll.Bootstrapping, KVPoll.Failed],
            ],
        )
        self.assertEqual([req.rid for req in rank0.queue], ["rid-a"])
        self.assertEqual(empty_follower.queue, [])
        self.assertEqual(
            [req.rid for req in partial_follower.queue], ["rid-a", "rid-extra"]
        )
        self.assertEqual(broadcast.call_count, 3)
        self.assertEqual(broadcast.call_args_list[0].args[0], [["rid-a", "rid-b"]])
        self.assertIsNone(broadcast.call_args_list[1].args[0])
        self.assertIsNone(broadcast.call_args_list[2].args[0])
        for call in broadcast.call_args_list:
            self.assertEqual(call.kwargs["src"], 0)

    def test_tp1_fast_path_skips_collectives(self):
        queue = _queue(0, ["rid-a"])
        queue.tp_size = 1
        with (
            mock.patch(
                "sglang.srt.utils.common.broadcast_pyobj"
            ) as broadcast,
            mock.patch(
                "sglang.srt.disaggregation.prefill.poll_and_all_reduce_attn_cp_tp_group"
            ) as reduce,
        ):
            queue.pop_bootstrapped()
        broadcast.assert_not_called()
        reduce.assert_not_called()
        self.assertEqual([req.rid for req in queue.queue], ["rid-a"])


if __name__ == "__main__":
    unittest.main(verbosity=3)
