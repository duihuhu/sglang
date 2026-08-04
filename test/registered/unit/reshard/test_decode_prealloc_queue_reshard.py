"""Regression tests for PD decode queues across AFD TP changes."""
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.disaggregation.base import KVPoll
from sglang.srt.disaggregation.decode import (
    DecodePreallocQueue,
    DecodeRequest,
    DecodeTransferQueue,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="stage-a-cpu-only")


class _Receiver:
    def __init__(self, poll):
        self.value = poll
        self.aborted = False

    def poll(self):
        return self.value

    def abort(self):
        self.aborted = True
        self.value = KVPoll.Failed

    def failure_exception(self):
        raise RuntimeError("failed")


def _entry(rid, poll=KVPoll.WaitingForInput):
    req = SimpleNamespace(
        rid=rid, bootstrap_room=0, finished_reason=None, return_logprob=False,
        time_stats=SimpleNamespace(set_bootstrap_done_time=lambda: None),
    )
    return DecodeRequest(req=req, kv_receiver=_Receiver(poll))


def _queue(rank, rids):
    queue = object.__new__(DecodePreallocQueue)
    queue.tp_rank = rank
    queue.gloo_group = object()
    queue.tp_size = 2
    queue.queue = [_entry(rid) for rid in rids]
    queue.pending_reqs = []
    queue.retracted_queue = []
    queue.scheduler = SimpleNamespace(enable_metrics=False)
    return queue


class TestDecodePreallocQueueReshard(CustomTestCase):
    def test_rank0_order_fixes_mismatched_local_lengths(self):
        authoritative = [["rid-a", "rid-b"]]
        observed_lengths = []

        def reduce_pollers(pollers, group):
            observed_lengths.append(len(pollers))
            # rid-b is absent on the follower, so the collective decision fails it.
            return [KVPoll.WaitingForInput, KVPoll.Failed]

        rank0 = _queue(0, ["rid-a", "rid-b"])
        follower = _queue(1, ["rid-a"])
        with (
            mock.patch(
                "sglang.srt.utils.common.broadcast_pyobj",
                return_value=authoritative,
            ),
            mock.patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=reduce_pollers,
            ),
            mock.patch("sglang.srt.disaggregation.decode.prepare_abort"),
        ):
            rank0._update_handshake_waiters()
            follower._update_handshake_waiters()

        self.assertEqual(observed_lengths, [2, 2])
        self.assertTrue(rank0.queue[0].waiting_for_input)
        self.assertTrue(follower.queue[0].waiting_for_input)

    def test_tp1_fast_path_skips_collectives(self):
        queue = _queue(0, ["rid-a"])
        queue.tp_size = 1
        with (
            mock.patch(
                "sglang.srt.utils.common.broadcast_pyobj"
            ) as broadcast,
            mock.patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce"
            ) as reduce,
        ):
            queue._update_handshake_waiters()
        broadcast.assert_not_called()
        reduce.assert_not_called()
        self.assertTrue(queue.queue[0].waiting_for_input)

    def test_transfer_rank0_order_fixes_mismatched_local_lengths(self):
        authoritative = [["rid-a", "rid-b"]]
        observed_lengths = []

        def transfer_queue(rank, rids):
            queue = object.__new__(DecodeTransferQueue)
            queue.tp_rank = rank
            queue.tp_size = 2
            queue.gloo_group = object()
            queue.queue = [_entry(rid, KVPoll.Transferring) for rid in rids]
            for index, entry in enumerate(queue.queue):
                entry.metadata_buffer_index = index
            queue.req_to_metadata_buffer_idx_allocator = SimpleNamespace(
                free=mock.Mock()
            )
            queue.scheduler = SimpleNamespace(
                enable_metrics=False, stream_output=mock.Mock()
            )
            queue.tree_cache = SimpleNamespace()
            return queue

        def reduce_pollers(pollers, group):
            observed_lengths.append(len(pollers))
            return [KVPoll.Transferring, KVPoll.Failed]

        rank0 = transfer_queue(0, ["rid-a", "rid-b"])
        follower = transfer_queue(1, ["rid-a"])
        with (
            mock.patch(
                "sglang.srt.utils.common.broadcast_pyobj",
                return_value=authoritative,
            ),
            mock.patch(
                "sglang.srt.disaggregation.decode.poll_and_all_reduce",
                side_effect=reduce_pollers,
            ),
            mock.patch("sglang.srt.disaggregation.decode.prepare_abort"),
            mock.patch("sglang.srt.disaggregation.decode.release_kv_cache"),
        ):
            rank0.pop_transferred()
            follower.pop_transferred()

        self.assertEqual(observed_lengths, [2, 2])
        self.assertEqual([entry.req.rid for entry in rank0.queue], ["rid-a"])
        self.assertEqual([entry.req.rid for entry in follower.queue], ["rid-a"])

    def test_snapshot_exposes_all_preallocation_subqueues(self):
        queue = _queue(0, ["active"])
        queue.pending_reqs = [SimpleNamespace(rid="pending")]
        queue.retracted_queue = [SimpleNamespace(rid="retracted")]
        snapshot = queue.snapshot()
        self.assertEqual(snapshot["queue"]["rids"], ["active"])
        self.assertEqual(snapshot["pending"]["rids"], ["pending"])
        self.assertEqual(snapshot["retracted"]["rids"], ["retracted"])
        self.assertEqual(len(snapshot["queue"]["rid_hash"]), 16)

    def test_idle_reason_reports_pending_and_retracted(self):
        from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin

        scheduler = SimpleNamespace(
            is_fully_idle=lambda: False,
            waiting_queue=[],
            running_batch=[],
            cur_batch=None,
            last_batch=None,
            result_queue=[],
            grammar_manager=SimpleNamespace(grammar_queue=[]),
            disagg_decode_prealloc_queue=SimpleNamespace(
                queue=[],
                pending_reqs=[SimpleNamespace(rid="pending")],
                retracted_queue=[SimpleNamespace(rid="retracted")],
            ),
            disagg_decode_transfer_queue=SimpleNamespace(queue=[]),
            disagg_prefill_bootstrap_queue=SimpleNamespace(queue=[]),
            disagg_prefill_inflight_queue=[],
            chunked_req=None,
        )
        reason = SchedulerAFDMixin.afd_component_scheduler_idle_reason(scheduler)
        self.assertIn("decode_pending=1", reason)
        self.assertIn("decode_retracted=1", reason)


if __name__ == "__main__":
    unittest.main(verbosity=3)
