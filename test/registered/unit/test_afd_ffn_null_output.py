import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.managers.io_struct import AFDReqInput
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class _ForwardMode:
    def __init__(self, *, decode):
        self._decode = decode

    def is_decode(self):
        return self._decode

    def is_extend(self):
        return not self._decode

    def is_prebuilt(self):
        return False

    def is_idle(self):
        return False


class TestAFDFFNNullOutputIsolation(CustomTestCase):
    def setUp(self):
        self.scheduler = object.__new__(Scheduler)
        self.scheduler.disaggregation_mode = DisaggregationMode.NULL
        self.scheduler.server_args = SimpleNamespace(
            afd_perspective=AFDPerspective.AFD_PERSPECTIVE_FFN,
            afd_shared_pool=False,
        )
        self.scheduler.log_batch_result_stats = mock.Mock()
        self.scheduler._maybe_clear_mm_inputs = mock.Mock()
        self.scheduler.maybe_send_health_check_signal = mock.Mock()
        self.scheduler.process_batch_result_prefill = mock.Mock()
        self.scheduler.process_batch_result_disagg_prefill = mock.Mock()
        self.scheduler.process_batch_result_decode = mock.Mock()
        self.scheduler.stream_output = mock.Mock()
        self.scheduler.send_to_tokenizer = mock.Mock()
        self.scheduler.tree_cache = mock.Mock()
        self.scheduler.cur_batch = None
        self.scheduler.last_batch = None
        self.scheduler.running_batch = None

    @staticmethod
    def _req(rid="rid-1", output_ids=None):
        return SimpleNamespace(
            rid=rid,
            origin_input_ids=[11, 12],
            output_ids=list(output_ids or [23]),
            fill_ids=[11, 12] + list(output_ids or [23]),
            prefix_indices=torch.arange(2, dtype=torch.int64),
            extend_input_len=1,
            finished_reason=None,
            grammar=mock.Mock(),
            req_pool_idx=0,
            set_extend_input_len=mock.Mock(),
            check_finished=mock.Mock(
                side_effect=AssertionError("FFN dummy must not check finish")
            ),
            finished=lambda: False,
            afd_pa_instance_id=None,
            afd_pf_instance_id=None,
            afd_lease_id=None,
            afd_pair_epoch=0,
        )

    def _assert_dummy_isolated(self, *, decode, overlap_copy=False):
        req = self._req()
        dummy = torch.tensor([0])
        live_batch = SimpleNamespace(
            reqs=[req],
            forward_mode=_ForwardMode(decode=decode),
            output_ids=dummy,
            device="cpu",
        )
        result_batch = (
            SimpleNamespace(
                reqs=[req],
                forward_mode=live_batch.forward_mode,
                output_ids=dummy,
                device="cpu",
            )
            if overlap_copy
            else live_batch
        )
        self.scheduler.cur_batch = live_batch
        self.scheduler.last_batch = live_batch
        result = SimpleNamespace(next_token_ids=dummy, copy_done=None)

        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=False):
            self.scheduler.process_batch_result(result_batch, result)

        self.assertEqual(req.output_ids, [23])
        self.assertIsNone(result_batch.output_ids)
        self.assertIsNone(live_batch.output_ids)
        req.check_finished.assert_not_called()
        req.grammar.accept_token.assert_not_called()
        self.scheduler.stream_output.assert_not_called()
        self.scheduler.send_to_tokenizer.assert_not_called()
        self.scheduler.tree_cache.cache_unfinished_req.assert_not_called()
        self.scheduler.tree_cache.cache_finished_req.assert_not_called()
        self.scheduler.process_batch_result_prefill.assert_not_called()
        self.scheduler.process_batch_result_decode.assert_not_called()
        self.scheduler.log_batch_result_stats.assert_not_called()
        self.scheduler._maybe_clear_mm_inputs.assert_not_called()
        self.scheduler.maybe_send_health_check_signal.assert_not_called()
        return req, live_batch, dummy

    def test_prefill_dummy_does_not_enter_authoritative_lifecycle(self):
        self._assert_dummy_isolated(decode=False)

    def test_decode_dummy_does_not_enter_authoritative_lifecycle_in_overlap(self):
        self._assert_dummy_isolated(decode=True, overlap_copy=True)

    def test_next_metadata_overwrites_req_and_decode_batch_geometry(self):
        req, batch, dummy = self._assert_dummy_isolated(
            decode=True, overlap_copy=True
        )
        authoritative = AFDReqInput(
            dispatch_id=7,
            batch_size=1,
            forward_mode=ForwardMode.DECODE,
            req_ids=[req.rid],
            seq_lens=[4],
            extend_lens=[1],
            input_ids_per_req=[[11, 12]],
            output_ids_per_req=[[31, 32]],
            max_new_tokens_per_req=[8],
        )
        self.scheduler.waiting_queue = []
        self.scheduler.running_batch = batch
        self.scheduler.last_batch = batch

        self.scheduler._afd_sync_output_ids(authoritative)

        self.assertEqual(req.output_ids, [31, 32])
        self.assertEqual(req.fill_ids, [11, 12, 31, 32])
        self.assertEqual(req.prefix_indices.tolist(), [0, 1, 2])
        req.set_extend_input_len.assert_called_once_with(1)
        self.assertEqual(batch.output_ids.tolist(), [32])
        self.assertIsNot(batch.output_ids, dummy)
        self.assertEqual(
            len(req.origin_input_ids) + len(req.output_ids),
            authoritative.seq_lens[0],
        )

    @staticmethod
    def _metadata(pa_id, rid, tokens, dispatch_id):
        return AFDReqInput(
            dispatch_id=dispatch_id,
            batch_size=1,
            forward_mode=ForwardMode.DECODE,
            req_ids=[rid],
            seq_lens=[2 + len(tokens)],
            extend_lens=[1],
            input_ids_per_req=[[11, 12]],
            output_ids_per_req=[tokens],
            max_new_tokens_per_req=[8],
            pa_instance_id=pa_id,
            pf_instance_id="F0",
            lease_id=f"lease-{pa_id}",
            lease_ids=[f"lease-{pa_id}"],
            pair_epoch=1,
        )

    def _enable_shared_ffn_lanes(self):
        self.scheduler.server_args.afd_shared_pool = True
        self.scheduler.waiting_queue = []
        self.scheduler.chunked_req = None
        self.scheduler._afd_batchsize_attn = None
        self.scheduler._afd_req_ids = None
        self.scheduler._afd_ffn_lane_req_ids = None
        self.scheduler._afd_ffn_active_pa = None
        self.scheduler._afd_ffn_pa_states = {}

    def test_nonshared_next_request_retires_warmup_mirror_before_sync(self):
        warmup = self._req("warmup-rid", [21])
        next_metadata = self._metadata(None, "request-rid", [31], 1)
        self.scheduler.waiting_queue = []
        self.scheduler.running_batch = SimpleNamespace(
            reqs=[warmup], output_ids=None, device="cpu", batch_is_full=True
        )
        self.scheduler.last_batch = self.scheduler.running_batch
        self.scheduler.chunked_req = None
        self.scheduler._afd_pending_batch_infos = deque([next_metadata])
        self.scheduler._afd_batchsize_attn = None
        self.scheduler._afd_req_ids = None
        self.scheduler._afd_ffn_lane_req_ids = None
        self.scheduler._afd_current_metadata = {
            "req_ids": [warmup.rid],
            "dispatch_id": 0,
        }
        self.scheduler.req_to_token_pool = SimpleNamespace(
            no_kv_bookkeeping=True, free=mock.Mock()
        )
        self.scheduler.process_input_requests = mock.Mock()
        self.scheduler._on_afd_batch_info_received = mock.Mock()
        self.scheduler._afd_ensure_reqs_from_afdreq = mock.Mock()

        with (
            mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False),
            mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True),
        ):
            self.scheduler._afd_process_input_requests([])

        self.assertIsNotNone(warmup.finished_reason)
        self.assertEqual(self.scheduler.running_batch.reqs, [])
        self.assertEqual(self.scheduler._afd_req_ids, ["request-rid"])
        self.assertIsNone(next_metadata.pa_instance_id)
        self.scheduler.req_to_token_pool.free.assert_called_once_with(warmup)

    def test_shared_ffn_rejects_true_foreign_rid_in_same_pa_lane(self):
        self._enable_shared_ffn_lanes()
        foreign = self._req("foreign-rid", [21])
        metadata = self._metadata("A0", "current-rid", [31], 15)
        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True):
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(
                self.scheduler, metadata
            )
        self.scheduler.running_batch = SimpleNamespace(
            reqs=[foreign], output_ids=None, device="cpu"
        )

        with self.assertRaisesRegex(RuntimeError, "foreign_rids=.*foreign-rid"):
            self.scheduler._afd_sync_output_ids(metadata)

    def test_shared_ffn_switches_pa_lanes_without_cross_pa_token_check(self):
        self._enable_shared_ffn_lanes()
        a0_req = self._req("a0-rid", [21])
        a1_req = self._req("a1-rid", [31])
        a0 = self._metadata("A0", a0_req.rid, [21], 15)
        a1 = self._metadata("A1", a1_req.rid, [31], 16)

        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True):
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(self.scheduler, a0)
            self.scheduler.running_batch = SimpleNamespace(
                reqs=[a0_req], output_ids=None, device="cpu"
            )
            self.scheduler._afd_ffn_lane_req_ids = [a0_req.rid]
            self.scheduler._afd_sync_output_ids(a0)

            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(self.scheduler, a1)
            self.assertEqual(self.scheduler.running_batch.reqs, [])
            self.scheduler.running_batch = SimpleNamespace(
                reqs=[a1_req], output_ids=None, device="cpu"
            )
            self.scheduler._afd_ffn_lane_req_ids = [a1_req.rid]
            self.scheduler._afd_sync_output_ids(a1)

            a0_next = self._metadata("A0", a0_req.rid, [21, 22], 17)
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(
                self.scheduler, a0_next
            )
            self.scheduler._afd_sync_output_ids(a0_next)

        self.assertIs(self.scheduler.running_batch.reqs[0], a0_req)
        self.assertEqual(a0_req.output_ids, [21, 22])
        self.assertEqual(self.scheduler.running_batch.output_ids.tolist(), [22])
        self.assertEqual(a1_req.output_ids, [31])

    def test_shared_ffn_stale_cleanup_is_scoped_to_active_pa_lane(self):
        self._enable_shared_ffn_lanes()
        a0_req = self._req("a0-rid", [21])
        a1_req = self._req("a1-rid", [31])
        a0 = self._metadata("A0", a0_req.rid, [21], 15)
        a1 = self._metadata("A1", a1_req.rid, [31], 16)

        with (
            mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True),
            mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False),
        ):
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(self.scheduler, a0)
            self.scheduler.waiting_queue = [a0_req]
            self.scheduler._afd_ffn_lane_req_ids = [a0_req.rid]
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(self.scheduler, a1)
            self.scheduler.waiting_queue = [a1_req]
            self.scheduler._afd_ffn_lane_req_ids = [a1_req.rid]
            with mock.patch(
                "sglang.srt.mem_cache.common.release_kv_cache"
            ) as release_kv:
                self.scheduler._afd_ffn_cleanup_stale(set(), {a1_req.rid})

        release_kv.assert_called_once_with(a1_req, self.scheduler.tree_cache)
        self.assertEqual(self.scheduler.waiting_queue, [])
        self.assertEqual(
            self.scheduler._afd_ffn_pa_states["A0"]["waiting_queue"], [a0_req]
        )

    def test_no_kv_unallocated_stale_never_releases_kv(self):
        req = self._req("unallocated")
        req.req_pool_idx = None
        self.scheduler.waiting_queue = [req]
        self.scheduler.req_to_token_pool = SimpleNamespace(
            no_kv_bookkeeping=True, free=mock.Mock()
        )

        with (
            mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False),
            mock.patch("sglang.srt.mem_cache.common.release_kv_cache") as release_kv,
        ):
            self.scheduler._afd_ffn_cleanup_stale(set(), {req.rid})

        release_kv.assert_not_called()
        self.scheduler.req_to_token_pool.free.assert_not_called()
        self.assertTrue(req.finished_reason is not None)
        self.assertEqual(self.scheduler.waiting_queue, [])

    def test_real_kv_allocated_stale_releases_once_and_is_idempotent(self):
        req = self._req("allocated")
        self.scheduler.waiting_queue = [req]
        self.scheduler.req_to_token_pool = SimpleNamespace(no_kv_bookkeeping=False)

        with (
            mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False),
            mock.patch("sglang.srt.mem_cache.common.release_kv_cache") as release_kv,
        ):
            self.scheduler._afd_ffn_cleanup_stale(set(), {req.rid})
            req.req_pool_idx = None
            self.scheduler._afd_ffn_cleanup_stale(set(), {req.rid})

        release_kv.assert_called_once_with(req, self.scheduler.tree_cache)
        self.assertEqual(self.scheduler.waiting_queue, [])

    def test_no_kv_allocated_stale_frees_only_request_slot_once(self):
        req = self._req("logical-slot")
        pool = SimpleNamespace(no_kv_bookkeeping=True)

        def free(freed_req):
            freed_req.req_pool_idx = None

        pool.free = mock.Mock(side_effect=free)
        self.scheduler.req_to_token_pool = pool
        shared_batch = SimpleNamespace(
            reqs=[req], output_ids=None, device="cpu", batch_is_full=True
        )
        self.scheduler.waiting_queue = [req]
        self.scheduler.running_batch = shared_batch
        self.scheduler.last_batch = shared_batch

        with (
            mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False),
            mock.patch("sglang.srt.mem_cache.common.release_kv_cache") as release_kv,
        ):
            self.scheduler._afd_ffn_cleanup_stale(set(), {req.rid})
            self.scheduler._afd_ffn_cleanup_stale(set(), {req.rid})

        release_kv.assert_not_called()
        pool.free.assert_called_once_with(req)
        self.assertEqual(shared_batch.reqs, [])
        self.assertFalse(shared_batch.batch_is_full)

    def test_shared_ffn_cleaned_lane_state_persists_across_pa_switch(self):
        self._enable_shared_ffn_lanes()
        self.scheduler.req_to_token_pool = SimpleNamespace(no_kv_bookkeeping=True)
        self.scheduler.req_to_token_pool.free = mock.Mock(
            side_effect=lambda req: setattr(req, "req_pool_idx", None)
        )
        a0_req = self._req("a0-stale", [21])
        a1_req = self._req("a1-live", [31])
        a0 = self._metadata("A0", a0_req.rid, [21], 15)
        a1 = self._metadata("A1", a1_req.rid, [31], 16)

        with (
            mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True),
            mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False),
        ):
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(self.scheduler, a0)
            self.scheduler.waiting_queue = [a0_req]
            self.scheduler._afd_ffn_lane_req_ids = [a0_req.rid]
            self.scheduler._afd_ffn_cleanup_stale(set(), {a0_req.rid})
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(self.scheduler, a1)
            self.scheduler.waiting_queue = [a1_req]
            self.scheduler._afd_ffn_lane_req_ids = [a1_req.rid]
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(self.scheduler, a0)

        self.assertEqual(self.scheduler.waiting_queue, [])
        self.assertEqual(self.scheduler._afd_ffn_lane_req_ids, [a0_req.rid])
        self.assertEqual(
            self.scheduler._afd_ffn_pa_states["A1"]["waiting_queue"], [a1_req]
        )
        self.scheduler.req_to_token_pool.free.assert_called_once_with(a0_req)

    def test_shared_ffn_rejects_same_pa_missing_authoritative_token(self):
        self._enable_shared_ffn_lanes()
        req = self._req("a0-rid", [21])
        metadata = self._metadata("A0", req.rid, [], 15)
        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True):
            SchedulerAFDMixin.afd_shared_select_ffn_pa_lane(
                self.scheduler, metadata
            )
        self.scheduler.running_batch = SimpleNamespace(
            reqs=[req], output_ids=None, device="cpu"
        )
        with self.assertRaisesRegex(RuntimeError, "same-PA mirror requests"):
            self.scheduler._afd_sync_output_ids(metadata)

    def test_attention_and_pdaf_keep_normal_result_processing(self):
        batch = SimpleNamespace(
            forward_mode=_ForwardMode(decode=False),
            output_ids=torch.tensor([5]),
            is_dllm=lambda: False,
        )
        result = SimpleNamespace(next_token_ids=batch.output_ids)

        self.scheduler.server_args.afd_perspective = (
            AFDPerspective.AFD_PERSPECTIVE_ATTN
        )
        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=False):
            self.scheduler.process_batch_result(batch, result)
        self.scheduler.process_batch_result_prefill.assert_called_once_with(
            batch, result
        )

        self.scheduler.process_batch_result_prefill.reset_mock()
        self.scheduler.disaggregation_mode = DisaggregationMode.PREFILL
        self.scheduler.server_args.afd_perspective = (
            AFDPerspective.AFD_PERSPECTIVE_FFN
        )
        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=False):
            self.scheduler.process_batch_result(batch, result)
        self.scheduler.process_batch_result_prefill.assert_not_called()
        self.scheduler.process_batch_result_disagg_prefill.assert_called_once_with(
            batch, result
        )
        self.assertIs(batch.output_ids, result.next_token_ids)

    def test_direct_normal_processor_fails_fast_for_null_ffn(self):
        batch = SimpleNamespace()
        result = SimpleNamespace()
        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True):
            with self.assertRaisesRegex(AssertionError, "bookkeeping-only"):
                Scheduler.process_batch_result_prefill(self.scheduler, batch, result)
            with self.assertRaisesRegex(AssertionError, "bookkeeping-only"):
                Scheduler.process_batch_result_decode(self.scheduler, batch, result)


if __name__ == "__main__":
    unittest.main(verbosity=3)
