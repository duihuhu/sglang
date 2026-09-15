import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import torch
from sglang.srt.disaggregation.utils import DisaggregationMode
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.managers.io_struct import AFDReqInput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


def _req(rid, *, seq_len=3, token=9):
    req = SimpleNamespace(
        rid=rid,
        req_pool_idx=None,
        prefix_indices=torch.empty(0, dtype=torch.int64),
        extend_input_len=seq_len,
        fill_ids=list(range(seq_len)),
        origin_input_ids=list(range(seq_len)),
        output_ids=[token],
        return_logprob=False,
        stream=False,
        grammar=None,
        finished_reason=None,
    )
    req.finished = lambda: req.finished_reason is not None
    req.init_next_round_input = mock.Mock()
    req.set_extend_input_len = lambda value: setattr(req, "extend_input_len", value)
    return req


def _metadata(mode, req_ids, seq_lens, extend_lens, dispatch_id=71):
    return {
        "dispatch_id": dispatch_id,
        "req_ids": list(req_ids),
        "seq_lens": list(seq_lens),
        "extend_lens": list(extend_lens),
        "geometry_by_rid": {
            rid: (seq_len, extend_len)
            for rid, seq_len, extend_len in zip(req_ids, seq_lens, extend_lens)
        },
    }


def _scheduler(mode, req_ids, seq_lens, extend_lens):
    scheduler = object.__new__(Scheduler)
    scheduler.server_args = SimpleNamespace(
        afd_perspective=AFDPerspective.AFD_PERSPECTIVE_FFN
    )
    scheduler._afd_forward_mode = mode
    scheduler._afd_req_ids = list(req_ids)
    scheduler._afd_batchsize_attn = len(req_ids)
    scheduler._afd_current_metadata = _metadata(mode, req_ids, seq_lens, extend_lens)
    scheduler.get_next_batch_to_run = mock.Mock(
        side_effect=AssertionError("NULL FFN must not run native admission")
    )
    return scheduler


class _DecodeBatch:
    def __init__(self, reqs, seq_lens):
        self.reqs = list(reqs)
        self.seq_lens_cpu = torch.tensor(seq_lens, dtype=torch.int64)
        self.seq_lens = self.seq_lens_cpu.clone()
        self.orig_seq_lens = self.seq_lens_cpu.to(torch.int32)
        self.req_pool_indices = torch.arange(len(reqs), dtype=torch.int64)
        self.output_ids = torch.tensor([req.output_ids[-1] for req in reqs])
        self.device = "cpu"
        self.forward_mode = ForwardMode.EXTEND
        self.sampling_info = SimpleNamespace(
            filter_batch=lambda keep, device_keep: None
        )
        self.model_config = SimpleNamespace(is_encoder_decoder=False)
        self.multimodal_inputs = None
        self.return_logprob = False
        self.spec_info = None
        self.is_spec_v2 = False
        self.mamba_track_indices = None
        self.mamba_track_mask = None
        self.mamba_track_seqlens = None
        self.enable_overlap = False
        self.spec_algorithm = SimpleNamespace(is_none=lambda: True)
        self.input_embeds = None

    def batch_size(self):
        return len(self.reqs)

    def is_empty(self):
        return not self.reqs

    def maybe_wait_verify_done(self):
        pass

    filter_batch = ScheduleBatch.filter_batch

    def merge_batch(self, other):
        self.reqs.extend(other.reqs)
        self.seq_lens_cpu = torch.cat([self.seq_lens_cpu, other.seq_lens_cpu])
        self.seq_lens = torch.cat([self.seq_lens, other.seq_lens])
        self.orig_seq_lens = torch.cat([self.orig_seq_lens, other.orig_seq_lens])
        self.req_pool_indices = torch.cat(
            [self.req_pool_indices, other.req_pool_indices]
        )

    def prepare_for_decode(self):
        self.forward_mode = ForwardMode.DECODE
        self.input_ids = self.output_ids
        self.output_ids = None
        self.seq_lens_cpu = self.seq_lens_cpu + 1


class TestAFDFFNNullAuthoritativeBatch(CustomTestCase):
    def test_perspective_mismatch_global_unavailable_is_nonfatal_once(self):
        scheduler = _scheduler(ForwardMode.DECODE, ["rid"], [4], [1])
        with (
            mock.patch(
                "sglang.srt.layers.afd.afd_is_ffn",
                side_effect=ValueError("global server args are not set"),
            ) as is_ffn,
            mock.patch("sglang.srt.layers.afd.afd_is_attn") as is_attn,
            mock.patch("sglang.srt.managers.scheduler.logger.debug") as debug,
        ):
            scheduler._afd_log_perspective_mismatch()
            scheduler._afd_log_perspective_mismatch()

        is_ffn.assert_called_once_with()
        is_attn.assert_not_called()
        debug.assert_called_once()
        self.assertIn("global=unavailable", debug.call_args.args[0])

    def test_null_debug_global_unavailable_is_safe(self):
        scheduler = _scheduler(ForwardMode.DECODE, ["rid"], [4], [1])
        with (
            mock.patch("sglang.srt.managers.scheduler.AFD_NULL_DEBUG_ENABLED", True),
            mock.patch(
                "sglang.srt.layers.afd.get_afd_perspective",
                side_effect=ValueError("global server args are not set"),
            ),
            mock.patch("sglang.srt.managers.scheduler.logger.info") as info,
        ):
            scheduler._afd_null_debug_log("global_unavailable")

        info.assert_called_once()
        self.assertIn('"global_perspective":"unavailable"', info.call_args.args[1])

    def test_null_debug_helper_is_silent_by_default(self):
        scheduler = _scheduler(ForwardMode.DECODE, ["rid"], [4], [1])
        with (
            mock.patch("sglang.srt.managers.scheduler.AFD_NULL_DEBUG_ENABLED", False),
            mock.patch("sglang.srt.managers.scheduler.logger.info") as info,
        ):
            scheduler._afd_null_debug_log("disabled")
        info.assert_not_called()

    def test_null_debug_helper_emits_structured_batch_fields(self):
        scheduler = _scheduler(ForwardMode.DECODE, ["rid"], [4], [1])
        req = _req("rid", token=23)
        batch = _DecodeBatch([req], [3])
        batch.prepare_for_decode()
        with (
            mock.patch("sglang.srt.managers.scheduler.AFD_NULL_DEBUG_ENABLED", True),
            mock.patch(
                "sglang.srt.layers.afd.get_afd_perspective",
                return_value=AFDPerspective.AFD_PERSPECTIVE_FFN,
            ),
            mock.patch("sglang.srt.managers.scheduler.logger.info") as info,
        ):
            scheduler._afd_null_debug_log(
                "unit",
                dispatch_id=71,
                **scheduler._afd_null_debug_batch_fields(batch),
            )
        args = info.call_args.args
        self.assertEqual(args[0], "[AFD_NULL_DEBUG] %s")
        self.assertIn('"dispatch_id":71', args[1])
        self.assertIn('"rids":["rid"]', args[1])
        self.assertIn('"last":23', args[1])
        self.assertIn('"batch_input_ids":{"shape":[1],"value":[23]}', args[1])

    def test_local_attn_ignores_stale_global_ffn_for_null_batch(self):
        scheduler = _scheduler(ForwardMode.EXTEND, ["wanted"], [3], [2])
        scheduler.server_args.afd_perspective = AFDPerspective.AFD_PERSPECTIVE_ATTN
        native_batch = object()
        scheduler.get_next_batch_to_run = mock.Mock(return_value=native_batch)
        scheduler._afd_build_authoritative_null_batch = mock.Mock(
            side_effect=AssertionError("local ATTN must not build FFN mirror batch")
        )

        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True):
            batch = scheduler._afd_get_next_batch(DisaggregationMode.NULL)

        self.assertIs(batch, native_batch)
        scheduler.get_next_batch_to_run.assert_called_once_with()
        scheduler._afd_build_authoritative_null_batch.assert_not_called()

    def test_extend_uses_only_snapshot_order_and_skips_native_scheduler(self):
        scheduler = _scheduler(ForwardMode.EXTEND, ["wanted"], [3], [2])
        wanted = _req("wanted", seq_len=3)
        scheduler.waiting_queue = [
            _req("extra"),
            wanted,
            _req("extra"),
            _req("other-extra"),
        ]
        scheduler.req_to_token_pool = SimpleNamespace(
            available_size=lambda: 8, device="cpu"
        )
        scheduler.token_to_kv_pool_allocator = SimpleNamespace(
            available_size=lambda: 32
        )
        scheduler.tree_cache = object()
        scheduler.model_config = object()
        scheduler.enable_overlap = False
        scheduler.spec_algorithm = object()
        scheduler.tp_rank = 0
        scheduler.tp_size = 1
        scheduler.new_token_ratio = 0.5
        scheduler.running_batch = SimpleNamespace(reqs=[])
        scheduler.enable_priority_scheduling = False
        scheduler.maybe_prepare_mlp_sync_batch = mock.Mock(
            side_effect=lambda batch: batch
        )

        def init_new(cls, reqs, *args, **kwargs):
            batch = SimpleNamespace(reqs=list(reqs), prefill_stats=None)
            batch.batch_size = lambda: len(batch.reqs)

            def prepare():
                batch.forward_mode = ForwardMode.EXTEND
                batch.extend_lens = [req.extend_input_len for req in batch.reqs]
                batch.input_ids = torch.tensor(
                    [
                        token
                        for req in batch.reqs
                        for token in req.fill_ids[-req.extend_input_len :]
                    ]
                )

            batch.prepare_for_extend = prepare
            return batch

        with (
            mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=False),
            mock.patch.object(ScheduleBatch, "init_new", classmethod(init_new)),
            mock.patch(
                "sglang.srt.observability.scheduler_metrics_mixin.PrefillStats.from_authoritative",
                return_value=object(),
            ),
        ):
            batch = scheduler._afd_get_next_batch(DisaggregationMode.NULL)

        self.assertEqual([req.rid for req in batch.reqs], ["wanted"])
        self.assertEqual(batch.extend_lens, [2])
        self.assertEqual(
            [req.rid for req in scheduler.waiting_queue],
            ["extra", "extra", "other-extra"],
        )
        scheduler.get_next_batch_to_run.assert_not_called()

    def test_decode_reorders_mirror_and_uses_authoritative_last_tokens(self):
        scheduler = _scheduler(ForwardMode.DECODE, ["b", "a"], [5, 4], [1, 1])
        req_a = _req("a", token=101)
        req_b = _req("b", token=202)
        scheduler.waiting_queue = [_req("waiting-extra")]
        scheduler.running_batch = _DecodeBatch(
            [_req("running-extra"), req_a, req_b], [7, 3, 4]
        )
        scheduler.last_batch = None

        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=False):
            batch = scheduler._afd_get_next_batch(DisaggregationMode.NULL)

        self.assertEqual([req.rid for req in batch.reqs], ["b", "a"])
        self.assertEqual(batch.input_ids.tolist(), [202, 101])
        self.assertEqual(batch.seq_lens_cpu.tolist(), [5, 4])
        self.assertEqual(batch.forward_mode, ForwardMode.DECODE)
        scheduler.get_next_batch_to_run.assert_not_called()

    def test_concurrent_extend_merges_into_active_decode_mirror(self):
        req1 = _req("req1", seq_len=3, token=31)
        req2 = _req("req2", seq_len=2, token=41)
        running = _DecodeBatch([req1], [3])
        running.prepare_for_decode()
        running.output_ids = None

        scheduler = _scheduler(ForwardMode.DECODE, ["req1"], [4], [1])
        scheduler.server_args.afd_shared_pool = False
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.waiting_queue = []
        scheduler.running_batch = running
        scheduler.cur_batch = None
        scheduler.last_batch = None
        scheduler._afd_ffn_active_req_ids = ["req1"]
        scheduler._afd_batchsize_attn = None
        scheduler._afd_pending_batch_infos = deque()
        scheduler.process_input_requests = mock.Mock()
        scheduler._on_afd_batch_info_received = mock.Mock()
        scheduler._afd_ensure_reqs_from_afdreq = mock.Mock()
        scheduler.req_to_token_pool = SimpleNamespace(no_kv_bookkeeping=True)
        scheduler.tree_cache = object()

        extend = AFDReqInput(
            dispatch_id=72,
            batch_size=1,
            forward_mode=ForwardMode.EXTEND,
            req_ids=["req2"],
            seq_lens=[2],
            extend_lens=[2],
            input_ids_per_req=[[0, 1]],
            output_ids_per_req=[[]],
            max_new_tokens_per_req=[8],
        )
        scheduler._afd_pending_batch_infos.append(extend)
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            scheduler._afd_process_input_requests([])

        self.assertEqual([req.rid for req in scheduler.running_batch.reqs], ["req1"])
        self.assertFalse(req1.finished())
        self.assertEqual(scheduler._afd_ffn_active_req_ids, ["req1", "req2"])

        extend_batch = _DecodeBatch([req2], [2])
        extend_batch.forward_mode = ForwardMode.EXTEND
        extend_batch.output_ids = torch.tensor([999])
        scheduler.cur_batch = extend_batch
        scheduler.last_batch = extend_batch
        result = SimpleNamespace(next_token_ids=extend_batch.output_ids, copy_done=None)
        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True):
            scheduler.process_batch_result_afd_ffn_null(extend_batch, result)

        self.assertEqual(
            [req.rid for req in scheduler.running_batch.reqs], ["req1", "req2"]
        )

        decode = AFDReqInput(
            dispatch_id=73,
            batch_size=2,
            forward_mode=ForwardMode.DECODE,
            req_ids=["req1", "req2"],
            seq_lens=[5, 3],
            extend_lens=[1, 1],
            input_ids_per_req=[[0, 1, 2], [0, 1]],
            output_ids_per_req=[[30, 31], [41]],
            max_new_tokens_per_req=[8, 8],
        )
        scheduler._afd_batchsize_attn = None
        scheduler._afd_pending_batch_infos.append(decode)
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            scheduler._afd_process_input_requests([])
        batch = scheduler._afd_build_authoritative_decode_batch()

        self.assertEqual([req.rid for req in batch.reqs], ["req1", "req2"])
        self.assertEqual(batch.input_ids.tolist(), [31, 41])
        self.assertEqual(scheduler._afd_ffn_active_req_ids, ["req1", "req2"])

    def test_null_ffn_mirror_survives_extend_and_repeated_decode(self):
        scheduler = _scheduler(ForwardMode.EXTEND, ["live"], [3], [3])
        req = _req("live", seq_len=3, token=17)
        extend_batch = _DecodeBatch([req], [3])
        req.output_ids = []
        req.fill_ids = [0, 1, 2]
        extend_batch.output_ids = torch.tensor([999])
        extend_batch.forward_mode = ForwardMode.EXTEND
        scheduler.running_batch = _DecodeBatch([], [])
        scheduler.cur_batch = extend_batch
        scheduler.last_batch = None
        scheduler.disaggregation_mode = DisaggregationMode.NULL
        scheduler.waiting_queue = []
        scheduler.tree_cache = object()

        dummy = extend_batch.output_ids
        result = SimpleNamespace(next_token_ids=dummy, copy_done=None)
        with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True):
            scheduler.process_batch_result_afd_ffn_null(extend_batch, result)

        self.assertIs(scheduler.running_batch, extend_batch)
        self.assertEqual([r.rid for r in scheduler.running_batch.reqs], ["live"])
        self.assertIs(scheduler.running_batch.reqs[0], req)
        self.assertIsNone(scheduler.running_batch.output_ids)
        self.assertEqual(req.output_ids, [])

        for dispatch_id, token, seq_len in ((72, 41, 4), (73, 42, 5)):
            req.output_ids = [token]
            scheduler._afd_forward_mode = ForwardMode.DECODE
            scheduler._afd_req_ids = ["live"]
            scheduler._afd_batchsize_attn = 1
            scheduler._afd_current_metadata = _metadata(
                ForwardMode.DECODE, ["live"], [seq_len], [1], dispatch_id
            )
            batch = scheduler._afd_build_authoritative_decode_batch()
            self.assertIs(batch.reqs[0], req)
            self.assertEqual(batch.input_ids.tolist(), [token])

            decode_dummy = torch.tensor([999])
            batch.output_ids = decode_dummy
            scheduler.cur_batch = batch
            with mock.patch("sglang.srt.layers.afd.afd_is_ffn", return_value=True):
                scheduler.process_batch_result_afd_ffn_null(
                    batch, SimpleNamespace(next_token_ids=decode_dummy, copy_done=None)
                )
            self.assertIs(scheduler.running_batch, batch)
            self.assertIsNone(scheduler.running_batch.output_ids)

        scheduler.req_to_token_pool = SimpleNamespace(no_kv_bookkeeping=True)
        scheduler.req_to_token_pool.free = mock.Mock()
        with (
            mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False),
            mock.patch("sglang.srt.mem_cache.common.release_kv_cache") as release,
        ):
            scheduler._afd_ffn_cleanup_stale(set(), {"live"})
        self.assertTrue(req.finished())
        release.assert_not_called()
        scheduler.req_to_token_pool.free.assert_not_called()
        self.assertEqual(scheduler.running_batch.reqs, [])

    def test_missing_duplicate_and_geometry_fail_before_forward(self):
        cases = []
        missing = _scheduler(ForwardMode.DECODE, ["missing"], [4], [1])
        missing.running_batch = _DecodeBatch([_req("other")], [3])
        missing.last_batch = None
        cases.append((missing, "missing"))

        duplicate = _scheduler(ForwardMode.DECODE, ["dup"], [4], [1])
        duplicate.running_batch = _DecodeBatch([_req("dup"), _req("dup")], [3, 3])
        duplicate.last_batch = None
        cases.append((duplicate, "duplicates"))

        geometry = _scheduler(ForwardMode.DECODE, ["rid"], [99], [1])
        geometry.running_batch = _DecodeBatch([_req("rid")], [3])
        geometry.last_batch = None
        cases.append((geometry, "geometry mismatch"))

        for scheduler, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(RuntimeError, message):
                    scheduler._afd_build_authoritative_null_batch()
                scheduler.get_next_batch_to_run.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=3)
