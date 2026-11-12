import threading
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation.prefill import SchedulerDisaggregationPrefillMixin
from sglang.srt.layers.afd import AsyncTensorCommunicator
from sglang.srt.managers.io_struct import (
    AFDReqInput,
    AbortReq,
    BatchTokenizedGenerateReqInput,
    TokenizedEmbeddingReqInput,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import Scheduler
from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.observability.scheduler_metrics_mixin import (
    PrefillStats,
    SchedulerMetricsMixin,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


def _afd_req(
    *,
    dispatch_id=41,
    req_ids=None,
    seq_lens=None,
    extend_lens=None,
    input_ids_per_req=None,
    output_ids_per_req=None,
):
    return AFDReqInput(
        dispatch_id=dispatch_id,
        req_ids=req_ids or ["rid-1"],
        seq_lens=seq_lens or [9],
        extend_lens=extend_lens or [1],
        input_ids_per_req=input_ids_per_req or [list(range(9))],
        output_ids_per_req=output_ids_per_req,
        max_new_tokens_per_req=[8] * len(req_ids or ["rid-1"]),
    )


def _scheduler(*, waiting=None, running=None, last=None):
    scheduler = object.__new__(Scheduler)
    scheduler.waiting_queue = list(waiting or [])
    scheduler.running_batch = running
    scheduler.last_batch = last
    scheduler.model_config = SimpleNamespace(vocab_size=128)
    return scheduler


def _fake_req(rid="rid-1"):
    req = SimpleNamespace(
        rid=rid,
        origin_input_ids=[],
        output_ids=[],
        fill_ids=[],
        prefix_indices=torch.empty(0, dtype=torch.int64),
        logprob_start_len=-1,
        req_pool_idx=None,
        extend_input_len=0,
    )

    def set_extend_input_len(value):
        req.extend_input_len = value

    req.set_extend_input_len = set_extend_input_len
    return req


class TestAFDPrefillReqMetadata(CustomTestCase):
    def test_new_req_preserves_repeated_prompt_prefix_hit_geometry(self):
        scheduler = _scheduler()
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            scheduler._afd_ensure_reqs_from_afdreq(_afd_req())

        self.assertEqual(len(scheduler.waiting_queue), 1)
        req = scheduler.waiting_queue[0]
        self.assertEqual(len(req.fill_ids), 9)
        self.assertEqual(len(req.prefix_indices), 8)
        self.assertEqual(req.extend_input_len, 1)

    def test_existing_waiting_and_last_batch_reqs_are_updated_without_duplicates(self):
        waiting_req = _fake_req()
        last_req = _fake_req()
        scheduler = _scheduler(
            waiting=[waiting_req], last=SimpleNamespace(reqs=[last_req])
        )
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            scheduler._afd_ensure_reqs_from_afdreq(_afd_req())

        self.assertEqual(scheduler.waiting_queue, [waiting_req])
        for req in (waiting_req, last_req):
            self.assertEqual(len(req.fill_ids), 9)
            self.assertEqual(len(req.prefix_indices), 8)
            self.assertEqual(req.extend_input_len, 1)

    def test_authoritative_prefill_snapshot_repairs_joining_rank_and_order(self):
        req_ids = [f"rid-{i}" for i in range(5)]
        afd_req = _afd_req(
            req_ids=req_ids,
            seq_lens=[9] * 5,
            extend_lens=[9] * 5,
            input_ids_per_req=[[i] * 9 for i in range(5)],
        )
        survivor = _scheduler(waiting=[_fake_req(rid) for rid in req_ids])
        joining = _scheduler(waiting=[_fake_req(req_ids[-1])])

        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            survivor._afd_ensure_reqs_from_afdreq(
                afd_req, authoritative_waiting=True
            )
            joining._afd_ensure_reqs_from_afdreq(
                afd_req, authoritative_waiting=True
            )

        self.assertEqual([req.rid for req in survivor.waiting_queue], req_ids)
        self.assertEqual([req.rid for req in joining.waiting_queue], req_ids)
        self.assertEqual(
            [req.origin_input_ids for req in joining.waiting_queue],
            [[i] * 9 for i in range(5)],
        )

    def test_authoritative_prefill_snapshot_is_idempotent_and_drops_extras(self):
        afd_req = _afd_req(
            req_ids=["a", "b"],
            seq_lens=[2, 2],
            extend_lens=[2, 2],
            input_ids_per_req=[[1, 2], [3, 4]],
        )
        a = _fake_req("a")
        b = _fake_req("b")
        scheduler = _scheduler(waiting=[_fake_req("extra"), b, a])

        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            scheduler._afd_ensure_reqs_from_afdreq(
                afd_req, authoritative_waiting=True
            )
            first_objects = list(scheduler.waiting_queue)
            scheduler._afd_ensure_reqs_from_afdreq(
                afd_req, authoritative_waiting=True
            )

        self.assertEqual([req.rid for req in scheduler.waiting_queue], ["a", "b"])
        self.assertEqual(scheduler.waiting_queue, first_objects)
        self.assertIs(scheduler.waiting_queue[0], a)
        self.assertIs(scheduler.waiting_queue[1], b)

    def test_authoritative_prefill_snapshot_rejects_duplicate_local_req(self):
        afd_req = _afd_req()
        scheduler = _scheduler(waiting=[_fake_req(), _fake_req()])

        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            with self.assertRaisesRegex(
                RuntimeError, "dispatch=41 rid=rid-1.*found=2"
            ):
                scheduler._afd_ensure_reqs_from_afdreq(
                    afd_req, authoritative_waiting=True
                )

    def test_authoritative_prefill_snapshot_requires_creation_metadata(self):
        scheduler = _scheduler(waiting=[_fake_req("rid-1")])
        afd_req = _afd_req()
        afd_req.input_ids_per_req = None

        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            with self.assertRaisesRegex(
                ValueError, "dispatch=41.*input_ids_per_req"
            ):
                scheduler._afd_ensure_reqs_from_afdreq(
                    afd_req, authoritative_waiting=True
                )

    def test_invalid_metadata_fails_with_dispatch_and_rid_context(self):
        invalid = (
            _afd_req(extend_lens=[10]),
            _afd_req(seq_lens=[8]),
            _afd_req(req_ids=["rid-1", "rid-2"]),
        )
        for afd_req in invalid:
            with self.subTest(afd_req=afd_req):
                with self.assertRaises(ValueError) as raised:
                    Scheduler._afd_validate_req_metadata(afd_req)
                message = str(raised.exception)
                self.assertIn("dispatch=41", message)
                self.assertIn("rid", message)


class TestAFDPrefillGeometryRestore(CustomTestCase):
    def setUp(self):
        self.scheduler = SimpleNamespace(_afd_current_metadata=None)
        SchedulerAFDMixin.afd_set_current_metadata(
            self.scheduler, _afd_req(dispatch_id=52)
        )

    def test_restores_prefix_after_cache_init_clears_it(self):
        req = _fake_req()
        req.prefix_indices = torch.empty(0, dtype=torch.int64)
        req.extend_input_len = 9

        SchedulerAFDMixin.afd_restore_req_geometry(self.scheduler, req)

        self.assertEqual(req.prefix_indices.tolist(), list(range(8)))
        self.assertEqual(req.extend_input_len, 1)

    def test_unknown_rid_fails_with_dispatch_context(self):
        req = _fake_req("unknown")
        with self.assertRaisesRegex(
            RuntimeError, "dispatch=52 rid=unknown"
        ):
            SchedulerAFDMixin.afd_restore_req_geometry(self.scheduler, req)


class TestAFDComponentReshardAdmissionGate(CustomTestCase):
    def setUp(self):
        self.scheduler = SimpleNamespace(
            _afd_component_runtime=SimpleNamespace(admission_blocked=True),
            _afd_component_fence_installed=True,
        )
        self.generate = object.__new__(TokenizedGenerateReqInput)
        self.embedding = object.__new__(TokenizedEmbeddingReqInput)
        self.batch = BatchTokenizedGenerateReqInput(batch=[self.generate])

    def _gate(self, recv_reqs):
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            return SchedulerAFDMixin.afd_gate_work_requests(
                self.scheduler, recv_reqs
            )

    def test_pending_request_is_not_effective_until_active_sync(self):
        same_round = object.__new__(TokenizedGenerateReqInput)
        self.scheduler._afd_component_fence_installed = False
        self.scheduler._afd_component_pending_fence_payload = {"operation_id": "op"}

        rank0 = self._gate([same_round])
        self.assertEqual(rank0, [same_round])
        self.assertEqual(
            list(getattr(self.scheduler, "_afd_deferred_work_requests", [])), []
        )
        rank0_waiting = []
        rank0_waiting.extend(rank0)

        # The follower has the same effective state in the request round.
        follower = SimpleNamespace(_afd_component_fence_installed=False)
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            follower_same_round = SchedulerAFDMixin.afd_gate_work_requests(
                follower, [same_round]
            )
        self.assertEqual(follower_same_round, [same_round])
        follower_waiting = []
        follower_waiting.extend(follower_same_round)
        self.assertEqual(rank0_waiting, follower_waiting)

        # Next iteration's serving-TP sync atomically installs the effective
        # fence on both ranks; only work received after that point is deferred.
        self.scheduler._afd_component_fence_installed = True
        follower._afd_component_fence_installed = True
        rank0_new = object.__new__(TokenizedGenerateReqInput)
        follower_new = object.__new__(TokenizedGenerateReqInput)
        self.assertEqual(self._gate([rank0_new]), [])
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            self.assertEqual(
                SchedulerAFDMixin.afd_gate_work_requests(follower, [follower_new]),
                [],
            )
        self.assertEqual(
            len(self.scheduler._afd_deferred_work_requests), 1
        )
        self.assertEqual(len(follower._afd_deferred_work_requests), 1)

    def test_fenced_work_is_deferred_and_not_processable(self):
        self.assertEqual(self._gate([self.generate, self.batch, self.embedding]), [])
        self.assertEqual(
            list(self.scheduler._afd_deferred_work_requests),
            [self.generate, self.batch, self.embedding],
        )

    def test_reopen_flushes_deferred_before_current_without_repacking(self):
        self._gate([self.batch, self.embedding])
        self.scheduler._afd_component_runtime.admission_blocked = False
        self.scheduler._afd_component_fence_installed = False
        current = object.__new__(TokenizedGenerateReqInput)

        processable = self._gate([current])

        self.assertEqual(processable, [self.batch, self.embedding, current])
        self.assertIs(processable[0], self.batch)
        self.assertEqual(list(self.scheduler._afd_deferred_work_requests), [])

    def test_follower_fence_does_not_depend_on_rank_zero_runtime(self):
        del self.scheduler._afd_component_runtime
        self.assertEqual(self._gate([self.generate]), [])
        self.assertEqual(
            list(self.scheduler._afd_deferred_work_requests), [self.generate]
        )

    def test_abort_and_afd_control_are_not_deferred(self):
        abort = AbortReq(rid="rid-1")
        afd_req = _afd_req()

        self.assertEqual(self._gate([abort, afd_req, self.generate]), [abort, afd_req])
        self.assertEqual(
            list(self.scheduler._afd_deferred_work_requests), [self.generate]
        )

    def test_fenced_abort_is_forwarded_while_new_work_stays_blocked(self):
        abort = AbortReq(rid="rid-before-cut")
        new_work = object.__new__(TokenizedGenerateReqInput)
        processable = self._gate([abort, new_work])
        self.assertEqual(processable, [abort])

        socket = mock.Mock()
        self.scheduler.afd_send_to_ffn = socket
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            SchedulerAFDMixin.afd_forward_work_requests(
                self.scheduler, processable
            )
        socket.send_pyobj.assert_called_once_with(abort)
        self.assertEqual(
            list(self.scheduler._afd_deferred_work_requests), [new_work]
        )

    def test_ffn_is_unchanged_even_when_runtime_is_fenced(self):
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            processable = SchedulerAFDMixin.afd_gate_work_requests(
                self.scheduler, [self.generate]
            )
        self.assertEqual(processable, [self.generate])
        self.assertFalse(hasattr(self.scheduler, "_afd_deferred_work_requests"))


class TestAFDPrefillDispatchMetadata(CustomTestCase):
    @staticmethod
    def _send(cases, batch_extend_lens, forward_mode):
        class Socket:
            def send_pyobj(self, value):
                self.value = value

        reqs = []
        for rid, seq_len, extend_len, input_len, output_len in cases:
            reqs.append(
                SimpleNamespace(
                    rid=rid,
                    extend_input_len=extend_len,
                    seqlen=seq_len,
                    origin_input_ids=list(range(input_len)),
                    output_ids=list(range(output_len)),
                    sampling_params=SimpleNamespace(max_new_tokens=8),
                )
            )
        batch = SimpleNamespace(
            reqs=reqs,
            extend_lens=batch_extend_lens,
            forward_mode=forward_mode,
            batch_size=lambda: len(reqs),
        )
        socket = Socket()
        scheduler = SimpleNamespace(
            _afd_component_runtime=None,
            _afd_reshard_drain=None,
            afd_send_to_ffn=socket,
            server_args=SimpleNamespace(afd_multi_pf_continuation=False),
        )
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            SchedulerAFDMixin.afd_send_batch_info(scheduler, batch)
        return socket.value

    def test_prefill_preserves_per_request_extend_geometry(self):
        message = self._send(
            (
                ("a", 9, 1, 9, 0),
                ("b", 10, 5, 10, 0),
                ("c", 11, 9, 11, 0),
            ),
            [1, 5, 9],
            ForwardMode.EXTEND,
        )
        self.assertEqual(message.req_ids, ["a", "b", "c"])
        self.assertEqual(message.seq_lens, [9, 10, 11])
        self.assertEqual(message.extend_lens, [1, 5, 9])

    def test_decode_ignores_stale_batch_extend_lens(self):
        message = self._send(
            (("a", 7, 1, 4, 3), ("b", 6, 1, 4, 2)),
            [1],
            ForwardMode.DECODE,
        )
        self.assertEqual(message.req_ids, ["a", "b"])
        self.assertEqual(message.seq_lens, [7, 6])
        self.assertEqual(message.extend_lens, [1, 1])

    def test_prefence_batch_dispatches_while_admission_is_blocked(self):
        class Drain:
            def __init__(self):
                self.calls = 0

            def next_dispatch(self):
                self.calls += 1
                return 17

        drain = Drain()
        with mock.patch(
            "sglang.srt.layers.afd.afd_is_attn", return_value=True
        ):
            class Socket:
                def send_pyobj(self, value):
                    self.value = value

            socket = Socket()
            req = SimpleNamespace(
                rid="pre-fence",
                extend_input_len=1,
                seqlen=4,
                origin_input_ids=[1, 2, 3, 4],
                output_ids=[],
                sampling_params=SimpleNamespace(max_new_tokens=8),
            )
            batch = SimpleNamespace(
                reqs=[req],
                extend_lens=[1],
                forward_mode=ForwardMode.EXTEND,
                batch_size=lambda: 1,
            )
            scheduler = SimpleNamespace(
                _afd_component_runtime=SimpleNamespace(admission_blocked=True),
                _afd_reshard_drain=drain,
                afd_send_to_ffn=socket,
                server_args=SimpleNamespace(afd_multi_pf_continuation=False),
            )

            SchedulerAFDMixin.afd_send_batch_info(scheduler, batch)

        self.assertEqual(drain.calls, 1)
        self.assertEqual(socket.value.dispatch_id, 17)
        self.assertEqual(socket.value.req_ids, ["pre-fence"])


class TestAFDPrefillBatchMetadata(CustomTestCase):
    def setUp(self):
        self.scheduler = SimpleNamespace(
            _afd_current_metadata={
                "dispatch_id": 73,
                "req_ids": ["a", "b", "c"],
                "extend_lens": [1, 5, 9],
                "seq_lens": [1, 5, 9],
            }
        )

    def test_matching_extend_lens_pass(self):
        batch = SimpleNamespace(
            reqs=[SimpleNamespace(rid=rid) for rid in ("a", "b", "c")],
            extend_lens=[1, 5, 9],
        )
        SchedulerAFDMixin.afd_validate_prefill_batch(self.scheduler, batch)

    def test_rid_or_extend_mismatch_reports_both_geometries(self):
        batches = (
            SimpleNamespace(
                reqs=[SimpleNamespace(rid=rid) for rid in ("a", "x", "c")],
                extend_lens=[1, 5, 9],
            ),
            SimpleNamespace(
                reqs=[SimpleNamespace(rid=rid) for rid in ("a", "b", "c")],
                extend_lens=[1, 4, 9],
            ),
        )
        for batch in batches:
            with self.subTest(batch=batch):
                with self.assertRaises(RuntimeError) as raised:
                    SchedulerAFDMixin.afd_validate_prefill_batch(
                        self.scheduler, batch
                    )
                message = str(raised.exception)
                self.assertIn("dispatch=73", message)
                self.assertIn("attn_extend", message)
                self.assertIn("ffn_extend", message)


class TestAFDAuthoritativePrefillBatchConstruction(CustomTestCase):
    @staticmethod
    def _scheduler(req_ids, *, req_capacity):
        reqs = [_fake_req(rid) for rid in req_ids]
        for req in reqs:
            req.init_next_round_input = mock.Mock()
        extend_lens = list(range(1, len(req_ids) + 1))
        seq_lens = [extend_len + 2 for extend_len in extend_lens]
        metadata = {
            "dispatch_id": 115,
            "req_ids": list(req_ids),
            "extend_lens": extend_lens,
            "seq_lens": seq_lens,
            "geometry_by_rid": {
                rid: (seq_len, extend_len)
                for rid, seq_len, extend_len in zip(
                    req_ids, seq_lens, extend_lens
                )
            },
        }
        pool = SimpleNamespace(
            available_size=mock.Mock(return_value=req_capacity),
            device="cpu",
        )
        token_pool = SimpleNamespace(
            available_size=mock.Mock(return_value=1000),
        )
        return SimpleNamespace(
            _afd_current_metadata=metadata,
            _afd_req_ids=list(req_ids),
            _afd_batchsize_attn=len(req_ids),
            _afd_ffn_authoritative_waiting=True,
            waiting_queue=reqs,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=token_pool,
            tree_cache=object(),
            model_config=object(),
            enable_overlap=False,
            spec_algorithm=object(),
            tp_rank=0,
            tp_size=1,
            new_token_ratio=0.37,
            running_batch=SimpleNamespace(reqs=[_fake_req("running")]),
            enable_priority_scheduling=False,
            maybe_prepare_mlp_sync_batch=mock.Mock(side_effect=lambda batch: batch),
            get_new_batch_prefill=mock.Mock(
                return_value=SimpleNamespace(
                    reqs=reqs[:1], batch_size=lambda: 1
                )
            ),
        )

    def test_local_adder_capacity_one_does_not_truncate_authoritative_batch(self):
        req_ids = [f"rid-{i}" for i in range(5)]
        survivor = self._scheduler(req_ids, req_capacity=5)
        joining = self._scheduler(req_ids, req_capacity=5)

        def prepare(batch):
            batch.extend_lens = [req.extend_input_len for req in batch.reqs]

        def init_new(cls, reqs, *args, **kwargs):
            batch = SimpleNamespace(
                reqs=list(reqs),
                batch_size=lambda: len(reqs),
            )
            batch.prepare_for_extend = lambda: prepare(batch)
            batch.prefill_stats = None
            return batch

        with mock.patch.object(
            ScheduleBatch, "init_new", classmethod(init_new)
        ):
            survivor_batch = (
                SchedulerDisaggregationPrefillMixin._afd_build_authoritative_prefill_batch(
                    survivor
                )
            )
            joining_batch = (
                SchedulerDisaggregationPrefillMixin._afd_build_authoritative_prefill_batch(
                    joining
                )
            )

        self.assertEqual([req.rid for req in survivor_batch.reqs], req_ids)
        self.assertEqual([req.rid for req in joining_batch.reqs], req_ids)
        self.assertEqual(survivor_batch.batch_size(), 5)
        self.assertEqual(joining_batch.batch_size(), 5)
        survivor.get_new_batch_prefill.assert_not_called()
        joining.get_new_batch_prefill.assert_not_called()
        for batch in (survivor_batch, joining_batch):
            self.assertIsInstance(batch.prefill_stats, PrefillStats)
            self.assertEqual(batch.prefill_stats.log_input_tokens, 15)
            self.assertEqual(batch.prefill_stats.log_hit_tokens, 10)
            self.assertEqual(batch.prefill_stats.new_token_ratio, 0.37)
            self.assertEqual(batch.prefill_stats.num_running_reqs.total, 1)
            self.assertEqual(batch.prefill_stats.num_new_seqs, 5)
            reporter = mock.Mock()
            reporter.report_prefill_stats(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=False,
                dp_cooperation_info=None,
            )
            reporter.report_prefill_stats.assert_called_once_with(
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=False,
                dp_cooperation_info=None,
            )
            metrics_scheduler = SimpleNamespace(
                is_stats_logging_rank=False,
                current_scheduler_metrics_enabled=False,
            )
            SchedulerMetricsMixin.report_prefill_stats(
                metrics_scheduler,
                prefill_stats=batch.prefill_stats,
                can_run_cuda_graph=False,
                dp_cooperation_info=None,
            )

    def test_authoritative_batch_capacity_shortage_fails_instead_of_bs_one(self):
        req_ids = [f"rid-{i}" for i in range(5)]
        joining = self._scheduler(req_ids, req_capacity=1)

        with self.assertRaisesRegex(
            RuntimeError,
            "capacity failure dispatch=115.*req_pool_needed.*5.*req_pool_available.*1",
        ):
            SchedulerDisaggregationPrefillMixin._afd_build_authoritative_prefill_batch(
                joining
            )

        self.assertEqual([req.rid for req in joining.waiting_queue], req_ids)


class TestAFDDecodeMetadataConsumption(CustomTestCase):
    def test_two_metadata_messages_are_consumed_one_per_iteration(self):
        scheduler = object.__new__(Scheduler)
        SchedulerAFDMixin.afd_init_state(scheduler)
        scheduler.server_args = SimpleNamespace(afd_multi_pf_continuation=False)
        scheduler.afd_send_to_ffn = None
        scheduler.waiting_queue = []
        scheduler.running_batch = None
        scheduler.last_batch = None
        scheduler.model_config = SimpleNamespace(vocab_size=128)
        scheduler.process_input_requests = mock.Mock()
        scheduler._on_afd_batch_info_received = mock.Mock()
        scheduler._afd_sync_output_ids = mock.Mock()
        scheduler._afd_ensure_reqs_from_afdreq = mock.Mock()
        first = _afd_req(dispatch_id=101, req_ids=["a"])
        second = _afd_req(dispatch_id=102, req_ids=["b"])

        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            scheduler._afd_process_input_requests(
                [first, second], work_already_forwarded=True
            )
            self.assertEqual(scheduler._afd_current_metadata["dispatch_id"], 101)
            self.assertEqual(list(scheduler._afd_pending_batch_infos), [second])
            scheduler._afd_ffn_authoritative_waiting = True
            scheduler.afd_reset_state()
            self.assertFalse(scheduler._afd_ffn_authoritative_waiting)
            scheduler._afd_process_input_requests([], work_already_forwarded=True)

        self.assertEqual(scheduler._afd_current_metadata["dispatch_id"], 102)
        self.assertEqual(list(scheduler._afd_pending_batch_infos), [])
        self.assertEqual(
            scheduler._afd_ensure_reqs_from_afdreq.call_args_list,
            [
                mock.call(first, authoritative_waiting=False),
                mock.call(second, authoritative_waiting=False),
            ],
        )


class TestAsyncTensorCommunicatorRecvCapability(CustomTestCase):
    def test_nonconcurrent_backend_receives_synchronously(self):
        caller_thread = threading.get_ident()

        class Inner:
            supports_concurrent_recv = False

            def __init__(self):
                self.recv_thread = None

            def recv_tensor(self):
                self.recv_thread = threading.get_ident()
                return torch.tensor([7])

        inner = Inner()
        with mock.patch("torch.cuda.is_available", return_value=False):
            communicator = AsyncTensorCommunicator(inner)
            communicator.recv_start()

        self.assertEqual(inner.recv_thread, caller_thread)
        self.assertEqual(communicator._pending_recv_count, 1)
        self.assertTrue(all(thread is None for thread in communicator._recv_threads))
        self.assertEqual(communicator._recv_ring[0].item(), 7)


if __name__ == "__main__":
    unittest.main(verbosity=3)
