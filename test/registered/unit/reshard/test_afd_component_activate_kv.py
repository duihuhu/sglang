"""CPU/mock coverage for AFD component reshard activate KV-runtime bring-up.

Regression guard for the joining-rank activate crash:
``AttributeError: 'ModelRunner' object has no attribute 'token_to_kv_pool'``.

A standby joining rank skips ``initialize()``/``_init_pools()``, so the KV
runtime attributes never get created. Activation must build them (ATTN) or
explicitly run the no-KV path (FFN) without touching missing attributes.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.reshard.afd_component_standby import (
    AFDComponentRankLifecycle,
    AFDComponentRankState,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


def _bare_runner():
    runner = object.__new__(ModelRunner)
    runner.device = "cpu"
    runner.tp_size = 2
    runner.tp_rank = 2
    runner.server_args = SimpleNamespace(tp_size=2)
    return runner


class TestAFDComponentActivateKV(CustomTestCase):
    def test_free_kv_pools_tolerates_missing_attributes(self):
        """A promoted standby rank has no KV attrs yet; teardown must not crash."""
        runner = _bare_runner()
        # Emulate the standby ModelRunner state: attributes were never created
        # by _init_pools(). getattr-safe teardown must run cleanly and seed None.
        self.assertFalse(hasattr(runner, "token_to_kv_pool"))
        runner._free_inplace_reshard_kv_pools()
        self.assertIsNone(runner.token_to_kv_pool)
        self.assertIsNone(runner.token_to_kv_pool_allocator)
        self.assertIsNone(runner.req_to_token_pool)
        self.assertEqual(runner.max_total_num_tokens, 0)

    def test_attn_refresh_runtime_builds_kv_via_commit(self):
        """ATTN joining rank activate must drop old KV then build a new one."""
        runner = _bare_runner()
        calls = []
        runner._free_inplace_reshard_kv_pools = lambda: calls.append("free")
        runner._update_model_tp_metadata = lambda tp: calls.append(("meta", tp))

        def _commit(tp, *, runtime_prepped):
            calls.append(("commit", tp, runtime_prepped))
            # The real commit path allocates the KV runtime.
            runner.token_to_kv_pool = SimpleNamespace(size=4096)
            runner.token_to_kv_pool_allocator = SimpleNamespace(size=4096)
            runner.req_to_token_pool = SimpleNamespace(size=64)

        runner._commit_inplace_reshard_runtime_post_xfer = _commit
        runner.rebuild_afd_ffn_memory_pool_after_reshard = lambda: calls.append(
            "ffn_rebuild"
        )
        runner._refresh_afd_component_data_plane = lambda: calls.append("data_plane")

        runner.afd_component_refresh_runtime(4, AFDPerspective.AFD_PERSPECTIVE_ATTN)

        self.assertEqual(calls[0], "free")
        self.assertEqual(calls[1], ("meta", 4))
        self.assertEqual(calls[2], ("commit", 4, False))
        self.assertEqual(calls[3], "data_plane")
        self.assertNotIn("ffn_rebuild", calls)
        # KV runtime attributes now exist and are populated (equivalent to a
        # normal attn ModelRunner).
        self.assertIsNotNone(runner.token_to_kv_pool)
        self.assertIsNotNone(runner.token_to_kv_pool_allocator)
        self.assertIsNotNone(runner.req_to_token_pool)

    def test_ffn_refresh_runtime_uses_no_kv_path(self):
        """FFN joining rank activate must not build an attention KV pool."""
        runner = _bare_runner()
        calls = []
        runner._free_inplace_reshard_kv_pools = lambda: calls.append("free")
        runner._commit_inplace_reshard_runtime_post_xfer = (
            lambda tp, *, runtime_prepped: calls.append("commit")
        )

        def _ffn_rebuild():
            calls.append("ffn_rebuild")
            # Mirrors rebuild_afd_ffn_memory_pool_after_reshard: never a KV pool.
            runner.token_to_kv_pool = None
            runner.token_to_kv_pool_allocator = SimpleNamespace(size=1)
            runner.req_to_token_pool = SimpleNamespace(size=1)

        runner.rebuild_afd_ffn_memory_pool_after_reshard = _ffn_rebuild

        def init_ngram_embedding():
            calls.append("ngram")
            runner.use_ngram_embedding = True

        def init_attention_backend():
            calls.append("attn_backend")
            runner.attn_backend = object()

        runner.maybe_init_ngram_embedding = init_ngram_embedding
        runner.init_attention_backend = init_attention_backend
        runner._refresh_afd_component_data_plane = lambda: calls.append("data_plane")

        runner.afd_component_refresh_runtime(4, AFDPerspective.AFD_PERSPECTIVE_FFN)

        self.assertEqual(
            calls, ["ffn_rebuild", "ngram", "attn_backend", "data_plane"]
        )
        self.assertNotIn("free", calls)
        self.assertNotIn("commit", calls)
        self.assertIsNone(runner.token_to_kv_pool)
        self.assertEqual(runner.tp_size, 4)
        self.assertIsNone(runner.graph_runner)
        self.assertIsNone(runner.piecewise_cuda_graph_runner)
        self.assertEqual(runner.graph_mem_usage, 0)

    def test_ffn_refresh_keeps_existing_attention_backend(self):
        runner = _bare_runner()
        sentinel = object()
        runner.attn_backend = sentinel
        runner.use_ngram_embedding = True
        runner.token_table = object()
        token_table = runner.token_table
        runner.graph_runner = object()
        runner.piecewise_cuda_graph_runner = object()
        runner.graph_mem_usage = 17
        graph_runner = runner.graph_runner
        piecewise_runner = runner.piecewise_cuda_graph_runner
        calls = []
        runner.rebuild_afd_ffn_memory_pool_after_reshard = lambda: calls.append(
            "ffn_rebuild"
        )
        runner.maybe_init_ngram_embedding = lambda: calls.append("ngram")
        runner.init_attention_backend = lambda: calls.append("attn_backend")
        runner._refresh_afd_component_data_plane = lambda: calls.append("data_plane")

        runner.afd_component_refresh_runtime(4, AFDPerspective.AFD_PERSPECTIVE_FFN)

        self.assertEqual(calls, ["ffn_rebuild", "data_plane"])
        self.assertIs(runner.attn_backend, sentinel)
        self.assertIs(runner.token_table, token_table)
        self.assertIs(runner.graph_runner, graph_runner)
        self.assertIs(runner.piecewise_cuda_graph_runner, piecewise_runner)
        self.assertEqual(runner.graph_mem_usage, 17)

    def test_broadcast_communicator_captures_tp_size_per_instance(self):
        from sglang.srt.layers.afd import BroadcastTensorCommunicator

        inner = SimpleNamespace()
        old = BroadcastTensorCommunicator(inner, local_tp_size=2, local_tp_rank=0)
        new = BroadcastTensorCommunicator(inner, local_tp_size=4, local_tp_rank=0)
        self.assertEqual(old.local_tp_size, 2)
        self.assertEqual(new.local_tp_size, 4)
        self.assertIsNone(old._tp_group)
        self.assertIsNone(new._tp_group)

    def test_data_plane_reset_clears_raw_and_async_once(self):
        import sglang.srt.layers.afd as afd

        perspective = AFDPerspective.AFD_PERSPECTIVE_FFN
        closed = []

        class FakeCommunicator:
            def close(self):
                closed.append(self)

        raw = FakeCommunicator()
        async_comm = SimpleNamespace(inner=raw)
        afd._experimental_tensor_communicators[perspective] = raw
        afd._experimental_async_communicators[perspective] = async_comm
        try:
            ModelRunner._refresh_afd_component_data_plane()
            self.assertEqual(afd._experimental_tensor_communicators, {})
            self.assertEqual(afd._experimental_async_communicators, {})
            self.assertEqual(closed, [raw])
            ModelRunner._refresh_afd_component_data_plane()
            self.assertEqual(closed, [raw])
        finally:
            afd.reset_afd_communicators()

    def test_ffn_noop_attention_backend_is_cpu_safe(self):
        from sglang.srt.layers.attention.tbo_backend import AFDFFNNoOpAttnBackend

        backend = AFDFFNNoOpAttnBackend()
        self.assertIsInstance(backend, AFDFFNNoOpAttnBackend)


class TestAFDComponentRefreshPDTopology(CustomTestCase):
    @staticmethod
    def _scheduler(mode):
        scheduler = object.__new__(_MixinHost)
        scheduler.server_args = SimpleNamespace(disaggregation_mode=mode)
        scheduler.is_afd_component_standby_rank = True
        calls = []
        scheduler.tp_worker = SimpleNamespace(
            finalize_inplace_reshard_activation=lambda: calls.append("finalize"),
            get_worker_info=lambda: tuple(range(12)),
        )
        scheduler.init_cache_with_memory_pool = lambda: calls.append("cache")
        scheduler.init_running_status = lambda: calls.append("status")

        def init_disaggregation():
            from sglang.srt.disaggregation.utils import DisaggregationMode

            calls.append("disaggregation")
            scheduler.disaggregation_mode = DisaggregationMode(mode)

        scheduler.init_disaggregation = init_disaggregation
        return scheduler, calls

    def test_joining_prefill_uses_server_arg_before_scheduler_init(self):
        scheduler, calls = self._scheduler("prefill")
        self.assertFalse(hasattr(scheduler, "disaggregation_mode"))

        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            scheduler.afd_component_refresh_pd_topology(4, 1)

        self.assertEqual(calls, ["finalize", "cache", "status", "disaggregation"])
        self.assertEqual(scheduler.server_args.afd_component_bootstrap_generation, 1)
        self.assertTrue(scheduler._afd_component_pd_runtime_initialized)

    def test_unrelated_mode_returns_without_joining_init(self):
        scheduler, calls = self._scheduler("null")

        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=True):
            scheduler.afd_component_refresh_pd_topology(4, 1)

        self.assertEqual(calls, [])
        self.assertFalse(
            hasattr(scheduler.server_args, "afd_component_bootstrap_generation")
        )


class TestAFDComponentFakeActivateStandby(CustomTestCase):
    def _fake_scheduler(self, rank, active_tp=2, max_tp=4):
        sched = object.__new__(_MixinHost)
        sched._afd_component_lifecycle = AFDComponentRankLifecycle(
            rank, active_tp, max_tp, "attn"
        )
        runner = SimpleNamespace(is_afd_component_standby_rank=None)
        sched.tp_worker = SimpleNamespace(
            model_runner=runner, is_afd_component_standby_rank=None
        )
        return sched

    def test_fake_activate_reset_keeps_joining_rank_standby(self):
        """Fake-mode activate advances epoch but must not promote a joining rank."""
        sched = self._fake_scheduler(rank=3, active_tp=2, max_tp=4)
        # Simulate the ACTIVATE safe point having promoted the rank to active_tp=4.
        sched._afd_component_lifecycle.active_tp = 4
        sched._afd_component_lifecycle.state = AFDComponentRankState.ACTIVE

        state = sched._afd_component_reset_lifecycle_active_tp(2)

        self.assertEqual(state, AFDComponentRankState.STANDBY)
        self.assertEqual(sched._afd_component_lifecycle.active_tp, 2)
        self.assertTrue(sched.is_afd_component_standby_rank)
        self.assertTrue(sched.tp_worker.model_runner.is_afd_component_standby_rank)

    def test_fake_activate_reset_keeps_initial_active_rank_active(self):
        """Initial active ranks (TP0/TP1) stay active and keep serving."""
        sched = self._fake_scheduler(rank=1, active_tp=2, max_tp=4)
        sched._afd_component_lifecycle.active_tp = 4
        sched._afd_component_lifecycle.state = AFDComponentRankState.ACTIVE

        state = sched._afd_component_reset_lifecycle_active_tp(2)

        self.assertEqual(state, AFDComponentRankState.ACTIVE)
        self.assertFalse(sched.is_afd_component_standby_rank)


class _MixinHost(SchedulerAFDMixin):
    pass


if __name__ == "__main__":
    unittest.main(verbosity=3)
