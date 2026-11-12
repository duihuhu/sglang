import importlib.util
import sys
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[4]
    / "python/sglang/srt/reshard/inplace_reshard_background.py"
)
_SPEC = importlib.util.spec_from_file_location("inplace_reshard_background", _MODULE_PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
InplaceReshardTimingContext = _MODULE.InplaceReshardTimingContext
ensure_timing_context = _MODULE.ensure_timing_context
status_operation_transition = _MODULE.status_operation_transition
should_defer_async_kv_grow = _MODULE.should_defer_async_kv_grow
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class TestInplaceReshardTiming(CustomTestCase):
    def test_deferred_context_aggregation(self):
        ctx = InplaceReshardTimingContext(
            operation_id="op", old_tp=1, new_tp=2, accepted_at_s=9.5,
            scheduler_pickup_at_s=10.0, admission_block_at_s=10.1,
            drained_at_s=10.6, prep_start_at_s=9.8, prep_done_at_s=10.4,
            execute_queued_at_s=10.6, execute_start_at_s=10.8, done_at_s=11.8,
            scheduler_ms={"legacy_pause_call_ms": 3.0},
        )
        got = ctx.snapshot({"weight_transfer_total_ms": 50.0})
        self.assertAlmostEqual(got["wait_for_drain_ms"], 500.0)
        self.assertAlmostEqual(got["background_prep_ms"], 600.0)
        self.assertAlmostEqual(got["background_prep_critical_wait_ms"], 300.0)
        self.assertAlmostEqual(got["execute_queue_wait_ms"], 200.0)
        self.assertAlmostEqual(got["request_to_done_ms"], 2300.0)
        self.assertEqual(got["model_runner"]["weight_transfer_total_ms"], 50.0)

    def test_already_drained_is_zero(self):
        ctx = InplaceReshardTimingContext("op", 1, 2, 1.0, 1.1, 1.2)
        ctx.mark_drained(1.2)
        self.assertEqual(ctx.snapshot()["wait_for_drain_ms"], 0.0)

    def test_deferred_context_identity_is_not_replaced(self):
        first = ensure_timing_context(
            None, operation_id="http-op", accepted_at_s=100.0,
            old_tp=1, new_tp=2, now_s=101.0,
        )
        first.prep_start_at_s = 101.5
        pending = ensure_timing_context(
            first, operation_id="http-op", accepted_at_s=100.0,
            old_tp=1, new_tp=2, now_s=105.0,
        )
        executing = ensure_timing_context(
            pending, operation_id="http-op", accepted_at_s=100.0,
            old_tp=1, new_tp=2, now_s=110.0,
        )
        self.assertIs(first, pending)
        self.assertIs(first, executing)
        self.assertEqual(executing.operation_id, "http-op")
        self.assertEqual(executing.accepted_at_s, 100.0)
        self.assertEqual(executing.prep_start_at_s, 101.5)

    def test_generation_increments_once_across_deferred_phases(self):
        prev = {"phase": "done", "generation": 7, "operation_id": "old", "started_at": 1.0}
        generation, started, changed = status_operation_transition(
            prev, "preparing", "http-op", 10.0
        )
        self.assertEqual(generation, 8)
        self.assertEqual(started, 10.0)
        self.assertTrue(changed)

        prev = {
            "phase": "preparing", "generation": generation,
            "operation_id": "http-op", "started_at": started,
        }
        for phase in ("prepared", "draining", "executing", "done"):
            generation, next_started, changed = status_operation_transition(
                prev, phase, "http-op", 20.0
            )
            self.assertEqual(generation, 8)
            self.assertEqual(next_started, 10.0)
            self.assertFalse(changed)
            prev = {
                "phase": phase, "generation": generation,
                "operation_id": "http-op", "started_at": next_started,
            }


class TestAsyncKVGrowPriority(CustomTestCase):
    def test_control_file_defers_without_consuming_pending(self):
        self.assertTrue(
            should_defer_async_kv_grow(
                control_file_pending=True, pending_reshard=False,
                execute_pending=False, prep_pending=False,
            )
        )

    def test_any_reshard_state_has_priority(self):
        for key in ("pending_reshard", "execute_pending", "prep_pending"):
            args = dict(
                control_file_pending=False, pending_reshard=False,
                execute_pending=False, prep_pending=False,
            )
            args[key] = True
            self.assertTrue(should_defer_async_kv_grow(**args), key)

    def test_no_reshard_work_allows_grow(self):
        self.assertFalse(
            should_defer_async_kv_grow(
                control_file_pending=False, pending_reshard=False,
                execute_pending=False, prep_pending=False,
            )
        )
