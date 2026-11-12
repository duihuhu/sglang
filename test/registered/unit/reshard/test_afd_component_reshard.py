import threading
import time
import unittest

from sglang.srt.managers.io_struct import (
    AFDComponentReshardCancelInput,
    AFDComponentReshardReqInput,
)
from sglang.srt.reshard.afd_component_reshard import (
    AFDComponentReshardConflict,
    AFDComponentReshardCoordinator,
    AFDComponentReshardDisabled,
    AFDComponentReshardPhase,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-cpu-only")


def make_coordinator(*, enabled=True, stage_id=None):
    return AFDComponentReshardCoordinator(
        enabled=enabled, max_tp=8, stage_id=stage_id, pair_id="pair-0",
        attn_tp=1, ffn_tp=2, channel_base=1700, control_base=1800,
    )


def make_request(operation_id, stage="prefill", **overrides):
    values = dict(
        stage=stage, expected_attn_tp=1, expected_ffn_tp=2,
        target_attn_tp=2, target_ffn_tp=4, expected_epoch=0,
        operation_id=operation_id, dry_run=False,
    )
    values.update(overrides)
    return AFDComponentReshardReqInput(**values)


def wait_terminal(coordinator, operation_id, timeout=2):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = coordinator.get_status(operation_id)
        if status.completed_at is not None:
            return status
        time.sleep(0.005)
    raise AssertionError(f"operation {operation_id} did not terminate")


class TestAFDComponentReshard(CustomTestCase):
    def test_dry_run_validation_and_topology_unchanged(self):
        coordinator = make_coordinator()
        status = coordinator.submit(make_request("dry", dry_run=True))
        self.assertEqual(status.phase, AFDComponentReshardPhase.SUCCEEDED.value)
        self.assertEqual(status.capability, "validation_only")
        self.assertFalse(status.runtime_supported)
        topology = coordinator.get_topology("prefill")
        self.assertEqual((topology.epoch, topology.attn_tp, topology.ffn_tp), (0, 1, 2))

    def test_unregistered_runtime_is_explicitly_unsupported(self):
        coordinator = make_coordinator()
        coordinator.submit(make_request("unsupported"))
        status = wait_terminal(coordinator, "unsupported")
        self.assertEqual(status.phase, AFDComponentReshardPhase.UNSUPPORTED.value)
        self.assertIn("not registered", status.message)

    def test_success_state_transition_and_epoch_commit(self):
        coordinator = make_coordinator()
        phases = []

        def executor(req, transition, is_cancelled):
            transition(AFDComponentReshardPhase.PREPARED.value, "prepared")
            phases.append(coordinator.get_status(req.operation_id).phase)
            transition(AFDComponentReshardPhase.COMMITTING.value, "committing")
            phases.append(coordinator.get_status(req.operation_id).phase)

        coordinator.register_executor(executor)
        coordinator.submit(make_request("success"))
        status = wait_terminal(coordinator, "success")
        self.assertEqual(phases, ["prepared", "committing"])
        self.assertEqual(status.phase, "succeeded")
        self.assertEqual(status.epoch, 1)
        topology = coordinator.get_topology("prefill")
        self.assertEqual((topology.epoch, topology.attn_tp, topology.ffn_tp), (1, 2, 4))

    def test_prefill_decode_can_run_in_parallel(self):
        coordinator = make_coordinator()
        barrier = threading.Barrier(2)

        def executor(req, transition, is_cancelled):
            barrier.wait(timeout=1)
            transition("committing")

        coordinator.register_executor(executor)
        coordinator.submit(make_request("p", stage="prefill"))
        coordinator.submit(make_request("d", stage="decode"))
        self.assertEqual(wait_terminal(coordinator, "p").phase, "succeeded")
        self.assertEqual(wait_terminal(coordinator, "d").phase, "succeeded")

    def test_same_stage_is_mutually_exclusive(self):
        coordinator = make_coordinator()
        first_entered = threading.Event()
        release_first = threading.Event()
        second_entered = threading.Event()

        def executor(req, transition, is_cancelled):
            if req.operation_id == "first":
                first_entered.set()
                release_first.wait(timeout=1)
            else:
                second_entered.set()
            transition("committing")

        coordinator.register_executor(executor)
        coordinator.submit(make_request("first"))
        self.assertTrue(first_entered.wait(timeout=1))
        coordinator.submit(make_request("second"))
        self.assertFalse(second_entered.wait(timeout=0.05))
        release_first.set()
        self.assertEqual(wait_terminal(coordinator, "first").phase, "succeeded")
        self.assertEqual(wait_terminal(coordinator, "second").phase, "failed_pre_commit")
        self.assertFalse(second_entered.is_set())

    def test_idempotency_and_conflicting_reuse(self):
        coordinator = make_coordinator()
        first = coordinator.submit(make_request("same", dry_run=True))
        second = coordinator.submit(make_request("same", dry_run=True))
        self.assertEqual(first.operation_id, second.operation_id)
        self.assertEqual(len(coordinator.get_status()), 1)
        with self.assertRaises(AFDComponentReshardConflict):
            coordinator.submit(make_request("same", dry_run=True, target_ffn_tp=8))

    def test_epoch_conflict(self):
        coordinator = make_coordinator()
        with self.assertRaises(AFDComponentReshardConflict):
            coordinator.submit(make_request("stale", expected_epoch=3))

    def test_cancel_before_commit(self):
        coordinator = make_coordinator()
        entered = threading.Event()

        def executor(req, transition, is_cancelled):
            entered.set()
            while not is_cancelled():
                time.sleep(0.005)

        coordinator.register_executor(executor)
        coordinator.submit(make_request("cancel"))
        self.assertTrue(entered.wait(timeout=1))
        coordinator.cancel(AFDComponentReshardCancelInput("cancel", expected_epoch=0))
        self.assertEqual(wait_terminal(coordinator, "cancel").phase, "cancelled")
        self.assertEqual(coordinator.get_topology("prefill").epoch, 0)

    def test_pre_and_post_commit_failures_are_distinct(self):
        coordinator = make_coordinator()

        def pre_failure(req, transition, is_cancelled):
            raise RuntimeError("prepare failed")

        coordinator.register_executor(pre_failure)
        coordinator.submit(make_request("pre"))
        self.assertEqual(wait_terminal(coordinator, "pre").phase, "failed_pre_commit")

        def post_failure(req, transition, is_cancelled):
            transition("committing")
            raise RuntimeError("commit uncertain")

        coordinator.register_executor(post_failure)
        coordinator.submit(make_request("post"))
        self.assertEqual(wait_terminal(coordinator, "post").phase, "failed_post_commit")

    def test_single_stage_topology_has_no_peer_stage(self):
        coordinator = make_coordinator(stage_id="decode")
        self.assertEqual([topology.stage for topology in coordinator.get_topology()], ["decode"])
        with self.assertRaises(ValueError):
            coordinator.get_topology("prefill")

    def test_submit_captures_executor(self):
        coordinator = make_coordinator()
        entered = threading.Event()
        release = threading.Event()
        calls = []

        def first(req, transition, is_cancelled):
            calls.append("first")
            entered.set()
            release.wait(timeout=1)
            transition("committing")

        def second(req, transition, is_cancelled):
            calls.append("second")
            transition("committing")

        coordinator.register_executor(first)
        coordinator.submit(make_request("captured"))
        self.assertTrue(entered.wait(timeout=1))
        coordinator.register_executor(second)
        release.set()
        self.assertEqual(wait_terminal(coordinator, "captured").phase, "succeeded")
        self.assertEqual(calls, ["first"])

    def test_cancel_worker_transition_race_is_idempotent(self):
        coordinator = make_coordinator()
        entered = threading.Event()
        release = threading.Event()

        def executor(req, transition, is_cancelled):
            entered.set()
            release.wait(timeout=1)
            transition("prepared")

        coordinator.register_executor(executor)
        coordinator.submit(make_request("race"))
        self.assertTrue(entered.wait(timeout=1))
        coordinator.cancel(AFDComponentReshardCancelInput("race"))
        release.set()
        self.assertEqual(wait_terminal(coordinator, "race").phase, "cancelled")

    def test_terminal_history_trim_skips_old_active(self):
        coordinator = AFDComponentReshardCoordinator(
            enabled=True, max_tp=8, stage_id=None, pair_id="pair-0",
            attn_tp=1, ffn_tp=2, channel_base=1700, control_base=1800,
            history_limit=2,
        )
        entered = threading.Event()
        release = threading.Event()

        def executor(req, transition, is_cancelled):
            if req.operation_id == "active":
                entered.set()
                release.wait(timeout=1)
            transition("committing")

        coordinator.register_executor(executor)
        coordinator.submit(make_request("active", stage="prefill"))
        self.assertTrue(entered.wait(timeout=1))
        coordinator.submit(make_request("done-1", stage="decode", dry_run=True))
        coordinator.submit(make_request("done-2", stage="decode", dry_run=True))
        operation_ids = {status.operation_id for status in coordinator.get_status()}
        self.assertEqual(operation_ids, {"active", "done-2"})
        release.set()
        wait_terminal(coordinator, "active")

    def test_feature_disabled(self):
        coordinator = make_coordinator(enabled=False)
        with self.assertRaises(AFDComponentReshardDisabled):
            coordinator.submit(make_request("disabled", dry_run=True))
        with self.assertRaises(AFDComponentReshardDisabled):
            coordinator.get_topology()

    def test_local_stage_restriction(self):
        coordinator = make_coordinator(stage_id="decode")
        with self.assertRaises(ValueError):
            coordinator.submit(make_request("wrong", stage="prefill", dry_run=True))


if __name__ == "__main__":
    unittest.main(verbosity=3)
