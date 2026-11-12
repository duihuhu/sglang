import multiprocessing as mp
import os
import tempfile
import time
import unittest

import torch
import torch.distributed as dist

from sglang.srt.layers.afd_reshard_comm import (
    AFDEpochChannelIdentity,
    AFDEpochCommunicatorRegistry,
    AFDPairDrainProtocol,
    AFDReshardCommunicatorCache,
    EpochChannelState,
    TorchDistributedEpochControl,
    TorchDistributedP2PAdapter,
    refresh_afd_reshard_communicator,
)
from sglang.srt.managers.io_struct import AFDComponentReshardReqInput
from sglang.srt.reshard.afd_component_reshard import AFDComponentReshardCoordinator
from sglang.srt.reshard.afd_component_runtime import (
    AFDComponent,
    AFDComponentRuntime,
    AdapterReceipt,
    RankLifecycle,
    register_afd_colocated_runtime,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="stage-a-cpu-only")


def _gloo_worker(rank, init_file, result_queue):
    dist.init_process_group("gloo", init_method=f"file://{init_file}", rank=rank, world_size=2)
    try:
        identity = AFDEpochChannelIdentity("pair", "prefill", 3, "weights")
        TorchDistributedEpochControl().consensus(identity, marker=1)
        adapter = TorchDistributedP2PAdapter()
        if rank == 0:
            adapter.send(torch.tensor([3, 1, 4], dtype=torch.int64), dst=1, tag=17)
        else:
            output = torch.empty(3, dtype=torch.int64)
            adapter.recv(output, src=0, tag=17)
            result_queue.put(output.tolist())
    finally:
        dist.destroy_process_group()


def make_coordinator():
    return AFDComponentReshardCoordinator(
        enabled=True, max_tp=4, stage_id="prefill", pair_id="p0",
        attn_tp=2, ffn_tp=2, channel_base=1700, control_base=1800,
    )


def make_request(operation_id):
    return AFDComponentReshardReqInput(
        stage="prefill", expected_attn_tp=2, expected_ffn_tp=2,
        target_attn_tp=4, target_ffn_tp=1, expected_epoch=0,
        operation_id=operation_id, dry_run=False,
    )


def wait_terminal(coordinator, operation_id):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        status = coordinator.get_status(operation_id)
        if status.completed_at is not None:
            return status
        time.sleep(0.005)
    raise AssertionError("operation did not terminate")


class FakeAdapter:
    def __init__(self):
        self.calls = []

    def _done(self, name):
        self.calls.append(name)
        return AdapterReceipt(True, name)

    def prepare(self, request, transitions):
        return self._done("prepare")

    def drain(self, request, cut_watermark):
        return self._done("drain")

    def activate(self, request, transitions):
        return self._done("activate")

    def retire(self, request, transitions):
        return self._done("retire")


class TestAFDColocatedRuntime(CustomTestCase):
    def test_lifecycle_scale_up_down_and_atomic_swap(self):
        runtime = AFDComponentRuntime(
            pair_id="pair", stage="prefill", max_world_size=4,
            attn_tp=2, ffn_tp=4, rank_devices=(0, 1, 2, 3),
        )
        original_attn = runtime.topology(AFDComponent.ATTN)
        prepared = runtime.prepare(target_attn_tp=4, target_ffn_tp=1)
        self.assertEqual(
            prepared[AFDComponent.ATTN].lifecycles,
            (RankLifecycle.ACTIVE, RankLifecycle.ACTIVE, RankLifecycle.PREPARED, RankLifecycle.PREPARED),
        )
        self.assertEqual(
            prepared[AFDComponent.FFN].lifecycles,
            (RankLifecycle.ACTIVE, RankLifecycle.DEMOTING, RankLifecycle.DEMOTING, RankLifecycle.DEMOTING),
        )
        self.assertIs(runtime.topology(AFDComponent.ATTN), original_attn)
        swapped = runtime.activate()
        self.assertEqual(swapped[AFDComponent.ATTN].active_tp, 4)
        self.assertEqual(swapped[AFDComponent.FFN].active_tp, 1)
        self.assertEqual(swapped[AFDComponent.ATTN].epoch, 1)

    def test_prefill_decode_identity_isolated(self):
        prefill = AFDEpochChannelIdentity("pair", "prefill", 1, "hidden")
        decode = AFDEpochChannelIdentity("pair", "decode", 1, "hidden")
        self.assertNotEqual(prefill, decode)
        cache = AFDReshardCommunicatorCache()
        self.assertIsNot(cache.refresh(prefill, object), cache.refresh(decode, object))
        values = []
        value = refresh_afd_reshard_communicator(
            cache, AFDEpochChannelIdentity("pair", "prefill", 2, "broadcast"),
            active_tp=4, factory=lambda tp: values.append(tp) or object(),
        )
        self.assertIsNotNone(value)
        self.assertEqual(values, [4])

    def test_epoch_registry_atomic_activate_fence_retire(self):
        registry = AFDEpochCommunicatorRegistry()
        identities = tuple(AFDEpochChannelIdentity("pair", "decode", 2, channel) for channel in ("a2f", "f2a"))
        for identity in identities:
            registry.prepare(identity)
            registry.ready(identity, "attn")
            registry.ready(identity, "ffn")
        registry.activate(identities)
        for identity in identities:
            self.assertEqual(registry.snapshot(identity).state, EpochChannelState.ACTIVE)
            registry.fence(identity)
            registry.retire(identity)
            self.assertEqual(registry.snapshot(identity).state, EpochChannelState.RETIRED)

    def test_strict_drain_watermark(self):
        drain = AFDPairDrainProtocol()
        self.assertEqual((drain.next_dispatch(), drain.next_dispatch()), (1, 2))
        cut = drain.cut()
        with self.assertRaises(RuntimeError):
            drain.next_dispatch()
        drain.set_pending(sends=1, recvs=0, fences=0)
        drain.ack("attn", cut)
        drain.ack("ffn", cut)
        self.assertFalse(drain.drained)
        drain.set_pending(sends=0, recvs=0, fences=0)
        self.assertTrue(drain.drained)
        drain.reopen_after_transition()
        self.assertIsNone(drain.cut_watermark)
        self.assertEqual(drain.next_dispatch(), 3)

    def test_reopen_rejects_incomplete_drain(self):
        drain = AFDPairDrainProtocol()
        drain.next_dispatch()
        cut = drain.cut()
        drain.ack("attn", cut)
        with self.assertRaisesRegex(RuntimeError, "strict drain"):
            drain.reopen_after_transition()
        with self.assertRaisesRegex(RuntimeError, "fenced"):
            drain.next_dispatch()

    def test_executor_unsupported_without_adapter(self):
        coordinator = make_coordinator()
        runtime = AFDComponentRuntime(pair_id="p0", stage="prefill", max_world_size=4, attn_tp=2, ffn_tp=2)
        register_afd_colocated_runtime(coordinator, runtime, None)
        coordinator.submit(make_request("unsupported"))
        status = wait_terminal(coordinator, "unsupported")
        self.assertEqual(status.phase, "unsupported")
        self.assertEqual(coordinator.get_topology("prefill").epoch, 0)

    def test_executor_rejected_receipt_does_not_publish_epoch(self):
        class RejectingAdapter(FakeAdapter):
            def activate(self, request, transitions):
                return AdapterReceipt(False, "CUDA copy not complete")

        coordinator = make_coordinator()
        runtime = AFDComponentRuntime(
            pair_id="p0", stage="prefill", max_world_size=4, attn_tp=2, ffn_tp=2
        )
        register_afd_colocated_runtime(coordinator, runtime, RejectingAdapter())
        coordinator.submit(make_request("rejected"))
        status = wait_terminal(coordinator, "rejected")
        self.assertEqual(status.phase, "failed_post_commit")
        self.assertEqual(coordinator.get_topology("prefill").epoch, 0)
        self.assertEqual(runtime.topology(AFDComponent.ATTN).epoch, 0)

    def test_executor_success_with_receipted_adapter(self):
        coordinator = make_coordinator()
        runtime = AFDComponentRuntime(pair_id="p0", stage="prefill", max_world_size=4, attn_tp=2, ffn_tp=2)
        adapter = FakeAdapter()
        register_afd_colocated_runtime(coordinator, runtime, adapter)
        coordinator.submit(make_request("success"))
        status = wait_terminal(coordinator, "success")
        self.assertEqual(status.phase, "succeeded")
        self.assertEqual(status.epoch, 1)
        self.assertEqual(adapter.calls, ["prepare", "drain", "activate", "retire"])
        self.assertEqual(runtime.topology(AFDComponent.ATTN).active_tp, 4)
        self.assertEqual(runtime.topology(AFDComponent.FFN).active_tp, 1)

    def test_executor_merges_activate_breakdown_into_final_status(self):
        class BreakdownAdapter(FakeAdapter):
            def prepare(self, request, transitions):
                self.calls.append("prepare")
                return AdapterReceipt(True, "prepare", {"prepare_s": 0.2})
            def drain(self, request, cut_watermark):
                self.calls.append("drain")
                return AdapterReceipt(True, "drain", {"quiesce_detail": 0.01})
            def activate(self, request, transitions):
                self.calls.append("activate")
                return AdapterReceipt(True, "activate", {
                    "attn": {"critical": {"peer_transfer_bytes": 64}, "ranks": []}
                })

        coordinator = make_coordinator()
        runtime = AFDComponentRuntime(
            pair_id="p0", stage="prefill", max_world_size=4, attn_tp=2, ffn_tp=2
        )
        register_afd_colocated_runtime(coordinator, runtime, BreakdownAdapter())
        coordinator.submit(make_request("breakdown"))
        status = wait_terminal(coordinator, "breakdown")
        self.assertEqual(status.breakdown["attn"]["critical"]["peer_transfer_bytes"], 64)
        self.assertIn("quiesce_drain_s", status.breakdown)
        self.assertIn("redirect_readiness_s", status.breakdown)

    def test_drain_advances_prepared_to_draining_to_committing(self):
        # Regression for the silent stall at PREPARED: the executor must emit a
        # DRAINING phase before the (bounded) drain and only then COMMITTING, so
        # a slow/blocked drain is observable in status polls rather than
        # appearing frozen at PREPARED.
        phases = []

        class SlowDrainAdapter(FakeAdapter):
            def __init__(self, phase_sink):
                super().__init__()
                self._sink = phase_sink

            def drain(self, request, cut_watermark):
                # By the time drain runs, the phase must already be DRAINING.
                self._sink.append("drain-observed")
                return self._done("drain")

        coordinator = make_coordinator()
        runtime = AFDComponentRuntime(
            pair_id="p0", stage="prefill", max_world_size=4, attn_tp=2, ffn_tp=2
        )
        adapter = SlowDrainAdapter(phases)
        register_afd_colocated_runtime(coordinator, runtime, adapter)

        seen = []
        original = coordinator._transition

        def spy(operation_id, phase, message=None, breakdown=None):
            seen.append(phase)
            return original(operation_id, phase, message, breakdown)

        coordinator._transition = spy
        coordinator.submit(make_request("draining"))
        status = wait_terminal(coordinator, "draining")
        coordinator._transition = original

        self.assertEqual(status.phase, "succeeded")
        # PREPARED must be followed by DRAINING before COMMITTING.
        self.assertIn("prepared", seen)
        self.assertIn("draining", seen)
        self.assertIn("committing", seen)
        self.assertLess(seen.index("prepared"), seen.index("draining"))
        self.assertLess(seen.index("draining"), seen.index("committing"))
        self.assertEqual(adapter.calls, ["prepare", "drain", "activate", "retire"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "Gloo unavailable")
    def test_cross_process_gloo_tensor_p2p(self):
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "gloo-init")
            processes = [ctx.Process(target=_gloo_worker, args=(rank, init_file, result_queue)) for rank in range(2)]
            for process in processes:
                process.start()
            for process in processes:
                process.join(timeout=10)
                self.assertFalse(process.is_alive())
                self.assertEqual(process.exitcode, 0)
            self.assertEqual(result_queue.get(timeout=1), [3, 1, 4])


if __name__ == "__main__":
    unittest.main(verbosity=3)
