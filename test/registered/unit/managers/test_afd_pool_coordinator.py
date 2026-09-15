import pickle
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

try:
    import zmq
except ImportError:
    zmq = None

from sglang.srt.managers.afd_pool_coordinator import (
    AFDCoordinatorRemoteError,
    AFDPoolCoordinator,
    AFDPoolCoordinatorClient,
    AFDPoolCoordinatorRPCServer,
    PFCapacityWaitTimeout,
    make_dispatch_identity,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="stage-a-cpu-only")


class TestAFDPoolCoordinator(CustomTestCase):
    def setUp(self):
        self.coordinator = AFDPoolCoordinator()
        self.coordinator.register("pf-b")
        self.coordinator.register("pf-a")

    def test_deterministic_tie_break_and_rid_affinity(self):
        first = self.coordinator.acquire("rid-1", "pa-1")
        self.assertEqual(first.pf_instance_id, "pf-a")
        self.coordinator.begin_dispatch("pa-1:0:1", first.lease_id)
        second = self.coordinator.acquire("rid-2", "pa-2")
        self.assertEqual(second.pf_instance_id, "pf-b")
        self.assertEqual(self.coordinator.acquire("rid-1", "pa-1"), first)

    def test_idempotent_begin_continuation_completion_and_release(self):
        lease = self.coordinator.acquire("rid-1", "pa-1")
        first = self.coordinator.begin_dispatch("pa-1:0:1", lease.lease_id)
        self.assertEqual(
            self.coordinator.begin_dispatch("pa-1:0:1", lease.lease_id), first
        )
        # A new dispatch_id on the admitted lease is a continuation batch and
        # must not consume a second PF capacity slot.
        self.assertTrue(self.coordinator.complete_dispatch(first.dispatch_id))
        second = self.coordinator.begin_dispatch("pa-1:0:2", lease.lease_id)
        self.assertEqual(self.coordinator.snapshot()["loads"][0]["inflight"], 1)
        self.assertTrue(
            self.coordinator.complete_dispatch(second.dispatch_id, lease.lease_id)
        )
        # COMPLETE releases dispatch state, but the lease retains capacity until
        # request-terminal RELEASE.
        self.assertEqual(self.coordinator.snapshot()["loads"][0]["inflight"], 1)
        self.assertTrue(self.coordinator.release("rid-1", "pa-1"))
        self.assertEqual(self.coordinator.snapshot()["loads"][0]["inflight"], 0)

    def test_wrong_lease_is_rejected_for_active_and_completed_dispatch(self):
        first = self.coordinator.acquire("rid-1", "pa-1")
        second = self.coordinator.acquire("rid-2", "pa-2")
        dispatch_id = "pa-1:0:1"
        self.coordinator.begin_dispatch(dispatch_id, first.lease_id)
        with self.assertRaisesRegex(ValueError, "another lease"):
            self.coordinator.begin_dispatch(dispatch_id, second.lease_id)
        self.assertFalse(
            self.coordinator.complete_dispatch(dispatch_id, second.lease_id)
        )
        self.assertTrue(self.coordinator.complete_dispatch(dispatch_id, first.lease_id))
        with self.assertRaisesRegex(ValueError, "another lease"):
            self.coordinator.begin_dispatch(dispatch_id, second.lease_id)
        replay = self.coordinator.begin_dispatch(dispatch_id, first.lease_id)
        self.assertEqual(replay.lease_id, first.lease_id)
        self.assertNotIn(
            dispatch_id,
            {edge["dispatch_id"] for edge in self.coordinator.snapshot()["dispatches"]},
        )

    def test_failure_epoch_releases_affinity_and_fences_stale_completion(self):
        lease = self.coordinator.acquire("rid-1", "pa-1")
        dispatch_id = make_dispatch_identity("pa-1", lease.pair_epoch, 1)
        self.coordinator.begin_dispatch(dispatch_id, lease.lease_id)
        self.assertEqual(self.coordinator.mark_failed("pf-a"), 1)
        self.assertFalse(
            self.coordinator.complete_dispatch(dispatch_id, lease.lease_id)
        )
        replacement = self.coordinator.acquire("rid-1", "pa-1")
        self.assertEqual(replacement.pf_instance_id, "pf-b")
        self.coordinator.register("pf-a")
        self.assertEqual(self.coordinator.snapshot()["epochs"]["pf-a"], 1)
        with self.assertRaisesRegex(ValueError, "stale lease"):
            self.coordinator.begin_dispatch("pa-1:0:stale", lease.lease_id)

    def test_two_pas_cannot_use_same_rid_lease(self):
        lease = self.coordinator.acquire("shared-rid", "pa-1")
        with self.assertRaisesRegex(ValueError, "another PA"):
            self.coordinator.acquire("shared-rid", "pa-2")
        with self.assertRaisesRegex(ValueError, "another PA"):
            self.coordinator.begin_dispatch("pa-2:0:1", lease.lease_id, pa_id="pa-2")

    def test_concurrent_single_lane_admission_is_atomic(self):
        self.coordinator.mark_failed("pf-b")
        leases = [
            self.coordinator.acquire(f"rid-{index}", f"pa-{index}")
            for index in range(16)
        ]
        barrier = threading.Barrier(len(leases))

        def begin(index):
            barrier.wait()
            try:
                self.coordinator.begin_dispatch(
                    f"pa-{index}:0:1", leases[index].lease_id
                )
                return True
            except RuntimeError:
                return False

        with ThreadPoolExecutor(max_workers=len(leases)) as pool:
            admitted = list(pool.map(begin, range(len(leases))))
        self.assertEqual(sum(admitted), 1)
        self.assertEqual(self.coordinator.snapshot()["loads"][0]["inflight"], 1)

    def test_sequential_continuations_reuse_one_admitted_lease(self):
        self.coordinator.mark_failed("pf-b")
        lease = self.coordinator.acquire("rid", "pa")
        for index in range(8):
            edge = self.coordinator.begin_dispatch(
                f"pa:0:{index}", lease.lease_id, pa_id="pa"
            )
            self.assertEqual(self.coordinator.snapshot()["loads"][0]["inflight"], 1)
            self.assertTrue(
                self.coordinator.complete_dispatch(edge.dispatch_id, lease.lease_id)
            )


    def test_release_frees_capacity_for_another_lease(self):
        self.coordinator.mark_failed("pf-b")
        first = self.coordinator.acquire("rid-1", "pa-1")
        first_dispatch = self.coordinator.begin_dispatch("pa-1:0:1", first.lease_id)
        second = self.coordinator.acquire("rid-2", "pa-2")
        with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
            self.coordinator.begin_dispatch("pa-2:0:1", second.lease_id)
        self.coordinator.complete_dispatch(first_dispatch.dispatch_id, first.lease_id)
        self.coordinator.release("rid-1", "pa-1")
        admitted = self.coordinator.begin_dispatch("pa-2:0:1", second.lease_id)
        self.assertEqual(admitted.lease_id, second.lease_id)

    def test_capacity_allows_configured_number(self):
        coordinator = AFDPoolCoordinator()
        coordinator.register("pf", capacity=2)
        leases = [coordinator.acquire(f"r{i}", f"p{i}") for i in range(3)]
        first = coordinator.begin_dispatch(
            "p0:0:1",
            leases[0].lease_id,
            lease_ids=[leases[0].lease_id, leases[1].lease_id],
        )
        coordinator.complete_dispatch(first.dispatch_id, first.lease_id)
        with self.assertRaisesRegex(RuntimeError, "capacity exhausted"):
            coordinator.begin_dispatch("p2:0:1", leases[2].lease_id)


    def test_multi_lease_batch_admits_same_pf_and_rejects_mixed_pf(self):
        coordinator = AFDPoolCoordinator()
        coordinator.register("pf-0", capacity=4)
        coordinator.register("pf-1", capacity=4)
        leases = [
            coordinator.acquire(
                f"rid-{index}",
                "pa-0",
                preferred_pf_instance_id="pf-0",
            )
            for index in range(3)
        ]
        edge = coordinator.begin_dispatch(
            "pa-0:0:1",
            leases[0].lease_id,
            pa_id="pa-0",
            lease_ids=[lease.lease_id for lease in leases],
        )
        self.assertEqual(edge.lease_ids, tuple(lease.lease_id for lease in leases))
        loads = {item["pf_instance_id"]: item for item in coordinator.snapshot()["loads"]}
        self.assertEqual(loads["pf-0"]["inflight"], 3)

        other = coordinator.acquire(
            "rid-other", "pa-0", preferred_pf_instance_id="pf-1"
        )
        coordinator.complete_dispatch(edge.dispatch_id, edge.lease_id)
        with self.assertRaisesRegex(ValueError, "share one PA/PF"):
            coordinator.begin_dispatch(
                "pa-0:0:2",
                leases[0].lease_id,
                lease_ids=[leases[0].lease_id, other.lease_id],
            )

    def test_pf_dispatch_lane_serializes_two_pas_independent_of_capacity(self):
        coordinator = AFDPoolCoordinator()
        coordinator.register("pf", capacity=64)
        first = coordinator.acquire("rid-a", "pa-a")
        second = coordinator.acquire("rid-b", "pa-b")
        coordinator.begin_dispatch("pa-a:0:1", first.lease_id, pa_id="pa-a")
        with self.assertRaisesRegex(RuntimeError, "dispatch lane busy"):
            coordinator.begin_dispatch("pa-b:0:1", second.lease_id, pa_id="pa-b")
        coordinator.complete_dispatch("pa-a:0:1", first.lease_id)
        edge = coordinator.begin_dispatch(
            "pa-b:0:1", second.lease_id, pa_id="pa-b"
        )
        self.assertEqual(edge.pa_instance_id, "pa-b")

    def test_request_fields_are_backward_compatible_with_pickle(self):
        try:
            from sglang.srt.managers.io_struct import AFDReqInput
        except ImportError as exc:
            self.skipTest(f"io_struct runtime dependencies unavailable: {exc}")
        request = AFDReqInput(dispatch_id=7, req_ids=["rid"])
        restored = pickle.loads(pickle.dumps(request))
        self.assertIsNone(restored.pa_instance_id)
        self.assertIsNone(restored.pf_instance_id)
        self.assertIsNone(restored.lease_id)
        self.assertEqual(restored.pair_epoch, 0)


@unittest.skipIf(zmq is None, "pyzmq is not installed")
class TestAFDPoolCoordinatorRPC(CustomTestCase):
    def setUp(self):
        self.server = AFDPoolCoordinatorRPCServer("tcp://127.0.0.1:0")
        self.endpoint = self.server._socket.getsockopt_string(zmq.LAST_ENDPOINT)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server._stop.set()
        self.thread.join(timeout=2)
        self.server.close()

    def test_rpc_roundtrip(self):
        client = AFDPoolCoordinatorClient(self.endpoint, timeout_ms=2000)
        client.register(pf_instance_id="pf-1", capacity=1)
        lease = client.acquire("rid", "pa")
        dispatch_id = make_dispatch_identity("pa", lease["pair_epoch"], 1)
        edge = client.begin_dispatch(dispatch_id, lease["lease_id"], pa_id="pa")
        self.assertEqual(edge["pf_instance_id"], "pf-1")
        self.assertTrue(client.complete_dispatch(dispatch_id, lease["lease_id"]))
        self.assertTrue(client.release("rid", "pa"))
        self.assertEqual(client.snapshot()["loads"][0]["inflight"], 0)

    def test_capacity_wait_succeeds_after_release_from_another_thread(self):
        client = AFDPoolCoordinatorClient(self.endpoint, timeout_ms=2000)
        client.register(pf_instance_id="pf-1", capacity=1)
        first = client.acquire("rid-1", "pa-1")
        first_dispatch = make_dispatch_identity("pa-1", first["pair_epoch"], 1)
        client.begin_dispatch(first_dispatch, first["lease_id"], pa_id="pa-1")
        second = client.acquire("rid-2", "pa-2")
        second_dispatch = make_dispatch_identity("pa-2", second["pair_epoch"], 1)

        def release_first():
            time.sleep(0.05)
            client.complete_dispatch(first_dispatch, first["lease_id"])
            client.release("rid-1", "pa-1")

        releaser = threading.Thread(target=release_first)
        releaser.start()
        edge = client.begin_dispatch(
            second_dispatch,
            second["lease_id"],
            pa_id="pa-2",
            wait_timeout_s=1.0,
            retry_backoff_s=0.01,
        )
        releaser.join(timeout=1)
        self.assertEqual(edge["lease_id"], second["lease_id"])

    def test_capacity_wait_timeout_is_explicit(self):
        client = AFDPoolCoordinatorClient(self.endpoint, timeout_ms=2000)
        client.register(pf_instance_id="pf-1", capacity=1)
        first = client.acquire("rid-1", "pa-1")
        client.begin_dispatch("pa-1:0:1", first["lease_id"], pa_id="pa-1")
        second = client.acquire("rid-2", "pa-2")

        with self.assertRaisesRegex(PFCapacityWaitTimeout, "Timed out waiting"):
            client.begin_dispatch(
                "pa-2:0:1",
                second["lease_id"],
                pa_id="pa-2",
                wait_timeout_s=0.03,
                retry_backoff_s=0.01,
            )

    def test_non_capacity_error_is_not_retried(self):
        client = AFDPoolCoordinatorClient(self.endpoint, timeout_ms=2000)
        client.register(pf_instance_id="pf-1", capacity=1)
        lease = client.acquire("rid", "pa")
        original_begin = self.server.coordinator.begin_dispatch
        self.server.coordinator.begin_dispatch = Mock(wraps=original_begin)
        with self.assertRaisesRegex(AFDCoordinatorRemoteError, "another PA"):
            client.begin_dispatch(
                "other:0:1",
                lease["lease_id"],
                pa_id="other",
                wait_timeout_s=0.5,
                retry_backoff_s=0.01,
            )
        self.server.coordinator.begin_dispatch.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=3)
