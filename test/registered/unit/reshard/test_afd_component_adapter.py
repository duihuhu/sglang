import threading
import time
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.layers.afd_reshard_comm import AFDPairDrainProtocol
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.managers.scheduler_afd_mixin import SchedulerAFDMixin
from sglang.srt.reshard.afd_component_adapter import (
    AFDComponentSchedulerProxyAdapter,
    AFDComponentSchedulerRuntime,
    ModelRunnerComponentAdapter,
    SchedulerCommandQueue,
    ZMQSchedulerControlClient,
    ZMQSchedulerControlServer,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-cpu-only")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        self.model.layers[0].self_attn = torch.nn.Linear(2, 2, bias=False)


class _Scheduler:
    def __init__(self, perspective):
        args = SimpleNamespace(
            afd_perspective=perspective, afd_component_max_tp=4,
            afd_micro_batch=1, disable_cuda_graph=True,
        )
        runner = SimpleNamespace(
            model=_Model(), model_config=SimpleNamespace(
                hf_config=SimpleNamespace(model_type="qwen3", num_experts=None, num_local_experts=None)
            ), server_args=args, pp_size=1, tp_size=2,
        )
        self.server_args = args
        self.tp_worker = SimpleNamespace(model_runner=runner)
        self._afd_component_command_queue = SchedulerCommandQueue()
        self._afd_reshard_drain = AFDPairDrainProtocol()
        self.tp_rank = 0
        self._afd_component_active_control_generation = 0
        # Number of is_fully_idle() polls that must occur before the scheduler
        # reports itself idle. Default 0 keeps historical always-idle behavior.
        self._idle_after_polls = 0
        self._idle_poll_count = 0
        self.fanout_calls = []
        self._afd_component_pending_fence_payload = None
        self._afd_component_fence_installed = False
        self._afd_component_fence_synced = False
        self._afd_component_quiesce_world_done = False
        self._afd_component_quiesce_world_operation = None
        self._afd_component_fence_payload = None
        # AFD-quiescent inputs consulted by AFDComponentSchedulerRuntime.
        self._afd_batchsize_attn = None
        self._afd_pending_batch_infos = deque()

    def is_fully_idle(self):
        self._idle_poll_count += 1
        return self._idle_poll_count > self._idle_after_polls

    def afd_reshard_drain_state(self): return self._afd_reshard_drain

    def afd_component_request_quiesce(self, payload):
        self._afd_component_pending_fence_payload = dict(payload)
        self._afd_component_fence_installed = False
        self._afd_component_fence_synced = False

    def afd_component_clear_quiesce(self):
        self._afd_component_pending_fence_payload = None
        self._afd_component_fence_installed = False
        self._afd_component_fence_synced = False
        self._afd_component_fence_payload = None
        self._afd_component_quiesce_world_done = False
        self._afd_component_quiesce_world_operation = None

    def sync_pending_quiesce(self):
        if self._afd_component_pending_fence_payload is None:
            return False
        self._afd_component_fence_payload = dict(
            self._afd_component_pending_fence_payload
        )
        self._afd_component_pending_fence_payload = None
        self._afd_component_fence_installed = True
        self._afd_component_fence_synced = True
        return True

    def mark_world_quiesced(self, operation_id):
        self.sync_pending_quiesce()
        self._afd_component_quiesce_world_done = True
        self._afd_component_quiesce_world_operation = operation_id
        self.fanout_calls.append({"action": "quiesce", "operation_id": operation_id})

    def afd_component_fanout_command(self, payload):
        self.fanout_calls.append(payload)
        return payload


class _QuiesceRuntime(AFDComponentSchedulerRuntime):
    """Runtime that skips CUDA communicator construction for CPU control tests."""

    def _strict_comm_drain(self):
        self.scheduler.afd_reshard_drain_state().set_pending(sends=0, recvs=0, fences=0)


class TestAFDComponentAdapter(CustomTestCase):
    def test_serving_readiness_bypasses_scheduler_command_queue(self):
        commands = SchedulerCommandQueue()
        provider_calls = []

        def readiness(payload):
            provider_calls.append(dict(payload))
            return {"ok": True, "operation_id": payload["operation_id"]}

        server = ZMQSchedulerControlServer(
            "tcp://127.0.0.1:38991", commands,
            default_timeout=1, readiness_provider=readiness,
        )
        try:
            client = ZMQSchedulerControlClient("tcp://127.0.0.1:38991")
            response = client.request(
                {
                    "action": "serving_readiness",
                    "operation_id": "ready-op",
                    "epoch": 1,
                    "_transport_grace": 0.0,
                },
                0.2,
            )
            self.assertTrue(response["ok"])
            self.assertEqual(len(provider_calls), 1)
            self.assertTrue(commands._commands.empty())
        finally:
            server.close()

    def test_transport_pending_preempts_post_receive_data_plane(self):
        """A real REP thread wakes the scheduler before data-plane work starts."""
        commands = SchedulerCommandQueue()
        external_pending = threading.Event()
        handled = []

        class LoopScheduler:
            tp_rank = 0
            is_afd_component_standby_rank = False
            _afd_component_control_group = object()
            _afd_component_active_control_group = object()
            _afd_component_active_control_generation = 0
            _afd_component_command_queue = commands
            _afd_component_external_control_pending = external_pending

            def __init__(self):
                self.runtime = SimpleNamespace(
                    poll=lambda: commands.poll(self.handle, limit=1)
                )
                self.safe_points = 0

            def handle(self, payload):
                handled.append(dict(payload))
                return {"ok": True}

            def afd_component_poll_runtime(self):
                return self.runtime.poll()

            def afd_component_poll_through_wakeup(self):
                return commands.poll_through_wakeup(self.handle)

            def afd_component_safe_point(self):
                self.safe_points += 1

        scheduler = LoopScheduler()
        endpoint = "tcp://127.0.0.1:38995"
        server = ZMQSchedulerControlServer(
            endpoint, commands, default_timeout=1,
            command_pending_callback=external_pending.set,
        )
        result = {}
        request_phase_entered = threading.Event()
        release_request_phase = threading.Event()
        retained_requests = [object()]

        def loop_once():
            # Model rank 0 after recv_requests()/TP request broadcast.  The
            # received request remains local while transport submits epoch 2.
            request_phase_entered.set()
            release_request_phase.wait(1)
            deadline = time.monotonic() + 1
            while not handled and time.monotonic() < deadline:
                with mock.patch(
                    "sglang.srt.utils.common.broadcast_pyobj",
                    side_effect=lambda data, *args, **kwargs: data,
                ):
                    SchedulerAFDMixin.afd_component_post_receive_control_checkpoint(
                        scheduler
                    )
                time.sleep(0.001)
            result["retained"] = retained_requests

        loop_thread = threading.Thread(target=loop_once)
        client_thread = threading.Thread(
            target=lambda: result.update(
                response=ZMQSchedulerControlClient(endpoint).request(
                    {
                        "action": "prepare", "operation_id": "epoch2",
                        "epoch": 1, "_transport_grace": 0.0,
                    },
                    1,
                )
            )
        )
        started = time.monotonic()
        try:
            loop_thread.start()
            self.assertTrue(request_phase_entered.wait(0.5))
            client_thread.start()
            self.assertTrue(external_pending.wait(0.5))
            release_request_phase.set()
            client_thread.join(1)
            loop_thread.join(1)
            self.assertFalse(client_thread.is_alive())
            self.assertFalse(loop_thread.is_alive())
            self.assertLess(time.monotonic() - started, 1)
            self.assertTrue(result["response"]["ok"])
            self.assertEqual([item["operation_id"] for item in handled], ["epoch2"])
            self.assertEqual(result["retained"], retained_requests)
            self.assertEqual(scheduler.safe_points, 1)
            self.assertFalse(commands.has_pending())
        finally:
            release_request_phase.set()
            server.close()
            client_thread.join(1)
            loop_thread.join(1)

    def test_transport_read_only_status_does_not_call_wakeup_callback(self):
        commands = SchedulerCommandQueue()
        callback = mock.Mock()
        endpoint = "tcp://127.0.0.1:38996"
        server = ZMQSchedulerControlServer(
            endpoint, commands, default_timeout=1,
            command_pending_callback=callback,
        )
        result = {}
        submitter = threading.Thread(
            target=lambda: result.update(
                response=ZMQSchedulerControlClient(endpoint).request(
                    {
                        "action": "prepare_status", "operation_id": "op",
                        "epoch": 0, "_transport_grace": 0.0,
                    },
                    1,
                )
            )
        )
        try:
            submitter.start()
            deadline = time.monotonic() + 0.5
            while not commands.has_pending() and time.monotonic() < deadline:
                time.sleep(0.001)
            self.assertTrue(commands.has_pending())
            self.assertFalse(commands.has_wakeup_pending())
            callback.assert_not_called()
            commands.poll(lambda payload: {"ok": True})
            submitter.join(1)
            self.assertTrue(result["response"]["ok"])
        finally:
            server.close()
            submitter.join(1)

    def test_read_only_status_does_not_publish_wakeup(self):
        commands = SchedulerCommandQueue()
        result = {}
        submitter = threading.Thread(
            target=lambda: result.update(
                commands.submit({"action": "prepare_status"}, 1)
            )
        )
        submitter.start()
        time.sleep(0.02)
        self.assertTrue(commands.has_pending())
        self.assertFalse(commands.has_wakeup_pending())
        commands.poll(lambda payload: {"ok": True})
        submitter.join(1)
        self.assertTrue(result["ok"])

    def test_poll_through_status_backlog_executes_one_wakeup_once(self):
        commands = SchedulerCommandQueue()
        results = [{}, {}, {}]
        payloads = [
            {"action": "prepare_status", "operation_id": "op", "epoch": 0},
            {"action": "prepare_status", "operation_id": "op", "epoch": 0},
            {
                "action": "activate", "operation_id": "op", "epoch": 1,
                "_wake_scheduler": True,
            },
        ]
        threads = [
            threading.Thread(
                target=lambda i=i: results[i].update(
                    commands.submit(payloads[i], 1)
                )
            )
            for i in range(3)
        ]
        # Preserve deterministic FIFO order while each submit blocks.
        for thread in threads:
            thread.start()
            time.sleep(0.01)
        self.assertTrue(commands.has_wakeup_pending())
        handled = []
        count, consumed = commands.poll_through_wakeup(
            lambda payload: handled.append(payload["action"]) or {"ok": True}
        )
        for thread in threads:
            thread.join(1)
        self.assertTrue(consumed)
        self.assertEqual(count, 3)
        self.assertEqual(
            handled, ["prepare_status", "prepare_status", "activate"]
        )
        self.assertFalse(commands.has_wakeup_pending())
        self.assertEqual([item["ok"] for item in results], [True, True, True])

    def test_checkpoint_activate_restart_is_consumed_before_data_plane(self):
        commands = SchedulerCommandQueue()
        external_pending = threading.Event()
        processed = []

        class Scheduler:
            tp_rank = 0
            is_afd_component_standby_rank = False
            _afd_component_control_group = object()
            _afd_component_active_control_group = object()
            _afd_component_active_control_generation = 0
            _afd_component_command_queue = commands
            _afd_component_external_control_pending = external_pending
            _afd_component_restart_active_iteration = False

            def __init__(self):
                self._afd_component_runtime = SimpleNamespace(handle=self.handle)

            def handle(self, payload):
                processed.append(payload["action"])
                if payload["action"] == "activate":
                    self._afd_component_restart_active_iteration = True
                return {"ok": True}

            def afd_component_poll_through_wakeup(self):
                return commands.poll_through_wakeup(self.handle)

            def afd_component_safe_point(self):
                pass

        scheduler = Scheduler()
        result = {}
        submitter = threading.Thread(
            target=lambda: result.update(
                commands.submit(
                    {
                        "action": "activate", "operation_id": "op",
                        "epoch": 1, "_wake_scheduler": True,
                    },
                    1, on_queued=external_pending.set,
                )
            )
        )
        submitter.start()
        self.assertTrue(external_pending.wait(0.5))
        with mock.patch(
            "sglang.srt.utils.common.broadcast_pyobj",
            side_effect=lambda data, *args, **kwargs: data,
        ):
            self.assertTrue(
                SchedulerAFDMixin.afd_component_post_receive_control_checkpoint(
                    scheduler
                )
            )
        self.assertTrue(
            SchedulerAFDMixin.afd_component_should_restart_active_iteration(
                scheduler
            )
        )
        submitter.join(1)
        self.assertEqual(processed, ["activate"])
        self.assertTrue(result["ok"])

    def test_command_executes_only_when_scheduler_polls(self):
        commands = SchedulerCommandQueue()
        result = {}
        thread = threading.Thread(target=lambda: result.update(commands.submit({"action": "x"}, 1)))
        thread.start()
        time.sleep(0.02)
        self.assertTrue(thread.is_alive())
        commands.poll(lambda payload: {"ok": True, "thread": threading.get_ident()})
        thread.join(1)
        self.assertTrue(result["ok"])
        self.assertEqual(result["thread"], threading.get_ident())

    def test_configured_timeout_allows_prepare_slower_than_old_limit(self):
        commands = SchedulerCommandQueue()
        result = {}

        def slow_prepare(payload):
            time.sleep(0.12)
            return {"ok": True}

        thread = threading.Thread(
            target=lambda: result.update(commands.submit({"action": "prepare"}, 0.25))
        )
        thread.start()
        time.sleep(0.02)
        poller = threading.Thread(target=lambda: commands.poll(slow_prepare))
        poller.start()
        thread.join(0.5)
        poller.join(0.5)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["ok"])

        timed_out = {}
        thread = threading.Thread(
            target=lambda: self._capture_timeout(commands, timed_out), daemon=True
        )
        thread.start()
        thread.join(0.2)
        self.assertIn("scheduler command timed out: prepare", timed_out["error"])
        # Drain the expired command so a real scheduler loop cannot retain it.
        commands.poll(lambda payload: {"ok": False})

    @staticmethod
    def _capture_timeout(commands, result):
        try:
            commands.submit({"action": "prepare"}, 0.03)
        except TimeoutError as exc:
            result["error"] = str(exc)

    def test_slow_prepare_timeout_runs_abort_after_handler_returns(self):
        commands = SchedulerCommandQueue()
        events = []
        result = {}

        def slow_handler(payload):
            events.append(payload["action"])
            if payload["action"] == "prepare":
                time.sleep(0.08)
            return {"ok": True}

        submitter = threading.Thread(
            target=lambda: self._capture_submit_timeout(
                commands, result, {
                    "action": "prepare", "operation_id": "slow", "epoch": 0
                }, 0.02,
            )
        )
        submitter.start()
        time.sleep(0.005)
        poller = threading.Thread(target=lambda: commands.poll(slow_handler))
        poller.start()
        submitter.join(0.2)
        poller.join(0.2)

        self.assertIn("scheduler command timed out: prepare", result["error"])
        self.assertEqual(events, ["prepare", "abort"])

    @staticmethod
    def _capture_submit_timeout(commands, result, payload, timeout):
        try:
            commands.submit(payload, timeout)
        except TimeoutError as exc:
            result["error"] = str(exc)

    def test_unsupported_commit_gate(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        adapter = ModelRunnerComponentAdapter(scheduler.tp_worker.model_runner, scheduler.server_args.afd_perspective, 4)
        capability = adapter.prepare({"target_attn_tp": 4, "target_ffn_tp": 1})
        self.assertTrue(capability.supported)
        self.assertFalse(capability.can_commit)
        self.assertIn("staging driver is not attached", capability.reason)

    def test_commit_gate_opens_only_with_ready_stager(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        events = []

        class Stager:
            prepared = None

            def safety_reason(self, request):
                return ""

            def prepare(self, request):
                events.append("prepare")
                self.prepared = SimpleNamespace(
                    state=SimpleNamespace(value="READY"), shadow={"weight": object()}
                )
                return self.prepared

            def abort(self, reason):
                events.append("abort")
                self.prepared.shadow.clear()
                self.prepared = None

            def activate(self, request):
                events.append("activate")

            def retire(self, request):
                events.append("retire")

        runner = scheduler.tp_worker.model_runner
        runner._afd_component_stager = Stager()
        adapter = ModelRunnerComponentAdapter(
            runner, scheduler.server_args.afd_perspective, 4
        )
        request = {
            "operation_id": "op",
            "epoch": 0,
            "expected_attn_tp": 2,
            "expected_ffn_tp": 2,
            "target_attn_tp": 4,
            "target_ffn_tp": 4,
        }
        self.assertTrue(adapter.capability(request).can_commit)
        self.assertTrue(adapter.prepare(request).can_commit)
        adapter.activate(request)
        adapter.retire(request)
        self.assertEqual(events, ["prepare", "activate", "retire"])


    def test_prepare_failure_aborts_local_shadow(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)

        class Stager:
            def __init__(self):
                self.prepared = SimpleNamespace(shadow={"weight": object()})

            def safety_reason(self, request):
                return ""

            def abort(self, reason):
                self.prepared.shadow.clear()
                self.prepared = None

        stager = Stager()
        scheduler.tp_worker.model_runner._afd_component_stager = stager
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=0.1)
        runtime.adapter.capability = lambda request: SimpleNamespace(
            supported=True, can_commit=True, reason="",
            model_family="qwen3", perspective="attn", active_tp=2, max_tp=4,
        )
        scheduler.afd_component_fanout_command = lambda payload: (_ for _ in ()).throw(
            TimeoutError("collective timed out")
        )
        request = {
            "action": "prepare", "operation_id": "op", "epoch": 0,
            "expected_attn_tp": 2, "expected_ffn_tp": 2,
            "target_attn_tp": 4, "target_ffn_tp": 4,
        }
        with self.assertRaisesRegex(TimeoutError, "collective timed out"):
            runtime.handle(request)
        self.assertIsNone(stager.prepared)

    def test_attention_ffn_quiesce_ack_over_control_channel(self):
        # Avoid CUDA communicator construction in this CPU control-path test.
        class Runtime(AFDComponentSchedulerRuntime):
            def _strict_comm_drain(self):
                self.scheduler.afd_reshard_drain_state().set_pending(sends=0, recvs=0, fences=0)

        ffn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        ffn_runtime = Runtime(ffn)
        server = ZMQSchedulerControlServer("tcp://127.0.0.1:38991", ffn._afd_component_command_queue)
        stop = threading.Event()
        poller = threading.Thread(target=lambda: self._poll_until(stop, ffn_runtime), daemon=True)
        poller.start()
        try:
            attn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
            attn_runtime = Runtime(attn, ZMQSchedulerControlClient("tcp://127.0.0.1:38991"))
            request = {
                "action": "quiesce", "operation_id": "op", "epoch": 0,
                "target_attn_tp": 4, "target_ffn_tp": 1, "timeout": 1,
            }
            first = attn_runtime.handle(dict(request))
            self.assertTrue(first["retryable"])
            self.assertEqual(attn.fanout_calls, [])
            attn.mark_world_quiesced("op")
            # Start the first peer request and wait on its completion event so
            # the FFN fence is definitely installed before fake world progress.
            self.assertTrue(attn_runtime.handle(dict(request))["retryable"])
            attempt = attn_runtime._quiesce_peer_attempt
            self.assertIsNotNone(attempt)
            self.assertTrue(attempt["done"].wait(1.0), "initial FFN fence RPC hung")
            peer_pending = attn_runtime.handle(dict(request))
            self.assertTrue(peer_pending["retryable"])
            ffn.mark_world_quiesced("op")
            response = self._poll_handle_until_terminal(
                attn_runtime, request, timeout=1.0
            )
            self.assertTrue(response["ok"])
            self.assertTrue(attn._afd_reshard_drain.drained)
            self.assertEqual(response["watermark"], 0)
        finally:
            stop.set()
            server.close()
            poller.join(1)

    @staticmethod
    def _poll_until(stop, runtime):
        while not stop.is_set():
            runtime.poll()
            time.sleep(0.001)

    def _poll_handle_until_terminal(self, runtime, request, timeout=1.0):
        deadline = time.monotonic() + timeout
        while True:
            response = runtime.handle(dict(request))
            if response.get("ok") or not response.get("retryable", False):
                return response
            attempt = runtime._quiesce_peer_attempt
            remaining = deadline - time.monotonic()
            self.assertGreater(remaining, 0, response.get("detail", "quiesce timeout"))
            if attempt is not None:
                attempt["done"].wait(remaining)
            else:
                # Yield until the fake scheduler consumes the queued peer RPC;
                # the condition-variable wait avoids timing assumptions.
                threading.Event().wait(min(remaining, 0.01))

    def test_quiesce_returns_retryable_while_busy_then_ok_when_idle(self):
        # FFN is mid-forward (not idle) on the first poll, becomes idle after
        # the second poll. Quiesce must fence admission on the first attempt and
        # return a retryable ACK rather than failing.
        ffn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        ffn._idle_after_polls = 1
        ffn_runtime = _QuiesceRuntime(ffn)
        base = {
            "action": "quiesce", "operation_id": "op", "epoch": 0,
            "target_attn_tp": 4, "target_ffn_tp": 1, "timeout": 1, "watermark": 0,
        }
        started = time.perf_counter()
        first = ffn_runtime.handle(dict(base))
        self.assertLess(time.perf_counter() - started, 0.1)
        self.assertFalse(first["ok"])
        self.assertTrue(first["retryable"])
        self.assertTrue(ffn_runtime.admission_blocked)
        # The handler itself never enters world fanout. The event-loop state
        # machine marks completion only after active-ready consensus.
        self.assertEqual(len(ffn.fanout_calls), 0)
        ffn.mark_world_quiesced("op")
        second = ffn_runtime.handle(dict(base))
        self.assertTrue(second["ok"])
        self.assertEqual(second["watermark"], 0)
        # Re-issued quiesce must NOT re-run the world fanout.
        self.assertEqual(len(ffn.fanout_calls), 1)

    def test_proxy_retire_does_not_contact_scheduler(self):
        class RejectingClient:
            def request(self, payload, timeout):
                raise AssertionError("proxy retire must not send a scheduler command")

        proxy = AFDComponentSchedulerProxyAdapter(RejectingClient(), timeout=1)
        request = SimpleNamespace(operation_id="op", expected_epoch=1)

        receipt = proxy.retire(request, transitions={})

        self.assertTrue(receipt.completed)
        self.assertFalse(receipt.retryable)
        self.assertIn("ACTIVATE", receipt.detail)
        self.assertIn("no scheduler command", receipt.detail)

    def test_proxy_drain_polls_until_pair_idle(self):
        # End-to-end: Attn coordinator drives quiesce through the proxy adapter.
        # FFN needs a couple polls to drain; the proxy must retry with backoff
        # and return a completed receipt rather than failing pre-commit.
        ffn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        ffn._idle_after_polls = 2

        class SafePointRuntime(_QuiesceRuntime):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.request_observed_before_sync = False

            def poll(self):
                consumed = super().poll()
                if (
                    self.admission_blocked
                    and self.scheduler._afd_component_pending_fence_payload
                ):
                    # The command poll requested quiesce in this round, but the
                    # effective gate was still open until this fake next-loop
                    # active-TP sync step.
                    self.request_observed_before_sync = not (
                        self.scheduler._afd_component_fence_installed
                    )
                    self.scheduler.sync_pending_quiesce()
                if (
                    self.admission_blocked
                    and not self.scheduler._afd_component_quiesce_world_done
                    and self._afd_quiescent_reason() is None
                ):
                    # After effective sync, model active-ready consensus and the
                    # max-world completion; production code is not bypassed.
                    self.scheduler.mark_world_quiesced(self.operation_id)
                return consumed

        ffn_runtime = SafePointRuntime(ffn)
        server = ZMQSchedulerControlServer(
            "tcp://127.0.0.1:38992", ffn._afd_component_command_queue
        )
        stop = threading.Event()
        poller = threading.Thread(
            target=lambda: self._poll_until(stop, ffn_runtime), daemon=True
        )
        poller.start()
        try:
            # Drive quiesce through the proxy bounded-retry loop.
            proxy = AFDComponentSchedulerProxyAdapter(
                ZMQSchedulerControlClient("tcp://127.0.0.1:38992"), timeout=2
            )
            request = SimpleNamespace(
                stage="prefill", expected_attn_tp=2, expected_ffn_tp=2,
                target_attn_tp=4, target_ffn_tp=1, expected_epoch=0,
                operation_id="op", dry_run=False,
            )
            # The proxy talks to whichever endpoint it was given; here it drives
            # FFN directly to validate the poll-until-idle contract in isolation.
            receipt = proxy.drain(request, 0)
            self.assertTrue(receipt.completed)
            self.assertTrue(ffn_runtime.request_observed_before_sync)
        finally:
            stop.set()
            server.close()
            poller.join(1)

    def test_quiescent_reason_requires_fence_and_no_pending(self):
        # AFD-quiescent is stricter than is_fully_idle: it must also see a fenced
        # admission, no in-flight batch info, an empty pending queue, and no
        # pending recvs. Each unsatisfied condition yields a retryable reason.
        ffn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        runtime = _QuiesceRuntime(ffn)
        runtime._afd_pending_recv_count = lambda: 0

        # Not fenced yet -> not quiescent even though is_fully_idle() is True.
        self.assertFalse(runtime.admission_blocked)
        self.assertIn("not requested", runtime._afd_quiescent_reason())

        runtime.admission_blocked = True
        self.assertIn("fence sync pending", runtime._afd_quiescent_reason())
        ffn._afd_component_fence_installed = True
        # In-flight AFD batch info blocks quiescence.
        ffn._afd_batchsize_attn = 4
        self.assertIn("batch info", runtime._afd_quiescent_reason())
        ffn._afd_batchsize_attn = None

        # Pending queued batch info blocks quiescence.
        ffn._afd_pending_batch_infos.append(object())
        self.assertIn("pending", runtime._afd_quiescent_reason())
        ffn._afd_pending_batch_infos.clear()

        # Pending recvs block quiescence.
        runtime._afd_pending_recv_count = lambda: 2
        self.assertIn("in flight", runtime._afd_quiescent_reason())
        runtime._afd_pending_recv_count = lambda: 0

        # is_fully_idle still guards as additional protection.
        ffn._idle_after_polls = 10_000
        self.assertIn("not fully idle", runtime._afd_quiescent_reason())
        ffn._idle_after_polls = 0
        ffn._idle_poll_count = 0

        # All satisfied -> quiescent.
        self.assertIsNone(runtime._afd_quiescent_reason())

    def test_strict_comm_drain_without_communicator_does_not_construct(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        scheduler._afd_reshard_drain.set_pending(sends=2, recvs=3, fences=4)

        with mock.patch(
            "sglang.srt.layers.afd.peek_async_communicator", return_value=None
        ) as peek, mock.patch(
            "sglang.srt.layers.afd.get_async_communicator",
            side_effect=AssertionError("finalize must not construct"),
        ) as construct:
            runtime._strict_comm_drain()

        peek.assert_called_once_with()
        construct.assert_not_called()
        self.assertEqual(scheduler._afd_reshard_drain.pending_sends, 0)
        self.assertEqual(scheduler._afd_reshard_drain.pending_recvs, 0)
        self.assertEqual(scheduler._afd_reshard_drain.pending_fences, 0)

    def test_strict_comm_drain_existing_pending_receive_raises(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        comm = SimpleNamespace(drain_sends=mock.Mock(), _pending_recv_count=2)

        with mock.patch(
            "sglang.srt.layers.afd.peek_async_communicator", return_value=comm
        ), self.assertRaisesRegex(RuntimeError, "2 pending receives"):
            runtime._strict_comm_drain()

        comm.drain_sends.assert_called_once_with()

    def test_pending_receive_poll_does_not_construct_communicator(self):
        ffn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        runtime = _QuiesceRuntime(ffn)
        with mock.patch(
            "sglang.srt.layers.afd.peek_async_communicator", return_value=None
        ) as peek, mock.patch(
            "sglang.srt.layers.afd.get_async_communicator",
            side_effect=AssertionError("status polling must not construct"),
        ) as construct:
            self.assertEqual(runtime._afd_pending_recv_count(), 0)

        peek.assert_called_once_with()
        construct.assert_not_called()

    def test_quiesce_retryable_when_afd_work_in_flight_then_ok(self):
        # Even when is_fully_idle() reports idle, an in-flight AFD batch info
        # must keep quiesce retryable (fence installed), then succeed once the
        # AFD ledger drains.
        ffn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        ffn._afd_batchsize_attn = 8  # mid-forward AFD work
        runtime = _QuiesceRuntime(ffn)
        runtime._afd_pending_recv_count = lambda: 0
        base = {
            "action": "quiesce", "operation_id": "op", "epoch": 0,
            "target_attn_tp": 4, "target_ffn_tp": 1, "timeout": 1, "watermark": 0,
        }
        first = runtime.handle(dict(base))
        self.assertFalse(first["ok"])
        self.assertTrue(first["retryable"])
        self.assertTrue(runtime.admission_blocked)
        self.assertEqual(len(ffn.fanout_calls), 0)

        # AFD work drains, then the event-loop consensus performs one world fanout.
        ffn._afd_batchsize_attn = None
        ffn.mark_world_quiesced("op")
        second = runtime.handle(dict(base))
        self.assertTrue(second["ok"])
        self.assertEqual(second["watermark"], 0)
        # Fence fanout is not re-run.
        self.assertEqual(len(ffn.fanout_calls), 1)

    def test_activate_success_releases_both_local_fences(self):
        ffn_scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        ffn_runtime = AFDComponentSchedulerRuntime(ffn_scheduler, command_timeout=1)
        ffn_runtime.admission_blocked = True
        ffn_runtime.operation_id = "op"

        class Peer:
            def __init__(self):
                self.calls = []

            def request(self, payload, timeout):
                self.calls.append((dict(payload), timeout))
                return ffn_runtime.handle(dict(payload))

        attn_scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        peer = Peer()
        attn_runtime = AFDComponentSchedulerRuntime(
            attn_scheduler, peer, command_timeout=1
        )
        attn_runtime.admission_blocked = True
        attn_runtime.operation_id = "op"
        for scheduler in (attn_scheduler, ffn_scheduler):
            scheduler._afd_reshard_drain.dispatch_seq = 7
            scheduler._afd_reshard_drain.cut_watermark = 7
            scheduler._afd_reshard_drain.attn_ack = 7
            scheduler._afd_reshard_drain.ffn_ack = 7
        request = {
            "action": "activate", "operation_id": "op", "epoch": 1,
            "target_attn_tp": 4, "target_ffn_tp": 1,
        }

        response = attn_runtime.handle(request)

        self.assertTrue(response["ok"])
        self.assertEqual([call[0] for call in peer.calls], [request])
        self.assertFalse(ffn_runtime.admission_blocked)
        self.assertIsNone(ffn_runtime.operation_id)
        self.assertFalse(attn_runtime.admission_blocked)
        self.assertIsNone(attn_runtime.operation_id)
        self.assertEqual(len(ffn_scheduler.fanout_calls), 1)
        self.assertEqual(len(attn_scheduler.fanout_calls), 1)
        self.assertEqual(attn_scheduler._afd_reshard_drain.next_dispatch(), 8)
        self.assertEqual(ffn_scheduler._afd_reshard_drain.next_dispatch(), 8)

    def test_activate_idempotent_readiness_never_calls_peer_from_event_loop(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)

        class Peer:
            def request(self, payload, timeout):
                raise AssertionError(
                    "idempotent readiness must not query peer in scheduler loop"
                )

        runtime = AFDComponentSchedulerRuntime(scheduler, Peer(), command_timeout=1)
        runtime._activated_operation = "ready-op"
        runtime._activated_epoch = 1
        runtime._activation_fanout_started = True
        runtime._activation_fanout_operation = "ready-op"
        runtime._activation_fanout_epoch = 1
        scheduler._afd_component_serving_ready = True
        response = runtime.handle({
            "action": "activate", "operation_id": "ready-op", "epoch": 1,
            "target_attn_tp": 4, "target_ffn_tp": 2,
        })

        self.assertTrue(response["ok"])
        self.assertFalse(runtime._activation_fanout_started)
        self.assertEqual(scheduler.fanout_calls, [])

    def test_serving_readiness_is_local_read_only_state(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        runtime._activated_operation = "ready-op"
        runtime._activated_epoch = 1
        runtime._activation_fanout_started = True
        runtime._activation_fanout_operation = "ready-op"
        runtime._activation_fanout_epoch = 1
        scheduler._afd_component_serving_ready = False
        payload = {"operation_id": "ready-op", "epoch": 1}

        pending = runtime.serving_readiness(payload)
        self.assertFalse(pending["ok"])
        self.assertTrue(pending["retryable"])
        self.assertTrue(runtime._activation_fanout_started)
        self.assertEqual(scheduler.fanout_calls, [])

        mismatch = runtime.serving_readiness(
            {"operation_id": "different-op", "epoch": 2}
        )
        self.assertFalse(mismatch["ok"])
        self.assertTrue(runtime._activation_fanout_started)

        scheduler._afd_component_serving_ready = True
        mismatched_ready = runtime.serving_readiness(
            {"operation_id": "different-op", "epoch": 2}
        )
        self.assertFalse(mismatched_ready["ok"])
        self.assertTrue(runtime._activation_fanout_started)

        ready = runtime.serving_readiness(payload)
        self.assertTrue(ready["ok"])
        self.assertFalse(runtime._activation_fanout_started)
        self.assertEqual(scheduler.fanout_calls, [])

        # The completion identity remains published after resolving the guard.
        retry = runtime.serving_readiness(payload)
        self.assertTrue(retry["ok"])
        self.assertEqual(scheduler.fanout_calls, [])

    def test_direct_readiness_resolution_allows_next_activate_fanout(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        runtime._activated_operation = "op1"
        runtime._activated_epoch = 1
        runtime._activation_fanout_started = True
        runtime._activation_fanout_operation = "op1"
        runtime._activation_fanout_epoch = 1
        scheduler._afd_component_serving_ready = True

        op1 = {"operation_id": "op1", "epoch": 1}
        self.assertTrue(runtime.serving_readiness(op1)["ok"])
        self.assertFalse(runtime._activation_fanout_started)
        self.assertTrue(runtime.serving_readiness(op1)["ok"])

        op2 = {
            "action": "activate",
            "operation_id": "op2",
            "epoch": 2,
            "target_attn_tp": 1,
            "target_ffn_tp": 4,
        }
        response = runtime.handle(op2)

        self.assertTrue(response["ok"])
        self.assertEqual(len(scheduler.fanout_calls), 1)
        self.assertEqual(scheduler.fanout_calls[0]["operation"], "op2")
        self.assertEqual(scheduler.fanout_calls[0]["action"], "activate")
        self.assertEqual(runtime._activated_operation, "op2")
        self.assertEqual(runtime._activated_epoch, 2)

    def test_activate_waits_for_serving_ready_without_duplicate_fanout(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        runtime.admission_blocked = True
        runtime.operation_id = "ready-op"
        scheduler._afd_component_serving_ready = True

        def activate_fanout(payload):
            scheduler.fanout_calls.append(dict(payload))
            scheduler._afd_component_serving_ready = False

        scheduler.afd_component_fanout_command = activate_fanout
        request = {
            "action": "activate", "operation_id": "ready-op", "epoch": 1,
            "target_attn_tp": 4, "target_ffn_tp": 2,
        }

        first = runtime.handle(dict(request))
        self.assertFalse(first["ok"])
        self.assertTrue(first["retryable"])
        self.assertIn("serving readiness", first["detail"])
        self.assertEqual(len(scheduler.fanout_calls), 1)
        self.assertFalse(runtime.admission_blocked)

        second = runtime.handle(dict(request))
        self.assertFalse(second["ok"])
        self.assertTrue(second["retryable"])
        self.assertEqual(len(scheduler.fanout_calls), 1)

        scheduler._afd_component_serving_ready = True
        completed = runtime.handle(dict(request))
        self.assertTrue(completed["ok"])
        self.assertEqual(len(scheduler.fanout_calls), 1)

    def test_historical_dispatch_sequence_does_not_block_empty_scheduler(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        scheduler._afd_component_fence_installed = True
        scheduler._afd_reshard_drain.dispatch_seq = 30
        scheduler.afd_component_scheduler_idle_reason = lambda: None
        with mock.patch(
            "sglang.srt.layers.afd.peek_async_communicator", return_value=None
        ) as peek, mock.patch(
            "sglang.srt.layers.afd.get_async_communicator",
            side_effect=AssertionError("observation must not construct"),
        ) as construct:
            reason = SchedulerAFDMixin.afd_component_local_quiescent_reason(
                scheduler
            )
        self.assertIsNone(reason)
        peek.assert_called_once_with()
        construct.assert_not_called()
        self.assertEqual(scheduler._afd_reshard_drain.dispatch_seq, 30)

    def test_fence_allows_admitted_batch_once_before_sealing_cut(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        scheduler._afd_component_active_control_group = object()
        scheduler._afd_component_fence_payload = {
            "operation_id": "cut-op", "component": "attn", "action": "quiesce"
        }
        for _ in range(30):
            scheduler._afd_reshard_drain.next_dispatch()

        # Phase 1 is admission-only: a batch scheduled before fence sync may
        # still perform its first and only dispatch.
        scheduler._afd_component_fence_installed = True
        self.assertIsNone(scheduler._afd_reshard_drain.cut_watermark)
        self.assertEqual(scheduler._afd_reshard_drain.next_dispatch(), 31)

        with mock.patch(
            "sglang.srt.utils.common.broadcast_pyobj",
            side_effect=lambda data, *args, **kwargs: data,
        ):
            watermark = SchedulerAFDMixin.afd_component_seal_quiesce_cut(
                scheduler
            )
        self.assertEqual(watermark, 31)
        self.assertEqual(scheduler._afd_reshard_drain.cut_watermark, 31)
        self.assertTrue(scheduler._afd_component_fence_sealed)
        with self.assertRaisesRegex(RuntimeError, "dispatch is fenced"):
            scheduler._afd_reshard_drain.next_dispatch()

    def test_attention_follower_installs_rank_zero_cut(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        scheduler.tp_rank = 1
        scheduler._afd_component_active_control_group = object()
        scheduler._afd_component_fence_payload = {
            "operation_id": "cut-op", "component": "attn", "action": "quiesce"
        }
        authoritative = {
            "action": "seal_cut", "operation_id": "cut-op",
            "generation": 0, "watermark": 31,
        }
        with mock.patch(
            "sglang.srt.utils.common.broadcast_pyobj",
            return_value=[authoritative],
        ):
            watermark = SchedulerAFDMixin.afd_component_seal_quiesce_cut(
                scheduler
            )
        self.assertEqual(scheduler._afd_reshard_drain.dispatch_seq, 0)
        self.assertEqual(watermark, 31)
        self.assertEqual(scheduler._afd_reshard_drain.cut_watermark, 31)
        with self.assertRaisesRegex(RuntimeError, "dispatch is fenced"):
            scheduler._afd_reshard_drain.next_dispatch()

    def test_seal_cut_rejects_operation_mismatch(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        scheduler.tp_rank = 1
        scheduler._afd_component_active_control_group = object()
        scheduler._afd_component_fence_payload = {
            "operation_id": "cut-op", "component": "attn", "action": "quiesce"
        }
        polluted = {
            "action": "seal_cut", "operation_id": "other-op",
            "generation": 0, "watermark": 31,
        }
        with mock.patch(
            "sglang.srt.utils.common.broadcast_pyobj", return_value=[polluted]
        ), self.assertRaisesRegex(RuntimeError, "operation or generation mismatch"):
            SchedulerAFDMixin.afd_component_seal_quiesce_cut(scheduler)
        self.assertIsNone(scheduler._afd_reshard_drain.cut_watermark)

    def test_ffn_installs_attention_owned_external_cut(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        scheduler._afd_component_active_control_group = object()
        scheduler._afd_component_fence_payload = {
            "operation_id": "cut-op", "component": "ffn", "action": "quiesce",
            "watermark": 31,
        }
        watermark = SchedulerAFDMixin.afd_component_seal_quiesce_cut(
            scheduler
        )
        self.assertEqual(watermark, 31)
        self.assertEqual(scheduler._afd_reshard_drain.cut_watermark, 31)

    def test_abort_force_reopens_dispatch_ledger(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        scheduler._afd_reshard_drain.next_dispatch()
        scheduler._afd_reshard_drain.cut()
        runtime.admission_blocked = True
        runtime.operation_id = "abort-op"

        response = runtime.handle(
            {
                "action": "abort",
                "operation_id": "abort-op",
                "epoch": 0,
                "target_attn_tp": 2,
                "target_ffn_tp": 2,
            }
        )

        self.assertTrue(response["ok"])
        self.assertFalse(runtime.admission_blocked)
        self.assertEqual(scheduler._afd_reshard_drain.next_dispatch(), 2)

    def test_activate_starts_peer_before_local_world_fanout_completes(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        peer_started = threading.Event()
        release_peer = threading.Event()
        local_started = threading.Event()
        events = []

        class Peer:
            def request(self, payload, timeout):
                events.append("peer-start")
                peer_started.set()
                release_peer.wait(timeout)
                events.append("peer-done")
                return {"ok": True}

        def fanout(payload):
            events.append("local-start")
            local_started.set()
            self.assertTrue(peer_started.wait(0.5))
            events.append("local-done")

        scheduler.afd_component_fanout_command = fanout
        runtime = AFDComponentSchedulerRuntime(scheduler, Peer(), command_timeout=1)
        request = {
            "action": "activate", "operation_id": "parallel-activate", "epoch": 1,
            "target_attn_tp": 4, "target_ffn_tp": 4,
        }
        result = {}
        thread = threading.Thread(target=lambda: result.update(runtime.handle(request)))
        thread.start()
        self.assertTrue(local_started.wait(0.5))
        self.assertTrue(peer_started.is_set())
        release_peer.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["ok"])
        self.assertLess(events.index("peer-start"), events.index("local-done"))

    def test_activate_peer_timeout_is_bounded(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        release = threading.Event()

        class Peer:
            def request(self, payload, timeout):
                release.wait(2)
                return {"ok": True}

        runtime = AFDComponentSchedulerRuntime(scheduler, Peer(), command_timeout=0.02)
        request = {
            "action": "activate", "operation_id": "activate-timeout", "epoch": 1,
            "target_attn_tp": 4, "target_ffn_tp": 4, "timeout": 0.02,
        }
        try:
            with self.assertRaisesRegex(TimeoutError, "FFN activate did not finish"):
                runtime.handle(request)
        finally:
            release.set()

    def test_activate_peer_reject_keeps_attn_fenced(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)

        class RejectingPeer:
            def request(self, payload, timeout):
                return {"ok": False, "detail": "activation failed"}

        runtime = AFDComponentSchedulerRuntime(
            scheduler, RejectingPeer(), command_timeout=1
        )
        runtime.admission_blocked = True
        runtime.operation_id = "op"
        request = {
            "action": "activate", "operation_id": "op", "epoch": 1,
            "target_attn_tp": 4, "target_ffn_tp": 1,
        }

        response = runtime.handle(request)

        self.assertFalse(response["ok"])
        self.assertIn("activation failed", response["detail"])
        self.assertTrue(runtime.admission_blocked)
        self.assertEqual(runtime.operation_id, "op")
        self.assertEqual(len(scheduler.fanout_calls), 1)

    def test_retire_is_local_idempotent_and_releases_fence(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)

        class Peer:
            def request(self, payload, timeout):
                raise AssertionError("retire must not contact the peer")

        peer = Peer()
        runtime = AFDComponentSchedulerRuntime(scheduler, peer, command_timeout=1)
        retire_calls = []
        runtime.adapter.retire = lambda payload: retire_calls.append(dict(payload))
        runtime.admission_blocked = True
        runtime.operation_id = "op"
        request = {
            "action": "retire", "operation_id": "op", "epoch": 1,
            "target_attn_tp": 4, "target_ffn_tp": 1,
        }

        response = runtime.handle(request)

        self.assertTrue(response["ok"])
        self.assertEqual(retire_calls, [request])
        self.assertEqual(scheduler.fanout_calls, [])
        self.assertFalse(runtime.admission_blocked)
        self.assertIsNone(runtime.operation_id)

        # Repeated external RETIRE remains rank-zero-local and harmless after
        # ACTIVATE has already cleared the prepared state and admission fence.
        self.assertTrue(runtime.handle(dict(request))["ok"])
        self.assertEqual(retire_calls, [request, request])
        self.assertEqual(scheduler.fanout_calls, [])

    def test_command_queue_still_consumed_after_admission_fenced(self):
        # Regression for the silent drain stall: after quiesce installs the
        # admission fence, the scheduler event loop must keep polling and
        # consuming subsequent commands (retryable quiesce re-polls, then
        # activate/retire). If poll() stopped consuming once fenced, drain would
        # hang forever with no further progress.
        ffn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        ffn._idle_after_polls = 1  # not idle on the first quiesce
        runtime = _QuiesceRuntime(ffn)
        base = {
            "action": "quiesce", "operation_id": "op", "epoch": 0,
            "target_attn_tp": 4, "target_ffn_tp": 1, "timeout": 1, "watermark": 0,
        }
        # First quiesce fences admission and returns retryable (still busy).
        first = runtime.handle(dict(base))
        self.assertTrue(runtime.admission_blocked)
        self.assertTrue(first["retryable"])

        # Simulate the event-loop active-ready/world phase before re-poll.
        ffn.mark_world_quiesced("op")
        # Submit a follow-up command through the real MPSC queue and confirm a
        # single poll() consumes it even though admission is already fenced.
        result = {}
        submitter = threading.Thread(
            target=lambda: result.update(
                ffn._afd_component_command_queue.submit(dict(base), 1)
            )
        )
        submitter.start()
        time.sleep(0.02)
        self.assertTrue(submitter.is_alive())  # blocked until the loop polls
        consumed = runtime.poll()
        submitter.join(1)
        self.assertEqual(consumed, 1)
        self.assertIn("ok", result)
        # Now idle -> the re-polled quiesce reports drained.
        self.assertTrue(result["ok"])

    def test_proxy_drain_times_out_when_never_idle(self):
        ffn = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_FFN)
        ffn._idle_after_polls = 10_000  # never idle within the test window
        ffn_runtime = _QuiesceRuntime(ffn)
        server = ZMQSchedulerControlServer(
            "tcp://127.0.0.1:38993", ffn._afd_component_command_queue
        )
        stop = threading.Event()
        poller = threading.Thread(
            target=lambda: self._poll_until(stop, ffn_runtime), daemon=True
        )
        poller.start()
        try:
            proxy = AFDComponentSchedulerProxyAdapter(
                ZMQSchedulerControlClient("tcp://127.0.0.1:38993"), timeout=0.3
            )
            request = SimpleNamespace(
                stage="prefill", expected_attn_tp=2, expected_ffn_tp=2,
                target_attn_tp=4, target_ffn_tp=1, expected_epoch=0,
                operation_id="timeout-op", dry_run=False,
            )
            receipt = proxy.drain(request, 0)
            self.assertFalse(receipt.completed)
            self.assertFalse(receipt.retryable)
            self.assertIn("did not drain", receipt.detail)
        finally:
            stop.set()
            server.close()
            poller.join(1)


    def test_prepare_is_retryable_until_post_activate_serving_ready(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        scheduler._afd_component_serving_ready = False
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        request = {
            "action": "prepare", "operation_id": "epoch2", "epoch": 1,
            "expected_attn_tp": 4, "expected_ffn_tp": 4,
            "target_attn_tp": 2, "target_ffn_tp": 2,
        }

        pending = runtime.handle(dict(request))
        self.assertFalse(pending["ok"])
        self.assertTrue(pending["retryable"])
        self.assertIn("serving readiness", pending["detail"])
        self.assertEqual(scheduler.fanout_calls, [])

        scheduler._afd_component_serving_ready = True
        runtime.adapter.capability = lambda payload: SimpleNamespace(
            supported=True, can_commit=True, reason="", model_family="qwen3",
            perspective="attn", active_tp=4, max_tp=4,
        )
        accepted = runtime.handle(dict(request))
        self.assertTrue(accepted["ok"])
        self.assertEqual(len(scheduler.fanout_calls), 1)
        self.assertEqual(scheduler.fanout_calls[0]["operation"], "epoch2")

    def test_runtime_prepare_and_status_never_call_peer(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)

        class RejectingPeer:
            def request(self, payload, timeout):
                raise AssertionError(
                    f"scheduler-loop {payload['action']} must not call peer"
                )

        runtime = AFDComponentSchedulerRuntime(
            scheduler, RejectingPeer(), command_timeout=1
        )
        runtime.adapter.capability = lambda payload: SimpleNamespace(
            supported=True, can_commit=True, reason="", model_family="qwen3",
            perspective="attn", active_tp=4, max_tp=4,
        )
        runtime.adapter.prepare_status = lambda payload: {
            "state": "PREPARING", "error": None,
            "timings_s": {"load": 0.2}, "rank_timings_s": [{"load": 0.2}],
        }
        request = {
            "operation_id": "local-only", "epoch": 1,
            "target_attn_tp": 2, "target_ffn_tp": 2,
        }

        prepared = runtime.handle({**request, "action": "prepare"})
        status = runtime.handle({**request, "action": "prepare_status"})

        self.assertTrue(prepared["ok"])
        self.assertTrue(prepared["accepted"])
        self.assertEqual(len(scheduler.fanout_calls), 1)
        self.assertTrue(status["ok"])
        self.assertEqual(status["state"], "PREPARING")
        self.assertEqual(status["breakdown"]["attn"]["critical"], {"load": 0.2})

    def test_proxy_prepare_initial_commands_are_concurrent(self):
        both_started = threading.Event()
        both_entered = threading.Barrier(2, action=both_started.set)
        release = threading.Event()

        class Client:
            def __init__(self, component):
                self.component = component

            def request(self, payload, timeout):
                if payload["action"] == "prepare":
                    both_entered.wait(timeout=0.5)
                    release.wait(timeout=0.5)
                    return {"ok": True, "accepted": True}
                if payload["action"] == "prepare_status":
                    return {
                        "ok": True, "state": "READY",
                        "breakdown": {
                            self.component: {"critical": {}, "ranks": []}
                        },
                    }
                if payload["action"] == "abort":
                    return {"ok": True}
                raise AssertionError(payload)

        proxy = AFDComponentSchedulerProxyAdapter(
            Client("attn"), timeout=1, peer_client=Client("ffn")
        )
        request = SimpleNamespace(
            stage="prefill", expected_attn_tp=4, expected_ffn_tp=4,
            target_attn_tp=2, target_ffn_tp=2, expected_epoch=1,
            operation_id="parallel-prepare", dry_run=False,
        )
        result = {}
        thread = threading.Thread(
            target=lambda: result.update(
                receipt=proxy.prepare(request, transitions={})
            )
        )
        thread.start()
        try:
            # The barrier can only pass if both RPCs started before either one
            # returned; checking call order alone would not prove concurrency.
            self.assertTrue(
                both_started.wait(0.5),
                "both component PREPARE requests were not concurrently in flight",
            )
        finally:
            release.set()
        thread.join(1)
        self.assertFalse(thread.is_alive())
        self.assertTrue(result["receipt"].completed)

    def test_proxy_prepare_retries_components_independently(self):
        class Client:
            def __init__(self, component, pending_count):
                self.component = component
                self.pending_count = pending_count
                self.prepare_attempts = 0

            def request(self, payload, timeout):
                if payload["action"] == "prepare":
                    self.prepare_attempts += 1
                    if self.prepare_attempts <= self.pending_count:
                        return {"ok": False, "retryable": True, "detail": "pending"}
                    return {"ok": True, "accepted": True}
                if payload["action"] == "prepare_status":
                    return {
                        "ok": True, "state": "READY",
                        "breakdown": {
                            self.component: {"critical": {}, "ranks": []}
                        },
                    }
                if payload["action"] == "abort":
                    return {"ok": True}
                raise AssertionError(payload)

        attn, ffn = Client("attn", 1), Client("ffn", 2)
        proxy = AFDComponentSchedulerProxyAdapter(
            attn, timeout=1, peer_client=ffn
        )
        request = SimpleNamespace(
            stage="prefill", expected_attn_tp=4, expected_ffn_tp=4,
            target_attn_tp=2, target_ffn_tp=2, expected_epoch=1,
            operation_id="independent-retry", dry_run=False,
        )
        with mock.patch.dict(
            "os.environ", {"AFD_RESHARD_PREPARE_POLL_INTERVAL": "0.001"}
        ):
            receipt = proxy.prepare(request, transitions={})

        self.assertTrue(receipt.completed)
        self.assertEqual(attn.prepare_attempts, 2)
        self.assertEqual(ffn.prepare_attempts, 3)

    def test_proxy_prepare_waits_for_ffn_ready_and_merges_breakdown(self):
        class Client:
            def __init__(self, component, status_states):
                self.component = component
                self.status_states = iter(status_states)
                self.status_calls = 0

            def request(self, payload, timeout):
                if payload["action"] == "prepare":
                    return {"ok": True, "accepted": True}
                if payload["action"] == "prepare_status":
                    self.status_calls += 1
                    return {
                        "ok": True, "state": next(self.status_states),
                        "breakdown": {
                            self.component: {
                                "critical": {"load": self.status_calls},
                                "ranks": [{"rank": 0}],
                            }
                        },
                    }
                if payload["action"] == "abort":
                    return {"ok": True}
                raise AssertionError(payload)

        attn = Client("attn", ["READY"])
        ffn = Client("ffn", ["PREPARING", "READY"])
        proxy = AFDComponentSchedulerProxyAdapter(
            attn, timeout=1, peer_client=ffn
        )
        request = SimpleNamespace(
            stage="prefill", expected_attn_tp=4, expected_ffn_tp=4,
            target_attn_tp=2, target_ffn_tp=2, expected_epoch=1,
            operation_id="joined-status", dry_run=False,
        )
        with mock.patch.dict(
            "os.environ", {"AFD_RESHARD_PREPARE_POLL_INTERVAL": "0.001"}
        ):
            receipt = proxy.prepare(request, transitions={})

        self.assertTrue(receipt.completed)
        self.assertEqual(attn.status_calls, 1)
        self.assertEqual(ffn.status_calls, 2)
        self.assertEqual(set(receipt.breakdown), {"attn", "ffn", "prepare_s"})
        self.assertEqual(receipt.breakdown["ffn"]["critical"], {"load": 2})

    def test_proxy_prepare_component_failure_aborts_both(self):
        class Client:
            def __init__(self, component, fail=False):
                self.component = component
                self.fail = fail
                self.aborts = 0

            def request(self, payload, timeout):
                if payload["action"] == "prepare":
                    if self.fail:
                        return {"ok": False, "detail": "staging rejected"}
                    return {"ok": True, "accepted": True}
                if payload["action"] == "abort":
                    self.aborts += 1
                    return {"ok": True}
                raise AssertionError(payload)

        attn, ffn = Client("attn"), Client("ffn", fail=True)
        proxy = AFDComponentSchedulerProxyAdapter(
            attn, timeout=1, peer_client=ffn
        )
        request = SimpleNamespace(
            expected_epoch=1, operation_id="failed-prepare",
            target_attn_tp=2, target_ffn_tp=2,
        )
        receipt = proxy.prepare(request, transitions={})

        self.assertFalse(receipt.completed)
        self.assertFalse(receipt.retryable)
        self.assertIn("ffn prepare rejected", receipt.detail)
        self.assertEqual(attn.aborts, 1)
        self.assertEqual(ffn.aborts, 1)

    def test_proxy_activate_polls_both_components_until_ready(self):
        class Client:
            def __init__(self, responses):
                self.responses = iter(responses)
                self.calls = []

            def request(self, payload, timeout):
                self.calls.append((dict(payload), timeout))
                response = next(self.responses)
                if isinstance(response, Exception):
                    raise response
                return response

        initial_breakdown = {
            "attn": {"critical": {"peer_transfer_s": 0.4}, "ranks": []},
            "ffn": {"critical": {"local_repartition_s": 0.2}, "ranks": []},
        }
        pending = {
            "ok": False,
            "retryable": True,
            "detail": "post-activate serving readiness pending",
            "breakdown": initial_breakdown,
        }
        attn = Client([
            pending,  # Sole full-timeout request owns real A/F fanout.
            TimeoutError("ATTN event loop still resuming"),
            {"ok": True, "breakdown": {"attn": initial_breakdown["attn"]}},
        ])
        ffn = Client([
            {"ok": False, "retryable": True, "detail": "pending"},
            {"ok": True, "breakdown": {"ffn": initial_breakdown["ffn"]}},
        ])
        proxy = AFDComponentSchedulerProxyAdapter(
            attn, timeout=1, peer_client=ffn
        )
        request = SimpleNamespace(
            stage="prefill", expected_attn_tp=2, expected_ffn_tp=2,
            target_attn_tp=4, target_ffn_tp=4, expected_epoch=0,
            operation_id="activate-ready", dry_run=False,
        )
        with mock.patch.dict(
            "os.environ",
            {
                "AFD_RESHARD_ACTIVATE_POLL_INTERVAL": "0.001",
                "AFD_RESHARD_ACTIVATE_RETRY_TIMEOUT": "0.01",
            },
        ):
            receipt = proxy.activate(request, transitions={})

        self.assertTrue(receipt.completed)
        self.assertEqual(receipt.breakdown, initial_breakdown)
        self.assertEqual(len(attn.calls), 3)
        self.assertEqual(len(ffn.calls), 2)
        self.assertEqual(attn.calls[0][1], 1)
        for payload, timeout in attn.calls[1:] + ffn.calls:
            self.assertEqual(payload["action"], "serving_readiness")
            self.assertTrue(payload["_readiness_retry"])
            self.assertEqual(payload["_transport_grace"], 0.0)
            self.assertLessEqual(timeout, 0.01)

    def test_proxy_activate_does_not_finish_before_ffn_ready(self):
        class AttnClient:
            def __init__(self):
                self.calls = 0

            def request(self, payload, timeout):
                self.calls += 1
                if self.calls == 1:
                    return {"ok": False, "retryable": True, "detail": "pending"}
                return {"ok": True}

        class FfnClient:
            def __init__(self):
                self.calls = 0

            def request(self, payload, timeout):
                self.calls += 1
                if self.calls < 3:
                    return {"ok": False, "retryable": True, "detail": "ffn pending"}
                return {"ok": True}

        attn, ffn = AttnClient(), FfnClient()
        proxy = AFDComponentSchedulerProxyAdapter(
            attn, timeout=1, peer_client=ffn
        )
        request = SimpleNamespace(
            expected_epoch=0, operation_id="both-ready",
            target_attn_tp=4, target_ffn_tp=4,
        )
        with mock.patch.dict(
            "os.environ", {"AFD_RESHARD_ACTIVATE_POLL_INTERVAL": "0.001"}
        ):
            receipt = proxy.activate(request, transitions={})

        self.assertTrue(receipt.completed)
        self.assertEqual(attn.calls, 2)
        self.assertEqual(ffn.calls, 3)

    def test_prepare_safe_point_only_starts_background_work(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        runtime.adapter.capability = lambda request: SimpleNamespace(
            supported=True, can_commit=True, reason="", model_family="qwen3",
            perspective="attn", active_tp=2, max_tp=4,
        )
        started = threading.Event()
        release = threading.Event()

        def fanout(payload):
            started.set()
            # Simulate safe-point start: no staging wait is performed here.
            return payload

        scheduler.afd_component_fanout_command = fanout
        request = {"action": "prepare", "operation_id": "async-op", "epoch": 0,
                   "expected_attn_tp": 2, "expected_ffn_tp": 2,
                   "target_attn_tp": 4, "target_ffn_tp": 4}
        before = time.perf_counter()
        response = runtime.handle(request)
        self.assertLess(time.perf_counter() - before, 0.1)
        self.assertTrue(response["ok"])
        self.assertTrue(response["accepted"])
        self.assertTrue(started.is_set())
        release.set()

    def test_prepare_status_propagates_background_error(self):
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        runtime.adapter.prepare_status = lambda request: {
            "state": "ERROR", "error": "injected background error", "timings_s": {}
        }
        response = runtime.handle({
            "action": "prepare_status", "operation_id": "op", "epoch": 0,
            "target_attn_tp": 4, "target_ffn_tp": 4,
        })
        self.assertFalse(response["ok"])
        self.assertEqual(response["state"], "ERROR")
        self.assertIn("injected background error", response["detail"])

    def test_prepare_status_never_uses_scheduler_world_fanout(self):
        # Status is rank-zero-local. Background staging itself uses only the
        # dedicated staging control group, so polling must not consume a command
        # generation while standby ranks block in their world receiver.
        scheduler = _Scheduler(AFDPerspective.AFD_PERSPECTIVE_ATTN)
        runtime = AFDComponentSchedulerRuntime(scheduler, command_timeout=1)
        runtime.adapter.prepare_status = lambda request: {
            "state": "PREPARING", "error": None, "timings_s": {},
            "rank_timings_s": [],
        }
        response = runtime.handle({
            "action": "prepare_status", "operation_id": "op", "epoch": 0,
            "target_attn_tp": 4, "target_ffn_tp": 4,
        })
        self.assertTrue(response["ok"])
        self.assertEqual(response["state"], "PREPARING")
        self.assertEqual(scheduler.fanout_calls, [])


class _FakeBatch:
    def __init__(self, reqs=None):
        self.reqs = list(reqs or [])
        self.batch_is_full = False

    def is_empty(self):
        return len(self.reqs) == 0


class _FakeReq:
    def __init__(self, rid):
        self.rid = rid
        # req_pool_idx=None means release_kv_cache is skipped (no CUDA needed).
        self.req_pool_idx = None
        self._finished = False

    def finished(self):
        return self._finished


class _FakeFFNScheduler:
    """Minimal FFN scheduler exercising the real idle-ledger reset methods."""

    from sglang.srt.disaggregation.utils import DisaggregationMode
    from sglang.srt.managers.scheduler import Scheduler

    _afd_ffn_reset_idle_ledger = Scheduler._afd_ffn_reset_idle_ledger
    _afd_ffn_cleanup_all = Scheduler._afd_ffn_cleanup_all
    is_fully_idle = Scheduler.is_fully_idle

    def __init__(self, *, admission_blocked, last_batch_reqs):
        self._afd_pending_batch_infos = deque()
        self._afd_component_runtime = SimpleNamespace(
            admission_blocked=admission_blocked
        )
        self._afd_component_fence_installed = admission_blocked
        self.running_batch = _FakeBatch()
        self.last_batch = _FakeBatch(last_batch_reqs)
        self.cur_batch = self.last_batch  # cur aliases the last forward's batch
        self.chunked_req = None
        self.waiting_queue = []
        self.tree_cache = SimpleNamespace()
        self.dllm_manager = SimpleNamespace(any_staging_reqs=lambda: False)
        self.enable_overlap = False
        self.result_queue = deque()
        self.pp_size = 1
        self.running_mbs = []
        self.grammar_manager = SimpleNamespace(grammar_queue=[])
        self.enable_hierarchical_cache = False
        self.disaggregation_mode = self.DisaggregationMode.PREFILL
        self.disagg_prefill_inflight_queue = []
        self.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])


class TestAFDFFNIdleLedgerReset(CustomTestCase):
    """Fix 1: FFN idle wait branch must reset the batch ledger to idle."""

    def _run_reset(self, sched):
        # afd_is_attn() must be False for the FFN cleanup path to run.
        with mock.patch("sglang.srt.layers.afd.afd_is_attn", return_value=False):
            sched._afd_ffn_reset_idle_ledger()

    def test_fenced_admission_resets_ledger_to_idle(self):
        sched = _FakeFFNScheduler(
            admission_blocked=True, last_batch_reqs=[_FakeReq("a"), _FakeReq("b")]
        )
        # Before: last_batch/cur_batch non-empty -> not idle.
        self.assertFalse(sched.is_fully_idle())
        self._run_reset(sched)
        # After: ledger drained -> idle True.
        self.assertIsNone(sched.last_batch)
        self.assertIsNone(sched.cur_batch)
        self.assertTrue(sched.is_fully_idle())

    def test_stale_running_batch_resets_ledger(self):
        sched = _FakeFFNScheduler(
            admission_blocked=False, last_batch_reqs=[_FakeReq("x")]
        )
        # Legacy path: stale running_batch present even without a fence.
        sched.running_batch = _FakeBatch([_FakeReq("stale")])
        self.assertFalse(sched.is_fully_idle())
        self._run_reset(sched)
        self.assertTrue(sched.running_batch.is_empty())
        self.assertIsNone(sched.last_batch)
        self.assertTrue(sched.is_fully_idle())

    def test_missing_communicator_does_not_construct_and_resets(self):
        sched = _FakeFFNScheduler(
            admission_blocked=True, last_batch_reqs=[_FakeReq("epoch1")]
        )
        with mock.patch(
            "sglang.srt.layers.afd.afd_is_attn", return_value=False
        ), mock.patch(
            "sglang.srt.layers.afd.peek_async_communicator", return_value=None
        ) as peek, mock.patch(
            "sglang.srt.layers.afd.get_async_communicator",
            side_effect=AssertionError("idle observation must not construct"),
        ) as construct:
            sched._afd_ffn_reset_idle_ledger()

        peek.assert_called_once_with()
        construct.assert_not_called()
        self.assertIsNone(sched.last_batch)
        self.assertIsNone(sched.cur_batch)
        self.assertTrue(sched.is_fully_idle())

    def test_existing_communicator_pending_work_blocks_reset(self):
        sched = _FakeFFNScheduler(
            admission_blocked=True, last_batch_reqs=[_FakeReq("in-flight")]
        )
        comm = SimpleNamespace(_pending_recv_count=1)
        with mock.patch(
            "sglang.srt.layers.afd.afd_is_attn", return_value=False
        ), mock.patch(
            "sglang.srt.layers.afd.peek_async_communicator", return_value=comm
        ):
            sched._afd_ffn_reset_idle_ledger()

        self.assertIsNotNone(sched.last_batch)
        self.assertEqual([req.rid for req in sched.last_batch.reqs], ["in-flight"])
        self.assertIs(sched.cur_batch, sched.last_batch)

    def test_no_reset_during_normal_serving_gap(self):
        # Not fenced, running_batch empty: live decode reqs in last_batch must
        # be preserved (Attn is only briefly paused between dispatches).
        reqs = [_FakeReq("live1"), _FakeReq("live2")]
        sched = _FakeFFNScheduler(admission_blocked=False, last_batch_reqs=reqs)
        self._run_reset(sched)
        self.assertIsNotNone(sched.last_batch)
        self.assertEqual(len(sched.last_batch.reqs), 2)
        self.assertFalse(sched.is_fully_idle())

    def test_no_reset_when_afd_work_pending(self):
        # A queued AFDReqInput means work is arriving; do not clobber the ledger.
        sched = _FakeFFNScheduler(
            admission_blocked=True, last_batch_reqs=[_FakeReq("q")]
        )
        sched._afd_pending_batch_infos.append(object())
        self._run_reset(sched)
        self.assertIsNotNone(sched.last_batch)
        self.assertEqual(len(sched.last_batch.reqs), 1)


class _TransferReq:
    def __init__(self, rid):
        self.rid = rid
        self.bootstrap_room = 17
        self.return_logprob = False
        self.finished_reason = None
        self.req_pool_idx = 3


class _TransferReceiver:
    def __init__(self, poll):
        self._poll = poll
        self.aborted = False
        self.conclude_state = None

    def poll(self):
        return self._poll

    def abort(self):
        from sglang.srt.disaggregation.base import KVPoll

        self.aborted = True
        self._poll = KVPoll.Failed

    def failure_exception(self):
        raise RuntimeError("injected transfer failure")


class TestAFDDecodeTransferQuiesce(CustomTestCase):
    def _make_queue(self, poll):
        from sglang.srt.disaggregation.decode import (
            DecodeRequest,
            DecodeTransferQueue,
        )

        scheduler = SimpleNamespace(
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            stream_output=mock.Mock(),
            enable_metrics=False,
        )
        allocator = SimpleNamespace(free=mock.Mock())
        queue = DecodeTransferQueue(
            gloo_group=object(),
            req_to_metadata_buffer_idx_allocator=allocator,
            tp_rank=0,
            metadata_buffers=SimpleNamespace(),
            scheduler=scheduler,
            tree_cache=SimpleNamespace(),
        )
        receiver = _TransferReceiver(poll)
        decode_req = DecodeRequest(
            req=_TransferReq("rid-1"),
            kv_receiver=receiver,
            metadata_buffer_index=9,
            transfer_started_at=time.monotonic(),
        )
        queue.add(decode_req)
        return queue, decode_req, allocator

    def test_pending_transfer_blocks_without_releasing(self):
        from sglang.srt.disaggregation.base import KVPoll

        queue, decode_req, allocator = self._make_queue(KVPoll.Transferring)
        with mock.patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Transferring],
        ), mock.patch(
            "sglang.srt.disaggregation.decode.release_kv_cache"
        ) as release:
            self.assertEqual(queue.pop_transferred(), [])
        self.assertEqual(queue.queue, [decode_req])
        allocator.free.assert_not_called()
        release.assert_not_called()

    def test_failed_or_abort_reaps_and_releases_resources(self):
        from sglang.srt.disaggregation.base import KVPoll

        queue, decode_req, allocator = self._make_queue(KVPoll.Failed)
        with mock.patch(
            "sglang.srt.disaggregation.decode.poll_and_all_reduce",
            return_value=[KVPoll.Failed],
        ), mock.patch(
            "sglang.srt.disaggregation.decode.prepare_abort"
        ), mock.patch(
            "sglang.srt.disaggregation.decode.release_kv_cache"
        ) as release:
            queue.abort_and_reap(["rid-1"], "client AbortReq")
        self.assertTrue(decode_req.kv_receiver.aborted)
        self.assertEqual(queue.queue, [])
        allocator.free.assert_called_once_with(9)
        release.assert_called_once_with(
            decode_req.req, queue.tree_cache, is_insert=False
        )

    def test_fence_grace_uses_rank0_rids_on_every_rank(self):
        payload = {
            "action": "abort_decode_transfers",
            "operation_id": "op",
            "generation": 4,
            "rids": ["rid-1"],
        }

        def make_scheduler(rank):
            queue = SimpleNamespace(
                queue=[SimpleNamespace(req=SimpleNamespace(rid="rid-1"))],
                pop_transferred=mock.Mock(return_value=[]),
                describe=mock.Mock(return_value=[]),
                abort_and_reap=mock.Mock(return_value=[]),
            )
            return SimpleNamespace(
                tp_rank=rank,
                server_args=SimpleNamespace(
                    afd_reshard_transfer_abort_grace=0.01,
                    afd_reshard_timeout=1.0,
                ),
                _afd_component_fence_installed=True,
                _afd_component_fence_payload={"operation_id": "op"},
                _afd_component_fence_monotonic=time.monotonic() - 1.0,
                _afd_component_transfer_log_monotonic=time.monotonic(),
                _afd_component_active_control_generation=4,
                _afd_component_active_control_group=object(),
                disagg_decode_transfer_queue=queue,
                waiting_queue=[],
            )

        ranks = [make_scheduler(0), make_scheduler(1)]
        with mock.patch(
            "sglang.srt.layers.afd.afd_is_attn", return_value=True
        ), mock.patch(
            "sglang.srt.utils.common.broadcast_pyobj", return_value=[payload]
        ):
            for scheduler in ranks:
                SchedulerAFDMixin.afd_component_progress_decode_transfers(scheduler)

        for scheduler in ranks:
            scheduler.disagg_decode_transfer_queue.abort_and_reap.assert_called_once()
            args = scheduler.disagg_decode_transfer_queue.abort_and_reap.call_args.args
            self.assertEqual(args[0], ["rid-1"])
            self.assertEqual(scheduler._afd_component_active_control_generation, 5)

    def test_fence_grace_keeps_pending_and_new_work_is_deferred(self):
        from sglang.srt.managers.io_struct import TokenizedGenerateReqInput

        queue = SimpleNamespace(
            queue=[SimpleNamespace(req=SimpleNamespace(rid="rid-1"))],
            pop_transferred=mock.Mock(return_value=[]),
            describe=mock.Mock(return_value=[]),
            abort_and_reap=mock.Mock(),
        )
        scheduler = SimpleNamespace(
            tp_rank=0,
            server_args=SimpleNamespace(
                afd_reshard_transfer_abort_grace=2.0, afd_reshard_timeout=10.0
            ),
            _afd_component_fence_installed=True,
            _afd_component_fence_payload={"operation_id": "op"},
            _afd_component_fence_monotonic=time.monotonic(),
            _afd_component_transfer_log_monotonic=time.monotonic(),
            _afd_component_active_control_generation=0,
            _afd_component_active_control_group=object(),
            disagg_decode_transfer_queue=queue,
            waiting_queue=[],
        )
        no_abort = {
            "action": "abort_decode_transfers",
            "operation_id": "op",
            "generation": 0,
            "rids": [],
        }
        with mock.patch(
            "sglang.srt.layers.afd.afd_is_attn", return_value=True
        ), mock.patch(
            "sglang.srt.utils.common.broadcast_pyobj", return_value=[no_abort]
        ):
            SchedulerAFDMixin.afd_component_progress_decode_transfers(scheduler)
        queue.pop_transferred.assert_called_once_with()
        queue.abort_and_reap.assert_not_called()

        scheduler._afd_deferred_work_requests = deque()
        work = mock.create_autospec(TokenizedGenerateReqInput, instance=True)
        with mock.patch(
            "sglang.srt.layers.afd.afd_is_attn", return_value=True
        ):
            self.assertEqual(
                SchedulerAFDMixin.afd_gate_work_requests(scheduler, [work]), []
            )
        self.assertEqual(list(scheduler._afd_deferred_work_requests), [work])


if __name__ == "__main__":
    unittest.main(verbosity=3)
