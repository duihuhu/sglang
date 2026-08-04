# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Production runtime boundary for the AFD component reshard MVP.

CUDA/model operations are submitted to the scheduler-owned queue and are only
executed by :meth:`AFDComponentSchedulerRuntime.poll` from the event-loop thread.
The HTTP process only owns :class:`AFDComponentSchedulerProxyAdapter`.
"""
from __future__ import annotations

import logging
import os
import queue
import threading
from dataclasses import asdict, dataclass, field, is_dataclass
from typing import Any, Dict, Optional

import zmq

from sglang.srt.reshard.afd_component_runtime import AdapterReceipt

logger = logging.getLogger(__name__)


def _capability_dict(capability: Any) -> Dict[str, Any]:
    """Serialize production dataclasses and lightweight test/extension objects."""
    if is_dataclass(capability) and not isinstance(capability, type):
        return asdict(capability)
    try:
        return dict(vars(capability))
    except TypeError as exc:
        raise TypeError(
            "component capability must be a dataclass or expose attributes"
        ) from exc


@dataclass(frozen=True)
class AFDComponentCapability:
    supported: bool
    can_commit: bool
    reason: str
    model_family: str = ""
    perspective: str = ""
    active_tp: int = 0
    max_tp: int = 0


# Only phase-changing commands must preempt the serving data plane.  Read-only
# queries (capability/prepare_status/status) remain FIFO scheduler commands but
# cannot continuously wake the active-world checkpoint.  serving_readiness has
# its dedicated transport-thread read-only provider and never enters this queue.
SCHEDULER_WAKEUP_ACTIONS = frozenset(
    {"prepare", "quiesce", "drain", "activate", "abort", "demote"}
)


@dataclass
class _Command:
    payload: Dict[str, Any]
    done: threading.Event
    wake_scheduler: bool = False
    response: Optional[Dict[str, Any]] = None
    expired: threading.Event = field(default_factory=threading.Event)


class SchedulerCommandQueue:
    """MPSC FIFO whose consumer is exclusively the scheduler event loop."""

    def __init__(self) -> None:
        self._commands: "queue.Queue[_Command]" = queue.Queue()
        self._wakeup_lock = threading.Lock()
        self._wakeup_count = 0
        self.pending = threading.Event()

    def submit(
        self, payload: Dict[str, Any], timeout: float, on_queued=None
    ) -> Dict[str, Any]:
        wake_scheduler = bool(payload.get("_wake_scheduler", False))
        command = _Command(payload, threading.Event(), wake_scheduler)
        if wake_scheduler:
            with self._wakeup_lock:
                self._wakeup_count += 1
                self.pending.set()
        self._commands.put(command)
        # Notify only after the command and wakeup metadata are both published.
        # This closes the callback-before-queue race without executing scheduler
        # work on the transport thread.
        if on_queued is not None:
            on_queued()
        if not command.done.wait(timeout):
            command.expired.set()
            raise TimeoutError(f"scheduler command timed out: {payload.get('action')}")
        assert command.response is not None
        return command.response

    def _finish_wakeup(self, command: _Command) -> None:
        if not command.wake_scheduler:
            return
        with self._wakeup_lock:
            self._wakeup_count -= 1
            if self._wakeup_count <= 0:
                self._wakeup_count = 0
                self.pending.clear()

    def _poll_one(self, handler):
        try:
            command = self._commands.get_nowait()
        except queue.Empty:
            return None
        try:
            if command.expired.is_set():
                command.response = {
                    "ok": False,
                    "retryable": bool(
                        command.payload.get("_readiness_retry", False)
                    ),
                    "detail": (
                        "scheduler command expired before execution: "
                        f"{command.payload.get('action')}"
                    ),
                }
            else:
                command.response = handler(command.payload)
            # A timed-out PREPARE must not leave late READY shadows behind.
            if (
                command.expired.is_set()
                and command.payload.get("action") == "prepare"
            ):
                try:
                    handler({**command.payload, "action": "abort"})
                except Exception:
                    pass
        except Exception as exc:
            command.response = {"ok": False, "detail": str(exc)}
        finally:
            self._finish_wakeup(command)
            command.done.set()
        return command

    def poll(self, handler, limit: int = 1) -> int:
        count = 0
        while count < limit and self._poll_one(handler) is not None:
            count += 1
        return count

    def poll_through_wakeup(self, handler) -> tuple[int, bool]:
        """Drain FIFO readers ahead of, and including, one wakeup command."""
        count = 0
        while self.has_wakeup_pending():
            command = self._poll_one(handler)
            if command is None:
                break
            count += 1
            if command.wake_scheduler:
                return count, True
        return count, False

    def has_pending(self) -> bool:
        return not self._commands.empty()

    def has_wakeup_pending(self) -> bool:
        with self._wakeup_lock:
            return self._wakeup_count > 0


class ZMQSchedulerControlServer:
    """Transport thread. It never invokes model/CUDA code directly."""

    def __init__(
        self,
        endpoint: str,
        command_queue: SchedulerCommandQueue,
        default_timeout: float = 600.0,
        readiness_provider=None,
        command_pending_callback=None,
    ) -> None:
        self.endpoint = endpoint
        self.command_queue = command_queue
        self.default_timeout = float(default_timeout)
        # A serving scheduler may be blocked in a TP request collective and
        # unable to poll its command queue. This provider is deliberately
        # read-only and lets the transport thread report local readiness without
        # entering model/CUDA code or waiting for the scheduler event loop.
        self.readiness_provider = readiness_provider
        # Callback is restricted to a threading.Event-style notification.  In
        # particular it must not touch a ZMQ socket, CUDA, or a process group.
        self.command_pending_callback = command_pending_callback
        self._closed = threading.Event()
        self._thread = threading.Thread(target=self._run, name="afd-component-control", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(self.endpoint)
        poller = zmq.Poller()
        poller.register(socket, zmq.POLLIN)
        try:
            while not self._closed.is_set():
                if not poller.poll(50):
                    continue
                payload = socket.recv_pyobj()
                timeout = max(float(payload.get("timeout", self.default_timeout)), 0.001)
                transport_grace = max(
                    float(
                        payload.get(
                            "_transport_grace",
                            os.getenv("AFD_RESHARD_TRANSPORT_GRACE", "5"),
                        )
                    ),
                    0.0,
                )
                try:
                    if (
                        payload.get("action") == "serving_readiness"
                        and self.readiness_provider is not None
                    ):
                        response = self.readiness_provider(payload)
                    else:
                        action = str(payload.get("action", ""))
                        wake_scheduler = action in SCHEDULER_WAKEUP_ACTIONS
                        queued_payload = {
                            **payload, "_wake_scheduler": wake_scheduler
                        }
                        # Only phase-changing commands wake active serving ranks.
                        # Read-only/status commands stay FIFO but cannot starve an
                        # ACTIVATE restart/readiness boundary.
                        callback = (
                            self.command_pending_callback
                            if wake_scheduler else None
                        )
                        if wake_scheduler:
                            logger.info(
                                "AFD control transport queued wakeup action=%s "
                                "operation=%s",
                                action, payload.get("operation_id"),
                            )
                        response = self.command_queue.submit(
                            queued_payload, timeout + transport_grace,
                            on_queued=callback,
                        )
                except Exception as exc:
                    response = {
                        "ok": False,
                        "retryable": bool(payload.get("_readiness_retry", False)),
                        "detail": str(exc),
                    }
                socket.send_pyobj(response)
        finally:
            socket.close()
            context.term()

    def close(self) -> None:
        self._closed.set()
        self._thread.join(timeout=1)


class ZMQSchedulerControlClient:
    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def request(self, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
        context = zmq.Context()
        socket = context.socket(zmq.REQ)
        transport_grace = max(
            float(
                payload.get(
                    "_transport_grace",
                    os.getenv("AFD_RESHARD_TRANSPORT_GRACE", "5"),
                )
            ),
            0.0,
        )
        timeout_ms = max(int((timeout + 2 * transport_grace) * 1000), 1)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
        socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
        socket.connect(self.endpoint)
        try:
            socket.send_pyobj({**payload, "timeout": timeout})
            return socket.recv_pyobj()
        except zmq.ZMQError as exc:
            raise TimeoutError(f"AFD control request to {self.endpoint} failed: {exc}") from exc
        finally:
            socket.close()
            context.term()


class ModelRunnerComponentAdapter:
    """Capability and transaction adapter over a real ModelRunner.

    Commit is advertised only when the runner exposes a max-world collective
    stager. This fail-closed gate prevents rank-0-only control from reporting a
    false READY while peer ranks have not staged their shadows.
    """

    def __init__(self, model_runner, perspective: str, max_tp: int) -> None:
        self.model_runner = model_runner
        self.perspective = perspective
        self.max_tp = max_tp

    def capability(self, request: Dict[str, Any]) -> AFDComponentCapability:
        runner = self.model_runner
        hf = getattr(getattr(runner, "model_config", None), "hf_config", None)
        family = str(getattr(hf, "model_type", "")).lower()
        model = getattr(runner, "model", None)
        reasons = []
        if family != "qwen3":
            reasons.append(f"model family {family!r} is not Qwen3 dense")
        if getattr(hf, "num_experts", None) or getattr(
            hf, "num_local_experts", None
        ):
            reasons.append("MoE is unsupported")
        if int(getattr(runner.server_args, "afd_micro_batch", 0)) != 1:
            reasons.append("M must be 1")
        if not getattr(runner.server_args, "disable_cuda_graph", False):
            reasons.append("CUDA graph must be disabled")
        if getattr(runner, "pp_size", 1) != 1:
            reasons.append("only one stage/PP=1 is supported")
        if model is None:
            reasons.append("rank has no loaded model")
        if self.max_tp < max(
            int(request["target_attn_tp"]), int(request["target_ffn_tp"])
        ):
            reasons.append("target exceeds launched max-world")
        if model is not None and not any(True for _ in model.named_parameters()):
            reasons.append("model has no named parameters")
        supported = not reasons
        commit_reasons = list(reasons)
        stager = getattr(runner, "_afd_component_stager", None)
        if stager is None:
            commit_reasons.append("max-world component staging driver is not attached")
        else:
            safety_reason = stager.safety_reason(request)
            if safety_reason:
                commit_reasons.append(safety_reason)
        can_commit = not commit_reasons
        return AFDComponentCapability(
            supported,
            can_commit,
            "; ".join(commit_reasons) if commit_reasons else "component staging/commit hooks ready",
            family,
            str(getattr(self.perspective, "value", self.perspective)),
            int(getattr(runner, "tp_size", 0)),
            self.max_tp,
        )

    def prepare(self, request: Dict[str, Any]) -> AFDComponentCapability:
        capability = self.capability(request)
        if not capability.can_commit:
            return capability
        state = self.model_runner._afd_component_stager.prepare(request)
        self.last_prepare_timings = dict(getattr(state, "timings_s", None) or {})
        if state.state.value != "READY":
            return AFDComponentCapability(
                False, False, f"staging did not reach READY: {state.state.value}",
                capability.model_family, capability.perspective,
                capability.active_tp, capability.max_tp,
            )
        return capability

    def prepare_status(self, request: Dict[str, Any]) -> Dict[str, Any]:
        stager = self.model_runner._afd_component_stager
        return stager.prepare_status(request.get("operation_id"))

    def activate(self, request: Dict[str, Any]) -> None:
        capability = self.capability(request)
        if not capability.can_commit:
            raise RuntimeError(capability.reason)
        self.model_runner._afd_component_stager.activate(request)

    def completed_breakdown(self, operation_id: str) -> Dict[str, Any]:
        stager = getattr(self.model_runner, "_afd_component_stager", None)
        completed = dict(getattr(stager, "last_completed_status", {}) or {})
        if completed.get("operation_id") != str(operation_id):
            return {}
        return {
            "critical": dict(completed.get("timings_s", {})),
            "ranks": list(completed.get("rank_timings_s", [])),
            "aggregation": "rank-local timings; critical is coordinator rank, ranks contains all-rank samples",
        }

    def retire(self, request: Dict[str, Any]) -> None:
        stager = getattr(self.model_runner, "_afd_component_stager", None)
        if stager is not None:
            stager.retire(request)


class AFDComponentSchedulerRuntime:
    """Scheduler-thread command handler for coordinator or participant."""

    def __init__(self, scheduler, peer_client: Optional[ZMQSchedulerControlClient] = None, command_timeout: float = 600.0) -> None:
        self.scheduler = scheduler
        self.peer_client = peer_client
        self.command_timeout = float(command_timeout)
        self.adapter = ModelRunnerComponentAdapter(scheduler.tp_worker.model_runner, scheduler.server_args.afd_perspective, int(scheduler.server_args.afd_component_max_tp))
        self.admission_blocked = False
        self.operation_id: Optional[str] = None
        self._quiesce_peer_attempt: Optional[Dict[str, Any]] = None
        self._quiesce_peer_terminal: Optional[Dict[str, Any]] = None
        # ACTIVATE's max-world fanout is destructive and must run exactly once.
        # Keep its identity after releasing the admission fence so HTTP retries
        # can wait for the scheduler-owned serving-readiness boundary without
        # repeating any collective.
        self._activated_operation: Optional[str] = None
        self._activated_epoch: Optional[int] = None
        self._activation_fanout_started = False
        self._activation_fanout_operation: Optional[str] = None
        self._activation_fanout_epoch: Optional[int] = None
        # serving_readiness runs on the transport thread while handle runs on
        # the scheduler thread. Keep activation identity publication and fanout
        # resolution in one small critical section.
        self._activation_state_lock = threading.Lock()

    def serving_readiness(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """Read local post-ACTIVATE state without scheduler-loop participation."""
        operation_id = str(payload.get("operation_id", ""))
        epoch = int(payload.get("epoch", -1))
        with self._activation_state_lock:
            matches = (
                self._activated_operation == operation_id
                and self._activated_epoch == epoch
            )
            if not matches:
                return {
                    "ok": False,
                    "retryable": self._activation_fanout_started,
                    "detail": (
                        "activation identity not published"
                        if self._activation_fanout_started
                        else "activation identity mismatch"
                    ),
                    "operation_id": operation_id,
                    "epoch": epoch,
                }
            ready = bool(
                getattr(self.scheduler, "_afd_component_serving_ready", True)
            )
            if ready and (
                self._activation_fanout_operation == operation_id
                and self._activation_fanout_epoch == epoch
            ):
                # Preserve the identity as the idempotent completion record, but
                # release this operation's guard so the next epoch may begin.
                # The owner check prevents a late retry for the prior identity
                # from clearing a newer operation's in-progress fanout.
                self._activation_fanout_started = False
        component = "ffn" if self._is_ffn() else "attn"
        return {
            "ok": ready,
            "retryable": not ready,
            "detail": "" if ready else "post-activate serving readiness pending",
            "operation_id": operation_id,
            "epoch": epoch,
            "breakdown": {
                component: self.adapter.completed_breakdown(operation_id)
            },
        }

    def poll(self) -> int:
        return self.scheduler._afd_component_command_queue.poll(self.handle, limit=1)

    def handle(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        action = payload.get("action")
        operation_id = str(payload.get("operation_id", ""))
        epoch = int(payload.get("epoch", -1))
        if not operation_id or epoch < 0:
            raise ValueError("operation_id and non-negative epoch are required")
        if action == "capability":
            return {"ok": True, "capability": _capability_dict(self.adapter.capability(payload))}
        if action == "prepare":
            if not bool(getattr(self.scheduler, "_afd_component_serving_ready", True)):
                return {
                    "ok": False,
                    "retryable": True,
                    "detail": "post-activate serving readiness pending",
                    "operation_id": operation_id,
                    "epoch": epoch,
                }
            cap = self.adapter.capability(payload)
            if not cap.supported or not cap.can_commit:
                return {"ok": False, "detail": cap.reason, "capability": _capability_dict(cap)}
            try:
                # PREPARE is scheduler-loop local. In particular, never wait on
                # the other component here: both component worlds may need to
                # enter communicator staging concurrently. The HTTP proxy owns
                # both clients and starts those requests in parallel.
                self._fanout(payload, "prepare")
            except Exception:
                # Fanout may fail after one or more ranks started their background
                # thread. Propagate ABORT best-effort and always clean rank zero.
                self._abort_after_fanout_failure(payload)
                raise
            return {
                "ok": True,
                "accepted": True,
                "capability": _capability_dict(cap),
                "detail": "background prepare started",
            }
        if action == "prepare_status":
            # Status is also scheduler-loop local; pair aggregation belongs to
            # the HTTP proxy, which can poll both component loops independently.
            local = self.adapter.prepare_status(payload)
            component = "ffn" if self._is_ffn() else "attn"
            breakdown = {
                component: {
                    "critical": dict(local.get("timings_s", {})),
                    "ranks": list(local.get("rank_timings_s", [])),
                }
            }
            state = local["state"]
            if state == "ERROR":
                return {
                    "ok": False,
                    "state": "ERROR",
                    "detail": str(local.get("error") or "component prepare failed"),
                    "breakdown": breakdown,
                }
            return {"ok": True, "state": state, "breakdown": breakdown,
                    "local": dict(local.get("timings_s", {}))}
        if action == "quiesce":
            component = "ffn" if self._is_ffn() else "attn"
            rank = int(getattr(self.scheduler, "tp_rank", 0))
            newly_fenced = (
                not self.admission_blocked or self.operation_id != operation_id
            )
            if newly_fenced:
                # This handler runs on rank 0's scheduler event-loop thread.  It
                # may only publish local state; active-rank propagation and the
                # max-world rendezvous are advanced by the common safe point.
                self.admission_blocked = True
                self.operation_id = operation_id
                self.scheduler.afd_component_request_quiesce(payload)
                logger.info(
                    "[AFD-reshard] op=%s stage=quiesce comp=%s rank=%d "
                    "fence requested locally; returning retryable",
                    operation_id, component, rank,
                )
                return {
                    "ok": False, "retryable": True,
                    "detail": f"{component} fence requested; active consensus pending",
                    "operation_id": operation_id, "epoch": epoch,
                }

            world_done = bool(
                getattr(self.scheduler, "_afd_component_quiesce_world_done", False)
                and getattr(
                    self.scheduler, "_afd_component_quiesce_world_operation", None
                ) == operation_id
            )
            if not world_done:
                reason = self._afd_quiescent_reason()
                return {
                    "ok": False, "retryable": True,
                    "detail": (
                        f"{component} active/world quiesce pending: "
                        f"{reason or 'active ready consensus pending'}"
                    ),
                    "operation_id": operation_id, "epoch": epoch,
                }

            # The max-world quiesce generation is complete exactly once.  Only
            # now is it safe to finalize local communication and ask the peer;
            # the peer's first call merely requests its own fence and returns.
            self._strict_comm_drain()
            drain = self.scheduler.afd_reshard_drain_state()
            watermark = int(payload.get("watermark", 0)) if self._is_ffn() else drain.cut()
            if self._is_ffn():
                # Attention owns the cut. FFN only installs the watermark carried
                # through the pair request after its max-world quiesce completed.
                drain.install_external_cut(watermark)
                drain.ack("attn", watermark)
                drain.ack("ffn", watermark)
            else:
                drain.ack("attn", watermark)
            if self.peer_client is not None:
                peer_payload = {**payload, "watermark": watermark}
                peer = self._poll_quiesce_peer(peer_payload)
                if peer is None:
                    return {
                        "ok": False, "retryable": True,
                        "detail": "FFN quiesce status pending",
                        "operation_id": operation_id, "epoch": epoch,
                    }
                if not peer.get("ok"):
                    if peer.get("retryable"):
                        return {
                            "ok": False, "retryable": True,
                            "detail": f"FFN quiesce pending: {peer.get('detail', '')}",
                            "operation_id": operation_id, "epoch": epoch,
                        }
                    return {"ok": False, "detail": f"invalid FFN quiesce ACK: {peer}"}
                if (
                    peer.get("operation_id") != operation_id
                    or int(peer.get("epoch", -1)) != epoch
                    or int(peer.get("watermark", -1)) != watermark
                ):
                    return {"ok": False, "detail": f"invalid FFN quiesce ACK: {peer}"}
                drain.ack("ffn", watermark)
            acknowledged = self._is_ffn() or drain.drained
            return {
                "ok": acknowledged,
                "retryable": not acknowledged,
                "detail": "pair did not strictly drain" if not acknowledged else "",
                "operation_id": operation_id, "epoch": epoch,
                "watermark": watermark,
            }
        if action == "activate":
            component = "ffn" if self._is_ffn() else "attn"
            rank = int(getattr(self.scheduler, "tp_rank", 0))
            with self._activation_state_lock:
                already_activated = (
                    self._activated_operation == operation_id
                    and self._activated_epoch == epoch
                )
                prior_fanout_unresolved = (
                    not already_activated and self._activation_fanout_started
                )
                if not already_activated and not prior_fanout_unresolved:
                    # Publish ownership before starting either destructive world
                    # transition. A concurrent readiness read cannot resolve this
                    # operation until its identity is published after fanout.
                    self._activation_fanout_started = True
                    self._activation_fanout_operation = operation_id
                    self._activation_fanout_epoch = epoch
            if prior_fanout_unresolved:
                return {
                    "ok": False,
                    "detail": (
                        "a different ACTIVATE operation arrived while the prior "
                        "activation fanout state was unresolved"
                    ),
                    "operation_id": operation_id,
                    "epoch": epoch,
                }
            if already_activated:
                if not bool(
                    getattr(self.scheduler, "_afd_component_serving_ready", True)
                ):
                    return {
                        "ok": False,
                        "retryable": True,
                        "detail": "post-activate serving readiness pending",
                        "operation_id": operation_id,
                        "epoch": epoch,
                    }
                # This handler runs in the scheduler event loop. Never query
                # the other component here: its rank zero may already be inside
                # a serving TP collective while this rank zero is handling the
                # control command. Cross-component readiness is joined by the
                # HTTP proxy, outside both scheduler event loops.
                with self._activation_state_lock:
                    if (
                        self._activated_operation == operation_id
                        and self._activated_epoch == epoch
                        and self._activation_fanout_operation == operation_id
                        and self._activation_fanout_epoch == epoch
                    ):
                        self._activation_fanout_started = False
                return {
                    "ok": True,
                    "operation_id": operation_id,
                    "epoch": epoch,
                    "breakdown": {
                        component: self.adapter.completed_breakdown(operation_id)
                    },
                }

            # Ownership was set under _activation_state_lock before this
            # destructive world transition. If fanout partially fails, keep it
            # set so a different operation fails closed.
            peer_result: Dict[str, Any] = {}
            peer_thread = None
            # A/F are colocated on the same physical GPUs. Rebuilding Attention
            # first and only then asking FFN to rebuild serializes two destructive
            # process-group transitions and can strand Decode FFN in old AFD/serving
            # state while Decode Attention waits at its final max-world barrier.
            # Start both component worlds concurrently, as PREPARE already does.
            if self.peer_client is not None:
                def activate_peer() -> None:
                    try:
                        peer_result["response"] = self.peer_client.request(
                            payload, self._timeout(payload)
                        )
                    except Exception as exc:
                        peer_result["error"] = exc

                peer_thread = threading.Thread(
                    target=activate_peer,
                    name="afd-ffn-activate",
                    daemon=True,
                )
                peer_thread.start()
            logger.info(
                "[AFD-reshard] op=%s stage=activate comp=%s rank=%d "
                "world fanout begin peer_concurrent=%s",
                operation_id, component, rank, peer_thread is not None,
            )
            try:
                self._fanout(payload, "activate")
                logger.info(
                    "[AFD-reshard] op=%s stage=activate comp=%s rank=%d "
                    "world fanout complete",
                    operation_id, component, rank,
                )
                if peer_thread is not None:
                    peer_thread.join(self._timeout(payload))
                    if peer_thread.is_alive():
                        raise TimeoutError(
                            "FFN activate did not finish after configured timeout"
                        )
                    if "error" in peer_result:
                        raise peer_result["error"]
                    peer = peer_result.get("response", {})
                    if not peer.get("ok") and not peer.get("retryable"):
                        return {
                            "ok": False,
                            "detail": (
                                "FFN activate rejected: "
                                f"{peer.get('detail', '')}"
                            ),
                        }
            except Exception:
                logger.exception(
                    "[AFD-reshard] op=%s stage=activate comp=%s rank=%d failed",
                    operation_id, component, rank,
                )
                raise
            # Activation is the last operation that both scheduler worlds can
            # safely coordinate. Release each rank-zero runtime's local fence
            # only after its world fanout (and, on ATTN, the FFN peer activate)
            # succeeded, before either side resumes its serving broadcast.
            self.scheduler.afd_reshard_drain_state().reopen_after_transition()
            self.scheduler.afd_component_clear_quiesce()
            self.admission_blocked = False
            self.operation_id = None
            self._reset_quiesce_peer_state()
            with self._activation_state_lock:
                self._activated_operation = operation_id
                self._activated_epoch = epoch
            # The max-world ACTIVATE is complete, but the scheduler must return
            # to its active loop and finish post_activate_ready before the HTTP
            # coordinator may publish this epoch as succeeded. Returning a
            # retryable receipt makes that lifecycle boundary explicit.
            if not bool(
                getattr(self.scheduler, "_afd_component_serving_ready", True)
            ) or (
                self.peer_client is not None
                and not peer_result.get("response", {}).get("ok", False)
            ):
                breakdown = {
                    component: self.adapter.completed_breakdown(operation_id)
                }
                peer_breakdown = peer_result.get("response", {}).get(
                    "breakdown", {}
                )
                if isinstance(peer_breakdown, dict):
                    breakdown.update(peer_breakdown)
                return {
                    "ok": False,
                    "retryable": True,
                    "detail": "post-activate serving readiness pending",
                    "operation_id": operation_id,
                    "epoch": epoch,
                    "breakdown": breakdown,
                }
            with self._activation_state_lock:
                if (
                    self._activated_operation == operation_id
                    and self._activated_epoch == epoch
                    and self._activation_fanout_operation == operation_id
                    and self._activation_fanout_epoch == epoch
                ):
                    self._activation_fanout_started = False
            breakdown = {
                component: self.adapter.completed_breakdown(operation_id)
            }
            peer_breakdown = peer_result.get("response", {}).get("breakdown", {})
            if isinstance(peer_breakdown, dict):
                breakdown.update(peer_breakdown)
            return {
                "ok": True,
                "operation_id": operation_id,
                "epoch": epoch,
                "breakdown": breakdown,
            }
        if action == "abort":
            self._abort_everywhere(payload)
            self.scheduler.afd_reshard_drain_state().force_reopen_after_abort()
            self.scheduler.afd_component_clear_quiesce()
            self.admission_blocked = False
            self.operation_id = None
            self._reset_quiesce_peer_state()
            return {"ok": True}
        if action == "retire":
            # ACTIVATE already retired committed shadows and released both local
            # rank-zero fences before serving resumes. External RETIRE is therefore
            # a rank-zero-local, idempotent cleanup only: peer RPC or world fanout
            # can deadlock once followers are back in the serving broadcast.
            self.adapter.retire(payload)
            self.admission_blocked = False
            self.operation_id = None
            return {"ok": True}
        raise ValueError(f"unknown AFD component action: {action!r}")


    def _reset_quiesce_peer_state(self) -> None:
        self._quiesce_peer_attempt = None
        self._quiesce_peer_terminal = None

    def _poll_quiesce_peer(self, payload: Dict[str, Any]):
        """Poll one immutable peer attempt without blocking the scheduler.

        A completed terminal ACK remains cached for the operation. A retryable
        ACK is returned once, then the next scheduler poll starts a fresh status
        request. The worker owns its attempt dictionary, so starting a later
        attempt cannot redirect a late result into newly allocated state.
        """
        operation = str(payload["operation_id"])
        terminal = self._quiesce_peer_terminal
        if terminal is not None:
            if terminal["operation_id"] == operation:
                return terminal["response"]
            self._quiesce_peer_terminal = None

        attempt = self._quiesce_peer_attempt
        if attempt is not None:
            if attempt["operation_id"] != operation:
                if not attempt["done"].is_set():
                    return {
                        "ok": False,
                        "retryable": True,
                        "detail": "previous FFN quiesce request still pending",
                    }
                self._quiesce_peer_attempt = None
            elif not attempt["done"].is_set():
                return None
            else:
                self._quiesce_peer_attempt = None
                if "error" in attempt:
                    return {
                        "ok": False,
                        "retryable": True,
                        "detail": str(attempt["error"]),
                    }
                response = dict(attempt.get("response", {}))
                if response.get("ok") or not response.get("retryable", False):
                    self._quiesce_peer_terminal = {
                        "operation_id": operation,
                        "response": response,
                    }
                return response

        attempt = {
            "operation_id": operation,
            "done": threading.Event(),
        }
        self._quiesce_peer_attempt = attempt

        def request_peer(current_attempt):
            try:
                current_attempt["response"] = self.peer_client.request(
                    payload, self._timeout(payload)
                )
            except Exception as exc:
                current_attempt["error"] = exc
            finally:
                current_attempt["done"].set()

        threading.Thread(
            target=request_peer,
            args=(attempt,),
            name="afd-ffn-quiesce-status",
            daemon=True,
        ).start()
        return None

    def _timeout(self, payload: Dict[str, Any]) -> float:
        return max(float(payload.get("timeout", self.command_timeout)), 0.001)

    def _abort_after_fanout_failure(self, payload: Dict[str, Any]) -> None:
        """Clean this component after failed PREPARE without peer RPC."""
        abort_payload = {**payload, "action": "abort"}
        try:
            self._fanout(abort_payload, "abort")
        except Exception:
            pass
        finally:
            self._cancel_local_prepare("component prepare fanout failed")

    def _cancel_local_prepare(self, reason: str) -> None:
        stager = getattr(
            self.scheduler.tp_worker.model_runner, "_afd_component_stager", None
        )
        if stager is None:
            return
        cancel = getattr(stager, "cancel_prepare", None)
        if cancel is not None:
            cancel(reason)
        else:
            stager.abort(reason)

    def _abort_everywhere(self, payload: Dict[str, Any]) -> None:
        """Best-effort peer/world abort with unconditional local shadow cleanup."""
        peer_abort = None
        if self.peer_client is not None:
            peer_abort = threading.Thread(
                target=self._best_effort_peer_abort,
                args=(payload, self._timeout(payload)),
                name="afd-ffn-abort", daemon=True,
            )
            peer_abort.start()
        try:
            self._fanout(payload, "abort")
        except Exception:
            pass
        finally:
            # A timed-out/failed process group may reject the abort fanout. The
            # rank-zero shadow still must be released; follower safe-point error
            # handlers provide the same unconditional local cleanup per rank.
            self._cancel_local_prepare("component operation aborted")
        if peer_abort is not None:
            peer_abort.join(self._timeout(payload) + 10.0)

    def _best_effort_peer_abort(
        self, payload: Dict[str, Any], timeout: float
    ) -> None:
        try:
            self.peer_client.request(payload, timeout)
        except Exception:
            pass

    def _fanout(self, payload: Dict[str, Any], action: str) -> None:
        component = "ffn" if self._is_ffn() else "attn"
        target_key = "target_ffn_tp" if self._is_ffn() else "target_attn_tp"
        world_action = action
        if action == "quiesce":
            world_action = "quiesce"
        world_payload = {
            **payload,
            "operation": str(payload["operation_id"]),
            "component": component,
            "action": world_action,
            "target_tp": int(payload[target_key]),
        }
        self.scheduler.afd_component_fanout_command(world_payload)

    def _is_ffn(self) -> bool:
        return str(getattr(self.scheduler.server_args.afd_perspective, "value", self.scheduler.server_args.afd_perspective)).lower().endswith("ffn")

    def _strict_comm_drain(self) -> None:
        from sglang.srt.layers.afd import peek_async_communicator

        comm = peek_async_communicator()
        if comm is not None:
            comm.drain_sends()
            pending_recvs = int(getattr(comm, "_pending_recv_count", 0))
            if pending_recvs:
                raise RuntimeError(
                    f"AFD communicator still has {pending_recvs} pending receives"
                )
        self.scheduler.afd_reshard_drain_state().set_pending(
            sends=0, recvs=0, fences=0
        )

    def _afd_pending_recv_count(self) -> int:
        """Best-effort count of in-flight AFD receives (0 when no communicator).

        Overridable for CPU control-path tests where no CUDA communicator is
        constructed. In production this reflects the async communicator ring.
        """
        try:
            from sglang.srt.layers.afd import peek_async_communicator

            comm = peek_async_communicator()
            return int(getattr(comm, "_pending_recv_count", 0))
        except Exception:
            return 0

    def _afd_quiescent_reason(self) -> Optional[str]:
        """Return None when this side is AFD-quiescent, else a retry reason.

        Being AFD-quiescent is stricter and more accurate than the generic
        ``is_fully_idle()`` for the operator-disaggregated event loop:

        * admission must already be fenced (no new dispatches can appear), and
        * no AFD batch info is mid-flight (``_afd_batchsize_attn is None``), and
        * the pending AFD batch-info queue is empty, and
        * no AFD receives are still in flight, and
        * ``is_fully_idle()`` holds as an additional protection (it is no longer
          perpetually False on the FFN side once the idle wait branch resets the
          batch ledger).

        Any unsatisfied condition is transient during pair drain, so the caller
        returns a retryable ACK and re-polls with backoff.
        """
        if not self.admission_blocked:
            return "quiesce operation is not requested"
        if not getattr(self.scheduler, "_afd_component_fence_installed", False):
            return "fence sync pending"
        reason_fn = getattr(
            self.scheduler, "afd_component_local_quiescent_reason", None
        )
        if reason_fn is not None:
            return reason_fn()
        # Lightweight CPU tests may not mix in SchedulerAFDMixin.
        sched = self.scheduler
        if getattr(sched, "_afd_batchsize_attn", None) is not None:
            return "AFD batch info is still being processed"
        pending = getattr(sched, "_afd_pending_batch_infos", None)
        if pending:
            return f"{len(pending)} AFD batch info(s) still pending"
        pending_recvs = self._afd_pending_recv_count()
        if pending_recvs:
            return f"{pending_recvs} AFD receive(s) still in flight"
        if not sched.is_fully_idle():
            return "scheduler is not fully idle yet"
        return None


class AFDComponentSchedulerProxyAdapter:
    """HTTP-side adapter; all actions execute in scheduler event loops."""

    def __init__(
        self,
        attn_client: ZMQSchedulerControlClient,
        timeout: float = 600.0,
        peer_client: Optional[ZMQSchedulerControlClient] = None,
    ) -> None:
        self.client = attn_client
        # The HTTP worker owns this FFN client. It must never be passed into or
        # called from the ATTN scheduler event loop.
        self.peer_client = peer_client
        self.timeout = timeout

    def _response_receipt(self, response: Dict[str, Any]) -> AdapterReceipt:
        return AdapterReceipt(
            bool(response.get("ok")), str(response.get("detail", "")),
            dict(response.get("breakdown", {})),
            bool(response.get("retryable", False)),
        )

    def _call(self, action, request, **extra):
        payload = {**request.__dict__, "action": action, "epoch": request.expected_epoch, **extra}
        response = self.client.request(payload, self.timeout)
        return self._response_receipt(response)

    def prepare(self, request, transitions):
        import time

        started = time.perf_counter()
        deadline = time.monotonic() + self.timeout
        interval = max(
            float(os.getenv("AFD_RESHARD_PREPARE_POLL_INTERVAL", "0.05")),
            0.001,
        )
        clients = [("attn", self.client)]
        if self.peer_client is not None:
            clients.append(("ffn", self.peer_client))
        accepted = {component: False for component, _ in clients}
        last_detail = {component: "not attempted" for component, _ in clients}
        prepare_payload = {
            **request.__dict__,
            "action": "prepare",
            "epoch": request.expected_epoch,
        }

        # Both initial PREPARE commands must be in flight at the same time. Each
        # component's max-world staging may rendezvous with the other component;
        # a serial HTTP call can therefore deadlock before the second command is
        # sent. Retryable readiness is tracked independently thereafter.
        while not all(accepted.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._best_effort_abort(request)
                pending = ", ".join(
                    f"{component}: {last_detail[component]}"
                    for component in accepted if not accepted[component]
                )
                return AdapterReceipt(
                    False,
                    f"prepare readiness did not finish within {self.timeout:.1f}s "
                    f"({pending})",
                )
            pending_clients = [
                (component, client)
                for component, client in clients
                if not accepted[component]
            ]
            responses = self._parallel_component_requests(
                pending_clients, prepare_payload, remaining
            )
            for component, _ in pending_clients:
                result = responses[component]
                if isinstance(result, Exception):
                    self._best_effort_abort(request)
                    return AdapterReceipt(
                        False, f"{component} prepare request failed: {result}"
                    )
                receipt = self._response_receipt(result)
                last_detail[component] = receipt.detail
                if receipt.completed:
                    accepted[component] = True
                elif not receipt.retryable:
                    self._best_effort_abort(request)
                    return AdapterReceipt(
                        False,
                        f"{component} prepare rejected: {receipt.detail}",
                        receipt.breakdown,
                    )
            if not all(accepted.values()):
                time.sleep(min(interval, max(deadline - time.monotonic(), 0.0)))

        status_payload = {
            **request.__dict__,
            "action": "prepare_status",
            "epoch": request.expected_epoch,
        }
        ready = {component: False for component, _ in clients}
        breakdown = {}
        while not all(ready.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._best_effort_abort(request)
                pending = ", ".join(
                    component for component in ready if not ready[component]
                )
                return AdapterReceipt(
                    False,
                    f"prepare did not finish within {self.timeout:.1f}s "
                    f"(pending: {pending})",
                )
            pending_clients = [
                (component, client)
                for component, client in clients
                if not ready[component]
            ]
            responses = self._parallel_component_requests(
                pending_clients, status_payload, remaining
            )
            for component, _ in pending_clients:
                result = responses[component]
                if isinstance(result, Exception):
                    self._best_effort_abort(request)
                    return AdapterReceipt(
                        False, f"{component} prepare_status failed: {result}"
                    )
                if not result.get("ok"):
                    self._best_effort_abort(request)
                    receipt = self._response_receipt(result)
                    return AdapterReceipt(
                        False,
                        f"{component} prepare failed: {receipt.detail}",
                        receipt.breakdown,
                        receipt.retryable,
                    )
                component_breakdown = result.get("breakdown", {}).get(component)
                if component_breakdown is None:
                    # Compatibility with older/local-only endpoints whose sole
                    # component was historically labelled ``attn``.
                    component_breakdown = result.get("breakdown", {}).get(
                        "attn",
                        {
                            "critical": dict(result.get("local", {})),
                            "ranks": list(result.get("rank_timings_s", [])),
                        },
                    )
                breakdown[component] = dict(component_breakdown)
                if result.get("state") == "READY":
                    ready[component] = True
            if not all(ready.values()):
                time.sleep(min(interval, max(deadline - time.monotonic(), 0.0)))

        breakdown["prepare_s"] = time.perf_counter() - started
        return AdapterReceipt(True, "background prepare ready", breakdown)

    @staticmethod
    def _parallel_component_requests(clients, payload, timeout):
        """Start all component RPCs before waiting for any one response."""
        results = {}
        done = {component: threading.Event() for component, _ in clients}

        def invoke(component, client):
            try:
                results[component] = client.request(dict(payload), timeout)
            except Exception as exc:
                results[component] = exc
            finally:
                done[component].set()

        for component, client in clients:
            threading.Thread(
                target=invoke,
                args=(component, client),
                name=f"afd-{component}-{payload.get('action', 'control')}",
                daemon=True,
            ).start()
        deadline = __import__("time").monotonic() + timeout
        for component, _ in clients:
            remaining = deadline - __import__("time").monotonic()
            if remaining <= 0 or not done[component].wait(remaining):
                results.setdefault(
                    component,
                    TimeoutError(
                        f"{component} {payload.get('action')} did not finish "
                        f"within {timeout:.3f}s"
                    ),
                )
        return results

    def _best_effort_abort(self, request) -> None:
        abort_timeout = max(
            float(os.getenv("AFD_RESHARD_ABORT_BEST_EFFORT_TIMEOUT", "1")),
            0.001,
        )
        payload = {
            **request.__dict__,
            "action": "abort",
            "epoch": request.expected_epoch,
            "_transport_grace": 0.0,
        }
        clients = [("attn", self.client)]
        if self.peer_client is not None:
            clients.append(("ffn", self.peer_client))
        # The bounded helper uses daemon workers, so a wedged abort endpoint can
        # neither hold the HTTP request forever nor prevent aborting its peer.
        self._parallel_component_requests(clients, payload, abort_timeout)

    def drain(self, request, cut_watermark):
        # The pair may still be processing in-flight AFD work when quiesce first
        # arrives (F is running A's forward). Each quiesce call fences admission
        # and returns quickly; we poll with a bounded backoff until both sides
        # are strictly idle or the configured reshard timeout elapses. Sleeping
        # between attempts releases the scheduler event-loop threads so they can
        # drain, avoiding an A/F deadlock.
        import time

        payload = {
            **request.__dict__, "action": "quiesce",
            "epoch": request.expected_epoch, "watermark": cut_watermark,
        }
        operation_id = getattr(request, "operation_id", "")
        deadline = time.monotonic() + self.timeout
        poll_interval = max(
            float(os.getenv("AFD_RESHARD_QUIESCE_POLL_INTERVAL", "0.05")), 0.001
        )
        # Bound how often we emit per-attempt logs so a fast poll interval does
        # not flood the log; always log the first attempt and completion.
        log_every = max(int(os.getenv("AFD_RESHARD_QUIESCE_LOG_EVERY", "20")), 1)
        last = AdapterReceipt(False, "quiesce not attempted", retryable=True)
        attempts = 0
        logger.info(
            "[AFD-reshard] op=%s stage=drain comp=proxy sending quiesce "
            "(cut_watermark=%d timeout=%.1fs poll=%.3fs)",
            operation_id, cut_watermark, self.timeout, poll_interval,
        )
        while True:
            attempts += 1
            response = self.client.request(payload, self.timeout)
            last = self._response_receipt(response)
            if attempts == 1 or attempts % log_every == 0:
                logger.info(
                    "[AFD-reshard] op=%s stage=drain comp=proxy attempt=%d "
                    "ok=%s retryable=%s detail=%s",
                    operation_id, attempts, last.completed, last.retryable,
                    last.detail,
                )
            if last.completed or not last.retryable:
                logger.info(
                    "[AFD-reshard] op=%s stage=drain comp=proxy finished after "
                    "%d attempt(s): ok=%s detail=%s",
                    operation_id, attempts, last.completed, last.detail,
                )
                return last
            if time.monotonic() >= deadline:
                logger.warning(
                    "[AFD-reshard] op=%s stage=drain comp=proxy TIMEOUT after "
                    "%d attempts (%.1fs): %s",
                    operation_id, attempts, self.timeout, last.detail,
                )
                return AdapterReceipt(
                    False,
                    f"pair did not drain within {self.timeout:.1f}s "
                    f"({attempts} attempts): {last.detail}",
                )
            time.sleep(min(poll_interval, max(deadline - time.monotonic(), 0.0)))

    def activate(self, request, transitions):
        import time

        deadline = time.monotonic() + self.timeout
        interval = max(
            float(os.getenv("AFD_RESHARD_ACTIVATE_POLL_INTERVAL", "0.005")),
            0.001,
        )
        retry_timeout = max(
            float(os.getenv("AFD_RESHARD_ACTIVATE_RETRY_TIMEOUT", "5")),
            0.001,
        )
        operation_id = getattr(request, "operation_id", "")

        # The first request is the sole owner of the real concurrent A/F fanout
        # and therefore retains the full collective timeout. Only after it
        # returns retryable do the HTTP-owned clients poll each scheduler's
        # idempotent, local-only readiness state.
        receipt = self._call("activate", request)
        activation_breakdown = dict(receipt.breakdown)
        logger.info(
            "[AFD-reshard] op=%s stage=activate comp=proxy initial "
            "ok=%s retryable=%s detail=%s",
            operation_id, receipt.completed, receipt.retryable, receipt.detail,
        )
        if receipt.completed or not receipt.retryable:
            return receipt

        payload = {
            **request.__dict__,
            "action": "serving_readiness",
            "epoch": request.expected_epoch,
            "_transport_grace": 0.0,
            "_readiness_retry": True,
        }
        clients = [("attn", self.client)]
        if self.peer_client is not None:
            clients.append(("ffn", self.peer_client))
        ready = {component: False for component, _ in clients}
        attempts = {component: 0 for component, _ in clients}
        last = {component: receipt for component, _ in clients}

        while not all(ready.values()):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                pending = ", ".join(
                    f"{component}: {last[component].detail}"
                    for component in ready if not ready[component]
                )
                return AdapterReceipt(
                    False,
                    f"post-activate serving readiness did not finish within "
                    f"{self.timeout:.1f}s ({pending})",
                )
            time.sleep(min(interval, remaining))
            for component, client in clients:
                if ready[component]:
                    continue
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                attempts[component] += 1
                try:
                    response = client.request(
                        payload, min(retry_timeout, remaining)
                    )
                    current = self._response_receipt(response)
                except TimeoutError as exc:
                    current = AdapterReceipt(False, str(exc), retryable=True)
                last[component] = current
                if current.breakdown:
                    activation_breakdown.update(current.breakdown)
                attempt = attempts[component]
                if attempt == 1 or attempt % 100 == 0 or current.completed:
                    logger.info(
                        "[AFD-reshard] op=%s stage=activate comp=proxy-%s "
                        "readiness_attempt=%d ok=%s retryable=%s detail=%s",
                        operation_id, component, attempt, current.completed,
                        current.retryable, current.detail,
                    )
                if current.completed:
                    ready[component] = True
                elif not current.retryable:
                    return current

        logger.info(
            "[AFD-reshard] op=%s stage=activate comp=proxy readiness complete "
            "attn_attempts=%d ffn_attempts=%d",
            operation_id,
            attempts.get("attn", 0),
            attempts.get("ffn", 0),
        )
        return AdapterReceipt(
            True, "post-activate serving readiness complete", activation_breakdown
        )

    def retire(self, request, transitions):
        # ACTIVATE's final barrier is the retirement boundary: every rank has
        # already discarded staged state and both scheduler runtimes have
        # released their admission fences before rank zero replies. Rank zero
        # then resumes serving TP broadcasts and can no longer poll a second
        # control command, so RETIRE must complete in the HTTP coordinator.
        return AdapterReceipt(
            True,
            "retirement completed by ACTIVATE; no scheduler command required",
        )
