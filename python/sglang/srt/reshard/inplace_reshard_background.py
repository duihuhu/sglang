"""Background prepare + fast commit for experimental in-place TP reshard.

While the server is still serving at ``old_tp``, joining standby ranks can load a
dummy model shell, receive weight shards, allocate KV, and warm up attention
backends.  The pause window should then only rebuild process groups and flip
active ranks (plus rank0 KV teardown / narrow, which needs the serving drain).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def background_prep_enabled() -> bool:
    return os.environ.get("SGLANG_INPLACE_RESHARD_BACKGROUND_PREP", "1") == "1"


def runtime_prep_enabled() -> bool:
    """Pre-allocate KV and warm attention on standby joining ranks during prep."""
    return os.environ.get("SGLANG_INPLACE_RESHARD_RUNTIME_PREP", "1") == "1"


@dataclass
class InplaceReshardPrepState:
    old_tp: int = 0
    new_tp: int = 0
    weights_ready: bool = False
    runtime_ready: bool = False
    joiners_prepared: bool = False
    rank0_exported_ipc: bool = False
    timings_s: Dict[str, float] = field(default_factory=dict)
    kv_cfg_hint: Optional[Any] = None
    started_at_s: Optional[float] = None
    done_at_s: Optional[float] = None

    @property
    def ready(self) -> bool:
        return self.weights_ready and self.runtime_ready

    def reset(self) -> None:
        self.old_tp = 0
        self.new_tp = 0
        self.weights_ready = False
        self.runtime_ready = False
        self.joiners_prepared = False
        self.rank0_exported_ipc = False
        self.timings_s.clear()
        self.kv_cfg_hint = None
        self.started_at_s = None
        self.done_at_s = None


@dataclass
class InplaceReshardTimingContext:
    """Rank-0 wall-clock context preserved across deferred scheduler callbacks."""

    operation_id: str
    old_tp: int
    new_tp: int
    accepted_at_s: float
    scheduler_pickup_at_s: float
    admission_block_at_s: float
    drained_at_s: Optional[float] = None
    prep_start_at_s: Optional[float] = None
    prep_done_at_s: Optional[float] = None
    execute_queued_at_s: Optional[float] = None
    execute_start_at_s: Optional[float] = None
    done_at_s: Optional[float] = None
    scheduler_ms: Dict[str, float] = field(default_factory=dict)

    def mark_drained(self, now: Optional[float] = None) -> None:
        if self.drained_at_s is None:
            self.drained_at_s = now if now is not None else time.time()

    def snapshot(self, model_runner: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        done = self.done_at_s or time.time()
        drained = self.drained_at_s or self.admission_block_at_s
        execute = self.execute_start_at_s or done
        prep_start = self.prep_start_at_s
        prep_done = self.prep_done_at_s
        ready = max(drained, prep_done or drained)
        prep_wall = max(0.0, (prep_done - prep_start) * 1000.0) if prep_start is not None and prep_done is not None else 0.0
        prep_critical = 0.0
        if prep_start is not None and prep_done is not None:
            prep_critical = max(0.0, min(prep_done, execute) - max(prep_start, self.admission_block_at_s)) * 1000.0
        out = {
            "schema_version": 1,
            "clock": "rank0_wall",
            "operation_id": self.operation_id,
            "old_tp": int(self.old_tp),
            "new_tp": int(self.new_tp),
            "scheduler_pickup_to_admission_block_ms": max(0.0, self.admission_block_at_s - self.scheduler_pickup_at_s) * 1000.0,
            "http_accept_to_scheduler_pickup_ms": max(0.0, self.scheduler_pickup_at_s - self.accepted_at_s) * 1000.0,
            "wait_for_drain_ms": max(0.0, drained - self.admission_block_at_s) * 1000.0,
            "background_prep_ms": prep_wall,
            "background_prep_critical_wait_ms": prep_critical,
            "background_prep_exclusive_critical_ms": (
                max(0.0, (prep_done or drained) - drained) * 1000.0
            ),
            "execute_queue_wait_ms": max(0.0, execute - ready) * 1000.0,
            "synchronized_execute_wait_ms": max(0.0, execute - self.execute_queued_at_s) * 1000.0 if self.execute_queued_at_s is not None else 0.0,
            "request_to_done_ms": max(0.0, done - self.accepted_at_s) * 1000.0,
            "critical_path_ms": max(0.0, done - self.admission_block_at_s) * 1000.0,
            "scheduler": dict(self.scheduler_ms),
            "model_runner": dict(model_runner or {}),
        }
        out["total_ms"] = out["request_to_done_ms"]
        return out


def ensure_timing_context(
    existing: Optional[InplaceReshardTimingContext],
    *,
    operation_id: Optional[str],
    accepted_at_s: Optional[float],
    old_tp: int,
    new_tp: int,
    now_s: Optional[float] = None,
) -> InplaceReshardTimingContext:
    """Return the current operation context without replacing deferred state."""
    now = time.time() if now_s is None else float(now_s)
    requested_id = str(operation_id) if operation_id else None
    same_operation = (
        existing is not None
        and int(existing.old_tp) == int(old_tp)
        and int(existing.new_tp) == int(new_tp)
        and (requested_id is None or existing.operation_id == requested_id)
    )
    if same_operation:
        return existing
    accepted = float(accepted_at_s) if accepted_at_s is not None else now
    resolved_id = requested_id or f"reshard-{int(accepted * 1_000_000)}"
    return InplaceReshardTimingContext(
        operation_id=resolved_id,
        old_tp=int(old_tp),
        new_tp=int(new_tp),
        accepted_at_s=accepted,
        scheduler_pickup_at_s=now,
        admission_block_at_s=now,
    )


def status_operation_transition(
    prev: Dict[str, Any], phase: str, operation_id: Optional[str], now_s: float
) -> tuple[int, Optional[float], bool]:
    """Compute generation/start identity once per logical reshard operation."""
    lifecycle = {
        "pre_draining", "preparing", "prepared", "draining", "executing",
        "gang_restart_ready", "restarting", "done", "failed",
    }
    prev_id = prev.get("operation_id")
    changed = bool(
        operation_id is not None
        and phase in lifecycle
        and operation_id != prev_id
    )
    generation = int(prev.get("generation") or 0)
    started_at = prev.get("started_at")
    if changed:
        generation += 1
        started_at = float(now_s)
    elif phase in lifecycle and started_at is None:
        started_at = float(now_s)
    return generation, started_at, changed


def should_defer_async_kv_grow(
    *,
    control_file_pending: bool,
    pending_reshard: bool,
    execute_pending: bool,
    prep_pending: bool,
) -> bool:
    """Pure scheduling predicate: reshard work always outranks post-done KV grow."""
    return bool(
        control_file_pending
        or pending_reshard
        or execute_pending
        or prep_pending
    )


def prep_state_key(old_tp: int, new_tp: int) -> tuple[int, int]:
    return (int(old_tp), int(new_tp))
