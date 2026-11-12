# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""In-memory control plane for AFD component tensor-parallel resharding.

This module deliberately has no dependency on the native TP reshard protocol or
its filesystem control files. Runtime implementations attach through
``register_executor``; until then non-dry-run requests terminate as unsupported.
"""

from __future__ import annotations

import dataclasses
import threading
import time
import uuid
from collections import OrderedDict
from enum import Enum
from typing import Callable, Dict, Optional, Protocol

from sglang.srt.managers.io_struct import (
    AFDComponentReshardCancelInput,
    AFDComponentReshardReqInput,
    AFDComponentReshardStatus,
    AFDComponentReshardTopology,
)


class AFDComponentReshardError(RuntimeError):
    """Base error carrying an HTTP-compatible status code."""

    status_code = 400


class AFDComponentReshardDisabled(AFDComponentReshardError):
    status_code = 404


class AFDComponentReshardConflict(AFDComponentReshardError):
    status_code = 409


class AFDComponentReshardNotFound(AFDComponentReshardError):
    status_code = 404


class AFDComponentReshardPhase(str, Enum):
    VALIDATING = "validating"
    PREPARING = "preparing"
    PREPARED = "prepared"
    DRAINING = "draining"
    ACTIVATING = "activating"
    RETIRING = "retiring"
    COMMITTING = "committing"
    SUCCEEDED = "succeeded"
    UNSUPPORTED = "unsupported"
    CANCELLED = "cancelled"
    FAILED_PRE_COMMIT = "failed_pre_commit"
    FAILED_POST_COMMIT = "failed_post_commit"


_ALLOWED_TRANSITIONS = {
    AFDComponentReshardPhase.VALIDATING.value: {
        AFDComponentReshardPhase.PREPARING.value,
        AFDComponentReshardPhase.SUCCEEDED.value,
        AFDComponentReshardPhase.CANCELLED.value,
        AFDComponentReshardPhase.FAILED_PRE_COMMIT.value,
    },
    AFDComponentReshardPhase.PREPARING.value: {
        AFDComponentReshardPhase.PREPARED.value,
        AFDComponentReshardPhase.COMMITTING.value,
        AFDComponentReshardPhase.UNSUPPORTED.value,
        AFDComponentReshardPhase.CANCELLED.value,
        AFDComponentReshardPhase.FAILED_PRE_COMMIT.value,
    },
    AFDComponentReshardPhase.PREPARED.value: {
        AFDComponentReshardPhase.DRAINING.value,
        AFDComponentReshardPhase.COMMITTING.value,
        AFDComponentReshardPhase.CANCELLED.value,
        AFDComponentReshardPhase.FAILED_PRE_COMMIT.value,
    },
    AFDComponentReshardPhase.DRAINING.value: {
        AFDComponentReshardPhase.ACTIVATING.value,
        AFDComponentReshardPhase.COMMITTING.value,
        AFDComponentReshardPhase.CANCELLED.value,
        AFDComponentReshardPhase.FAILED_PRE_COMMIT.value,
    },
    AFDComponentReshardPhase.ACTIVATING.value: {
        AFDComponentReshardPhase.RETIRING.value,
        AFDComponentReshardPhase.COMMITTING.value,
        AFDComponentReshardPhase.FAILED_POST_COMMIT.value,
    },
    AFDComponentReshardPhase.RETIRING.value: {
        AFDComponentReshardPhase.COMMITTING.value,
        AFDComponentReshardPhase.SUCCEEDED.value,
        AFDComponentReshardPhase.FAILED_POST_COMMIT.value,
    },
    AFDComponentReshardPhase.COMMITTING.value: {
        AFDComponentReshardPhase.ACTIVATING.value,
        AFDComponentReshardPhase.RETIRING.value,
        AFDComponentReshardPhase.SUCCEEDED.value,
        AFDComponentReshardPhase.FAILED_POST_COMMIT.value,
    },
}


TERMINAL_PHASES = {
    AFDComponentReshardPhase.SUCCEEDED.value,
    AFDComponentReshardPhase.UNSUPPORTED.value,
    AFDComponentReshardPhase.CANCELLED.value,
    AFDComponentReshardPhase.FAILED_PRE_COMMIT.value,
    AFDComponentReshardPhase.FAILED_POST_COMMIT.value,
}


class AFDComponentReshardExecutor(Protocol):
    def __call__(
        self,
        request: AFDComponentReshardReqInput,
        transition: Callable[[str, Optional[str]], None],
        is_cancelled: Callable[[], bool],
    ) -> None: ...


class AFDComponentReshardCoordinator:
    """Thread-safe, per-stage AFD reshard operation registry."""

    def __init__(
        self,
        *,
        enabled: bool,
        max_tp: int,
        stage_id: Optional[str],
        pair_id: str,
        attn_tp: int,
        ffn_tp: int,
        channel_base: int,
        control_base: int,
        history_limit: int = 128,
    ) -> None:
        if stage_id not in (None, "prefill", "decode"):
            raise ValueError("stage_id must be 'prefill' or 'decode'")
        if max_tp < 1 or attn_tp < 1 or ffn_tp < 1:
            raise ValueError("TP sizes must be positive")
        if attn_tp > max_tp or ffn_tp > max_tp:
            raise ValueError("initial TP exceeds max_tp")
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.enabled = enabled
        self.max_tp = max_tp
        self.stage_id = stage_id
        self.pair_id = pair_id
        self.channel_base = channel_base
        self.control_base = control_base
        self._history_limit = history_limit
        self._lock = threading.RLock()
        managed_stages = (stage_id,) if stage_id is not None else ("prefill", "decode")
        self._stage_locks = {stage: threading.Lock() for stage in managed_stages}
        self._operations: "OrderedDict[str, AFDComponentReshardStatus]" = OrderedDict()
        self._fingerprints: Dict[str, tuple] = {}
        self._cancelled: set[str] = set()
        self._executor: Optional[AFDComponentReshardExecutor] = None
        self._topologies = {
            stage: AFDComponentReshardTopology(
                stage=stage,
                pair_id=pair_id,
                epoch=0,
                attn_tp=attn_tp,
                ffn_tp=ffn_tp,
                max_tp=max_tp,
                channel_base=channel_base,
                control_base=control_base,
            )
            for stage in managed_stages
        }

    @property
    def capability(self) -> str:
        with self._lock:
            return "runtime_executor" if self._executor is not None else "validation_only"

    def register_executor(self, executor: Optional[AFDComponentReshardExecutor]) -> None:
        with self._lock:
            self._executor = executor

    def _require_enabled(self) -> None:
        if not self.enabled:
            raise AFDComponentReshardDisabled("AFD component reshard is disabled")

    @staticmethod
    def _fingerprint(req: AFDComponentReshardReqInput) -> tuple:
        return (
            req.stage, req.expected_attn_tp, req.expected_ffn_tp,
            req.target_attn_tp, req.target_ffn_tp, req.expected_epoch, req.dry_run,
        )

    def _validate(self, req: AFDComponentReshardReqInput) -> None:
        if req.stage not in ("prefill", "decode"):
            raise ValueError("stage must be 'prefill' or 'decode'")
        if (
            not isinstance(req.expected_epoch, int)
            or isinstance(req.expected_epoch, bool)
            or req.expected_epoch < 0
        ):
            raise ValueError("expected_epoch must be a non-negative integer")
        if self.stage_id is not None and req.stage != self.stage_id:
            raise ValueError(f"this endpoint controls stage {self.stage_id!r}, not {req.stage!r}")
        values = (
            req.expected_attn_tp, req.expected_ffn_tp,
            req.target_attn_tp, req.target_ffn_tp,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) or value < 1 for value in values):
            raise ValueError("expected and target A/F TP values must be positive integers")
        if req.target_attn_tp > self.max_tp or req.target_ffn_tp > self.max_tp:
            raise ValueError(f"target TP exceeds configured max_tp={self.max_tp}")
        topo = self._topologies[req.stage]
        if req.expected_epoch != topo.epoch:
            raise AFDComponentReshardConflict(
                f"epoch conflict: expected {req.expected_epoch}, current {topo.epoch}"
            )
        if (req.expected_attn_tp, req.expected_ffn_tp) != (topo.attn_tp, topo.ffn_tp):
            raise AFDComponentReshardConflict(
                "topology conflict: expected A/F TP "
                f"{req.expected_attn_tp}/{req.expected_ffn_tp}, current "
                f"{topo.attn_tp}/{topo.ffn_tp}"
            )

    def submit(self, req: AFDComponentReshardReqInput) -> AFDComponentReshardStatus:
        self._require_enabled()
        operation_id = req.operation_id or uuid.uuid4().hex
        req.operation_id = operation_id
        fingerprint = self._fingerprint(req)
        with self._lock:
            existing = self._operations.get(operation_id)
            if existing is not None:
                if self._fingerprints[operation_id] != fingerprint:
                    raise AFDComponentReshardConflict(
                        f"operation_id {operation_id!r} was reused with a different request"
                    )
                return dataclasses.replace(existing)
            self._validate(req)
            now = time.time()
            executor = self._executor
            capability = (
                "runtime_executor" if executor is not None else "validation_only"
            )
            status = AFDComponentReshardStatus(
                operation_id=operation_id, stage=req.stage,
                phase=AFDComponentReshardPhase.VALIDATING.value,
                epoch=req.expected_epoch, expected_attn_tp=req.expected_attn_tp,
                expected_ffn_tp=req.expected_ffn_tp,
                target_attn_tp=req.target_attn_tp, target_ffn_tp=req.target_ffn_tp,
                dry_run=req.dry_run,
                capability=capability,
                runtime_supported=executor is not None,
                message="request validated", created_at=now, updated_at=now,
            )
            self._operations[operation_id] = status
            self._fingerprints[operation_id] = fingerprint
            self._trim_history_locked()
        if req.dry_run:
            self._transition(operation_id, AFDComponentReshardPhase.SUCCEEDED.value, "validation completed")
        else:
            thread = threading.Thread(
                target=self._run, args=(dataclasses.replace(req), executor),
                name=f"afd-reshard-{req.stage}-{operation_id}", daemon=True,
            )
            thread.start()
        return self.get_status(operation_id)

    def _trim_history_locked(
        self, preserve_operation_id: Optional[str] = None
    ) -> None:
        excess = len(self._operations) - self._history_limit
        if excess <= 0:
            return
        terminal_ids = [
            operation_id
            for operation_id, status in self._operations.items()
            if status.phase in TERMINAL_PHASES
            and operation_id != preserve_operation_id
        ]
        for operation_id in terminal_ids[:excess]:
            self._operations.pop(operation_id, None)
            self._fingerprints.pop(operation_id, None)
            self._cancelled.discard(operation_id)

    def _transition(
        self, operation_id: str, phase: str, message: Optional[str] = None,
        breakdown: Optional[Dict[str, float]] = None,
    ) -> None:
        with self._lock:
            status = self._operations[operation_id]
            if status.phase in TERMINAL_PHASES:
                return
            if phase not in _ALLOWED_TRANSITIONS.get(status.phase, set()):
                raise AFDComponentReshardConflict(
                    f"invalid phase transition {status.phase!r} -> {phase!r}"
                )
            status.phase = phase
            status.updated_at = time.time()
            if message is not None:
                status.message = message
            if breakdown:
                status.breakdown.update(breakdown)
            if phase in TERMINAL_PHASES:
                status.completed_at = status.updated_at
                self._trim_history_locked(preserve_operation_id=operation_id)

    def _run(
        self,
        req: AFDComponentReshardReqInput,
        executor: Optional[AFDComponentReshardExecutor],
    ) -> None:
        operation_id = req.operation_id
        stage_lock = self._stage_locks[req.stage]
        with stage_lock:
            if self._is_cancelled(operation_id):
                self._transition(operation_id, AFDComponentReshardPhase.CANCELLED.value, "cancelled before preparation")
                return
            try:
                with self._lock:
                    self._validate(req)
            except Exception as exc:
                self._transition(
                    operation_id,
                    AFDComponentReshardPhase.FAILED_PRE_COMMIT.value,
                    str(exc),
                )
                return
            self._transition(
                operation_id,
                AFDComponentReshardPhase.PREPARING.value,
                "runtime preparation started",
            )
            if executor is None:
                self._transition(
                    operation_id, AFDComponentReshardPhase.UNSUPPORTED.value,
                    "runtime executor is not registered; validation capability only",
                )
                return
            committed = False

            def transition(
                phase: str, message: Optional[str] = None,
                breakdown: Optional[Dict[str, float]] = None,
            ) -> None:
                nonlocal committed
                if phase in (
                    AFDComponentReshardPhase.COMMITTING.value,
                    AFDComponentReshardPhase.ACTIVATING.value,
                    AFDComponentReshardPhase.RETIRING.value,
                ):
                    committed = True
                self._transition(operation_id, phase, message, breakdown)

            try:
                executor(req, transition, lambda: self._is_cancelled(operation_id))
                with self._lock:
                    phase = self._operations[operation_id].phase
                if phase == AFDComponentReshardPhase.CANCELLED.value:
                    return
                if phase not in (
                    AFDComponentReshardPhase.COMMITTING.value,
                    AFDComponentReshardPhase.RETIRING.value,
                ):
                    raise RuntimeError(
                        "executor returned without entering a committed phase"
                    )
                with self._lock:
                    topo = self._topologies[req.stage]
                    topo.attn_tp = req.target_attn_tp
                    topo.ffn_tp = req.target_ffn_tp
                    topo.epoch += 1
                    topo.updated_at = time.time()
                    self._operations[operation_id].epoch = topo.epoch
                self._transition(operation_id, AFDComponentReshardPhase.SUCCEEDED.value, "runtime commit completed")
            except Exception as exc:
                with self._lock:
                    current_phase = self._operations[operation_id].phase
                if current_phase in TERMINAL_PHASES:
                    return
                phase = (
                    AFDComponentReshardPhase.FAILED_POST_COMMIT.value
                    if committed else AFDComponentReshardPhase.FAILED_PRE_COMMIT.value
                )
                self._transition(operation_id, phase, str(exc))

    def _is_cancelled(self, operation_id: str) -> bool:
        with self._lock:
            return operation_id in self._cancelled

    def cancel(self, req: AFDComponentReshardCancelInput) -> AFDComponentReshardStatus:
        self._require_enabled()
        with self._lock:
            status = self._operations.get(req.operation_id)
            if status is None:
                raise AFDComponentReshardNotFound(f"unknown operation_id {req.operation_id!r}")
            if req.expected_epoch is not None and req.expected_epoch != status.epoch:
                raise AFDComponentReshardConflict(
                    f"epoch conflict: expected {req.expected_epoch}, operation has {status.epoch}"
                )
            if status.phase in TERMINAL_PHASES:
                return dataclasses.replace(status)
            if status.phase in (
                AFDComponentReshardPhase.COMMITTING.value,
                AFDComponentReshardPhase.ACTIVATING.value,
                AFDComponentReshardPhase.RETIRING.value,
            ):
                raise AFDComponentReshardConflict(
                    "operation can no longer be cancelled after commit started"
                )
            self._cancelled.add(req.operation_id)
            self._transition(
                req.operation_id,
                AFDComponentReshardPhase.CANCELLED.value,
                "cancelled before commit",
            )
            return dataclasses.replace(self._operations[req.operation_id])

    def get_status(self, operation_id: Optional[str] = None):
        self._require_enabled()
        with self._lock:
            if operation_id is not None:
                status = self._operations.get(operation_id)
                if status is None:
                    raise AFDComponentReshardNotFound(f"unknown operation_id {operation_id!r}")
                return dataclasses.replace(status)
            return [dataclasses.replace(value) for value in reversed(self._operations.values())]

    def get_topology(self, stage: Optional[str] = None):
        self._require_enabled()
        with self._lock:
            if stage is not None:
                if stage not in self._topologies:
                    raise ValueError("stage must be 'prefill' or 'decode'")
                return dataclasses.replace(self._topologies[stage])
            return [dataclasses.replace(value) for value in self._topologies.values()]
