# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Feature-local AFD component standby lifecycle.

This protocol deliberately has no compatibility with native in-place reshard
commands.  Transport is injected so unit tests do not need torch.distributed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Dict, Optional


class AFDComponentWorldAction(str, Enum):
    PREPARE = "prepare"
    PREPARE_STATUS = "prepare_status"
    QUIESCE = "quiesce"
    ACTIVATE = "activate"
    DEMOTE = "demote"
    ABORT = "abort"
    RETIRE = "retire"
    SHUTDOWN = "shutdown"


class AFDComponentRankState(str, Enum):
    ACTIVE = "active"
    STANDBY = "standby"
    PREPARED = "prepared"
    SHUTDOWN = "shutdown"


@dataclass(frozen=True)
class AFDComponentWorldCommand:
    operation: str
    epoch: int
    component: str
    action: AFDComponentWorldAction
    target_tp: Optional[int] = None

    @classmethod
    def parse(cls, payload: Dict[str, Any]) -> "AFDComponentWorldCommand":
        required = {"operation", "epoch", "component", "action"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"AFD component world command misses {sorted(missing)}")
        operation = str(payload["operation"])
        epoch = int(payload["epoch"])
        component = str(payload["component"]).lower()
        action = AFDComponentWorldAction(payload["action"])
        target_tp = payload.get("target_tp")
        if not operation or epoch < 0 or component not in ("attn", "ffn"):
            raise ValueError("invalid AFD component command identity")
        if action in (
            AFDComponentWorldAction.PREPARE,
            AFDComponentWorldAction.ACTIVATE,
            AFDComponentWorldAction.DEMOTE,
        ):
            if target_tp is None or int(target_tp) < 1:
                raise ValueError(f"{action.value} requires positive target_tp")
            target_tp = int(target_tp)
        return cls(operation, epoch, component, action, target_tp)

    def as_payload(self) -> Dict[str, Any]:
        return {
            "operation": self.operation,
            "epoch": self.epoch,
            "component": self.component,
            "action": self.action.value,
            "target_tp": self.target_tp,
        }


class AFDComponentRankLifecycle:
    """State-only safe-point hook; it never moves or commits model weights."""

    def __init__(self, rank: int, active_tp: int, max_tp: int, component: str):
        if not 1 <= active_tp <= max_tp:
            raise ValueError("active_tp must be within max-world")
        self.rank = rank
        self.active_tp = active_tp
        self.max_tp = max_tp
        self.component = component
        self.epoch = 0
        self.operation: Optional[str] = None
        self.prepared_tp: Optional[int] = None
        self.state = (
            AFDComponentRankState.ACTIVE
            if rank < active_tp
            else AFDComponentRankState.STANDBY
        )

    @property
    def is_standby(self) -> bool:
        # PREPARE does not activate a joining rank; it must keep consuming the
        # standby control world until ACTIVATE, ABORT, or SHUTDOWN.
        return (
            self.state != AFDComponentRankState.SHUTDOWN
            and self.rank >= self.active_tp
        )

    def apply_safe_point(self, payload: Dict[str, Any]) -> AFDComponentRankState:
        cmd = AFDComponentWorldCommand.parse(payload)
        if cmd.component != self.component:
            raise ValueError(f"component identity mismatch: {cmd.component} != {self.component}")
        if cmd.epoch < self.epoch:
            raise ValueError("stale AFD component epoch")
        if cmd.action == AFDComponentWorldAction.PREPARE:
            if cmd.target_tp > self.max_tp:
                raise ValueError("target_tp exceeds AFD component max-world")
            self.operation, self.prepared_tp = cmd.operation, cmd.target_tp
            self.state = AFDComponentRankState.PREPARED
        elif cmd.action == AFDComponentWorldAction.PREPARE_STATUS:
            # Polling observes the background thread without changing topology.
            pass
        elif cmd.action == AFDComponentWorldAction.QUIESCE:
            # Ordering fence only; topology and staged state stay unchanged.
            pass
        elif cmd.action == AFDComponentWorldAction.ACTIVATE:
            if self.operation != cmd.operation or self.prepared_tp != cmd.target_tp:
                raise ValueError("activate does not match prepared command")
            self.active_tp, self.epoch = cmd.target_tp, cmd.epoch
            self.state = (
                AFDComponentRankState.ACTIVE
                if self.rank < cmd.target_tp
                else AFDComponentRankState.STANDBY
            )
            self.operation = self.prepared_tp = None
        elif cmd.action == AFDComponentWorldAction.DEMOTE:
            self.active_tp, self.epoch = cmd.target_tp, cmd.epoch
            self.state = (
                AFDComponentRankState.ACTIVE
                if self.rank < cmd.target_tp
                else AFDComponentRankState.STANDBY
            )
            self.operation = self.prepared_tp = None
        elif cmd.action in (
            AFDComponentWorldAction.ABORT,
            AFDComponentWorldAction.RETIRE,
        ):
            self.operation = self.prepared_tp = None
            self.state = (
                AFDComponentRankState.ACTIVE
                if self.rank < self.active_tp
                else AFDComponentRankState.STANDBY
            )
        else:
            self.state = AFDComponentRankState.SHUTDOWN
        return self.state


class AFDComponentStandbyLoop:
    """Blocking loop over a dedicated AFD component world receiver."""

    def __init__(
        self,
        lifecycle: AFDComponentRankLifecycle,
        receive: Callable[[], Dict[str, Any]],
    ):
        self.lifecycle, self.receive = lifecycle, receive

    def run(self) -> AFDComponentRankState:
        while self.lifecycle.is_standby:
            payload = self.receive()
            if payload is None:
                continue
            state = self.lifecycle.apply_safe_point(payload)
            if state in (AFDComponentRankState.ACTIVE, AFDComponentRankState.SHUTDOWN):
                return state
        return self.lifecycle.state
