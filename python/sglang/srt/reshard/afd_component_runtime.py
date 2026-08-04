# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Isolated runtime state machine for colocated AFD component resharding.

This module owns no CUDA resources. A topology is published only after an
explicit adapter receipt says that the corresponding runtime operation has
completed. It does not use the native standby/control-file implementation.
"""
from __future__ import annotations

import dataclasses
import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Mapping, Optional, Protocol, Tuple

from sglang.srt.managers.io_struct import AFDComponentReshardReqInput
from sglang.srt.reshard.afd_component_reshard import (
    AFDComponentReshardCoordinator,
    AFDComponentReshardPhase,
)

logger = logging.getLogger(__name__)


class AFDComponent(str, Enum):
    ATTN = "attn"
    FFN = "ffn"


class RankLifecycle(str, Enum):
    ACTIVE = "active"
    STANDBY = "standby"
    PREPARED = "prepared"
    DEMOTING = "demoting"


@dataclass(frozen=True)
class AFDComponentTopology:
    pair_id: str
    stage: str
    component: AFDComponent
    epoch: int
    max_world_size: int
    active_ranks: Tuple[int, ...]
    rank_devices: Tuple[int, ...]

    def __post_init__(self) -> None:
        if self.stage not in ("prefill", "decode"):
            raise ValueError("stage must be 'prefill' or 'decode'")
        if not 1 <= self.max_world_size <= 4:
            raise ValueError("max_world_size must be in [1, 4] for the first runtime")
        if len(self.rank_devices) != self.max_world_size:
            raise ValueError("rank_devices must map every max-world rank")
        if self.active_ranks != tuple(range(len(self.active_ranks))):
            raise ValueError("active_ranks must be the dense prefix [0, active_tp)")
        if not self.active_ranks or len(self.active_ranks) > self.max_world_size:
            raise ValueError("active_ranks must be non-empty and within max world")
        if self.epoch < 0:
            raise ValueError("epoch must be non-negative")

    @property
    def active_tp(self) -> int:
        return len(self.active_ranks)


@dataclass(frozen=True)
class PreparedComponentTransition:
    source: AFDComponentTopology
    target: AFDComponentTopology
    lifecycles: Tuple[RankLifecycle, ...]


@dataclass(frozen=True)
class AdapterReceipt:
    completed: bool
    detail: str = ""
    breakdown: Mapping[str, float] = dataclasses.field(default_factory=dict)
    # Set when the boundary rejected the action for a transient reason (e.g. the
    # pair has not finished draining in-flight AFD work yet) and the caller may
    # retry after a short backoff instead of failing pre-commit.
    retryable: bool = False


class AFDColocatedRuntimeAdapter(Protocol):
    """GPU-owning integration boundary. Each method must return a receipt."""

    def prepare(self, request: AFDComponentReshardReqInput, transitions: Mapping[AFDComponent, PreparedComponentTransition]) -> AdapterReceipt: ...
    def drain(self, request: AFDComponentReshardReqInput, cut_watermark: int) -> AdapterReceipt: ...
    def activate(self, request: AFDComponentReshardReqInput, transitions: Mapping[AFDComponent, PreparedComponentTransition]) -> AdapterReceipt: ...
    def retire(self, request: AFDComponentReshardReqInput, transitions: Mapping[AFDComponent, PreparedComponentTransition]) -> AdapterReceipt: ...


class AFDComponentRuntime:
    """Thread-safe immutable topology publisher for one P/D pair."""

    def __init__(self, *, pair_id: str, stage: str, max_world_size: int, attn_tp: int, ffn_tp: int, rank_devices: Optional[Tuple[int, ...]] = None) -> None:
        devices = rank_devices or tuple(range(max_world_size))
        self._lock = threading.RLock()
        self._prepared: Optional[Dict[AFDComponent, PreparedComponentTransition]] = None
        self._topologies: Dict[AFDComponent, AFDComponentTopology] = {
            component: AFDComponentTopology(pair_id, stage, component, 0, max_world_size, tuple(range(tp)), devices)
            for component, tp in ((AFDComponent.ATTN, attn_tp), (AFDComponent.FFN, ffn_tp))
        }

    def topology(self, component: AFDComponent) -> AFDComponentTopology:
        with self._lock:
            return self._topologies[component]

    def lifecycles(self, component: AFDComponent) -> Tuple[RankLifecycle, ...]:
        with self._lock:
            if self._prepared is not None:
                return self._prepared[component].lifecycles
            active = set(self._topologies[component].active_ranks)
            return tuple(RankLifecycle.ACTIVE if rank in active else RankLifecycle.STANDBY for rank in range(self._topologies[component].max_world_size))

    def prepare(self, *, target_attn_tp: int, target_ffn_tp: int) -> Mapping[AFDComponent, PreparedComponentTransition]:
        with self._lock:
            if self._prepared is not None:
                raise RuntimeError("a component transition is already prepared")
            targets = {AFDComponent.ATTN: target_attn_tp, AFDComponent.FFN: target_ffn_tp}
            prepared: Dict[AFDComponent, PreparedComponentTransition] = {}
            for component, target_tp in targets.items():
                source = self._topologies[component]
                target = dataclasses.replace(source, epoch=source.epoch + 1, active_ranks=tuple(range(target_tp)))
                old, new = set(source.active_ranks), set(target.active_ranks)
                states = tuple(
                    RankLifecycle.ACTIVE if rank in old & new else
                    RankLifecycle.PREPARED if rank in new - old else
                    RankLifecycle.DEMOTING if rank in old - new else
                    RankLifecycle.STANDBY
                    for rank in range(source.max_world_size)
                )
                prepared[component] = PreparedComponentTransition(source, target, states)
            if len({item.target.epoch for item in prepared.values()}) != 1:
                raise RuntimeError("A/F component epochs diverged")
            self._prepared = prepared
            return dict(prepared)

    def activate(self) -> Mapping[AFDComponent, AFDComponentTopology]:
        """Atomically publish both A/F target topologies after adapter activation."""
        with self._lock:
            if self._prepared is None:
                raise RuntimeError("no prepared component transition")
            self._topologies = {component: transition.target for component, transition in self._prepared.items()}
            self._prepared = None
            return dict(self._topologies)

    def abort(self) -> None:
        with self._lock:
            self._prepared = None


class AFDColocatedRuntimeExecutor:
    def __init__(self, runtime: AFDComponentRuntime, adapter: AFDColocatedRuntimeAdapter):
        self.runtime = runtime
        self.adapter = adapter

    @staticmethod
    def _require(receipt: AdapterReceipt, action: str) -> None:
        if not isinstance(receipt, AdapterReceipt) or not receipt.completed:
            detail = receipt.detail if isinstance(receipt, AdapterReceipt) else "missing receipt"
            raise RuntimeError(f"GPU adapter did not complete {action}: {detail}")

    def __call__(self, request, transition: Callable, is_cancelled: Callable[[], bool]) -> None:
        operation_id = getattr(request, "operation_id", "")
        stage = getattr(request, "stage", "")
        log_ctx = "op=%s stage=%s comp=coordinator" % (operation_id, stage)
        prepared = self.runtime.prepare(target_attn_tp=request.target_attn_tp, target_ffn_tp=request.target_ffn_tp)
        try:
            logger.info("[AFD-reshard] %s executor.prepare begin", log_ctx)
            prepare_receipt = self.adapter.prepare(request, prepared)
            self._require(prepare_receipt, "prepare")
            breakdown = dict(prepare_receipt.breakdown)
            prepare_s = float(breakdown.get("prepare_s", 0.0))
            if is_cancelled():
                self.runtime.abort()
                transition(AFDComponentReshardPhase.CANCELLED.value, "cancelled after preparation")
                logger.info("[AFD-reshard] %s cancelled after prepare", log_ctx)
                return
            transition(
                AFDComponentReshardPhase.PREPARED.value,
                f"GPU adapter prepared shadow state in {prepare_s:.3f}s",
                breakdown,
            )
            logger.info(
                "[AFD-reshard] %s stage=prepared prepare_s=%.3f -> entering drain",
                log_ctx, prepare_s,
            )
            cut = int(getattr(request, "dispatch_cut_watermark", 0))
            # Announce DRAINING before the (potentially long) bounded quiesce
            # retry so the operation phase reflects that we left PREPARED. This
            # makes a stuck drain visible in status polls instead of appearing
            # frozen at PREPARED.
            transition(
                AFDComponentReshardPhase.DRAINING.value,
                f"draining in-flight AFD work (cut_watermark={cut})",
            )
            logger.info(
                "[AFD-reshard] %s stage=draining cut_watermark=%d calling adapter.drain",
                log_ctx, cut,
            )
            import time
            drain_started = time.perf_counter()
            drain_receipt = self.adapter.drain(request, cut)
            self._require(drain_receipt, "drain")
            drain_s = time.perf_counter() - drain_started
            transition(
                AFDComponentReshardPhase.COMMITTING.value,
                "atomic activation started",
                {
                    "drain_s": drain_s,
                    "quiesce_drain_s": drain_s,
                    **dict(drain_receipt.breakdown),
                },
            )
            logger.info(
                "[AFD-reshard] %s stage=drained detail=%s -> committing",
                log_ctx, getattr(drain_receipt, "detail", ""),
            )
            logger.info("[AFD-reshard] %s stage=committing calling adapter.activate", log_ctx)
            commit_started = time.perf_counter()
            activate_started = time.perf_counter()
            activate_receipt = self.adapter.activate(request, prepared)
            self._require(activate_receipt, "activate")
            activate_s = time.perf_counter() - activate_started
            logger.info("[AFD-reshard] %s stage=activated calling adapter.retire", log_ctx)
            retire_started = time.perf_counter()
            self._require(self.adapter.retire(request, prepared), "retire")
            retire_s = time.perf_counter() - retire_started
            transition(
                AFDComponentReshardPhase.RETIRING.value,
                "activation complete",
                {
                    **dict(activate_receipt.breakdown),
                    "activate_s": activate_s,
                    "redirect_readiness_s": activate_s,
                    "retire_s": retire_s,
                    "commit_s": time.perf_counter() - commit_started,
                },
            )
            # Publish only after every GPU-owning hook has produced a receipt.
            self.runtime.activate()
            logger.info("[AFD-reshard] %s stage=retired topology published", log_ctx)
        except Exception as exc:
            logger.warning("[AFD-reshard] %s executor failed: %s", log_ctx, exc)
            self.runtime.abort()
            # Best-effort release of admission fences on both pair participants.
            try:
                self.adapter.retire(request, prepared)
            except Exception:
                pass
            raise


def register_afd_colocated_runtime(coordinator: AFDComponentReshardCoordinator, runtime: AFDComponentRuntime, adapter: Optional[AFDColocatedRuntimeAdapter]) -> Optional[AFDColocatedRuntimeExecutor]:
    """Register only a real adapter; None preserves explicit unsupported behavior."""
    executor = AFDColocatedRuntimeExecutor(runtime, adapter) if adapter is not None else None
    coordinator.register_executor(executor)
    return executor
