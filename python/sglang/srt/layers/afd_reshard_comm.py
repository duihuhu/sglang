# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0
"""Epoch-scoped communication primitives for the isolated AFD reshard path."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, FrozenSet, Optional, Tuple

import torch
import torch.distributed as dist


@dataclass(frozen=True, order=True)
class AFDEpochChannelIdentity:
    pair_id: str
    stage: str
    epoch: int
    channel: str

    def __post_init__(self) -> None:
        if not self.pair_id or self.stage not in ("prefill", "decode"):
            raise ValueError("identity requires pair_id and prefill/decode stage")
        if self.epoch < 0 or not self.channel:
            raise ValueError("identity requires non-negative epoch and channel")


class EpochChannelState(str, Enum):
    PREPARING = "preparing"
    READY = "ready"
    ACTIVE = "active"
    FENCED = "fenced"
    RETIRED = "retired"


@dataclass(frozen=True)
class EpochChannelSnapshot:
    identity: AFDEpochChannelIdentity
    state: EpochChannelState
    ready_participants: FrozenSet[str] = frozenset()


class AFDEpochCommunicatorRegistry:
    """Atomic registry; transport creation remains an explicit adapter action."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: Dict[AFDEpochChannelIdentity, EpochChannelSnapshot] = {}

    def prepare(self, identity: AFDEpochChannelIdentity) -> EpochChannelSnapshot:
        with self._lock:
            if identity in self._entries:
                raise RuntimeError(f"channel already exists: {identity}")
            value = EpochChannelSnapshot(identity, EpochChannelState.PREPARING)
            self._entries[identity] = value
            return value

    def ready(self, identity: AFDEpochChannelIdentity, participant: str) -> EpochChannelSnapshot:
        if participant not in ("attn", "ffn"):
            raise ValueError("participant must be attn or ffn")
        with self._lock:
            value = self._entries[identity]
            if value.state not in (EpochChannelState.PREPARING, EpochChannelState.READY):
                raise RuntimeError(f"cannot mark ready from {value.state.value}")
            participants = value.ready_participants | {participant}
            state = EpochChannelState.READY if participants == {"attn", "ffn"} else EpochChannelState.PREPARING
            value = EpochChannelSnapshot(identity, state, frozenset(participants))
            self._entries[identity] = value
            return value

    def activate(self, identities: Tuple[AFDEpochChannelIdentity, ...]) -> None:
        with self._lock:
            values = [self._entries[item] for item in identities]
            if not values or any(value.state != EpochChannelState.READY for value in values):
                raise RuntimeError("all epoch channels must be ready before activation")
            epochs = {(value.identity.pair_id, value.identity.stage, value.identity.epoch) for value in values}
            if len(epochs) != 1:
                raise RuntimeError("atomic activation requires one pair/stage/epoch")
            for value in values:
                self._entries[value.identity] = EpochChannelSnapshot(value.identity, EpochChannelState.ACTIVE, value.ready_participants)

    def fence(self, identity: AFDEpochChannelIdentity) -> None:
        with self._lock:
            value = self._entries[identity]
            if value.state != EpochChannelState.ACTIVE:
                raise RuntimeError("only active channels can be fenced")
            self._entries[identity] = EpochChannelSnapshot(identity, EpochChannelState.FENCED, value.ready_participants)

    def retire(self, identity: AFDEpochChannelIdentity) -> None:
        with self._lock:
            value = self._entries[identity]
            if value.state != EpochChannelState.FENCED:
                raise RuntimeError("channel must be fenced before retire")
            self._entries[identity] = EpochChannelSnapshot(identity, EpochChannelState.RETIRED, value.ready_participants)

    def snapshot(self, identity: AFDEpochChannelIdentity) -> EpochChannelSnapshot:
        with self._lock:
            return self._entries[identity]


@dataclass
class AFDPairDrainProtocol:
    """Strict dispatch watermark state shared by a single A/F pair."""
    dispatch_seq: int = 0
    cut_watermark: Optional[int] = None
    attn_ack: int = -1
    ffn_ack: int = -1
    pending_sends: int = 0
    pending_recvs: int = 0
    pending_fences: int = 0
    _lock: threading.RLock = field(default_factory=threading.RLock, repr=False, compare=False)

    def next_dispatch(self) -> int:
        with self._lock:
            if self.cut_watermark is not None:
                raise RuntimeError("dispatch is fenced at the cut watermark")
            self.dispatch_seq += 1
            return self.dispatch_seq

    def cut(self) -> int:
        with self._lock:
            if self.cut_watermark is None:
                self.cut_watermark = self.dispatch_seq
            return self.cut_watermark

    def install_external_cut(self, watermark: int) -> int:
        """Install the Attention-owned cut on its paired FFN scheduler."""
        watermark = int(watermark)
        if watermark < 0:
            raise ValueError("cut watermark cannot be negative")
        with self._lock:
            if self.cut_watermark is None:
                self.cut_watermark = watermark
            elif self.cut_watermark != watermark:
                raise RuntimeError(
                    "external cut does not match the active cut watermark"
                )
            return self.cut_watermark

    def ack(self, participant: str, watermark: int) -> None:
        with self._lock:
            if self.cut_watermark is None or watermark != self.cut_watermark:
                raise RuntimeError("ACK must exactly match the active cut watermark")
            if participant == "attn":
                self.attn_ack = watermark
            elif participant == "ffn":
                self.ffn_ack = watermark
            else:
                raise ValueError("participant must be attn or ffn")

    def set_pending(self, *, sends: int, recvs: int, fences: int) -> None:
        if min(sends, recvs, fences) < 0:
            raise ValueError("pending counts cannot be negative")
        with self._lock:
            self.pending_sends, self.pending_recvs, self.pending_fences = sends, recvs, fences

    @property
    def drained(self) -> bool:
        with self._lock:
            cut = self.cut_watermark
            return cut is not None and self.attn_ack == cut and self.ffn_ack == cut and self.pending_sends == self.pending_recvs == self.pending_fences == 0

    def reopen_after_transition(self) -> None:
        """Open dispatch admission after a successfully drained transition."""
        with self._lock:
            if self.cut_watermark is None:
                return
            cut = self.cut_watermark
            drained = (
                self.attn_ack == cut
                and self.ffn_ack == cut
                and self.pending_sends == self.pending_recvs == self.pending_fences == 0
            )
            if not drained:
                raise RuntimeError("cannot reopen AFD dispatch before strict drain")
            self._clear_transition_locked()

    def reopen_at_transition_boundary(self) -> None:
        """Reset a follower ledger at the all-rank activation boundary.

        Pair ACKs are exchanged only by the two rank-zero scheduler runtimes.
        Followers nevertheless install the same cut and must not carry it into
        the next epoch.  The final max-world activation handshake proves that
        commit completed everywhere; local in-flight counters must still be zero.
        """
        with self._lock:
            if self.cut_watermark is None:
                raise RuntimeError(
                    "cannot reopen AFD follower dispatch without an installed cut"
                )
            if self.pending_sends or self.pending_recvs or self.pending_fences:
                raise RuntimeError(
                    "cannot reopen AFD follower dispatch with pending work"
                )
            self._clear_transition_locked()

    def force_reopen_after_abort(self) -> None:
        """Clear a transition fence after abort cleanup restores source serving."""
        with self._lock:
            self._clear_transition_locked()

    def _clear_transition_locked(self) -> None:
        self.cut_watermark = None
        self.attn_ack = -1
        self.ffn_ack = -1
        self.pending_sends = 0
        self.pending_recvs = 0
        self.pending_fences = 0


@dataclass(frozen=True)
class TorchP2PCapability:
    available: bool
    backend: Optional[str]
    supports_cpu: bool
    supports_cuda: bool
    reason: str = ""


class TorchDistributedEpochControl:
    """Cross-process epoch handshake on a Gloo control process group."""

    def __init__(self, group: Optional[Any] = None) -> None:
        self.group = group

    def consensus(self, identity: AFDEpochChannelIdentity, marker: int) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        if str(dist.get_backend(self.group)).lower() != "gloo":
            raise RuntimeError("epoch control requires a Gloo process group")
        # A stable identity digest catches pair/stage/epoch/channel mismatches.
        import hashlib

        payload = f"{identity.pair_id}|{identity.stage}|{identity.epoch}|{identity.channel}"
        # Seven bytes always fit signed int64 used by the control tensor.
        digest = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:7], "big")
        value = torch.tensor([digest, marker], dtype=torch.int64)
        low, high = value.clone(), value.clone()
        dist.all_reduce(low, op=dist.ReduceOp.MIN, group=self.group)
        dist.all_reduce(high, op=dist.ReduceOp.MAX, group=self.group)
        if not torch.equal(low, high):
            raise RuntimeError("epoch control participants disagree on identity or marker")

    def barrier(self) -> None:
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("torch.distributed is not initialized")
        dist.barrier(group=self.group)


class TorchDistributedP2PAdapter:
    """Real cross-process tensor P2P over an initialized torch process group."""

    def __init__(self, group: Optional[Any] = None) -> None:
        self.group = group

    @staticmethod
    def capability(group: Optional[Any] = None) -> TorchP2PCapability:
        if not dist.is_available() or not dist.is_initialized():
            return TorchP2PCapability(False, None, False, False, "torch.distributed is not initialized")
        backend = str(dist.get_backend(group)).lower()
        cpu = backend == "gloo"
        cuda = backend == "nccl" and torch.cuda.is_available()
        return TorchP2PCapability(cpu or cuda, backend, cpu, cuda, "")

    def _check(self, tensor: torch.Tensor) -> None:
        capability = self.capability(self.group)
        supported = capability.supports_cuda if tensor.is_cuda else capability.supports_cpu
        if not supported:
            raise RuntimeError(f"backend {capability.backend!r} cannot transport tensor on {tensor.device}: {capability.reason}")

    def send(self, tensor: torch.Tensor, *, dst: int, tag: int = 0) -> None:
        self._check(tensor)
        dist.send(tensor.contiguous(), dst=dst, group=self.group, tag=tag)

    def recv(self, output: torch.Tensor, *, src: int, tag: int = 0) -> torch.Tensor:
        self._check(output)
        dist.recv(output, src=src, group=self.group, tag=tag)
        return output


class LocalLoopbackTensorCommunicator:
    """Test/local adapter explicitly restricted to one process."""
    cross_process = False

    def __init__(self) -> None:
        self._queue: queue.Queue[torch.Tensor] = queue.Queue()

    def send(self, tensor: torch.Tensor) -> None:
        self._queue.put(tensor.detach().clone())

    def recv(self) -> torch.Tensor:
        return self._queue.get()


class AFDReshardCommunicatorCache:
    """Epoch-keyed cache isolated from all legacy AFD singletons."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._cache: Dict[AFDEpochChannelIdentity, Any] = {}

    def refresh(self, identity: AFDEpochChannelIdentity, factory: Callable[[], Any]) -> Any:
        with self._lock:
            value = self._cache.get(identity)
            if value is None:
                value = factory()
                self._cache[identity] = value
            return value

    def retire(self, identity: AFDEpochChannelIdentity) -> None:
        with self._lock:
            value = self._cache.pop(identity, None)
        close = getattr(value, "close", None)
        if close is not None:
            close()


def refresh_afd_reshard_communicator(
    cache: AFDReshardCommunicatorCache,
    identity: AFDEpochChannelIdentity,
    *,
    active_tp: int,
    factory: Callable[[int], Any],
) -> Any:
    """Build an epoch-local Broadcast/Layer communicator for dynamic TP.

    The caller supplies the owning factory because process groups and CUDA
    streams are scheduler-owned. This helper never clears legacy globals.
    """
    if active_tp < 1:
        raise ValueError("active_tp must be positive")
    return cache.refresh(identity, lambda: factory(active_tp))
