"""Fixed multi-peer transports for shared Attention/FFN disaggregation.

A pool is process-local and immutable after construction. Batch routing uses a
``ContextVar`` scoped by :meth:`AFDCrossNodePeerPool.select`; it never mutates a
module-global communicator override.
"""

from __future__ import annotations
import contextvars
import json
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Mapping, Optional
from sglang.srt.layers.afd_type import AFDPerspective


@dataclass(frozen=True)
class AFDCrossNodePeerSpec:
    """One bidirectional edge. Port bases are named for the tensor sender.

    FFN listens on ``attn_base_port`` (A->F), and Attention listens on
    ``ffn_base_port`` (F->A). Handshake ports use the same direction rule.
    """

    peer_id: str
    peer_host: str
    ffn_base_port: int
    attn_base_port: int
    ffn_handshake_base_port: int = 60000
    attn_handshake_base_port: int = 61000
    channel: int = 0
    timeout_ms: int = 60000
    backend: str = "zmq"
    control_endpoint: Optional[str] = None

    def __post_init__(self):
        if not self.peer_id or not self.peer_host:
            raise ValueError("AFD peer_id and peer_host must be non-empty")
        if self.backend not in ("zmq", "ucx"):
            raise ValueError(f"Unsupported shared AFD backend: {self.backend}")
        if self.channel < 0 or self.timeout_ms <= 0:
            raise ValueError("AFD channel must be non-negative and timeout positive")
        ports = (
            self.ffn_base_port,
            self.attn_base_port,
            self.ffn_handshake_base_port,
            self.attn_handshake_base_port,
        )
        if any(not 0 <= port <= 65534 for port in ports):
            raise ValueError(f"Invalid AFD port base in peer {self.peer_id}: {ports}")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]):
        aliases = {
            "ffn_port_base": "ffn_base_port",
            "attn_port_base": "attn_base_port",
            "ffn_handshake_base": "ffn_handshake_base_port",
            "attn_handshake_base": "attn_handshake_base_port",
        }
        return cls(**{aliases.get(k, k): v for k, v in value.items()})


def parse_shared_peer_specs(value: str) -> tuple[AFDCrossNodePeerSpec, ...]:
    """Parse JSON or ``id@host:ffn:attn[:ffn_hs:attn_hs:channel:timeout]``."""
    if not value or not value.strip():
        raise ValueError("Shared AFD peer specs must be non-empty")
    value = value.strip()
    if value[0] in "[{":
        payload = json.loads(value)
        if isinstance(payload, dict):
            payload = payload.get("peers", [payload])
        if not isinstance(payload, list):
            raise ValueError("Shared AFD peer specs JSON must be a list")
        specs = tuple(AFDCrossNodePeerSpec.from_mapping(item) for item in payload)
    else:
        parsed = []
        for item in value.split(","):
            peer_id, endpoint = item.strip().split("@", 1)
            parts = endpoint.rsplit(":", 7)
            if len(parts) < 3:
                raise ValueError(f"Invalid shared AFD peer spec: {item}")
            host, numbers = parts[0], [int(part) for part in parts[1:]]
            kwargs = dict(
                peer_id=peer_id,
                peer_host=host,
                ffn_base_port=numbers[0],
                attn_base_port=numbers[1],
            )
            kwargs.update(
                zip(
                    (
                        "ffn_handshake_base_port",
                        "attn_handshake_base_port",
                        "channel",
                        "timeout_ms",
                    ),
                    numbers[2:],
                )
            )
            parsed.append(AFDCrossNodePeerSpec(**kwargs))
        specs = tuple(parsed)
    if not specs:
        raise ValueError("At least one shared AFD peer spec is required")
    if len({spec.peer_id for spec in specs}) != len(specs):
        raise ValueError("Shared AFD peer IDs must be unique in one process")
    ports = []
    for spec in specs:
        ports.extend(
            (
                spec.ffn_base_port + 1,
                spec.attn_base_port + 1,
                spec.ffn_handshake_base_port + 1,
                spec.attn_handshake_base_port + 1,
            )
        )
    if len(set(ports)) != len(ports):
        raise ValueError("Shared AFD edge data/handshake ports must be unique")
    return specs


@dataclass(frozen=True)
class AFDPeerStatus:
    peer_id: str
    state: str
    reconnects: int
    error: Optional[str] = None


class AFDUcxPeerAdapter:
    """Reserved adapter; shared-pool UCX wiring is intentionally blocked."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "Shared AFD UCX is not implemented; use backend='zmq'"
        )


class AFDCrossNodePeerPool:
    """Fixed process-local registry of independent preconstructed edges."""

    _registries: Dict[str, "AFDCrossNodePeerPool"] = {}
    _registry_lock = threading.Lock()

    def __init__(
        self,
        perspective: AFDPerspective,
        specs: Iterable[AFDCrossNodePeerSpec],
        *,
        local_tp_size: int = 1,
        local_tp_rank: int = 0,
        comm_factory: Optional[Callable[..., object]] = None,
        broadcast_factory: Optional[Callable[..., object]] = None,
        async_factory: Optional[Callable[[object], object]] = None,
    ):
        specs = tuple(specs)
        if not specs or len({s.peer_id for s in specs}) != len(specs):
            raise ValueError("Shared AFD peer specs must be non-empty and unique")
        if local_tp_size < 1 or not 0 <= local_tp_rank < local_tp_size:
            raise ValueError("Invalid local TP size/rank")
        if comm_factory is None:
            from sglang.srt.layers.afd import ZMQSimpleTensorCommunicator

            comm_factory = ZMQSimpleTensorCommunicator
        if broadcast_factory is None:
            from sglang.srt.layers.afd import BroadcastTensorCommunicator

            broadcast_factory = BroadcastTensorCommunicator
        if async_factory is None:
            from sglang.srt.layers.afd import AsyncTensorCommunicator

            async_factory = AsyncTensorCommunicator
        self.perspective, self.specs = perspective, {s.peer_id: s for s in specs}
        self.local_tp_size, self.local_tp_rank = local_tp_size, local_tp_rank
        self._comm_factory, self._broadcast_factory = comm_factory, broadcast_factory
        self._async_factory = async_factory
        self._channels, self._states = {}, {}
        self._selection = contextvars.ContextVar(f"afd_peer_{id(self)}", default=None)
        self._lock = threading.RLock()
        for spec in specs:
            self._construct(spec, 0)

    @classmethod
    def process_registry(cls, registry_id, factory):
        with cls._registry_lock:
            if registry_id not in cls._registries:
                cls._registries[registry_id] = factory()
            return cls._registries[registry_id]

    def _construct(self, spec, reconnects):
        if spec.backend == "ucx":
            AFDUcxPeerAdapter(self.perspective, spec)
        inner = None
        try:
            if self.local_tp_rank == 0:
                inner = self._comm_factory(
                    self.perspective,
                    peer_host=spec.peer_host,
                    ffn_base_port=spec.ffn_base_port,
                    attn_base_port=spec.attn_base_port,
                    ffn_handshake_base_port=spec.ffn_handshake_base_port,
                    attn_handshake_base_port=spec.attn_handshake_base_port,
                    channel=spec.channel,
                    timeout_ms=spec.timeout_ms,
                )
            comm = inner
            if self.local_tp_size > 1:
                comm = self._broadcast_factory(
                    inner_comm=inner,
                    local_tp_size=self.local_tp_size,
                    local_tp_rank=self.local_tp_rank,
                )
            self._channels[spec.peer_id] = self._async_factory(comm)
            self._states[spec.peer_id] = AFDPeerStatus(
                spec.peer_id, "ready", reconnects
            )
        except Exception as exc:
            if inner is not None and hasattr(inner, "close"):
                inner.close()
            self._states[spec.peer_id] = AFDPeerStatus(
                spec.peer_id, "failed", reconnects, str(exc)
            )
            raise

    @property
    def peer_ids(self):
        return tuple(sorted(self.specs))

    def __len__(self):
        return len(self.specs)

    def __contains__(self, peer_id):
        return peer_id in self.specs

    def get(self, peer_id=None):
        peer_id = peer_id if peer_id is not None else self._selection.get()
        if peer_id is None:
            raise RuntimeError("No shared AFD peer selected for current batch")
        try:
            return self._channels[peer_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown shared AFD peer {peer_id!r}; available={self.peer_ids}"
            ) from exc

    @contextmanager
    def select(self, peer_id):
        self.get(peer_id)
        token = self._selection.set(peer_id)
        try:
            yield self.get(peer_id)
        finally:
            self._selection.reset(token)

    def status(self):
        return {p: self._states[p] for p in self.peer_ids}

    @staticmethod
    def _close_communicator(comm):
        """Close the first owning wrapper, or the deepest closeable transport."""
        seen = set()
        current = comm
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            close = getattr(current, "close", None)
            if callable(close):
                close()
                return
            current = getattr(current, "inner", getattr(current, "inner_comm", None))

    def reconnect(self, peer_id):
        with self._lock:
            old = self.get(peer_id)
            reconnects = self._states[peer_id].reconnects + 1
            self._close_communicator(old)
            try:
                self._construct(self.specs[peer_id], reconnects)
            except Exception:
                self._channels.pop(peer_id, None)
                raise
            return self._channels[peer_id]

    def close(self):
        with self._lock:
            for peer_id, comm in tuple(self._channels.items()):
                self._close_communicator(comm)
                self._states[peer_id] = AFDPeerStatus(
                    peer_id, "closed", self._states[peer_id].reconnects
                )
            self._channels.clear()


# Existing same-node ipc_cpp continuation API.
@dataclass(frozen=True)
class AFDPeerSpec:
    group_id: int
    channel_id: int
    peer_device: int

    def __post_init__(self):
        if min(self.group_id, self.channel_id, self.peer_device) < 0:
            raise ValueError("AFD peer fields must be non-negative")


@dataclass
class AFDPeerChannel:
    spec: AFDPeerSpec
    async_comm: object
    lane: object

    @property
    def mb_id(self):
        return self.spec.group_id

    @property
    def inner(self):
        return self.async_comm.inner

    def recv_start(self, on_ready):
        return self.lane.recv_start(on_ready)

    def recv_wait(self):
        return self.lane.recv_wait()

    def drain(self):
        self.lane.drain()


class AFDPeerChannelPool:
    def __init__(self, perspective, specs, *, comm_factory=None, async_factory=None):
        specs = list(specs)
        if not specs:
            raise ValueError("At least one AFD peer spec is required")
        if len({s.group_id for s in specs}) != len(specs):
            raise ValueError("AFD peer group IDs must be unique")
        if len({s.channel_id for s in specs}) != len(specs):
            raise ValueError("AFD peer channel IDs must be unique")
        if comm_factory is None:
            from sglang.srt.layers.afd_ipc_cpp.communicator import (
                CppIpcTensorCommunicator,
            )

            comm_factory = CppIpcTensorCommunicator
        if async_factory is None:
            from sglang.srt.layers.afd import AsyncTensorCommunicator

            def async_factory(inner):
                return AsyncTensorCommunicator(inner, allow_background_recv=True)

        from sglang.srt.layers.afd_per_mb_channel import PerMbChannel

        self.perspective, self._channels = perspective, {}
        for spec in specs:
            inner = comm_factory(
                perspective, peer_device=spec.peer_device, channel_id=spec.channel_id
            )
            async_comm = async_factory(inner)
            self._channels[spec.group_id] = AFDPeerChannel(
                spec, async_comm, PerMbChannel(spec.group_id, async_comm)
            )

    @property
    def group_ids(self):
        return tuple(sorted(self._channels))

    def __len__(self):
        return len(self._channels)

    def __contains__(self, group_id):
        return group_id in self._channels

    def __getitem__(self, group_id):
        try:
            return self._channels[group_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown AFD PF group {group_id}; available={self.group_ids}"
            ) from exc

    def drain(self):
        for group_id in self.group_ids:
            self._channels[group_id].drain()
