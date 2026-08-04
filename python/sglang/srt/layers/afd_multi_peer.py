"""Multi-peer communication primitives for shared-PA AFD.

The production AFD path historically owns one process-global communicator.
Shared PA deployments need one persistent channel per PF group so responses
can complete independently without sharing FIFO/ring state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional

from sglang.srt.layers.afd_type import AFDPerspective


@dataclass(frozen=True)
class AFDPeerSpec:
    """Stable addressing for one PF group."""

    group_id: int
    channel_id: int
    peer_device: int

    def __post_init__(self):
        if self.group_id < 0:
            raise ValueError("group_id must be non-negative")
        if self.channel_id < 0:
            raise ValueError("channel_id must be non-negative")
        if self.peer_device < 0:
            raise ValueError("peer_device must be non-negative")


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
    """A fixed process-local registry of independent PF channels."""

    def __init__(
        self,
        perspective: AFDPerspective,
        specs: Iterable[AFDPeerSpec],
        *,
        comm_factory: Optional[Callable[..., object]] = None,
        async_factory: Optional[Callable[[object], object]] = None,
    ):
        specs = list(specs)
        if not specs:
            raise ValueError("At least one AFD peer spec is required")

        group_ids = [spec.group_id for spec in specs]
        channel_ids = [spec.channel_id for spec in specs]
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("AFD peer group IDs must be unique")
        if len(set(channel_ids)) != len(channel_ids):
            raise ValueError("AFD peer channel IDs must be unique")

        if comm_factory is None:
            from sglang.srt.layers.afd_ipc_cpp.communicator import (
                CppIpcTensorCommunicator,
            )

            comm_factory = CppIpcTensorCommunicator
        if async_factory is None:
            from sglang.srt.layers.afd import AsyncTensorCommunicator

            async_factory = lambda inner: AsyncTensorCommunicator(
                inner, allow_background_recv=True
            )
        from sglang.srt.layers.afd_per_mb_channel import PerMbChannel

        self.perspective = perspective
        self._channels: Dict[int, AFDPeerChannel] = {}
        for spec in specs:
            inner = comm_factory(
                perspective,
                peer_device=spec.peer_device,
                channel_id=spec.channel_id,
            )
            async_comm = async_factory(inner)
            self._channels[spec.group_id] = AFDPeerChannel(
                spec=spec,
                async_comm=async_comm,
                lane=PerMbChannel(spec.group_id, async_comm),
            )

    @property
    def group_ids(self):
        return tuple(sorted(self._channels))

    def __len__(self):
        return len(self._channels)

    def __contains__(self, group_id: int):
        return group_id in self._channels

    def __getitem__(self, group_id: int) -> AFDPeerChannel:
        try:
            return self._channels[group_id]
        except KeyError as exc:
            raise KeyError(
                f"Unknown AFD PF group {group_id}; "
                f"available={self.group_ids}"
            ) from exc

    def drain(self):
        errors = []
        for group_id in self.group_ids:
            try:
                self._channels[group_id].drain()
            except Exception as exc:
                errors.append((group_id, exc))
        if errors:
            details = ", ".join(
                f"group={group_id}: {exc}" for group_id, exc in errors
            )
            raise RuntimeError(f"Failed to drain AFD peer channels: {details}")
