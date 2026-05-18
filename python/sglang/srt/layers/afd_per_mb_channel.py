"""Per-mb communication channel for data-driven AFD schedule.

This module provides a thin abstraction that splits the single shared
``AsyncTensorCommunicator`` (used by the legacy `AFDStageScheduleGenerator`
path) into M independent channels — one per micro-batch.

Each channel owns its own backend ``FifoTensorCommunicator`` instance with a
distinct port (UCX), SHM path (IPC) or ZMQ socket pair, so that:

* Sends from different mbs do **not** share an asyncio Lock or a single
  ring slot ordering.
* Recvs from different mbs can complete out-of-order — whoever's data
  lands first on the wire is consumed first by the driver.

Backends are selected via ``server_args.afd_comm_backend`` exactly like
``afd.get_tensor_communicator()``.  We re-use the existing per-backend
classes; only the routing keys (port / SHM file / ZMQ port) are made
mb-aware so multiple instances coexist.
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Callable, List, Optional

import torch
import torch.distributed as dist

from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.server_args import get_global_server_args

logger = logging.getLogger(__name__)


class PerMbChannel:
    """One bidirectional tensor channel pinned to a specific micro-batch id.

    Wraps an ``AsyncTensorCommunicator`` for ``send_async`` /
    ``recv_start`` / ``recv_wait`` semantics, plus a small bit of state
    so the driver can ask "is this channel's recv finished?" without
    blocking.
    """

    def __init__(self, mb_id: int, async_comm):
        self.mb_id = mb_id
        self.async_comm = async_comm
        self._recv_pending = False
        self._on_ready: Optional[Callable[[int], None]] = None
        self._inflight_thread: Optional[threading.Thread] = None

    # ── send ──────────────────────────────────────────────────────────
    def send_async(self, x: torch.Tensor):
        self.async_comm.send_async(x)

    # ── recv ──────────────────────────────────────────────────────────
    def recv_start(self, on_ready: Callable[[int], None]):
        """Begin a background recv. ``on_ready(mb_id)`` is invoked from a
        background thread once the bytes have landed (data is *not* yet
        copied to the compute stream — that happens lazily inside
        ``recv_wait``)."""
        if self._recv_pending:
            raise RuntimeError(
                f"PerMbChannel(mb={self.mb_id}): recv_start called while a recv "
                "is already pending"
            )
        self._on_ready = on_ready
        self._recv_pending = True

        # Thin wrapper: kick off the underlying recv_start, then poll the
        # ring slot in a tiny daemon to know when bytes have arrived.
        # AsyncTensorCommunicator.recv_start spawns its own bg thread that
        # blocks on inner.recv_tensor; we hold onto that thread reference
        # via the ring index (write_idx - 1).
        self.async_comm.recv_start()
        # Capture the index that AsyncTensorCommunicator.recv_start just
        # wrote to (write was advanced post-spawn).
        ring_size = self.async_comm._RING_SIZE
        write_idx_after = self.async_comm._recv_idx_write
        # Find the most recent thread slot
        slot = (write_idx_after - 1) % ring_size
        bg_thread = self.async_comm._recv_threads[slot]
        if bg_thread is None:
            # The recv may have completed synchronously.
            self._recv_pending = True
            self._invoke_ready_async()
            return

        def _watcher():
            try:
                bg_thread.join()
            except Exception:
                logger.exception("PerMbChannel(mb=%d): watcher join failed",
                                 self.mb_id)
            # Mark the moment bytes have landed (before main-thread GPU sync).
            try:
                from sglang.srt.layers.afd_mixin import _afd_host_events
                _afd_host_events.append({
                    "ts_ms": round(time.time() * 1000, 3),
                    "role": "DRIVER", "layer": -1, "mb": self.mb_id,
                    "event": "recv_done",
                })
            except Exception:
                pass
            # Clear _recv_pending NOW (bytes have landed; the GPU copy
            # in recv_wait can still happen later).  Driver code may call
            # comm.recv_wait directly via the per-mb override (bypassing
            # PerMbChannel.recv_wait), so we cannot rely on recv_wait to
            # clear it.  Doing it here lets _issue_recv for the next
            # layer reuse this channel without spurious "already pending".
            self._recv_pending = False
            cb = self._on_ready
            self._on_ready = None
            if cb is not None:
                try:
                    cb(self.mb_id)
                except Exception:
                    logger.exception(
                        "PerMbChannel(mb=%d): on_ready callback failed",
                        self.mb_id,
                    )

        watcher = threading.Thread(
            target=_watcher,
            daemon=True,
            name=f"afd-mb{self.mb_id}-recv-watch",
        )
        watcher.start()
        self._inflight_thread = watcher

    def _invoke_ready_async(self):
        """Schedule the on_ready callback on a tiny daemon thread."""
        cb = self._on_ready

        def _fire():
            if cb is not None:
                try:
                    cb(self.mb_id)
                except Exception:
                    logger.exception(
                        "PerMbChannel(mb=%d): on_ready callback failed",
                        self.mb_id,
                    )

        threading.Thread(
            target=_fire, daemon=True,
            name=f"afd-mb{self.mb_id}-recv-ready",
        ).start()

    def recv_wait(self) -> torch.Tensor:
        """Drain the most recent recv; performs the GPU copy + sync."""
        if not self._recv_pending:
            raise RuntimeError(
                f"PerMbChannel(mb={self.mb_id}): recv_wait called with no "
                "pending recv"
            )
        try:
            tensor = self.async_comm.recv_wait()
        finally:
            self._recv_pending = False
            self._on_ready = None
        return tensor

    # ── lifecycle ────────────────────────────────────────────────────
    def drain(self):
        try:
            self.async_comm.drain_sends()
        except Exception:
            logger.exception("PerMbChannel(mb=%d): drain_sends failed",
                             self.mb_id)
        try:
            self.async_comm.drain_recvs()
        except Exception:
            logger.exception("PerMbChannel(mb=%d): drain_recvs failed",
                             self.mb_id)
        self._recv_pending = False
        self._on_ready = None
        self._inflight_thread = None


class MultiMbChannelSet:
    """One ``PerMbChannel`` per micro-batch id, sharing the same backend
    type but distinct backend instances (different ports / SHM paths)."""

    def __init__(self, num_mb: int, channels: List[PerMbChannel]):
        if len(channels) != num_mb:
            raise ValueError(
                f"MultiMbChannelSet: expected {num_mb} channels, got "
                f"{len(channels)}"
            )
        self.num_mb = num_mb
        self.channels = channels

    def __getitem__(self, mb_id: int) -> PerMbChannel:
        return self.channels[mb_id]

    def __len__(self) -> int:
        return self.num_mb

    def drain(self):
        for c in self.channels:
            try:
                c.drain()
            except Exception:
                logger.exception("MultiMbChannelSet: drain mb=%d failed",
                                 c.mb_id)


# --------------- backend factory -------------------------------------


def _make_inner_comm_for_mb(mb_id: int, defer_connect: bool = False):
    """Construct one backend ``FifoTensorCommunicator`` keyed by mb_id.

    The mb_id is forwarded to the backend so SHM paths / TCP ports
    differ per mb, allowing M instances to coexist on the same node.
    """
    from sglang.srt.layers.afd import (
        BroadcastTensorCommunicator,
        ZMQSimpleTensorCommunicator,
        get_afd_perspective,
    )

    perspective = get_afd_perspective()
    if perspective is None:
        raise RuntimeError(
            "afd_per_mb_channel: AFD perspective is not set"
        )

    server_args = get_global_server_args()
    comm_backend = getattr(server_args, "afd_comm_backend", None) or "auto"

    if comm_backend == "auto":
        if os.environ.get("AFD_UCX_TLS"):
            comm_backend = "ucx"
        elif os.environ.get("MLC_INTERFACE"):
            comm_backend = "stepmesh"
        else:
            comm_backend = "zmq"

    if comm_backend == "stepmesh":
        # StepMesh's fserver_lib is a process-singleton (only one
        # init() per process); we can't instantiate M of them.
        raise RuntimeError(
            "afd_per_mb_channel: stepmesh backend cannot run with "
            "--afd-async-schedule. Use --afd-comm-backend=ucx or ipc."
        )

    if comm_backend == "ucx":
        from sglang.srt.layers.rdma_comm import UcxTensorCommunicator
        return UcxTensorCommunicator(perspective, mb_id=mb_id,
                                     defer_connect=defer_connect)

    if comm_backend == "ipc":
        from sglang.srt.layers.ipc_comm import IpcTensorCommunicator
        return IpcTensorCommunicator(perspective, mb_id=mb_id)

    if comm_backend == "zmq":
        local_tp = server_args.tp_size
        local_tp_rank = (
            dist.get_rank() % local_tp if dist.is_initialized() else 0
        )
        ffn_base = "AFD_FFN_BASE_PORT"
        attn_base = "AFD_ATTN_BASE_PORT"
        old_ffn = os.environ.get(ffn_base)
        old_attn = os.environ.get(attn_base)
        try:
            ffn_port = int(old_ffn) if old_ffn else 40000
            attn_port = int(old_attn) if old_attn else 50000
            stride = max(local_tp + 4, 16)
            os.environ[ffn_base] = str(ffn_port + mb_id * stride)
            os.environ[attn_base] = str(attn_port + mb_id * stride)
            if local_tp > 1:
                base = ZMQSimpleTensorCommunicator(perspective) \
                    if local_tp_rank == 0 else None
                comm = BroadcastTensorCommunicator(
                    inner_comm=base,
                    local_tp_size=local_tp,
                    local_tp_rank=local_tp_rank,
                )
            else:
                comm = ZMQSimpleTensorCommunicator(perspective)
        finally:
            if old_ffn is None:
                os.environ.pop(ffn_base, None)
            else:
                os.environ[ffn_base] = old_ffn
            if old_attn is None:
                os.environ.pop(attn_base, None)
            else:
                os.environ[attn_base] = old_attn
        return comm

    raise ValueError(f"afd_per_mb_channel: unknown backend {comm_backend}")


_per_mb_channel_set: Optional[MultiMbChannelSet] = None


def get_per_mb_channel_set(num_mb: int) -> MultiMbChannelSet:
    """Process-wide cached factory for ``MultiMbChannelSet``.

    The cache always holds ``afd_micro_batch`` channels (the maximum M
    the operator configured).  Callers ask for ``num_mb <= afd_micro_batch``
    and get a view over the first ``num_mb`` channels; this lets the
    runtime adapt M down (e.g. M=1 during warmup) without rebuilding
    the cache or tearing down the IPC handshake.
    """
    global _per_mb_channel_set
    server_args = get_global_server_args()
    max_mb = max(int(getattr(server_args, "afd_micro_batch", 1) or 1), num_mb)

    if _per_mb_channel_set is not None:
        if _per_mb_channel_set.num_mb >= num_mb:
            if _per_mb_channel_set.num_mb == num_mb:
                return _per_mb_channel_set
            # Return a thin view with only the first num_mb channels.
            view_channels = _per_mb_channel_set.channels[:num_mb]
            return MultiMbChannelSet(num_mb=num_mb, channels=view_channels)
        raise RuntimeError(
            f"get_per_mb_channel_set: requested num_mb={num_mb} but cache "
            f"only has {_per_mb_channel_set.num_mb}.  Increase "
            "--afd-micro-batch."
        )

    from sglang.srt.layers.afd import AsyncTensorCommunicator

    channels: List[PerMbChannel] = []

    if max_mb > 1:
        # Two-phase channel creation for FFN side to avoid deadlock:
        # Phase 1: FFN creates all listeners (non-blocking), Attn connects.
        # Phase 2: FFN waits for all connections to complete.
        #
        # For Attn side, connect() already retries, so serial is fine.
        from sglang.srt.layers.afd import get_afd_perspective
        from sglang.srt.layers.afd_type import AFDPerspective
        perspective = get_afd_perspective()
        is_ffn = (perspective == AFDPerspective.AFD_PERSPECTIVE_FFN)

        if is_ffn:
            # Phase 1: Create all inner comms with defer_connect=True
            # This only creates the _p2p object without calling connect(),
            # then we manually start_listen() on each.
            inners = []
            for mb_id in range(max_mb):
                t0 = time.time()
                inner = _make_inner_comm_for_mb(mb_id, defer_connect=True)
                inner.start_listen()
                inners.append((mb_id, inner, t0))
                logger.info(
                    "afd_per_mb_channel: FFN mb=%d listener started", mb_id
                )

            # Phase 2: Wait for all connections from Attn side
            for mb_id, inner, t0 in inners:
                inner.wait_connected()
                async_comm = AsyncTensorCommunicator(inner)
                channels.append(PerMbChannel(mb_id, async_comm))
                logger.info(
                    "afd_per_mb_channel: built channel mb=%d in %.1fms",
                    mb_id, (time.time() - t0) * 1000,
                )
        else:
            # Attn side: two-phase — create all comms with defer_connect,
            # then connect them serially (UCX retries until FFN listens).
            inners = []
            for mb_id in range(max_mb):
                t0 = time.time()
                inner = _make_inner_comm_for_mb(mb_id, defer_connect=True)
                inners.append((mb_id, inner, t0))
                logger.info(
                    "afd_per_mb_channel: Attn mb=%d created (deferred)", mb_id
                )

            # Now connect each serially
            for mb_id, inner, t0 in inners:
                if inner._p2p is not None:
                    inner._p2p.connect()
                inner._warmup_buffer_pool()
                logger.info("UcxTensorCommunicator: ready (K=%d)", inner._K)
                async_comm = AsyncTensorCommunicator(inner)
                channels.append(PerMbChannel(mb_id, async_comm))
                logger.info(
                    "afd_per_mb_channel: built channel mb=%d in %.1fms",
                    mb_id, (time.time() - t0) * 1000,
                )
    else:
        for mb_id in range(max_mb):
            t0 = time.time()
            inner = _make_inner_comm_for_mb(mb_id)
            async_comm = AsyncTensorCommunicator(inner)
            channels.append(PerMbChannel(mb_id, async_comm))
            logger.info(
                "afd_per_mb_channel: built channel mb=%d in %.1fms",
                mb_id, (time.time() - t0) * 1000,
            )

    _per_mb_channel_set = MultiMbChannelSet(num_mb=max_mb, channels=channels)
    if max_mb == num_mb:
        return _per_mb_channel_set
    view_channels = _per_mb_channel_set.channels[:num_mb]
    return MultiMbChannelSet(num_mb=num_mb, channels=view_channels)


def reset_per_mb_channel_set():
    """Test helper — drop the cached channel set so the next call rebuilds."""
    global _per_mb_channel_set
    if _per_mb_channel_set is not None:
        try:
            _per_mb_channel_set.drain()
        except Exception:
            pass
    _per_mb_channel_set = None
