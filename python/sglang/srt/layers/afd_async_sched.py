"""Data-driven AFD scheduler.

Replaces the static ``AFDStageScheduleGenerator`` two-phase order with a
fully event-driven loop:

* Each micro-batch is treated as an independent layer chain.
* Compute runs as soon as that mb's recv dependency is satisfied.
* As soon as compute finishes, the result is dispatched fire-and-forget
  on that mb's per-mb channel (``send_async`` inside
  ``prepare_mlp``/``postprocess_layer``).
* The main thread blocks only when every mb is stalled on a recv.

This removes the layer-boundary alignment of the legacy schedule.

Per-mb chain (same on both sides):

    A(0) → F(0) → A(1) → F(1) → ... → A(N-1) → F(N-1)

Where the role of each step depends on perspective:

==========  ==========================  ==========================
Step type   ATTN side (DA)              FFN side (DF)
==========  ==========================  ==========================
A(L)        compute attn + send         no-op + recv (from DA)
F(L)        no-op + recv (from DF)      compute MLP + send
==========  ==========================  ==========================

So each mb chain alternates "compute-and-send" and "recv-and-merge"
steps; whichever is which depends on perspective.  The driver only
needs to know which steps require a recv before they can start.
"""

from __future__ import annotations

import enum
import logging
import os
import queue
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch

from sglang.srt.layers.afd_type import AFDPerspective

logger = logging.getLogger(__name__)


class _State(enum.Enum):
    READY_TO_COMPUTE = enum.auto()
    WAIT_RECV = enum.auto()
    DONE = enum.auto()


@dataclass
class _MbState:
    mb_id: int
    state: _State
    next_layer: int = 0
    next_stage: str = "A"  # "A" or "F"
    hidden_states: Optional[torch.Tensor] = None
    residual: Optional[torch.Tensor] = None


class AsyncMbDriver:
    """Event-driven AFD execution loop."""

    def __init__(
        self,
        perspective: AFDPerspective,
        layers,
        num_layers: int,
        m_stage: int,
        channels,
        input_arrs: List[Dict],
        detailed_timeline: Optional[list] = None,
    ):
        self.perspective = perspective
        self.layers = layers
        self.num_layers = num_layers
        self.m_stage = m_stage
        self.channels = channels
        self.input_arrs = input_arrs
        self.detailed_timeline = detailed_timeline

        self._is_attn = perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN

        # Initialize per-mb state. Both sides start with the same input
        # hidden_states/residual snapshots (DF's will be overwritten by
        # the first recv inside forward_afd_A's prepare_mlp).
        self.mbs: List[_MbState] = []
        for mb_id in range(m_stage):
            inp = input_arrs[mb_id]
            mb = _MbState(
                mb_id=mb_id,
                state=_State.READY_TO_COMPUTE,
                next_layer=0,
                next_stage="A",
                hidden_states=inp["hidden_states"],
                residual=inp["residual"],
            )
            # Decide initial state based on whether the very first step
            # needs a recv.
            if self._needs_recv(stage="A"):
                mb.state = _State.WAIT_RECV
            self.mbs.append(mb)

        self._recv_done: "queue.Queue[int]" = queue.Queue()

    # ── public API ────────────────────────────────────────────────────
    def run(self) -> List[torch.Tensor]:
        # Issue initial recvs for any mb starting in WAIT_RECV.
        for mb in self.mbs:
            if mb.state == _State.WAIT_RECV:
                self._issue_recv(mb)

        while not all(mb.state == _State.DONE for mb in self.mbs):
            ready = [
                mb for mb in self.mbs if mb.state == _State.READY_TO_COMPUTE
            ]
            if ready:
                ready.sort(key=lambda m: (m.next_layer, m.mb_id))
                self._step(ready[0])
            else:
                mb_id = self._recv_done.get()
                self.mbs[mb_id].state = _State.READY_TO_COMPUTE

        return self.mbs  # caller reads mb.hidden_states / mb.residual

    # ── core step ─────────────────────────────────────────────────────
    def _needs_recv(self, stage: str) -> bool:
        """Does the upcoming ``stage`` for this perspective require a
        recv to land before compute can proceed?

        Truth table:
          DA, A-step → compute attn + send         → no recv
          DA, F-step → recv from DF + merge        → YES recv
          DF, A-step → recv from DA + (no-op attn) → YES recv
          DF, F-step → compute MLP + send back     → no recv
        """
        if stage == "A":
            return not self._is_attn
        return self._is_attn  # stage == "F"

    def _step(self, mb: _MbState):
        layer_id = mb.next_layer
        layer = self.layers[layer_id]
        inp = self.input_arrs[mb.mb_id]
        forward_batch = inp["forward_batch"]
        positions = inp["positions"]

        # Per-mb override for the AFDCommunicator's send/recv calls.
        chan = self.channels[mb.mb_id]
        from sglang.srt.layers import afd as _afd
        prev_override = getattr(_afd, "_per_mb_async_override", None)
        _afd._per_mb_async_override = chan.async_comm
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "AsyncMbDriver: step mb=%d layer=%d stage=%s "
                "override_inner_id=%s",
                mb.mb_id, layer_id, mb.next_stage,
                id(getattr(chan.async_comm, "inner", None)),
            )

        from sglang.srt.layers.afd_mixin import _afd_ctx, _afd_host_events
        _afd_ctx["mb"] = mb.mb_id
        _afd_ctx["layer"] = layer_id
        _afd_ctx["stage"] = mb.next_stage

        t0 = time.perf_counter()
        try:
            _afd_host_events.append({
                "ts_ms": round(time.time() * 1000, 3),
                "role": "DRIVER", "layer": layer_id, "mb": mb.mb_id,
                "event": "ready_pick", "stage": mb.next_stage,
            })
        except Exception:
            pass
        try:
            if mb.next_stage == "A":
                hs, res = layer.forward_afd_A(
                    positions, mb.hidden_states, forward_batch, mb.residual,
                )
            else:
                hs, res = layer.forward_afd_F(
                    mb.hidden_states, forward_batch, mb.residual,
                )
        finally:
            _afd._per_mb_async_override = prev_override

        t1 = time.perf_counter()
        mb.hidden_states = hs
        mb.residual = res

        if self.detailed_timeline is not None:
            stage_name = (
                "AFD_FORWARD_STAGE_A" if mb.next_stage == "A"
                else "AFD_FORWARD_STAGE_F"
            )
            self.detailed_timeline.append({
                "step": len(self.detailed_timeline),
                "stage": stage_name,
                "layer": layer_id,
                "mb": mb.mb_id,
                "t_start_ms": t0 * 1000,
                "t_end_ms": t1 * 1000,
                "dur_ms": (t1 - t0) * 1000,
            })

        # Advance
        if mb.next_stage == "A":
            mb.next_stage = "F"
        else:
            mb.next_stage = "A"
            mb.next_layer += 1

        if mb.next_layer >= self.num_layers:
            mb.state = _State.DONE
            return

        if self._needs_recv(mb.next_stage):
            mb.state = _State.WAIT_RECV
            self._issue_recv(mb)
        else:
            mb.state = _State.READY_TO_COMPUTE

    # ── recv plumbing ────────────────────────────────────────────────
    def _issue_recv(self, mb: _MbState):
        chan = self.channels[mb.mb_id]
        chan.recv_start(self._on_recv_arrived)

    def _on_recv_arrived(self, mb_id: int):
        # Fire-and-forget — main loop will pick this up via queue.get().
        self._recv_done.put(mb_id)


__all__ = ["AsyncMbDriver"]
