"""Cross-layer async pipeline executor for AFD micro-batch overlap.

Replaces the serial schedule-driven loop with a multi-stream pipeline that
overlaps DA's Attention compute with DF's FFN compute across layers.

Key design:
- Each micro-batch gets its own CUDA stream for compute.
- Communication (send/recv) uses a shared comm_stream.
- CUDA events establish minimal dependencies (no CPU blocking on GPU ops).
- CPU only blocks on SHM flag polling (~77us per recv in ipc_event mode).

Data flow per layer (DA side, M=2):
  stream_mb0: [wait recv_event] → Attn(mb0) → [record compute_done]
  stream_mb1: [wait recv_event] → Attn(mb1) → [record compute_done]
  comm_stream: [wait compute_done(mb0)] → send(mb0) → [wait compute_done(mb1)] → send(mb1)
               → poll_recv(mb0) → copy → [record recv_event(mb0)]
               → poll_recv(mb1) → copy → [record recv_event(mb1)]
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import torch

from sglang.srt.layers.afd import (
    afd_is_attn,
    afd_is_ffn,
    get_async_communicator,
    get_tensor_communicator,
)
from sglang.srt.layers.afd_mixin import AFDDecoderLayerMixin
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)


class AsyncPipelineExecutor:
    """Multi-stream pipeline executor for AFD micro-batch overlap."""

    def __init__(
        self,
        layers,
        m_stage: int,
        input_arrs: List[Dict],
        perspective: AFDPerspective,
    ):
        self.layers = layers
        self.num_layers = len(layers)
        self.m_stage = m_stage
        self.input_arrs = input_arrs
        self.perspective = perspective
        self._is_attn = perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN

        device = torch.cuda.current_device()

        # Per-micro-batch compute streams
        self.mb_streams = [
            torch.cuda.Stream(device=device) for _ in range(m_stage)
        ]
        # Shared communication stream (high priority)
        self.comm_stream = torch.cuda.Stream(device=device, priority=-1)

        # Pre-allocate events: compute_done[mb] and recv_ready[layer][mb]
        self.compute_done_events = [
            torch.cuda.Event() for _ in range(m_stage)
        ]
        # recv_ready_events[layer][mb] — signals that recv data is on comm_stream
        self.recv_ready_events = [
            [torch.cuda.Event() for _ in range(m_stage)]
            for _ in range(self.num_layers)
        ]

        # Get the raw IPC communicator for stream-ordered ops
        self._inner_comm = get_tensor_communicator()

    def run(self) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """Execute the full pipeline and return per-mb final (hidden, residual)."""
        if self._is_attn:
            return self._run_attn_side()
        else:
            return self._run_ffn_side()

    def _run_attn_side(self) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """DA (Attention) side — interleaved pipeline.

        Key insight: after sending mb0's Attn result to DF, DA does NOT wait
        for mb0's FFN result. Instead, it immediately computes mb1's Attention
        (which only depends on mb1's PREVIOUS layer FFN result, not mb0's).

        Timeline (M=2):
          DA: A_L0(mb0) → send(mb0) → A_L0(mb1) → send(mb1) → recv(mb0) → A_L1(mb0) → send(mb0) → recv(mb1) → A_L1(mb1) → ...
          DF:                          F_L0(mb0) →                          F_L0(mb1) → F_L1(mb0) → ...

        The overlap: while DA computes A_L0(mb1), DF computes F_L0(mb0) in parallel.
        When DA finishes A_L0(mb1) and polls recv(mb0), DF has likely already
        finished F_L0(mb0) — so the poll returns immediately (no GPU idle time).
        """
        mb_hidden = [self.input_arrs[i]["hidden_states"] for i in range(self.m_stage)]
        mb_residual = [self.input_arrs[i]["residual"] for i in range(self.m_stage)]

        for layer_id in range(self.num_layers):
            layer = self.layers[layer_id]
            inner_lc = layer.layer_communicator.layer_communicator

            # Interleaved: compute Attn(mb_i) → send(mb_i) → recv(mb_{i-1})
            for mb in range(self.m_stage):
                # Compute Attention for this mb
                hs, res = AFDDecoderLayerMixin.forward_afd_A_compute(
                    layer,
                    self.input_arrs[mb]["positions"],
                    mb_hidden[mb],
                    self.input_arrs[mb]["forward_batch"],
                    mb_residual[mb],
                )
                mb_hidden[mb] = hs
                mb_residual[mb] = res

                # Send this mb's result to DF immediately
                self._send_tensor(mb_hidden[mb])

                # After sending mb_i, recv the PREVIOUS mb's FFN result
                # (it has been computing on DF while we did Attn for this mb)
                if mb > 0:
                    prev_mb = mb - 1
                    recv_hs = self._recv_tensor_default_stream()
                    recv_hs, mb_residual[prev_mb] = inner_lc.postprocess_layer(
                        recv_hs, mb_residual[prev_mb],
                        self.input_arrs[prev_mb]["forward_batch"],
                    )
                    mb_hidden[prev_mb] = recv_hs

            # Recv the last mb's FFN result
            last_mb = self.m_stage - 1
            recv_hs = self._recv_tensor_default_stream()
            recv_hs, mb_residual[last_mb] = inner_lc.postprocess_layer(
                recv_hs, mb_residual[last_mb],
                self.input_arrs[last_mb]["forward_batch"],
            )
            mb_hidden[last_mb] = recv_hs

        results = []
        for mb in range(self.m_stage):
            results.append((mb_hidden[mb], mb_residual[mb]))
        return results

    def _run_ffn_side(self) -> List[Tuple[torch.Tensor, Optional[torch.Tensor]]]:
        """DF (FFN) side — matches DA's interleaved send order.

        DA sends in order: mb0, mb1, mb0, mb1, ... (per layer).
        DF receives in the same order, computes FFN, sends back immediately.
        """
        mb_hidden = [self.input_arrs[i]["hidden_states"] for i in range(self.m_stage)]
        mb_residual = [self.input_arrs[i]["residual"] for i in range(self.m_stage)]

        for layer_id in range(self.num_layers):
            layer = self.layers[layer_id]

            # DF receives and processes each mb in the order DA sent them
            for mb in range(self.m_stage):
                # Recv this mb's Attn output from DA
                mb_hidden[mb] = self._recv_tensor_default_stream()

                # Compute FFN
                hs, res = AFDDecoderLayerMixin.forward_afd_F_compute(
                    layer,
                    mb_hidden[mb],
                    self.input_arrs[mb]["forward_batch"],
                    mb_residual[mb],
                )
                mb_hidden[mb] = hs
                mb_residual[mb] = res

                # Send result back to DA immediately
                self._send_tensor(mb_hidden[mb])

        results = []
        for mb in range(self.m_stage):
            results.append((mb_hidden[mb], mb_residual[mb]))
        return results

    def _send_tensor(self, x: torch.Tensor):
        """Send tensor — uses GPU-only path if available, else stream-ordered."""
        comm = self._inner_comm
        if hasattr(comm, 'send_tensor_gpu_only'):
            comm.send_tensor_gpu_only(x)
        elif hasattr(comm, 'send_stream_ordered'):
            comm.send_stream_ordered(x)
        else:
            comm.send_tensor(x)

    def _recv_tensor_default_stream(self) -> torch.Tensor:
        """Recv tensor on default stream. Uses recv_zero_sync for minimal overhead."""
        comm = self._inner_comm
        if hasattr(comm, 'recv_zero_sync'):
            result = comm.recv_zero_sync()
        else:
            result = comm.recv_tensor()
        if not result.is_contiguous():
            result = result.contiguous()
        return result

    def _recv_tensor_on_stream(self, stream: torch.cuda.Stream) -> torch.Tensor:
        """Recv tensor with minimal CPU blocking (SHM poll only)."""
        comm = self._inner_comm
        if hasattr(comm, 'recv_poll_and_enqueue'):
            tensor, ev = comm.recv_poll_and_enqueue(stream)
            return tensor
        else:
            # Fallback: use recv_zero_sync or recv_tensor on the given stream
            with torch.cuda.stream(stream):
                if hasattr(comm, 'recv_zero_sync'):
                    return comm.recv_zero_sync()
                else:
                    return comm.recv_tensor()
