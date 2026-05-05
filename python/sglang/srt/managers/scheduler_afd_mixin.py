"""AFD (Attention-FFN Disaggregation) scheduler mixin.

Provides reusable AFD scheduling helpers that can be mixed into any event loop
(normal, disagg_prefill, disagg_decode, etc.).
"""

from __future__ import annotations

import logging
from collections import deque
from typing import TYPE_CHECKING, Optional

import zmq

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

logger = logging.getLogger(__name__)


class SchedulerAFDMixin:
    """Mixin providing AFD helpers for scheduler event loops.

    Usage: call these methods from within any event loop to add AFD support.
    The host scheduler must have `self.afd_send_to_ffn` and `self.afd_recv_from_attn`
    initialized (done in `init_ipc_channels`).
    """

    def afd_init_state(self: "Scheduler"):
        """Initialize per-loop AFD state. Call once at the start of event_loop."""
        self._afd_batchsize_attn = None
        self._afd_forward_mode = None
        self._afd_req_ids = None
        self._afd_pending_batch_infos: deque = deque()

        self._afd_poller = None
        from sglang.srt.layers.afd import afd_is_ffn

        if afd_is_ffn() and getattr(self, "afd_recv_from_attn", None) is not None:
            self._afd_poller = zmq.Poller()
            self._afd_poller.register(self.afd_recv_from_attn, zmq.POLLIN)

    def afd_recv_messages(self: "Scheduler"):
        """Poll for messages from the Attn scheduler.

        Returns ALL messages (including AFDReqInput) so they can be
        broadcast to all TP ranks. AFDReqInput state is extracted later
        in _afd_process_input_requests on every rank.
        """
        recv_socket = getattr(self, "afd_recv_from_attn", None)
        if recv_socket is None:
            return []
        extra_reqs = []
        while True:
            try:
                msg = recv_socket.recv_pyobj(zmq.NOBLOCK)
                extra_reqs.append(msg)
            except zmq.ZMQError:
                break
        return extra_reqs

    def afd_ffn_should_wait(self: "Scheduler") -> bool:
        """Return True if FFN side should wait for Attn sync before running a batch."""
        from sglang.srt.layers.afd import afd_is_ffn

        if afd_is_ffn() and self._afd_batchsize_attn is None:
            if self._afd_poller is not None:
                self._afd_poller.poll(timeout=10)
            return True
        return False

    def afd_send_batch_info(self: "Scheduler", batch: "ScheduleBatch"):
        """Attn side: send AFDReqInput to FFN so it knows the current batch."""
        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.managers.io_struct import AFDReqInput

        if not afd_is_attn():
            return
        send_socket = getattr(self, "afd_send_to_ffn", None)
        if send_socket is None:
            return

        afd_req = AFDReqInput(
            batch_size=batch.batch_size(),
            forward_mode=batch.forward_mode,
            req_ids=[r.rid for r in batch.reqs],
            seq_lens=[r.extend_input_len + r.seqlen for r in batch.reqs],
            extend_lens=(
                batch.extend_lens if hasattr(batch, "extend_lens") else None
            ),
            max_input_len=max(
                (r.extend_input_len for r in batch.reqs), default=0
            ),
            repr_output_len=int(sum(
                max(r.seqlen - len(r.origin_input_ids), 1) for r in batch.reqs
            ) / max(len(batch.reqs), 1)) if batch.reqs else 1,
            # PD+AF decode: FFN needs output_ids so fill_ids matches Attn side
            output_ids_per_req=[list(r.output_ids) for r in batch.reqs],
            # FFN-side Req creation: carry origin_input_ids and max_new_tokens
            input_ids_per_req=[list(r.origin_input_ids) for r in batch.reqs],
            max_new_tokens_per_req=[r.sampling_params.max_new_tokens for r in batch.reqs],
        )
        send_socket.send_pyobj(afd_req)

    def afd_prepare_overlap(self: "Scheduler", batch: "ScheduleBatch"):
        """Compute microbatch split points for AFD on the batch."""
        from sglang.srt.batch_overlap.afd_overlap import _split_seq_indices_m_way
        from sglang.srt.layers.afd import get_afd_micro_batch
        from sglang.srt.model_executor.forward_batch_info import ForwardMode

        m = get_afd_micro_batch()
        if batch.batch_size() < m:
            batch.afd_split_seq_index = None
            return

        forward_mode = batch.forward_mode
        if forward_mode == ForwardMode.EXTEND:
            extend_lens = batch.extend_lens
            split_indices = _split_seq_indices_m_way(len(extend_lens), m, extend_lens)
        elif forward_mode.is_decode() or forward_mode.is_target_verify():
            split_indices = _split_seq_indices_m_way(batch.batch_size(), m, None)
        else:
            return

        batch.afd_split_seq_index = split_indices

    def afd_forward_work_requests(self: "Scheduler", recv_reqs):
        """Attn side: forward work requests to FFN via ZMQ."""
        from sglang.srt.layers.afd import afd_is_attn
        from sglang.srt.managers.io_struct import (
            AFDReqInput,
            BatchTokenizedGenerateReqInput,
            TokenizedEmbeddingReqInput,
            TokenizedGenerateReqInput,
        )

        if not afd_is_attn():
            return
        send_socket = getattr(self, "afd_send_to_ffn", None)
        if send_socket is None:
            return

        for recv_req in recv_reqs:
            if isinstance(recv_req, AFDReqInput):
                continue
            if isinstance(recv_req, BatchTokenizedGenerateReqInput):
                for sub_req in recv_req:
                    send_socket.send_pyobj(sub_req)
            elif isinstance(
                recv_req, (TokenizedGenerateReqInput, TokenizedEmbeddingReqInput)
            ):
                send_socket.send_pyobj(recv_req)

    def afd_reset_state(self: "Scheduler"):
        """Reset per-iteration AFD state after processing a batch."""
        self._afd_batchsize_attn = None
        self._afd_forward_mode = None
        self._afd_req_ids = None
