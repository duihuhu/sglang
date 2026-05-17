"""AFD (Attention-FFN Disaggregation) Mixin and utilities for multi-model support.

Provides:
- AFDDecoderLayerMixin: mix into any DecoderLayer to gain AFD capability
- AFDWeightFilter: generic weight filter for selective loading
"""

import logging
import time
from typing import Optional, Tuple

import torch

from sglang.srt.layers.afd import (
    AFDCommunicator,
    AFDProxyAttention,
    AFDProxyMLP,
    get_afd_perspective,
)
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_npu

logger = logging.getLogger(__name__)
_is_npu = is_npu()

# ── AFD TPOT breakdown timing ──────────────────────────────────────────────
# Global accumulator reset per forward pass from model_forward_afd()
_afd_timing_records: list = []  # list of dicts with per-stage timing breakdown
_afd_timing_enabled: bool = True

# Scheduler-level wall-clock anchors for cross-GPU latency breakdown
_afd_sched_ts: dict = {}  # keys: zmq_sent, zmq_recv, forward_start, forward_end

# Host wall-clock events for send/recv pipeline breakdown
# Each entry: {"ts_ms": float, "role": "DA"|"DF", "layer": int, "mb": int,
#               "event": str, "stage": "A"|"F"}
_afd_host_events: list = []

# Pipeline context for host event labeling (set by model_forward_afd loop)
# Mutable dict so assignments are visible across all importers
_afd_ctx: dict = {"mb": -1, "layer": -1, "stage": ""}



class AFDDecoderLayerMixin:
    """Mixin that adds AFD (A/F disaggregation) capability to any DecoderLayer.

    Usage::

        class MyDecoderLayer(AFDDecoderLayerMixin, nn.Module):
            def __init__(self, ...):
                super().__init__()
                ...  # original init
                self._afd_init()  # call at the end

    The mixin expects the host class to have:
    - self.self_attn: the attention module
    - self.mlp: the MLP / MoE module
    - self.layer_communicator: a LayerCommunicator instance
    - self.layer_id: int (for logging)
    """

    def _afd_init(self):
        """Inject AFD communicator and proxy modules. Call at the end of __init__."""
        import types

        perspective = get_afd_perspective()
        if perspective is None:
            return

        if perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            self.self_attn = AFDProxyAttention()
        elif perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            self.mlp = AFDProxyMLP()

        self.layer_communicator = AFDCommunicator(
            layer_communicator=self.layer_communicator,
            perspective=perspective,
            layer_id=getattr(self, "layer_id", -1),
        )

        # Bind default _run_attn/_run_mlp if the host class doesn't define them
        # (needed when the host does not inherit AFDDecoderLayerMixin).
        if not hasattr(self, "_run_attn"):
            self._run_attn = types.MethodType(AFDDecoderLayerMixin._run_attn, self)
        if not hasattr(self, "_run_mlp"):
            self._run_mlp = types.MethodType(AFDDecoderLayerMixin._run_mlp, self)

        from sglang.srt.server_args import get_global_server_args

        server_args = get_global_server_args()
        if getattr(server_args, "enable_torch_compile", False):
            if perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
                self._run_attn = torch.compile(
                    self._run_attn, dynamic=True, disable=_is_npu
                )
            elif perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
                self._run_mlp = torch.compile(
                    self._run_mlp, dynamic=True, disable=_is_npu
                )
            logger.info(
                "AFD layer %d: torch.compile enabled for %s",
                getattr(self, "layer_id", -1),
                "_run_attn" if perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN else "_run_mlp",
            )

    # --- Overridable hooks for model-specific Attention / MLP interfaces ---

    def _run_attn(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Default attention call. Override for models with extra args."""
        return self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )

    def _run_mlp(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Default MLP call. Override for models with extra args."""
        return self.mlp(hidden_states, forward_batch)

    # --- Generic AFD forward stages ---

    def forward_afd_A(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Attention stage: prepare_attn -> attn -> prepare_mlp."""
        global _afd_timing_records, _afd_timing_enabled

        ev_prep_attn_start = ev_prep_attn_end = None
        ev_attn_start = ev_attn_end = None
        ev_prep_mlp_start = ev_prep_mlp_end = None

        if _afd_timing_enabled and torch.cuda.is_available():
            ev_prep_attn_start = torch.cuda.Event(enable_timing=True)
            ev_prep_attn_end = torch.cuda.Event(enable_timing=True)
            ev_prep_attn_start.record()

        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )

        if ev_prep_attn_end is not None:
            ev_prep_attn_end.record()
            ev_attn_start = torch.cuda.Event(enable_timing=True)
            ev_attn_end = torch.cuda.Event(enable_timing=True)
            ev_attn_start.record()

        if hidden_states.shape[0] != 0:
            hidden_states = self._run_attn(positions, hidden_states, forward_batch)

        if ev_attn_end is not None:
            ev_attn_end.record()
            ev_prep_mlp_start = torch.cuda.Event(enable_timing=True)
            ev_prep_mlp_end = torch.cuda.Event(enable_timing=True)
            ev_prep_mlp_start.record()

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        if ev_prep_mlp_end is not None:
            ev_prep_mlp_end.record()
            _afd_timing_records.append({
                "stage": "A",
                "layer_id": getattr(self, "layer_id", -1),
                "perspective": "attn" if get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_ATTN else "ffn",
                "events": {
                    "prep_attn": (ev_prep_attn_start, ev_prep_attn_end),
                    "attn": (ev_attn_start, ev_attn_end),
                    "prep_mlp": (ev_prep_mlp_start, ev_prep_mlp_end),
                },
            })

        return hidden_states, residual

    def forward_afd_F(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FFN stage: mlp -> postprocess_layer."""
        global _afd_timing_records, _afd_timing_enabled

        ev_mlp_start = ev_mlp_end = None
        ev_post_start = ev_post_end = None

        if _afd_timing_enabled and torch.cuda.is_available():
            ev_mlp_start = torch.cuda.Event(enable_timing=True)
            ev_mlp_end = torch.cuda.Event(enable_timing=True)
            ev_mlp_start.record()

        is_attn = get_afd_perspective() == AFDPerspective.AFD_PERSPECTIVE_ATTN
        bsz = hidden_states.shape[0]
        hidden_dim = hidden_states.shape[1]
        dtype_sz = hidden_states.element_size()

        hidden_states = self._run_mlp(hidden_states, forward_batch)

        if ev_mlp_end is not None:
            ev_mlp_end.record()
            ev_post_start = torch.cuda.Event(enable_timing=True)
            ev_post_end = torch.cuda.Event(enable_timing=True)
            ev_post_start.record()

        if is_attn:
            _layer_id = getattr(self, "layer_id", -1)
            tx_kb = bsz * hidden_dim * dtype_sz / 1024
            print(f"[AFD_DBG] L{_layer_id:>2} batch={bsz:>2} | DA→DF {tx_kb:>5.0f}KB | DA←DF {tx_kb:>5.0f}KB", flush=True)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        if ev_post_end is not None:
            ev_post_end.record()
            _afd_timing_records.append({
                "stage": "F",
                "layer_id": getattr(self, "layer_id", -1),
                "perspective": "attn" if is_attn else "ffn",
                "events": {
                    "mlp": (ev_mlp_start, ev_mlp_end),
                    "postprocess": (ev_post_start, ev_post_end),
                },
            })

        return hidden_states, residual


class AFDWeightFilter:
    """Generic weight filter for AFD selective loading.

    Attn nodes skip FFN/expert weights. FFN nodes skip attention weights.
    Shared weights (embed, norm, lm_head) are always loaded.
    """

    ATTN_PATTERNS = [
        "self_attn",
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "qkv_proj",
        "kv_a_proj_with_mqa",
        "kv_b_proj",
        "q_a_proj",
        "q_b_proj",
    ]

    FFN_PATTERNS = [
        "mlp",
        "expert",
        "gate_proj",
        "up_proj",
        "down_proj",
        "gate_up_proj",
        "shared_expert",
    ]

    SHARED_PATTERNS = [
        "embed_tokens",
        "lm_head",
        "norm",
        "layernorm",
        "input_layernorm",
        "post_attention_layernorm",
    ]

    @classmethod
    def should_load(cls, param_name: str, perspective: AFDPerspective) -> bool:
        name_lower = param_name.lower()

        if any(p in name_lower for p in cls.SHARED_PATTERNS):
            return True

        is_attn_weight = any(p in name_lower for p in cls.ATTN_PATTERNS)
        is_ffn_weight = any(p in name_lower for p in cls.FFN_PATTERNS)

        if perspective == AFDPerspective.AFD_PERSPECTIVE_ATTN:
            if is_ffn_weight and not is_attn_weight:
                return False
            return True
        elif perspective == AFDPerspective.AFD_PERSPECTIVE_FFN:
            if is_attn_weight and not is_ffn_weight:
                return False
            return True
        return True
