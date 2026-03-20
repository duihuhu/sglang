"""AFD (Attention-FFN Disaggregation) Mixin and utilities for multi-model support.

Provides:
- AFDDecoderLayerMixin: mix into any DecoderLayer to gain AFD capability
- AFDWeightFilter: generic weight filter for selective loading
"""

import logging
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

logger = logging.getLogger(__name__)


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
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        if hidden_states.shape[0] != 0:
            hidden_states = self._run_attn(positions, hidden_states, forward_batch)
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual

    def forward_afd_F(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """FFN stage: mlp -> postprocess_layer."""
        hidden_states = self._run_mlp(hidden_states, forward_batch)
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
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
