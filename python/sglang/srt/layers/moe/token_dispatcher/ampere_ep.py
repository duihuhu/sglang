"""Ampere (SM80+) MoE EP dispatcher using FlashInfer MoeAlltoAll.

This backend reuses FlashInfer's rank-to-rank token dispatch/combine kernels
(which run on SM80) while pairing with standard MoE runners such as Triton,
unlike ``moe_a2a_backend=flashinfer`` which requires Blackwell MoE runners.
"""

from __future__ import annotations

from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
    FlashinferCombineInput,
    FlashinferDispatcher,
    FlashinferDispatchOutput,
)

__all__ = [
    "AmpereEPCombineInput",
    "AmpereEPDispatcher",
    "AmpereEPDispatchOutput",
]

# Reuse FlashInfer dispatch/combine tensor layouts for the Triton runner path.
AmpereEPDispatchOutput = FlashinferDispatchOutput
AmpereEPCombineInput = FlashinferCombineInput


class AmpereEPDispatcher(FlashinferDispatcher):
    """FlashInfer MoeAlltoAll dispatcher tuned for Ampere-class GPUs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Triton/deep_gemm runners consume activations directly instead of writing
        # into the FlashInfer workspace combine buffer used by cutlass.
        self.payload_in_workspace = False
        self.invalid_token_expert_id = self.num_experts
