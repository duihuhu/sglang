from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional

import torch
import triton.language as tl

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.utils import cpu_has_amx_support, is_cpu, is_cuda, is_hip, is_xpu

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
        DeepEPLLDispatchOutput,
        DeepEPNormalCombineInput,
        DeepEPNormalDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
        FlashinferDispatchOutput,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


_is_hip = is_hip()
_is_cuda = is_cuda()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_use_aiter = bool(int(os.getenv("SGLANG_USE_AITER", "0")))
_is_xpu = is_xpu()
_MOE_PADDING_SIZE = 128 if bool(int(os.getenv("SGLANG_MOE_PADDING", "0"))) else 0


if _is_cuda or _is_hip:
    from sgl_kernel import gelu_and_mul, silu_and_mul

    if _is_hip:
        _has_vllm = False
        if _use_aiter:
            try:
                from aiter import moe_sum
            except ImportError:
                raise ImportError(
                    "aiter is required when SGLANG_USE_AITER is set to True"
                )
        else:
            try:
                from vllm import _custom_ops as vllm_ops  # moe_sum

                _has_vllm = True
            except ImportError:
                # Fallback: vllm not available, will use triton moe_sum
                _has_vllm = False
elif _is_cpu and _is_cpu_amx_available:
    pass
elif _is_xpu:
    from sgl_kernel import moe_sum_reduce, silu_and_mul


if _is_cuda or _is_hip or _is_xpu:
    from sgl_kernel import (  # noqa: F401
        moe_align_block_size as sgl_moe_align_block_size,
    )


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
    ) -> TritonRunnerOutput:

        # TODO: move these functions to the triton runner
        from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
            _swiglu_gpt_oss_sigmoid_alpha,
            _swiglu_silu_clamp_mul,
            invoke_fused_moe_kernel,
            moe_sum_reduce_torch_compile,
            moe_sum_reduce_triton,
        )

        hidden_states = runner_input.hidden_states
        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        sorted_token_ids = runner_input.sorted_token_ids
        expert_ids = runner_input.expert_ids
        num_tokens_post_padded = runner_input.num_tokens_post_padded

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        b13 = quant_info.b13
        b2 = quant_info.b2
        a13_scale = quant_info.a13_scale
        a2_scale = quant_info.a2_scale
        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        w13_zp = quant_info.w13_zp
        w2_zp = quant_info.w2_zp
        block_shape = quant_info.block_shape
        per_channel_quant = quant_info.per_channel_quant
        use_fp8_w8a8 = quant_info.use_fp8_w8a8
        use_int8_w8a8 = quant_info.use_int8_w8a8
        use_int8_w8a16 = quant_info.use_int8_w8a16
        use_int4_w4a16 = quant_info.use_int4_w4a16

        activation = self.config.activation
        no_combine = self.config.no_combine
        inplace = self.config.inplace
        gemm1_alpha = self.config.gemm1_alpha
        gemm1_limit = self.config.gemm1_clamp_limit
        routed_scaling_factor = self.config.routed_scaling_factor
        apply_router_weight_on_input = self.config.apply_router_weight_on_input

        assert self.config.is_gated, "Only gated MoEs are supported for Triton runner"

        M = hidden_states.shape[0]
        E, N, _ = w13.shape
        compute_type = (
            tl.bfloat16 if hidden_states.dtype == torch.bfloat16 else tl.float16
        )

        intermediate_cache1 = torch.empty(
            (M, topk_ids.shape[1], N),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        invoke_fused_moe_kernel(
            hidden_states,
            w13,
            b13,
            intermediate_cache1,
            a13_scale,
            w13_scale,
            w13_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            apply_router_weight_on_input,
            topk_ids.shape[1],
            running_state["config"],
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )

        intermediate_cache2 = torch.empty(
            (M * topk_ids.shape[1], N // 2),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if activation == "silu":
            if gemm1_alpha is not None:
                assert gemm1_limit is not None
                intermediate_cache2 = _swiglu_gpt_oss_sigmoid_alpha(
                    intermediate_cache1.view(-1, N), gemm1_alpha, gemm1_limit
                )
            elif gemm1_limit is not None:
                intermediate_cache2 = _swiglu_silu_clamp_mul(
                    intermediate_cache1.view(-1, N), gemm1_limit
                )
            elif _is_cuda or _is_hip or _is_xpu:
                silu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                vllm_ops.silu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        elif activation == "gelu":
            assert gemm1_alpha is None, "gemm1_alpha is not supported for gelu"
            assert gemm1_limit is None, "gemm1_limit is not supported for gelu"
            if _is_cuda or _is_hip:
                gelu_and_mul(intermediate_cache1.view(-1, N), intermediate_cache2)
            else:
                vllm_ops.gelu_and_mul(
                    intermediate_cache2, intermediate_cache1.view(-1, N)
                )
        else:
            raise ValueError(f"Unsupported activation: {activation=}")

        intermediate_cache3 = torch.empty(
            (M, topk_ids.shape[1], w2.shape[1]),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )

        if no_combine:
            assert not inplace
            out_hidden_states = torch.empty(
                (M, topk_ids.shape[1], w2.shape[1]),
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        elif inplace:
            out_hidden_states = hidden_states
        else:
            out_hidden_states = torch.empty_like(hidden_states)

        invoke_fused_moe_kernel(
            intermediate_cache2,
            w2,
            b2,
            (
                intermediate_cache3
                if not no_combine and topk_ids.shape[1] != 1
                else out_hidden_states.unsqueeze(0)
            ),
            a2_scale,
            w2_scale,
            w2_zp,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            not apply_router_weight_on_input,
            1,
            running_state["config"],
            compute_type=compute_type,
            use_fp8_w8a8=use_fp8_w8a8,
            use_int8_w8a8=use_int8_w8a8,
            use_int8_w8a16=use_int8_w8a16,
            use_int4_w4a16=use_int4_w4a16,
            per_channel_quant=per_channel_quant,
            block_shape=block_shape,
        )

        if routed_scaling_factor is None:
            routed_scaling_factor = 1.0

        if no_combine:
            pass
        elif _is_cuda:
            if topk_ids.shape[1] == 1 and routed_scaling_factor == 1.0:
                pass  # we write directly into out_hidden_states
            elif topk_ids.shape[1] == 2 and routed_scaling_factor == 1.0:
                torch.add(
                    intermediate_cache3[:, 0],
                    intermediate_cache3[:, 1],
                    out=out_hidden_states,
                ).squeeze(dim=1)
            else:
                # According to micro benchmark results, torch.compile can get better performance for small token.
                if M <= 32:
                    moe_sum_reduce_torch_compile(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
                else:
                    moe_sum_reduce_triton(
                        intermediate_cache3.view(*intermediate_cache3.shape),
                        out_hidden_states,
                        routed_scaling_factor,
                    )
        elif _is_hip:
            if _use_aiter:
                moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                )
            elif _has_vllm:
                vllm_ops.moe_sum(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                )
            else:
                # Fallback: use triton moe_sum when vllm is not available
                moe_sum_reduce_triton(
                    intermediate_cache3.view(*intermediate_cache3.shape),
                    out_hidden_states,
                    routed_scaling_factor,
                )
        elif _is_xpu:
            moe_sum_reduce(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
                routed_scaling_factor,
            )
        else:
            vllm_ops.moe_sum(
                intermediate_cache3.view(*intermediate_cache3.shape),
                out_hidden_states,
            )

        return TritonRunnerOutput(
            hidden_states=out_hidden_states,
        )

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import fused_experts
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    output = fused_experts(
        hidden_states=dispatch_output.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=runner_config,
        b1=quant_info.b13,
        b2=quant_info.b2,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        w1_scale=quant_info.w13_scale,
        w2_scale=quant_info.w2_scale,
        w1_zp=quant_info.w13_zp,
        w2_zp=quant_info.w2_zp,
        a1_scale=quant_info.a13_scale,
        a2_scale=quant_info.a2_scale,
        block_shape=quant_info.block_shape,
    )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # NOTE: this is dead code as a fused func for standard format is registered.
    # This is left here for testing and examples.

    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        get_config_dtype_str,
        moe_align_block_size,
        try_get_optimal_moe_config,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    num_tokens = hidden_states.shape[0]
    num_local_experts = runner_config.num_local_experts

    if (
        not (quant_info.use_fp8_w8a8 or quant_info.use_int8_w8a8)
        or quant_info.block_shape is not None
        or _use_aiter
    ):
        padding_size = 0
    else:
        padding_size = _MOE_PADDING_SIZE

    config_dtype = get_config_dtype_str(
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        dtype=hidden_states.dtype,
    )

    get_config_func = functools.partial(
        try_get_optimal_moe_config,
        quant_info.w13_weight.shape,
        (
            num_local_experts,
            quant_info.w2_weight.shape[1],
            quant_info.w2_weight.shape[2] - padding_size,
        ),
        topk_output.topk_ids.shape[1],
        config_dtype,
        block_shape=quant_info.block_shape,
        per_channel_quant=quant_info.per_channel_quant,
    )

    config = get_config_func(num_tokens)

    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_output.topk_ids, config["BLOCK_SIZE_M"], num_local_experts
    )

    running_state["config"] = config

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # NOTE: this is dead code as a fused func for standard format is registered.
    # This is left here for testing and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )


def _map_global_expert_ids_to_local(
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    runner_config: MoeRunnerConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global routed expert ids to this EP rank's local weight indices."""
    from sglang.srt.distributed.parallel_state import get_moe_expert_parallel_rank

    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk ids/weights shapes differ: {tuple(topk_ids.shape)} vs "
            f"{tuple(topk_weights.shape)}."
        )
    num_local_experts = runner_config.num_local_experts
    if not num_local_experts:
        raise ValueError("num_local_experts must be positive for EP dispatch.")
    local_start = get_moe_expert_parallel_rank() * num_local_experts
    local_end = local_start + num_local_experts
    valid = (topk_ids >= local_start) & (topk_ids < local_end)
    local_ids = torch.where(valid, topk_ids - local_start, -1).to(torch.int32)
    local_weights = torch.where(valid, topk_weights, 0.0)
    return local_ids, local_weights


@register_pre_permute("flashinfer", "triton")
def pre_permute_flashinfer_to_triton(
    dispatch_output: FlashinferDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    """Adapt FlashInfer's fixed-slot global routing layout to local Triton MoE."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    hidden_states, hidden_states_scale, topk_output, moe_output = dispatch_output
    if hidden_states_scale is not None or hidden_states.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError(
            "Flashinfer A2A with Triton requires BF16/FP16 dispatch output."
        )
    if hidden_states.ndim != 2:
        raise ValueError(
            f"Expected token-major [M, H] hidden states, got {tuple(hidden_states.shape)}."
        )
    if moe_output is not None:
        raise ValueError(
            "Flashinfer workspace output is reserved for the CUTLASS runner and "
            "must be disabled for Triton."
        )
    if runner_config.no_combine:
        raise NotImplementedError(
            "Flashinfer A2A with Triton requires local expert contributions to "
            "be combined before communication combine."
        )
    local_ids, local_weights = _map_global_expert_ids_to_local(
        topk_output.topk_ids, topk_output.topk_weights, runner_config
    )
    if local_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"Received {hidden_states.shape[0]} hidden rows but "
            f"{local_ids.shape[0]} routing rows."
        )
    standard_dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=local_weights,
            topk_ids=local_ids,
            router_logits=hidden_states.new_empty((hidden_states.shape[0], 0)),
        ),
    )
    running_state["flashinfer_hidden_states_shape"] = hidden_states.shape
    return pre_permute_standard_to_triton(
        standard_dispatch_output, quant_info, runner_config, running_state
    )


@register_post_permute("triton", "flashinfer")
def post_permute_triton_to_flashinfer(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> FlashinferCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.flashinfer import (
        FlashinferCombineInput,
    )

    if (
        runner_output.hidden_states.shape
        != running_state["flashinfer_hidden_states_shape"]
    ):
        raise ValueError(
            f"Triton output shape {tuple(runner_output.hidden_states.shape)} does "
            "not preserve Flashinfer fixed-slot shape "
            f"{tuple(running_state['flashinfer_hidden_states_shape'])}."
        )
    return FlashinferCombineInput(hidden_states=runner_output.hidden_states)


@register_pre_permute("deepep_normal", "triton")
def pre_permute_deepep_normal_to_triton(
    dispatch_output: DeepEPNormalDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    """Adapt token-major DeepEP normal output to the generic Triton runner."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    (
        hidden_states,
        hidden_states_scale,
        topk_ids,
        topk_weights,
        num_recv_tokens_per_expert,
    ) = dispatch_output
    if hidden_states_scale is not None or hidden_states.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError(
            "The Triton deepep_normal adapter requires BF16/FP16 dispatch "
            "output; disable FP8 dispatch for the selected A2A backend."
        )
    if hidden_states.ndim != 2:
        raise ValueError(
            "Expected token-major hidden states with shape [num_tokens, "
            f"hidden_size], got {tuple(hidden_states.shape)}."
        )
    if topk_ids.shape != topk_weights.shape:
        raise ValueError(
            f"topk ids/weights shapes differ: {tuple(topk_ids.shape)} vs "
            f"{tuple(topk_weights.shape)}."
        )
    if topk_ids.shape[0] != hidden_states.shape[0]:
        raise ValueError(
            f"Received {hidden_states.shape[0]} hidden-state rows but "
            f"{topk_ids.shape[0]} routing rows."
        )
    if len(num_recv_tokens_per_expert) != runner_config.num_local_experts:
        raise ValueError(
            f"Received counts for {len(num_recv_tokens_per_expert)} experts, "
            f"expected {runner_config.num_local_experts}."
        )
    if runner_config.no_combine:
        raise NotImplementedError(
            "The deepep_normal Triton adapter requires the runner to combine "
            "local expert results before DeepEP combine."
        )

    local_topk_ids, local_topk_weights = _map_global_expert_ids_to_local(
        topk_ids, topk_weights, runner_config
    )
    standard_dispatch_output = StandardDispatchOutput(
        hidden_states=hidden_states,
        hidden_states_scale=None,
        topk_output=StandardTopKOutput(
            topk_weights=local_topk_weights,
            topk_ids=local_topk_ids,
            router_logits=hidden_states.new_empty((hidden_states.shape[0], 0)),
        ),
    )
    running_state["deepep_normal_hidden_states_shape"] = hidden_states.shape
    running_state["deepep_normal_topk_ids"] = topk_ids
    running_state["deepep_normal_topk_weights"] = topk_weights
    return pre_permute_standard_to_triton(
        standard_dispatch_output, quant_info, runner_config, running_state
    )


@register_post_permute("triton", "deepep_normal")
def post_permute_triton_to_deepep_normal(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPNormalCombineInput:
    """Keep token order and routing metadata expected by normal combine."""
    from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPNormalCombineInput

    if (
        runner_output.hidden_states.shape
        != running_state["deepep_normal_hidden_states_shape"]
    ):
        raise ValueError(
            f"Triton output shape {tuple(runner_output.hidden_states.shape)} does "
            "not match the DeepEP normal dispatch shape "
            f"{tuple(running_state['deepep_normal_hidden_states_shape'])}."
        )

    return DeepEPNormalCombineInput(
        hidden_states=runner_output.hidden_states,
        topk_ids=running_state["deepep_normal_topk_ids"],
        topk_weights=running_state["deepep_normal_topk_weights"],
    )


@register_pre_permute("deepep_ll", "triton")
def pre_permute_deepep_ll_to_triton(
    dispatch_output: DeepEPLLDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:
    """Adapt the expert-major DeepEP/Mooncake LL layout to Triton MoE.

    Low-latency dispatch returns one padded token matrix per local expert.  The
    regular Triton runner can consume the same data after flattening it and
    assigning each row to that local expert.  Invalid padded rows are marked
    with expert id ``-1`` and zero routing weight.

    This is a BF16/FP16 compatibility path for GPUs where DeepGEMM is
    unavailable (for example, SM80).  Routing weights remain in the dispatcher
    and are applied exactly once by the subsequent LL combine.
    """
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardDispatchOutput,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput

    (
        hidden_states,
        hidden_states_scale,
        origin_topk_ids,
        origin_topk_weights,
        masked_m,
        _,
    ) = dispatch_output

    if hidden_states_scale is not None or hidden_states.dtype not in (
        torch.bfloat16,
        torch.float16,
    ):
        raise ValueError(
            "The Triton deepep_ll adapter requires BF16/FP16 dispatch output; "
            "disable FP8 dispatch for the selected A2A backend."
        )
    if hidden_states.ndim != 3:
        raise ValueError(
            "Expected expert-major hidden states with shape "
            f"[num_local_experts, token_capacity, hidden_size], got "
            f"{tuple(hidden_states.shape)}."
        )
    if runner_config.apply_router_weight_on_input:
        raise NotImplementedError(
            "The deepep_ll Triton adapter does not support "
            "apply_router_weight_on_input."
        )
    if runner_config.no_combine:
        raise NotImplementedError(
            "The deepep_ll Triton adapter requires the runner to combine its "
            "single local expert result per dispatched row."
        )

    num_local_experts, token_capacity, hidden_size = hidden_states.shape
    if masked_m.numel() != num_local_experts:
        raise ValueError(
            f"masked_m has {masked_m.numel()} entries, expected "
            f"{num_local_experts}."
        )

    token_offsets = torch.arange(token_capacity, device=hidden_states.device)
    valid_rows = token_offsets.unsqueeze(0) < masked_m.reshape(-1, 1)
    local_expert_ids = torch.arange(
        num_local_experts, dtype=torch.int32, device=hidden_states.device
    ).unsqueeze(1)
    local_expert_ids = local_expert_ids.expand(-1, token_capacity)
    local_expert_ids = local_expert_ids.masked_fill(~valid_rows, -1).reshape(-1, 1)

    local_topk_weights = valid_rows.reshape(-1, 1).to(torch.float32)
    if runner_config.routed_scaling_factor not in (None, 1.0):
        local_topk_weights.div_(runner_config.routed_scaling_factor)

    flat_hidden_states = hidden_states.reshape(-1, hidden_size)
    local_topk_output = StandardTopKOutput(
        topk_weights=local_topk_weights,
        topk_ids=local_expert_ids,
        router_logits=flat_hidden_states.new_empty((flat_hidden_states.shape[0], 0)),
    )
    standard_dispatch_output = StandardDispatchOutput(
        hidden_states=flat_hidden_states,
        hidden_states_scale=None,
        topk_output=local_topk_output,
    )

    running_state["deepep_ll_hidden_states_shape"] = hidden_states.shape
    running_state["deepep_ll_origin_topk_ids"] = origin_topk_ids
    running_state["deepep_ll_origin_topk_weights"] = origin_topk_weights
    return pre_permute_standard_to_triton(
        standard_dispatch_output,
        quant_info,
        runner_config,
        running_state,
    )


@register_post_permute("triton", "deepep_ll")
def post_permute_triton_to_deepep_ll(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> DeepEPLLCombineInput:
    """Restore the expert-major layout expected by LL combine."""
    from sglang.srt.layers.moe.token_dispatcher.deepep import (
        DeepEPLLCombineInput,
    )

    return DeepEPLLCombineInput(
        hidden_states=runner_output.hidden_states.reshape(
            running_state["deepep_ll_hidden_states_shape"]
        ),
        topk_ids=running_state["deepep_ll_origin_topk_ids"],
        topk_weights=running_state["deepep_ll_origin_topk_weights"],
    )
