from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, Tuple

import torch

from sglang.srt.configs.model_config import get_nsa_index_head_dim, is_deepseek_nsa
from sglang.srt.distributed.parallel_state import get_world_group, get_tp_group
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    TokenToKVPoolAllocator,
)
from sglang.srt.mem_cache.memory_pool import (
    DoubleSparseTokenToKVPool,
    HybridLinearKVPool,
    HybridReqToTokenPool,
    MHATokenToKVPool,
    MHATokenToKVPoolFP4,
    MLATokenToKVPool,
    MLATokenToKVPoolFP4,
    NSATokenToKVPool,
    ReqToTokenPool,
)
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool, SWATokenToKVPoolAllocator
from sglang.srt.utils.common import (
    get_available_gpu_memory,
    is_float4_e2m1fn_x2,
    is_npu,
)
from sglang.srt.utils import broadcast_pyobj

if TYPE_CHECKING:
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class MemoryPoolConfig:
    """Resolved memory pool config, shared between target and draft workers."""

    max_total_num_tokens: int
    max_running_requests: int
    full_max_total_num_tokens: Optional[int] = None
    swa_max_total_num_tokens: Optional[int] = None

    mem_fraction_static: Optional[float] = None

    def __post_init__(self):
        if self.max_total_num_tokens <= 0:
            msg = "Not enough memory. Please try to increase --mem-fraction-static."
            if self.mem_fraction_static is not None:
                msg += f" Current value: mem_fraction_static={self.mem_fraction_static}"
            raise RuntimeError(msg)


# the ratio of mamba cache pool size to max_running_requests
MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO = 3
MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP = 2
MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP = 1

logger = logging.getLogger(__name__)

_is_npu = is_npu()


class ModelRunnerKVCacheMixin:
    def get_cell_size_per_token(self: ModelRunner, num_layers: int) -> int:
        kv_size = torch._utils._element_size(self.kv_cache_dtype)
        if self.use_mla_backend:
            cell_size = (
                (self.model_config.kv_lora_rank + self.model_config.qk_rope_head_dim)
                * num_layers
                * kv_size
            )
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16
                cell_size = (cell_size // 2) + (
                    (
                        (
                            self.model_config.kv_lora_rank
                            + self.model_config.qk_rope_head_dim
                        )
                        // scale_block_size
                    )
                    * num_layers
                    * kv_size
                )

            # Add indexer KV cache overhead for NSA models (DeepSeek V3.2)
            if is_deepseek_nsa(self.model_config.hf_config):
                index_head_dim = get_nsa_index_head_dim(self.model_config.hf_config)
                indexer_size_per_token = (
                    index_head_dim
                    + index_head_dim // NSATokenToKVPool.quant_block_size * 4
                )
                element_size = torch._utils._element_size(
                    NSATokenToKVPool.index_k_with_scale_buffer_dtype
                )
                cell_size += indexer_size_per_token * num_layers * element_size
        else:
            if self.model_config.is_hybrid_swa:
                full_layers_num = len(self.model_config.full_attention_layer_ids)
                swa_layers_num = len(self.model_config.swa_attention_layer_ids)

                full_per_token = self.model_config.get_num_kv_heads(
                    get_attention_tp_size()
                ) * (self.model_config.head_dim + (self.model_config.v_head_dim or self.model_config.head_dim))

                swa_per_token = self.model_config.get_swa_num_kv_heads(
                    get_attention_tp_size()
                ) * (self.model_config.swa_head_dim + self.model_config.swa_v_head_dim)

                cell_size = (
                    full_per_token * full_layers_num + swa_per_token * swa_layers_num
                ) * kv_size
            else:
                cell_size = (
                    self.model_config.get_num_kv_heads(get_attention_tp_size())
                    * (self.model_config.head_dim + (self.model_config.v_head_dim or self.model_config.head_dim))
                    * num_layers
                    * kv_size
                )

            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                # kv_scale_buffer
                scale_block_size = 16

                n = self.model_config.get_num_kv_heads(get_attention_tp_size())
                k = self.model_config.head_dim
                cell_size = (cell_size // 2) + (
                    (n * k * num_layers * 2 * kv_size) // scale_block_size
                )
        return cell_size

    def profile_max_num_token(
        self: ModelRunner,
        pre_model_load_memory: int,
        *,
        distributed: Optional[bool] = None,
        empty_cache: bool = True,
    ):
        if distributed is None:
            distributed = self.tp_group.world_size > 1
        post_model_load_memory = get_available_gpu_memory(
            self.device,
            self.gpu_id,
            distributed=distributed,
            empty_cache=empty_cache,
            cpu_group=self.tp_group.cpu_group,
        )

        # Get the number of layers used for KV cache calculation
        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        elif mambaish := self.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if self.start_layer <= i < self.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = self.num_effective_layers

        cell_size = self.get_cell_size_per_token(num_layers)

        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)

        return int(rest_memory * (1 << 30)) // cell_size

    def _inplace_reshard_analytic_avail_gb(self: ModelRunner) -> float:
        """Estimate post-weight free memory like cold ``--tp N`` (ignores allocator drift)."""
        if self.device != "cuda":
            return get_available_gpu_memory(
                self.device, self.gpu_id, distributed=False, empty_cache=True
            )
        weight_gb = sum(
            p.numel() * p.element_size() for p in self.model.parameters()
        ) / (1024**3)
        total_gb = torch.cuda.get_device_properties(self.gpu_id).total_memory / (
            1024**3
        )
        return max(total_gb - weight_gb - 1.0, 0.0)

    def _probe_inplace_reshard_kv_tokens(
        self: ModelRunner,
        upper: int,
        *,
        max_running_requests: int = 0,
    ) -> int:
        """Best-effort probe of how many KV tokens this rank can actually allocate."""
        if upper <= 0 or self.device != "cuda":
            return upper
        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        elif mambaish := self.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if self.start_layer <= i < self.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = self.num_effective_layers
        cell_size = max(self.get_cell_size_per_token(num_layers), 1)
        ctx_len = int(getattr(self.model_config, "context_len", 0) or 0)
        req_overhead = 0
        if max_running_requests > 0 and ctx_len > 0:
            req_overhead = max_running_requests * (ctx_len + 4) * 4
        self._compact_inplace_reshard_cuda_memory()

        lo, hi = 1, upper
        best = 0
        while lo <= hi:
            mid = (lo + hi) // 2
            nbytes = mid * cell_size + req_overhead
            try:
                buf = torch.empty(
                    max(nbytes, 1),
                    dtype=torch.uint8,
                    device=self.device,
                )
                del buf
                torch.cuda.synchronize()
                best = mid
                lo = mid + 1
            except RuntimeError:
                hi = mid - 1
        self._compact_inplace_reshard_cuda_memory()
        if best < upper:
            logger.warning(
                "In-place reshard rank %d KV probe shrank %d -> %d tokens",
                self.tp_rank,
                upper,
                best,
            )
        return max(best, 1)

    def profile_max_num_token_after_inplace_reshard(
        self: ModelRunner,
        pre_model_load_memory: float = 0.0,
        *,
        distributed: bool = True,
    ) -> int:
        """Profile KV capacity after in-place TP reshard (cold ``--tp N`` parity).

        Uses the same formula as ``profile_max_num_token`` but with analytic
        post-weight free memory so rank0 allocator drift from earlier TP degrees
        does not drag down the whole group's KV budget."""
        import torch.distributed as dist

        post_model_load_memory = self._inplace_reshard_analytic_avail_gb()
        if distributed:
            post_t = torch.tensor(
                [post_model_load_memory], dtype=torch.float64, device="cpu"
            )
            dist.all_reduce(
                post_t, op=dist.ReduceOp.MIN, group=self.tp_group.cpu_group
            )
            post_model_load_memory = float(post_t.item())

        if pre_model_load_memory <= 0:
            pre_model_load_memory = float(
                getattr(self, "_pre_model_load_memory_gb", 0.0) or 0.0
            )
        if pre_model_load_memory <= 0:
            pre_model_load_memory = post_model_load_memory

        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        elif mambaish := self.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if self.start_layer <= i < self.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = self.num_effective_layers

        cell_size = self.get_cell_size_per_token(num_layers)
        headroom_gb = self._inplace_reshard_kv_headroom_gb()
        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if rest_memory <= 0:
            rest_memory = post_model_load_memory * self.mem_fraction_static
        rest_memory = max(rest_memory - headroom_gb, 0.0)
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)
        return int(rest_memory * (1024**3)) // cell_size

    def _inplace_reshard_effective_avail_gb(self: ModelRunner) -> float:
        """Free memory plus PyTorch pool bytes that are reserved but unallocated."""
        if self.device != "cuda":
            return get_available_gpu_memory(
                self.device, self.gpu_id, distributed=False, empty_cache=True
            )
        torch.cuda.empty_cache()
        free_b, _ = torch.cuda.mem_get_info(self.gpu_id)
        reserved = torch.cuda.memory_reserved(self.gpu_id)
        allocated = torch.cuda.memory_allocated(self.gpu_id)
        pool_extra = max(int(reserved) - int(allocated), 0)
        return (free_b + pool_extra) / (1024**3)

    def profile_max_num_token_measured_inplace_reshard(
        self: ModelRunner,
        *,
        distributed: bool = True,
    ) -> int:
        """Profile KV capacity from measured free memory (safety floor for rank0 drift)."""
        import torch.distributed as dist

        post_model_load_memory = self._inplace_reshard_effective_avail_gb()
        if distributed:
            post_t = torch.tensor(
                [post_model_load_memory], dtype=torch.float64, device="cpu"
            )
            dist.all_reduce(
                post_t, op=dist.ReduceOp.MIN, group=self.tp_group.cpu_group
            )
            post_model_load_memory = float(post_t.item())

        pre_model_load_memory = float(
            getattr(self, "_pre_model_load_memory_gb", 0.0) or 0.0
        )
        if pre_model_load_memory <= 0:
            pre_model_load_memory = post_model_load_memory

        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        elif mambaish := self.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if self.start_layer <= i < self.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = self.num_effective_layers

        cell_size = self.get_cell_size_per_token(num_layers)
        headroom_gb = self._inplace_reshard_kv_headroom_gb()
        rest_memory = post_model_load_memory - pre_model_load_memory * (
            1 - self.mem_fraction_static
        )
        if rest_memory <= 0:
            rest_memory = post_model_load_memory * self.mem_fraction_static
        rest_memory = max(rest_memory - headroom_gb, 0.0)
        if self.mambaish_config is not None:
            rest_memory = self.handle_max_mamba_cache(rest_memory)
        return int(rest_memory * (1024**3)) // cell_size

    def _inplace_reshard_kv_headroom_gb(self: ModelRunner) -> float:
        """GPU memory reserved for forward activations after KV pool allocation."""
        return 6.0

    def _inplace_reshard_kv_cell_size(self: ModelRunner) -> int:
        if self.is_draft_worker:
            num_layers = getattr(
                self.model_config.hf_config,
                "num_nextn_predict_layers",
                self.num_effective_layers,
            )
        elif mambaish := self.mambaish_config:
            effective_layer_ids = [
                i
                for i in mambaish.full_attention_layer_ids
                if self.start_layer <= i < self.end_layer
            ]
            num_layers = len(effective_layer_ids)
        else:
            num_layers = self.num_effective_layers
        return max(self.get_cell_size_per_token(num_layers), 1)

    def _shrink_inplace_reshard_kv_cfg_for_headroom(
        self: ModelRunner,
        cfg: MemoryPoolConfig,
        *,
        min_headroom_gb: float,
        free_gb: float,
    ) -> MemoryPoolConfig:
        """Drop enough tokens so ``free_gb`` can reach ``min_headroom_gb`` after realloc."""
        deficit = min_headroom_gb - free_gb
        if deficit <= 0:
            return cfg
        drop = int(deficit * (1024**3) / self._inplace_reshard_kv_cell_size()) + 1
        shrink = max(int(cfg.max_total_num_tokens) - drop, 1)
        return MemoryPoolConfig(
            max_total_num_tokens=shrink,
            max_running_requests=self._resolve_max_num_reqs(shrink),
            full_max_total_num_tokens=cfg.full_max_total_num_tokens,
            swa_max_total_num_tokens=cfg.swa_max_total_num_tokens,
            mem_fraction_static=cfg.mem_fraction_static,
        )

    def _resolve_memory_pool_config_fill_inplace_reshard(
        self: ModelRunner,
        *,
        distributed: bool = False,
    ) -> MemoryPoolConfig:
        analytic = self.profile_max_num_token_after_inplace_reshard(
            distributed=distributed
        )
        measured = self.profile_max_num_token_measured_inplace_reshard(
            distributed=distributed
        )
        profiled_tokens = min(analytic, measured)
        token_capacity = self._resolve_token_capacity(profiled_tokens)

        full_tokens = None
        swa_tokens = None
        if self.is_hybrid_swa:
            token_capacity, full_tokens, swa_tokens = self._resolve_hybrid_swa_tokens(
                token_capacity
            )

        return MemoryPoolConfig(
            max_total_num_tokens=token_capacity,
            max_running_requests=self._resolve_max_num_reqs(token_capacity),
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
            mem_fraction_static=self.server_args.mem_fraction_static,
        )

    def handle_max_mamba_cache(self: ModelRunner, total_rest_memory):
        config = self.mambaish_config
        server_args = self.server_args
        assert config is not None

        # reserve the memory for the intermediate mamba states used for spec dec
        if not self.spec_algorithm.is_none():
            assert server_args.speculative_num_draft_tokens is not None
            assert server_args.max_running_requests is not None

            max_running_requests = server_args.max_running_requests // (
                self.dp_size if server_args.enable_dp_attention else 1
            )
            mamba_state_intermediate_size = (
                config.mamba2_cache_params.mamba_cache_per_req
                * max_running_requests
                * server_args.speculative_num_draft_tokens
            )
            total_rest_memory = total_rest_memory - (
                mamba_state_intermediate_size / (1 << 30)
            )

        if server_args.max_mamba_cache_size is not None:
            # Use explicitly set max_mamba_cache_size
            server_args.max_mamba_cache_size = server_args.max_mamba_cache_size // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        elif (
            server_args.disable_radix_cache
            and server_args.max_running_requests is not None
        ):
            # Use explicitly set max_running_requests when radix cache is disabled
            server_args.max_mamba_cache_size = server_args.max_running_requests // (
                server_args.dp_size if server_args.enable_dp_attention else 1
            )
        else:
            # Use ratio-based calculation to auto-fit available memory
            assert config.mamba2_cache_params.mamba_cache_per_req > 0

            # allocate the memory based on the ratio between mamba state memory vs. full kv cache memory
            # solve the equations:
            # 1. mamba_state_memory + full_kv_cache_memory == total_rest_memory
            # 2. mamba_state_memory / full_kv_cache_memory == server_args.mamba_full_memory_ratio
            mamba_state_memory_raw = (
                total_rest_memory
                * server_args.mamba_full_memory_ratio
                / (1 + server_args.mamba_full_memory_ratio)
            )
            # calculate the max_mamba_cache_size based on the given total mamba memory
            server_args.max_mamba_cache_size = int(
                (mamba_state_memory_raw * (1 << 30))
                // config.mamba2_cache_params.mamba_cache_per_req
            )

        mamba_state_memory = (
            server_args.max_mamba_cache_size
            * config.mamba2_cache_params.mamba_cache_per_req
            / (1 << 30)
        )
        return total_rest_memory - mamba_state_memory

    def calculate_mla_kv_cache_dim(self: ModelRunner) -> int:
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        kv_cache_dtype = self.kv_cache_dtype
        kv_lora_rank = self.model_config.kv_lora_rank
        qk_rope_head_dim = self.model_config.qk_rope_head_dim
        kv_cache_dim = kv_lora_rank + qk_rope_head_dim  # default mla kv cache dim

        # For non-NSA models, MLA kv cache dim is simply kv_lora_rank + qk_rope_head_dim
        if not is_nsa_model:
            return kv_cache_dim

        # TRTLLM backend does not override kv_cache_dim for MLA kv cache
        # Assuming nsa prefill and decode backends are the same when using trtllm MLA backend,
        # since it is not compatible for trtllm and other mla attn backend due to the different
        # kv cache layout.
        if (
            self.server_args.nsa_prefill_backend == "trtllm"
            or self.server_args.nsa_decode_backend == "trtllm"
        ):
            return kv_cache_dim

        quant_block_size = NSATokenToKVPool.quant_block_size
        rope_storage_dtype = NSATokenToKVPool.rope_storage_dtype
        # Calculate override_kv_cache_dim for FP8 storage for non-trtllm attention backends:
        # kv_lora_rank + scale storage (kv_lora_rank // quant_block_size * 4 bytes) + rope dimension storage
        # Note: rope dimension is stored in original dtype (bf16), not quantized to fp8
        if kv_cache_dtype == torch.float8_e4m3fn:
            assert (
                kv_lora_rank % quant_block_size == 0
            ), f"kv_lora_rank {kv_lora_rank} must be multiple of quant_block_size {quant_block_size}"

            return (
                kv_lora_rank
                + kv_lora_rank // quant_block_size * 4
                + qk_rope_head_dim * rope_storage_dtype.itemsize
            )

        return kv_cache_dim

    def _resolve_hybrid_swa_tokens(
        self: ModelRunner, token_capacity: int
    ) -> Tuple[int, int, int]:
        """Split token_capacity into full/swa pools.

        Returns (effective_capacity, full_max_total_num_tokens, swa_max_total_num_tokens).
        """
        page_size = self.server_args.page_size

        assert self.sliding_window_size is not None and self.sliding_window_size > 0
        full_layers_num = len(self.model_config.full_attention_layer_ids)
        swa_layers_num = len(self.model_config.swa_attention_layer_ids)

        assert swa_layers_num > 0, "Hybrid SWA model must have at least one SWA layer"

        def align_page_size(x: int) -> int:
            return (x // page_size) * page_size

        if full_layers_num == 0:
            # all layers are SWA
            swa_tokens = align_page_size(token_capacity)
            logger.info(
                f"Use sliding window memory pool (all SWA). swa_layer_tokens={swa_tokens}"
            )
            return swa_tokens, 0, swa_tokens

        swa_full_tokens_ratio = self.server_args.swa_full_tokens_ratio

        # Use unified memory-based allocation for all hybrid SWA models.
        #
        # Let:
        #   F = Full layer per-token memory
        #   S = SWA layer per-token memory (may differ from F)
        #   r = swa_full_tokens_ratio = swa_tokens / full_tokens
        #
        # The profile phase computed:
        #   cell_size = F * n_full + S * n_swa
        #   token_capacity = rest_memory / cell_size
        #   => total_memory = token_capacity * (F * n_full + S * n_swa)
        #
        # We need to solve:
        #   full_tokens * F * n_full + swa_tokens * S * n_swa = total_memory
        #   swa_tokens = full_tokens * r
        #
        # Solution:
        #   full_tokens = total_memory / (F * n_full + r * S * n_swa)
        #               = token_capacity * (F * n_full + S * n_swa) / (F * n_full + r * S * n_swa)

        kv_size = torch._utils._element_size(self.kv_cache_dtype)

        # Full layer per-token memory
        full_per_token = (
            self.model_config.get_num_kv_heads(get_attention_tp_size())
            * (self.model_config.head_dim + (self.model_config.v_head_dim or self.model_config.head_dim))
            * kv_size
        )

        # SWA layer per-token memory
        swa_per_token = (
            self.model_config.get_swa_num_kv_heads(get_attention_tp_size())
            * (self.model_config.swa_head_dim + self.model_config.swa_v_head_dim)
            * kv_size
        )

        # Total memory available from profile
        total_memory = token_capacity * (
            full_per_token * full_layers_num + swa_per_token * swa_layers_num
        )

        # Solve the equations
        denominator = (
            full_per_token * full_layers_num
            + swa_full_tokens_ratio * swa_per_token * swa_layers_num
        )
        assert (
            denominator > 0
        ), f"Invalid denominator={denominator} for memory-based allocation. full_per_token={full_per_token}, full_layers_num={full_layers_num}, swa_per_token={swa_per_token}, swa_layers_num={swa_layers_num}, swa_full_tokens_ratio={swa_full_tokens_ratio}"

        full_tokens = align_page_size(int(total_memory / denominator))
        swa_tokens = align_page_size(int(full_tokens * swa_full_tokens_ratio))

        logger.info(
            f"Use sliding window memory pool. full_layer_tokens={full_tokens}, swa_layer_tokens={swa_tokens}"
        )
        return full_tokens, full_tokens, swa_tokens

    def _calculate_mamba_ratio(self: ModelRunner) -> int:
        if self.server_args.disable_radix_cache:
            return 1

        additional_ratio = 0
        if self.server_args.enable_mamba_extra_buffer():
            # ping-pong buffer size is 2 when overlap schedule is on, 1 otherwise.
            if not self.server_args.disable_overlap_schedule:
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_OVERLAP
            else:
                additional_ratio = MAMBA_CACHE_V2_ADDITIONAL_RATIO_NO_OVERLAP

        return MAMBA_CACHE_SIZE_MAX_RUNNING_REQUESTS_RATIO + additional_ratio

    def _init_pools(self: ModelRunner):
        """Initialize the memory pools."""
        max_num_reqs = self.max_running_requests

        # Initialize req_to_token_pool
        if self.req_to_token_pool is None:
            # FIXME(lsyin): this is the temporary fix for the context length issue when using speculative decoding
            extra_max_context_len = 4
            if self.server_args.speculative_num_draft_tokens is not None:
                extra_max_context_len += self.server_args.speculative_num_draft_tokens

            if self.server_args.disaggregation_mode == "decode":
                from sglang.srt.disaggregation.decode import (
                    DecodeReqToTokenPool,
                    HybridMambaDecodeReqToTokenPool,
                )

                # subscribe memory for pre-allocated requests
                # if max_num_reqs <= 32, we pre-allocate 2x requests
                pre_alloc_size = max_num_reqs * 2 if max_num_reqs <= 32 else 0
                if config := self.mambaish_config:
                    self.req_to_token_pool = HybridMambaDecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        cache_params=config.mamba2_cache_params,
                        speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                        enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                        pre_alloc_size=pre_alloc_size,
                        enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                        mamba_size=self.server_args.max_mamba_cache_size,
                    )
                else:
                    self.req_to_token_pool = DecodeReqToTokenPool(
                        size=max_num_reqs,
                        max_context_len=self.model_config.context_len
                        + extra_max_context_len,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        pre_alloc_size=pre_alloc_size,
                    )
            elif config := self.mambaish_config:
                self.req_to_token_pool = HybridReqToTokenPool(
                    size=max_num_reqs,
                    mamba_size=self.server_args.max_mamba_cache_size,
                    mamba_spec_state_size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    cache_params=config.mamba2_cache_params,
                    enable_mamba_extra_buffer=self.server_args.enable_mamba_extra_buffer(),
                    speculative_num_draft_tokens=self.server_args.speculative_num_draft_tokens,
                    enable_overlap_schedule=not self.server_args.disable_overlap_schedule,
                )
            else:
                self.req_to_token_pool = ReqToTokenPool(
                    size=max_num_reqs,
                    max_context_len=self.model_config.context_len
                    + extra_max_context_len,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                )
        else:
            # Draft worker shares req_to_token_pool with the target worker.
            assert self.is_draft_worker

        # Initialize token_to_kv_pool
        is_nsa_model = is_deepseek_nsa(self.model_config.hf_config)
        if self.server_args.attention_backend == "ascend" and not self.mambaish_config:
            if self.use_mla_backend:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMLATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    index_head_dim=(
                        self.model_config.index_head_dim if is_nsa_model else None
                    ),
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                from sglang.srt.hardware_backend.npu.memory_pool_npu import (
                    NPUMHATokenToKVPool,
                )

                self.token_to_kv_pool = NPUMHATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.use_mla_backend and is_nsa_model:
            self.token_to_kv_pool = NSATokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                kv_lora_rank=self.model_config.kv_lora_rank,
                qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                kv_cache_dim=self.calculate_mla_kv_cache_dim(),
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
                index_head_dim=get_nsa_index_head_dim(self.model_config.hf_config),
            )
        elif self.use_mla_backend and not self.mambaish_config:
            assert not is_nsa_model
            if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                self.token_to_kv_pool = MLATokenToKVPoolFP4(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
            else:
                self.token_to_kv_pool = MLATokenToKVPool(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    kv_lora_rank=self.model_config.kv_lora_rank,
                    qk_rope_head_dim=self.model_config.qk_rope_head_dim,
                    layer_num=self.num_effective_layers,
                    device=self.device,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    start_layer=self.start_layer,
                    end_layer=self.end_layer,
                )
        elif self.server_args.enable_double_sparsity:
            self.token_to_kv_pool = DoubleSparseTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(get_attention_tp_size()),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                heavy_channel_num=self.server_args.ds_heavy_channel_num,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )
        else:
            if self.is_hybrid_swa:
                kwargs = {}
                if self.is_hybrid_swa_compress:
                    kwargs = {
                        "swa_head_num": max(
                            1,
                            self.model_config.hf_text_config.swa_num_key_value_heads
                            // get_attention_tp_size(),
                        ),
                        "swa_head_dim": self.model_config.hf_text_config.swa_head_dim,
                        "swa_v_head_dim": self.model_config.hf_text_config.swa_v_head_dim,
                        "v_head_dim": self.model_config.hf_text_config.v_head_dim,
                    }
                self.token_to_kv_pool = SWAKVPool(
                    size=self.full_max_total_num_tokens,
                    size_swa=self.swa_max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    swa_attention_layer_ids=self.model_config.swa_attention_layer_ids,
                    full_attention_layer_ids=self.model_config.full_attention_layer_ids,
                    enable_kvcache_transpose=False,
                    device=self.device,
                    **kwargs,
                )
            elif config := self.mambaish_config:
                extra_args = {}
                if self.use_mla_backend:
                    extra_args = {
                        "kv_lora_rank": self.model_config.kv_lora_rank,
                        "qk_rope_head_dim": self.model_config.qk_rope_head_dim,
                    }
                self.token_to_kv_pool = HybridLinearKVPool(
                    page_size=self.page_size,
                    size=self.max_total_num_tokens,
                    dtype=self.kv_cache_dtype,
                    head_num=self.model_config.get_num_kv_heads(
                        get_attention_tp_size()
                    ),
                    head_dim=self.model_config.head_dim,
                    # if draft worker, we only need 1 attention layer's kv pool
                    full_attention_layer_ids=(
                        [0]
                        if self.is_draft_worker
                        else [
                            i
                            for i in config.full_attention_layer_ids
                            if self.start_layer <= i < self.end_layer
                        ]
                    ),
                    enable_kvcache_transpose=False,
                    device=self.device,
                    mamba_pool=self.req_to_token_pool.mamba_pool,
                    enable_memory_saver=self.server_args.enable_memory_saver,
                    use_mla=self.use_mla_backend,
                    **extra_args,
                )
            else:
                if is_float4_e2m1fn_x2(self.kv_cache_dtype):
                    self.token_to_kv_pool = MHATokenToKVPoolFP4(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )
                else:
                    self.token_to_kv_pool = MHATokenToKVPool(
                        self.max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        head_num=self.model_config.get_num_kv_heads(
                            get_attention_tp_size()
                        ),
                        head_dim=self.model_config.head_dim,
                        layer_num=self.num_effective_layers,
                        device=self.device,
                        enable_memory_saver=self.server_args.enable_memory_saver,
                        start_layer=self.start_layer,
                        end_layer=self.end_layer,
                        enable_alt_stream=not self.server_args.enable_pdmux,
                        enable_kv_cache_copy=(
                            self.server_args.speculative_algorithm is not None
                        ),
                    )

        # Initialize token_to_kv_pool_allocator
        need_sort = self.server_args.disaggregation_mode in ("decode", "prefill")
        if self.token_to_kv_pool_allocator is None:
            if _is_npu and (
                self.server_args.attention_backend == "ascend"
                or self.hybrid_gdn_config is not None
            ):
                from sglang.srt.hardware_backend.npu.allocator_npu import (
                    NPUPagedTokenToKVPoolAllocator,
                )

                self.token_to_kv_pool_allocator = NPUPagedTokenToKVPoolAllocator(
                    self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=need_sort,
                )
            else:
                if self.is_hybrid_swa:
                    self.token_to_kv_pool_allocator = SWATokenToKVPoolAllocator(
                        self.full_max_total_num_tokens,
                        self.swa_max_total_num_tokens,
                        page_size=self.page_size,
                        dtype=self.kv_cache_dtype,
                        device=self.device,
                        kvcache=self.token_to_kv_pool,
                        need_sort=need_sort,
                    )
                else:
                    if self.page_size == 1:
                        self.token_to_kv_pool_allocator = TokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )
                    else:
                        self.token_to_kv_pool_allocator = PagedTokenToKVPoolAllocator(
                            self.max_total_num_tokens,
                            page_size=self.page_size,
                            dtype=self.kv_cache_dtype,
                            device=self.device,
                            kvcache=self.token_to_kv_pool,
                            need_sort=need_sort,
                        )

        else:
            assert self.is_draft_worker
            if self.is_hybrid_swa:
                assert (
                    self.token_to_kv_pool_allocator.__class__
                    == SWATokenToKVPoolAllocator
                )
                self.token_to_kv_pool.full_to_swa_index_mapping = (
                    self.token_to_kv_pool_allocator.full_to_swa_index_mapping
                )

    def _resolve_token_capacity(self: ModelRunner, profiled_tokens: int) -> int:
        """Compute final token pool capacity from profiled value,
        applying user cap, page alignment, and PP sync"""
        user_limit = self.server_args.max_total_tokens

        # Apply user-specified upper bound
        if user_limit is not None:
            if user_limit > profiled_tokens:
                logging.warning(
                    f"max_total_tokens={user_limit} is larger than the profiled value "
                    f"{profiled_tokens}. Use the profiled value instead."
                )
            capacity = min(profiled_tokens, user_limit)
        else:
            capacity = profiled_tokens

        # Align to page boundary
        page_size = self.server_args.page_size
        capacity = capacity // page_size * page_size

        # Sync across PP ranks (each may have different layer counts)
        if self.pp_size > 1:
            tensor = torch.tensor(capacity, dtype=torch.int64)
            torch.distributed.all_reduce(
                tensor,
                op=torch.distributed.ReduceOp.MIN,
                group=get_world_group().cpu_group,
            )
            capacity = tensor.item()

        return capacity

    def _resolve_max_num_reqs(self: ModelRunner, token_capacity: int) -> int:
        """Compute max concurrent requests (per dp worker) from the finalized
        token capacity."""
        # Estimate pool size (used as upper bound when user specifies max_running_requests)
        estimated = int(token_capacity / self.model_config.context_len * 512)
        estimated = max(min(estimated, 4096), 2048)

        max_num_reqs = self.server_args.max_running_requests
        if max_num_reqs is not None:
            max_num_reqs = min(max_num_reqs // self.dp_size, estimated)
        else:
            max_num_reqs = min(estimated, token_capacity // 2)

        if self.mambaish_config is not None:
            ratio = self._calculate_mamba_ratio()
            max_num_reqs = min(
                max_num_reqs, self.server_args.max_mamba_cache_size // ratio
            )

        return max_num_reqs

    def _apply_memory_pool_config(self: ModelRunner, config: MemoryPoolConfig):
        """Apply a resolved MemoryPoolConfig and initialize pools."""
        self.max_total_num_tokens = config.max_total_num_tokens
        self.max_running_requests = config.max_running_requests
        if self.is_hybrid_swa:
            self.full_max_total_num_tokens = config.full_max_total_num_tokens
            self.swa_max_total_num_tokens = config.swa_max_total_num_tokens

        self._init_pools()

    def _resolve_memory_pool_config(
        self: ModelRunner,
        pre_model_load_memory: int,
        *,
        distributed: Optional[bool] = None,
        empty_cache: bool = True,
    ) -> MemoryPoolConfig:
        """Profile GPU memory and resolve all pool parameters into a config."""
        profiled_tokens = self.profile_max_num_token(
            pre_model_load_memory,
            distributed=distributed,
            empty_cache=empty_cache,
        )
        token_capacity = self._resolve_token_capacity(profiled_tokens)

        full_tokens = None
        swa_tokens = None
        if self.is_hybrid_swa:
            token_capacity, full_tokens, swa_tokens = self._resolve_hybrid_swa_tokens(
                token_capacity
            )

        return MemoryPoolConfig(
            max_total_num_tokens=token_capacity,
            max_running_requests=self._resolve_max_num_reqs(token_capacity),
            full_max_total_num_tokens=full_tokens,
            swa_max_total_num_tokens=swa_tokens,
            mem_fraction_static=self.server_args.mem_fraction_static,
        )

    def init_memory_pool(self: ModelRunner, pre_model_load_memory: int):
        if not self.spec_algorithm.is_none() and self.is_draft_worker:
            assert (
                self.memory_pool_config is not None
            ), "Draft worker requires memory_pool_config"
        else:
            self.memory_pool_config = self._resolve_memory_pool_config(
                pre_model_load_memory
            )

        self._apply_memory_pool_config(self.memory_pool_config)

        logger.info(
            f"Memory pool end. "
            f"avail mem={get_available_gpu_memory(self.device, self.gpu_id):.2f} GB"
        )

    def _compact_inplace_reshard_cuda_memory(self: ModelRunner) -> None:
        """Best-effort reclaim of peak transfer / narrow fragmentation before KV profile."""
        import gc as _gc

        _gc.collect()
        if self.device == "cuda":
            torch.cuda.synchronize()
            if hasattr(torch.cuda, "ipc_collect"):
                torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

    def _maybe_reclaim_inplace_reshard_memory_imbalance(
        self: ModelRunner, tp_max_avail: Optional[float] = None
    ) -> float:
        """Reclaim CUDA caching-allocator drift when this rank trails the TP group."""
        local_avail = get_available_gpu_memory(
            self.device, self.gpu_id, distributed=False, empty_cache=True
        )
        max_avail = tp_max_avail if tp_max_avail is not None else local_avail
        if local_avail >= 0.95 * max_avail:
            return local_avail
        logger.warning(
            "In-place reshard rank %d avail=%.2fGB trails tp_max=%.2fGB; reclaiming",
            self.tp_rank,
            local_avail,
            max_avail,
        )
        for _ in range(3):
            self._compact_inplace_reshard_cuda_memory()
        return get_available_gpu_memory(
            self.device, self.gpu_id, distributed=False, empty_cache=True
        )

    def rebuild_memory_pool_after_inplace_reshard(self: ModelRunner) -> MemoryPoolConfig:
        """Re-profile GPU memory and rebuild KV pools after in-place TP reshard.

        Each rank fills ``mem_fraction_static`` of its post-weight free memory; the
        TP group takes MIN(tokens) so all ranks stay consistent while maximizing KV
        on clean GPUs (standby activations)."""
        import torch.distributed as dist

        tp_cpu = get_tp_group().cpu_group
        dist.barrier(group=tp_cpu)
        if self.tp_rank == 0 and getattr(
            self, "_needs_inplace_reshard_rank0_cold_reload", False
        ):
            self._needs_inplace_reshard_rank0_cold_reload = False
            self._cold_reload_inplace_reshard_rank0_weights(self.tp_size)
        elif self.tp_rank == 0 and not getattr(
            self, "_rank0_inplace_reshard_cold_reloaded", False
        ):
            self._maybe_reload_inplace_reshard_rank0_weights_for_memory(self.tp_size)
        if self.tp_rank == 0:
            self._rank0_inplace_reshard_cold_reloaded = False
        dist.barrier(group=tp_cpu)
        self.configure_kv_cache_dtype()
        self._release_inplace_reshard_attention_state()
        self._compact_inplace_reshard_cuda_memory()
        local_avail = self._inplace_reshard_effective_avail_gb()
        avail_t = torch.tensor([local_avail], dtype=torch.float64)
        max_t = avail_t.clone()
        dist.all_reduce(max_t, op=dist.ReduceOp.MAX, group=tp_cpu)
        tp_max_avail = float(max_t.item())
        self._maybe_reclaim_inplace_reshard_memory_imbalance(tp_max_avail)

        dist.barrier(group=tp_cpu)

        prev_tokens = int(getattr(self, "max_total_num_tokens", 0) or 0)
        local_cfg = self._resolve_memory_pool_config_fill_inplace_reshard(
            distributed=False
        )
        tokens_t = torch.tensor(
            [local_cfg.max_total_num_tokens], dtype=torch.int64
        )
        reqs_t = torch.tensor(
            [local_cfg.max_running_requests], dtype=torch.int64
        )
        dist.all_reduce(tokens_t, op=dist.ReduceOp.MIN, group=tp_cpu)
        dist.all_reduce(reqs_t, op=dist.ReduceOp.MIN, group=tp_cpu)
        feasible_t = torch.tensor(
            [
                self._probe_inplace_reshard_kv_tokens(
                    int(tokens_t.item()),
                    max_running_requests=int(reqs_t.item()),
                )
            ],
            dtype=torch.int64,
        )
        dist.all_reduce(feasible_t, op=dist.ReduceOp.MIN, group=tp_cpu)
        cfg = MemoryPoolConfig(
            max_total_num_tokens=int(feasible_t.item()),
            max_running_requests=int(reqs_t.item()),
            full_max_total_num_tokens=local_cfg.full_max_total_num_tokens,
            swa_max_total_num_tokens=local_cfg.swa_max_total_num_tokens,
            mem_fraction_static=local_cfg.mem_fraction_static,
        )
        dist.barrier(group=tp_cpu)
        post_avail = self._inplace_reshard_effective_avail_gb()
        min_meas_t = torch.tensor([post_avail], dtype=torch.float64)
        max_meas_t = min_meas_t.clone()
        dist.all_reduce(min_meas_t, op=dist.ReduceOp.MIN, group=tp_cpu)
        dist.all_reduce(max_meas_t, op=dist.ReduceOp.MAX, group=tp_cpu)
        if self.tp_rank == 0:
            analytic_avail = self._inplace_reshard_analytic_avail_gb()
            pre_gb = float(getattr(self, "_pre_model_load_memory_gb", 0.0) or 0.0)
            logger.info(
                "In-place reshard KV re-profile: %d -> %d tokens, %d -> %d reqs "
                "(pre=%.2fGB analytic_post=%.2fGB measured_post rank0=%.2fGB "
                "tp_meas_min=%.2fGB tp_meas_max=%.2fGB)",
                prev_tokens,
                cfg.max_total_num_tokens,
                getattr(self, "max_running_requests", 0),
                cfg.max_running_requests,
                pre_gb,
                analytic_avail,
                post_avail,
                float(min_meas_t.item()),
                float(max_meas_t.item()),
            )
        self.memory_pool_config = cfg
        min_headroom_gb = self._inplace_reshard_kv_headroom_gb()

        def _sync_cfg_tokens(local_cfg: MemoryPoolConfig) -> MemoryPoolConfig:
            shrink_t = torch.tensor(
                [local_cfg.max_total_num_tokens], dtype=torch.int64
            )
            dist.all_reduce(shrink_t, op=dist.ReduceOp.MIN, group=tp_cpu)
            n = int(shrink_t.item())
            return MemoryPoolConfig(
                max_total_num_tokens=n,
                max_running_requests=self._resolve_max_num_reqs(n),
                full_max_total_num_tokens=local_cfg.full_max_total_num_tokens,
                swa_max_total_num_tokens=local_cfg.swa_max_total_num_tokens,
                mem_fraction_static=local_cfg.mem_fraction_static,
            )

        def _drop_kv_pools() -> None:
            self.req_to_token_pool = None
            self.token_to_kv_pool = None
            self.token_to_kv_pool_allocator = None
            self._compact_inplace_reshard_cuda_memory()

        for attempt in range(10):
            try:
                self._apply_memory_pool_config(cfg)
                free_gb = self._inplace_reshard_effective_avail_gb()
                free_t = torch.tensor([free_gb], dtype=torch.float64)
                dist.all_reduce(free_t, op=dist.ReduceOp.MIN, group=tp_cpu)
                min_free = float(free_t.item())
                if min_free >= min_headroom_gb:
                    break
                next_cfg = self._shrink_inplace_reshard_kv_cfg_for_headroom(
                    cfg, min_headroom_gb=min_headroom_gb, free_gb=min_free
                )
                if int(next_cfg.max_total_num_tokens) >= int(cfg.max_total_num_tokens):
                    next_cfg = MemoryPoolConfig(
                        max_total_num_tokens=max(
                            int(cfg.max_total_num_tokens * 0.9), 1
                        ),
                        max_running_requests=self._resolve_max_num_reqs(
                            max(int(cfg.max_total_num_tokens * 0.9), 1)
                        ),
                        full_max_total_num_tokens=cfg.full_max_total_num_tokens,
                        swa_max_total_num_tokens=cfg.swa_max_total_num_tokens,
                        mem_fraction_static=cfg.mem_fraction_static,
                    )
                logger.warning(
                    "In-place reshard rank %d KV headroom retry: %d -> %d tokens "
                    "(min free %.2fGB < %.1fGB)",
                    self.tp_rank,
                    cfg.max_total_num_tokens,
                    next_cfg.max_total_num_tokens,
                    min_free,
                    min_headroom_gb,
                )
                _drop_kv_pools()
                cfg = _sync_cfg_tokens(next_cfg)
                if attempt >= 9:
                    logger.warning(
                        "In-place reshard rank %d KV headroom low (%.2fGB min) after %d tokens",
                        self.tp_rank,
                        min_free,
                        cfg.max_total_num_tokens,
                    )
                    break
            except RuntimeError as exc:
                msg = str(exc).lower()
                if attempt >= 9 or (
                    "not enough memory" not in msg and "out of memory" not in msg
                ):
                    raise
                shrink = max(int(cfg.max_total_num_tokens * 0.9), 1)
                logger.warning(
                    "In-place reshard rank %d KV alloc retry: %d -> %d tokens (%s)",
                    self.tp_rank,
                    cfg.max_total_num_tokens,
                    shrink,
                    exc,
                )
                _drop_kv_pools()
                cfg = _sync_cfg_tokens(
                    MemoryPoolConfig(
                        max_total_num_tokens=shrink,
                        max_running_requests=self._resolve_max_num_reqs(shrink),
                        full_max_total_num_tokens=cfg.full_max_total_num_tokens,
                        swa_max_total_num_tokens=cfg.swa_max_total_num_tokens,
                        mem_fraction_static=cfg.mem_fraction_static,
                    )
                )
        dist.barrier(group=tp_cpu)
        saved_pre = getattr(self, "_pre_reshard_tp", self.tp_size)
        self._pre_reshard_tp = self.tp_size
        self._update_model_tp_metadata(self.tp_size)
        self._pre_reshard_tp = saved_pre
        dist.barrier(group=tp_cpu)
        return cfg
