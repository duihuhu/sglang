"""AFD (Attention-FFN Disaggregation) batch overlap preparation.

Separated from two_batch_overlap.py (optimization E2) to keep AFD logic
isolated and maintainable.
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence

import torch

from sglang.srt.batch_overlap.two_batch_overlap import (
    _compute_extend_num_tokens,
    get_token_num_per_seq,
    split_spec_info,
)
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.dp_attention import get_attention_tp_size
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.server_args import get_global_server_args

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


def _get_afd_micro_batch() -> int:
    return getattr(get_global_server_args(), "afd_micro_batch", 3)


def _afd_is_ffn() -> bool:
    from sglang.srt.layers.afd import afd_is_ffn

    return afd_is_ffn()


# ------------- E3 optimization: generic m-way split -------------


def _split_seq_indices_m_way(
    num_seqs: int,
    m: int,
    extend_lens: Optional[Sequence[int]] = None,
) -> List[int]:
    """Compute m-1 split indices that divide num_seqs into m roughly equal parts.

    G3 optimization: when extend_lens is provided, use token-count-aware
    greedy splitting instead of naive num_seqs // m.
    """
    if m <= 1:
        return []

    if extend_lens is not None and len(extend_lens) > 0:
        total_tokens = sum(extend_lens)
        target_per_part = total_tokens / m
        splits = []
        running_sum = 0
        part_idx = 1
        for i in range(len(extend_lens)):
            running_sum += extend_lens[i]
            if running_sum >= target_per_part * part_idx and part_idx < m:
                splits.append(i + 1)
                part_idx += 1
                if part_idx >= m:
                    break
        while len(splits) < m - 1:
            last = splits[-1] if splits else 0
            remaining = num_seqs - last
            step = max(1, remaining // (m - len(splits)))
            splits.append(min(last + step, num_seqs))
        return splits[:m - 1]

    interval = max(1, num_seqs // m)
    return [interval * (i + 1) for i in range(m - 1)]


def _compute_token_indices_m_way(
    split_seq_indices: List[int],
    forward_mode: ForwardMode,
    extend_seq_lens: Optional[Sequence[int]],
    token_num_per_seq: Optional[int],
) -> List[int]:
    """Compute token split indices from seq split indices.

    Self-contained implementation that avoids TBO's two-chunk-split logic
    which ignores split_seq_index when token distribution is skewed.
    """
    if forward_mode == ForwardMode.EXTEND:
        assert extend_seq_lens is not None
        return [sum(extend_seq_lens[:idx]) for idx in split_seq_indices]
    elif forward_mode.is_target_verify() or forward_mode.is_decode():
        assert token_num_per_seq is not None
        return [idx * token_num_per_seq for idx in split_seq_indices]
    elif forward_mode.is_idle():
        return [0] * len(split_seq_indices)
    else:
        raise NotImplementedError(f"Unsupported forward_mode: {forward_mode}")


# ------------- AfdForwardBatchPreparer -------------


class AfdForwardBatchPreparer:
    """Prepares AFD sub-batches (micro-batches) for pipelined execution.

    E3 optimization: supports arbitrary m (not just 2 or 3).
    G3 optimization: token-aware splitting in EXTEND mode.
    """

    @classmethod
    def prepare(cls, batch: ForwardBatch):
        m = _get_afd_micro_batch()
        if (
            batch.afd_split_seq_index is None
            or m is None
            or m <= 1
        ):
            return

        from sglang.srt.layers.attention.tbo_backend import AfdAttnBackend

        if not isinstance(batch.attn_backend, AfdAttnBackend):
            return

        token_indices = cls._compute_split_token_indices(batch, m)
        seq_indices = batch.afd_split_seq_index

        children_backends = batch.attn_backend.children
        assert len(children_backends) == m, (
            f"AfdAttnBackend has {len(children_backends)} children but m={m}"
        )

        children = []
        num_tokens_total = batch.input_ids.shape[0]

        # Build m children from m-1 split points
        boundaries_seq = [0] + list(seq_indices) + [batch.batch_size]
        boundaries_tok = [0] + list(token_indices) + [num_tokens_total]

        num_token_non_padded_values = []
        for i in range(m):
            tok_start = boundaries_tok[i]
            tok_end = boundaries_tok[i + 1]
            num_token_non_padded_values.append(max(0, tok_end - tok_start))

        num_token_non_padded_tensor = torch.tensor(
            num_token_non_padded_values, dtype=torch.int32
        ).to(device=batch.input_ids.device, non_blocking=True)

        for i in range(m):
            child = cls._filter_batch(
                batch,
                start_token_index=boundaries_tok[i],
                end_token_index=boundaries_tok[i + 1],
                start_seq_index=boundaries_seq[i],
                end_seq_index=boundaries_seq[i + 1],
                output_attn_backend=children_backends[i],
                out_num_token_non_padded=num_token_non_padded_tensor[i],
            )
            children.append(child)

        batch.afd_children = children
        batch.can_run_afd_overlap = True

    @classmethod
    def _compute_split_token_indices(
        cls, batch: ForwardBatch, m: int
    ) -> List[int]:
        token_num_per_seq = get_token_num_per_seq(
            forward_mode=batch.forward_mode, spec_info=batch.spec_info
        )
        return _compute_token_indices_m_way(
            split_seq_indices=batch.afd_split_seq_index,
            forward_mode=batch.forward_mode,
            extend_seq_lens=batch.extend_seq_lens_cpu
            if hasattr(batch, "extend_seq_lens_cpu")
            else None,
            token_num_per_seq=token_num_per_seq,
        )

    @classmethod
    def _filter_batch(
        cls,
        batch: ForwardBatch,
        *,
        start_token_index: int,
        end_token_index: int,
        start_seq_index: int,
        end_seq_index: int,
        output_attn_backend: AttentionBackend,
        out_num_token_non_padded: torch.Tensor,
    ) -> ForwardBatch:
        assert end_token_index >= start_token_index
        num_tokens = batch.input_ids.shape[0]
        num_seqs = batch.batch_size

        output_dict = dict()
        is_ffn = _afd_is_ffn()

        skip_keys_ffn = {"out_cache_loc", "req_pool_indices"}

        if is_ffn:
            for key in skip_keys_ffn:
                output_dict[key] = None

        for key in ["input_ids", "positions", "out_cache_loc"]:
            if is_ffn and key in skip_keys_ffn:
                continue
            old_value = getattr(batch, key)
            assert old_value.shape[0] == num_tokens
            output_dict[key] = old_value[start_token_index:end_token_index]

        attention_tp_size = get_attention_tp_size()
        tbo_padded_len = (
            (end_token_index - start_token_index - 1) // attention_tp_size + 1
        ) * attention_tp_size

        for key in [
            "req_pool_indices",
            "seq_lens",
            "seq_lens_cpu",
            "extend_seq_lens",
            "extend_prefix_lens",
            "extend_start_loc",
            "extend_prefix_lens_cpu",
            "extend_seq_lens_cpu",
            "extend_logprob_start_lens_cpu",
            "lora_ids",
            "rids",
        ]:
            if is_ffn and key in skip_keys_ffn:
                continue
            old_value = getattr(batch, key, None)
            if old_value is None:
                continue
            elif batch.forward_mode.is_target_verify() and key in (
                "extend_seq_lens",
                "extend_prefix_lens",
                "extend_start_loc",
                "extend_prefix_lens_cpu",
                "extend_seq_lens_cpu",
                "extend_logprob_start_lens_cpu",
            ):
                output_dict[key] = None
                continue
            assert len(old_value) == num_seqs
            output_dict[key] = old_value[start_seq_index:end_seq_index]

        spec_info = getattr(batch, "spec_info", None)
        output_spec_info = split_spec_info(
            spec_info=spec_info,
            start_token_index=start_token_index,
            end_token_index=end_token_index,
            start_seq_index=start_seq_index,
            end_seq_index=end_seq_index,
        )
        output_dict["spec_info"] = output_spec_info

        for key in [
            "forward_mode",
            "is_extend_in_batch",
            "all_extend_in_batch",
            "return_logprob",
            "req_to_token_pool",
            "token_to_kv_pool",
            "can_run_dp_cuda_graph",
            "dp_padding_mode",
            "global_forward_mode",
            "is_prefill_only",
            "spec_algorithm",
            "capture_hidden_mode",
            "padded_static_len",
            "mrope_positions",
            "split_index",
            "orig_seq_lens",
        ]:
            if is_ffn and key in skip_keys_ffn:
                continue
            output_dict[key] = getattr(batch, key, None)

        if not batch.forward_mode.is_target_verify():
            assert (
                _compute_extend_num_tokens(batch.input_ids, batch.forward_mode)
                == batch.extend_num_tokens
            )
        extend_num_tokens = _compute_extend_num_tokens(
            output_dict["input_ids"], output_dict["forward_mode"]
        )

        if (
            get_global_server_args().moe_dense_tp_size == 1
            and batch.global_dp_buffer_len is not None
        ):
            global_dp_buffer_len = end_token_index - start_token_index
        else:
            global_dp_buffer_len = None

        output_dict.update(
            dict(
                batch_size=end_seq_index - start_seq_index,
                seq_lens_sum=(
                    output_dict["seq_lens_cpu"].sum()
                    if "seq_lens_cpu" in output_dict
                    and output_dict.get("seq_lens_cpu") is not None
                    else (
                        output_dict["seq_lens"].sum().item()
                        if "seq_lens" in output_dict
                        and output_dict.get("seq_lens") is not None
                        else 0
                    )
                ),
                extend_num_tokens=extend_num_tokens,
                attn_backend=output_attn_backend,
                num_token_non_padded=out_num_token_non_padded,
                num_token_non_padded_cpu=None,
                tbo_split_seq_index=None,
                tbo_parent_token_range=None,
                tbo_padded_len=tbo_padded_len,
                tbo_children=None,
                afd_split_seq_index=None,
                afd_parent_token_range=(start_token_index, end_token_index),
                afd_children=None,
                can_run_afd_overlap=False,
                original_global_num_tokens_cpu=None,
                global_num_tokens_gpu=None,
                global_num_tokens_cpu=None,
                global_dp_buffer_len=global_dp_buffer_len,
                global_num_tokens_for_logprob_gpu=None,
                global_num_tokens_for_logprob_cpu=None,
                sampling_info=None,
                temp_scaled_logprobs=False,
                temperature=None,
                top_p_normalized_logprobs=False,
                top_p=None,
                mm_inputs=None,
                top_logprobs_nums=None,
                token_ids_logprobs=None,
                next_token_logits_buffer=None,
                return_hidden_states_before_norm=False,
            )
        )

        errors = []
        for field in dataclasses.fields(ForwardBatch):
            if (
                getattr(batch, field.name) is not None
                and field.name not in output_dict
            ):
                errors.append(
                    f"Field {field.name} has value but is not yet supported "
                    f"(value={getattr(batch, field.name)})"
                )
        if errors:
            raise Exception(f"{len(errors)} errors:\n" + "\n".join(errors))

        return ForwardBatch(**output_dict)
