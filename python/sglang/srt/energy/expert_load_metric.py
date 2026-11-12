"""Expert Load Imbalance metrics for MoE DVFS awareness.

Provides lightweight metrics that quantify how unevenly tokens are
distributed across experts in a single decode step.  These metrics
serve as an additional feature for the energy/latency predictor so
that DVFS can account for MoE routing skew.

Metrics:
    ELS  – Expert Load Skew: max(expert_counts) / mean(expert_counts)
    GLR  – GPU Load Ratio:   max(gpu_load) / mean(gpu_load)  (TP>1)
    LIF  – Load Imbalance Factor: GLR * sqrt(ELS)  (unified scalar)
"""

from __future__ import annotations

import math
from typing import Union

import torch


def compute_els(
    topk_ids: torch.Tensor,
    num_experts: int,
) -> float:
    """Compute Expert Load Skew.

    Args:
        topk_ids: (num_tokens, top_k) tensor of expert indices.
        num_experts: Total number of experts in the model.

    Returns:
        ELS value >= 1.0.  Returns 1.0 for empty input.
    """
    if topk_ids.numel() == 0:
        return 1.0

    counts = torch.bincount(topk_ids.flatten().int(), minlength=num_experts)
    max_count = counts.max().item()
    mean_count = counts.float().mean().item()

    if mean_count <= 0:
        return 1.0
    return max_count / mean_count


def compute_glr(
    topk_ids: torch.Tensor,
    num_experts: int,
    tp: int,
) -> float:
    """Compute GPU Load Ratio (inter-GPU imbalance).

    Assumes experts are evenly sharded across TP ranks:
    rank g owns experts [g*E/T, (g+1)*E/T).

    Args:
        topk_ids: (num_tokens, top_k) tensor of expert indices.
        num_experts: Total number of experts.
        tp: Tensor parallelism degree.

    Returns:
        GLR value >= 1.0.  Returns 1.0 when tp <= 1 or empty input.
    """
    if tp <= 1 or topk_ids.numel() == 0:
        return 1.0

    counts = torch.bincount(topk_ids.flatten().int(), minlength=num_experts)
    experts_per_gpu = num_experts // tp

    gpu_loads = torch.zeros(tp, dtype=counts.dtype, device=counts.device)
    for g in range(tp):
        start = g * experts_per_gpu
        end = start + experts_per_gpu
        gpu_loads[g] = counts[start:end].sum()

    max_load = gpu_loads.max().item()
    mean_load = gpu_loads.float().mean().item()

    if mean_load <= 0:
        return 1.0
    return max_load / mean_load


def compute_lif(
    topk_ids: torch.Tensor,
    num_experts: int,
    tp: int = 1,
) -> float:
    """Compute unified Load Imbalance Factor.

    LIF = GLR * sqrt(ELS)

    When TP=1, GLR=1 and LIF = sqrt(ELS).

    Args:
        topk_ids: (num_tokens, top_k) tensor of expert indices.
        num_experts: Total number of experts.
        tp: Tensor parallelism degree.

    Returns:
        LIF value >= 1.0.
    """
    els = compute_els(topk_ids, num_experts)
    glr = compute_glr(topk_ids, num_experts, tp)
    return glr * math.sqrt(els)


def compute_lif_from_counts(
    expert_counts: Union[torch.Tensor, "numpy.ndarray"],
    num_experts: int,
    tp: int = 1,
) -> float:
    """Compute LIF from pre-computed expert counts (useful in profiling).

    Args:
        expert_counts: 1-D array of length num_experts with per-expert token counts.
        num_experts: Total number of experts.
        tp: Tensor parallelism degree.

    Returns:
        LIF value >= 1.0.
    """
    import numpy as np

    if isinstance(expert_counts, torch.Tensor):
        counts = expert_counts.cpu().numpy().astype(float)
    else:
        counts = np.asarray(expert_counts, dtype=float)

    if counts.sum() <= 0:
        return 1.0

    mean_count = counts.mean()
    max_count = counts.max()
    els = max_count / mean_count if mean_count > 0 else 1.0

    glr = 1.0
    if tp > 1:
        experts_per_gpu = num_experts // tp
        gpu_loads = np.array([
            counts[g * experts_per_gpu: (g + 1) * experts_per_gpu].sum()
            for g in range(tp)
        ])
        mean_load = gpu_loads.mean()
        glr = gpu_loads.max() / mean_load if mean_load > 0 else 1.0

    return glr * math.sqrt(els)
