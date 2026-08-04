"""Helpers for mapping CUDA logical GPU indices to NVML physical GPU indices."""
from __future__ import annotations

import os


def nvml_gpu_id(logical_gpu_id: int) -> int:
    """Map CUDA logical GPU index to physical NVML GPU index.

    When CUDA_VISIBLE_DEVICES is set (e.g. "2,3"), torch cuda:0 maps to physical
    GPU 2, but NVML still uses physical indices. DVFS must lock the physical GPU.
    """
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not cvd:
        return logical_gpu_id
    parts = [p.strip() for p in cvd.split(",") if p.strip()]
    if len(parts) == 1:
        return int(parts[0])
    if logical_gpu_id < len(parts):
        return int(parts[logical_gpu_id])
    return logical_gpu_id
