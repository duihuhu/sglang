#!/usr/bin/env python3
"""Deprecated: use Ablation/model_scalibility/charts/plot_moe_energy.py."""
from __future__ import annotations

import runpy
from pathlib import Path

runpy.run_path(
    str(Path(__file__).resolve().parent / "plot_moe_energy.py"),
    run_name="__main__",
)
