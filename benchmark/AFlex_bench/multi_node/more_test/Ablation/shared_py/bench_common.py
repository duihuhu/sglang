#!/usr/bin/env python3
"""Shared paths and Energy/Token helpers for Ablation breakdown figures."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.figure
    import matplotlib.pyplot as plt

SHARED_ROOT = Path(__file__).resolve().parent
ABLATION_ROOT = SHARED_ROOT.parent
AFLEX_ROOT = SHARED_ROOT.parents[3]
WORKLOAD_DIR = AFLEX_ROOT / "multi_node/more_test/macro/data/workloads"
PANEL_GAP_RATIO = 0.20  # middle spacer width ratio in gridspec
FIG_SIZE = (13.0, 3.5)
Y_LABEL = "Energy/Token (J)"
SAVE_PAD_INCHES = 0.0

# Paper typography aligned with macro/charts/plot_e2e_dashboard.py.
MICRO_REFERENCE_FIG_WIDTH = 9.0
MICRO_REFERENCE_FONT_SIZE = 14
MICRO_REFERENCE_PANEL_TITLE = 15
MICRO_REFERENCE_LEGEND = 13
PAPER_FONT_SCALE = FIG_SIZE[0] / MICRO_REFERENCE_FIG_WIDTH
FONT_SIZE = MICRO_REFERENCE_FONT_SIZE * PAPER_FONT_SCALE
PANEL_TITLE_FONTSIZE = MICRO_REFERENCE_PANEL_TITLE * PAPER_FONT_SCALE
LEGEND_FONT_SIZE = MICRO_REFERENCE_LEGEND * PAPER_FONT_SCALE

if str(AFLEX_ROOT) not in sys.path:
    sys.path.insert(0, str(AFLEX_ROOT))

from common.energy_per_token import (  # noqa: E402
    energy_per_token_j_all_tokens,
    energy_per_token_mj_all_tokens,
    parse_workload_key,
    workload_token_totals,
)


def apply_plot_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "figure.dpi": 150,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def create_ab_axes(fig: "matplotlib.figure.Figure") -> list:
    gs = fig.add_gridspec(1, 3, width_ratios=[1, PANEL_GAP_RATIO, 1], wspace=0)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2)]
    # Invisible middle panel keeps the (a)-(b) gap when bbox_inches='tight'.
    spacer = fig.add_subplot(gs[0, 1])
    spacer.axis("off")
    spacer.plot([0.5], [0.5], alpha=0.0, transform=spacer.transAxes)
    return axes


def style_energy_ylabel(ax) -> None:
    ax.set_ylabel(Y_LABEL, labelpad=2)


def finalize_ab_figure(fig: "matplotlib.figure.Figure") -> None:
    fig.subplots_adjust(left=0.11, right=0.99, top=0.80, bottom=0.16, wspace=0)


def save_ablation_clustered_figure(
    fig: "matplotlib.figure.Figure",
    pdf_path: Path,
    *,
    png_dpi: int = 200,
    save_png: bool = True,
) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"bbox_inches": "tight", "pad_inches": SAVE_PAD_INCHES}
    fig.savefig(pdf_path, **save_kw)
    print(f"Saved: {pdf_path}")
    if save_png:
        png_path = pdf_path.with_suffix(".png")
        fig.savefig(png_path, dpi=png_dpi, **save_kw)
        print(f"Saved: {png_path}")


def save_breakdown_figure(
    fig: "matplotlib.figure.Figure",
    stem: Path,
    *,
    paper_pdf: Path | None = None,
) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    save_kw = {"bbox_inches": "tight", "pad_inches": SAVE_PAD_INCHES}
    fig.savefig(stem.with_suffix(".pdf"), **save_kw)
    fig.savefig(stem.with_suffix(".png"), dpi=300, **save_kw)
    print(f"Saved: {stem.with_suffix('.pdf')}")
    if paper_pdf is not None:
        paper_pdf.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(paper_pdf, **save_kw)
        print(f"Saved: {paper_pdf}")


__all__ = [
    "ABLATION_ROOT",
    "AFLEX_ROOT",
    "FIG_SIZE",
    "PANEL_GAP_RATIO",
    "SHARED_ROOT",
    "WORKLOAD_DIR",
    "SAVE_PAD_INCHES",
    "FONT_SIZE",
    "PANEL_TITLE_FONTSIZE",
    "LEGEND_FONT_SIZE",
    "apply_plot_style",
    "create_ab_axes",
    "energy_per_token_j_all_tokens",
    "energy_per_token_mj_all_tokens",
    "finalize_ab_figure",
    "parse_workload_key",
    "save_ablation_clustered_figure",
    "save_breakdown_figure",
    "style_energy_ylabel",
    "Y_LABEL",
    "workload_token_totals",
]
