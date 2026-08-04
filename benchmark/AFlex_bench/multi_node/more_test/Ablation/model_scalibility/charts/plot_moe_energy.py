#!/usr/bin/env python3
"""Plot Mixtral MoE Energy/Token clustered bar chart (code + conv)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SHARED_ROOT = ROOT.parent / "shared_py"
sys.path.insert(0, str(SHARED_ROOT))
import bench_common as BC

DEFAULT_INPUT = ROOT / "data" / "plan_moe_e2e.json"
CHARTS_DIR = HERE
PAPER_PDF = HERE / "moe_energy_clustered.pdf"

SYSTEMS = [
    ("native_dp_baseline", "SGLang"),
    ("native_dp_tier", "DynamoLLM"),
    ("pd_dp_baseline", "DistServe"),
    ("pd_dp_tier", "BiScale"),
    ("pdaf_baseline", "MegaScale"),
    ("pdaf_tier", "AFlex"),
]
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#9467bd", "#d62728"]
QPS_LIST = [2, 4, 8, 16]
PANEL_GAP_RATIO = 0.13
SUBTITLE_LABELPAD = 2
ENERGY_YLABEL = "Energy per Token (J)"
LEGEND_PAD = -0.04


def _qps_tick_labels() -> list[str]:
    return [str(q) for q in QPS_LIST]


def _subtitle_below_xlabel(ax, text: str) -> None:
    ax.set_xlabel(text, fontsize=BC.PANEL_TITLE_FONTSIZE, labelpad=SUBTITLE_LABELPAD)


def _create_ab_axes(fig: plt.Figure) -> list[plt.Axes]:
    gs = fig.add_gridspec(1, 3, width_ratios=[1, PANEL_GAP_RATIO, 1], wspace=0)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2)]
    spacer = fig.add_subplot(gs[0, 1])
    spacer.axis("off")
    spacer.plot([0.5], [0.5], alpha=0.0, transform=spacer.transAxes)
    return axes


def load_data(path: Path) -> dict:
    return json.loads(path.read_text())["results"]


def get_energy_array(results: dict, dataset: str, systems: list[tuple[str, str]]) -> np.ndarray:
    rows = []
    for qps in QPS_LIST:
        key = f"{dataset}_qps{qps}"
        row = []
        for sys_key, _ in systems:
            entry = results.get(sys_key, {}).get(key, {})
            val = BC.energy_per_token_j_all_tokens(entry, key)
            row.append(val if entry.get("status") == "PASS" and val > 0 else 0)
        rows.append(row)
    return np.asarray(rows)


def plot_panel(
    ax: plt.Axes,
    values: np.ndarray,
    title: str,
    systems: list[tuple[str, str]],
    colors: list[str],
) -> None:
    x = np.arange(len(QPS_LIST), dtype=float)
    n_sys = len(systems)
    bar_width = 0.84 / n_sys
    for idx, ((_, label), color) in enumerate(zip(systems, colors)):
        offset = (idx - (n_sys - 1) / 2) * bar_width
        vals = values[:, idx]
        mask = vals > 0
        ax.bar(
            x[mask] + offset,
            vals[mask],
            width=bar_width * 0.92,
            label=label,
            color=color,
            edgecolor="none",
            linewidth=0,
            zorder=3,
        )
    ax.set_xticks(x, _qps_tick_labels())
    ax.set_ylabel(ENERGY_YLABEL, labelpad=2)
    _subtitle_below_xlabel(ax, title)
    ax.grid(True, alpha=0.3)
    ax.spines["right"].set_visible(True)


def plot_figure(
    results: dict,
    *,
    exclude_systems: tuple[str, ...] = (),
    output_stem: Path | None = None,
    paper_pdf: Path | None = None,
) -> None:
    systems = [s for s in SYSTEMS if s[0] not in exclude_systems]
    colors = [c for (s, _), c in zip(SYSTEMS, COLORS) if s not in exclude_systems]

    BC.apply_plot_style()
    fig = plt.figure(figsize=BC.FIG_SIZE)
    axes = _create_ab_axes(fig)
    conv_vals = get_energy_array(results, "conv", systems)
    code_vals = get_energy_array(results, "code", systems)
    plot_panel(axes[0], conv_vals, "(a) Conversation", systems, colors)
    plot_panel(axes[1], code_vals, "(b) Coding", systems, colors)
    for ax, vals in zip(axes, (conv_vals, code_vals)):
        positive = vals[vals > 0]
        if positive.size:
            top = float(positive.max()) * 1.12
            locator = MaxNLocator(nbins=4)
            ticks = locator.tick_values(0, top)
            top_tick = ticks[ticks >= positive.max()][0]
            visible_ticks = ticks[(ticks >= 0) & (ticks <= top_tick)]
            ax.set_ylim(0, top_tick)
            ax.set_yticks(visible_ticks)
    # Panel (b) Code: fixed y ticks
    axes[1].set_ylabel("")
    axes[1].set_yticks([0, 0.2, 0.4, 0.6])
    axes[1].set_ylim(0, 0.6)
    # Format y=0 without decimals
    from matplotlib.ticker import FuncFormatter
    def _fmt_y(val, _pos):
        if val == 0:
            return "0"
        return f"{val:g}"
    for ax in axes:
        ax.yaxis.set_major_formatter(FuncFormatter(_fmt_y))
    fig.subplots_adjust(left=0.11, right=0.95, top=0.90, bottom=0.20, wspace=0)
    legend_handles = [
        Patch(facecolor=color, edgecolor="none", label=label)
        for (_, label), color in zip(systems, colors)
    ]
    fig.canvas.draw()
    pos_a = axes[0].get_position()
    pos_b = axes[1].get_position()
    legend_x = (pos_a.x0 + pos_b.x1) / 2
    legend_y = max(pos_a.y1, pos_b.y1) + LEGEND_PAD
    fig.legend(
        handles=legend_handles,
        fontsize=BC.LEGEND_FONT_SIZE,
        loc="lower center",
        ncol=len(systems),
        frameon=False,
        columnspacing=2,
        handletextpad=0.4,
        bbox_to_anchor=(legend_x, legend_y),
        bbox_transform=fig.transFigure,
    )
    output_stem = output_stem or (CHARTS_DIR / "moe_energy_clustered")
    BC.save_ablation_clustered_figure(fig, output_stem.with_suffix(".pdf"))
    if paper_pdf is not None and paper_pdf != output_stem.with_suffix(".pdf"):
        BC.save_ablation_clustered_figure(fig, paper_pdf, save_png=False)
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--paper", action="store_true", help="exclude MegaScale for paper figure")
    args = parser.parse_args()
    exclude = ("pdaf_baseline",)
    plot_figure(
        load_data(args.input),
        exclude_systems=exclude,
        paper_pdf=PAPER_PDF,
    )


if __name__ == "__main__":
    main()
