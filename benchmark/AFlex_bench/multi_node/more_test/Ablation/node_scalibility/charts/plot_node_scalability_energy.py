#!/usr/bin/env python3
"""Plot node-scalability Energy/Token for code and conversation workloads."""
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

DATA_FILE = ROOT / "data" / "node_scalability_all.json"
CHARTS_DIR = ROOT / "charts"
OUTPUT_PDF = CHARTS_DIR / "node_scalability_energy_clustered.pdf"

SYSTEMS = ["sglang", "dynamollm", "distserve", "biscale", "aflex"]
SYSTEM_LABELS = ["SGLang", "DynamoLLM", "DistServe", "BiScale", "AFlex"]
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#d62728"]
NODES = [1, 2, 4]
PANEL_GAP_RATIO = 0.13
SUBTITLE_LABELPAD = 2
ENERGY_YLABEL = "Energy per Token (J)"
LEGEND_PAD = -0.04


def _node_tick_labels() -> list[str]:
    return [str(n) for n in NODES]


def _subtitle_below_xlabel(ax, text: str) -> None:
    ax.set_xlabel(text, fontsize=BC.PANEL_TITLE_FONTSIZE, labelpad=SUBTITLE_LABELPAD)


def _create_ab_axes(fig: plt.Figure) -> list[plt.Axes]:
    gs = fig.add_gridspec(1, 3, width_ratios=[1, PANEL_GAP_RATIO, 1], wspace=0)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2)]
    spacer = fig.add_subplot(gs[0, 1])
    spacer.axis("off")
    spacer.plot([0.5], [0.5], alpha=0.0, transform=spacer.transAxes)
    return axes


def _load_all() -> dict:
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def load_energy_per_token(dataset: str) -> np.ndarray:
    data = _load_all()
    rows = []
    for node in NODES:
        bucket = data["results"][dataset][str(node)]
        rows.append(
            [
                BC.energy_per_token_j_all_tokens(bucket[system], bucket[system]["workload"])
                if system in bucket
                else float("nan")
                for system in SYSTEMS
            ]
        )
    return np.asarray(rows)


def plot_panel(ax: plt.Axes, values: np.ndarray, title: str) -> None:
    x = np.arange(len(NODES), dtype=float)
    bar_width = 0.84 / len(SYSTEMS)
    for system_idx, (label, color) in enumerate(zip(SYSTEM_LABELS, COLORS)):
        offsets = (system_idx - (len(SYSTEMS) - 1) / 2) * bar_width
        ax.bar(
            x + offsets,
            values[:, system_idx],
            width=bar_width * 0.92,
            label=label,
            color=color,
            edgecolor="none",
            linewidth=0,
            zorder=3,
        )
    ax.set_xticks(x, _node_tick_labels())
    ax.set_ylabel(ENERGY_YLABEL, labelpad=2)
    _subtitle_below_xlabel(ax, title)
    vmax = float(np.nanmax(values)) if values.size else 1.0
    locator = MaxNLocator(nbins=4)
    ticks = locator.tick_values(0, vmax * 1.05)
    top_tick = ticks[ticks >= vmax][0]
    visible_ticks = ticks[(ticks >= 0) & (ticks <= top_tick)]
    ax.set_ylim(0, top_tick)
    ax.set_yticks(visible_ticks)
    ax.grid(True, alpha=0.3)
    ax.spines["right"].set_visible(True)


def plot_figure(*, exclude_systems: tuple[str, ...] = ()) -> None:
    global SYSTEMS, SYSTEM_LABELS, COLORS
    orig = (SYSTEMS, SYSTEM_LABELS, COLORS)
    retained = [
        (s, label, color)
        for s, label, color in zip(SYSTEMS, SYSTEM_LABELS, COLORS)
        if s not in exclude_systems
    ]
    SYSTEMS = [x[0] for x in retained]
    SYSTEM_LABELS = [x[1] for x in retained]
    COLORS = [x[2] for x in retained]

    BC.apply_plot_style()
    fig = plt.figure(figsize=BC.FIG_SIZE)
    axes = _create_ab_axes(fig)
    plot_panel(axes[0], load_energy_per_token("conv"), "(a) Conversation")
    plot_panel(axes[1], load_energy_per_token("code"), "(b) Coding")
    # Panel (b) Code: fixed y ticks
    axes[1].set_ylabel("")
    axes[1].set_yticks([0, 0.1, 0.2])
    axes[1].set_ylim(0, 0.2)
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
        for label, color in zip(SYSTEM_LABELS, COLORS)
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
        ncol=len(SYSTEMS),
        frameon=False,
        columnspacing=2,
        handletextpad=0.4,
        bbox_to_anchor=(legend_x, legend_y),
        bbox_transform=fig.transFigure,
    )
    BC.save_ablation_clustered_figure(fig, OUTPUT_PDF)
    plt.close(fig)

    SYSTEMS, SYSTEM_LABELS, COLORS = orig


def main() -> None:
    plot_figure()


if __name__ == "__main__":
    main()
