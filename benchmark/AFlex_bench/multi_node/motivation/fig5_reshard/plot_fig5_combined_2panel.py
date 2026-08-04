#!/usr/bin/env python3
"""Fig.5 two-panel — layout aligned with fig2_combined_2panel.pdf."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 14,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
AF_JSON = DATA_DIR / "af_all_tp_results.json"
DVFS_CSV = DATA_DIR / "dvfs_switch_summary_latest.csv"
STARTUP_JSON = DATA_DIR / "startup_overhead_by_tp.json"

FONT_BASE = 14
FONT_TITLE = 15
BAR_LABEL_KW = dict(
    ha="center", va="center", fontsize=FONT_BASE - 2,
    fontweight="bold", color="white",
)

TP_ORDER = [1, 2, 4, 8]
TP_LABELS = ["TP1", "TP2", "TP4", "TP8"]
XS = np.arange(len(TP_ORDER))

# Startup breakdown (no CUDA graph) — loaded from data/startup_overhead_by_tp.json
STARTUP_BUILD: list[float] = []
STARTUP_MAT: list[float] = []
STARTUP_INIT: list[float] = []
STARTUP_COLORS = {"build": "#3B6FA0", "mat": "#5BA67E", "init": "#E8853A"}


def load_startup_breakdown() -> None:
    global STARTUP_BUILD, STARTUP_MAT, STARTUP_INIT
    doc = json.loads(STARTUP_JSON.read_text())
    STARTUP_BUILD = list(doc["build_tp_comm_group_s"])
    STARTUP_MAT = list(doc["materialize_weight_kv_s"])
    STARTUP_INIT = list(doc["init_inference_engine_s"])

Y_TOP = (100, 530)
Y_BOT = (0, 80)
TOP_TICKS = [100, 300, 500]
BOT_TICKS = [25, 50, 75]

LEGEND_KW = dict(
    loc="lower center",
    fontsize=FONT_BASE - 1,
    frameon=True,
    framealpha=0.25,
    facecolor="white",
    edgecolor="none",
    handletextpad=0.5,
    handlelength=1.8,
)


def _bbox_fig(fig: plt.Figure, artist) -> object:
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer()).transformed(
        fig.transFigure.inverted()
    )


def _place_aligned_legends(
    fig: plt.Figure,
    handles_a, labels_a, cx_a: float,
    handles_b, labels_b, cx_b: float,
    subplot_top: float,
) -> float:
    """Place panel-(a) and panel-(b) legends with matched top/bottom edges."""
    h_build, h_mat, h_init = handles_a
    l_build, l_mat, l_init = labels_a
    row_kw = {
        "fontsize": FONT_BASE - 1,
        "frameon": True,
        "framealpha": 0.25,
        "facecolor": "white",
        "edgecolor": "none",
        "handletextpad": 0.25,
        "handlelength": 1.2,
        "borderaxespad": 0.0,
    }
    row_gap = 0.0
    legend_bottom = subplot_top + 0.006

    def _stack_panel_a(bottom: float) -> tuple[Legend, Legend, float]:
        leg2 = Legend(
            fig, [h_mat], [l_mat],
            bbox_to_anchor=(cx_a, bottom),
            bbox_transform=fig.transFigure,
            ncol=1, loc="lower center", **row_kw,
        )
        leg2.set_clip_on(False)
        fig.add_artist(leg2)
        bbox2 = _bbox_fig(fig, leg2)

        leg1 = Legend(
            fig, [h_build, h_init], [l_build, l_init],
            bbox_to_anchor=(cx_a, bbox2.y1 + row_gap),
            bbox_transform=fig.transFigure,
            ncol=2, columnspacing=0.2, loc="lower center", **row_kw,
        )
        leg1.set_clip_on(False)
        fig.add_artist(leg1)
        bbox1 = _bbox_fig(fig, leg1)

        leg2.remove()
        leg2 = Legend(
            fig, [h_mat], [l_mat],
            bbox_to_anchor=(bbox1.x0, bottom),
            bbox_transform=fig.transFigure,
            ncol=1, loc="lower left", **row_kw,
        )
        leg2.set_clip_on(False)
        fig.add_artist(leg2)
        return leg1, leg2, bbox1.y1

    _, _, legend_top_a = _stack_panel_a(legend_bottom)

    leg_b = None
    best_ls = 0.85
    for labelspacing in (0.30, 0.40, 0.50, 0.60, 0.70, 0.85, 1.0, 1.2, 1.4):
        if leg_b is not None:
            leg_b.remove()
        leg_b = Legend(
            fig, handles_b, labels_b,
            bbox_to_anchor=(cx_b, legend_bottom),
            bbox_transform=fig.transFigure,
            ncol=3, columnspacing=0.8, labelspacing=labelspacing,
            loc="lower center", **row_kw,
        )
        leg_b.set_clip_on(False)
        fig.add_artist(leg_b)
        bbox_b = _bbox_fig(fig, leg_b)
        best_ls = labelspacing
        if bbox_b.y1 >= legend_top_a - 0.001:
            break

    bbox_b = _bbox_fig(fig, leg_b)
    for leg in list(fig.artists):
        if isinstance(leg, Legend):
            leg.remove()

    legend_bottom = bbox_b.y0
    legend_top = bbox_b.y1

    leg_b = Legend(
        fig, handles_b, labels_b,
        bbox_to_anchor=(cx_b, legend_bottom),
        bbox_transform=fig.transFigure,
        ncol=3, columnspacing=0.8, labelspacing=best_ls,
        loc="lower center", **row_kw,
    )
    leg_b.set_clip_on(False)
    fig.add_artist(leg_b)

    leg1, leg2, top_a = _stack_panel_a(legend_bottom)

    # If panel-(b) is taller, keep row1 at the top edge.
    bbox1 = _bbox_fig(fig, leg1)
    if top_a < legend_top - 0.001:
        leg1.remove()
        leg1 = Legend(
            fig, [h_build, h_init], [l_build, l_init],
            bbox_to_anchor=(cx_a, legend_top),
            bbox_transform=fig.transFigure,
            ncol=2, columnspacing=0.2, loc="upper center", **row_kw,
        )
        leg1.set_clip_on(False)
        fig.add_artist(leg1)
        bbox1 = _bbox_fig(fig, leg1)
        leg2.remove()
        leg2 = Legend(
            fig, [h_mat], [l_mat],
            bbox_to_anchor=(bbox1.x0, legend_bottom),
            bbox_transform=fig.transFigure,
            ncol=1, loc="lower left", **row_kw,
        )
        leg2.set_clip_on(False)
        fig.add_artist(leg2)

    return legend_top


def _fig_legend(
    fig: plt.Figure, handles, labels, cx: float, legend_y: float, ncol: int,
    **overrides,
):
    kw = {**LEGEND_KW, "columnspacing": 1.0, **overrides}
    leg = Legend(
        fig, handles, labels,
        bbox_to_anchor=(cx, legend_y),
        bbox_transform=fig.transFigure,
        ncol=ncol,
        **kw,
    )
    leg.set_clip_on(False)
    fig.add_artist(leg)
    return leg


def _draw_break_marks(ax_top, ax_bottom, d=0.012):
    kw = dict(color="k", clip_on=False, linewidth=0.8)
    ax_top.plot((-d, +d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_top.plot((1 - d, 1 + d), (-d, +d), transform=ax_top.transAxes, **kw)
    ax_bottom.plot((-d, +d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kw)
    ax_bottom.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_bottom.transAxes, **kw)


def plot_panel_startup(ax: plt.Axes) -> tuple[list, list]:
    width = 0.55
    bottom = np.zeros(len(TP_ORDER))
    segments = [
        (STARTUP_BUILD, "Build TP comm group", STARTUP_COLORS["build"], True),
        (STARTUP_MAT, "Materialize weight & KV cache", STARTUP_COLORS["mat"], False),
        (STARTUP_INIT, "Init inference engine", STARTUP_COLORS["init"], False),
    ]
    for vals, label, color, skip_small in segments:
        ax.bar(XS, vals, width, bottom=bottom, color=color,
               edgecolor="white", linewidth=0.6, label=label)
        for i, (b, v) in enumerate(zip(bottom, vals)):
            if skip_small and i < 2:
                continue
            if v < 0.1:
                continue
            ax.text(XS[i], b + v / 2, f"{v:.1f}", **BAR_LABEL_KW)
        bottom = bottom + np.array(vals)

    ax.set_xticks(XS, TP_LABELS)
    ax.set_ylabel("Overhead (s)", fontsize=FONT_BASE, labelpad=4)
    ax.set_ylim(0, 14)
    ax.set_yticks([0, 2, 4, 6, 8, 10, 12, 14])
    ax.margins(x=0.04, y=0)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="both", labelsize=FONT_BASE - 1)
    return ax.get_legend_handles_labels()


def load_af_data() -> dict:
    return json.loads(AF_JSON.read_text())


def load_dvfs_p50() -> list[float]:
    df = pd.read_csv(DVFS_CSV)
    sub = df[df["ngpus"].isin(TP_ORDER)].sort_values("ngpus")
    return [float(r.wall_median_ms) for r in sub.itertuples()]


def plot_broken_af_lines(fig: plt.Figure, gs_spec) -> tuple:
    sub = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_spec, height_ratios=[1.0, 1.45], hspace=0.06,
    )
    ax_top = fig.add_subplot(sub[0])
    ax_bot = fig.add_subplot(sub[1])

    data = load_af_data()
    ttft_n = [data["native"][f"TP{t}"]["ttft_ms"] for t in TP_ORDER]
    ttft_a = [data["af"][f"TP{t}"]["ttft_ms"] for t in TP_ORDER]
    tpot_n = [data["native"][f"TP{t}"]["tpot_ms"] for t in TP_ORDER]
    tpot_a = [data["af"][f"TP{t}"]["tpot_ms"] for t in TP_ORDER]
    dvfs = load_dvfs_p50()

    ax_top.plot(XS, ttft_n, "o-", color="#4C72B0", lw=2, ms=5, label="TTFT")
    ax_top.plot(XS, ttft_a, "s--", color="#4C72B0", lw=2, ms=5, label="TTFT (AF)")
    ax_bot.plot(XS, tpot_n, "o-", color="#DD8452", lw=2, ms=5, label="TPOT")
    ax_bot.plot(XS, tpot_a, "s--", color="#DD8452", lw=2, ms=5, label="TPOT (AF)")
    ax_bot.plot(XS, dvfs, "^-.", color="#55A868", lw=2, ms=5, label="DVFS Switch")

    ax_top.set_ylim(*Y_TOP)
    ax_bot.set_ylim(*Y_BOT)
    ax_top.set_yticks(TOP_TICKS)
    ax_bot.set_yticks(BOT_TICKS)
    ax_bot.set_xticks(XS, TP_LABELS)

    for ax in (ax_top, ax_bot):
        ax.grid(axis="y", alpha=0.3)
        ax.margins(x=0.04)
        ax.tick_params(axis="both", labelsize=FONT_BASE - 1)

    ax_top.spines["bottom"].set_visible(False)
    ax_bot.spines["top"].set_visible(False)
    ax_top.tick_params(labelbottom=False, bottom=False)
    ax_bot.tick_params(top=False)

    _draw_break_marks(ax_top, ax_bot)

    handles, labels = [], []
    for ax in (ax_top, ax_bot):
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    # Reorder: group native + AF pairs for each metric
    # Desired: TTFT, TTFT (AF), DVFS Switch, TPOT, TPOT (AF)
    order = [0, 1, 4, 2, 3]
    handles = [handles[i] for i in order]
    labels = [labels[i] for i in order]

    return ax_top, ax_bot, handles, labels


def _panel_subtitle(fig: plt.Figure, ax: plt.Axes, text: str, y_fig: float) -> None:
    pos = ax.get_position()
    fig.text((pos.x0 + pos.x1) / 2, y_fig, text, ha="center", va="top",
             fontsize=FONT_TITLE)


def main():
    load_startup_breakdown()
    fig = plt.figure(figsize=(10.9, 4.6))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1.15, 1.17], wspace=0.18)
    ax_a = fig.add_subplot(gs[0])
    handles_a, labels_a = plot_panel_startup(ax_a)
    ax_top, ax_bot, handles_b, labels_b = plot_broken_af_lines(fig, gs[1])

    subplot_top = max(ax_a.get_position().y1, ax_top.get_position().y1)
    legend_top = 1.0
    for top_margin in (0.82, 0.78, 0.74, 0.70):
        for artist in list(fig.artists):
            if isinstance(artist, Legend):
                artist.remove()
        fig.subplots_adjust(wspace=0.18, left=0.10, right=0.97, top=top_margin, bottom=0.20)
        subplot_top = max(ax_a.get_position().y1, ax_top.get_position().y1)
        cx_a = (ax_a.get_position().x0 + ax_a.get_position().x1) / 2
        cx_b = (ax_bot.get_position().x0 + ax_bot.get_position().x1) / 2
        legend_top = _place_aligned_legends(
            fig, handles_a, labels_a, cx_a, handles_b, labels_b, cx_b, subplot_top,
        )
        if legend_top <= 0.998:
            break

    top_pos = ax_top.get_position()
    bot_pos = ax_bot.get_position()
    mid_y = (top_pos.y1 + bot_pos.y0) / 2
    fig.text(bot_pos.x0 - 0.056, mid_y, "Overhead (ms)", rotation=90,
             va="center", ha="center", fontsize=FONT_BASE)

    subtitle_y = min(ax_a.get_position().y0, ax_bot.get_position().y0) - 0.070
    _panel_subtitle(fig, ax_a, "(a) Instance Startup", subtitle_y)
    _panel_subtitle(fig, ax_bot, "(b) AF-Disaggregation & DVFS Overhead", subtitle_y)

    out = HERE / "fig5_combined_2panel.pdf"
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0)
    print(f"Wrote {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
