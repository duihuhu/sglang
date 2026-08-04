#!/usr/bin/env python3
"""Plot a single figure with 4 subplots showing Energy per Token for all 4 datasets.
- Shared legend at bottom
- Font size increased by 4pt from base (10 -> 14)
- Line width increased by 2pt from base
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FixedLocator, FuncFormatter
import numpy as np

import bench_common as BC

BASE_FONT_SIZE = 10
FONT_INCREASE = 4
FONT_SIZE = BASE_FONT_SIZE + FONT_INCREASE
AXIS_FONT_SIZE = 16
LEGEND_FONT_SIZE = 15

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MICRO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = MICRO_ROOT / "data"
CHARTS_DIR = Path(__file__).resolve().parent

QPS_LIST = [2, 4, 8, 16]
QPS_TO_X = {q: i for i, q in enumerate(QPS_LIST)}

BASE_LW = 1.8
LW_INCREASE = 2
BASE_MS = 5
MS_INCREASE = 2

PLOT_SERIES = [
    ("native_tp1_baseline", "SGLang", "#1f77b4", "-", "o"),
    ("native_tp1_tier", "DynamoLLM", "#aec7e8", "--", "s"),
    ("pd_hetero_baseline", "DistServe", "#ff7f0e", "-", "^"),
    ("pd_hetero_tier_biscale", "BiScale", "#ffbb78", "--", "v"),
    ("aflex", "AFlex", "#d62728", "-", "*"),
]

def _dataset_label(ds: str, name: str) -> str:
    meta = BC.DATASET_META.get(ds, {})
    inp = meta.get("input_len", "?")
    out = meta.get("output_len", "?")
    return f"{name}({inp}-input/{out}-output)"


DATASETS = [
    ("qa_lpld", _dataset_label("qa_lpld", "QA")),
    ("chatbot_lphd", _dataset_label("chatbot_lphd", "Chatbot")),
    ("rag_hpld", _dataset_label("rag_hpld", "RAG")),
    ("summary_hphd", _dataset_label("summary_hphd", "Summary")),
]

PANEL_SUBTITLE_Y = -0.16
ENERGY_YLABEL = "Energy per Token (J)"
LEFT_YLABEL_AXES = (0, 2)
RIGHT_COLUMN_AXES = (1, 3)
LEFT_YLABEL_X = -0.14
WSPACE = 0.15
HSPACE = 0.32
FIG_LEFT = 0.12
FIG_RIGHT = 0.97
FIG_TOP = 0.91
FIG_BOTTOM = 0.17
LEGEND_PAD = 0.001


def _qps_tick_labels() -> list[str]:
    return [f"{q}" for q in QPS_LIST]


def _subtitle_below_xlabel(ax, text: str) -> None:
    ax.text(
        0.5,
        PANEL_SUBTITLE_Y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=AXIS_FONT_SIZE,
        clip_on=False,
    )


def _format_y_tick(val: float, _pos: int) -> str:
    if abs(val) < 1e-9:
        return "0"
    return f"{val:g}"


def load_data() -> dict:
    flat: dict[str, dict] = {}
    for ds in BC.DATASETS:
        path = BC.dataset_data_file(ds)
        if not path.exists():
            continue
        part = BC.flatten_results(json.loads(path.read_text()))
        for scheme, bucket in part.items():
            flat.setdefault(scheme, {}).update(bucket)
    return flat


def get_energy_series(data: dict, scheme: str, dataset: str) -> dict[int, float]:
    pts = {}
    for q in QPS_LIST:
        entry = data.get(scheme, {}).get(BC.wl_key(dataset, q), {})
        if isinstance(entry, dict) and entry.get("status") == "PASS":
            v = entry.get("energy_per_token_mj")
            if v is not None:
                pts[q] = v
    return pts


def _force_ylim_from_zero(ax) -> None:
    ax.relim()
    ax.autoscale_view()
    _top = ax.get_ylim()[1]
    if _top <= 0:
        _top = 1.0
    loc = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10])
    ticks = loc.tick_values(0, _top)
    ticks = [t for t in ticks if t >= 0]
    if not ticks or ticks[-1] < _top:
        ticks.append(_top * 1.05)
    ax.set_ylim(0, ticks[-1])
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_y_tick))


def _align_axes_column(ax_top, ax_bottom) -> None:
    """Keep the plot boxes in one column aligned despite different tick-label widths."""
    pos_top = ax_top.get_position()
    pos_bottom = ax_bottom.get_position()
    x0 = max(pos_top.x0, pos_bottom.x0)
    width = min(pos_top.width, pos_bottom.width)
    ax_top.set_position([x0, pos_top.y0, width, pos_top.height])
    ax_bottom.set_position([x0, pos_bottom.y0, width, pos_bottom.height])


def _align_left_ylabels(ax_top, ax_bottom) -> None:
    for ax in (ax_top, ax_bottom):
        ax.yaxis.set_label_coords(LEFT_YLABEL_X, 0.5)


def main():
    data = load_data()
    if not data:
        print("No data found. Check data files in", DATA_DIR)
        return

    fig, axes = plt.subplots(2, 2, figsize=(10, 5.8))
    axes = axes.flatten()

    all_handles = []
    all_labels = []

    for idx, (ds, title) in enumerate(DATASETS):
        ax = axes[idx]
        for scheme, label, color, ls, mk in PLOT_SERIES:
            if scheme not in data:
                continue
            pts = get_energy_series(data, scheme, ds)
            if not pts:
                continue
            xs = [QPS_TO_X[q] for q in sorted(pts)]
            ys = [pts[q] / 1000.0 for q in sorted(pts)]
            lw = BASE_LW + LW_INCREASE
            ms = BASE_MS + MS_INCREASE
            if scheme == "aflex":
                lw = 2.6 + LW_INCREASE
                ms = 8 + MS_INCREASE
            line, = ax.plot(xs, ys, color=color, linestyle=ls, marker=mk,
                           markersize=ms, linewidth=lw, label=label)
            if idx == 0:
                all_handles.append(line)
                all_labels.append(label)

        ax.set_xticks(range(len(QPS_LIST)))
        ax.set_xticklabels(_qps_tick_labels(), fontsize=AXIS_FONT_SIZE)
        ax.tick_params(axis="both", labelsize=AXIS_FONT_SIZE)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.yaxis.set_major_formatter(FuncFormatter(_format_y_tick))
        ax.grid(True, alpha=0.3)
        _force_ylim_from_zero(ax)

        panel_letter = chr(ord("a") + idx)
        _subtitle_below_xlabel(ax, f"({panel_letter}) {title}")

        if idx in LEFT_YLABEL_AXES:
            ax.set_ylabel(ENERGY_YLABEL, labelpad=2, fontsize=AXIS_FONT_SIZE)
        elif idx in RIGHT_COLUMN_AXES:
            ax.set_ylabel(" ", labelpad=2, fontsize=AXIS_FONT_SIZE)

    fig.subplots_adjust(
        left=FIG_LEFT,
        right=FIG_RIGHT,
        top=FIG_TOP,
        bottom=FIG_BOTTOM,
        wspace=WSPACE,
        hspace=HSPACE,
    )
    _align_axes_column(axes[0], axes[2])
    _align_axes_column(axes[1], axes[3])
    _align_left_ylabels(axes[0], axes[2])

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tight = fig.get_tightbbox(renderer)
    fig_w, fig_h = fig.get_size_inches()
    left_plot_frac = axes[0].get_position().x0
    legend_x = (left_plot_frac + tight.x1 / fig_w) / 2
    legend_y = tight.y1 / fig_h + LEGEND_PAD
    fig.legend(
        all_handles, all_labels,
        fontsize=LEGEND_FONT_SIZE,
        loc="lower center",
        ncol=len(all_handles),
        frameon=False,
        columnspacing=0.6,
        handletextpad=0.3,
        bbox_to_anchor=(legend_x, legend_y),
        bbox_transform=fig.transFigure,
    )

    out_pdf = CHARTS_DIR / "micro_energy_only.pdf"
    out_png = CHARTS_DIR / "micro_energy_only.png"
    save_kw = {"dpi": 200, "bbox_inches": "tight", "pad_inches": 0.05}
    fig.savefig(out_pdf, **save_kw)
    fig.savefig(out_png, **save_kw)
    plt.close(fig)
    print(f"Saved {out_pdf}")
    print(f"Saved {out_png}")


if __name__ == "__main__":
    main()
