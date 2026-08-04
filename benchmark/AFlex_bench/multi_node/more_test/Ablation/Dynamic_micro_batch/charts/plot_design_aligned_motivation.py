#!/usr/bin/env python3
"""Design-aligned neighboring-layer A/F timelines for the paper."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_FILE = ROOT / "data" / "design_aligned_cases.json"
CHARTS_DIR = HERE
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

CASES = json.loads(DATA_FILE.read_text())["cases"]

A_COLORS = ["#BDD7EE", "#5B9BD5"]
F_COLORS = ["#FFE699", "#EDB84E"]
IDLE_COLOR = "#E7EAF0"
EDGE = "#425466"
TRANSFER_GAP = 0.0  # ms, A→F transmission overhead (set to 0 per paper convention)


def schedule(case, layers=1):
    """Earliest-start interleaved schedule on serialized A and F pools."""
    parts, a_lat, f_lat = case["parts"], case["A"], case["F"]
    a_available = f_available = 0.0
    f_end = {}
    events = []
    for layer in range(layers):
        for mb in range(len(parts)):
            a_start = max(a_available, f_end.get((layer - 1, mb), 0.0))
            a_end = a_start + a_lat[mb]
            events.append(
                {
                    "pool": "A",
                    "layer": layer,
                    "mb": mb,
                    "start": a_start,
                    "end": a_end,
                    "lat": a_lat[mb],
                    "bs": parts[mb],
                }
            )
            a_available = a_end
            f_start = max(f_available, a_end + TRANSFER_GAP)
            f_finish = f_start + f_lat[mb]
            events.append(
                {
                    "pool": "F",
                    "layer": layer,
                    "mb": mb,
                    "start": f_start,
                    "end": f_finish,
                    "lat": f_lat[mb],
                    "bs": parts[mb],
                }
            )
            f_available = f_finish
            f_end[layer, mb] = f_finish
    return events, max(a_available, f_available)


def idle_intervals(events, pool, start, end):
    intervals = sorted((e["start"], e["end"]) for e in events if e["pool"] == pool)
    idle, cursor = [], start
    for s, f in intervals:
        if s > cursor + 1e-9 and min(s, end) > cursor:
            idle.append((cursor, min(s, end)))
        cursor = max(cursor, f)
    if cursor < end - 1e-9:
        idle.append((cursor, end))
    return idle


PANEL_TITLE_GAP = 0.010
TIME_XLABEL_FONTSIZE = 18
TIME_XTICK_FONTSIZE = 15
LEGEND_FONTSIZE = 18


def _place_panel_title(fig: plt.Figure, ax_bot: plt.Axes, text: str, renderer) -> None:
    xlabel_bbox = ax_bot.xaxis.label.get_window_extent(renderer=renderer)
    fig_bbox = fig.transFigure.inverted().transform_bbox(xlabel_bbox)
    fig.text(
        fig_bbox.x0 + fig_bbox.width / 2,
        fig_bbox.y0 - PANEL_TITLE_GAP,
        text,
        ha="center",
        va="top",
        fontsize=18,
        fontweight="bold",
        transform=fig.transFigure,
    )


def draw_timeline(ax, case, label, show_y=False, label_fontsize=18):
    """Draw timeline; return (x_left, x_right) for caller to align across rows."""
    is_m2 = len(case["parts"]) == 2
    all_events, _ = schedule(case, layers=2)
    events = [
        e
        for e in all_events
        if e["layer"] == 0 or (e["layer"] == 1 and e["pool"] == "A")
    ]
    if is_m2:
        cycle_start = next(
            e["start"]
            for e in all_events
            if e["pool"] == "A" and e["layer"] == 0 and e["mb"] == 1
        )
        cycle_end = next(
            e["start"]
            for e in all_events
            if e["pool"] == "A" and e["layer"] == 1 and e["mb"] == 1
        )
    else:
        cycle_start = case["A"][0]
        a1_event = next(e for e in all_events if e["pool"] == "A" and e["layer"] == 1)
        cycle_end = a1_event["end"]
    layer_time = cycle_end - cycle_start
    if is_m2:
        a_next = next(e for e in events if e["layer"] == 1 and e["pool"] == "A")
        display_end = max(cycle_end, a_next["end"])
    else:
        display_end = cycle_end
    trimmed = []
    for e in events:
        s = max(e["start"], 0.0)
        en = min(e["end"], display_end)
        if s < en - 1e-9:
            trimmed.append({**e, "start": s, "end": en})
    events = trimmed
    x_left, x_right = 0.0, display_end

    lane_y = {"F": 0, "A": 1}
    h = 0.58
    idle_start_bound = cycle_start
    for pool in ("A", "F"):
        y = lane_y[pool]
        for idle_start, idle_end in idle_intervals(events, pool, idle_start_bound, x_right):
            ax.add_patch(
                Rectangle(
                    (idle_start, y - h / 2),
                    idle_end - idle_start,
                    h,
                    facecolor=IDLE_COLOR,
                    edgecolor="white",
                    linewidth=0.4,
                    hatch="////",
                    zorder=1,
                )
            )

    for e in events:
        y = lane_y[e["pool"]]
        palette = A_COLORS if e["pool"] == "A" else F_COLORS
        color = palette[e["mb"] % len(palette)]
        ax.barh(
            y,
            e["end"] - e["start"],
            left=e["start"],
            height=h,
            color=color,
            edgecolor=EDGE,
            linewidth=0.65,
            zorder=3,
        )
        symbol = "A" if e["pool"] == "A" else "F"
        mb = f",{e['mb']}" if is_m2 else ""
        width = e["end"] - e["start"]
        ax.text(
            e["start"] + width / 2,
            y,
            rf"${symbol}_{{{e['layer']}{mb}}}$",
            ha="center",
            va="center",
            fontsize=label_fontsize,
            zorder=4,
        )

    ax.annotate(
        "",
        xy=(cycle_end, 1.52),
        xytext=(cycle_start, 1.52),
        arrowprops=dict(arrowstyle="<->", color="black", lw=1.15),
    )
    ax.text(
        (cycle_start + cycle_end) / 2,
        1.56,
        f"{layer_time:.2f} ms",
        ha="center",
        va="bottom",
        fontsize=11.2,
        color="black",
        fontweight="bold",
    )

    total_bubble = 0.0
    all_cycle_spans = []
    for pool, y_arrow in [("A", 0.58), ("F", 0.58)]:
        cycle_compute = sorted(
            (e["start"], e["end"])
            for e in all_events
            if e["pool"] == pool and cycle_start <= e["start"] < cycle_end
        )
        cursor = cycle_start
        for comp_start, comp_end in cycle_compute:
            if comp_start > cursor + 1e-9:
                ax.annotate(
                    "",
                    xy=(comp_start, y_arrow),
                    xytext=(cursor, y_arrow),
                    arrowprops=dict(arrowstyle="<->", color="#C00000", lw=0.9),
                )
                all_cycle_spans.append((cursor, comp_start))
                total_bubble += comp_start - cursor
            cursor = max(cursor, comp_end)
        if cursor < cycle_end - 1e-9:
            ax.annotate(
                "",
                xy=(cycle_end, y_arrow),
                xytext=(cursor, y_arrow),
                arrowprops=dict(arrowstyle="<->", color="#C00000", lw=0.9),
            )
            all_cycle_spans.append((cursor, cycle_end))
            total_bubble += cycle_end - cursor
    if total_bubble > 1e-9 and all_cycle_spans:
        label_x = sum((a + b) / 2 for a, b in all_cycle_spans) / len(all_cycle_spans)
        ax.text(
            label_x,
            0.51,
            f"{total_bubble:.2f} ms",
            ha="center",
            va="top",
            fontsize=10.8,
            color="#C00000",
            fontweight="bold",
        )

    ax.text(
        0.5,
        1.08,
        label,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="#2ca02c",
    )
    ax.set_ylim(-0.62, 1.82)
    ax.set_yticks([0, 1], ["", ""])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.tick_params(axis="x", labelsize=10.5, length=2)
    ax.grid(axis="x", linestyle=":", alpha=0.25, zorder=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return x_left, x_right


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 17,
            "font.weight": "bold",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig = plt.figure(figsize=(12.5, 6.8))
    outer = fig.add_gridspec(
        1, 3, wspace=0.08, left=0.06, right=0.99, top=0.84, bottom=0.30
    )

    panels = [
        {
            "title": "(a) Low Load",
            "top": ("low_m1", "M=1, total bs=2"),
            "bottom": ("low_m2", "M=2, bs=1+1"),
        },
        {
            "title": "(b) High Load",
            "top": ("balanced_m1", "M=1, total bs=384"),
            "bottom": ("balanced_m2", "M=2, bs=192+192"),
        },
        {
            "title": "(c) Adaptive A/F batching",
            "top": ("balanced_m2", "Equal: bs=192+192"),
            "bottom": ("unequal", "Searched: bs=128+256"),
        },
    ]

    bottom_axes = []
    for col, panel in enumerate(panels):
        inner = outer[col].subgridspec(2, 1, hspace=0.32)
        top_case = CASES[panel["top"][0]]
        bot_case = CASES[panel["bottom"][0]]
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1])
        top_xl, top_xr = draw_timeline(ax_top, top_case, panel["top"][1], show_y=(col == 0))
        bot_fs = 16 if col in (0, 2) else 18
        bot_xl, bot_xr = draw_timeline(ax_bot, bot_case, panel["bottom"][1], show_y=(col == 0), label_fontsize=bot_fs)
        xl, xr = min(top_xl, bot_xl), max(top_xr, bot_xr) * 1.02
        ax_top.set_xlim(xl, xr)
        ax_bot.set_xlim(xl, xr)
        ax_top.set_xticklabels([])
        # xlabel removed per request
        # arrowhead at right end of x-axis spine for both top and bottom axes
        for ax in (ax_top, ax_bot):
            ax.annotate(
                "",
                xy=(1.0, 0),
                xytext=(0.985, 0),
                xycoords=("axes fraction", ax.get_xaxis_transform()),
                textcoords=("axes fraction", ax.get_xaxis_transform()),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.0),
                clip_on=False,
            )
        ax_bot.tick_params(axis="x", labelsize=TIME_XTICK_FONTSIZE, length=2)
        bottom_axes.append(ax_bot)

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax_bot, panel in zip(bottom_axes, panels):
        _place_panel_title(fig, ax_bot, panel["title"], renderer)

    legend = [
        Patch(facecolor=A_COLORS[0], edgecolor=EDGE, label="Attention compute"),
        Patch(facecolor=F_COLORS[0], edgecolor=EDGE, label="FFN compute"),
        Patch(facecolor=IDLE_COLOR, edgecolor="white", hatch="////", label="Pool bubble / idle"),
    ]
    fig.legend(
        handles=legend,
        ncol=3,
        loc="upper center",
        frameon=False,
        bbox_to_anchor=(0.5, 0.96),
        fontsize=LEGEND_FONTSIZE,
    )

    output_pdf = CHARTS_DIR / "dynamic_m_design_aligned.pdf"
    output_png = CHARTS_DIR / "dynamic_m_design_aligned.png"
    save_kw = {"bbox_inches": "tight", "pad_inches": 0}
    fig.savefig(output_pdf, **save_kw)
    fig.savefig(output_png, dpi=200, **save_kw)
    print(output_pdf)
    print(output_png)
    plt.close(fig)


if __name__ == "__main__":
    main()
