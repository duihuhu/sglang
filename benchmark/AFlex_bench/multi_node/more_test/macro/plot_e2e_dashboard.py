#!/usr/bin/env python3
"""Plot end-to-end dashboards from macro_e2e_all.json (paper-aligned styling)."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, MaxNLocator

import bench_common as BC

# Paper typography aligned with micro/charts/plot_energy_only.py.
# Both figures are typically included at the same line width (e.g. \textwidth).
# Macro uses a wider canvas (13" vs 9"), so absolute pt sizes are scaled up so
# printed text matches micro's visual size.
MICRO_REFERENCE = {
    "fig_width": 9.0,
    "font_size": 14,
    "panel_title": 15,
    "legend": 13,
}
MACRO_FIG_WIDTH = 13.0
PAPER_FONT_SCALE = MACRO_FIG_WIDTH / MICRO_REFERENCE["fig_width"]

FONT_SIZE = MICRO_REFERENCE["font_size"] * PAPER_FONT_SCALE
PANEL_TITLE_FONTSIZE = MICRO_REFERENCE["panel_title"] * PAPER_FONT_SCALE
LEGEND_FONT_SIZE = MICRO_REFERENCE["legend"] * PAPER_FONT_SCALE

plt.rcParams.update({
    "font.size": FONT_SIZE,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MACRO_ROOT = Path(__file__).resolve().parent
CHARTS_DIR = MACRO_ROOT / "charts"

QPS_LIST = [2, 4, 8, 16]
QPS_TO_X = {q: i for i, q in enumerate(QPS_LIST)}

# Match Ablation/plot_without_megascale.py (MegaScale excluded).
PLOT_SERIES = [
    ("native_tp1_baseline", "SGLang", "#1f77b4", "-", "o", 3.8, 7),
    ("native_tp1_tier", "DynamoLLM", "#aec7e8", "--", "s", 3.8, 7),
    ("pd_hetero_baseline", "DistServe", "#ff7f0e", "-", "^", 3.8, 7),
    ("pd_hetero_tier_biscale", "BiScale", "#ffbb78", "--", "v", 3.8, 7),
    ("aflex_tier1", "AFlex", "#d62728", "-", "*", 4.6, 10),
]

DATASETS = [
    ("code", "Code", "With Code Trace", "end_to_end_code_trace"),
    ("conv", "Conversation", "With Conversation Trace", "end_to_end_conversation_trace"),
]

DEFAULT_PERCENTILE = "p90"
TTFT_SLO_MS = 400
TPOT_SLO_MS = 120
SLO_LINESTYLE = (0, (4, 3))
SLO_COLOR = "#333333"
SLO_ALPHA = 0.65
SLO_LINEWIDTH = 1.8
FIG_SIZE = (MACRO_FIG_WIDTH, 4.2)
LINE_WIDTH = 3.8
MARKER_SIZE = 7
WSPACE_AB = 0.34
WSPACE_BC = 0.32
PANEL_SUBTITLE_Y = -0.20
PANEL_C_SUBTITLE_X = 0.0
PANEL_C_YLABEL_PAD = 2
SLO_TEXT_OFFSET_FRAC = 0.06
XTICK_PAD = 8
ENERGY_YLABEL = "Energy per Token (J)"


def load_results(path: Path) -> dict:
    import json
    print(f"Loading {path}")
    return BC.flatten_results(json.loads(path.read_text()))


def wl_key(dataset: str, qps: int) -> str:
    return f"{dataset}_qps{qps}"


def get_series(data: dict, scheme: str, dataset: str, metric: str) -> dict[int, float]:
    pts = {}
    for q in QPS_LIST:
        m = data.get(scheme, {}).get(wl_key(dataset, q), {})
        if isinstance(m, dict) and m.get("status") == "PASS":
            v = m.get(metric)
            if v is not None:
                pts[q] = v
    return pts


def interp_percentile(p50: float | None, p99: float | None, pct: int) -> float | None:
    if p50 is None or p99 is None:
        return None
    return p50 + (pct - 50) / (99 - 50) * (p99 - p50)


def get_latency_series(
    data: dict, scheme: str, dataset: str, prefix: str, percentile: str,
) -> dict[int, float]:
    if percentile in ("p50", "p99", "avg"):
        return get_series(data, scheme, dataset, f"{prefix}_{percentile}_ms")
    if not (percentile.startswith("p") and percentile[1:].isdigit()):
        raise ValueError(f"Unsupported percentile: {percentile}")
    pct = int(percentile[1:])
    if pct == 90 or pct == 95:
        direct = get_series(data, scheme, dataset, f"{prefix}_p{pct}_ms")
        if direct:
            return direct
    if not 50 < pct < 99:
        raise ValueError(f"Unsupported percentile: {percentile}")
    p50_pts = get_series(data, scheme, dataset, f"{prefix}_p50_ms")
    p99_pts = get_series(data, scheme, dataset, f"{prefix}_p99_ms")
    pts = {}
    for q in p50_pts:
        v = interp_percentile(p50_pts[q], p99_pts.get(q), pct)
        if v is not None:
            pts[q] = v
    return pts


def _normalize_series(series: tuple) -> tuple:
    if len(series) == 5:
        return (*series, LINE_WIDTH, MARKER_SIZE)
    if len(series) == 7:
        return series
    raise ValueError(f"PLOT_SERIES entries must have 5 or 7 fields: {series!r}")


def _active_series(data: dict, dataset: str) -> list[tuple]:
    active = []
    for series in PLOT_SERIES:
        scheme, label, color, ls, mk, lw, ms = _normalize_series(series)
        if scheme not in data:
            continue
        if not get_series(data, scheme, dataset, "throughput_tok_s"):
            continue
        active.append((scheme, label, color, ls, mk, lw, ms))
    return active


def _qps_xs(qps_values: list[int]) -> list[int]:
    return [QPS_TO_X[q] for q in qps_values]


def _qps_tick_labels() -> list[str]:
    return [str(q) for q in QPS_LIST]


def _format_energy_y_tick(val: float, _pos: int) -> str:
    if abs(val) < 1e-9:
        return "0"
    return f"{val:.1f}"


def _subtitle_below_xlabel(
    ax,
    text: str,
    *,
    x: float = 0.5,
    ha: str = "center",
) -> None:
    ax.text(
        x,
        PANEL_SUBTITLE_Y,
        text,
        transform=ax.transAxes,
        ha=ha,
        va="top",
        fontsize=PANEL_TITLE_FONTSIZE,
        clip_on=False,
    )


def _style_axis(ax) -> None:
    ax.set_xticks(range(len(QPS_LIST)))
    ax.set_xticklabels(_qps_tick_labels())
    ax.grid(True, alpha=0.3)


def _apply_xtick_pad(ax) -> None:
    ax.tick_params(axis="x", pad=XTICK_PAD)


def _set_zero_based_ylim(ax) -> None:
    """Use major ticks as the exact zero-based y-axis bounds."""
    ax.autoscale(enable=True, axis="y")
    _, ymax = ax.get_ylim()
    if ymax <= 0:
        ymax = 1.0
    ticks = MaxNLocator(nbins=4).tick_values(0, ymax)
    ticks = ticks[ticks >= 0]
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_ylim(0, ticks[-1])
    ax.set_autoscaley_on(False)


def _latency_panel_config(prefix: str, percentile: str) -> tuple[str, str]:
    name = "TTFT" if prefix == "ttft_proc" else "TPOT"
    if percentile == "avg":
        return f"Avg {name} (ms)", f"Avg {name}"
    return f"{percentile.upper()} {name} (ms)", name


def _create_dashboard_axes(fig: plt.Figure) -> list:
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1, WSPACE_AB, 1, WSPACE_BC, 1],
        wspace=0,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2, 4)]
    for spacer_idx in (1, 3):
        spacer = fig.add_subplot(gs[0, spacer_idx])
        spacer.axis("off")
    return axes


def plot_dataset_dashboard_3panel(
    data: dict,
    dataset: str,
    trace_suffix: str,
    out_path: Path,
    ttft_percentile: str = "p90",
    tpot_percentile: str = "p90",
) -> None:
    fig = plt.figure(figsize=FIG_SIZE)
    axes = _create_dashboard_axes(fig)

    ax = axes[0]
    for scheme, label, color, ls, mk, lw, ms in _active_series(data, dataset):
        pts = get_series(data, scheme, dataset, "energy_per_token_mj")
        if not pts:
            continue
        xs = _qps_xs(sorted(pts))
        ys = [pts[q] / 1000.0 for q in sorted(pts)]
        ax.plot(
            xs, ys, color=color, linestyle=ls, marker=mk,
            markersize=ms, linewidth=lw, label=label,
        )
    ax.set_ylabel(ENERGY_YLABEL)
    _style_axis(ax)
    _subtitle_below_xlabel(ax, "(a) Energy per Token")

    latency_panels = [
        ("ttft_proc", ttft_percentile),
        ("tpot", tpot_percentile),
    ]
    for ax_idx, (prefix, percentile) in enumerate(latency_panels, start=1):
        ylabel, panel_title = _latency_panel_config(prefix, percentile)
        ax = axes[ax_idx]
        for scheme, label, color, ls, mk, lw, ms in _active_series(data, dataset):
            pts = get_latency_series(data, scheme, dataset, prefix, percentile)
            if not pts:
                continue
            xs = _qps_xs(sorted(pts))
            ys = [pts[q] for q in sorted(pts)]
            ax.plot(
                xs, ys, color=color, linestyle=ls, marker=mk,
                markersize=ms, linewidth=lw, label=label,
            )
        if prefix == "tpot":
            ax.set_ylabel(ylabel, labelpad=PANEL_C_YLABEL_PAD)
        else:
            ax.set_ylabel(ylabel)
        panel_letter = "b" if prefix == "ttft_proc" else "c"
        _style_axis(ax)
        slo_ms = TTFT_SLO_MS if prefix == "ttft_proc" else TPOT_SLO_MS
        ax.axhline(
            y=slo_ms,
            color=SLO_COLOR,
            linestyle=SLO_LINESTYLE,
            linewidth=SLO_LINEWIDTH,
            alpha=SLO_ALPHA,
        )
        _subtitle_below_xlabel(ax, f"({panel_letter}) {panel_title}")

    fig.subplots_adjust(left=0.08, right=0.99, top=0.82, bottom=0.24)
    for ax in axes:
        _set_zero_based_ylim(ax)
        _apply_xtick_pad(ax)
    axes[0].yaxis.set_major_formatter(FuncFormatter(_format_energy_y_tick))
    tpot_ticks = [0, 50, 100, 150]
    axes[2].yaxis.set_major_locator(FixedLocator(tpot_ticks))
    axes[2].set_ylim(0, tpot_ticks[-1])

    for ax, slo_ms in [(axes[1], TTFT_SLO_MS), (axes[2], TPOT_SLO_MS)]:
        _, ymax = ax.get_ylim()
        slo_text_y = slo_ms - ymax * SLO_TEXT_OFFSET_FRAC
        ax.text(
            0.02, slo_text_y,
            f"SLO={slo_ms}ms",
            color=SLO_COLOR,
            fontsize=FONT_SIZE * 0.75,
            alpha=SLO_ALPHA,
            ha="left",
            va="top",
            transform=ax.get_yaxis_transform(),
        )

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=LEGEND_FONT_SIZE,
        loc="upper center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.535, 0.98),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {pdf_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=BC.ALL_DATA_FILE)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(f"Data file not found: {args.input}")

    data = load_results(args.input)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    for ds, _title, _trace_suffix, out_stem in DATASETS:
        plot_dataset_dashboard_3panel(
            data,
            ds,
            _trace_suffix,
            CHARTS_DIR / out_stem,
            ttft_percentile=DEFAULT_PERCENTILE,
            tpot_percentile=DEFAULT_PERCENTILE,
        )


if __name__ == "__main__":
    main()
