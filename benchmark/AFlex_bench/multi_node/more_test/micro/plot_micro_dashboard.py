#!/usr/bin/env python3
"""Plot micro-benchmark dashboards from per-dataset micro_e2e_*.json files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

import bench_common as BC

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

MICRO_ROOT = Path(__file__).resolve().parent
CHARTS_DIR = MICRO_ROOT / "charts"

QPS_LIST = [2, 4, 8, 16]
QPS_TO_X = {q: i for i, q in enumerate(QPS_LIST)}

PLOT_SERIES = [
    ("native_tp1_baseline", "SGLang", "#1f77b4", "-", "o", 1.8, 5),
    ("native_tp1_tier", "DynamoLLM", "#aec7e8", "--", "s", 1.8, 5),
    ("pd_hetero_baseline", "DistServe", "#ff7f0e", "-", "^", 1.8, 5),
    ("pd_hetero_tier_biscale", "BiScale", "#ffbb78", "--", "v", 1.8, 5),
    ("aflex", "AFlex", "#d62728", "-", "*", 2.6, 8),
]

DATASETS = [
    ("qa_lpld", "QA (il=128, ol=64)", "Low Prefill Low Decode"),
    ("chatbot_lphd", "Chatbot (il=128, ol=1024)", "Low Prefill High Decode"),
    ("rag_hpld", "RAG (il=4096, ol=64)", "High Prefill Low Decode"),
    ("summary_hphd", "Summary (il=4096, ol=1024)", "High Prefill High Decode"),
]

PLOT_PERCENTILES = ("p50", "p90", "p95", "avg")
COMBINED_PERCENTILES = ("p50", "p90", "p95", "p99")
COMBINED_GRID_NAME = "micro_latency_4x8"
COMBINED_ROWS = [
    ("qa_lpld", "ttft_proc", "QA TTFT"),
    ("qa_lpld", "tpot", "QA TPOT"),
    ("chatbot_lphd", "ttft_proc", "Chatbot TTFT"),
    ("chatbot_lphd", "tpot", "Chatbot TPOT"),
    ("rag_hpld", "ttft_proc", "RAG TTFT"),
    ("rag_hpld", "tpot", "RAG TPOT"),
    ("summary_hphd", "ttft_proc", "Summary TTFT"),
    ("summary_hphd", "tpot", "Summary TPOT"),
]

FIG_TITLE_BASE = "Microbenchmark"
FIG_TITLE_FONTSIZE = 14
PANEL_TITLE_FONTSIZE = 12
PANEL_TITLE_PAD = 5
FIG_SIZE = (10.8, 3.5)
LINE_WIDTH = 1.8
MARKER_SIZE = 5
WSPACE_AB = 0.22
WSPACE_BC = 0.21

REQUEST_LATENCY_FIELDS = {
    "ttft_proc": "ttft_proc_ms",
    "tpot": "tpot_ms",
}


def load_results(path: Path | None = None) -> dict:
    if path is not None:
        print(f"Loading {path}")
        return BC.flatten_results(json.loads(path.read_text()))
    flat = BC.load_resume()
    if not flat:
        raise FileNotFoundError(f"No micro dataset files found under {BC.DATA_DIR}")
    print(f"Loading {len(BC.DATASETS)} dataset files from {BC.DATA_DIR}")
    return flat


def get_series(data: dict, scheme: str, dataset: str, metric: str) -> dict[int, float]:
    pts = {}
    for q in QPS_LIST:
        m = data.get(scheme, {}).get(BC.wl_key(dataset, q), {})
        if isinstance(m, dict) and m.get("status") == "PASS":
            v = m.get(metric)
            if v is not None:
                pts[q] = v
    return pts


def interp_percentile(p50: float | None, p99: float | None, pct: int) -> float | None:
    if p50 is None or p99 is None:
        return None
    return p50 + (pct - 50) / (99 - 50) * (p99 - p50)


def _percentile_from_requests(entry: dict, field: str, pct: int) -> float | None:
    values = [
        req[field]
        for req in entry.get("request_results") or []
        if req.get("success") and req.get(field) is not None
    ]
    if not values:
        return None
    return float(np.percentile(values, pct))


def get_latency_value(entry: dict, prefix: str, percentile: str) -> float | None:
    if percentile == "avg":
        return entry.get(f"{prefix}_avg_ms")
    if percentile in ("p50", "p99"):
        return entry.get(f"{prefix}_{percentile}_ms")
    if not (percentile.startswith("p") and percentile[1:].isdigit()):
        raise ValueError(f"Unsupported percentile: {percentile}")
    pct = int(percentile[1:])
    if not 50 < pct < 99:
        raise ValueError(f"Unsupported percentile: {percentile}")
    stored = entry.get(f"{prefix}_{percentile}_ms")
    if stored is not None:
        return stored
    field = REQUEST_LATENCY_FIELDS[prefix]
    direct = _percentile_from_requests(entry, field, pct)
    if direct is not None:
        return direct
    return interp_percentile(entry.get(f"{prefix}_p50_ms"), entry.get(f"{prefix}_p99_ms"), pct)


def get_latency_series(
    data: dict, scheme: str, dataset: str, prefix: str, percentile: str,
) -> dict[int, float]:
    pts = {}
    for q in QPS_LIST:
        entry = data.get(scheme, {}).get(BC.wl_key(dataset, q), {})
        if not isinstance(entry, dict) or entry.get("status") != "PASS":
            continue
        value = get_latency_value(entry, prefix, percentile)
        if value is not None:
            pts[q] = value
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


def _style_axis(ax) -> None:
    ax.set_xticks(range(len(QPS_LIST)))
    ax.set_xticklabels([str(q) for q in QPS_LIST])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, alpha=0.3)


def _force_ylim_from_zero(ax, integer_ticks: bool = False) -> None:
    """Pin y-axis from 0, upper bound at a nice tick above data."""
    ax.relim()
    ax.autoscale_view()
    _top = ax.get_ylim()[1]
    if _top <= 0:
        _top = 1.0
    from matplotlib.ticker import MaxNLocator, FixedLocator
    loc = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10], integer=integer_ticks)
    ticks = loc.tick_values(0, _top)
    ticks = [t for t in ticks if t >= 0]
    if not ticks or ticks[-1] < _top:
        ticks.append(_top * 1.05)
    ax.set_ylim(0, ticks[-1])
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.set_autoscaley_on(False)


def _latency_panel_config(prefix: str, percentile: str) -> tuple[str, str]:
    name = "TTFT" if prefix == "ttft_proc" else "TPOT"
    if percentile == "avg":
        return f"Avg {name} (ms)", f"Avg {name}"
    return f"{percentile.upper()} {name} (ms)", f"{percentile.upper()} {name}"


def _output_suffix(percentile: str) -> str:
    return {"p50": "", "p90": "_p90", "p95": "_p95", "p99": "_p99", "avg": "_avg"}[percentile]


def plot_dataset_dashboard_3panel(
    data: dict,
    dataset: str,
    trace_suffix: str,
    out_path: Path,
    ttft_percentile: str = "p90",
    tpot_percentile: str = "p90",
) -> None:
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(
        1, 5,
        width_ratios=[1, WSPACE_AB, 1, WSPACE_BC, 1],
        wspace=0,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2, 4)]

    ax = axes[0]
    for scheme, label, color, ls, mk, lw, ms in _active_series(data, dataset):
        pts = get_series(data, scheme, dataset, "energy_per_token_mj")
        if not pts:
            continue
        xs = _qps_xs(sorted(pts))
        ys = [pts[q] / 1000.0 for q in sorted(pts)]
        ax.plot(xs, ys, color=color, linestyle=ls, marker=mk, markersize=ms, linewidth=lw, label=label)
    ax.set_ylabel("Energy/Token (J)")
    ax.set_title("(a) Energy per Token By RPS", fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD)
    _style_axis(ax)
    _force_ylim_from_zero(ax)
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)

    for ax_idx, (prefix, percentile) in enumerate(
        [("ttft_proc", ttft_percentile), ("tpot", tpot_percentile)], start=1,
    ):
        ylabel, panel_title = _latency_panel_config(prefix, percentile)
        ax = axes[ax_idx]
        for scheme, label, color, ls, mk, lw, ms in _active_series(data, dataset):
            pts = get_latency_series(data, scheme, dataset, prefix, percentile)
            if not pts:
                continue
            xs = _qps_xs(sorted(pts))
            ys = [pts[q] for q in sorted(pts)]
            ax.plot(xs, ys, color=color, linestyle=ls, marker=mk, markersize=ms, linewidth=lw, label=label)
        ax.set_ylabel(ylabel)
        panel_letter = "b" if prefix == "ttft_proc" else "c"
        ax.set_title(
            f"({panel_letter}) {panel_title} By RPS",
            fontsize=PANEL_TITLE_FONTSIZE,
            pad=PANEL_TITLE_PAD,
        )
        _style_axis(ax)
        _force_ylim_from_zero(ax, integer_ticks=True)

    plt.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.15)
    fig.text(0.5, 0.03, f"{FIG_TITLE_BASE} {trace_suffix}", ha="center", va="bottom", fontsize=FIG_TITLE_FONTSIZE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    print(f"Saved {pdf_path}")


def _plot_latency_panel(ax, data: dict, dataset: str, prefix: str, percentile: str) -> None:
    for scheme, label, color, ls, mk, lw, ms in _active_series(data, dataset):
        pts = get_latency_series(data, scheme, dataset, prefix, percentile)
        if not pts:
            continue
        xs = _qps_xs(sorted(pts))
        ys = [pts[q] for q in sorted(pts)]
        ax.plot(xs, ys, color=color, linestyle=ls, marker=mk, markersize=ms, linewidth=lw, label=label)
    _style_axis(ax)


def plot_combined_latency_4x8(data: dict, out_path: Path) -> None:
    """4x8 grid: rows = dataset x metric (8), cols = P50/P90/P95/P99 (4)."""
    nrows, ncols = len(COMBINED_ROWS), len(COMBINED_PERCENTILES)
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.0, 18.0), sharex=True, squeeze=False)
    for row_idx, (dataset, prefix, row_label) in enumerate(COMBINED_ROWS):
        for col_idx, percentile in enumerate(COMBINED_PERCENTILES):
            ax = axes[row_idx, col_idx]
            _plot_latency_panel(ax, data, dataset, prefix, percentile)
            if row_idx == 0:
                ax.set_title(percentile.upper(), fontsize=PANEL_TITLE_FONTSIZE, pad=PANEL_TITLE_PAD)
            if col_idx == 0:
                ax.set_ylabel(f"{row_label}\n(ms)", fontsize=9)
            if row_idx == nrows - 1:
                ax.set_xlabel("RPS")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        fontsize=8,
        loc="upper center",
        ncol=len(PLOT_SERIES),
        frameon=False,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(f"{FIG_TITLE_BASE} — Latency Percentiles", fontsize=FIG_TITLE_FONTSIZE, y=1.01)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.95, bottom=0.04, hspace=0.35, wspace=0.22)
    for ax in axes.flat:
        _force_ylim_from_zero(ax)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = out_path.with_suffix(".pdf")
    png_path = out_path.with_suffix(".png")
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"Saved {pdf_path}")
    print(f"Saved {png_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None, help="Single dataset JSON (default: all 4 files)")
    parser.add_argument("--dataset", default=None, help="Only plot one dataset key")
    parser.add_argument(
        "--percentile",
        choices=("p50", "p90", "p95", "p99", "avg", "all"),
        default="p90",
    )
    parser.add_argument("--ttft-percentile", choices=("p50", "p90", "p95", "p99", "avg"), default=None)
    parser.add_argument("--tpot-percentile", choices=("p50", "p90", "p95", "p99", "avg"), default=None)
    parser.add_argument("--all-percentiles", action="store_true", help="Also generate p50/p95/avg dashboards")
    parser.add_argument("--combined-4x8", action="store_true", help="Only generate the 4x8 latency grid")
    parser.add_argument("--skip-combined-4x8", action="store_true", help="Skip the 4x8 latency grid")
    args = parser.parse_args()

    if args.input is not None and not args.input.exists():
        raise SystemExit(f"Data file not found: {args.input}")

    data = load_results(args.input)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS
    if args.dataset:
        datasets = [t for t in DATASETS if t[0] == args.dataset]

    if args.all_percentiles or args.percentile == "all":
        percentiles = list(PLOT_PERCENTILES)
    else:
        percentiles = [args.percentile]

    if not args.combined_4x8:
        for ds, _title, trace_suffix in datasets:
            for percentile in percentiles:
                if percentile == "p90":
                    out_name = f"micro_{ds}"
                elif percentile == "p50":
                    out_name = f"micro_{ds}_p50"
                else:
                    out_name = f"micro_{ds}{_output_suffix(percentile)}"
                plot_dataset_dashboard_3panel(
                    data,
                    ds,
                    trace_suffix,
                    CHARTS_DIR / out_name,
                    ttft_percentile=args.ttft_percentile or percentile,
                    tpot_percentile=args.tpot_percentile or percentile,
                )

    if args.combined_4x8 or not args.skip_combined_4x8:
        plot_combined_latency_4x8(data, CHARTS_DIR / COMBINED_GRID_NAME)


if __name__ == "__main__":
    main()
