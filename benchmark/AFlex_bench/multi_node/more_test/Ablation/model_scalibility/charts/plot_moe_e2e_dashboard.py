#!/usr/bin/env python3
"""Plot Mixtral MoE end-to-end dashboards in the Dense E2E style."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE / "results" / "plan_moe_e2e.json"
CHARTS_DIR = HERE / "charts"
QPS_LIST = [2, 4, 8, 16]
QPS_TO_X = {q: i for i, q in enumerate(QPS_LIST)}

PLOT_SERIES = [
    ("native_dp_baseline", "SGLang", "#1f77b4", "-", "o"),
    ("native_dp_tier", "DynamoLLM", "#aec7e8", "--", "s"),
    # ("pd_dp_baseline", "DistServe", "#ff7f0e", "-", "^"),  # needs rerun
    # ("pd_dp_tier", "BiScale", "#ffbb78", "--", "v"),        # needs rerun
    ("pdaf_baseline", "MegaScale", "#2ca02c", "-", "D"),
    ("pdaf_tier", "AFlex", "#d62728", "-", "P"),
]
DATASETS = [
    ("code", "With Code Trace", "moe_end_to_end_code_trace"),
    ("conv", "With Conversation Trace", "moe_end_to_end_conversation_trace"),
]
PLOT_PERCENTILES = ("p50", "p95", "avg")
FIG_TITLE_BASE = "Mixtral-8x7B End-to-end Performance"
FIG_SIZE = (10.8, 3.5)


def load_results(path: Path) -> dict:
    payload = json.loads(path.read_text())
    return payload["results"]


def get_series(data: dict, scheme: str, dataset: str, metric: str) -> dict[int, float]:
    points = {}
    for qps in QPS_LIST:
        row = data.get(scheme, {}).get(f"{dataset}_qps{qps}", {})
        if row.get("status") == "PASS" and row.get(metric) is not None:
            points[qps] = row[metric]
    return points


def get_latency(data: dict, scheme: str, dataset: str, prefix: str, percentile: str):
    if percentile == "p50":
        return get_series(data, scheme, dataset, f"{prefix}_p50_ms")
    if percentile == "avg":
        return get_series(data, scheme, dataset, f"{prefix}_avg_ms")
    p50 = get_series(data, scheme, dataset, f"{prefix}_p50_ms")
    p99 = get_series(data, scheme, dataset, f"{prefix}_p99_ms")
    return {q: p50[q] + (95 - 50) / (99 - 50) * (p99[q] - p50[q])
            for q in p50 if q in p99}


def style_axis(ax):
    ax.set_xticks(range(len(QPS_LIST)))
    ax.set_xticklabels([str(q) for q in QPS_LIST])
    ax.set_xlabel("Request Rate (req/s)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.grid(True, alpha=0.3)


def plot_dashboard(data: dict, dataset: str, trace_title: str, output: Path,
                   percentile: str):
    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 5, width_ratios=[1, 0.22, 1, 0.18, 1], wspace=0)
    axes = [fig.add_subplot(gs[0, i]) for i in (0, 2, 4)]

    panels = [
        ("energy", "(a) Energy per Token By RPS", "Energy/Token (J)"),
        ("ttft_proc", f"(b) {percentile.upper()} TTFT By RPS",
         f"{percentile.upper()} TTFT (ms)"),
        ("tpot", f"(c) {percentile.upper()} TPOT By RPS",
         f"{percentile.upper()} TPOT (ms)"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for scheme, label, color, linestyle, marker in PLOT_SERIES:
            if metric == "energy":
                points = get_series(data, scheme, dataset, "energy_per_token_mj")
                points = {q: value / 1000.0 for q, value in points.items()}
            else:
                points = get_latency(data, scheme, dataset, metric, percentile)
            qps_values = sorted(points)
            if not qps_values:
                continue
            ax.plot([QPS_TO_X[q] for q in qps_values],
                    [points[q] for q in qps_values],
                    color=color, linestyle=linestyle, marker=marker,
                    markersize=5, linewidth=1.8, label=label)
        ax.set_title(title, fontsize=12, pad=5)
        ax.set_ylabel(ylabel)
        style_axis(ax)
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.text(0.5, 0.01, f"{FIG_TITLE_BASE} {trace_title}",
             ha="center", va="bottom", fontsize=14)
    plt.subplots_adjust(left=0.07, right=0.99, top=0.88, bottom=0.19)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--dataset", choices=("code", "conv"))
    parser.add_argument("--percentile", choices=("p50", "p95", "avg", "all"),
                        default="all")
    args = parser.parse_args()
    data = load_results(args.input)
    datasets = [entry for entry in DATASETS if not args.dataset or entry[0] == args.dataset]
    percentiles = PLOT_PERCENTILES if args.percentile == "all" else (args.percentile,)
    suffixes = {"p50": "", "p95": "_p95", "avg": "_avg"}
    for dataset, trace_title, stem in datasets:
        for percentile in percentiles:
            output = CHARTS_DIR / f"{stem}{suffixes[percentile]}"
            plot_dashboard(data, dataset, trace_title, output, percentile)
            print(f"Saved {output.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
