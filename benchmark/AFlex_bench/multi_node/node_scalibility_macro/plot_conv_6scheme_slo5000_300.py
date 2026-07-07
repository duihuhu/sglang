#!/usr/bin/env python3
"""Plot conv 6-scheme dashboard from the SLO=5000/300 test results."""
from __future__ import annotations
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

QPS_LIST = [2, 4, 6, 8, 12, 16]

SCHEME_ORDER = [
    "native_tp_baseline",
    "native_tp_tier",
    "pd_hetero_baseline",
    "pd_hetero_tier",
    "pdaf_baseline",
    "pdaf_tier",
]
SCHEME_LABELS = {
    "native_tp_baseline": "SGLang",
    "native_tp_tier": "DynamoLLM",
    "pd_hetero_baseline": "DistServe",
    "pd_hetero_tier": "BiScale",
    "pdaf_baseline": "MegaScale",
    "pdaf_tier": "AFlex",
}
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]
MARKERS = ["o", "s", "^", "v", "D", "*"]
STYLES = ["-", "--", "-", "--", "-", "--"]


def load_results():
    files = sorted(RESULTS_DIR.glob("conv_6scheme_slo5000_300_*.json"))
    if not files:
        raise FileNotFoundError("No conv result file found")
    latest = files[-1]
    print(f"Loading: {latest}")
    data = json.load(open(latest))
    return data.get("results", {})


def get_series(data, scheme, metric):
    pts = {}
    for q in QPS_LIST:
        key = f"conv_qps{q}"
        m = data.get(scheme, {}).get(key, {})
        if isinstance(m, dict) and m.get("status") == "PASS":
            v = m.get(metric)
            if v is not None:
                pts[q] = v
    return pts


def plot_energy_panel(ax, data):
    for i, scheme in enumerate(SCHEME_ORDER):
        pts = get_series(data, scheme, "energy_per_token_mj")
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] / 1000.0 for q in xs]
        ax.plot(xs, ys, color=COLORS[i], linestyle=STYLES[i],
                marker=MARKERS[i], markersize=7, linewidth=2.2,
                label=SCHEME_LABELS[scheme])
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel("Energy/Token (J)", fontsize=11)
    ax.set_title("(a) Energy per Token", fontsize=12, fontweight="bold")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper right")


def plot_throughput_panel(ax, data):
    for i, scheme in enumerate(SCHEME_ORDER):
        pts = get_series(data, scheme, "throughput_tok_s")
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] for q in xs]
        ax.plot(xs, ys, color=COLORS[i], linestyle=STYLES[i],
                marker=MARKERS[i], markersize=7, linewidth=2.2,
                label=SCHEME_LABELS[scheme])
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel("Throughput (tok/s)", fontsize=11)
    ax.set_title("(b) Throughput", fontsize=12, fontweight="bold")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")


def plot_tpot_panel(ax, data):
    for i, scheme in enumerate(SCHEME_ORDER):
        p50 = get_series(data, scheme, "tpot_p50_ms")
        p99 = get_series(data, scheme, "tpot_p99_ms")
        if not p50:
            continue
        xs = sorted(p50)
        ax.plot(xs, [p50[q] for q in xs], color=COLORS[i], linestyle="-",
                marker=MARKERS[i], markersize=6, linewidth=2.0,
                label=SCHEME_LABELS[scheme])
        if p99:
            ax.plot(xs, [p99.get(q, 0) for q in xs], color=COLORS[i],
                    linestyle=":", linewidth=1.4, alpha=0.7)
    ax.axhline(300, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="SLO=300ms")
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel("TPOT (ms)", fontsize=11)
    ax.set_title("(c) TPOT — solid=P50, dotted=P99", fontsize=12, fontweight="bold")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)


def plot_ttft_panel(ax, data):
    for i, scheme in enumerate(SCHEME_ORDER):
        p50 = get_series(data, scheme, "ttft_proc_p50_ms")
        p99 = get_series(data, scheme, "ttft_proc_p99_ms")
        if not p50:
            continue
        xs = sorted(p50)
        ax.plot(xs, [p50[q] for q in xs], color=COLORS[i], linestyle="-",
                marker=MARKERS[i], markersize=6, linewidth=2.0,
                label=SCHEME_LABELS[scheme])
        if p99:
            ax.plot(xs, [p99.get(q, 0) for q in xs], color=COLORS[i],
                    linestyle=":", linewidth=1.4, alpha=0.7)
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel("TTFT (ms)", fontsize=11)
    ax.set_title("(d) TTFT — solid=P50, dotted=P99", fontsize=12, fontweight="bold")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)


def main():
    data = load_results()

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Conv Dataset — 6-Scheme Comparison | 16-Card Qwen3-32B | SLO: TTFT=5s, TPOT=300ms",
        fontsize=14, fontweight="bold", y=0.98)

    plot_energy_panel(axes[0, 0], data)
    plot_throughput_panel(axes[0, 1], data)
    plot_tpot_panel(axes[1, 0], data)
    plot_ttft_panel(axes[1, 1], data)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = CHARTS_DIR / "conv_6scheme_slo5000_300_dashboard.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
