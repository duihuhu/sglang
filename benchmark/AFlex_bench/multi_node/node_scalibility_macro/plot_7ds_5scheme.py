#!/usr/bin/env python3
"""Plot 5-scheme comparison dashboards (excluding AFlex) from full_7ds results.

Generates dashboard for each dataset: Energy/Token, TTFT, TPOT, SLO%.
Uses the latest full_7ds_*.json result file.
"""
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
]
SCHEME_LABELS = {
    "native_tp_baseline": "SGLang",
    "native_tp_tier": "DynamoLLM",
    "pd_hetero_baseline": "DistServe",
    "pd_hetero_tier": "BiScale",
    "pdaf_baseline": "MegaScale",
}
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c"]
MARKERS = ["o", "s", "^", "v", "D"]
STYLES = ["-", "--", "-", "--", "-"]

DATASETS = [
    ("code", "Code (variable-len)"),
    ("conv", "Conversation (variable-len)"),
    ("qa_lpld", "QA (in128, out64)"),
    ("chatbot_lphd", "Chatbot (in128, out1024)"),
    ("balanced_mpmd", "Balanced (in512, out256)"),
    ("rag_hpld", "RAG (in4096, out64)"),
    ("summary_hphd", "Summary (in4096, out1024)"),
]


def load_results() -> dict:
    """Load the latest full_7ds result file."""
    files = sorted(RESULTS_DIR.glob("full_7ds_*.json"))
    if not files:
        raise FileNotFoundError("No full_7ds_*.json found in results/")
    latest = files[-1]
    print(f"Loading: {latest}")
    data = json.load(open(latest))
    return data.get("results", {})


def get_metric(data: dict, scheme: str, dataset: str, qps: int, metric: str):
    key = f"{dataset}_qps{qps}"
    m = data.get(scheme, {}).get(key, {})
    if isinstance(m, dict) and m.get("status") == "PASS":
        return m.get(metric)
    return None


def get_series(data: dict, scheme: str, dataset: str, metric: str):
    pts = {}
    for q in QPS_LIST:
        v = get_metric(data, scheme, dataset, q, metric)
        if v is not None:
            pts[q] = v
    return pts


def plot_energy_panel(ax, data, dataset):
    for i, scheme in enumerate(SCHEME_ORDER):
        if scheme not in data:
            continue
        pts = get_series(data, scheme, dataset, "energy_per_token_mj")
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] / 1000.0 for q in xs]
        ax.plot(xs, ys, color=COLORS[i], linestyle=STYLES[i],
                marker=MARKERS[i], markersize=7, linewidth=2.0,
                label=SCHEME_LABELS[scheme])
    ax.set_xlabel("QPS")
    ax.set_ylabel("Energy/Token (J)")
    ax.set_title("(a) Energy per Token")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_ttft_panel(ax, data, dataset):
    for i, scheme in enumerate(SCHEME_ORDER):
        if scheme not in data:
            continue
        p50 = get_series(data, scheme, dataset, "ttft_proc_p50_ms")
        p99 = get_series(data, scheme, dataset, "ttft_proc_p99_ms")
        if not p50:
            continue
        xs = sorted(p50)
        ax.plot(xs, [p50[q] for q in xs], color=COLORS[i], linestyle="-",
                marker=MARKERS[i], markersize=6, linewidth=2.0,
                label=f"{SCHEME_LABELS[scheme]} P50")
        if p99:
            ax.plot(xs, [p99.get(q, 0) for q in xs], color=COLORS[i],
                    linestyle=":", linewidth=1.5, alpha=0.7)
    ax.set_xlabel("QPS")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("(b) TTFT — solid=P50, dotted=P99")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)


def plot_tpot_panel(ax, data, dataset):
    for i, scheme in enumerate(SCHEME_ORDER):
        if scheme not in data:
            continue
        p50 = get_series(data, scheme, dataset, "tpot_p50_ms")
        p99 = get_series(data, scheme, dataset, "tpot_p99_ms")
        if not p50:
            continue
        xs = sorted(p50)
        ax.plot(xs, [p50[q] for q in xs], color=COLORS[i], linestyle="-",
                marker=MARKERS[i], markersize=6, linewidth=2.0,
                label=f"{SCHEME_LABELS[scheme]} P50")
        if p99:
            ax.plot(xs, [p99.get(q, 0) for q in xs], color=COLORS[i],
                    linestyle=":", linewidth=1.5, alpha=0.7)
    ax.axhline(100, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="SLO=100ms")
    ax.set_xlabel("QPS")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("(c) TPOT — solid=P50, dotted=P99")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper left", ncol=2)


def plot_slo_panel(ax, data, dataset):
    for i, scheme in enumerate(SCHEME_ORDER):
        if scheme not in data:
            continue
        pts = get_series(data, scheme, dataset, "slo_violation_rate")
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] for q in xs]
        ax.plot(xs, ys, color=COLORS[i], linestyle=STYLES[i],
                marker=MARKERS[i], markersize=7, linewidth=2.0,
                label=SCHEME_LABELS[scheme])
    ax.set_xlabel("QPS")
    ax.set_ylabel("SLO Violation (%)")
    ax.set_title("(d) SLO Violation Rate")
    ax.set_xticks(QPS_LIST)
    ax.set_ylim(-2, 105)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")


def plot_dashboard(data, dataset, title, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"16-Card Dashboard — {title} | Qwen3-32B 2-Node (5 schemes, excl. AFlex)",
        fontsize=14, fontweight="bold", y=0.98)

    plot_energy_panel(axes[0, 0], data, dataset)
    plot_ttft_panel(axes[0, 1], data, dataset)
    plot_tpot_panel(axes[1, 0], data, dataset)
    plot_slo_panel(axes[1, 1], data, dataset)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    data = load_results()
    for dataset, title in DATASETS:
        out_path = CHARTS_DIR / f"7ds_{dataset}_dashboard_5scheme.png"
        plot_dashboard(data, dataset, title, out_path)


if __name__ == "__main__":
    main()
