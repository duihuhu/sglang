#!/usr/bin/env python3
"""Plot 6-scheme dashboard from fixed_6scheme_complete_final results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE.parent / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

QPS_LIST = [2, 4, 6, 8, 12, 16]

SCHEME_ORDER = [
    "native_tp1_baseline",
    "native_tp1_tier",
    "pd_xnode_tp1_baseline",
    "pd_xnode_tp1_tier",
    "pdaf_tp1_baseline",
    "pdaf_tp1_tier",
]
SCHEME_LABELS = {
    "native_tp1_baseline": "SGLang",
    "native_tp1_tier": "DynamoLLM",
    "pd_xnode_tp1_baseline": "DistServe",
    "pd_xnode_tp1_tier": "BiScale",
    "pdaf_tp1_baseline": "MegaScale",
    "pdaf_tp1_tier": "AFlex",
}
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]
MARKERS = ["o", "s", "^", "v", "D", "*"]
STYLES = ["-", "--", "-", "--", "-", "--"]


def load_results(path: Path | None) -> dict:
    if path is None:
        files = sorted(RESULTS_DIR.glob("fixed_6scheme_complete_final_*.json"))
        if not files:
            raise FileNotFoundError("No fixed_6scheme_complete_final_*.json found")
        path = files[-1]
    print(f"Loading: {path}")
    data = json.loads(path.read_text())
    return data.get("results", data)


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
        ax.plot(
            xs, ys,
            color=COLORS[i], linestyle=STYLES[i],
            marker=MARKERS[i], markersize=7, linewidth=2.2,
            label=SCHEME_LABELS[scheme],
        )
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
        ax.plot(
            xs, ys,
            color=COLORS[i], linestyle=STYLES[i],
            marker=MARKERS[i], markersize=7, linewidth=2.2,
            label=SCHEME_LABELS[scheme],
        )
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
        ax.plot(
            xs, [p50[q] for q in xs],
            color=COLORS[i], linestyle="-",
            marker=MARKERS[i], markersize=6, linewidth=2.0,
            label=SCHEME_LABELS[scheme],
        )
        if p99:
            ax.plot(
                xs, [p99.get(q, 0) for q in xs],
                color=COLORS[i], linestyle=":", linewidth=1.4, alpha=0.7,
            )
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
        ax.plot(
            xs, [p50[q] for q in xs],
            color=COLORS[i], linestyle="-",
            marker=MARKERS[i], markersize=6, linewidth=2.0,
            label=SCHEME_LABELS[scheme],
        )
        if p99:
            ax.plot(
                xs, [p99.get(q, 0) for q in xs],
                color=COLORS[i], linestyle=":", linewidth=1.4, alpha=0.7,
            )
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel("TTFT (ms)", fontsize=11)
    ax.set_title("(d) TTFT — solid=P50, dotted=P99", fontsize=12, fontweight="bold")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left", ncol=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=CHARTS_DIR / "fixed_6scheme_slo5000_300_dashboard.png",
    )
    args = parser.parse_args()

    data = load_results(args.input)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Conv Dataset — 6-Scheme (node3+node4) | 16-Card Qwen3-32B | SLO: TTFT=5s, TPOT=300ms\n"
        "SGLang/DynamoLLM: TP1×16 | DistServe/BiScale: P=8×TP1 + D=8×TP1 (xnode) | "
        "MegaScale/AFlex: PDAF TP1×4 | DVFS: min-energy",
        fontsize=12, fontweight="bold", y=0.98,
    )

    plot_energy_panel(axes[0, 0], data)
    plot_throughput_panel(axes[0, 1], data)
    plot_tpot_panel(axes[1, 0], data)
    plot_ttft_panel(axes[1, 1], data)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
