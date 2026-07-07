#!/usr/bin/env python3
"""Generate 16-card macro benchmark charts with sparse QPS ticks."""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"

QPS_LIST = [2, 4, 6, 8, 12, 16]

SCHEME_ORDER = [
    "native_tp2_baseline",
    "native_tp2_tier",
    "pd_dp_baseline",
    "pd_dp_tier",
    "pdaf_baseline",
    "pdaf_tier",
]
SCHEME_LABELS = {
    "native_tp2_baseline": "SGLang",
    "native_tp2_tier": "DynamoLLM",
    "pd_dp_baseline": "DistServe",
    "pd_dp_tier": "BiScale",
    "pdaf_baseline": "MegaScale",
    "pdaf_tier": "AFlex",
}
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]
MARKERS = ["o", "s", "^", "v", "D", "P"]
STYLES = ["-", "--", "-", "--", "-", "--"]

MACRO_DATASETS = ["conv", "code"]
MACRO_LABELS = {"conv": "Conversation", "code": "Code"}


def load_results(prefix):
    merged = {}
    for f in sorted(RESULTS_DIR.glob(f"{prefix}*.json")):
        data = json.load(open(f))
        for scheme, entries in data.get("results", {}).items():
            sname = scheme.replace("native_tp_", "native_tp2_")
            merged.setdefault(sname, {}).update(entries)
    return merged


def extract(data, scenario, metric, qps_list):
    qv, mv = [], []
    for q in qps_list:
        key = f"{scenario}_qps{q}"
        entry = data.get(key, {})
        if isinstance(entry, dict) and entry.get("status") == "PASS":
            val = entry.get(metric)
            if val is not None:
                qv.append(q)
                mv.append(val)
    return qv, mv


def plot_metric_grid(all_data, metric_key, ylabel, title, outpath, scale=1.0):
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.995)

    for idx, dataset in enumerate(MACRO_DATASETS):
        ax = axes[idx]
        for i, scheme in enumerate(SCHEME_ORDER):
            if scheme not in all_data:
                continue
            qv, mv = extract(all_data[scheme], dataset, metric_key, QPS_LIST)
            if mv:
                ax.plot(
                    qv,
                    [v / scale for v in mv],
                    color=COLORS[i],
                    linestyle=STYLES[i],
                    marker=MARKERS[i],
                    markersize=6,
                    linewidth=2.0,
                    label=SCHEME_LABELS[scheme],
                )
        ax.set_xlabel("QPS", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(MACRO_LABELS[dataset], fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xticks(QPS_LIST)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {outpath}")


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    macro16 = load_results("scal_16card_")

    plot_metric_grid(
        macro16,
        "energy_per_token_mj",
        "Energy/Token (J)",
        "16-Card Macro: Energy per Token | Qwen3-32B 2-Node",
        CHARTS_DIR / "16card_energy_per_token.png",
        scale=1000.0,
    )

    for pname, metric_key, label in [
        ("p50", "ttft_proc_p50_ms", "TTFT P50 (ms)"),
        ("avg", "ttft_proc_avg_ms", "TTFT Avg (ms)"),
        ("p99", "ttft_proc_p99_ms", "TTFT P99 (ms)"),
    ]:
        plot_metric_grid(
            macro16,
            metric_key,
            label,
            f"16-Card Macro: {label} | Qwen3-32B 2-Node",
            CHARTS_DIR / f"16card_ttft_{pname}.png",
        )

    for pname, metric_key, label in [
        ("p50", "tpot_p50_ms", "TPOT P50 (ms)"),
        ("avg", "tpot_avg_ms", "TPOT Avg (ms)"),
        ("p99", "tpot_p99_ms", "TPOT P99 (ms)"),
    ]:
        plot_metric_grid(
            macro16,
            metric_key,
            label,
            f"16-Card Macro: {label} | Qwen3-32B 2-Node",
            CHARTS_DIR / f"16card_tpot_{pname}.png",
        )


if __name__ == "__main__":
    main()
