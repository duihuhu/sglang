#!/usr/bin/env python3
"""Plot MoE benchmark comparison (6 schemes × 3 workloads)."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results" / "json"
FIG_DIR = Path(__file__).resolve().parent.parent / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = [
    ("native_dp8", "Native DP8"),
    ("native_dp8_tier", "Native DP8\n+Tier"),
    ("pd_dp4", "PD DP4"),
    ("pd_dp4_tier", "PD DP4\n+Tier"),
    ("pdaf_tp2", "PDAF DynM"),
    ("pdaf_tp2_tier", "PDAF DynM\n+Tier"),
]

WORKLOADS = [
    ("workload_azure_code_light_real", "Light"),
    ("workload_azure_code_medium_real", "Medium"),
    ("workload_azure_code_heavy_real", "Heavy"),
]

COLORS = {
    "native_dp8": "#8172B3",
    "native_dp8_tier": "#B8AFDA",
    "pd_dp4": "#C44E52",
    "pd_dp4_tier": "#E88E8E",
    "pdaf_tp2": "#4C72B0",
    "pdaf_tp2_tier": "#55A868",
}

HATCHES = {
    "native_dp8": "",
    "native_dp8_tier": "//",
    "pd_dp4": "",
    "pd_dp4_tier": "//",
    "pdaf_tp2": "",
    "pdaf_tp2_tier": "//",
}


def load_data():
    data = {}
    for scheme_key, _ in SCHEMES:
        data[scheme_key] = {}
        for wl_key, _ in WORKLOADS:
            fpath = RESULTS_DIR / f"{scheme_key}_{wl_key}.json"
            if fpath.exists():
                with open(fpath) as f:
                    data[scheme_key][wl_key] = json.load(f)
    return data


def plot_comparison(data):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("MoE Benchmark: Qwen3-30B-A3B (8×A800 GPU)", fontsize=14, fontweight="bold")

    metrics = [
        ("throughput_tok_s", "Throughput (tok/s)", axes[0, 0]),
        ("ttft_proc_avg_ms", "TTFT Processing (ms)", axes[0, 1]),
        ("tpot_avg_ms", "TPOT (ms)", axes[1, 0]),
        ("total_energy_j", "Total Energy (J)", axes[1, 1]),
    ]

    n_schemes = len(SCHEMES)
    n_workloads = len(WORKLOADS)
    bar_width = 0.12
    group_gap = 0.15

    for metric_key, metric_label, ax in metrics:
        for i, (scheme_key, scheme_label) in enumerate(SCHEMES):
            vals = []
            for wl_key, _ in WORKLOADS:
                d = data.get(scheme_key, {}).get(wl_key, {})
                vals.append(d.get(metric_key, 0))

            x = np.arange(n_workloads) * (n_schemes * bar_width + group_gap)
            offset = (i - n_schemes / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, vals, bar_width,
                          label=scheme_label,
                          color=COLORS[scheme_key],
                          hatch=HATCHES[scheme_key],
                          edgecolor="white", linewidth=0.5)

            for bar, v in zip(bars, vals):
                if v > 0:
                    fontsize = 6.5
                    if metric_key == "total_energy_j":
                        txt = f"{v/1000:.0f}k"
                    elif v >= 1000:
                        txt = f"{v:.0f}"
                    else:
                        txt = f"{v:.1f}"
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                            txt, ha="center", va="bottom", fontsize=fontsize)

        ax.set_ylabel(metric_label)
        ax.set_xticks(np.arange(n_workloads) * (n_schemes * bar_width + group_gap))
        ax.set_xticklabels([wl_label for _, wl_label in WORKLOADS])
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    axes[0, 0].legend(loc="upper left", fontsize=8, ncol=2)

    plt.tight_layout()
    out_path = FIG_DIR / "moe_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")

    # Also plot SLO violation separately
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 5))
    fig2.suptitle("MoE Benchmark: SLO Violation Rate (%)", fontsize=13, fontweight="bold")

    for i, (scheme_key, scheme_label) in enumerate(SCHEMES):
        vals = []
        for wl_key, _ in WORKLOADS:
            d = data.get(scheme_key, {}).get(wl_key, {})
            vals.append(d.get("slo_violation_rate", 0))

        x = np.arange(n_workloads) * (n_schemes * bar_width + group_gap)
        offset = (i - n_schemes / 2 + 0.5) * bar_width
        bars = ax2.bar(x + offset, vals, bar_width,
                       label=scheme_label,
                       color=COLORS[scheme_key],
                       hatch=HATCHES[scheme_key],
                       edgecolor="white", linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f"{v:.1f}", ha="center", va="bottom", fontsize=7)

    ax2.set_ylabel("SLO Violation Rate (%)")
    ax2.set_xticks(np.arange(n_workloads) * (n_schemes * bar_width + group_gap))
    ax2.set_xticklabels([wl_label for _, wl_label in WORKLOADS])
    ax2.legend(loc="upper left", fontsize=8, ncol=3)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_axisbelow(True)

    plt.tight_layout()
    out_path2 = FIG_DIR / "moe_slo_violations.png"
    plt.savefig(out_path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path2}")


if __name__ == "__main__":
    data = load_data()
    plot_comparison(data)
