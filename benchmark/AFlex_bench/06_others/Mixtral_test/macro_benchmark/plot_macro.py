#!/usr/bin/env python3
"""Plot macro benchmark comparison charts for Mixtral-8x7B 8-GPU."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULT_FILE = Path(__file__).parent / "results" / "macro_mixtral_20260622_204406.json"
OUT_DIR = Path(__file__).parent / "charts"
OUT_DIR.mkdir(exist_ok=True)

with open(RESULT_FILE) as f:
    data = json.load(f)

DEPLOYS = ["native_dp4", "native_dp4_tier", "pd_2p2d", "pd_2p2d_tier",
           "pdaf_tp2", "pdaf_tp2_tier", "pdaf_hetero", "pdaf_hetero_tier"]

DEPLOY_LABELS = {
    "native_dp4": "Native DP4",
    "native_dp4_tier": "Native DP4\n+Tier",
    "pd_2p2d": "PD 2P2D",
    "pd_2p2d_tier": "PD 2P2D\n+Tier",
    "pdaf_tp2": "PDAF TP2",
    "pdaf_tp2_tier": "PDAF TP2\n+Tier",
    "pdaf_hetero": "PDAF Hetero",
    "pdaf_hetero_tier": "PDAF Hetero\n+Tier",
}

WORKLOADS = [
    "workload_azure_code_heavy_real",
    "workload_azure_code_light_real",
    "workload_azure_code_medium_real",
    "workload_azure_conv_heavy_real",
    "workload_azure_conv_light_real",
    "workload_azure_conv_medium",
]

WL_SHORT = {
    "workload_azure_code_heavy_real": "Code Heavy",
    "workload_azure_code_light_real": "Code Light",
    "workload_azure_code_medium_real": "Code Medium",
    "workload_azure_conv_heavy_real": "Conv Heavy",
    "workload_azure_conv_light_real": "Conv Light",
    "workload_azure_conv_medium": "Conv Medium",
}

COLORS = plt.cm.tab10(np.linspace(0, 1, 8))

def get_metric(deploy, workload, metric):
    """Get a metric value, return None if CRASHED/FAIL."""
    if deploy not in data:
        return None
    if workload not in data[deploy]:
        return None
    entry = data[deploy][workload]
    if entry.get("status") not in ("PASS",):
        return None
    return entry.get(metric)


def plot_grouped_bar(metric, ylabel, title, filename, log_scale=False):
    """Grouped bar chart: X=workload, grouped by deploy."""
    fig, ax = plt.subplots(figsize=(16, 7))
    n_wl = len(WORKLOADS)
    n_dep = len(DEPLOYS)
    bar_width = 0.1
    x = np.arange(n_wl)

    for i, dep in enumerate(DEPLOYS):
        vals = []
        for wl in WORKLOADS:
            v = get_metric(dep, wl, metric)
            vals.append(v if v is not None else 0)
        offset = (i - n_dep / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width, label=DEPLOY_LABELS[dep],
                      color=COLORS[i], edgecolor='white', linewidth=0.5)
        for j, v in enumerate(vals):
            if v == 0:
                ax.text(x[j] + offset, 0, '✗', ha='center', va='bottom',
                        fontsize=8, color='red')

    ax.set_xticks(x)
    ax.set_xticklabels([WL_SHORT[w] for w in WORKLOADS], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper left', fontsize=8, ncol=4)
    if log_scale:
        ax.set_yscale('log')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved {filename}")


def plot_energy_savings():
    """Bar chart showing energy savings of Tier vs Baseline."""
    fig, ax = plt.subplots(figsize=(14, 6))
    pairs = [("native_dp4", "native_dp4_tier"),
             ("pd_2p2d", "pd_2p2d_tier")]
    pair_labels = ["Native DP4", "PD 2P2D"]

    n_wl = len(WORKLOADS)
    bar_width = 0.35
    x = np.arange(n_wl)

    for i, (base, tier) in enumerate(pairs):
        savings = []
        for wl in WORKLOADS:
            e_base = get_metric(base, wl, "total_energy_j")
            e_tier = get_metric(tier, wl, "total_energy_j")
            if e_base and e_tier and e_base > 0:
                savings.append((e_base - e_tier) / e_base * 100)
            else:
                savings.append(0)
        offset = (i - 0.5) * bar_width
        ax.bar(x + offset, savings, bar_width, label=f"{pair_labels[i]} Tier savings",
               color=COLORS[i * 2 + 1], edgecolor='white')

    ax.set_xticks(x)
    ax.set_xticklabels([WL_SHORT[w] for w in WORKLOADS], fontsize=10)
    ax.set_ylabel("Energy Savings (%)", fontsize=11)
    ax.set_title("Energy Savings: Tier vs Baseline (Mixtral-8x7B, 8 GPU)", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "macro_energy_savings.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved macro_energy_savings.png")


def plot_latency_vs_energy():
    """Scatter: TTFT_avg vs Energy per token for stable deploys."""
    stable_deploys = ["native_dp4", "native_dp4_tier", "pd_2p2d", "pd_2p2d_tier"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for wi, wl in enumerate(WORKLOADS):
        ax = axes[wi]
        for i, dep in enumerate(stable_deploys):
            ttft = get_metric(dep, wl, "ttft_avg_ms")
            ept = get_metric(dep, wl, "energy_per_token_mj")
            if ttft and ept:
                ax.scatter(ttft, ept, s=120, color=COLORS[DEPLOYS.index(dep)],
                           label=DEPLOY_LABELS[dep].replace('\n', ' '),
                           edgecolors='black', linewidth=0.5, zorder=5)
        ax.set_xlabel("TTFT avg (ms)", fontsize=9)
        ax.set_ylabel("Energy/Token (mJ)", fontsize=9)
        ax.set_title(WL_SHORT[wl], fontsize=10, fontweight='bold')
        ax.grid(alpha=0.3)
        if wi == 0:
            ax.legend(fontsize=7)

    plt.suptitle("Latency vs Energy Efficiency (Mixtral-8x7B, 8 GPU)", fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_DIR / "macro_latency_vs_energy.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved macro_latency_vs_energy.png")


if __name__ == "__main__":
    plot_grouped_bar("ttft_avg_ms", "TTFT avg (ms)", "TTFT Average (Mixtral-8x7B, 8 GPU Macro)", "macro_ttft.png", log_scale=True)
    plot_grouped_bar("tpot_avg_ms", "TPOT avg (ms)", "TPOT Average (Mixtral-8x7B, 8 GPU Macro)", "macro_tpot.png")
    plot_grouped_bar("total_energy_j", "Total Energy (J)", "Total Energy Consumption (Mixtral-8x7B, 8 GPU Macro)", "macro_energy.png")
    plot_grouped_bar("throughput_tok_s", "Throughput (tok/s)", "Throughput (Mixtral-8x7B, 8 GPU Macro)", "macro_throughput.png")
    plot_grouped_bar("slo_violation_rate", "SLO Violation (%)", "SLO Violation Rate (Mixtral-8x7B, 8 GPU Macro)", "macro_slo.png")
    plot_energy_savings()
    plot_latency_vs_energy()
    print("\nAll charts saved to:", OUT_DIR)
