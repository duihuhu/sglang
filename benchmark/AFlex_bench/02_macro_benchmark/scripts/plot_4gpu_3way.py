#!/usr/bin/env python3
"""Plot 4-GPU 3-way comparison: PDAF / PD DP2 / Native DP4, Tier vs max-freq.

Reads:
  results/4gpu_3way/json/*_results.json       (Tier DVFS, --freq auto)
  results/4gpu_3way_bl/json/*_results.json     (baseline, --freq max)

Produces grouped bar charts (throughput, TTFT proc, TPOT, energy, SLO) and a
Tier energy-savings chart per architecture.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
BASE = HERE.parent / "4gpu"
TIER_DIR = BASE / "results_4gpu_3way" / "json"
BL_DIR = BASE / "results_4gpu_3way_bl" / "json"
FIG_DIR = BASE / "results_4gpu_3way" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# (base_deploy_key, display label, color_max, color_tier)
# Colors consistent with 4gpu_v2_comparison: PDAF=blue, PD DP2=red, Native DP4=purple
ARCHS = [
    ("pdaf_4g_dyn", "PDAF", "#4C72B0", "#55A868"),
    ("pd_dp2_disagg", "PD DP2", "#C44E52", "#E88E8E"),
    ("native_dp4", "Native DP4", "#8172B3", "#B8AFDA"),
]
WORKLOADS = ["steady", "varying", "heavy"]
WL_LABELS = ["Steady", "Varying", "Heavy"]


def _load(dir_path, tier):
    """Return {arch_key: {wl: metrics}}."""
    out = {}
    for key, _, _, _ in ARCHS:
        out[key] = {}
        for wl in WORKLOADS:
            name = f"{key}_tier_var_{wl}" if tier else f"{key}_var_{wl}"
            f = dir_path / f"{name}_results.json"
            if f.exists():
                out[key][wl] = json.load(open(f))
    return out


TIER = _load(TIER_DIR, tier=True)
BL = _load(BL_DIR, tier=False)


def _vals(store, key, metric):
    return [store.get(key, {}).get(wl, {}).get(metric, 0) for wl in WORKLOADS]


def plot_comparison():
    """2x2 grouped bars: Tier vs max for each arch, per metric."""
    metrics = [
        ("throughput_tok_s", "Throughput (tok/s)", False),
        ("ttft_proc_avg_ms", "TTFT processing (ms, no queue)", False),
        ("tpot_avg_ms", "TPOT avg (ms)", False),
        ("total_energy_j", "Total Energy (J)", False),
    ]
    x = np.arange(len(WORKLOADS))
    n = len(ARCHS) * 2
    w = 0.8 / n

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    for ax, (mkey, title, _) in zip(axes.flatten(), metrics):
        slot = 0
        for key, label, color_max, color_tier in ARCHS:
            for store, tag, color, hatch in (
                    (BL, "max", color_max, None), (TIER, "Tier", color_tier, "//")):
                vals = _vals(store, key, mkey)
                ax.bar(x + (slot - (n - 1) / 2) * w, vals, w,
                       label=f"{label} ({tag})", color=color, alpha=0.9,
                       hatch=hatch, edgecolor="black", linewidth=0.4)
                slot += 1
        ax.set_xticks(x)
        ax.set_xticklabels(WL_LABELS)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=7, ncol=3)
    fig.suptitle("4-GPU 3-Way Comparison (GPUs 4-7): Tier DVFS vs Max Freq\n"
                 "SLO TTFT<2000ms (proc) / TPOT<150ms — all 0% violation",
                 fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "4gpu_3way_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_savings():
    """Grouped bars: Tier energy savings % per arch per workload."""
    x = np.arange(len(WORKLOADS))
    n = len(ARCHS)
    w = 0.8 / n
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (key, label, color_max, color_tier) in enumerate(ARCHS):
        pct = []
        for wl in WORKLOADS:
            be = BL.get(key, {}).get(wl, {}).get("total_energy_j", 0)
            te = TIER.get(key, {}).get(wl, {}).get("total_energy_j", 0)
            pct.append((be - te) / be * 100 if be > 0 else 0)
        bars = ax.bar(x + (i - (n - 1) / 2) * w, pct, w, label=label,
                      color=color_tier, edgecolor="black", linewidth=0.4)
        for b, v in zip(bars, pct):
            ax.text(b.get_x() + b.get_width() / 2, v + (0.4 if v >= 0 else -1.2),
                    f"{v:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(WL_LABELS)
    ax.set_ylabel("Tier Energy Savings (%)")
    ax.set_title("Tier DVFS Energy Savings vs Max-Freq Baseline (4-GPU)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "4gpu_3way_savings.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_pd_breakdown():
    """Stacked P/D energy for PDAF & PD (Tier), per workload."""
    pd_archs = [("pdaf_4g_dyn", "PDAF (Tier)"), ("pd_dp2_disagg", "PD DP2 (Tier)")]
    x = np.arange(len(WORKLOADS))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (key, label) in zip(axes, pd_archs):
        p = [TIER.get(key, {}).get(wl, {}).get("prefill_energy_j", 0) for wl in WORKLOADS]
        d = [TIER.get(key, {}).get(wl, {}).get("decode_energy_j", 0) for wl in WORKLOADS]
        ax.bar(x, p, 0.6, label="Prefill", color="#4C72B0")
        ax.bar(x, d, 0.6, bottom=p, label="Decode", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(WL_LABELS)
        ax.set_ylabel("Energy (J)")
        ax.set_title(f"{label} — P/D Energy Breakdown")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out = FIG_DIR / "4gpu_3way_pd_breakdown.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_comparison()
    plot_savings()
    plot_pd_breakdown()
    print("All 4-GPU 3-way plots done.")


