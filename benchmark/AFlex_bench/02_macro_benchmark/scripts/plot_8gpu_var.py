#!/usr/bin/env python3
"""Plot 8-GPU variable-length benchmark results (4 schemes x 5 workloads)."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent / "results" / "8gpu_var"
FIG_DIR = BASE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = [
    ("pdaf_dyn", "PDAF DynM (max)"),
    ("pdaf_dyn_tier", "PDAF DynM+DVFS"),
    ("pd_dp4_slo2000", "PD DP4 (max)"),
    ("native_tp8", "Native TP8 (max)"),
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

WORKLOADS = ["heavy", "steady", "varying", "overload", "tier1_demo"]
WL_LABELS = ["Heavy", "Steady", "Varying", "Overload", "Tier1-Demo"]


def load_all():
    data = {}
    for folder, label in SCHEMES:
        summary_path = BASE / folder / "json" / "8gpu_summary.json"
        if not summary_path.exists():
            print(f"Missing: {summary_path}")
            continue
        with open(summary_path) as f:
            raw = json.load(f)
        deploy_key = list(raw.keys())[0]
        entries = raw[deploy_key]
        scheme_data = {}
        for wl_group_key, qps_dict in entries.items():
            wl_name = wl_group_key.replace("var_", "")
            inner_key = list(qps_dict.keys())[0]
            scheme_data[wl_name] = qps_dict[inner_key]
        data[folder] = scheme_data
    return data


def plot_comparison(data):
    n_wl = len(WORKLOADS)
    n_s = len(SCHEMES)
    x = np.arange(n_wl)
    w = 0.8 / n_s

    metrics = [
        ("throughput_tok_s", "Throughput (tok/s)", False),
        ("ttft_avg_ms", "TTFT avg (ms)", True),
        ("tpot_avg_ms", "TPOT avg (ms)", False),
        ("total_energy_j", "Total Energy (J)", False),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax, (key, title, use_log) in zip(axes, metrics):
        for i, (folder, label) in enumerate(SCHEMES):
            if folder not in data:
                continue
            vals = []
            for wl in WORKLOADS:
                if wl in data[folder]:
                    vals.append(data[folder][wl].get(key, 0))
                else:
                    vals.append(0)
            ax.bar(x + (i - (n_s - 1) / 2) * w, vals, w,
                   label=label, color=COLORS[i])
        ax.set_xticks(x)
        ax.set_xticklabels(WL_LABELS, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        if use_log:
            ax.set_yscale("log")

    fig.suptitle("8-GPU Variable-Length Workload Comparison\n"
                 "SLO: TTFT<2000ms, TPOT<150ms", fontsize=13)
    fig.tight_layout()
    out = FIG_DIR / "8gpu_var_comparison.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_energy_breakdown(data):
    """Energy bar with P/D breakdown for PDAF schemes."""
    n_wl = len(WORKLOADS)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (folder, label) in zip(axes, [SCHEMES[0], SCHEMES[1]]):
        if folder not in data:
            continue
        p_vals, d_vals = [], []
        for wl in WORKLOADS:
            if wl in data[folder]:
                p_vals.append(data[folder][wl].get("prefill_energy_j", 0))
                d_vals.append(data[folder][wl].get("decode_energy_j", 0))
            else:
                p_vals.append(0)
                d_vals.append(0)
        x = np.arange(n_wl)
        ax.bar(x, p_vals, 0.6, label="Prefill Energy", color="#4C72B0")
        ax.bar(x, d_vals, 0.6, bottom=p_vals, label="Decode Energy", color="#DD8452")
        ax.set_xticks(x)
        ax.set_xticklabels(WL_LABELS, fontsize=9)
        ax.set_ylabel("Energy (J)")
        ax.set_title(f"{label} — P/D Energy Breakdown")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    out = FIG_DIR / "8gpu_var_energy_breakdown.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_slo_violations(data):
    """SLO violation rates across schemes and workloads."""
    n_wl = len(WORKLOADS)
    n_s = len(SCHEMES)
    x = np.arange(n_wl)
    w = 0.8 / n_s

    fig, ax = plt.subplots(figsize=(12, 5))
    for i, (folder, label) in enumerate(SCHEMES):
        if folder not in data:
            continue
        vals = []
        for wl in WORKLOADS:
            if wl in data[folder]:
                vals.append(data[folder][wl].get("slo_violation_rate", 0))
            else:
                vals.append(0)
        ax.bar(x + (i - (n_s - 1) / 2) * w, vals, w,
               label=label, color=COLORS[i])

    ax.axhline(y=0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(WL_LABELS, fontsize=10)
    ax.set_ylabel("SLO Violation Rate (%)")
    ax.set_title("SLO Violation Rate (TTFT<2000ms, TPOT<150ms)\n8-GPU Variable-Length")
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 100)

    fig.tight_layout()
    out = FIG_DIR / "8gpu_var_slo_violations.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_energy_savings(data):
    """Energy savings of DVFS vs max-freq baseline."""
    if "pdaf_dyn" not in data or "pdaf_dyn_tier" not in data:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    savings = []
    labels = []
    for wl, wl_label in zip(WORKLOADS, WL_LABELS):
        if wl in data["pdaf_dyn"] and wl in data["pdaf_dyn_tier"]:
            base = data["pdaf_dyn"][wl]["total_energy_j"]
            dvfs = data["pdaf_dyn_tier"][wl]["total_energy_j"]
            pct = (1 - dvfs / base) * 100 if base > 0 else 0
            savings.append(pct)
            labels.append(wl_label)

    colors = ["#55A868" if s > 0 else "#C44E52" for s in savings]
    ax.bar(range(len(savings)), savings, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(savings)))
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Energy Savings (%)")
    ax.set_title("DVFS V2 Energy Savings vs PDAF DynM (max freq)\n8-GPU Variable-Length")
    ax.axhline(y=0, color='black', linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    for i, v in enumerate(savings):
        ax.text(i, v + 0.5, f"{v:.1f}%", ha='center', fontsize=9, fontweight='bold')

    fig.tight_layout()
    out = FIG_DIR / "8gpu_var_dvfs_savings.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    data = load_all()
    if not data:
        print("No data loaded!")
        exit(1)
    print(f"Loaded {len(data)} schemes")
    plot_comparison(data)
    plot_energy_breakdown(data)
    plot_slo_violations(data)
    plot_energy_savings(data)
    print("All plots done.")
