#!/usr/bin/env python3
"""Plot 8-GPU benchmark results.

Generates:
  - 8gpu_scaling_*.png : Per-workload QPS scaling (throughput, TPOT, TTFT, J/tok)
  - 8gpu_collapse.png  : SLO collapse point comparison across topologies
  - 8gpu_energy_breakdown.png : Prefill vs Decode energy at key QPS points
  - 8gpu_summary_table.png : Heatmap of max sustainable QPS per topology×workload
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "results" / "8gpu" / "json"
FIG_DIR = HERE / "results" / "8gpu" / "figures"

DEPLOYS = ["pd_p4d4", "pdaf_8g_m1", "pdaf_8g_m2", "pdaf_8g_m1_tier", "pdaf_8g_m2_tier"]
LABELS = {
    "pd_p4d4": "Pure PD (TP4+TP4)",
    "pdaf_8g_m1": "AF M=1",
    "pdaf_8g_m2": "AF M=2",
    "pdaf_8g_m1_tier": "AF M=1+DVFS",
    "pdaf_8g_m2_tier": "AF M=2+DVFS",
}
COLORS = {
    "pd_p4d4": "#4C72B0",
    "pdaf_8g_m1": "#DD8452",
    "pdaf_8g_m2": "#DA8BC3",
    "pdaf_8g_m1_tier": "#55A868",
    "pdaf_8g_m2_tier": "#8172B3",
}
MARKERS = {
    "pd_p4d4": "o",
    "pdaf_8g_m1": "s",
    "pdaf_8g_m2": "D",
    "pdaf_8g_m1_tier": "^",
    "pdaf_8g_m2_tier": "v",
}
WORKLOADS = ["il128_ol1024", "il512_ol256", "il2048_ol64", "il4096_ol64"]
WL_TITLES = {
    "il128_ol1024": "il128/ol1024 (decode-heavy)",
    "il512_ol256": "il512/ol256 (balanced)",
    "il2048_ol64": "il2048/ol64 (prefill-heavy)",
    "il4096_ol64": "il4096/ol64 (extreme prefill)",
}


def load():
    data = {}
    for f in glob.glob(str(JSON_DIR / "*_results.json")):
        d = json.load(open(f))
        dep = d.get("deploy")
        if not dep:
            continue
        key = f"il{d.get('il')}_ol{d.get('ol')}"
        data.setdefault(key, {}).setdefault(dep, {})[float(d["qps"])] = d
    return data


def plot_scaling(data):
    """Per-workload 4-panel QPS scaling."""
    for wl in WORKLOADS:
        grp = data.get(wl, {})
        if not grp:
            continue
        panels = [("throughput_tok_s", "Throughput (tok/s)", False),
                  ("tpot_avg_ms", "TPOT avg (ms)", False),
                  ("ttft_avg_ms", "TTFT avg (ms, log)", True),
                  ("energy_per_token_mj", "Energy/token (mJ)", False)]
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        for ax, (key, title, logy) in zip(axes.flat, panels):
            for dep in DEPLOYS:
                if dep not in grp:
                    continue
                qps = sorted(grp[dep])
                ys = [grp[dep][q].get(key, 0) for q in qps]
                ax.plot(qps, ys, marker=MARKERS[dep], color=COLORS[dep],
                        label=LABELS[dep], linewidth=1.5, markersize=5)
            ax.set_title(title, fontsize=11)
            ax.set_xlabel("QPS")
            ax.grid(True, alpha=0.3)
            if logy:
                ax.set_yscale("log")
        axes.flat[0].legend(fontsize=8, loc="upper left")
        fig.suptitle(f"8-GPU Scaling: {WL_TITLES.get(wl, wl)}", fontsize=13)
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"8gpu_scaling_{wl}.png", dpi=120)
        plt.close(fig)
        print(f"wrote 8gpu_scaling_{wl}.png")


def plot_collapse(data):
    """Bar chart: max sustainable QPS (SLO viol <50%) per topology × workload."""
    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(len(WORKLOADS))
    n = len(DEPLOYS)
    w = 0.8 / n

    for i, dep in enumerate(DEPLOYS):
        max_qps = []
        for wl in WORKLOADS:
            grp = data.get(wl, {}).get(dep, {})
            sustainable = [q for q, d in grp.items()
                          if d.get("slo_violation_rate", 0) <= 50]
            max_qps.append(max(sustainable) if sustainable else 0)
        ax.bar(x + (i - (n-1)/2) * w, max_qps, w,
               label=LABELS[dep], color=COLORS[dep])

    ax.set_xticks(x)
    ax.set_xticklabels([WL_TITLES.get(wl, wl) for wl in WORKLOADS], fontsize=9)
    ax.set_ylabel("Max sustainable QPS (SLO viol < 50%)")
    ax.set_title("8-GPU: Throughput ceiling per topology × workload", fontsize=13)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "8gpu_collapse.png", dpi=120)
    plt.close(fig)
    print("wrote 8gpu_collapse.png")


def plot_energy_breakdown(data):
    """Stacked bar: Prefill vs Decode energy at a representative QPS per workload."""
    target_qps = {"il128_ol1024": 8.0, "il512_ol256": 6.0,
                  "il2048_ol64": 4.0, "il4096_ol64": 3.0}
    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(18, 5), sharey=False)
    if len(WORKLOADS) == 1:
        axes = [axes]

    for ax, wl in zip(axes, WORKLOADS):
        q = target_qps.get(wl, 4.0)
        grp = data.get(wl, {})
        deps_present = [d for d in DEPLOYS if d in grp and q in grp[d]]
        if not deps_present:
            ax.set_title(f"{wl} (no data @ q{q:.0f})")
            continue
        x = np.arange(len(deps_present))
        p_energy = [grp[d][q].get("prefill_energy_j", 0) for d in deps_present]
        d_energy = [grp[d][q].get("decode_energy_j", 0) for d in deps_present]
        ax.bar(x, p_energy, 0.6, label="Prefill", color="#4C72B0", alpha=0.8)
        ax.bar(x, d_energy, 0.6, bottom=p_energy, label="Decode", color="#C44E52", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([LABELS[d].replace(" ", "\n") for d in deps_present],
                           fontsize=8)
        ax.set_title(f"{WL_TITLES.get(wl, wl)}\n@ QPS={q:.0f}", fontsize=10)
        ax.set_ylabel("Energy (J)")
        ax.grid(True, axis="y", alpha=0.3)
        if ax == axes[0]:
            ax.legend(fontsize=8)

    fig.suptitle("8-GPU Energy Breakdown: Prefill vs Decode", fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "8gpu_energy_breakdown.png", dpi=120)
    plt.close(fig)
    print("wrote 8gpu_energy_breakdown.png")


def plot_efficiency(data):
    """J/tok vs throughput scatter for all runs — Pareto frontier visualization."""
    fig, ax = plt.subplots(figsize=(12, 7))
    for dep in DEPLOYS:
        xs, ys = [], []
        for wl in WORKLOADS:
            grp = data.get(wl, {}).get(dep, {})
            for q, d in grp.items():
                if d.get("slo_violation_rate", 0) <= 50:
                    xs.append(d["throughput_tok_s"])
                    ys.append(d.get("energy_per_token_mj", 0))
        if xs:
            ax.scatter(xs, ys, marker=MARKERS[dep], color=COLORS[dep],
                      label=LABELS[dep], alpha=0.7, s=40)
    ax.set_xlabel("Throughput (tok/s)")
    ax.set_ylabel("Energy/token (mJ)")
    ax.set_title("8-GPU: Energy efficiency vs Throughput (SLO-compliant runs only)",
                 fontsize=13)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "8gpu_efficiency.png", dpi=120)
    plt.close(fig)
    print("wrote 8gpu_efficiency.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    if not data:
        print("No data found in", JSON_DIR)
        return
    plot_scaling(data)
    plot_collapse(data)
    plot_energy_breakdown(data)
    plot_efficiency(data)
    print("All 8-GPU figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
