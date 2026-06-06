#!/usr/bin/env python3
"""Plot 4-GPU v2 comparison chart — uses TTFT Processing Time (excl. queuing)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).parent.parent / "4gpu"
RESULT_DIR = BASE / "results_4gpu_var_v2"
FIG_DIR = BASE / "charts_4gpu_var_v2"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = [
    ("pdaf_dyn/json/pdaf_4g_dyn_var_{wl}_results.json", "PDAF DynM"),
    ("pdaf_dyn_tier/json/pdaf_4g_dyn_tier_var_{wl}_results.json", "PDAF DynM+Tier"),
    ("pd_dp2/json/pd_dp2_disagg_var_{wl}_results.json", "PD DP2"),
    ("native_dp4/json/native_dp4_var_{wl}_results.json", "Native DP4"),
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]

WORKLOADS = ["steady", "varying", "heavy", "overload", "tier1_demo"]
WL_LABELS = ["Steady", "Varying", "Heavy", "Overload", "Tier1-Demo"]


def load_data():
    data = {}
    for tmpl, label in SCHEMES:
        scheme_data = {}
        for wl in WORKLOADS:
            fpath = RESULT_DIR / tmpl.format(wl=wl)
            if fpath.exists():
                with open(fpath) as f:
                    scheme_data[wl] = json.load(f)
        data[label] = scheme_data
    return data


def plot_comparison(data):
    n_wl = len(WORKLOADS)
    n_s = len([label for _, label in SCHEMES if label in data and data[label]])
    x = np.arange(n_wl)
    w = 0.8 / n_s

    metrics = [
        ("throughput_tok_s", "Throughput (tok/s)"),
        ("ttft_proc_avg_ms", "TTFT Processing Time (ms)"),
        ("tpot_avg_ms", "TPOT (ms)"),
        ("total_energy_j", "Total Energy (J)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for ax, (key, title) in zip(axes, metrics):
        scheme_idx = 0
        for (_, label), color in zip(SCHEMES, COLORS):
            if label not in data or not data[label]:
                continue
            vals = []
            for wl in WORKLOADS:
                if wl in data[label]:
                    vals.append(data[label][wl].get(key, 0))
                else:
                    vals.append(0)
            ax.bar(x + (scheme_idx - (n_s - 1) / 2) * w, vals, w,
                   label=label, color=color, edgecolor="white", linewidth=0.5)
            scheme_idx += 1

        ax.set_xticks(x)
        ax.set_xticklabels(WL_LABELS, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("4-GPU Deployment: Variable-Length Workload Comparison\n"
                 "Model: Qwen3-32B | SLO: TTFT<2000ms, TPOT<150ms",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "4gpu_v2_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    data = load_data()
    loaded = {k: len(v) for k, v in data.items() if v}
    print(f"Loaded: {loaded}")
    if not any(data.values()):
        print("No data found!")
        exit(1)
    plot_comparison(data)
    print("Done.")
