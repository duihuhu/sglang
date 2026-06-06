#!/usr/bin/env python3
"""Plot 8-GPU v2 comparison chart (4 schemes x 5 workloads) — optimized layout."""
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_DIR = Path(__file__).parent.parent.parent / "results" / "8gpu_v2"
JSON_DIR = RESULT_DIR / "json"
FIG_DIR = RESULT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = [
    ("pdaf_8g_dyn", "PDAF DynM"),
    ("pdaf_8g_dyn_tier", "PDAF DynM+Tier"),
    ("pd_dp4", "PD DP4"),
    ("native_dp8", "Native DP8"),
]
COLORS = ["#4C72B0", "#55A868", "#C44E52", "#8172B3"]
HATCHES = ["", "", "", ""]

WORKLOADS = ["steady", "varying", "heavy", "overload", "tier1_demo"]
WL_LABELS = ["Steady", "Varying", "Heavy", "Overload", "Tier1-Demo"]


def load_data():
    """Load per-scheme per-workload results from individual JSON files."""
    data = {}
    for prefix, label in SCHEMES:
        scheme_data = {}
        for wl in WORKLOADS:
            fname = JSON_DIR / f"{prefix}_var_{wl}_results.json"
            if fname.exists():
                with open(fname) as f:
                    scheme_data[wl] = json.load(f)
        data[label] = scheme_data
    return data


def plot_comparison(data):
    n_wl = len(WORKLOADS)
    n_s = len([s for _, s in SCHEMES if s in data and data[s]])
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
            bars = ax.bar(x + (scheme_idx - (n_s - 1) / 2) * w, vals, w,
                          label=label, color=color, edgecolor="white", linewidth=0.5)
            scheme_idx += 1

        ax.set_xticks(x)
        ax.set_xticklabels(WL_LABELS, fontsize=10)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=9, loc="best")

    fig.suptitle("8-GPU Deployment: Variable-Length Workload Comparison\n"
                 "Model: Qwen3-32B | SLO: TTFT<5000ms, TPOT<300ms",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = FIG_DIR / "8gpu_v2_comparison.png"
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
