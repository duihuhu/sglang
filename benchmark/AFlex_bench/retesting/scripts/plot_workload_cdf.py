#!/usr/bin/env python3
"""Plot CDF charts for micro-benchmark (varying) and macro-benchmark (Azure) workloads."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
CHARTS_DIR = BASE / "workloads" / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

WL_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads")


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def plot_cdf(ax, data, label, color, linestyle="-"):
    sorted_d = np.sort(data)
    cdf = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, cdf, label=label, color=color, linestyle=linestyle, linewidth=1.5)


plt.rcParams.update({"font.size": 9, "figure.dpi": 150})

# ============================================================
# Figure 1: Micro-benchmark varying workload
# ============================================================
varying = load_jsonl(WL_DIR / "workload_varying.jsonl")
ils = [r["input_len"] for r in varying]
ols = [r["output_len"] for r in varying]
arrivals = [r["arrival_time_s"] for r in varying]
# Compute inter-arrival QPS (instantaneous)
iat = [arrivals[i] - arrivals[i-1] for i in range(1, len(arrivals))]
inst_qps = [1.0 / t if t > 0 else 0 for t in iat]

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle("Micro-Benchmark: Varying Workload Distribution (n=280)",
             fontsize=12, fontweight="bold")

plot_cdf(axes[0], ils, "Input Length", "#2196F3")
axes[0].set_xlabel("Input Length (tokens)")
axes[0].set_ylabel("CDF")
axes[0].set_title("Input Length CDF")
axes[0].grid(alpha=0.3)
axes[0].axvline(np.median(ils), color="red", linestyle="--", alpha=0.5,
                label=f"median={int(np.median(ils))}")
axes[0].legend(fontsize=8)

plot_cdf(axes[1], ols, "Output Length", "#4CAF50")
axes[1].set_xlabel("Output Length (tokens)")
axes[1].set_title("Output Length CDF")
axes[1].grid(alpha=0.3)
axes[1].axvline(np.median(ols), color="red", linestyle="--", alpha=0.5,
                label=f"median={int(np.median(ols))}")
axes[1].legend(fontsize=8)

plot_cdf(axes[2], inst_qps, "Instantaneous QPS", "#FF9800")
axes[2].set_xlabel("QPS (req/s)")
axes[2].set_title("Instantaneous QPS CDF")
axes[2].grid(alpha=0.3)
axes[2].set_xlim(0, min(20, max(inst_qps)))
axes[2].axvline(np.median(inst_qps), color="red", linestyle="--", alpha=0.5,
                label=f"median={np.median(inst_qps):.1f}")
axes[2].legend(fontsize=8)

plt.tight_layout()
out = CHARTS_DIR / "micro_varying_cdf.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()


# ============================================================
# Figure 2: Macro-benchmark Azure workloads (6 datasets)
# ============================================================
azure_workloads = {
    "Conv Light\n(n=979, ~3.3 qps)": "workload_azure_conv_light_real.jsonl",
    "Conv Medium\n(n=1761, ~5.9 qps)": "workload_azure_conv_medium_real.jsonl",
    "Conv Heavy\n(n=2997, ~10.0 qps)": "workload_azure_conv_heavy_real.jsonl",
    "Code Light\n(n=681, ~2.3 qps)": "workload_azure_code_light_real.jsonl",
    "Code Medium\n(n=1600, ~5.3 qps)": "workload_azure_code_medium_real.jsonl",
    "Code Heavy\n(n=2678, ~8.9 qps)": "workload_azure_code_heavy_real.jsonl",
}

colors_il = ["#1565C0", "#2196F3", "#64B5F6", "#E65100", "#FF9800", "#FFCC80"]
colors_ol = ["#1B5E20", "#4CAF50", "#81C784", "#4A148C", "#9C27B0", "#CE93D8"]

# --- Input Length CDF ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Macro-Benchmark: Azure Workload Distributions",
             fontsize=13, fontweight="bold")

ax = axes[0]
for i, (label, fname) in enumerate(azure_workloads.items()):
    rows = load_jsonl(WL_DIR / fname)
    plot_cdf(ax, [r["input_len"] for r in rows], label.split("\n")[0],
             colors_il[i])
ax.set_xlabel("Input Length (tokens)")
ax.set_ylabel("CDF")
ax.set_title("Input Length")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(0, 8000)

# --- Output Length CDF ---
ax = axes[1]
for i, (label, fname) in enumerate(azure_workloads.items()):
    rows = load_jsonl(WL_DIR / fname)
    plot_cdf(ax, [r["output_len"] for r in rows], label.split("\n")[0],
             colors_ol[i])
ax.set_xlabel("Output Length (tokens)")
ax.set_title("Output Length")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(0, 550)

# --- QPS (inter-arrival) CDF ---
ax = axes[2]
qps_colors = ["#1565C0", "#2196F3", "#64B5F6", "#E65100", "#FF9800", "#FFCC80"]
for i, (label, fname) in enumerate(azure_workloads.items()):
    rows = load_jsonl(WL_DIR / fname)
    arrivals = [r["arrival_time_s"] for r in rows]
    iat = [arrivals[j] - arrivals[j-1] for j in range(1, len(arrivals))]
    iqps = [1.0 / t if t > 0.001 else 100 for t in iat]
    plot_cdf(ax, iqps, label.split("\n")[0], qps_colors[i])
ax.set_xlabel("Instantaneous QPS (req/s)")
ax.set_title("Request Arrival Rate")
ax.legend(fontsize=7, loc="lower right")
ax.grid(alpha=0.3)
ax.set_xlim(0, 50)

plt.tight_layout()
out = CHARTS_DIR / "macro_azure_cdf.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()


# ============================================================
# Figure 3: Combined summary (both datasets overview)
# ============================================================
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Workload Dataset Overview\nTop: Micro-Benchmark (Varying)  |  Bottom: Macro-Benchmark (Azure)",
             fontsize=12, fontweight="bold")

# Top row: Varying
plot_cdf(axes[0, 0], ils, "Input Len", "#2196F3")
axes[0, 0].set_title("Varying: Input Length")
axes[0, 0].set_xlabel("tokens")
axes[0, 0].set_ylabel("CDF")
axes[0, 0].grid(alpha=0.3)

plot_cdf(axes[0, 1], ols, "Output Len", "#4CAF50")
axes[0, 1].set_title("Varying: Output Length")
axes[0, 1].set_xlabel("tokens")
axes[0, 1].grid(alpha=0.3)

plot_cdf(axes[0, 2], inst_qps, "QPS", "#FF9800")
axes[0, 2].set_title("Varying: QPS")
axes[0, 2].set_xlabel("req/s")
axes[0, 2].set_xlim(0, 15)
axes[0, 2].grid(alpha=0.3)

# Bottom row: Azure
ax = axes[1, 0]
for i, (label, fname) in enumerate(azure_workloads.items()):
    rows = load_jsonl(WL_DIR / fname)
    plot_cdf(ax, [r["input_len"] for r in rows], label.split("\n")[0], colors_il[i])
ax.set_title("Azure: Input Length")
ax.set_xlabel("tokens")
ax.set_ylabel("CDF")
ax.legend(fontsize=6)
ax.grid(alpha=0.3)
ax.set_xlim(0, 8000)

ax = axes[1, 1]
for i, (label, fname) in enumerate(azure_workloads.items()):
    rows = load_jsonl(WL_DIR / fname)
    plot_cdf(ax, [r["output_len"] for r in rows], label.split("\n")[0], colors_ol[i])
ax.set_title("Azure: Output Length")
ax.set_xlabel("tokens")
ax.legend(fontsize=6)
ax.grid(alpha=0.3)
ax.set_xlim(0, 550)

ax = axes[1, 2]
for i, (label, fname) in enumerate(azure_workloads.items()):
    rows = load_jsonl(WL_DIR / fname)
    arrivals = [r["arrival_time_s"] for r in rows]
    iat = [arrivals[j] - arrivals[j-1] for j in range(1, len(arrivals))]
    iqps = [1.0 / t if t > 0.001 else 100 for t in iat]
    plot_cdf(ax, iqps, label.split("\n")[0], qps_colors[i])
ax.set_title("Azure: QPS")
ax.set_xlabel("req/s")
ax.legend(fontsize=6)
ax.grid(alpha=0.3)
ax.set_xlim(0, 50)

plt.tight_layout()
out = CHARTS_DIR / "workload_overview_cdf.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

print("\nDone!")
