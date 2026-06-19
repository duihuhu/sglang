#!/usr/bin/env python3
"""Plot static benchmark comparison: Native DP8 vs PD DP4 vs PDAF TP2."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Find latest result file
result_files = sorted(RESULTS_DIR.glob("static_bench_*.json"))
if not result_files:
    raise FileNotFoundError("No result files found in results/")
result_file = result_files[-1]
print(f"Using: {result_file.name}")

with open(result_file) as f:
    data = json.load(f)

DEPLOYS = ["native_dp8", "pd_dp4", "pdaf_tp2"]
DEPLOY_LABELS = ["Native DP8", "PD DP4", "PDAF TP2"]
COLORS = ["#2196F3", "#4CAF50", "#FF9800"]
WORKLOADS = list(data.get("native_dp8", {}).keys())
WL_SHORT = [w.replace("il", "IL").replace("_ol", "\nOL").replace("_qps", "\nQPS")
            for w in WORKLOADS]

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.dpi": 150,
})


def get_metric(deploy, workload, key):
    return data.get(deploy, {}).get(workload, {}).get(key, 0)


# --- Figure 1: Main comparison (4 subplots) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("MoE (Qwen3-30B-A3B) Static Benchmark\nNative DP8 vs PD DP4 vs PDAF TP2",
             fontsize=14, fontweight="bold")

x = np.arange(len(WORKLOADS))
width = 0.25

# 1) Throughput
ax = axes[0, 0]
for i, (dep, label, color) in enumerate(zip(DEPLOYS, DEPLOY_LABELS, COLORS)):
    vals = [get_metric(dep, w, "throughput_tok_s") for w in WORKLOADS]
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
ax.set_ylabel("Throughput (tok/s)")
ax.set_title("Throughput")
ax.set_xticks(x + width)
ax.set_xticklabels(WL_SHORT, fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# 2) TPOT
ax = axes[0, 1]
for i, (dep, label, color) in enumerate(zip(DEPLOYS, DEPLOY_LABELS, COLORS)):
    vals = [get_metric(dep, w, "tpot_avg_ms") for w in WORKLOADS]
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
ax.set_ylabel("TPOT avg (ms)")
ax.set_title("Time Per Output Token (TPOT)")
ax.set_xticks(x + width)
ax.set_xticklabels(WL_SHORT, fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# 3) Total Energy
ax = axes[1, 0]
for i, (dep, label, color) in enumerate(zip(DEPLOYS, DEPLOY_LABELS, COLORS)):
    vals = [get_metric(dep, w, "total_energy_j") / 1000 for w in WORKLOADS]
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
ax.set_ylabel("Total Energy (kJ)")
ax.set_title("Total Energy Consumption")
ax.set_xticks(x + width)
ax.set_xticklabels(WL_SHORT, fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

# 4) Energy Efficiency (mJ/tok)
ax = axes[1, 1]
for i, (dep, label, color) in enumerate(zip(DEPLOYS, DEPLOY_LABELS, COLORS)):
    vals = [get_metric(dep, w, "energy_per_token_mj") for w in WORKLOADS]
    ax.bar(x + i * width, vals, width, label=label, color=color, alpha=0.85)
ax.set_ylabel("Energy per Token (mJ/tok)")
ax.set_title("Energy Efficiency")
ax.set_xticks(x + width)
ax.set_xticklabels(WL_SHORT, fontsize=8)
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out1 = CHARTS_DIR / "static_4panel_comparison.png"
plt.savefig(out1, dpi=150, bbox_inches="tight")
print(f"Saved: {out1}")
plt.close()


# --- Figure 2: Energy savings bar chart ---
fig, ax = plt.subplots(figsize=(12, 5))
fig.suptitle("Energy Savings vs Native DP8 (Qwen3-30B-A3B, Max Freq)",
             fontsize=13, fontweight="bold")

x = np.arange(len(WORKLOADS))
width = 0.35

for i, (dep, label, color) in enumerate(zip(DEPLOYS[1:], DEPLOY_LABELS[1:], COLORS[1:])):
    savings = []
    for w in WORKLOADS:
        native_e = get_metric("native_dp8", w, "total_energy_j")
        dep_e = get_metric(dep, w, "total_energy_j")
        if native_e > 0:
            savings.append((native_e - dep_e) / native_e * 100)
        else:
            savings.append(0)
    bars = ax.bar(x + i * width, savings, width, label=label, color=color, alpha=0.85)
    for bar, val in zip(bars, savings):
        y = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, y + (1 if y >= 0 else -3),
                f"{val:.1f}%", ha="center", va="bottom" if y >= 0 else "top",
                fontsize=8)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Energy Saving (%)")
ax.set_xlabel("Workload")
ax.set_xticks(x + width / 2)
ax.set_xticklabels(WL_SHORT, fontsize=8)
ax.legend(fontsize=10, loc="lower left")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(min(-55, ax.get_ylim()[0]), max(15, ax.get_ylim()[1]))

plt.tight_layout()
out2 = CHARTS_DIR / "static_energy_savings.png"
plt.savefig(out2, dpi=150, bbox_inches="tight")
print(f"Saved: {out2}")
plt.close()


# --- Figure 3: Summary table image ---
fig, ax = plt.subplots(figsize=(14, 4))
ax.axis("off")

col_labels = ["Workload", "Thpt\n(tok/s)", "TPOT\n(ms)", "Energy\n(kJ)", "mJ/tok",
              "Thpt\n(tok/s)", "TPOT\n(ms)", "Energy\n(kJ)", "mJ/tok",
              "Thpt\n(tok/s)", "TPOT\n(ms)", "Energy\n(kJ)", "mJ/tok"]
table_data = []
for w, ws in zip(WORKLOADS, WL_SHORT):
    row = [w.replace("_", " ")]
    for dep in DEPLOYS:
        row.append(f"{get_metric(dep, w, 'throughput_tok_s'):.0f}")
        row.append(f"{get_metric(dep, w, 'tpot_avg_ms'):.0f}")
        row.append(f"{get_metric(dep, w, 'total_energy_j')/1000:.1f}")
        row.append(f"{get_metric(dep, w, 'energy_per_token_mj'):.0f}")
    table_data.append(row)

table = ax.table(cellText=table_data, colLabels=col_labels, loc="center",
                 cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1.0, 1.4)

# Color header groups
for j in range(1, 5):
    table[0, j].set_facecolor("#BBDEFB")
for j in range(5, 9):
    table[0, j].set_facecolor("#C8E6C9")
for j in range(9, 13):
    table[0, j].set_facecolor("#FFE0B2")

ax.set_title("Native DP8              PD DP4              PDAF TP2",
             fontsize=11, pad=20)
fig.suptitle("Static Benchmark Summary (Qwen3-30B-A3B, 8×A800, Max Freq)",
             fontsize=12, fontweight="bold", y=0.98)

plt.tight_layout()
out3 = CHARTS_DIR / "static_summary_table.png"
plt.savefig(out3, dpi=150, bbox_inches="tight")
print(f"Saved: {out3}")
plt.close()

print("\nDone! All charts saved to:", CHARTS_DIR)
