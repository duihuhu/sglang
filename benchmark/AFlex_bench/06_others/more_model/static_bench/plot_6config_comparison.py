#!/usr/bin/env python3
"""Plot full 6-config comparison: with and without Tier DVFS."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Load both result files and merge
result_files = sorted(RESULTS_DIR.glob("static_bench_*.json"))
merged = {}
for rf in result_files:
    with open(rf) as f:
        d = json.load(f)
    merged.update(d)

print(f"Loaded {len(merged)} deploy configs: {list(merged.keys())}")

DEPLOYS = ["native_dp8", "native_dp8_tier", "pd_dp4", "pd_dp4_tier",
           "pdaf_tp2", "pdaf_tp2_tier"]
DEPLOY_LABELS = ["Native DP8", "Native DP8\n+Tier", "PD DP4",
                 "PD DP4\n+Tier", "PDAF TP2", "PDAF TP2\n+Tier"]
COLORS = ["#2196F3", "#1565C0", "#4CAF50", "#2E7D32", "#FF9800", "#E65100"]
HATCHES = ["", "//", "", "//", "", "//"]

WORKLOADS = list(merged.get("native_dp8", {}).keys())
WL_SHORT = [w.replace("il", "IL").replace("_ol", "/OL").replace("_qps", "/Q")
            for w in WORKLOADS]

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def get_m(deploy, workload, key):
    return merged.get(deploy, {}).get(workload, {}).get(key, 0)


# === Figure 1: Energy per token (mJ/tok) - main comparison ===
fig, ax = plt.subplots(figsize=(14, 6))
x = np.arange(len(WORKLOADS))
width = 0.13

for i, (dep, label, color, hatch) in enumerate(
        zip(DEPLOYS, DEPLOY_LABELS, COLORS, HATCHES)):
    vals = [get_m(dep, w, "energy_per_token_mj") for w in WORKLOADS]
    ax.bar(x + i * width, vals, width, label=label, color=color,
           alpha=0.85, hatch=hatch, edgecolor="white")

ax.set_ylabel("Energy per Token (mJ/tok)")
ax.set_title("MoE (Qwen3-30B-A3B) Energy Efficiency: 6 Configs × 6 Workloads",
             fontsize=13, fontweight="bold")
ax.set_xticks(x + width * 2.5)
ax.set_xticklabels(WL_SHORT, fontsize=9)
ax.legend(fontsize=8, ncol=3, loc="upper right")
ax.grid(axis="y", alpha=0.3)
ax.set_xlabel("Workload (IL/OL/QPS)")

plt.tight_layout()
out = CHARTS_DIR / "static_6config_energy_efficiency.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()


# === Figure 2: Energy savings vs native_dp8 (baseline) ===
fig, ax = plt.subplots(figsize=(14, 6))

compare_deploys = ["native_dp8_tier", "pd_dp4", "pd_dp4_tier",
                   "pdaf_tp2", "pdaf_tp2_tier"]
compare_labels = ["Native+Tier", "PD DP4", "PD DP4+Tier",
                  "PDAF TP2", "PDAF TP2+Tier"]
compare_colors = ["#1565C0", "#4CAF50", "#2E7D32", "#FF9800", "#E65100"]

x = np.arange(len(WORKLOADS))
width = 0.15

for i, (dep, label, color) in enumerate(
        zip(compare_deploys, compare_labels, compare_colors)):
    savings = []
    for w in WORKLOADS:
        native_e = get_m("native_dp8", w, "total_energy_j")
        dep_e = get_m(dep, w, "total_energy_j")
        if native_e > 0:
            savings.append((native_e - dep_e) / native_e * 100)
        else:
            savings.append(0)
    bars = ax.bar(x + i * width, savings, width, label=label,
                  color=color, alpha=0.85)
    for bar, val in zip(bars, savings):
        y = bar.get_height()
        if abs(val) > 3:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    y + (0.8 if y >= 0 else -2.5),
                    f"{val:.0f}%", ha="center",
                    va="bottom" if y >= 0 else "top", fontsize=7)

ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Energy Saving vs Native DP8 (%)")
ax.set_title("Energy Savings vs Native DP8 (Max Freq Baseline)",
             fontsize=13, fontweight="bold")
ax.set_xticks(x + width * 2)
ax.set_xticklabels(WL_SHORT, fontsize=9)
ax.set_xlabel("Workload (IL/OL/QPS)")
ax.legend(fontsize=9, loc="lower left")
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "static_6config_savings_vs_native.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()


# === Figure 3: 4-panel with Tier ===
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("MoE Static Benchmark: 6 Configs (Qwen3-30B-A3B, 8×A800)",
             fontsize=14, fontweight="bold")

x = np.arange(len(WORKLOADS))
width = 0.13

metrics = [
    ("throughput_tok_s", "Throughput (tok/s)", "Throughput"),
    ("tpot_avg_ms", "TPOT avg (ms)", "Time Per Output Token"),
    ("total_energy_j", "Total Energy (J)", "Total Energy"),
    ("energy_per_token_mj", "Energy/Token (mJ/tok)", "Energy Efficiency"),
]

for ax_idx, (key, ylabel, title) in enumerate(metrics):
    ax = axes[ax_idx // 2, ax_idx % 2]
    for i, (dep, label, color, hatch) in enumerate(
            zip(DEPLOYS, DEPLOY_LABELS, COLORS, HATCHES)):
        vals = [get_m(dep, w, key) for w in WORKLOADS]
        if key == "total_energy_j":
            vals = [v / 1000 for v in vals]
            ylabel = "Total Energy (kJ)"
        ax.bar(x + i * width, vals, width, label=label, color=color,
               alpha=0.85, hatch=hatch, edgecolor="white")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xticks(x + width * 2.5)
    ax.set_xticklabels(WL_SHORT, fontsize=7)
    ax.legend(fontsize=7, ncol=3)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "static_6config_4panel.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()


# === Figure 4: Tier effect (paired bars) ===
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Tier DVFS Effect: Energy Savings within Each Architecture",
             fontsize=13, fontweight="bold")

pairs = [("native_dp8", "native_dp8_tier", "Native DP8", "#2196F3"),
         ("pd_dp4", "pd_dp4_tier", "PD DP4", "#4CAF50"),
         ("pdaf_tp2", "pdaf_tp2_tier", "PDAF TP2", "#FF9800")]

for ax_idx, (base, tier, name, color) in enumerate(pairs):
    ax = axes[ax_idx]
    savings = []
    for w in WORKLOADS:
        base_e = get_m(base, w, "total_energy_j")
        tier_e = get_m(tier, w, "total_energy_j")
        if base_e > 0:
            savings.append((base_e - tier_e) / base_e * 100)
        else:
            savings.append(0)
    bars = ax.bar(range(len(WORKLOADS)), savings, color=color, alpha=0.8)
    for bar, val in zip(bars, savings):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}%", ha="center", fontsize=8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"{name}: Tier vs Max Freq")
    ax.set_ylabel("Energy Saving (%)")
    ax.set_xticks(range(len(WORKLOADS)))
    ax.set_xticklabels(WL_SHORT, fontsize=7, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(-30, max(savings) + 5 if max(savings) > 0 else 10)

plt.tight_layout()
out = CHARTS_DIR / "static_tier_effect.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
plt.close()

print("\nDone!")
