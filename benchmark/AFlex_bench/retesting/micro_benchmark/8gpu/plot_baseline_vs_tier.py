#!/usr/bin/env python3
"""Plot Qwen3-32B 8GPU: Baseline vs Tier comparison (chatbot + qa)."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = HERE / "results"

# Load results
baseline_files = sorted(RESULTS_DIR.glob("micro_8gpu_2*.json"))
tier_files = sorted(RESULTS_DIR.glob("micro_8gpu_tier_*.json"))

with open(baseline_files[-1]) as f:
    baseline_raw = json.load(f)
with open(tier_files[-1]) as f:
    tier_raw = json.load(f)

# Normalize keys (strip _tier suffix for tier data)
baseline = {}
for k, v in baseline_raw.items():
    baseline[k] = v
tier = {}
for k, v in tier_raw.items():
    tier[k.replace("_tier", "")] = v

SCENARIOS = ["chatbot", "qa"]
QPS_LIST = [1, 2, 3]
DEPLOYS = ["native_dp", "pd_dp", "pdaf"]
DEPLOY_LABELS = {"native_dp": "Native DP8", "pd_dp": "PD DP4", "pdaf": "PDAF TP2"}
COLORS_BASE = {"native_dp": "#3498db", "pd_dp": "#2ecc71", "pdaf": "#e74c3c"}
COLORS_TIER = {"native_dp": "#85c1e9", "pd_dp": "#82e0aa", "pdaf": "#f1948a"}


def get(data, deploy, scenario, qps, metric):
    key = f"{scenario}_qps{qps}"
    entry = data.get(deploy, {}).get(key, {})
    if entry.get("status") != "PASS":
        return 0
    mapping = {
        "thpt": "throughput_tok_s",
        "ttft": "ttft_proc_avg_ms",
        "tpot": "tpot_avg_ms",
        "energy": "total_energy_j",
        "mj_tok": "energy_per_token_mj",
        "slo": "slo_violation_rate",
    }
    return entry.get(mapping.get(metric, metric), 0)


# --- Figure 1: Baseline vs Tier (QPS=3) bar chart ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Qwen3-32B 8-GPU: Baseline (MaxFreq) vs DVFS-Tier (QPS=3)",
             fontsize=14, fontweight='bold')

metrics_cfg = [
    ("mj_tok", "Energy per Token (mJ/tok)", True),
    ("energy", "Total Energy (J)", True),
    ("tpot", "TPOT (ms)", True),
    ("thpt", "Throughput (tok/s)", False),
]

for idx, (metric, ylabel, lower_better) in enumerate(metrics_cfg):
    ax = axes[idx // 2, idx % 2]
    x = np.arange(len(SCENARIOS))
    width = 0.13
    offset = 0

    for deploy in DEPLOYS:
        vals_b = [get(baseline, deploy, s, 3, metric) for s in SCENARIOS]
        vals_t = [get(tier, deploy, s, 3, metric) for s in SCENARIOS]

        ax.bar(x + offset, vals_b, width,
               label=f"{DEPLOY_LABELS[deploy]} Base",
               color=COLORS_BASE[deploy], edgecolor='black', linewidth=0.5)
        offset += width
        ax.bar(x + offset, vals_t, width,
               label=f"{DEPLOY_LABELS[deploy]} Tier",
               color=COLORS_TIER[deploy], edgecolor='black', linewidth=0.5,
               hatch='//')
        offset += width + 0.02

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels([s.capitalize() for s in SCENARIOS])
    ax.set_ylabel(ylabel)
    ax.legend(fontsize=6.5, ncol=2)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "baseline_vs_tier_qps3.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 2: Energy saving percentage ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Qwen3-32B 8-GPU: Energy Saving (Tier vs Baseline) by Scenario & QPS",
             fontsize=13, fontweight='bold')

x = np.arange(len(SCENARIOS) * len(QPS_LIST))
labels = [f"{s.capitalize()}\nQPS={q}" for s in SCENARIOS for q in QPS_LIST]
width = 0.25

for i, deploy in enumerate(DEPLOYS):
    savings = []
    for s in SCENARIOS:
        for q in QPS_LIST:
            b = get(baseline, deploy, s, q, "mj_tok")
            t = get(tier, deploy, s, q, "mj_tok")
            pct = (b - t) / b * 100 if b > 0 else 0
            savings.append(pct)
    ax.bar(x + i * width, savings, width, label=DEPLOY_LABELS[deploy],
           color=COLORS_BASE[deploy], edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Energy Saving (%)")
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "energy_saving_pct.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 3: Per-architecture energy/token vs QPS ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Qwen3-32B 8-GPU: Energy/Token - Baseline vs Tier by QPS",
             fontsize=13, fontweight='bold')

for i, deploy in enumerate(DEPLOYS):
    ax = axes[i]
    for scenario in SCENARIOS:
        vals_b = [get(baseline, deploy, scenario, q, "mj_tok") for q in QPS_LIST]
        vals_t = [get(tier, deploy, scenario, q, "mj_tok") for q in QPS_LIST]
        ax.plot(QPS_LIST, vals_b, 'o-', label=f"{scenario} Base", linewidth=2, markersize=7)
        ax.plot(QPS_LIST, vals_t, 's--', label=f"{scenario} Tier", linewidth=1.5, markersize=6, alpha=0.7)
    ax.set_xlabel("QPS")
    ax.set_ylabel("mJ/tok")
    ax.set_title(DEPLOY_LABELS[deploy])
    ax.legend(fontsize=9, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xticks(QPS_LIST)

plt.tight_layout()
out = CHARTS_DIR / "energy_per_tok_by_qps.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# --- Figure 4: Total Energy saving percentage ---
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Qwen3-32B 8-GPU: Total Energy Saving (Tier vs Baseline) by Scenario & QPS",
             fontsize=13, fontweight='bold')

x = np.arange(len(SCENARIOS) * len(QPS_LIST))
labels = [f"{s.capitalize()}\nQPS={q}" for s in SCENARIOS for q in QPS_LIST]
width = 0.25

for i, deploy in enumerate(DEPLOYS):
    savings = []
    for s in SCENARIOS:
        for q in QPS_LIST:
            b = get(baseline, deploy, s, q, "energy")
            t = get(tier, deploy, s, q, "energy")
            pct = (b - t) / b * 100 if b > 0 else 0
            savings.append(pct)
    bars = ax.bar(x + i * width, savings, width, label=DEPLOY_LABELS[deploy],
           color=COLORS_BASE[deploy], edgecolor='black', linewidth=0.5)
    for bar, v in zip(bars, savings):
        if abs(v) > 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f"{v:.1f}%", ha='center', va='bottom', fontsize=7)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Total Energy Saving (%)")
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "total_energy_saving_pct.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 5: Total Energy cross-architecture comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Qwen3-32B 8-GPU: Total Energy (J) Cross-Architecture Comparison",
             fontsize=14, fontweight='bold')

for idx, scenario in enumerate(SCENARIOS):
    ax = axes[idx]
    x = np.arange(len(QPS_LIST))
    width = 0.13
    offset = 0

    for deploy in DEPLOYS:
        vals_b = [get(baseline, deploy, scenario, q, "energy") for q in QPS_LIST]
        vals_t = [get(tier, deploy, scenario, q, "energy") for q in QPS_LIST]

        ax.bar(x + offset, vals_b, width,
               label=f"{DEPLOY_LABELS[deploy]} Base",
               color=COLORS_BASE[deploy], edgecolor='black', linewidth=0.5)
        offset += width
        ax.bar(x + offset, vals_t, width,
               label=f"{DEPLOY_LABELS[deploy]} Tier",
               color=COLORS_TIER[deploy], edgecolor='black', linewidth=0.5,
               hatch='//')
        offset += width + 0.02

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels([f"QPS={q}" for q in QPS_LIST])
    ax.set_ylabel("Total Energy (J)")
    ax.set_title(f"{scenario.capitalize()}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "total_energy_cross_arch.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 6: Energy per token cross-architecture comparison ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Qwen3-32B 8-GPU: Energy per Token (mJ/tok) Cross-Architecture Comparison",
             fontsize=14, fontweight='bold')

for idx, scenario in enumerate(SCENARIOS):
    ax = axes[idx]
    x = np.arange(len(QPS_LIST))
    width = 0.13
    offset = 0

    for deploy in DEPLOYS:
        vals_b = [get(baseline, deploy, scenario, q, "mj_tok") for q in QPS_LIST]
        vals_t = [get(tier, deploy, scenario, q, "mj_tok") for q in QPS_LIST]

        ax.bar(x + offset, vals_b, width,
               label=f"{DEPLOY_LABELS[deploy]} Base",
               color=COLORS_BASE[deploy], edgecolor='black', linewidth=0.5)
        offset += width
        ax.bar(x + offset, vals_t, width,
               label=f"{DEPLOY_LABELS[deploy]} Tier",
               color=COLORS_TIER[deploy], edgecolor='black', linewidth=0.5,
               hatch='//')
        offset += width + 0.02

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels([f"QPS={q}" for q in QPS_LIST])
    ax.set_ylabel("Energy per Token (mJ/tok)")
    ax.set_title(f"{scenario.capitalize()}")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "energy_per_tok_cross_arch.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

print("\nDone! All charts saved to:", CHARTS_DIR)
