#!/usr/bin/env python3
"""Plot Energy per Token cross-arch comparison: ignore-eos=True vs False."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = HERE / "results"

SCENARIOS = ["chatbot", "qa"]
QPS_LIST = [1, 2, 3]
DEPLOYS = ["native_dp", "pd_dp", "pdaf"]
DEPLOY_LABELS = {"native_dp": "Native DP8", "pd_dp": "PD DP4", "pdaf": "PDAF TP2"}
COLORS_BASE = {"native_dp": "#3498db", "pd_dp": "#2ecc71", "pdaf": "#e74c3c"}
COLORS_TIER = {"native_dp": "#85c1e9", "pd_dp": "#82e0aa", "pdaf": "#f1948a"}


def load_pair(baseline_path, tier_path):
    with open(baseline_path) as f:
        baseline_raw = json.load(f)
    with open(tier_path) as f:
        tier_raw = json.load(f)
    baseline = dict(baseline_raw)
    tier = {k.replace("_tier", ""): v for k, v in tier_raw.items()}
    return baseline, tier


def get(data, deploy, scenario, qps, metric="mj_tok"):
    key = f"{scenario}_qps{qps}"
    entry = data.get(deploy, {}).get(key, {})
    if entry.get("status") != "PASS":
        return 0
    mapping = {
        "energy": "total_energy_j",
        "mj_tok": "energy_per_token_mj",
    }
    return entry.get(mapping.get(metric, metric), 0)


def plot_cross_arch_panel(ax, baseline, tier, scenario, title_suffix=""):
    x = np.arange(len(QPS_LIST))
    width = 0.13
    offset = 0

    for deploy in DEPLOYS:
        vals_b = [get(baseline, deploy, scenario, q, "mj_tok") for q in QPS_LIST]
        vals_t = [get(tier, deploy, scenario, q, "mj_tok") for q in QPS_LIST]

        ax.bar(x + offset, vals_b, width,
               label=f"{DEPLOY_LABELS[deploy]} Base",
               color=COLORS_BASE[deploy], edgecolor="black", linewidth=0.5)
        offset += width
        ax.bar(x + offset, vals_t, width,
               label=f"{DEPLOY_LABELS[deploy]} Tier",
               color=COLORS_TIER[deploy], edgecolor="black", linewidth=0.5,
               hatch="//")
        offset += width + 0.02

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels([f"QPS={q}" for q in QPS_LIST])
    ax.set_ylabel("Energy per Token (mJ/tok)")
    title = scenario.capitalize()
    if title_suffix:
        title = f"{title} ({title_suffix})"
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(axis="y", alpha=0.3)


# ignore-eos=True (fixed output length)
baseline_true, tier_true = load_pair(
    RESULTS_DIR / "micro_8gpu_20260622_133455.json",
    RESULTS_DIR / "micro_8gpu_tier_20260622_151338.json",
)

# ignore-eos=False (early stop at EOS)
baseline_false, tier_false = load_pair(
    RESULTS_DIR / "micro_8gpu_20260622_105617.json",
    RESULTS_DIR / "micro_8gpu_tier_20260622_121538.json",
)


# --- Figure 1: ignore-eos=False only (same format as energy_per_tok_cross_arch) ---
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Qwen3-32B 8-GPU: Energy per Token (mJ/tok) — ignore-eos=False",
             fontsize=14, fontweight="bold")
for idx, scenario in enumerate(SCENARIOS):
    plot_cross_arch_panel(axes[idx], baseline_false, tier_false, scenario)
plt.tight_layout()
out = CHARTS_DIR / "energy_per_tok_cross_arch_ignore_false.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")


# --- Figure 2: merged ignore-eos=True vs False ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Qwen3-32B 8-GPU: Energy per Token (mJ/tok) — ignore-eos=True vs False",
             fontsize=14, fontweight="bold")

datasets = [
    (baseline_true, tier_true, "ignore-eos=True"),
    (baseline_false, tier_false, "ignore-eos=False"),
]
for row, (bl, tr, eos_label) in enumerate(datasets):
    for col, scenario in enumerate(SCENARIOS):
        plot_cross_arch_panel(axes[row, col], bl, tr, scenario, eos_label)

plt.tight_layout()
out = CHARTS_DIR / "energy_per_tok_cross_arch_ignore_eos_merged.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: {out}")

print("\nDone!")
