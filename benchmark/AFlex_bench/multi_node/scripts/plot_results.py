#!/usr/bin/env python3
"""Plot multi-node 16-GPU multi-architecture benchmark results.

Reads the latest results/multi_arch_16gpu_*.json and renders:
  1. Per-metric bar charts comparing architectures (baseline) at each QPS.
  2. Baseline vs Tier energy comparison per architecture.
  3. Energy-per-token and SLO across scenarios.

Also still supports the legacy pdaf_16gpu_tp*.json single-deploy file.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
RESULTS_DIR = BASE / "results"
CHARTS_DIR = BASE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["chatbot", "qa", "rag", "summary"]
ARCH_ORDER = ["native_dp", "pd_dp_xnode", "pd_dp_intra", "pdaf"]
ARCH_LABELS = {
    "native_dp": "Native DP16", "pd_dp_xnode": "PD DP8 x-node",
    "pd_dp_intra": "PD DP8 intra", "pdaf": "PDAF",
}
ARCH_COLORS = {
    "native_dp": "#3498db", "pd_dp_xnode": "#2ecc71",
    "pd_dp_intra": "#27ae60", "pdaf": "#e74c3c",
}

files = sorted(RESULTS_DIR.glob("multi_arch_16gpu_*.json"))
if not files:
    raise SystemExit("no multi_arch results in " + str(RESULTS_DIR))
with open(files[-1]) as f:
    blob = json.load(f)
results = blob["results"]          # {"<arch>_<mode>": {wl_key: metrics}}
meta = blob.get("meta", {})
qps_list = meta.get("qps", [1, 2, 4])


def get(deploy, scenario, qps, metric):
    d = results.get(deploy, {})
    if "__status__" in d:
        return None
    e = d.get(f"{scenario}_qps{qps}", {})
    return e.get(metric) if e.get("status") == "PASS" else None


def avg_over_scenarios(deploy, qps, metric):
    vals = [get(deploy, s, qps, metric) for s in SCENARIOS]
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


# --- Figure 1: arch comparison (baseline), avg over scenarios, per QPS ---
metrics = [
    ("throughput_tok_s", "Throughput (tok/s)"),
    ("ttft_proc_avg_ms", "TTFT proc avg (ms)"),
    ("tpot_avg_ms", "TPOT avg (ms)"),
    ("total_energy_j", "Total Energy (J)"),
    ("energy_per_token_mj", "Energy/token (mJ/tok)"),
    ("slo_violation_rate", "SLO violation (%)"),
]
present_archs = [a for a in ARCH_ORDER if f"{a}_baseline" in results]

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("16-GPU Architecture Comparison (baseline, avg over scenarios)\n"
             "Qwen3-32B, 2 nodes x 8 A800", fontsize=14, fontweight="bold")
for idx, (metric, label) in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    x = np.arange(len(qps_list))
    w = 0.8 / max(len(present_archs), 1)
    for i, arch in enumerate(present_archs):
        vals = [avg_over_scenarios(f"{arch}_baseline", q, metric) or 0 for q in qps_list]
        ax.bar(x + i * w, vals, w, label=ARCH_LABELS[arch], color=ARCH_COLORS[arch])
    ax.set_xticks(x + w * (len(present_archs) - 1) / 2)
    ax.set_xticklabels([f"QPS{q}" for q in qps_list])
    ax.set_ylabel(label)
    ax.set_title(label)
    ax.legend(fontsize=7)
    ax.grid(axis="y", alpha=0.3)
plt.tight_layout(rect=[0, 0, 1, 0.95])
out1 = CHARTS_DIR / "arch_comparison_baseline.png"
plt.savefig(out1, dpi=130, bbox_inches="tight")
print("saved", out1)

# --- Figure 2: Baseline vs Tier energy per arch (avg over scenarios+qps) ---
fig2, ax2 = plt.subplots(figsize=(11, 6))
labels, base_e, tier_e = [], [], []
for arch in present_archs:
    bvals = [avg_over_scenarios(f"{arch}_baseline", q, "energy_per_token_mj") for q in qps_list]
    tvals = [avg_over_scenarios(f"{arch}_tier", q, "energy_per_token_mj") for q in qps_list]
    bvals = [v for v in bvals if v is not None]
    tvals = [v for v in tvals if v is not None]
    if bvals:
        labels.append(ARCH_LABELS[arch])
        base_e.append(float(np.mean(bvals)))
        tier_e.append(float(np.mean(tvals)) if tvals else 0)
if labels:
    x = np.arange(len(labels))
    ax2.bar(x - 0.2, base_e, 0.4, label="baseline (lock 1410)", color="#95a5a6")
    ax2.bar(x + 0.2, tier_e, 0.4, label="tier (DVFS)", color="#f39c12")
    for i, (b, t) in enumerate(zip(base_e, tier_e)):
        if b and t:
            pct = (b - t) / b * 100
            ax2.text(i + 0.2, t, f"{pct:+.0f}%", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.set_ylabel("Energy/token (mJ/tok)")
    ax2.set_title("Baseline vs Tier (DVFS): Energy per Token (avg over scenarios+QPS)")
    ax2.legend()
    ax2.grid(axis="y", alpha=0.3)
    out2 = CHARTS_DIR / "baseline_vs_tier_energy.png"
    plt.savefig(out2, dpi=130, bbox_inches="tight")
    print("saved", out2)

print("done")
