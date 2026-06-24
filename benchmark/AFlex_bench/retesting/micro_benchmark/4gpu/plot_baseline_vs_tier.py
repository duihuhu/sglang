#!/usr/bin/env python3
"""Plot Qwen3-32B 4GPU: Baseline vs Tier comparison."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["chatbot", "qa", "rag", "summary"]
QPS_LIST = [1, 2, 3]

# Data from benchmark logs (Baseline = full_baseline_v2, Tier = full_tier_v3 + pdaf_tier_v2)
baseline = {
    "native_dp": {
        "chatbot_qps1": {"thpt": 14.9, "ttft": 90.5, "tpot": 43.6, "energy": 107135, "mj_tok": 35759.3},
        "chatbot_qps2": {"thpt": 29.7, "ttft": 89.7, "tpot": 43.8, "energy": 70856, "mj_tok": 23658.0},
        "chatbot_qps3": {"thpt": 44.3, "ttft": 88.5, "tpot": 44.2, "energy": 58068, "mj_tok": 19433.6},
        "qa_qps1": {"thpt": 242.1, "ttft": 90.5, "tpot": 46.1, "energy": 282531, "mj_tok": 5518.2},
        "qa_qps2": {"thpt": 458.7, "ttft": 91.1, "tpot": 47.4, "energy": 149216, "mj_tok": 2914.4},
        "qa_qps3": {"thpt": 652.7, "ttft": 91.4, "tpot": 48.4, "energy": 105161, "mj_tok": 2053.9},
        "rag_qps1": {"thpt": 63.1, "ttft": 90.2, "tpot": 44.4, "energy": 201529, "mj_tok": 15744.4},
        "rag_qps2": {"thpt": 124.4, "ttft": 90.1, "tpot": 45.7, "energy": 132208, "mj_tok": 10328.7},
        "rag_qps3": {"thpt": 184.1, "ttft": 90.5, "tpot": 46.8, "energy": 93663, "mj_tok": 7317.4},
        "summary_qps1": {"thpt": 21.4, "ttft": 90.6, "tpot": 44.0, "energy": 122876, "mj_tok": 28609.2},
        "summary_qps2": {"thpt": 63.1, "ttft": 89.5, "tpot": 44.6, "energy": 93395, "mj_tok": 14673.3},
        "summary_qps3": {"thpt": 156.1, "ttft": 90.3, "tpot": 46.4, "energy": 87130, "mj_tok": 8258.8},
    },
    "pd_dp": {
        "chatbot_qps1": {"thpt": 14.3, "ttft": 47.9, "tpot": 47.9, "energy": 101900, "mj_tok": 35592.0},
        "chatbot_qps2": {"thpt": 28.8, "ttft": 47.6, "tpot": 47.5, "energy": 62462, "mj_tok": 21546.1},
        "chatbot_qps3": {"thpt": 43.5, "ttft": 48.4, "tpot": 47.9, "energy": 46749, "mj_tok": 15944.3},
        "qa_qps1": {"thpt": 241.8, "ttft": 46.1, "tpot": 46.8, "energy": 182767, "mj_tok": 3569.7},
        "qa_qps2": {"thpt": 457.3, "ttft": 46.9, "tpot": 47.6, "energy": 97784, "mj_tok": 1909.8},
        "qa_qps3": {"thpt": 649.9, "ttft": 47.1, "tpot": 47.8, "energy": 68636, "mj_tok": 1340.5},
        "rag_qps1": {"thpt": 63.1, "ttft": 46.8, "tpot": 46.0, "energy": 149042, "mj_tok": 11643.9},
        "rag_qps2": {"thpt": 124.3, "ttft": 46.8, "tpot": 46.8, "energy": 86882, "mj_tok": 6787.6},
        "rag_qps3": {"thpt": 183.9, "ttft": 48.8, "tpot": 47.3, "energy": 62069, "mj_tok": 4849.1},
        "summary_qps1": {"thpt": 57.3, "ttft": 47.3, "tpot": 46.3, "energy": 149368, "mj_tok": 12843.3},
        "summary_qps2": {"thpt": 116.1, "ttft": 47.2, "tpot": 47.1, "energy": 86304, "mj_tok": 7225.1},
        "summary_qps3": {"thpt": 166.2, "ttft": 49.4, "tpot": 47.7, "energy": 60989, "mj_tok": 5264.4},
    },
    "pdaf": {
        "chatbot_qps1": {"thpt": 14.5, "ttft": 49.5, "tpot": 47.5, "energy": 98747, "mj_tok": 34015.5},
        "chatbot_qps2": {"thpt": 29.2, "ttft": 49.4, "tpot": 48.0, "energy": 58274, "mj_tok": 19848.0},
        "chatbot_qps3": {"thpt": 43.9, "ttft": 49.3, "tpot": 48.4, "energy": 42818, "mj_tok": 14494.8},
        "qa_qps1": {"thpt": 240.9, "ttft": 52.3, "tpot": 52.0, "energy": 143174, "mj_tok": 2796.4},
        "qa_qps2": {"thpt": 452.6, "ttft": 52.1, "tpot": 55.1, "energy": 76561, "mj_tok": 1495.3},
        "qa_qps3": {"thpt": 637.1, "ttft": 52.3, "tpot": 58.7, "energy": 55476, "mj_tok": 1083.5},
        "rag_qps1": {"thpt": 63.0, "ttft": 81.6, "tpot": 49.5, "energy": 134337, "mj_tok": 10495.0},
        "rag_qps2": {"thpt": 124.1, "ttft": 83.9, "tpot": 50.8, "energy": 73161, "mj_tok": 5715.7},
        "rag_qps3": {"thpt": 183.3, "ttft": 84.5, "tpot": 52.2, "energy": 51524, "mj_tok": 4025.3},
        "summary_qps1": {"thpt": 54.2, "ttft": 150.8, "tpot": 49.6, "energy": 136044, "mj_tok": 12469.7},
        "summary_qps2": {"thpt": 102.7, "ttft": 161.6, "tpot": 51.0, "energy": 77216, "mj_tok": 7350.4},
        "summary_qps3": {"thpt": 150.8, "ttft": 187.7, "tpot": 52.2, "energy": 56278, "mj_tok": 5334.4},
    },
}

tier = {
    "native_dp": {
        "chatbot_qps1": {"thpt": 14.9, "ttft": 92.3, "tpot": 43.7, "energy": 98573, "mj_tok": 32901.5},
        "chatbot_qps2": {"thpt": 29.8, "ttft": 92.4, "tpot": 45.1, "energy": 68374, "mj_tok": 22798.8},
        "chatbot_qps3": {"thpt": 44.3, "ttft": 88.8, "tpot": 44.3, "energy": 57838, "mj_tok": 19369.8},
        "qa_qps1": {"thpt": 242.2, "ttft": 90.7, "tpot": 46.1, "energy": 281094, "mj_tok": 5490.1},
        "qa_qps2": {"thpt": 458.8, "ttft": 91.1, "tpot": 47.4, "energy": 150995, "mj_tok": 2949.1},
        "qa_qps3": {"thpt": 652.5, "ttft": 91.5, "tpot": 48.5, "energy": 106784, "mj_tok": 2085.6},
        "rag_qps1": {"thpt": 63.1, "ttft": 91.4, "tpot": 44.5, "energy": 196940, "mj_tok": 15386.0},
        "rag_qps2": {"thpt": 124.4, "ttft": 90.3, "tpot": 45.7, "energy": 132786, "mj_tok": 10374.0},
        "rag_qps3": {"thpt": 184.0, "ttft": 90.7, "tpot": 46.9, "energy": 92353, "mj_tok": 7215.1},
        "summary_qps1": {"thpt": 22.0, "ttft": 90.5, "tpot": 44.0, "energy": 120928, "mj_tok": 27297.5},
        "summary_qps2": {"thpt": 62.2, "ttft": 89.6, "tpot": 44.6, "energy": 92288, "mj_tok": 14707.2},
        "summary_qps3": {"thpt": 157.4, "ttft": 90.5, "tpot": 46.5, "energy": 88301, "mj_tok": 8299.0},
    },
    "pd_dp": {
        "chatbot_qps1": {"thpt": 14.3, "ttft": 46.9, "tpot": 48.0, "energy": 103893, "mj_tok": 36288.1},
        "chatbot_qps2": {"thpt": 28.9, "ttft": 46.7, "tpot": 47.7, "energy": 62389, "mj_tok": 21446.8},
        "chatbot_qps3": {"thpt": 43.6, "ttft": 49.0, "tpot": 48.0, "energy": 47420, "mj_tok": 16151.4},
        "qa_qps1": {"thpt": 242.3, "ttft": 45.8, "tpot": 46.9, "energy": 184226, "mj_tok": 3598.2},
        "qa_qps2": {"thpt": 457.4, "ttft": 46.1, "tpot": 47.8, "energy": 98506, "mj_tok": 1923.9},
        "qa_qps3": {"thpt": 652.0, "ttft": 46.8, "tpot": 48.0, "energy": 71207, "mj_tok": 1390.8},
        "rag_qps1": {"thpt": 63.1, "ttft": 47.4, "tpot": 46.0, "energy": 150947, "mj_tok": 11792.7},
        "rag_qps2": {"thpt": 124.4, "ttft": 48.1, "tpot": 46.8, "energy": 87923, "mj_tok": 6868.9},
        "rag_qps3": {"thpt": 184.0, "ttft": 48.1, "tpot": 47.3, "energy": 61207, "mj_tok": 4781.8},
        "summary_qps1": {"thpt": 57.5, "ttft": 45.9, "tpot": 46.3, "energy": 146669, "mj_tok": 12562.6},
        "summary_qps2": {"thpt": 110.0, "ttft": 46.9, "tpot": 47.2, "energy": 87061, "mj_tok": 7694.3},
        "summary_qps3": {"thpt": 164.6, "ttft": 47.4, "tpot": 47.8, "energy": 61480, "mj_tok": 5369.5},
    },
    "pdaf": {
        "chatbot_qps1": {"thpt": 29.7, "ttft": 67.9, "tpot": 52.3, "energy": 81042, "mj_tok": 13586.3},
        "chatbot_qps2": {"thpt": 111.6, "ttft": 76.6, "tpot": 55.6, "energy": 64286, "mj_tok": 4266.1},
        "chatbot_qps3": {"thpt": 176.8, "ttft": 81.0, "tpot": 56.4, "energy": 33654, "mj_tok": 2813.7},
        "qa_qps1": {"thpt": 239.3, "ttft": 76.0, "tpot": 56.4, "energy": 103833, "mj_tok": 2028.0},
        "qa_qps2": {"thpt": 447.1, "ttft": 92.1, "tpot": 60.1, "energy": 57232, "mj_tok": 1117.8},
        "qa_qps3": {"thpt": 623.1, "ttft": 112.4, "tpot": 66.2, "energy": 41460, "mj_tok": 809.8},
        "rag_qps1": {"thpt": 62.9, "ttft": 127.7, "tpot": 54.1, "energy": 97751, "mj_tok": 7636.8},
        "rag_qps2": {"thpt": 123.5, "ttft": 157.1, "tpot": 55.5, "energy": 52728, "mj_tok": 4119.4},
        "rag_qps3": {"thpt": 182.0, "ttft": 196.2, "tpot": 56.4, "energy": 37029, "mj_tok": 2892.9},
        "summary_qps1": {"thpt": 51.6, "ttft": 253.7, "tpot": 54.3, "energy": 99367, "mj_tok": 9540.7},
        "summary_qps2": {"thpt": 100.1, "ttft": 360.9, "tpot": 56.0, "energy": 54494, "mj_tok": 5300.9},
        "summary_qps3": {"thpt": 147.1, "ttft": 493.3, "tpot": 57.7, "energy": 39404, "mj_tok": 3799.8},
    },
}

DEPLOYS = ["native_dp", "pd_dp", "pdaf"]
DEPLOY_LABELS = {"native_dp": "Native DP4", "pd_dp": "PD DP2", "pdaf": "PDAF TP1"}
COLORS_BASE = {"native_dp": "#3498db", "pd_dp": "#2ecc71", "pdaf": "#e74c3c"}
COLORS_TIER = {"native_dp": "#85c1e9", "pd_dp": "#82e0aa", "pdaf": "#f1948a"}
HATCHES = {"baseline": "", "tier": "//"}


# --- Figure: Baseline vs Tier energy per token (QPS=3) ---
fig, axes = plt.subplots(2, 2, figsize=(15, 11))
fig.suptitle("Qwen3-32B 4-GPU: Baseline vs DVFS-Tier Comparison (QPS=3)",
             fontsize=14, fontweight='bold')

metrics_cfg = [
    ("mj_tok", "Energy per Token (mJ/tok)", True),
    ("energy", "Total Energy (J)", True),
    ("tpot", "TPOT (ms)", True),
    ("ttft", "TTFT (ms)", True),
]

for idx, (metric, ylabel, lower_better) in enumerate(metrics_cfg):
    ax = axes[idx // 2, idx % 2]
    x = np.arange(len(SCENARIOS))
    width = 0.13
    offset = 0

    for deploy in DEPLOYS:
        key = lambda s: f"{s}_qps3"
        vals_b = [baseline[deploy][key(s)][metric] for s in SCENARIOS]
        vals_t = [tier[deploy][key(s)][metric] for s in SCENARIOS]

        bars_b = ax.bar(x + offset, vals_b, width, label=f"{DEPLOY_LABELS[deploy]} Base",
                        color=COLORS_BASE[deploy], edgecolor='black', linewidth=0.5)
        offset += width
        bars_t = ax.bar(x + offset, vals_t, width, label=f"{DEPLOY_LABELS[deploy]} Tier",
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
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_title("Qwen3-32B 4-GPU: Energy Saving (Tier vs Baseline) by Scenario & QPS",
             fontsize=13, fontweight='bold')

x = np.arange(len(SCENARIOS) * len(QPS_LIST))
labels = [f"{s[:4]}\nQ{q}" for s in SCENARIOS for q in QPS_LIST]
width = 0.25

for i, deploy in enumerate(DEPLOYS):
    savings = []
    for s in SCENARIOS:
        for q in QPS_LIST:
            key = f"{s}_qps{q}"
            b = baseline[deploy][key]["mj_tok"]
            t = tier[deploy][key]["mj_tok"]
            pct = (b - t) / b * 100
            savings.append(pct)
    ax.bar(x + i * width, savings, width, label=DEPLOY_LABELS[deploy],
           color=COLORS_BASE[deploy], edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(labels, fontsize=7)
ax.set_ylabel("Energy Saving (%)")
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "energy_saving_pct.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 3: Per-architecture line plot (Energy per token vs QPS) ---
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Qwen3-32B 4-GPU: Energy/Token - Baseline vs Tier by QPS",
             fontsize=13, fontweight='bold')

for i, deploy in enumerate(DEPLOYS):
    ax = axes[i]
    for scenario in SCENARIOS:
        vals_b = [baseline[deploy][f"{scenario}_qps{q}"]["mj_tok"] for q in QPS_LIST]
        vals_t = [tier[deploy][f"{scenario}_qps{q}"]["mj_tok"] for q in QPS_LIST]
        ax.plot(QPS_LIST, vals_b, 'o-', label=f"{scenario} Base", linewidth=2, markersize=7)
        ax.plot(QPS_LIST, vals_t, 's--', label=f"{scenario} Tier", linewidth=1.5, markersize=6, alpha=0.7)
    ax.set_xlabel("QPS")
    ax.set_ylabel("mJ/tok")
    ax.set_title(DEPLOY_LABELS[deploy])
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax.set_xticks(QPS_LIST)

plt.tight_layout()
out = CHARTS_DIR / "energy_per_tok_by_qps.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

print("\nDone! All charts saved to:", CHARTS_DIR)
