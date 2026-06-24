#!/usr/bin/env python3
"""Compare old machine vs new machine baseline results (QPS=3)."""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

SCENARIOS = ["chatbot", "qa", "rag", "summary"]
DEPLOYS = ["native_dp", "pd_dp", "pdaf"]
DEPLOY_LABELS = {"native_dp": "Native DP4", "pd_dp": "PD DP2", "pdaf": "PDAF TP1"}

# Old machine data (read from micro-test-old.png image)
# QPS=3 data from old machine charts:
# Panel 1 (Throughput): Native~44, PD~43, PDAF~44 for chatbot
#   qa: Native~659, PD~656, PDAF~643
#   rag: Native~184, PD~183, PDAF~183
#   summary: Native~155, PD~166, PDAF~150
# Panel 2 (TPOT): Native~43, PD~46, PDAF~47
#   qa: Native~47, PD~47, PDAF~57
#   rag: Native~46, PD~46, PDAF~51
#   summary: Native~45, PD~47, PDAF~51
# Panel 3 (Total Energy): Native~55000, PD~43000, PDAF~40000
#   qa: Native~100000, PD~65000, PDAF~53000
#   rag: Native~90000, PD~59000, PDAF~49000
#   summary: Native~83000, PD~57000, PDAF~53000
# Panel 4 (mJ/tok): Native~18400, PD~14600, PDAF~13600
#   qa: Native~1960, PD~1280, PDAF~1040
#   rag: Native~7000, PD~4600, PDAF~3800
#   summary: Native~7900, PD~4900, PDAF~5000

old_machine = {
    "native_dp": {
        "chatbot_qps3": {"thpt": 44.4, "tpot": 43.4, "energy": 55000, "mj_tok": 18400, "ttft": 86.0},
        "qa_qps3": {"thpt": 659.0, "tpot": 47.0, "energy": 100000, "mj_tok": 1960, "ttft": 88.0},
        "rag_qps3": {"thpt": 184.0, "tpot": 46.0, "energy": 90000, "mj_tok": 7000, "ttft": 87.0},
        "summary_qps3": {"thpt": 155.0, "tpot": 45.0, "energy": 83000, "mj_tok": 7900, "ttft": 87.0},
    },
    "pd_dp": {
        "chatbot_qps3": {"thpt": 43.6, "tpot": 46.0, "energy": 43000, "mj_tok": 14600, "ttft": 47.0},
        "qa_qps3": {"thpt": 656.0, "tpot": 47.0, "energy": 65000, "mj_tok": 1280, "ttft": 46.0},
        "rag_qps3": {"thpt": 183.0, "tpot": 46.0, "energy": 59000, "mj_tok": 4600, "ttft": 47.0},
        "summary_qps3": {"thpt": 166.0, "tpot": 47.0, "energy": 57000, "mj_tok": 4900, "ttft": 48.0},
    },
    "pdaf": {
        "chatbot_qps3": {"thpt": 44.0, "tpot": 47.0, "energy": 40000, "mj_tok": 13600, "ttft": 48.0},
        "qa_qps3": {"thpt": 643.0, "tpot": 57.0, "energy": 53000, "mj_tok": 1040, "ttft": 51.0},
        "rag_qps3": {"thpt": 183.0, "tpot": 51.0, "energy": 49000, "mj_tok": 3800, "ttft": 82.0},
        "summary_qps3": {"thpt": 150.0, "tpot": 51.0, "energy": 53000, "mj_tok": 5000, "ttft": 150.0},
    },
}

# New machine data (from full_baseline_v2.log)
new_machine = {
    "native_dp": {
        "chatbot_qps3": {"thpt": 44.3, "tpot": 44.2, "energy": 58068, "mj_tok": 19433.6, "ttft": 88.5},
        "qa_qps3": {"thpt": 652.7, "tpot": 48.4, "energy": 105161, "mj_tok": 2053.9, "ttft": 91.4},
        "rag_qps3": {"thpt": 184.1, "tpot": 46.8, "energy": 93663, "mj_tok": 7317.4, "ttft": 90.5},
        "summary_qps3": {"thpt": 156.1, "tpot": 46.4, "energy": 87130, "mj_tok": 8258.8, "ttft": 90.3},
    },
    "pd_dp": {
        "chatbot_qps3": {"thpt": 43.5, "tpot": 47.9, "energy": 46749, "mj_tok": 15944.3, "ttft": 48.4},
        "qa_qps3": {"thpt": 649.9, "tpot": 47.8, "energy": 68636, "mj_tok": 1340.5, "ttft": 47.1},
        "rag_qps3": {"thpt": 183.9, "tpot": 47.3, "energy": 62069, "mj_tok": 4849.1, "ttft": 48.8},
        "summary_qps3": {"thpt": 166.2, "tpot": 47.7, "energy": 60989, "mj_tok": 5264.4, "ttft": 49.4},
    },
    "pdaf": {
        "chatbot_qps3": {"thpt": 43.9, "tpot": 48.4, "energy": 42818, "mj_tok": 14494.8, "ttft": 49.3},
        "qa_qps3": {"thpt": 637.1, "tpot": 58.7, "energy": 55476, "mj_tok": 1083.5, "ttft": 52.3},
        "rag_qps3": {"thpt": 183.3, "tpot": 52.2, "energy": 51524, "mj_tok": 4025.3, "ttft": 84.5},
        "summary_qps3": {"thpt": 150.8, "tpot": 52.2, "energy": 56278, "mj_tok": 5334.4, "ttft": 187.7},
    },
}

COLORS = {"native_dp": "#3498db", "pd_dp": "#2ecc71", "pdaf": "#e74c3c"}

# --- Figure: Old vs New Machine (QPS=3) ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("Qwen3-32B 4-GPU Baseline (QPS=3): Old Machine vs New Machine",
             fontsize=14, fontweight='bold')

metrics_cfg = [
    ("thpt", "Throughput (tok/s)", False),
    ("tpot", "TPOT (ms)", True),
    ("ttft", "TTFT (ms)", True),
    ("energy", "Total Energy (J)", True),
    ("mj_tok", "Energy/Token (mJ/tok)", True),
]

for idx, (metric, ylabel, lower_better) in enumerate(metrics_cfg):
    ax = axes[idx // 3, idx % 3]
    x = np.arange(len(SCENARIOS))
    width = 0.12

    for i, deploy in enumerate(DEPLOYS):
        vals_old = [old_machine[deploy][f"{s}_qps3"][metric] for s in SCENARIOS]
        vals_new = [new_machine[deploy][f"{s}_qps3"][metric] for s in SCENARIOS]

        offset_old = i * (2 * width + 0.04)
        offset_new = offset_old + width

        bars_old = ax.bar(x + offset_old, vals_old, width,
                          label=f"{DEPLOY_LABELS[deploy]} Old" if idx == 0 else "",
                          color=COLORS[deploy], alpha=0.6, edgecolor='black', linewidth=0.5)
        bars_new = ax.bar(x + offset_new, vals_new, width,
                          label=f"{DEPLOY_LABELS[deploy]} New" if idx == 0 else "",
                          color=COLORS[deploy], alpha=1.0, edgecolor='black', linewidth=0.5,
                          hatch='//')

    ax.set_xticks(x + width * 3)
    ax.set_xticklabels([s.capitalize() for s in SCENARIOS])
    ax.set_ylabel(ylabel)
    ax.set_title(ylabel)
    ax.grid(axis='y', alpha=0.3)

# Legend in the last (empty) panel
axes[1, 2].axis('off')
handles = []
for deploy in DEPLOYS:
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=COLORS[deploy], alpha=0.6, ec='black', lw=0.5))
    handles.append(plt.Rectangle((0, 0), 1, 1, fc=COLORS[deploy], alpha=1.0, ec='black', lw=0.5, hatch='//'))
labels = []
for deploy in DEPLOYS:
    labels.extend([f"{DEPLOY_LABELS[deploy]} (Old)", f"{DEPLOY_LABELS[deploy]} (New)"])
axes[1, 2].legend(handles, labels, loc='center', fontsize=11, frameon=True)
axes[1, 2].set_title("Legend", fontsize=12)

plt.tight_layout()
out = CHARTS_DIR / "old_vs_new_machine.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 2: Percentage difference (new vs old) ---
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_title("New Machine vs Old Machine: Performance Difference (%) at QPS=3\n"
             "(Positive = New is higher/worse for latency/energy, better for throughput)",
             fontsize=12, fontweight='bold')

metrics_pct = ["thpt", "tpot", "energy", "mj_tok"]
metric_labels = ["Throughput", "TPOT", "Total Energy", "mJ/tok"]
x = np.arange(len(SCENARIOS) * len(metrics_pct))
labels_x = []
for s in SCENARIOS:
    for m in metric_labels:
        labels_x.append(f"{s[:4]}\n{m[:5]}")

width = 0.25
for i, deploy in enumerate(DEPLOYS):
    diffs = []
    for s in SCENARIOS:
        for m in metrics_pct:
            old_v = old_machine[deploy][f"{s}_qps3"][m]
            new_v = new_machine[deploy][f"{s}_qps3"][m]
            pct = (new_v - old_v) / old_v * 100
            diffs.append(pct)
    ax.bar(x + i * width, diffs, width, label=DEPLOY_LABELS[deploy],
           color=COLORS[deploy], edgecolor='black', linewidth=0.5)

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_xticks(x + width)
ax.set_xticklabels(labels_x, fontsize=6.5, rotation=0)
ax.set_ylabel("Difference (%)")
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "old_vs_new_pct_diff.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# Print text summary
print("\n=== Old vs New Machine Summary (QPS=3) ===")
print(f"{'Deploy':<12} {'Scenario':<10} {'Metric':<8} {'Old':>8} {'New':>8} {'Diff%':>7}")
print("-" * 60)
for deploy in DEPLOYS:
    for s in SCENARIOS:
        for m in ["thpt", "tpot", "energy", "mj_tok"]:
            old_v = old_machine[deploy][f"{s}_qps3"][m]
            new_v = new_machine[deploy][f"{s}_qps3"][m]
            pct = (new_v - old_v) / old_v * 100
            print(f"{DEPLOY_LABELS[deploy]:<12} {s:<10} {m:<8} {old_v:>8.1f} {new_v:>8.1f} {pct:>+6.1f}%")
    print()

print("\nDone!")
