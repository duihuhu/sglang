#!/usr/bin/env python3
"""Plot Dense 4GPU micro-benchmark comparison (no Tier)."""
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# Load latest results
result_files = sorted(RESULTS_DIR.glob("micro_4gpu_*.json"))
with open(result_files[-1]) as f:
    data = json.load(f)

DEPLOYS = ["pdaf", "native_dp", "pd_dp"]
DEPLOY_LABELS = {"pdaf": "PDAF TP1", "native_dp": "Native DP4", "pd_dp": "PD DP2"}
COLORS = {"pdaf": "#e74c3c", "native_dp": "#3498db", "pd_dp": "#2ecc71"}
SCENARIOS = ["chatbot", "qa", "rag", "summary"]
QPS_LIST = [1, 2, 3]

# Collect data per scenario
def get_metric(deploy, scenario, qps, metric):
    key = f"{scenario}_qps{qps}"
    entry = data.get(deploy, {}).get(key, {})
    if entry.get("status") != "PASS":
        return None
    return entry.get(metric)


# --- Figure 1: 4-panel per-scenario comparison (QPS=3, highest load) ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dense Qwen3-32B 4-GPU Micro-Benchmark (No Tier, Max Freq)\nQPS=3 Comparison", fontsize=14, fontweight='bold')

metrics = [
    ("throughput_tok_s", "Throughput (tok/s)", False),
    ("tpot_avg_ms", "TPOT (ms)", True),
    ("total_energy_j", "Total Energy (J)", True),
    ("energy_per_token_mj", "Energy per Token (mJ/tok)", True),
]

for idx, (metric, label, lower_better) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]
    x = np.arange(len(SCENARIOS))
    width = 0.25
    for i, deploy in enumerate(DEPLOYS):
        vals = [get_metric(deploy, s, 3, metric) or 0 for s in SCENARIOS]
        bars = ax.bar(x + i * width, vals, width, label=DEPLOY_LABELS[deploy], color=COLORS[deploy])
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f"{v:.0f}" if v > 10 else f"{v:.1f}",
                        ha='center', va='bottom', fontsize=7)
    ax.set_xticks(x + width)
    ax.set_xticklabels([s.capitalize() for s in SCENARIOS])
    ax.set_ylabel(label)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
out = CHARTS_DIR / "dense_4gpu_qps3_comparison.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 2: Energy per token across all QPS ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dense Qwen3-32B 4-GPU: Energy Efficiency by Scenario & QPS (No Tier)", fontsize=14, fontweight='bold')

for idx, scenario in enumerate(SCENARIOS):
    ax = axes[idx // 2, idx % 2]
    for deploy in DEPLOYS:
        vals = [get_metric(deploy, scenario, q, "energy_per_token_mj") or 0 for q in QPS_LIST]
        ax.plot(QPS_LIST, vals, 'o-', label=DEPLOY_LABELS[deploy], color=COLORS[deploy], linewidth=2, markersize=8)
    ax.set_xlabel("QPS")
    ax.set_ylabel("Energy per Token (mJ/tok)")
    ax.set_title(f"{scenario.capitalize()}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(QPS_LIST)

plt.tight_layout()
out = CHARTS_DIR / "dense_4gpu_energy_by_qps.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 3: TPOT comparison across all QPS ---
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("Dense Qwen3-32B 4-GPU: TPOT by Scenario & QPS (No Tier)", fontsize=14, fontweight='bold')

for idx, scenario in enumerate(SCENARIOS):
    ax = axes[idx // 2, idx % 2]
    for deploy in DEPLOYS:
        vals = [get_metric(deploy, scenario, q, "tpot_avg_ms") or 0 for q in QPS_LIST]
        ax.plot(QPS_LIST, vals, 'o-', label=DEPLOY_LABELS[deploy], color=COLORS[deploy], linewidth=2, markersize=8)
    ax.set_xlabel("QPS")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title(f"{scenario.capitalize()}")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_xticks(QPS_LIST)

plt.tight_layout()
out = CHARTS_DIR / "dense_4gpu_tpot_by_qps.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")


# --- Figure 4: Summary table ---
fig, ax = plt.subplots(figsize=(16, 8))
ax.axis('off')
ax.set_title("Dense Qwen3-32B 4-GPU Micro-Benchmark Summary (No Tier, Max Freq)", fontsize=13, fontweight='bold', pad=20)

headers = ["Deploy", "Scenario", "QPS", "Thpt\n(tok/s)", "TTFT\n(ms)", "TPOT\n(ms)", "Energy\n(J)", "mJ/tok", "SLO%"]
rows = []
for deploy in DEPLOYS:
    for scenario in SCENARIOS:
        for qps in QPS_LIST:
            key = f"{scenario}_qps{qps}"
            entry = data.get(deploy, {}).get(key, {})
            if entry.get("status") != "PASS":
                continue
            rows.append([
                DEPLOY_LABELS[deploy], scenario.capitalize(), str(qps),
                f"{entry['throughput_tok_s']:.1f}",
                f"{entry['ttft_proc_avg_ms']:.1f}",
                f"{entry['tpot_avg_ms']:.1f}",
                f"{entry['total_energy_j']:.0f}",
                f"{entry['energy_per_token_mj']:.1f}",
                f"{entry['slo_violation_rate']:.1f}"
            ])

table = ax.table(cellText=rows, colLabels=headers, loc='center', cellLoc='center')
table.auto_set_font_size(False)
table.set_fontsize(7)
table.scale(1.0, 1.2)

for i, key in enumerate(table._cells):
    cell = table._cells[key]
    if key[0] == 0:
        cell.set_facecolor('#2c3e50')
        cell.set_text_props(color='white', fontweight='bold')
    elif key[0] % 2 == 0:
        cell.set_facecolor('#ecf0f1')

plt.tight_layout()
out = CHARTS_DIR / "dense_4gpu_summary_table.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

print("\nDone! All charts saved to:", CHARTS_DIR)
