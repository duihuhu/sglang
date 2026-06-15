#!/usr/bin/env python3
"""Generate 6gpu_azure_simplified_comparison.png — similar to the 8GPU chart."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "results_6gpu_azure" / "json"
OUT_DIR = HERE / "charts_6gpu_azure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Load summary
summary = json.loads((JSON_DIR / "6gpu_summary.json").read_text())

DATASETS = ['Code Medium', 'Conv Light', 'Conv Medium', 'Conv Heavy']

# Schemes: baseline + tier pairs (only fully successful ones)
schemes_baseline = ['native_dp6', 'pd_dp3', 'pdaf_6g_1p2d', 'pdaf_6g_2p1d']
schemes_tier = ['native_dp6_tier', 'pd_dp3_tier', 'pdaf_6g_1p2d_tier', 'pdaf_6g_2p1d_tier']
scheme_labels = ['Native DP6', 'PD DP3', 'PDAF\n1P+2D', 'PDAF\n2P+1D']

colors_base = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47']


def get_val(deploy, dataset, key, divisor=1.0):
    d = summary.get(deploy, {}).get(dataset, {})
    if not d or d.get('throughput_tok_s', 0) == 0:
        return None
    return d.get(key, 0) / divisor


# === Plot: 1 row, 3 cols (Energy, TPOT, SLO) ===
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("6-GPU Azure Workload — Deployment Comparison (TTFT SLO=5s, TPOT SLO=200ms)",
             fontsize=12, fontweight='bold')

bar_width = 0.07
n_datasets = len(DATASETS)
n_schemes = len(schemes_baseline)
group_width = n_schemes * 2 * bar_width + bar_width * 2

metrics = [
    ('total_energy_j', 'Total Energy (kJ)', 'Total Energy', 1000.0, None),
    ('tpot_avg_ms', 'TPOT Avg (ms)', 'TPOT (Avg)', 1.0, 200),
    ('slo_violation_rate', 'SLO Violation (%)', 'SLO Violation', 1.0, None),
]

for mi, (key, ylabel, title, divisor, slo_line) in enumerate(metrics):
    ax = axes[mi]
    for di, ds_label in enumerate(DATASETS):
        x_base = di * group_width
        for si in range(n_schemes):
            x_pos = x_base + si * 2 * bar_width
            # Baseline (solid)
            val_base = get_val(schemes_baseline[si], ds_label, key, divisor)
            if val_base is not None:
                ax.bar(x_pos, val_base, bar_width,
                       color=colors_base[si], edgecolor='black', linewidth=0.4)
            # Tier (hatched)
            val_tier = get_val(schemes_tier[si], ds_label, key, divisor)
            if val_tier is not None:
                ax.bar(x_pos + bar_width, val_tier, bar_width,
                       color=colors_base[si], edgecolor='black', linewidth=0.4, hatch='//')

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ticks = [di * group_width + (n_schemes - 0.5) * bar_width for di in range(n_datasets)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(DATASETS, fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    if slo_line:
        ax.axhline(y=slo_line, color='red', linestyle='--', linewidth=1.2,
                   alpha=0.7, label=f'SLO={slo_line}ms')
        ax.legend(loc='upper left', fontsize=8)

# === Legend ===
legend_elements = []
for si, label in enumerate(scheme_labels):
    legend_elements.append(Patch(facecolor=colors_base[si], edgecolor='black',
                                 linewidth=0.5, label=label))
    legend_elements.append(Patch(facecolor=colors_base[si], edgecolor='black',
                                 linewidth=0.5, hatch='//', label=f'{label} +Tier'))

fig.legend(handles=legend_elements, loc='lower center', ncol=4,
           fontsize=8.5, bbox_to_anchor=(0.5, -0.04), frameon=True)

plt.tight_layout(rect=[0, 0.08, 1, 0.94])
save_path = OUT_DIR / "6gpu_azure_simplified_comparison.png"
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

# === Energy savings table ===
print("\n=== Energy Savings (Tier vs No-Tier) ===")
for ds_label in DATASETS:
    print(f"\n{ds_label}:")
    for si in range(n_schemes):
        e_b = get_val(schemes_baseline[si], ds_label, 'total_energy_j')
        e_t = get_val(schemes_tier[si], ds_label, 'total_energy_j')
        if e_b and e_t and e_b > 0:
            save = (1 - e_t / e_b) * 100
            slo_b = get_val(schemes_baseline[si], ds_label, 'slo_violation_rate') or 0
            slo_t = get_val(schemes_tier[si], ds_label, 'slo_violation_rate') or 0
            print(f"  {scheme_labels[si].replace(chr(10),' '):18s}: "
                  f"{e_b/1000:.0f}kJ -> {e_t/1000:.0f}kJ ({save:+.1f}%) | "
                  f"SLO: {slo_b:.1f}% -> {slo_t:.1f}%")
        else:
            print(f"  {scheme_labels[si].replace(chr(10),' '):18s}: N/A (crash)")

# === Throughput comparison ===
print("\n=== Throughput Comparison (tok/s) ===")
print(f"{'Deploy':<25} {'Code Med':>10} {'Conv Lt':>10} {'Conv Med':>10} {'Conv Hvy':>10}")
for si in range(n_schemes):
    for deploy in [schemes_baseline[si], schemes_tier[si]]:
        label = deploy
        vals = []
        for ds in DATASETS:
            v = get_val(deploy, ds, 'throughput_tok_s')
            vals.append(f"{v:.0f}" if v else "N/A")
        print(f"  {label:<23} {vals[0]:>10} {vals[1]:>10} {vals[2]:>10} {vals[3]:>10}")
