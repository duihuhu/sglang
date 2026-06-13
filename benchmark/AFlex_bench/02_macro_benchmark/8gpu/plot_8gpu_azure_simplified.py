#!/usr/bin/env python3
"""Generate 8gpu_azure_simplified_comparison.png — original format (1x3 subplots)."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "results_8gpu_azure_slo2" / "json"
OUT_DIR = HERE / "charts_8gpu_azure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Workloads as x-axis groups (only complete datasets)
DATASETS = {
    'Code\nMedium': 'azure_code_medium_real',
    'Conv\nLight': 'azure_conv_light_real',
    'Conv\nMedium': 'azure_conv_medium_real',
    'Conv\nHeavy': 'azure_conv_heavy_real',
}

# Schemes: baseline + tier pairs
schemes_baseline = ['native_dp8', 'pd_dp4', 'pdaf_8g_dyn', 'pdaf_8g_asym_1p6d']
schemes_tier = ['native_dp8_tier', 'pd_dp4_tier', 'pdaf_8g_dyn_tier', 'pdaf_8g_asym_1p6d_tier']
scheme_labels = ['Native DP8', 'PD DP4', 'PDAF Sym', 'PDAF Asym\n(1P+6D)']

colors_base = ['#4472C4', '#ED7D31', '#A5A5A5', '#70AD47']

# Load all data
results = {}
for ds_label, ds_key in DATASETS.items():
    results[ds_label] = {}
    for s in schemes_baseline + schemes_tier:
        fname = JSON_DIR / f"{s}_var_{ds_key}_results.json"
        if fname.exists():
            with open(fname) as f:
                results[ds_label][s] = json.load(f)

# === Plot: 1 row, 2 cols ===
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("8-GPU Azure Workload — Tier DVFS Comparison (TTFT SLO=4s, TPOT SLO=200ms)",
             fontsize=12, fontweight='bold')

bar_width = 0.08
ds_labels = list(DATASETS.keys())
n_datasets = len(ds_labels)
n_schemes = len(schemes_baseline)
group_width = n_schemes * 2 * bar_width + bar_width * 2

metrics = [
    ('total_energy_j', 'Total Energy (kJ)', 'Total Energy', 1000.0, None),
    ('tpot_avg_ms', 'TPOT Avg (ms)', 'TPOT (Avg)', 1.0, 200),
]

for mi, (key, ylabel, title, divisor, slo_line) in enumerate(metrics):
    ax = axes[mi]
    for di, ds_label in enumerate(ds_labels):
        x_base = di * group_width
        for si in range(n_schemes):
            x_pos = x_base + si * 2 * bar_width
            # Baseline (solid)
            d_base = results[ds_label].get(schemes_baseline[si], {})
            val_base = d_base.get(key, 0) / divisor
            if d_base.get('throughput_tok_s', 0) == 0 and key != 'total_energy_j':
                val_base = 0
            ax.bar(x_pos, val_base, bar_width,
                   color=colors_base[si], edgecolor='black', linewidth=0.4)
            # Tier (hatched)
            d_tier = results[ds_label].get(schemes_tier[si], {})
            val_tier = d_tier.get(key, 0) / divisor
            if d_tier.get('throughput_tok_s', 0) == 0 and key != 'total_energy_j':
                val_tier = 0
            ax.bar(x_pos + bar_width, val_tier, bar_width,
                   color=colors_base[si], edgecolor='black', linewidth=0.4, hatch='//')

    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ticks = [di * group_width + (n_schemes - 0.5) * bar_width for di in range(n_datasets)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ds_labels, fontsize=7.5)
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
save_path = OUT_DIR / "8gpu_azure_simplified_comparison.png"
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

# Print energy savings summary
print("\n=== Energy Savings (Tier vs No-Tier) ===")
for ds_label in ds_labels:
    print(f"\n{ds_label}:")
    for si in range(n_schemes):
        b = results[ds_label].get(schemes_baseline[si], {})
        t = results[ds_label].get(schemes_tier[si], {})
        e_b = b.get('total_energy_j', 0)
        e_t = t.get('total_energy_j', 0)
        if e_b > 0 and e_t > 0:
            save = (1 - e_t / e_b) * 100
            slo_b = b.get('slo_violation_rate', 0)
            slo_t = t.get('slo_violation_rate', 0)
            print(f"  {scheme_labels[si]:18s}: {e_b/1000:.0f}kJ -> {e_t/1000:.0f}kJ"
                  f" ({save:+.1f}%) | SLO: {slo_b:.1f}% -> {slo_t:.1f}%")
