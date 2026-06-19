#!/usr/bin/env python3
"""Generate 8gpu_azure_simplified_comparison_v2.png — without Code Medium & PDAF Asym."""
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

DATASETS = {
    'Conv\nLight': 'azure_conv_light_real',
    'Conv\nMedium': 'azure_conv_medium_real',
}

schemes_baseline = ['native_dp8', 'pd_dp4', 'pdaf_8g_dyn']
schemes_tier = ['native_dp8_tier', 'pd_dp4_tier', 'pdaf_8g_dyn_tier']
scheme_labels = ['Native DP8', 'PD DP4', 'PDAF Sym']

colors_base = ['#4472C4', '#ED7D31', '#A5A5A5']

results = {}
for ds_label, ds_key in DATASETS.items():
    results[ds_label] = {}
    for s in schemes_baseline + schemes_tier:
        fname = JSON_DIR / f"{s}_var_{ds_key}_results.json"
        if fname.exists():
            with open(fname) as f:
                results[ds_label][s] = json.load(f)

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle("8-GPU Azure Workload — Tier DVFS Comparison (TTFT SLO=4s, TPOT SLO=200ms)",
             fontsize=12, fontweight='bold')

bar_width = 0.10
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
            d_base = results[ds_label].get(schemes_baseline[si], {})
            val_base = d_base.get(key, 0) / divisor
            if d_base.get('throughput_tok_s', 0) == 0 and key != 'total_energy_j':
                val_base = 0
            ax.bar(x_pos, val_base, bar_width,
                   color=colors_base[si], edgecolor='black', linewidth=0.4)
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
    ax.set_xticklabels(ds_labels, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_axisbelow(True)
    if slo_line:
        ax.axhline(y=slo_line, color='red', linestyle='--', linewidth=1.2,
                   alpha=0.7, label=f'SLO={slo_line}ms')
        ax.legend(loc='upper left', fontsize=8)

legend_elements = []
for si, label in enumerate(scheme_labels):
    legend_elements.append(Patch(facecolor=colors_base[si], edgecolor='black',
                                 linewidth=0.5, label=label))
    legend_elements.append(Patch(facecolor=colors_base[si], edgecolor='black',
                                 linewidth=0.5, hatch='//', label=f'{label} +Tier'))

fig.legend(handles=legend_elements, loc='lower center', ncol=3,
           fontsize=9, bbox_to_anchor=(0.5, -0.04), frameon=True)

plt.tight_layout(rect=[0, 0.08, 1, 0.94])
save_path = OUT_DIR / "8gpu_azure_simplified_comparison_v2.png"
fig.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"Saved: {save_path}")

print("\n=== PDAF Sym+Tier vs Native+Tier Energy Savings ===")
print(f"{'Dataset':<14} {'Native+Tier(kJ)':>15} {'PDAF Sym+Tier(kJ)':>17} {'Savings':>10}")
print("-" * 60)
for ds_label in ds_labels:
    e_native_tier = results[ds_label].get('native_dp8_tier', {}).get('total_energy_j', 0)
    e_pdaf_tier = results[ds_label].get('pdaf_8g_dyn_tier', {}).get('total_energy_j', 0)
    if e_native_tier > 0 and e_pdaf_tier > 0:
        saving_pct = (1 - e_pdaf_tier / e_native_tier) * 100
        print(f"{ds_label.replace(chr(10), ' '):<14} {e_native_tier/1000:>15.1f} {e_pdaf_tier/1000:>17.1f} {saving_pct:>9.1f}%")
