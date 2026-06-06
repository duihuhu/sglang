#!/usr/bin/env python3
"""Visualize TTFT SLO sweep results: TTFT processing time heatmap/chart."""
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent / "results_ttft_sweep"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

summary_path = RESULTS_DIR / "ttft_sweep_summary.json"
with open(summary_path) as f:
    data = json.load(f)

TTFT_SLOS = [5000, 2000, 1000, 500, 300, 200]

bl_proc = []
v2_proc = []
v2_total = []
bl_total = []
v2_slo_viol = []
v2_saving = []

for ttft_slo in TTFT_SLOS:
    bl = data.get(f"baseline_ttft{ttft_slo}", {})
    v2 = data.get(f"v2_ttft{ttft_slo}", {})

    bl_proc.append(bl.get("ttft_proc_avg_ms", 0))
    bl_total.append(bl.get("ttft_avg_ms", 0))
    v2_proc.append(v2.get("ttft_proc_avg_ms", 0))
    v2_total.append(v2.get("ttft_avg_ms", 0))
    v2_slo_viol.append(v2.get("slo_violation_rate", 0))

    e_bl = bl.get("total_energy_j", 1)
    e_v2 = v2.get("total_energy_j", 0)
    saving = (1 - e_v2 / e_bl) * 100 if e_bl > 0 else 0
    v2_saving.append(saving)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("TTFT SLO Sweep: Processing Time (Excluding Queuing)\n"
             "PDAF DynM, 4-GPU, workload_steady, TPOT SLO=150ms",
             fontsize=12, fontweight='bold')

x = np.arange(len(TTFT_SLOS))
labels = [str(s) for s in TTFT_SLOS]

# Panel 1: TTFT Processing Time comparison
ax = axes[0, 0]
w = 0.35
bars1 = ax.bar(x - w/2, bl_proc, w, label='Baseline', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x + w/2, v2_proc, w, label='V2 (DVFS)', color='#FF9800', alpha=0.8)
ax.set_xlabel('TTFT SLO (ms)')
ax.set_ylabel('TTFT Processing Time (ms)')
ax.set_title('TTFT Processing Time (Excl. Queuing)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.axhline(y=115, color='#2196F3', linestyle='--', alpha=0.3)
for bar in bars1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)
for bar in bars2:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{bar.get_height():.0f}', ha='center', va='bottom', fontsize=8)

# Panel 2: SLO Violation Rate
ax = axes[0, 1]
colors = ['#4CAF50' if v < 5 else '#FF9800' if v < 50 else '#F44336' for v in v2_slo_viol]
bars = ax.bar(x, v2_slo_viol, color=colors, alpha=0.8)
ax.set_xlabel('TTFT SLO (ms)')
ax.set_ylabel('SLO Violation Rate (%)')
ax.set_title('V2 SLO Violation Rate')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.axhline(y=5, color='red', linestyle='--', alpha=0.5, label='5% threshold')
ax.legend()
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

# Panel 3: Energy Saving
ax = axes[1, 0]
colors = ['#4CAF50' if s > 25 else '#FF9800' if s > 15 else '#F44336' for s in v2_saving]
bars = ax.bar(x, v2_saving, color=colors, alpha=0.8)
ax.set_xlabel('TTFT SLO (ms)')
ax.set_ylabel('Energy Saving vs Baseline (%)')
ax.set_title('V2 Energy Saving')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim(0, 35)
for bar in bars:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{bar.get_height():.1f}%', ha='center', va='bottom', fontsize=8)

# Panel 4: Heatmap - TTFT proc vs SLO threshold
ax = axes[1, 1]
heatmap_data = np.array([
    v2_proc,
    bl_proc,
    v2_slo_viol,
    v2_saving
])
row_labels = ['V2 TTFT proc (ms)', 'BL TTFT proc (ms)', 'V2 SLO Viol (%)', 'V2 Saving (%)']

im = ax.imshow(heatmap_data, aspect='auto', cmap='YlOrRd')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_yticks(range(4))
ax.set_yticklabels(row_labels)
ax.set_xlabel('TTFT SLO (ms)')
ax.set_title('Summary Heatmap')

for i in range(4):
    for j in range(len(TTFT_SLOS)):
        val = heatmap_data[i, j]
        fmt = f"{val:.0f}" if val >= 1 else f"{val:.1f}"
        ax.text(j, i, fmt, ha='center', va='center', fontsize=8,
                color='white' if val > np.nanmax(heatmap_data) * 0.6 else 'black')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

plt.tight_layout()
out_path = OUT_DIR / "ttft_slo_sweep_proc.png"
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path}")

# Also create a focused heatmap: TTFT processing vs TTFT SLO (what user asked)
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
fig.suptitle("TTFT SLO vs Actual TTFT Processing Time (Excl. Queuing)\n"
             "PDAF DynM 4-GPU | workload_steady | TPOT SLO=150ms",
             fontsize=11, fontweight='bold')

# Show as bar chart with SLO line
x_pos = np.arange(len(TTFT_SLOS))
w = 0.3
bars1 = ax.bar(x_pos - w/2, bl_proc, w, label='Baseline TTFT proc', color='#2196F3', alpha=0.8)
bars2 = ax.bar(x_pos + w/2, v2_proc, w, label='V2 TTFT proc', color='#FF9800', alpha=0.8)

# Draw SLO thresholds as horizontal markers
for i, slo in enumerate(TTFT_SLOS):
    if slo <= 600:  # only draw reasonable ones on this scale
        ax.plot([i-0.4, i+0.4], [slo, slo], 'r--', alpha=0.6, linewidth=1.5)
        ax.text(i+0.42, slo, f'SLO={slo}', fontsize=7, color='red', va='center')

ax.set_xlabel('TTFT SLO Setting (ms)')
ax.set_ylabel('Actual TTFT Processing Time (ms)')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.legend(loc='upper left')

# Annotate violation rates
for i, (proc, viol) in enumerate(zip(v2_proc, v2_slo_viol)):
    if viol > 0:
        ax.text(i + w/2, proc + 15, f'viol={viol:.1f}%', ha='center', fontsize=7, color='red')

# Add saving on secondary axis
ax2 = ax.twinx()
ax2.plot(x_pos, v2_saving, 's-', color='green', alpha=0.7, label='Energy Saving %')
ax2.set_ylabel('Energy Saving (%)', color='green')
ax2.set_ylim(0, 35)
ax2.legend(loc='upper right')

plt.tight_layout()
out_path2 = OUT_DIR / "ttft_slo_sweep_proc_focused.png"
plt.savefig(out_path2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out_path2}")
