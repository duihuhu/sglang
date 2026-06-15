#!/usr/bin/env python3
"""Compare prefill-heavy hetero PDAF (P6+D2) vs existing layouts on Azure workloads.

New schemes (this experiment):
  - pdaf_8g_2pa4pf : Prefill PF-TP4+PA-TP2 (6 GPU) + Decode TP1+TP1 (2 GPU)
  - pdaf_8g_4pa2pf : Prefill PF-TP2+PA-TP4 (6 GPU) + Decode TP1+TP1 (2 GPU)
Reference schemes (already in repo):
  - pdaf_8g_asym_1p6d : Prefill 2 GPU + Decode 6 GPU
  - pdaf_8g_dyn       : PDAF Symmetric, Prefill 4 GPU + Decode 4 GPU
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "results_8gpu_azure_slo2" / "json"
OUT_DIR = HERE / "charts_8gpu_azure"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "Code\nMedium": "azure_code_medium_real",
    "Conv\nLight": "azure_conv_light_real",
    "Conv\nMedium": "azure_conv_medium_real",
    "Conv\nHeavy": "azure_conv_heavy_real",
}

schemes_baseline = ["pdaf_8g_2pa4pf", "pdaf_8g_4pa2pf",
                    "pdaf_8g_asym_1p6d", "pdaf_8g_dyn"]
schemes_tier = [s + "_tier" for s in schemes_baseline]
scheme_labels = ["2PA4PF\n(P6+D2)", "4PA2PF\n(P6+D2)",
                 "Asym\n(P2+D6)", "Sym\n(P4+D4)"]
colors_base = ["#C0504D", "#E59C00", "#4472C4", "#70AD47"]


def load(scheme, ds_key):
    f = JSON_DIR / f"{scheme}_var_{ds_key}_results.json"
    if f.exists():
        try:
            return json.load(open(f))
        except Exception:
            return {}
    return {}


results = {}
for ds_label, ds_key in DATASETS.items():
    results[ds_label] = {}
    for s in schemes_baseline + schemes_tier:
        results[ds_label][s] = load(s, ds_key)

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle("8-GPU Azure — Prefill-Heavy (P6+D2) vs Existing Layouts "
             "(TTFT SLO=4s, TPOT SLO=200ms)", fontsize=13, fontweight="bold")

bar_width = 0.09
ds_labels = list(DATASETS.keys())
n_datasets = len(ds_labels)
n_schemes = len(schemes_baseline)
group_width = n_schemes * 2 * bar_width + bar_width * 2

metrics = [
    ("throughput_tok_s", "Throughput (tok/s)", "Throughput", 1.0, None),
    ("total_energy_j", "Total Energy (kJ)", "Total Energy", 1000.0, None),
    ("ttft_avg_ms", "TTFT Avg (s, log)", "TTFT (queueing)", 1000.0, 4.0),
]

for mi, (key, ylabel, title, divisor, slo_line) in enumerate(metrics):
    ax = axes[mi]
    for di, ds_label in enumerate(ds_labels):
        x_base = di * group_width
        for si in range(n_schemes):
            x_pos = x_base + si * 2 * bar_width
            d_base = results[ds_label].get(schemes_baseline[si], {})
            val_base = d_base.get(key, 0) / divisor
            ax.bar(x_pos, val_base, bar_width,
                   color=colors_base[si], edgecolor="black", linewidth=0.4)
            d_tier = results[ds_label].get(schemes_tier[si], {})
            val_tier = d_tier.get(key, 0) / divisor
            ax.bar(x_pos + bar_width, val_tier, bar_width,
                   color=colors_base[si], edgecolor="black", linewidth=0.4,
                   hatch="//")
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ticks = [di * group_width + (n_schemes - 0.5) * bar_width
             for di in range(n_datasets)]
    ax.set_xticks(ticks)
    ax.set_xticklabels(ds_labels, fontsize=8.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    if key == "ttft_avg_ms":
        ax.set_yscale("log")
    if slo_line:
        ax.axhline(y=slo_line, color="red", linestyle="--", linewidth=1.2,
                   alpha=0.7, label=f"SLO={slo_line:.0f}s")
        ax.legend(loc="upper left", fontsize=8)

legend_elements = []
for si, label in enumerate(scheme_labels):
    legend_elements.append(Patch(facecolor=colors_base[si], edgecolor="black",
                                 linewidth=0.5, label=label.replace("\n", " ")))
    legend_elements.append(Patch(facecolor=colors_base[si], edgecolor="black",
                                 linewidth=0.5, hatch="//",
                                 label=label.replace("\n", " ") + " +Tier"))
fig.legend(handles=legend_elements, loc="lower center", ncol=4,
           fontsize=9, bbox_to_anchor=(0.5, -0.06), frameon=True)

plt.tight_layout(rect=[0, 0.07, 1, 0.94])
save_path = OUT_DIR / "8gpu_azure_prefill_heavy_comparison.png"
fig.savefig(save_path, dpi=150, bbox_inches="tight")
print(f"Saved: {save_path}")

print("\n=== Tier energy savings (vs own baseline) ===")
for ds_label in ds_labels:
    print(f"\n{ds_label.replace(chr(10),' ')}:")
    for si in range(n_schemes):
        b = results[ds_label].get(schemes_baseline[si], {})
        t = results[ds_label].get(schemes_tier[si], {})
        eb, et = b.get("total_energy_j", 0), t.get("total_energy_j", 0)
        if eb and et:
            print(f"  {scheme_labels[si].replace(chr(10),' '):14s}: "
                  f"{eb/1000:.0f}kJ -> {et/1000:.0f}kJ ({(1-et/eb)*100:+.1f}%) | "
                  f"thpt {b.get('throughput_tok_s',0):.0f}->{t.get('throughput_tok_s',0):.0f} "
                  f"| SLO {b.get('slo_violation_rate',0):.0f}%->{t.get('slo_violation_rate',0):.0f}%")
