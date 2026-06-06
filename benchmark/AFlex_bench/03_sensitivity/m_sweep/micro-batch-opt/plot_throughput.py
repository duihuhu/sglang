#!/usr/bin/env python3
"""Plot throughput vs concurrency for all architectures, combining old + new data."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PNG = os.path.join(HERE, "results", "plot_throughput.png")

# Old data from all_concurrency_results.csv (in=32, out=32 rows)
old_data = {
    "PD_TP2": [(256, 1704.9), (512, 3212.5), (768, 3704.0), (1024, 3506.5), (1536, 3460.2), (2048, 3504.6)],
    "AF_M2":  [(256, 1560.4), (512, 2507.1), (1024, 2953.4), (1536, 3182.4), (2048, 3280.9)],
    "AF_M1":  [(256, 1522.7), (512, 1830.6), (768, 1882.5), (1024, 1944.7)],
    "PD_TP1": [(256, 734.1), (512, 716.1), (768, 700.6), (1024, 710.7), (1536, 712.8), (2048, 706.7)],
}

# New data from high_conc_v2 run (in=32, out=32, higher concurrency)
new_data = {
    "PD_TP2": [(2048, 3516.8), (3072, 3592.3), (4096, 3469.8), (5120, 3498.6), (6144, 3437.5)],
    "AF_M2":  [(2048, 3276.8), (3072, 3315.5), (4096, 3389.1), (5120, 3435.2), (6144, 3436.9)],
}

# Decode batch stats from v2 (for annotation)
decode_batch = {
    "PD_TP2": {2048: 538, 3072: 539, 4096: 539, 5120: 539, 6144: 538},
    "AF_M2":  {2048: 495, 3072: 495, 4096: 743, 5120: 595, 6144: 850},
}

# Merge old + new (deduplicate by taking new value for same concurrency)
def merge(old, new):
    d = dict(old)
    for c, v in new:
        d[c] = v
    return sorted(d.items())

merged = {}
for k in ["PD_TP2", "AF_M2", "AF_M1", "PD_TP1"]:
    merged[k] = merge(old_data.get(k, []), new_data.get(k, []))

# Plot
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

styles = {
    "PD_TP2": {"color": "#2196F3", "marker": "o", "linewidth": 2.5, "label": "PD TP=2 (4 GPU)"},
    "AF_M2":  {"color": "#FF5722", "marker": "s", "linewidth": 2.5, "label": "AF M=2 async (4 GPU)"},
    "AF_M1":  {"color": "#4CAF50", "marker": "^", "linewidth": 2.0, "label": "AF M=1 (4 GPU)"},
    "PD_TP1": {"color": "#9C27B0", "marker": "D", "linewidth": 2.0, "label": "PD TP=1 (2 GPU)"},
}

for config, points in merged.items():
    concs = [p[0] for p in points]
    thrus = [p[1] for p in points]
    s = styles[config]
    ax.plot(concs, thrus, marker=s["marker"], color=s["color"],
            linewidth=s["linewidth"], markersize=7, label=s["label"])

# Mark the new v2 region
ax.axvline(x=2048, color="gray", linestyle="--", alpha=0.4, linewidth=1)
ax.text(2200, 500, "← prev data | new v2 →", fontsize=9, color="gray", alpha=0.7)

# Annotate decode batch for key points
for config in ["PD_TP2", "AF_M2"]:
    if config in decode_batch:
        for conc, db_max in decode_batch[config].items():
            points_dict = dict(merged[config])
            if conc in points_dict and conc >= 3072:
                thru = points_dict[conc]
                offset_y = 120 if config == "PD_TP2" else -180
                ax.annotate(f"D_bs={db_max}", xy=(conc, thru),
                           xytext=(conc, thru + offset_y),
                           fontsize=7.5, color=styles[config]["color"], alpha=0.8,
                           ha="center")

ax.set_xlabel("Concurrency", fontsize=12)
ax.set_ylabel("Output Throughput (tok/s)", fontsize=12)
ax.set_title("Throughput vs Concurrency (Qwen3-32B, in=32 out=32, A800×4)\nPD vs AF Disaggregation", fontsize=13)
ax.legend(loc="lower right", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 6500)
ax.set_ylim(0, 4200)

xticks = [256, 512, 1024, 1536, 2048, 3072, 4096, 5120, 6144]
ax.set_xticks(xticks)
ax.set_xticklabels([str(x) for x in xticks], fontsize=9)

plt.tight_layout()
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
print(f"Saved: {OUT_PNG}")
