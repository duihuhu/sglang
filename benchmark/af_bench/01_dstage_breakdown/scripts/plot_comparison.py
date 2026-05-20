#!/usr/bin/env python3
"""Plot Native vs PD+AF comparison charts."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

JSON_PATH = Path("/workspace/sglang/af_launch_logs/arch_comparison.json")
OUT_DIR = JSON_PATH.parent

with open(JSON_PATH) as f:
    raw = json.load(f)

# Map to English labels
name_map = {
    "原生sglang": "Native (tp=4)",
    "PD+AF分离": "PD+AF",
}
orig_names = list(raw.keys())
labels = [name_map[n] for n in orig_names]
colors = ["#4C72B0", "#DD8452"]

metrics = {
    "Mean TTFT (ms)":         [raw[n]["mean_ttft_ms"] for n in orig_names],
    "P50 TTFT (ms)":          [raw[n]["p50_ttft_ms"] for n in orig_names],
    "Mean TPOT (ms)":         [raw[n]["mean_tpot_ms"] for n in orig_names],
    "Output Throughput (tok/s)": [raw[n]["output_throughput_tok_s"] for n in orig_names],
    "Total Energy (J)":       [raw[n]["total_energy_j"] for n in orig_names],
    "Wall Duration (s)":      [raw[n]["wall_duration_s"] for n in orig_names],
}

# ── Figure 1: 2x3 side-by-side bar charts ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for idx, (title, vals) in enumerate(metrics.items()):
    ax = axes[idx // 3][idx % 3]
    bars = ax.bar(labels, vals, color=colors, width=0.4, edgecolor="white", linewidth=1.2)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=12)

    for bar, v in zip(bars, vals):
        if v >= 1e4:
            label = f"{v:.0f}"
        elif v >= 100:
            label = f"{v:.1f}"
        else:
            label = f"{v:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                label, ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Highlight the better (lower) value for all metrics except throughput
    if "throughput" in title.lower():
        best_idx = int(np.argmax(vals))
    else:
        best_idx = int(np.argmin(vals))
    bars[best_idx].set_edgecolor("#2ca02c")
    bars[best_idx].set_linewidth(3)

fig.suptitle("Native (tp=4) vs PD+AF — Qwen3-32B, 1024 in / 128 out",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
path1 = OUT_DIR / "comparison_bar.png"
fig.savefig(path1, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path1}")

# ── Figure 2: Normalized to Native ─────────────────────────────────────────
native_vals = {m: vals[0] for m, vals in metrics.items()}
normalized = {}
for m, vals in metrics.items():
    nv = native_vals[m]
    normalized[m] = [v / nv if nv != 0 else 1.0 for v in vals]

fig, axes = plt.subplots(2, 3, figsize=(14, 9))

for idx, (title, norm_vals) in enumerate(normalized.items()):
    ax = axes[idx // 3][idx % 3]
    bars = ax.bar(labels, norm_vals, color=colors, width=0.4, edgecolor="white", linewidth=1.2)
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.tick_params(labelsize=12)

    for bar, v in zip(bars, norm_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{v:.2f}x", ha="center", va="bottom", fontsize=9, fontweight="bold")

    if "throughput" in title.lower():
        ax.text(0.98, 0.95, "higher is better", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="green", fontstyle="italic")
    else:
        ax.text(0.98, 0.95, "lower is better", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="red", fontstyle="italic")

fig.suptitle("Native (tp=4) vs PD+AF — Normalized to Native",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
path2 = OUT_DIR / "comparison_normalized.png"
fig.savefig(path2, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path2}")

# ── Figure 3: TTFT vs TPOT scatter + summary table ─────────────────────────
fig, (ax_scatter, ax_table) = plt.subplots(1, 2, figsize=(14, 6),
    gridspec_kw={"width_ratios": [1, 1]})

# Scatter
for i, name in enumerate(labels):
    ax_scatter.scatter(metrics["Mean TTFT (ms)"][i], metrics["Mean TPOT (ms)"][i],
                       s=300, color=colors[i], zorder=5, edgecolors="white", linewidth=2)
ax_scatter.set_xlabel("Mean TTFT (ms)", fontsize=12)
ax_scatter.set_ylabel("Mean TPOT (ms)", fontsize=12)
ax_scatter.set_title("TTFT vs TPOT Trade-off", fontsize=13, fontweight="bold")
ax_scatter.grid(True, alpha=0.3)
for i, name in enumerate(labels):
    ax_scatter.annotate(name,
        (metrics["Mean TTFT (ms)"][i], metrics["Mean TPOT (ms)"][i]),
        (metrics["Mean TTFT (ms)"][i] * 1.02, metrics["Mean TPOT (ms)"][i] * 1.02),
        fontsize=11, fontweight="bold")

# Summary table
table_data = [
    ["Metric", "Native (tp=4)", "PD+AF", "Winner"],
    ["Mean TTFT",  f"{metrics['Mean TTFT (ms)'][0]:.0f} ms",  f"{metrics['Mean TTFT (ms)'][1]:.0f} ms",  "PD+AF" if metrics["Mean TTFT (ms)"][1] < metrics["Mean TTFT (ms)"][0] else "Native"],
    ["P50 TTFT",   f"{metrics['P50 TTFT (ms)'][0]:.0f} ms",   f"{metrics['P50 TTFT (ms)'][1]:.0f} ms",   "PD+AF" if metrics["P50 TTFT (ms)"][1] < metrics["P50 TTFT (ms)"][0] else "Native"],
    ["Mean TPOT",  f"{metrics['Mean TPOT (ms)'][0]:.1f} ms",  f"{metrics['Mean TPOT (ms)'][1]:.1f} ms",  "Native" if metrics["Mean TPOT (ms)"][1] > metrics["Mean TPOT (ms)"][0] else "PD+AF"],
    ["Output Tput", f"{metrics['Output Throughput (tok/s)'][0]:.1f} tok/s", f"{metrics['Output Throughput (tok/s)'][1]:.1f} tok/s", "Native" if metrics["Output Throughput (tok/s)"][1] < metrics["Output Throughput (tok/s)"][0] else "PD+AF"],
    ["Energy",     f"{metrics['Total Energy (J)'][0]:.0f} J",  f"{metrics['Total Energy (J)'][1]:.0f} J",  "Native" if metrics["Total Energy (J)"][1] > metrics["Total Energy (J)"][0] else "PD+AF"],
    ["Duration",   f"{metrics['Wall Duration (s)'][0]:.1f} s", f"{metrics['Wall Duration (s)'][1]:.1f} s", "Native" if metrics["Wall Duration (s)"][1] > metrics["Wall Duration (s)"][0] else "PD+AF"],
]

ax_table.axis("off")
tbl = ax_table.table(cellText=table_data, cellLoc="center", loc="center",
                     colWidths=[0.28, 0.28, 0.28, 0.16])
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
# Style header
for j in range(4):
    tbl[0, j].set_facecolor("#40466e")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
# Style winner column
for i in range(1, 7):
    w = table_data[i][3]
    color = "#c8e6c9" if w == "PD+AF" else "#bbdefb"
    tbl[i, 3].set_facecolor(color)
    tbl[i, 3].set_text_props(fontweight="bold")
ax_table.set_title("Summary", fontsize=13, fontweight="bold", pad=10)

fig.suptitle("Native (tp=4) vs PD+AF — Qwen3-32B",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
path3 = OUT_DIR / "comparison_summary.png"
fig.savefig(path3, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved {path3}")

print("\nAll charts generated.")
