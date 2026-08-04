"""
Figure 5: 2x2 boxplot grid combining TP-wise and BS-wise views.
  (a) F/A ratio by TP      (b) F/A ratio by BS
  (c) Bubble overhead by TP (d) Bubble overhead by BS
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
import numpy as np
import pandas as pd

IDLE_POWER_TABLE = {
    210: 65.81,
    450: 66.31,
    690: 67.28,
    930: 68.86,
    1170: 73.50,
    1410: 90.02,
}

HERE = Path(__file__).resolve().parent
MOTIVATION_ROOT = HERE.parent
DATA_DIR = MOTIVATION_ROOT / "data" / "v1_layer_profile"
OUT_DIR = HERE

decode_df = pd.read_csv(DATA_DIR / "decode_data_v1.txt", sep="\t", skiprows=1)
prefill_df = pd.read_csv(DATA_DIR / "prefill_data_v1.txt", sep="\t", skiprows=1)
decode_df.columns = decode_df.columns.str.strip()
prefill_df.columns = prefill_df.columns.str.strip()

decode_df["FA_ratio"] = decode_df["F"] / decode_df["A"]
prefill_df["FA_ratio"] = prefill_df["F"] / prefill_df["A"]


def compute_bubble_pct(df):
    time_diff_us = np.abs(df["A"].values - df["F"].values)
    tp = df["tp"].values
    freq = df["gpu_clock"].values
    idle_power = np.array([IDLE_POWER_TABLE.get(int(f), 80.0) for f in freq])
    bubble_energy_mj = idle_power * time_diff_us * tp * 1e-3
    total_energy_mj = df["A_energy_mj"].values + df["F_energy_mj"].values
    return bubble_energy_mj / total_energy_mj * 100


decode_df["bubble_pct"] = compute_bubble_pct(decode_df)
prefill_df["bubble_pct"] = compute_bubble_pct(prefill_df)

tp_values = [1, 2, 4, 8]
bs_values = [1, 4, 16, 64]

color_prefill = "#4C72B0"
color_decode = "#DD8452"

GROUP_SPACING = 1.0
PAIR_OFFSET = 0.21

box_kw = dict(
    widths=0.36,
    patch_artist=True,
    showfliers=False,
    medianprops=dict(color="black", linewidth=1.5),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
)


def make_positions(n):
    return np.arange(1, n + 1) * GROUP_SPACING


def set_subtitle_below(ax, text, fontsize=11):
    ax.set_title(text, fontsize=fontsize, pad=14, y=-0.40)


def plot_paired_boxplot(ax, prefill_data, decode_data, positions, xlabels, title, title_center=False):
    positions_p = positions - PAIR_OFFSET
    positions_d = positions + PAIR_OFFSET

    bp_p = ax.boxplot(prefill_data, positions=positions_p, **box_kw)
    bp_d = ax.boxplot(decode_data, positions=positions_d, **box_kw)

    for patch in bp_p["boxes"]:
        patch.set_facecolor(color_prefill)
        patch.set_alpha(0.7)
    for patch in bp_d["boxes"]:
        patch.set_facecolor(color_decode)
        patch.set_alpha(0.7)

    ax.set_xticks(positions)
    ax.set_xticklabels(xlabels, fontsize=10)
    set_subtitle_below(ax, title)
    half_group = PAIR_OFFSET + box_kw["widths"] / 2
    ax.set_xlim(positions[0] - half_group - 0.08, positions[-1] + half_group + 0.08)
    return bp_p, bp_d


fig, axes = plt.subplots(2, 2, figsize=(8.0, 5.0))

# (a) F/A by TP
tp_positions = make_positions(len(tp_values))
bs_positions = make_positions(len(bs_values))

pdata_tp = [prefill_df[prefill_df["tp"] == tp]["FA_ratio"].values for tp in tp_values]
ddata_tp = [decode_df[decode_df["tp"] == tp]["FA_ratio"].values for tp in tp_values]
plot_paired_boxplot(
    axes[0, 0],
    pdata_tp,
    ddata_tp,
    tp_positions,
    [str(tp) for tp in tp_values],
    "(a) Latency F/A Ratio By TP",
    title_center=True,
)
axes[0, 0].axhline(y=1, color="gray", linestyle="--", linewidth=1, zorder=0)
axes[0, 0].set_ylabel("Latency F/A Ratio", fontsize=11)
axes[0, 0].set_ylim(0, 5.5)
axes[0, 0].set_yticks([0, 1, 2, 3, 4, 5])

# (b) Bubble by TP
pdata_tp_b = [prefill_df[prefill_df["tp"] == tp]["bubble_pct"].values for tp in tp_values]
ddata_tp_b = [decode_df[decode_df["tp"] == tp]["bubble_pct"].values for tp in tp_values]
plot_paired_boxplot(
    axes[0, 1],
    pdata_tp_b,
    ddata_tp_b,
    tp_positions,
    [str(tp) for tp in tp_values],
    "(b) Stall Energy Percent By TP",
    title_center=True,
)
Y_LABEL_X = -0.10
Y_LABEL_Y = 0.45

axes[0, 1].set_ylabel("Stall Energy Percent (%)", fontsize=11)
axes[0, 1].set_ylim(-2.5, 50)
axes[0, 1].set_yticks([0, 10, 20, 30, 40, 50])
axes[0, 1].grid(True, alpha=0.2, axis="y")

# (c) F/A by BS
pdata_bs = [prefill_df[prefill_df["batch_size"] == bs]["FA_ratio"].values for bs in bs_values]
ddata_bs = [decode_df[decode_df["batch_size"] == bs]["FA_ratio"].values for bs in bs_values]
plot_paired_boxplot(
    axes[1, 0],
    pdata_bs,
    ddata_bs,
    bs_positions,
    [str(bs) for bs in bs_values],
    "(c) Latency F/A Ratio By Batch Size",
    title_center=True,
)
axes[1, 0].axhline(y=1, color="gray", linestyle="--", linewidth=1, zorder=0)
axes[1, 0].set_ylabel("Latency F/A Ratio", fontsize=11)
axes[1, 0].set_ylim(0, 5.5)
axes[1, 0].set_yticks([0, 1, 2, 3, 4, 5])

# (d) Bubble by BS
pdata_bs_b = [prefill_df[prefill_df["batch_size"] == bs]["bubble_pct"].values for bs in bs_values]
ddata_bs_b = [decode_df[decode_df["batch_size"] == bs]["bubble_pct"].values for bs in bs_values]
plot_paired_boxplot(
    axes[1, 1],
    pdata_bs_b,
    ddata_bs_b,
    bs_positions,
    [str(bs) for bs in bs_values],
    "(d) Stall Energy Percent By Batch Size",
    title_center=True,
)
axes[1, 1].set_ylabel("Stall Energy Percent (%)", fontsize=11)
axes[1, 1].set_ylim(-2.5, 50)
axes[1, 1].set_yticks([0, 10, 20, 30, 40, 50])
axes[1, 1].grid(True, alpha=0.2, axis="y")

legend_patches = [
    mpatches.Patch(color=color_prefill, alpha=0.7, label="P"),
    mpatches.Patch(color=color_decode, alpha=0.7, label="D"),
]
axes[0, 0].legend(handles=legend_patches, loc="upper left", fontsize=10)

plt.subplots_adjust(hspace=0.38, wspace=0.22, left=0.11, right=0.96, top=0.96, bottom=0.16)
axes[0, 1].yaxis.set_label_coords(Y_LABEL_X, Y_LABEL_Y)
axes[1, 1].yaxis.set_label_coords(Y_LABEL_X, Y_LABEL_Y)
plt.savefig(
    OUT_DIR / "fig4_combined_4panel.pdf",
    bbox_inches="tight",
    pad_inches=0.03,
)
print("Saved: fig4_combined_4panel.pdf")
