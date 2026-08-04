"""
Fig.3: Decode relative latency vs relative energy heatmap.
Generates fig3_combined_4panel.pdf and .png.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize

plt.rcParams.update({
    "font.size": 10,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

HERE = Path(__file__).resolve().parent
MOTIVATION_ROOT = HERE.parent
DATA_DIR = MOTIVATION_ROOT / "data" / "v1_layer_profile"
OUT_DIR = HERE
TARGET_TPS = [1, 2, 4, 8]
TARGET_BS = [1, 4, 16, 64, 256]


def load_decode():
    fpath = DATA_DIR / "decode_data_v1.txt"
    df = pd.read_csv(fpath, sep="\t", skiprows=1)
    df.columns = [
        "tp", "input_len", "output_len", "gpu_clock", "batch_size",
        "A_us", "F_us", "lat_ms", "AF64_ms", "A_energy_mj", "F_energy_mj",
    ]
    return df


def compute_ratios(df):
    results = []
    grouped = df.groupby(["tp", "batch_size", "input_len"])
    for (tp, bs, _il), group in grouped:
        if tp not in TARGET_TPS or bs not in TARGET_BS:
            continue
        if group["A_us"].min() <= 0 or group["F_us"].min() <= 0:
            continue
        if group["A_energy_mj"].min() <= 0 or group["F_energy_mj"].min() <= 0:
            continue
        results.append({
            "tp": tp,
            "batch_size": bs,
            "A_lat_range": 1.0 - group["A_us"].min() / group["A_us"].max(),
            "F_lat_range": 1.0 - group["F_us"].min() / group["F_us"].max(),
            "A_energy_range": 1.0 - group["A_energy_mj"].min() / group["A_energy_mj"].max(),
            "F_energy_range": 1.0 - group["F_energy_mj"].min() / group["F_energy_mj"].max(),
        })
    return pd.DataFrame(results)


def make_pivot(df_ratio, col):
    avg = df_ratio.groupby(["tp", "batch_size"])[[col]].mean().reset_index()
    pivot = avg.pivot(index="batch_size", columns="tp", values=col)
    return pivot.reindex(index=sorted(TARGET_BS, reverse=True), columns=TARGET_TPS)


def draw_heatmap(ax, pivot, title, norm, cmap):
    im = ax.imshow(pivot.values, cmap=cmap, norm=norm, aspect="auto")
    ax.set_xticks(range(len(TARGET_TPS)))
    ax.set_xticklabels([f"TP{t}" for t in TARGET_TPS])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([str(int(b)) for b in pivot.index])
    ax.set_title(title, fontsize=12, pad=20, y=-0.44)
    for i in range(len(pivot.index)):
        for j in range(len(TARGET_TPS)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color="black")
    return im


def main():
    df = load_decode()
    ratios = compute_ratios(df)
    da_lat = make_pivot(ratios, "A_lat_range")
    df_lat = make_pivot(ratios, "F_lat_range")
    da_energy = make_pivot(ratios, "A_energy_range")
    df_energy = make_pivot(ratios, "F_energy_range")

    norm_top = Normalize(vmin=0.0, vmax=1.0)
    norm_bot = Normalize(vmin=0.0, vmax=1.0)
    cmap = "YlOrRd"

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8))
    draw_heatmap(axes[0, 0], da_lat, "(a) DA - Latency Sensitivity", norm_top, cmap)
    im_top = draw_heatmap(axes[0, 1], df_lat, "(b) DF - Latency Sensitivity", norm_top, cmap)
    draw_heatmap(axes[1, 0], da_energy, "(c) DA - Energy Sensitivity", norm_bot, cmap)
    im_bot = draw_heatmap(axes[1, 1], df_energy, "(d) DF - Energy Sensitivity", norm_bot, cmap)

    plt.subplots_adjust(wspace=0.18, hspace=0.36, left=0.095, right=0.88, top=0.96, bottom=0.22)

    pos_top = axes[0, 1].get_position()
    pos_bot = axes[1, 1].get_position()
    cbar_w, cbar_gap = 0.015, 0.02
    cbar_ticks = np.arange(0.0, 1.01, 0.2)
    cbar_ax_top = fig.add_axes([pos_top.x1 + cbar_gap, pos_top.y0, cbar_w, pos_top.height])
    cbar_top = fig.colorbar(im_top, cax=cbar_ax_top)
    cbar_top.set_ticks(cbar_ticks)
    cbar_top.set_ticklabels([f"{t:.1f}" for t in cbar_ticks])
    cbar_top.ax.tick_params(labelsize=9)
    cbar_ax_top.set_ylabel("Relative Range", fontsize=10, rotation=270, labelpad=16)
    cbar_ax_bot = fig.add_axes([pos_bot.x1 + cbar_gap, pos_bot.y0, cbar_w, pos_bot.height])
    cbar_bot = fig.colorbar(im_bot, cax=cbar_ax_bot)
    cbar_bot.set_ticks(cbar_ticks)
    cbar_bot.set_ticklabels([f"{t:.1f}" for t in cbar_ticks])
    cbar_bot.ax.tick_params(labelsize=9)
    cbar_ax_bot.set_ylabel("Relative Range", fontsize=10, rotation=270, labelpad=16)

    pos_top_left = axes[0, 0].get_position()
    pos_bot_left = axes[1, 0].get_position()
    batch_size_x = pos_top_left.x0 - 0.05
    fig.text(
        batch_size_x,
        (pos_top_left.y0 + pos_top_left.y1) / 2,
        "Batch Size",
        ha="right",
        va="center",
        rotation=90,
        fontsize=12,
    )
    fig.text(
        batch_size_x,
        (pos_bot_left.y0 + pos_bot_left.y1) / 2,
        "Batch Size",
        ha="right",
        va="center",
        rotation=90,
        fontsize=12,
    )
    out_path = OUT_DIR / "fig3_combined_4panel.pdf"
    png_path = OUT_DIR / "fig3_combined_4panel.png"
    fig.savefig(out_path, format="pdf", bbox_inches="tight", pad_inches=0)
    fig.savefig(png_path, format="png", dpi=200, bbox_inches="tight", pad_inches=0)
    print(f"Saved: {out_path.resolve()}")
    print(f"Saved: {png_path.resolve()}")
    plt.close()


if __name__ == "__main__":
    main()
