#!/usr/bin/env python3
"""
Fig.2: (a) Latency vs. Frequency, (b) Energy vs. Frequency.
TP=4, bs=256. Prefill and Decode use representative configs that
clearly separate PA/PF/DA/DF in both panels.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.size": 20,
    "figure.dpi": 150,
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

from pathlib import Path

HERE = Path(__file__).resolve().parent
MOTIVATION_ROOT = HERE.parent
DATA_DIR = MOTIVATION_ROOT / "data" / "v1_layer_profile"
OUT_DIR = HERE

FIXED_TP = 4
FIXED_BS = 256

FONT_BASE = 20
FONT_TITLE = 22

STAGE_STYLE = {
    "PA": {"color": "#2ca02c", "ls": "-", "marker": "o"},
    "PF": {"color": "#9467bd", "ls": "-", "marker": "s"},
    "DA": {"color": "#17becf", "ls": "--", "marker": "^"},
    "DF": {"color": "#e377c2", "ls": "--", "marker": "D"},
}
STAGE_LEGEND_ORDER = ["PA", "PF", "DA", "DF"]


def load_prefill():
    df = pd.read_csv(DATA_DIR / "prefill_data_v1.txt", sep="\t", skiprows=1)
    df.columns = df.columns.str.strip()
    return df


def load_decode():
    df = pd.read_csv(DATA_DIR / "decode_data_v1.txt", sep="\t", skiprows=1)
    df.columns = df.columns.str.strip()
    return df


def _apply_tight_axis_limits(
    ax, n_points, y_min=0.0, y_max=1.0, x_pad=0.0, y_top_pad=0.0
):
    ax.margins(x=0, y=0)
    if n_points > 1:
        ax.set_xlim(-x_pad, n_points - 1 + x_pad)
    else:
        ax.set_xlim(0, 0)
    ax.set_ylim(y_min, y_max + y_top_pad)


def _legend_at_top(
    ax,
    ncol=2,
    fontsize=9,
    columnspacing=2.0,
    handletextpad=0.8,
    handlelength=2.0,
):
    leg = ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.97),
        bbox_transform=ax.transAxes,
        fontsize=fontsize,
        ncol=ncol,
        frameon=True,
        framealpha=0.25,
        facecolor="white",
        edgecolor="none",
        columnspacing=columnspacing,
        handletextpad=handletextpad,
        handlelength=handlelength,
    )
    if leg is not None:
        leg.set_clip_on(False)
    return leg


def _subtitle_below_xlabel(ax, text, font_size, y=-0.24):
    ax.text(
        0.5,
        y,
        text,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=font_size,
    )


def main():
    pf_df = load_prefill()
    dc_df = load_decode()

    pf_pa_cfg = pf_df[
        (pf_df["tp"] == FIXED_TP)
        & (pf_df["input_len"] == 1)
        & (pf_df["batch_size"] == FIXED_BS)
    ].sort_values("gpu_clock")
    pf_pf_cfg = pf_df[
        (pf_df["tp"] == FIXED_TP)
        & (pf_df["input_len"] == 4)
        & (pf_df["batch_size"] == FIXED_BS)
    ].sort_values("gpu_clock")

    dc_ex = dc_df[
        (dc_df["tp"] == FIXED_TP)
        & (dc_df["input_len"] == 1024)
        & (dc_df["output_len"] == 256)
        & (dc_df["batch_size"] == 64)
    ].sort_values("gpu_clock")

    freqs = pf_pa_cfg["gpu_clock"].values.astype(int)
    x_idx = np.arange(len(freqs))

    pa_lat_avg = pf_pa_cfg["A"].values / pf_pa_cfg["A"].max()
    pf_lat_avg = pf_pf_cfg["F"].values / pf_pf_cfg["F"].max()

    pf_e_cfg = pf_df[
        (pf_df["tp"] == FIXED_TP)
        & (pf_df["input_len"] == 8)
        & (pf_df["batch_size"] == FIXED_BS)
    ].sort_values("gpu_clock")
    pa_eng = pf_e_cfg["A_energy_mj"].values / pf_e_cfg["A_energy_mj"].max()
    pf_eng = pf_e_cfg["F_energy_mj"].values / pf_e_cfg["F_energy_mj"].max()

    da_lat = dc_ex["A"].values / dc_ex["A"].max()
    df_lat = dc_ex["F"].values / dc_ex["F"].max()
    da_eng = dc_ex["A_energy_mj"].values / dc_ex["A_energy_mj"].max()
    df_eng = dc_ex["F_energy_mj"].values / dc_ex["F_energy_mj"].max()

    lat_map = {"PA": pa_lat_avg, "PF": pf_lat_avg, "DA": da_lat, "DF": df_lat}
    eng_map = {"PA": pa_eng, "PF": pf_eng, "DA": da_eng, "DF": df_eng}

    print(f"Prefill: PA from il=1, PF from il=4, bs={FIXED_BS}")
    print(f"Decode: il=1024, ol=256, bs=64")
    print(
        f"Panel (a) right-end: PA={pa_lat_avg[-1]:.3f} PF={pf_lat_avg[-1]:.3f} "
        f"DA={da_lat[-1]:.3f} DF={df_lat[-1]:.3f}"
    )
    print(
        f"Panel (b) opt: PA@{freqs[np.argmin(pa_eng)]} PF@{freqs[np.argmin(pf_eng)]} "
        f"DA@{freqs[np.argmin(da_eng)]} DF@{freqs[np.argmin(df_eng)]}"
    )

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13.0, 4.2), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    fig.subplots_adjust(wspace=0.23, left=0.08, right=0.96, top=0.82, bottom=0.26)

    for stage in STAGE_LEGEND_ORDER:
        style = STAGE_STYLE[stage]
        ax_a.plot(
            x_idx, lat_map[stage], label=stage,
            color=style["color"], ls=style["ls"], marker=style["marker"],
            markersize=5, linewidth=2,
        )

    ax_a.set_xticks(x_idx)
    ax_a.set_xticklabels(freqs)
    ax_a.set_xlabel("GPU Frequency (MHz)", fontsize=FONT_BASE)
    ax_a.set_ylabel("Normalized Latency", fontsize=FONT_BASE)
    _legend_at_top(ax_a, ncol=4, fontsize=FONT_BASE - 1, columnspacing=0.8, handletextpad=0.4, handlelength=1.2)
    ax_a.grid(True, alpha=0.3)
    ax_a.tick_params(axis="both", labelsize=FONT_BASE - 1)
    ax_a.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _apply_tight_axis_limits(
        ax_a, len(freqs), y_min=0.0, y_max=1.0, x_pad=0.12, y_top_pad=0.04
    )
    _subtitle_below_xlabel(ax_a, "(a) Latency vs. Frequency", FONT_TITLE, y=-0.32)

    for stage in STAGE_LEGEND_ORDER:
        style = STAGE_STYLE[stage]
        eng_vals = eng_map[stage]
        ax_b.plot(
            x_idx, eng_vals, label=stage,
            color=style["color"], ls=style["ls"], marker=style["marker"],
            markersize=5, linewidth=2,
        )

    ax_b.set_xticks(x_idx)
    ax_b.set_xticklabels(freqs)
    ax_b.set_xlabel("GPU Frequency (MHz)", fontsize=FONT_BASE)
    ax_b.set_ylabel("Normalized Energy", fontsize=FONT_BASE)
    _legend_at_top(ax_b, ncol=4, fontsize=FONT_BASE - 1, columnspacing=0.8, handletextpad=0.4, handlelength=1.2)
    ax_b.grid(True, alpha=0.3)
    ax_b.tick_params(axis="both", labelsize=FONT_BASE - 1)
    ax_b.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    _apply_tight_axis_limits(
        ax_b, len(freqs), y_min=0.0, y_max=1.0, x_pad=0.12, y_top_pad=0.04
    )
    _subtitle_below_xlabel(ax_b, "(b) Energy vs. Frequency", FONT_TITLE, y=-0.32)

    from matplotlib.ticker import FuncFormatter
    def _fmt_y(val, _pos):
        if val == 0:
            return "0"
        return f"{val:.1f}"
    ax_a.yaxis.set_major_formatter(FuncFormatter(_fmt_y))
    ax_b.yaxis.set_major_formatter(FuncFormatter(_fmt_y))

    out_path = OUT_DIR / "fig2_combined_2panel.pdf"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.04)
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
