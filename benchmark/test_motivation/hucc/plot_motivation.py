#!/usr/bin/env python3
"""Generate all motivation figures for AF separation analysis."""

import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

STAGE_COLORS = {
    "PA": "#1565C0",
    "PF": "#E65100",
    "DA": "#2E7D32",
    "DF": "#7B1FA2",
}
FREQ_COLORS = {210: "#E53935", 540: "#FB8C00", 870: "#43A047", 1200: "#1E88E5"}
INPUT_MARKERS = {128: "o", 512: "s", 4096: "D"}

BASE_DIR = Path(__file__).parent
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)


def load_data():
    p = pd.read_csv(BASE_DIR / "prefill_data.txt", sep="\t")
    d = pd.read_csv(BASE_DIR / "decode_data.txt", sep="\t")
    p.columns = p.columns.str.strip()
    d.columns = d.columns.str.strip()
    if "P_FFFN" in p.columns:
        p = p.rename(columns={"P_FFFN": "P_FFN"})
    return p, d


# ── Fig 1.1  DVFS Speedup Bar ──────────────────────────────────────────────
def fig1_1(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    input_lens = [128, 512, 4096]
    x = np.arange(len(input_lens))
    w = 0.35

    # Prefill (tp=8, bs=1)
    ax = axes[0]
    pa_sp, pf_sp = [], []
    for il in input_lens:
        row_lo = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == 210) & (pf.batch_size == 1)]
        row_hi = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == 1200) & (pf.batch_size == 1)]
        pa_sp.append(row_lo.P_Attention.values[0] / row_hi.P_Attention.values[0])
        pf_sp.append(row_lo.P_FFN.values[0] / row_hi.P_FFN.values[0])
    ax.bar(x - w / 2, pa_sp, w, label="P-Attention", color=STAGE_COLORS["PA"])
    ax.bar(x + w / 2, pf_sp, w, label="P-FFN", color=STAGE_COLORS["PF"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(il) for il in input_lens])
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Speedup (210→1200 MHz)")
    ax.set_title("Prefill DVFS Speedup (tp=8, bs=1)")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    # Decode (tp=8, bs=1, output=256)
    ax = axes[1]
    da_sp, df_sp = [], []
    for il in input_lens:
        row_lo = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == 210) & (dc.batch_size == 1)]
        row_hi = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == 1200) & (dc.batch_size == 1)]
        da_sp.append(row_lo.D_Attention.values[0] / row_hi.D_Attention.values[0])
        df_sp.append(row_lo.D_FFN.values[0] / row_hi.D_FFN.values[0])
    ax.bar(x - w / 2, da_sp, w, label="D-Attention", color=STAGE_COLORS["DA"])
    ax.bar(x + w / 2, df_sp, w, label="D-FFN", color=STAGE_COLORS["DF"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(il) for il in input_lens])
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Speedup (210→1200 MHz)")
    ax.set_title("Decode DVFS Speedup (tp=8, bs=1, out=256)")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.5)

    fig.suptitle("Fig 1.1: DVFS Speedup — Attention vs FFN", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_1_dvfs_speedup_bar.png")
    plt.close(fig)
    print("  [done] fig1_1")


# ── Fig 1.3  PA Characteristic Drift — PA speedup varies across configs ───
def fig1_3(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: PA speedup across different (tp, bs) configs, fixed input=4096
    ax = axes[0]
    configs_pa = [
        ("tp=1\nbs=1", 1, 1),
        ("tp=2\nbs=1", 2, 1),
        ("tp=2\nbs=16", 2, 16),
        ("tp=4\nbs=1", 4, 1),
        ("tp=4\nbs=16", 4, 16),
        ("tp=8\nbs=1", 8, 1),
        ("tp=8\nbs=16", 8, 16),
        ("tp=8\nbs=256", 8, 256),
    ]
    labels, pa_sp, pf_sp = [], [], []
    for name, tp, bs in configs_pa:
        row_lo = pf[(pf.tp == tp) & (pf.input_len == 4096) & (pf.gpu_clock == 210) & (pf.batch_size == bs)]
        row_hi = pf[(pf.tp == tp) & (pf.input_len == 4096) & (pf.gpu_clock == 1200) & (pf.batch_size == bs)]
        if len(row_lo) > 0 and len(row_hi) > 0:
            labels.append(name)
            pa_sp.append(row_lo.P_Attention.values[0] / row_hi.P_Attention.values[0])
            pf_sp.append(row_lo.P_FFN.values[0] / row_hi.P_FFN.values[0])
    x = np.arange(len(labels))
    w = 0.35
    ax.bar(x - w / 2, pa_sp, w, label="P-Attention", color=STAGE_COLORS["PA"])
    ax.bar(x + w / 2, pf_sp, w, label="P-FFN", color=STAGE_COLORS["PF"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Configuration (TP, Batch Size)")
    ax.set_ylabel("Speedup (210→1200 MHz)")
    ax.set_title("Prefill: PA vs PF Speedup (input=4096)")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    ax.axhspan(1.0, 2.0, alpha=0.06, color="blue", label="_mem-bound zone")
    ax.axhspan(3.0, 6.0, alpha=0.06, color="red", label="_compute-bound zone")
    ax.text(0.02, 0.92, "memory-bound zone", transform=ax.transAxes, fontsize=8, color="blue", alpha=0.6)
    ax.text(0.02, 0.72, "compute-bound zone", transform=ax.transAxes, fontsize=8, color="red", alpha=0.6)

    # Right: DA speedup across configs for comparison — stays uniformly low
    ax = axes[1]
    configs_da = [
        ("tp=1\nbs=1", 1, 1),
        ("tp=2\nbs=1", 2, 1),
        ("tp=2\nbs=16", 2, 16),
        ("tp=4\nbs=1", 4, 1),
        ("tp=4\nbs=16", 4, 16),
        ("tp=8\nbs=1", 8, 1),
        ("tp=8\nbs=16", 8, 16),
        ("tp=8\nbs=256", 8, 256),
    ]
    labels, da_sp, df_sp = [], [], []
    for name, tp, bs in configs_da:
        row_lo = dc[(dc.tp == tp) & (dc.input_len == 4096) & (dc.output_len == 256) & (dc.gpu_clock == 210) & (dc.batch_size == bs)]
        row_hi = dc[(dc.tp == tp) & (dc.input_len == 4096) & (dc.output_len == 256) & (dc.gpu_clock == 1200) & (dc.batch_size == bs)]
        if len(row_lo) > 0 and len(row_hi) > 0:
            labels.append(name)
            da_sp.append(row_lo.D_Attention.values[0] / row_hi.D_Attention.values[0])
            df_sp.append(row_lo.D_FFN.values[0] / row_hi.D_FFN.values[0])
    x = np.arange(len(labels))
    ax.bar(x - w / 2, da_sp, w, label="D-Attention", color=STAGE_COLORS["DA"])
    ax.bar(x + w / 2, df_sp, w, label="D-FFN", color=STAGE_COLORS["DF"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_xlabel("Configuration (TP, Batch Size)")
    ax.set_ylabel("Speedup (210→1200 MHz)")
    ax.set_title("Decode: DA vs DF Speedup (input=4096, out=256)")
    ax.legend()
    ax.axhline(y=1, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(axes[0].get_ylim())

    fig.suptitle("Fig 1.3: PA Characteristic Drift — PA varies while DA stays memory-bound", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_3_pa_characteristic_drift.png")
    plt.close(fig)
    print("  [done] fig1_3")


# ── Fig 1.2  Normalized Time vs Freq ───────────────────────────────────────
def fig1_2(pf, dc):
    fig, ax = plt.subplots(figsize=(8, 5))
    freqs = [210, 540, 870, 1200]

    configs = {
        "PA": (pf, "P_Attention", {"tp": 8, "input_len": 4096, "batch_size": 16}),
        "PF": (pf, "P_FFN", {"tp": 8, "input_len": 4096, "batch_size": 16}),
        "DA": (dc, "D_Attention", {"tp": 8, "input_len": 4096, "output_len": 256, "batch_size": 16}),
        "DF": (dc, "D_FFN", {"tp": 8, "input_len": 4096, "output_len": 256, "batch_size": 16}),
    }

    for stage, (df, col, filt) in configs.items():
        mask = pd.Series(True, index=df.index)
        for k, v in filt.items():
            mask &= df[k] == v
        times = []
        for f in freqs:
            row = df[mask & (df.gpu_clock == f)]
            times.append(row[col].values[0])
        base = times[0]
        ax.plot(freqs, [t / base for t in times], "o-", color=STAGE_COLORS[stage], label=stage, linewidth=2, markersize=6)

    ideal = [1.0, 210 / 540, 210 / 870, 210 / 1200]
    ax.plot(freqs, ideal, "k--", alpha=0.4, label="Ideal (linear)")
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Normalized Time (baseline=210MHz)")
    ax.set_title("Fig 1.2: Normalized Time vs GPU Frequency\n(tp=8, input=4096, bs=16)")
    ax.legend()
    ax.set_xticks(freqs)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig1_2_normalized_time_vs_freq.png")
    plt.close(fig)
    print("  [done] fig1_2")


# ── Fig 2.1  Attention Fraction vs Input Length ────────────────────────────
def fig2_1(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    input_lens = [128, 512, 4096]
    freqs = [210, 540, 870, 1200]

    # Prefill (tp=8, bs=16)
    ax = axes[0]
    for f in freqs:
        fracs = []
        for il in input_lens:
            row = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == f) & (pf.batch_size == 16)]
            if len(row) > 0:
                pa, pff = row.P_Attention.values[0], row.P_FFN.values[0]
                fracs.append(pa / (pa + pff))
            else:
                fracs.append(np.nan)
        ax.plot(input_lens, fracs, "o-", color=FREQ_COLORS[f], label=f"{f} MHz", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Attention Fraction  A/(A+F)")
    ax.set_title("Prefill (tp=8, bs=16)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(input_lens)
    ax.set_xticklabels([str(il) for il in input_lens])
    ax.legend()
    ax.set_ylim(0, 0.8)

    # Decode (tp=8, bs=16, output=256)
    ax = axes[1]
    for f in freqs:
        fracs = []
        for il in input_lens:
            row = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == f) & (dc.batch_size == 16)]
            if len(row) > 0:
                da, dff = row.D_Attention.values[0], row.D_FFN.values[0]
                fracs.append(da / (da + dff))
            else:
                fracs.append(np.nan)
        ax.plot(input_lens, fracs, "o-", color=FREQ_COLORS[f], label=f"{f} MHz", linewidth=2)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Attention Fraction  A/(A+F)")
    ax.set_title("Decode (tp=8, bs=16, out=256)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(input_lens)
    ax.set_xticklabels([str(il) for il in input_lens])
    ax.legend()
    ax.set_ylim(0.3, 0.7)

    fig.suptitle("Fig 2.1: Attention Fraction vs Input Length", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_1_attn_fraction_vs_inputlen.png")
    plt.close(fig)
    print("  [done] fig2_1")


# ── Fig 2.2  Attention Fraction vs Batch Size ──────────────────────────────
def fig2_2(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    input_lens = [128, 512, 4096]

    # Prefill (tp=8, clock=870)
    ax = axes[0]
    for il in input_lens:
        bss, fracs = [], []
        for bs in [1, 16, 256]:
            row = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == 870) & (pf.batch_size == bs)]
            if len(row) > 0:
                pa, pff = row.P_Attention.values[0], row.P_FFN.values[0]
                bss.append(bs)
                fracs.append(pa / (pa + pff))
        ax.plot(bss, fracs, "-", label=f"input={il}", linewidth=2, marker=INPUT_MARKERS[il])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Attention Fraction  A/(A+F)")
    ax.set_title("Prefill (tp=8, clock=870)")
    ax.set_xscale("log", base=2)
    ax.legend()

    # Decode (tp=8, clock=870, output=256)
    ax = axes[1]
    for il in input_lens:
        bss, fracs = [], []
        for bs in [1, 16, 256]:
            row = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == 870) & (dc.batch_size == bs)]
            if len(row) > 0:
                da, dff = row.D_Attention.values[0], row.D_FFN.values[0]
                bss.append(bs)
                fracs.append(da / (da + dff))
        ax.plot(bss, fracs, "-", label=f"input={il}", linewidth=2, marker=INPUT_MARKERS[il])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Batch Size")
    ax.set_ylabel("Attention Fraction  A/(A+F)")
    ax.set_title("Decode (tp=8, clock=870, out=256)")
    ax.set_xscale("log", base=2)
    ax.legend()

    fig.suptitle("Fig 2.2: Attention Fraction vs Batch Size", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_2_attn_fraction_vs_batchsize.png")
    plt.close(fig)
    print("  [done] fig2_2")


# ── Fig 2.3  Attention Fraction vs GPU Frequency ──────────────────────────
def fig2_3(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    freqs = [210, 540, 870, 1200]
    input_lens = [128, 512, 4096]

    # Prefill (tp=8, bs=16)
    ax = axes[0]
    for il in input_lens:
        fracs = []
        for f in freqs:
            row = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == f) & (pf.batch_size == 16)]
            if len(row) > 0:
                pa, pff = row.P_Attention.values[0], row.P_FFN.values[0]
                fracs.append(pa / (pa + pff))
            else:
                fracs.append(np.nan)
        ax.plot(freqs, fracs, "-", label=f"input={il}", linewidth=2, marker=INPUT_MARKERS[il])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Attention Fraction  A/(A+F)")
    ax.set_title("Prefill (tp=8, bs=16)")
    ax.set_xticks(freqs)
    ax.legend()

    # Decode (tp=8, bs=16, output=256)
    ax = axes[1]
    for il in input_lens:
        fracs = []
        for f in freqs:
            row = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == f) & (dc.batch_size == 16)]
            if len(row) > 0:
                da, dff = row.D_Attention.values[0], row.D_FFN.values[0]
                fracs.append(da / (da + dff))
            else:
                fracs.append(np.nan)
        ax.plot(freqs, fracs, "-", label=f"input={il}", linewidth=2, marker=INPUT_MARKERS[il])
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Attention Fraction  A/(A+F)")
    ax.set_title("Decode (tp=8, bs=16, out=256)")
    ax.set_xticks(freqs)
    ax.legend()

    fig.suptitle("Fig 2.3: Attention Fraction vs GPU Frequency", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_3_attn_fraction_vs_freq.png")
    plt.close(fig)
    print("  [done] fig2_3")


# ── Fig 2.4  Stacked Bar Breakdown ────────────────────────────────────────
def fig2_4(pf, dc):
    configs = [
        ("short-seq\nbs=1", 8, 128, 1, 256),
        ("short-seq\nbs=16", 8, 128, 16, 256),
        ("short-seq\nbs=256", 8, 128, 256, 256),
        ("long-seq\nbs=1", 8, 4096, 1, 256),
        ("long-seq\nbs=16", 8, 4096, 16, 256),
        ("long-seq\nbs=256", 8, 4096, 256, 256),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    clock = 870

    # Prefill
    ax = axes[0]
    labels, pa_vals, pf_vals = [], [], []
    for name, tp, il, bs, _ in configs:
        row = pf[(pf.tp == tp) & (pf.input_len == il) & (pf.gpu_clock == clock) & (pf.batch_size == bs)]
        if len(row) > 0:
            labels.append(name)
            pa_vals.append(row.P_Attention.values[0])
            pf_vals.append(row.P_FFN.values[0])
    x = np.arange(len(labels))
    totals = [a + f for a, f in zip(pa_vals, pf_vals)]
    pa_frac = [a / t * 100 for a, t in zip(pa_vals, totals)]
    pf_frac = [f / t * 100 for f, t in zip(pf_vals, totals)]
    ax.bar(x, pa_frac, label="P-Attention", color=STAGE_COLORS["PA"])
    ax.bar(x, pf_frac, bottom=pa_frac, label="P-FFN", color=STAGE_COLORS["PF"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time Fraction (%)")
    ax.set_title(f"Prefill Breakdown (tp=8, clock={clock})")
    ax.legend()
    ax.set_ylim(0, 105)

    # Decode
    ax = axes[1]
    labels, da_vals, df_vals = [], [], []
    for name, tp, il, bs, ol in configs:
        row = dc[(dc.tp == tp) & (dc.input_len == il) & (dc.output_len == ol) & (dc.gpu_clock == clock) & (dc.batch_size == bs)]
        if len(row) > 0:
            labels.append(name)
            da_vals.append(row.D_Attention.values[0])
            df_vals.append(row.D_FFN.values[0])
    x = np.arange(len(labels))
    totals = [a + f for a, f in zip(da_vals, df_vals)]
    da_frac = [a / t * 100 for a, t in zip(da_vals, totals)]
    df_frac = [f / t * 100 for f, t in zip(df_vals, totals)]
    ax.bar(x, da_frac, label="D-Attention", color=STAGE_COLORS["DA"])
    ax.bar(x, df_frac, bottom=da_frac, label="D-FFN", color=STAGE_COLORS["DF"])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Time Fraction (%)")
    ax.set_title(f"Decode Breakdown (tp=8, clock={clock}, out=256)")
    ax.legend()
    ax.set_ylim(0, 105)

    fig.suptitle("Fig 2.4: A/F Time Fraction Breakdown", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig2_4_stacked_bar_breakdown.png")
    plt.close(fig)
    print("  [done] fig2_4")


# ── Fig 3.1 / 3.2  A/F Ratio Heatmap ──────────────────────────────────────
def fig3_1_3_2(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    input_lens = [128, 512, 4096]
    batch_sizes = [1, 16, 256]
    clock = 870

    # Prefill heatmap (tp=8)
    ax = axes[0]
    mat = np.full((len(batch_sizes), len(input_lens)), np.nan)
    for i, bs in enumerate(batch_sizes):
        for j, il in enumerate(input_lens):
            row = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == clock) & (pf.batch_size == bs)]
            if len(row) > 0:
                mat[i, j] = row.P_Attention.values[0] / row.P_FFN.values[0]
    im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto", vmin=0.2, vmax=1.2)
    for i in range(len(batch_sizes)):
        for j in range(len(input_lens)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(input_lens)))
    ax.set_xticklabels([str(il) for il in input_lens])
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Batch Size")
    ax.set_title(f"Prefill PA/PF Ratio (tp=8, clock={clock})")
    plt.colorbar(im, ax=ax, shrink=0.8, label="PA / PF")

    # Decode heatmap (tp=8, output=256)
    ax = axes[1]
    mat = np.full((len(batch_sizes), len(input_lens)), np.nan)
    for i, bs in enumerate(batch_sizes):
        for j, il in enumerate(input_lens):
            row = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == clock) & (dc.batch_size == bs)]
            if len(row) > 0:
                mat[i, j] = row.D_Attention.values[0] / row.D_FFN.values[0]
    im = ax.imshow(mat, cmap="RdYlBu_r", aspect="auto", vmin=0.5, vmax=1.6)
    for i in range(len(batch_sizes)):
        for j in range(len(input_lens)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=11, fontweight="bold")
    ax.set_xticks(range(len(input_lens)))
    ax.set_xticklabels([str(il) for il in input_lens])
    ax.set_yticks(range(len(batch_sizes)))
    ax.set_yticklabels([str(bs) for bs in batch_sizes])
    ax.set_xlabel("Input Length")
    ax.set_ylabel("Batch Size")
    ax.set_title(f"Decode DA/DF Ratio (tp=8, clock={clock}, out=256)")
    plt.colorbar(im, ax=ax, shrink=0.8, label="DA / DF")

    fig.suptitle("Fig 3.1/3.2: A/F Ratio Heatmap", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_1_3_2_af_ratio_heatmap.png")
    plt.close(fig)
    print("  [done] fig3_1_3_2")


# ── Fig 3.3  A/F Ratio Box Plot ───────────────────────────────────────────
def fig3_3(pf, dc):
    fig, ax = plt.subplots(figsize=(8, 5))

    pf_ratios = (pf.P_Attention / pf.P_FFN).dropna().values
    dc_ratios = (dc.D_Attention / dc.D_FFN).dropna().values

    bp = ax.boxplot(
        [pf_ratios, dc_ratios],
        labels=["Prefill\nPA/PF", "Decode\nDA/DF"],
        patch_artist=True,
        widths=0.5,
        showmeans=True,
        meanprops=dict(marker="D", markerfacecolor="red", markersize=7),
    )
    bp["boxes"][0].set_facecolor("#BBDEFB")
    bp["boxes"][1].set_facecolor("#C8E6C9")

    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.6, label="A=F (ratio=1.0)")
    ax.set_ylabel("A/F Time Ratio")
    ax.set_title("Fig 3.3: A/F Ratio Distribution Across All Configs")
    ax.legend()

    stats_text = (
        f"Prefill: min={pf_ratios.min():.2f}, max={pf_ratios.max():.2f}, "
        f"range={pf_ratios.max()-pf_ratios.min():.2f}\n"
        f"Decode:  min={dc_ratios.min():.2f}, max={dc_ratios.max():.2f}, "
        f"range={dc_ratios.max()-dc_ratios.min():.2f}"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_3_af_ratio_boxplot.png")
    plt.close(fig)
    print("  [done] fig3_3")


# ── Fig 3.4  Sensitivity Tornado ──────────────────────────────────────────
def fig3_4(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    def sensitivity(df, a_col, f_col, var, other_filters, var_vals):
        ratios = []
        for v in var_vals:
            filt = (df[var] == v)
            for k, val in other_filters.items():
                filt &= (df[k] == val)
            rows = df[filt]
            if len(rows) > 0:
                ratios.append((rows[a_col] / rows[f_col]).mean())
        if len(ratios) >= 2:
            return max(ratios) - min(ratios)
        return 0

    # Prefill sensitivity (tp=8)
    ax = axes[0]
    base_p = {"tp": 8}
    vars_p = {
        "input_len": [128, 512, 4096],
        "gpu_clock": [210, 540, 870, 1200],
        "batch_size": [1, 16, 256],
    }
    sens_p = {}
    for var, vals in vars_p.items():
        others = {k: v for k, v in {"input_len": 4096, "gpu_clock": 870, "batch_size": 16}.items() if k != var}
        others.update(base_p)
        sens_p[var] = sensitivity(pf, "P_Attention", "P_FFN", var, others, vals)
    sorted_vars = sorted(sens_p.keys(), key=lambda k: sens_p[k])
    y_pos = range(len(sorted_vars))
    ax.barh(y_pos, [sens_p[v] for v in sorted_vars], color=["#42A5F5", "#66BB6A", "#FFA726"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_vars)
    ax.set_xlabel("A/F Ratio Range")
    ax.set_title("Prefill Sensitivity (tp=8)")

    # Decode sensitivity (tp=8, output_len=256)
    ax = axes[1]
    base_d = {"tp": 8, "output_len": 256}
    vars_d = {
        "input_len": [128, 512, 4096],
        "gpu_clock": [210, 540, 870, 1200],
        "batch_size": [1, 16, 256],
    }
    sens_d = {}
    for var, vals in vars_d.items():
        others = {k: v for k, v in {"input_len": 4096, "gpu_clock": 870, "batch_size": 16}.items() if k != var}
        others.update(base_d)
        sens_d[var] = sensitivity(dc, "D_Attention", "D_FFN", var, others, vals)
    sorted_vars = sorted(sens_d.keys(), key=lambda k: sens_d[k])
    y_pos = range(len(sorted_vars))
    ax.barh(y_pos, [sens_d[v] for v in sorted_vars], color=["#42A5F5", "#66BB6A", "#FFA726"])
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_vars)
    ax.set_xlabel("A/F Ratio Range")
    ax.set_title("Decode Sensitivity (tp=8, out=256)")

    fig.suptitle("Fig 3.4: Sensitivity — Which Variable Impacts A/F Ratio Most?", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig3_4_sensitivity_tornado.png")
    plt.close(fig)
    print("  [done] fig3_4")


# ── Fig 4.1  Absolute Time vs Freq ────────────────────────────────────────
def fig4_1(pf, dc):
    fig, ax = plt.subplots(figsize=(8, 5))
    freqs = [210, 540, 870, 1200]

    configs = {
        "PA": (pf, "P_Attention", {"tp": 8, "input_len": 4096, "batch_size": 1}),
        "PF": (pf, "P_FFN", {"tp": 8, "input_len": 4096, "batch_size": 1}),
        "DA": (dc, "D_Attention", {"tp": 8, "input_len": 4096, "output_len": 256, "batch_size": 1}),
        "DF": (dc, "D_FFN", {"tp": 8, "input_len": 4096, "output_len": 256, "batch_size": 1}),
    }

    for stage, (df, col, filt) in configs.items():
        mask = pd.Series(True, index=df.index)
        for k, v in filt.items():
            mask &= df[k] == v
        times = [df[mask & (df.gpu_clock == f)][col].values[0] for f in freqs]
        ls = "-" if stage.startswith("P") else "--"
        ax.plot(freqs, times, f"o{ls}", color=STAGE_COLORS[stage], label=stage, linewidth=2, markersize=6)

    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Execution Time (μs)")
    ax.set_title("Fig 4.1: Execution Time vs GPU Frequency\n(tp=8, input=4096, bs=1)")
    ax.legend()
    ax.set_xticks(freqs)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_1_time_vs_freq_absolute.png")
    plt.close(fig)
    print("  [done] fig4_1")


# ── Fig 4.2  Marginal Speedup per Freq Interval ──────────────────────────
def fig4_2(pf, dc):
    fig, ax = plt.subplots(figsize=(9, 5))
    freqs = [210, 540, 870, 1200]
    intervals = ["210→540", "540→870", "870→1200"]

    configs = {
        "PA": (pf, "P_Attention", {"tp": 8, "input_len": 4096, "batch_size": 1}),
        "PF": (pf, "P_FFN", {"tp": 8, "input_len": 4096, "batch_size": 1}),
        "DA": (dc, "D_Attention", {"tp": 8, "input_len": 128, "output_len": 256, "batch_size": 1}),
        "DF": (dc, "D_FFN", {"tp": 8, "input_len": 128, "output_len": 256, "batch_size": 1}),
    }

    x = np.arange(len(intervals))
    w = 0.2
    for idx, (stage, (df, col, filt)) in enumerate(configs.items()):
        mask = pd.Series(True, index=df.index)
        for k, v in filt.items():
            mask &= df[k] == v
        times = [df[mask & (df.gpu_clock == f)][col].values[0] for f in freqs]
        marginals = [(times[i] - times[i + 1]) / times[i] * 100 for i in range(3)]
        ax.bar(x + (idx - 1.5) * w, marginals, w, label=stage, color=STAGE_COLORS[stage])

    ax.set_xticks(x)
    ax.set_xticklabels(intervals)
    ax.set_xlabel("Frequency Interval")
    ax.set_ylabel("Marginal Speedup (%)")
    ax.set_title("Fig 4.2: Marginal Speedup per Frequency Interval (tp=8, bs=1)")
    ax.legend()
    ax.axhline(y=0, color="gray", linestyle="-", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig4_2_marginal_speedup.png")
    plt.close(fig)
    print("  [done] fig4_2")


# ── Fig 5.1  Frequency Scaling Efficiency ─────────────────────────────────
def fig5_1(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    freqs = [210, 540, 870, 1200]

    # Large workload
    ax = axes[0]
    configs = {
        "PA": (pf, "P_Attention", {"tp": 8, "input_len": 4096, "batch_size": 1}),
        "PF": (pf, "P_FFN", {"tp": 8, "input_len": 4096, "batch_size": 1}),
        "DA": (dc, "D_Attention", {"tp": 8, "input_len": 4096, "output_len": 256, "batch_size": 1}),
        "DF": (dc, "D_FFN", {"tp": 8, "input_len": 4096, "output_len": 256, "batch_size": 1}),
    }
    for stage, (df, col, filt) in configs.items():
        mask = pd.Series(True, index=df.index)
        for k, v in filt.items():
            mask &= df[k] == v
        times = [df[mask & (df.gpu_clock == f)][col].values[0] for f in freqs]
        base = times[0]
        eff = [(base / t) / (f / 210) for t, f in zip(times, freqs)]
        ax.plot(freqs, eff, "o-", color=STAGE_COLORS[stage], label=stage, linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect scaling")
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Scaling Efficiency")
    ax.set_title("input=4096, bs=1")
    ax.set_xticks(freqs)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)

    # Small workload - decode bs=1, short seq
    ax = axes[1]
    configs2 = {
        "PA": (pf, "P_Attention", {"tp": 8, "input_len": 128, "batch_size": 1}),
        "PF": (pf, "P_FFN", {"tp": 8, "input_len": 128, "batch_size": 1}),
        "DA": (dc, "D_Attention", {"tp": 8, "input_len": 128, "output_len": 256, "batch_size": 1}),
        "DF": (dc, "D_FFN", {"tp": 8, "input_len": 128, "output_len": 256, "batch_size": 1}),
    }
    for stage, (df, col, filt) in configs2.items():
        mask = pd.Series(True, index=df.index)
        for k, v in filt.items():
            mask &= df[k] == v
        times = [df[mask & (df.gpu_clock == f)][col].values[0] for f in freqs]
        base = times[0]
        eff = [(base / t) / (f / 210) for t, f in zip(times, freqs)]
        ax.plot(freqs, eff, "o-", color=STAGE_COLORS[stage], label=stage, linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect scaling")
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Scaling Efficiency")
    ax.set_title("input=128, bs=1")
    ax.set_xticks(freqs)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=8)

    fig.suptitle("Fig 5.1: Frequency Scaling Efficiency (tp=8)", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_1_scaling_efficiency.png")
    plt.close(fig)
    print("  [done] fig5_1")


# ── Fig 5.2  Frequency Knee-point (DA focus) ──────────────────────────────
def fig5_2(pf, dc):
    fig, ax = plt.subplots(figsize=(9, 5))
    freqs = [210, 540, 870, 1200]
    intervals = ["210→540", "540→870", "870→1200"]

    configs = [
        ("DA bs=1 in=128", dc, "D_Attention", {"tp": 8, "input_len": 128, "output_len": 256, "batch_size": 1}),
        ("DA bs=1 in=4096", dc, "D_Attention", {"tp": 8, "input_len": 4096, "output_len": 256, "batch_size": 1}),
        ("DA bs=256 in=128", dc, "D_Attention", {"tp": 8, "input_len": 128, "output_len": 256, "batch_size": 256}),
        ("DA bs=256 in=4096", dc, "D_Attention", {"tp": 8, "input_len": 4096, "output_len": 256, "batch_size": 256}),
    ]

    colors = ["#1565C0", "#42A5F5", "#E65100", "#FF8A65"]
    x = np.arange(len(intervals))
    w = 0.2
    for idx, (name, df, col, filt) in enumerate(configs):
        mask = pd.Series(True, index=df.index)
        for k, v in filt.items():
            mask &= df[k] == v
        times = [df[mask & (df.gpu_clock == f)][col].values[0] for f in freqs]
        marginals = [(times[i] - times[i + 1]) / times[i] * 100 for i in range(3)]
        ax.bar(x + (idx - 1.5) * w, marginals, w, label=name, color=colors[idx])

    ax.set_xticks(x)
    ax.set_xticklabels(intervals)
    ax.set_xlabel("Frequency Interval")
    ax.set_ylabel("Marginal Speedup (%)")
    ax.set_title("Fig 5.2: Decode Attention Knee-point Analysis (tp=8)")
    ax.legend(fontsize=8)
    ax.axhline(y=5, color="red", linestyle=":", alpha=0.5)
    ax.text(2.3, 5.5, "5% threshold", color="red", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_2_freq_kneepoint.png")
    plt.close(fig)
    print("  [done] fig5_2")


# ── Fig 5.3  A/F Ratio vs Freq for different workload scales ─────────────
def fig5_3(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    freqs = [210, 540, 870, 1200]

    # Prefill
    ax = axes[0]
    workloads = [
        ("in=128,bs=1", 128, 1, "o-", "#E53935"),
        ("in=128,bs=16", 128, 16, "s-", "#FB8C00"),
        ("in=4096,bs=1", 4096, 1, "D-", "#43A047"),
        ("in=4096,bs=16", 4096, 16, "^-", "#1E88E5"),
        ("in=128,bs=256", 128, 256, "v-", "#8E24AA"),
        ("in=4096,bs=256", 4096, 256, "P-", "#546E7A"),
    ]
    for name, il, bs, style, color in workloads:
        ratios = []
        for f in freqs:
            row = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == f) & (pf.batch_size == bs)]
            if len(row) > 0:
                ratios.append(row.P_Attention.values[0] / row.P_FFN.values[0])
            else:
                ratios.append(np.nan)
        if not all(np.isnan(r) for r in ratios):
            ax.plot(freqs, ratios, style, color=color, label=name, linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("PA / PF Ratio")
    ax.set_title("Prefill (tp=8)")
    ax.set_xticks(freqs)
    ax.legend(fontsize=8)

    # Decode
    ax = axes[1]
    workloads_d = [
        ("in=128,bs=1", 128, 1, "o-", "#E53935"),
        ("in=128,bs=16", 128, 16, "s-", "#FB8C00"),
        ("in=4096,bs=1", 4096, 1, "D-", "#43A047"),
        ("in=4096,bs=16", 4096, 16, "^-", "#1E88E5"),
        ("in=128,bs=256", 128, 256, "v-", "#8E24AA"),
        ("in=4096,bs=256", 4096, 256, "P-", "#546E7A"),
    ]
    for name, il, bs, style, color in workloads_d:
        ratios = []
        for f in freqs:
            row = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == f) & (dc.batch_size == bs)]
            if len(row) > 0:
                ratios.append(row.D_Attention.values[0] / row.D_FFN.values[0])
            else:
                ratios.append(np.nan)
        if not all(np.isnan(r) for r in ratios):
            ax.plot(freqs, ratios, style, color=color, label=name, linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("DA / DF Ratio")
    ax.set_title("Decode (tp=8, out=256)")
    ax.set_xticks(freqs)
    ax.legend(fontsize=8)

    fig.suptitle("Fig 5.3: A/F Ratio vs Frequency — Workload Dependence", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_3_af_ratio_vs_freq_workloads.png")
    plt.close(fig)
    print("  [done] fig5_3")


# ── Fig 5.5  Energy Proxy (Time × Freq) ──────────────────────────────────
def fig5_5(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    freqs = [210, 540, 870, 1200]

    # Prefill (tp=1, input=4096, bs=1) - clearest signal
    ax = axes[0]
    configs = {
        "PA": (pf, "P_Attention", {"tp": 1, "input_len": 4096, "batch_size": 1}),
        "PF": (pf, "P_FFN", {"tp": 1, "input_len": 4096, "batch_size": 1}),
    }
    for stage, (df, col, filt) in configs.items():
        mask = pd.Series(True, index=df.index)
        for k, v in filt.items():
            mask &= df[k] == v
        times = [df[mask & (df.gpu_clock == f)][col].values[0] for f in freqs]
        energy = [t * f / 1e6 for t, f in zip(times, freqs)]
        base = energy[0]
        ax.plot(freqs, [e / base for e in energy], "o-", color=STAGE_COLORS[stage], label=stage, linewidth=2)
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Normalized Energy Proxy\n(Time × Freq, baseline=210MHz)")
    ax.set_title("Prefill (tp=1, input=4096)")
    ax.set_xticks(freqs)
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    # Decode (tp=8, input=128, bs=1)
    ax = axes[1]
    configs = {
        "DA": (dc, "D_Attention", {"tp": 8, "input_len": 128, "output_len": 256, "batch_size": 1}),
        "DF": (dc, "D_FFN", {"tp": 8, "input_len": 128, "output_len": 256, "batch_size": 1}),
    }
    for stage, (df, col, filt) in configs.items():
        mask = pd.Series(True, index=df.index)
        for k, v in filt.items():
            mask &= df[k] == v
        times = [df[mask & (df.gpu_clock == f)][col].values[0] for f in freqs]
        energy = [t * f / 1e6 for t, f in zip(times, freqs)]
        base = energy[0]
        ax.plot(freqs, [e / base for e in energy], "o-", color=STAGE_COLORS[stage], label=stage, linewidth=2)
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Normalized Energy Proxy\n(Time × Freq, baseline=210MHz)")
    ax.set_title("Decode (tp=8, input=128, bs=1)")
    ax.set_xticks(freqs)
    ax.legend()
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.3)

    fig.suptitle("Fig 5.5: Energy Proxy — Higher = More Wasted Energy", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_5_energy_proxy.png")
    plt.close(fig)
    print("  [done] fig5_5")


# ── Fig 5.6  Batch-dependent Compute/Memory Characterization ─────────────
def fig5_6(dc):
    fig, ax = plt.subplots(figsize=(9, 6))
    batch_sizes = [1, 16, 256]
    freqs = [210, 540, 870, 1200]

    for bs, marker in zip(batch_sizes, ["o", "s", "D"]):
        da_effs, df_effs = [], []
        for il in [128, 512, 4096]:
            for f_idx in range(len(freqs)):
                if f_idx == 0:
                    continue
                row_lo = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == freqs[0]) & (dc.batch_size == bs)]
                row_hi = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == freqs[f_idx]) & (dc.batch_size == bs)]
                if len(row_lo) > 0 and len(row_hi) > 0:
                    freq_r = freqs[f_idx] / freqs[0]
                    da_sp = row_lo.D_Attention.values[0] / row_hi.D_Attention.values[0]
                    df_sp = row_lo.D_FFN.values[0] / row_hi.D_FFN.values[0]
                    da_effs.append(da_sp / freq_r)
                    df_effs.append(df_sp / freq_r)
        if da_effs:
            ax.scatter(np.mean(da_effs), np.mean(df_effs), s=150, marker=marker,
                       label=f"bs={bs}", zorder=5, edgecolors="black")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("D-Attention Avg Scaling Efficiency")
    ax.set_ylabel("D-FFN Avg Scaling Efficiency")
    ax.set_title("Fig 5.6: Compute/Memory Characterization by Batch Size\n(tp=8, decode)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.fill_between([0, 0.5], [0, 0], [1, 1], alpha=0.05, color="blue")
    ax.fill_between([0.5, 1], [0, 0], [1, 1], alpha=0.05, color="red")
    ax.text(0.15, 0.85, "Memory\nBound", fontsize=10, alpha=0.4, ha="center")
    ax.text(0.85, 0.85, "Compute\nBound", fontsize=10, alpha=0.4, ha="center")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_6_zone_classification.png")
    plt.close(fig)
    print("  [done] fig5_6")


# ── Fig 5.7  TP Scaling Efficiency ────────────────────────────────────────
def fig5_7(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    tps = [1, 2, 4, 8]
    x = np.arange(len(tps))
    w = 0.35

    # Prefill (input=4096, clock=1200, bs=1)
    ax = axes[0]
    pa_times, pf_times = [], []
    for tp in tps:
        row = pf[(pf.tp == tp) & (pf.input_len == 4096) & (pf.gpu_clock == 1200) & (pf.batch_size == 1)]
        if len(row) > 0:
            pa_times.append(row.P_Attention.values[0])
            pf_times.append(row.P_FFN.values[0])
    pa_sp = [pa_times[0] / t for t in pa_times]
    pf_sp = [pf_times[0] / t for t in pf_times]
    ax.bar(x - w / 2, pa_sp, w, label="P-Attention", color=STAGE_COLORS["PA"])
    ax.bar(x + w / 2, pf_sp, w, label="P-FFN", color=STAGE_COLORS["PF"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(tp) for tp in tps])
    ax.set_xlabel("Tensor Parallelism (TP)")
    ax.set_ylabel("Speedup (vs TP=1)")
    ax.set_title("Prefill (input=4096, clock=1200)")
    ax.legend()

    # Decode (input=4096, clock=1200, bs=1, output=256)
    ax = axes[1]
    da_times, df_times = [], []
    for tp in tps:
        row = dc[(dc.tp == tp) & (dc.input_len == 4096) & (dc.output_len == 256) & (dc.gpu_clock == 1200) & (dc.batch_size == 1)]
        if len(row) > 0:
            da_times.append(row.D_Attention.values[0])
            df_times.append(row.D_FFN.values[0])
    da_sp = [da_times[0] / t for t in da_times]
    df_sp = [df_times[0] / t for t in df_times]
    ax.bar(x - w / 2, da_sp, w, label="D-Attention", color=STAGE_COLORS["DA"])
    ax.bar(x + w / 2, df_sp, w, label="D-FFN", color=STAGE_COLORS["DF"])
    ax.set_xticks(x)
    ax.set_xticklabels([str(tp) for tp in tps])
    ax.set_xlabel("Tensor Parallelism (TP)")
    ax.set_ylabel("Speedup (vs TP=1)")
    ax.set_title("Decode (input=4096, clock=1200, out=256)")
    ax.legend()

    fig.suptitle("Fig 5.7: TP Scaling Efficiency — A vs F", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig5_7_tp_scaling.png")
    plt.close(fig)
    print("  [done] fig5_7")


# ── Fig 6.1  Output Length Impact on DA/DF by Batch Size ──────────────────
def fig6_1(dc):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    output_lens = [64, 256, 512]
    batch_configs = [
        (1, "bs=1"),
        (16, "bs=16"),
        (256, "bs=256"),
    ]

    for idx, (bs, title) in enumerate(batch_configs):
        ax = axes[idx]
        for il, color in zip([128, 4096], ["#1565C0", "#E65100"]):
            da_vals, df_vals = [], []
            for ol in output_lens:
                row = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == ol) & (dc.gpu_clock == 870) & (dc.batch_size == bs)]
                if len(row) > 0:
                    da_vals.append(row.D_Attention.values[0])
                    df_vals.append(row.D_FFN.values[0])
                else:
                    da_vals.append(np.nan)
                    df_vals.append(np.nan)
            ax.plot(output_lens, da_vals, "o-", color=color, label=f"DA in={il}", linewidth=2)
            ax.plot(output_lens, df_vals, "s--", color=color, label=f"DF in={il}", linewidth=2, alpha=0.6)
        ax.set_xlabel("Output Length")
        ax.set_ylabel("Execution Time (μs)")
        ax.set_title(f"{title} (tp=8, clock=870)")
        ax.legend(fontsize=7)

    fig.suptitle("Fig 6.1: Output Length Impact by Batch Size", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig6_1_output_len_impact.png")
    plt.close(fig)
    print("  [done] fig6_1")


# ── Fig 7.1  Why PD-only is insufficient — Frequency Dilemma ──────────────
def fig7_1(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    freqs = np.array([210, 540, 870, 1200])

    # --- Left: Prefill (tp=1, input=4096, bs=1) ---
    ax = axes[0]
    pa_times, pf_times = [], []
    for f in freqs:
        row = pf[(pf.tp == 1) & (pf.input_len == 4096) & (pf.gpu_clock == f) & (pf.batch_size == 1)]
        pa_times.append(row.P_Attention.values[0])
        pf_times.append(row.P_FFN.values[0])
    pa_times, pf_times = np.array(pa_times), np.array(pf_times)

    pd_total = pa_times + pf_times
    pa_speedup = pa_times[0] / pa_times
    pf_speedup = pf_times[0] / pf_times
    ideal_speedup = freqs / freqs[0]

    ax.plot(freqs, pf_speedup / ideal_speedup * 100, "s-", color=STAGE_COLORS["PF"],
            label="PF scaling eff.", linewidth=2)
    ax.plot(freqs, pa_speedup / ideal_speedup * 100, "o-", color=STAGE_COLORS["PA"],
            label="PA scaling eff.", linewidth=2)

    ax.fill_between(freqs,
                     pa_speedup / ideal_speedup * 100,
                     pf_speedup / ideal_speedup * 100,
                     alpha=0.15, color="red", label="Wasted freq. on PA\n(PD mode penalty)")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_title("Prefill (tp=1, input=4096, bs=1)\nPD mode: same freq forces PA to waste power")
    ax.set_xticks(freqs)
    ax.set_ylim(40, 110)
    ax.legend(fontsize=8, loc="lower left")

    # Add annotation: how much power PA wastes
    waste_pct = (1 - pa_speedup[-1] / ideal_speedup[-1]) * 100
    ax.annotate(f"PA wastes {waste_pct:.0f}% of\nfrequency investment",
                xy=(1200, pa_speedup[-1] / ideal_speedup[-1] * 100),
                xytext=(800, 55), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontweight="bold")

    # --- Right: Decode (tp=8, input=128, bs=1, output=256) ---
    ax = axes[1]
    da_times, df_times = [], []
    for f in freqs:
        row = dc[(dc.tp == 8) & (dc.input_len == 128) & (dc.output_len == 256) &
                 (dc.gpu_clock == f) & (dc.batch_size == 1)]
        da_times.append(row.D_Attention.values[0])
        df_times.append(row.D_FFN.values[0])
    da_times, df_times = np.array(da_times), np.array(df_times)

    da_speedup = da_times[0] / da_times
    df_speedup = df_times[0] / df_times

    ax.plot(freqs, df_speedup / ideal_speedup * 100, "s-", color=STAGE_COLORS["DF"],
            label="DF scaling eff.", linewidth=2)
    ax.plot(freqs, da_speedup / ideal_speedup * 100, "o-", color=STAGE_COLORS["DA"],
            label="DA scaling eff.", linewidth=2)
    ax.fill_between(freqs,
                     da_speedup / ideal_speedup * 100,
                     df_speedup / ideal_speedup * 100,
                     alpha=0.15, color="red", label="Wasted freq.\n(PD mode penalty)")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("GPU Clock (MHz)")
    ax.set_ylabel("Scaling Efficiency (%)")
    ax.set_title("Decode (tp=8, input=128, bs=1, out=256)\nBoth stages are memory-bound → DVFS has limited effect")
    ax.set_xticks(freqs)
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8, loc="lower left")

    da_waste = (1 - da_speedup[-1] / ideal_speedup[-1]) * 100
    ax.annotate(f"DA wastes {da_waste:.0f}% of\nfrequency investment",
                xy=(1200, da_speedup[-1] / ideal_speedup[-1] * 100),
                xytext=(700, 15), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="red"),
                color="red", fontweight="bold")

    fig.suptitle("Fig 7.1: Why PD-only is Insufficient — Frequency Dilemma", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_1_pd_only_frequency_dilemma.png")
    plt.close(fig)
    print("  [done] fig7_1")


# ── Fig 7.2  Why static AF ratio fails — Pipeline Utilization ────────────
def fig7_2(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Pipeline utilization = min(time_A, time_F) / max(time_A, time_F)
    # Perfect balance = 1.0, worst = 0.0

    # --- Left: Prefill pipeline utilization across workloads ---
    ax = axes[0]
    pf_util = pf.copy()
    pf_util["ratio"] = pf_util["P_Attention"] / pf_util["P_FFN"]
    pf_util["util"] = pf_util[["P_Attention", "P_FFN"]].min(axis=1) / pf_util[["P_Attention", "P_FFN"]].max(axis=1)

    static_ratios = [0.5, 0.75, 1.0]
    bins = np.linspace(0, 1, 21)
    ax.hist(pf_util["util"].values, bins=bins, alpha=0.7, color="#42A5F5",
            edgecolor="white", label="Actual A/F balance\nacross all configs")
    ax.axvline(x=pf_util["util"].median(), color="red", linestyle="-", linewidth=2,
               label=f"Median = {pf_util['util'].median():.2f}")
    ax.axvline(x=1.0, color="green", linestyle="--", linewidth=2, alpha=0.5,
               label="Perfect balance (1.0)")
    ax.set_xlabel("Pipeline Utilization  min(A,F)/max(A,F)")
    ax.set_ylabel("Count (# configurations)")
    ax.set_title("Prefill: Pipeline Balance Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)

    pct_below_50 = (pf_util["util"] < 0.5).sum() / len(pf_util) * 100
    pct_below_70 = (pf_util["util"] < 0.7).sum() / len(pf_util) * 100
    ax.text(0.03, 0.55, f"{pct_below_50:.0f}% configs < 50% util.\n{pct_below_70:.0f}% configs < 70% util.",
            transform=ax.transAxes, fontsize=9, color="red",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Right: Decode pipeline utilization ---
    ax = axes[1]
    dc_util = dc.copy()
    dc_util["ratio"] = dc_util["D_Attention"] / dc_util["D_FFN"]
    dc_util["util"] = dc_util[["D_Attention", "D_FFN"]].min(axis=1) / dc_util[["D_Attention", "D_FFN"]].max(axis=1)

    ax.hist(dc_util["util"].values, bins=bins, alpha=0.7, color="#66BB6A",
            edgecolor="white", label="Actual A/F balance\nacross all configs")
    ax.axvline(x=dc_util["util"].median(), color="red", linestyle="-", linewidth=2,
               label=f"Median = {dc_util['util'].median():.2f}")
    ax.axvline(x=1.0, color="green", linestyle="--", linewidth=2, alpha=0.5,
               label="Perfect balance (1.0)")
    ax.set_xlabel("Pipeline Utilization  min(A,F)/max(A,F)")
    ax.set_ylabel("Count (# configurations)")
    ax.set_title("Decode: Pipeline Balance Distribution")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1.05)

    pct_below_50 = (dc_util["util"] < 0.5).sum() / len(dc_util) * 100
    pct_below_70 = (dc_util["util"] < 0.7).sum() / len(dc_util) * 100
    ax.text(0.03, 0.55, f"{pct_below_50:.0f}% configs < 50% util.\n{pct_below_70:.0f}% configs < 70% util.",
            transform=ax.transAxes, fontsize=9, color="red",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Fig 7.2: Why Static A/F Ratio Fails — Pipeline Utilization Distribution", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_2_static_af_pipeline_util.png")
    plt.close(fig)
    print("  [done] fig7_2")


# ── Fig 7.3  Three-way comparison: PD-only vs AF-static vs AF-dynamic ────
def fig7_3(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    freqs = np.array([210, 540, 870, 1200])

    # --- Prefill comparison ---
    ax = axes[0]
    configs = [
        ("in=128\nbs=1", 8, 128, 1),
        ("in=128\nbs=16", 8, 128, 16),
        ("in=512\nbs=1", 8, 512, 1),
        ("in=512\nbs=16", 8, 512, 16),
        ("in=4096\nbs=1", 8, 4096, 1),
        ("in=4096\nbs=16", 8, 4096, 16),
        ("in=128\nbs=256", 8, 128, 256),
        ("in=4096\nbs=256", 8, 4096, 256),
    ]

    labels, pd_energy, af_static_energy, af_dynamic_energy = [], [], [], []
    for name, tp, il, bs in configs:
        pa_t = {}
        pf_t = {}
        for f in freqs:
            row = pf[(pf.tp == tp) & (pf.input_len == il) & (pf.gpu_clock == f) & (pf.batch_size == bs)]
            if len(row) > 0:
                pa_t[f] = row.P_Attention.values[0]
                pf_t[f] = row.P_FFN.values[0]
        if 1200 not in pa_t:
            continue
        labels.append(name)

        pd_e = (pa_t[1200] + pf_t[1200]) * 1200
        af_static_e = pa_t[870] * 870 + pf_t[1200] * 1200

        best_af_e = float("inf")
        for fa in freqs:
            for ff in freqs:
                if fa in pa_t and ff in pf_t:
                    e = pa_t[fa] * fa + pf_t[ff] * ff
                    lat = pa_t[fa] + pf_t[ff]
                    pd_lat = pa_t[1200] + pf_t[1200]
                    if lat <= pd_lat * 1.05:
                        best_af_e = min(best_af_e, e)
        if best_af_e == float("inf"):
            best_af_e = af_static_e

        pd_energy.append(pd_e)
        af_static_energy.append(af_static_e)
        af_dynamic_energy.append(best_af_e)

    x = np.arange(len(labels))
    w = 0.25
    pd_norm = np.array(pd_energy)
    ax.bar(x - w, pd_norm / pd_norm * 100, w, label="PD-only\n(same freq=1200)", color="#E53935")
    ax.bar(x, np.array(af_static_energy) / pd_norm * 100, w,
           label="AF-static\n(PA=870, PF=1200)", color="#FB8C00")
    ax.bar(x + w, np.array(af_dynamic_energy) / pd_norm * 100, w,
           label="AF-dynamic\n(optimal per-workload)", color="#43A047")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Relative Energy (PD-only = 100%)")
    ax.set_title("Prefill Energy Comparison (tp=8)\n(same latency budget, lower = better)")
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(60, 105)

    # --- Decode comparison ---
    ax = axes[1]
    configs_d = [
        ("in=128\nbs=1", 8, 128, 256, 1),
        ("in=128\nbs=16", 8, 128, 256, 16),
        ("in=512\nbs=1", 8, 512, 256, 1),
        ("in=512\nbs=16", 8, 512, 256, 16),
        ("in=4096\nbs=1", 8, 4096, 256, 1),
        ("in=4096\nbs=16", 8, 4096, 256, 16),
        ("in=128\nbs=256", 8, 128, 256, 256),
        ("in=4096\nbs=256", 8, 4096, 256, 256),
    ]

    labels, pd_energy, af_static_energy, af_dynamic_energy = [], [], [], []
    for name, tp, il, ol, bs in configs_d:
        da_t = {}
        df_t = {}
        for f in freqs:
            row = dc[(dc.tp == tp) & (dc.input_len == il) & (dc.output_len == ol) &
                     (dc.gpu_clock == f) & (dc.batch_size == bs)]
            if len(row) > 0:
                da_t[f] = row.D_Attention.values[0]
                df_t[f] = row.D_FFN.values[0]
        if 1200 not in da_t:
            continue
        labels.append(name)

        pd_e = (da_t[1200] + df_t[1200]) * 1200
        af_static_e = da_t[870] * 870 + df_t[1200] * 1200

        best_af_e = float("inf")
        for fa in freqs:
            for ff in freqs:
                if fa in da_t and ff in df_t:
                    e = da_t[fa] * fa + df_t[ff] * ff
                    lat = da_t[fa] + df_t[ff]
                    pd_lat = da_t[1200] + df_t[1200]
                    if lat <= pd_lat * 1.05:
                        best_af_e = min(best_af_e, e)
        if best_af_e == float("inf"):
            best_af_e = af_static_e

        pd_energy.append(pd_e)
        af_static_energy.append(af_static_e)
        af_dynamic_energy.append(best_af_e)

    x = np.arange(len(labels))
    pd_norm = np.array(pd_energy)
    ax.bar(x - w, pd_norm / pd_norm * 100, w, label="PD-only\n(same freq=1200)", color="#E53935")
    ax.bar(x, np.array(af_static_energy) / pd_norm * 100, w,
           label="AF-static\n(DA=870, DF=1200)", color="#FB8C00")
    ax.bar(x + w, np.array(af_dynamic_energy) / pd_norm * 100, w,
           label="AF-dynamic\n(optimal per-workload)", color="#43A047")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Relative Energy (PD-only = 100%)")
    ax.set_title("Decode Energy Comparison (tp=8, out=256)\n(same latency budget, lower = better)")
    ax.legend(fontsize=7, loc="upper right")
    ax.axhline(y=100, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylim(60, 105)

    fig.suptitle("Fig 7.3: PD-only vs AF-static vs AF-dynamic Energy Comparison", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_3_three_way_comparison.png")
    plt.close(fig)
    print("  [done] fig7_3")


# ── Fig 7.4  PD-only dilemma by Zone — Decode's batch-dependent severity ──
def fig7_4(pf, dc):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    freqs = np.array([210, 540, 870, 1200])

    # --- Left: Prefill — scaling eff gap across workloads ---
    ax = axes[0]
    configs_p = [
        ("in=128\nbs=1", 128, 1),
        ("in=128\nbs=16", 128, 16),
        ("in=128\nbs=256", 128, 256),
        ("in=4096\nbs=1", 4096, 1),
        ("in=4096\nbs=16", 4096, 16),
        ("in=4096\nbs=256", 4096, 256),
    ]
    labels, pa_effs, pf_effs = [], [], []
    for name, il, bs in configs_p:
        row_lo = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == 210) & (pf.batch_size == bs)]
        row_hi = pf[(pf.tp == 8) & (pf.input_len == il) & (pf.gpu_clock == 1200) & (pf.batch_size == bs)]
        if len(row_lo) > 0 and len(row_hi) > 0:
            ideal_sp = 1200 / 210
            pa_sp = row_lo.P_Attention.values[0] / row_hi.P_Attention.values[0]
            pf_sp = row_lo.P_FFN.values[0] / row_hi.P_FFN.values[0]
            labels.append(name)
            pa_effs.append(pa_sp / ideal_sp * 100)
            pf_effs.append(pf_sp / ideal_sp * 100)
    x = np.arange(len(labels))
    w = 0.35
    bars_pf = ax.bar(x - w / 2, pf_effs, w, label="PF scaling eff.", color=STAGE_COLORS["PF"], alpha=0.85)
    bars_pa = ax.bar(x + w / 2, pa_effs, w, label="PA scaling eff.", color=STAGE_COLORS["PA"], alpha=0.85)
    for i in range(len(x)):
        gap = pf_effs[i] - pa_effs[i]
        if gap > 0:
            ax.annotate(f"gap\n{gap:.0f}%", xy=(x[i], max(pa_effs[i], pf_effs[i]) + 1),
                        fontsize=7, ha="center", color="red", fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Scaling Efficiency at 1200MHz (%)")
    ax.set_title("Prefill: PA vs PF frequency efficiency gap\n(tp=8, gap = PD mode waste)")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 60)

    # --- Right: Decode — scaling eff gap by batch size ---
    ax = axes[1]
    configs_d = [
        ("in=128\nbs=1", 128, 1, "Zone 1\n(both mem.)"),
        ("in=128\nbs=16", 128, 16, "Zone 2\n(A mem, F comp)"),
        ("in=128\nbs=256", 128, 256, "Zone 2→3"),
        ("in=4096\nbs=1", 4096, 1, "Zone 1"),
        ("in=4096\nbs=16", 4096, 16, "Zone 2"),
        ("in=4096\nbs=256", 4096, 256, "Zone 3\n(both comp.)"),
    ]
    labels, da_effs, df_effs, zone_labels = [], [], [], []
    for name, il, bs, zone in configs_d:
        row_lo = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == 210) & (dc.batch_size == bs)]
        row_hi = dc[(dc.tp == 8) & (dc.input_len == il) & (dc.output_len == 256) & (dc.gpu_clock == 1200) & (dc.batch_size == bs)]
        if len(row_lo) > 0 and len(row_hi) > 0:
            ideal_sp = 1200 / 210
            da_sp = row_lo.D_Attention.values[0] / row_hi.D_Attention.values[0]
            df_sp = row_lo.D_FFN.values[0] / row_hi.D_FFN.values[0]
            labels.append(name)
            da_effs.append(da_sp / ideal_sp * 100)
            df_effs.append(df_sp / ideal_sp * 100)
            zone_labels.append(zone)
    x = np.arange(len(labels))
    ax.bar(x - w / 2, df_effs, w, label="DF scaling eff.", color=STAGE_COLORS["DF"], alpha=0.85)
    ax.bar(x + w / 2, da_effs, w, label="DA scaling eff.", color=STAGE_COLORS["DA"], alpha=0.85)
    for i in range(len(x)):
        gap = abs(df_effs[i] - da_effs[i])
        color = "red" if gap > 3 else "gray"
        ax.annotate(f"gap\n{gap:.0f}%", xy=(x[i], max(da_effs[i], df_effs[i]) + 1),
                    fontsize=7, ha="center", color=color, fontweight="bold")
    # Zone annotations at bottom
    for i, zl in enumerate(zone_labels):
        ax.text(x[i], -5, zl, fontsize=6, ha="center", va="top", color="#555555", style="italic")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Scaling Efficiency at 1200MHz (%)")
    ax.set_title("Decode: DA vs DF frequency efficiency gap\n(tp=8, out=256, gap = PD mode waste)")
    ax.legend(fontsize=8)
    ax.set_ylim(-12, 60)

    fig.suptitle("Fig 7.4: PD-only Frequency Dilemma Severity — Per Phase × Workload", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_4_pd_dilemma_by_zone.png")
    plt.close(fig)
    print("  [done] fig7_4")


# ── Fig 7.5  Static AF — Optimal A/(A+F) fraction per config ─────────────
def fig7_5(pf, dc):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # --- Top: Prefill — ALL TP levels, sorted by A/(A+F) ---
    ax = axes[0]
    p_configs = []
    for _, row in pf.iterrows():
        label = f"tp={int(row.tp)}\nin={int(row.input_len)}\nbs={int(row.batch_size)}\n{int(row.gpu_clock)}MHz"
        frac = row.P_Attention / (row.P_Attention + row.P_FFN)
        p_configs.append((label, frac, int(row.tp)))

    p_configs.sort(key=lambda c: c[1])
    labels = [c[0] for c in p_configs]
    fracs = [c[1] for c in p_configs]
    tps = [c[2] for c in p_configs]
    tp_colors = {1: "#E53935", 2: "#FB8C00", 4: "#43A047", 8: "#1E88E5"}
    colors_p = [tp_colors[tp] for tp in tps]

    x = np.arange(len(labels))
    ax.bar(x, fracs, color=colors_p, edgecolor="white", linewidth=0.3)
    ax.axhline(y=0.5, color="black", linestyle="--", linewidth=2, label="A:F = 1:1 (balance)")
    ax.axhline(y=0.33, color="gray", linestyle=":", linewidth=1.5, label="A:F = 1:2")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=tp_colors[1], label="tp=1"),
        Patch(facecolor=tp_colors[2], label="tp=2"),
        Patch(facecolor=tp_colors[4], label="tp=4"),
        Patch(facecolor=tp_colors[8], label="tp=8"),
        plt.Line2D([0], [0], color="black", linestyle="--", label="A=F (ratio=1:1)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left", ncol=3)
    ax.set_xticks(x[::3])
    ax.set_xticklabels([labels[i] for i in range(0, len(labels), 3)], fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("Attention Fraction  A/(A+F)")
    n_above = sum(1 for f in fracs if f > 0.5)
    ax.set_title(f"Prefill (ALL TPs): A/(A+F) from {min(fracs):.2f} to {max(fracs):.2f} — "
                 f"{n_above}/{len(fracs)} configs ({n_above*100//len(fracs)}%) have PA > PF!")
    ax.set_ylim(0, 0.9)
    ax.fill_between([0, len(fracs)], 0.5, 0.9, alpha=0.05, color="red")
    ax.text(0.98, 0.95,
            f"Range: {min(fracs):.2f} – {max(fracs):.2f}\n"
            f"Crosses 0.5: {n_above}/{len(fracs)} ({n_above*100//len(fracs)}%)\n"
            f"tp=1: 58%, tp=2: 38%, tp=4: 33%, tp=8: 0%\n"
            f"High freq + short seq + low TP → PA bottleneck",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    # --- Bottom: Decode — optimal A/(A+F) across configs ---
    ax = axes[1]
    d_configs = []
    for _, row in dc[dc.tp == 8].iterrows():
        label = f"in={int(row.input_len)}\nbs={int(row.batch_size)}\n{int(row.gpu_clock)}MHz"
        frac = row.D_Attention / (row.D_Attention + row.D_FFN)
        d_configs.append((label, frac, row.batch_size, row.input_len, row.gpu_clock))

    d_configs.sort(key=lambda c: c[1])
    labels = [c[0] for c in d_configs]
    fracs = [c[1] for c in d_configs]
    colors_d = ["#2E7D32" if f < 0.45 else "#D32F2F" if f > 0.55 else "#FDD835" for f in fracs]

    x = np.arange(len(labels))
    ax.bar(x, fracs, color=colors_d, edgecolor="white", linewidth=0.5)
    ax.axhline(y=0.5, color="red", linestyle="--", linewidth=2, label="A:F = 1:1 (balance)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=5, rotation=45, ha="right")
    ax.set_ylabel("Attention Fraction  A/(A+F)")
    ax.set_title("Decode: Optimal A/(A+F) varies from {:.2f} to {:.2f} — crosses 0.5 (bottleneck reversal!)".format(
        min(fracs), max(fracs)))
    ax.legend(fontsize=8, loc="upper left")
    ax.set_ylim(0.3, 0.65)

    n_above = sum(1 for f in fracs if f > 0.5)
    n_total = len(fracs)
    ax.text(0.98, 0.95,
            f"Range: {min(fracs):.2f} – {max(fracs):.2f}\n"
            f"Reversal: {n_above}/{n_total} configs have A > F\n"
            f"No static ratio works for both sides!",
            transform=ax.transAxes, fontsize=8, ha="right", va="top",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle("Fig 7.5: Why Static A/F Ratio Fails — Per-Phase Optimal Fraction", fontsize=14, y=1.01)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig7_5_static_af_optimal_fraction.png")
    plt.close(fig)
    print("  [done] fig7_5")


def _orthogonal_path_vertical_first(xs, ys):
    """Connect consecutive points with L-shaped segments: vertical then horizontal."""
    if xs is None or len(xs) < 2:
        return list(xs) if xs is not None else [], list(ys) if ys is not None else []
    ox, oy = [xs[0]], [ys[0]]
    for i in range(len(xs) - 1):
        ox.extend([xs[i], xs[i + 1]])
        oy.extend([ys[i + 1], ys[i + 1]])
    return ox, oy


def _pareto_le(points):
    """Non-dominated points when minimizing both latency L and energy proxy E."""
    pts = [(float(a), float(b)) for a, b in points if np.isfinite(a) and np.isfinite(b)]
    eff = []
    for p in pts:
        dominated = False
        for q in pts:
            if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                dominated = True
                break
        if not dominated:
            eff.append(p)
    return sorted(eff, key=lambda x: (x[0], x[1]))


def _unified_le_series_prefill(pfdf, il, bs, tp, freqs, exp=2):
    Ls, Es = [], []
    for f in freqs:
        row = pfdf[(pfdf.tp == tp) & (pfdf.input_len == il) & (pfdf.gpu_clock == f) & (pfdf.batch_size == bs)]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        L = float(r.P_Attention + r.P_FFN)
        Es.append(f**exp * L / 1e6)
        Ls.append(L)
    return Ls, Es


def _unified_le_series_decode(dcdf, il, bs, tp, out_len, freqs, exp=2):
    Ls, Es = [], []
    for f in freqs:
        row = dcdf[
            (dcdf.tp == tp)
            & (dcdf.input_len == il)
            & (dcdf.output_len == out_len)
            & (dcdf.gpu_clock == f)
            & (dcdf.batch_size == bs)
        ]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        L = float(r.D_Attention + r.D_FFN)
        Es.append(f**exp * L / 1e6)
        Ls.append(L)
    return Ls, Es


def _af_grid_prefill(pfdf, il, bs, tp, freqs, exp=2):
    pts = []
    for fa in freqs:
        for ff in freqs:
            ra = pfdf[(pfdf.tp == tp) & (pfdf.input_len == il) & (pfdf.gpu_clock == fa) & (pfdf.batch_size == bs)]
            rf = pfdf[(pfdf.tp == tp) & (pfdf.input_len == il) & (pfdf.gpu_clock == ff) & (pfdf.batch_size == bs)]
            if len(ra) == 0 or len(rf) == 0:
                continue
            pa = float(ra.iloc[0].P_Attention)
            pff = float(rf.iloc[0].P_FFN)
            L = pa + pff
            E = (fa**exp * pa + ff**exp * pff) / 1e6
            pts.append((L, E))
    return pts


def _af_grid_decode(dcdf, il, bs, tp, out_len, freqs, exp=2):
    pts = []
    for fa in freqs:
        for ff in freqs:
            base = (
                (dcdf.tp == tp)
                & (dcdf.input_len == il)
                & (dcdf.output_len == out_len)
                & (dcdf.batch_size == bs)
            )
            ra = dcdf[base & (dcdf.gpu_clock == fa)]
            rf = dcdf[base & (dcdf.gpu_clock == ff)]
            if len(ra) == 0 or len(rf) == 0:
                continue
            da = float(ra.iloc[0].D_Attention)
            dff = float(rf.iloc[0].D_FFN)
            L = da + dff
            E = (fa**exp * da + ff**exp * dff) / 1e6
            pts.append((L, E))
    return pts


_FIG8_ABS_CFGS = [
    (128, 16, "in=128, bs=16"),
    (4096, 1, "in=4096, bs=1"),
    (4096, 16, "in=4096, bs=16"),
]


# ── Fig 8: multi-TP, dual energy model (f^exp * L) ─────────────────────────
_FIG8_TPS = [2, 4, 8]


def _fig8_draw_unified(ax, Ls, Es, freqs, title, exp, ngpus):
    """One subplot: unified-f trajectory with SLA."""
    if Ls is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=9)
        ax.set_title(title, fontsize=9)
        return
    Es = [e * ngpus for e in Es]
    ox, oy = _orthogonal_path_vertical_first(Ls, Es)
    ax.plot(ox, oy, "-", color="#888", lw=1.4, zorder=2, alpha=0.7)
    for i, (f, l, e) in enumerate(zip(freqs, Ls, Es)):
        ax.scatter(l, e, color=FREQ_COLORS[f], s=70, zorder=5,
                   edgecolors="black", linewidths=0.7)
        ax.annotate(f" {f}", (l, e), fontsize=6.5, fontweight="bold",
                    color=FREQ_COLORS[f], va="bottom")
    sla_l = (Ls[2] + Ls[3]) / 2
    ax.axvline(sla_l, color="#FF6F00", ls="--", lw=1.5, alpha=0.8, zorder=1)
    ax.axvspan(0, sla_l, alpha=0.05, color="#FFE0B2", zorder=0)
    feas = [(l, e, f) for l, e, f in zip(Ls, Es, freqs) if l <= sla_l * 1.001]
    if feas:
        bl, be, bf = min(feas, key=lambda t: t[1])
        ax.plot(bl, be, "*", color="#D32F2F", ms=16, zorder=7,
                markeredgecolor="black", markeredgewidth=0.8)
        ax.text(0.97, 0.97,
                f"SLA={sla_l:.0f}\u03bcs\n\u2605 {bf}MHz E={be:.0f}",
                transform=ax.transAxes, fontsize=6.5, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", alpha=0.9))
    ftag = "f\u00b2" if exp == 2 else "f"
    ax.set_xlabel("Latency (\u03bcs)", fontsize=8)
    ax.set_ylabel(f"E = {ftag}\u00b7L\u00b7{ngpus}/10\u2076", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)


def _fig8_draw_af(ax, uLs, uEs, grid_raw, freqs, title, exp, ngpus):
    """One subplot: AF grid + Pareto + unified ref with SLA."""
    if uLs is None:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", fontsize=9)
        ax.set_title(title, fontsize=9)
        return
    uEg = [e * ngpus for e in uEs]
    grid = [(l, e * ngpus) for l, e in grid_raw] if grid_raw else []

    ux, uy = _orthogonal_path_vertical_first(uLs, uEg)
    ax.plot(ux, uy, "--", color="#999", lw=1.1, alpha=0.5, zorder=1, label="unified")
    for f, l, e in zip(freqs, uLs, uEg):
        ax.scatter(l, e, color=FREQ_COLORS[f], s=35, zorder=3, alpha=0.45,
                   edgecolors="black", linewidths=0.4)
    if grid:
        gx, gy = zip(*grid)
        ax.scatter(gx, gy, color="#1976D2", s=28, alpha=0.35, zorder=2,
                   edgecolors="none", label=f"AF ({len(grid)})")
        par = _pareto_le(grid)
        if len(par) >= 2:
            ax.plot([p[0] for p in par], [p[1] for p in par],
                    "-", color="#1565C0", lw=2.2, alpha=0.9, zorder=4, label="Pareto")

    sla_l = (uLs[2] + uLs[3]) / 2
    ax.axvline(sla_l, color="#FF6F00", ls="--", lw=1.5, alpha=0.8, zorder=1)
    ax.axvspan(0, sla_l, alpha=0.05, color="#FFE0B2", zorder=0)

    feas_u = [(l, e) for l, e in zip(uLs, uEg) if l <= sla_l * 1.001]
    feas_a = [(l, e) for l, e in grid if l <= sla_l * 1.001]
    bu = min(feas_u, key=lambda t: t[1]) if feas_u else None
    ba = min(feas_a, key=lambda t: t[1]) if feas_a else None

    info = f"SLA={sla_l:.0f}\u03bcs\n"
    if ba:
        ax.plot(ba[0], ba[1], "*", color="#D32F2F", ms=16, zorder=7,
                markeredgecolor="black", markeredgewidth=0.8)
    if bu and ba:
        if bu[1] > ba[1] * 1.005:
            pct = (1 - ba[1] / bu[1]) * 100
            info += f"save {pct:.1f}%\n"
            ax.annotate("", xy=ba, xytext=bu,
                        arrowprops=dict(arrowstyle="->", color="#C62828", lw=1.5), zorder=6)
        info += f"({len(feas_u)}\u2192{len(feas_a)} pts)"
    ax.text(0.97, 0.97, info, transform=ax.transAxes, fontsize=6.5,
            va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#888", alpha=0.9))
    ftag = "f\u00b2" if exp == 2 else "f"
    ax.set_xlabel("Latency (\u03bcs)", fontsize=8)
    ax.set_ylabel(f"E = {ftag}\u00b7L\u00b7{ngpus}/10\u2076", fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    ax.legend(fontsize=6, loc="lower left")


def _fig8_gen(pf, dc, exp, tag):
    """Generate unified + AF figures for all TPs with given energy exponent."""
    tps = _FIG8_TPS
    cfgs = _FIG8_ABS_CFGS
    freqs = [210, 540, 870, 1200]
    out_len = 256
    nrows = len(tps) * 2
    ncols = len(cfgs)
    fig_u, axes_u = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows))
    fig_a, axes_a = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows))

    for ti, tp in enumerate(tps):
        ngpus = tp
        for col, (il, bs, wl) in enumerate(cfgs):
            row_p = ti
            row_d = len(tps) + ti
            Lp, Ep = _unified_le_series_prefill(pf, il, bs, tp, freqs, exp)
            gp = _af_grid_prefill(pf, il, bs, tp, freqs, exp)
            _fig8_draw_unified(axes_u[row_p, col], Lp, Ep, freqs,
                               f"Prefill tp={tp} \u2014 {wl}", exp, ngpus)
            _fig8_draw_af(axes_a[row_p, col], Lp, Ep, gp, freqs,
                          f"Prefill tp={tp} \u2014 {wl}", exp, ngpus)

            Ld, Ed = _unified_le_series_decode(dc, il, bs, tp, out_len, freqs, exp)
            gd = _af_grid_decode(dc, il, bs, tp, out_len, freqs, exp)
            _fig8_draw_unified(axes_u[row_d, col], Ld, Ed, freqs,
                               f"Decode tp={tp} \u2014 {wl}", exp, ngpus)
            _fig8_draw_af(axes_a[row_d, col], Ld, Ed, gd, freqs,
                          f"Decode tp={tp} \u2014 {wl}", exp, ngpus)

    ftag = "f\u00b2" if exp == 2 else "f"
    fig_u.suptitle(
        f"No AF \u2014 unified DVFS across tp=2,4,8  (E \u221d {ftag}\u00b7L)",
        fontsize=14, y=1.005)
    fig_a.suptitle(
        f"With AF \u2014 Pareto vs unified across tp=2,4,8  (E \u221d {ftag}\u00b7L)",
        fontsize=14, y=1.005)
    for fg in (fig_u, fig_a):
        fg.tight_layout(rect=[0, 0, 1, 0.99])
    fig_u.savefig(FIG_DIR / f"fig8_unified_{tag}.png", dpi=150)
    fig_a.savefig(FIG_DIR / f"fig8_af_{tag}.png", dpi=150)
    plt.close(fig_u)
    plt.close(fig_a)
    print(f"  [done] fig8 unified+af ({tag})")


def _fig8_savings_pct(pf, dc, exp):
    """Compute AF savings % for every (phase, tp, workload) combo."""
    freqs = [210, 540, 870, 1200]
    out_len = 256
    all_tps = [1, 2, 4, 8]
    wl_list = [
        (128, 1), (512, 1), (4096, 1),
        (128, 16), (512, 16), (4096, 16),
    ]
    results = {}
    for phase in ("Prefill", "Decode"):
        for tp in all_tps:
            for il, bs in wl_list:
                ngpus = tp
                if phase == "Prefill":
                    uL, uE = _unified_le_series_prefill(pf, il, bs, tp, freqs, exp)
                    grid = _af_grid_prefill(pf, il, bs, tp, freqs, exp)
                else:
                    uL, uE = _unified_le_series_decode(dc, il, bs, tp, out_len, freqs, exp)
                    grid = _af_grid_decode(dc, il, bs, tp, out_len, freqs, exp)
                if uL is None or not grid:
                    results[(phase, tp, il, bs)] = None
                    continue
                uE = [e * ngpus for e in uE]
                grid = [(l, e * ngpus) for l, e in grid]
                sla = (uL[2] + uL[3]) / 2
                fu = [(l, e) for l, e in zip(uL, uE) if l <= sla * 1.001]
                fa = [(l, e) for l, e in grid if l <= sla * 1.001]
                bu = min(fu, key=lambda t: t[1]) if fu else None
                ba = min(fa, key=lambda t: t[1]) if fa else None
                if bu and ba and bu[1] > ba[1] * 1.001:
                    results[(phase, tp, il, bs)] = (1 - ba[1] / bu[1]) * 100
                else:
                    results[(phase, tp, il, bs)] = 0.0
    return results, wl_list, all_tps


def _fig8_summary(pf, dc):
    """Summary bar chart: AF savings % across tp=1,2,4,8 and workloads."""
    tp_colors = {1: "#E53935", 2: "#FB8C00", 4: "#43A047", 8: "#1E88E5"}
    fig, axes = plt.subplots(2, 2, figsize=(17, 10))

    for col, (exp, etag) in enumerate([(2, "f\u00b2\u00b7L"), (1, "f\u00b7L")]):
        res, wl_list, all_tps = _fig8_savings_pct(pf, dc, exp)
        wl_labels = [f"il={il}\nbs={bs}" for il, bs in wl_list]
        x = np.arange(len(wl_list))
        w = 0.19

        for row, phase in enumerate(["Prefill", "Decode"]):
            ax = axes[row, col]
            for ti, tp in enumerate(all_tps):
                vals = []
                for il, bs in wl_list:
                    v = res.get((phase, tp, il, bs))
                    vals.append(v if v is not None else -1)
                offset = (ti - (len(all_tps) - 1) / 2) * w
                colors = []
                heights = []
                for v in vals:
                    if v is None or v < 0:
                        heights.append(0)
                        colors.append("#E0E0E0")
                    else:
                        heights.append(v)
                        colors.append(tp_colors[tp])
                bars = ax.bar(x + offset, heights, w * 0.88,
                              color=colors, edgecolor="black", linewidth=0.4,
                              label=f"tp={tp}" if row == 0 and col == 0 else "")
                for bar, v in zip(bars, vals):
                    if v is not None and v > 0.5:
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.4,
                                f"{v:.0f}", ha="center", va="bottom", fontsize=6.5,
                                fontweight="bold")

            ax.set_xticks(x)
            ax.set_xticklabels(wl_labels, fontsize=8)
            ax.set_ylabel("AF energy saving (%)", fontsize=10)
            ax.set_title(f"{phase}  (E \u221d {etag})", fontsize=12)
            ax.grid(axis="y", alpha=0.25)
            ymax = max(ax.get_ylim()[1], 5)
            ax.set_ylim(0, ymax * 1.12)

    handles = [matplotlib.patches.Patch(facecolor=tp_colors[tp], edgecolor="black",
               linewidth=0.5, label=f"tp={tp}") for tp in [1, 2, 4, 8]]
    handles.append(matplotlib.patches.Patch(facecolor="#E0E0E0", edgecolor="black",
                   linewidth=0.5, label="no data"))
    fig.legend(handles=handles, fontsize=10, loc="lower center", ncol=5,
               title="Tensor Parallelism", title_fontsize=11)
    fig.suptitle(
        "AF separation: energy saving % across tp=1,2,4,8\n"
        "SLA = (L\u2088\u2087\u2080+L\u2081\u2082\u2080\u2080)/2  |  grey = no data for that (tp, workload)",
        fontsize=13, y=1.02,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])
    fig.savefig(FIG_DIR / "fig8_summary_savings.png", dpi=150)
    plt.close(fig)
    print("  [done] fig8_summary")


def fig8(pf, dc):
    """Generate all Fig-8 variants: f^2*L and f*L, multi-TP."""
    _fig8_gen(pf, dc, exp=2, tag="f2")
    _fig8_gen(pf, dc, exp=1, tag="f1")
    _fig8_summary(pf, dc)


# ═══════════════════════════════════════════════════════════════════════════
def main():
    print("Loading data...")
    pf, dc = load_data()
    print(f"  Prefill: {len(pf)} rows, Decode: {len(dc)} rows")
    print(f"  Saving figures to {FIG_DIR}\n")

    print("Generating figures...")
    fig1_1(pf, dc)
    fig1_3(pf, dc)
    fig1_2(pf, dc)
    fig2_1(pf, dc)
    fig2_2(pf, dc)
    fig2_3(pf, dc)
    fig2_4(pf, dc)
    fig3_1_3_2(pf, dc)
    fig3_3(pf, dc)
    fig3_4(pf, dc)
    fig4_1(pf, dc)
    fig4_2(pf, dc)
    fig5_1(pf, dc)
    fig5_2(pf, dc)
    fig5_3(pf, dc)
    fig5_5(pf, dc)
    fig5_6(dc)
    fig5_7(pf, dc)
    fig6_1(dc)
    fig7_1(pf, dc)
    fig7_2(pf, dc)
    fig7_3(pf, dc)
    fig7_4(pf, dc)
    fig7_5(pf, dc)
    fig8(pf, dc)

    print(f"\nAll {len(list(FIG_DIR.glob('*.png')))} figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
