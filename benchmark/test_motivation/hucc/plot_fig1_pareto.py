#!/usr/bin/env python3
"""
Figure 1: Pareto frontier comparison — Unified DVFS vs AF grid.
Shows 2 representative configs side-by-side, with AFlex/OptB optimal points annotated.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.family": "sans-serif",
})

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "decode_data.txt"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FREQS = [210, 450, 690, 930, 1170, 1410]
MAX_FREQ = 1410
FREQ_COLORS = {
    210: "#E53935", 450: "#FB8C00", 690: "#43A047",
    930: "#8E24AA", 1170: "#1E88E5", 1410: "#00897B",
}

# Representative configs to show
CONFIGS = [
    {"tp": 4, "il": 512, "ol": 64, "bs": 16, "title": "tp=4, bs=16, il=512\n(sweet spot)"},
    {"tp": 8, "il": 1024, "ol": 64, "bs": 32, "title": "tp=8, bs=32, il=1024\n(sweet spot)"},
]


def get_unified_points(df, tp, il, ol, bs):
    """Unified DVFS: same freq for A & F."""
    points = []
    for f in FREQS:
        row = df[(df.tp == tp) & (df.input_len == il) & (df.output_len == ol)
                 & (df.gpu_clock == f) & (df.batch_size == bs)]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lat = float(r.D_A_lat) + float(r.D_F_lat)
        eng = float(r.D_A_energy) + float(r.D_F_energy)
        points.append((lat, eng, f))
    return points


def get_af_grid(df, tp, il, ol, bs):
    """AF grid: all (f_A, f_F) combinations."""
    base = (df.tp == tp) & (df.input_len == il) & (df.output_len == ol) & (df.batch_size == bs)
    points = []
    for fa in FREQS:
        ra = df[base & (df.gpu_clock == fa)]
        if len(ra) == 0:
            continue
        for ff in FREQS:
            rf = df[base & (df.gpu_clock == ff)]
            if len(rf) == 0:
                continue
            lat = float(ra.iloc[0].D_A_lat) + float(rf.iloc[0].D_F_lat)
            eng = float(ra.iloc[0].D_A_energy) + float(rf.iloc[0].D_F_energy)
            points.append((lat, eng, fa, ff))
    return points


def find_optb(unified_pts, slo_budget):
    """OptB: min-energy unified freq under SLO."""
    best = None
    for lat, eng, f in unified_pts:
        if lat <= slo_budget:
            if best is None or eng < best[1]:
                best = (lat, eng, f)
    return best


def find_aflex(af_pts, slo_budget):
    """AFlex: min-energy (f_A, f_F) under SLO."""
    best = None
    for lat, eng, fa, ff in af_pts:
        if lat <= slo_budget:
            if best is None or eng < best[1]:
                best = (lat, eng, fa, ff)
    return best


def draw_config(ax, df, cfg, ngpus):
    tp, il, ol, bs = cfg["tp"], cfg["il"], cfg["ol"], cfg["bs"]
    uni_pts = get_unified_points(df, tp, il, ol, bs)
    af_pts = get_af_grid(df, tp, il, ol, bs)

    if not uni_pts or not af_pts:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        return

    # Scale energy by ngpus
    uni_pts = [(l, e * ngpus, f) for l, e, f in uni_pts]
    af_pts = [(l, e * ngpus, fa, ff) for l, e, fa, ff in af_pts]

    # SLO = max-freq latency
    max_freq_pt = [p for p in uni_pts if p[2] == MAX_FREQ]
    if not max_freq_pt:
        return
    slo_budget = max_freq_pt[0][0]

    # Draw unified staircase
    uni_sorted = sorted(uni_pts, key=lambda p: p[0])
    ux = [p[0] for p in uni_sorted]
    uy = [p[1] for p in uni_sorted]
    ax.plot(ux, uy, "--", color="#888", lw=1.5, alpha=0.6, zorder=1)

    # Draw unified points with freq colors
    for lat, eng, f in uni_pts:
        ax.scatter(lat, eng, color=FREQ_COLORS[f], s=70, zorder=4,
                   edgecolors="black", linewidths=0.6)
        ax.annotate(f"{f}", (lat, eng), fontsize=6, ha="center", va="bottom",
                    xytext=(0, 5), textcoords="offset points", color="#555")

    # Draw AF grid
    gx = [p[0] for p in af_pts]
    gy = [p[1] for p in af_pts]
    ax.scatter(gx, gy, color="#90CAF9", s=22, alpha=0.45, zorder=2,
               edgecolors="none", label="AF grid (36 combos)")

    # Find and mark OptB and AFlex
    optb = find_optb(uni_pts, slo_budget)
    aflex = find_aflex(af_pts, slo_budget)

    if optb:
        ax.scatter(optb[0], optb[1], color="#FF6F00", s=140, zorder=5,
                   marker="D", edgecolors="black", linewidths=1.2, label=f"OptB ({optb[2]}MHz)")

    if aflex:
        ax.scatter(aflex[0], aflex[1], color="#D32F2F", s=140, zorder=5,
                   marker="*", edgecolors="black", linewidths=0.8,
                   label=f"AFlex (A={aflex[2]}/F={aflex[3]})")

    # Draw arrow showing saving
    if optb and aflex:
        saving_pct = (optb[1] - aflex[1]) / optb[1] * 100
        mid_x = (optb[0] + aflex[0]) / 2
        ax.annotate("", xy=(aflex[0], aflex[1]), xytext=(optb[0], optb[1]),
                    arrowprops=dict(arrowstyle="->", color="#D32F2F", lw=2.0,
                                    connectionstyle="arc3,rad=0.2"))
        # Label the saving
        label_x = max(optb[0], aflex[0]) + (max(ux) - min(ux)) * 0.02
        label_y = (optb[1] + aflex[1]) / 2
        ax.text(label_x, label_y, f"−{saving_pct:.1f}%\nenergy",
                fontsize=10, color="#D32F2F", fontweight="bold",
                ha="left", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#D32F2F", alpha=0.9))

    # SLO line
    ax.axvline(slo_budget, color="#E53935", ls=":", lw=1.0, alpha=0.5)
    ax.text(slo_budget, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else min(gy) * 0.95,
            " SLO", fontsize=8, color="#E53935", va="bottom")

    ax.set_xlabel("Latency (μs)")
    ax.set_ylabel(f"Energy (mJ) × {ngpus} GPUs")
    ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    ncols = len(CONFIGS)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 5.0))
    if ncols == 1:
        axes = [axes]

    for ax, cfg in zip(axes, CONFIGS):
        ngpus = cfg["tp"]
        draw_config(ax, df, cfg, ngpus)

    fig.suptitle("Unified DVFS vs AF Differential DVFS: Pareto Frontier (Decode, SLO×1.0)",
                 fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    out = FIG_DIR / "fig1_pareto_comparison.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
