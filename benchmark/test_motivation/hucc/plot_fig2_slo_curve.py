#!/usr/bin/env python3
"""
Figure 2: Energy saving vs SLO multiplier curves.
Two line groups per tp:
  - Solid: AFlex vs BiScale (total saving, grows with SLO slack)
  - Dashed: AFlex vs OptB (pure AF differentiation, saturates after SLO×1.5)
Shaded area between = BiScale U-shape trap contribution.
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
SLO_MULTS = [1.0, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]

# Mixed workload weights
IL_WEIGHTS = {128: 0.30, 512: 0.30, 1024: 0.20, 2048: 0.10, 4096: 0.07, 8192: 0.03}

TP_COLORS = {1: "#1E88E5", 2: "#43A047", 4: "#E53935", 8: "#8E24AA"}
TP_MARKERS = {1: "o", 2: "s", 4: "D", 8: "^"}


def biscale_energy(df_cfg, slo_budget):
    """BiScale: lowest-freq-first that satisfies SLO."""
    for f in sorted(FREQS):
        row = df_cfg[df_cfg.gpu_clock == f]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lat = float(r.D_A_lat) + float(r.D_F_lat)
        eng = float(r.D_A_energy) + float(r.D_F_energy)
        if lat <= slo_budget:
            return eng
    return None


def optb_energy(df_cfg, slo_budget):
    """OptB: min-energy unified freq under SLO."""
    best = None
    for f in FREQS:
        row = df_cfg[df_cfg.gpu_clock == f]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        lat = float(r.D_A_lat) + float(r.D_F_lat)
        eng = float(r.D_A_energy) + float(r.D_F_energy)
        if lat <= slo_budget:
            if best is None or eng < best:
                best = eng
    return best


def aflex_energy(df_cfg, slo_budget):
    """AFlex: min-energy (f_A, f_F) under SLO."""
    best = None
    for fa in FREQS:
        ra = df_cfg[df_cfg.gpu_clock == fa]
        if len(ra) == 0:
            continue
        for ff in FREQS:
            rf = df_cfg[df_cfg.gpu_clock == ff]
            if len(rf) == 0:
                continue
            lat = float(ra.iloc[0].D_A_lat) + float(rf.iloc[0].D_F_lat)
            eng = float(ra.iloc[0].D_A_energy) + float(rf.iloc[0].D_F_energy)
            if lat <= slo_budget:
                if best is None or eng < best:
                    best = eng
    return best


def compute_weighted_savings(df, tp, slo_mult):
    """Compute mixed-workload weighted savings for a given tp and SLO multiplier."""
    ils = sorted(df.input_len.unique())
    ols = sorted(df.output_len.unique())
    bss = sorted(df.batch_size.unique())

    ws_biscale, ws_optb, ws_weights = [], [], []

    for il in ils:
        w = IL_WEIGHTS.get(il, 0.0)
        if w == 0:
            continue
        for ol in ols:
            for bs in bss:
                cfg = df[(df.tp == tp) & (df.input_len == il) &
                         (df.output_len == ol) & (df.batch_size == bs)]
                if len(cfg) == 0:
                    continue
                ref = cfg[cfg.gpu_clock == MAX_FREQ]
                if len(ref) == 0:
                    continue
                ref_lat = float(ref.iloc[0].D_A_lat) + float(ref.iloc[0].D_F_lat)
                slo = ref_lat * slo_mult

                e_bi = biscale_energy(cfg, slo)
                e_ob = optb_energy(cfg, slo)
                e_af = aflex_energy(cfg, slo)

                if e_bi is not None and e_af is not None and e_bi > 0:
                    s_bi = (e_bi - e_af) / e_bi * 100.0
                    ws_biscale.append(s_bi * w)
                    ws_weights.append(w)

                if e_ob is not None and e_af is not None and e_ob > 0:
                    s_ob = (e_ob - e_af) / e_ob * 100.0
                    ws_optb.append(s_ob * w)

    total_w = sum(ws_weights) if ws_weights else 1
    mean_bi = sum(ws_biscale) / total_w if ws_biscale else 0
    mean_ob = sum(ws_optb) / total_w if ws_optb else 0
    return mean_bi, mean_ob


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    tps = sorted(df.tp.unique())

    fig, ax = plt.subplots(1, 1, figsize=(9, 5.5))

    for tp in tps:
        color = TP_COLORS.get(tp, "#333")
        marker = TP_MARKERS.get(tp, "o")

        savings_bi = []
        savings_ob = []
        for slo_m in SLO_MULTS:
            s_bi, s_ob = compute_weighted_savings(df, tp, slo_m)
            savings_bi.append(s_bi)
            savings_ob.append(s_ob)

        # Solid: vs BiScale
        ax.plot(SLO_MULTS, savings_bi, "-", color=color, marker=marker,
                markersize=7, lw=2.0, label=f"tp={tp} vs BiScale", zorder=3)

        # Dashed: vs OptB
        ax.plot(SLO_MULTS, savings_ob, "--", color=color, marker=marker,
                markersize=5, lw=1.5, alpha=0.7, label=f"tp={tp} vs OptB", zorder=2)

        # Shaded area between (BiScale U-trap contribution)
        ax.fill_between(SLO_MULTS, savings_ob, savings_bi,
                        color=color, alpha=0.08, zorder=1)

    # Annotations
    ax.axhline(y=0, color="#999", lw=0.5)

    # Add region labels
    ax.axvspan(1.0, 1.15, alpha=0.06, color="#4CAF50", zorder=0)
    ax.text(1.07, ax.get_ylim()[0] + 1 if ax.get_ylim()[0] >= 0 else 1, "strict SLO\n(production)",
            fontsize=8, ha="center", va="bottom", color="#2E7D32", fontstyle="italic")

    ax.set_xlabel("SLO Multiplier (×min latency at max freq)", fontsize=12)
    ax.set_ylabel("Energy Saving (%)", fontsize=12)
    ax.set_title("Decode: AFlex Energy Saving vs SLO Strictness\n"
                 "(solid = vs BiScale, dashed = vs OptB, shaded = U-trap fix contribution)",
                 fontsize=12, fontweight="bold")

    ax.set_xticks(SLO_MULTS)
    ax.set_xticklabels([f"×{s}" for s in SLO_MULTS])

    # Legend: two columns
    ax.legend(fontsize=8, loc="upper left", ncol=2, framealpha=0.9)

    fig.tight_layout()
    out = FIG_DIR / "fig2_saving_vs_slo.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
