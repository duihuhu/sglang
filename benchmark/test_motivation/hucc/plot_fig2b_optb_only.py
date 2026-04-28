#!/usr/bin/env python3
"""
Figure 2b: AFlex vs OptB only — pure AF differentiation saving vs SLO.
Zoomed-in view showing the saturation effect clearly.
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
SLO_MULTS_FULL = [1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0]
SLO_MULTS_ZOOM = [1.0, 1.02, 1.05, 1.08, 1.1, 1.15, 1.2, 1.3, 1.5]

IL_WEIGHTS = {128: 0.30, 512: 0.30, 1024: 0.20, 2048: 0.10, 4096: 0.07, 8192: 0.03}

TP_COLORS = {1: "#1E88E5", 2: "#43A047", 4: "#E53935", 8: "#8E24AA"}
TP_MARKERS = {1: "o", 2: "s", 4: "D", 8: "^"}


def optb_energy(df_cfg, slo_budget):
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


def compute_optb_savings(df, tp, slo_mult):
    """Return (weighted_mean, all_raw_savings_list) for error bar."""
    ils = sorted(df.input_len.unique())
    ols = sorted(df.output_len.unique())
    bss = sorted(df.batch_size.unique())

    ws, ww, raw = [], [], []
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

                e_ob = optb_energy(cfg, slo)
                e_af = aflex_energy(cfg, slo)

                if e_ob is not None and e_af is not None and e_ob > 0:
                    s = (e_ob - e_af) / e_ob * 100.0
                    ws.append(s * w)
                    ww.append(w)
                    raw.append(s)

    mean = sum(ws) / sum(ww) if ww else 0
    return mean, raw


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    tps = sorted(df.tp.unique())

    # ── 2×2 subplots: one panel per tp ──
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    axes_flat = axes.flatten()

    for idx, tp in enumerate(tps):
        ax = axes_flat[idx]
        color = TP_COLORS.get(tp, "#333")

        means, p25s, p75s, p10s, p90s = [], [], [], [], []
        for slo_m in SLO_MULTS_ZOOM:
            mean, raw = compute_optb_savings(df, tp, slo_m)
            means.append(mean)
            if raw:
                p10s.append(np.percentile(raw, 10))
                p25s.append(np.percentile(raw, 25))
                p75s.append(np.percentile(raw, 75))
                p90s.append(np.percentile(raw, 90))
            else:
                p10s.append(0); p25s.append(0)
                p75s.append(0); p90s.append(0)
            print(f"tp={tp} SLO×{slo_m}: mean={mean:.2f}%  "
                  f"P25={p25s[-1]:.1f}%  P75={p75s[-1]:.1f}%")

        # P10~P90 light band
        ax.fill_between(SLO_MULTS_ZOOM, p10s, p90s,
                        color=color, alpha=0.08, zorder=1, label="P10~P90")

        # P25~P75 darker band
        ax.fill_between(SLO_MULTS_ZOOM, p25s, p75s,
                        color=color, alpha=0.22, zorder=2, label="P25~P75")

        # Mean line
        ax.plot(SLO_MULTS_ZOOM, means, "-o", color=color,
                markersize=6, lw=2.2, zorder=4, label="weighted mean")

        # Annotate peak and tail
        ax.annotate(f"{means[0]:.1f}%", (SLO_MULTS_ZOOM[0], means[0]),
                    fontsize=11, fontweight="bold", color=color,
                    xytext=(-5, 10), textcoords="offset points", ha="center")
        ax.annotate(f"{means[-1]:.1f}%", (SLO_MULTS_ZOOM[-1], means[-1]),
                    fontsize=10, color=color,
                    xytext=(6, 0), textcoords="offset points", ha="left", va="center")

        # Saturation band
        ax.axhspan(2, 6, alpha=0.08, color="#FF9800", zorder=0)

        # Peak region
        ax.axvspan(0.99, 1.05, alpha=0.08, color="#4CAF50", zorder=0)

        ax.set_title(f"tp = {tp}", fontsize=13, fontweight="bold", color=color)
        ax.set_ylim(bottom=0, top=max(max(p90s) * 1.15, 14))
        ax.set_xlim(0.98, 1.52)
        ax.set_xticks(SLO_MULTS_ZOOM)
        ax.set_xticklabels([f"×{s}" for s in SLO_MULTS_ZOOM], fontsize=8)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    # Shared labels
    for ax in axes[1]:
        ax.set_xlabel("SLO Multiplier", fontsize=11)
    for ax in axes[:, 0]:
        ax.set_ylabel("Energy Saving vs OptB (%)", fontsize=11)

    fig.suptitle("AFlex vs OptB: Pure AF Differentiation Saving (Decode)\n"
                 "line = weighted mean, dark band = P25~P75, light band = P10~P90",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = FIG_DIR / "fig2b_aflex_vs_optb_zoom.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
