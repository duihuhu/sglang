#!/usr/bin/env python3
"""
Figure 3: Sweet-spot heatmap — tp × bs matrix of AFlex vs OptB saving at SLO×1.0.
Color intensity = energy saving %. Annotated with ★ (>10%) and ▲ (5~10%).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "font.family": "sans-serif",
})

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "decode_data.txt"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FREQS = [210, 450, 690, 930, 1170, 1410]
MAX_FREQ = 1410


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
    best_pair = None
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
                    best_pair = (fa, ff)
    return best, best_pair


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    tps = sorted(df.tp.unique())
    bss = sorted(df.batch_size.unique())
    ils = sorted(df.input_len.unique())
    ols = sorted(df.output_len.unique())

    slo_mult = 1.0

    # Build matrix: rows=tp, cols=bs
    matrix = np.full((len(tps), len(bss)), np.nan)
    freq_labels = [[None] * len(bss) for _ in range(len(tps))]

    for ri, tp in enumerate(tps):
        for ci, bs in enumerate(bss):
            savings = []
            best_pair_example = None
            for il in ils:
                for ol in ols:
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
                    e_af, pair = aflex_energy(cfg, slo)

                    if e_ob is not None and e_af is not None and e_ob > 0:
                        s = (e_ob - e_af) / e_ob * 100.0
                        savings.append(s)
                        if pair and pair[0] != pair[1]:
                            best_pair_example = pair

            if savings:
                matrix[ri, ci] = np.mean(savings)
                if best_pair_example:
                    freq_labels[ri][ci] = best_pair_example

    # Plot heatmap
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))

    # Custom colormap: white -> light blue -> deep blue -> red for >10%
    cmap = matplotlib.colormaps.get_cmap("YlOrRd")
    norm = mcolors.Normalize(vmin=0, vmax=max(15, np.nanmax(matrix)))

    im = ax.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")

    # Add text annotations
    for ri in range(len(tps)):
        for ci in range(len(bss)):
            val = matrix[ri, ci]
            if np.isnan(val):
                ax.text(ci, ri, "N/A", ha="center", va="center",
                        fontsize=9, color="#999")
                continue

            # Symbol
            if val >= 10:
                sym = "★"
                color = "white"
                fontweight = "bold"
            elif val >= 5:
                sym = "▲"
                color = "black"
                fontweight = "bold"
            else:
                sym = ""
                color = "black"
                fontweight = "normal"

            text = f"{sym}{val:.1f}%"
            ax.text(ci, ri, text, ha="center", va="center",
                    fontsize=10, color=color, fontweight=fontweight)

            # Show freq pair for sweet-spot cells
            pair = freq_labels[ri][ci]
            if pair and val >= 5:
                ax.text(ci, ri + 0.32, f"A={pair[0]}/F={pair[1]}",
                        ha="center", va="center", fontsize=6, color="#555")

    ax.set_xticks(range(len(bss)))
    ax.set_xticklabels([f"bs={b}" for b in bss], fontsize=10)
    ax.set_yticks(range(len(tps)))
    ax.set_yticklabels([f"tp={t}" for t in tps], fontsize=11)

    ax.set_xlabel("Batch Size", fontsize=12)
    ax.set_ylabel("Tensor Parallelism", fontsize=12)
    ax.set_title("Decode: AFlex vs OptB Energy Saving (%) at SLO×1.0\n"
                 "★ ≥ 10% (sweet spot)   ▲ 5~10%   blank < 5%",
                 fontsize=12, fontweight="bold")

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Energy Saving (%)", fontsize=10)

    fig.tight_layout()
    out = FIG_DIR / "fig3_sweetspot_heatmap.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
