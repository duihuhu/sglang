#!/usr/bin/env python3
"""Prefill latency vs energy proxy: unified DVFS path + AF grid scatter (prefill_data_v0).

Outputs two figures: E ∝ f^2·L (default name) and E ∝ f·L (suffix _f1).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "figure.dpi": 150,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)

BASE_DIR = Path(__file__).parent
DATA_FILE = BASE_DIR / "prefill_data_v0.txt"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FREQ_COLORS = {210: "#E53935", 540: "#FB8C00", 870: "#43A047", 1200: "#1E88E5"}
FREQS = [210, 540, 870, 1200]


def orthogonal_path_vertical_first(xs, ys):
    """Connect consecutive points with L-shaped segments: vertical then horizontal (step goes down then across)."""
    if xs is None or len(xs) < 2:
        return list(xs) if xs is not None else [], list(ys) if ys is not None else []
    ox, oy = [xs[0]], [ys[0]]
    for i in range(len(xs) - 1):
        ox.extend([xs[i], xs[i + 1]])
        oy.extend([ys[i + 1], ys[i + 1]])
    return ox, oy


def unified_series(df, tp, il, bs, freqs, exp=2):
    ls, es = [], []
    for f in freqs:
        row = df[
            (df.tp == tp) & (df.input_len == il) & (df.gpu_clock == f) & (df.batch_size == bs)
        ]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        pa, pf = float(r.P_A), float(r.P_F)
        l = pa + pf
        ls.append(l)
        es.append((f**exp) * l / 1e6)
    return ls, es


def af_grid(df, tp, il, bs, freqs, exp=2):
    pts = []
    for fa in freqs:
        for ff in freqs:
            base = (df.tp == tp) & (df.input_len == il) & (df.batch_size == bs)
            ra = df[base & (df.gpu_clock == fa)]
            rf = df[base & (df.gpu_clock == ff)]
            if len(ra) == 0 or len(rf) == 0:
                continue
            pa = float(ra.iloc[0].P_A)
            pff = float(rf.iloc[0].P_F)
            l = pa + pff
            e = (fa**exp * pa + ff**exp * pff) / 1e6
            pts.append((l, e))
    return pts


def draw_subplot(ax, df, tp, il, bs, freqs, exp, ngpus):
    u_ls, u_es = unified_series(df, tp, il, bs, freqs, exp)
    grid_raw = af_grid(df, tp, il, bs, freqs, exp)

    if u_ls is None or not grid_raw:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"tp={tp} in={il} bs={bs}", fontsize=8)
        return

    u_es = [e * ngpus for e in u_es]
    grid = [(l, e * ngpus) for l, e in grid_raw]

    ux, uy = orthogonal_path_vertical_first(u_ls, u_es)
    ax.plot(ux, uy, "--", color="#999", lw=1.0, alpha=0.55, zorder=1, label="unified DVFS")
    for f, l, e in zip(freqs, u_ls, u_es):
        ax.scatter(l, e, color=FREQ_COLORS[f], s=28, zorder=3, edgecolors="black", linewidths=0.35)

    gx, gy = zip(*grid)
    ax.scatter(gx, gy, color="#1976D2", s=14, alpha=0.35, zorder=2, edgecolors="none", label="AF grid")

    ftag = "f^2" if exp == 2 else "f"
    ax.set_xlabel("Latency (us)", fontsize=7)
    ax.set_ylabel(f"E={ftag}·L·{ngpus}/10^6", fontsize=7)
    ax.set_title(f"tp={tp}  in={il}  bs={bs}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="upper right")


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    all_il = sorted(df.input_len.unique())
    tps = sorted(df.tp.unique())
    bss = sorted(df.batch_size.unique())
    # exp=2 (f^2·L): default filename; exp=1 (f·L): extra figure
    exp_configs = [
        (2, ""),  # prefill_pareto_v0_all_tp_input_bs1.png
        (1, "_f1"),  # prefill_pareto_v0_all_tp_input_bs1_f1.png
    ]

    for exp, name_suffix in exp_configs:
        e_note = "f^2" if exp == 2 else "f (linear in f)"

        for bs in bss:
            nrows = len(tps)
            ncols = len(all_il)
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.0 * ncols, 2.35 * nrows), sharex=False, sharey=False)
            if nrows == 1:
                axes = np.array([axes])
            if ncols == 1:
                axes = axes.reshape(-1, 1)

            for ri, tp in enumerate(tps):
                ngpus = int(tp)
                for ci, il in enumerate(all_il):
                    ax = axes[ri, ci]
                    sub = df[(df.tp == tp) & (df.input_len == il) & (df.batch_size == bs)]
                    if len(sub) == 0:
                        ax.set_facecolor("#F5F5F5")
                        ax.text(
                            0.5,
                            0.5,
                            "N/A",
                            ha="center",
                            va="center",
                            transform=ax.transAxes,
                            fontsize=9,
                            color="#9E9E9E",
                        )
                        ax.set_xticks([])
                        ax.set_yticks([])
                        ax.set_title(f"tp={tp}  in={il}  bs={bs}", fontsize=8)
                        continue
                    draw_subplot(ax, df, tp, il, bs, FREQS, exp, ngpus)

            tag = f"bs{bs}"
            fig.suptitle(
                f"Prefill: latency vs energy proxy  |  AF grid {len(FREQS)}x{len(FREQS)} freqs (MHz)  |  "
                f"E proportional to {e_note}  |  {tag}",
                fontsize=11,
                y=1.01,
            )
            fig.tight_layout(rect=[0, 0, 1, 0.98])
            out = FIG_DIR / f"prefill_pareto_v0_all_tp_input_{tag}{name_suffix}.png"
            fig.savefig(out, dpi=150)
            plt.close(fig)
            print(f"Saved {out}")


if __name__ == "__main__":
    main()
