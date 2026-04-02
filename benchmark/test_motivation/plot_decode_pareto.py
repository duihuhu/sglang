#!/usr/bin/env python3
"""Decode latency vs energy proxy: unified DVFS path + AF grid scatter (decode_data.txt).

Two mosaic figures: all (output_len x batch_size) panels on one page each for E~f^2*L and E~f*L.
Each panel: rows=TP, cols=input_len.
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
DATA_FILE = BASE_DIR / "decode_data.txt"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FREQ_COLORS = {210: "#E53935", 540: "#FB8C00", 870: "#43A047", 1200: "#1E88E5"}
FREQS = [210, 540, 870, 1200]


def orthogonal_path_vertical_first(xs, ys):
    """Connect consecutive points with L-shaped segments: vertical then horizontal."""
    if xs is None or len(xs) < 2:
        return list(xs) if xs is not None else [], list(ys) if ys is not None else []
    ox, oy = [xs[0]], [ys[0]]
    for i in range(len(xs) - 1):
        ox.extend([xs[i], xs[i + 1]])
        oy.extend([ys[i + 1], ys[i + 1]])
    return ox, oy


def unified_series(df, tp, il, ol, bs, freqs, exp=2):
    ls, es = [], []
    for f in freqs:
        row = df[
            (df.tp == tp)
            & (df.input_len == il)
            & (df.output_len == ol)
            & (df.gpu_clock == f)
            & (df.batch_size == bs)
        ]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        da, df_ = float(r.D_Attention), float(r.D_FFN)
        l = da + df_
        ls.append(l)
        es.append((f**exp) * l / 1e6)
    return ls, es


def af_grid(df, tp, il, ol, bs, freqs, exp=2):
    pts = []
    for fa in freqs:
        for ff in freqs:
            base = (
                (df.tp == tp)
                & (df.input_len == il)
                & (df.output_len == ol)
                & (df.batch_size == bs)
            )
            ra = df[base & (df.gpu_clock == fa)]
            rf = df[base & (df.gpu_clock == ff)]
            if len(ra) == 0 or len(rf) == 0:
                continue
            da = float(ra.iloc[0].D_Attention)
            dff = float(rf.iloc[0].D_FFN)
            l = da + dff
            e = (fa**exp * da + ff**exp * dff) / 1e6
            pts.append((l, e))
    return pts


def draw_subplot(ax, df, tp, il, ol, bs, freqs, exp, ngpus):
    u_ls, u_es = unified_series(df, tp, il, ol, bs, freqs, exp)
    grid_raw = af_grid(df, tp, il, ol, bs, freqs, exp)

    if u_ls is None or not grid_raw:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"tp={tp}  in={il}", fontsize=6)
        return

    u_es = [e * ngpus for e in u_es]
    grid = [(l, e * ngpus) for l, e in grid_raw]

    ux, uy = orthogonal_path_vertical_first(u_ls, u_es)
    ax.plot(ux, uy, "--", color="#999", lw=0.9, alpha=0.55, zorder=1, label="unified DVFS")
    for f, l, e in zip(freqs, u_ls, u_es):
        ax.scatter(l, e, color=FREQ_COLORS[f], s=22, zorder=3, edgecolors="black", linewidths=0.3)

    gx, gy = zip(*grid)
    ax.scatter(gx, gy, color="#1976D2", s=11, alpha=0.35, zorder=2, edgecolors="none", label="AF grid")

    ftag = "f^2" if exp == 2 else "f"
    ax.set_xlabel("Latency (us)", fontsize=6)
    ax.set_ylabel(f"E={ftag}·L·{ngpus}/10^6", fontsize=6)
    ax.set_title(f"tp={tp}  in={il}", fontsize=6)
    ax.tick_params(labelsize=5)
    ax.legend(fontsize=4, loc="upper right")


def _fill_panel(subfig, df, tps, all_il, ol, bs, freqs, exp):
    nrows = len(tps)
    ncols = len(all_il)
    axes = subfig.subplots(nrows, ncols, sharex=False, sharey=False)
    if nrows == 1:
        axes = np.array([axes])
    if ncols == 1:
        axes = axes.reshape(-1, 1)

    for ri, tp in enumerate(tps):
        ngpus = int(tp)
        for ci, il in enumerate(all_il):
            a = axes[ri, ci]
            sub = df[
                (df.tp == tp)
                & (df.input_len == il)
                & (df.output_len == ol)
                & (df.batch_size == bs)
            ]
            if len(sub) == 0:
                a.set_facecolor("#F5F5F5")
                a.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=a.transAxes,
                    fontsize=8,
                    color="#9E9E9E",
                )
                a.set_xticks([])
                a.set_yticks([])
                a.set_title(f"tp={tp}  in={il}", fontsize=6)
                continue
            draw_subplot(a, df, tp, il, ol, bs, freqs, exp, ngpus)


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    all_il = sorted(df.input_len.unique())
    tps = sorted(df.tp.unique())
    output_lens = sorted(df.output_len.unique())
    batch_sizes = sorted(df.batch_size.unique())

    exp_configs = [
        (2, "decode_pareto_mosaic_f2.png"),
        (1, "decode_pareto_mosaic_f1.png"),
    ]

    for exp, fname in exp_configs:
        e_note = "f^2·L" if exp == 2 else "f·L"

        fig = plt.figure(figsize=(16.5, 18.0), layout="constrained")
        subfigs = fig.subfigures(
            len(output_lens),
            len(batch_sizes),
            wspace=0.03,
            hspace=0.11,
        )

        for i, ol in enumerate(output_lens):
            for j, bs in enumerate(batch_sizes):
                sf = subfigs[i, j]
                sf.suptitle(f"output_len={ol}, batch_size={bs}", fontsize=10, y=1.02)
                _fill_panel(sf, df, tps, all_il, ol, bs, FREQS, exp)

        fig.suptitle(
            f"Decode: latency vs energy (all configs)  |  AF grid {len(FREQS)}×{len(FREQS)} MHz  |  "
            f"E ∝ {e_note} · tp",
            fontsize=12,
        )
        out = FIG_DIR / fname
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
