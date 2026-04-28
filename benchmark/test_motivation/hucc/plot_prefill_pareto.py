#!/usr/bin/env python3
"""Prefill latency vs real energy: unified DVFS staircase + AF grid scatter.

Uses measured P_A_energy / P_F_energy (mJ) directly.
Layout: rows=tp, cols=input_len, one figure per batch_size.
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
DATA_FILE = BASE_DIR / "prefill_data.txt"
FIG_DIR = BASE_DIR / "figures"
FIG_DIR.mkdir(exist_ok=True)

FREQ_COLORS = {
    210: "#E53935",
    450: "#FB8C00",
    690: "#43A047",
    930: "#8E24AA",
    1170: "#1E88E5",
    1410: "#00897B",
}
FREQS = [210, 450, 690, 930, 1170, 1410]


def staircase_pareto(xs, ys):
    """Pareto-style staircase from right-top to left-bottom.

    Walk from the rightmost point leftward. Between consecutive points:
      - if current y >= next y: go left first, then down  (horizontal, vertical)
      - if current y <  next y: go up first, then left    (vertical, horizontal)
    This creates an outer envelope around the unified DVFS points."""
    if xs is None or len(xs) < 2:
        return list(xs) if xs is not None else [], list(ys) if ys is not None else []
    # Sort by x descending (right to left)
    pairs = sorted(zip(xs, ys), key=lambda p: -p[0])
    ox, oy = [pairs[0][0]], [pairs[0][1]]
    for i in range(len(pairs) - 1):
        cx, cy = pairs[i]
        nx, ny = pairs[i + 1]
        if cy >= ny:
            # current is higher: go left first, then down
            ox.append(nx)
            oy.append(cy)
            ox.append(nx)
            oy.append(ny)
        else:
            # current is lower: go up first, then left
            ox.append(cx)
            oy.append(ny)
            ox.append(nx)
            oy.append(ny)
    return ox, oy


def unified_series(df, tp, il, bs, freqs):
    """Unified DVFS: same freq for attention & FFN. Returns (latencies, energies)."""
    ls, es = [], []
    for f in freqs:
        row = df[
            (df.tp == tp) & (df.input_len == il) & (df.gpu_clock == f) & (df.batch_size == bs)
        ]
        if len(row) == 0:
            return None, None
        r = row.iloc[0]
        l = float(r.P_A_lat) + float(r.P_F_lat)
        e = float(r.P_A_energy) + float(r.P_F_energy)
        ls.append(l)
        es.append(e)
    return ls, es


def af_grid(df, tp, il, bs, freqs):
    """AF grid: attention@freq_a, FFN@freq_f. Returns list of (lat, energy)."""
    pts = []
    for fa in freqs:
        for ff in freqs:
            base = (df.tp == tp) & (df.input_len == il) & (df.batch_size == bs)
            ra = df[base & (df.gpu_clock == fa)]
            rf = df[base & (df.gpu_clock == ff)]
            if len(ra) == 0 or len(rf) == 0:
                continue
            pa_lat = float(ra.iloc[0].P_A_lat)
            pf_lat = float(rf.iloc[0].P_F_lat)
            pa_e = float(ra.iloc[0].P_A_energy)
            pf_e = float(rf.iloc[0].P_F_energy)
            l = pa_lat + pf_lat
            e = pa_e + pf_e
            pts.append((l, e))
    return pts


def draw_subplot(ax, df, tp, il, bs, freqs, ngpus):
    u_ls, u_es = unified_series(df, tp, il, bs, freqs)
    grid_raw = af_grid(df, tp, il, bs, freqs)

    if u_ls is None or not grid_raw:
        ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title(f"tp={tp} in={il} bs={bs}", fontsize=8)
        return

    # Scale by ngpus
    u_es = [e * ngpus for e in u_es]
    grid = [(l, e * ngpus) for l, e in grid_raw]

    # Staircase path for unified DVFS
    ux, uy = staircase_pareto(u_ls, u_es)
    ax.plot(ux, uy, "--", color="#999", lw=1.0, alpha=0.55, zorder=1, label="unified DVFS")

    # Colored dots per frequency
    for f, l, e in zip(freqs, u_ls, u_es):
        ax.scatter(l, e, color=FREQ_COLORS[f], s=28, zorder=3, edgecolors="black", linewidths=0.35)

    # AF grid scatter
    gx, gy = zip(*grid)
    ax.scatter(gx, gy, color="#1976D2", s=14, alpha=0.35, zorder=2, edgecolors="none", label="AF grid")

    ax.set_xlabel("Latency (us)", fontsize=7)
    ax.set_ylabel(f"Energy (mJ)x{ngpus}", fontsize=7)
    ax.set_title(f"tp={tp}  in={il}  bs={bs}", fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=5, loc="upper right")


def run_for_freqs(df, freqs, suffix, freq_label):
    """Generate one set of figures for a given freq list."""
    all_il = sorted(df.input_len.unique())
    tps = sorted(df.tp.unique())
    bss = sorted(df.batch_size.unique())

    for bs in bss:
        nrows = len(tps)
        ncols = len(all_il)
        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(2.0 * ncols, 2.35 * nrows),
            sharex=False, sharey=False,
        )
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
                        0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, fontsize=9, color="#9E9E9E",
                    )
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f"tp={tp}  in={il}  bs={bs}", fontsize=8)
                    continue
                draw_subplot(ax, df, tp, il, bs, freqs, ngpus)

        tag = f"bs{bs}"
        fig.suptitle(
            f"Prefill: latency vs real energy  |  AF grid {len(freqs)}x{len(freqs)}  |  {freq_label}  |  {tag}",
            fontsize=11, y=1.01,
        )
        fig.tight_layout(rect=[0, 0, 1, 0.98])
        out = FIG_DIR / f"prefill_pareto_real_energy_{suffix}_{tag}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print(f"Saved {out}")


def main():
    df = pd.read_csv(DATA_FILE, sep="\t")
    df.columns = df.columns.str.strip()

    # Set 1: all 6 freqs
    run_for_freqs(df, FREQS, "6freq", "210-1410 MHz")

    # Set 2: without 210 MHz
    freqs_no210 = [f for f in FREQS if f != 210]
    run_for_freqs(df, freqs_no210, "no210", "450-1410 MHz (no 210)")


if __name__ == "__main__":
    main()
