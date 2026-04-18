#!/usr/bin/env python3
"""Analyze P_data.csv: A/F latency and energy across tp, input_len, gpu_clock, batch_size."""

import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import FuncFormatter

CSV_PATH = os.path.join(os.path.dirname(__file__), "P_data.csv")
OUT_DIR = os.path.join(os.path.dirname(__file__), "figures", "analysis_af")
# Omit very long contexts from plots (e.g. heatmap / facet grids).
EXCLUDE_INPUT_LENS = frozenset({32000, 40000})
# Heatmap slice for `heatmaps()` only.
HEATMAP_GPU_CLOCK = 1410


def _trend_slice_clock(rows):
    clocks = sorted({r["gpu_clock"] for r in rows})
    return HEATMAP_GPU_CLOCK if HEATMAP_GPU_CLOCK in clocks else clocks[-1]


# Plots 3–5: facet grids — inner spacing (split latency/energy into separate figures like plots 1/2).
_FACET_INNER_HSPACE = 0.12
_FACET_INNER_WSPACE = 0.10
_FACET_CELL_W = 2.45
_FACET_CELL_H = 3.05
# Linear x uses scientific ticks when max(|x|) >= this (plot 3/5); plot 4 uses log-x rule below.
_FACET_X_SCI_MIN = 1e4


def _facet_sci_y(ax):
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)


def _facet_sci_x_if_linear_large(ax, x_values):
    arr = np.asarray(x_values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0 or float(np.nanmax(np.abs(arr))) < _FACET_X_SCI_MIN:
        return
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0), useMathText=True)


def _facet_sci_x_if_log_large(ax, x_values):
    """Log-scaled x; use scientific-style tick labels when values reach 1e4+."""
    arr = np.asarray(x_values, dtype=float)
    if arr.size == 0 or float(np.nanmax(arr)) < _FACET_X_SCI_MIN:
        return

    def _fmt(v, pos):
        if not np.isfinite(v):
            return ""
        vf = float(v)
        av = abs(vf)
        if av >= 1e4 or (0 < av < 1e-2):
            return f"{vf:.2e}"
        if abs(vf - round(vf)) < max(1e-6 * av, 1e-9):
            return str(int(round(vf)))
        return f"{vf:g}"

    ax.xaxis.set_major_formatter(FuncFormatter(_fmt))


def load_rows(path):
    rows = []
    with open(path, newline="") as f:
        r = csv.reader(f)
        next(r)  # title row
        header = next(r)
        idx = {name: i for i, name in enumerate(header)}
        for line in r:
            if len(line) < len(header):
                continue
            rows.append(
                {
                    "tp": int(line[idx["tp"]]),
                    "input_len": int(line[idx["input_len"]]),
                    "gpu_clock": int(line[idx["gpu_clock"]]),
                    "batch_size": int(line[idx["batch_size"]]),
                    "A": float(line[idx["A"]]),
                    "F": float(line[idx["F"]]),
                    "TTFT_ms": float(line[idx["TTFT_ms"]]),
                    "A_energy_mj": float(line[idx["A_energy_mj"]]),
                    "F_energy_mj": float(line[idx["F_energy_mj"]]),
                }
            )
    return rows


def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def plot_by_batch_size(rows, out_dir):
    """Facet: tp x input_len; x=batch_size. Writes 1_latency_*.png and 1_energy_*.png (y-axis sci)."""
    by_key = defaultdict(list)
    for rec in rows:
        key = (rec["tp"], rec["input_len"], rec["gpu_clock"])
        by_key[key].append(rec)
    # pick median gpu_clock slice for overview grid (too many clocks otherwise)
    clocks = sorted({r["gpu_clock"] for r in rows})
    mid_clock = clocks[len(clocks) // 2]

    tps = sorted({r["tp"] for r in rows})
    ilens = sorted({r["input_len"] for r in rows})

    fig, axes = plt.subplots(len(tps), len(ilens), figsize=(4 * len(ilens), 3 * len(tps)), sharex=True)
    if len(tps) == 1 and len(ilens) == 1:
        axes = np.array([[axes]])
    elif len(tps) == 1:
        axes = axes.reshape(1, -1)
    elif len(ilens) == 1:
        axes = axes.reshape(-1, 1)

    for i, tp in enumerate(tps):
        for j, ilen in enumerate(ilens):
            ax = axes[i, j]
            sub = sorted(
                [r for r in rows if r["tp"] == tp and r["input_len"] == ilen and r["gpu_clock"] == mid_clock],
                key=lambda x: x["batch_size"],
            )
            if not sub:
                ax.set_visible(False)
                continue
            bs = [r["batch_size"] for r in sub]
            ax.plot(bs, [r["A"] for r in sub], "o-", label="A", color="#1f77b4")
            ax.plot(bs, [r["F"] for r in sub], "s-", label="F", color="#ff7f0e")
            ax.set_title(f"tp={tp}, input_len={ilen}, clock={mid_clock}")
            ax.set_xlabel("batch_size")
            ax.set_ylabel("A / F (CSV units)")
            ax.set_xscale("log", base=2)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            _facet_sci_y(ax)

    fig.suptitle(f"A/F latency vs batch_size (gpu_clock={mid_clock} MHz)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_latency_AF_vs_batch_size.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(len(tps), len(ilens), figsize=(4 * len(ilens), 3 * len(tps)), sharex=True)
    if len(tps) == 1 and len(ilens) == 1:
        axes = np.array([[axes]])
    elif len(tps) == 1:
        axes = axes.reshape(1, -1)
    elif len(ilens) == 1:
        axes = axes.reshape(-1, 1)

    for i, tp in enumerate(tps):
        for j, ilen in enumerate(ilens):
            ax = axes[i, j]
            sub = sorted(
                [r for r in rows if r["tp"] == tp and r["input_len"] == ilen and r["gpu_clock"] == mid_clock],
                key=lambda x: x["batch_size"],
            )
            if not sub:
                ax.set_visible(False)
                continue
            bs = [r["batch_size"] for r in sub]
            ax.plot(bs, [r["A_energy_mj"] for r in sub], "o-", label="A_energy", color="#2ca02c")
            ax.plot(bs, [r["F_energy_mj"] for r in sub], "s-", label="F_energy", color="#d62728")
            ax.set_title(f"tp={tp}, input_len={ilen}, clock={mid_clock}")
            ax.set_xlabel("batch_size")
            ax.set_ylabel("Energy (mJ)")
            ax.set_xscale("log", base=2)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            _facet_sci_y(ax)

    fig.suptitle(f"A/F energy vs batch_size (gpu_clock={mid_clock} MHz)", y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "1_energy_AF_vs_batch_size.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_by_tp(rows, out_dir):
    """Facet input_len × batch_size; x=tp; fixed gpu_clock. Two PNGs: latency A/F, then energy."""
    clock = _trend_slice_clock(rows)
    ilens = sorted({r["input_len"] for r in rows})
    bss = sorted({r["batch_size"] for r in rows})
    n_il, n_bs = len(ilens), len(bss)
    fig_w = _FACET_CELL_W * n_bs + 0.5
    fig_h = _FACET_CELL_H * n_il + 0.35
    for want_energy, fname, sup in [
        (False, "3_latency_AF_vs_tp.png", f"(3) A/F vs tp — facet input_len×batch_size, gpu_clock={clock} MHz"),
        (True, "3_energy_AF_vs_tp.png", f"(3) A_energy & F_energy vs tp — same facet, gpu_clock={clock} MHz"),
    ]:
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        inner = fig.add_gridspec(
            n_il, n_bs, hspace=_FACET_INNER_HSPACE, wspace=_FACET_INNER_WSPACE
        )
        for i, ilen in enumerate(ilens):
            for j, bs in enumerate(bss):
                ax = fig.add_subplot(inner[i, j])
                sub = sorted(
                    [r for r in rows if r["input_len"] == ilen and r["batch_size"] == bs and r["gpu_clock"] == clock],
                    key=lambda x: x["tp"],
                )
                if not sub:
                    ax.set_axis_off()
                    continue
                xs = [r["tp"] for r in sub]
                if want_energy:
                    ax.plot(xs, [r["A_energy_mj"] for r in sub], "o-", color="#2ca02c", markersize=3, label="A_e")
                    ax.plot(xs, [r["F_energy_mj"] for r in sub], "s-", color="#d62728", markersize=3, label="F_e")
                else:
                    ax.plot(xs, [r["A"] for r in sub], "o-", color="#1f77b4", markersize=3, label="A")
                    ax.plot(xs, [r["F"] for r in sub], "s-", color="#ff7f0e", markersize=3, label="F")
                ax.set_xticks(xs)
                ax.set_title(f"ilen={ilen}, bs={bs}", fontsize=7)
                if i == n_il - 1:
                    ax.set_xlabel("tp", fontsize=8)
                if j == 0:
                    ax.set_ylabel("mJ" if want_energy else "A / F", fontsize=8)
                ax.legend(fontsize=5, loc="upper left")
                ax.grid(True, alpha=0.28)
                _facet_sci_y(ax)
                _facet_sci_x_if_linear_large(ax, xs)
        fig.suptitle(sup, fontsize=11)
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_by_input_len(rows, out_dir):
    """Facet tp × batch_size; x=input_len (log2); fixed gpu_clock. Two PNGs."""
    clock = _trend_slice_clock(rows)
    tps = sorted({r["tp"] for r in rows})
    bss = sorted({r["batch_size"] for r in rows})
    n_tp, n_bs = len(tps), len(bss)
    fig_w = _FACET_CELL_W * n_bs + 0.5
    fig_h = _FACET_CELL_H * n_tp + 0.35
    for want_energy, fname, sup in [
        (False, "4_latency_AF_vs_input_len.png", f"(4) A/F vs input_len — facet tp×batch_size, gpu_clock={clock} MHz"),
        (
            True,
            "4_energy_AF_vs_input_len.png",
            f"(4) A_energy & F_energy vs input_len — same facet, gpu_clock={clock} MHz",
        ),
    ]:
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        inner = fig.add_gridspec(
            n_tp, n_bs, hspace=_FACET_INNER_HSPACE, wspace=_FACET_INNER_WSPACE
        )
        for i, tp in enumerate(tps):
            for j, bs in enumerate(bss):
                ax = fig.add_subplot(inner[i, j])
                sub = sorted(
                    [r for r in rows if r["tp"] == tp and r["batch_size"] == bs and r["gpu_clock"] == clock],
                    key=lambda x: x["input_len"],
                )
                if not sub:
                    ax.set_axis_off()
                    continue
                xs = [r["input_len"] for r in sub]
                if want_energy:
                    ax.plot(xs, [r["A_energy_mj"] for r in sub], "o-", color="#2ca02c", markersize=3, label="A_e")
                    ax.plot(xs, [r["F_energy_mj"] for r in sub], "s-", color="#d62728", markersize=3, label="F_e")
                else:
                    ax.plot(xs, [r["A"] for r in sub], "o-", color="#1f77b4", markersize=3, label="A")
                    ax.plot(xs, [r["F"] for r in sub], "s-", color="#ff7f0e", markersize=3, label="F")
                ax.set_xscale("log", base=2)
                ax.set_title(f"tp={tp}, bs={bs}", fontsize=7)
                if i == n_tp - 1:
                    ax.set_xlabel("input_len", fontsize=8)
                if j == 0:
                    ax.set_ylabel("mJ" if want_energy else "A / F", fontsize=8)
                ax.legend(fontsize=5, loc="upper left")
                ax.grid(True, alpha=0.28)
                _facet_sci_y(ax)
                _facet_sci_x_if_log_large(ax, xs)
        fig.suptitle(sup, fontsize=11)
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)


def plot_by_gpu_clock(rows, out_dir):
    """Facet tp × input_len; x=gpu_clock; batch_size=1. Two PNGs."""
    fixed_bs = 1
    tps = sorted({r["tp"] for r in rows})
    ilens = sorted({r["input_len"] for r in rows})
    n_tp, n_il = len(tps), len(ilens)
    fig_w = _FACET_CELL_W * n_il + 0.5
    fig_h = _FACET_CELL_H * n_tp + 0.35
    for want_energy, fname, sup in [
        (
            False,
            "5_latency_AF_vs_gpu_clock.png",
            f"(5) A/F vs gpu_clock — facet tp×input_len, batch_size={fixed_bs}",
        ),
        (
            True,
            "5_energy_AF_vs_gpu_clock.png",
            f"(5) A_energy & F_energy vs gpu_clock — same facet, batch_size={fixed_bs}",
        ),
    ]:
        fig = plt.figure(figsize=(fig_w, fig_h), constrained_layout=True)
        inner = fig.add_gridspec(
            n_tp, n_il, hspace=_FACET_INNER_HSPACE, wspace=_FACET_INNER_WSPACE
        )
        for i, tp in enumerate(tps):
            for j, ilen in enumerate(ilens):
                ax = fig.add_subplot(inner[i, j])
                sub = sorted(
                    [r for r in rows if r["tp"] == tp and r["input_len"] == ilen and r["batch_size"] == fixed_bs],
                    key=lambda x: x["gpu_clock"],
                )
                if not sub:
                    ax.set_axis_off()
                    continue
                xs = [r["gpu_clock"] for r in sub]
                if want_energy:
                    ax.plot(xs, [r["A_energy_mj"] for r in sub], "o-", color="#2ca02c", markersize=3, label="A_e")
                    ax.plot(xs, [r["F_energy_mj"] for r in sub], "s-", color="#d62728", markersize=3, label="F_e")
                else:
                    ax.plot(xs, [r["A"] for r in sub], "o-", color="#1f77b4", markersize=3, label="A")
                    ax.plot(xs, [r["F"] for r in sub], "s-", color="#ff7f0e", markersize=3, label="F")
                ax.set_title(f"tp={tp}, ilen={ilen}, bs={fixed_bs}", fontsize=7)
                if i == n_tp - 1:
                    ax.set_xlabel("gpu_clock (MHz)", fontsize=8)
                if j == 0:
                    ax.set_ylabel("mJ" if want_energy else "A / F", fontsize=8)
                ax.legend(fontsize=5, loc="upper right")
                ax.grid(True, alpha=0.28)
                _facet_sci_y(ax)
                _facet_sci_x_if_linear_large(ax, xs)
        fig.suptitle(sup, fontsize=11)
        fig.savefig(os.path.join(out_dir, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)


def _heatmap_color_norm(mat):
    """Stretch colormap: log norm when span is large, else robust linear percentiles."""
    v = mat[np.isfinite(mat)].astype(float).ravel()
    if v.size == 0:
        return Normalize(0.0, 1.0)
    vmin, vmax = float(np.min(v)), float(np.max(v))
    if vmax <= vmin:
        return Normalize(vmin=vmin, vmax=vmin + 1e-30)
    if np.all(v > 0) and (vmax / vmin) >= 12.0:
        return LogNorm(vmin=vmin, vmax=vmax)
    p2, p98 = np.percentile(v, [2.0, 98.0])
    if p98 <= p2:
        return Normalize(vmin=vmin, vmax=vmax)
    return Normalize(vmin=p2, vmax=p98)


def heatmaps(rows, out_dir):
    """2D heatmap: batch_size x one dim, for A ratio F/A and energy ratio."""
    tps = sorted({r["tp"] for r in rows})
    tp = tps[len(tps) // 2]
    clocks = sorted({r["gpu_clock"] for r in rows})
    clock = HEATMAP_GPU_CLOCK if HEATMAP_GPU_CLOCK in clocks else clocks[-1]

    ilens = sorted({r["input_len"] for r in rows})
    bss = sorted({r["batch_size"] for r in rows})

    mat_a = np.full((len(ilens), len(bss)), np.nan)
    mat_f = np.full_like(mat_a, np.nan)
    mat_ea = np.full_like(mat_a, np.nan)
    mat_ef = np.full_like(mat_a, np.nan)
    bi = {b: j for j, b in enumerate(bss)}
    ii = {x: i for i, x in enumerate(ilens)}
    for r in rows:
        if r["tp"] != tp or r["gpu_clock"] != clock:
            continue
        i = ii[r["input_len"]]
        j = bi[r["batch_size"]]
        mat_a[i, j] = r["A"]
        mat_f[i, j] = r["F"]
        mat_ea[i, j] = r["A_energy_mj"]
        mat_ef[i, j] = r["F_energy_mj"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, mat, title in [
        (axes[0, 0], mat_a, "A"),
        (axes[0, 1], mat_f, "F"),
        (axes[1, 0], mat_ea, "A_energy (mJ)"),
        (axes[1, 1], mat_ef, "F_energy (mJ)"),
    ]:
        norm = _heatmap_color_norm(mat)
        im = ax.imshow(
            mat,
            aspect="auto",
            origin="lower",
            cmap="turbo",
            norm=norm,
            interpolation="nearest",
        )
        ax.set_xticks(range(len(bss)))
        ax.set_xticklabels(bss, rotation=45)
        ax.set_yticks(range(len(ilens)))
        ax.set_yticklabels(ilens)
        ax.set_xlabel("batch_size")
        ax.set_ylabel("input_len")
        ax.set_title(f"{title} (tp={tp}, clock={clock})")
        plt.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(
        f"Heatmaps: input_len × batch_size | gpu_clock={clock} MHz "
        f"(turbo, log/robust norm)"
    )
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "6_heatmap_inputlen_batch.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_iso_inputlen_batch_product(rows, out_dir, product: int = 16384):
    """Along input_len * batch_size = product: A/F, TTFT, and energies vs input_len."""
    sub_all = [r for r in rows if r["input_len"] * r["batch_size"] == product]
    if not sub_all:
        return

    tps = sorted({r["tp"] for r in sub_all})
    clocks = sorted({r["gpu_clock"] for r in sub_all})
    n_r, n_c = len(tps), len(clocks)
    n_blk = 3
    fig_h = 2.35 * n_r * n_blk + 0.6
    fig, axes = plt.subplots(
        n_blk * n_r,
        n_c,
        figsize=(3.0 * n_c, fig_h),
        sharex=True,
        squeeze=False,
    )
    if n_blk * n_r == 1 and n_c == 1:
        axes = np.array([[axes]])
    elif n_blk * n_r == 1:
        axes = axes.reshape(1, -1)
    elif n_c == 1:
        axes = axes.reshape(-1, 1)

    def row_index(block: int, i_tp: int) -> int:
        return block * n_r + i_tp

    for i, tp in enumerate(tps):
        for j, clk in enumerate(clocks):
            chunk = sorted(
                [r for r in sub_all if r["tp"] == tp and r["gpu_clock"] == clk],
                key=lambda x: x["input_len"],
            )
            if not chunk:
                for b in range(n_blk):
                    axes[row_index(b, i), j].set_visible(False)
                continue
            xs = [r["input_len"] for r in chunk]

            ax_af = axes[row_index(0, i), j]
            ax_af.plot(xs, [r["A"] for r in chunk], "o-", label="A", color="#1f77b4", markersize=4)
            ax_af.plot(xs, [r["F"] for r in chunk], "s-", label="F", color="#ff7f0e", markersize=4)
            ax_af.set_xscale("log", base=2)
            ax_af.set_yscale("log")
            ax_af.set_title(f"tp={tp}, clock={clk}", fontsize=9)
            ax_af.grid(True, which="both", alpha=0.25)
            if j == 0:
                ax_af.set_ylabel("A / F")
            if i == 0:
                ax_af.text(0.02, 0.98, "A / F", transform=ax_af.transAxes, va="top", fontsize=8)

            ax_ttft = axes[row_index(1, i), j]
            ax_ttft.plot(xs, [r["TTFT_ms"] for r in chunk], "^-", color="#2ca02c", markersize=4, label="TTFT_ms")
            ax_ttft.set_xscale("log", base=2)
            ax_ttft.set_yscale("log")
            ax_ttft.grid(True, which="both", alpha=0.25)
            if j == 0:
                ax_ttft.set_ylabel("TTFT (ms)")
            if i == 0:
                ax_ttft.text(0.02, 0.98, "TTFT", transform=ax_ttft.transAxes, va="top", fontsize=8)

            ax_en = axes[row_index(2, i), j]
            ax_en.plot(xs, [r["A_energy_mj"] for r in chunk], "o-", label="A_energy", color="#9467bd", markersize=4)
            ax_en.plot(xs, [r["F_energy_mj"] for r in chunk], "s-", label="F_energy", color="#d62728", markersize=4)
            ax_en.set_xscale("log", base=2)
            ax_en.set_yscale("log")
            ax_en.grid(True, which="both", alpha=0.25)
            if j == 0:
                ax_en.set_ylabel("Energy (mJ)")
            if i == n_r - 1:
                ax_en.set_xlabel("input_len (ticks: len×bs)")
            if i == 0:
                ax_en.text(0.02, 0.98, "Energy", transform=ax_en.transAxes, va="top", fontsize=8)

    sample = sorted({r["input_len"] for r in sub_all})
    labels = [f"{il}\n×{product // il}" for il in sample]
    ax_bot = axes[n_blk * n_r - 1, 0]
    if ax_bot.get_visible():
        ax_bot.set_xticks(sample)
        ax_bot.set_xticklabels(labels, fontsize=7)

    legend_handles = []
    legend_labels = []
    for ax in axes.ravel():
        h, lab = ax.get_legend_handles_labels()
        for hi, li in zip(h, lab):
            if li and li not in legend_labels:
                legend_handles.append(hi)
                legend_labels.append(li)
    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper right",
            fontsize=7,
            ncol=3,
            bbox_to_anchor=(0.99, 1.0),
        )

    fig.suptitle(
        f"Prefill: input_len×batch_size={product} | A/F, TTFT (ms), A_energy & F_energy (mJ) vs len/bs mix",
        fontsize=11,
        y=1.005,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(
        os.path.join(out_dir, f"9_AF_iso_inputlen_bs_{product}.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(fig)


def plot_F_freq_sensitivity_ratio_bs4(
    rows,
    out_dir,
    *,
    batch_size: int = 4,
    clk_low: int = 210,
    clk_high: int = 1410,
) -> None:
    """F latency ratio F(clk_low)/F(clk_high) at fixed batch_size, vs input_len, per tp."""
    rows_b = [
        r
        for r in rows
        if r["batch_size"] == batch_size
        and r["gpu_clock"] in (clk_low, clk_high)
    ]
    by_key = defaultdict(dict)
    for r in rows_b:
        by_key[(r["tp"], r["input_len"])][r["gpu_clock"]] = r["F"]

    series = defaultdict(list)
    for (tp, ilen), clocks in sorted(by_key.items()):
        if clk_low not in clocks or clk_high not in clocks:
            continue
        f_lo = float(clocks[clk_low])
        f_hi = float(clocks[clk_high])
        if f_lo <= 0 or f_hi <= 0 or not (np.isfinite(f_lo) and np.isfinite(f_hi)):
            continue
        series[tp].append((ilen, f_lo / f_hi))

    for tp in series:
        series[tp].sort(key=lambda t: t[0])

    fig, ax = plt.subplots(figsize=(9, 5.2))
    for tp in sorted(series.keys()):
        pts = series[tp]
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, marker="o", ms=5, lw=1.8, label=f"tp={tp}")

    ax.axhline(1.0, color="0.45", ls="--", lw=1.1, alpha=0.85, label="ratio = 1")
    ax.set_xlabel("input_len (prefill tokens)")
    ax.set_ylabel(f"F latency ratio\nF({clk_low} MHz) / F({clk_high} MHz)")
    ax.set_title(
        f"Prefill F latency frequency sensitivity (batch_size={batch_size})\n"
        f"ratio > 1 means slower F at {clk_low} MHz than at {clk_high} MHz"
    )
    if series:
        all_x = [x for pts in series.values() for x, _ in pts]
        if all_x and max(all_x) / max(min(all_x), 1) >= 8:
            ax.set_xscale("log", base=2)
    ax.grid(True, which="both", alpha=0.35)
    ax.legend(loc="best", framealpha=0.92)
    fig.tight_layout()
    out_png = os.path.join(
        out_dir,
        f"10_F_lat_{clk_low}_over_{clk_high}_MHz_bs{batch_size}.png",
    )
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def correlation_and_summary(rows, out_dir):
    """Scatter matrix style: log metrics vs log batch_size colored by clock."""
    bs = np.array([r["batch_size"] for r in rows], dtype=float)
    A = np.array([r["A"] for r in rows])
    F = np.array([r["F"] for r in rows])
    EA = np.array([r["A_energy_mj"] for r in rows])
    EF = np.array([r["F_energy_mj"] for r in rows])
    clk = np.array([r["gpu_clock"] for r in rows], dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    sc = axes[0, 0].scatter(bs, A, c=clk, cmap="viridis", alpha=0.6, s=12)
    axes[0, 0].set_xscale("log", base=2)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xlabel("batch_size")
    axes[0, 0].set_ylabel("A")
    plt.colorbar(sc, ax=axes[0, 0], label="gpu_clock")

    sc = axes[0, 1].scatter(bs, F, c=clk, cmap="viridis", alpha=0.6, s=12)
    axes[0, 1].set_xscale("log", base=2)
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_xlabel("batch_size")
    axes[0, 1].set_ylabel("F")

    sc = axes[1, 0].scatter(bs, EA, c=clk, cmap="plasma", alpha=0.6, s=12)
    axes[1, 0].set_xscale("log", base=2)
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_xlabel("batch_size")
    axes[1, 0].set_ylabel("A_energy (mJ)")
    plt.colorbar(sc, ax=axes[1, 0], label="gpu_clock")

    sc = axes[1, 1].scatter(bs, EF, c=clk, cmap="plasma", alpha=0.6, s=12)
    axes[1, 1].set_xscale("log", base=2)
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_xlabel("batch_size")
    axes[1, 1].set_ylabel("F_energy (mJ)")
    plt.colorbar(sc, ax=axes[1, 1], label="gpu_clock")

    fig.suptitle("Global view: batch_size vs metrics (color=gpu_clock)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "7_scatter_batch_colored_clock.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # F/A ratio vs batch_size by tp
    tps = sorted({r["tp"] for r in rows})
    fig, ax = plt.subplots(figsize=(7, 5))
    for tp in tps:
        sub = [r for r in rows if r["tp"] == tp]
        ax.scatter(
            [r["batch_size"] for r in sub],
            [r["F"] / r["A"] if r["A"] else np.nan for r in sub],
            alpha=0.5,
            s=14,
            label=f"tp={tp}",
        )
    ax.set_xscale("log", base=2)
    ax.set_xlabel("batch_size")
    ax.set_ylabel("F / A")
    ax.set_title("F:A latency ratio vs batch_size (all clocks/input_len)")
    ax.legend(markerscale=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "8_ratio_F_over_A_vs_batch.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    lines = []
    lines.append("# P_data summary (auto-generated)\n")
    lines.append(f"n_rows={len(rows)}\n")
    lines.append(f"A: min={A.min():.2f} max={A.max():.2f} mean={A.mean():.2f}\n")
    lines.append(f"F: min={F.min():.2f} max={F.max():.2f} mean={F.mean():.2f}\n")
    lines.append(f"F/A: mean={(F/A).mean():.3f} median={np.median(F/A):.3f}\n")
    lines.append(f"A_energy_mj: mean={EA.mean():.2f}  F_energy_mj: mean={EF.mean():.2f}\n")
    with open(os.path.join(out_dir, "summary_stats.txt"), "w") as f:
        f.writelines(lines)


def main():
    rows = load_rows(CSV_PATH)
    rows = [r for r in rows if r["input_len"] not in EXCLUDE_INPUT_LENS]
    ensure_dir(OUT_DIR)
    plot_by_batch_size(rows, OUT_DIR)
    plot_by_tp(rows, OUT_DIR)
    plot_by_input_len(rows, OUT_DIR)
    plot_by_gpu_clock(rows, OUT_DIR)
    heatmaps(rows, OUT_DIR)
    plot_iso_inputlen_batch_product(rows, OUT_DIR, product=16384)
    plot_F_freq_sensitivity_ratio_bs4(rows, OUT_DIR)
    correlation_and_summary(rows, OUT_DIR)
    clk_trend = _trend_slice_clock(rows)
    with open(os.path.join(OUT_DIR, "summary_stats.txt"), "a") as f:
        f.write(
            f"\nplot3/4: 3_latency_AF_vs_tp.png + 3_energy_AF_vs_tp.png; "
            f"4_latency_AF_vs_input_len.png + 4_energy_AF_vs_input_len.png; "
            f"gpu_clock={clk_trend} MHz.\n"
        )
        f.write(
            "plot5: 5_latency_AF_vs_gpu_clock.png + 5_energy_AF_vs_gpu_clock.png; batch_size=1.\n"
        )
    print("Wrote figures to", OUT_DIR)


if __name__ == "__main__":
    main()
