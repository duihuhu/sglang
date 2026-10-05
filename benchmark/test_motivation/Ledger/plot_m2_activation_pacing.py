#!/usr/bin/env python3
"""Plot all frozen-screen cells and repeats without selecting a winning policy."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Patch


MODES = ("separate-4096", "mixed-4096", "mixed-1024", "unchunked")
LABELS = ("Separate 4K", "Mixed 4K", "Mixed 1K", "Unchunked")
ORDERS = ("short-first", "long-first")
COLORS = {0: "#777777", 1: "#b34d5e", 4: "#237b71"}


def save(fig, output, name):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def absolute_outcomes(data, output):
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8), sharey="col")
    for row, order in enumerate(ORDERS):
        for col, (key, title) in enumerate((("all_mean_jct", "Mean JCT, all requests"),
                                           ("makespan", "Replay makespan"))):
            axis = axes[row, col]
            for index, limit in enumerate(COLORS):
                xs, means, samples = [], [], []
                for mode_index, mode in enumerate(MODES):
                    runs = sorted((r for r in data["rows"] if
                                   (r["experiment"]["mode"], r["experiment"]["order"],
                                    r["experiment"]["activation_limit"]) == (mode, order, limit)),
                                  key=lambda r: r["experiment"]["rep"])
                    if len(runs) != 2:
                        raise ValueError("This figure requires two repeats for every frozen screen cell")
                    values = [r["metrics"][key] for r in runs]
                    xs.append(mode_index + (index - 1) * .24)
                    means.append(statistics.fmean(values))
                    samples.append(values)
                axis.bar(xs, means, width=.21, color=COLORS[limit], alpha=.75)
                for x, values in zip(xs, samples):
                    axis.scatter([x-.025, x+.025], values, s=16, color="#222222", zorder=3)
            axis.set_xticks(range(len(MODES)), LABELS, fontsize=9)
            axis.set_title(f"{order}: {title}", loc="left", fontsize=11)
            axis.set_ylabel("Seconds")
            axis.set_ylim(bottom=0)
            axis.grid(axis="y", alpha=.18)
            axis.set_axisbelow(True)
    legend = [Patch(facecolor=COLORS[k], alpha=.75, label=("Corrected default" if k == 0 else f"At most {k} fresh / pass"))
              for k in COLORS]
    fig.legend(handles=legend, ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, .945))
    fig.suptitle("M2: does fresh-activation pacing improve corrected batching?", fontsize=14, y=.985)
    fig.text(.075, .055, "48 screening runs, 80 requests/run; four finite waves. Bars: two-run means; dots: individual server-start repeats.", fontsize=10)
    fig.text(.075, .025, "Same token-VTC and state8 budget. This is exploratory evidence, not a steady-state or fairness guarantee.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.075, right=.985, bottom=.14, top=.85, wspace=.19, hspace=.3)
    save(fig, output, "m2_activation_pacing_outcomes")


def all_tradeoffs(data, output):
    endpoints = (("all_mean_jct", "All mean\nJCT"), ("short_mean_jct", "Short mean\nJCT"),
                 ("long_mean_jct", "Long mean\nJCT"), ("makespan", "Replay\nmakespan"))
    lookup = {(e["mode"], e["order"], e["activation_limit"]): e for e in data["paired_effects"]}
    matrices = []
    for order in ORDERS:
        matrices.append([[100 * lookup[(mode, order, limit)]["endpoints"][key]["mean_reduction_fraction"]
                          for key, _ in endpoints] for mode in MODES for limit in (1, 4)])
    bound = max(5, max(abs(value) for matrix in matrices for row in matrix for value in row))
    cmap = LinearSegmentedColormap.from_list("pacing_effect", ["#b34d5e", "#ffffff", "#237b71"])
    norm = TwoSlopeNorm(vmin=-bound, vcenter=0, vmax=bound)
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.7))
    row_labels = [f"{label}, k={limit}" for label in LABELS for limit in (1, 4)]
    for axis, order, matrix in zip(axes, ORDERS, matrices):
        rendered = axis.imshow(matrix, cmap=cmap, norm=norm, aspect="auto")
        axis.set_xticks(range(len(endpoints)), [label for _, label in endpoints], fontsize=10)
        axis.set_yticks(range(len(row_labels)), row_labels, fontsize=9)
        axis.set_title(order, loc="left", fontsize=12, pad=12)
        for row, values in enumerate(matrix):
            mode, limit = MODES[row // 2], (1, 4)[row % 2]
            inactive = lookup[(mode, order, limit)]["endpoints"]["blocks_activation_pacing"]["paced_mean"] == 0
            for col, value in enumerate(values):
                axis.text(col, row, f"{value:+.1f}%" + ("*" if inactive else ""), ha="center", va="center", fontsize=10,
                          color="white" if abs(value) / bound > .68 else "#222222")
        for boundary in (1.5, 3.5, 5.5):
            axis.axhline(boundary, color="white", linewidth=2)
        axis.tick_params(length=0)
        for spine in axis.spines.values():
            spine.set_visible(False)
    cax = fig.add_axes((.925, .23, .015, .58))
    fig.colorbar(rendered, cax=cax, label="Reduction versus corrected default (%)")
    fig.suptitle("M2: report both groups, including costs of delayed activation", fontsize=14, y=.98)
    fig.text(.125, .085, "Each cell: mean of two paired relative changes; positive means lower latency. No confidence intervals or winner selection.", fontsize=10)
    fig.text(.125, .055, "* Pacing never triggered in either repeat: do not attribute these differences to a changed admission decision.", fontsize=10, color="#555555")
    fig.text(.125, .025, "k limits fresh activations per prefill pass, not total active requests. All baselines already have correct slot accounting.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.125, right=.90, bottom=.19, top=.86, wspace=.52)
    save(fig, output, "m2_activation_pacing_tradeoffs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if not data["complete_planned_matrix"] or data["artifact_count"] != 48:
        raise ValueError("Require the complete predeclared 48-run screen")
    if {r["experiment"]["regime"] for r in data["rows"]} != {"state8"}:
        raise ValueError("Do not combine different resource regimes in this figure")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    absolute_outcomes(data, args.output)
    all_tradeoffs(data, args.output)


if __name__ == "__main__":
    main()
