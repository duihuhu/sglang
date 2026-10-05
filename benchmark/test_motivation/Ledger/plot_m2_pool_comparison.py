"""Plot complete budget-matched pool/cap cells, including negative controls."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if not data["complete"] or data["stage"] != "measure" or data["counts"]["runs"] != 24:
        raise ValueError("Plot only the complete formal matrix")
    order = [("static-natural", 485, "A\n485"), ("unified-old-cap", 485, "D\n485"),
             ("static-rebalanced", 485, "B\n485"), ("static-rebalanced", 607, "B\n607"),
             ("unified-raised-cap", 485, "C\n485"), ("unified-raised-cap", 607, "C\n607")]
    colors = ["#70757b", "#b5706d", "#258b78", "#258b78", "#ce8a2d", "#ce8a2d"]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 3, figsize=(11.8, 7.3))
    for y, mode in enumerate(("separate-4096", "mixed-4096")):
        for x, (metric, title) in enumerate((("mean_jct_s", "Mean JCT"), ("mean_queue_s", "Mean queue"), ("p95_jct_s", "p95 JCT"))):
            ax = axes[y, x]
            values = []
            for arm, cap, label in order:
                values.append([r["metrics"][metric] for r in data["rows"] if
                               (r["experiment"]["pool"], r["experiment"]["mode"], r["experiment"]["cap"]) == (arm, mode, cap)])
            if any(len(v) != 2 for v in values):
                raise ValueError("Incomplete cell")
            ax.bar(range(6), [sum(v) / 2 for v in values], color=colors, width=.72, alpha=.85)
            for i, samples in enumerate(values):
                ax.scatter([i-.10, i+.10], samples, color="#222222", s=18, zorder=3)
            ax.set_xticks(range(6), [row[2] for row in order])
            ax.set_ylim(0, max(max(v) for v in values) * 1.17)
            ax.set_title(f"{'Separate4K' if y == 0 else 'Mixed4K'}: {title}", fontsize=11)
            ax.set_ylabel("Seconds")
            ax.grid(axis="y", alpha=.18)
            ax.set_axisbelow(True)
    fig.suptitle("Pool management and runtime cap at matched resident-memory budgets", fontsize=14, y=.97)
    fig.text(.5, .07, "A: original 485-slot split   B: rebalanced static   C: unified, request table 607   D: unified, request table 485",
             ha="center", fontsize=9)
    fig.text(.5, .035, "607 requests, P512/O128, TP2. Bars: two-session means; dots: individual runs, not confidence intervals.",
             ha="center", fontsize=9)
    fig.subplots_adjust(left=.065, right=.985, top=.90, bottom=.16, hspace=.43, wspace=.29)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "svg"):
        fig.savefig(args.output.with_suffix("." + suffix), dpi=180, facecolor="white")


if __name__ == "__main__":
    main()
