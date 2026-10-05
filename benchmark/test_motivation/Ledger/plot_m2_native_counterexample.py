"""Plot native calibration and its new-request cost without hiding repeats."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if not data["complete"]:
        parser.error("Need a complete matrix")
    policies = ("native607", "cap512", "slots96")
    colors = ("#267C9B", "#37886D", "#86699D")
    labels = ("Native607", "Cap512", "Slots96")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for col, phase in enumerate(("far", "near")):
        for row, metric in enumerate(("jct", "ttft")):
            ax, maximum = axes[row, col], 0
            for x, (policy, color) in enumerate(zip(policies, colors)):
                own = sorted([r for r in data["rows"] if r["experiment"]["policy"] == policy and r["experiment"]["phase"] == phase], key=lambda r: r["experiment"]["rep"])
                if [r["experiment"]["rep"] for r in own] != [61, 62]:
                    raise ValueError("Incomplete paired cell")
                values = [r["metrics"]["mean_jct_s"] if metric == "jct" else r["metrics"]["candidate"]["mean_ttft_s"] for r in own]
                average = statistics.fmean(values)
                ax.bar(x, average, width=.55, color=color)
                for offset, marker, value in zip((-.07, .07), ("o", "s"), values):
                    ax.scatter(x + offset, value, marker=marker, s=22, facecolors="white", edgecolors="#252525", zorder=3)
                ax.annotate(f"{average:.2f}", (x, max(values)), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=10)
                maximum = max(maximum, *values)
            ax.set_xticks(range(3), labels)
            ax.set_ylim(0, maximum * 1.25)
            ax.set_title(f"STATE / {phase}")
            ax.set_ylabel("Overall mean JCT (s)" if metric == "jct" else "New-request mean TTFT (s)")
            ax.set_axisbelow(True)
            ax.grid(axis="y", alpha=.18)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Same unified pool / Separate4K: native control calibration", fontsize=14)
    fig.text(.5, .025, "Bars: two-session means. Circle: rep61; square: rep62. Points are observations, not confidence intervals.", ha="center", va="bottom", fontsize=8.5)
    fig.subplots_adjust(left=.09, right=.98, top=.9, bottom=.12, hspace=.37, wspace=.25)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(args.output.with_suffix(suffix), dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
