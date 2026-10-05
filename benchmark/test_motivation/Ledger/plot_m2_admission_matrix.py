"""Show completed two-session admission comparisons, including cohort costs."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ("native", "decode32", "slots64", "delayer32")
COLORS = {"native": "#267C9B", "decode32": "#D16E45", "slots64": "#37886D", "delayer32": "#86699D"}
LABELS = {"native": "Native", "decode32": "Decode32", "slots64": "Slots64", "delayer32": "Delayer32"}


def draw(result, output, metric):
    policies = [p for p in POLICIES if any(r["experiment"]["policy"] == p for r in result["rows"])]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    invalid = False
    for ax, (profile, phase) in zip(axes.flat, [("state", "far"), ("state", "near"), ("kv", "far"), ("kv", "near")]):
        maximum = 0
        for x, policy in enumerate(policies):
            own = sorted([r for r in result["rows"] if r["experiment"]["policy"] == policy
                          and r["experiment"]["profile"] == profile and r["experiment"]["phase"] == phase],
                         key=lambda r: r["experiment"]["rep"])
            if [r["experiment"]["rep"] for r in own] != [51, 52]:
                raise ValueError("Plot only complete two-session cells")
            values = [r["metrics"]["mean_jct_s"] if metric == "jct" else r["metrics"]["candidate"]["mean_ttft_s"] for r in own]
            bad = any(r["scene_errors"] for r in own)
            invalid |= bad
            average = statistics.fmean(values)
            ax.bar(x, average, width=.56, color=COLORS[policy], hatch="///" if bad else None,
                   edgecolor="#444444" if bad else COLORS[policy], linewidth=.5)
            for dx, marker, value in zip((-.075, .075), ("o", "s"), values):
                ax.scatter(x+dx, value, marker=marker, s=20, facecolors="white", edgecolors="#252525", zorder=3)
            ax.annotate(f"{average:.2f}", (x, max(values)), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9)
            maximum = max(maximum, *values)
        ax.set_xticks(range(len(policies)), [LABELS[p] for p in policies])
        ax.set_ylim(0, maximum*1.25)
        ax.set_title(f"{profile.upper()} / {phase}", fontsize=11)
        ax.set_ylabel("Overall mean JCT (s)" if metric == "jct" else "New-request mean TTFT (s)")
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Unified pool / Separate4K: " + ("completion time" if metric == "jct" else "new-request first-token cost"), fontsize=14)
    footnote = "Bars: two-session means. Circle: rep51; square: rep52. Points are observations, not confidence intervals."
    if invalid:
        footnote += "\nHatched cells have invalid arrival scenes; do not interpret them as phase-matched effects."
    fig.text(.5, .025, footnote, ha="center", va="bottom", fontsize=9, color="#444444")
    fig.subplots_adjust(left=.09, right=.98, top=.9, bottom=.13, hspace=.37, wspace=.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        fig.savefig(output.with_suffix(suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = json.loads(args.analysis.read_text())
    if not result["complete"]:
        parser.error("Need a complete matrix; do not silently impute missing cells")
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})
    for metric in ("jct", "ttft"):
        draw(result, args.output.with_name(args.output.name+"_"+metric), metric)


if __name__ == "__main__":
    main()
