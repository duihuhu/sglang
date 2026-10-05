#!/usr/bin/env python3
"""Plot every calibration cell and the prespecified cap intervention."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


COLORS = ("#ad4d62", "#237b71", "#bf8b26")
MODES = ("separate-4096", "mixed-4096", "mixed-1024")


def values(data, pool, mode, cap, metric):
    result = [r["metrics"][metric] for r in sorted(data["rows"], key=lambda r: r["experiment"]["rep"])
              if (r["experiment"]["pool"], r["experiment"]["mode"], r["experiment"]["cap"]) == (pool, mode, cap)]
    if len(result) != 2:
        raise ValueError("Expected exactly two restarts per cell")
    return result


def save(fig, output, name):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def calibration(data, output):
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.5))
    caps = (128, 256, 485)
    for axis, (key, title) in zip(axes, (("all_mean_jct_s", "Mean JCT (s)"), ("mean_queue_s", "Mean queue (s)"),
                                       ("state_slot_seconds", "Active state (slot-s)"))):
        for mode, color in zip(MODES, COLORS):
            points = [values(data, "natural", mode, cap, key) for cap in caps]
            axis.plot(caps, [statistics.fmean(p) for p in points], color=color, marker="o", label=mode)
            for cap, point in zip(caps, points):
                axis.scatter([cap-3, cap+3], point, color=color, s=14, alpha=.6)
        axis.set_xticks(caps)
        axis.set_xlabel("Runtime cap (physical state pool = 485)")
        axis.set_ylabel(title)
        axis.grid(alpha=.2)
    fig.suptitle("M2: native batching calibration on the same 607 P512/O128 requests", fontsize=13, y=.99)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(.5, .925), frameon=False)
    fig.text(.065, .035, "Lines: two-start means; dots: starts. Large variability remains; no stable winner or confirmatory CI.", fontsize=10)
    fig.subplots_adjust(left=.065, right=.985, top=.77, bottom=.19, wspace=.35)
    save(fig, output, "m2_state_cap_calibration")


def mediation(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.7))
    configs = (("natural", 485), ("wide", 485), ("wide", 607))
    labels = ("A: state485 / cap485", "B: state728 / cap485", "C: state728 / cap607")
    for axis, (key, title) in zip(axes, (("all_mean_jct_s", "Mean JCT (s)"), ("state_slot_seconds", "Active state (slot-s)"))):
        for index, (mode, color) in enumerate(zip(MODES[:2], COLORS[:2])):
            for i, (pool, cap) in enumerate(configs):
                points = values(data, pool, mode, cap, key)
                x = i + (index-.5)*.30
                axis.bar(x, statistics.fmean(points), width=.26, color=color, alpha=.8, label=mode if i == 0 else None)
                axis.scatter([x-.025, x+.025], points, color="#222222", s=15, zorder=3)
        axis.set_xticks(range(3), labels, fontsize=9)
        axis.set_ylabel(title)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
    handles, legend = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend, ncol=2, loc="upper center", bbox_to_anchor=(.5, .925), frameon=False)
    delta = [c["delta_tight_minus_relaxed"] for c in data["contrasts"] if c["metric"] == "all_mean_jct_s"]
    fig.suptitle("M2: pool-layout control and within-pool cap intervention", fontsize=14, y=.99)
    fig.text(.07, .15, f"JCT interaction: D(B) - D(C) = {delta[0]:+.3f}s, {delta[1]:+.3f}s. D = Mixed minus Separate.", fontsize=10)
    fig.text(.07, .11, "Both mean JCTs increase at C: the smaller gap is not a speedup. Only two independent starts.", fontsize=10)
    fig.text(.07, .07, "B/C share the physical pool within each mode/start. A/B also change auxiliary memory and server session.", fontsize=10)
    fig.text(.07, .03, "Exploratory cap interaction only; no hybrid exclusivity, equal-total-HBM benefit, or new-algorithm claim.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.07, right=.985, top=.77, bottom=.27, wspace=.25)
    save(fig, output, "m2_state_cap_mediation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if not data["complete"] or len(data["rows"]) != 26:
        raise ValueError("Require both complete planned stages")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "svg.fonttype": "none",
                         "axes.spines.top": False, "axes.spines.right": False})
    calibration(data, args.output)
    mediation(data, args.output)


if __name__ == "__main__":
    main()
