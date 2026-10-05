#!/usr/bin/env python3
"""Plot natural pool pressure and explicitly labeled participation proxies."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


MODES = ("separate-4096", "mixed-4096")
COLORS = ("#ad4d62", "#237b71")
SHAPES = ("long", "medium", "short", "mixed", "state-pressure")
LABELS = ("16 x 8K", "64 x 2K", "256 x 512", "128 short + 8 long", "State-pressure")


def save(fig, output, name):
    for suffix in ("png", "svg"):
        fig.savefig(output / f"{name}.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def pressure(data, output):
    state, kv, _, _ = data["manifest"]["capacity_reference"]["resolved_capacity"]
    fig, axes = plt.subplots(2, 2, figsize=(12.4, 7.4), sharey=True)
    for row, rep in enumerate((1, 2)):
        for col, mode in enumerate(MODES):
            record = next(r for r in data["rows"] if
                          (r["experiment"]["shape"], r["experiment"]["mode"], r["experiment"]["rep"])
                          == ("state-pressure", mode, rep))
            t = record["resource_timeline"]
            axis = axes[row, col]
            axis.step([e["time_s"] for e in t], [100 * e["state_slots"] / state for e in t],
                      where="post", color="#ad4d62", label="Active state / state capacity")
            axis.step([e["time_s"] for e in t], [100 * e["kv_tokens"] / kv for e in t],
                      where="post", color="#237b71", label="Active KV / KV capacity")
            axis.set_title(f"{mode}, restart {rep}", loc="left", fontsize=11)
            axis.set_xlabel("Seconds from replay origin")
            axis.set_ylabel("Capacity used (%)")
            axis.set_ylim(0, 105)
            axis.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, loc="upper center", bbox_to_anchor=(.5, .935), frameon=False)
    count = data["manifest"]["pressure_count"]
    fig.suptitle(f"M2: natural capacity under {count} short requests (P512 / O128)", fontsize=14, y=.99)
    fig.text(.075, .035, f"State capacity {state}; KV capacity {kv:,} tokens. Corrected gate, no activation pacing, same requests.", fontsize=10)
    fig.subplots_adjust(top=.84, bottom=.13, left=.075, right=.985, hspace=.39, wspace=.21)
    save(fig, output, "m2_natural_pressure")


def participation(data, output):
    fig, axes = plt.subplots(2, 1, figsize=(12.4, 7.5))
    endpoints = (("all_mean_jct_s", "Mean JCT (s)"),
                 ("decode_ready_unselected_with_pending_slot_seconds_proxy", "Ready, unselected with pending\nCPU-launch proxy (slot-s)"))
    for axis, (key, title) in zip(axes, endpoints):
        for index, (mode, color) in enumerate(zip(MODES, COLORS)):
            for shape_index, shape in enumerate(SHAPES):
                values = [r["metrics"][key] for r in sorted(data["rows"], key=lambda r: r["experiment"]["rep"])
                          if r["experiment"]["mode"] == mode and r["experiment"]["shape"] == shape]
                if len(values) != 2:
                    raise ValueError("Expected two restarts per cell")
                x = shape_index + (index - .5) * .30
                axis.bar(x, statistics.fmean(values), width=.26, color=color, alpha=.75,
                         label=mode if shape_index == 0 else None)
                axis.scatter([x-.025, x+.025], values, color="#222222", s=14, zorder=3)
        axis.set_ylabel(title, fontsize=10)
        axis.set_xticks(range(len(SHAPES)), LABELS, fontsize=9)
        axis.grid(axis="y", alpha=.2)
        axis.set_axisbelow(True)
        axis.set_ylim(bottom=0)
    axes[0].legend(frameon=False, ncol=2, loc="upper left")
    fig.suptitle("M2: execution participation and service outcomes at natural capacity", fontsize=14, y=.99)
    fig.text(.10, .07, "Bars: two-run means; dots: individual restarts. Compare modes within each workload, not shapes with different output work.", fontsize=10)
    fig.text(.10, .035, "Unselected does not mean wasted GPU time or avoidable loss. Overlap and normal scheduling can contribute to this proxy.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.10, right=.985, top=.91, bottom=.16, hspace=.29)
    save(fig, output, "m2_natural_participation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if not data["complete"] or len(data["rows"]) != 20:
        raise ValueError("Require complete planned observation matrix")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    pressure(data, args.output)
    participation(data, args.output)


if __name__ == "__main__":
    main()
