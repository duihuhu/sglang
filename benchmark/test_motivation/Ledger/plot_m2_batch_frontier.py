#!/usr/bin/env python3
"""Plot measured M2 native batch controls with explicit screening labels."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from confirm_m2_batch_frontier import evaluate


COLORS = {"separate-4096": "#28775e", "mixed-4096": "#c14d48",
          "mixed-1024": "#376ea1"}
LABELS = {"separate-4096": "Separate prefill, chunk 4096",
          "mixed-4096": "Mixed prefill/decode, chunk 4096",
          "mixed-1024": "Mixed prefill/decode, chunk 1024"}


def plot(data, output, stage, factor):
    rows = [r for r in data["rows"] if r["stage"] == stage and r["factor"] == factor]
    if not rows:
        raise ValueError("No data for requested stage/shape")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.4))
    panels = [("split_p95", "(a) Short-request tenant p95 JCT", "Seconds"),
              ("packed_p95", "(b) Long-request tenant p95 JCT", "Seconds"),
              ("makespan", "(c) Whole replay completion time", "Seconds"),
              ("split_active_per_request", "(d) Short-request active residency", "Seconds / request")]
    for axis, (endpoint, title, unit) in zip(axes.flat, panels):
        for mode in COLORS:
            for policy, style, marker in [("vtc", "-", "o"), ("instant", "--", "s")]:
                selected = [r for r in rows if r["config"].startswith(mode + "/")
                            and r["config"].endswith("/" + policy)]
                selected.sort(key=lambda r: int(r["config"].split("/")[1][1:]))
                if not selected:
                    continue
                xs = [int(r["config"].split("/")[1][1:]) for r in selected]
                ys = [r["mean"][endpoint] for r in selected]
                axis.plot(xs, ys, color=COLORS[mode], linestyle=style, marker=marker,
                          linewidth=2, markersize=5)
        axis.set_title(title, loc="left", fontsize=12, pad=10)
        axis.set_xlabel("Runtime admission cap (fixed pool and graph)")
        axis.set_ylabel(unit)
        axis.set_xticks([9, 24, 48, 96])
        axis.grid(axis="y", alpha=0.22)
        axis.set_ylim(bottom=0)
    handles = [plt.Line2D([], [], color=COLORS[m], linewidth=2, label=LABELS[m]) for m in COLORS]
    handles += [plt.Line2D([], [], color="#333333", linestyle=ls, marker=mk, label=label)
                for ls, mk, label in [("-", "o", "Token VTC"), ("--", "s", "Instant-share control")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=3,
               frameon=False, fontsize=10)
    count = sorted({len(r["runs"]) for r in rows})
    fig.suptitle("M2: Admission and batch rules change latency and efficiency", fontsize=17, y=0.99)
    fig.text(0.08, 0.065, f"{stage.upper()}; split factor {factor}; runs per point: {count}. "
             "Native SGLang controls, not complete SOTA reproductions.", color="#555555")
    fig.text(0.08, 0.035, "Kimi-Linear-48B-A3B, TP2, GPU 6/7; state 480, KV 196608; "
             "identical paired inputs; no prefix reuse.", color="#555555")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.15, top=0.81, hspace=0.46, wspace=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def plot_confirmation(data, selection, output):
    factors = selection["factors"]
    fig, axes = plt.subplots(len(factors), 2, figsize=(12.8, 4.0 * len(factors)),
                             squeeze=False)
    endpoints = ["split_p95", "packed_p95", "makespan"]
    labels = ["Short p95", "Long p95", "Makespan"]
    for row_index, factor in enumerate(factors):
        result = evaluate(data, selection, factor)
        left, right = axes[row_index]
        for side, shift, color, label in [("baseline", -0.19, "#376ea1", "Cap 96"),
                                           ("candidate", 0.19, "#28775e", "Cap 48")]:
            row = next(r for r in data["rows"] if r["stage"] == "confirm"
                       and r["factor"] == factor and r["config"] == selection[side])
            ys = [row["mean"][key] for key in endpoints]
            bars = left.bar([i + shift for i in range(3)], ys, width=0.36,
                            color=color, label=label)
            left.bar_label(bars, fmt="%.2f", fontsize=9, padding=3)
        left.set_xticks(range(3), labels)
        left.set_ylabel("Seconds; mean of run-level metrics")
        left.set_ylim(0, left.get_ylim()[1] * 1.14)
        left.legend(frameon=False, ncol=2, fontsize=10)
        left.set_title(f"Split factor {factor}: absolute outcomes", loc="left", fontsize=12)
        for i, key in enumerate(endpoints):
            effect = result["effects"][key]
            point = 100 * effect["mean_improvement"]
            low, high = [100 * v for v in effect["ci95"]]
            right.errorbar(point, i, xerr=[[point - low], [high - point]],
                           fmt="o", color="#28775e", capsize=5)
            right.annotate(f"{point:+.1f}% [{low:+.1f}, {high:+.1f}]",
                           (point, i), xytext=(0, 13), textcoords="offset points",
                           ha="center", fontsize=9)
        right.axvline(0, color="#333333", linewidth=1)
        right.axvline(-5, color="#c14d48", linestyle=":", linewidth=1)
        right.set_yticks(range(3), labels)
        right.set_ylim(2.7, -0.7)
        right.set_xlim(min(-12, right.get_xlim()[0] - 10), max(15, right.get_xlim()[1] + 10))
        right.set_xlabel("Paired improvement (%); positive is better")
        right.set_title("5 fresh pairs; bootstrap 95% CI", loc="left", fontsize=12)
        for axis in (left, right):
            axis.spines[["top", "right"]].set_visible(False)
            axis.grid(axis="y" if axis is left else "x", alpha=0.16)
            axis.set_axisbelow(True)
    fig.suptitle("M2 confirmation: same VTC, pools and graphs; admission cap 96 to 48",
                 fontsize=15, y=0.985)
    fig.text(0.07, 0.055, "Preselected pair; screening excluded; full workload warmup. "
             "Dotted line: -5% secondary-endpoint margin.", fontsize=10, color="#555555")
    fig.text(0.07, 0.025, "This is a native-control comparison, not SOTA inferiority "
             "or a token-fairness guarantee.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=0.07, right=0.97, bottom=0.14, top=0.90,
                        hspace=0.51, wspace=0.31)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=180, facecolor="white")
    fig.savefig(output.with_suffix(".svg"), facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--stage", default="screen")
    parser.add_argument("--factor", type=int, default=16)
    parser.add_argument("--confirmation", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if args.confirmation:
        plot_confirmation(data, json.loads(args.confirmation.read_text()), args.output)
    else:
        plot(data, args.output, args.stage, args.factor)


if __name__ == "__main__":
    main()
