#!/usr/bin/env python3
"""Show the slot-gate mechanism and all native-batching controls."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


CONFIGS = ["state8-prefill4096", "state8-mixed4096", "state8-mixed1024",
           "state8-unchunked", "wide-cap8-prefill4096"]
LABELS = ["Separate 4K", "Mixed 4K", "Mixed 1K", "Unchunked", "Wide + cap8"]
COLORS = {"legacy": "#a74656", "reuse-aware": "#237b71"}


def save(fig, directory, name):
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{name}.{extension}", dpi=180, facecolor="white")
    plt.close(fig)


def mechanism(data, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for gate in COLORS:
        key = f"m2s-state8-prefill4096-{gate}-short-first-rep20"
        sample = data["first_rep_samples"][key]
        events = sample["resource_events"]
        xs = [0] + [e["observed_s"] + sample["clock_offset"] for e in events]
        ys = [0] + [e["active_requests"] for e in events]
        label = "Original gate" if gate == "legacy" else "Reuse-aware gate"
        axes[0].step(xs, ys, where="post", label=label, color=COLORS[gate], linewidth=2)
    axes[0].axhline(8, color="#666666", linestyle=":", linewidth=1, label="Runtime / state cap")
    axes[0].set_ylim(0, 9)
    axes[0].set_xlabel("Seconds since replay start")
    axes[0].set_ylabel("Active request / recurrent-state holders")
    axes[0].set_title("(a) Exact activation trajectory, repetition 20", loc="left", fontsize=11)
    axes[0].legend(frameon=False, loc="lower left", fontsize=9)
    endpoints = [("short_p95", "Short p95"), ("long_p95", "Long p95"), ("makespan", "Makespan")]
    for gate, shift in (("legacy", -.18), ("reuse-aware", .18)):
        row = next(r for r in data["rows"] if (r["config"], r["gate"], r["order"]) ==
                   (CONFIGS[0], gate, "short-first"))
        xs = [i + shift for i in range(3)]
        bars = axes[1].bar(xs, [row["mean"][k] for k, _ in endpoints], width=.32,
                           color=COLORS[gate], alpha=.8)
        axes[1].bar_label(bars, fmt="%.2f", padding=5, fontsize=9)
        for x, (key, _) in zip(xs, endpoints):
            axes[1].scatter([x + (i - 2) * .025 for i in range(len(row["runs"]))],
                            [r[key] for r in row["runs"]], s=14, color="#333333", zorder=3)
    axes[1].set_xticks(range(3), [label for _, label in endpoints])
    axes[1].set_ylabel("Seconds")
    axes[1].set_ylim(0, axes[1].get_ylim()[1] * 1.13)
    axes[1].set_title("(b) Service consequences, all five repetitions", loc="left", fontsize=11)
    for axis in axes:
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle("M2: a reused slot should not be charged as a fresh allocation", fontsize=15, y=.99)
    fig.text(.075, .055, "Same token-VTC, pools, runtime cap and workload. State8 / separate 4K / short-first; no prefix caching.", fontsize=10)
    fig.text(.075, .015, "Trajectory is a predeclared repetition, not an average; dots show all formal runs. No universal batching claim.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.075, right=.98, bottom=.23, top=.82, wspace=.3)
    save(fig, directory, "m2_slot_reuse_mechanism")


def controls(data, directory):
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.4), sharex="col")
    endpoints = [("short_p95", "Short p95 JCT"), ("long_p95", "Long p95 JCT"),
                 ("makespan", "Replay makespan")]
    for i, order in enumerate(("short-first", "long-first")):
        for j, (endpoint, title) in enumerate(endpoints):
            axis = axes[i, j]
            for k, config in enumerate(CONFIGS):
                row = next(r for r in data["paired_gate_effects"]
                           if r["config"] == config and r["order"] == order)
                value = row["effects"][endpoint]
                mid = 100 * value["mean_improvement"]
                lo, hi = [100 * x for x in value["ci95"]]
                axis.errorbar(mid, k, xerr=[[mid-lo], [hi-mid]], fmt="o", capsize=4,
                              color="#237b71" if k < 3 else "#536378")
            axis.axvline(0, color="#777777", linewidth=1)
            axis.set_yticks(range(5), LABELS if j == 0 else [])
            axis.set_ylim(4.6, -.6)
            axis.set_title(f"{order}: {title}", loc="left", fontsize=11)
            axis.grid(axis="x", alpha=.18)
            if i == 1:
                axis.set_xlabel("Improvement versus original gate (%)")
    fig.suptitle("M2: does the opportunity survive other batching choices?", fontsize=15, y=.99)
    fig.text(.11, .05, "All 100 formal runs; five paired repeats per contrast. Points / bars: run-level paired means / bootstrap 95% CI.", fontsize=10)
    fig.text(.11, .014, "Positive is better. Within-session gate A/B; two mixed chunk sizes are one batching family. All use the same token-VTC.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.11, right=.98, bottom=.17, top=.88, wspace=.2, hspace=.3)
    save(fig, directory, "m2_slot_reuse_controls")


def distribution(data, directory):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for gate, color in COLORS.items():
        sample = data["first_rep_samples"][f"m2s-state8-prefill4096-{gate}-short-first-rep20"]
        values = sorted(r["latency_s"] for r in sample["request_latencies"] if r["group"] == "short")
        label = "Original gate" if gate == "legacy" else "Reuse-aware gate"
        axes[0].step([0, *values], [0, *[(i + 1) / len(values) for i in range(len(values))]],
                     where="post", color=color, label=label, linewidth=2)
        axes[0].plot(values, [(i + 1) / len(values) for i in range(len(values))], ".", color=color)
    axes[0].axhline(.95, color="#777777", linestyle=":", linewidth=1)
    axes[0].set_xlabel("Short-request JCT (seconds)")
    axes[0].set_ylabel("Fraction completed")
    axes[0].set_ylim(0, 1.07)
    axes[0].set_title("(a) Empirical CDF, repetition 20 (16 requests)", loc="left", fontsize=11)
    axes[0].legend(frameon=False, loc="lower right", fontsize=9)
    row = next(r for r in data["paired_gate_effects"]
               if r["config"] == CONFIGS[0] and r["order"] == "short-first")
    for i, key in enumerate(("short_p95", "short_mean_jct", "short_max_jct")):
        value = row["effects"][key]
        mid = 100 * value["mean_improvement"]
        lo, hi = [100 * x for x in value["ci95"]]
        axes[1].errorbar(mid, i, xerr=[[mid-lo], [hi-mid]], fmt="o", capsize=5, color="#237b71")
        axes[1].annotate(f"{mid:.1f}% [{lo:.1f}, {hi:.1f}]", (mid, i), xytext=(0, 14),
                         textcoords="offset points", ha="center", fontsize=10)
    axes[1].set_yticks(range(3), ["p95", "Mean", "Maximum"])
    axes[1].set_ylim(2.55, -.55)
    axes[1].axvline(0, color="#777777", linewidth=1)
    axes[1].set_xlim(-5, 37)
    axes[1].set_xlabel("Short-request JCT improvement (%)")
    axes[1].set_title("(b) Five paired repetitions, mean and 95% CI", loc="left", fontsize=11)
    for axis in axes:
        axis.grid(alpha=.18)
    fig.suptitle("M2: one fewer late request is not a faster final completion", fontsize=15, y=.99)
    fig.text(.075, .05, "State8 / separate 4K / short-first. Empirical CDF is not interpolated; the reported p95 uses linear interpolation.", fontsize=10)
    fig.text(.075, .015, "Both gates retain a final slow short request. Report the mean and maximum alongside the small-cohort p95.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.075, right=.97, bottom=.23, top=.82, wspace=.3)
    save(fig, directory, "m2_slot_reuse_distribution")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if not data["complete_planned_matrix"]:
        raise ValueError("Require the full predeclared 100-run matrix")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False, "svg.fonttype": "none"})
    mechanism(data, args.output)
    controls(data, args.output)
    distribution(data, args.output)


if __name__ == "__main__":
    main()
