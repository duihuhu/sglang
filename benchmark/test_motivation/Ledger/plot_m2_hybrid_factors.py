#!/usr/bin/env python3
"""Scientific figures for the measured M2 factors, including negative controls."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


REGIMES = ["wide", "state-low", "cap-control", "kv-low"]
LABELS = ["Wide pools", "State 8", "Wide + cap 8", "KV 16K"]
COLORS = {"vtc": "#326a9b", "instant": "#b04a62"}


def save(fig, output, name):
    for ext in ("png", "svg"):
        fig.savefig(output / f"{name}.{ext}", dpi=180, facecolor="white")
    plt.close(fig)


def lookup(data, family, regime, variant, policy):
    return next(r for r in data["rows"] if (r["family"], r["regime"], r["variant"],
                r["policy"]) == (family, regime, variant, policy))


def activation(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
    counts = [1, 4, 16, 32]
    rows = [lookup(data, "E1", "wide", f"n{n}", "fcfs") for n in counts]
    bars = axes[0].bar(range(4), [r["mean"]["peak_state"] for r in rows], color="#267969", width=.55)
    axes[0].bar_label(bars, fmt="%.0f", padding=4)
    axes[0].set_xticks(range(4), ["1 x 16K", "4 x 4K", "16 x 1K", "32 x 512"])
    axes[0].set_ylim(0, 38)
    axes[0].set_ylabel("Peak allocated recurrent-state slots")
    axes[0].set_xlabel("Same 16K aggregate input tokens")
    axes[0].set_title("(a) Request partition changes state activation", loc="left", fontsize=12)
    trace = data["first_rep_trajectories"]["m2h-wide-E1-n1-fcfs-rep10"]
    events = trace["events"]
    first = events[0]["observed_s"]
    xs = [e["observed_s"] - first for e in events]
    state_axis = axes[1]
    kv_axis = state_axis.twinx()
    state_axis.step(xs, [e["state_slots"] for e in events], where="post", color="#267969", label="State slots", linewidth=2.2)
    kv_axis.step(xs, [e["kv_tokens"] / 1024 for e in events], where="post", color="#326a9b", linestyle="--", label="KV tokens", linewidth=2.2)
    state_axis.set_ylim(0, 1.35)
    state_axis.set_yticks([0, 1])
    kv_axis.set_ylim(0, 20)
    state_axis.set_ylabel("Allocated state slots", color="#267969")
    kv_axis.set_ylabel("Allocated KV tokens (Ki tokens)", color="#326a9b")
    state_axis.set_xlabel("Seconds since first observed activation")
    state_axis.set_title("(b) Chunk continuation adds KV, not another state", loc="left", fontsize=12)
    handles = state_axis.get_lines() + kv_axis.get_lines()
    state_axis.legend(handles, [h.get_label() for h in handles], loc="upper right", frameon=False)
    for axis in axes:
        axis.grid(axis="y", alpha=.18)
        axis.set_axisbelow(True)
    fig.suptitle("M2 / E1: activation is a request-level resource event", fontsize=16, y=.99)
    fig.text(.075, .055, "3 runs per shape; state peaks agree. Right: first predeclared repetition. No prefix caching.", fontsize=10, color="#555555")
    fig.text(.075, .015, "Output is 128 tokens per request: total output work differs. This is not a latency or fairness comparison.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.075, right=.93, bottom=.23, top=.83, wspace=.38)
    save(fig, output, "m2_hybrid_activation")


def candidates(data, output):
    fig, axes = plt.subplots(2, 3, figsize=(13.8, 7.4), sharey="col")
    endpoints = [("short_p95", "Short-request p95 JCT"), ("long_p95", "Long-request p95 JCT"),
                 ("makespan", "Replay completion time")]
    for i, order in enumerate(("short-first", "long-first")):
        for j, (endpoint, title) in enumerate(endpoints):
            axis = axes[i, j]
            for policy, shift in (("vtc", -.18), ("instant", .18)):
                rows = [lookup(data, "E2", r, order, policy) for r in REGIMES]
                xs = [k + shift for k in range(4)]
                axis.bar(xs, [r["mean"][endpoint] for r in rows], width=.32,
                         color=COLORS[policy], alpha=.8, label="Token VTC" if policy == "vtc" else "Instant-share control")
                for x, row in zip(xs, rows):
                    jitter = [(k - 2) * .022 for k in range(len(row["runs"]))]
                    axis.scatter([x + v for v in jitter], [r[endpoint] for r in row["runs"]],
                                 s=12, c="#252525", alpha=.65, zorder=3)
            axis.set_xticks(range(4), LABELS, rotation=20, ha="right")
            axis.set_ylim(bottom=0)
            axis.set_ylabel("Seconds" if j == 0 else "")
            axis.grid(axis="y", alpha=.18)
            axis.set_axisbelow(True)
            axis.set_title(f"{order}: {title}", loc="left", fontsize=11)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", bbox_to_anchor=(.5, .945), ncol=2, frameon=False)
    fig.suptitle("M2 / E2: the same candidate requests encounter different admission constraints", fontsize=15, y=.995)
    fig.text(.065, .052, "16 short (512/64) + 4 long (8192/64); dots: all 5 repetitions. Identical paired inputs; two submission orders.", fontsize=10, color="#555555")
    fig.text(.065, .017, "Native capacity checks retained. Instant-share is a diagnostic comparator, not an equal-fairness SOTA baseline.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.065, right=.99, top=.86, bottom=.18, hspace=.55, wspace=.25)
    save(fig, output, "m2_hybrid_candidates")


def holding(data, output):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 4.8))
    for variant, shift, color, label in (("o32", -.18, "#267969", "Background output 32"),
                                          ("o256", .18, "#b04a62", "Background output 256")):
        rows = [lookup(data, "E3", r, variant, "fcfs") for r in REGIMES]
        xs = [i + shift for i in range(4)]
        bars = axes[0].bar(xs, [r["mean"]["probe_queue"] for r in rows], width=.32, color=color, alpha=.8, label=label)
        axes[0].bar_label(bars, fmt="%.2f", fontsize=9, padding=5)
        for x, row in zip(xs, rows):
            axes[0].scatter([x + (i-2)*.024 for i in range(len(row["runs"]))],
                            [r["probe_queue"] for r in row["runs"]], s=13, c="#252525", zorder=3)
    axes[0].set_xticks(range(4), LABELS)
    axes[0].set_ylabel("Probe queue delay (seconds)")
    axes[0].set_ylim(bottom=0)
    axes[0].set_ylim(top=axes[0].get_ylim()[1] * 1.2)
    axes[0].set_title("(a) Longer holders delay the next admission", loc="left", fontsize=12)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    values = data["holding_vs_wide_difference_in_differences"]
    for i, regime in enumerate(REGIMES[1:]):
        row = values[regime]
        if not row["contrast_valid"]:
            raise ValueError("E3 intended lifecycle contrast invalid")
        m, (lo, hi) = row["mean"], row["ci95"]
        axes[1].errorbar(m, i, xerr=[[m-lo], [hi-m]], fmt="o", color="#326a9b", capsize=5)
        axes[1].annotate(f"{m:.3f} [{lo:.3f}, {hi:.3f}]", (m, i), xytext=(0, 12),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_yticks(range(3), LABELS[1:])
    axes[1].set_ylim(2.5, -.6)
    axes[1].axvline(0, color="#777777", linewidth=1)
    axes[1].set_xlim(-2.5, max(v["ci95"][1] for v in values.values()) + 3)
    axes[1].set_xlabel("Extra long-vs-short queue effect relative to wide (s)")
    axes[1].set_title("(b) Cap-only control tests the attribution", loc="left", fontsize=12)
    for axis in axes:
        axis.grid(alpha=.17)
        axis.set_axisbelow(True)
    fig.suptitle("M2 / E3: holding time matters when admission slots are exhausted", fontsize=15, y=.99)
    fig.text(.065, .055, "FCFS; 8 background requests + identical probe at 3s. All 5 pairs shown; run-level bootstrap 95% intervals.", fontsize=10, color="#555555")
    fig.text(.065, .015, "Background work differs intentionally. Queue is first scheduler-observed ownership minus client submission.", fontsize=10, color="#555555")
    fig.subplots_adjust(left=.065, right=.98, top=.82, bottom=.22, wspace=.36)
    save(fig, output, "m2_hybrid_holding")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    if not data["complete_planned_matrix"]:
        raise ValueError("Expected all 132 formal runs before publication figures")
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "svg.fonttype": "none"})
    activation(data, args.output)
    candidates(data, args.output)
    holding(data, args.output)


if __name__ == "__main__":
    main()
