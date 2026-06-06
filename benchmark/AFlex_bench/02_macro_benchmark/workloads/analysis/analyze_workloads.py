"""
Workload analysis script: generates characterization figures for all variable workloads.
Outputs: QPS over time, input/output length distributions, and summary comparison.
"""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

WORKLOAD_DIR = Path(__file__).parent.parent
OUTPUT_DIR = Path(__file__).parent / "figures"
OUTPUT_DIR.mkdir(exist_ok=True)

WORKLOADS = {
    "steady": "workload_steady.jsonl",
    "varying": "workload_varying.jsonl",
    "heavy": "workload_heavy.jsonl",
    "overload": "workload_overload.jsonl",
    "tier1_demo": "workload_tier1_demo.jsonl",
}

COLORS = {
    "steady": "#2196F3",
    "varying": "#FF9800",
    "heavy": "#F44336",
    "overload": "#9C27B0",
    "tier1_demo": "#4CAF50",
}


def load_workload(name):
    path = WORKLOAD_DIR / WORKLOADS[name]
    data = [json.loads(line) for line in open(path)]
    return data


def compute_qps_over_time(data, bin_size=5.0):
    """Compute QPS in sliding bins."""
    times = np.array([d["arrival_time_s"] for d in data])
    max_t = times.max()
    bins = np.arange(0, max_t + bin_size, bin_size)
    counts, _ = np.histogram(times, bins=bins)
    qps = counts / bin_size
    bin_centers = (bins[:-1] + bins[1:]) / 2
    return bin_centers, qps


def plot_qps_over_time():
    """Plot QPS distribution over time for all workloads."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, (name, _) in enumerate(WORKLOADS.items()):
        ax = axes[i]
        data = load_workload(name)
        t, qps = compute_qps_over_time(data, bin_size=5.0)
        ax.bar(t, qps, width=4.5, color=COLORS[name], alpha=0.7, edgecolor="white", linewidth=0.5)
        ax.axhline(np.mean(qps), color="black", linestyle="--", linewidth=1, alpha=0.6, label=f"avg={np.mean(qps):.1f}")
        ax.set_title(f"{name} (n={len(data)})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("QPS")
        ax.legend(loc="upper right", fontsize=9)
        ax.set_ylim(bottom=0)

        # annotate phases
        phases = {}
        for d in data:
            p = d["phase"]
            if p not in phases:
                phases[p] = [d["arrival_time_s"], d["arrival_time_s"]]
            phases[p][1] = max(phases[p][1], d["arrival_time_s"])

        for j, (phase, (t_start, t_end)) in enumerate(sorted(phases.items(), key=lambda x: x[1][0])):
            y_pos = ax.get_ylim()[1] * (0.85 - 0.08 * (j % 3))
            ax.annotate(phase, xy=((t_start + t_end) / 2, y_pos), fontsize=7, ha="center", color="gray")

    axes[-1].axis("off")
    fig.suptitle("QPS Over Time (5s bins)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "qps_over_time.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'qps_over_time.png'}")


def plot_length_distributions():
    """Plot input/output length distributions for all workloads."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 6))

    for i, (name, _) in enumerate(WORKLOADS.items()):
        data = load_workload(name)
        input_lens = [d["input_len"] for d in data]
        output_lens = [d["output_len"] for d in data]

        ax_in = axes[0, i]
        ax_in.hist(input_lens, bins=30, color=COLORS[name], alpha=0.7, edgecolor="white")
        ax_in.axvline(np.mean(input_lens), color="black", linestyle="--", linewidth=1)
        ax_in.set_title(f"{name}\n(μ={np.mean(input_lens):.0f})")
        if i == 0:
            ax_in.set_ylabel("Count (Input Len)")

        ax_out = axes[1, i]
        ax_out.hist(output_lens, bins=30, color=COLORS[name], alpha=0.7, edgecolor="white")
        ax_out.axvline(np.mean(output_lens), color="black", linestyle="--", linewidth=1)
        ax_out.set_title(f"(μ={np.mean(output_lens):.0f})")
        ax_out.set_xlabel("Length (tokens)")
        if i == 0:
            ax_out.set_ylabel("Count (Output Len)")

    fig.suptitle("Input / Output Length Distributions", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "length_distributions.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'length_distributions.png'}")


def plot_io_scatter():
    """Scatter plot of input vs output length, colored by workload."""
    fig, ax = plt.subplots(figsize=(8, 6))

    for name in WORKLOADS:
        data = load_workload(name)
        il = [d["input_len"] for d in data]
        ol = [d["output_len"] for d in data]
        ax.scatter(il, ol, color=COLORS[name], alpha=0.3, s=12, label=name)

    ax.set_xlabel("Input Length (tokens)")
    ax.set_ylabel("Output Length (tokens)")
    ax.set_title("Input vs Output Length (all workloads)")
    ax.legend(markerscale=3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "io_scatter.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'io_scatter.png'}")


def plot_summary_comparison():
    """Bar chart comparing key stats across workloads."""
    names = list(WORKLOADS.keys())
    stats = {}
    for name in names:
        data = load_workload(name)
        times = [d["arrival_time_s"] for d in data]
        duration = max(times) - min(times)
        stats[name] = {
            "requests": len(data),
            "duration_s": duration,
            "avg_qps": len(data) / duration if duration > 0 else 0,
            "avg_il": np.mean([d["input_len"] for d in data]),
            "avg_ol": np.mean([d["output_len"] for d in data]),
            "total_tokens": sum(d["input_len"] + d["output_len"] for d in data),
            "peak_qps": max(compute_qps_over_time(data, bin_size=5.0)[1]),
        }

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    metrics = [
        ("requests", "Request Count", ""),
        ("avg_qps", "Average QPS", "req/s"),
        ("peak_qps", "Peak QPS (5s bin)", "req/s"),
        ("avg_il", "Avg Input Length", "tokens"),
        ("avg_ol", "Avg Output Length", "tokens"),
        ("total_tokens", "Total Tokens (I+O)", "tokens"),
    ]

    for idx, (key, title, unit) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        vals = [stats[n][key] for n in names]
        bars = ax.bar(names, vals, color=[COLORS[n] for n in names], alpha=0.8, edgecolor="white")
        ax.set_title(title)
        if unit:
            ax.set_ylabel(unit)
        ax.tick_params(axis="x", rotation=15)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{v:.0f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Workload Summary Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "summary_comparison.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'summary_comparison.png'}")


def plot_phase_breakdown():
    """Stacked bar chart showing phase composition of each workload."""
    fig, axes = plt.subplots(1, 5, figsize=(16, 4), sharey=False)

    for i, (name, _) in enumerate(WORKLOADS.items()):
        ax = axes[i]
        data = load_workload(name)
        phases = {}
        for d in data:
            p = d["phase"]
            if p not in phases:
                phases[p] = 0
            phases[p] += 1

        sorted_phases = sorted(phases.items(), key=lambda x: -x[1])
        labels = [p[0] for p in sorted_phases]
        sizes = [p[1] for p in sorted_phases]
        ax.pie(sizes, labels=labels, autopct="%1.0f%%", textprops={"fontsize": 8})
        ax.set_title(f"{name}\n({len(data)} reqs)")

    fig.suptitle("Phase Composition", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "phase_breakdown.png")
    plt.close()
    print(f"Saved: {OUTPUT_DIR / 'phase_breakdown.png'}")


if __name__ == "__main__":
    print("Analyzing workloads...")
    plot_qps_over_time()
    plot_length_distributions()
    plot_io_scatter()
    plot_summary_comparison()
    plot_phase_breakdown()
    print("Done! All figures saved to:", OUTPUT_DIR)
