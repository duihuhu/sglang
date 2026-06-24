#!/usr/bin/env python3
"""Plot reshard benchmark results as timeline charts.

Reads reshard_results.json and generates:
1. Per-request TTFT timeline with reload events
2. Per-request TPOT timeline with reload events
3. Phase-level throughput bar chart
4. Phase-level energy comparison (total + per-token)
5. Combined overview panel

Usage:
    python plot_reshard.py [--input results/reshard_results.json]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"


def load_results(path):
    with open(path) as f:
        return json.load(f)


def plot_timeline_metric(results, metric, ylabel, title, events, out_path,
                         slo_line=None):
    """Plot per-request metric over time with reload event markers."""
    fig, ax = plt.subplots(figsize=(14, 5))

    requests = results["requests"]
    ok = [r for r in requests if r["success"]]

    timestamps = [r["timestamp"] for r in ok]
    values = [r[metric] for r in ok]

    # Color by phase
    phase_colors = {
        "phase1_low": "#2196F3",
        "phase2_stress": "#FF9800",
        "phase4_high": "#4CAF50",
        "phase5_reduce": "#9C27B0",
        "phase6_low": "#00BCD4",
    }
    colors = [phase_colors.get(r["phase"], "#999") for r in ok]

    ax.scatter(timestamps, values, c=colors, alpha=0.6, s=12, edgecolors="none")

    # Draw reload event bands
    phases = results["phases"]
    for phase in phases:
        if phase.get("is_reload"):
            ax.axvspan(phase["start_time"], phase["end_time"],
                       alpha=0.15, color="red", label=phase["phase"])
            mid = (phase["start_time"] + phase["end_time"]) / 2
            ax.axvline(mid, color="red", linestyle="--", alpha=0.5)
            ax.text(mid, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else max(values) * 0.9,
                    phase["phase"].replace("reload_", "⟳ "),
                    ha="center", fontsize=9, color="red", fontweight="bold")

    # Event annotations
    for ts, name in events:
        if "reload" not in name:
            ax.axvline(ts, color="gray", linestyle=":", alpha=0.4)

    # SLO line
    if slo_line:
        ax.axhline(slo_line, color="red", linestyle="-", alpha=0.3, linewidth=1.5)
        ax.text(max(timestamps) * 0.98, slo_line * 1.05, "SLO",
                ha="right", color="red", fontsize=9)

    # Legend
    legend_patches = [mpatches.Patch(color=c, label=p.replace("_", " ").title())
                      for p, c in phase_colors.items()]
    legend_patches.append(mpatches.Patch(color="red", alpha=0.3, label="Reload"))
    ax.legend(handles=legend_patches, loc="upper left", fontsize=8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_phase_bars(results, out_path):
    """Bar chart comparing throughput and energy across phases."""
    phases = [p for p in results["phases"] if not p.get("is_reload")]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    names = [p["phase"].replace("phase", "P").replace("_", "\n") for p in phases]
    x = np.arange(len(names))
    width = 0.6

    # Throughput
    ax = axes[0, 0]
    thpts = [p["throughput_tok_s"] for p in phases]
    bars = ax.bar(x, thpts, width, color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"])
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Throughput by Phase")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    for bar, v in zip(bars, thpts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{v:.0f}", ha="center", fontsize=8)

    # TTFT avg
    ax = axes[0, 1]
    ttfts = [p["ttft_avg_ms"] for p in phases]
    bars = ax.bar(x, ttfts, width, color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"])
    ax.set_ylabel("TTFT avg (ms)")
    ax.set_title("TTFT Average by Phase")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    ax.axhline(5000, color="red", linestyle="--", alpha=0.5, label="SLO")
    ax.legend(fontsize=8)

    # Energy per token
    ax = axes[1, 0]
    epts = [p["energy_per_token_mj"] for p in phases]
    bars = ax.bar(x, epts, width, color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"])
    ax.set_ylabel("Energy per Token (mJ)")
    ax.set_title("Energy Efficiency by Phase")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)
    for bar, v in zip(bars, epts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f"{v:.1f}", ha="center", fontsize=8)

    # Total energy
    ax = axes[1, 1]
    energies = [p["total_energy_j"] for p in phases]
    bars = ax.bar(x, energies, width, color=["#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#00BCD4"])
    ax.set_ylabel("Total Energy (J)")
    ax.set_title("Total Energy by Phase")
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=8)

    # Add TP/GPU annotations
    for i, p in enumerate(phases):
        ax.text(i, -max(energies) * 0.08,
                f"TP{p['tp']} / {p['ngpu']}G",
                ha="center", fontsize=8, color="gray")

    fig.suptitle("Reshard Benchmark: Phase Comparison", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def plot_combined_timeline(results, out_path):
    """4-panel timeline: TTFT, TPOT, throughput (windowed), config."""
    fig, axes = plt.subplots(4, 1, figsize=(16, 14), sharex=True)

    requests = results["requests"]
    ok = [r for r in requests if r["success"]]
    phases = results["phases"]
    events = results.get("events", [])

    timestamps = np.array([r["timestamp"] for r in ok])
    ttfts = np.array([r["ttft_ms"] for r in ok])
    tpots = np.array([r["tpot_ms"] for r in ok])

    # Reload bands
    for ax in axes:
        for phase in phases:
            if phase.get("is_reload"):
                ax.axvspan(phase["start_time"], phase["end_time"],
                           alpha=0.12, color="red")

    # Panel 1: TTFT
    ax = axes[0]
    ax.scatter(timestamps, ttfts, s=8, alpha=0.5, c="#2196F3", edgecolors="none")
    ax.axhline(TTFT_SLO_MS, color="red", linestyle="--", alpha=0.4)
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Per-Request TTFT")
    ax.set_ylim(0, min(max(ttfts) * 1.2, TTFT_SLO_MS * 3) if len(ttfts) > 0 else 10000)
    ax.grid(True, alpha=0.2)

    # Panel 2: TPOT
    ax = axes[1]
    ax.scatter(timestamps, tpots, s=8, alpha=0.5, c="#4CAF50", edgecolors="none")
    ax.axhline(TPOT_SLO_MS, color="red", linestyle="--", alpha=0.4)
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("Per-Request TPOT")
    ax.grid(True, alpha=0.2)

    # Panel 3: Windowed throughput (10s windows)
    ax = axes[2]
    if len(timestamps) > 0:
        window_s = 10
        t_max = max(timestamps)
        windows = np.arange(0, t_max, window_s)
        thpts = []
        for t_start in windows:
            mask = (timestamps >= t_start) & (timestamps < t_start + window_s)
            tokens = sum(r["tokens"] for r, m in zip(ok, mask) if m)
            thpts.append(tokens / window_s)
        ax.bar(windows, thpts, width=window_s * 0.9, alpha=0.7, color="#FF9800")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Windowed Throughput (10s)")
    ax.grid(True, alpha=0.2)

    # Panel 4: Configuration timeline
    ax = axes[3]
    for phase in phases:
        if not phase.get("is_reload"):
            color = "#2196F3" if phase["ngpu"] == 4 else "#4CAF50"
            ax.barh(0, phase["end_time"] - phase["start_time"],
                    left=phase["start_time"], height=0.5,
                    color=color, alpha=0.7)
            mid = (phase["start_time"] + phase["end_time"]) / 2
            ax.text(mid, 0, f"TP{phase['tp']}\n{phase['ngpu']}GPU\nQPS={phase['qps']}",
                    ha="center", va="center", fontsize=7, fontweight="bold")
    ax.set_ylim(-0.5, 0.5)
    ax.set_yticks([])
    ax.set_ylabel("Config")
    ax.set_xlabel("Time (s)")
    ax.set_title("Deployment Configuration")

    # Reload annotations on all panels
    for phase in phases:
        if phase.get("is_reload"):
            mid = (phase["start_time"] + phase["end_time"]) / 2
            axes[3].axvline(mid, color="red", linestyle="--", alpha=0.7)
            axes[3].text(mid, 0.3, phase["phase"].replace("reload_", "⟳"),
                         ha="center", fontsize=8, color="red")

    fig.suptitle("Reshard Benchmark: Graceful Reload Timeline\n"
                 f"Qwen3-32B PDAF | 4GPU TP1 → 8GPU TP2 → 4GPU TP1 | "
                 f"IL={INPUT_LEN} OL={OUTPUT_LEN}",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0
INPUT_LEN = 512
OUTPUT_LEN = 128


def main():
    parser = argparse.ArgumentParser(description="Plot reshard benchmark results")
    parser.add_argument("--input", type=str,
                        default=str(RESULTS_DIR / "reshard_results.json"),
                        help="Path to reshard_results.json")
    args = parser.parse_args()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from: {args.input}")
    results = load_results(args.input)
    events = results.get("events", [])

    print("Generating charts...")

    # 1. TTFT timeline
    plot_timeline_metric(
        results, "ttft_ms", "TTFT (ms)",
        "Per-Request TTFT During Reshard (Qwen3-32B PDAF)",
        events, CHARTS_DIR / "reshard_ttft_timeline.png",
        slo_line=TTFT_SLO_MS)

    # 2. TPOT timeline
    plot_timeline_metric(
        results, "tpot_ms", "TPOT (ms)",
        "Per-Request TPOT During Reshard (Qwen3-32B PDAF)",
        events, CHARTS_DIR / "reshard_tpot_timeline.png",
        slo_line=TPOT_SLO_MS)

    # 3. Phase bars
    plot_phase_bars(results, CHARTS_DIR / "reshard_phase_comparison.png")

    # 4. Combined timeline
    plot_combined_timeline(results, CHARTS_DIR / "reshard_combined_timeline.png")

    print(f"\nAll charts saved to: {CHARTS_DIR}")


if __name__ == "__main__":
    main()
