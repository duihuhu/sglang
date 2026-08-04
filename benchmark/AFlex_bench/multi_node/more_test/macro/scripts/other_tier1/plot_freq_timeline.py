#!/usr/bin/env python3
"""Plot GPU frequency timeline from tier1 benchmark results."""
from __future__ import annotations

import argparse, json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
FREQ_DIR = HERE / "freq_timelines"
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

FREQ_COLORS = {
    210: "#d73027", 450: "#fc8d59", 690: "#fee090",
    930: "#91bfdb", 1170: "#4575b4", 1410: "#313695",
}


def plot_single_timeline(data: dict, out_path: Path):
    """Plot frequency timeline for one config with per-GPU subplots."""
    timeline = data.get("nvidia_smi_timeline", [])
    if not timeline:
        print(f"No timeline data in {data.get('config', '?')}")
        return

    cfg_name = data.get("config", "unknown")

    # Group by (node, gpu)
    from collections import defaultdict
    grouped = defaultdict(list)
    for s in timeline:
        grouped[(s.get("node", "?"), s["gpu"])].append((s["t"], s["freq_mhz"]))

    nodes = sorted(set(k[0] for k in grouped))
    ngpu = max(k[1] for k in grouped) + 1

    fig, axes = plt.subplots(len(nodes), 1, figsize=(16, 3 * len(nodes)), sharex=True)
    if len(nodes) == 1:
        axes = [axes]

    t0 = timeline[0]["t"] if timeline else 0

    for ax_idx, node in enumerate(nodes):
        ax = axes[ax_idx]
        for gpu in range(ngpu):
            key = (node, gpu)
            if key not in grouped:
                continue
            points = grouped[key]
            ts = [p[0] - t0 for p in points]
            fs = [p[1] for p in points]

            # Color by dominant frequency
            dom_freq = max(set(fs), key=fs.count) if fs else 1410
            color = FREQ_COLORS.get(dom_freq, "#999999")

            ax.scatter(ts, [gpu] * len(ts), c=[color], s=2, alpha=0.6, marker='s')
            ax.set_ylabel(f"{node}\nGPU index")
            ax.set_yticks(range(ngpu))
            ax.set_ylim(-0.5, ngpu - 0.5)

    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(f"GPU Frequency Timeline — {cfg_name} | QPS={data.get('qps','?')} {data.get('dataset','?')}",
                 fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_comparison(files: list[Path], out_path: Path):
    """Overlay frequency distributions from multiple configs."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, node_label in enumerate(["n3", "n4"]):
        ax = axes[ax_idx]
        for fi, fp in enumerate(files):
            data = json.loads(fp.read_text())
            timeline = data.get("nvidia_smi_timeline", [])
            freqs = [s["freq_mhz"] for s in timeline if s.get("node") == node_label]
            if not freqs:
                continue
            label = data.get("config", fp.stem)
            ax.hist(freqs, bins=30, alpha=0.5, density=True, label=label)
        ax.set_xlabel("GPU SM Clock (MHz)")
        ax.set_ylabel("Density")
        ax.set_title(f"{node_label} — Frequency Distribution")
        ax.legend(fontsize=8)
        ax.axvline(930, color='green', linestyle='--', alpha=0.5, label='_930 MHz target')
        ax.axvline(1170, color='orange', linestyle='--', alpha=0.5, label='_1170 MHz target')

    fig.suptitle("GPU Frequency Distribution Comparison", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_freq_time_series(files: list[Path], out_path: Path):
    """Plot average GPU frequency over time for each config."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax_idx, node_label in enumerate(["n3", "n4"]):
        ax = axes[ax_idx]
        for fp in files:
            data = json.loads(fp.read_text())
            timeline = data.get("nvidia_smi_timeline", [])
            if not timeline:
                continue

            # Group by time bins and average freq
            node_samples = [s for s in timeline if s.get("node") == node_label]
            if not node_samples:
                continue
            t0 = node_samples[0]["t"]

            # Bin by 1-second windows
            from collections import defaultdict
            bins = defaultdict(list)
            for s in node_samples:
                bin_t = int(s["t"] - t0)
                bins[bin_t].append(s["freq_mhz"])

            ts = sorted(bins.keys())
            avg_freqs = [np.mean(bins[t]) for t in ts]
            label = data.get("config", fp.stem)
            ax.plot(ts, avg_freqs, linewidth=1.5, alpha=0.8, label=label)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Avg GPU Freq (MHz)")
        ax.set_title(f"{node_label} — Mean Freq over Time")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1600)

    fig.suptitle("GPU Frequency Time Series", fontweight="bold", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="*", default=None,
                        help="Specific freq timeline JSON files")
    parser.add_argument("--all", action="store_true",
                        help="Process all timeline files in freq_timelines/")
    args = parser.parse_args()

    if args.input:
        files = [Path(p) for p in args.input]
    elif args.all:
        files = sorted(FREQ_DIR.glob("tier1_*_code_qps16.json"))
    else:
        files = sorted(FREQ_DIR.glob("tier1_*_code_qps16.json"))

    if not files:
        print("No timeline files found!")
        return

    # Individual plots
    for fp in files:
        data = json.loads(fp.read_text())
        cfg = data.get("config", fp.stem)
        out = CHARTS_DIR / f"freq_timeline_{cfg}.png"
        plot_single_timeline(data, out)

    # Comparison plots (if multiple files)
    if len(files) >= 2:
        plot_comparison(files, CHARTS_DIR / "freq_dist_comparison.png")
        plot_freq_time_series(files, CHARTS_DIR / "freq_timeseries_comparison.png")


if __name__ == "__main__":
    main()
