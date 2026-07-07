#!/usr/bin/env python3
"""Plot per-instance GPU frequency timeline from monitor_biscale_freq_timeline.py."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CHARTS = HERE / "charts"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def plot_timeline(samples: list[dict], out: Path, title: str):
    if not samples:
        print("no samples")
        return
    t0 = samples[0]["t"]
    by_inst: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for s in samples:
        by_inst[s["instance"]].append((s["t"] - t0, s["freq_mhz"]))

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, phase, label in [(axes[0], "P", "Prefill (node1)"),
                             (axes[1], "D", "Decode (node2)")]:
        insts = sorted(k for k in by_inst if k.startswith(phase))
        for inst in insts:
            xs, ys = zip(*by_inst[inst])
            ax.step(xs, ys, where="post", linewidth=1.2, label=inst)
        ax.set_ylabel("SM freq (MHz)")
        ax.set_title(f"{label} — {title}")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=4, fontsize=7, loc="upper right")
    axes[1].set_xlabel("Time since capture start (s)")
    fig.suptitle(title, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def plot_workload_panels(samples: list[dict], out: Path):
    by_wl: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        by_wl[s.get("workload", "unknown")].append(s)
    wls = [w for w in sorted(by_wl) if w.startswith("code_qps")]
    if not wls:
        plot_timeline(samples, out, "BiScale freq timeline")
        return
    n = len(wls)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    for ax, wl in zip(axes, wls):
        sub = by_wl[wl]
        t0 = sub[0]["t"]
        by_inst = defaultdict(list)
        for s in sub:
            by_inst[s["instance"]].append((s["t"] - t0, s["freq_mhz"]))
        for inst in sorted(by_inst):
            xs, ys = zip(*by_inst[inst])
            ax.step(xs, ys, where="post", linewidth=1.0, label=inst)
        ax.set_title(wl)
        ax.set_ylabel("MHz")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=8, fontsize=6, loc="upper right")
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("BiScale per-instance frequency by workload", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path,
                        default=HERE / "results" / "freq_timeline_biscale_code")
    parser.add_argument("--out-dir", type=Path, default=CHARTS)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = args.data_dir / "nvidia_smi_timeline.jsonl"
    samples = load_jsonl(jsonl) if jsonl.exists() else []
    plot_timeline(samples, args.out_dir / "biscale_code_freq_timeline.png",
                  "BiScale code — all workloads")
    plot_workload_panels(samples, args.out_dir / "biscale_code_freq_by_qps.png")


if __name__ == "__main__":
    main()
