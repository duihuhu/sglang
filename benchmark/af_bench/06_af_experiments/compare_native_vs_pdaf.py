#!/usr/bin/env python3
"""Compare Native vs PD+AF: TTFT, TPOT, throughput, energy charts."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_results(path: str, label: str):
    """Load a benchmark result JSON, return extracted metrics."""
    with open(path) as f:
        raw = json.load(f)
    ok = [r for r in raw.get("results", []) if r.get("success")]
    if not ok:
        print(f"WARNING: {path} has no successful requests")
        return None
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    ttft_total = [r["ttft_total_ms"] for r in ok if r.get("ttft_total_ms", 0) > 0]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    latency = [r["latency_s"] for r in ok]
    total_in = sum(r.get("input_tokens", 0) for r in ok)
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    wall = raw.get("wall_duration_s", 1)
    energy_mj = raw.get("energy_mj_delta", {})
    total_energy_j = sum(energy_mj.values()) / 1000 if energy_mj else 0
    return {
        "label": label,
        "n": len(ok),
        "ttft": ttft,
        "ttft_total": ttft_total,
        "tpot": tpot,
        "latency": latency,
        "total_in": total_in,
        "total_out": total_out,
        "wall_s": wall,
        "energy_j": total_energy_j,
        "out_tps": total_out / wall if wall > 0 else 0,
    }


def plot_comparison(native, pdaf, save_dir: str):
    """Generate 2x2 comparison chart panel."""
    archs = [native, pdaf]
    colors = ["#4C72B0", "#DD8452"]

    # ── 2x2 panel ────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Native vs PD+AF — Performance Comparison", fontsize=16, fontweight="bold", y=0.98)

    # 1. TTFT box plot
    ax = axes[0, 0]
    bp = ax.boxplot([a["ttft"] for a in archs], patch_artist=True, widths=0.4)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([a["label"] for a in archs], fontsize=11)
    ax.set_ylabel("TTFT (ms)", fontsize=12)
    ax.set_title("TTFT Comparison (lower is better)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    # Annotate medians
    for i, a in enumerate(archs):
        med = np.median(a["ttft"])
        ax.annotate(f"P50={med:.0f}ms", xy=(i+1, med), xytext=(i+1.35, med),
                    fontsize=9, color=colors[i], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=colors[i], alpha=0.5))

    # 2. TPOT box plot
    ax = axes[0, 1]
    bp = ax.boxplot([a["tpot"] for a in archs], patch_artist=True, widths=0.4)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticklabels([a["label"] for a in archs], fontsize=11)
    ax.set_ylabel("TPOT (ms)", fontsize=12)
    ax.set_title("TPOT Comparison (lower is better)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, a in enumerate(archs):
        med = np.median(a["tpot"])
        ax.annotate(f"P50={med:.0f}ms", xy=(i+1, med), xytext=(i+1.35, med),
                    fontsize=9, color=colors[i], fontweight="bold",
                    arrowprops=dict(arrowstyle="-", color=colors[i], alpha=0.5))

    # 3. Output throughput + energy bar chart
    ax = axes[1, 0]
    x = np.arange(len(archs))
    w = 0.3
    bars1 = ax.bar(x - w/2, [a["out_tps"] for a in archs], w, label="Output tok/s",
                   color=colors, alpha=0.8, edgecolor="gray", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([a["label"] for a in archs], fontsize=11)
    ax.set_ylabel("Output Tokens / s", fontsize=12, color=colors[0])
    ax.set_title("Throughput & Energy", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, a in zip(bars1, archs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{a['out_tps']:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax2 = ax.twinx()
    bars2 = ax2.bar(x + w/2, [a["energy_j"] for a in archs], w, label="Energy (J)",
                    color=colors, alpha=0.3, hatch="///", edgecolor="gray", linewidth=0.5)
    ax2.set_ylabel("Energy (J)", fontsize=12, color=colors[1])
    for bar, a in zip(bars2, archs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                f"{a['energy_j']:.0f}J", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # 4. Efficiency: Throughput / Energy
    ax = axes[1, 1]
    eff = [a["out_tps"] / a["energy_j"] * 1000 if a["energy_j"] > 0 else 0 for a in archs]
    bars = ax.bar([a["label"] for a in archs], eff, color=colors, alpha=0.8,
                  edgecolor="gray", linewidth=0.5, width=0.4)
    ax.set_ylabel("Output tok / kJ", fontsize=12)
    ax.set_title("Energy Efficiency (higher is better)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for bar, v in zip(bars, eff):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{v:.1f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    path = Path(save_dir) / "native_vs_pdaf_comparison.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {path}")
    plt.close(fig)

    # ── Summary table ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"  {'Metric':<30} {'Native':<18} {'PD+AF':<18}")
    print("  " + "-" * 66)
    rows = [
        ("Requests succeeded", f"{native['n']}", f"{pdaf['n']}"),
        ("Wall duration (s)", f"{native['wall_s']:.1f}", f"{pdaf['wall_s']:.1f}"),
        ("Output throughput (tok/s)", f"{native['out_tps']:.1f}", f"{pdaf['out_tps']:.1f}"),
        ("Mean TTFT (ms)", f"{np.mean(native['ttft']):.1f}", f"{np.mean(pdaf['ttft']):.1f}"),
        ("P50 TTFT (ms)", f"{np.median(native['ttft']):.1f}", f"{np.median(pdaf['ttft']):.1f}"),
        ("P95 TTFT (ms)", f"{np.percentile(native['ttft'], 95):.1f}", f"{np.percentile(pdaf['ttft'], 95):.1f}"),
        ("Mean TPOT (ms)", f"{np.mean(native['tpot']):.1f}", f"{np.mean(pdaf['tpot']):.1f}"),
        ("P50 TPOT (ms)", f"{np.median(native['tpot']):.1f}", f"{np.median(pdaf['tpot']):.1f}"),
        ("P95 TPOT (ms)", f"{np.percentile(native['tpot'], 95):.1f}", f"{np.percentile(pdaf['tpot'], 95):.1f}"),
        ("Total Energy (J)", f"{native['energy_j']:.0f}", f"{pdaf['energy_j']:.0f}"),
        ("Energy Efficiency (tok/kJ)", f"{eff[0]:.2f}", f"{eff[1]:.2f}"),
    ]
    for key, v1, v2 in rows:
        print(f"  {key:<30} {v1:<18} {v2:<18}")
    print("=" * 70)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare Native vs PD+AF")
    parser.add_argument("--native", default="/workspace/sglang/af_launch_logs/results/results_native.json")
    parser.add_argument("--pdaf", default="/workspace/sglang/af_launch_logs/results/results_pdaf.json")
    parser.add_argument("--save-dir", default="/workspace/sglang/af_launch_logs/reports")
    args = parser.parse_args()

    native = load_results(args.native, "Native")
    pdaf = load_results(args.pdaf, "PD+AF")
    if native is None or pdaf is None:
        print("ERROR: missing result files")
        sys.exit(1)

    plot_comparison(native, pdaf, args.save_dir)


if __name__ == "__main__":
    main()
