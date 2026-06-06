#!/usr/bin/env python3
"""Analyze joint TTFT x TPOT SLO sweep v2: heatmaps with ttft_proc (excl. queuing).

Generates a 2x2 heatmap similar to joint_sweep_slo_violations.png but using
ttft_proc_avg_ms instead of ttft_avg_ms for the TTFT panel.
"""
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results_joint_sweep_v2"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

TTFT_SLOS = [5000, 2000, 1000, 500, 300, 200]
TPOT_SLOS = [300, 250, 200, 150, 100, 90, 80, 70]


def load_result(ttft, tpot, scheme):
    json_dir = RESULTS_DIR / f"ttft_{ttft}_tpot_{tpot}" / scheme / "json"
    if not json_dir.exists():
        return None
    files = list(json_dir.glob("*_results.json"))
    if not files:
        # Try 4gpu_summary.json
        summary = json_dir / "4gpu_summary.json"
        if summary.exists():
            with open(summary) as f:
                data = json.load(f)
            for dk, wl in data.items():
                if isinstance(wl, dict):
                    for wk, qps in wl.items():
                        if isinstance(qps, dict):
                            for qk, r in qps.items():
                                if isinstance(r, dict) and "throughput_tok_s" in r:
                                    return r
        return None
    with open(files[0]) as f:
        return json.load(f)


def main():
    n_ttft = len(TTFT_SLOS)
    n_tpot = len(TPOT_SLOS)

    # Data arrays
    saving = np.full((n_ttft, n_tpot), np.nan)
    slo_viol_v2 = np.full((n_ttft, n_tpot), np.nan)
    slo_viol_bl = np.full((n_ttft, n_tpot), np.nan)
    ttft_proc_v2 = np.full((n_ttft, n_tpot), np.nan)
    ttft_proc_bl = np.full((n_ttft, n_tpot), np.nan)
    tok_slo_v2 = np.full((n_ttft, n_tpot), np.nan)

    for i, ttft in enumerate(TTFT_SLOS):
        for j, tpot in enumerate(TPOT_SLOS):
            r_v2 = load_result(ttft, tpot, "v2")
            r_bl = load_result(ttft, tpot, "baseline")

            if r_v2:
                e_v2 = r_v2.get("total_energy_j", 0)
                slo_viol_v2[i, j] = r_v2.get("slo_violation_rate", 0)
                ttft_proc_v2[i, j] = r_v2.get("ttft_proc_avg_ms", 0)
                tok_slo_v2[i, j] = r_v2.get("tpot_per_token_viol_rate", 0) * 100

                if r_bl:
                    e_bl = r_bl.get("total_energy_j", 0)
                    slo_viol_bl[i, j] = r_bl.get("slo_violation_rate", 0)
                    ttft_proc_bl[i, j] = r_bl.get("ttft_proc_avg_ms", 0)
                    if e_bl > 0:
                        saving[i, j] = (1 - e_v2 / e_bl) * 100

    # Check data availability
    valid = np.sum(~np.isnan(saving))
    print(f"Valid data points: {valid}/{n_ttft * n_tpot}")
    if valid == 0:
        print("No data available yet. Exiting.")
        return

    # Plot heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Joint TTFT x TPOT SLO Sweep — V2 DVFS\n"
                 "(TTFT = Processing Time, Excl. Queuing)",
                 fontsize=13, fontweight='bold')

    ttft_labels = [str(s) for s in TTFT_SLOS]
    tpot_labels = [str(s) for s in TPOT_SLOS]

    def plot_heatmap(ax, data, title, cmap, vmin=None, vmax=None, fmt=".1f"):
        im = ax.imshow(data, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(n_tpot))
        ax.set_xticklabels(tpot_labels)
        ax.set_yticks(range(n_ttft))
        ax.set_yticklabels(ttft_labels)
        ax.set_xlabel("TPOT SLO (ms)")
        ax.set_ylabel("TTFT SLO (ms)")
        ax.set_title(title)
        for ii in range(n_ttft):
            for jj in range(n_tpot):
                val = data[ii, jj]
                if not np.isnan(val):
                    threshold = (vmax or np.nanmax(data)) * 0.6
                    color = 'white' if val > threshold else 'black'
                    ax.text(jj, ii, f"{val:{fmt}}", ha='center', va='center',
                            fontsize=7, color=color)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plot_heatmap(axes[0, 0], saving,
                 "Energy Saving vs Baseline (%)",
                 'RdYlGn', vmin=0, vmax=35, fmt=".1f")

    plot_heatmap(axes[0, 1], slo_viol_v2,
                 "Overall SLO Violation Rate — V2 (%)",
                 'RdYlGn_r', vmin=0, vmax=100, fmt=".0f")

    plot_heatmap(axes[1, 0], ttft_proc_v2,
                 "Actual TTFT Processing Time — V2 (ms)\n(Excl. Queuing)",
                 'YlOrRd', fmt=".0f")

    plot_heatmap(axes[1, 1], tok_slo_v2,
                 "Per-Token TPOT SLO Violation — V2 (%)",
                 'RdYlGn_r', vmin=0, vmax=100, fmt=".1f")

    plt.tight_layout()
    out_path = OUT_DIR / "joint_sweep_slo_violations_v2.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {out_path}")

    # Print summary
    print(f"\n{'TTFT':>6} {'TPOT':>6} | {'Save%':>6} | {'SLO%':>6} | {'TTFT proc':>10} | {'TokViol%':>8}")
    print("-" * 60)
    for i, ttft in enumerate(TTFT_SLOS):
        for j, tpot in enumerate(TPOT_SLOS):
            sv = saving[i, j]
            sl = slo_viol_v2[i, j]
            tp = ttft_proc_v2[i, j]
            tv = tok_slo_v2[i, j]
            print(f"{ttft:>6} {tpot:>6} | {sv:>5.1f}% | {sl:>5.1f}% | {tp:>9.0f}ms | {tv:>7.1f}%")


if __name__ == "__main__":
    main()
