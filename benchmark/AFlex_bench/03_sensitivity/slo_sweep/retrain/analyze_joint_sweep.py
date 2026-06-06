#!/usr/bin/env python3
"""Analyze joint TTFT x TPOT SLO sweep results: heatmaps + summary CSV."""
import json
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).resolve().parent / "results_joint_sweep"
OUT_DIR = Path(__file__).resolve().parent / "figures"
OUT_DIR.mkdir(exist_ok=True)

TTFT_SLOS = [5000, 2000, 1000, 500, 300, 200]
TPOT_SLOS = [300, 250, 200, 150, 100, 90, 80, 70]


def load_result(ttft, tpot, scheme):
    json_dir = RESULTS_DIR / f"ttft_{ttft}_tpot_{tpot}" / scheme / "json"
    if not json_dir.exists():
        return None
    files = list(json_dir.glob("*.json"))
    if not files:
        return None
    with open(files[0]) as f:
        data = json.load(f)
    for dk, wl in data.items():
        if isinstance(wl, dict):
            for wk, qps in wl.items():
                if isinstance(qps, dict):
                    for qk, r in qps.items():
                        if isinstance(r, dict) and "throughput_tok_s" in r:
                            return r
    return None


def main():
    n_ttft = len(TTFT_SLOS)
    n_tpot = len(TPOT_SLOS)

    energy_v2 = np.full((n_ttft, n_tpot), np.nan)
    energy_bl = np.full((n_ttft, n_tpot), np.nan)
    saving = np.full((n_ttft, n_tpot), np.nan)
    slo_viol_v2 = np.full((n_ttft, n_tpot), np.nan)
    slo_viol_bl = np.full((n_ttft, n_tpot), np.nan)
    ttft_actual_v2 = np.full((n_ttft, n_tpot), np.nan)
    tpot_actual_v2 = np.full((n_ttft, n_tpot), np.nan)
    tok_slo_v2 = np.full((n_ttft, n_tpot), np.nan)

    csv_rows = []

    for i, ttft in enumerate(TTFT_SLOS):
        for j, tpot in enumerate(TPOT_SLOS):
            r_v2 = load_result(ttft, tpot, "v2")
            r_bl = load_result(ttft, tpot, "baseline")

            row = {"ttft_slo": ttft, "tpot_slo": tpot}

            if r_v2:
                e_v2 = r_v2.get("total_energy_j", 0)
                energy_v2[i, j] = e_v2
                slo_viol_v2[i, j] = r_v2.get("slo_violation_rate", 0)
                ttft_actual_v2[i, j] = r_v2.get("ttft_avg_ms", 0)
                tpot_actual_v2[i, j] = r_v2.get("tpot_avg_ms", 0)
                tok_slo_v2[i, j] = r_v2.get("tpot_per_token_viol_rate", 0)
                row.update({
                    "v2_energy_j": e_v2,
                    "v2_thpt": r_v2.get("throughput_tok_s", 0),
                    "v2_ttft_ms": r_v2.get("ttft_avg_ms", 0),
                    "v2_tpot_ms": r_v2.get("tpot_avg_ms", 0),
                    "v2_slo_viol": r_v2.get("slo_violation_rate", 0),
                    "v2_tok_slo_viol": r_v2.get("tpot_per_token_viol_rate", 0),
                })

            if r_bl:
                e_bl = r_bl.get("total_energy_j", 0)
                energy_bl[i, j] = e_bl
                slo_viol_bl[i, j] = r_bl.get("slo_violation_rate", 0)
                row.update({
                    "bl_energy_j": e_bl,
                    "bl_thpt": r_bl.get("throughput_tok_s", 0),
                    "bl_ttft_ms": r_bl.get("ttft_avg_ms", 0),
                    "bl_tpot_ms": r_bl.get("tpot_avg_ms", 0),
                    "bl_slo_viol": r_bl.get("slo_violation_rate", 0),
                })

            if r_v2 and r_bl and e_bl > 0:
                saving[i, j] = (1 - e_v2 / e_bl) * 100
                row["energy_saving_pct"] = saving[i, j]

            csv_rows.append(row)

    # Save CSV
    csv_path = RESULTS_DIR / "joint_sweep_summary.csv"
    if csv_rows:
        keys = csv_rows[0].keys()
        all_keys = sorted(set(k for row in csv_rows for k in row.keys()))
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=all_keys)
            w.writeheader()
            w.writerows(csv_rows)
        print(f"CSV saved: {csv_path}")

    # Plot heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Joint TTFT x TPOT SLO Sweep (V2 DVFS Model)",
                 fontsize=14, fontweight='bold')

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
                    ax.text(jj, ii, f"{val:{fmt}}", ha='center', va='center',
                            fontsize=7, color='white' if val > (vmax or np.nanmax(data)) * 0.6 else 'black')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plot_heatmap(axes[0, 0], saving, "Energy Saving vs Baseline (%)",
                 'RdYlGn', vmin=0, vmax=35, fmt=".1f")
    plot_heatmap(axes[0, 1], slo_viol_v2, "SLO Violation Rate - V2 (%)",
                 'RdYlGn_r', vmin=0, vmax=100, fmt=".0f")
    plot_heatmap(axes[1, 0], ttft_actual_v2, "Actual TTFT - V2 (ms)",
                 'YlOrRd', fmt=".0f")
    plot_heatmap(axes[1, 1], tok_slo_v2, "Per-Token TPOT SLO Violation - V2 (%)",
                 'RdYlGn_r', vmin=0, vmax=50, fmt=".1f")

    plt.tight_layout()
    out_path = OUT_DIR / "joint_slo_sweep_heatmap.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Heatmap saved: {out_path}")

    # Print summary table
    print(f"\n{'TTFT':>6} {'TPOT':>6} | {'V2 E(J)':>8} {'BL E(J)':>8} {'Save%':>6} | "
          f"{'V2 SLO%':>8} {'BL SLO%':>8} | {'V2 TTFT':>8} {'V2 TPOT':>8}")
    print("-" * 90)
    for i, ttft in enumerate(TTFT_SLOS):
        for j, tpot in enumerate(TPOT_SLOS):
            ev = energy_v2[i, j]
            eb = energy_bl[i, j]
            sv = saving[i, j]
            sv2 = slo_viol_v2[i, j]
            sb = slo_viol_bl[i, j]
            tv = ttft_actual_v2[i, j]
            pv = tpot_actual_v2[i, j]
            print(f"{ttft:>6} {tpot:>6} | {ev:>8.0f} {eb:>8.0f} {sv:>5.1f}% | "
                  f"{sv2:>7.1f}% {sb:>7.1f}% | {tv:>7.0f}ms {pv:>7.0f}ms")


if __name__ == "__main__":
    main()
