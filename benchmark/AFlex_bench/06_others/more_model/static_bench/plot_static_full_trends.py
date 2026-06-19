#!/usr/bin/env python3
"""Plot trend lines from full Cartesian-product static benchmark results.

Generates multi-panel line charts showing how metrics vary with QPS, IL, OL.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results_full"
CHARTS_DIR = HERE / "charts_full"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = RESULTS_DIR / "static_full_results.json"
if not RESULTS_FILE.exists():
    raise FileNotFoundError(f"Results not found: {RESULTS_FILE}")

with open(RESULTS_FILE) as f:
    data = json.load(f)

print(f"Loaded deploys: {list(data.keys())}")

DEPLOYS = ["native_dp8", "native_dp8_tier", "pd_dp4", "pd_dp4_tier",
           "pdaf_tp2", "pdaf_tp2_tier"]
DEPLOY_LABELS = ["Native DP8", "Native DP8+Tier", "PD DP4",
                 "PD DP4+Tier", "PDAF TP2", "PDAF TP2+Tier"]
COLORS = ["#2196F3", "#1565C0", "#4CAF50", "#2E7D32", "#FF9800", "#E65100"]
MARKERS = ["o", "s", "^", "v", "D", "P"]
LINESTYLES = ["-", "--", "-", "--", "-", "--"]

QPS_LIST = [1, 2, 3, 4, 5, 6]
IL_LIST = [128, 256, 512, 1024]
OL_LIST = [128, 256, 512, 1024]

plt.rcParams.update({"font.size": 9, "figure.dpi": 150})


def get_m(deploy, wl_name, key):
    entry = data.get(deploy, {}).get(wl_name, {})
    if entry.get("status") != "PASS":
        return None
    return entry.get(key)


def wl_name(il, ol, qps):
    return f"il{il}_ol{ol}_qps{qps}"


# ============================================================
# Chart 1: Throughput vs QPS (one subplot per IL×OL combination)
# ============================================================
def plot_metric_vs_qps(metric_key, ylabel, title_prefix, filename):
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True)
    fig.suptitle(f"{title_prefix} vs QPS (Qwen3-30B-A3B, 8×A800)",
                 fontsize=14, fontweight="bold")

    for row_idx, il in enumerate(IL_LIST):
        for col_idx, ol in enumerate(OL_LIST):
            ax = axes[row_idx, col_idx]
            for dep_idx, (dep, label, color, marker, ls) in enumerate(
                    zip(DEPLOYS, DEPLOY_LABELS, COLORS, MARKERS, LINESTYLES)):
                vals = []
                valid_qps = []
                for qps in QPS_LIST:
                    v = get_m(dep, wl_name(il, ol, qps), metric_key)
                    if v is not None:
                        vals.append(v)
                        valid_qps.append(qps)
                if vals:
                    ax.plot(valid_qps, vals, color=color, marker=marker,
                            linestyle=ls, linewidth=1.5, markersize=4,
                            label=label, alpha=0.85)

            ax.set_title(f"IL={il}, OL={ol}", fontsize=9)
            ax.grid(True, alpha=0.3)
            if row_idx == 3:
                ax.set_xlabel("QPS")
            if col_idx == 0:
                ax.set_ylabel(ylabel)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, 0.97))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = CHARTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ============================================================
# Chart 2: Metric vs IL (one subplot per QPS×OL)
# ============================================================
def plot_metric_vs_il(metric_key, ylabel, title_prefix, filename):
    fig, axes = plt.subplots(4, 6, figsize=(24, 16), sharex=True)
    fig.suptitle(f"{title_prefix} vs Input Length (Qwen3-30B-A3B, 8×A800)",
                 fontsize=14, fontweight="bold")

    for row_idx, ol in enumerate(OL_LIST):
        for col_idx, qps in enumerate(QPS_LIST):
            ax = axes[row_idx, col_idx]
            for dep_idx, (dep, label, color, marker, ls) in enumerate(
                    zip(DEPLOYS, DEPLOY_LABELS, COLORS, MARKERS, LINESTYLES)):
                vals = []
                valid_il = []
                for il in IL_LIST:
                    v = get_m(dep, wl_name(il, ol, qps), metric_key)
                    if v is not None:
                        vals.append(v)
                        valid_il.append(il)
                if vals:
                    ax.plot(valid_il, vals, color=color, marker=marker,
                            linestyle=ls, linewidth=1.5, markersize=4,
                            label=label, alpha=0.85)

            ax.set_title(f"OL={ol}, QPS={qps}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if row_idx == 3:
                ax.set_xlabel("Input Length")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
            ax.set_xticks(IL_LIST)
            ax.set_xticklabels([str(x) for x in IL_LIST], fontsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, 0.97))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = CHARTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ============================================================
# Chart 3: Metric vs OL (one subplot per QPS×IL)
# ============================================================
def plot_metric_vs_ol(metric_key, ylabel, title_prefix, filename):
    fig, axes = plt.subplots(4, 6, figsize=(24, 16), sharex=True)
    fig.suptitle(f"{title_prefix} vs Output Length (Qwen3-30B-A3B, 8×A800)",
                 fontsize=14, fontweight="bold")

    for row_idx, il in enumerate(IL_LIST):
        for col_idx, qps in enumerate(QPS_LIST):
            ax = axes[row_idx, col_idx]
            for dep_idx, (dep, label, color, marker, ls) in enumerate(
                    zip(DEPLOYS, DEPLOY_LABELS, COLORS, MARKERS, LINESTYLES)):
                vals = []
                valid_ol = []
                for ol in OL_LIST:
                    v = get_m(dep, wl_name(il, ol, qps), metric_key)
                    if v is not None:
                        vals.append(v)
                        valid_ol.append(ol)
                if vals:
                    ax.plot(valid_ol, vals, color=color, marker=marker,
                            linestyle=ls, linewidth=1.5, markersize=4,
                            label=label, alpha=0.85)

            ax.set_title(f"IL={il}, QPS={qps}", fontsize=8)
            ax.grid(True, alpha=0.3)
            if row_idx == 3:
                ax.set_xlabel("Output Length")
            if col_idx == 0:
                ax.set_ylabel(ylabel)
            ax.set_xscale("log", base=2)
            ax.set_xticks(OL_LIST)
            ax.set_xticklabels([str(x) for x in OL_LIST], fontsize=7)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6,
               fontsize=9, bbox_to_anchor=(0.5, 0.97))
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    out = CHARTS_DIR / filename
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


# ============================================================
# Chart 4: Energy savings heatmap (Tier vs non-Tier)
# ============================================================
def plot_energy_savings_heatmap():
    pairs = [("native_dp8", "native_dp8_tier", "Native DP8"),
             ("pd_dp4", "pd_dp4_tier", "PD DP4"),
             ("pdaf_tp2", "pdaf_tp2_tier", "PDAF TP2")]

    for qps in QPS_LIST:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Tier Energy Savings (%) at QPS={qps}",
                     fontsize=13, fontweight="bold")

        for ax_idx, (base, tier, name) in enumerate(pairs):
            ax = axes[ax_idx]
            matrix = np.zeros((len(IL_LIST), len(OL_LIST)))
            for i, il in enumerate(IL_LIST):
                for j, ol in enumerate(OL_LIST):
                    base_e = get_m(base, wl_name(il, ol, qps), "total_energy_j")
                    tier_e = get_m(tier, wl_name(il, ol, qps), "total_energy_j")
                    if base_e and tier_e and base_e > 0:
                        matrix[i, j] = (base_e - tier_e) / base_e * 100
                    else:
                        matrix[i, j] = np.nan

            im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto",
                           vmin=-20, vmax=40)
            ax.set_xticks(range(len(OL_LIST)))
            ax.set_xticklabels([str(x) for x in OL_LIST])
            ax.set_yticks(range(len(IL_LIST)))
            ax.set_yticklabels([str(x) for x in IL_LIST])
            ax.set_xlabel("Output Length")
            ax.set_ylabel("Input Length")
            ax.set_title(f"{name}: Tier Savings (%)")

            for i in range(len(IL_LIST)):
                for j in range(len(OL_LIST)):
                    val = matrix[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.1f}%", ha="center",
                                va="center", fontsize=8,
                                color="black" if abs(val) < 20 else "white")

            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        plt.tight_layout()
        out = CHARTS_DIR / f"tier_savings_heatmap_qps{qps}.png"
        plt.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close()


# ============================================================
# Chart 5: Compact 4-panel summary (aggregated across QPS)
# ============================================================
def plot_summary_4panel():
    metrics = [
        ("throughput_tok_s", "Throughput (tok/s)"),
        ("tpot_avg_ms", "TPOT (ms)"),
        ("total_energy_j", "Total Energy (J)"),
        ("energy_per_token_mj", "Energy/Token (mJ/tok)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Full Sweep Summary: Metric vs QPS (averaged over IL×OL)",
                 fontsize=14, fontweight="bold")

    for ax_idx, (key, ylabel) in enumerate(metrics):
        ax = axes[ax_idx // 2, ax_idx % 2]
        for dep_idx, (dep, label, color, marker, ls) in enumerate(
                zip(DEPLOYS, DEPLOY_LABELS, COLORS, MARKERS, LINESTYLES)):
            avg_vals = []
            for qps in QPS_LIST:
                vals = []
                for il in IL_LIST:
                    for ol in OL_LIST:
                        v = get_m(dep, wl_name(il, ol, qps), key)
                        if v is not None:
                            vals.append(v)
                if vals:
                    avg_vals.append(np.mean(vals))
                else:
                    avg_vals.append(np.nan)
            ax.plot(QPS_LIST, avg_vals, color=color, marker=marker,
                    linestyle=ls, linewidth=2, markersize=6,
                    label=label, alpha=0.85)

        ax.set_xlabel("QPS")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = CHARTS_DIR / "summary_4panel_vs_qps.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


if __name__ == "__main__":
    print("Generating trend charts from full sweep data...")
    print()

    plot_metric_vs_qps("throughput_tok_s", "Throughput (tok/s)",
                       "Throughput", "trend_throughput_vs_qps.png")
    plot_metric_vs_qps("tpot_avg_ms", "TPOT (ms)",
                       "TPOT", "trend_tpot_vs_qps.png")
    plot_metric_vs_qps("energy_per_token_mj", "Energy/Token (mJ/tok)",
                       "Energy Efficiency", "trend_energy_eff_vs_qps.png")
    plot_metric_vs_qps("total_energy_j", "Total Energy (J)",
                       "Total Energy", "trend_total_energy_vs_qps.png")

    plot_metric_vs_il("throughput_tok_s", "Throughput (tok/s)",
                      "Throughput", "trend_throughput_vs_il.png")
    plot_metric_vs_il("energy_per_token_mj", "Energy/Token (mJ/tok)",
                      "Energy Efficiency", "trend_energy_eff_vs_il.png")

    plot_metric_vs_ol("throughput_tok_s", "Throughput (tok/s)",
                      "Throughput", "trend_throughput_vs_ol.png")
    plot_metric_vs_ol("energy_per_token_mj", "Energy/Token (mJ/tok)",
                      "Energy Efficiency", "trend_energy_eff_vs_ol.png")

    plot_energy_savings_heatmap()
    plot_summary_4panel()

    print("\nAll charts saved to:", CHARTS_DIR)
