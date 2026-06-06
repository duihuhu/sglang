#!/usr/bin/env python3
"""Aggregate 8-GPU v2 benchmark results and generate comparison charts."""
import json
import os
import sys
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_BASE = Path(__file__).parent / "results" / "8gpu_v2"
CHART_DIR = Path(__file__).parent.parent.parent / "versions" / "charts_v4"
CHART_DIR.mkdir(parents=True, exist_ok=True)

SCHEME_MAP = {
    "pd_tp4": ("PD TP4", "pd_p4d4"),
    "pd_dp4": ("PD DP4", "pd_dp4"),
    "pdaf_dyn": ("PDAF DynM", "pdaf_8g_dyn"),
    "pdaf_dyn_tier": ("PDAF Tier+DynM", "pdaf_8g_dyn_tier"),
}

COLORS = {
    "PD TP4": "#1f77b4",
    "PD DP4": "#ff7f0e",
    "PDAF DynM": "#2ca02c",
    "PDAF Tier+DynM": "#d62728",
}
MARKERS = {"PD TP4": "o", "PD DP4": "s", "PDAF DynM": "^", "PDAF Tier+DynM": "D"}

WORKLOADS = ["il128_ol1024", "il512_ol256", "il2048_ol64", "il4096_ol64"]
WL_LABELS = {
    "il128_ol1024": "IL128/OL1024\n(Decode-heavy)",
    "il512_ol256": "IL512/OL256\n(Balanced)",
    "il2048_ol64": "IL2048/OL64\n(Prefill-heavy)",
    "il4096_ol64": "IL4096/OL64\n(Long-prefill)",
}


def load_all_results():
    """Load all JSON results into a structured dict."""
    data = defaultdict(lambda: defaultdict(list))
    for scheme_dir, (label, prefix) in SCHEME_MAP.items():
        json_dir = RESULT_BASE / scheme_dir / "json"
        if not json_dir.exists():
            continue
        for f in sorted(json_dir.glob(f"{prefix}_*_results.json")):
            with open(f) as fp:
                r = json.load(fp)
            if r.get("throughput_tok_s", 0) == 0 and r.get("failed", 0) > 0:
                continue  # skip crashed runs
            wl = f"{r['il']}_{r['ol']}" if 'il' in r else None
            if wl is None:
                continue
            wl_key = f"il{r['il']}_ol{r['ol']}"
            data[label][wl_key].append(r)
    # sort by QPS
    for label in data:
        for wl in data[label]:
            data[label][wl].sort(key=lambda x: x.get("qps", 0))
    return data


def make_table(data):
    """Generate markdown summary table."""
    lines = []
    lines.append("## 8-GPU 方案对比汇总\n")
    lines.append("| 方案 | 负载 | QPS | 吞吐(tok/s) | TTFT_proc(ms) | TPOT(ms) | 总能耗(J) | 能效(mJ/tok) | SLO违背(%) |")
    lines.append("|------|------|-----|------------|--------------|---------|----------|------------|-----------|")
    for label in ["PD TP4", "PD DP4", "PDAF DynM", "PDAF Tier+DynM"]:
        if label not in data:
            continue
        for wl in WORKLOADS:
            if wl not in data[label]:
                continue
            for r in data[label][wl]:
                thpt = r.get("throughput_tok_s", 0)
                ttft = r.get("ttft_proc_avg_ms", 0) or r.get("ttft_avg_ms", 0)
                tpot = r.get("tpot_avg_ms", 0)
                energy = r.get("total_energy_j", 0)
                eff = r.get("energy_per_token_mj", 0)
                slo = r.get("slo_violation_rate", 0)
                qps = r.get("qps", 0)
                lines.append(f"| {label} | {wl} | {qps} | {thpt:.1f} | {ttft:.0f} | {tpot:.0f} | {energy:.0f} | {eff:.1f} | {slo:.1f} |")
    return "\n".join(lines)


def plot_metric_vs_qps(data, metric_key, ylabel, title_suffix, filename,
                       use_proc_ttft=False):
    """Plot a metric vs QPS for all schemes, one subplot per workload."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=False)
    fig.suptitle(f"8-GPU Deployment: {title_suffix}", fontsize=14, y=1.02)

    for idx, wl in enumerate(WORKLOADS):
        ax = axes[idx]
        for label in ["PD TP4", "PD DP4", "PDAF DynM", "PDAF Tier+DynM"]:
            if label not in data or wl not in data[label]:
                continue
            results = data[label][wl]
            qps_vals = [r["qps"] for r in results]
            if use_proc_ttft:
                metric_vals = [r.get("ttft_proc_avg_ms", 0) or r.get("ttft_avg_ms", 0) for r in results]
            else:
                metric_vals = [r.get(metric_key, 0) for r in results]
            # slo_violation_rate is already in percentage (e.g. 58.33)
            if metric_key == "slo_violation_rate":
                metric_vals = [v for v in metric_vals]
            ax.plot(qps_vals, metric_vals, marker=MARKERS[label],
                    color=COLORS[label], label=label, linewidth=2, markersize=6)
        ax.set_xlabel("QPS")
        ax.set_title(WL_LABELS[wl], fontsize=10)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.set_ylabel(ylabel)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    plt.tight_layout()
    plt.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {CHART_DIR / filename}")


def plot_energy_breakdown(data):
    """Bar chart: energy at SLO-safe max QPS for each workload."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("8-GPU: Energy at Max SLO-safe QPS (per workload)", fontsize=14, y=1.02)

    labels_order = ["PD TP4", "PD DP4", "PDAF DynM", "PDAF Tier+DynM"]

    for idx, wl in enumerate(WORKLOADS):
        ax = axes[idx]
        bar_labels = []
        prefill_e = []
        decode_e = []

        for label in labels_order:
            if label not in data or wl not in data[label]:
                bar_labels.append(label)
                prefill_e.append(0)
                decode_e.append(0)
                continue
            results = data[label][wl]
            # Find max QPS where SLO < 50%
            safe = [r for r in results if r.get("slo_violation_rate", 0) < 50
                    and r.get("throughput_tok_s", 0) > 0]
            if not safe:
                bar_labels.append(label)
                prefill_e.append(0)
                decode_e.append(0)
                continue
            best = safe[-1]  # highest QPS that's SLO-safe
            bar_labels.append(f"{label}\nQPS={best['qps']}")
            prefill_e.append(best.get("prefill_energy_j", 0) / 1000)
            decode_e.append(best.get("decode_energy_j", 0) / 1000)

        x = np.arange(len(bar_labels))
        w = 0.6
        ax.bar(x, prefill_e, w, label="Prefill Energy", color="#4c72b0", alpha=0.8)
        ax.bar(x, decode_e, w, bottom=prefill_e, label="Decode Energy", color="#dd8452", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8)
        ax.set_title(WL_LABELS[wl], fontsize=10)
        ax.set_ylabel("Energy (kJ)" if idx == 0 else "")
        ax.grid(True, alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.0), fontsize=10)
    plt.tight_layout()
    plt.savefig(CHART_DIR / "energy_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {CHART_DIR / 'energy_breakdown.png'}")


def plot_energy_saving_summary(data):
    """Summary: energy saving percentage of Tier vs non-Tier at same QPS."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.set_title("Energy Saving: PDAF Tier+DynM vs PDAF DynM\n(at same QPS, SLO-safe points only)", fontsize=12)

    savings = []
    labels = []
    for wl in WORKLOADS:
        if "PDAF DynM" not in data or wl not in data["PDAF DynM"]:
            continue
        if "PDAF Tier+DynM" not in data or wl not in data["PDAF Tier+DynM"]:
            continue
        dyn_results = {r["qps"]: r for r in data["PDAF DynM"][wl]}
        tier_results = {r["qps"]: r for r in data["PDAF Tier+DynM"][wl]}
        common_qps = sorted(set(dyn_results.keys()) & set(tier_results.keys()))
        for qps in common_qps:
            dr = dyn_results[qps]
            tr = tier_results[qps]
            if dr["throughput_tok_s"] == 0 or tr["throughput_tok_s"] == 0:
                continue
            if dr["slo_violation_rate"] >= 50 or tr["slo_violation_rate"] >= 50:
                continue
            e_dyn = dr["total_energy_j"]
            e_tier = tr["total_energy_j"]
            if e_dyn > 0:
                saving = (e_dyn - e_tier) / e_dyn * 100
                savings.append(saving)
                labels.append(f"{wl}\nQPS={qps}")

    if savings:
        x = np.arange(len(savings))
        colors = ["#2ca02c" if s > 0 else "#d62728" for s in savings]
        ax.bar(x, savings, color=colors, alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Energy Saving (%)")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(savings):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "energy_saving_tier_vs_dyn.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {CHART_DIR / 'energy_saving_tier_vs_dyn.png'}")


def main():
    print("Loading results...")
    data = load_all_results()
    for label in data:
        total = sum(len(v) for v in data[label].values())
        print(f"  {label}: {total} data points")

    print("\nGenerating charts...")
    plot_metric_vs_qps(data, "throughput_tok_s", "Throughput (tok/s)",
                       "Throughput vs QPS", "throughput_vs_qps.png")
    plot_metric_vs_qps(data, "ttft_proc_avg_ms", "TTFT (ms)",
                       "TTFT (Server Processing, excl. queue) vs QPS",
                       "ttft_vs_qps.png", use_proc_ttft=True)
    plot_metric_vs_qps(data, "tpot_avg_ms", "TPOT (ms)",
                       "TPOT vs QPS", "tpot_vs_qps.png")
    plot_metric_vs_qps(data, "slo_violation_rate", "SLO Violation (%)",
                       "SLO Violation Rate vs QPS", "slo_vs_qps.png")
    plot_metric_vs_qps(data, "total_energy_j", "Total Energy (J)",
                       "Total Energy vs QPS", "energy_vs_qps.png")
    plot_metric_vs_qps(data, "energy_per_token_mj", "Energy/Token (mJ)",
                       "Energy Efficiency vs QPS", "efficiency_vs_qps.png")
    plot_energy_breakdown(data)
    plot_energy_saving_summary(data)

    print("\nGenerating markdown table...")
    table = make_table(data)
    return table, data


if __name__ == "__main__":
    table, data = main()
    print("\n" + table)
