#!/usr/bin/env python3
"""Plot PDAF affinity benchmark results: affine vs non-affine deployment comparison.

Generates:
  1. TTFT comparison (avg + p99) vs QPS
  2. Throughput comparison vs QPS
  3. TPOT comparison (avg + p99) vs QPS
  4. Energy efficiency comparison vs QPS
  5. SLO violation rate vs QPS

Usage:
    python3 plot_affinity_comparison.py <results.json>
    python3 plot_affinity_comparison.py  # uses latest file in results/
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"


def find_latest_results():
    results_dir = RESULTS_DIR
    files = sorted(results_dir.glob("affinity_bench_*.json"))
    if not files:
        print(f"No result files found in {results_dir}")
        sys.exit(1)
    return files[-1]


def load_results(path):
    with open(path) as f:
        data = json.load(f)
    return data


def extract_series(deploy_results):
    """Extract ordered (qps, metrics) from one deployment's results."""
    qps_list = []
    metrics = []
    for key, val in deploy_results.items():
        if key.startswith("__"):
            continue
        if val.get("status") != "PASS":
            continue
        qps = int(key.replace("qps_", ""))
        qps_list.append(qps)
        metrics.append(val)
    order = np.argsort(qps_list)
    return [qps_list[i] for i in order], [metrics[i] for i in order]


def plot_ttft(ax, data):
    """TTFT avg + p99 vs QPS for both deployments."""
    colors = {"pdaf_affine": "#2196F3", "pdaf_nonaffine": "#FF5722"}
    labels = {"pdaf_affine": "Affine (interleaved)", "pdaf_nonaffine": "Non-affine (continuous)"}

    for dep_name, dep_res in data["results"].items():
        if "__status__" in dep_res:
            continue
        qps_list, metrics = extract_series(dep_res)
        if not qps_list:
            continue
        c = colors.get(dep_name, "#333")
        lbl = labels.get(dep_name, dep_name)
        avg = [m["ttft_avg_ms"] for m in metrics]
        p99 = [m["ttft_p99_ms"] for m in metrics]
        ax.plot(qps_list, avg, "o-", color=c, label=f"{lbl} avg", linewidth=2)
        ax.plot(qps_list, p99, "s--", color=c, alpha=0.6, label=f"{lbl} p99", linewidth=1.5)

    ax.axhline(y=5000, color="red", linestyle=":", alpha=0.5, label="SLO (5s)")
    ax.set_xlabel("QPS")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("TTFT vs QPS")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_throughput(ax, data):
    """Throughput vs QPS."""
    colors = {"pdaf_affine": "#2196F3", "pdaf_nonaffine": "#FF5722"}
    labels = {"pdaf_affine": "Affine (interleaved)", "pdaf_nonaffine": "Non-affine (continuous)"}

    for dep_name, dep_res in data["results"].items():
        if "__status__" in dep_res:
            continue
        qps_list, metrics = extract_series(dep_res)
        if not qps_list:
            continue
        c = colors.get(dep_name, "#333")
        lbl = labels.get(dep_name, dep_name)
        thpt = [m["throughput_tok_s"] for m in metrics]
        ax.plot(qps_list, thpt, "o-", color=c, label=lbl, linewidth=2, markersize=8)

    ax.set_xlabel("QPS")
    ax.set_ylabel("Throughput (tok/s)")
    ax.set_title("Throughput vs QPS")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_tpot(ax, data):
    """TPOT avg + p99 vs QPS."""
    colors = {"pdaf_affine": "#2196F3", "pdaf_nonaffine": "#FF5722"}
    labels = {"pdaf_affine": "Affine (interleaved)", "pdaf_nonaffine": "Non-affine (continuous)"}

    for dep_name, dep_res in data["results"].items():
        if "__status__" in dep_res:
            continue
        qps_list, metrics = extract_series(dep_res)
        if not qps_list:
            continue
        c = colors.get(dep_name, "#333")
        lbl = labels.get(dep_name, dep_name)
        avg = [m["tpot_avg_ms"] for m in metrics]
        p99 = [m["tpot_p99_ms"] for m in metrics]
        ax.plot(qps_list, avg, "o-", color=c, label=f"{lbl} avg", linewidth=2)
        ax.plot(qps_list, p99, "s--", color=c, alpha=0.6, label=f"{lbl} p99", linewidth=1.5)

    ax.axhline(y=300, color="red", linestyle=":", alpha=0.5, label="SLO (300ms)")
    ax.set_xlabel("QPS")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("TPOT vs QPS")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def plot_energy(ax, data):
    """Energy per token vs QPS."""
    colors = {"pdaf_affine": "#2196F3", "pdaf_nonaffine": "#FF5722"}
    labels = {"pdaf_affine": "Affine (interleaved)", "pdaf_nonaffine": "Non-affine (continuous)"}

    for dep_name, dep_res in data["results"].items():
        if "__status__" in dep_res:
            continue
        qps_list, metrics = extract_series(dep_res)
        if not qps_list:
            continue
        c = colors.get(dep_name, "#333")
        lbl = labels.get(dep_name, dep_name)
        ept = [m["energy_per_token_mj"] for m in metrics]
        ax.plot(qps_list, ept, "o-", color=c, label=lbl, linewidth=2, markersize=8)

    ax.set_xlabel("QPS")
    ax.set_ylabel("Energy per token (mJ)")
    ax.set_title("Energy Efficiency vs QPS")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)


def plot_slo(ax, data):
    """SLO violation rate vs QPS."""
    colors = {"pdaf_affine": "#2196F3", "pdaf_nonaffine": "#FF5722"}
    labels = {"pdaf_affine": "Affine (interleaved)", "pdaf_nonaffine": "Non-affine (continuous)"}

    for dep_name, dep_res in data["results"].items():
        if "__status__" in dep_res:
            continue
        qps_list, metrics = extract_series(dep_res)
        if not qps_list:
            continue
        c = colors.get(dep_name, "#333")
        lbl = labels.get(dep_name, dep_name)
        slo = [m["slo_violation_rate"] for m in metrics]
        ax.bar([q + (0.2 if "affine" in dep_name and "non" not in dep_name else -0.2)
                for q in qps_list],
               slo, width=0.35, color=c, alpha=0.8, label=lbl)

    ax.set_xlabel("QPS")
    ax.set_ylabel("SLO Violation Rate (%)")
    ax.set_title("SLO Violation Rate vs QPS")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")


def plot_improvement_bar(fig, data):
    """Add a bar chart showing % improvement of affine over non-affine for key metrics."""
    affine = data["results"].get("pdaf_affine", {})
    nonaffine = data["results"].get("pdaf_nonaffine", {})
    if "__status__" in affine or "__status__" in nonaffine:
        return

    qps_a, metrics_a = extract_series(affine)
    qps_n, metrics_n = extract_series(nonaffine)

    common_qps = sorted(set(qps_a) & set(qps_n))
    if not common_qps:
        return

    improvements = {"TTFT": [], "Throughput": [], "TPOT": [], "Energy/tok": []}
    for q in common_qps:
        ia = qps_a.index(q)
        ina = qps_n.index(q)
        ma, mn = metrics_a[ia], metrics_n[ina]
        if mn["ttft_avg_ms"] > 0:
            improvements["TTFT"].append(
                (mn["ttft_avg_ms"] - ma["ttft_avg_ms"]) / mn["ttft_avg_ms"] * 100)
        if mn["throughput_tok_s"] > 0:
            improvements["Throughput"].append(
                (ma["throughput_tok_s"] - mn["throughput_tok_s"]) / mn["throughput_tok_s"] * 100)
        if mn["tpot_avg_ms"] > 0:
            improvements["TPOT"].append(
                (mn["tpot_avg_ms"] - ma["tpot_avg_ms"]) / mn["tpot_avg_ms"] * 100)
        if mn["energy_per_token_mj"] > 0:
            improvements["Energy/tok"].append(
                (mn["energy_per_token_mj"] - ma["energy_per_token_mj"]) / mn["energy_per_token_mj"] * 100)

    avg_imp = {k: np.mean(v) if v else 0 for k, v in improvements.items()}

    ax = fig.add_subplot(2, 3, 6)
    keys = list(avg_imp.keys())
    vals = [avg_imp[k] for k in keys]
    bars = ax.bar(keys, vals, color=["#2196F3" if v >= 0 else "#FF5722" for v in vals],
                  alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.5)
    ax.set_ylabel("Improvement (%)")
    ax.set_title("Affine vs Non-affine\n(avg % improvement)")
    ax.grid(True, alpha=0.3, axis="y")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{val:.1f}%", ha="center", va="bottom" if val >= 0 else "top",
                fontsize=9, fontweight="bold")


def main():
    if len(sys.argv) > 1:
        result_file = Path(sys.argv[1])
    else:
        result_file = find_latest_results()

    print(f"Loading results from: {result_file}")
    data = load_results(result_file)

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle("PDAF NIC-Affinity Benchmark: Affine (interleaved) vs Non-affine (continuous)\n"
                 f"Model: Qwen3-32B | TP=4 | Workload: summary (in=4096, out=64)",
                 fontsize=13, fontweight="bold")

    ax1 = fig.add_subplot(2, 3, 1)
    plot_ttft(ax1, data)

    ax2 = fig.add_subplot(2, 3, 2)
    plot_throughput(ax2, data)

    ax3 = fig.add_subplot(2, 3, 3)
    plot_tpot(ax3, data)

    ax4 = fig.add_subplot(2, 3, 4)
    plot_energy(ax4, data)

    ax5 = fig.add_subplot(2, 3, 5)
    plot_slo(ax5, data)

    plot_improvement_bar(fig, data)

    plt.tight_layout()
    out_png = CHARTS_DIR / "pdaf_affinity_comparison.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out_png}")
    plt.close()

    # Also generate individual charts for clarity
    for name, plot_fn in [("ttft", plot_ttft), ("throughput", plot_throughput),
                          ("tpot", plot_tpot), ("energy", plot_energy),
                          ("slo", plot_slo)]:
        fig2, ax = plt.subplots(figsize=(8, 5))
        plot_fn(ax, data)
        fig2.tight_layout()
        fig2.savefig(CHARTS_DIR / f"pdaf_affinity_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig2)
    print(f"Individual charts saved to: {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
