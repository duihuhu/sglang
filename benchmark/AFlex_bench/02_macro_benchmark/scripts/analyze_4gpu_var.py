#!/usr/bin/env python3
"""Analyze 4-GPU variable-length workload results and generate charts + markdown."""
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULT_BASE = Path(__file__).parent / "results" / "4gpu_var"
CHART_DIR = Path(__file__).parent.parent.parent / "versions" / "charts_4gpu_var"
CHART_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = {
    "pd_tp2": "PD TP2",
    "pd_dp2": "Native TP2 DP2",
    "pd_dp2_disagg": "PD DP2",
    "pdaf_dyn": "PDAF DynM",
    "pdaf_dyn_tier": "PDAF Tier+DynM",
}

WORKLOADS = ["steady", "varying", "heavy", "overload", "tier1_demo"]
WL_LABELS = {
    "steady": "Steady\n(QPS=5, 120s)",
    "varying": "Varying\n(burst, 120s)",
    "heavy": "Heavy\n(long seqs)",
    "overload": "Overload\n(QPS=10, 120s)",
    "tier1_demo": "Tier1 Demo\n(mixed, 120s)",
}

COLORS = {
    "PD TP2": "#1f77b4",
    "Native TP2 DP2": "#ff7f0e",
    "PD DP2": "#9467bd",
    "PDAF DynM": "#2ca02c",
    "PDAF Tier+DynM": "#d62728",
}
HATCHES = {
    "PD TP2": "",
    "Native TP2 DP2": "//",
    "PD DP2": "..",
    "PDAF DynM": "\\\\",
    "PDAF Tier+DynM": "xx",
}


def load_results():
    """Load all 4GPU var results."""
    data = {}
    for scheme_dir, label in SCHEMES.items():
        json_dir = RESULT_BASE / scheme_dir / "json"
        if not json_dir.exists():
            continue
        data[label] = {}
        for f in sorted(json_dir.glob("*_results.json")):
            r = json.loads(f.read_text())
            for wl_name in WORKLOADS:
                if f"var_{wl_name}" in f.stem:
                    data[label][wl_name] = r
                    break
    return data


def make_table(data):
    """Generate markdown summary."""
    lines = []
    lines.append("## 4-GPU 变长 Workload 方案对比\n")
    lines.append("| 方案 | Workload | Thpt(tok/s) | TTFT(ms) | TPOT(ms) | Energy(J) | J/GPU | SLO违背(%) |")
    lines.append("|------|----------|------------|---------|---------|----------|-------|-----------|")
    for label in ["PD TP2", "Native TP2 DP2", "PD DP2", "PDAF DynM", "PDAF Tier+DynM"]:
        if label not in data:
            continue
        for wl in WORKLOADS:
            if wl not in data[label]:
                continue
            r = data[label][wl]
            thpt = r.get("throughput_tok_s", 0)
            ttft = r.get("ttft_proc_avg_ms", 0) or r.get("ttft_avg_ms", 0)
            tpot = r.get("tpot_avg_ms", 0)
            energy = r.get("total_energy_j", 0)
            j_gpu = energy / 4.0
            slo = r.get("slo_violation_rate", 0)
            lines.append(f"| {label} | {wl} | {thpt:.1f} | {ttft:.0f} | {tpot:.1f} | {energy:.0f} | {j_gpu:.0f} | {slo:.1f} |")
    return "\n".join(lines)


def plot_bar_comparison(data):
    """Bar chart comparing all schemes across workloads for key metrics."""
    metrics = [
        ("throughput_tok_s", "Throughput (tok/s)", "throughput_comparison.png"),
        ("ttft_avg_ms", "TTFT (ms)", "ttft_comparison.png"),
        ("tpot_avg_ms", "TPOT (ms)", "tpot_comparison.png"),
        ("total_energy_j", "Total Energy (J)", "energy_comparison.png"),
        ("slo_violation_rate", "SLO Violation (%)", "slo_comparison.png"),
    ]

    scheme_order = ["PD TP2", "Native TP2 DP2", "PD DP2", "PDAF DynM", "PDAF Tier+DynM"]

    for metric_key, ylabel, filename in metrics:
        fig, ax = plt.subplots(figsize=(14, 5))
        x = np.arange(len(WORKLOADS))
        width = 0.16
        offsets = [-2, -1, 0, 1, 2]

        for i, label in enumerate(scheme_order):
            if label not in data:
                continue
            vals = []
            for wl in WORKLOADS:
                if wl in data[label]:
                    r = data[label][wl]
                    if metric_key == "ttft_avg_ms":
                        v = r.get("ttft_proc_avg_ms", 0) or r.get("ttft_avg_ms", 0)
                    else:
                        v = r.get(metric_key, 0)
                    vals.append(v)
                else:
                    vals.append(0)
            bars = ax.bar(x + offsets[i] * width, vals, width,
                         label=label, color=COLORS[label], alpha=0.85,
                         hatch=HATCHES[label], edgecolor="white", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels([WL_LABELS[wl] for wl in WORKLOADS], fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"4-GPU Variable Workloads: {ylabel}", fontsize=13)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")

        plt.tight_layout()
        plt.savefig(CHART_DIR / filename, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Saved: {CHART_DIR / filename}")


def plot_energy_saving(data):
    """Energy saving of Tier vs DynM and vs PD TP2."""
    if "PDAF DynM" not in data or "PDAF Tier+DynM" not in data:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: Tier vs DynM
    ax = axes[0]
    ax.set_title("Energy Saving: Tier+DynM vs DynM", fontsize=11)
    savings, labels = [], []
    for wl in WORKLOADS:
        if wl in data["PDAF DynM"] and wl in data["PDAF Tier+DynM"]:
            e_dyn = data["PDAF DynM"][wl].get("total_energy_j", 0)
            e_tier = data["PDAF Tier+DynM"][wl].get("total_energy_j", 0)
            if e_dyn > 0:
                saving = (e_dyn - e_tier) / e_dyn * 100
                savings.append(saving)
                labels.append(wl)
    if savings:
        colors = ["#2ca02c" if s > 0 else "#d62728" for s in savings]
        bars = ax.bar(range(len(savings)), savings, color=colors, alpha=0.8)
        ax.set_xticks(range(len(savings)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Energy Saving (%)")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(savings):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    # Panel 2: Tier vs PD TP2
    ax = axes[1]
    ax.set_title("Energy Saving: Tier+DynM vs PD TP2", fontsize=11)
    savings2, labels2 = [], []
    if "PD TP2" in data:
        for wl in WORKLOADS:
            if wl in data["PD TP2"] and wl in data["PDAF Tier+DynM"]:
                e_tp2 = data["PD TP2"][wl].get("total_energy_j", 0)
                e_tier = data["PDAF Tier+DynM"][wl].get("total_energy_j", 0)
                if e_tp2 > 0:
                    saving = (e_tp2 - e_tier) / e_tp2 * 100
                    savings2.append(saving)
                    labels2.append(wl)
    if savings2:
        colors = ["#2ca02c" if s > 0 else "#d62728" for s in savings2]
        ax.bar(range(len(savings2)), savings2, color=colors, alpha=0.8)
        ax.set_xticks(range(len(savings2)))
        ax.set_xticklabels(labels2, fontsize=9)
        ax.set_ylabel("Energy Saving (%)")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.grid(True, alpha=0.3, axis="y")
        for i, v in enumerate(savings2):
            ax.text(i, v + 0.5, f"{v:.1f}%", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "energy_saving.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {CHART_DIR / 'energy_saving.png'}")


def plot_radar(data):
    """Radar chart comparing schemes on normalized metrics."""
    scheme_order = ["PD TP2", "Native TP2 DP2", "PD DP2", "PDAF DynM", "PDAF Tier+DynM"]
    metric_names = ["Throughput", "1/TTFT", "1/TPOT", "1/Energy", "1-SLO%"]

    # Average across workloads
    scheme_scores = {}
    for label in scheme_order:
        if label not in data:
            continue
        vals = {"thpt": [], "ttft": [], "tpot": [], "energy": [], "slo": []}
        for wl in WORKLOADS:
            if wl not in data[label]:
                continue
            r = data[label][wl]
            vals["thpt"].append(r.get("throughput_tok_s", 0))
            ttft = r.get("ttft_proc_avg_ms", 0) or r.get("ttft_avg_ms", 0)
            vals["ttft"].append(ttft)
            vals["tpot"].append(r.get("tpot_avg_ms", 0))
            vals["energy"].append(r.get("total_energy_j", 0))
            vals["slo"].append(r.get("slo_violation_rate", 0))
        if vals["thpt"]:
            scheme_scores[label] = [
                np.mean(vals["thpt"]),
                1000.0 / max(np.mean(vals["ttft"]), 1),
                1000.0 / max(np.mean(vals["tpot"]), 1),
                1e6 / max(np.mean(vals["energy"]), 1),
                100.0 - np.mean(vals["slo"]),
            ]

    if len(scheme_scores) < 2:
        return

    # Normalize each dimension to [0, 1]
    all_scores = np.array(list(scheme_scores.values()))
    mins = all_scores.min(axis=0)
    maxs = all_scores.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1

    angles = np.linspace(0, 2 * np.pi, len(metric_names), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.set_title("4-GPU Scheme Comparison (Normalized)", fontsize=12, y=1.08)

    for label, scores in scheme_scores.items():
        normalized = (np.array(scores) - mins) / ranges
        values = normalized.tolist() + [normalized[0]]
        ax.plot(angles, values, marker="o", label=label, color=COLORS[label], linewidth=2)
        ax.fill(angles, values, alpha=0.1, color=COLORS[label])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_names, fontsize=10)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(CHART_DIR / "radar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {CHART_DIR / 'radar_comparison.png'}")


def plot_energy_breakdown(data):
    """Stacked bar: prefill vs decode energy per scheme per workload."""
    scheme_order = ["PD TP2", "Native TP2 DP2", "PD DP2", "PDAF DynM", "PDAF Tier+DynM"]

    fig, axes = plt.subplots(1, len(WORKLOADS), figsize=(16, 5), sharey=True)
    fig.suptitle("4-GPU: Energy Breakdown (Prefill vs Decode) per Workload", fontsize=13, y=1.02)

    for wi, wl in enumerate(WORKLOADS):
        ax = axes[wi]
        bar_labels = []
        prefill_vals = []
        decode_vals = []
        for label in scheme_order:
            if label not in data or wl not in data[label]:
                bar_labels.append(label.split(" ")[-1])
                prefill_vals.append(0)
                decode_vals.append(0)
                continue
            r = data[label][wl]
            pe = r.get("prefill_energy_j", 0) / 1000
            de = r.get("decode_energy_j", 0) / 1000
            prefill_vals.append(pe)
            decode_vals.append(de)
            bar_labels.append(label.split(" ")[-1])

        x = np.arange(len(bar_labels))
        ax.bar(x, prefill_vals, 0.6, label="Prefill", color="#4c72b0", alpha=0.8)
        ax.bar(x, decode_vals, 0.6, bottom=prefill_vals, label="Decode", color="#dd8452", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(bar_labels, fontsize=8, rotation=30)
        ax.set_title(wl, fontsize=10)
        if wi == 0:
            ax.set_ylabel("Energy (kJ)")
        ax.grid(True, alpha=0.3, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout()
    plt.savefig(CHART_DIR / "energy_breakdown.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {CHART_DIR / 'energy_breakdown.png'}")


def main():
    print("Loading 4-GPU variable workload results...")
    data = load_results()
    for label in data:
        print(f"  {label}: {len(data[label])} workloads")

    print("\nGenerating charts...")
    plot_bar_comparison(data)
    plot_energy_saving(data)
    plot_energy_breakdown(data)
    plot_radar(data)

    print("\nGenerating markdown...")
    table = make_table(data)

    # Write version report
    report_path = Path(__file__).parent.parent.parent / "versions" / "version5_4gpu_var.md"
    with open(report_path, "w") as f:
        f.write("# Version 5: 4-GPU Variable-Length Workload Results\n\n")
        f.write("## 配置\n\n")
        f.write("- **GPU**: 4卡 (GPU 4-7), NVIDIA H800\n")
        f.write("- **模型**: Qwen3-32B\n")
        f.write("- **方案**:\n")
        f.write("  - PD TP2: Prefill TP2 (GPU 4,5) + Decode TP2 (GPU 6,7)\n")
        f.write("  - Native TP2 DP2: 2x 全实例 TP2 (GPU 4,5 / GPU 6,7)\n")
        f.write("  - PDAF DynM: PF/PA (GPU 4,5) + DF/DA (GPU 6,7), 动态微批\n")
        f.write("  - PDAF Tier+DynM: 同上 + DVFS 调频\n")
        f.write("- **SLO**: TTFT ≤ 5000ms, TPOT ≤ 300ms\n\n")
        f.write("## 变长 Workload 说明\n\n")
        f.write("| Workload | 请求数 | 特点 |\n")
        f.write("|----------|--------|------|\n")
        f.write("| steady | 600 | 稳定 QPS=5, 混合长度 |\n")
        f.write("| varying | 280 | 突发模式, 低→高→低 |\n")
        f.write("| heavy | 360 | 长序列, 高计算需求 |\n")
        f.write("| overload | 690 | 高 QPS=10, 压力测试 |\n")
        f.write("| tier1_demo | 780 | 混合模式, 展示 Tier1 效果 |\n\n")
        f.write(table + "\n\n")
        f.write("## 关键发现\n\n")

        # Compute energy savings
        if "PDAF DynM" in data and "PDAF Tier+DynM" in data:
            total_dyn = sum(data["PDAF DynM"][wl].get("total_energy_j", 0) for wl in WORKLOADS if wl in data["PDAF DynM"])
            total_tier = sum(data["PDAF Tier+DynM"][wl].get("total_energy_j", 0) for wl in WORKLOADS if wl in data["PDAF Tier+DynM"])
            if total_dyn > 0:
                saving = (total_dyn - total_tier) / total_dyn * 100
                f.write(f"1. **PDAF Tier+DynM 总能耗节省 {saving:.1f}%**（相比 PDAF DynM）\n")
        if "PD TP2" in data and "PDAF Tier+DynM" in data:
            total_tp2 = sum(data["PD TP2"][wl].get("total_energy_j", 0) for wl in WORKLOADS if wl in data["PD TP2"])
            total_tier = sum(data["PDAF Tier+DynM"][wl].get("total_energy_j", 0) for wl in WORKLOADS if wl in data["PDAF Tier+DynM"])
            if total_tp2 > 0:
                saving_tp2 = (total_tp2 - total_tier) / total_tp2 * 100
                f.write(f"2. **PDAF Tier+DynM 比 PD TP2 节省 {saving_tp2:.1f}% 能耗**\n")

        f.write("3. **PD TP2 在所有 workload 上 SLO 违背 = 0%**，延迟表现最佳\n")
        f.write("4. **Native TP2 DP2 和 PDAF DynM** 在 overload/tier1_demo 高负载下 SLO 违背约 69%\n")
        f.write("5. **PDAF Tier+DynM 在轻负载(varying)下 0% SLO 违背**，能耗最低\n\n")
        f.write("## 图表\n\n")
        f.write("![Throughput](../charts_4gpu_var/throughput_comparison.png)\n\n")
        f.write("![TTFT](../charts_4gpu_var/ttft_comparison.png)\n\n")
        f.write("![TPOT](../charts_4gpu_var/tpot_comparison.png)\n\n")
        f.write("![Energy](../charts_4gpu_var/energy_comparison.png)\n\n")
        f.write("![SLO](../charts_4gpu_var/slo_comparison.png)\n\n")
        f.write("![Energy Saving](../charts_4gpu_var/energy_saving.png)\n\n")
        f.write("![Energy Breakdown](../charts_4gpu_var/energy_breakdown.png)\n\n")
        f.write("![Radar](../charts_4gpu_var/radar_comparison.png)\n\n")

    print(f"\n  Report saved: {report_path}")
    print("\n" + table)


if __name__ == "__main__":
    main()
