#!/usr/bin/env python3
"""Plot 6-scheme comparison for node scalability results.

Bar charts comparing Native/PD/PDAF (baseline vs tier) across four metrics:
TTFT, TPOT, total energy, energy per token. Throughput annotated.

Usage:
    python3 plot_6scheme_comparison.py [results.json]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"

ORDER = ["native_dp", "pd_dp", "pdaf"]
ARCH_LABEL = {"native_dp": "Native DP", "pd_dp": "PD", "pdaf": "PDAF"}
C_BASE = "#90A4AE"
C_TIER = "#43A047"


def load(path):
    return json.load(open(path))


def collect(data):
    """Return {arch: {mode: metrics}} from results (single workload per scheme)."""
    out = {}
    for scheme, wl in data["results"].items():
        if not isinstance(wl, dict) or "__status__" in wl:
            continue
        if scheme.endswith("_tier"):
            arch, mode = scheme[:-5], "tier"
        elif scheme.endswith("_baseline"):
            arch, mode = scheme[:-9], "baseline"
        else:
            continue
        for _, m in wl.items():
            if isinstance(m, dict) and m.get("status") == "PASS":
                out.setdefault(arch, {})[mode] = m
                break
    return out


def main():
    result_file = Path(sys.argv[1]) if len(sys.argv) > 1 else sorted(
        RESULTS_DIR.glob("scal_8card_*.json"))[-1]
    data = load(result_file)
    meta = data.get("meta", {})
    scen = ",".join(meta.get("scenarios", ["?"]))
    qps = ",".join(str(q) for q in meta.get("qps", ["?"]))
    ngpu = meta.get("ngpu_total", "?")

    d = collect(data)
    archs = [a for a in ORDER if a in d]

    metrics = [
        ("ttft_proc_avg_ms", "TTFT (ms)", "TTFT (lower=better)"),
        ("tpot_avg_ms", "TPOT (ms)", "TPOT (lower=better)"),
        ("total_energy_j", "Total Energy (J)", "Total Energy (lower=better)"),
        ("energy_per_token_mj", "Energy/token (mJ)", "Energy per Token (lower=better)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(
        f"6-Scheme Comparison — {ngpu}-card cross-node | Qwen3-32B | "
        f"scenario={scen} qps={qps}\nNative / PD / PDAF  ×  baseline / tier(DVFS)",
        fontsize=13, fontweight="bold")

    x = np.arange(len(archs))
    w = 0.36

    for ax, (key, ylabel, title) in zip(axes.flat, metrics):
        base_vals = [d[a].get("baseline", {}).get(key, 0) for a in archs]
        tier_vals = [d[a].get("tier", {}).get(key, 0) for a in archs]
        b1 = ax.bar(x - w / 2, base_vals, w, label="baseline", color=C_BASE)
        b2 = ax.bar(x + w / 2, tier_vals, w, label="tier (DVFS)", color=C_TIER)
        ax.set_xticks(x)
        ax.set_xticklabels([ARCH_LABEL[a] for a in archs])
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.legend(fontsize=9)
        for bars in (b1, b2):
            for bar in bars:
                h = bar.get_height()
                if h > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, h,
                            f"{h:.0f}" if h >= 100 else f"{h:.1f}",
                            ha="center", va="bottom", fontsize=8)
        # tier vs baseline % delta on energy charts
        if key in ("total_energy_j", "energy_per_token_mj"):
            for i, a in enumerate(archs):
                bv, tv = base_vals[i], tier_vals[i]
                if bv > 0 and tv > 0:
                    pct = (tv - bv) / bv * 100
                    ax.text(x[i], max(bv, tv) * 1.08, f"{pct:+.0f}%",
                            ha="center", fontsize=8, color="#1565C0",
                            fontweight="bold")

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / f"6scheme_{ngpu}card_{scen}_qps{qps}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {out}")
    plt.close()


if __name__ == "__main__":
    main()
