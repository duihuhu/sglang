#!/usr/bin/env python3
"""8-GPU tier comparison figures from results/8gpu/json/.

Mirrors the deploy_tier_tradeoff.png style: 5-way head-to-head of
pure-PD vs bare-AF vs AF+DVFS at M=1 / M=2, on the 8-GPU topologies
(all TP=2 per stage). Generated after the multi-GPU DVFS fix.

Generates:
  - 8gpu_tier_tradeoff.png : 3-panel (energy/tok, TTFT log, TPOT) on
    il2048_ol64 (the load with the widest 5-way QPS overlap).
  - 8gpu_tier_generalization.png : cross-load J/tok bars at a common
    healthy QPS for the 5 schemes.
"""
import json
import glob
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "results" / "8gpu" / "json"
FIG_DIR = HERE / "results" / "8gpu" / "figures"

ORDER = ["pd_p4d4", "pdaf_8g_m1", "pdaf_8g_m1_tier",
         "pdaf_8g_m2", "pdaf_8g_m2_tier"]
COLS = {"pd_p4d4": "#8172B3", "pdaf_8g_m1": "#DD8452",
        "pdaf_8g_m1_tier": "#4C72B0", "pdaf_8g_m2": "#DA8BC3",
        "pdaf_8g_m2_tier": "#55A868"}
LABELS = {"pd_p4d4": "pure-PD (P4D4)", "pdaf_8g_m1": "bare-AF M=1",
          "pdaf_8g_m1_tier": "AF+DVFS M=1", "pdaf_8g_m2": "bare-AF M=2",
          "pdaf_8g_m2_tier": "AF+DVFS M=2"}


def load():
    data = defaultdict(lambda: defaultdict(dict))
    for f in glob.glob(str(JSON_DIR / "*_results.json")):
        d = json.load(open(f))
        dep = d.get("deploy")
        if not dep:
            continue
        # drop watchdog-cancelled / hung points (thpt==0)
        if d.get("throughput_tok_s", 0) == 0:
            continue
        key = (d["il"], d["ol"])
        data[key][dep][float(d["qps"])] = d
    return data


def plot_tradeoff(data, grp_key=("2048", "64"), max_qps=3):
    """3-panel 5-way comparison on the widest-overlap load."""
    grp = data.get(grp_key, {})
    order = [d for d in ORDER if d in grp]
    if not order:
        print("tier data incomplete, skip plot_tradeoff"); return
    qps = sorted({q for d in order for q in grp[d] if q <= max_qps})
    x = np.arange(len(qps)); n = len(order); w = 0.8 / n
    panels = [("energy_per_token_mj", "Energy/token (mJ) - lower=greener"),
              ("ttft_avg_ms", "TTFT avg (ms, log)"),
              ("tpot_avg_ms", "TPOT avg (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (key, title) in zip(axes, panels):
        for i, d in enumerate(order):
            ys = [grp[d][q][key] if q in grp[d] else 0 for q in qps]
            ax.bar(x + (i - (n - 1) / 2) * w, ys, w,
                   label=LABELS[d], color=COLS[d])
        ax.set_xticks(x); ax.set_xticklabels([f"q{int(q)}" for q in qps])
        ax.set_title(title); ax.grid(True, axis="y", alpha=0.3)
        ax.legend(fontsize=8)
        if key == "ttft_avg_ms":
            ax.set_yscale("log")
    il, ol = grp_key
    fig.suptitle(f"8-GPU: pure-PD vs bare-AF vs AF+DVFS at M=1/M=2 "
                 f"@ il{il}_ol{ol} (all TP=2, multi-card DVFS fixed)",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "8gpu_tier_tradeoff.png", dpi=110)
    plt.close(fig)
    print("wrote 8gpu_tier_tradeoff.png")


def plot_generalization(data, q=2.0):
    """Cross-load J/tok bars for the 5 schemes at a common healthy QPS."""
    loads = [("128", "1024"), ("512", "256"), ("2048", "64"), ("4096", "64")]
    order = ORDER
    x = np.arange(len(loads)); n = len(order); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(13, 6))
    for i, d in enumerate(order):
        ys = []
        for ld in loads:
            g = data.get(ld, {}).get(d, {})
            ys.append(g[q]["energy_per_token_mj"] if q in g else 0)
        ax.bar(x + (i - (n - 1) / 2) * w, ys, w,
               label=LABELS[d], color=COLS[d])
    ax.set_xticks(x)
    ax.set_xticklabels([f"il{a}_ol{b}" for a, b in loads])
    ax.set_ylabel("Energy per token (mJ/tok)")
    ax.set_title(f"8-GPU energy efficiency across load types @ QPS{int(q)} "
                 f"(all 0-viol healthy points)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "8gpu_tier_generalization.png", dpi=110)
    plt.close(fig)
    print("wrote 8gpu_tier_generalization.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    plot_tradeoff(data)
    plot_generalization(data)
    print("All figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()


