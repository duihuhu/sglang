#!/usr/bin/env python3
"""Plot deployment-topology comparison figures from results/deploy/json/.

Generates:
  - deploy_il512_scaling.png : 4-panel QPS scaling on il512_ol128 (throughput,
    TPOT, TTFT, energy/token) for all 7 topologies.
  - deploy_pd_vs_af.png      : PD vs PD+AF head-to-head at equal GPU count.
  - deploy_card_efficiency.png : energy & throughput vs GPU count.
  - deploy_generalization.png : il256_ol512 + il1024_ol128 cross-load bars.
"""
import json
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
JSON_DIR = HERE / "results" / "deploy" / "json"
FIG_DIR = HERE / "results" / "deploy" / "figures"

# Topology display order, colors, GPU counts
DEPLOYS = ["pd_p1d1", "pd_p1d2", "pd_p2d1", "pd_p2d2", "pd_p2d4",
           "pdaf_m2", "pdaf_d_tp2"]
NG = {"pd_p1d1": 2, "pd_p1d2": 3, "pd_p2d1": 3, "pd_p2d2": 4,
      "pd_p2d4": 6, "pdaf_m2": 4, "pdaf_d_tp2": 6}
COLORS = {
    "pd_p1d1": "#4C72B0", "pd_p1d2": "#55A868", "pd_p2d1": "#C44E52",
    "pd_p2d2": "#8172B3", "pd_p2d4": "#CCB974", "pdaf_m2": "#DA8BC3",
    "pdaf_d_tp2": "#937860",
}


def load():
    data = {}
    for f in glob.glob(str(JSON_DIR / "*_results.json")):
        d = json.load(open(f))
        dep = d.get("deploy")
        if not dep:
            continue
        key = (str(d.get("il")), str(d.get("ol")))
        data.setdefault(key, {}).setdefault(dep, {})[float(d["qps"])] = d
    return data


def plot_scaling(data):
    """4-panel QPS scaling on il512_ol128 for all topologies."""
    grp = data.get(("512", "128"), {})
    panels = [("throughput_tok_s", "Throughput (tok/s)", False),
              ("tpot_avg_ms", "TPOT avg (ms)", False),
              ("ttft_avg_ms", "TTFT avg (ms, log)", True),
              ("energy_per_token_mj", "Energy per token (mJ/tok)", False)]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (key, title, logy) in zip(axes.flat, panels):
        for dep in DEPLOYS:
            if dep not in grp:
                continue
            qps = sorted(grp[dep])
            ys = [grp[dep][q][key] for q in qps]
            ax.plot(qps, ys, marker="o", color=COLORS[dep],
                    label=f"{dep} ({NG[dep]}gpu)")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("QPS")
        ax.grid(True, alpha=0.3)
        if logy:
            ax.set_yscale("log")
    axes.flat[0].legend(fontsize=8, ncol=2)
    fig.suptitle("Deployment scaling on il512_ol128 (freq=auto)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deploy_il512_scaling.png", dpi=110)
    plt.close(fig)
    print("wrote deploy_il512_scaling.png")


# [PLOT_PDAF]
def plot_pd_vs_af(data):
    """PD vs PD+AF at equal 4-GPU on il512_ol128: TPOT & energy bars."""
    grp = data.get(("512", "128"), {})
    pd, af = grp.get("pd_p2d2", {}), grp.get("pdaf_m2", {})
    qps = sorted(set(pd) & set(af))
    if not qps:
        return
    import numpy as np
    x = np.arange(len(qps)); w = 0.35
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, key, title in [(axes[0], "tpot_avg_ms", "TPOT avg (ms)"),
                           (axes[1], "energy_per_token_mj", "Energy/token (mJ)")]:
        ax.bar(x - w/2, [pd[q][key] for q in qps], w, label="pd_p2d2 (PD, 4gpu)",
               color="#8172B3")
        ax.bar(x + w/2, [af[q][key] for q in qps], w, label="pdaf_m2 (PD+AF, 4gpu)",
               color="#DA8BC3")
        ax.set_xticks(x); ax.set_xticklabels([f"q{int(q)}" for q in qps])
        ax.set_title(title); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=9)
    fig.suptitle("PD vs PD+AF at equal 4-GPU (il512_ol128) — PD wins on both",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deploy_pd_vs_af.png", dpi=110)
    plt.close(fig)
    print("wrote deploy_pd_vs_af.png")


# [PLOT_CARD]
def plot_card(data):
    """Decode-vs-prefill card allocation: TTFT at high QPS shows decode is the
    bottleneck. Bar chart of TTFT at QPS6 across topologies on il512_ol128."""
    grp = data.get(("512", "128"), {})
    import numpy as np
    deps = [d for d in DEPLOYS if d in grp]
    q = 6.0
    deps = [d for d in deps if q in grp[d]]
    ttft = [grp[d][q]["ttft_avg_ms"] for d in deps]
    thpt = [grp[d][q]["throughput_tok_s"] for d in deps]
    viol = [grp[d][q]["slo_violation_rate"] for d in deps]
    x = np.arange(len(deps))
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    bars = axes[0].bar(x, ttft, color=[COLORS[d] for d in deps])
    for b, v in zip(bars, viol):
        axes[0].text(b.get_x()+b.get_width()/2, b.get_height(),
                     f"{v:.0f}%viol", ha="center", va="bottom", fontsize=8)
    axes[0].set_yscale("log")
    axes[0].set_title("TTFT avg @ QPS6 (log) — spikes = decode saturated")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"{d}\n{NG[d]}gpu" for d in deps], fontsize=8)
    axes[1].bar(x, thpt, color=[COLORS[d] for d in deps])
    axes[1].set_title("Throughput @ QPS6 (tok/s)")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"{d}\n{NG[d]}gpu" for d in deps], fontsize=8)
    axes[1].grid(True, axis="y", alpha=0.3)
    fig.suptitle("Card allocation @ QPS6: p1d2(D2) survives, p2d1(P2)/p1d1 collapse",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deploy_card_efficiency.png", dpi=110)
    plt.close(fig)
    print("wrote deploy_card_efficiency.png")


def plot_generalization(data):
    """Cross-load J/tok bars for the 4 key topologies at QPS2 (all 0 viol)."""
    import numpy as np
    loads = [("256", "512"), ("512", "128"), ("1024", "128")]
    keydeps = ["pd_p1d2", "pd_p2d2", "pdaf_m2", "pdaf_d_tp2"]
    q = 2.0
    x = np.arange(len(loads)); w = 0.2
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for i, dep in enumerate(keydeps):
        ys = []
        for ld in loads:
            g = data.get(ld, {}).get(dep, {})
            ys.append(g[q]["energy_per_token_mj"] if q in g else 0)
        ax.bar(x + (i - 1.5) * w, ys, w, label=f"{dep} ({NG[dep]}gpu)",
               color=COLORS[dep])
    ax.set_xticks(x)
    ax.set_xticklabels([f"il{a}_ol{b}" for a, b in loads])
    ax.set_ylabel("Energy per token (mJ/tok)")
    ax.set_title("Energy efficiency across load types @ QPS2 — PD beats AF everywhere")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deploy_generalization.png", dpi=110)
    plt.close(fig)
    print("wrote deploy_generalization.png")


def plot_tier(data):
    """5-way on il512: pure-PD vs bare-AF / AF+DVFS at M=1 and M=2.
    Focus: does DVFS recover AF energy, and how does M=1 vs M=2 behave?"""
    import numpy as np
    grp = data.get(("512", "128"), {})
    order = ["pd_p2d2", "pd_dp2", "pdaf_m2", "pdaf_m2_tier", "pdaf_m1", "pdaf_m1_tier"]
    cols = {"pd_p2d2": "#8172B3", "pd_dp2": "#C44E52", "pdaf_m2": "#DA8BC3",
            "pdaf_m2_tier": "#55A868", "pdaf_m1": "#DD8452", "pdaf_m1_tier": "#4C72B0"}
    labels = {"pd_p2d2": "pure-PD TP2", "pd_dp2": "pure-PD DP2",
              "pdaf_m2": "bare-AF M=2", "pdaf_m2_tier": "AF+DVFS M=2",
              "pdaf_m1": "bare-AF M=1", "pdaf_m1_tier": "AF+DVFS M=1"}
    order = [d for d in order if d in grp]
    if not order:
        print("tier data incomplete, skip plot_tier"); return
    qps = sorted({q for d in order for q in grp[d]})
    x = np.arange(len(qps)); n = len(order); w = 0.8 / n
    panels = [("energy_per_token_mj", "Energy/token (mJ) — lower=greener"),
              ("ttft_avg_ms", "TTFT avg (ms, log) — DVFS hurts latency"),
              ("tpot_avg_ms", "TPOT avg (ms)")]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2))
    for ax, (key, title) in zip(axes, panels):
        for i, d in enumerate(order):
            ys = [grp[d][q][key] if q in grp[d] else 0 for q in qps]
            ax.bar(x + (i - (n - 1) / 2) * w, ys, w,
                   label=labels[d], color=cols[d])
        ax.set_xticks(x); ax.set_xticklabels([f"q{int(q)}" for q in qps])
        ax.set_title(title); ax.grid(True, axis="y", alpha=0.3); ax.legend(fontsize=8)
        if key == "ttft_avg_ms":
            ax.set_yscale("log")
    fig.suptitle("PD (TP2 vs DP2) vs bare-AF vs AF+DVFS at M=1 / M=2 @ il512_ol128 "
                 "(4 GPU) — TP2 lowest latency+energy; DP2 close but weaker",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "deploy_tier_tradeoff.png", dpi=110)
    plt.close(fig)
    print("wrote deploy_tier_tradeoff.png")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    data = load()
    plot_scaling(data)
    plot_pd_vs_af(data)
    plot_card(data)
    plot_generalization(data)
    plot_tier(data)
    print("All figures saved to", FIG_DIR)


if __name__ == "__main__":
    main()
