#!/usr/bin/env python3
"""Generate exp-micro style figures (Figure 4-7) for a given card count.

Each figure corresponds to one scenario (dataset type):
  Figure 4: HPHD (qa)
  Figure 5: HPLD (rag)
  Figure 6: HPLD-extreme (summary)
  Figure 7: LPHD (chatbot)

Each figure has 3 subplots:
  (a) Energy per token (mJ) vs QPS — 6 scheme lines
  (b) TTFT (P50, P99) vs QPS — 6 schemes x 2 percentiles
  (c) TPOT (P50, P99) vs QPS — 6 schemes x 2 percentiles

Usage:
    python3 plot_exp_micro.py --ngpu 8
    python3 plot_exp_micro.py --ngpu 16
    python3 plot_exp_micro.py --ngpu 8 --input results/scal_8card_merged.json
"""
import argparse
import glob
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"

SCENARIO_MAP = {
    "qa": ("Figure 4: HPHD (qa)", "figure_4_hphd_qa"),
    "rag": ("Figure 5: HPLD (rag)", "figure_5_hpld_rag"),
    "summary": ("Figure 6: HPLD-extreme (summary)", "figure_6_hpld_summary"),
    "chatbot": ("Figure 7: LPHD (chatbot)", "figure_7_lphd_chatbot"),
}

SCHEMES = [
    ("native_dp", "baseline"),
    ("native_dp", "tier"),
    ("pd_dp", "baseline"),
    ("pd_dp", "tier"),
    ("pdaf", "baseline"),
    ("pdaf", "tier"),
]

COLORS = {
    ("native_dp", "baseline"): "#90A4AE",
    ("native_dp", "tier"): "#546E7A",
    ("pd_dp", "baseline"): "#42A5F5",
    ("pd_dp", "tier"): "#1565C0",
    ("pdaf", "baseline"): "#66BB6A",
    ("pdaf", "tier"): "#2E7D32",
}

MARKERS = {
    ("native_dp", "baseline"): "o",
    ("native_dp", "tier"): "o",
    ("pd_dp", "baseline"): "s",
    ("pd_dp", "tier"): "s",
    ("pdaf", "baseline"): "^",
    ("pdaf", "tier"): "^",
}

LINESTYLES = {
    "baseline": "-",
    "tier": "--",
}


def load_merged(ngpu, input_path=None):
    if input_path:
        return json.load(open(input_path))
    files = sorted(glob.glob(str(RESULTS_DIR / f"scal_{ngpu}card*.json")))
    merged = {}
    for f in files:
        d = json.load(open(f))
        r = d.get("results", d)
        for deploy, wl in r.items():
            if isinstance(wl, dict):
                merged.setdefault(deploy, {}).update(wl)
    return {"results": merged, "meta": {"ngpu_total": ngpu}}


def get_series(results, scheme, mode, scenario, metric):
    key = f"{scheme}_{mode}"
    wl = results.get(key, {})
    if not wl:
        key = scheme if mode == "baseline" else f"{scheme}_tier"
        wl = results.get(key, {})
    pts = {}
    for k, v in wl.items():
        m = re.match(rf"{scenario}_qps(\d+)$", k)
        if m and isinstance(v, dict) and v.get("status") == "PASS":
            val = v.get(metric)
            if val is not None:
                pts[int(m.group(1))] = val
    return pts


def plot_figure(results, scenario, ngpu, out_dir):
    title, fname = SCENARIO_MAP[scenario]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{title} — {ngpu}-card cross-node | Qwen3-32B",
                 fontsize=13, fontweight="bold", y=0.98)

    # (a) Energy per token
    ax = axes[0]
    for scheme, mode in SCHEMES:
        pts = get_series(results, scheme, mode, scenario, "energy_per_token_mj")
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] for q in xs]
        label = f"{scheme}/{mode}"
        ax.plot(xs, ys, marker=MARKERS[(scheme, mode)],
                linestyle=LINESTYLES[mode],
                color=COLORS[(scheme, mode)],
                linewidth=1.8, markersize=5, label=label)
    ax.set_xlabel("QPS")
    ax.set_ylabel("Energy/token (mJ)")
    ax.set_title("(a) Energy", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, ncol=2, loc="best")

    # (b) TTFT P50/P99
    ax = axes[1]
    for scheme, mode in SCHEMES:
        for pct, ls_suffix in [("p50", "-"), ("p99", ":")]:
            metric = f"ttft_proc_{pct}_ms"
            pts = get_series(results, scheme, mode, scenario, metric)
            if not pts:
                continue
            xs = sorted(pts)
            ys = [pts[q] for q in xs]
            ls = LINESTYLES[mode] if pct == "p50" else ":"
            label = f"{scheme}/{mode} {pct.upper()}"
            alpha = 1.0 if pct == "p50" else 0.6
            ax.plot(xs, ys, marker=MARKERS[(scheme, mode)],
                    linestyle=ls,
                    color=COLORS[(scheme, mode)],
                    linewidth=1.5 if pct == "p50" else 1.0,
                    markersize=4, label=label, alpha=alpha)
    ax.set_xlabel("QPS")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("(b) TTFT (P50 solid, P99 dotted)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5.5, ncol=2, loc="best")

    # (c) TPOT P50/P99
    ax = axes[2]
    for scheme, mode in SCHEMES:
        for pct, ls_suffix in [("p50", "-"), ("p99", ":")]:
            metric = f"tpot_{pct}_ms"
            pts = get_series(results, scheme, mode, scenario, metric)
            if not pts:
                continue
            xs = sorted(pts)
            ys = [pts[q] for q in xs]
            ls = LINESTYLES[mode] if pct == "p50" else ":"
            label = f"{scheme}/{mode} {pct.upper()}"
            alpha = 1.0 if pct == "p50" else 0.6
            ax.plot(xs, ys, marker=MARKERS[(scheme, mode)],
                    linestyle=ls,
                    color=COLORS[(scheme, mode)],
                    linewidth=1.5 if pct == "p50" else 1.0,
                    markersize=4, label=label, alpha=alpha)
    ax.set_xlabel("QPS")
    ax.set_ylabel("TPOT (ms)")
    ax.set_title("(c) TPOT (P50 solid, P99 dotted)", fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=5.5, ncol=2, loc="best")

    plt.tight_layout()
    out_path = out_dir / f"{fname}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpu", type=int, required=True, choices=[4, 8, 16])
    parser.add_argument("--input", type=str, default=None)
    args = parser.parse_args()

    data = load_merged(args.ngpu, args.input)
    results = data.get("results", data)
    ngpu = args.ngpu

    out_dir = CHARTS_DIR / f"{ngpu}card"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating exp-micro figures for {ngpu}-card...")
    for scenario in SCENARIO_MAP:
        plot_figure(results, scenario, ngpu, out_dir)

    # Bonus: energy per token comparison across all datasets
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.suptitle(f"{ngpu}-card — Energy/token across datasets | Qwen3-32B",
                 fontsize=13, fontweight="bold")
    scenarios = list(SCENARIO_MAP.keys())
    width = 0.12
    x_base = np.arange(len(scenarios))

    for i, (scheme, mode) in enumerate(SCHEMES):
        vals = []
        for sc in scenarios:
            pts = get_series(results, scheme, mode, sc, "energy_per_token_mj")
            if pts:
                avg = np.mean(list(pts.values()))
                vals.append(avg)
            else:
                vals.append(0)
        offset = (i - len(SCHEMES) / 2 + 0.5) * width
        ax.bar(x_base + offset, vals, width * 0.9,
               color=COLORS[(scheme, mode)], label=f"{scheme}/{mode}",
               edgecolor="white", linewidth=0.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels([f"{sc}\n({SCENARIO_MAP[sc][0].split(':')[0]})"
                        for sc in scenarios], fontsize=9)
    ax.set_ylabel("Avg Energy/token (mJ)")
    ax.legend(fontsize=8, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out_path = out_dir / "energy_per_token_4datasets.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")

    print("Done!")


if __name__ == "__main__":
    main()
