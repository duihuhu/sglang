#!/usr/bin/env python3
"""Plot per-card-count 6-scheme curves across scenarios vs QPS.

For one merged results JSON (single ngpu), draw a grid: rows = scenarios,
cols = {energy/token, throughput}, each subplot has 6 lines
(native/pd/pdaf x baseline/tier) vs QPS.

Usage:
    python3 plot_scheme_curves.py results/scal_16card_merged_ALL.json
"""
import json
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
CHARTS_DIR = HERE / "charts"

SCENARIOS = ["chatbot", "qa", "rag", "summary"]
SCHEMES = ["native_dp", "pd_dp", "pdaf"]
STYLE = {
    ("native_dp", "baseline"): ("#90A4AE", "o-"),
    ("native_dp", "tier"): ("#546E7A", "o--"),
    ("pd_dp", "baseline"): ("#42A5F5", "s-"),
    ("pd_dp", "tier"): ("#1565C0", "s--"),
    ("pdaf", "baseline"): ("#66BB6A", "^-"),
    ("pdaf", "tier"): ("#2E7D32", "^--"),
}
METRICS = [("energy_per_token_mj", "Energy/token (mJ)"),
           ("throughput_tok_s", "Throughput (tok/s)")]


def series(results, scheme, mode, scenario):
    sk = scheme if mode == "baseline" else f"{scheme}_tier"
    wl = results.get(sk, {})
    pts = {}
    for k, m in wl.items():
        mm = re.match(rf"{scenario}_qps(\d+)$", k)
        if mm and isinstance(m, dict) and m.get("status") == "PASS":
            pts[int(mm.group(1))] = m
    return pts


def main():
    path = Path(sys.argv[1])
    data = json.load(open(path))
    results = data.get("results", data)
    meta = data.get("meta", {})
    ngpu = meta.get("ngpu_total", "?")

    fig, axes = plt.subplots(len(SCENARIOS), len(METRICS),
                             figsize=(7 * len(METRICS), 4 * len(SCENARIOS)),
                             squeeze=False)
    fig.suptitle(f"{ngpu}-card cross-node — 6 schemes vs QPS | Qwen3-32B",
                 fontsize=15, fontweight="bold")

    for ri, scenario in enumerate(SCENARIOS):
        for ci, (mkey, mlabel) in enumerate(METRICS):
            ax = axes[ri][ci]
            for scheme in SCHEMES:
                for mode in ("baseline", "tier"):
                    pts = series(results, scheme, mode, scenario)
                    vals = {q: m.get(mkey) for q, m in pts.items()
                            if m.get(mkey) is not None}
                    if not vals:
                        continue
                    xs = sorted(vals)
                    ys = [vals[q] for q in xs]
                    color, mk = STYLE[(scheme, mode)]
                    ax.plot(xs, ys, mk, color=color, linewidth=1.8,
                            markersize=5, label=f"{scheme}/{mode}")
            ax.set_title(f"{scenario} — {mlabel}", fontsize=10)
            ax.set_xlabel("QPS")
            ax.set_ylabel(mlabel)
            ax.grid(True, alpha=0.3)
            if ri == 0 and ci == 0:
                ax.legend(fontsize=7, ncol=2)

    plt.tight_layout()
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CHARTS_DIR / f"{ngpu}card_6scheme_curves.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
