#!/usr/bin/env python3
"""Visualize DVFS decision logs: predictor accuracy + selected-frequency mix.

Produces two figures under results/fixed_qps/:

  1) dvfs_pred_accuracy.png  — decode predictor error vs QPS, one line per
     il/ol group (median signed error + |error| band). Quantifies the
     systematic under-estimation of decode iteration time.
  2) dvfs_freq_mix.png       — stacked bar of selected SM-clock distribution
     for decode-attn and decode-ffn across every (group, QPS), showing how
     frequency choice shifts with load.

Run from scripts/bench/ (reads logs/, writes results/fixed_qps/).
"""

import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
LOG_ROOT = HERE / "logs"
OUT = HERE / "results" / "fixed_qps" / "figures"

WARMUP_SKIP = 5
LEGAL_FREQS = [210, 450, 690, 930, 1170, 1410]
FREQ_COLORS = {
    210: "#2c7bb6", 450: "#abd9e9", 690: "#ffffbf",
    930: "#fdae61", 1170: "#f46d43", 1410: "#d7191c",
}
GROUP_COLORS = {
    "il512_ol128": "#1f77b4", "il1024_ol128": "#ff7f0e",
    "il256_ol512": "#2ca02c", "il2048_ol256": "#d62728",
}


def load_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


def parse_meta(path):
    name = Path(path).name
    m = re.search(
        r"(il\d+_ol\d+)_qps([0-9p]+)_tier1_freq_dvfs_decisions_(\w+)_(\w+)_gpu(\d+)",
        name)
    if not m:
        return None
    return {
        "group": m.group(1), "qps": float(m.group(2).replace("p", ".")),
        "persp": m.group(3), "disagg": m.group(4), "gpu": int(m.group(5)),
    }


def collect(disagg="decode"):
    """Return rows keyed by (group, qps, persp) for the given disagg stage."""
    files = sorted(glob.glob(str(LOG_ROOT / "**/*dvfs_decisions*.jsonl"),
                             recursive=True))
    files = [f for f in files if "_archive" not in f]
    data = defaultdict(list)
    for f in files:
        meta = parse_meta(f)
        if not meta or meta["disagg"] != disagg:
            continue
        rows = [r for r in load_jsonl(f) if r.get("phase") == disagg][WARMUP_SKIP:]
        if rows:
            data[(meta["group"], meta["qps"], meta["persp"])].extend(rows)
    return data


def plot_accuracy(data):
    """Median signed predictor error vs QPS, one line per group (attn side)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, persp in zip(axes, ["attn", "ffn"]):
        groups = sorted({g for (g, _q, p) in data if p == persp})
        for g in groups:
            pts = sorted((q, rows) for (gg, q, p), rows in data.items()
                         if gg == g and p == persp)
            xs, med, lo, hi = [], [], [], []
            for q, rows in pts:
                errs = [r["pred_iter_err_pct"] for r in rows
                        if r.get("pred_iter_err_pct") is not None]
                if not errs:
                    continue
                xs.append(q)
                med.append(np.median(errs))
                lo.append(np.percentile(errs, 10))
                hi.append(np.percentile(errs, 90))
            if not xs:
                continue
            c = GROUP_COLORS.get(g, "#555555")
            ax.plot(xs, med, marker="o", color=c, label=g, linewidth=2)
            ax.fill_between(xs, lo, hi, color=c, alpha=0.12)
        ax.axhline(0, color="k", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("QPS (req/s)")
        ax.set_ylabel("pred_iter_err_pct  (pred - obs)/obs ×100")
        ax.set_title(f"Decode-{persp} predictor error vs QPS\n"
                     "(negative ⇒ predictor UNDER-estimates iter time)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("DVFS predictor accuracy (decode side, post-warmup)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / "dvfs_pred_accuracy.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def plot_freq_mix(data, stage="decode", attn_key="sel_f_a", ffn_key="sel_f_f"):
    """Stacked bar of selected-frequency share per QPS, one row per il/ol group
    (columns = <stage>-attn / <stage>-ffn). Splitting by group keeps the x-axis
    labels short (just the QPS) so they never overlap."""
    if not data:
        print(f"No {stage} decision rows — skipping {stage} freq mix")
        return
    groups = sorted({g for (g, _q, _p) in data})
    n = len(groups)
    fig, axes = plt.subplots(n, 2, figsize=(12, 3.2 * n), squeeze=False)
    for gi, grp in enumerate(groups):
        for pj, persp in enumerate(["attn", "ffn"]):
            ax = axes[gi][pj]
            key = attn_key if persp == "attn" else ffn_key
            cols = sorted((k for k in data if k[0] == grp and k[2] == persp),
                          key=lambda k: k[1])
            labels = [f"q{q:g}" for (_g, q, _p) in cols]
            x = np.arange(len(cols))
            bottoms = np.zeros(len(cols))
            for fr in LEGAL_FREQS:
                shares = []
                for (g, q, p) in cols:
                    rows = data[(g, q, p)]
                    tot = sum(1 for r in rows if key in r)
                    cnt = sum(1 for r in rows if r.get(key) == fr)
                    shares.append(100.0 * cnt / tot if tot else 0.0)
                shares = np.array(shares)
                ax.bar(x, shares, bottom=bottoms, color=FREQ_COLORS[fr],
                       edgecolor="white", linewidth=0.5, label=f"{fr} MHz")
                bottoms += shares
            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=8)
            ax.set_ylim(0, 100)
            if pj == 0:
                ax.set_ylabel("freq share (%)")
            ax.set_title(f"{grp}  {stage}-{persp}", fontsize=10)
            # Legend only once (top-right subplot), referencing all freq bands.
            if gi == 0 and pj == 1:
                ax.legend(title="SM clock", bbox_to_anchor=(1.01, 1),
                          loc="upper left", fontsize=8)
    fig.suptitle(f"DVFS selected-frequency distribution ({stage} side, post-warmup)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT / f"dvfs_freq_mix_{stage}.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    # Predictor accuracy is only logged in a useful form on the decode side.
    data_decode = collect(disagg="decode")
    if not data_decode:
        print("No decode decision logs found under", LOG_ROOT)
    else:
        print(f"Loaded {len(data_decode)} (group,qps,persp) decode series")
        plot_accuracy(data_decode)
        plot_freq_mix(data_decode, stage="decode",
                      attn_key="sel_f_a", ffn_key="sel_f_f")
    # Prefill side: frequency-choice distribution (fields are f_a / f_f).
    data_prefill = collect(disagg="prefill")
    print(f"Loaded {len(data_prefill)} (group,qps,persp) prefill series")
    plot_freq_mix(data_prefill, stage="prefill",
                  attn_key="f_a", ffn_key="f_f")


if __name__ == "__main__":
    main()
