#!/usr/bin/env python3
"""Validate and plot two node3 Qwen3-32B decode A/F batch-curve runs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHARTS_DIR = ROOT / "charts"
CHARTS_DIR.mkdir(exist_ok=True)
FILES = [
    DATA_DIR / "decode_bs_curve_node3_shortctx_run1.txt",
    DATA_DIR / "decode_bs_curve_node3_shortctx_run2.txt",
]


def load(path):
    with path.open() as f:
        return {
            int(r["batch_size"]): (float(r["D_A_lat"]) / 1000, float(r["D_F_lat"]) / 1000)
            for r in csv.DictReader(f, delimiter="\t")
        }


runs = [load(p) for p in FILES]
bs = np.array(sorted(set().union(*runs)))
mean = []
err = []
count = []
for b in bs:
    vals = np.array([r[b] for r in runs if b in r])
    mean.append(vals.mean(0))
    count.append(len(vals))
    err.append((vals.max(0) - vals.min(0)) / 2 if len(vals) > 1 else np.zeros(2))
mean = np.array(mean)
err = np.array(err)


def pred(n, j):
    if n in bs:
        return mean[np.where(bs == n)[0][0], j]
    k = np.searchsorted(bs, n)
    x0, x1 = bs[k - 1], bs[k]
    y0, y1 = mean[k - 1, j], mean[k, j]
    return y0 + (n - x0) * (y1 - y0) / (x1 - x0)


def evaluate(x, L=64):
    parts = (x, 512 - x)
    wa = sum(pred(v, 0) for v in parts)
    wf = sum(pred(v, 1) for v in parts)
    return wa + (L - 1) * max(wa, wf) + wf, wa, wf


search = sorted((evaluate(x)[0], x, 512 - x, *evaluate(x)[1:]) for x in range(128, 385))
best = search[0]
equal = evaluate(256)
edge = evaluate(128)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
fig, axs = plt.subplots(1, 2, figsize=(10.8, 4.2), gridspec_kw={"wspace": 0.28})
colors = ["#4C78A8", "#F2B84B"]
names = ["Attention", "FFN"]
for j in range(2):
    axs[0].errorbar(
        bs,
        mean[:, j],
        yerr=err[:, j],
        marker="o" if j == 0 else "s",
        lw=1.6,
        ms=4,
        capsize=2,
        color=colors[j],
        label=names[j],
    )
axs[0].axvline(160, color="#888", ls="--", lw=0.8)
axs[0].axvline(288, color="#888", ls="--", lw=0.8)
axs[0].text(160, 8.7, "wave/tile\ntransition", ha="center", fontsize=7.5, color="#555")
axs[0].text(288, 8.7, "wave/tile\ntransition", ha="center", fontsize=7.5, color="#555")
axs[0].set(
    xlabel="Decode batch size",
    ylabel="Per-layer latency (ms)",
    title="(a) Node3 measured A/F batch curve",
)
axs[0].grid(alpha=0.22, ls="--")
axs[0].legend(frameon=False)
axs[0].spines[["top", "right"]].set_visible(False)

x = np.array([r[1] for r in search])
y = np.array([r[0] for r in search])
axs[1].plot(x, y, color="#6F4E7C", lw=1.7)
axs[1].scatter(
    [256],
    [equal[0]],
    color="#2E7D5B",
    s=42,
    zorder=3,
    label=f"Best: 256+256 ({equal[0]:.1f} ms)",
)
axs[1].scatter(
    [128, 384],
    [edge[0], edge[0]],
    color="#B33A3A",
    s=28,
    zorder=3,
    label=f"128+384 ({edge[0]:.1f} ms)",
)
axs[1].set(
    xlabel="First micro-batch size (second = 512 - first)",
    ylabel="Predicted forward time (ms)",
    title="(b) Partition search with measured curve",
)
axs[1].grid(alpha=0.22, ls="--")
axs[1].legend(frameon=False, fontsize=8)
axs[1].spines[["top", "right"]].set_visible(False)
axs[1].text(
    0.04,
    0.08,
    f"Best split: {best[1]}+{best[2]}\n128+384 is {(edge[0] / equal[0] - 1) * 100:.2f}% slower",
    transform=axs[1].transAxes,
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="#ccc", boxstyle="round,pad=.28"),
)
fig.suptitle(
    "Qwen3-32B decode batch scaling and partition search (node3, TP=1, 210 MHz)",
    fontweight="bold",
    fontsize=11.3,
)
fig.text(
    0.5,
    -0.015,
    "Input/output length = 16/16; error bars show half-range across two independent runs where available.",
    ha="center",
    fontsize=8,
    color="#555",
)
fig.tight_layout()
for ext in ("png", "pdf"):
    path = CHARTS_DIR / f"node3_af_batch_curve_and_search.{ext}"
    fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(path)
plt.close(fig)

summary = {
    "configuration": {
        "node": "node3",
        "container": "operator_test",
        "model": "Qwen3-32B",
        "tp": 1,
        "gpu": 4,
        "clock_mhz": 210,
        "input_len": 16,
        "output_len": 16,
        "warmup": 10,
        "repeat": 50,
    },
    "points": [
        {
            "batch_size": int(b),
            "attention_ms_mean": float(m[0]),
            "ffn_ms_mean": float(m[1]),
            "attention_half_range_ms": float(e[0]),
            "ffn_half_range_ms": float(e[1]),
            "runs": int(n),
        }
        for b, m, e, n in zip(bs, mean, err, count)
    ],
    "search_model": (
        "T_fill + (L-1)*T_cycle + T_drain; T_fill=sum(A); T_cycle=max(sum(A),sum(F)); "
        "T_drain=sum(F); L=64"
    ),
    "best": {"split": [best[1], best[2]], "forward_ms": best[0]},
    "equal": {"split": [256, 256], "forward_ms": equal[0]},
    "edge": {
        "split": [128, 384],
        "forward_ms": edge[0],
        "slowdown_vs_equal_pct": (edge[0] / equal[0] - 1) * 100,
    },
}
out_json = DATA_DIR / "node3_af_batch_curve_validation.json"
out_json.write_text(json.dumps(summary, indent=2) + "\n")
print(out_json)
