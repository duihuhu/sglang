#!/usr/bin/env python3
"""Plot Decode MoE breakdown: balanced vs skewed_rank0."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROFILE_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROFILE_ROOT / "data" / "ep_decode_breakdown"
FIG_DIR = PROFILE_ROOT / "fig"

COMPONENTS = ("gate", "topk", "dispatch", "moe_core", "combine", "ep_allreduce")
COMP_LABELS = ("gate", "topk", "dispatch", "moe_core", "combine", "EP AR")
COLORS = ("#9ECAE1", "#6BAED6", "#4292C6", "#2171B5", "#084594", "#B2182B")

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)


def _to_us(ms: float) -> float:
    return ms * 1000.0


def load_breakdown(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for row in rows:
        routing = row["routing"]
        max_us = {k: _to_us(v) for k, v in row["max_rank_us"].items()}
        per_rank = [
            {k: _to_us(v) for k, v in rank_row.items()} for rank_row in row["per_rank_us"]
        ]
        out[routing] = {"max": max_us, "per_rank": per_rank, "meta": row}
    return out


def plot_breakdown_b32(data: dict[str, dict], out_path: Path) -> None:
    bal = data["balanced"]
    skew = data["skewed_rank0"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.1, 1.3, 1.1]})

    # Panel A: stacked components (max rank)
    ax = axes[0]
    x = np.arange(2)
    width = 0.55
    bottoms = np.zeros(2)
    for comp, label, color in zip(COMPONENTS, COMP_LABELS, COLORS):
        vals = [bal["max"][comp], skew["max"][comp]]
        ax.bar(x, vals, width, bottom=bottoms, label=label, color=color, edgecolor="white", linewidth=0.5)
        bottoms += vals
    ax.set_xticks(x)
    ax.set_xticklabels(["balanced", "skewed_rank0"])
    ax.set_ylabel("latency (µs)")
    ax.set_title("(a) Component breakdown (max rank)")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)
    for i, total in enumerate([bal["max"]["total"], skew["max"]["total"]]):
        ax.text(i, total + 30, f"{total:.0f} µs", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Panel B: per-rank moe_core + ep_allreduce
    ax = axes[1]
    ranks = np.arange(4)
    w = 0.18
    for idx, (name, d, offset) in enumerate(
        [("balanced", bal, -1.5), ("skewed", skew, 0.5)]
    ):
        core = [r["moe_core"] for r in d["per_rank"]]
        ar = [r["ep_allreduce"] for r in d["per_rank"]]
        ax.bar(ranks + offset * w, core, w, label=f"{name} moe_core", color=COLORS[3], alpha=0.85 if idx == 0 else 0.55)
        ax.bar(
            ranks + offset * w,
            ar,
            w,
            bottom=core,
            label=f"{name} EP AR",
            color=COLORS[5],
            alpha=0.85 if idx == 0 else 0.55,
        )
        totals = [r["total"] for r in d["per_rank"]]
        for r_i, t in enumerate(totals):
            ax.text(ranks[r_i] + offset * w, t + 25, f"{t:.0f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(ranks)
    ax.set_xticklabels([f"rank {i}" for i in ranks])
    ax.set_ylabel("latency (µs)")
    ax.set_title("(b) Per-rank moe_core + EP AR")
    ax.legend(loc="upper left", ncol=2, framealpha=0.9)

    # Panel C: component delta table as horizontal bars (skew - bal)
    ax = axes[2]
    deltas = [skew["max"][c] - bal["max"][c] for c in COMPONENTS]
    y = np.arange(len(COMPONENTS))
    colors_delta = ["#E45756" if d > 0 else "#4C78A8" for d in deltas]
    ax.barh(y, deltas, color=colors_delta, edgecolor="white")
    ax.axvline(0, color="#333", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(COMP_LABELS)
    ax.set_xlabel("skewed − balanced (µs)")
    ax.set_title("(c) Skewed overhead vs balanced (max rank)")
    total_delta = skew["max"]["total"] - bal["max"]["total"]
    ax.text(
        0.98,
        0.02,
        f"total Δ = {total_delta:+.0f} µs\n(skew/bal = {skew['max']['total']/bal['max']['total']:.2f}×)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    meta = bal["meta"]
    fig.suptitle(
        f"Decode MoE breakdown — EP={meta['world_size']}, length={meta['length']}, "
        f"batch={meta['batch_size']}, {meta['freq_mhz']} MHz (all ranks locked)",
        fontsize=13,
        y=1.02,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out_path}")


def main() -> None:
    src = DATA_DIR / "breakdown_ep4_l512_f930_allrank_lock.json"
    if not src.exists():
        raise SystemExit(f"missing {src}")
    data = load_breakdown(src)
    plot_breakdown_b32(data, FIG_DIR / "ep_decode_breakdown_b32.png")


if __name__ == "__main__":
    main()
