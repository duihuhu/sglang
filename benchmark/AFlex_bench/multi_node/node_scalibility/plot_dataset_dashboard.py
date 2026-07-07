#!/usr/bin/env python3
"""Per-dataset dashboard: Energy/token, TTFT P50/P90, TPOT P50/P90 vs QPS.

Usage:
    python3 plot_dataset_dashboard.py --input results/scal_16card_v2_final.json
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
CHARTS_DIR = HERE / "charts" / "16card_datasets"

SCHEMES = [
    ("native_tp2_baseline", "SGLang"),
    ("native_tp2_tier", "DynamoLLM"),
    ("pd_dp_baseline", "DistServe"),
    ("pd_dp_tier", "BiScale"),
    ("pdaf_baseline", "MegaScale"),
    ("pdaf_tier", "AFlex"),
]

DATASETS = [
    ("qa_lpld", "QA", "in128, out64"),
    ("chatbot_lphd", "Chatbot", "in128, out1024"),
    ("balanced_mpmd", "Balanced", "in512, out256"),
    ("rag_hpld", "RAG", "in4096, out64"),
    ("summary_hphd", "Summary", "in4096, out1024"),
]

QPS_LIST = [2, 4, 6, 8, 12, 16]

STYLE = {
    "SGLang": {"color": "#1f77b4", "marker": "o"},
    "DynamoLLM": {"color": "#6baed6", "marker": "s"},
    "DistServe": {"color": "#ff7f0e", "marker": "^"},
    "BiScale": {"color": "#fdbf6f", "marker": "v"},
    "MegaScale": {"color": "#2ca02c", "marker": "D"},
    "AFlex": {"color": "#98df8a", "marker": "*"},
}


def interp_p90(p50, p99):
    """Linear interpolation between stored P50 and P99."""
    if p50 is None or p99 is None:
        return None
    return p50 + (90 - 50) / (99 - 50) * (p99 - p50)


def get_series(results, scheme_key, dataset, metric_fn):
    wl = results.get(scheme_key, {})
    pts = {}
    for q in QPS_LIST:
        key = f"{dataset}_qps{q}"
        v = wl.get(key)
        if not isinstance(v, dict) or v.get("status") != "PASS":
            continue
        val = metric_fn(v)
        if val is not None:
            pts[q] = val
    return pts


def plot_dataset(results, dataset_key, title, subtitle, out_path, ngpu=16):
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    fig.suptitle(
        f"16-Card Micro — {title} ({subtitle}) | Qwen3-32B 2-Node",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    panels = [
        (axes[0], "Energy per Token (mJ/tok)", "energy", None),
        (axes[1], "TTFT (ms)", "ttft", "ttft_proc"),
        (axes[2], "TPOT (ms)", "tpot", "tpot"),
    ]

    for ax, ylabel, panel_kind, lat_prefix in panels:
        for scheme_key, label in SCHEMES:
            st = STYLE[label]
            color = st["color"]
            marker = st["marker"]

            if panel_kind == "energy":
                pts = get_series(
                    results, scheme_key, dataset_key,
                    lambda v: v.get("energy_per_token_mj"),
                )
                if not pts:
                    continue
                xs = sorted(pts)
                ys = [pts[q] for q in xs]
                ax.plot(
                    xs, ys, color=color, marker=marker, linewidth=2.2,
                    markersize=7, label=label,
                )
            else:
                p50_pts = get_series(
                    results, scheme_key, dataset_key,
                    lambda v, p=lat_prefix: v.get(f"{p}_p50_ms"),
                )
                p99_pts = get_series(
                    results, scheme_key, dataset_key,
                    lambda v, p=lat_prefix: v.get(f"{p}_p99_ms"),
                )
                if not p50_pts:
                    continue
                xs = sorted(p50_pts)
                p50_ys = [p50_pts[q] for q in xs]
                p90_ys = [
                    interp_p90(p50_pts[q], p99_pts.get(q)) for q in xs
                ]
                ax.plot(
                    xs, p50_ys, color=color, marker=marker, linewidth=2.2,
                    markersize=6, label=label,
                )
                ax.plot(
                    xs, p90_ys, color=color, marker=marker, linewidth=1.6,
                    markersize=5, linestyle="--", alpha=0.85,
                )

        ax.set_xlabel("QPS", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        if panel_kind != "energy":
            ax.set_title(
                f"{ylabel} — solid: P50, dashed: P90 (P90 interpolated from P50/P99)",
                fontsize=12,
            )
        else:
            ax.set_title(ylabel, fontsize=12)
        ax.set_xticks(QPS_LIST)
        ax.grid(True, alpha=0.35)
        if panel_kind == "energy" or ax.has_data():
            ax.legend(fontsize=10, loc="best", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default=str(HERE / "results" / "scal_16card_v2_final.json"),
    )
    parser.add_argument("--out-dir", default=str(CHARTS_DIR))
    args = parser.parse_args()

    data = json.load(open(args.input))
    results = data.get("results", data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating dataset dashboards from {args.input}...")
    for ds_key, title, subtitle in DATASETS:
        out_path = out_dir / f"{ds_key}_dashboard.png"
        plot_dataset(results, ds_key, title, subtitle, out_path)

    print("Done!")


if __name__ == "__main__":
    main()
