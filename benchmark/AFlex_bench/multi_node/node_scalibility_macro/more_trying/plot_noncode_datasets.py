#!/usr/bin/env python3
"""Plot non-code datasets with available 7-scheme QPS6 results + Tier1 MS/AFlex."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_7scheme_6dataset import (
    CHARTS_DIR,
    DATASET_TITLES,
    RESULTS_DIR,
    _get_series,
    _plot_lat_p50_p99,
    _plot_lines,
    plot_dataset_dashboard_4panel,
)

PLOT_5SCHEME = [
    ("native_tp1_baseline", "SGLang", "#9E9E9E", "-", "o"),
    ("native_tp1_tier", "DynamoLLM", "#607D8B", "--", "s"),
    ("pd_hetero_baseline", "DistServe", "#FF9800", "-", "^"),
    ("pd_hetero_tier_biscale", "BiScale", "#FFC107", "--", "v"),
    ("tier1_ilp_tier2", "AFlex-Tier1", "#4CAF50", "-.", "*"),
]

PLOT_TIER1_MS_AFLEX = [
    ("megascale_tier1", "MegaScale", "#2196F3", "-", "D"),
    ("aflex_tier1", "AFlex", "#4CAF50", "--", "*"),
]

SUBTITLE_5 = (
    "SGLang/DynamoLLM: TP1×16 | DistServe/BiScale: PD hetero xnode | "
    "AFlex-Tier1: fixed Tier1 QPS4 1P+3D + compositional DVFS"
)
SUBTITLE_TIER1 = "MegaScale: Tier1 topology + lock 1410 MHz | AFlex: Tier1 topology + compositional DVFS (per-QPS layout)"


def _load_latest(pattern: str) -> tuple[dict, dict]:
    files = sorted(RESULTS_DIR.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern}")
    payload = json.loads(files[-1].read_text())
    return payload.get("results", payload), payload.get("meta", {})


def _scheme_complete(data: dict, schemes: list[str], dataset: str, qps_list: list[int]) -> bool:
    for sk in schemes:
        for q in qps_list:
            m = data.get(sk, {}).get(f"{dataset}_qps{q}", {})
            if m.get("status") != "PASS":
                return False
    return True


def _count_pass(data: dict, schemes: list[str], dataset: str, qps_list: list[int]) -> int:
    n = 0
    for sk in schemes:
        for q in qps_list:
            if data.get(sk, {}).get(f"{dataset}_qps{q}", {}).get("status") == "PASS":
                n += 1
    return n


def plot_tier1_ms_aflex_panel(
    data: dict, meta: dict, dataset: str, out: Path,
):
    qps_list = meta.get("qps", [2, 4, 6, 8, 12, 16])
    ttft_slo = meta.get("ttft_slo_ms", 5000)
    tpot_slo = meta.get("tpot_slo_ms", 300)
    ds_title = DATASET_TITLES.get(dataset, dataset)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{ds_title} — Tier1 MegaScale vs AFlex | 16-Card Qwen3-32B | "
        f"SLO: TTFT={ttft_slo/1000:.0f}s, TPOT={tpot_slo:.0f}ms\n{SUBTITLE_TIER1}",
        fontsize=11, fontweight="bold", y=0.98,
    )

    ax = axes[0, 0]
    _plot_lines(ax, data, dataset, "energy_per_token_mj", qps_list,
                scale=1000.0, ylabel="Energy/Token (J)", plot_series=PLOT_TIER1_MS_AFLEX)
    ax.set_title("(a) Energy per Token", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    _plot_lines(ax, data, dataset, "throughput_tok_s", qps_list,
                ylabel="Throughput (tok/s)", plot_series=PLOT_TIER1_MS_AFLEX)
    ax.set_title("(b) Throughput", fontweight="bold")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    _plot_lat_p50_p99(ax, data, dataset, "tpot", "TPOT (ms)", qps_list,
                      slo=tpot_slo, plot_series=PLOT_TIER1_MS_AFLEX)
    ax.set_title("(c) TPOT", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 1]
    _plot_lat_p50_p99(ax, data, dataset, "ttft_proc", "TTFT (ms)", qps_list,
                      plot_series=PLOT_TIER1_MS_AFLEX)
    ax.set_title("(d) TTFT", fontweight="bold")
    ax.legend(fontsize=8, ncol=2)

    # Energy saving annotation
    saves = []
    for q in qps_list:
        ms = data.get("megascale_tier1", {}).get(f"{dataset}_qps{q}", {})
        af = data.get("aflex_tier1", {}).get(f"{dataset}_qps{q}", {})
        if ms.get("status") == "PASS" and af.get("status") == "PASS":
            b, a = ms["energy_per_token_mj"], af["energy_per_token_mj"]
            if b > 0:
                saves.append((b - a) / b * 100)
    if saves:
        ax = axes[0, 0]
        ax.text(
            0.98, 0.55,
            f"AFlex ↓ energy vs MegaScale:\n  {min(saves):.0f}–{max(saves):.0f}%",
            transform=ax.transAxes, fontsize=9, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#4CAF50", alpha=0.9),
        )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="conv,qa_lpld")
    args = parser.parse_args()

    data7, meta7 = _load_latest("7scheme_6dataset_*.json")
    qps_list = meta7.get("qps", [2, 4, 6, 8, 12, 16])
    schemes5 = [s[0] for s in PLOT_5SCHEME]

    try:
        data_t1, meta_t1 = _load_latest("tier1_megascale_aflex_*.json")
    except FileNotFoundError:
        data_t1, meta_t1 = {}, {"qps": qps_list}

    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        n = _count_pass(data7, schemes5, ds, qps_list)
        total = len(schemes5) * len(qps_list)
        if n == 0:
            print(f"SKIP {ds}: no 7scheme data")
            continue

        out5 = CHARTS_DIR / f"{ds}_5scheme_dashboard_4panel.png"
        plot_dataset_dashboard_4panel(
            data7, meta7, ds, out5,
            plot_series=PLOT_5SCHEME,
            scheme_count=5,
            subtitle=SUBTITLE_5,
            annotate_savings=False,
        )
        print(f"  {ds} 7scheme: {n}/{total} points plotted")

        n_t1 = _count_pass(data_t1, ["megascale_tier1", "aflex_tier1"], ds, qps_list)
        if n_t1 > 0:
            out_t1 = CHARTS_DIR / f"{ds}_tier1_ms_aflex_dashboard_4panel.png"
            plot_tier1_ms_aflex_panel(data_t1, meta_t1, ds, out_t1)
            print(f"  {ds} tier1 MS/AFlex: {n_t1}/12 points plotted")


if __name__ == "__main__":
    main()
