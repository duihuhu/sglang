#!/usr/bin/env python3
"""Plot 7-scheme x 6-dataset benchmark results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
CHARTS_DIR.mkdir(exist_ok=True)

SCHEME_ORDER = [
    "native_tp1_baseline",
    "native_tp1_tier",
    "pd_hetero_baseline",
    "pd_hetero_tier_biscale",
    "pdaf_best_baseline",
    "pdaf_best_tier",
    "tier1_ilp_tier2",
]

LABELS = {
    "native_tp1_baseline": "SGLang",
    "native_tp1_tier": "DynamoLLM",
    "pd_hetero_baseline": "DistServe",
    "pd_hetero_tier_biscale": "BiScale",
    "pdaf_best_baseline": "MegaScale",
    "pdaf_best_tier": "AFlex",
    "tier1_ilp_tier2": "AFlex-Tier1",
}

# (scheme_key, label, color, linestyle, marker)
PLOT_SERIES = [
    ("native_tp1_baseline", "SGLang", "#9E9E9E", "-", "o"),
    ("native_tp1_tier", "DynamoLLM", "#607D8B", "--", "s"),
    ("pd_hetero_baseline", "DistServe", "#FF9800", "-", "^"),
    ("pd_hetero_tier_biscale", "BiScale", "#FFC107", "--", "v"),
    ("pdaf_best_baseline", "MegaScale", "#2196F3", "-", "D"),
    ("pdaf_best_tier", "AFlex", "#4CAF50", "--", "*"),
    ("tier1_ilp_tier2", "AFlex-Tier1", "#E91E63", "-.", "P"),
]

DATASET_TITLES = {
    "code": "Code",
    "conv": "Conversation",
    "qa_lpld": "QA (in128, out64)",
    "chatbot_lphd": "Chatbot (in128, out1024)",
    "balanced_mpmd": "Balanced (in512, out256)",
    "summary_hphd": "Summary (in4096, out1024)",
}

ARCH_SUBTITLE = (
    "SGLang/DynamoLLM: TP1×16 | DistServe/BiScale: PD hetero xnode | "
    "MegaScale: best PDAF (c2 intra tp1×4) | AFlex: Tier1 1P+3D + compositional DVFS"
)

# 6-scheme view: drop sweep AFlex, label Tier1 as AFlex
PLOT_SERIES_AFLEX_TIER1 = [
    ("native_tp1_baseline", "SGLang", "#9E9E9E", "-", "o"),
    ("native_tp1_tier", "DynamoLLM", "#607D8B", "--", "s"),
    ("pd_hetero_baseline", "DistServe", "#FF9800", "-", "^"),
    ("pd_hetero_tier_biscale", "BiScale", "#FFC107", "--", "v"),
    ("pdaf_best_baseline", "MegaScale", "#2196F3", "-", "D"),
    ("tier1_ilp_tier2", "AFlex", "#4CAF50", "--", "*"),
]

# AFlex (Tier1) vs these baselines for energy-saving annotations
SAVING_BASELINES = [
    ("DistServe", "pd_hetero_baseline"),
    ("DynamoLLM", "native_tp1_tier"),
    ("BiScale", "pd_hetero_tier_biscale"),
]
AFLEX_SCHEME_KEYS = ("tier1_ilp_tier2", "pdaf_best_tier")


def _load(path: Path | None) -> tuple[dict, dict]:
    if path is None:
        for pattern in (
            "7scheme_6dataset_final_*.json",
            "7scheme_6dataset_dataset_*.json",
            "7scheme_6dataset_partial_*.json",
        ):
            files = sorted(RESULTS_DIR.glob(pattern))
            if files:
                path = files[-1]
                break
        if path is None:
            raise FileNotFoundError("No 7scheme_6dataset results found")
    print(f"Loading {path.name}")
    payload = json.loads(path.read_text())
    return payload.get("results", payload), payload.get("meta", {})


def _wl_key(dataset: str, qps: int) -> str:
    return f"{dataset}_qps{qps}"


def _get_series(
    data: dict, scheme: str, dataset: str, metric: str, qps_list: list[int]
) -> dict[int, float]:
    pts = {}
    for q in qps_list:
        m = data.get(scheme, {}).get(_wl_key(dataset, q), {})
        if isinstance(m, dict) and m.get("status") == "PASS":
            v = m.get(metric)
            if v is not None:
                pts[q] = v
    return pts


def _active_series(
    data: dict, dataset: str, qps_list: list[int], plot_series: list | None = None
):
    series = plot_series or PLOT_SERIES
    for scheme, label, color, ls, mk in series:
        if any(
            data.get(scheme, {}).get(_wl_key(dataset, q), {}).get("status") == "PASS"
            for q in qps_list
        ):
            yield scheme, label, color, ls, mk


def _plot_lines(ax, data, dataset, metric, qps_list, scale=1.0, ylabel="", plot_series=None):
    for scheme, label, color, ls, mk in _active_series(data, dataset, qps_list, plot_series):
        pts = _get_series(data, scheme, dataset, metric, qps_list)
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] / scale for q in xs]
        ax.plot(
            xs, ys, color=color, linestyle=ls,
            marker=mk, markersize=8, linewidth=2.2, label=label,
        )
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(qps_list)
    ax.grid(True, alpha=0.3)


def _plot_lat_p50_p99(
    ax, data, dataset, prefix: str, ylabel: str, qps_list: list[int],
    slo: float | None = None,
    plot_series: list | None = None,
):
    for scheme, label, color, ls, mk in _active_series(data, dataset, qps_list, plot_series):
        p50 = _get_series(data, scheme, dataset, f"{prefix}_p50_ms", qps_list)
        p99 = _get_series(data, scheme, dataset, f"{prefix}_p99_ms", qps_list)
        if not p50:
            continue
        xs = sorted(p50)
        ax.plot(
            xs, [p50[q] for q in xs], color=color, linestyle=ls,
            marker=mk, markersize=7, linewidth=2.0, label=label,
        )
        if p99:
            ax.plot(
                xs, [p99.get(q, 0) for q in xs],
                color=color, linestyle=":", linewidth=1.4, alpha=0.7,
            )
    if slo is not None:
        ax.axhline(slo, color="red", linestyle="--", linewidth=1.2, alpha=0.6, label="SLO")
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(qps_list)
    ax.grid(True, alpha=0.3)


def _resolve_aflex_key(plot_series: list | None) -> str | None:
    if plot_series:
        for scheme, label, *_ in plot_series:
            if label == "AFlex":
                return scheme
    for key in AFLEX_SCHEME_KEYS:
        if key in LABELS:
            return key
    return None


def _energy_saving_ranges(
    data: dict,
    dataset: str,
    qps_list: list[int],
    aflex_key: str,
) -> list[tuple[str, float, float]]:
    """Return (baseline_label, min%, max%) energy saving for AFlex."""
    rows = []
    for label, base_key in SAVING_BASELINES:
        saves = []
        for q in qps_list:
            bm = data.get(base_key, {}).get(_wl_key(dataset, q), {})
            am = data.get(aflex_key, {}).get(_wl_key(dataset, q), {})
            if bm.get("status") != "PASS" or am.get("status") != "PASS":
                continue
            b = bm.get("energy_per_token_mj")
            a = am.get("energy_per_token_mj")
            if b and a is not None and b > 0:
                saves.append((b - a) / b * 100.0)
        if saves:
            rows.append((label, min(saves), max(saves)))
    return rows


def _annotate_aflex_savings(ax, ranges: list[tuple[str, float, float]], qps_list: list[int]):
    if not ranges:
        return
    qlo, qhi = min(qps_list), max(qps_list)
    lines = [f"AFlex ↓ energy (QPS {qlo}–{qhi}):"]
    for label, lo, hi in ranges:
        if abs(lo - hi) < 0.5:
            lines.append(f"  vs {label}: {lo:.0f}%")
        else:
            lines.append(f"  vs {label}: {lo:.0f}–{hi:.0f}%")
    ax.text(
        0.98, 0.55, "\n".join(lines),
        transform=ax.transAxes,
        fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="#4CAF50", alpha=0.92),
        zorder=10,
    )


def plot_dataset_dashboard_4panel(
    data: dict,
    meta: dict,
    dataset: str,
    out: Path,
    title: str | None = None,
    plot_series: list | None = None,
    subtitle: str | None = None,
    scheme_count: int | None = None,
    annotate_savings: bool = False,
):
    qps_list = meta.get("qps", [2, 8, 16])
    ds_title = title or DATASET_TITLES.get(dataset, dataset)
    ttft_slo = meta.get("ttft_slo_ms", 5000)
    tpot_slo = meta.get("tpot_slo_ms", 300)
    n_schemes = scheme_count or (len(plot_series) if plot_series else len(PLOT_SERIES))
    arch = subtitle or ARCH_SUBTITLE

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{ds_title} — {n_schemes}-Scheme (node34+33) | 16-Card Qwen3-32B | "
        f"SLO: TTFT={ttft_slo/1000:.0f}s, TPOT={tpot_slo:.0f}ms\n{arch}",
        fontsize=11, fontweight="bold", y=0.98,
    )

    ax = axes[0, 0]
    _plot_lines(ax, data, dataset, "energy_per_token_mj", qps_list,
                scale=1000.0, ylabel="Energy/Token (J)", plot_series=plot_series)
    ax.set_title("(a) Energy per Token", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")
    if annotate_savings:
        aflex_key = _resolve_aflex_key(plot_series)
        if aflex_key:
            ranges = _energy_saving_ranges(data, dataset, qps_list, aflex_key)
            _annotate_aflex_savings(ax, ranges, qps_list)

    ax = axes[0, 1]
    _plot_lines(ax, data, dataset, "throughput_tok_s", qps_list,
                ylabel="Throughput (tok/s)", plot_series=plot_series)
    ax.set_title("(b) Throughput", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left")

    ax = axes[1, 0]
    _plot_lat_p50_p99(ax, data, dataset, "tpot", "TPOT (ms)", qps_list,
                      slo=tpot_slo, plot_series=plot_series)
    ax.set_title("(c) TPOT — solid=P50, dotted=P99", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    ax = axes[1, 1]
    _plot_lat_p50_p99(ax, data, dataset, "ttft_proc", "TTFT (ms)", qps_list,
                      plot_series=plot_series)
    ax.set_title("(d) TTFT — solid=P50, dotted=P99", fontweight="bold")
    ax.legend(fontsize=7, loc="upper left", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_energy_by_dataset(data: dict, meta: dict, out: Path):
    results = data
    datasets = meta["datasets"]
    qps_list = meta["qps"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, ds in zip(axes, datasets):
        for scheme, label, color, ls, mk in PLOT_SERIES:
            ys = []
            for q in qps_list:
                m = results.get(scheme, {}).get(_wl_key(ds, q))
                ys.append(
                    m.get("energy_per_token_mj")
                    if m and m.get("status") == "PASS" else np.nan
                )
            ax.plot(qps_list, ys, linestyle=ls, marker=mk,
                    label=label, color=color, linewidth=2)
        ax.set_title(ds)
        ax.set_xlabel("QPS")
        ax.set_ylabel("mJ/tok")
        ax.grid(True, alpha=0.3)

    handles, labels_ = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels_, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("7-Scheme Energy per Token (6 datasets)", fontsize=14, y=1.06)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--dataset", default="code")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="4-panel dashboard output (default: charts/{dataset}_7scheme_dashboard_4panel.png)",
    )
    parser.add_argument(
        "--aflex-tier1-only",
        action="store_true",
        help="Drop sweep AFlex (pdaf_best_tier); plot Tier1 as AFlex (6 schemes)",
    )
    parser.add_argument(
        "--annotate-savings",
        action="store_true",
        help="Annotate AFlex energy saving %% range vs DistServe/DynamoLLM/BiScale on panel (a)",
    )
    parser.add_argument("--overview", action="store_true", help="Also plot 6-dataset energy overview")
    args = parser.parse_args()

    data, meta = _load(args.input)
    out = args.output or CHARTS_DIR / f"{args.dataset}_7scheme_dashboard_4panel.png"
    plot_series = PLOT_SERIES_AFLEX_TIER1 if args.aflex_tier1_only else None
    scheme_count = 6 if args.aflex_tier1_only else None
    annotate = args.aflex_tier1_only or args.annotate_savings
    plot_dataset_dashboard_4panel(
        data, meta, args.dataset, out,
        plot_series=plot_series,
        scheme_count=scheme_count,
        annotate_savings=annotate,
    )

    if args.overview:
        plot_energy_by_dataset(data, meta, CHARTS_DIR / "7scheme_6dataset_energy.png")


if __name__ == "__main__":
    main()
