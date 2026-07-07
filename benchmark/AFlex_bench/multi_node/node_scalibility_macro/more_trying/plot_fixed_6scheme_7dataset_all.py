#!/usr/bin/env python3
"""Plot all comparison charts for fixed 6-scheme x 7-dataset results."""
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

QPS_LIST = [2, 4, 6, 8, 12, 16]

SCHEME_ORDER = [
    "native_tp1_baseline",
    "native_tp1_tier",
    "pd_xnode_tp1_baseline",
    "pd_xnode_tp1_tier",
    "pdaf_tp1_baseline",
    "pdaf_tp1_tier",
]
SCHEME_LABELS = {
    "native_tp1_baseline": "SGLang",
    "native_tp1_tier": "DynamoLLM",
    "pd_xnode_tp1_baseline": "DistServe",
    "pd_xnode_tp1_tier": "BiScale",
    "pdaf_tp1_baseline": "MegaScale",
    "pdaf_tp1_tier": "AFlex",
}
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]
MARKERS = ["o", "s", "^", "v", "D", "*"]
STYLES = ["-", "--", "-", "--", "-", "--"]

# (scheme_key, label, color_idx, linestyle, marker)
PLOT_SERIES = [
    ("native_tp1_baseline", "SGLang", 0, "-", "o"),
    ("native_tp1_tier", "DynamoLLM", 1, "--", "s"),
    ("native_tp1_tier_minfreq", "DynamoLLM (min-F)", 1, ":", "s"),
    ("pd_xnode_tp1_baseline", "DistServe", 2, "-", "^"),
    ("pd_hetero_baseline", "DistServe", 2, "-", "^"),
    ("pd_xnode_tp1_tier", "BiScale (min-E)", 3, "--", "v"),
    ("pd_xnode_tp1_tier_biscale", "BiScale (paper, TP1×8)", 3, "-.", "v"),
    ("pd_hetero_tier_biscale", "BiScale", 3, "--", "<"),
    ("pd_xnode_tp1_tier_minfreq", "BiScale (min-F)", 3, ":", "v"),
    ("pdaf_tp1_baseline", "MegaScale", 4, "-", "D"),
    ("pdaf_tp1_tier", "AFlex", 5, "--", "*"),
]

ARCH_SUBTITLE = (
    "SGLang/DynamoLLM: TP1×16 | DistServe/BiScale: P=8×TP1+D=8×TP1 (xnode) | "
    "MegaScale/AFlex: PDAF TP1×4 | Tier DVFS: min-E (Dynamo/BiScale) vs min-F"
)
DATASETS = [
    ("conv", "Conversation", "macro"),
    ("code", "Code", "macro"),
    ("qa_lpld", "QA", "micro", "in128, out64"),
    ("chatbot_lphd", "Chatbot", "micro", "in128, out1024"),
    ("balanced_mpmd", "Balanced", "micro", "in512, out256"),
    ("rag_hpld", "RAG", "micro", "in4096, out64"),
    ("summary_hphd", "Summary", "micro", "in4096, out1024"),
]


def load_results(path: Path | None) -> tuple[dict, dict]:
    if path is None:
        files = sorted(RESULTS_DIR.glob("fixed_6scheme_7dataset_final_*.json"))
        if files:
            path = files[-1]
        elif (RESULTS_DIR / "pdaf_code_final.json").exists():
            path = RESULTS_DIR / "pdaf_code_final.json"
        else:
            raise FileNotFoundError(
                "No fixed_6scheme_7dataset_final_*.json or pdaf_code_final.json"
            )
    print(f"Loading: {path}")
    payload = json.loads(path.read_text())
    return payload.get("results", payload), payload.get("meta", {})


def load_merged(
    main_path: Path | None = None,
    minfreq_path: Path | None = None,
    biscale_path: Path | None = None,
    hetero_biscale_path: Path | None = None,
    noradix_path: Path | None = None,
    distserve_hetero_path: Path | None = None,
    supplement_path: Path | None = None,
    include_minfreq: bool = True,
    include_biscale_paper: bool = True,
) -> dict:
    data, _ = load_results(main_path)
    if supplement_path is not None and supplement_path.exists():
        sup = json.loads(supplement_path.read_text()).get("results", {})
        for scheme, wl in sup.items():
            if not isinstance(wl, dict) or "__status__" in wl:
                continue
            dst = data.setdefault(scheme, {})
            for k, v in wl.items():
                if isinstance(v, dict) and v.get("status") == "PASS":
                    if k not in dst or dst.get(k, {}).get("status") != "PASS":
                        dst[k] = v
        print(f"Supplemented from: {supplement_path.name}")
    if include_minfreq:
        if minfreq_path is None:
            for pattern in (
                "fixed_6scheme_7dataset_minfreq_tier_final_*.json",
                "fixed_6scheme_7dataset_minfreq_tier_partial_*.json",
            ):
                files = sorted(RESULTS_DIR.glob(pattern))
                if files:
                    minfreq_path = files[-1]
                    break
        if minfreq_path is not None and minfreq_path.exists():
            mf = json.loads(minfreq_path.read_text()).get("results", {})
            data.update(mf)
            print(f"Merged min-freq tier: {minfreq_path.name}")
    if include_biscale_paper:
        if biscale_path is None:
            for pattern in (
                "fixed_6scheme_7dataset_final_20260706_183348.json",
                "fixed_6scheme_code_biscale_*.json",
            ):
                files = sorted(RESULTS_DIR.glob(pattern))
                if files:
                    biscale_path = files[-1]
                    break
        if biscale_path is not None and biscale_path.exists():
            bs = json.loads(biscale_path.read_text()).get("results", {})
            tier = bs.get("pd_xnode_tp1_tier", {})
            code_pts = {k: v for k, v in tier.items()
                        if k.startswith("code_qps") and isinstance(v, dict)}
            if code_pts:
                data["pd_xnode_tp1_tier_biscale"] = code_pts
                print(f"Merged BiScale paper (code): {biscale_path.name} "
                      f"({len(code_pts)} points)")
    if hetero_biscale_path is None:
        files = sorted(RESULTS_DIR.glob("biscale_pd_hetero_code_final_*.json"))
        if files:
            hetero_biscale_path = files[-1]
    if hetero_biscale_path is not None and hetero_biscale_path.exists():
        hb = json.loads(hetero_biscale_path.read_text()).get("results", {})
        het = hb.get("pd_hetero_tier_biscale", hb)
        code_pts = {k: v for k, v in het.items()
                    if k.startswith("code_qps") and isinstance(v, dict)}
        if code_pts:
            data["pd_hetero_tier_biscale"] = code_pts
            print(f"Merged BiScale hetero (code): {hetero_biscale_path.name} "
                  f"({len(code_pts)} points)")
    if noradix_path is None:
        files = sorted(RESULTS_DIR.glob("noradix_3scheme_code_final_*.json"))
        if files:
            noradix_path = files[-1]
    if noradix_path is not None and noradix_path.exists():
        nr = json.loads(noradix_path.read_text()).get("results", {})
        n_pts = 0
        for scheme, wl in nr.items():
            if not isinstance(wl, dict) or "__status__" in wl:
                continue
            code_pts = {k: v for k, v in wl.items()
                        if k.startswith("code_qps") and isinstance(v, dict)
                        and v.get("status") == "PASS"}
            if code_pts:
                data.setdefault(scheme, {}).update(code_pts)
                n_pts += len(code_pts)
        if n_pts:
            print(f"Merged no-radix 3-scheme (code): {noradix_path.name} "
                  f"({n_pts} points)")
    if distserve_hetero_path is None:
        files = sorted(RESULTS_DIR.glob("distserve_pd_hetero_code_final_*.json"))
        if files:
            distserve_hetero_path = files[-1]
    if distserve_hetero_path is not None and distserve_hetero_path.exists():
        dh = json.loads(distserve_hetero_path.read_text()).get("results", {})
        het = dh.get("pd_hetero_baseline", dh)
        code_pts = {k: v for k, v in het.items()
                    if k.startswith("code_qps") and isinstance(v, dict)
                    and v.get("status") == "PASS"}
        if code_pts:
            data["pd_hetero_baseline"] = code_pts
            print(f"Merged DistServe hetero (code): {distserve_hetero_path.name} "
                  f"({len(code_pts)} points)")
    return data


def _has_biscale_paper(data: dict, dataset: str) -> bool:
    bs = data.get("pd_xnode_tp1_tier_biscale", {})
    if "__status__" in bs:
        return False
    return any(
        bs.get(wl_key(dataset, q), {}).get("status") == "PASS"
        for q in QPS_LIST
    )


def _has_hetero_biscale(data: dict, dataset: str) -> bool:
    hb = data.get("pd_hetero_tier_biscale", {})
    if "__status__" in hb:
        return False
    return any(
        hb.get(wl_key(dataset, q), {}).get("status") == "PASS"
        for q in QPS_LIST
    )


def _has_hetero_distserve(data: dict, dataset: str) -> bool:
    hd = data.get("pd_hetero_baseline", {})
    if "__status__" in hd:
        return False
    return any(
        hd.get(wl_key(dataset, q), {}).get("status") == "PASS"
        for q in QPS_LIST
    )


def _active_series(data: dict, dataset: str | None = None) -> list[tuple]:
    active = []
    for scheme, label, ci, ls, mk in PLOT_SERIES:
        if scheme == "pd_xnode_tp1_tier_biscale":
            continue
        if (
            scheme == "pd_xnode_tp1_baseline"
            and dataset is not None
            and _has_hetero_distserve(data, dataset)
        ):
            continue
        if (
            scheme == "pd_xnode_tp1_tier"
            and dataset is not None
            and (_has_biscale_paper(data, dataset) or _has_hetero_biscale(data, dataset))
        ):
            continue
        if (
            scheme == "native_tp1_tier_minfreq"
            and dataset is not None
            and get_series(data, "native_tp1_tier", dataset, "throughput_tok_s")
        ):
            continue
        if scheme not in data:
            continue
        wl = data.get(scheme, {})
        if "__status__" in wl:
            continue
        if dataset is not None and not get_series(data, scheme, dataset, "throughput_tok_s"):
            continue
        active.append((scheme, label, ci, ls, mk))
    return active


def wl_key(dataset: str, qps: int) -> str:
    return f"{dataset}_qps{qps}"


def get_series(data: dict, scheme: str, dataset: str, metric: str) -> dict[int, float]:
    pts = {}
    for q in QPS_LIST:
        m = data.get(scheme, {}).get(wl_key(dataset, q), {})
        if isinstance(m, dict) and m.get("status") == "PASS":
            v = m.get(metric)
            if v is not None:
                pts[q] = v
    return pts


def interp_p90(p50: float | None, p99: float | None) -> float | None:
    if p50 is None or p99 is None:
        return None
    return p50 + (90 - 50) / (99 - 50) * (p99 - p50)


def _plot_lines(ax, data, dataset, metric, scale=1.0, ylabel=""):
    for scheme, label, ci, ls, mk in _active_series(data, dataset):
        pts = get_series(data, scheme, dataset, metric)
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] / scale for q in xs]
        ax.plot(
            xs, ys,
            color=COLORS[ci], linestyle=ls,
            marker=mk, markersize=7, linewidth=2.2,
            label=label,
        )
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)


def _plot_lat_p50_p99(ax, data, dataset, prefix: str, ylabel: str, slo: float | None = None):
    for scheme, label, ci, ls, mk in _active_series(data, dataset):
        p50 = get_series(data, scheme, dataset, f"{prefix}_p50_ms")
        p99 = get_series(data, scheme, dataset, f"{prefix}_p99_ms")
        if not p50:
            continue
        xs = sorted(p50)
        ax.plot(
            xs, [p50[q] for q in xs],
            color=COLORS[ci], linestyle=ls,
            marker=mk, markersize=6, linewidth=2.0,
            label=label,
        )
        if p99:
            ax.plot(
                xs, [p99.get(q, 0) for q in xs],
                color=COLORS[ci], linestyle=":", linewidth=1.4, alpha=0.7,
            )
    if slo is not None:
        ax.axhline(slo, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
    ax.set_xlabel("QPS", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)


def plot_dataset_dashboard_4panel(data: dict, dataset: str, title: str, subtitle: str, out: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"{title} — 6-Scheme (node3+node4) | 16-Card Qwen3-32B | SLO: TTFT=5s, TPOT=300ms",
        fontsize=12, fontweight="bold", y=0.98,
    )

    ax = axes[0, 0]
    _plot_lines(ax, data, dataset, "energy_per_token_mj", scale=1000.0, ylabel="Energy/Token (J)")
    ax.set_title("(a) Energy per Token", fontweight="bold")
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[0, 1]
    _plot_lines(ax, data, dataset, "throughput_tok_s", ylabel="Throughput (tok/s)")
    ax.set_title("(b) Throughput", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1, 0]
    _plot_lat_p50_p99(ax, data, dataset, "tpot", "TPOT (ms)", slo=300)
    ax.set_title("(c) TPOT — solid=P50, dotted=P99", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    ax = axes[1, 1]
    _plot_lat_p50_p99(ax, data, dataset, "ttft_proc", "TTFT (ms)")
    ax.set_title("(d) TTFT — solid=P50, dotted=P99", fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_dataset_dashboard_3panel(data: dict, dataset: str, title: str, subtitle: str, out: Path):
    """Macro-style 3-panel: energy, TTFT P90, TPOT P90."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(
        f"16-Card Macro Dashboard — {title} | Qwen3-32B 2-Node\n{ARCH_SUBTITLE}",
        fontsize=13, fontweight="bold", y=1.02,
    )

    ax = axes[0]
    for scheme, label, ci, ls, mk in _active_series(data, dataset):
        pts = get_series(data, scheme, dataset, "energy_per_token_mj")
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] / 1000.0 for q in xs]
        ax.plot(xs, ys, color=COLORS[ci], linestyle=ls,
                marker=mk, markersize=7, linewidth=2.2, label=label)
    ax.set_xlabel("QPS")
    ax.set_ylabel("Energy/Token (J)")
    ax.set_title("(a) Energy per Token", fontweight="bold")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")

    for ax_idx, (prefix, ylabel, panel) in enumerate([
        ("ttft_proc", "TTFT P90 (ms)", "(b) TTFT P90"),
        ("tpot", "TPOT P90 (ms)", "(c) TPOT P90"),
    ], start=1):
        ax = axes[ax_idx]
        for scheme, label, ci, ls, mk in _active_series(data, dataset):
            p50_pts = get_series(data, scheme, dataset, f"{prefix}_p50_ms")
            p99_pts = get_series(data, scheme, dataset, f"{prefix}_p99_ms")
            if not p50_pts:
                continue
            xs = sorted(p50_pts)
            p90_ys = [interp_p90(p50_pts[q], p99_pts.get(q)) for q in xs]
            ax.plot(xs, p90_ys, color=COLORS[ci], linestyle=ls,
                    marker=mk, markersize=7, linewidth=2.2, label=label)
        ax.set_xlabel("QPS")
        ax.set_ylabel(ylabel)
        ax.set_title(panel, fontweight="bold")
        ax.set_xticks(QPS_LIST)
        ax.grid(True, alpha=0.3)
        if ax_idx == 1:
            ax.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_micro_3panel(data: dict, dataset: str, title: str, subtitle: str, out: Path):
    fig, axes = plt.subplots(3, 1, figsize=(14, 15))
    fig.suptitle(
        f"16-Card Micro — {title} ({subtitle}) | Qwen3-32B 2-Node\n{ARCH_SUBTITLE}",
        fontsize=14, fontweight="bold", y=0.995,
    )

    ax = axes[0]
    for scheme, label, ci, ls, mk in _active_series(data, dataset):
        pts = get_series(data, scheme, dataset, "energy_per_token_mj")
        if not pts:
            continue
        xs = sorted(pts)
        ax.plot(xs, [pts[q] for q in xs], color=COLORS[ci], linestyle=ls,
                marker=mk, markersize=7, linewidth=2.2, label=label)
    ax.set_xlabel("QPS", fontsize=12)
    ax.set_ylabel("Energy per Token (mJ/tok)", fontsize=12)
    ax.set_title("Energy per Token", fontsize=12)
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.35)
    ax.legend(fontsize=10, loc="best", ncol=2)

    for ax, prefix, ylabel in [
        (axes[1], "ttft_proc", "TTFT (ms)"),
        (axes[2], "tpot", "TPOT (ms)"),
    ]:
        for scheme, label, ci, ls, mk in _active_series(data, dataset):
            p50 = get_series(data, scheme, dataset, f"{prefix}_p50_ms")
            p99 = get_series(data, scheme, dataset, f"{prefix}_p99_ms")
            if not p50:
                continue
            xs = sorted(p50)
            ax.plot(xs, [p50[q] for q in xs], color=COLORS[ci], linestyle="-",
                    marker=mk, markersize=6, linewidth=2.0, label=label)
            ax.plot(xs, [interp_p90(p50[q], p99.get(q)) for q in xs],
                    color=COLORS[ci], linestyle="--", linewidth=1.6, alpha=0.85)
        ax.set_xlabel("QPS", fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(f"{ylabel} — solid: P50, dashed: P90", fontsize=12)
        ax.set_xticks(QPS_LIST)
        ax.grid(True, alpha=0.35)

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_energy_overview_grid(data: dict, out: Path):
    """7-dataset energy grid (2 rows: macro + micro)."""
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        "Fixed 6-Scheme — Energy per Token (J) | All 7 Datasets\n" + ARCH_SUBTITLE,
        fontsize=14, fontweight="bold", y=1.0,
    )
    axes_flat = axes.flatten()
    for idx, ds_info in enumerate(DATASETS):
        ax = axes_flat[idx]
        dataset = ds_info[0]
        title = ds_info[1]
        for scheme, label, ci, ls, mk in _active_series(data, dataset):
            pts = get_series(data, scheme, dataset, "energy_per_token_mj")
            if not pts:
                continue
            xs = sorted(pts)
            ys = [pts[q] / 1000.0 for q in xs]
            ax.plot(xs, ys, color=COLORS[ci], linestyle=ls,
                    marker=mk, markersize=5, linewidth=1.8, label=label)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("QPS", fontsize=9)
        ax.set_ylabel("J/tok", fontsize=9)
        ax.set_xticks(QPS_LIST)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper right")

    axes_flat[7].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_energy_qps16_bars(data: dict, out: Path):
    """Grouped bar chart: energy @ QPS16 across datasets."""
    datasets = [d[0] for d in DATASETS]
    labels = [d[1] for d in DATASETS]
    x = np.arange(len(datasets))
    width = 0.10
    fig, ax = plt.subplots(figsize=(20, 7))
    series = _active_series(data)
    n = len(series)
    for j, (scheme, label, ci, ls, mk) in enumerate(series):
        vals = []
        for ds in datasets:
            if scheme not in {s for s, *_ in _active_series(data, ds)}:
                vals.append(0)
                continue
            m = data.get(scheme, {}).get(wl_key(ds, 16), {})
            vals.append(m.get("energy_per_token_mj", 0) / 1000.0 if m.get("status") == "PASS" else 0)
        offset = (j - (n - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=label, color=COLORS[ci])

    ax.set_xlabel("Dataset", fontsize=12)
    ax.set_ylabel("Energy/Token (J) @ QPS16", fontsize=12)
    ax.set_title(
        "Energy Comparison @ QPS16 — All Datasets\n" + ARCH_SUBTITLE,
        fontsize=13, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.legend(fontsize=9, ncol=3, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_throughput_overview_grid(data: dict, out: Path):
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    fig.suptitle(
        "Fixed 6-Scheme — Throughput (tok/s) | All 7 Datasets\n" + ARCH_SUBTITLE,
        fontsize=14, fontweight="bold", y=1.0,
    )
    for idx, ds_info in enumerate(DATASETS):
        ax = axes.flatten()[idx]
        dataset, title = ds_info[0], ds_info[1]
        for scheme, label, ci, ls, mk in _active_series(data, dataset):
            pts = get_series(data, scheme, dataset, "throughput_tok_s")
            if not pts:
                continue
            xs = sorted(pts)
            ax.plot(xs, [pts[q] for q in xs], color=COLORS[ci], linestyle=ls,
                    marker=mk, markersize=5, linewidth=1.8, label=label)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("QPS", fontsize=9)
        ax.set_ylabel("tok/s", fontsize=9)
        ax.set_xticks(QPS_LIST)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7, loc="upper left")
    axes.flatten()[7].axis("off")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_savings_heatmap(data: dict, out: Path):
    """AFlex energy saving % vs SGLang @ each dataset x QPS."""
    datasets = [d[0] for d in DATASETS]
    mat = np.full((len(datasets), len(QPS_LIST)), np.nan)
    for ri, ds in enumerate(datasets):
        for ci, q in enumerate(QPS_LIST):
            b = data.get("native_tp1_baseline", {}).get(wl_key(ds, q), {})
            a = data.get("pdaf_tp1_tier", {}).get(wl_key(ds, q), {})
            if b.get("status") == "PASS" and a.get("status") == "PASS":
                mat[ri, ci] = (b["energy_per_token_mj"] - a["energy_per_token_mj"]) / b["energy_per_token_mj"] * 100

    fig, ax = plt.subplots(figsize=(10, 7))
    im = ax.imshow(mat, aspect="auto", cmap="Greens", vmin=0, vmax=70)
    ax.set_xticks(range(len(QPS_LIST)))
    ax.set_xticklabels([str(q) for q in QPS_LIST])
    ax.set_yticks(range(len(datasets)))
    ax.set_yticklabels([d[1] for d in DATASETS])
    ax.set_xlabel("QPS")
    ax.set_ylabel("Dataset")
    ax.set_title("AFlex vs SGLang Energy Saving (%)", fontweight="bold")
    for ri in range(len(datasets)):
        for ci in range(len(QPS_LIST)):
            if not np.isnan(mat[ri, ci]):
                ax.text(ci, ri, f"{mat[ri, ci]:.0f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, label="Saving %")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--minfreq-input", type=Path, default=None)
    parser.add_argument("--biscale-input", type=Path, default=None)
    parser.add_argument("--hetero-biscale-input", type=Path, default=None)
    parser.add_argument("--noradix-input", type=Path, default=None)
    parser.add_argument("--distserve-hetero-input", type=Path, default=None)
    parser.add_argument("--supplement-input", type=Path, default=None)
    parser.add_argument("--no-minfreq", action="store_true")
    parser.add_argument("--no-biscale-paper", action="store_true")
    parser.add_argument("--out-dir", type=Path, default=CHARTS_DIR)
    parser.add_argument("--dataset", type=str, default=None,
                        help="Only plot this dataset (e.g. code)")
    args = parser.parse_args()

    data = load_merged(
        args.input,
        args.minfreq_input,
        args.biscale_input,
        args.hetero_biscale_input,
        args.noradix_input,
        args.distserve_hetero_input,
        args.supplement_input,
        include_minfreq=not args.no_minfreq,
        include_biscale_paper=not args.no_biscale_paper,
    )
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS
    if args.dataset:
        datasets = [d for d in DATASETS if d[0] == args.dataset]
        if not datasets:
            raise SystemExit(f"Unknown dataset: {args.dataset}")

    print("Per-dataset 4-panel dashboards:")
    for ds_info in datasets:
        ds, title = ds_info[0], ds_info[1]
        sub = ds_info[3] if len(ds_info) > 3 else "macro trace"
        plot_dataset_dashboard_4panel(
            data, ds, title, sub,
            out_dir / f"{ds}_dashboard_4panel.png",
        )

    print("Macro 3-panel dashboards (conv/code):")
    for ds, title in [("conv", "Conversation"), ("code", "Code")]:
        plot_dataset_dashboard_3panel(
            data, ds, title, "macro trace",
            out_dir / f"{ds}_dashboard_macro.png",
        )

    print("Micro 3-panel dashboards:")
    for ds_info in DATASETS:
        if ds_info[2] != "micro":
            continue
        ds, title, _, sub = ds_info
        plot_micro_3panel(data, ds, title, sub, out_dir / f"{ds}_dashboard.png")

    print("Overview charts:")
    plot_energy_overview_grid(data, out_dir / "overview_energy_all_datasets.png")
    plot_throughput_overview_grid(data, out_dir / "overview_throughput_all_datasets.png")
    plot_energy_qps16_bars(data, out_dir / "overview_energy_qps16_bars.png")
    plot_savings_heatmap(data, out_dir / "overview_aflex_saving_heatmap.png")

    plot_dataset_dashboard_4panel(
        data, "conv", "Conv Dataset", "macro trace",
        out_dir / "fixed_6scheme_slo5000_300_dashboard.png",
    )

    print(f"\nAll charts saved to: {out_dir}")


if __name__ == "__main__":
    main()
