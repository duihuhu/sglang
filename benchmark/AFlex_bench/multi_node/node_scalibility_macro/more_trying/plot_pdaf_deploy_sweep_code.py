#!/usr/bin/env python3
"""Plot PDAF deployment-class sweep on code (QPS 2/8/16)."""
from __future__ import annotations

import argparse
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

QPS_LIST = [2, 8, 16]
CAT_COLORS = {"c1": "#1f77b4", "c2": "#ff7f0e", "c3": "#2ca02c"}
TIER_COLOR = {"baseline": "#aec7e8", "tier": "#d62728"}


def _latest(pattern: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(pattern))
    return files[-1] if files else None


def load_results(sweep_path: Path | None, c3_path: Path | None) -> dict:
    if sweep_path is None:
        sweep_path = _latest("pdaf_deploy_sweep_code_final_*.json")
        if sweep_path is None:
            sweep_path = _latest("pdaf_deploy_sweep_code_partial_*.json")
    if sweep_path is None:
        raise FileNotFoundError("No pdaf_deploy_sweep_code_*.json")

    data = json.loads(sweep_path.read_text())
    results = dict(data.get("results", {}))
    print(f"Sweep: {sweep_path.name} ({len(results)} schemes)")

    if c3_path is None:
        c3_path = RESULTS_DIR / "pdaf_code_final.json"
    if c3_path.exists():
        c3 = json.loads(c3_path.read_text()).get("results", {})
        remap = {
            "pdaf_tp1_baseline": "pdaf_c3_xnode_tp1x4_baseline",
            "pdaf_tp1_tier": "pdaf_c3_xnode_tp1x4_tier",
        }
        for old, new in remap.items():
            if old in c3 and new not in results:
                wl = {k: v for k, v in c3[old].items()
                      if k.startswith("code_qps") and int(k.replace("code_qps", "")) in QPS_LIST}
                if wl:
                    results[new] = wl
                    print(f"  merged C3 from {c3_path.name}: {new} ({len(wl)} pts)")
    return results


def _parse_scheme(key: str) -> tuple[str, str, bool]:
    """Return (category, short_label, is_tier)."""
    tier = key.endswith("_tier")
    if key.startswith("pdaf_c3_"):
        return "c3", "C3 xnode 4×TP1", tier
    m = re.match(r"pdaf_c2_intra_(.+)_(baseline|tier)$", key)
    if m:
        return "c2", f"C2 intra {m.group(1)}", tier
    m = re.match(r"pdaf_c1_xnode_p(.+)_d(.+)_(baseline|tier)$", key)
    if m:
        return "c1", f"C1 P={m.group(1)} D={m.group(2)}", tier
    return "?", key, tier


def _metric(results: dict, scheme: str, qps: int, field: str) -> float | None:
    m = results.get(scheme, {}).get(f"code_qps{qps}", {})
    if not isinstance(m, dict) or m.get("status") != "PASS":
        return None
    return m.get(field)


def _rank_schemes(results: dict, qps: int = 16, tier_only: bool = True) -> list[tuple[str, float]]:
    ranked = []
    for key in results:
        if key.startswith("__") or isinstance(results[key], str):
            continue
        if tier_only and not key.endswith("_tier"):
            continue
        v = _metric(results, key, qps, "energy_per_token_mj")
        if v is not None:
            ranked.append((key, v / 1000.0))
    ranked.sort(key=lambda x: x[1])
    return ranked


def plot_top_energy_bar(results: dict, out: Path, top_n: int = 12, qps: int = 16):
    ranked = _rank_schemes(results, qps=qps, tier_only=True)
    if not ranked:
        ranked = _rank_schemes(results, qps=qps, tier_only=False)
    ranked = ranked[:top_n]

    labels, vals, colors = [], [], []
    for key, e in ranked:
        cat, short, _ = _parse_scheme(key)
        labels.append(short.replace("C1 ", "").replace("C2 ", "").replace("C3 ", "C3 "))
        vals.append(e)
        colors.append(CAT_COLORS.get(cat, "#888"))

    fig, ax = plt.subplots(figsize=(14, max(5, 0.4 * len(labels) + 2)))
    y = np.arange(len(labels))
    ax.barh(y, vals, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Energy/Token (J)")
    ax.set_title(
        f"Code — PDAF Deploy Sweep (AFlex tier) | QPS={qps}\n"
        "Lower is better | SLO: TTFT=5s, TPOT=300ms",
        fontweight="bold",
    )
    ax.grid(True, axis="x", alpha=0.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=c, label=l) for l, c in
                       [("C1 xnode PD", CAT_COLORS["c1"]),
                        ("C2 intra", CAT_COLORS["c2"]),
                        ("C3 xnode TP1×4", CAT_COLORS["c3"])]],
              loc="lower right", fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_category_best_lines(results: dict, out: Path):
    """Best tier config per category vs QPS."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "Code — Best PDAF Deploy per Class (AFlex tier)\n"
        "SLO: TTFT=5s, TPOT=300ms",
        fontsize=12, fontweight="bold", y=0.98,
    )
    panels = [
        (0, 0, "energy_per_token_mj", 1000.0, "Energy/Token (J)"),
        (0, 1, "throughput_tok_s", 1.0, "Throughput (tok/s)"),
        (1, 0, "tpot_p50_ms", 1.0, "TPOT P50 (ms)"),
        (1, 1, "ttft_proc_p50_ms", 1.0, "TTFT P50 (ms)"),
    ]

    best_per_cat: dict[str, str] = {}
    for cat in ("c1", "c2", "c3"):
        candidates = []
        for key in results:
            if not key.endswith("_tier"):
                continue
            c, _, _ = _parse_scheme(key)
            if c != cat:
                continue
            v = _metric(results, key, 16, "energy_per_token_mj")
            if v is not None:
                candidates.append((key, v))
        if candidates:
            best_per_cat[cat] = min(candidates, key=lambda x: x[1])[0]

    for r, c, field, scale, ylabel in panels:
        ax = axes[r, c]
        for cat, key in best_per_cat.items():
            xs, ys = [], []
            for q in QPS_LIST:
                v = _metric(results, key, q, field)
                if v is not None:
                    xs.append(q)
                    ys.append(v / scale)
            if xs:
                _, short, _ = _parse_scheme(key)
                ax.plot(xs, ys, color=CAT_COLORS[cat], marker="o", linewidth=2,
                        label=f"{cat.upper()}: {short}")
        ax.set_xlabel("QPS")
        ax.set_ylabel(ylabel)
        ax.set_xticks(QPS_LIST)
        ax.grid(True, alpha=0.3)
        ax.set_title(ylabel, fontweight="bold")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_c1_heatmap(results: dict, out: Path, qps: int = 16):
    layouts = ("tp4x1", "tp2x2", "tp1x4")
    mat = np.full((3, 3), np.nan)
    for i, p in enumerate(layouts):
        for j, d in enumerate(layouts):
            key = f"pdaf_c1_xnode_p{p}_d{d}_tier"
            v = _metric(results, key, qps, "energy_per_token_mj")
            if v is not None:
                mat[i, j] = v / 1000.0

    if np.all(np.isnan(mat)):
        print("  skip c1 heatmap (no tier data)")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels([f"D={x}" for x in layouts])
    ax.set_yticklabels([f"P={x}" for x in layouts])
    for i in range(3):
        for j in range(3):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                        color="black", fontsize=10, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Energy/Token (J)")
    ax.set_title(f"C1 xnode PD — AFlex tier Energy Heatmap | QPS={qps}", fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def print_summary(results: dict, qps: int = 16):
    ranked = _rank_schemes(results, qps=qps, tier_only=True)
    print(f"\n=== Top 5 AFlex @ QPS={qps} (energy/token J) ===")
    for i, (key, e) in enumerate(ranked[:5], 1):
        cat, short, _ = _parse_scheme(key)
        thr = _metric(results, key, qps, "throughput_tok_s")
        print(f"  {i}. [{cat}] {short}: {e:.3f} J/tok, thr={thr:.1f}" if thr else
              f"  {i}. [{cat}] {short}: {e:.3f} J/tok")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--c3-input", type=Path, default=None)
    args = parser.parse_args()

    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results(args.input, args.c3_input)
    print_summary(results)

    plot_top_energy_bar(results, CHARTS_DIR / "pdaf_deploy_sweep_code_top_energy_qps16.png")
    plot_category_best_lines(results, CHARTS_DIR / "pdaf_deploy_sweep_code_category_best_4panel.png")
    plot_c1_heatmap(results, CHARTS_DIR / "pdaf_deploy_sweep_code_c1_heatmap_qps16.png")
    print(f"Charts -> {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
