#!/usr/bin/env python3
"""Plot best-PDAF 6-scheme benchmark (4 datasets × 6 QPS)."""
from __future__ import annotations

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
SCHEMES = [
    ("native_tp1_baseline", "SGLang", "#1f77b4", "-"),
    ("native_tp1_tier", "DynamoLLM", "#aec7e8", "--"),
    ("pd_hetero_baseline", "DistServe", "#ff7f0e", "-"),
    ("pd_hetero_tier_biscale", "BiScale", "#ffbb78", "--"),
    ("pdaf_best_baseline", "MegaScale", "#2ca02c", "-"),
    ("pdaf_best_tier", "AFlex", "#d62728", "--"),
]
DATASETS = [
    ("code", "Code"),
    ("conv", "Conversation"),
    ("balanced_mpmd", "Balanced"),
    ("summary_hphd", "Summary"),
]


def load() -> tuple[dict, dict]:
    files = sorted(RESULTS_DIR.glob("best_pdaf_6scheme_final_*.json"))
    if not files:
        raise FileNotFoundError("No best_pdaf_6scheme_final_*.json")
    p = files[-1]
    print(f"Loading {p.name}")
    payload = json.loads(p.read_text())
    return payload["results"], payload.get("meta", {})


def _get(results, scheme, ds, qps, field):
    m = results.get(scheme, {}).get(f"{ds}_qps{qps}", {})
    if m.get("status") != "PASS":
        return None
    return m.get(field)


def plot_dataset_dashboard(results: dict, meta: dict):
    best = meta.get("best_pdaf_deploy", {}).get("scheme_key", "?")
    for ds, title in DATASETS:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"{title} — 6-Scheme (best PDAF: {best})\n"
            "SLO: TTFT=5s, TPOT=300ms | no radix cache",
            fontsize=12, fontweight="bold", y=0.98,
        )
        panels = [
            (0, 0, "energy_per_token_mj", 1000.0, "Energy/Token (J)"),
            (0, 1, "throughput_tok_s", 1.0, "Throughput (tok/s)"),
            (1, 0, "tpot_p50_ms", 1.0, "TPOT P50 (ms)"),
            (1, 1, "ttft_proc_p50_ms", 1.0, "TTFT P50 (ms)"),
        ]
        for r, c, field, scale, ylab in panels:
            ax = axes[r, c]
            for sk, label, color, ls in SCHEMES:
                xs, ys = [], []
                for q in QPS_LIST:
                    v = _get(results, sk, ds, q, field)
                    if v is not None:
                        xs.append(q)
                        ys.append(v / scale)
                if xs:
                    ax.plot(xs, ys, color=color, linestyle=ls, marker="o",
                            linewidth=2, label=label)
            ax.set_xlabel("QPS")
            ax.set_ylabel(ylab)
            ax.set_xticks(QPS_LIST)
            ax.grid(True, alpha=0.3)
            ax.set_title(ylab, fontweight="bold")
        axes[0, 0].legend(fontsize=8, loc="best")
        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out = CHARTS_DIR / f"best_pdaf_6scheme_{ds}_4panel.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  {out.name}")


def plot_qps16_bars(results: dict, meta: dict):
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [s[1] for s in SCHEMES]
    x = np.arange(len(SCHEMES))
    width = 0.2
    colors_ds = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (ds, dtitle) in enumerate(DATASETS):
        vals = []
        for sk, _, _, _ in SCHEMES:
            v = _get(results, sk, ds, 16, "energy_per_token_mj")
            vals.append(v / 1000.0 if v else 0)
        ax.bar(x + (i - 1.5) * width, vals, width, label=dtitle, color=colors_ds[i])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Energy/Token (J)")
    best = meta.get("best_pdaf_deploy", {}).get("scheme_key", "")
    ax.set_title(f"QPS=16 Energy — best PDAF: {best}", fontweight="bold")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    out = CHARTS_DIR / "best_pdaf_6scheme_energy_qps16_bars.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    results, meta = load()
    plot_dataset_dashboard(results, meta)
    plot_qps16_bars(results, meta)
    print(f"Charts -> {CHARTS_DIR}/")


if __name__ == "__main__":
    main()
