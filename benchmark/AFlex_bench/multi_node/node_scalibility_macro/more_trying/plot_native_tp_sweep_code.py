#!/usr/bin/env python3
"""Plot native TP2/4/8 sweep: SGLang vs DynamoLLM horizontal TP comparison (code)."""
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
TP_LIST = [2, 4, 8]
COLORS = {"SGLang": "#1f77b4", "DynamoLLM": "#aec7e8"}


def load_sweep(path: Path | None) -> dict:
    if path is None:
        files = sorted(RESULTS_DIR.glob("native_tp_sweep_code_final_*.json"))
        if not files:
            raise FileNotFoundError("No native_tp_sweep_code_final_*.json")
        path = files[-1]
    print(f"Loading: {path}")
    return json.loads(path.read_text())["results"]


def _get(results: dict, tp: int, tier: bool, qps: int, metric: str) -> float | None:
    key = f"native_tp{tp}_{'tier' if tier else 'baseline'}"
    m = results.get(key, {}).get(f"code_qps{qps}", {})
    if m.get("status") != "PASS":
        return None
    return m.get(metric)


def plot_tp_bars_at_qps16(results: dict, out: Path):
    """Horizontal TP comparison at QPS16 (bar chart)."""
    metrics = [
        ("energy_per_token_mj", "Energy/Token (J)", 1000.0),
        ("throughput_tok_s", "Throughput (tok/s)", 1.0),
        ("ttft_proc_p50_ms", "TTFT P50 (ms)", 1.0),
        ("tpot_p50_ms", "TPOT P50 (ms)", 1.0),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
    fig.suptitle(
        "Code — SGLang vs DynamoLLM | Native TP Sweep (no radix cache) | QPS=16",
        fontsize=13, fontweight="bold", y=1.02,
    )
    x = np.arange(len(TP_LIST))
    width = 0.35
    qps = 16
    for ax, (metric, ylabel, scale) in zip(axes, metrics):
        sg, dyn = [], []
        for tp in TP_LIST:
            v0 = _get(results, tp, False, qps, metric)
            v1 = _get(results, tp, True, qps, metric)
            sg.append(v0 / scale if v0 is not None else 0)
            dyn.append(v1 / scale if v1 is not None else 0)
        ax.bar(x - width / 2, sg, width, label="SGLang", color=COLORS["SGLang"])
        ax.bar(x + width / 2, dyn, width, label="DynamoLLM", color=COLORS["DynamoLLM"])
        ax.set_xticks(x)
        ax.set_xticklabels([f"TP{t}" for t in TP_LIST])
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].legend(fontsize=9)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def plot_dashboard_4panel(results: dict, out: Path):
    """Energy / throughput / latency vs QPS, one line pair per TP."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Code — Native TP Sweep (no radix cache) | SGLang (solid) vs DynamoLLM (dashed)\n"
        "SLO: TTFT=5s, TPOT=300ms",
        fontsize=12, fontweight="bold", y=0.98,
    )

    panels = [
        (0, 0, "energy_per_token_mj", 1000.0, "Energy/Token (J)"),
        (0, 1, "throughput_tok_s", 1.0, "Throughput (tok/s)"),
        (1, 0, "tpot_p50_ms", 1.0, "TPOT P50 (ms)"),
        (1, 1, "ttft_proc_p50_ms", 1.0, "TTFT P50 (ms)"),
    ]
    tp_colors = {2: "#1f77b4", 4: "#ff7f0e", 8: "#2ca02c"}
    for r, c, metric, scale, ylabel in panels:
        ax = axes[r, c]
        for tp in TP_LIST:
            color = tp_colors[tp]
            for tier, ls, prefix in [(False, "-", "SGLang"), (True, "--", "DynamoLLM")]:
                ys = []
                xs = []
                for q in QPS_LIST:
                    v = _get(results, tp, tier, q, metric)
                    if v is not None:
                        xs.append(q)
                        ys.append(v / scale)
                if xs:
                    ax.plot(xs, ys, color=color, linestyle=ls, marker="o", linewidth=2,
                            markersize=6, label=f"{prefix} TP{tp}")
        ax.set_xlabel("QPS")
        ax.set_ylabel(ylabel)
        ax.set_xticks(QPS_LIST)
        ax.grid(True, alpha=0.3)
        ax.set_title(ylabel, fontweight="bold")
    axes[0, 0].legend(fontsize=7, loc="upper right", ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {out.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=CHARTS_DIR)
    args = parser.parse_args()

    results = load_sweep(args.input)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_tp_bars_at_qps16(results, args.out_dir / "code_native_tp_compare_qps16.png")
    plot_dashboard_4panel(results, args.out_dir / "code_native_tp_sweep_4panel.png")


if __name__ == "__main__":
    main()
