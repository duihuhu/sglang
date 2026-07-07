#!/usr/bin/env python3
"""Per-dataset macro dashboard: Energy, TTFT, TPOT, Frequency CDF.

Generates one 2x2 figure per dataset (conv, code) with:
  (a) Energy/Token vs QPS
  (b) TTFT vs QPS (P50 / P95 / P99)
  (c) TPOT vs QPS (P50 / P95 / P99)
  (d) Frequency selection CDF (baseline locked at 1410 MHz; tier from DVFS logs)

Usage:
    python3 plot_macro_dataset_dashboard.py
"""
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
FREQ_DIR = RESULTS_DIR / "quantify_pdaf_tier"

QPS_LIST = [2, 4, 6, 8, 12, 16]
MAX_GPU_FREQ = 1410

SCHEME_ORDER = [
    "native_tp2_baseline",
    "native_tp2_tier",
    "pd_dp_baseline",
    "pd_dp_tier",
    "pdaf_baseline",
    "pdaf_tier",
]
SCHEME_LABELS = {
    "native_tp2_baseline": "SGLang",
    "native_tp2_tier": "DynamoLLM",
    "pd_dp_baseline": "DistServe",
    "pd_dp_tier": "BiScale",
    "pdaf_baseline": "MegaScale",
    "pdaf_tier": "AFlex",
}
COLORS = ["#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c", "#98df8a"]
MARKERS = ["o", "s", "^", "v", "D", "P"]
STYLES = ["-", "--", "-", "--", "-", "--"]

DATASETS = [
    ("conv", "Conversation"),
    ("code", "Code"),
]

# DVFS jsonl directories keyed by (scheme, dataset)
FREQ_LOG_DIRS: dict[tuple[str, str], list[Path]] = {
    ("pdaf_tier", "conv"): [
        FREQ_DIR / "session_conv_xnode",
        FREQ_DIR / "session_conv",
        FREQ_DIR / "qps1_conv",
    ],
}


def load_results(prefix: str = "scal_16card_") -> dict:
    merged: dict[str, dict] = {}
    for f in sorted(RESULTS_DIR.glob(f"{prefix}*.json")):
        data = json.load(open(f))
        for scheme, entries in data.get("results", {}).items():
            sname = scheme.replace("native_tp_", "native_tp2_")
            merged.setdefault(sname, {}).update(entries)
    return merged


def get_metric(data: dict, scheme: str, dataset: str, qps: int, metric: str):
    key = f"{dataset}_qps{qps}"
    m = data.get(scheme, {}).get(key, {})
    if isinstance(m, dict) and m.get("status") == "PASS":
        return m.get(metric)
    return None


def interp_px(p50, p99, pct):
    if p50 is None or p99 is None:
        return None
    return p50 + (pct - 50) / (99 - 50) * (p99 - p50)


def get_series(data: dict, scheme: str, dataset: str, metric: str) -> dict[int, float]:
    pts = {}
    for q in QPS_LIST:
        v = get_metric(data, scheme, dataset, q, metric)
        if v is not None:
            pts[q] = v
    return pts


def parse_freq_logs(log_dirs: list[Path]) -> list[int]:
    freqs: list[int] = []
    keys = ("sel_f_a", "sel_f_f", "f_a", "f_f", "cur_f_a", "cur_f_f")
    for log_dir in log_dirs:
        if not log_dir.exists():
            continue
        for path in sorted(log_dir.glob("dvfs_decisions_*.jsonl")):
            for line in path.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                for key in keys:
                    val = rec.get(key)
                    if val:
                        freqs.append(int(val))
    return freqs


def load_freq_samples(scheme: str, dataset: str) -> list[int] | None:
    if scheme.endswith("_baseline"):
        return [MAX_GPU_FREQ] * 200

    log_dirs = FREQ_LOG_DIRS.get((scheme, dataset))
    if not log_dirs:
        return None
    freqs = parse_freq_logs(log_dirs)
    return freqs or None


def plot_cdf(ax, freqs: list[int], label: str, color: str, linestyle: str = "-"):
    xs = np.sort(freqs)
    ys = np.arange(1, len(xs) + 1) / len(xs)
    ax.step(xs, ys, where="post", label=label, color=color, linestyle=linestyle, linewidth=2)


def plot_energy_panel(ax, data: dict, dataset: str):
    for i, scheme in enumerate(SCHEME_ORDER):
        if scheme not in data:
            continue
        pts = get_series(data, scheme, dataset, "energy_per_token_mj")
        if not pts:
            continue
        xs = sorted(pts)
        ys = [pts[q] / 1000.0 for q in xs]
        ax.plot(
            xs, ys, color=COLORS[i], linestyle=STYLES[i], marker=MARKERS[i],
            markersize=6, linewidth=2.0, label=SCHEME_LABELS[scheme],
        )
    ax.set_xlabel("QPS")
    ax.set_ylabel("Energy/Token (J)")
    ax.set_title("(a) Energy per Token")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")


def plot_latency_panel(ax, data: dict, dataset: str, prefix: str, panel_id: str, title: str):
    pct_styles = [(50, "-", 2.0, 6), (95, "--", 1.6, 5), (99, ":", 1.4, 4)]
    legend_handles = []

    for i, scheme in enumerate(SCHEME_ORDER):
        if scheme not in data:
            continue
        p50_pts = get_series(data, scheme, dataset, f"{prefix}_p50_ms")
        p99_pts = get_series(data, scheme, dataset, f"{prefix}_p99_ms")
        if not p50_pts:
            continue
        xs = sorted(p50_pts)
        color = COLORS[i]
        first = True
        for pct, ls, lw, ms in pct_styles:
            valid_x, valid_y = [], []
            for q in xs:
                y = interp_px(p50_pts[q], p99_pts.get(q), pct)
                if y is not None:
                    valid_x.append(q)
                    valid_y.append(y)
            if not valid_y:
                continue
            use_marker = pct == 50
            line, = ax.plot(
                valid_x, valid_y, color=color, linestyle=ls, linewidth=lw,
                marker=MARKERS[i] if use_marker else None,
                markersize=ms if use_marker else 0,
                label=SCHEME_LABELS[scheme] if first else "_nolegend_",
            )
            if first:
                legend_handles.append(line)
                first = False

    ax.set_xlabel("QPS")
    ax.set_ylabel(f"{title} (ms)")
    ax.set_title(f"({panel_id}) {title} — solid P50, dashed P95, dotted P99")
    ax.set_xticks(QPS_LIST)
    ax.grid(True, alpha=0.3)
    ax.legend(handles=legend_handles, fontsize=8, loc="upper left", ncol=2)


def plot_freq_cdf_panel(ax, dataset: str):
    plotted = 0
    missing = []
    for i, scheme in enumerate(SCHEME_ORDER):
        freqs = load_freq_samples(scheme, dataset)
        if freqs is None:
            if scheme.endswith("_tier"):
                missing.append(SCHEME_LABELS[scheme])
            continue
        plot_cdf(ax, freqs, SCHEME_LABELS[scheme], COLORS[i], STYLES[i])
        plotted += 1

    ax.set_xlabel("Selected Frequency (MHz)")
    ax.set_ylabel("CDF")
    ax.set_title("(d) Frequency Selection CDF")
    ax.set_xlim(0, MAX_GPU_FREQ + 120)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    note_lines = []
    if missing:
        note_lines.append("Tier logs unavailable: " + ", ".join(missing))
    note_lines.append("Baseline schemes locked at 1410 MHz.")
    if plotted <= 4:
        note_lines.append("PDAF+Tier conv uses quantify DVFS logs.")
    ax.text(
        0.02, 0.02, "\n".join(note_lines), transform=ax.transAxes, fontsize=7.5,
        va="bottom", ha="left",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffde7", alpha=0.9, edgecolor="#ccc"),
    )


def plot_dataset_dashboard(data: dict, dataset: str, title: str, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        f"16-Card Macro Dashboard — {title} | Qwen3-32B 2-Node",
        fontsize=15, fontweight="bold", y=0.98,
    )

    plot_energy_panel(axes[0, 0], data, dataset)
    plot_latency_panel(axes[0, 1], data, dataset, "ttft_proc", "b", "TTFT")
    plot_latency_panel(axes[1, 0], data, dataset, "tpot", "c", "TPOT")
    plot_freq_cdf_panel(axes[1, 1], dataset)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_results()
    for dataset, title in DATASETS:
        out_path = CHARTS_DIR / f"16card_{dataset}_dashboard.png"
        plot_dataset_dashboard(data, dataset, title, out_path)


if __name__ == "__main__":
    main()
