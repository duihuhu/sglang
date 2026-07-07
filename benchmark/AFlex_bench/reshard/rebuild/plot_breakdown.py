#!/usr/bin/env python3
"""Plot TP=8 cold-start breakdown as horizontal stacked bar (paper figure)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 9,
})

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR / "results"
OUT_DIR = SCRIPT_DIR / "charts"


def load_data() -> dict:
    path = RESULTS_DIR / "startup_breakdown_precise.json"
    with open(path) as f:
        return json.load(f)


def plot_breakdown(data: dict, out_dir: Path) -> None:
    trials = data["trials"]

    # Use average values
    n = len(trials)
    tp_comm = sum(t["tp_comm_group_s"] for t in trials) / n
    weight = sum(t["weight_load_s"] for t in trials) / n
    kv_cache = sum(t["kv_cache_alloc_s"] for t in trials) / n
    cuda_graph_std = sum(t["cuda_graph_standard_s"] for t in trials) / n
    cuda_graph_pw = sum(t["cuda_graph_piecewise_s"] for t in trials) / n
    http_misc = sum(
        t["total_s"] - t["tp_comm_group_s"] - t["materialize_weight_kv_s"] - t["cuda_graph_total_s"]
        for t in trials
    ) / n

    # Define phases for paper figure
    phases = [
        ("Build TP\ncomm group", tp_comm, "#4C78A8"),
        ("Load model\nweights", weight, "#72B7B2"),
        ("Allocate\nKV cache", kv_cache, "#54A24B"),
        ("CUDA graph\ncapture", cuda_graph_std, "#F58518"),
        ("Piecewise\nCUDA graph", cuda_graph_pw, "#E45756"),
        ("HTTP/misc\nstartup", http_misc, "#B279A2"),
    ]

    fig, ax = plt.subplots(figsize=(10, 2.2))

    left = 0.0
    y = 0
    for label, duration, color in phases:
        bar = ax.barh(y, duration, left=left, height=0.55, color=color, label=label, edgecolor="white", linewidth=0.5)
        # Add time annotation
        if duration >= 2.0:
            ax.text(
                left + duration / 2, y,
                f"{duration:.1f}s",
                ha="center", va="center",
                fontsize=9, fontweight="bold",
                color="white" if duration >= 5.0 else "black",
            )
        elif duration >= 0.3:
            ax.text(
                left + duration / 2, y + 0.35,
                f"{duration:.1f}s",
                ha="center", va="bottom",
                fontsize=7, color="#555",
            )
        left += duration

    # Total annotation
    total = sum(d for _, d, _ in phases)
    ax.axvline(total, color="#333", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(total + 0.5, y, f"Total: {total:.1f}s", va="center", fontsize=9, color="#333")

    ax.set_yticks([0])
    ax.set_yticklabels([f"Qwen3-32B\nTP=8"])
    ax.set_xlabel("Time (s)")
    ax.set_title("SGLang Instance Cold-Start Breakdown (8×A800-80GB)", fontsize=11, pad=10)
    ax.set_xlim(0, total + 8)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False, ncols=6, fontsize=8,
        loc="lower center", bbox_to_anchor=(0.45, -0.55),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"startup_breakdown_tp8.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


def plot_breakdown_3trials(data: dict, out_dir: Path) -> None:
    """Plot all 3 trials as separate bars for variance visibility."""
    trials = data["trials"]

    phases_def = [
        ("tp_comm_group_s", "Build TP comm group", "#4C78A8"),
        ("weight_load_s", "Load model weights", "#72B7B2"),
        ("kv_cache_alloc_s", "Allocate KV cache", "#54A24B"),
        ("cuda_graph_standard_s", "CUDA graph capture", "#F58518"),
        ("cuda_graph_piecewise_s", "Piecewise CUDA graph", "#E45756"),
    ]

    fig, ax = plt.subplots(figsize=(10, 3.0))
    y_positions = list(range(len(trials)))
    y_labels = [f"Trial {t['trial']}" for t in trials]

    for yi, trial in enumerate(trials):
        left = 0.0
        for key, label, color in phases_def:
            val = trial[key]
            bar = ax.barh(
                yi, val, left=left, height=0.5,
                color=color, label=label if yi == 0 else None,
                edgecolor="white", linewidth=0.5,
            )
            if val >= 3.0:
                ax.text(
                    left + val / 2, yi,
                    f"{val:.1f}s",
                    ha="center", va="center",
                    fontsize=8, fontweight="bold",
                    color="white" if val >= 8.0 else "black",
                )
            left += val

        # Total annotation at end
        ax.text(left + 0.5, yi, f"{trial['total_s']:.1f}s", va="center", fontsize=8, color="#555")

    ax.set_yticks(y_positions)
    ax.set_yticklabels(y_labels)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_title("SGLang TP=8 Cold-Start Breakdown (Qwen3-32B, 8×A800-80GB)", fontsize=11)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False, ncols=5, fontsize=8,
        loc="lower center", bbox_to_anchor=(0.45, -0.4),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"startup_breakdown_tp8_3trials.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


def main():
    data = load_data()
    plot_breakdown(data, OUT_DIR)
    plot_breakdown_no_cudagraph(OUT_DIR)
    plot_tp4_vs_tp8(OUT_DIR)


def plot_breakdown_no_cudagraph(out_dir: Path) -> None:
    """Plot no-CUDA-graph startup breakdown — fine-grained, elongated."""
    with open(RESULTS_DIR / "startup_breakdown_fine_grained.json") as f:
        data = json.load(f)

    import statistics

    trials = data["trials"]
    tp_comm = statistics.mean([t["tp_comm_group_s"] for t in trials])
    weight = statistics.mean([t["weight_load_s"] for t in trials])
    kv_cache = statistics.mean([t["kv_cache_alloc_s"] for t in trials])
    scheduler = statistics.mean([t["scheduler_init_s"] for t in trials])
    http = statistics.mean([t["http_startup_s"] for t in trials])
    warmup = statistics.mean([t["server_warmup_s"] for t in trials])

    phases = [
        ("Build TP\ncomm group", tp_comm, "#4C78A8"),
        ("Load model\nweights", weight, "#72B7B2"),
        ("Allocate\nKV cache", kv_cache, "#54A24B"),
        ("Init scheduler\n& attn backend", scheduler, "#F58518"),
        ("HTTP server\nstartup", http, "#EECA3B"),
        ("Server warmup\n(1st forward)", warmup, "#E45756"),
    ]

    total = sum(d for _, d, _ in phases)
    fig, ax = plt.subplots(figsize=(14, 1.8))

    left = 0.0
    y = 0
    for label, duration, color in phases:
        ax.barh(
            y, duration, left=left, height=0.55,
            color=color, label=label, edgecolor="white", linewidth=0.5,
        )
        # Annotate
        if duration >= 0.8:
            ax.text(
                left + duration / 2, y,
                f"{duration:.2f}s",
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white" if duration >= 2.5 else "black",
            )
        else:
            # Short phases: annotate above
            ax.text(
                left + duration / 2, y - 0.38,
                f"{duration:.2f}s",
                ha="center", va="top",
                fontsize=8, color="#333",
            )
        left += duration

    ax.axvline(total, color="#333", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.text(total + 0.12, y, f"Total: {total:.1f}s", va="center", fontsize=10, color="#333", fontweight="bold")

    ax.set_yticks([0])
    ax.set_yticklabels([f"Qwen3-32B\nTP=8"], fontsize=10)
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_title("SGLang TP=8 Instance Startup Breakdown — No CUDA Graph (8×A800-80GB)", fontsize=11, pad=10)
    ax.set_xlim(0, total + 2.5)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        frameon=False, ncols=6, fontsize=8.5,
        loc="lower center", bbox_to_anchor=(0.45, -0.62),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"startup_breakdown_no_cudagraph.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


def plot_comparison(out_dir: Path) -> None:
    """Plot side-by-side comparison: with vs without CUDA graph."""
    with open(RESULTS_DIR / "startup_breakdown_precise.json") as f:
        with_cg = json.load(f)
    with open(RESULTS_DIR / "startup_breakdown_no_cudagraph_precise.json") as f:
        no_cg = json.load(f)

    import statistics

    # With CUDA graph
    wcg_trials = with_cg["trials"]
    wcg_tp = statistics.mean([t["tp_comm_group_s"] for t in wcg_trials])
    wcg_weight = statistics.mean([t["weight_load_s"] for t in wcg_trials])
    wcg_kv = statistics.mean([t["kv_cache_alloc_s"] for t in wcg_trials])
    wcg_cg_std = statistics.mean([t["cuda_graph_standard_s"] for t in wcg_trials])
    wcg_cg_pw = statistics.mean([t["cuda_graph_piecewise_s"] for t in wcg_trials])
    wcg_total = statistics.mean([t["total_s"] for t in wcg_trials])
    wcg_http = wcg_total - wcg_tp - wcg_weight - wcg_kv - wcg_cg_std - wcg_cg_pw

    # Without CUDA graph
    ncg_trials = no_cg["trials"]
    ncg_tp = statistics.mean([t["tp_comm_group_s"] for t in ncg_trials])
    ncg_weight = statistics.mean([t["weight_load_s"] for t in ncg_trials])
    ncg_kv = statistics.mean([t["kv_cache_alloc_s"] for t in ncg_trials])
    ncg_total = statistics.mean([t["total_s"] for t in ncg_trials])
    ncg_http = ncg_total - ncg_tp - ncg_weight - ncg_kv

    fig, ax = plt.subplots(figsize=(11, 2.8))

    # Define bars: [label, phases]
    configs = [
        ("TP=8\n(default)", [
            ("Build TP\ncomm group", wcg_tp, "#4C78A8"),
            ("Load weights", wcg_weight, "#72B7B2"),
            ("KV cache", wcg_kv, "#54A24B"),
            ("CUDA graph", wcg_cg_std, "#F58518"),
            ("Piecewise\nCUDA graph", wcg_cg_pw, "#E45756"),
            ("HTTP/misc", wcg_http, "#B279A2"),
        ]),
        ("TP=8\n(no CUDA graph)", [
            ("Build TP\ncomm group", ncg_tp, "#4C78A8"),
            ("Load weights", ncg_weight, "#72B7B2"),
            ("KV cache", ncg_kv, "#54A24B"),
            ("HTTP/misc", ncg_http, "#B279A2"),
        ]),
    ]

    y_positions = [0, 1]
    legend_handles = {}

    for yi, (label, phases) in enumerate(configs):
        left = 0.0
        for phase_label, duration, color in phases:
            bar = ax.barh(
                yi, duration, left=left, height=0.5,
                color=color, edgecolor="white", linewidth=0.5,
            )
            if phase_label not in legend_handles:
                legend_handles[phase_label] = bar[0]
            if duration >= 3.0:
                ax.text(
                    left + duration / 2, yi,
                    f"{duration:.1f}s",
                    ha="center", va="center",
                    fontsize=9, fontweight="bold",
                    color="white" if duration >= 8.0 else "black",
                )
            elif duration >= 0.5:
                ax.text(
                    left + duration / 2, yi + 0.32,
                    f"{duration:.1f}s",
                    ha="center", va="bottom",
                    fontsize=7, color="#555",
                )
            left += duration

        # Total annotation
        ax.text(left + 0.8, yi, f"Total: {left:.1f}s", va="center", fontsize=9, color="#333", fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in configs])
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_title("SGLang TP=8 Instance Startup Breakdown (Qwen3-32B, 8×A800-80GB)", fontsize=11, pad=10)
    ax.set_xlim(0, 78)
    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        legend_handles.values(), legend_handles.keys(),
        frameon=False, ncols=6, fontsize=8,
        loc="lower center", bbox_to_anchor=(0.45, -0.50),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"startup_breakdown_comparison.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


def plot_tp4_vs_tp8(out_dir: Path) -> None:
    """Plot all TP configurations reconfiguration overhead (paper figure).
    Uses warm cache data (realistic reshard scenario where model was recently loaded).
    """
    with open(RESULTS_DIR / "all_tp_cold_warm.json") as f:
        data = json.load(f)

    # Use warm cache (realistic reshard: old instance just ran, page cache hot)
    warm = data["warm"]

    configs = []
    for tp_label in ["TP=1", "TP=2", "TP=4", "TP=8"]:
        d = warm[tp_label]
        configs.append((tp_label, [
            ("Build TP comm group", d["tp_comm_group_s"], "#3B6FA0"),
            ("Materialize weight\n& KV cache", d["weight_load_s"] + d["kv_cache_alloc_s"], "#5BA67E"),
            ("Init inference engine", d["scheduler_init_s"], "#E8853A"),
        ]))

    fig, ax = plt.subplots(figsize=(7, 2.8))

    y_positions = list(range(len(configs)))
    legend_handles = {}

    for yi, (label, phases) in enumerate(configs):
        left = 0.0
        for phase_label, duration, color in phases:
            bar = ax.barh(
                yi, duration, left=left, height=0.50,
                color=color, edgecolor="white", linewidth=0.6,
            )
            if phase_label not in legend_handles:
                legend_handles[phase_label] = bar[0]
            if duration >= 1.0:
                ax.text(
                    left + duration / 2, yi,
                    f"{duration:.1f}s",
                    ha="center", va="center",
                    fontsize=8.5, fontweight="bold",
                    color="white",
                )
            elif duration >= 0.15:
                ax.text(
                    left + duration / 2, yi - 0.34,
                    f"{duration:.2f}s",
                    ha="center", va="top",
                    fontsize=6.5, color="#333",
                )
            left += duration

        ax.text(left + 0.12, yi, f"{left:.1f}s", va="center", fontsize=9,
                color="#333", fontweight="bold")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([c[0] for c in configs], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)", fontsize=10)
    ax.set_xlim(0, 15)
    ax.grid(axis="x", alpha=0.2, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        legend_handles.values(), legend_handles.keys(),
        frameon=False, ncols=3, fontsize=8.5,
        loc="lower center", bbox_to_anchor=(0.45, 1.02),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"startup_breakdown_tp4_vs_tp8.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=300)
        print(f"wrote {path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
