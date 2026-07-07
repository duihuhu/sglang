#!/usr/bin/env python3
"""Plot QPS workload request timeline during TP scaling."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = REPO_ROOT / "benchmark/AFlex_bench/reshard/Baseline/results/node1_tp2_to_tp4_ipc_qps1_timeline.json"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark/AFlex_bench/reshard/Baseline/charts"

ROUTE_COLORS = {
    "source": "#4C78A8",
    "target": "#54A24B",
    "cutover": "#E45756",
}


def event_time(record: dict[str, Any], name: str) -> float | None:
    for event in record.get("events", []):
        if event.get("name") == name:
            return float(event["time_s"])
    return None


def plot_timeline(record: dict[str, Any], out_dir: Path) -> None:
    requests = record.get("requests", [])
    workload_start = event_time(record, "workload_start") or 0.0
    scale_start = event_time(record, "scale_start")
    target_ready = event_time(record, "target_ready")
    cutover_start = event_time(record, "cutover_start")
    cutover_end = event_time(record, "cutover_end")
    workload_done = event_time(record, "workload_done") or max(r.get("end_s", 0) for r in requests)

    fig, (ax_req, ax_evt) = plt.subplots(
        2,
        1,
        figsize=(13.5, 6.0),
        sharex=True,
        gridspec_kw={"height_ratios": [4.5, 1.0], "hspace": 0.08},
    )

    for req in requests:
        rid = int(req["request_id"])
        start = float(req["start_s"]) - workload_start
        end = float(req["end_s"]) - workload_start
        route = req.get("route", "unknown")
        success = bool(req.get("success"))
        color = ROUTE_COLORS.get(route, "#999999")
        if success:
            ax_req.barh(rid, max(end - start, 0.02), left=start, height=0.68, color=color, alpha=0.85)
        else:
            ax_req.scatter(start, rid, marker="x", s=55, color=ROUTE_COLORS["cutover"], linewidths=2.0, zorder=5)

    ax_req.set_ylabel("Request ID")
    ax_req.set_title("QPS=1 Requests During TP2→TP4 IPC/NVLink Scaling")
    ax_req.grid(axis="x", alpha=0.25)
    ax_req.set_ylim(len(requests), -1)

    def rel(t: float | None) -> float | None:
        return None if t is None else t - workload_start

    scale_rel = rel(scale_start)
    ready_rel = rel(target_ready)
    cut_s_rel = rel(cutover_start)
    cut_e_rel = rel(cutover_end)
    done_rel = workload_done - workload_start

    if scale_rel is not None:
        ax_req.axvline(scale_rel, color="#F58518", linestyle="--", linewidth=1.5)
    if ready_rel is not None:
        ax_req.axvline(ready_rel, color="#54A24B", linestyle="--", linewidth=1.5)
    if cut_s_rel is not None and cut_e_rel is not None:
        ax_req.axvspan(cut_s_rel, cut_e_rel, color="#E45756", alpha=0.15)
        ax_req.text((cut_s_rel + cut_e_rel) / 2, -3, f"cutover {cut_e_rel - cut_s_rel:.2f}s", ha="center", va="bottom", fontsize=9, color="#B00020")

    # Event lane
    ax_evt.set_yticks([])
    ax_evt.set_xlabel("Time since workload start (s)")
    ax_evt.set_xlim(0, done_rel + 2)
    ax_evt.grid(axis="x", alpha=0.25)
    event_specs = [
        ("scale start", scale_rel, "#F58518"),
        ("target ready", ready_rel, "#54A24B"),
        ("cutover start", cut_s_rel, "#E45756"),
        ("cutover end", cut_e_rel, "#E45756"),
    ]
    for label, t, color in event_specs:
        if t is None:
            continue
        ax_evt.axvline(t, color=color, linestyle="--", linewidth=1.4)
        ax_evt.text(t, 0.25, label, rotation=35, ha="right", va="bottom", fontsize=8, color=color)
    if scale_rel is not None and ready_rel is not None:
        ax_evt.barh(0, ready_rel - scale_rel, left=scale_rel, height=0.2, color="#4C78A8", alpha=0.35)
        ax_evt.text((scale_rel + ready_rel) / 2, -0.18, f"async prepare {ready_rel - scale_rel:.1f}s", ha="center", va="top", fontsize=9)
    if cut_s_rel is not None and cut_e_rel is not None:
        ax_evt.barh(0, cut_e_rel - cut_s_rel, left=cut_s_rel, height=0.35, color="#E45756", alpha=0.55)

    summary = record.get("summary", {})
    subtitle = (
        f"success={summary.get('success')}/{summary.get('total')}, "
        f"failed={summary.get('failed')}, "
        f"avg TTFT={summary.get('ttft_avg_ms', 0):.1f} ms"
    )
    ax_req.text(0.01, 0.98, subtitle, transform=ax_req.transAxes, ha="left", va="top", fontsize=10,
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "none"})

    legend = [
        Patch(facecolor=ROUTE_COLORS["source"], label="source TP2 success"),
        Patch(facecolor=ROUTE_COLORS["target"], label="target TP4 success"),
        Patch(facecolor=ROUTE_COLORS["cutover"], alpha=0.25, label="cutover unavailable"),
    ]
    ax_req.legend(handles=legend, frameon=False, loc="lower right")

    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"qps1_tp_scaling_timeline.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=220)
        print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    record = json.loads(args.input.read_text(encoding="utf-8"))
    plot_timeline(record, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
