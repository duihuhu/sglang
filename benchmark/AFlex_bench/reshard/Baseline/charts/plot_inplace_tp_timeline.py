#!/usr/bin/env python3
"""Plot in-place TP1→TP8 QPS timeline results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt


COLORS = {1: "#4C78A8", 2: "#72B7B2", 4: "#54A24B", 8: "#F58518"}


def save(fig, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=180, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")


def plot(data: Dict[str, Any], out_dir: Path, stem: str) -> None:
    requests: List[Dict[str, Any]] = data["requests"]
    events: List[Dict[str, Any]] = data["events"]

    fig, ax = plt.subplots(figsize=(15, 7))

    # Requests: one horizontal micro-bar per request.
    for req in requests:
        y = 2 + (req["request_id"] % 20) * 0.12
        start = req["start_s"]
        width = max(req["end_s"] - req["start_s"], 0.03)
        if req["status"] == "ok":
            color = COLORS.get(int(req["tp_size"]), "#999999")
            ax.barh(y, width, left=start, height=0.07, color=color, alpha=0.75)
        else:
            ax.scatter(start, y, marker="x", s=45, color="#E45756", linewidth=1.8)

    # TP phase bars.
    phase_start = 0.0
    for ev in events:
        old_tp = int(ev["old_tp"])
        ax.barh(0.65, ev["pause_start_s"] - phase_start, left=phase_start, height=0.32, color=COLORS[old_tp], alpha=0.9)
        ax.text((phase_start + ev["pause_start_s"]) / 2, 0.65, f"TP{old_tp}", ha="center", va="center", color="white", fontsize=9)
        phase_start = ev["resume_s"]
    if requests:
        end_t = max(r["end_s"] for r in requests)
    else:
        end_t = max(e["resume_s"] for e in events)
    ax.barh(0.65, max(0, end_t - phase_start), left=phase_start, height=0.32, color=COLORS[8], alpha=0.9)
    ax.text((phase_start + end_t) / 2, 0.65, "TP8", ha="center", va="center", color="white", fontsize=9)

    # Reshard windows.
    for ev in events:
        pause = ev["pause_start_s"]
        resume = ev["resume_s"]
        width = max(resume - pause, 0.02)
        ax.axvline(pause, color="#222222", linestyle="--", linewidth=0.9, alpha=0.65)
        ax.axvline(resume, color="#222222", linestyle=":", linewidth=0.9, alpha=0.65)
        ax.barh(1.25, width, left=pause, height=0.28, color="#E45756", alpha=0.55)
        label = f"TP{ev['old_tp']}→TP{ev['new_tp']}\n{width:.2f}s, {ev['bandwidth_gbps']:.1f} GB/s"
        ax.text(pause + width / 2, 1.25, label, ha="center", va="center", fontsize=8)

    summary = data.get("summary", {})
    title = (
        "In-place TP Reshard Transfer Microbench "
        f"(fail={summary.get('failed_requests', 0)}/{summary.get('total_requests', 0)}, "
        f"synthetic pause={summary.get('visible_pause_s', 0):.2f}s)"
    )
    ax.set_title(title)
    ax.set_xlabel("Time since workload start (s)")
    ax.set_yticks([0.65, 1.25, 2.6, 3.8], ["Serving TP", "Reshard window", "Requests", ""])
    ax.set_ylim(0.2, 4.8)
    ax.grid(axis="x", alpha=0.25)

    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[tp], alpha=0.75, label=f"TP{tp} request") for tp in [1, 2, 4, 8]]
    handles.append(plt.Line2D([0], [0], marker="x", color="#E45756", linestyle="None", label="failed during pause"))
    ax.legend(handles=handles, frameon=False, ncols=5, loc="lower center", bbox_to_anchor=(0.5, -0.18))

    save(fig, out_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--stem", default="inplace_tp1_to_tp8_qps_timeline")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    plot(data, args.out_dir, args.stem)


if __name__ == "__main__":
    main()
