#!/usr/bin/env python3
"""Plot real-workload in-place reshard timeline (bench_inplace_reshard_real_workload.py output)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = {1: "#4C78A8", 2: "#72B7B2", 4: "#54A24B", 8: "#F58518"}


def save(fig, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def plot(data: dict[str, Any], out_dir: Path, stem: str) -> tuple[Path, Path]:
    summary = data["summary"]
    timeline = data["timeline"]
    events = summary.get("reshard_events", [])
    downtime = summary.get("slo_downtime_windows_merged") or summary.get(
        "slo_downtime_windows", []
    )
    span = max((r["issue_s"] for r in timeline), default=0.0)

    fig, (ax_phase, ax_req) = plt.subplots(
        2,
        1,
        figsize=(14.5, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.1, 3.2], "hspace": 0.06},
    )

    # --- Top: TP serving phases + reshard pause windows ---
    phase_end = 0.0
    for ev in events:
        old_tp = 1 if ev["step"] == 0 else events[ev["step"] - 1]["new_tp"]
        pause_start = float(ev["trigger_s"])
        pause_end = float(ev["done_s"]) if ev.get("done_s") is not None else pause_start + float(ev.get("pause_s") or 1.0)
        if pause_start > phase_end:
            ax_phase.barh(
                0.5,
                pause_start - phase_end,
                left=phase_end,
                height=0.55,
                color=COLORS.get(old_tp, "#999999"),
                alpha=0.92,
            )
            ax_phase.text(
                (phase_end + pause_start) / 2,
                0.5,
                f"TP{old_tp}",
                ha="center",
                va="center",
                color="white",
                fontsize=10,
                fontweight="bold",
            )
        width = max(pause_end - pause_start, 0.05)
        ax_phase.barh(0.5, width, left=pause_start, height=0.55, color="#E45756", alpha=0.65)
        pause_label = ev.get("pause_s")
        pause_txt = f"{pause_label:.1f}s" if pause_label is not None else "?"
        ax_phase.text(
            pause_start + width / 2,
            0.5,
            f"TP{old_tp}→TP{ev['new_tp']}\n{pause_txt}",
            ha="center",
            va="center",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
        ax_phase.axvline(pause_start, color="#333333", linestyle="--", linewidth=0.8, alpha=0.5)
        phase_end = pause_end

    final_tp = events[-1]["new_tp"] if events else 1
    if span > phase_end:
        ax_phase.barh(
            0.5,
            span - phase_end,
            left=phase_end,
            height=0.55,
            color=COLORS.get(final_tp, "#999999"),
            alpha=0.92,
        )
        ax_phase.text(
            (phase_end + span) / 2,
            0.5,
            f"TP{final_tp}",
            ha="center",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )

    ax_phase.set_ylim(0.05, 1.05)
    ax_phase.set_yticks([])
    ax_phase.set_ylabel("Serving TP")
    ax_phase.grid(axis="x", alpha=0.25)

    # --- Bottom: per-request bars (issue → issue+e2e) ---
    for i, req in enumerate(timeline):
        start = float(req["issue_s"])
        e2e = float(req.get("e2e_s") or 0.03)
        width = max(e2e, 0.04)
        tp = int(req["tp_at_issue"])
        y = i
        slo_ok = req.get("slo_pass", req.get("ok"))
        if slo_ok:
            ax_req.barh(
                y,
                width,
                left=start,
                height=0.72,
                color=COLORS.get(tp, "#999999"),
                alpha=0.82,
            )
        elif req.get("ok"):
            ax_req.barh(
                y,
                width,
                left=start,
                height=0.72,
                color=COLORS.get(tp, "#999999"),
                alpha=0.35,
                edgecolor="#B00020",
                linewidth=0.8,
            )
        else:
            ax_req.scatter(start, y, marker="x", s=50, color="#B00020", linewidths=2, zorder=5)

    for w in downtime:
        ax_req.axvspan(
            float(w["start_s"]),
            float(w["end_s"]),
            color="#7B1E1E",
            alpha=0.10,
            zorder=0,
        )

    n_ok = sum(1 for r in timeline if r.get("ok"))
    n_slo = summary.get("n_slo_pass", sum(1 for r in timeline if r.get("slo_pass")))
    n_total = len(timeline)
    dt_total = summary.get("slo_downtime_total_s", 0)
    title = (
        f"In-place Reshard Real Workload Timeline  "
        f"(HTTP {n_ok}/{n_total}, SLO {n_slo}/{n_total}, downtime={dt_total:.1f}s, span={span:.1f}s)"
    )
    fig.suptitle(title, fontsize=13, y=0.98)
    ax_req.set_xlabel("Time since workload start (s)")
    ax_req.set_ylabel("Request index")
    ax_req.set_ylim(len(timeline), -1)
    ax_req.set_xlim(0, max(span * 1.02, 5))
    ax_req.grid(axis="x", alpha=0.25)

    legend_handles = [
        Patch(facecolor=COLORS[tp], alpha=0.82, label=f"TP{tp} @ issue")
        for tp in sorted({int(r["tp_at_issue"]) for r in timeline})
    ]
    legend_handles.append(Patch(facecolor="#E45756", alpha=0.65, label="reshard pause"))
    legend_handles.append(Patch(facecolor="#7B1E1E", alpha=0.10, label="SLO downtime"))
    if n_ok < n_total:
        legend_handles.append(
            plt.Line2D([0], [0], marker="x", color="#B00020", linestyle="None", label="HTTP fail")
        )
    if n_slo < n_ok:
        legend_handles.append(
            Patch(facecolor="#999999", alpha=0.35, edgecolor="#B00020", label="SLO fail (HTTP ok)")
        )
    ax_req.legend(handles=legend_handles, loc="upper right", frameon=True, fontsize=9)

    # Annotate reshard summary table
    lines = []
    for ev in events:
        pause = ev.get("pause_s")
        pause_txt = f"{pause:.1f}s" if pause is not None else "?"
        xfer = (ev.get("timings") or {}).get("transfer_ms", 0) / 1000
        lines.append(
            f"TP→{ev['new_tp']} @{ev['trigger_s']:.0f}s  pause={pause_txt}  xfer={xfer:.1f}s"
        )
    fig.text(
        0.01,
        0.01,
        "  |  ".join(lines),
        ha="left",
        va="bottom",
        fontsize=8.5,
        color="#444444",
    )

    return save(fig, out_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--stem", default="inplace_workload_timeline_tp1_8")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    png, pdf = plot(data, args.out_dir, args.stem)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
