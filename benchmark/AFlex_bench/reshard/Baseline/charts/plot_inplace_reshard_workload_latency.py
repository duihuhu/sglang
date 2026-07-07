#!/usr/bin/env python3
"""Plot per-request TTFT and TPOT vs time for in-place reshard workload runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch

COLORS = {1: "#4C78A8", 2: "#72B7B2", 4: "#54A24B", 8: "#F58518"}


def decode_tpot_s(req: dict[str, Any]) -> float | None:
    if req.get("tpot_s") is not None:
        return float(req["tpot_s"])
    ttft = req.get("ttft_s")
    e2e = req.get("e2e_s")
    out_len = int(req.get("output_len") or 0)
    if ttft is None or e2e is None:
        return None
    decode_s = max(float(e2e) - float(ttft), 0.0)
    denom = max(out_len - 1, 1)
    return decode_s / denom


def slo_pass(req: dict[str, Any], ttft_lim: float, tpot_lim: float) -> bool:
    if "slo_pass" in req:
        return bool(req["slo_pass"])
    if not req.get("ok"):
        return False
    ttft = req.get("ttft_s")
    tpot = decode_tpot_s(req)
    if ttft is None or tpot is None:
        return False
    return float(ttft) <= ttft_lim and tpot <= tpot_lim


def save(fig, out_dir: Path, stem: str) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def shade_reshard_windows(ax, events: list[dict[str, Any]]) -> None:
    for ev in events:
        start = float(ev["trigger_s"])
        end = float(ev["done_s"]) if ev.get("done_s") is not None else start + float(ev.get("pause_s") or 1.0)
        ax.axvspan(start, end, color="#E45756", alpha=0.12, zorder=0)
        ax.axvline(start, color="#E45756", linestyle="--", linewidth=0.9, alpha=0.55)
        ax.axvline(end, color="#E45756", linestyle=":", linewidth=0.9, alpha=0.55)


def shade_downtime_windows(ax, windows: list[dict[str, Any]]) -> None:
    for w in windows:
        ax.axvspan(
            float(w["start_s"]),
            float(w["end_s"]),
            color="#7B1E1E",
            alpha=0.18,
            zorder=1,
        )


def plot(data: dict[str, Any], out_dir: Path, stem: str) -> tuple[Path, Path]:
    summary = data["summary"]
    timeline = data["timeline"]
    events = summary.get("reshard_events", [])
    slo = summary.get("slo", {})
    ttft_lim = float(slo.get("ttft_ms", 2000)) / 1000.0
    tpot_lim = float(slo.get("tpot_ms", 100)) / 1000.0
    downtime = summary.get("slo_downtime_windows_merged") or summary.get(
        "slo_downtime_windows", []
    )
    span = max((float(r["issue_s"]) for r in timeline), default=1.0)

    fig, (ax_ttft, ax_tpot) = plt.subplots(
        2,
        1,
        figsize=(14.5, 8.5),
        sharex=True,
        gridspec_kw={"height_ratios": [1, 1], "hspace": 0.08},
    )

    for ax in (ax_ttft, ax_tpot):
        shade_reshard_windows(ax, events)
        shade_downtime_windows(ax, downtime)

    for tp in sorted({int(r["tp_at_issue"]) for r in timeline}):
        pts = [r for r in timeline if int(r["tp_at_issue"]) == tp]
        for req in pts:
            issue = float(req["issue_s"])
            passed = slo_pass(req, ttft_lim, tpot_lim)
            marker = "o" if passed else "x"
            edge = "white" if passed else "#B00020"
            lw = 0.4 if passed else 1.2
            if req.get("ttft_s") is not None:
                ax_ttft.scatter(
                    issue,
                    float(req["ttft_s"]),
                    s=42,
                    c=COLORS.get(tp, "#999999"),
                    alpha=0.85,
                    marker=marker,
                    edgecolors=edge,
                    linewidths=lw,
                    zorder=3,
                )
            tpot = decode_tpot_s(req)
            if tpot is not None:
                ax_tpot.scatter(
                    issue,
                    tpot,
                    s=42,
                    c=COLORS.get(tp, "#999999"),
                    alpha=0.85,
                    marker=marker,
                    edgecolors=edge,
                    linewidths=lw,
                    zorder=3,
                )

        n_tp = len(pts)
        ax_ttft.scatter([], [], s=42, c=COLORS.get(tp, "#999"), label=f"TP{tp} (n={n_tp})")

    ax_ttft.axhline(ttft_lim, color="#333333", linestyle="-.", linewidth=1.0, alpha=0.7, zorder=2)
    ax_tpot.axhline(tpot_lim, color="#333333", linestyle="-.", linewidth=1.0, alpha=0.7, zorder=2)
    ax_ttft.text(span * 0.01, ttft_lim * 1.02, f"SLO {ttft_lim*1000:.0f}ms", fontsize=8, color="#333")
    ax_tpot.text(span * 0.01, tpot_lim * 1.02, f"SLO {tpot_lim*1000:.0f}ms", fontsize=8, color="#333")

    ttft_max = max(
        (float(r["ttft_s"]) for r in timeline if r.get("ttft_s") is not None),
        default=ttft_lim,
    )
    for ev in events:
        end = ev.get("done_s")
        if end is None:
            continue
        mid = (float(ev["trigger_s"]) + float(end)) / 2
        ax_ttft.text(
            mid,
            ttft_max * 1.05,
            f"→TP{ev['new_tp']}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#B00020",
            fontweight="bold",
        )

    n_ok = summary.get("n_ok", 0)
    n_slo = summary.get("n_slo_pass", sum(1 for r in timeline if slo_pass(r, ttft_lim, tpot_lim)))
    n_total = summary.get("n_total", len(timeline))
    dt_total = summary.get("slo_downtime_total_s", sum(w.get("duration_s", 0) for w in downtime))
    fig.suptitle(
        f"Per-request latency vs time  (HTTP {n_ok}/{n_total}, SLO {n_slo}/{n_total}, "
        f"downtime={dt_total:.1f}s, span={span:.1f}s)",
        fontsize=12,
        y=0.98,
    )

    ax_ttft.set_ylabel("TTFT (s)")
    ax_ttft.set_title("Time to first token (× = SLO fail)", fontsize=10, loc="left")
    ax_ttft.grid(axis="y", alpha=0.3)
    ax_tpot.set_ylabel("TPOT (s/token)")
    ax_tpot.set_xlabel("Request issue time since workload start (s)")
    ax_tpot.set_title("Decode TPOT (× = SLO fail)", fontsize=10, loc="left")
    ax_tpot.grid(axis="y", alpha=0.3)

    tp_list = sorted({int(r["tp_at_issue"]) for r in timeline})
    legend_handles = [Patch(facecolor=COLORS[tp], label=f"TP{tp}") for tp in tp_list]
    legend_handles.append(Patch(facecolor="#E45756", alpha=0.12, label="reshard pause"))
    legend_handles.append(Patch(facecolor="#7B1E1E", alpha=0.18, label="SLO downtime"))
    legend_handles.append(
        plt.Line2D([0], [0], marker="x", color="#B00020", linestyle="None", label="SLO fail")
    )
    ax_ttft.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
    ax_tpot.legend(handles=legend_handles, loc="upper right", fontsize=8, frameon=True)
    ax_ttft.set_xlim(0, span * 1.02)

    return save(fig, out_dir, stem)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    parser.add_argument("--stem", default="inplace_workload_ttft_tpot_tp1_8")
    args = parser.parse_args()
    data = json.loads(args.input.read_text())
    png, pdf = plot(data, args.out_dir, args.stem)
    print(f"wrote {png}")
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
