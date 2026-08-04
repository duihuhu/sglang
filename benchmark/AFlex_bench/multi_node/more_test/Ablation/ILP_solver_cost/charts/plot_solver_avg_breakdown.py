#!/usr/bin/env python3
"""Plot Code and Conversation average solver latency breakdown versus GPU budget."""
from __future__ import annotations
import argparse, csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent.parent
PARTS = [
    ("decode_enumeration_mean_ms", "Decode enumeration", "#4c78a8"),
    ("prefill_enumeration_mean_ms", "Prefill enumeration", "#72b7b2"),
    ("pareto_mean_ms", "Pareto", "#f58518"),
    ("global_search_mean_ms", "Global search", "#54a24b"),
    ("other_mean_ms", "Other", "#bab0ac"),
]

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=ROOT / "data/solver_avg_breakdown_summary.csv")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "charts")
    return parser.parse_args()

def number(row, key):
    return float(row[key])

def draw_panel(ax, rows, title):
    rows = sorted(rows, key=lambda row: int(row["G"]))
    if [int(row["G"]) for row in rows] != [8, 16, 32]:
        raise ValueError(f"{title} must contain exactly G=8,16,32")
    x = np.arange(len(rows))
    bottom = np.zeros(len(rows))
    infeasible = np.array([number(row, "feasible_rate") < 1.0 for row in rows])
    for key, label, color in PARTS:
        values = np.array([number(row, key) for row in rows])
        bars = ax.bar(x, values, bottom=bottom, width=.62, color=color, label=label,
                      linewidth=.35, edgecolor="white")
        for index, bar in enumerate(bars):
            if infeasible[index]:
                bar.set_hatch("////")
                bar.set_edgecolor("#555555")
                bar.set_linewidth(.45)
        bottom += values
    ax.set_xticks(x, [row["G"] for row in rows])
    ax.set_xlabel("GPU budget G")
    ax.set_title(title)
    ax.grid(axis="y", alpha=.22, linewidth=.5)
    ax.set_axisbelow(True)
    ax.margins(y=.06)

def main():
    args = parse_args()
    with args.summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 6:
        raise ValueError(f"expected 6 summary rows, got {len(rows)}")
    code = [row for row in rows if row["dataset"] == "code"]
    conversation = [row for row in rows if row["dataset"] == "conversation"]
    if len(code) != 3 or len(conversation) != 3:
        raise ValueError("expected three Code and three Conversation rows")
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 7.2,
        "axes.titlesize": 8.2, "axes.labelsize": 7.7, "legend.fontsize": 6.5,
        "xtick.labelsize": 6.8, "ytick.labelsize": 6.8,
    })
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.5), sharey=True, constrained_layout=True)
    draw_panel(axes[0], code, "(a) Code")
    draw_panel(axes[1], conversation, "(b) Conversation")
    axes[0].set_ylabel("Average latency (ms)")
    handles = [Patch(facecolor=color, label=label) for _, label, color in PARTS]
    if any(number(row, "feasible_rate") < 1.0 for row in rows):
        handles.append(Patch(facecolor="white", edgecolor="#555555", hatch="////", label="Infeasible"))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.035),
               ncol=len(handles), frameon=False, columnspacing=.95, handlelength=1.05)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "solver_avg_breakdown"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {stem}.pdf")

if __name__ == "__main__":
    main()
