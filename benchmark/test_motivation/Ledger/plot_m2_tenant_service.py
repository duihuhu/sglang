#!/usr/bin/env python3
"""Plot historical tenant outcomes without labelling latency as fairness."""

import argparse
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot(data, output):
    rows = [row for row in data["mixed_rows"] if row["experiment"]["split_factor"] == 16]
    if len(rows) != 10 or not data["complete"]:
        raise ValueError("Expected all ten f16 mixed confirmation runs")
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    panels = [
        ("p95_jct_s", "Request p95 JCT (s)", 1),
        ("output_50", "Time to deliver 50% of tenant output (s)", 1),
        ("state_slot_seconds", "Active state occupancy (slot-seconds)", 1),
        ("kv_token_seconds", "Active KV occupancy (million token-seconds)", 1e6),
    ]
    for ax, (metric, title, scale) in zip(axes.flat, panels):
        for group, color, label in [("packed", "#705894", "Packed tenant"),
                                     ("split", "#167c87", "Split tenant")]:
            by_cap = {}
            for cap in (96, 48):
                by_cap[cap] = {}
                for row in rows:
                    if row["experiment"]["admission_cap"] != cap:
                        continue
                    own = row["tenants"][group]
                    value = own["output_milestones_s"]["50"] if metric == "output_50" else own[metric]
                    by_cap[cap][row["experiment"]["repetition"]] = value / scale
            if set(by_cap[96]) != set(by_cap[48]) or len(by_cap[96]) != 5:
                raise ValueError("Unpaired f16 measurements")
            for rep in sorted(by_cap[96]):
                ax.plot([0, 1], [by_cap[96][rep], by_cap[48][rep]], color=color,
                        alpha=0.22, linewidth=1, marker="o", markersize=3)
            means = [statistics.fmean(by_cap[cap].values()) for cap in (96, 48)]
            ax.plot([0, 1], means, color=color, linewidth=2.5, marker="D", markersize=6, label=label)
            for x, value in enumerate(means):
                ax.annotate(f"{value:.2f}", (x, value), xytext=(0, 9), textcoords="offset points",
                            ha="center", color=color, fontsize=11)
        ax.set_title(title, fontsize=12, pad=12)
        ax.set_xticks([0, 1], ["Cap96", "Cap48"])
        ax.set_xlim(-0.18, 1.18)
        ax.set_ylim(bottom=0, top=ax.get_ylim()[1] * 1.13)
        ax.grid(axis="y", alpha=0.18)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    fig.suptitle("Equal final token charge; different resource and service outcomes\nf16 / same token-VTC / historical paired cap control", fontsize=16, y=0.985)
    fig.text(0.5, 0.055,
             "Each tenant: 131,072 input + 4,096 output tokens. Thin lines: five paired repeats; diamonds: means.\n"
             "Output timing is not VTC fairness. State counts include three assigned slots per active request.\n"
             "Post-hoc audit of September 9 data; no new GPU runs or new fairness PASS.",
             ha="center", va="center", fontsize=10)
    fig.subplots_adjust(top=0.82, bottom=0.16, left=0.09, right=0.97, hspace=0.45, wspace=0.25)
    output.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"m2_tenant_service_f16.{suffix}", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audit", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plot(json.loads(args.audit.read_text()), args.output)


if __name__ == "__main__":
    main()
