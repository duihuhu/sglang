#!/usr/bin/env python3
"""Render the three M1 evidence figures from existing confirmation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_vtc_state_tax_observation import CELLS
from plot_m1_opening_observation import LONG, SHORT, figure_data, plot as plot_opening

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import PercentFormatter


RESOURCE_CELLS = ("f16-base", "f16-state-relaxed")
PARTITION_CELLS = ("f1-base", "f4-base", "f16-base", "f32-base")


def series_data(analysis_path: Path) -> dict:
    opening = figure_data(analysis_path)
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    cells = analysis["decisions"]["cells"]
    resources, partitions = [], []
    for cell in dict.fromkeys(RESOURCE_CELLS + PARTITION_CELLS):
        metrics = cells[cell]["token"]
        if metrics["input_imbalance"] != 0 or metrics["output_imbalance"] != 0:
            raise ValueError(f"{cell}: unequal aggregate input or output")
        factor, interval, state_cap, kv_cap = CELLS[cell]
        if cell in RESOURCE_CELLS:
            rows = []
            for tenant in ("packed", "split"):
                rows.append({
                    "tenant": tenant,
                    "peak_state_slots": metrics[f"{tenant}_peak_state"],
                    "peak_kv_tokens": metrics[f"{tenant}_peak_kv"],
                    "peak_state_fraction": metrics[f"{tenant}_peak_state"] / state_cap,
                    "peak_kv_fraction": metrics[f"{tenant}_peak_kv"] / kv_cap,
                })
            resources.append({
                "cell": cell, "state_capacity": state_cap,
                "kv_capacity": kv_cap, "tenants": rows,
            })
        if cell in PARTITION_CELLS:
            partitions.append({
                "cell": cell, "split_factor": factor,
                "split_input_per_request": 16384 // factor,
                "split_output_per_request": 512 // factor,
                "wave_interval_s": interval,
                "state_capacity": state_cap, "kv_capacity": kv_cap,
                "packed_p95_slowdown_ratio": 1 + metrics["packed_slowdown"],
                "split_p95_slowdown_ratio": 1 + metrics["split_slowdown"],
                "mean_absolute_slowdown_gap_pp": 100 * metrics["gap"],
                "packed_queue_p95_s": metrics["packed_queue_p95"],
                "split_queue_p95_s": metrics["split_queue_p95"],
                "offered_token_charge_per_tenant_wave": 17408,
            })
    partitions.sort(key=lambda row: row["split_factor"])
    return {
        "source": opening["source"],
        "source_sha256": opening["source_sha256"],
        "opening": opening,
        "resources": resources,
        "partitions": partitions,
        "scope": "Local token-VTC, Kimi-Linear-48B-A3B, TP2 A800 GPUs 6/7, no prefix reuse",
        "aggregation": "Three confirmation runs per policy/cell; compact source has no per-run CI",
        "resource_note": "Mean per-run peaks; state and KV peaks need not be simultaneous. Pools have separate denominators. Raw slots include the configured state allocation path.",
        "partition_note": "Input length, output length, and request count change together; equal totals do not imply equal compute or application semantics. Gap is mean absolute per-run difference, not absolute difference of mean ratios.",
        "conclusion_scope": "Length/partition-related resource mismatch and service asymmetry; not a causal claim that state tax alone accounts for the latency gap.",
    }


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#E1E5E6", linewidth=0.7)
    ax.tick_params(labelsize=10)


def save(fig, directory, stem):
    directory.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(directory / f"{stem}.{extension}", dpi=200, facecolor="white")
    plt.close(fig)


def plot_resources(data, directory):
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.9))
    fig.subplots_adjust(left=0.085, right=0.97, bottom=0.27, top=0.71, wspace=0.34)
    fig.suptitle("Equal token budgets produce different state / KV occupancy", y=0.98, fontsize=15)
    fig.legend(
        handles=[Patch(color=LONG, label="Long / packed"), Patch(color=SHORT, label="Short / split")],
        loc="upper center", bbox_to_anchor=(0.5, 0.90), ncol=2, frameon=False,
    )
    width = 0.30
    for ax, dimension, title in zip(
        axes, ("state", "kv"),
        ("(a) Recurrent-state slots", "(b) MLA KV tokens"),
    ):
        for index, (tenant, color) in enumerate((("packed", LONG), ("split", SHORT))):
            positions = [x + (index - 0.5) * width for x in range(len(data["resources"]))]
            records = [row["tenants"][index] for row in data["resources"]]
            values = [row[f"peak_{dimension}_fraction"] for row in records]
            bars = ax.bar(positions, values, width=width, color=color)
            for bar, record, value in zip(bars, records, values):
                raw_field = "peak_state_slots" if dimension == "state" else "peak_kv_tokens"
                label = f"{record[raw_field]:,.0f}\n{value:.1%}"
                ax.annotate(
                    label, (bar.get_x() + width / 2, value), xytext=(0, 7),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9,
                )
        ax.set_xticks((0, 1), ("Base\nState cap 48", "State relaxed\nState cap 480"))
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, 0.88)
        ax.set_yticks((0, 0.2, 0.4, 0.6, 0.8))
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set_ylabel("Peak occupancy / configured pool capacity", fontsize=10)
        ax.set_title(title, fontsize=12, loc="left", pad=12)
        ax.tick_params(axis="x", length=0, pad=9)
        style(ax)
    fig.text(
        0.085, 0.095,
        "Both cells: split factor 16; KV cap 196,608; same offered tokens per tenant. "
        "State-cap change also changes admission concurrency.", fontsize=8.5, color="#555555",
    )
    fig.text(
        0.085, 0.035,
        "Measured means of 3 runs. Peaks need not coincide; the two pool fractions are not additive. "
        "Source: 2026-08-26 confirmation.", fontsize=8.5, color="#555555",
    )
    save(fig, directory, "m1_measured_resource_pressure")


def plot_partitions(data, directory):
    rows = data["partitions"]
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 5.0))
    fig.subplots_adjust(left=0.075, right=0.97, bottom=0.30, top=0.73, wspace=0.30)
    fig.suptitle("Token VTC: service asymmetry emerges as requests become shorter", y=0.98, fontsize=15)
    x = list(range(len(rows)))
    xticks = ("f=1\n16K / 512", "f=4\n4K / 128", "f=16\n1K / 32", "f=32\n512 / 16")
    for ax, suffix, title, ylabel in zip(
        axes,
        ("p95_slowdown_ratio", "queue_p95_s"),
        ("(a) Relative latency inflation", "(b) Admission waiting time"),
        ("Mixed p95 / isolated p95 JCT", "Queue waiting p95 (seconds)"),
    ):
        ax.axvspan(-0.23, 0.23, color="#EDF0F0", zorder=0)
        for tenant, color, marker, offset in (
            ("packed", LONG, "o", -19), ("split", SHORT, "s", 10)
        ):
            values = [row[f"{tenant}_{suffix}"] for row in rows]
            ax.plot(x, values, color=color, marker=marker, linewidth=2, markersize=6,
                    label="Long / packed" if tenant == "packed" else "Split tenant")
            for pos, value in zip(x, values):
                label_offset = offset
                if suffix == "p95_slowdown_ratio" and tenant == "packed":
                    label_offset = -13 if pos == 0 else 9
                ax.annotate(
                    f"{value:.2f}", (pos, value), xytext=(0, label_offset),
                    textcoords="offset points", ha="center", va="bottom",
                    fontsize=9, color=color,
                )
        ax.set_xticks(x, xticks)
        ax.set_xlim(-0.35, 3.35)
        ax.set_ylim(0, 13.8)
        ax.set_title(title, fontsize=12, loc="left", pad=12)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Split factor / (input, output) per split request", fontsize=9, labelpad=11)
        ax.tick_params(axis="x", length=0, pad=9)
        style(ax)
    axes[0].axhline(1, linestyle="--", color="#7B858C", linewidth=0.8)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.91), ncol=2, frameon=False)
    fig.text(
        0.075, 0.090,
        "Packed remains (16K, 512); both tenants keep the same total input/output per wave. "
        "Shaded f=1: identical request shapes.", fontsize=8.5, color="#555555",
    )
    fig.text(
        0.075, 0.035,
        "State cap 48; KV cap 196,608; wave interval 1.5s. "
        "Means of 3 runs; no CI in compact source. 2026-08-26 confirmation.",
        fontsize=8.5, color="#555555",
    )
    save(fig, directory, "m1_partition_sensitivity")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output_directory", type=Path)
    args = parser.parse_args()
    data = series_data(args.analysis)
    plot_opening(data["opening"], args.output_directory / "m1_token_state_fairness")
    plot_resources(data, args.output_directory)
    plot_partitions(data, args.output_directory)
    print(json.dumps(data, indent=2, allow_nan=False))
