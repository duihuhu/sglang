#!/usr/bin/env python3
"""Separate offered service, analytical state tax, and measured slowdown."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


LONG = "#27755D"
SHORT = "#C04B48"


def figure_data(analysis_path: Path) -> dict:
    source = analysis_path.read_bytes()
    analysis = json.loads(source)
    if not analysis["integrity_pass"]:
        raise ValueError("Source integrity checks did not pass")
    metrics = analysis["decisions"]["cells"]["f16-base"]["token"]
    if metrics["input_imbalance"] != 0 or metrics["output_imbalance"] != 0:
        raise ValueError("The selected cell does not have equal aggregate tokens")
    shapes = {"packed": (1, 16384, 512), "split": (16, 1024, 32)}
    rows = []
    for tenant, (count, prompt, output) in shapes.items():
        rows.append({
            "tenant": tenant,
            "requests_per_wave": count,
            "input_per_request": prompt,
            "output_per_request": output,
            "offered_token_charge": count * (prompt + 2 * output),
            "fixed_state_per_initial_resident_token_relative_to_packed": 16384 / prompt,
            "mean_p95_slowdown_ratio": 1 + metrics[f"{tenant}_slowdown"],
        })
    return {
        "source": str(analysis_path),
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "cell": "f16-base",
        "policy": "local token-VTC",
        "input_weight": 1,
        "output_weight": 2,
        "rows": rows,
        "panel_a": "Offered workload charge, not measured backlogged-window fairness",
        "panel_b": "Analytical S/L for identical active state size S; no aggregate memory or state-time claim",
        "panel_c": "Mean of per-run mixed/isolated p95 ratios, not a ratio of pooled request quantiles; CI unavailable in this compact input",
        "causality": "Juxtaposition of observations, not proof that state tax alone causes slowdown",
    }


def plot(data: dict, output_stem: Path):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelcolor": "#333333",
        "text.color": "#252525",
        "axes.edgecolor": "#555555",
        "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 5.0))
    fig.subplots_adjust(left=0.073, right=0.975, bottom=0.28, top=0.71, wspace=0.65)
    fig.suptitle("Equal token budgets, unequal state tax and slowdown", y=0.975, fontsize=16)
    fig.legend(
        handles=[
            Patch(facecolor=LONG, label="Long / packed: 1 x (16K input, 512 output)"),
            Patch(facecolor=SHORT, label="Short / split: 16 x (1K input, 32 output)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, 0.905), ncol=2,
        frameon=False, columnspacing=2.8, fontsize=10,
    )
    rows = data["rows"]
    fields = (
        "offered_token_charge",
        "fixed_state_per_initial_resident_token_relative_to_packed",
        "mean_p95_slowdown_ratio",
    )
    titles = (
        "(a) Token-accounting view\nWorkload identity",
        "(b) Fixed-state cost view\nAnalytical: S / resident length",
        "(c) Service-isolation view\nMeasured: f16-base, token VTC",
    )
    labels = (
        "Input + 2 x output tokens / wave",
        "State bytes / resident token\n(relative to long request)",
        "Mixed p95 / isolated p95 JCT",
    )
    for ax, field, title, label in zip(axes, fields, titles, labels):
        values = [row[field] for row in rows]
        bars = ax.bar((0, 1), values, color=[LONG, SHORT], width=0.53, zorder=3)
        ax.set_xticks((0, 1), ("Long / packed", "Short / split"))
        ax.set_title(title, fontsize=11, loc="left", pad=15, linespacing=1.5)
        ax.set_ylabel(label, fontsize=10, labelpad=8)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#E2E5E6", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", length=0, pad=9, labelsize=9)
        ax.tick_params(axis="y", labelsize=9)
        ax.set_xlim(-0.55, 1.55)
        ax.set_ylim(0, max(values) * 1.25)
        if field == "offered_token_charge":
            ax.set_yticks((0, 8192, 16384), ("0", "8,192", "16,384"))
            texts = [f"{int(v):,}" for v in values]
        else:
            texts = [f"{v:.2f}x" if field.startswith("mean_") else f"{v:g}x" for v in values]
            ax.axhline(1, color="#697781", linewidth=0.9, linestyle="--", zorder=2)
        for bar, text in zip(bars, texts):
            ax.annotate(
                text, (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 6), textcoords="offset points", ha="center",
                va="bottom", fontsize=11, fontweight="semibold",
            )
    notes = (
        "16,384 input + 512 output tokens\nper tenant, per wave",
        "Same active-state size S\nInitial context: 16K vs. 1K",
        "Matched isolated baselines\nMean across 3 runs; CI not supplied",
    )
    for ax, note in zip(axes, notes):
        bounds = ax.get_position()
        fig.text(
            bounds.x0 + bounds.width / 2, 0.145, note,
            ha="center", va="center", fontsize=9, color="#555555", linespacing=1.5,
        )
    fig.text(
        0.073, 0.035,
        "Kimi-Linear-48B-A3B | TP2, A800 GPUs 6/7 | No prefix reuse | "
        "Measured panel: 2026-08-26 confirmation summary",
        fontsize=8.5, color="#666666",
    )
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "svg"):
        fig.savefig(output_stem.with_suffix(f".{extension}"), dpi=200, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output_stem", type=Path)
    args = parser.parse_args()
    data = figure_data(args.analysis)
    plot(data, args.output_stem)
    print(json.dumps(data, indent=2, allow_nan=False))
