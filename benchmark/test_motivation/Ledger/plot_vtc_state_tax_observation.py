#!/usr/bin/env python3
"""Plot the confirmed VTC state-tax observation without changing its gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


CELLS = (
    "f1-base",
    "f4-base",
    "f16-base",
    "f32-base",
    "f16-state-relaxed",
    "f16-kv-relaxed",
    "f16-full",
    "f16-fast",
    "f16-slow",
)
CELL_LABELS = (
    "f1\nbase",
    "f4\nbase",
    "f16\nbase",
    "f32\nbase",
    "f16\nstate+",
    "f16\nKV+",
    "f16\nfull",
    "f16\nfast",
    "f16\nslow",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    analysis = json.loads(args.analysis.read_text(encoding="utf-8"))
    cells = analysis["decisions"]["cells"]

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.2))

    resource_cases = (
        ("Base\nsplit", "f16-base", "split", 48, 196608),
        ("Base\npacked", "f16-base", "packed", 48, 196608),
        ("State relaxed\nsplit", "f16-state-relaxed", "split", 480, 196608),
        ("State relaxed\npacked", "f16-state-relaxed", "packed", 480, 196608),
    )
    positions = np.arange(len(resource_cases))
    state_share = [
        cells[cell]["token"][f"{tenant}_peak_state"] / state_cap
        for _, cell, tenant, state_cap, _ in resource_cases
    ]
    kv_share = [
        cells[cell]["token"][f"{tenant}_peak_kv"] / kv_cap
        for _, cell, tenant, _, kv_cap in resource_cases
    ]
    width = 0.34
    axes[0].bar(
        positions - width / 2,
        state_share,
        width,
        color="#2F7D5B",
        label="Peak recurrent-state share",
    )
    axes[0].bar(
        positions + width / 2,
        kv_share,
        width,
        color="#3973B7",
        label="Peak active-KV share",
    )
    axes[0].set_xticks(positions, [item[0] for item in resource_cases])
    axes[0].set_ylabel("Fraction of configured capacity")
    axes[0].set_title("(a) Equal tokens, different resource vectors")
    axes[0].set_ylim(0, max(state_share + kv_share) * 1.22)
    axes[0].grid(axis="y", alpha=0.22, linewidth=0.8)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")

    policy_series = (
        ("Standard token VTC", "token", "#3B3B3B", "o"),
        ("Profiled scalar VTC", "profiled", "#D17A22", "s"),
        ("Instant state/KV share", "instant", "#168C8C", "^"),
    )
    x = np.arange(len(CELLS))
    for label, policy, color, marker in policy_series:
        values = [100 * cells[cell][policy]["gap"] for cell in CELLS]
        axes[1].plot(
            x,
            values,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
            label=label,
        )
    axes[1].set_yscale("log")
    axes[1].set_xticks(x, CELL_LABELS)
    axes[1].set_ylabel("p95 normalized-slowdown gap (%)")
    axes[1].set_title("(b) Runtime resource state beats scalar cost in 8/9 cells")
    axes[1].grid(axis="y", which="both", alpha=0.22, linewidth=0.8)
    axes[1].legend(frameon=False, fontsize=9, loc="lower left")

    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.tick_params(axis="x", labelsize=9)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
