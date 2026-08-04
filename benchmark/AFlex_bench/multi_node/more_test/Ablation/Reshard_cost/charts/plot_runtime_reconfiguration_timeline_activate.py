#!/usr/bin/env python3
"""Two clean stacked bars: Expand and Shrink ACTIVATE-phase breakdown."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

STAGES = ("prefill", "decode")
COMPONENTS = ("attn", "ffn")
TRANSITIONS = ("t1", "t2")
LABELS = {"t1": "Expand (TP2→TP4)", "t2": "Shrink (TP4→TP1)"}

MACRO_PHASES = ()  # PUBLISH removed — now async in background

ACTIVATE_STEPS = (
    ("Rebuild NCCL groups", "activate_rebuild_groups_s", "#4C78A8"),
    ("Rebuild KVCache", "activate_refresh_runtime_s", "#54A24B"),
    ("Mooncake RDMA register", "activate_refresh_bootstrap_topology_s", "#B279A2"),
)

# Sub-steps with negligible time (<0.1s) folded into parent steps
_MERGE_MAP = {
    "activate_refresh_scheduler_groups_s": "activate_rebuild_groups_s",
    "activate_materialize_s":              "activate_refresh_runtime_s",
    "activate_consensus_s":                "activate_refresh_bootstrap_topology_s",
}


def measured(item: Any) -> float | None:
    if isinstance(item, Mapping) and item.get("status") == "measured":
        v = item.get("value")
        return float(v) if isinstance(v, (int, float)) else None
    return None


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def load_paper_runs(result_dir: Path) -> list[dict[str, Mapping[str, Any]]]:
    paths = sorted(result_dir.glob("run_*_paper_breakdown.json"))
    if len(paths) != 3:
        raise ValueError(f"Need 3 paper_breakdown files, found {len(paths)}")
    out = []
    for p in paths:
        doc = json.loads(p.read_text())
        out.append({t["id"]: t for t in doc["transitions"]})
    return out


def load_full_runs(result_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(result_dir.glob("run_*_full_sequence.json"))
    if len(paths) != 3:
        raise ValueError(f"Need 3 full_sequence files, found {len(paths)}")
    return [json.loads(p.read_text()) for p in paths]


def wall_mean(runs, tid: str, metric: str) -> float:
    vals = []
    for run in runs:
        vals.append(max(
            measured(run[tid]["stages"][s]["wall_clock"][metric]) for s in STAGES
        ))
    return mean_std(vals)[0]


@dataclass
class CriticalRank:
    activate_s: float
    steps: dict[str, float]
    envelope: float


def critical_rank(transition: Mapping[str, Any], macro_activate: float) -> CriticalRank:
    best = None
    for stage in STAGES:
        for comp in COMPONENTS:
            for rec in transition[stage]["breakdown"][comp]["ranks"]:
                a = float(rec["activate_s"])
                if best is None or a > best[0] + 1e-4:
                    best = (a, rec)
    assert best is not None
    act_s, rec = best
    steps = {}
    for _, field, _ in ACTIVATE_STEPS:
        if field == "__envelope__":
            continue
        v = rec.get(field)
        if isinstance(v, (int, float)) and v > 0:
            steps[field] = float(v)
    envelope = max(macro_activate - act_s, 0.0)
    return CriticalRank(act_s, steps, envelope)


def aggregate(result_dir: Path) -> dict[str, dict[str, float]]:
    paper = load_paper_runs(result_dir)
    full = load_full_runs(result_dir)
    out: dict[str, dict[str, float]] = {}
    for tid in TRANSITIONS:
        row: dict[str, float] = {}
        for _, metric, _ in MACRO_PHASES:
            row[metric] = wall_mean(paper, tid, metric)
        macro_act = wall_mean(paper, tid, "activate_s")

        # ACTIVATE sub-steps: mean across 3 runs (each run picks its critical rank)
        step_lists: dict[str, list[float]] = {f: [] for _, f, _ in ACTIVATE_STEPS if f != "__envelope__"}
        env_list: list[float] = []
        for doc in full:
            cr = critical_rank(doc[tid], max(
                doc[tid][s]["breakdown"]["activate_s"] for s in STAGES
            ))
            for f, v in cr.steps.items():
                step_lists[f].append(v)
            env_list.append(cr.envelope)
        for _, f, _ in ACTIVATE_STEPS:
            if f == "__envelope__":
                row[f] = mean_std(env_list)[0]
            elif step_lists.get(f):
                row[f] = mean_std(step_lists[f])[0]
            else:
                row[f] = 0.0
        row["activate_s"] = macro_act
        # Fold negligible sub-steps into their parent steps
        for child, parent in _MERGE_MAP.items():
            row[parent] = row.get(parent, 0.0) + row.get(child, 0.0)
            row[child] = 0.0
        out[tid] = row
    return out


def plot(result_dir: Path, stem: str, out_dir: Path) -> tuple[Path, Path]:
    data = aggregate(result_dir)

    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(10, 2.8))

    y_pos = [1, 0]  # Expand (t1) on top
    bar_h = 0.55

    legend_items = [(l, c) for l, _, c in ACTIVATE_STEPS]

    for yi, tid in zip(y_pos, TRANSITIONS):
        row = data[tid]
        left = 0.0
        for label, field, color in ACTIVATE_STEPS:
            w = row.get(field, 0.0)
            if w <= 0:
                continue
            ax.barh(yi, w, left=left, height=bar_h, color=color, edgecolor="white", linewidth=0.6)
            if w >= 0.25:
                ax.text(left + w / 2, yi, f"{w:.2f}s", ha="center", va="center", fontsize=7, color="white", fontweight="bold")
            left += w
        total = left
        ax.text(total + 0.10, yi, f"{total:.2f}s", ha="left", va="center", fontsize=9, fontweight="bold")

    ax.set_yticks(y_pos, [LABELS[t] for t in TRANSITIONS])
    ax.set_xlabel("Duration (s)")
    ax.set_xlim(0, 4)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    handles = [Patch(facecolor=c, label=l) for l, c in legend_items]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=3, frameon=False, fontsize=8)

    fig.subplots_adjust(left=0.14, right=0.95, top=0.82, bottom=0.15)

    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def main() -> int:
    here = Path(__file__).resolve().parent
    root = here.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", type=Path, default=root / "data")
    parser.add_argument("--output-dir", type=Path, default=root / "charts")
    parser.add_argument("--output-stem", default="runtime_reconfiguration_timeline_activate")
    args = parser.parse_args()
    for p in plot(args.result_dir, args.output_stem, args.output_dir):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
