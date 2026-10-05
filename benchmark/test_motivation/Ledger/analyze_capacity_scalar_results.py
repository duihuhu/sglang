#!/usr/bin/env python3
"""Analyze the state-cap x KV-cap x scalar-weight Topic B experiment."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def mean(values: list[float]) -> float:
    return statistics.fmean(values)


def mean_std(values: list[float]) -> str:
    if len(values) == 1:
        return f"{values[0]:.3f}"
    return f"{mean(values):.3f} +/- {statistics.stdev(values):.3f}"


def capacities(artifact: dict[str, Any]) -> tuple[int, int]:
    state_capacity = next(
        int(state["max_mamba_cache_size"])
        for state in artifact["server"]["internal_states"]
        if state.get("max_mamba_cache_size") is not None
    )
    return state_capacity, int(artifact["server"]["max_total_num_tokens"])


def makespan(artifact: dict[str, Any]) -> float:
    return max(req["finished_s"] for req in artifact["requests"]) - min(
        req["submitted_s"] for req in artifact["requests"]
    )


def debts(artifact: dict[str, Any]) -> tuple[float, float]:
    state_capacity, kv_capacity = capacities(artifact)
    tenants = artifact["scheduler_resource_trace"]["tenants"]

    def dominant(group: str) -> float:
        usage = tenants[group]
        return max(
            usage["state_slot_seconds"] / state_capacity,
            usage["kv_token_seconds"] / kv_capacity,
        )

    return dominant("state-heavy"), dominant("kv-heavy")


def relative_gap(pair: tuple[float, float]) -> float:
    left, right = pair
    midpoint = (left + right) / 2
    return abs(left - right) / midpoint if midpoint else 0.0


def pressure_ratio(artifact: dict[str, Any]) -> float:
    state_capacity, kv_capacity = capacities(artifact)
    tenants = artifact["scheduler_resource_trace"]["tenants"]
    state_time = sum(item["state_slot_seconds"] for item in tenants.values())
    kv_time = sum(item["kv_token_seconds"] for item in tenants.values())
    return (state_time / state_capacity) / (kv_time / kv_capacity)


def effective_max_running(artifact: dict[str, Any]) -> int:
    return next(
        int(state["effective_max_running_requests_per_dp"])
        for state in artifact["server"]["internal_states"]
        if state.get("effective_max_running_requests_per_dp") is not None
    )


def peak_utilization(artifact: dict[str, Any]) -> tuple[float, float]:
    trace = artifact["scheduler_resource_trace"]
    if "system" not in trace:
        raise ValueError("resource trace schema v2 with system peaks is required")
    system = trace["system"]
    _, kv_capacity = capacities(artifact)
    return (
        system["peak_active_requests"] / effective_max_running(artifact),
        system["peak_kv_tokens"] / kv_capacity,
    )


def pressure_class(state_util: float, kv_util: float) -> str:
    state_saturated = state_util >= 0.90
    kv_saturated = kv_util >= 0.90
    if state_saturated and kv_saturated:
        return "dual-bound"
    if state_saturated:
        return "state-bound"
    if kv_saturated:
        return "KV-bound"
    return "underloaded"


def policy_key(artifact: dict[str, Any]) -> tuple[str, float | None]:
    experiment = artifact["experiment"]
    label = experiment.get("policy_label")
    if not label:
        raise ValueError(
            f"{experiment['run_id']} lacks policy_label; use the cross-sweep runner"
        )
    weight = experiment.get("scalar_weight") if label.startswith("scalar-") else None
    return label, weight


def load(input_dir: Path) -> dict[str, dict[tuple[str, float | None], list[dict[str, Any]]]]:
    grouped: dict[
        str, dict[tuple[str, float | None], list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for path in sorted(input_dir.glob("cross-*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        summary = artifact["summary"]["overall"]
        if summary["completed"] != summary["requests"]:
            raise ValueError(f"{path} contains failed requests")
        if artifact.get("scheduler_resource_trace") is None:
            raise ValueError(f"{path} lacks exact scheduler resource accounting")
        capacity_label = artifact["experiment"].get("capacity_label")
        if not capacity_label:
            raise ValueError(f"{path} lacks capacity_label")
        grouped[capacity_label][policy_key(artifact)].append(artifact)
    if not grouped:
        raise ValueError(f"No cross-*.json artifacts found in {input_dir}")
    return grouped


def aggregate(items: list[dict[str, Any]]) -> dict[str, float | int | list[float]]:
    pairs = [debts(item) for item in items]
    return {
        "n": len(items),
        "relative_gap": [relative_gap(pair) for pair in pairs],
        "absolute_gap": [abs(pair[0] - pair[1]) for pair in pairs],
        "makespan": [makespan(item) for item in items],
        "goodput": [
            item["summary"]["overall"]["completed_within_slo"] for item in items
        ],
        "retractions": [
            item["summary"]["overall"]["total_retractions"] for item in items
        ],
        "state_p95": [
            item["summary"]["groups"]["state-heavy"]["p95_latency_s"]
            for item in items
        ],
        "kv_p95": [
            item["summary"]["groups"]["kv-heavy"]["p95_latency_s"]
            for item in items
        ],
    }


def build_report(
    grouped: dict[str, dict[tuple[str, float | None], list[dict[str, Any]]]]
) -> tuple[str, dict[str, Any]]:
    lines = [
        "# Topic B Capacity x Scalar-Weight Cross Experiment",
        "",
        "> The screening phase uses one repetition. It identifies bottleneck regimes and "
        "candidate weights; it is not statistical evidence until finalists are repeated.",
        "",
        "## Capacity calibration",
        "",
        "| capacity | actual state cap | actual KV cap | state admission util. | KV util. | normalized pressure state/KV | regime | FCFS retractions |",
        "|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    summary: dict[str, Any] = {"capacities": {}, "global_scalar": {}}
    cells: dict[str, dict[str, Any]] = {}
    scalar_weights: set[float] = set()

    for capacity_label in sorted(grouped):
        policies = grouped[capacity_label]
        fcfs_items = policies.get(("fcfs", None))
        if not fcfs_items:
            raise ValueError(f"{capacity_label} has no FCFS calibration run")
        actual_state, actual_kv = capacities(fcfs_items[0])
        ratios = [pressure_ratio(item) for item in fcfs_items]
        utilizations = [peak_utilization(item) for item in fcfs_items]
        state_utils = [item[0] for item in utilizations]
        kv_utils = [item[1] for item in utilizations]
        regime = pressure_class(mean(state_utils), mean(kv_utils))
        fcfs_metrics = aggregate(fcfs_items)
        cells[capacity_label] = {
            "policies": policies,
            "fcfs": fcfs_metrics,
            "regime": regime,
            "state_capacity": actual_state,
            "kv_capacity": actual_kv,
            "pressure_ratio": mean(ratios),
            "state_utilization": mean(state_utils),
            "kv_utilization": mean(kv_utils),
        }
        for label, weight in policies:
            if label.startswith("scalar-") and weight is not None:
                scalar_weights.add(float(weight))
        lines.append(
            f"| {capacity_label} | {actual_state} | {actual_kv:,} | "
            f"{mean_std(state_utils)} | {mean_std(kv_utils)} | "
            f"{mean_std(ratios)} | {regime} | "
            f"{mean(fcfs_metrics['retractions']):.1f} |"
        )

    has_policy_results = any(
        any(label != "fcfs" for label, _ in cell["policies"])
        for cell in cells.values()
    )
    if not has_policy_results:
        lines.extend(
            [
                "",
                "## Decision",
                "",
                "Calibration only. Run the `screen` stage after the grid contains state-bound, mixed, and KV-bound cells.",
                "",
            ]
        )
        return "\n".join(lines), summary

    lines.extend(
        [
            "",
            "## Per-capacity oracle comparison",
            "",
            "A scalar candidate is eligible only when its mean makespan is no more than 5% above FCFS. "
            "The gap below is scale-free: `|d_state-d_KV| / mean(d_state,d_KV)`.",
            "",
            "| capacity | regime | best eligible scalar | scalar rel. gap | vector rel. gap | vector vs oracle | scalar makespan vs FCFS | vector makespan vs FCFS |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )

    oracle_wins = 0
    meaningful_vector_wins = 0
    meaningful_vector_regime_wins: set[str] = set()
    oracle_gaps: list[float] = []
    vector_gaps: list[float] = []
    for capacity_label in sorted(cells):
        cell = cells[capacity_label]
        policies = cell["policies"]
        fcfs_span = mean(cell["fcfs"]["makespan"])
        scalar_candidates = []
        for (label, weight), items in policies.items():
            if not label.startswith("scalar-"):
                continue
            metrics = aggregate(items)
            if mean(metrics["makespan"]) <= 1.05 * fcfs_span:
                scalar_candidates.append((mean(metrics["relative_gap"]), weight, metrics))
        vector_items = policies.get(("vector", None))
        if not scalar_candidates or not vector_items:
            continue
        scalar_gap, best_weight, scalar_metrics = min(scalar_candidates)
        vector_metrics = aggregate(vector_items)
        vector_gap = mean(vector_metrics["relative_gap"])
        improvement = (scalar_gap - vector_gap) / scalar_gap if scalar_gap else 0.0
        scalar_delta = mean(scalar_metrics["makespan"]) / fcfs_span - 1
        vector_delta = mean(vector_metrics["makespan"]) / fcfs_span - 1
        if improvement > 0:
            oracle_wins += 1
        if improvement >= 0.10:
            meaningful_vector_wins += 1
            meaningful_vector_regime_wins.add(cell["regime"])
        oracle_gaps.append(scalar_gap)
        vector_gaps.append(vector_gap)
        lines.append(
            f"| {capacity_label} | {cell['regime']} | {best_weight:g} | "
            f"{mean_std(scalar_metrics['relative_gap'])} | "
            f"{mean_std(vector_metrics['relative_gap'])} | {improvement * 100:+.1f}% | "
            f"{scalar_delta * 100:+.1f}% | {vector_delta * 100:+.1f}% |"
        )
        summary["capacities"][capacity_label] = {
            "regime": cell["regime"],
            "best_scalar_weight": best_weight,
            "best_scalar_relative_gap": scalar_gap,
            "vector_relative_gap": vector_gap,
            "vector_improvement": improvement,
            "scalar_makespan_delta": scalar_delta,
            "vector_makespan_delta": vector_delta,
        }

    weight_scores: dict[float, list[float]] = defaultdict(list)
    weight_regrets: dict[float, list[float]] = defaultdict(list)
    for capacity_label, cell_summary in summary["capacities"].items():
        cell = cells[capacity_label]
        oracle_gap = cell_summary["best_scalar_relative_gap"]
        for weight in scalar_weights:
            items = cell["policies"].get((f"scalar-w{weight:g}", weight))
            if not items:
                continue
            metrics = aggregate(items)
            gap = mean(metrics["relative_gap"])
            weight_scores[weight].append(gap)
            weight_regrets[weight].append(gap - oracle_gap)

    complete_weights = {
        weight: scores
        for weight, scores in weight_scores.items()
        if len(scores) == len(summary["capacities"])
    }
    lines.extend(["", "## Fixed scalar robustness", ""])
    best_fixed = None
    if complete_weights:
        best_fixed = min(complete_weights, key=lambda item: mean(complete_weights[item]))
        lines.extend(
            [
                f"Best fixed scalar weight across all completed cells: `{best_fixed:g}`; "
                f"mean relative gap {mean(complete_weights[best_fixed]):.3f}, "
                f"worst-cell gap {max(complete_weights[best_fixed]):.3f}, "
                f"mean regret versus per-cell oracle {mean(weight_regrets[best_fixed]):.3f}.",
                "",
            ]
        )
        summary["global_scalar"] = {
            "best_fixed_weight": best_fixed,
            "mean_relative_gap": mean(complete_weights[best_fixed]),
            "worst_relative_gap": max(complete_weights[best_fixed]),
            "mean_oracle_regret": mean(weight_regrets[best_fixed]),
        }
    else:
        lines.extend(["No scalar weight has completed every analyzed capacity cell.", ""])

    mean_oracle_gap = mean(oracle_gaps) if oracle_gaps else float("nan")
    mean_vector_gap = mean(vector_gaps) if vector_gaps else float("nan")
    vector_vs_oracle = (
        (mean_oracle_gap - mean_vector_gap) / mean_oracle_gap
        if mean_oracle_gap
        else 0.0
    )
    total_retractions = sum(
        mean(aggregate(items)["retractions"])
        for cell in cells.values()
        for items in cell["policies"].values()
    )
    lines.extend(
        [
            "## Aggregate checks",
            "",
            f"- Mean per-cell oracle relative gap: {mean_oracle_gap:.3f}; "
            f"mean vector gap: {mean_vector_gap:.3f}; vector change: "
            f"{vector_vs_oracle * 100:+.1f}%.",
            f"- Vector wins by at least 10% in {meaningful_vector_wins} cells "
            f"across {len(meaningful_vector_regime_wins)} regimes: "
            f"{', '.join(sorted(meaningful_vector_regime_wins)) or 'none'}.",
            f"- Total retractions across all analyzed policy runs: {total_retractions:.0f}.",
            "",
        ]
    )

    lines.extend(["## Decision", ""])
    completed_policy_cells = len(summary["capacities"])
    if completed_policy_cells < len(cells):
        lines.append(
            f"Incomplete screening: {completed_policy_cells}/{len(cells)} capacity "
            "cells have both a throughput-eligible scalar sweep and vector result. "
            "Do not apply the pass/kill gate yet."
        )
    elif (
        len(meaningful_vector_regime_wins) >= 2
        and meaningful_vector_wins >= 2
        and vector_vs_oracle >= 0.10
    ):
        lines.append(
            "Screening condition met: vector beats the per-capacity scalar oracle in at least two bottleneck regimes. "
            "Repeat the winning cells and adjacent scalar weights three times before treating this as a paper-level result."
        )
    elif vector_vs_oracle > 0:
        lines.append(
            "Robustness-only evidence: vector improves the cross-cell mean but does "
            "not beat the per-cell scalar oracle by at least 10% across two bottleneck "
            "regimes. This cannot support scalar representational insufficiency."
        )
    else:
        lines.append(
            "Kill condition: a throughput-bounded per-capacity scalar oracle matches or beats vector. "
            "Do not claim that vector accounting is necessary for this workload."
        )
    lines.append("")
    summary["decision"] = {
        "completed_policy_cells": completed_policy_cells,
        "total_capacity_cells": len(cells),
        "vector_oracle_wins": oracle_wins,
        "meaningful_vector_wins": meaningful_vector_wins,
        "meaningful_winning_regimes": sorted(meaningful_vector_regime_wins),
        "mean_oracle_relative_gap": mean_oracle_gap,
        "mean_vector_relative_gap": mean_vector_gap,
        "vector_mean_improvement": vector_vs_oracle,
        "total_retractions": total_retractions,
    }
    return "\n".join(lines), summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary-json", type=Path)
    args = parser.parse_args()
    report, summary = build_report(load(args.input_dir))
    args.output.write_text(report, encoding="utf-8")
    if args.summary_json:
        args.summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
