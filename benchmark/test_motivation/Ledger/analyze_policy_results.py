#!/usr/bin/env python3
"""Compare FCFS, VTC, tuned scalar VTC, and vector resource debt."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


POLICIES = (
    "fcfs",
    "vtc-token",
    "vtc-r8076",
    "vtc-r13460",
    "vector",
)


def infer_policy(run_id: str) -> str:
    for policy in sorted(POLICIES, key=len, reverse=True):
        if run_id.startswith(f"policy-{policy}-r"):
            return policy
    raise ValueError(f"Cannot infer policy from {run_id}")


def load(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped = {policy: [] for policy in POLICIES}
    for path in sorted(input_dir.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        policy = infer_policy(artifact["experiment"]["run_id"])
        if artifact["summary"]["overall"]["completed"] != artifact["summary"][
            "overall"
        ]["requests"]:
            raise ValueError(f"{path} contains failed requests")
        grouped[policy].append(artifact)
    counts = {policy: len(items) for policy, items in grouped.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise ValueError(f"Unbalanced repetitions: {counts}")
    return grouped


def mean_std(values: list[float], digits: int = 2) -> str:
    return (
        f"{statistics.fmean(values):.{digits}f} +/- "
        f"{statistics.stdev(values):.{digits}f}"
    )


def capacities(artifact: dict[str, Any]) -> tuple[int, int]:
    state_capacity = next(
        int(state["max_mamba_cache_size"])
        for state in artifact["server"]["internal_states"]
        if state.get("max_mamba_cache_size") is not None
    )
    return state_capacity, int(artifact["server"]["max_total_num_tokens"])


def dominant_debts(artifact: dict[str, Any]) -> tuple[float, float]:
    state_capacity, kv_capacity = capacities(artifact)
    tenants = artifact["scheduler_resource_trace"]["tenants"]

    def debt(group: str) -> float:
        usage = tenants[group]
        return max(
            usage["state_slot_seconds"] / state_capacity,
            usage["kv_token_seconds"] / kv_capacity,
        )

    return debt("state-heavy"), debt("kv-heavy")


def service_lag(artifact: dict[str, Any]) -> tuple[int, float]:
    events = []
    for req in artifact["requests"]:
        previous = 0
        for index, (event_time, count) in enumerate(req["token_events"]):
            delta = count - previous
            if index == 0:
                delta += req["input_len"]
            events.append((event_time, req["group"], delta))
            previous = count
    events.sort()

    service = {"state-heavy": 0, "kv-heavy": 0}
    max_lag = 0
    lag_area = 0.0
    previous_time = events[0][0]
    for event_time, group, delta in events:
        lag_area += abs(service["state-heavy"] - service["kv-heavy"]) * (
            event_time - previous_time
        )
        service[group] += delta
        max_lag = max(max_lag, abs(service["state-heavy"] - service["kv-heavy"]))
        previous_time = event_time
    return max_lag, lag_area


def makespan(artifact: dict[str, Any]) -> float:
    return max(req["finished_s"] for req in artifact["requests"]) - min(
        req["submitted_s"] for req in artifact["requests"]
    )


def metrics(artifacts: list[dict[str, Any]]) -> dict[str, list[float]]:
    debts = [dominant_debts(item) for item in artifacts]
    lags = [service_lag(item) for item in artifacts]
    return {
        "state_p95": [
            item["summary"]["groups"]["state-heavy"]["p95_latency_s"]
            for item in artifacts
        ],
        "kv_p95": [
            item["summary"]["groups"]["kv-heavy"]["p95_latency_s"]
            for item in artifacts
        ],
        "goodput": [
            item["summary"]["overall"]["completed_within_slo"]
            for item in artifacts
        ],
        "makespan": [makespan(item) for item in artifacts],
        "max_token_lag": [item[0] for item in lags],
        "token_lag_area": [item[1] for item in lags],
        "state_debt": [item[0] for item in debts],
        "kv_debt": [item[1] for item in debts],
        "debt_gap": [abs(item[0] - item[1]) for item in debts],
    }


def build_report(grouped: dict[str, list[dict[str, Any]]]) -> str:
    by_policy = {policy: metrics(items) for policy, items in grouped.items()}
    fcfs_makespan = statistics.fmean(by_policy["fcfs"]["makespan"])
    lines = [
        "# Topic B Sustained Policy Kill Test",
        "",
        "> 12 waves per run, 16 state-heavy plus 1 KV-heavy request per wave, "
        "3 repetitions, identical Kimi-Linear TP=2 capacity and arrival trace.",
        "",
        "| policy | state p95 JCT (s) | KV p95 JCT (s) | <=20s goodput | makespan (s) | vs FCFS makespan | max token lag | dominant-debt gap |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy in POLICIES:
        item = by_policy[policy]
        mean_makespan = statistics.fmean(item["makespan"])
        makespan_delta = (mean_makespan / fcfs_makespan - 1) * 100
        lines.append(
            f"| {policy} | {mean_std(item['state_p95'])} | "
            f"{mean_std(item['kv_p95'])} | "
            f"{statistics.fmean(item['goodput']):.1f} | "
            f"{mean_std(item['makespan'])} | {makespan_delta:+.1f}% | "
            f"{statistics.fmean(item['max_token_lag']):,.0f} | "
            f"{mean_std(item['debt_gap'])} |"
        )

    scalar_policies = ("vtc-token", "vtc-r8076", "vtc-r13460")
    best_scalar = min(
        scalar_policies,
        key=lambda policy: statistics.fmean(by_policy[policy]["debt_gap"]),
    )
    best_scalar_gap = statistics.fmean(by_policy[best_scalar]["debt_gap"])
    vector_gap = statistics.fmean(by_policy["vector"]["debt_gap"])
    vector_gap_improvement = (best_scalar_gap - vector_gap) / best_scalar_gap
    vector_makespan_delta = (
        statistics.fmean(by_policy["vector"]["makespan"]) / fcfs_makespan - 1
    )
    vector_goodput_delta = statistics.fmean(
        by_policy["vector"]["goodput"]
    ) - statistics.fmean(by_policy["fcfs"]["goodput"])

    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Best scalar baseline: `{best_scalar}`, dominant-debt gap {best_scalar_gap:.2f}.",
            f"- Vector debt gap: {vector_gap:.2f}, {vector_gap_improvement * 100:.1f}% below the best tested scalar VTC.",
            f"- Vector makespan change versus FCFS: {vector_makespan_delta * 100:+.1f}%; <=20s goodput change: {vector_goodput_delta:+.1f} requests.",
            "- Token VTC keeps token-service lag low, while vector debt intentionally permits token lag to equalize normalized physical resource use. These are different fairness objectives.",
            "- Scalar tuning is sensitive: a physical-slot conversion (8,076) is materially better than token-only VTC, while the 5-slot admission conversion (13,460) regresses close to FCFS on debt gap.",
            "",
            "## Decision",
            "",
            "This is a **partial pass**, not a paper-level pass. At the current capacity point vector debt beats all tested scalar VTC settings without throughput loss, but GPU-time accounting, broader scalar tuning, state/KV bottleneck sweeps, request-splitting resistance, and a fairness-efficiency frontier remain required.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(load(args.input_dir))
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
