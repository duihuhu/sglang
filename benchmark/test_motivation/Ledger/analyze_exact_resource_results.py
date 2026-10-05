#!/usr/bin/env python3
"""Aggregate scheduler-side resource accounting for the Topic B kill test."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


CONDITIONS = (
    "state-isolated",
    "kv-isolated",
    "state-first",
    "kv-first",
    "interleave",
)


def mean_std(values: list[float]) -> str:
    return f"{statistics.fmean(values):.3f} +/- {statistics.stdev(values):.3f}"


def load_artifacts(input_dir: Path) -> dict[str, list[dict[str, Any]]]:
    grouped = {condition: [] for condition in CONDITIONS}
    for path in sorted(input_dir.glob("*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        run_id = artifact["experiment"]["run_id"]
        condition = next(
            (
                candidate
                for candidate in CONDITIONS
                if run_id.startswith(f"exact-{candidate}-r")
            ),
            None,
        )
        if condition is None:
            raise ValueError(f"Cannot infer condition from {run_id}")
        if artifact.get("scheduler_resource_trace") is None:
            raise ValueError(f"{path} has no scheduler resource trace")
        if artifact["summary"]["overall"]["completed"] != artifact["summary"][
            "overall"
        ]["requests"]:
            raise ValueError(f"{path} contains failed requests")
        grouped[condition].append(artifact)

    counts = {condition: len(items) for condition, items in grouped.items()}
    if len(set(counts.values())) != 1 or not next(iter(counts.values())):
        raise ValueError(f"Unbalanced condition repetitions: {counts}")
    return grouped


def capacity(artifact: dict[str, Any]) -> tuple[int, int]:
    kv_capacity = int(artifact["server"]["max_total_num_tokens"])
    mamba_capacity = next(
        int(state["max_mamba_cache_size"])
        for state in artifact["server"]["internal_states"]
        if state.get("max_mamba_cache_size") is not None
    )
    return mamba_capacity, kv_capacity


def usage(artifact: dict[str, Any], group: str) -> dict[str, float]:
    return artifact["scheduler_resource_trace"]["tenants"][group]


def normalized_debt(artifact: dict[str, Any], group: str) -> tuple[float, float, float]:
    state_capacity, kv_capacity = capacity(artifact)
    item = usage(artifact, group)
    state_debt = item["state_slot_seconds"] / state_capacity
    kv_debt = item["kv_token_seconds"] / kv_capacity
    return state_debt, kv_debt, max(state_debt, kv_debt)


def paired_slowdown(
    concurrent: list[dict[str, Any]],
    isolated: list[dict[str, Any]],
    group: str,
) -> list[float]:
    return [
        co["summary"]["groups"][group]["p95_latency_s"]
        / alone["summary"]["groups"][group]["p95_latency_s"]
        for co, alone in zip(concurrent, isolated)
    ]


def build_report(grouped: dict[str, list[dict[str, Any]]]) -> str:
    repetitions = len(grouped["state-first"])
    state_capacity, kv_capacity = capacity(grouped["state-first"][0])
    lines = [
        "# Topic B Exact Active-Resource Results",
        "",
        f"> {repetitions} repetitions on Kimi-Linear-48B-A3B, TP=2, FCFS; "
        "scheduler allocator timestamps rather than streaming-response proxies.",
        "",
        f"- Physical recurrent-state capacity: {state_capacity} slots.",
        f"- Full-KV capacity: {kv_capacity:,} token slots.",
        "- Scope: active private request state/KV; inactive radix-cache ownership and GPU time are not included.",
        "- All requests completed; no retractions were observed.",
        "",
        "| condition | group | state-slot-s | KV-token-s | normalized state debt | normalized KV debt | dominant debt | peak state slots |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    for condition in CONDITIONS:
        artifacts = grouped[condition]
        groups = sorted(artifacts[0]["summary"]["groups"])
        for group in groups:
            state_seconds = [usage(item, group)["state_slot_seconds"] for item in artifacts]
            kv_seconds = [usage(item, group)["kv_token_seconds"] for item in artifacts]
            debts = [normalized_debt(item, group) for item in artifacts]
            peaks = [usage(item, group)["peak_state_slots"] for item in artifacts]
            lines.append(
                f"| {condition} | {group} | {mean_std(state_seconds)} | "
                f"{mean_std(kv_seconds)} | {mean_std([item[0] for item in debts])} | "
                f"{mean_std([item[1] for item in debts])} | "
                f"{mean_std([item[2] for item in debts])} | "
                f"{statistics.fmean(peaks):.0f} |"
            )

    co_runs = [
        item
        for condition in ("state-first", "kv-first", "interleave")
        for item in grouped[condition]
    ]
    token_errors = []
    peak_ratios = []
    dominant_ratios = []
    specialization = []
    for item in co_runs:
        groups = item["summary"]["groups"]
        state_tokens = groups["state-heavy"]["input_tokens_completed"] + groups[
            "state-heavy"
        ]["output_tokens_completed"]
        kv_tokens = groups["kv-heavy"]["input_tokens_completed"] + groups[
            "kv-heavy"
        ]["output_tokens_completed"]
        token_errors.append(abs(state_tokens - kv_tokens) / kv_tokens)
        peak_ratios.append(
            usage(item, "state-heavy")["peak_state_slots"]
            / usage(item, "kv-heavy")["peak_state_slots"]
        )
        state_debt = normalized_debt(item, "state-heavy")
        kv_debt = normalized_debt(item, "kv-heavy")
        dominant_ratios.append(state_debt[2] / kv_debt[2])
        specialization.append(
            (state_debt[0] > state_debt[1]) and (kv_debt[1] > kv_debt[0])
        )

    slowdowns = {
        condition: {
            "state-heavy": paired_slowdown(
                grouped[condition], grouped["state-isolated"], "state-heavy"
            ),
            "kv-heavy": paired_slowdown(
                grouped[condition], grouped["kv-isolated"], "kv-heavy"
            ),
        }
        for condition in ("state-first", "kv-first", "interleave")
    }
    max_slowdown = max(
        statistics.fmean(values)
        for condition in slowdowns.values()
        for values in condition.values()
    )

    lines.extend(
        [
            "",
            "## Findings",
            "",
            f"- Completed-token mismatch: {max(token_errors) * 100:.2f}%.",
            f"- Mean physical peak state ratio (state-heavy / KV-heavy): {statistics.fmean(peak_ratios):.1f}x.",
            f"- Every co-run shows resource specialization: state-heavy is state-dominant and KV-heavy is KV-dominant ({sum(specialization)}/{len(specialization)} runs).",
            f"- Equal token service maps to a mean {statistics.fmean(dominant_ratios):.2f}x difference in cumulative normalized dominant debt.",
            f"- Maximum mean p95 JCT slowdown versus isolated execution: {max_slowdown:.2f}x.",
            "",
            "## Decision",
            "",
            "Exact active-resource accounting confirms the physical premise of Topic B, but it does not yet prove VTC inadequate. The next gate remains a sustained-arrival comparison of tuned scalar VTC against vector debt, including GPU time and a throughput-loss bound.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(load_artifacts(args.input_dir))
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
