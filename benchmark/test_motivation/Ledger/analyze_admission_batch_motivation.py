#!/usr/bin/env python3
"""Analyze M2 admission-batch coupling and M3 runtime-cost motivation data."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CELLS = {
    "low-state-auto": {"state": 48, "max_running": 9},
    "high-state-capped": {"state": 480, "max_running": 9},
    "high-state-auto": {"state": 480, "max_running": 96},
}
PARTITION_POLICIES = ("token-vtc", "profiled-vtc", "instant-share")


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower)


def pct(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.1%}"


def num(value: float, digits: int = 2) -> str:
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def load_artifacts(result_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(result_dir.glob("ab-*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if item.get("experiment", {}).get("experiment_stage") == "smoke":
            continue
        item["_path"] = str(path)
        artifacts.append(item)
    return artifacts


def repetition(item: dict[str, Any]) -> int:
    match = re.search(r"rep(\d+)$", item["experiment"]["run_id"])
    return int(match.group(1)) if match else 0


def makespan(item: dict[str, Any]) -> float:
    return max(float(req["finished_s"]) for req in item["requests"])


def group_requests(item: dict[str, Any], group: str) -> list[dict[str, Any]]:
    return [req for req in item["requests"] if req["group"] == group]


def decode_metrics(requests: list[dict[str, Any]]) -> tuple[float, float]:
    intervals = []
    max_gaps = []
    for req in requests:
        events = req.get("token_events", [])
        completion = int(req.get("completion_tokens", 0))
        first = req.get("first_token_s")
        if first is not None and completion > 1:
            intervals.append(
                (float(req["finished_s"]) - float(first)) / (completion - 1)
            )
        times = [float(event[0]) for event in events]
        if len(times) > 1:
            max_gaps.append(max(b - a for a, b in zip(times, times[1:])))
    return mean(intervals), percentile(max_gaps, 0.95)


def batch_metrics(item: dict[str, Any]) -> dict[str, float]:
    trace = item.get("scheduler_batch_trace") or {}
    events = trace.get("events", [])
    decode = [event for event in events if event.get("mode") == "decode"]
    prefill = [
        event
        for event in events
        if event.get("mode") in {"extend", "mixed", "split_prefill"}
    ]
    decode_gaps = [
        float(event["since_previous_launch_us"]) / 1e6
        for event in events
        if event.get("previous_mode") == "decode"
        and event.get("since_previous_launch_us") is not None
    ]
    prefill_gaps = [
        float(event["since_previous_launch_us"]) / 1e6
        for event in events
        if event.get("previous_mode") in {"extend", "mixed", "split_prefill"}
        and event.get("since_previous_launch_us") is not None
    ]
    exposure: dict[str, float] = defaultdict(float)
    for previous, current in zip(events, events[1:]):
        gap = current.get("since_previous_launch_us")
        if gap is None:
            continue
        for tenant, count in previous.get("tenant_counts", {}).items():
            exposure[tenant] += float(count) * float(gap) / 1e6
    return {
        "batch_events": float(len(events)),
        "decode_events": float(len(decode)),
        "prefill_events": float(len(prefill)),
        "peak_batch": float(max((event.get("batch_size", 0) for event in events), default=0)),
        "mean_decode_batch": mean([float(event["batch_size"]) for event in decode]),
        "p95_decode_gap": percentile(decode_gaps, 0.95),
        "p95_prefill_gap": percentile(prefill_gaps, 0.95),
        "peak_waiting": float(
            max((event.get("waiting_requests", 0) for event in events), default=0)
        ),
        "split_batch_exposure": exposure.get("split", 0.0),
        "packed_batch_exposure": exposure.get("packed", 0.0),
        "incumbent_batch_exposure": exposure.get("incumbent", 0.0),
        "pulser_batch_exposure": exposure.get("pulser", 0.0),
    }


def isolation_map(
    artifacts: list[dict[str, Any]], mode: str
) -> dict[tuple[str, int, str], dict[str, Any]]:
    result = {}
    valid = (
        {"split-only", "packed-only"}
        if mode == "partition"
        else {"incumbent-only", "pulser-only"}
    )
    for item in artifacts:
        exp = item["experiment"]
        if exp.get("mode") != mode or exp.get("composition") not in valid:
            continue
        group = exp["composition"].removesuffix("-only")
        result[(exp["cell_id"], repetition(item), group)] = item
    return result


def mixed_metric(
    item: dict[str, Any], isolations: dict[tuple[str, int, str], dict[str, Any]]
) -> dict[str, float]:
    exp = item["experiment"]
    rep = repetition(item)
    split = item["summary"]["groups"]["split"]
    packed = item["summary"]["groups"]["packed"]
    split_iso = isolations[(exp["cell_id"], rep, "split")]["summary"]["groups"]["split"]
    packed_iso = isolations[(exp["cell_id"], rep, "packed")]["summary"]["groups"]["packed"]
    split_requests = group_requests(item, "split")
    packed_requests = group_requests(item, "packed")
    split_interval, split_max_gap = decode_metrics(split_requests)
    packed_interval, packed_max_gap = decode_metrics(packed_requests)
    split_resource = split["scheduler_active_resources"]
    packed_resource = packed["scheduler_active_resources"]
    vtc = item.get("scheduler_vtc_trace") or {}
    queue = vtc.get("queue_waits", {})
    values = {
        "split_p95": float(split["p95_jct_s"]),
        "packed_p95": float(packed["p95_jct_s"]),
        "split_iso_p95": float(split_iso["p95_jct_s"]),
        "packed_iso_p95": float(packed_iso["p95_jct_s"]),
        "split_slowdown": float(split["p95_jct_s"]) / float(split_iso["p95_jct_s"]) - 1,
        "packed_slowdown": float(packed["p95_jct_s"]) / float(packed_iso["p95_jct_s"]) - 1,
        "makespan": makespan(item),
        "peak_active": float(
            item["scheduler_resource_trace"]["system"]["peak_active_requests"]
        ),
        "split_active_per_req": float(split_resource["active_request_seconds"])
        / len(split_requests),
        "packed_active_per_req": float(packed_resource["active_request_seconds"])
        / len(packed_requests),
        "split_decode_interval": split_interval,
        "packed_decode_interval": packed_interval,
        "split_p95_max_itl": split_max_gap,
        "packed_p95_max_itl": packed_max_gap,
        "split_queue_p95": float(queue.get("split", {}).get("p95_seconds", math.nan)),
        "packed_queue_p95": float(queue.get("packed", {}).get("p95_seconds", math.nan)),
    }
    values["gap"] = abs(values["split_slowdown"] - values["packed_slowdown"])
    values.update(batch_metrics(item))
    return values


def delivered_service_metric(item: dict[str, Any]) -> dict[str, float]:
    groups = ("incumbent", "pulser")
    totals = {}
    events: list[tuple[float, str, float]] = []
    completion_times = {}
    for group in groups:
        requests = group_requests(item, group)
        totals[group] = sum(
            int(req["prompt_tokens"]) + 2 * int(req["completion_tokens"])
            for req in requests
        )
        completion_times[group] = max(float(req["finished_s"]) for req in requests)
        for req in requests:
            first = req.get("first_token_s")
            if first is not None:
                events.append((float(first), group, float(req["prompt_tokens"])))
            previous = 0
            for timestamp, completed in req.get("token_events", []):
                delta = int(completed) - previous
                if delta > 0:
                    events.append((float(timestamp), group, float(2 * delta)))
                    previous = int(completed)

    start = max(float(req["scheduled_arrival_s"]) for req in item["requests"])
    end = min(completion_times.values())
    delivered = {group: 0.0 for group in groups}
    for timestamp, group, amount in sorted(events):
        if timestamp <= start:
            delivered[group] += amount
    denominator = max(float(totals[group]) for group in groups)
    previous_t = start
    previous_gap = abs(delivered["incumbent"] - delivered["pulser"]) / denominator
    max_gap = previous_gap
    max_pulser_lead = (delivered["pulser"] - delivered["incumbent"]) / denominator
    gap_area = 0.0
    for timestamp, group, amount in sorted(events):
        if timestamp <= start or timestamp > end:
            continue
        gap_area += previous_gap * (timestamp - previous_t)
        delivered[group] += amount
        previous_t = timestamp
        previous_gap = abs(delivered["incumbent"] - delivered["pulser"]) / denominator
        max_gap = max(max_gap, previous_gap)
        max_pulser_lead = max(
            max_pulser_lead,
            (delivered["pulser"] - delivered["incumbent"]) / denominator,
        )
    if end > previous_t:
        gap_area += previous_gap * (end - previous_t)
    duration = max(end - start, 1e-12)
    return {
        "service_max_gap": max_gap,
        "service_gap_area": gap_area / duration,
        "max_pulser_lead": max_pulser_lead,
        "service_window_s": duration,
        "incumbent_finish": completion_times["incumbent"],
        "pulser_finish": completion_times["pulser"],
    }


def pulse_metric(
    item: dict[str, Any], isolations: dict[tuple[str, int, str], dict[str, Any]]
) -> dict[str, float]:
    exp = item["experiment"]
    rep = repetition(item)
    incumbent = item["summary"]["groups"]["incumbent"]
    pulser = item["summary"]["groups"]["pulser"]
    incumbent_iso = isolations[(exp["cell_id"], rep, "incumbent")]["summary"]["groups"]["incumbent"]
    pulser_iso = isolations[(exp["cell_id"], rep, "pulser")]["summary"]["groups"]["pulser"]
    values = delivered_service_metric(item)
    values.update(
        {
            "incumbent_slowdown": float(incumbent["p95_jct_s"])
            / float(incumbent_iso["p95_jct_s"])
            - 1,
            "pulser_slowdown": float(pulser["p95_jct_s"])
            / float(pulser_iso["p95_jct_s"])
            - 1,
            "makespan": makespan(item),
        }
    )
    values["gap"] = abs(values["incumbent_slowdown"] - values["pulser_slowdown"])
    values.update(batch_metrics(item))
    return values


def aggregate(rows: list[dict[str, float]]) -> dict[str, float]:
    return {key: mean([row[key] for row in rows]) for key in rows[0]} if rows else {}


def collect_partition(artifacts: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    isolations = isolation_map(artifacts, "partition")
    grouped: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(lambda: defaultdict(list))
    for item in artifacts:
        exp = item["experiment"]
        if exp.get("mode") == "partition" and exp.get("composition") == "mixed":
            grouped[exp["cell_id"]][exp["policy_label"]].append(
                mixed_metric(item, isolations)
            )
    return {
        cell: {policy: aggregate(rows) for policy, rows in policies.items()}
        for cell, policies in grouped.items()
    }


def collect_pulse(artifacts: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    isolations = isolation_map(artifacts, "pulse")
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in artifacts:
        exp = item["experiment"]
        if exp.get("mode") == "pulse" and exp.get("composition") == "mixed":
            grouped[exp["policy_label"]].append(pulse_metric(item, isolations))
    return {policy: aggregate(rows) for policy, rows in grouped.items()}


def integrity(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    requests = [request for item in artifacts for request in item.get("requests", [])]
    matrices = Counter()
    capacities_ok = True
    equal_token_mixed = 0
    for item in artifacts:
        exp = item["experiment"]
        matrices[(exp["mode"], exp["cell_id"], exp["composition"], exp["policy_label"], repetition(item))] += 1
        expected = (
            CELLS.get(exp["cell_id"])
            if exp["mode"] == "partition"
            else {"state": 480, "max_running": 4}
        )
        states = item["server"].get("internal_states", [])
        capacities_ok = capacities_ok and bool(states) and all(
            int(state["max_mamba_cache_size"]) == expected["state"]
            and int(state["effective_max_running_requests_per_dp"])
            == expected["max_running"]
            for state in states
        )
        if exp["composition"] == "mixed":
            groups = (
                ("split", "packed")
                if exp["mode"] == "partition"
                else ("incumbent", "pulser")
            )
            totals = {
                group: (
                    sum(int(req["prompt_tokens"]) for req in item["requests"] if req["group"] == group),
                    sum(int(req["completion_tokens"]) for req in item["requests"] if req["group"] == group),
                )
                for group in groups
            }
            equal_token_mixed += int(totals[groups[0]] == totals[groups[1]])

    expected_matrix = set()
    for cell in CELLS:
        for rep in (1, 2, 3):
            expected_matrix.add(("partition", cell, "split-only", "fcfs-split-only", rep))
            expected_matrix.add(("partition", cell, "packed-only", "fcfs-packed-only", rep))
            for policy in PARTITION_POLICIES:
                expected_matrix.add(("partition", cell, "mixed", policy, rep))
    for rep in (1, 2, 3):
        expected_matrix.add(("pulse", "pulse-capped4", "incumbent-only", "fcfs-incumbent-only", rep))
        expected_matrix.add(("pulse", "pulse-capped4", "pulser-only", "fcfs-pulser-only", rep))
        expected_matrix.add(("pulse", "pulse-capped4", "mixed", "token-vtc", rep))
        expected_matrix.add(("pulse", "pulse-capped4", "mixed", "instant-share", rep))
    matrix_ok = set(matrices) == expected_matrix and all(matrices[key] == 1 for key in expected_matrix)
    return {
        "artifacts": len(artifacts),
        "requests": len(requests),
        "failures": sum(not request.get("success", False) for request in requests),
        "cached_tokens": sum(int(request.get("cached_tokens", 0)) for request in requests),
        "retractions": sum(int(request.get("num_retractions", 0)) for request in requests),
        "resource_traces": sum(bool(item.get("scheduler_resource_trace")) for item in artifacts),
        "batch_traces": sum(bool(item.get("scheduler_batch_trace")) for item in artifacts),
        "dropped_batch_events": sum(
            int((item.get("scheduler_batch_trace") or {}).get("dropped_events", 0))
            for item in artifacts
        ),
        "resolved": sum(bool(item["server"].get("resolved_experiment_policy")) for item in artifacts),
        "gpu": sum(item["experiment"].get("physical_gpus") == "6,7" for item in artifacts),
        "equal_token_mixed": equal_token_mixed,
        "mixed": sum(item["experiment"].get("composition") == "mixed" for item in artifacts),
        "capacities_ok": capacities_ok,
        "matrix_ok": matrix_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args()

    artifacts = load_artifacts(args.result_dir)
    audit = integrity(artifacts)
    partition = collect_partition(artifacts)
    pulse = collect_pulse(artifacts)
    integrity_pass = (
        audit["artifacts"] == 57
        and audit["failures"] == 0
        and audit["cached_tokens"] == 0
        and audit["retractions"] == 0
        and audit["resource_traces"] == audit["artifacts"]
        and audit["batch_traces"] == audit["artifacts"]
        and audit["dropped_batch_events"] == 0
        and audit["resolved"] == audit["artifacts"]
        and audit["gpu"] == audit["artifacts"]
        and audit["equal_token_mixed"] == audit["mixed"]
        and audit["capacities_ok"]
        and audit["matrix_ok"]
    )

    decisions: dict[str, Any] = {"complete": False}
    lines = [
        "# Admission-batch and runtime-cost motivation report",
        "",
        "## Integrity",
        "",
        f"Artifacts {audit['artifacts']}; requests {audit['requests']}; failures "
        f"{audit['failures']}; cached tokens {audit['cached_tokens']}; retractions "
        f"{audit['retractions']}; exact resource traces {audit['resource_traces']}/"
        f"{audit['artifacts']}; batch traces {audit['batch_traces']}/{audit['artifacts']}; "
        f"dropped batch events {audit['dropped_batch_events']}; resolved policy metadata "
        f"{audit['resolved']}/{audit['artifacts']}; equal-token mixed "
        f"{audit['equal_token_mixed']}/{audit['mixed']}.",
        "",
    ]

    complete = all(
        cell in partition and all(policy in partition[cell] for policy in PARTITION_POLICIES)
        for cell in CELLS
    ) and all(policy in pulse for policy in ("token-vtc", "instant-share"))
    if complete:
        c0 = partition["low-state-auto"]["token-vtc"]
        c1 = partition["high-state-capped"]["token-vtc"]
        c2 = partition["high-state-auto"]["token-vtc"]
        fixed_jct_change = mean(
            [abs(c1["split_p95"] / c0["split_p95"] - 1), abs(c1["packed_p95"] / c0["packed_p95"] - 1)]
        )
        peak_ratio = c2["peak_active"] / max(c1["peak_active"], 1e-12)
        queue_reduction = 1 - c2["split_queue_p95"] / c1["split_queue_p95"]
        active_inflation = c2["split_active_per_req"] / c1["split_active_per_req"]
        decode_inflation = c2["split_decode_interval"] / c1["split_decode_interval"]
        packed_externality = c2["packed_p95"] / c1["packed_p95"] - 1
        m2_pass = (
            fixed_jct_change <= 0.10
            and peak_ratio >= 4
            and queue_reduction >= 0.50
            and max(active_inflation, decode_inflation) >= 2
            and packed_externality >= 0.15
        )
        decisions["m2"] = {
            "passed": m2_pass,
            "fixed_concurrency_mean_abs_jct_change": fixed_jct_change,
            "peak_active_ratio": peak_ratio,
            "split_queue_reduction": queue_reduction,
            "split_active_inflation": active_inflation,
            "split_decode_interval_inflation": decode_inflation,
            "packed_p95_externality": packed_externality,
        }
        lines.extend(
            [
                "## M2: admission-batch coupling",
                "",
                "| cell | state / max-running | split/packed p95 JCT | peak active | split active/request | split decode interval | split queue p95 | mean decode batch |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for cell in CELLS:
            value = partition[cell]["token-vtc"]
            expected = CELLS[cell]
            lines.append(
                f"| {cell} | {expected['state']} / {expected['max_running']} | "
                f"{num(value['split_p95'])}/{num(value['packed_p95'])} | "
                f"{num(value['peak_active'], 0)} | {num(value['split_active_per_req'])} | "
                f"{num(value['split_decode_interval'], 4)} | {num(value['split_queue_p95'])} | "
                f"{num(value['mean_decode_batch'])} |"
            )
        lines.extend(
            [
                "",
                f"Decision: **{'PASS' if m2_pass else 'FAIL'}**. At fixed max-running=9, "
                f"raising state 48->480 changes mean tenant p95 JCT by {pct(fixed_jct_change)}. "
                f"At fixed state=480, raising max-running 9->96 changes peak active by "
                f"{num(peak_ratio)}x, reduces split queue p95 by {pct(queue_reduction)}, "
                f"inflates split active residency by {num(active_inflation)}x and decode "
                f"interval by {num(decode_inflation)}x, while packed p95 changes by "
                f"{pct(packed_externality)}.",
                "",
                "## M3a: static cost versus runtime resource signal",
                "",
                "| cell | token gap | profiled gap | instant gap | instant improvement over strongest scalar |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        instant_improvements = {}
        profiled_improvements = {}
        for cell in CELLS:
            token = partition[cell]["token-vtc"]
            profiled = partition[cell]["profiled-vtc"]
            instant = partition[cell]["instant-share"]
            strongest = min(token["gap"], profiled["gap"])
            instant_improvement = (strongest - instant["gap"]) / max(strongest, 1e-12)
            profiled_improvement = (token["gap"] - profiled["gap"]) / max(token["gap"], 1e-12)
            instant_improvements[cell] = instant_improvement
            profiled_improvements[cell] = profiled_improvement
            lines.append(
                f"| {cell} | {pct(token['gap'])} | {pct(profiled['gap'])} | "
                f"{pct(instant['gap'])} | {pct(instant_improvement)} |"
            )
        capped_improvement = mean(
            [instant_improvements["low-state-auto"], instant_improvements["high-state-capped"]]
        )
        uncapped_improvement = instant_improvements["high-state-auto"]
        profiled_mean = mean(list(profiled_improvements.values()))
        m3a_pass = (
            profiled_mean < 0.15
            and capped_improvement >= 0.15
            and uncapped_improvement <= 0.10
        )
        decisions["m3a"] = {
            "passed": m3a_pass,
            "profiled_mean_improvement": profiled_mean,
            "instant_capped_mean_improvement": capped_improvement,
            "instant_uncapped_improvement": uncapped_improvement,
        }
        lines.extend(
            [
                "",
                f"Decision: **{'PASS' if m3a_pass else 'FAIL'}**. Profiled scalar mean "
                f"improvement over token VTC is {pct(profiled_mean)}; instant-share mean "
                f"improvement in the two max-running=9 cells is {pct(capped_improvement)}, "
                f"but only {pct(uncapped_improvement)} when max-running=96.",
                "",
                "## M3b: instantaneous-share history reset probe",
                "",
                "| policy | max delivered-service gap | gap area | max pulser lead | slowdown gap | makespan |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for policy in ("token-vtc", "instant-share"):
            value = pulse[policy]
            lines.append(
                f"| {policy} | {pct(value['service_max_gap'])} | "
                f"{pct(value['service_gap_area'])} | {pct(value['max_pulser_lead'])} | "
                f"{pct(value['gap'])} | {num(value['makespan'])} |"
            )
        token_pulse = pulse["token-vtc"]
        instant_pulse = pulse["instant-share"]
        max_gap_ratio = instant_pulse["service_max_gap"] / max(token_pulse["service_max_gap"], 1e-12)
        area_ratio = instant_pulse["service_gap_area"] / max(token_pulse["service_gap_area"], 1e-12)
        pulse_makespan = instant_pulse["makespan"] / token_pulse["makespan"] - 1
        m3b_pass = max(max_gap_ratio, area_ratio) >= 1.20 and pulse_makespan <= 0.05
        decisions["m3b"] = {
            "passed": m3b_pass,
            "max_service_gap_ratio": max_gap_ratio,
            "service_gap_area_ratio": area_ratio,
            "instant_makespan_change": pulse_makespan,
        }
        decisions["m3"] = {"passed": m3a_pass and m3b_pass}
        decisions["complete"] = True
        lines.extend(
            [
                "",
                f"Decision: **{'PASS' if m3b_pass else 'FAIL'}**. Instant-share/token-VTC "
                f"maximum service-gap ratio is {num(max_gap_ratio)}x, normalized gap-area "
                f"ratio is {num(area_ratio)}x, and makespan change is {pct(pulse_makespan)}.",
                "",
            ]
        )
    else:
        lines.extend(["Formal matrix incomplete.", ""])

    overall = bool(
        integrity_pass
        and decisions.get("m2", {}).get("passed")
        and decisions.get("m3", {}).get("passed")
    )
    lines.extend(
        [
            "## Overall",
            "",
            f"Motivation decision: **{'PASS' if overall else 'FAIL'}**. M2 and M3 are "
            "reported separately; a failed sub-gate is retained as a falsified explanation, "
            "not rewritten post hoc.",
            "",
        ]
    )
    report = "\n".join(lines)
    compact = {
        "integrity": audit,
        "integrity_pass": integrity_pass,
        "partition": partition,
        "pulse": pulse,
        "decisions": decisions,
        "overall": overall,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    if args.analysis_output:
        args.analysis_output.write_text(
            json.dumps(compact, indent=2) + "\n", encoding="utf-8"
        )
    print(report)


if __name__ == "__main__":
    main()
