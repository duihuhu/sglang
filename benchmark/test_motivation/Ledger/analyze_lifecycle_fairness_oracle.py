#!/usr/bin/env python3
"""Select and compare token, tuned-scalar, and realized-cost fair schedulers."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_artifacts(result_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(result_dir.glob("fair-*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    return artifacts


def repetition(artifact: dict[str, Any]) -> int:
    match = re.search(r"rep(\d+)$", artifact["experiment"]["run_id"])
    return int(match.group(1)) if match else 0


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def fmt(value: float, digits: int = 3) -> str:
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def pct(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.1%}"


def makespan(artifact: dict[str, Any]) -> float:
    return max(float(request["finished_s"]) for request in artifact["requests"])


def bootstrap_mean_ci(values: list[float], samples: int = 20000) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(20260825)
    means = sorted(
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    )
    return means[int(0.025 * samples)], means[int(0.975 * samples)]


def policy_label(artifact: dict[str, Any]) -> str:
    experiment = artifact["experiment"]
    policy = experiment["runtime_policy"]
    if policy == "vtc":
        weight = float(experiment.get("request_weight", 0))
        if weight == 0 and "scalar" not in experiment["run_id"]:
            return "token-vtc"
        return f"scalar-vtc-r{weight:g}"
    if policy == "lifecycle":
        state = float(experiment.get("lifecycle_state_weight", 1))
        kv = float(experiment.get("lifecycle_kv_weight", 1))
        return f"realized-cost-s{state:g}-k{kv:g}"
    return policy


def isolation_map(
    artifacts: list[dict[str, Any]], stage: str
) -> dict[tuple[int, str], dict[str, Any]]:
    result = {}
    for item in artifacts:
        experiment = item["experiment"]
        if experiment.get("experiment_stage") != stage:
            continue
        composition = experiment.get("composition")
        if composition in {"split-only", "packed-only"}:
            tenant = composition.removesuffix("-only")
            result[(repetition(item), tenant)] = item
    return result


def mixed_metric(
    artifact: dict[str, Any], isolations: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, float]:
    rep = repetition(artifact)
    split_iso = isolations[(rep, "split")]["summary"]["groups"]["split"]
    packed_iso = isolations[(rep, "packed")]["summary"]["groups"]["packed"]
    groups = artifact["summary"]["groups"]
    split_resources = groups["split"].get("scheduler_active_resources", {})
    packed_resources = groups["packed"].get("scheduler_active_resources", {})
    split_slowdown = (
        float(groups["split"]["p95_jct_s"]) / float(split_iso["p95_jct_s"]) - 1
    )
    packed_slowdown = (
        float(groups["packed"]["p95_jct_s"])
        / float(packed_iso["p95_jct_s"])
        - 1
    )
    return {
        "rep": float(rep),
        "split_p95": float(groups["split"]["p95_jct_s"]),
        "packed_p95": float(groups["packed"]["p95_jct_s"]),
        "split_slowdown": split_slowdown,
        "packed_slowdown": packed_slowdown,
        "slowdown_gap": abs(split_slowdown - packed_slowdown),
        "makespan": makespan(artifact),
        "slo_goodput": float(artifact["summary"]["overall"]["slo_goodput"]),
        "retractions": float(
            artifact["summary"]["overall"].get("total_retractions", 0)
        ),
        "split_state": float(split_resources.get("state_slot_seconds", 0)),
        "split_kv": float(split_resources.get("kv_token_seconds", 0)),
        "packed_state": float(packed_resources.get("state_slot_seconds", 0)),
        "packed_kv": float(packed_resources.get("kv_token_seconds", 0)),
    }


def calibration_rows(
    artifacts: list[dict[str, Any]], isolations: dict[tuple[int, str], dict[str, Any]]
) -> dict[str, dict[str, float]]:
    rows = {}
    for item in artifacts:
        experiment = item["experiment"]
        if (
            experiment.get("experiment_stage") != "calibration"
            or experiment.get("composition") != "mixed"
        ):
            continue
        rows[policy_label(item)] = mixed_metric(item, isolations)
    return rows


def select_oracles(rows: dict[str, dict[str, float]]) -> dict[str, Any]:
    token = rows["token-vtc"]
    rows.setdefault("scalar-vtc-r0", dict(token))

    def choose(prefix: str) -> str:
        candidates = {
            label: values
            for label, values in rows.items()
            if label.startswith(prefix) and values["makespan"] <= 1.05 * token["makespan"]
        }
        if not candidates:
            raise ValueError(f"no throughput-bounded {prefix} calibration candidate")
        return min(
            candidates,
            key=lambda label: (
                candidates[label]["slowdown_gap"],
                candidates[label]["makespan"],
            ),
        )

    scalar_label = choose("scalar-vtc-")
    lifecycle_label = choose("realized-cost-")
    scalar_weight = float(scalar_label.rsplit("r", 1)[1])
    match = re.fullmatch(r"realized-cost-s([0-9.]+)-k([0-9.]+)", lifecycle_label)
    if match is None:
        raise ValueError(lifecycle_label)
    return {
        "scalar_label": scalar_label,
        "scalar_request_weight": scalar_weight,
        "lifecycle_label": lifecycle_label,
        "lifecycle_state_weight": float(match.group(1)),
        "lifecycle_kv_weight": float(match.group(2)),
    }


def render_calibration(
    rows: dict[str, dict[str, float]], selection: dict[str, Any]
) -> list[str]:
    lines = [
        "## Calibration",
        "",
        "| policy | split slowdown | packed slowdown | slowdown gap | makespan (s) | SLO goodput | selected |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    selected = {"token-vtc", selection["scalar_label"], selection["lifecycle_label"]}
    for label, values in sorted(rows.items()):
        lines.append(
            f"| {label} | {pct(values['split_slowdown'])} | "
            f"{pct(values['packed_slowdown'])} | {pct(values['slowdown_gap'])} | "
            f"{fmt(values['makespan'])} | {fmt(values['slo_goodput'], 0)} | "
            f"{'yes' if label in selected else ''} |"
        )
    lines.append("")
    return lines


def render_confirmation(
    artifacts: list[dict[str, Any]], selection: dict[str, Any]
) -> tuple[list[str], bool | None]:
    isolations = isolation_map(artifacts, "confirm")
    grouped: dict[str, list[dict[str, float]]] = defaultdict(list)
    for item in artifacts:
        experiment = item["experiment"]
        if (
            experiment.get("experiment_stage") == "confirm"
            and experiment.get("composition") == "mixed"
        ):
            grouped[policy_label(item)].append(mixed_metric(item, isolations))
    for items in grouped.values():
        items.sort(key=lambda item: item["rep"])
    if not grouped:
        return ["## Confirmation", "", "Not run.", ""], None

    wanted = ["token-vtc", selection["scalar_label"], selection["lifecycle_label"]]
    stats = {}
    for label in wanted:
        items = grouped.get(label, [])
        gaps = [item["slowdown_gap"] for item in items]
        stats[label] = {
            "reps": len(items),
            "split": mean([item["split_slowdown"] for item in items]),
            "packed": mean([item["packed_slowdown"] for item in items]),
            "split_p95": mean([item["split_p95"] for item in items]),
            "packed_p95": mean([item["packed_p95"] for item in items]),
            "gap": mean(gaps),
            "gap_ci": bootstrap_mean_ci(gaps),
            "makespan": mean([item["makespan"] for item in items]),
            "slo": mean([item["slo_goodput"] for item in items]),
            "retractions": mean([item["retractions"] for item in items]),
            "split_state": mean([item["split_state"] for item in items]),
            "split_kv": mean([item["split_kv"] for item in items]),
            "packed_state": mean([item["packed_state"] for item in items]),
            "packed_kv": mean([item["packed_kv"] for item in items]),
        }

    token = stats["token-vtc"]
    scalar = stats[selection["scalar_label"]]
    lifecycle = stats[selection["lifecycle_label"]]
    improve_token = (token["gap"] - lifecycle["gap"]) / token["gap"]
    improve_scalar = (scalar["gap"] - lifecycle["gap"]) / scalar["gap"]
    makespan_change = (lifecycle["makespan"] - token["makespan"]) / token["makespan"]
    paired_direction = all(
        lifecycle_item["slowdown_gap"] < token_item["slowdown_gap"]
        and lifecycle_item["slowdown_gap"] < scalar_item["slowdown_gap"]
        for lifecycle_item, token_item, scalar_item in zip(
            grouped[selection["lifecycle_label"]],
            grouped["token-vtc"],
            grouped[selection["scalar_label"]],
        )
    )
    passed = (
        all(stats[label]["reps"] == 3 for label in wanted)
        and improve_token >= 0.15
        and improve_scalar >= 0.15
        and makespan_change <= 0.05
        and paired_direction
    )
    lines = [
        "## Confirmation",
        "",
        f"Decision: **{'PASS' if passed else 'FAIL'}**. Realized-cost gap improvement "
        f"vs token VTC: {pct(improve_token)}; vs tuned scalar VTC: "
        f"{pct(improve_scalar)}; makespan change: {pct(makespan_change)}; "
        f"paired direction: {'3/3' if paired_direction else 'not 3/3'}.",
        "",
        "| policy | reps | split/packed p95 (s) | split slowdown | packed slowdown | slowdown gap | gap bootstrap 95% CI | makespan (s) | SLO goodput | retractions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in wanted:
        values = stats[label]
        lines.append(
            f"| {label} | {values['reps']} | {fmt(values['split_p95'])}/{fmt(values['packed_p95'])} | {pct(values['split'])} | "
            f"{pct(values['packed'])} | {pct(values['gap'])} | "
            f"[{pct(values['gap_ci'][0])}, {pct(values['gap_ci'][1])}] | "
            f"{fmt(values['makespan'])} | {fmt(values['slo'], 1)} | {fmt(values['retractions'], 0)} |"
        )
    lines.extend(
        [
            "",
            "### Exact Active Resource-Time",
            "",
            "| policy | split state-slot-s | split KV-token-s | packed state-slot-s | packed KV-token-s |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for label in wanted:
        values = stats[label]
        lines.append(
            f"| {label} | {fmt(values['split_state'])} | {fmt(values['split_kv'], 0)} | "
            f"{fmt(values['packed_state'])} | {fmt(values['packed_kv'], 0)} |"
        )
    lines.append("")
    return lines, passed


def integrity(artifacts: list[dict[str, Any]]) -> dict[str, int]:
    requests = [request for item in artifacts for request in item.get("requests", [])]
    confirmation = [
        item
        for item in artifacts
        if item.get("experiment", {}).get("experiment_stage") == "confirm"
    ]
    resolved = sum(
        bool(item.get("server", {}).get("resolved_experiment_policy"))
        for item in confirmation
    )
    return {
        "artifacts": len(artifacts),
        "requests": len(requests),
        "failures": sum(not request.get("success", False) for request in requests),
        "retractions": sum(int(request.get("num_retractions", 0)) for request in requests),
        "exact_traces": sum(bool(item.get("scheduler_resource_trace")) for item in artifacts),
        "confirmation_resolved": resolved,
        "confirmation_artifacts": len(confirmation),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()
    artifacts = load_artifacts(args.result_dir)
    cal_isolations = isolation_map(artifacts, "calibration")
    rows = calibration_rows(artifacts, cal_isolations)
    selection = select_oracles(rows)
    confirmation, passed = render_confirmation(artifacts, selection)
    audit = integrity(artifacts)
    report = "\n".join(
        [
            "# Lifecycle-cost fairness oracle results",
            "",
            f"Artifacts analyzed: {len(artifacts)}",
            "",
            f"Overall decision: **{'PENDING' if passed is None else ('PASS' if passed else 'FAIL')}**",
            "",
            "## Integrity",
            "",
            f"Formal artifacts: {audit['artifacts']}; requests: {audit['requests']}; "
            f"failures: {audit['failures']}; retractions: {audit['retractions']}; "
            f"exact traces: {audit['exact_traces']}/{audit['artifacts']}; "
            f"confirmation resolved-policy metadata: "
            f"{audit['confirmation_resolved']}/{audit['confirmation_artifacts']}.",
            "",
            *render_calibration(rows, selection),
            *confirmation,
            "## Frozen selection",
            "",
            "```json",
            json.dumps(selection, indent=2),
            "```",
            "",
        ]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    if args.selection_output:
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.write_text(
            json.dumps(selection, indent=2) + "\n", encoding="utf-8"
        )
    print(report)


if __name__ == "__main__":
    main()
