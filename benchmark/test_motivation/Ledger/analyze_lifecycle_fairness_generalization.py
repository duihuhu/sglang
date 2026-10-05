#!/usr/bin/env python3
"""Analyze held-out lifecycle-cost fairness experiments."""

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
    for path in sorted(result_dir.glob("gen-*.json")):
        try:
            item = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        item["_path"] = str(path)
        artifacts.append(item)
    return artifacts


def repetition(item: dict[str, Any]) -> int:
    match = re.search(r"rep(\d+)$", item["experiment"]["run_id"])
    return int(match.group(1)) if match else 0


def policy_label(item: dict[str, Any]) -> str:
    exp = item["experiment"]
    policy = exp["runtime_policy"]
    if policy == "vtc":
        request = float(exp.get("request_weight", 0))
        input_weight = float(exp.get("input_weight", 1))
        output_weight = float(exp.get("output_weight", 1))
        if request == 1 and input_weight == 0 and output_weight == 0:
            return "request-vtc"
        if request == 0 and input_weight == 1 and output_weight == 1:
            return "token-vtc"
        return f"request-tax-r{request:g}-i{input_weight:g}-o{output_weight:g}"
    if policy == "instant":
        return "instant-share"
    if policy == "vector":
        return "vector-debt"
    if policy == "lifecycle":
        state = float(exp.get("lifecycle_state_weight", 1))
        kv = float(exp.get("lifecycle_kv_weight", 1))
        return f"lifecycle-s{state:g}-k{kv:g}"
    return policy


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def pct(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.1%}"


def num(value: float, digits: int = 3) -> str:
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def bootstrap_mean_ci(values: list[float], samples: int = 20000) -> tuple[float, float]:
    rng = random.Random(20260825)
    estimates = sorted(
        mean([rng.choice(values) for _ in values]) for _ in range(samples)
    )
    return estimates[int(samples * 0.025)], estimates[int(samples * 0.975)]


def makespan(item: dict[str, Any]) -> float:
    return max(float(req["finished_s"]) for req in item["requests"])


def isolation_map(
    artifacts: list[dict[str, Any]], stage: str
) -> dict[tuple[str, int, str], dict[str, Any]]:
    result = {}
    for item in artifacts:
        exp = item["experiment"]
        composition = exp.get("composition")
        if exp.get("experiment_stage") != stage or composition not in {
            "split-only",
            "packed-only",
        }:
            continue
        result[(exp["cell_id"], repetition(item), composition.removesuffix("-only"))] = item
    return result


def mixed_metric(
    item: dict[str, Any], isolations: dict[tuple[str, int, str], dict[str, Any]]
) -> dict[str, float]:
    exp = item["experiment"]
    key = (exp["cell_id"], repetition(item))
    split_iso = isolations[(*key, "split")]["summary"]["groups"]["split"]
    packed_iso = isolations[(*key, "packed")]["summary"]["groups"]["packed"]
    groups = item["summary"]["groups"]
    split = groups["split"]
    packed = groups["packed"]
    split_slowdown = float(split["p95_jct_s"]) / float(split_iso["p95_jct_s"]) - 1
    packed_slowdown = float(packed["p95_jct_s"]) / float(packed_iso["p95_jct_s"]) - 1
    split_resource = split.get("scheduler_active_resources", {})
    packed_resource = packed.get("scheduler_active_resources", {})
    return {
        "split_slowdown": split_slowdown,
        "packed_slowdown": packed_slowdown,
        "gap": abs(split_slowdown - packed_slowdown),
        "split_p95": float(split["p95_jct_s"]),
        "packed_p95": float(packed["p95_jct_s"]),
        "makespan": makespan(item),
        "slo": float(item["summary"]["overall"]["slo_goodput"]),
        "split_state": float(split_resource.get("state_slot_seconds", 0)),
        "split_kv": float(split_resource.get("kv_token_seconds", 0)),
        "packed_state": float(packed_resource.get("state_slot_seconds", 0)),
        "packed_kv": float(packed_resource.get("kv_token_seconds", 0)),
    }


def stage_rows(
    artifacts: list[dict[str, Any]], stage: str
) -> dict[str, dict[str, list[dict[str, float]]]]:
    isolations = isolation_map(artifacts, stage)
    rows: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for item in artifacts:
        exp = item["experiment"]
        if exp.get("experiment_stage") != stage or exp.get("composition") != "mixed":
            continue
        rows[exp["cell_id"]][policy_label(item)].append(
            mixed_metric(item, isolations)
        )
    return rows


def aggregate(items: list[dict[str, float]]) -> dict[str, float]:
    keys = items[0].keys() if items else []
    return {key: mean([item[key] for item in items]) for key in keys}


def select(screen: dict[str, dict[str, list[dict[str, float]]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for cell, grouped in sorted(screen.items()):
        rows = {label: aggregate(items) for label, items in grouped.items()}
        token = rows["token-vtc"]

        def choose(labels: list[str]) -> str:
            bounded = [
                label
                for label in labels
                if label in rows and rows[label]["makespan"] <= 1.05 * token["makespan"]
            ]
            if not bounded:
                raise ValueError(f"{cell}: no throughput-bounded candidate in {labels}")
            return min(bounded, key=lambda label: (rows[label]["gap"], rows[label]["makespan"]))

        request_labels = [
            label
            for label in rows
            if label == "token-vtc" or label.startswith("request-tax-")
        ]
        static_labels = [
            label for label in rows if not label.startswith("lifecycle-")
        ]
        lifecycle_labels = [
            label for label in rows if label.startswith("lifecycle-")
        ]
        result[cell] = {
            "request_tax": choose(request_labels),
            "strongest_static": choose(static_labels),
            "lifecycle_oracle": choose(lifecycle_labels),
            "screen_rows": rows,
        }
    return result


def integrity(artifacts: list[dict[str, Any]]) -> dict[str, int]:
    requests = [req for item in artifacts for req in item.get("requests", [])]
    mixed = [
        item
        for item in artifacts
        if item.get("experiment", {}).get("composition") == "mixed"
    ]
    return {
        "artifacts": len(artifacts),
        "requests": len(requests),
        "failures": sum(not req.get("success", False) for req in requests),
        "exact": sum(bool(item.get("scheduler_resource_trace")) for item in artifacts),
        "resolved": sum(
            bool(item.get("server", {}).get("resolved_experiment_policy"))
            for item in mixed
        ),
        "mixed": len(mixed),
        "gpu": sum(
            item.get("experiment", {}).get("physical_gpus") == "6,7"
            for item in artifacts
        ),
    }


def render_screen(selection: dict[str, Any]) -> list[str]:
    lines = [
        "## Screening selection",
        "",
        "| cell | best request-tax | strongest static | lifecycle oracle | token gap | fixed lifecycle gap |",
        "|---|---|---|---|---:|---:|",
    ]
    for cell, selected in selection.items():
        rows = selected["screen_rows"]
        lines.append(
            f"| {cell} | {selected['request_tax']} | {selected['strongest_static']} | "
            f"{selected['lifecycle_oracle']} | {pct(rows['token-vtc']['gap'])} | "
            f"{pct(rows['lifecycle-s0.25-k1']['gap'])} |"
        )
    lines.append("")
    return lines


def render_confirmation(
    confirm: dict[str, dict[str, list[dict[str, float]]]],
    selection: dict[str, Any],
) -> tuple[list[str], bool | None]:
    if not confirm:
        return ["## Confirmation", "", "Not run.", ""], None
    lines = [
        "## Confirmation by cell",
        "",
        "| cell | strongest confirmed static | lifecycle oracle | token gap | strongest static gap | fixed lifecycle gap | oracle gap | fixed improvement | price regret | makespan change | direction |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    cell_results = []
    complete = True
    for cell, selected in selection.items():
        grouped = confirm.get(cell, {})
        required_static = {
            "fcfs",
            "request-vtc",
            "token-vtc",
            "instant-share",
            "vector-debt",
            selected["request_tax"],
        }
        required_lifecycle = {
            "lifecycle-s0.25-k1",
            selected["lifecycle_oracle"],
        }
        if any(
            len(grouped.get(label, [])) != 3
            for label in required_static | required_lifecycle
        ):
            complete = False
            continue
        aggregated = {
            label: aggregate(items)
            for label, items in grouped.items()
            if len(items) == 3
        }
        token_value = aggregated["token-vtc"]
        bounded_static = {
            label: aggregated[label]
            for label in required_static
            if aggregated[label]["makespan"] <= 1.05 * token_value["makespan"]
        }
        if not bounded_static:
            complete = False
            continue
        strongest_static_label = min(
            bounded_static,
            key=lambda label: (
                bounded_static[label]["gap"],
                bounded_static[label]["makespan"],
            ),
        )
        lifecycle_oracle_label = min(
            required_lifecycle,
            key=lambda label: (
                aggregated[label]["gap"],
                aggregated[label]["makespan"],
            ),
        )
        labels = {
            "token": "token-vtc",
            "strong": strongest_static_label,
            "fixed": "lifecycle-s0.25-k1",
            "oracle": lifecycle_oracle_label,
        }
        values = {name: aggregated[label] for name, label in labels.items()}
        strong_gap = values["strong"]["gap"]
        fixed_gap = values["fixed"]["gap"]
        oracle_gap = values["oracle"]["gap"]
        improvement = (strong_gap - fixed_gap) / strong_gap if strong_gap else 0.0
        regret = (fixed_gap - oracle_gap) / max(oracle_gap, 1e-12)
        makespan_change = (
            values["fixed"]["makespan"] / values["token"]["makespan"] - 1
        )
        direction = fixed_gap < strong_gap
        cell_results.append(
            {
                "cell": cell,
                "improvement": improvement,
                "regret": regret,
                "makespan_change": makespan_change,
                "direction": direction,
            }
        )
        lines.append(
            f"| {cell} | {strongest_static_label} | {lifecycle_oracle_label} | "
            f"{pct(values['token']['gap'])} | {pct(strong_gap)} | "
            f"{pct(fixed_gap)} | {pct(oracle_gap)} | {pct(improvement)} | "
            f"{pct(regret)} | {pct(makespan_change)} | {'yes' if direction else 'no'} |"
        )
    lines.append("")
    if not complete or len(cell_results) != len(selection):
        lines.extend(["Confirmation incomplete.", ""])
        return lines, None

    mean_improvement = mean([row["improvement"] for row in cell_results])
    direction_count = sum(row["direction"] for row in cell_results)
    mean_makespan = mean([row["makespan_change"] for row in cell_results])
    median_regret = statistics.median(row["regret"] for row in cell_results)
    passed = (
        mean_improvement >= 0.15
        and direction_count >= 8
        and mean_makespan <= 0.05
        and median_regret < 0.15
    )
    lines.extend(
        [
            "## Generalization gate",
            "",
            f"Decision: **{'PASS' if passed else 'FAIL'}**. Mean gap improvement over "
            f"the strongest per-cell static baseline: {pct(mean_improvement)}; paired "
            f"direction: {direction_count}/{len(cell_results)} cells; mean makespan "
            f"change: {pct(mean_makespan)}; median fixed-price regret: "
            f"{pct(median_regret)}.",
            "",
        ]
    )
    return lines, passed


def render_instant_kill(
    kill: dict[str, dict[str, list[dict[str, float]]]],
    selection: dict[str, Any],
) -> tuple[list[str], bool | None]:
    if not kill:
        return ["## Independent instantaneous-share kill confirm", "", "Not run.", ""], None
    lines = [
        "## Independent instantaneous-share kill confirm",
        "",
        "| cell | token gap | instant-share gap | fixed lifecycle gap | lifecycle improvement | makespan change | direction |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    rows = []
    complete = True
    for cell in selection:
        grouped = kill.get(cell, {})
        required = {"token-vtc", "instant-share", "lifecycle-s0.25-k1"}
        if any(len(grouped.get(label, [])) != 3 for label in required):
            complete = False
            continue
        token = aggregate(grouped["token-vtc"])
        instant = aggregate(grouped["instant-share"])
        fixed = aggregate(grouped["lifecycle-s0.25-k1"])
        improvement = (instant["gap"] - fixed["gap"]) / max(instant["gap"], 1e-12)
        makespan_change = fixed["makespan"] / token["makespan"] - 1
        direction = fixed["gap"] < instant["gap"]
        rows.append(
            {
                "improvement": improvement,
                "makespan_change": makespan_change,
                "direction": direction,
            }
        )
        lines.append(
            f"| {cell} | {pct(token['gap'])} | {pct(instant['gap'])} | "
            f"{pct(fixed['gap'])} | {pct(improvement)} | {pct(makespan_change)} | "
            f"{'yes' if direction else 'no'} |"
        )
    lines.append("")
    if not complete or len(rows) != len(selection):
        lines.extend(["Kill confirmation incomplete.", ""])
        return lines, None

    improvements = [row["improvement"] for row in rows]
    mean_improvement = mean(improvements)
    improvement_ci = bootstrap_mean_ci(improvements)
    mean_makespan = mean([row["makespan_change"] for row in rows])
    directions = sum(row["direction"] for row in rows)
    passed = (
        mean_improvement >= 0.15
        and directions >= 8
        and mean_makespan <= 0.05
    )
    lines.extend(
        [
            f"Decision: **{'PASS' if passed else 'FAIL'}**. Mean gap improvement "
            f"over instantaneous-share: {pct(mean_improvement)}, cell bootstrap 95% "
            f"CI [{pct(improvement_ci[0])}, {pct(improvement_ci[1])}]; paired "
            f"direction: {directions}/{len(rows)} cells; mean makespan change vs token "
            f"VTC: {pct(mean_makespan)}.",
            "",
        ]
    )
    return lines, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection-output", type=Path)
    args = parser.parse_args()

    artifacts = load_artifacts(args.result_dir)
    screen = stage_rows(artifacts, "screen")
    selection = select(screen)
    confirmation, exploratory_passed = render_confirmation(
        stage_rows(artifacts, "confirm"), selection
    )
    instant_kill, passed = render_instant_kill(stage_rows(artifacts, "kill"), selection)
    audit = integrity(artifacts)
    integrity_pass = (
        audit["failures"] == 0
        and audit["exact"] == audit["artifacts"]
        and audit["resolved"] == audit["mixed"]
        and audit["gpu"] == audit["artifacts"]
    )
    if passed is True and not integrity_pass:
        passed = False
    report = "\n".join(
        [
            "# Lifecycle-cost fairness held-out generalization",
            "",
            f"Overall decision: **{'PENDING' if passed is None else ('PASS' if passed else 'FAIL')}**",
            "",
            "## Integrity",
            "",
            f"Formal artifacts: {audit['artifacts']}; requests: {audit['requests']}; "
            f"failures: {audit['failures']}; exact traces: "
            f"{audit['exact']}/{audit['artifacts']}; resolved mixed-policy metadata: "
            f"{audit['resolved']}/{audit['mixed']}; GPU 6,7 metadata: "
            f"{audit['gpu']}/{audit['artifacts']}.",
            "",
            *render_screen(selection),
            *confirmation,
            *instant_kill,
        ]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    if args.selection_output:
        compact = {
            cell: {
                key: value
                for key, value in selected.items()
                if key != "screen_rows"
            }
            for cell, selected in selection.items()
        }
        args.selection_output.write_text(
            json.dumps(compact, indent=2) + "\n", encoding="utf-8"
        )
    print(report)


if __name__ == "__main__":
    main()
