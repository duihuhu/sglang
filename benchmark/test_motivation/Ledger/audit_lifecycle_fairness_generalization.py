#!/usr/bin/env python3
"""Audit the formal lifecycle-fairness generalization artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CELLS = {
    "base": (16, 16384, 512, 1.5, 48, 196608),
    "split4": (4, 16384, 512, 1.5, 48, 196608),
    "split32": (32, 16384, 512, 1.5, 48, 196608),
    "output1k": (16, 8192, 1024, 1.5, 48, 196608),
    "output2k": (16, 4096, 2048, 1.5, 48, 196608),
    "arrival1": (16, 16384, 512, 1.0, 48, 196608),
    "arrival3": (16, 16384, 512, 3.0, 48, 196608),
    "state96": (16, 16384, 512, 1.5, 96, 196608),
    "kv393": (16, 16384, 512, 1.5, 48, 393216),
    "relaxed": (16, 16384, 512, 1.5, 96, 393216),
}
STAGE_WAVES = {"screen": 4, "confirm": 8, "kill": 8}
EXPECTED_SCHEDULE_POLICY = {
    "fcfs": "fcfs",
    "vtc": "tenant-vtc",
    "instant": "tenant-instant-share",
    "vector": "tenant-vector-debt",
    "lifecycle": "tenant-lifecycle-cost",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1e-9, abs_tol=1e-6)


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


def expected_confirm_count(selection: dict[str, Any], cell: str) -> int:
    selected = selection[cell]
    extras = int(selected["request_tax"] != "token-vtc")
    extras += int(selected["lifecycle_oracle"] != "lifecycle-s0.25-k1")
    return 3 * (8 + extras)


def audit(result_dir: Path) -> tuple[dict[str, Any], list[str]]:
    selection = load_json(result_dir / "selection.json")
    paths = sorted(result_dir.glob("gen-*.json"))
    items = [(path, load_json(path)) for path in paths]
    errors: list[str] = []
    stage_cell_counts: Counter[tuple[str, str]] = Counter()
    policy_counts: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    run_ids: set[str] = set()
    request_count = 0
    failure_count = 0
    retraction_count = 0
    equal_token_mixed = 0
    resolved_count = 0
    exact_trace_count = 0

    for path, item in items:
        exp = item.get("experiment", {})
        stage = exp.get("experiment_stage")
        cell = exp.get("cell_id")
        run_id = exp.get("run_id")
        where = path.name
        if stage not in STAGE_WAVES or cell not in CELLS:
            errors.append(f"{where}: unexpected stage/cell {stage}/{cell}")
            continue
        stage_cell_counts[(stage, cell)] += 1
        policy_counts[(stage, cell)][policy_label(item)] += 1
        if run_id in run_ids:
            errors.append(f"{where}: duplicate run_id {run_id}")
        run_ids.add(run_id)
        if path.stem != run_id:
            errors.append(f"{where}: filename/run_id mismatch")

        split, prompt, output, interval, state_cap, kv_cap = CELLS[cell]
        expected_metadata = {
            "split_factor": split,
            "packed_prompt": prompt,
            "packed_output": output,
            "wave_interval_s": interval,
            "waves": STAGE_WAVES[stage],
            "physical_gpus": "6,7",
        }
        for key, expected in expected_metadata.items():
            if exp.get(key) != expected:
                errors.append(
                    f"{where}: {key}={exp.get(key)!r}, expected {expected!r}"
                )

        server = item.get("server", {})
        if server.get("model_path") != "/models/Kimi-Linear-48B-A3B":
            errors.append(f"{where}: wrong model path {server.get('model_path')!r}")
        if server.get("max_total_num_tokens") != kv_cap:
            errors.append(f"{where}: KV cap mismatch")
        states = server.get("internal_states", [])
        if not states or any(s.get("max_mamba_cache_size") != state_cap for s in states):
            errors.append(f"{where}: state cap mismatch")

        runtime_policy = exp.get("runtime_policy")
        resolved = server.get("resolved_experiment_policy", [])
        expected_policy = EXPECTED_SCHEDULE_POLICY.get(runtime_policy)
        resolved_ok = bool(resolved) and all(
            state.get("schedule_policy") == expected_policy for state in resolved
        )
        if runtime_policy == "vtc":
            resolved_ok = resolved_ok and all(
                close(float(state.get("tenant_vtc_request_weight")), float(exp["request_weight"]))
                and close(float(state.get("tenant_vtc_input_weight")), float(exp["input_weight"]))
                and close(float(state.get("tenant_vtc_output_weight")), float(exp["output_weight"]))
                for state in resolved
            )
        if runtime_policy == "lifecycle":
            resolved_ok = resolved_ok and all(
                close(
                    float(state.get("tenant_lifecycle_state_weight")),
                    float(exp["lifecycle_state_weight"]),
                )
                and close(
                    float(state.get("tenant_lifecycle_kv_weight")),
                    float(exp["lifecycle_kv_weight"]),
                )
                for state in resolved
            )
        if resolved_ok:
            resolved_count += 1
        else:
            errors.append(f"{where}: resolved policy/weight mismatch")

        trace = item.get("scheduler_resource_trace")
        if trace and trace.get("run_id") == run_id:
            exact_trace_count += 1
            system = trace.get("system", {})
            tenants = trace.get("tenants", {})
            for key in (
                "active_request_seconds",
                "state_slot_seconds",
                "kv_token_seconds",
                "current_active_requests",
                "current_state_slots",
                "current_kv_tokens",
            ):
                total = sum(float(value.get(key, 0)) for value in tenants.values())
                if not close(float(system.get(key, 0)), total):
                    errors.append(f"{where}: trace tenant/system mismatch for {key}")
        else:
            errors.append(f"{where}: missing or mismatched exact trace")

        requests = item.get("requests", [])
        request_count += len(requests)
        failure_count += sum(not request.get("success", False) for request in requests)
        retraction_count += sum(int(request.get("num_retractions", 0)) for request in requests)
        composition = exp.get("composition")
        expected_requests = STAGE_WAVES[stage] * {
            "mixed": split + 1,
            "split-only": split,
            "packed-only": 1,
        }.get(composition, -1)
        if len(requests) != expected_requests:
            errors.append(
                f"{where}: {len(requests)} requests, expected {expected_requests}"
            )
        if composition == "mixed":
            by_group = defaultdict(lambda: {"input": 0, "output": 0})
            for request in requests:
                group = by_group[request.get("group")]
                group["input"] += int(request.get("prompt_tokens", 0))
                group["output"] += int(request.get("completion_tokens", 0))
            if by_group.get("split") == by_group.get("packed"):
                equal_token_mixed += 1
            else:
                errors.append(f"{where}: split/packed input or output tokens differ")

    for cell in CELLS:
        expected_counts = {
            "screen": 20,
            "confirm": expected_confirm_count(selection, cell),
            "kill": 15,
        }
        for stage, expected in expected_counts.items():
            observed = stage_cell_counts[(stage, cell)]
            if observed != expected:
                errors.append(
                    f"{stage}/{cell}: {observed} artifacts, expected {expected}"
                )

        for stage, repetitions in (("confirm", 3), ("kill", 3)):
            required = {"token-vtc": repetitions, "instant-share": repetitions}
            if stage == "confirm":
                required.update(
                    {
                        "fcfs": repetitions * 3,
                        "request-vtc": repetitions,
                        "vector-debt": repetitions,
                        "lifecycle-s0.25-k1": repetitions,
                    }
                )
            else:
                required.update(
                    {"fcfs": repetitions * 2, "lifecycle-s0.25-k1": repetitions}
                )
            for label, expected in required.items():
                observed = policy_counts[(stage, cell)][label]
                if observed != expected:
                    errors.append(
                        f"{stage}/{cell}/{label}: {observed}, expected {expected}"
                    )

    mixed_count = sum(
        item.get("experiment", {}).get("composition") == "mixed"
        for _, item in items
    )
    summary = {
        "decision": "PASS" if not errors else "FAIL",
        "artifacts": len(items),
        "requests": request_count,
        "failed_requests": failure_count,
        "retractions": retraction_count,
        "exact_traces": exact_trace_count,
        "resolved_policy_matches": resolved_count,
        "equal_input_output_token_mixed_artifacts": equal_token_mixed,
        "mixed_artifacts": mixed_count,
        "stage_artifacts": {
            stage: sum(value for (name, _), value in stage_cell_counts.items() if name == stage)
            for stage in STAGE_WAVES
        },
        "cell_artifacts": {
            cell: {
                stage: stage_cell_counts[(stage, cell)] for stage in STAGE_WAVES
            }
            for cell in CELLS
        },
        "errors": errors,
    }
    return summary, errors


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# Lifecycle fairness generalization audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        f"- Artifacts: {summary['artifacts']}",
        f"- Requests: {summary['requests']}",
        f"- Failed requests: {summary['failed_requests']}",
        f"- Retractions: {summary['retractions']}",
        f"- Exact traces: {summary['exact_traces']}/{summary['artifacts']}",
        f"- Exact resolved policy/weight matches: {summary['resolved_policy_matches']}/{summary['artifacts']}",
        f"- Equal input and output token mixed artifacts: {summary['equal_input_output_token_mixed_artifacts']}/{summary['mixed_artifacts']}",
        "",
        "| cell | screen | confirm | kill |",
        "|---|---:|---:|---:|",
    ]
    for cell, counts in summary["cell_artifacts"].items():
        lines.append(
            f"| {cell} | {counts['screen']} | {counts['confirm']} | {counts['kill']} |"
        )
    lines.extend(["", "## Errors", ""])
    if summary["errors"]:
        lines.extend(f"- {error}" for error in summary["errors"])
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    summary, errors = audit(args.result_dir)
    markdown = render(summary)
    if args.json_output:
        args.json_output.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    if args.markdown_output:
        args.markdown_output.write_text(markdown, encoding="utf-8")
    print(markdown, end="")
    raise SystemExit(bool(errors))


if __name__ == "__main__":
    main()
