#!/usr/bin/env python3
"""Analyze the preregistered no-prefix VTC state-tax observations."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


CELLS = {
    "f1-base": (1, 1.5, 48, 196608),
    "f4-base": (4, 1.5, 48, 196608),
    "f16-base": (16, 1.5, 48, 196608),
    "f32-base": (32, 1.5, 48, 196608),
    "f16-state-relaxed": (16, 1.5, 480, 196608),
    "f16-kv-relaxed": (16, 1.5, 48, 393216),
    "f16-full": (16, 1.5, 480, 393216),
    "f16-fast": (16, 1.0, 48, 196608),
    "f16-slow": (16, 3.0, 48, 196608),
}
CALIBRATION_CELLS = ("f4-base", "f16-base", "f16-kv-relaxed")
REQUEST_TAX_LABELS = tuple(
    f"request-tax-r{weight}" for weight in (1024, 4096, 8192, 16384, 32768, 65536)
)
SCREEN_MIXED_LABELS = (
    "fcfs",
    "request-vtc",
    "token-vtc",
    *REQUEST_TAX_LABELS,
    "profiled-vtc",
    "instant-share",
)


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def pct(value: float) -> str:
    return "-" if not math.isfinite(value) else f"{value:.1%}"


def num(value: float, digits: int = 3) -> str:
    return "-" if not math.isfinite(value) else f"{value:.{digits}f}"


def repetition(item: dict[str, Any]) -> int:
    match = re.search(r"rep(\d+)$", item["experiment"]["run_id"])
    return int(match.group(1)) if match else 0


def load_artifacts(result_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(result_dir.glob("st-*.json")):
        if path.name in {"profiled_vtc.json", "selection.json", "analysis.json"}:
            continue
        item = json.loads(path.read_text(encoding="utf-8"))
        item["_path"] = str(path)
        artifacts.append(item)
    return artifacts


def makespan(item: dict[str, Any]) -> float:
    return max(float(request["finished_s"]) for request in item["requests"])


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
        group = composition.removesuffix("-only")
        result[(exp["cell_id"], repetition(item), group)] = item
    return result


def request_cost(exp: dict[str, Any], prompt: int, output: int) -> float:
    return (
        float(exp["request_weight"])
        + float(exp["input_weight"]) * prompt
        + float(exp["input_quadratic_weight"]) * prompt * prompt
        + float(exp["output_weight"]) * output
        + float(exp["input_output_weight"]) * prompt * output
        + float(exp["output_quadratic_weight"]) * output * output
    )


def mixed_metric(
    item: dict[str, Any], isolations: dict[tuple[str, int, str], dict[str, Any]]
) -> dict[str, float]:
    exp = item["experiment"]
    cell = exp["cell_id"]
    rep = repetition(item)
    split_iso = isolations[(cell, rep, "split")]["summary"]["groups"]["split"]
    packed_iso = isolations[(cell, rep, "packed")]["summary"]["groups"]["packed"]
    groups = item["summary"]["groups"]
    split_slowdown = (
        float(groups["split"]["p95_jct_s"]) / float(split_iso["p95_jct_s"]) - 1
    )
    packed_slowdown = (
        float(groups["packed"]["p95_jct_s"]) / float(packed_iso["p95_jct_s"])
        - 1
    )
    totals: dict[str, dict[str, int]] = defaultdict(
        lambda: {"input": 0, "output": 0}
    )
    for request in item["requests"]:
        totals[request["group"]]["input"] += int(request["prompt_tokens"])
        totals[request["group"]]["output"] += int(request["completion_tokens"])
    state_cap = int(item["server"]["internal_states"][0]["max_mamba_cache_size"])
    kv_cap = int(item["server"]["max_total_num_tokens"])
    resource = {}
    for group in ("split", "packed"):
        usage = groups[group]["scheduler_active_resources"]
        tokens = totals[group]["input"] + totals[group]["output"]
        resource[group] = {
            "state_per_token": float(usage["state_slot_seconds"]) / tokens,
            "kv_per_token": float(usage["kv_token_seconds"]) / tokens,
            "dominant_per_token": max(
                float(usage["state_slot_seconds"]) / state_cap,
                float(usage["kv_token_seconds"]) / kv_cap,
            )
            / tokens,
            "peak_state": float(usage["peak_state_slots"]),
            "peak_kv": float(usage["peak_kv_tokens"]),
        }
    state_ratio = max(
        resource["split"]["state_per_token"],
        resource["packed"]["state_per_token"],
    ) / max(
        min(
            resource["split"]["state_per_token"],
            resource["packed"]["state_per_token"],
        ),
        1e-12,
    )
    kv_ratio = max(
        resource["split"]["kv_per_token"], resource["packed"]["kv_per_token"]
    ) / max(
        min(
            resource["split"]["kv_per_token"],
            resource["packed"]["kv_per_token"],
        ),
        1e-12,
    )
    composition = {}
    for group in ("split", "packed"):
        composition[group] = resource[group]["peak_state"] / max(
            resource[group]["peak_kv"], 1
        )
    composition_ratio = max(composition.values()) / max(
        min(composition.values()), 1e-12
    )
    vtc_trace = item.get("scheduler_vtc_trace") or {}
    max_request_cost = max(
        request_cost(exp, int(exp["packed_prompt"]), int(exp["packed_output"])),
        request_cost(
            exp,
            int(exp["packed_prompt"]) // int(exp["split_factor"]),
            int(exp["packed_output"]) // int(exp["split_factor"]),
        ),
    )
    counter_bound_ratio = float(vtc_trace.get("max_counter_gap", 0)) / max(
        2 * max_request_cost, 1e-12
    )
    queue_waits = vtc_trace.get("queue_waits", {})
    return {
        "split_slowdown": split_slowdown,
        "packed_slowdown": packed_slowdown,
        "gap": abs(split_slowdown - packed_slowdown),
        "makespan": makespan(item),
        "input_imbalance": abs(totals["split"]["input"] - totals["packed"]["input"])
        / max(totals["packed"]["input"], 1),
        "output_imbalance": abs(
            totals["split"]["output"] - totals["packed"]["output"]
        )
        / max(totals["packed"]["output"], 1),
        "state_ratio": state_ratio,
        "kv_ratio": kv_ratio,
        "max_dimension_ratio": max(state_ratio, kv_ratio),
        "composition_ratio": composition_ratio,
        "split_dominant_per_token": resource["split"]["dominant_per_token"],
        "packed_dominant_per_token": resource["packed"]["dominant_per_token"],
        "split_peak_state": resource["split"]["peak_state"],
        "packed_peak_state": resource["packed"]["peak_state"],
        "split_peak_kv": resource["split"]["peak_kv"],
        "packed_peak_kv": resource["packed"]["peak_kv"],
        "counter_bound_ratio": counter_bound_ratio,
        "max_counter_gap": float(vtc_trace.get("max_counter_gap", 0)),
        "split_queue_p95": float(
            queue_waits.get("split", {}).get("p95_seconds", 0)
        ),
        "packed_queue_p95": float(
            queue_waits.get("packed", {}).get("p95_seconds", 0)
        ),
        "split_peak_waiting": float(
            queue_waits.get("split", {}).get("peak_waiting_requests", 0)
        ),
        "packed_peak_waiting": float(
            queue_waits.get("packed", {}).get("peak_waiting_requests", 0)
        ),
    }


def aggregate(items: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: mean([item[key] for item in items])
        for key in (items[0].keys() if items else [])
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
        rows[exp["cell_id"]][exp["policy_label"]].append(
            mixed_metric(item, isolations)
        )
    return rows


def select(screen: dict[str, dict[str, list[dict[str, float]]]]) -> dict[str, Any]:
    per_cell = {}
    for cell, grouped in sorted(screen.items()):
        rows = {label: aggregate(items) for label, items in grouped.items()}
        token = rows["token-vtc"]
        candidates = {
            label: rows[label]
            for label in ("token-vtc", *REQUEST_TAX_LABELS)
            if label in rows and rows[label]["makespan"] <= 1.05 * token["makespan"]
        }
        per_cell[cell] = min(
            candidates,
            key=lambda label: (candidates[label]["gap"], candidates[label]["makespan"]),
        )

    global_candidates = []
    for label in ("token-vtc", *REQUEST_TAX_LABELS):
        values = []
        valid = True
        for cell in CALIBRATION_CELLS:
            rows = {name: aggregate(items) for name, items in screen[cell].items()}
            if label not in rows or rows[label]["makespan"] > 1.05 * rows["token-vtc"]["makespan"]:
                valid = False
                break
            values.append(rows[label])
        if valid:
            global_candidates.append(
                (mean([value["gap"] for value in values]), mean([value["makespan"] for value in values]), label)
            )
    if not global_candidates:
        raise ValueError("no throughput-bounded global request-tax candidate")
    _, _, global_label = min(global_candidates)
    return {"global_request_tax": global_label, "per_cell_request_tax": per_cell}


def integrity(artifacts: list[dict[str, Any]]) -> dict[str, Any]:
    formal = [
        item
        for item in artifacts
        if item["experiment"]["experiment_stage"]
        in {"profile", "screen", "confirm"}
    ]
    requests = [request for item in formal for request in item.get("requests", [])]
    vtc = [item for item in formal if item["experiment"]["runtime_policy"] == "vtc"]
    mixed = [
        item
        for item in formal
        if item["experiment"].get("composition") == "mixed"
        and item["experiment"]["mode"] == "partition"
    ]
    stage_counts = Counter(item["experiment"]["experiment_stage"] for item in formal)
    run_ids = Counter(item["experiment"]["run_id"] for item in formal)
    duplicate_run_ids = [run_id for run_id, count in run_ids.items() if count != 1]
    resolved = 0
    for item in formal:
        exp = item["experiment"]
        expected_policy = {
            "fcfs": "fcfs",
            "vtc": "tenant-vtc",
            "instant": "tenant-instant-share",
        }[exp["runtime_policy"]]
        states = item["server"].get("resolved_experiment_policy", [])
        ok = bool(states) and all(state.get("schedule_policy") == expected_policy for state in states)
        if exp["runtime_policy"] == "vtc":
            for state in states:
                for key in (
                    "request_weight",
                    "input_weight",
                    "input_quadratic_weight",
                    "output_weight",
                    "input_output_weight",
                    "output_quadratic_weight",
                ):
                    actual = float(state[f"tenant_vtc_{key}"])
                    ok = ok and math.isclose(
                        actual,
                        float(exp[key]),
                        rel_tol=1e-9,
                        abs_tol=1e-12,
                    )
        resolved += int(ok)

    profile_shapes = Counter()
    screen_matrix = Counter()
    confirm_matrix = Counter()
    for item in formal:
        exp = item["experiment"]
        stage = exp["experiment_stage"]
        if stage == "profile":
            profile_shapes[(exp["profile_prompt"], exp["profile_output"])] += 1
        elif stage == "screen":
            screen_matrix[(exp["cell_id"], exp["policy_label"])] += 1
        elif stage == "confirm":
            confirm_matrix[(
                exp["cell_id"],
                exp["policy_label"],
                repetition(item),
            )] += 1

    profile_matrix_ok = all(
        profile_shapes[(prompt, output)] == 3
        for prompt in (1024, 4096, 16384)
        for output in (32, 128, 512)
    ) and sum(profile_shapes.values()) == 27
    screen_matrix_ok = all(
        screen_matrix[(cell, label)] == 1
        for cell in CELLS
        for label in (
            "fcfs-split-only",
            "fcfs-packed-only",
            *SCREEN_MIXED_LABELS,
        )
    ) and sum(screen_matrix.values()) == 117
    filenames_ok = all(
        Path(item["_path"]).name == f"{item['experiment']['run_id']}.json"
        for item in formal
    )
    model_ok = all(
        item["server"].get("model_path") == "/models/Kimi-Linear-48B-A3B"
        for item in formal
    )
    capacities_ok = True
    for item in formal:
        exp = item["experiment"]
        if exp["experiment_stage"] == "profile":
            expected_state, expected_kv = 480, 393216
        else:
            _, _, expected_state, expected_kv = CELLS[exp["cell_id"]]
        states = item["server"].get("internal_states", [])
        capacities_ok = capacities_ok and bool(states)
        capacities_ok = capacities_ok and all(
            int(state["max_mamba_cache_size"]) == expected_state
            for state in states
        )
        capacities_ok = capacities_ok and (
            int(item["server"]["max_total_num_tokens"]) == expected_kv
        )
    return {
        "artifacts": len(formal),
        "requests": len(requests),
        "failures": sum(not request.get("success", False) for request in requests),
        "cached": sum(int(request.get("cached_tokens", 0)) for request in requests),
        "retractions": sum(int(request.get("num_retractions", 0)) for request in requests),
        "exact_traces": sum(bool(item.get("scheduler_resource_trace")) for item in formal),
        "vtc_traces": sum(bool(item.get("scheduler_vtc_trace")) for item in vtc),
        "vtc_artifacts": len(vtc),
        "resolved": resolved,
        "gpu": sum(item["experiment"].get("physical_gpus") == "6,7" for item in formal),
        "equal_token_mixed": sum(
            sum(int(request["prompt_tokens"]) for request in item["requests"] if request["group"] == "split")
            == sum(int(request["prompt_tokens"]) for request in item["requests"] if request["group"] == "packed")
            and sum(int(request["completion_tokens"]) for request in item["requests"] if request["group"] == "split")
            == sum(int(request["completion_tokens"]) for request in item["requests"] if request["group"] == "packed")
            for item in mixed
        ),
        "mixed": len(mixed),
        "stage_counts": dict(stage_counts),
        "duplicate_run_ids": duplicate_run_ids,
        "filenames_ok": filenames_ok,
        "model_ok": model_ok,
        "capacities_ok": capacities_ok,
        "profile_matrix_ok": profile_matrix_ok,
        "screen_matrix_ok": screen_matrix_ok,
        "confirm_matrix_entries": len(confirm_matrix),
    }


def confirm_matrix_ok(
    artifacts: list[dict[str, Any]], selection: dict[str, Any]
) -> bool:
    observed = Counter()
    for item in artifacts:
        exp = item["experiment"]
        if exp["experiment_stage"] == "confirm":
            observed[(exp["cell_id"], exp["policy_label"], repetition(item))] += 1

    global_label = f"global-{selection['global_request_tax']}"
    expected = set()
    for cell in CELLS:
        oracle_label = f"oracle-{selection['per_cell_request_tax'][cell]}"
        labels = (
            "fcfs-split-only",
            "fcfs-packed-only",
            "fcfs",
            "request-vtc",
            "token-vtc",
            "profiled-vtc",
            global_label,
            oracle_label,
            "instant-share",
        )
        for rep in (1, 2, 3):
            expected.update((cell, label, rep) for label in labels)
    return set(observed) == expected and all(observed[key] == 1 for key in expected)


def bootstrap_mean_ci(values: list[float], samples: int = 20000) -> tuple[float, float]:
    rng = random.Random(20260826)
    estimates = sorted(mean([rng.choice(values) for _ in values]) for _ in range(samples))
    return estimates[int(0.025 * samples)], estimates[int(0.975 * samples)]


def analyze_confirm(
    confirm: dict[str, dict[str, list[dict[str, float]]]], selection: dict[str, Any]
) -> tuple[list[str], dict[str, Any]]:
    lines = [
        "## Confirmation by cell", "",
        "| cell | token gap | profiled gap | global-tax gap | oracle-tax gap | instant gap | max resource ratio | composition ratio | split/packed queue p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    cells = {}
    complete = True
    global_label = selection["global_request_tax"]
    for cell in CELLS:
        grouped = confirm.get(cell, {})
        oracle_label = selection["per_cell_request_tax"][cell]
        labels = {
            "token": "token-vtc",
            "profiled": "profiled-vtc",
            "global": f"global-{global_label}",
            "oracle": f"oracle-{oracle_label}",
            "instant": "instant-share",
        }
        if any(len(grouped.get(label, [])) != 3 for label in labels.values()):
            complete = False
            continue
        values = {name: aggregate(grouped[label]) for name, label in labels.items()}
        cells[cell] = values
        token = values["token"]
        lines.append(
            f"| {cell} | {pct(token['gap'])} | {pct(values['profiled']['gap'])} | "
            f"{pct(values['global']['gap'])} | {pct(values['oracle']['gap'])} | "
            f"{pct(values['instant']['gap'])} | {num(token['max_dimension_ratio'], 2)}x | "
            f"{num(token['composition_ratio'], 2)}x | "
            f"{num(token['split_queue_p95'])}/{num(token['packed_queue_p95'])} |"
        )
    lines.append("")
    lines.extend(
        [
            "## Standard-VTC resource detail",
            "",
            "| cell | split/packed slowdown | peak state split/packed | peak KV split/packed | state-time ratio | KV-time ratio |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for cell in CELLS:
        if cell not in cells:
            continue
        token = cells[cell]["token"]
        lines.append(
            f"| {cell} | {pct(token['split_slowdown'])}/"
            f"{pct(token['packed_slowdown'])} | "
            f"{num(token['split_peak_state'], 1)}/"
            f"{num(token['packed_peak_state'], 1)} | "
            f"{num(token['split_peak_kv'], 0)}/"
            f"{num(token['packed_peak_kv'], 0)} | "
            f"{num(token['state_ratio'], 2)}x | "
            f"{num(token['kv_ratio'], 2)}x |"
        )
    lines.append("")
    if not complete or len(cells) != len(CELLS):
        lines.extend(["Confirmation incomplete.", ""])
        return lines, {"complete": False, "o1": None, "o2": None, "o3": None}

    o1_rows = [cells[cell]["token"] for cell in ("f16-base", "f32-base")]
    o1 = all(
        max(row["input_imbalance"], row["output_imbalance"]) <= 0.01
        and row["counter_bound_ratio"] <= 1.0
        and row["max_dimension_ratio"] >= 2.0
        and row["composition_ratio"] >= 4.0
        and row["gap"] >= 0.50
        for row in o1_rows
    )
    base_gap = cells["f16-base"]["token"]["gap"]
    state_gap = cells["f16-state-relaxed"]["token"]["gap"]
    kv_gap = cells["f16-kv-relaxed"]["token"]["gap"]
    f1_gap = cells["f1-base"]["token"]["gap"]
    state_reduction = (base_gap - state_gap) / max(base_gap, 1e-12)
    kv_reduction = (base_gap - kv_gap) / max(base_gap, 1e-12)
    o2 = state_reduction >= 0.20 and state_reduction >= kv_reduction and f1_gap < 0.25

    o3_rows = []
    for cell, values in cells.items():
        token_makespan = values["token"]["makespan"]
        scalars = [
            (name, values[name])
            for name in ("token", "profiled", "global", "oracle")
            if values[name]["makespan"] <= 1.05 * token_makespan
        ]
        strongest_name, strongest = min(
            scalars,
            key=lambda item: (item[1]["gap"], item[1]["makespan"]),
        )
        instant = values["instant"]
        improvement = (strongest["gap"] - instant["gap"]) / max(strongest["gap"], 1e-12)
        o3_rows.append(
            {
                "cell": cell,
                "strongest_scalar": strongest_name,
                "strongest_gap": strongest["gap"],
                "instant_gap": instant["gap"],
                "improvement": improvement,
                "direction": instant["gap"] < strongest["gap"],
                "makespan_change": instant["makespan"] / token_makespan - 1,
            }
        )
    improvements = [row["improvement"] for row in o3_rows]
    mean_improvement = mean(improvements)
    ci = bootstrap_mean_ci(improvements)
    wins = sum(row["direction"] for row in o3_rows)
    mean_makespan = mean([row["makespan_change"] for row in o3_rows])
    o3 = mean_improvement >= 0.15 and wins >= 7 and mean_makespan <= 0.05

    lines.extend(
        [
            "## Preregistered gates", "",
            f"- O1 VTC abstraction mismatch: **{'PASS' if o1 else 'FAIL'}**. "
            f"f16/f32 max resource ratios {num(o1_rows[0]['max_dimension_ratio'], 2)}x/"
            f"{num(o1_rows[1]['max_dimension_ratio'], 2)}x; composition ratios "
            f"{num(o1_rows[0]['composition_ratio'], 2)}x/{num(o1_rows[1]['composition_ratio'], 2)}x; "
            f"slowdown gaps {pct(o1_rows[0]['gap'])}/{pct(o1_rows[1]['gap'])}.",
            f"- O2 hybrid state specificity: **{'PASS' if o2 else 'FAIL'}**. "
            f"state-relaxed gap reduction {pct(state_reduction)}; KV-relaxed reduction "
            f"{pct(kv_reduction)}; identical-shape f1 gap {pct(f1_gap)}.",
            f"- O3 scalar repair insufficiency: **{'PASS' if o3 else 'FAIL'}**. "
            f"instant-share mean improvement {pct(mean_improvement)}, cell-bootstrap "
            f"95% CI [{pct(ci[0])}, {pct(ci[1])}], wins {wins}/9, mean makespan "
            f"change {pct(mean_makespan)}.", "",
        ]
    )
    return lines, {
        "complete": True,
        "o1": o1,
        "o2": o2,
        "o3": o3,
        "state_reduction": state_reduction,
        "kv_reduction": kv_reduction,
        "f1_gap": f1_gap,
        "o3_mean_improvement": mean_improvement,
        "o3_ci": ci,
        "o3_wins": wins,
        "o3_mean_makespan": mean_makespan,
        "cells": cells,
        "o3_rows": o3_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--selection-output", type=Path)
    parser.add_argument("--analysis-output", type=Path)
    args = parser.parse_args()

    artifacts = load_artifacts(args.result_dir)
    screen = stage_rows(artifacts, "screen")
    selection = select(screen)
    confirm_lines, decisions = analyze_confirm(stage_rows(artifacts, "confirm"), selection)
    audit = integrity(artifacts)
    audit["confirm_matrix_ok"] = (
        confirm_matrix_ok(artifacts, selection) if decisions["complete"] else None
    )
    profile = json.loads((args.result_dir / "profiled_vtc.json").read_text(encoding="utf-8"))
    expected_stage_counts = {"profile": 27, "screen": 117, "confirm": 243}
    integrity_pass = (
        audit["failures"] == 0
        and audit["cached"] == 0
        and audit["retractions"] == 0
        and audit["exact_traces"] == audit["artifacts"]
        and audit["vtc_traces"] == audit["vtc_artifacts"]
        and audit["resolved"] == audit["artifacts"]
        and audit["gpu"] == audit["artifacts"]
        and audit["equal_token_mixed"] == audit["mixed"]
        and not audit["duplicate_run_ids"]
        and audit["filenames_ok"]
        and audit["model_ok"]
        and audit["capacities_ok"]
        and audit["profile_matrix_ok"]
        and audit["screen_matrix_ok"]
        and (not decisions["complete"] or audit["confirm_matrix_ok"])
        and all(audit["stage_counts"].get(stage, 0) == count for stage, count in expected_stage_counts.items() if stage != "confirm" or decisions["complete"])
        and profile["gate"]["passed"]
    )
    overall = None if not decisions["complete"] else bool(
        integrity_pass and decisions["o1"] and decisions["o2"] and decisions["o3"]
    )
    report = "\n".join(
        [
            "# VTC state-tax observation report", "",
            f"Overall decision: **{'PENDING' if overall is None else ('PASS' if overall else 'FAIL')}**", "",
            "## Integrity", "",
            f"Artifacts {audit['artifacts']}; requests {audit['requests']}; failures "
            f"{audit['failures']}; cached tokens {audit['cached']}; retractions "
            f"{audit['retractions']}; exact resource traces {audit['exact_traces']}/"
            f"{audit['artifacts']}; VTC traces {audit['vtc_traces']}/{audit['vtc_artifacts']}; "
            f"resolved policy/weights {audit['resolved']}/{audit['artifacts']}; GPU 6,7 "
            f"metadata {audit['gpu']}/{audit['artifacts']}; equal-token mixed "
            f"{audit['equal_token_mixed']}/{audit['mixed']}.", "",
            f"Profile fit: **{'PASS' if profile['gate']['passed'] else 'FAIL'}**; median "
            f"relative error {pct(profile['median_relative_error'])}, max "
            f"{pct(profile['max_relative_error'])}.", "",
            "## Frozen selection", "",
            f"Global request-tax label: `{selection['global_request_tax']}`.", "",
            "| cell | frozen per-cell request-tax oracle |", "|---|---|",
            *[f"| {cell} | {selection['per_cell_request_tax'][cell]} |" for cell in CELLS],
            "", *confirm_lines,
        ]
    )
    compact = {
        "selection": selection,
        "integrity": audit,
        "integrity_pass": integrity_pass,
        "decisions": decisions,
        "overall": overall,
    }
    if args.output:
        args.output.write_text(report + "\n", encoding="utf-8")
    if args.selection_output:
        args.selection_output.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    if args.analysis_output:
        args.analysis_output.write_text(json.dumps(compact, indent=2) + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
