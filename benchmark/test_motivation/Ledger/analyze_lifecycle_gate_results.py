#!/usr/bin/env python3
"""Analyze the five preregistered lifecycle-cost motivation gates."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable


def load_artifacts(result_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(result_dir.glob("m[1-5]-*.json")):
        try:
            artifact = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    return artifacts


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else math.nan


def fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.{digits}f}"


def pct(value: float) -> str:
    if not math.isfinite(value):
        return "-"
    return f"{value:.1%}"


def group_by(
    items: list[dict[str, Any]], key: Callable[[dict[str, Any]], Any]
) -> dict[Any, list[dict[str, Any]]]:
    grouped: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        grouped[key(item)].append(item)
    return grouped


def repetition(artifact: dict[str, Any]) -> int:
    match = re.search(r"rep(\d+)$", artifact["experiment"]["run_id"])
    return int(match.group(1)) if match else 0


def p95(artifact: dict[str, Any], group: str = "overall") -> float:
    if group == "overall":
        return float(artifact["summary"]["overall"]["p95_jct_s"])
    return float(artifact["summary"]["groups"][group]["p95_jct_s"])


def makespan(artifact: dict[str, Any]) -> float:
    return max(float(request["finished_s"]) for request in artifact["requests"])


def resource(
    artifact: dict[str, Any], group: str, metric: str
) -> float:
    values = artifact["summary"]["groups"][group].get(
        "scheduler_active_resources", {}
    )
    return float(values.get(metric, 0))


def analyze_m1(artifacts: list[dict[str, Any]]) -> tuple[list[str], bool]:
    selected = [
        item
        for item in artifacts
        if item["experiment"]["run_id"].startswith("m1-")
    ]
    grouped = group_by(
        selected,
        lambda item: (
            int(item["experiment"]["prompt_len"]),
            int(item["experiment"]["output_len"]),
            float(item["experiment"]["reuse_fraction"]),
        ),
    )
    rows = []
    for key, items in grouped.items():
        rows.append(
            {
                "key": key,
                "raw_tokens": key[0] + key[1],
                "jct": mean([p95(item) for item in items]),
                "rep_jct": {repetition(item): p95(item) for item in items},
            }
        )

    comparable = 0
    inversions = []
    for left in rows:
        for right in rows:
            if left["raw_tokens"] >= right["raw_tokens"]:
                continue
            comparable += 1
            ratio = left["jct"] / right["jct"]
            if ratio <= 1.05:
                continue
            common_reps = set(left["rep_jct"]) & set(right["rep_jct"])
            stable = bool(common_reps) and all(
                left["rep_jct"][rep] > 1.05 * right["rep_jct"][rep]
                for rep in common_reps
            )
            inversions.append((ratio, stable, left["key"], right["key"]))
    inversions.sort(reverse=True)
    inversion_rate = len(inversions) / comparable if comparable else 0
    largest = inversions[0][0] if inversions else 1
    stable_large = any(ratio >= 1.15 and stable for ratio, stable, _, _ in inversions)
    passed = inversion_rate >= 0.10 and stable_large

    lines = [
        "## M1: token/length ranking",
        "",
        f"Decision: **{'PASS' if passed else 'FAIL'}**. Comparable pairs: {comparable}; "
        f">5% inversions: {len(inversions)} ({inversion_rate:.1%}); largest inversion: {largest:.2f}x.",
        "",
        "| expensive lower-token class | cheaper higher-token class | p95 ratio | stable across reps |",
        "|---|---|---:|---|",
    ]
    for ratio, stable, left, right in inversions[:10]:
        lines.append(
            f"| P={left[0]}, O={left[1]}, reuse={left[2]:.1f} | "
            f"P={right[0]}, O={right[1]}, reuse={right[2]:.1f} | "
            f"{ratio:.2f}x | {'yes' if stable else 'no'} |"
        )
    lines.append("")
    return lines, passed


def analyze_m2(artifacts: list[dict[str, Any]]) -> tuple[list[str], bool]:
    selected = [
        item
        for item in artifacts
        if item["experiment"]["run_id"].startswith("m2-")
    ]
    grouped = group_by(
        selected,
        lambda item: (item["experiment"]["regime"], item["experiment"]["order"]),
    )
    values = {
        key: mean([p95(item) for item in items]) for key, items in grouped.items()
    }
    regimes = sorted({key[0] for key in values})
    orders = ["decode-first", "prefill-first", "interleave"]
    best = {
        regime: min(orders, key=lambda order: values.get((regime, order), math.inf))
        for regime in regimes
    }
    worst_regret = {}
    for order in orders:
        regrets = []
        for regime in regimes:
            oracle = values.get((regime, best[regime]), math.nan)
            current = values.get((regime, order), math.nan)
            if math.isfinite(oracle) and math.isfinite(current):
                regrets.append((current - oracle) / oracle)
        worst_regret[order] = max(regrets) if regrets else math.nan
    reversal = (
        "state-bound" in best
        and "kv-bound" in best
        and best["state-bound"] != best["kv-bound"]
    )
    robust_fixed_regret = min(worst_regret.values()) if worst_regret else math.nan
    passed = reversal and robust_fixed_regret >= 0.15

    lines = [
        "## M2: state-dependent ranking",
        "",
        f"Decision: **{'PASS' if passed else 'FAIL'}**. State/KV best-order reversal: "
        f"{'yes' if reversal else 'no'}; best fixed order worst-regime regret: {pct(robust_fixed_regret)}.",
        "",
        "| regime | decode first p95 | prefill first p95 | interleave p95 | best |",
        "|---|---:|---:|---:|---|",
    ]
    for regime in regimes:
        lines.append(
            f"| {regime} | {fmt(values.get((regime, 'decode-first'), math.nan))} | "
            f"{fmt(values.get((regime, 'prefill-first'), math.nan))} | "
            f"{fmt(values.get((regime, 'interleave'), math.nan))} | {best[regime]} |"
        )
    lines.append("")
    return lines, passed


def analyze_m3(artifacts: list[dict[str, Any]]) -> tuple[list[str], bool]:
    selected = [
        item
        for item in artifacts
        if item["experiment"]["run_id"].startswith("m3-")
    ]
    grouped = group_by(
        selected,
        lambda item: (item["experiment"]["regime"], item["experiment"]["order"]),
    )
    rows = {}
    for key, items in grouped.items():
        rows[key] = {
            "p95": mean([p95(item) for item in items]),
            "background": mean([p95(item, "background") for item in items]),
            "cached": mean(
                [
                    float(
                        item["summary"]["groups"]["future-reuse"].get(
                            "mean_cached_tokens", 0
                        )
                    )
                    for item in items
                ]
            ),
            "makespan": mean([makespan(item) for item in items]),
        }
    fcfs_pressure = rows.get(("fcfs", "pressure-first"), {})
    fcfs_reuse = rows.get(("fcfs", "reuse-first"), {})
    lpm_pressure = rows.get(("lpm", "pressure-first"), {})
    ordering_gain = (
        (fcfs_pressure.get("p95", math.nan) - fcfs_reuse.get("p95", math.nan))
        / fcfs_pressure.get("p95", math.nan)
    )
    lpm_residual = (
        (lpm_pressure.get("p95", math.nan) - fcfs_reuse.get("p95", math.nan))
        / lpm_pressure.get("p95", math.nan)
    )
    cache_delta = fcfs_reuse.get("cached", 0) - fcfs_pressure.get("cached", 0)
    passed = ordering_gain >= 0.15 and cache_delta >= 8192 and lpm_residual >= 0.05

    lines = [
        "## M3: admission/batch/eviction coupling",
        "",
        f"Decision: **{'PASS' if passed else 'FAIL'}**. FCFS reuse-first p95 gain: "
        f"{pct(ordering_gain)}; cached-token delta: {fmt(cache_delta, 0)}; "
        f"residual gain over pressure-first LPM: {pct(lpm_residual)}.",
        "",
        "| server policy | emulated order | overall p95 | background p95 | reuse cached tokens | makespan |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, row in sorted(rows.items()):
        lines.append(
            f"| {key[0]} | {key[1]} | {fmt(row['p95'])} | "
            f"{fmt(row['background'])} | {fmt(row['cached'], 0)} | {fmt(row['makespan'])} |"
        )
    lines.append("")
    return lines, passed


def analyze_m4(artifacts: list[dict[str, Any]]) -> tuple[list[str], bool]:
    selected = [
        item
        for item in artifacts
        if item["experiment"]["run_id"].startswith("m4-")
    ]
    grouped = group_by(
        selected,
        lambda item: (
            item["experiment"]["runtime_policy"],
            item["experiment"]["composition"],
        ),
    )
    split_iso = grouped.get(("fcfs", "split-only"), [])
    packed_iso = grouped.get(("fcfs", "packed-only"), [])
    mixed_vtc = grouped.get(("vtc", "mixed"), [])
    split_tokens = (
        sum(int(req["input_len"]) + int(req["output_len"]) for req in split_iso[0]["requests"])
        if split_iso
        else 0
    )
    packed_tokens = (
        sum(int(req["input_len"]) + int(req["output_len"]) for req in packed_iso[0]["requests"])
        if packed_iso
        else 0
    )
    split_state_per_token = mean(
        [resource(item, "split", "state_slot_seconds") for item in split_iso]
    ) / max(split_tokens, 1)
    packed_state_per_token = mean(
        [resource(item, "packed", "state_slot_seconds") for item in packed_iso]
    ) / max(packed_tokens, 1)
    split_kv_per_token = mean(
        [resource(item, "split", "kv_token_seconds") for item in split_iso]
    ) / max(split_tokens, 1)
    packed_kv_per_token = mean(
        [resource(item, "packed", "kv_token_seconds") for item in packed_iso]
    ) / max(packed_tokens, 1)
    resource_ratio = max(
        split_state_per_token / max(packed_state_per_token, 1e-12),
        packed_state_per_token / max(split_state_per_token, 1e-12),
        split_kv_per_token / max(packed_kv_per_token, 1e-12),
        packed_kv_per_token / max(split_kv_per_token, 1e-12),
    )
    split_iso_p95 = mean([p95(item, "split") for item in split_iso])
    packed_iso_p95 = mean([p95(item, "packed") for item in packed_iso])
    split_mixed_p95 = mean([p95(item, "split") for item in mixed_vtc])
    packed_mixed_p95 = mean([p95(item, "packed") for item in mixed_vtc])
    split_slowdown = split_mixed_p95 / split_iso_p95 - 1
    packed_slowdown = packed_mixed_p95 / packed_iso_p95 - 1
    slowdown_gap = abs(split_slowdown - packed_slowdown)
    equal_tokens = split_tokens == packed_tokens and split_tokens > 0
    passed = equal_tokens and resource_ratio >= 2 and slowdown_gap >= 0.15

    lines = [
        "## M4: equal-token tenant cost transfer",
        "",
        f"Decision: **{'PASS' if passed else 'FAIL'}**. Requested tokens: "
        f"split={split_tokens}, packed={packed_tokens}; maximum per-token resource ratio: "
        f"{resource_ratio:.2f}x; VTC normalized-slowdown gap: {pct(slowdown_gap)}.",
        "",
        "| tenant shape | isolated p95 | mixed VTC p95 | normalized slowdown | state-slot-s/token | KV-token-s/token |",
        "|---|---:|---:|---:|---:|---:|",
        f"| split | {fmt(split_iso_p95)} | {fmt(split_mixed_p95)} | {pct(split_slowdown)} | {fmt(split_state_per_token, 6)} | {fmt(split_kv_per_token, 3)} |",
        f"| packed | {fmt(packed_iso_p95)} | {fmt(packed_mixed_p95)} | {pct(packed_slowdown)} | {fmt(packed_state_per_token, 6)} | {fmt(packed_kv_per_token, 3)} |",
        "",
    ]
    return lines, passed


def analyze_m5(artifacts: list[dict[str, Any]]) -> tuple[list[str], bool]:
    selected = [
        item
        for item in artifacts
        if item["experiment"]["run_id"].startswith("m5-")
    ]
    grouped = group_by(selected, lambda item: item["experiment"]["order"])
    rows = {
        order: {
            "p95": mean([p95(item) for item in items]),
            "makespan": mean([makespan(item) for item in items]),
            "slo": mean(
                [float(item["summary"]["overall"]["slo_goodput"]) for item in items]
            ),
        }
        for order, items in grouped.items()
    }
    baselines = {
        "fcfs": "decode,prefill,reuse,medium",
        "sjf": "prefill,reuse,medium,decode",
        "lpm": "reuse,decode,prefill,medium",
        "future-memory": "decode,medium,prefill,reuse",
        "composite": "reuse,decode,medium,prefill",
    }
    present_baselines = {
        name: rows[order]
        for name, order in baselines.items()
        if order in rows
    }
    strongest_name, strongest = min(
        present_baselines.items(), key=lambda item: item[1]["p95"]
    ) if present_baselines else ("missing", {"p95": math.nan, "makespan": math.nan, "slo": math.nan})
    oracle_order, oracle = min(
        rows.items(), key=lambda item: (-item[1]["slo"], item[1]["p95"])
    ) if rows else ("missing", {"p95": math.nan, "makespan": math.nan, "slo": math.nan})
    headroom = (strongest["p95"] - oracle["p95"]) / strongest["p95"]
    throughput_ok = oracle["makespan"] <= 1.05 * strongest["makespan"]
    passed = len(rows) == 24 and headroom >= 0.15 and throughput_ok

    lines = [
        "## M5: empirical oracle headroom",
        "",
        f"Decision: **{'PASS' if passed else 'FAIL'}**. Orders completed: {len(rows)}/24; "
        f"oracle={oracle_order}; strongest baseline={strongest_name}; p95 headroom={pct(headroom)}; "
        f"throughput bound={'pass' if throughput_ok else 'fail'}.",
        "",
        "| policy/order | class order | overall p95 | SLO goodput | makespan |",
        "|---|---|---:|---:|---:|",
    ]
    for name, order in baselines.items():
        row = rows.get(order)
        if row:
            lines.append(
                f"| {name} | {order} | {fmt(row['p95'])} | {fmt(row['slo'], 1)} | {fmt(row['makespan'])} |"
            )
    if oracle_order in rows:
        lines.append(
            f"| empirical oracle | {oracle_order} | {fmt(oracle['p95'])} | "
            f"{fmt(oracle['slo'], 1)} | {fmt(oracle['makespan'])} |"
        )
    lines.append("")
    return lines, passed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    artifacts = load_artifacts(args.result_dir)
    sections = []
    decisions = []
    for label, analyzer in (
        ("M1", analyze_m1),
        ("M2", analyze_m2),
        ("M3", analyze_m3),
        ("M4", analyze_m4),
        ("M5", analyze_m5),
    ):
        lines, passed = analyzer(artifacts)
        sections.extend(lines)
        decisions.append((label, passed))
    report = "\n".join(
        [
            "# Lifecycle-cost motivation gate results",
            "",
            f"Artifacts analyzed: {len(artifacts)}",
            "",
            "| gate | decision |",
            "|---|---|",
            *[f"| {label} | {'PASS' if passed else 'FAIL'} |" for label, passed in decisions],
            "",
            *sections,
        ]
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report + "\n", encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
