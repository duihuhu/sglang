#!/usr/bin/env python3
"""Aggregate replicated hybrid-motivation experiment artifacts."""

from __future__ import annotations

import argparse
import json
import random
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class Record:
    path: Path
    mode: str
    condition: str
    replicate: int
    artifact: dict[str, Any]


def infer_condition(artifact: dict[str, Any]) -> str:
    experiment = artifact["experiment"]
    if experiment["mode"] == "lifetime-order":
        return experiment["order"]
    groups = set(artifact["summary"]["groups"])
    if groups == {"state-heavy"}:
        return "state-isolated"
    if groups == {"kv-heavy"}:
        return "kv-isolated"
    return experiment["order"]


def load_records(input_dir: Path) -> list[Record]:
    records: list[Record] = []
    for path in sorted(input_dir.glob("*.json")):
        match = re.search(r"[-_]r(\d+)\.json$", path.name)
        if match is None:
            raise ValueError(f"Cannot infer replicate from {path.name}")
        artifact = json.loads(path.read_text(encoding="utf-8"))
        records.append(
            Record(
                path=path,
                mode=artifact["experiment"]["mode"],
                condition=infer_condition(artifact),
                replicate=int(match.group(1)),
                artifact=artifact,
            )
        )
    return records


def validate(records: list[Record]) -> None:
    keys = [(record.mode, record.condition, record.replicate) for record in records]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate mode/condition/replicate artifacts")
    for record in records:
        failed = [request for request in record.artifact["requests"] if not request["success"]]
        if failed:
            raise ValueError(f"{record.path} contains {len(failed)} failed requests")
        retractions = record.artifact["summary"]["overall"]["total_retractions"]
        if retractions:
            raise ValueError(f"{record.path} contains {retractions} retractions")


def select(records: list[Record], mode: str, condition: str) -> list[Record]:
    selected = [
        record
        for record in records
        if record.mode == mode and record.condition == condition
    ]
    return sorted(selected, key=lambda record: record.replicate)


def group_metric(record: Record, group: str, metric: str) -> float:
    return float(record.artifact["summary"]["groups"][group][metric])


def overall_metric(record: Record, metric: str) -> float:
    return float(record.artifact["summary"]["overall"][metric])


def mean_std(values: list[float]) -> tuple[float, float]:
    return statistics.fmean(values), statistics.stdev(values) if len(values) > 1 else 0.0


def paired_values(
    left: list[Record],
    right: list[Record],
    metric: Callable[[Record], float],
) -> list[tuple[float, float]]:
    left_by_rep = {record.replicate: record for record in left}
    right_by_rep = {record.replicate: record for record in right}
    if set(left_by_rep) != set(right_by_rep):
        raise ValueError("Replicate sets do not match")
    return [
        (metric(left_by_rep[rep]), metric(right_by_rep[rep]))
        for rep in sorted(left_by_rep)
    ]


def bootstrap_mean_ci(values: list[float], samples: int = 20_000) -> tuple[float, float]:
    rng = random.Random(0)
    boot = [
        statistics.fmean(rng.choice(values) for _ in values) for _ in range(samples)
    ]
    boot.sort()
    return boot[int(samples * 0.025)], boot[int(samples * 0.975)]


def fmt(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def lifetime_section(records: list[Record]) -> tuple[list[str], bool]:
    conditions = {
        name: select(records, "lifetime-order", name)
        for name in ("short-first", "long-first", "interleave")
    }
    lines = [
        "## M-A: Lifetime Order Sensitivity",
        "",
        "| condition | short p95 JCT (s) | short p95 TTFT (s) | overall p95 JCT (s) | <=10s completions |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, items in conditions.items():
        jct = [group_metric(item, "short", "p95_latency_s") for item in items]
        ttft = [group_metric(item, "short", "p95_ttft_s") for item in items]
        overall = [overall_metric(item, "p95_latency_s") for item in items]
        goodput = [overall_metric(item, "completed_within_slo") for item in items]
        jct_mean, jct_std = mean_std(jct)
        ttft_mean, ttft_std = mean_std(ttft)
        overall_mean, overall_std = mean_std(overall)
        lines.append(
            f"| {name} | {fmt(jct_mean)} +/- {fmt(jct_std)} | "
            f"{fmt(ttft_mean)} +/- {fmt(ttft_std)} | "
            f"{fmt(overall_mean)} +/- {fmt(overall_std)} | {fmt(statistics.fmean(goodput), 1)} |"
        )

    pairs = paired_values(
        conditions["short-first"],
        conditions["long-first"],
        lambda item: group_metric(item, "short", "p95_latency_s"),
    )
    relative_improvements = [(long - short) / long for short, long in pairs]
    mean_improvement = statistics.fmean(relative_improvements)
    improvement_ci = bootstrap_mean_ci(relative_improvements)

    short_goodput = [
        overall_metric(item, "completed_within_slo")
        for item in conditions["short-first"]
    ]
    long_goodput = [
        overall_metric(item, "completed_within_slo")
        for item in conditions["long-first"]
    ]
    goodput_improvement = (
        statistics.fmean(short_goodput) - statistics.fmean(long_goodput)
    ) / statistics.fmean(long_goodput)
    gate_passed = (
        (mean_improvement >= 0.20 or goodput_improvement >= 0.15)
        and improvement_ci[0] > 0
    )
    lines.extend(
        [
            "",
            f"- `short-first` 相对 `long-first` 的 short p95 JCT 改善：{fmt(mean_improvement * 100, 1)}% "
            f"(paired bootstrap 95% CI {fmt(improvement_ci[0] * 100, 1)}% to {fmt(improvement_ci[1] * 100, 1)}%)。",
            f"- 10s SLO goodput 改善：{fmt(goodput_improvement * 100, 1)}%。",
            f"- Preliminary gate: **{'PASS' if gate_passed else 'FAIL'}**。",
            "",
            "解释：FAIL 不表示调度顺序无影响，而是简单的 short-lifetime-first 没有表现出预期优势。当前结果不支持把 Topic A 直接包装成 SJF/lifetime heuristic；需要先做 state-cap sensitivity 和 resource-time-vs-SJF cross case。",
        ]
    )
    return lines, gate_passed


def paired_slowdowns(
    concurrent: list[Record],
    isolated: list[Record],
    group: str,
    metric: str,
) -> list[float]:
    pairs = paired_values(
        concurrent,
        isolated,
        lambda item: group_metric(item, group, metric),
    )
    return [co_run / alone for co_run, alone in pairs]


def fairness_section(records: list[Record]) -> tuple[list[str], bool]:
    conditions = {
        name: select(records, "fairness-pair", name)
        for name in (
            "state-first",
            "kv-first",
            "interleave",
            "state-isolated",
            "kv-isolated",
        )
    }
    lines = [
        "## M-B: Equal Tokens, Unequal State Pressure",
        "",
        "| condition | group | p95 JCT (s) | p95 TTFT (s) | peak decode state slots |",
        "|---|---|---:|---:|---:|",
    ]
    for condition in ("state-isolated", "kv-isolated", "state-first", "kv-first", "interleave"):
        items = conditions[condition]
        groups = sorted(items[0].artifact["summary"]["groups"])
        for group in groups:
            jct = [group_metric(item, group, "p95_latency_s") for item in items]
            ttft = [group_metric(item, group, "p95_ttft_s") for item in items]
            slots = [
                group_metric(item, group, "peak_decode_state_slots_lower_bound")
                for item in items
            ]
            lines.append(
                f"| {condition} | {group} | {fmt(statistics.fmean(jct))} | "
                f"{fmt(statistics.fmean(ttft))} | {fmt(statistics.fmean(slots), 0)} |"
            )

    state_slowdowns: dict[str, list[float]] = {}
    kv_jct_slowdowns: dict[str, list[float]] = {}
    kv_ttft_slowdowns: dict[str, list[float]] = {}
    for condition in ("state-first", "kv-first", "interleave"):
        state_slowdowns[condition] = paired_slowdowns(
            conditions[condition],
            conditions["state-isolated"],
            "state-heavy",
            "p95_latency_s",
        )
        kv_jct_slowdowns[condition] = paired_slowdowns(
            conditions[condition],
            conditions["kv-isolated"],
            "kv-heavy",
            "p95_latency_s",
        )
        kv_ttft_slowdowns[condition] = paired_slowdowns(
            conditions[condition],
            conditions["kv-isolated"],
            "kv-heavy",
            "p95_ttft_s",
        )

    state_peak = statistics.fmean(
        group_metric(item, "state-heavy", "peak_decode_state_slots_lower_bound")
        for item in conditions["state-first"]
    )
    kv_peak = statistics.fmean(
        group_metric(item, "kv-heavy", "peak_decode_state_slots_lower_bound")
        for item in conditions["state-first"]
    )
    peak_ratio = state_peak / kv_peak
    max_slowdown = max(
        statistics.fmean(values)
        for values in list(state_slowdowns.values()) + list(kv_jct_slowdowns.values())
    )
    order_effect = statistics.fmean(state_slowdowns["kv-first"]) / statistics.fmean(
        state_slowdowns["state-first"]
    )

    token_errors = []
    for condition in ("state-first", "kv-first", "interleave"):
        for item in conditions[condition]:
            groups = item.artifact["summary"]["groups"]
            state_tokens = groups["state-heavy"]["input_tokens_completed"] + groups[
                "state-heavy"
            ]["output_tokens_completed"]
            kv_tokens = groups["kv-heavy"]["input_tokens_completed"] + groups[
                "kv-heavy"
            ]["output_tokens_completed"]
            token_errors.append(abs(state_tokens - kv_tokens) / kv_tokens)

    gate_passed = (
        max(token_errors) < 0.01
        and peak_ratio >= 8.0
        and max_slowdown >= 1.5
        and order_effect >= 1.25
    )
    lines.extend(
        [
            "",
            f"- 两组 completed token service 最大误差：{fmt(max(token_errors) * 100, 2)}%。",
            f"- state-heavy / KV-heavy peak decode state-slot ratio：{fmt(peak_ratio, 1)}x。",
            f"- 最大 mean p95 JCT slowdown：{fmt(max_slowdown, 2)}x。",
            f"- state-heavy 的 slowdown 在 `kv-first` 相对 `state-first` 放大：{fmt(order_effect, 2)}x。",
            f"- KV-heavy 在 `state-first` 下的 mean p95 TTFT slowdown：{fmt(statistics.fmean(kv_ttft_slowdowns['state-first']), 2)}x。",
            f"- Preliminary gate: **{'PASS' if gate_passed else 'FAIL'}**。",
            "",
            "解释：PASS 只证明 equal-token groups 的物理资源压力与干扰不相等；还没有证明 VTC/DRF baseline 失败。下一阶段必须加入可信 tenant ID、精确 allocator timestamps 和 tuned VTC 对照。",
        ]
    )
    return lines, gate_passed


def build_report(records: list[Record]) -> str:
    versions = sorted(
        {
            record.artifact["server"].get("version", "unknown")
            for record in records
        }
    )
    lifetime_lines, lifetime_passed = lifetime_section(records)
    fairness_lines, fairness_passed = fairness_section(records)
    lines = [
        "# Preliminary Hybrid Motivation Results",
        "",
        "> 三次重复；同一 Kimi-Linear-48B-A3B TP=2 server process；结果仅用于 motivation go/no-go。",
        "",
        f"- Artifacts: {len(records)}",
        f"- Server version: {', '.join(versions)}",
        "- 所有请求成功，所有正式 condition 的 retraction 数为 0。",
        "",
        *lifetime_lines,
        "",
        *fairness_lines,
        "",
        "## Decision",
        "",
        f"- Topic A preliminary gate: **{'PASS' if lifetime_passed else 'FAIL'}**。不进入在线 lifetime predictor 实现，先做更便宜的 capacity sensitivity/cross-case。",
        f"- Topic B physical-motivation gate: **{'PASS' if fairness_passed else 'FAIL'}**。可以进入精确 tenant resource accounting 和 VTC kill test。",
        "- 当前不应把两个 topic 都当作已成立；证据优先支持 Topic B。",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = load_records(args.input_dir)
    validate(records)
    report = build_report(records)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
