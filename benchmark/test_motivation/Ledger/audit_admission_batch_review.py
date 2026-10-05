#!/usr/bin/env python3
"""Read existing artifacts for the 2026-09-08 topic review; run no inference."""

import argparse
import json
import statistics
from pathlib import Path

import analyze_admission_batch_motivation as analysis


def service_events(item):
    events = []
    for req in item["requests"]:
        group = req["group"]
        if req.get("first_token_s") is not None:
            events.append((req["first_token_s"], group, req["prompt_tokens"]))
        previous = 0
        for timestamp, completed in req.get("token_events", []):
            if completed > previous:
                events.append((timestamp, group, 2 * (completed - previous)))
                previous = completed
    return sorted(events)


def pulse_window(item, start, end):
    groups = ("incumbent", "pulser")
    totals = {
        g: sum(r["prompt_tokens"] + 2 * r["completion_tokens"]
               for r in item["requests"] if r["group"] == g)
        for g in groups
    }
    assert totals[groups[0]] == totals[groups[1]]
    assert end > start
    denominator = totals[groups[0]]
    received = dict.fromkeys(groups, 0)
    events = service_events(item)
    for timestamp, group, amount in events:
        if timestamp <= start:
            received[group] += amount
    initial = (received["pulser"] - received["incumbent"]) / denominator
    max_absolute = abs(initial)
    max_increment = 0.0
    for timestamp, group, amount in events:
        if start < timestamp <= end:
            received[group] += amount
            gap = (received["pulser"] - received["incumbent"]) / denominator
            max_absolute = max(max_absolute, abs(gap))
            max_increment = max(max_increment, abs(gap - initial))
    return {
        "start_s": start,
        "end_s": end,
        "initial_signed_gap": initial,
        "max_absolute_gap": max_absolute,
        "max_incremental_gap": max_increment,
    }


def audit(result_dir):
    items = analysis.load_artifacts(result_dir)
    assert len(items) == 57
    mismatches = []
    mapping = {"fcfs": "fcfs", "vtc": "tenant-vtc",
               "instant": "tenant-instant-share"}
    for item in items:
        exp, server = item["experiment"], item["server"]
        resolved = server.get("resolved_experiment_policy", [])
        if server.get("max_total_num_tokens") != 196608:
            mismatches.append([exp["run_id"], "kv_capacity"])
        if not resolved:
            mismatches.append([exp["run_id"], "missing_policy"])
        for policy in resolved:
            if policy.get("schedule_policy") != mapping[exp["runtime_policy"]]:
                mismatches.append([exp["run_id"], "policy"])
            if exp["runtime_policy"] == "vtc":
                for weight in ("request", "input", "input_quadratic", "output",
                               "input_output", "output_quadratic"):
                    expected = exp[weight + "_weight"]
                    actual = policy.get("tenant_vtc_" + weight + "_weight")
                    if actual is None or abs(actual - expected) > 1e-10:
                        mismatches.append([exp["run_id"], weight])

    iso = analysis.isolation_map(items, "partition")
    partition_rows = []
    for item in items:
        exp = item["experiment"]
        if exp["mode"] == "partition" and exp["composition"] == "mixed":
            metrics = analysis.mixed_metric(item, iso)
            partition_rows.append({
                "run": exp["run_id"], "cell": exp["cell_id"],
                "policy": exp["policy_label"], "rep": analysis.repetition(item),
                **{k: metrics[k] for k in (
                    "gap", "split_p95", "packed_p95", "split_iso_p95",
                    "packed_iso_p95", "makespan", "split_active_per_req",
                    "split_decode_interval")},
            })
    means = {}
    for cell in analysis.CELLS:
        rows = [r for r in partition_rows
                if r["cell"] == cell and r["policy"] == "token-vtc"]
        means[cell] = statistics.fmean(r["makespan"] for r in rows)
    pulse_items = [x for x in items if x["experiment"]["mode"] == "pulse"
                   and x["experiment"]["composition"] == "mixed"]
    start = max(r["scheduled_arrival_s"] for x in pulse_items for r in x["requests"])
    end = min(analysis.delivered_service_metric(x)[key] for x in pulse_items
              for key in ("incumbent_finish", "pulser_finish"))
    pulse_rows = [{
        "run": x["experiment"]["run_id"],
        "original": analysis.delivered_service_metric(x),
        "common_window_sensitivity": pulse_window(x, start, end),
    } for x in pulse_items]
    return {
        "review_date": "2026-09-08",
        "scope": "post-hoc read-only audit; no new experimental gate or confidence claim",
        "original_integrity": analysis.integrity(items),
        "effective_policy_weight_kv_mismatches": mismatches,
        "partition_runs": partition_rows,
        "token_vtc_mean_makespan_s": means,
        "high_state_uncapping_makespan_change":
            means["high-state-auto"] / means["high-state-capped"] - 1,
        "pulse_runs": pulse_rows,
        "pulse_limit": "Neither original nor common-window sensitivity establishes continuously waiting backlog or VTC service-counter bounds.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    args = parser.parse_args()
    result = audit(args.result_dir)
    print(json.dumps(result, indent=2, allow_nan=False))
    if result["effective_policy_weight_kv_mismatches"]:
        raise SystemExit(1)
