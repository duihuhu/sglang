#!/usr/bin/env python3
"""Read-only audit of historical two-tenant confirmation artifacts.

Client output progress is deliberately not called VTC service or fairness.
The audit creates a new output directory and never edits its source artifacts.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics


GROUPS = ("packed", "split")
CAPS = (48, 96)
FACTORS = (16, 32)
REPS = tuple(range(10, 15))
COMPOSITIONS = ("mixed", "packed-only", "split-only")


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected finite number, got {value!r}")
    return value


def output_events(requests):
    """Convert cumulative per-request stream counts to tenant output deltas."""
    deltas = defaultdict(int)
    ids = set()
    for req in requests:
        if req["rid"] in ids:
            raise ValueError("Duplicate request ID")
        ids.add(req["rid"])
        if (not req["success"] or req.get("error") or req["cached_tokens"]
                or req["num_retractions"] or req["prompt_tokens"] != req["input_len"]
                or req["completion_tokens"] != req["output_len"]):
            raise ValueError("Invalid request integrity")
        submitted, finished = number(req["submitted_s"]), number(req["finished_s"])
        previous_t, previous_n = submitted, 0
        if finished < submitted:
            raise ValueError("Negative request duration")
        for timestamp, count in req["token_events"]:
            timestamp, count = number(timestamp), number(count)
            if (timestamp < previous_t or timestamp > finished or count <= previous_n
                    or int(count) != count or count > req["completion_tokens"]):
                raise ValueError("Invalid cumulative stream event")
            deltas[(timestamp, req["group"])] += count - previous_n
            previous_t, previous_n = timestamp, count
        if previous_n != req["completion_tokens"] or previous_n <= 0:
            raise ValueError("Incomplete output stream")
        if req["first_token_s"] != req["token_events"][0][0]:
            raise ValueError("First-token timestamp mismatch")
    return [(t, group, count) for (t, group), count in sorted(deltas.items())]


def milestones(events, group, total):
    result = {}
    cumulative = 0
    for timestamp, tenant, count in events:
        if tenant != group:
            continue
        cumulative += count
        for percent in (25, 50, 75, 100):
            if cumulative >= math.ceil(total * percent / 100) and str(percent) not in result:
                result[str(percent)] = timestamp
    if len(result) != 4 or cumulative != total:
        raise ValueError("Output milestones do not cover the full stream")
    return result


def common_client_window(items):
    start = max(number(r["submitted_s"]) for item in items for r in item["requests"])
    end = min(max(number(r["finished_s"]) for r in item["requests"] if r["group"] == group)
              for item in items for group in GROUPS)
    return (start, end) if start < end else None


def output_window(events, start, end, totals):
    """Use the same client-time window for both caps; subtract starting offsets.

    This only guarantees client-outstanding work after all submissions. It
    neither proves server waiting backlog nor includes executed input service.
    """
    if start >= end or min(totals.values()) <= 0:
        raise ValueError("Invalid client output window")
    initial = {g: sum(n for t, group, n in events if group == g and t <= start)
               for g in GROUPS}
    increments = dict.fromkeys(GROUPS, 0)
    by_time = defaultdict(Counter)
    for timestamp, group, count in events:
        if start < timestamp <= end:
            by_time[timestamp][group] += count
    previous_t, gap, maximum, area = start, 0.0, 0.0, 0.0
    for timestamp, counts in sorted(by_time.items()):
        area += abs(gap) * (timestamp - previous_t)
        for group, count in counts.items():
            increments[group] += count
        gap = increments["split"] / totals["split"] - increments["packed"] / totals["packed"]
        maximum = max(maximum, abs(gap))
        previous_t = timestamp
    area += abs(gap) * (end - previous_t)
    return {
        "start_s": start, "end_s": end, "duration_s": end - start,
        "initial_delivered_output": initial, "incremental_delivered_output": increments,
        "max_abs_incremental_output_fraction_gap": maximum,
        "time_mean_abs_incremental_output_fraction_gap": area / (end - start),
        "end_split_minus_packed_output_fraction": gap,
        "denominator": "Each tenant's full-run output count; input excluded",
        "scope": "Client-outstanding drain window, NOT server-waiting fairness",
    }


def features(item):
    vtc = item.get("scheduler_vtc_trace") or {}
    resource = item.get("scheduler_resource_trace") or {}
    batch = item.get("scheduler_batch_trace") or {}
    events = batch.get("events", [])
    return {
        "client_output_stream": all(bool(r.get("token_events")) for r in item["requests"]),
        "vtc_counter_summary": bool(vtc.get("counters")),
        "vtc_counter_events": bool(vtc.get("events")),
        "resource_summary": bool(resource.get("tenants")),
        "resource_events": bool(resource.get("events")),
        "batch_shape_events": bool(events),
        "per_request_batch_work": bool(events) and all(bool(e.get("members")) for e in events),
        "per_tenant_waiting_snapshots": bool(events) and all("waiting_by_tenant" in e for e in events),
        "admission_block_records": bool(batch.get("admission_blocks")),
        "server_event_clock_origin": "event_clock_origin_s" in batch and "event_clock_origin_s" in resource,
        "client_absolute_origin": "client_started_s" in item or "experiment_start_s" in item,
    }


def inspect_item(item):
    exp = item["experiment"]
    if item["integrity_errors"]:
        raise ValueError("Historical artifact has integrity errors")
    requests = item["requests"]
    events = output_events(requests)
    groups = set(r["group"] for r in requests)
    expected = set(GROUPS) if exp["composition"] == "mixed" else {exp["composition"].removesuffix("-only")}
    if groups != expected:
        raise ValueError("Unexpected tenant set")
    resource = item["scheduler_resource_trace"]
    batch = item["scheduler_batch_trace"]
    if (batch.get("dropped_events", 0) or resource.get("dropped_events", 0)
            or batch.get("dropped_admission_blocks", 0)
            or resource["system"]["current_active_requests"] != 0):
        raise ValueError("Dropped trace or unfreed requests")
    if batch["event_count"] != len(batch["events"]):
        raise ValueError("Batch count mismatch")
    configs = []
    for endpoint in ("server_initial", "server_final"):
        states = item[endpoint]["internal_states"]
        if not states:
            raise ValueError("Missing resolved state")
        for state in states:
            for key, wanted in {"max_mamba_cache_size": 480, "max_total_tokens": 196608,
                                "pp_max_micro_batch_size": exp["admission_cap"],
                                "chunked_prefill_size": 4096}.items():
                if state[key] != wanted:
                    raise ValueError(f"Unexpected historical configuration: {key}")
            configs.append({key: state.get(key) for key in (
                "schedule_policy", "max_mamba_cache_size", "max_total_tokens",
                "pp_max_micro_batch_size", "disable_radix_cache", "disable_cuda_graph",
                "disable_prefill_cuda_graph", "enable_mixed_chunk")})
    if any(c != configs[0] for c in configs):
        raise ValueError("Resolved configuration drift within a run")
    result = {"experiment": exp, "request_count": len(requests), "features": features(item),
              "config": configs[0], "payload_fingerprint": item["payload_fingerprint"],
              "tenants": {}, "batch_events": len(batch["events"])}
    for group in sorted(groups):
        own = [r for r in requests if r["group"] == group]
        usage = resource["tenants"][group]
        inputs = sum(r["prompt_tokens"] for r in own)
        outputs = sum(r["completion_tokens"] for r in own)
        charge = inputs + 2 * outputs
        total_usage = number(usage["active_request_seconds"])
        if inputs <= 0 or total_usage <= 0:
            raise ValueError("Invalid workload or active time")
        result["tenants"][group] = {
            "requests": len(own), "input_tokens": inputs, "output_tokens": outputs,
            "completed_token_charge": charge,
            "p95_jct_s": number(item["summary"]["groups"][group]["p95_jct_s"]),
            "mean_jct_s": statistics.fmean(r["finished_s"] - r["submitted_s"] for r in own),
            "mean_ttft_s": statistics.fmean(r["first_token_s"] - r["submitted_s"] for r in own),
            "active_request_seconds": total_usage,
            "state_slot_seconds": number(usage["state_slot_seconds"]),
            "kv_token_seconds": number(usage["kv_token_seconds"]),
            "state_slots_per_active_request_time": usage["state_slot_seconds"] / total_usage,
            "state_slot_seconds_per_charge": usage["state_slot_seconds"] / charge,
            "kv_token_seconds_per_charge": usage["kv_token_seconds"] / charge,
            "output_milestones_s": milestones(events, group, outputs),
            "last_request_finished_s": max(r["finished_s"] for r in own),
        }
    if exp["composition"] == "mixed":
        vtc = item["scheduler_vtc_trace"]
        expected_weights = dict(request=0, input=1, input_quadratic=0, output=2,
                                input_output=0, output_quadratic=0)
        if vtc["weights"] != expected_weights or configs[0]["schedule_policy"] != "tenant-vtc":
            raise ValueError("Mixed policy or weights mismatch")
        for group in GROUPS:
            own = result["tenants"][group]
            if (own["input_tokens"] != 131072 or own["output_tokens"] != 4096
                    or own["requests"] != (8 if group == "packed" else 8 * exp["split_factor"])):
                raise ValueError("Unexpected confirmation workload")
        result["vtc_summary"] = {key: vtc[key] for key in (
            "observation_count", "current_counter_gap", "max_counter_gap", "counters", "queue_waits")}
    return result, events


def read_verified(path, expected_hash):
    raw = path.read_bytes()
    if digest(raw) != expected_hash:
        raise ValueError(f"Historical SHA256 mismatch: {path}")
    return json.loads(raw)


def audit(index_path):
    raw_index = index_path.read_bytes()
    index = json.loads(raw_index)
    selected = {path: sha for path, sha in index["source_hashes"].items()
                if Path(path).parent.name in ("confirm_f16", "confirm_f32")}
    if len(selected) != 60:
        raise ValueError(f"Expected 60 confirmation artifacts, found {len(selected)}")
    rows, streams, items, payloads = {}, {}, {}, defaultdict(set)
    for name, sha in sorted(selected.items()):
        item = read_verified(Path(name), sha)
        exp = item["experiment"]
        if exp["stage"] != "confirm" or exp["batch_mode"] != "separate-4096":
            raise ValueError("Unexpected confirmation stage or mode")
        key = (exp["split_factor"], exp["repetition"], exp["admission_cap"], exp["composition"])
        if key in rows:
            raise ValueError("Duplicate confirmation cell")
        row, events = inspect_item(item)
        row.update(source=name, sha256=sha)
        rows[key], streams[key], items[key] = row, events, item
        payloads[(key[0], key[1], key[3])].add(row["payload_fingerprint"])
    expected = set(itertools.product(FACTORS, REPS, CAPS, COMPOSITIONS))
    if set(rows) != expected or any(len(v) != 1 for v in payloads.values()):
        raise ValueError("Incomplete matrix or cap payload mismatch")
    for (factor, rep, cap, composition), row in rows.items():
        if composition != "mixed":
            continue
        for group in GROUPS:
            isolated = rows[(factor, rep, cap, group + "-only")]["tenants"][group]
            row["tenants"][group]["matched_p95_slowdown"] = row["tenants"][group]["p95_jct_s"] / isolated["p95_jct_s"]
    pairs = []
    for factor, rep in itertools.product(FACTORS, REPS):
        keys = [(factor, rep, cap, "mixed") for cap in CAPS]
        window = common_client_window([items[key] for key in keys])
        entry = {"factor": factor, "rep": rep, "client_window": window, "caps": {}}
        for key in keys:
            row = rows[key]
            totals = {g: row["tenants"][g]["output_tokens"] for g in GROUPS}
            entry["caps"][str(key[2])] = (output_window(streams[key], *window, totals)
                                             if window else None)
        pairs.append(entry)
    mixed = [row for key, row in sorted(rows.items()) if key[3] == "mixed"]
    aggregates = []
    for factor, cap in itertools.product(FACTORS, CAPS):
        subset = [r for r in mixed if r["experiment"]["split_factor"] == factor
                  and r["experiment"]["admission_cap"] == cap]
        per_tenant = {}
        for group in GROUPS:
            per_tenant[group] = {metric: statistics.fmean(r["tenants"][group][metric] for r in subset)
                                 for metric in ("p95_jct_s", "mean_ttft_s", "matched_p95_slowdown",
                                                "state_slot_seconds", "kv_token_seconds",
                                                "state_slots_per_active_request_time")}
            per_tenant[group]["output_50_s"] = statistics.fmean(r["tenants"][group]["output_milestones_s"]["50"] for r in subset)
        aggregate = {"factor": factor, "cap": cap, "repetitions": len(subset), "tenants": per_tenant}
        windows = [p["caps"][str(cap)] for p in pairs if p["factor"] == factor and p["client_window"]]
        aggregate["mean_client_window_s"] = statistics.fmean(w["duration_s"] for w in windows) if windows else None
        aggregate["mean_max_output_fraction_gap"] = statistics.fmean(w["max_abs_incremental_output_fraction_gap"] for w in windows) if windows else None
        aggregate["mean_state_time_split_over_packed"] = statistics.fmean(r["tenants"]["split"]["state_slot_seconds"] / r["tenants"]["packed"]["state_slot_seconds"] for r in subset)
        aggregate["mean_kv_time_split_over_packed"] = statistics.fmean(r["tenants"]["split"]["kv_token_seconds"] / r["tenants"]["packed"]["kv_token_seconds"] for r in subset)
        aggregates.append(aggregate)
    return {
        "complete": True, "new_gpu_runs": 0,
        "index_source": str(index_path), "index_sha256": digest(raw_index),
        "audit_script_sha256": digest(Path(__file__).read_bytes()),
        "counts": {"verified_artifacts": len(rows), "requests": sum(r["request_count"] for r in rows.values()),
                   "mixed_runs": len(mixed), "isolation_runs": len(rows) - len(mixed)},
        "source_hashes": selected,
        "feature_coverage_mixed_runs": {key: sum(r["features"][key] for r in mixed) for key in mixed[0]["features"]},
        "mixed_rows": mixed, "paired_client_windows": pairs, "aggregates": aggregates,
        "scope": "Post-hoc historical audit; no new fairness gate, no VTC-bound test, no state-only causality",
        "limitations": [
            "Client output milestones/gaps exclude input and do not reconstruct VTC counters.",
            "Client outstanding is not scheduler waiting backlog; common windows are outcome-dependent descriptive drain windows.",
            "Resource integrals cover the full run, not the client window; physical state slots can include extra buffers.",
            "Batch tenant counts are participation counts, not per-tenant work or GPU time.",
            "Paired payload verification applies within each factor/rep, not across factors or server sessions.",
            "F16/f32 use different full payload fingerprints and sessions; they are not a fixed-victim causal comparison.",
        ],
    }


def report(data):
    lines = ["# Historical M2 Tenant-Service Audit", "", f"Counts: {data['counts']}; new GPU runs: 0.",
             "", "All selected raw SHA256 hashes verified. Historical gates unchanged.",
             "", "## Trace Coverage (20 Mixed Runs)", "", "| Field | Runs |", "|---|---:|"]
    lines += [f"| {key} | {count} |" for key, count in data["feature_coverage_mixed_runs"].items()]
    lines += ["", "## Full-Run Tenant Results", "", "Five-repeat means; post-hoc descriptive audit, no new CI or PASS.", "",
              "| Shape | Cap | Tenant | P95 JCT s | Slowdown | Output 50% s | State slot-s | KV token-s |", "|---|---:|---|---:|---:|---:|---:|---:|"]
    for row in data["aggregates"]:
        for group, m in row["tenants"].items():
            lines.append(f"| f{row['factor']} | {row['cap']} | {group} | {m['p95_jct_s']:.3f} | {m['matched_p95_slowdown']:.3f} | {m['output_50_s']:.3f} | {m['state_slot_seconds']:.3f} | {m['kv_token_seconds']:.0f} |")
    lines += ["", "## Common Client Drain Windows", "", "Output-only incremental gaps, NOT token-VTC fairness. Same start/end for each paired cap run.", "",
              "| Shape | Cap | Mean window s | Mean max output fraction gap | State-time split/packed | KV-time split/packed |", "|---|---:|---:|---:|---:|---:|"]
    for row in data["aggregates"]:
        duration, gap = row["mean_client_window_s"], row["mean_max_output_fraction_gap"]
        duration_text = "NA" if duration is None else f"{duration:.3f}"
        gap_text = "NA" if gap is None else f"{100 * gap:.2f}%"
        lines.append(f"| f{row['factor']} | {row['cap']} | {duration_text} | {gap_text} | {row['mean_state_time_split_over_packed']:.3f} | {row['mean_kv_time_split_over_packed']:.3f} |")
    lines += ["", "## Limits", ""] + [f"- {note}" for note in data["limitations"]]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("index", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    data = audit(args.index)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "audit.json").open("x") as handle:
        json.dump(data, handle, indent=2, allow_nan=False)
        handle.write("\n")
    with (args.output / "report.md").open("x") as handle:
        handle.write(report(data))
    print(report(data))


if __name__ == "__main__":
    main()
