#!/usr/bin/env python3
"""Separate observed progress gaps from causal or GPU-idleness claims."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_m2_hybrid_factors import lifecycle


def union_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if end <= start:
            continue
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def overlap_duration(start, end, windows):
    return sum(max(0., min(end, b) - max(start, a)) for a, b in windows)


def intersect_intervals(first, second):
    first, second = union_intervals(first), union_intervals(second)
    result = []
    i = j = 0
    while i < len(first) and j < len(second):
        left, right = max(first[i][0], second[j][0]), min(first[i][1], second[j][1])
        if right > left:
            result.append((left, right))
        if first[i][1] <= second[j][1]:
            i += 1
        else:
            j += 1
    return result


def metrics(item):
    spans = lifecycle(item)
    requests = {r["rid"]: r for r in item["requests"]}
    trace = item["scheduler_batch_trace"]
    offset = trace["event_clock_origin_s"] - item["experiment_clock_origin_s"]
    events = trace["events"]
    launches = [offset + e["launch_s"] for e in events]
    if any(b < a for a, b in zip(launches, launches[1:])):
        raise ValueError("Nonmonotonic launch times")
    pending_windows = union_intervals((requests[rid]["submitted_s"], start) for rid, (start, _) in spans.items())
    resource = item["scheduler_resource_trace"]
    resource_offset = resource["event_clock_origin_s"] - item["experiment_clock_origin_s"]
    full_windows = union_intervals(
        (resource_offset + left["observed_s"], resource_offset + right["observed_s"])
        for left, right in zip(resource["events"], resource["events"][1:])
        if left["state_slots"] == item["resolved_capacity"][0])
    full_pending_windows = intersect_intervals(full_windows, pending_windows)
    per_req = defaultdict(lambda: {"prefill_chunks": 0, "decode_steps": 0, "first_work": None,
                                  "first_work_tokens": None, "prefill_recorded_end": None})
    for event, launch in zip(events, launches):
        for member in event["members"]:
            row = per_req[member["rid"]]
            if row["first_work"] is None:
                row["first_work"] = launch
                row["first_work_tokens"] = member["work_tokens"]
            if member["phase"] == "prefill":
                row["prefill_chunks"] += 1
                row["prefill_recorded_end"] = offset + event["recorded_s"]
            else:
                row["decode_steps"] += 1
    if set(per_req) != set(requests) or any(v["prefill_recorded_end"] is None for v in per_req.values()):
        raise ValueError("Incomplete per-request progress")
    unselected = defaultdict(float)
    ready_unselected = defaultdict(float)
    ready_unselected_pending = defaultdict(float)
    ready_unselected_full_pending = defaultdict(float)
    for index, event in enumerate(events[:-1]):
        left, right = launches[index:index + 2]
        selected = {m["rid"] for m in event["members"]}
        for rid, (start, end) in spans.items():
            if rid in selected:
                continue
            a, b = max(start, left), min(end, right)
            if b <= a:
                continue
            unselected[rid] += b - a
            a = max(a, per_req[rid]["prefill_recorded_end"])
            if b > a:
                ready_unselected[rid] += b - a
                ready_unselected_pending[rid] += overlap_duration(a, b, pending_windows)
                ready_unselected_full_pending[rid] += overlap_duration(a, b, full_pending_windows)
    result = {
        "requests": len(requests),
        "all_mean_jct_s": statistics.fmean(r["latency_s"] for r in requests.values()),
        "mean_ttft_s": statistics.fmean(r["ttft_s"] for r in requests.values()),
        "makespan_s": max(r["finished_s"] for r in requests.values()) - min(r["submitted_s"] for r in requests.values()),
        "mean_queue_s": statistics.fmean(spans[rid][0] - r["submitted_s"] for rid, r in requests.items()),
        "peak_active": item["scheduler_resource_trace"]["system"]["peak_active_requests"],
        "peak_kv_tokens": item["scheduler_resource_trace"]["system"]["peak_kv_tokens"],
        "state_slot_seconds": item["scheduler_resource_trace"]["system"]["state_slot_seconds"],
        "kv_token_seconds": item["scheduler_resource_trace"]["system"]["kv_token_seconds"],
        "unselected_slot_seconds_proxy": sum(unselected.values()),
        "decode_ready_unselected_slot_seconds_proxy": sum(ready_unselected.values()),
        "decode_ready_unselected_with_pending_slot_seconds_proxy": sum(ready_unselected_pending.values()),
        "decode_ready_unselected_state_full_with_pending_slot_seconds_proxy": sum(ready_unselected_full_pending.values()),
        "state_full_with_pending_seconds": sum(b - a for a, b in full_pending_windows),
        "max_decode_ready_unselected_s_proxy": max(ready_unselected.values(), default=0.),
        "mean_first_prefill_fraction": statistics.fmean(per_req[rid]["first_work_tokens"] / r["prompt_tokens"]
                                                        for rid, r in requests.items()),
        "mean_prefill_chunks": statistics.fmean(p["prefill_chunks"] for p in per_req.values()),
        "decode_work_minus_delivered_histogram": dict(Counter(
            per_req[rid]["decode_steps"] - (r["completion_tokens"] - 1) for rid, r in requests.items())),
        "blocks_by_reason": dict(Counter(b["reason"] for b in trace["admission_blocks"])),
        "blocks_state_empty_kv_free": sum(b.get("state_slots_free") == 0 and (b.get("kv_tokens_free") or 0) > 0
                                           for b in trace["admission_blocks"]),
        "max_arrival_lateness_ms": 1000 * max(r["submitted_s"] - r["scheduled_arrival_s"] for r in requests.values()),
        "groups": {},
    }
    for group in sorted({r["group"] for r in requests.values()}):
        own = [r for r in requests.values() if r["group"] == group]
        result["groups"][group] = {
            "mean_jct_s": statistics.fmean(r["latency_s"] for r in own),
            "mean_queue_s": statistics.fmean(spans[r["rid"]][0] - r["submitted_s"] for r in own),
            "mean_tpot_s": statistics.fmean((r["latency_s"] - r["ttft_s"]) / (r["completion_tokens"] - 1) for r in own),
        }
    details = [{"rid": rid, "group": requests[rid]["group"], "activation_s": spans[rid][0],
                "release_s": spans[rid][1], **per_req[rid],
                "unselected_s_proxy": unselected[rid],
                "decode_ready_unselected_s_proxy": ready_unselected[rid],
                "decode_ready_unselected_with_pending_s_proxy": ready_unselected_pending[rid],
                "decode_ready_unselected_state_full_with_pending_s_proxy": ready_unselected_full_pending[rid]}
               for rid in sorted(requests)]
    return result, details


def analyze(directory):
    from run_m2_progress_probe import SHAPES, integrity_errors, progress_errors, validate_natural_info
    manifest = json.loads((directory / "manifest.json").read_text())
    done_path = directory / "completed.json"
    complete = done_path.exists()
    if complete:
        done = json.loads(done_path.read_text())
        if not done.get("completed") or done["source_hashes"] != manifest["source_hashes"]:
            raise ValueError("Completion/hash mismatch")
    rows, hashes, seen = [], {}, set()
    payloads = defaultdict(set)
    for path in sorted(directory.glob("m2progress-*-observe.json")):
        item = json.loads(path.read_text())
        exp = item["experiment"]
        key = (exp["mode"], exp["shape"], exp["rep"])
        count = {"short": 256, "medium": 64, "long": 16, "mixed": 136,
                 "state-pressure": manifest["pressure_count"]}[exp["shape"]]
        if key in seen or item["integrity_errors"] or integrity_errors(item, count, .05) or progress_errors(item):
            raise ValueError(f"Invalid or repeated run: {path}")
        seen.add(key)
        for info in ("server_initial", "server_final"):
            if list(validate_natural_info(item[info], exp["mode"])) != manifest["capacity_reference"]["resolved_capacity"]:
                raise ValueError("Capacity mismatch")
        payloads[(exp["shape"], exp["rep"])].add(item["payload_fingerprint"])
        value, detail = metrics(item)
        resource = item["scheduler_resource_trace"]
        offset = resource["event_clock_origin_s"] - item["experiment_clock_origin_s"]
        timeline = [{"time_s": offset + e["observed_s"],
                     **{k: e[k] for k in ("active_requests", "state_slots", "kv_tokens")}}
                    for e in resource["events"]]
        rows.append({"experiment": exp, "metrics": value, "request_progress": detail,
                     "resource_timeline": timeline})
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    expected = {(mode, shape, rep) for mode in ("mixed-4096", "separate-4096") for shape in SHAPES for rep in (1, 2)}
    complete &= seen == expected and sum(r["metrics"]["requests"] for r in rows) == manifest["planned_measured_requests"]
    if any(len(v) != 1 for v in payloads.values()):
        raise ValueError("Unpaired payloads")
    return {"complete": complete, "manifest": manifest, "rows": rows, "artifact_hashes": hashes,
            "scope": "CPU-observed participation proxies; not kernel idle time, avoidable-loss proof, or fairness guarantee."}


def report(data):
    lines = ["# M2 natural-capacity progress probe", "", f"Complete: {data['complete']}; runs: {len(data['rows'])}", "",
             "All slot gates corrected; activation limit off. Unselected time is a CPU-launch proxy, not wasted GPU time.", "",
             "| Mode | Shape | Rep | Mean JCT s | Peak active | State slot-s | Decode-ready unselected slot-s | With pending slot-s |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in data["rows"]:
        e, m = row["experiment"], row["metrics"]
        lines.append(f"| {e['mode']} | {e['shape']} | {e['rep']} | {m['all_mean_jct_s']:.3f} | {m['peak_active']} | "
                     f"{m['state_slot_seconds']:.3f} | {m['decode_ready_unselected_slot_seconds_proxy']:.3f} | "
                     f"{m['decode_ready_unselected_with_pending_slot_seconds_proxy']:.3f} |")
    lines += ["", "Different shapes have different output work; their absolute latencies are not causal state-cost comparisons.",
              "Two restarts per configuration are exploratory. No confirmatory CI, winner selection, or SLO/fairness claim."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.directory)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.output / "report.md").write_text(report(data))
    print(report(data))


if __name__ == "__main__":
    main()
