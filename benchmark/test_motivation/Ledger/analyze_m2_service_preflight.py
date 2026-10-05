"""Summarize all fixed-target preflight cells without promoting them to evidence of causality."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from audit_m2_service_preflight import inspect


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process(item):
    exp = item["experiment"]
    trace = item["service_trace"]
    groups = item["summary"]["groups"]
    result = {"experiment": exp, "fingerprints": item["fingerprints"], "groups": groups,
              "integrity_errors": item["integrity_errors"],
              "makespan_s": max(r["finished_s"] for r in item["requests"]),
              "resource_system": item["scheduler_resource_trace"]["system"],
              "resource_tenants": item["scheduler_resource_trace"]["tenants"],
              "output50_s": {}, "max_submit_lateness_s": max(
                  r["submitted_s"] - r["scheduled_arrival_s"] for r in item["requests"])}
    resource = item["scheduler_resource_trace"]
    resource_offset = resource["event_clock_origin_s"] - item["experiment_clock_origin_s"]
    prefix = f"hybrid-tenant:{exp['run_id']}:"
    result["resource_curves"] = [{"at_s": resource_offset + e["observed_s"],
                                   "tenants": {k.removeprefix(prefix): v for k, v in e["tenants"].items()}}
                                  for e in resource["events"]]
    for group in groups:
        requests = [r for r in item["requests"] if r["group"] == group]
        changes = []
        for req in requests:
            previous = 0
            for at, current in req["token_events"]:
                changes.append((at, current - previous))
                previous = current
        total, goal = 0, sum(r["output_len"] for r in requests) / 2
        for at, delta in sorted(changes):
            total += delta
            if total >= goal:
                result["output50_s"][group] = at
                break
    if not trace.get("enabled"):
        return result
    result["audit"] = inspect(item)
    prefix = f"hybrid-tenant:{exp['run_id']}:"
    counters, lifted, inputs, outputs, outstanding = {}, Counter(), Counter(), Counter(), Counter()
    waiting, curves, launches, finished = {}, [], {}, set()
    service_origin = trace["event_clock_origin_s"] - item["experiment_clock_origin_s"]
    last_time = trace["events"][0]["observed_s"]
    both_outstanding = waiting_both = 0
    reasons, max_lead = Counter(), Counter()
    for event in trace["events"]:
        now = event["observed_s"]
        if all(outstanding[prefix + r] > 0 for r in ("A", "B")):
            both_outstanding += now - last_time
            if all(waiting.get(prefix + r, 0) > 0 for r in ("A", "B")):
                waiting_both += now - last_time
        last_time = now
        if "waiting" in event:
            waiting = event["waiting"]
        kind = event["kind"]
        if kind == "queue_inserted" and event["inserted"]:
            outstanding.update(row["tenant"] for row in event["members"])
        if "counters_after" in event:
            counters = event["counters_after"]
        if kind == "queue_counter":
            lifted[event["members"][0]["tenant"]] += event["lift_delta"]
        if kind == "output_charge":
            for row in event["members"]:
                outputs[row["tenant"]] += row["current_output"] - row["previous_output"]
        if kind == "batch_launch":
            launches[event["batch_id"]] = event["members"]
        if kind == "batch_result":
            for row in launches[event["batch_id"]]:
                if row["phase"] == "prefill":
                    inputs[row["tenant"]] += row["work_tokens"]
            for row in event["members"]:
                if row["rid"] in event["finished_rids"] and row["rid"] not in finished:
                    outstanding[row["tenant"]] -= 1
                    finished.add(row["rid"])
        if kind == "scan_stop":
            reasons[event["reason"]] += 1
        if kind in ("admission", "batch_result", "queue_counter"):
            record = {"at_s": service_origin + now, "kind": kind, "tenants": {}}
            for key in counters:
                role = key.removeprefix(prefix)
                charge = counters[key] - lifted[key]
                executed = inputs[key] + 2 * outputs[key]
                max_lead[role] = max(max_lead[role], charge - executed)
                record["tenants"][role] = {"counter": counters[key], "lift": lifted[key],
                    "charge_without_lift": charge, "completed_prefill_tokens": inputs[key],
                    "server_output_tokens": outputs[key], "executed_weighted_tokens": executed,
                    "waiting": waiting.get(key, 0), "outstanding": outstanding[key]}
            curves.append(record)
    result.update(counter_execution_curves=curves, max_charge_ahead_of_completed_work=dict(max_lead),
                  both_server_outstanding_seconds=both_outstanding,
                  both_waiting_seconds=waiting_both, scan_stop_reasons=dict(reasons))
    return result


def analyze(directory, output):
    manifest = directory / "manifest.json"
    completed = directory / "completed.json"
    if not completed.exists():
        raise ValueError("Incomplete preflight; do not silently summarize a partial matrix")
    completion = json.loads(completed.read_text())
    sources, rows = [], []
    for path in sorted(directory.glob("*/m2sf-*-preflight.json")):
        row = process(json.loads(path.read_text()))
        sources.append({"path": str(path), "sha256": sha(path)})
        rows.append(row)
    expected = {(o, a, b) for o in ("off", "on") for a in ("packed", "split") for b in ("alone", "packed", "split")}
    actual = [(r["experiment"]["observer"], r["experiment"]["target"], r["experiment"]["neighbor"]) for r in rows]
    if len(rows) != 12 or set(actual) != expected or any(r["integrity_errors"] or r.get("audit", {}).get("errors") for r in rows):
        raise ValueError("Incomplete matrix or failed observation integrity")
    result = {"scope": "one functional run per cell; descriptive only, not independent-repeat confirmation",
              "manifest": {"path": str(manifest), "sha256": sha(manifest)},
              "completed": {"path": str(completed), "sha256": sha(completed), "value": completion},
              "sources": sources, "rows": rows, "pairs": []}
    for shape in ("packed", "split"):
        alone = next(r for r in rows if r["experiment"]["observer"] == "on"
                     and r["experiment"]["target"] == shape and r["experiment"]["neighbor"] == "alone")
        for neighbor in ("alone", "packed", "split"):
            matching = {r["experiment"]["observer"]: r for r in rows
                        if r["experiment"]["target"] == shape and r["experiment"]["neighbor"] == neighbor}
            on, off = matching["on"], matching["off"]
            result["pairs"].append({"target": shape, "neighbor": neighbor,
                "on_target_p95_jct_s": on["groups"]["A"]["p95_jct_s"],
                "on_target_p95_slowdown": on["groups"]["A"]["p95_jct_s"] / alone["groups"]["A"]["p95_jct_s"],
                "on_target_output50_s": on["output50_s"]["A"],
                "on_off_makespan_ratio": on["makespan_s"] / off["makespan_s"],
                "both_waiting_seconds": on["both_waiting_seconds"],
                "both_server_outstanding_seconds": on["both_server_outstanding_seconds"],
                "max_target_charge_ahead_of_completed_work": on["max_charge_ahead_of_completed_work"].get("A"),
                "scan_stop_reasons": on["scan_stop_reasons"]})
    output.mkdir(parents=True, exist_ok=False)
    with (output / "summary.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    print(json.dumps({"pairs": result["pairs"], "completion": completion,
                      "event_count": sum(sum(r.get("audit", {}).get("event_kinds", {}).values()) for r in rows)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    analyze(args.directory, args.output)
