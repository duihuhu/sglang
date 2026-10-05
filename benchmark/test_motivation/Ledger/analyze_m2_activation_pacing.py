#!/usr/bin/env python3
"""Descriptive paired analysis, never a confirmatory gate, for activation pacing."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

from analyze_m2_hybrid_factors import lifecycle
from run_m2_activation_pacing import LIMITS, ORDERS, integrity_errors, validate_info


def nearest_rank(values, q=.95):
    if not values or not 0 < q <= 1:
        raise ValueError("Nonempty sample and quantile in (0,1] required")
    return sorted(values)[max(0, math.ceil(q * len(values)) - 1)]


def metrics(item):
    requests = item["requests"]
    spans = lifecycle(item)
    resource, batch = item["scheduler_resource_trace"], item["scheduler_batch_trace"]
    value = {
        "all_mean_jct": statistics.fmean(r["latency_s"] for r in requests),
        "makespan": max(r["finished_s"] for r in requests) - min(r["submitted_s"] for r in requests),
        "peak_state": resource["system"]["peak_state_slots"],
        "state_slot_seconds": resource["system"]["state_slot_seconds"],
        "kv_token_seconds": resource["system"]["kv_token_seconds"],
        "vtc_max_counter_gap": item["scheduler_vtc_trace"]["max_counter_gap"],
        "max_arrival_lateness_s": max(r["submitted_s"] - r["scheduled_arrival_s"] for r in requests),
        "first_release_s": min(b for a, b in spans.values()),
    }
    for group in ("short", "long"):
        own = [r for r in requests if r["group"] == group]
        queues = [spans[r["rid"]][0] - r["submitted_s"] for r in own]
        if min(queues) < -.01:
            raise ValueError("Activation precedes submission")
        value.update({
            f"{group}_mean_jct": statistics.fmean(r["latency_s"] for r in own),
            f"{group}_p95_jct": nearest_rank([r["latency_s"] for r in own]),
            f"{group}_max_jct": max(r["latency_s"] for r in own),
            f"{group}_p95_ttft": nearest_rank([r["ttft_s"] for r in own]),
            f"{group}_mean_tpot": statistics.fmean((r["latency_s"] - r["ttft_s"]) /
                                                    (r["completion_tokens"] - 1) for r in own),
            f"{group}_mean_queue": statistics.fmean(queues),
            f"{group}_mean_active": statistics.fmean(spans[r["rid"]][1] - spans[r["rid"]][0] for r in own),
            f"{group}_completed_charge": sum(r["prompt_tokens"] + 2*r["completion_tokens"] for r in own),
        })
    reasons = Counter(b["reason"] for b in batch["admission_blocks"])
    for reason in ("activation_pacing", "request_limit", "prefill_no_token", "prefill_other"):
        value[f"blocks_{reason}"] = reasons[reason]
    paced = [b for b in batch["admission_blocks"] if b["reason"] == "activation_pacing"]
    value["distinct_paced_candidates"] = len({b["candidate_rid"] for b in paced})
    value["pacing_with_running"] = sum(b["running_requests"] > 0 for b in paced)
    value["pacing_with_continuation"] = sum(b["selected_reusing_request_slots"] > 0 for b in paced)
    value["mixed_batches"] = sum(b["mode"] == "mixed" and b["decoding_requests"] > 0 for b in batch["events"])
    value["prefill_batches"] = sum(b["mode"] in ("mixed", "extend") for b in batch["events"])
    return value


def paired_summary(base, paced):
    if set(base) != set(paced) or not base:
        raise ValueError("Unpaired run identifiers")
    result = {}
    for endpoint in base[next(iter(base))]:
        a = [base[i][endpoint] for i in sorted(base)]
        b = [paced[i][endpoint] for i in sorted(base)]
        result[endpoint] = {
            "baseline_mean": statistics.fmean(a), "paced_mean": statistics.fmean(b),
            "mean_difference": statistics.fmean(y - x for x, y in zip(a, b)),
            "mean_reduction_fraction": statistics.fmean(1 - y/x for x, y in zip(a, b)) if min(a) > 0 else None,
        }
    return result


def analyze(directories):
    rows, hashes, manifests, pairs = [], {}, {}, defaultdict(dict)
    seen, payloads, budget_pairs = set(), defaultdict(set), defaultdict(set)
    complete = True
    for directory in directories:
        manifest = json.loads((directory / "manifest.json").read_text())
        manifests[str(directory)] = manifest
        args = manifest["args"]
        if args["stage"] != "screen":
            raise ValueError("Only screen runs may enter this exploratory comparison")
        done_path = directory / "completed.json"
        if not done_path.exists():
            complete = False
        else:
            done = json.loads(done_path.read_text())
            if not done.get("completed") or done["source_hashes"] != manifest["source_hashes"]:
                raise ValueError("Source hashes changed or completion invalid")
        expected = {(mode, limit, order, args["rep_start"] + session * args["reps_per_session"] + index)
                    for mode in args["modes"] for limit in LIMITS for order in ORDERS
                    for session in range(args["sessions"]) for index in range(args["reps_per_session"])}
        actual, measured_requests = set(), 0
        for path in sorted(directory.glob("m2p-*.json")):
            item = json.loads(path.read_text())
            exp = item["experiment"]
            if exp["stage"] != "screen":
                continue
            if exp["run_id"] in seen:
                raise ValueError("Duplicate experiment")
            seen.add(exp["run_id"])
            expected_count = args["waves"] * (args["short_per_wave"] + args["long_per_wave"])
            if item.get("physical_gpu_ids", [6, 7]) != args.get("gpus", [6, 7]):
                raise ValueError("Physical GPUs differ from execution manifest")
            if item["integrity_errors"] or integrity_errors(item, expected_count, args["max_lateness"]):
                raise ValueError(f"Invalid replay: {path}")
            for field in ("server_initial", "server_final"):
                capacity = validate_info(item[field], exp["mode"], exp["regime"], exp["activation_limit"])
                if list(capacity) != item["resolved_capacity"]:
                    raise ValueError("Capacity readback mismatch")
            cell = (exp["mode"], exp["activation_limit"], exp["order"], exp["rep"])
            if cell not in expected or cell in actual or exp["regime"] != args["regime"]:
                raise ValueError("Unexpected or repeated matrix cell")
            actual.add(cell)
            if any(exp[k] != args[k] for k in ("waves", "short_per_wave", "long_per_wave", "output_tokens")):
                raise ValueError("Workload metadata mismatch")
            counts = Counter(r["group"] for r in item["requests"])
            if counts != {"short": args["waves"] * args["short_per_wave"],
                          "long": args["waves"] * args["long_per_wave"]}:
                raise ValueError("Unexpected cohort sizes")
            signature = (args["waves"], args["interval"], args["short_per_wave"], args["long_per_wave"], args["output_tokens"])
            payloads[(signature, exp["order"], exp["rep"])].add(item["payload_fingerprint"])
            key = (signature, exp["regime"], exp["mode"], exp["order"])
            budget_pairs[(key, exp["rep"])].add(tuple(item["resolved_capacity"]))
            value = metrics(item)
            pairs[(key, exp["activation_limit"])][exp["rep"]] = value
            rows.append({"experiment": exp, "metrics": value})
            measured_requests += len(item["requests"])
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        complete &= actual == expected and measured_requests == manifest["planned_measured_requests"]
    if not rows or any(len(v) != 1 for v in [*payloads.values(), *budget_pairs.values()]):
        raise ValueError("No valid data or mismatched payloads/capacities")
    effects = []
    for (key, limit), candidate in sorted(pairs.items()):
        if not limit:
            continue
        baseline = pairs.get((key, 0), {})
        if set(baseline) != set(candidate):
            complete = False
            continue
        signature, regime, mode, order = key
        effects.append({"workload": signature, "regime": regime, "mode": mode, "order": order,
                        "activation_limit": limit, "repetitions": sorted(candidate),
                        "endpoints": paired_summary(baseline, candidate)})
    return {"artifact_count": len(rows), "request_count": sum(
                r["experiment"]["waves"] * (r["experiment"]["short_per_wave"] + r["experiment"]["long_per_wave"]) for r in rows),
            "complete_planned_matrix": bool(complete), "rows": rows, "paired_effects": effects,
            "artifact_hashes": hashes, "manifests": manifests,
            "scope": "Exploratory paired means; no confirmatory PASS, steady-state SLO or fairness claim."}


def report(data):
    lines = ["# M2 activation pacing: exploratory results", "",
             f"Runs: {data['artifact_count']}; requests: {data['request_count']}; complete={data['complete_planned_matrix']}", "",
             "All gates are reuse-aware. Positive reduction means a smaller measured value, not automatically a service improvement.", "",
             "| Regime / batching / order | New limit | Paired runs | All mean JCT reduction | Short mean reduction | Long mean reduction | Makespan reduction | Pacing events/run |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["paired_effects"]:
        values = []
        for name in ("all_mean_jct", "short_mean_jct", "long_mean_jct", "makespan"):
            fraction = row["endpoints"][name]["mean_reduction_fraction"]
            values.append("n/a" if fraction is None else f"{100*fraction:+.2f}%")
        paced = row["endpoints"]["blocks_activation_pacing"]["paced_mean"]
        lines.append(f"| {row['regime']} / {row['mode']} / {row['order']} | {row['activation_limit']} | {len(row['repetitions'])} | "
                     + " | ".join(values) + f" | {paced:.1f} |")
    lines += ["", "## Limits", "",
              "Screening averages are not independent confirmation; report every cell, including regressions and inactive interventions.",
              "Pacing events precede the remaining candidate budgets: they do not prove feasible admissions were rejected.",
              "Nearest-rank p95, per-group queues/lifetimes, TPOT, state/KV integrals and counter gaps are in analysis.json.",
              "VTC counter gaps include lifts and unconditioned backlog; they are not violations of its fairness guarantee.",
              "Finite-wave completion is not steady-state goodput. No SLO thresholds were inferred from intervention results.",
              "Native prefill-size and free-slot-delay controls, natural pressure, held-out traces and full literature baselines remain separate stages."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.directories)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.output / "report.md").write_text(report(data))
    print(report(data))


if __name__ == "__main__":
    main()
