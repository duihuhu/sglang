#!/usr/bin/env python3
"""Measure a slot-accounting intervention separately from batching policies."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_m2_batch_frontier import paired_effect, percentile
from analyze_m2_hybrid_factors import lifecycle, metrics
from m2_gpu_health import health_errors


CONFIGS = ("state8-prefill4096", "state8-mixed4096", "state8-mixed1024",
           "state8-unchunked", "wide-cap8-prefill4096")


def extra_metrics(item):
    value = metrics(item)
    spans = lifecycle(item)
    first_release = min(b for a, b in spans.values())
    for group in ("short", "long"):
        own = [r for r in item["requests"] if r["group"] == group]
        value[f"{group}_mean_jct"] = statistics.fmean(r["latency_s"] for r in own)
        value[f"{group}_max_jct"] = max(r["latency_s"] for r in own)
        value[f"{group}_p50"] = percentile([r["latency_s"] for r in own], .5)
        value[f"{group}_ttft_p95"] = percentile([r["ttft_s"] for r in own], .95)
        value[f"{group}_first_episode_admitted"] = sum(spans[r["rid"]][0] < first_release for r in own)
        value[f"{group}_tpot_mean"] = statistics.fmean(
            (r["latency_s"] - r["ttft_s"]) / (r["completion_tokens"] - 1) for r in own)
    blocks = item["scheduler_batch_trace"]["admission_blocks"]
    # E2 contains only fresh queued candidates; continuations are already selected.
    value["reuse_slot_gate_witnesses"] = sum(
        b["reason"] == "request_limit" and b["candidate_rid"] is not None
        and b["selected_reusing_request_slots"] > 0
        and b["selected_requests"] < b["runtime_cap"] - b["running_requests"]
        and b["selected_requests"] - b["selected_reusing_request_slots"] + 1 <= b["request_slots_free"]
        for b in blocks)
    batches = item["scheduler_batch_trace"]["events"]
    value["mixed_batches"] = sum(b["mode"] == "mixed" and b["decoding_requests"] > 0 for b in batches)
    value["prefill_batches"] = sum(b["mode"] in ("extend", "mixed") for b in batches)
    value["reused_request_gate_events"] = sum(b["selected_reusing_request_slots"] > 0 for b in blocks)
    return value


def analyze(directories):
    groups, fingerprints, contents = defaultdict(list), defaultdict(set), defaultdict(set)
    hashes, manifests, samples, hardware = {}, {}, {}, {}
    seen = set()
    total_requests = 0
    for directory in directories:
        manifests[str(directory)] = json.loads((directory / "manifest.json").read_text())
        for path in sorted(directory.glob("m2s-*.json")):
            item = json.loads(path.read_text())
            exp = item["experiment"]
            if exp["stage"] != "confirm":
                continue
            if exp["run_id"] in seen:
                raise ValueError("Duplicate formal run")
            seen.add(exp["run_id"])
            resource, batch = item["scheduler_resource_trace"], item["scheduler_batch_trace"]
            if item["integrity_errors"] or health_errors(item.get("gpu_health_samples", [])) or resource.get("dropped_events") or batch.get("dropped_events") or batch.get("dropped_admission_blocks"):
                raise ValueError("Invalid experiment")
            if any(not r["success"] or r["cached_tokens"] or r["num_retractions"]
                   or r["prompt_tokens"] != r["input_len"] or r["completion_tokens"] != r["output_len"]
                   for r in item["requests"]):
                raise ValueError("Invalid request")
            if (len(item["requests"]) != 20
                    or sum(r["group"] == "short" for r in item["requests"]) != 16
                    or sum(r["group"] == "long" for r in item["requests"]) != 4):
                raise ValueError("Unexpected cohort sizes")
            if resource["system"]["peak_active_requests"] > 8 or resource["system"]["current_active_requests"]:
                raise ValueError("Capacity/release integrity")
            fingerprints[(exp["variant"], exp["rep"])].add(item["payload_fingerprint"])
            contents[exp["rep"]].add(item["content_fingerprint"])
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            loaded = [gpu["sm_mhz"] for sample in item["gpu_health_samples"]
                      for gpu in sample["gpus"] if gpu["utilization"] >= 80]
            hardware[exp["run_id"]] = {
                "samples": len(item["gpu_health_samples"]), "loaded_gpu_samples": len(loaded),
                "loaded_sm_mhz_min": min(loaded) if loaded else None,
                "loaded_sm_mhz_max": max(loaded) if loaded else None,
            }
            total_requests += len(item["requests"])
            key = (exp["config"], exp["gate"], exp["variant"])
            value = extra_metrics(item)
            if not value["one_state_per_active"]:
                raise ValueError("State invariant")
            if exp["config"].startswith("state8-mixed") and value["mixed_batches"] == 0:
                raise ValueError("Mixed mode did not execute mixed batches")
            if "mixed" not in exp["config"] and value["mixed_batches"]:
                raise ValueError("Unexpected mixed batch")
            if value["short_completed_charge"] != 10240 or value["long_completed_charge"] != 33280:
                raise ValueError("Completed token service mismatch")
            groups[key].append({"rep": exp["rep"], **value})
            if exp["rep"] == 20:
                samples[exp["run_id"]] = {
                    "admission_blocks": batch["admission_blocks"],
                    "resource_events": resource["events"],
                    "request_latencies": [{"rid": r["rid"], "group": r["group"],
                                           "latency_s": r["latency_s"]} for r in item["requests"]],
                    "clock_offset": resource["event_clock_origin_s"] - item["experiment_clock_origin_s"],
                }
    if not groups or any(len(v) != 1 for v in [*fingerprints.values(), *contents.values()]):
        raise ValueError("No data or mismatched paired payloads")
    rows = []
    for (config, gate, order), runs in sorted(groups.items()):
        rows.append({"config": config, "gate": gate, "order": order,
                     "runs": sorted(runs, key=lambda r: r["rep"]),
                     "mean": {k: statistics.fmean(r[k] for r in runs) for k in runs[0] if k != "rep"}})
    lookup = {(r["config"], r["gate"], r["order"]): r for r in rows}
    effects = []
    endpoints = ("short_p95", "long_p95", "makespan", "short_mean_jct", "short_max_jct", "short_queue_p95", "short_tpot_mean")
    for config in CONFIGS:
        for order in ("short-first", "long-first"):
            old = lookup.get((config, "legacy", order))
            new = lookup.get((config, "reuse-aware", order))
            if old is None or new is None:
                continue
            a = {r["rep"]: r for r in old["runs"]}
            b = {r["rep"]: r for r in new["runs"]}
            if set(a) != set(b):
                continue
            reps = sorted(a)
            effects.append({"config": config, "order": order, "reps": reps,
                            "effects": {k: paired_effect([a[i][k] for i in reps], [b[i][k] for i in reps])
                                        for k in endpoints}})
    expected = {(c, g, o): set(range(20, 25)) for c in CONFIGS
                for g in ("legacy", "reuse-aware") for o in ("short-first", "long-first")}
    actual = {key: {r["rep"] for r in runs} for key, runs in groups.items()}
    reference = next((r for r in effects if r["config"] == "state8-prefill4096" and r["order"] == "short-first"), None)
    gate = None
    if reference is not None and reference["reps"] == list(range(20, 25)):
        e = reference["effects"]
        gate = {"passed": (e["short_p95"]["mean_improvement"] >= .05 and e["short_p95"]["ci95"][0] > 0
                           and e["long_p95"]["ci95"][0] >= -.05 and e["makespan"]["ci95"][0] >= -.05),
                "effects": e, "meaning": "Local finite-replay opportunity under predeclared secondary-cost margins."}
    return {"artifact_count": len(hashes), "request_count": total_requests,
            "complete_planned_matrix": actual == expected and total_requests == 2000 and len(hashes) == 100,
            "paired_payloads_match": True, "source_hashes": hashes, "execution_manifests": manifests,
            "hardware": hardware,
            "rows": rows, "paired_gate_effects": effects, "reference_opportunity": gate,
            "first_rep_samples": samples,
            "scope": "Same token-VTC; native batching families; local issue/opportunity, not a universal claim or complete external scheduler reproduction."}


def report(data):
    lines = ["# M2 slot accounting and batching", "",
             f"Formal runs: {data['artifact_count']}; requests: {data['request_count']}; complete={data['complete_planned_matrix']}", "",
             "| Configuration | Gate | Order | N | Short p95 | Short mean | Short max | Long p95 | Makespan | Premature slot denials | Mixed batches |",
             "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["rows"]:
        m = row["mean"]
        lines.append(f"| {row['config']} | {row['gate']} | {row['order']} | {len(row['runs'])} | {m['short_p95']:.3f} | {m['short_mean_jct']:.3f} | {m['short_max_jct']:.3f} | {m['long_p95']:.3f} | {m['makespan']:.3f} | {m['reuse_slot_gate_witnesses']:.1f} | {m['mixed_batches']:.1f} |")
    lines += ["", "## Within-session correction effects", "",
              "Positive is improvement; percentages, paired run-level bootstrap intervals.", "",
              "| Configuration | Order | Short p95 | Long p95 | Makespan |",
              "|---|---|---:|---:|---:|"]
    for row in data["paired_gate_effects"]:
        cells = []
        for endpoint in ("short_p95", "long_p95", "makespan"):
            e = row["effects"][endpoint]
            cells.append(f"{100*e['mean_improvement']:+.2f} [{100*e['ci95'][0]:+.2f}, {100*e['ci95'][1]:+.2f}]")
        lines.append(f"| {row['config']} | {row['order']} | " + " | ".join(cells) + " |")
    lines += ["", "Reference opportunity: " + str(None if data["reference_opportunity"] is None else data["reference_opportunity"]["passed"]),
              "", "All batching comparisons use the same token-VTC. No external SOTA system was faithfully reproduced.",
              "Cross-batching comparisons are session-blocked; gate A/B is randomized within-session.",
              "With only 16 short requests, linearly interpolated p95 is sensitive to the final two ranks; mean, max and first-repetition ECDF accompany it.",
              "Absolute latency differences between batching families include compute/batch effects; no KDA-only causality claim."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.directories)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.output / "report.md").write_text(report(data))
    print(report(data))


if __name__ == "__main__":
    main()
