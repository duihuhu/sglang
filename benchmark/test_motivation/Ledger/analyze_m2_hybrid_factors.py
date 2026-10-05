#!/usr/bin/env python3
"""Separate structural hybrid factors, observed constraints, and policy effects."""

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics

from analyze_m2_batch_frontier import paired_effect, percentile


def mean(values):
    return statistics.fmean(values) if values else None


def bootstrap_mean(values, draws=10000):
    if not values:
        raise ValueError("empty sample")
    rng = random.Random(20260910)
    samples = [mean(rng.choices(values, k=len(values))) for _ in range(draws)]
    return {"mean": mean(values), "ci95": [percentile(samples, .025), percentile(samples, .975)],
            "n": len(values)}


def lifecycle(item):
    trace = item["scheduler_resource_trace"]
    offset = trace["event_clock_origin_s"] - item["experiment_clock_origin_s"]
    starts, finishes = defaultdict(list), defaultdict(list)
    for event in trace["events"]:
        timestamp = offset + event["observed_s"]
        for added in event["added"]:
            starts[added["rid"]].append(timestamp)
        for rid in event["removed"]:
            finishes[rid].append(timestamp)
    expected = {r["rid"] for r in item["requests"]}
    if set(starts) != expected or set(finishes) != expected:
        raise ValueError("missing activation/release event")
    if any(len(starts[rid]) != 1 or len(finishes[rid]) != 1 for rid in expected):
        raise ValueError("unexpected reactivation; cannot interpret as one lifetime")
    return {rid: (starts[rid][0], finishes[rid][0]) for rid in expected}


def metrics(item):
    spans = lifecycle(item)
    requests = item["requests"]
    trace = item["scheduler_resource_trace"]
    blocks = item["scheduler_batch_trace"].get("admission_blocks", [])
    value = {"makespan": max(r["finished_s"] for r in requests),
             "peak_state": trace["system"]["peak_state_slots"],
             "peak_kv": trace["system"]["peak_kv_tokens"],
             "state_slot_seconds": trace["system"]["state_slot_seconds"],
             "max_arrival_lateness_s": max(r["submitted_s"] - r["scheduled_arrival_s"] for r in requests),
             "one_state_per_active": all(e["state_slots"] == e["active_requests"] for e in trace["events"])}
    for group, summary in item["summary"]["groups"].items():
        own = [r for r in requests if r["group"] == group]
        queues = [spans[r["rid"]][0] - r["submitted_s"] for r in own]
        if min(queues) < -0.01:
            raise ValueError("activation precedes submission: incompatible clocks")
        value.update({f"{group}_p95": summary["p95_jct_s"],
                      f"{group}_queue_p95": percentile(queues, .95),
                      f"{group}_active_mean": mean([b - a for a, b in (spans[r["rid"]] for r in own)]),
                      f"{group}_completed_charge": sum(r["prompt_tokens"] + 2*r["completion_tokens"] for r in own)})
    for reason in ("request_limit", "prefill_no_token", "prefill_other"):
        value[f"blocks_{reason}"] = sum(b["reason"] == reason for b in blocks)
    value["blocks_state_physically_empty"] = sum(b.get("state_slots_free") == 0 for b in blocks)
    value["blocks_kv_budget_with_state_free"] = sum(
        b["reason"] == "prefill_no_token" and (b.get("state_slots_free") or 0) > 0 for b in blocks)
    continuation = []
    previous = {"kv_tokens": 0, "state_slots": 0}
    for event in trace["events"]:
        delta_kv = event["kv_tokens"] - previous["kv_tokens"]
        delta_state = event["state_slots"] - previous["state_slots"]
        if delta_kv >= 2048 and delta_state == 0 and not event["added"] and not event["removed"]:
            continuation.append(delta_kv)
        previous = event
    value["kv_progress_without_new_state_events"] = len(continuation)
    if item["experiment"]["family"] == "E3":
        probe = next(r for r in requests if r["group"] == "probe")
        backgrounds = [r for r in requests if r["group"] == "background"]
        arrival = probe["submitted_s"]
        alive = sum(spans[r["rid"]][0] <= arrival < spans[r["rid"]][1] for r in backgrounds)
        started, released = spans[probe["rid"]]
        first_release = min(spans[r["rid"]][1] for r in backgrounds)
        value.update({"background_alive_at_probe": alive,
                      "probe_queue": started - arrival,
                      "probe_ttft": probe["ttft_s"], "probe_jct": probe["latency_s"],
                      "probe_activation_after_first_background_release_s": started - first_release,
                      "probe_arrival_s": arrival, "probe_activation_s": started,
                      "first_background_release_s": first_release,
                      "lifecycle_contrast_valid": alive == (8 if item["experiment"]["variant"] == "o256" else 0)})
    return value


def analyze(directories):
    groups, fingerprints, contents = defaultdict(list), defaultdict(set), defaultdict(set)
    hashes, manifests, seen, block_samples, trajectories = {}, {}, set(), {}, {}
    request_count = 0
    for directory in directories:
        manifest = directory / "manifest.json"
        if manifest.exists():
            manifests[str(directory)] = json.loads(manifest.read_text())
        for path in sorted(directory.glob("m2h-*.json")):
            item = json.loads(path.read_text())
            exp = item["experiment"]
            if exp["stage"] != "confirm":
                continue
            if exp["run_id"] in seen:
                raise ValueError("duplicate run")
            seen.add(exp["run_id"])
            if item["integrity_errors"]:
                raise ValueError(f"invalid run: {path}")
            if any(not r["success"] or r["cached_tokens"] or r["num_retractions"]
                   or r["completion_tokens"] != r["output_len"]
                   or r["prompt_tokens"] != r["input_len"] for r in item["requests"]):
                raise ValueError("request integrity mismatch")
            resource = item["scheduler_resource_trace"]
            batch = item["scheduler_batch_trace"]
            if (resource.get("dropped_events") or batch.get("dropped_events")
                    or batch.get("dropped_admission_blocks")
                    or resource["system"]["current_active_requests"]):
                raise ValueError("incomplete resource/batch trace")
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            request_count += len(item["requests"])
            fingerprints[(exp["family"], exp["variant"], exp["rep"])].add(item["payload_fingerprint"])
            if exp["family"] == "E2":
                contents[exp["rep"]].add(item["content_fingerprint"])
            key = (exp["family"], exp["regime"], exp["variant"], exp["policy"])
            groups[key].append({"rep": exp["rep"], **metrics(item)})
            if exp["rep"] == 10:
                block_samples[exp["run_id"]] = item["scheduler_batch_trace"].get("admission_blocks", [])[:30]
                trajectories[exp["run_id"]] = {
                    "experiment": exp, "events": item["scheduler_resource_trace"]["events"],
                    "experiment_clock_offset_s": item["scheduler_resource_trace"]["event_clock_origin_s"] - item["experiment_clock_origin_s"],
                }
    if not groups or any(len(v) != 1 for v in list(fingerprints.values()) + list(contents.values())):
        raise ValueError("missing data or unmatched paired inputs")
    rows = []
    for (family, regime, variant, policy), runs in sorted(groups.items()):
        means = {key: mean([r[key] for r in runs]) for key in runs[0] if key != "rep"}
        rows.append({"family": family, "regime": regime, "variant": variant, "policy": policy,
                     "runs": sorted(runs, key=lambda r: r["rep"]), "mean": means})
    lookup = {(r["family"], r["regime"], r["variant"], r["policy"]): r for r in rows}
    effects = []
    for key, baseline in lookup.items():
        family, regime, variant, policy = key
        if family != "E2" or policy != "vtc":
            continue
        candidate = lookup.get((family, regime, variant, "instant"))
        if candidate is None:
            continue
        a = {r["rep"]: r for r in baseline["runs"]}
        b = {r["rep"]: r for r in candidate["runs"]}
        if set(a) != set(b):
            continue
        reps = sorted(a)
        effects.append({"regime": regime, "variant": variant, "reps": reps,
                        "effects": {endpoint: paired_effect([a[i][endpoint] for i in reps],
                                    [b[i][endpoint] for i in reps]) for endpoint in
                                    ("short_p95", "long_p95", "makespan")}})
    holding = {}
    for regime in ("wide", "state-low", "cap-control", "kv-low"):
        short = lookup.get(("E3", regime, "o32", "fcfs"))
        long = lookup.get(("E3", regime, "o256", "fcfs"))
        if short is None or long is None:
            continue
        a = {r["rep"]: r for r in short["runs"]}
        b = {r["rep"]: r for r in long["runs"]}
        if set(a) != set(b):
            continue
        reps = sorted(a)
        differences = [b[i]["probe_queue"] - a[i]["probe_queue"] for i in reps]
        holding[regime] = {"reps": reps, "queue_difference_s": differences,
                           "valid": all(a[i]["lifecycle_contrast_valid"] and b[i]["lifecycle_contrast_valid"] for i in reps),
                           **bootstrap_mean(differences)}
    did = {}
    if "wide" in holding:
        for regime, data in holding.items():
            if regime == "wide" or data["reps"] != holding["wide"]["reps"]:
                continue
            differences = [a-b for a,b in zip(data["queue_difference_s"], holding["wide"]["queue_difference_s"])]
            did[regime] = {"contrast_valid": data["valid"] and holding["wide"]["valid"],
                           **bootstrap_mean(differences)}
    expected = {("E1", "wide", f"n{n}", "fcfs"): {10, 11, 12} for n in (1, 4, 16, 32)}
    for regime in ("wide", "state-low", "cap-control", "kv-low"):
        for variant in ("short-first", "long-first"):
            for policy in ("vtc", "instant"):
                expected[("E2", regime, variant, policy)] = set(range(10, 15))
        for variant in ("o32", "o256"):
            expected[("E3", regime, variant, "fcfs")] = set(range(10, 15))
    actual = {key: {r["rep"] for r in runs} for key, runs in groups.items()}
    complete = actual == expected and len(hashes) == 132 and request_count == 2119
    return {"artifact_count": len(hashes), "request_count": request_count,
            "complete_planned_matrix": complete,
            "all_one_state_per_active": all(r["one_state_per_active"] for runs in groups.values() for r in runs),
            "max_arrival_lateness_s": max(r["max_arrival_lateness_s"] for runs in groups.values() for r in runs),
            "source_hashes": hashes, "execution_manifests": manifests,
            "paired_payloads_match": True, "rows": rows,
            "instant_vs_vtc_exploratory_effects": effects,
            "holding_queue_differences": holding, "holding_vs_wide_difference_in_differences": did,
            "first_rep_block_samples": block_samples, "first_rep_trajectories": trajectories,
            "sota_inapplicability_demonstrated": False,
            "scope": "Hybrid factors and local-policy probes; correct capacity checks retained; no SOTA-wide conclusion."}


def report(data):
    lines = ["# M2 hybrid factors measurements", "",
             f"Formal runs: {data['artifact_count']}; requests: {data['request_count']}; paired payloads match.",
             f"Complete planned matrix: {data['complete_planned_matrix']}; one-state-per-active: {data['all_one_state_per_active']}.",
             "", "## E1 activation", "",
             "| Shape | Runs | Peak state slots | Peak KV tokens | KV-progress/no-new-state events |",
             "|---|---:|---:|---:|---:|"]
    for row in data["rows"]:
        if row["family"] == "E1":
            m = row["mean"]
            lines.append(f"| {row['variant']} | {len(row['runs'])} | {m['peak_state']:.1f} | {m['peak_kv']:.1f} | {m['kv_progress_without_new_state_events']:.1f} |")
    lines += ["", "## E2 fixed candidates", "",
              "| Regime | Submission | Policy | Runs | Short p95 | Long p95 | Makespan | Request-limit / no-token blocks |",
              "|---|---|---|---:|---:|---:|---:|---:|"]
    for row in data["rows"]:
        if row["family"] == "E2":
            m = row["mean"]
            lines.append(f"| {row['regime']} | {row['variant']} | {row['policy']} | {len(row['runs'])} | {m['short_p95']:.3f} | {m['long_p95']:.3f} | {m['makespan']:.3f} | {m['blocks_request_limit']:.1f} / {m['blocks_prefill_no_token']:.1f} |")
    lines += ["", "## E3 probe", "",
              "| Regime | Background output | Runs | Probe queue | TTFT | JCT | Live background at arrival | Valid fraction |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in data["rows"]:
        if row["family"] == "E3":
            m = row["mean"]
            lines.append(f"| {row['regime']} | {row['variant']} | {len(row['runs'])} | {m['probe_queue']:.3f} | {m['probe_ttft']:.3f} | {m['probe_jct']:.3f} | {m['background_alive_at_probe']:.1f} | {m['lifecycle_contrast_valid']:.1f} |")
    lines += ["", "## Holding interaction relative to wide", "",
              "Long-minus-short background probe queue effect, minus the same effect on wide pools; seconds.", ""]
    for regime, result in data["holding_vs_wide_difference_in_differences"].items():
        lines.append(f"- {regime}: {result['mean']:.3f}, CI [{result['ci95'][0]:.3f}, {result['ci95'][1]:.3f}], valid={result['contrast_valid']}.")
    lines += ["", "## Boundaries", "",
              "Activation is first scheduler-observed active ownership, not exact GPU start time.",
              "Admission reasons are observed return paths; physical pool counts can exclude pending selected reservations.",
              "A positive effect reproduced by cap-control is not unique KDA execution causality.",
              "E1 output work and E3 background work differ intentionally; neither is an equal-total-work fairness comparison.",
              "Policy effect intervals are exploratory multiple contrasts, not simultaneous significance claims.",
              "No complete SOTA implementation or calibrated-model comparison was performed."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.directories)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
    (args.output / "report.md").write_text(report(data))
    print(report(data))


if __name__ == "__main__":
    main()
