#!/usr/bin/env python3
"""Summarize M2 controls without promoting a configuration sweep to SOTA proof."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from analyze_admission_batch_motivation import batch_metrics, decode_metrics


ENDPOINTS = ("split_p95", "packed_p95", "makespan")


def finite(value):
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list):
        return [finite(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    lower, upper = math.floor(index), math.ceil(index)
    return ordered[lower] * (upper - index) + ordered[upper] * (index - lower) if lower != upper else ordered[lower]


def paired_effect(baseline, candidate, *, draws=10000, seed=20260909):
    if len(baseline) != len(candidate) or not baseline or min(baseline) <= 0:
        raise ValueError("positive, paired nonempty baseline values are required")
    effects = [1 - c / b for b, c in zip(baseline, candidate)]
    rng = random.Random(seed)
    samples = [statistics.fmean(rng.choices(effects, k=len(effects))) for _ in range(draws)]
    return {"mean_improvement": statistics.fmean(effects),
            "ci95": [percentile(samples, 0.025), percentile(samples, 0.975)],
            "n": len(effects), "claim_eligible_n": len(effects) >= 5}


def dominates(left, right):
    return all(left[k] <= right[k] for k in ENDPOINTS) and any(left[k] < right[k] for k in ENDPOINTS)


def config_name(exp):
    return f"{exp['batch_mode']}/n{exp['admission_cap']}/{exp['runtime_policy']}"


def metrics(item, isolations):
    exp = item["experiment"]
    requests = item["requests"]
    value = {"makespan": max(r["finished_s"] for r in requests),
             "max_arrival_lateness_s": max(r["submitted_s"] - r["scheduled_arrival_s"] for r in requests)}
    resource = item["scheduler_resource_trace"]
    value["peak_active"] = resource["system"]["peak_active_requests"]
    for group in ("split", "packed"):
        own = [r for r in requests if r["group"] == group]
        usage = resource["tenants"][group]
        p95 = item["summary"]["groups"][group]["p95_jct_s"]
        key = (exp["stage"], exp["batch_mode"], exp["admission_cap"],
               exp["split_factor"], exp["repetition"], group)
        isolated = isolations[key]["summary"]["groups"][group]["p95_jct_s"]
        value.update({
            f"{group}_p95": p95, f"{group}_slowdown_ratio": p95 / isolated,
            f"{group}_active_per_request": usage["active_request_seconds"] / len(own),
            f"{group}_decode_interval": decode_metrics(own)[0],
            f"{group}_peak_state": usage["peak_state_slots"],
            f"{group}_peak_kv": usage["peak_kv_tokens"],
            f"{group}_completed_charge": sum(r["prompt_tokens"] + 2 * r["completion_tokens"] for r in own),
        })
        vtc = item.get("scheduler_vtc_trace") or {}
        value[f"{group}_queue_p95"] = vtc.get("queue_waits", {}).get(group, {}).get("p95_seconds")
    value.update(finite(batch_metrics(item)))
    return value


def analyze(directories):
    artifacts = []
    hashes = {}
    manifests = {}
    seen = set()
    for directory in directories:
        manifest = directory / "manifest.json"
        if manifest.exists():
            manifests[str(directory)] = json.loads(manifest.read_text())
        for path in sorted(directory.glob("m2-*.json")):
            item = json.loads(path.read_text())
            exp = item["experiment"]
            if exp["stage"] in ("warmup", "smoke"):
                continue
            key = (exp["stage"], exp["run_id"])
            if key in seen:
                raise ValueError(f"Duplicate run: {key}")
            seen.add(key)
            if item["integrity_errors"]:
                raise ValueError(f"Invalid artifact {path}: {item['integrity_errors']}")
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            artifacts.append(item)
    if not artifacts:
        raise ValueError("No completed non-smoke artifacts")
    isolations = {}
    payloads = defaultdict(set)
    for item in artifacts:
        exp = item["experiment"]
        composition = exp["composition"]
        payloads[(exp["stage"], exp["split_factor"], exp["repetition"], composition)].add(item["payload_fingerprint"])
        if composition != "mixed":
            key = (exp["stage"], exp["batch_mode"], exp["admission_cap"],
                   exp["split_factor"], exp["repetition"], composition.removesuffix("-only"))
            if key in isolations:
                raise ValueError(f"Duplicate isolation: {key}")
            isolations[key] = item
    if any(len(values) != 1 for values in payloads.values()):
        raise ValueError("Paired request payloads do not match across policies/configurations")
    groups = defaultdict(list)
    for item in artifacts:
        exp = item["experiment"]
        if exp["composition"] == "mixed":
            key = (exp["stage"], exp["split_factor"], config_name(exp))
            groups[key].append({"rep": exp["repetition"], **metrics(item, isolations)})
    rows = []
    for (stage, factor, config), runs in sorted(groups.items()):
        means = {key: statistics.fmean(r[key] for r in runs if r[key] is not None)
                 if any(r[key] is not None for r in runs) else None
                 for key in runs[0] if key != "rep"}
        rows.append({"stage": stage, "factor": factor, "config": config,
                     "runs": sorted(runs, key=lambda r: r["rep"]), "mean": means})
    frontiers = {}
    for stage, factor in sorted({(row["stage"], row["factor"]) for row in rows}):
        subset = [r for r in rows if (r["stage"], r["factor"]) == (stage, factor)]
        frontiers[f"{stage}/f{factor}"] = [r["config"] for r in subset
            if not any(dominates(other["mean"], r["mean"]) for other in subset)]
    return {"artifact_count": len(artifacts),
            "request_count": sum(len(i["requests"]) for i in artifacts),
            "source_hashes": hashes, "paired_payloads_match": True,
            "execution_manifests": manifests,
            "rows": rows, "descriptive_frontiers": frontiers,
            "scope": "Native batching controls; no complete SOTA implementation or theoretical fairness claim"}


def report(data):
    lines = ["# M2 batch frontier measurements", "",
             f"Artifacts: {data['artifact_count']}; requests: {data['request_count']}. Paired payload identity verified.",
             "", "These are native scheduling controls, not full Sarathi/FairBatching reproductions.",
             "One-repeat screening is descriptive, not confirmatory evidence.", "",
             "| Stage / shape | Configuration | Runs | Split p95 (s) | Packed p95 (s) | Makespan (s) | Split slowdown | Packed slowdown |",
             "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in data["rows"]:
        m = row["mean"]
        lines.append(f"| {row['stage']}/f{row['factor']} | {row['config']} | {len(row['runs'])} | "
                     f"{m['split_p95']:.3f} | {m['packed_p95']:.3f} | {m['makespan']:.3f} | "
                     f"{m['split_slowdown_ratio']:.3f} | {m['packed_slowdown_ratio']:.3f} |")
    lines.extend(["", "## Interpretation", "",
                  "Frontiers minimize both tenants' absolute p95 and makespan. They are descriptive;",
                  "a dominated sample mean is not proof of a significant or SOTA-wide improvement.",
                  "Completed token equality is not a backlogged-window fairness guarantee.",
                  "Matched isolation uses the same payload, arrival offsets and configuration.",
                  "Decode intervals and launch gaps are not synchronized GPU kernel timings.",
                  "Missing instant-share queue measurements remain null.", "",
                  "## Descriptive frontiers", ""])
    for key, configs in data["descriptive_frontiers"].items():
        lines.append(f"- {key}: {', '.join(configs)}")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.directories)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis.json").write_text(json.dumps(finite(data), indent=2, allow_nan=False) + "\n")
    (args.output / "report.md").write_text(report(data))
    print(report(data))


if __name__ == "__main__":
    main()
