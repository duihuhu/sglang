#!/usr/bin/env python3
"""Validate and summarize independent M2 pool runs without merging diagnostics."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_m2_hybrid_factors import lifecycle
from analyze_m2_activation_pacing import nearest_rank
from run_m2_pool_comparison import COUNT, capacity, integrity_errors, progress_errors


def metrics(item):
    spans = lifecycle(item)
    requests = item["requests"]
    system = item["scheduler_resource_trace"]["system"]
    return {
        "mean_jct_s": statistics.fmean(r["latency_s"] for r in requests),
        "p95_jct_s": nearest_rank([r["latency_s"] for r in requests]),
        "mean_queue_s": statistics.fmean(spans[r["rid"]][0] - r["submitted_s"] for r in requests),
        "mean_ttft_s": statistics.fmean(r["ttft_s"] for r in requests),
        "mean_tpot_s": statistics.fmean((r["latency_s"] - r["ttft_s"]) / (r["completion_tokens"] - 1) for r in requests),
        "makespan_s": max(r["finished_s"] for r in requests) - min(r["submitted_s"] for r in requests),
        "state_slot_seconds": system["state_slot_seconds"],
        "peak_active": system["peak_active_requests"], "peak_kv_tokens": system["peak_kv_tokens"],
        "peak_state_slots": system["peak_state_slots"],
        "peak_live_cache_bytes_proxy": max(e["state_slots"] * 21708800 + e["kv_tokens"] * 8064
                                           for e in item["scheduler_resource_trace"]["events"]),
        "blocks": dict(Counter(b["reason"] for b in item["scheduler_batch_trace"]["admission_blocks"])),
        "max_arrival_lateness_ms": 1000 * max(r["submitted_s"] - r["scheduled_arrival_s"] for r in requests),
    }


def paired_contrasts(rows):
    index = {(r["experiment"]["pool"], r["experiment"]["mode"], r["experiment"]["cap"], r["experiment"]["rep"]): r
             for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate contrast cell")
    pairs = (
        ("static-repartition", ("static-natural", 485), ("static-rebalanced", 607)),
        ("unified-and-cap", ("static-natural", 485), ("unified-raised-cap", 607)),
        ("unified-switch-only", ("static-natural", 485), ("unified-old-cap", 485)),
        ("unified-vs-rebalanced-607", ("static-rebalanced", 607), ("unified-raised-cap", 607)),
        ("unified-vs-rebalanced-485", ("static-rebalanced", 485), ("unified-raised-cap", 485)),
        ("static-cap-relaxation", ("static-rebalanced", 485), ("static-rebalanced", 607)),
        ("unified-cap-relaxation", ("unified-raised-cap", 485), ("unified-raised-cap", 607)),
    )
    results = []
    for mode in ("separate-4096", "mixed-4096"):
        for name, a, b in pairs:
            samples = []
            for rep in (31, 32):
                left, right = index.get((a[0], mode, a[1], rep)), index.get((b[0], mode, b[1], rep))
                if left is None or right is None:
                    continue
                if left["payload_fingerprint"] != right["payload_fingerprint"]:
                    raise ValueError("Contrast workload differs")
                metrics_a, metrics_b = left["metrics"], right["metrics"]
                gains = {k: 100 * (metrics_a[k] - metrics_b[k]) / metrics_a[k]
                         for k in ("mean_jct_s", "p95_jct_s", "makespan_s", "mean_queue_s", "state_slot_seconds")}
                samples.append({"rep": rep, "gain_percent": gains,
                                "constrained_benefit": gains["mean_jct_s"] > 0 and
                                gains["p95_jct_s"] >= -5 and gains["makespan_s"] >= -5})
            if samples:
                results.append({"mode": mode, "comparison": name, "from": a, "to": b,
                                "paired_samples": samples,
                                "mean_gain_percent": {k: statistics.fmean(s["gain_percent"][k] for s in samples)
                                                      for k in samples[0]["gain_percent"]},
                                "all_repeats_constrained_benefit": len(samples) == 2 and all(s["constrained_benefit"] for s in samples)})
    return results


def analyze(directory, allow_partial=False):
    manifest = json.loads((directory / "manifest.json").read_text())
    done_file = directory / "completed.json"
    complete = done_file.exists()
    if not complete and not allow_partial:
        raise ValueError("Experiment has no completed marker")
    if complete:
        done = json.loads(done_file.read_text())
        if not done["completed"] or done["source_hashes"] != manifest["source_hashes"]:
            raise ValueError("Completion/source mismatch")
    rows, warmups, memory, correctness, gpu_samples, server_logs = [], [], [], [], [], []
    hashes, fingerprints, seen = {}, defaultdict(set), set()
    for session in sorted(directory.glob("session-*")):
        log_file = session / "server.log"
        if log_file.exists():
            log = log_file.read_text()
            server_logs.append({"session": session.name,
                                "allocation_retry_warnings": log.count("memory allocation failed with OOM"),
                                "custom_all_reduce_fallback_warnings": log.count("To silence this warning, specify --disable-custom-all-reduce explicitly."),
                                "sha256": hashlib.sha256(log_file.read_bytes()).hexdigest()})
        for path in sorted(session.glob("m2pool-*.json")):
            item = json.loads(path.read_text())
            e = item["experiment"]
            key = (session.name, e["rep"], e["cap"], e["stage"])
            if key in seen:
                raise ValueError("Duplicate run")
            seen.add(key)
            errors = item["integrity_errors"] + integrity_errors(item, COUNT, .05) + progress_errors(item)
            if errors:
                raise ValueError(f"Invalid run {path}: {errors}")
            for when in ("server_initial", "server_final"):
                if list(capacity(item[when], e["mode"], e["pool"], e["cap"])) != item["resolved_capacity"]:
                    raise ValueError("Readback/capacity mismatch")
            timing = item["client_timing"]
            if (timing["connection_events"]["created"].get("replay", 0)
                    or timing["connection_events"]["reused"].get("replay") != COUNT):
                raise ValueError("HTTP timing isolation failed")
            row = {"session": session.name, "experiment": e, "metrics": metrics(item),
                   "capacity": item["resolved_capacity"], "payload_fingerprint": item["payload_fingerprint"]}
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
            if e["stage"] == "warmup":
                warmups.append(row)
            else:
                rows.append(row)
                fingerprints[e["rep"]].add(item["payload_fingerprint"])
                gpu_samples.extend(g for sample in item["gpu_health_samples"] for g in sample["gpus"])
            for label in ("server_initial", "server_final"):
                audit = item[label]["internal_states"][0].get("m2_pool_audit")
                if audit:
                    row[label + "_audit"] = audit
        for path in sorted((session / "memory").glob("*.json")):
            data = json.loads(path.read_text())
            categories = defaultdict(int)
            for storage in data.pop("storages"):
                categories[storage["names"][0].split(".")[0]] += storage["bytes"]
            memory.append({"session": session.name, "file": str(path), **data,
                           "storage_categories_bytes": dict(categories)})
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        for path in session.glob("correctness-*.json"):
            data = json.loads(path.read_text())
            correctness.append({"session": session.name, "records": [
                {k: r[k] for k in ("repeat", "input_length", "output_ids")} for r in data["records"]]})
            hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    if any(len(values) > 1 for values in fingerprints.values()):
        raise ValueError("Unpaired workload payloads")
    expected_warm = {(r["session"], r["experiment"]["rep"], r["experiment"]["cap"]) for r in rows}
    actual_warm = {(r["session"], r["experiment"]["rep"], r["experiment"]["cap"]) for r in warmups}
    if complete and expected_warm != actual_warm:
        raise ValueError("Warmup and measured matrix differ")
    if complete and manifest["stage"] == "measure":
        expected = {(f"session-{i:02d}-{s['arm']}-{s['mode']}", rep, cap)
                    for i, s in enumerate(manifest["frozen_plan"]["sessions"])
                    for rep in s["reps"] for cap in s["caps"]}
        if expected_warm != expected or any(r["experiment"]["stage"] != "measure" for r in rows):
            raise ValueError("Measured matrix differs from the frozen plan")
    grouped = defaultdict(list)
    for row in rows:
        e = row["experiment"]
        grouped[(e["pool"], e["mode"], e["cap"])].append(row)
    means = []
    for (arm, mode, cap), own in sorted(grouped.items()):
        keys = [k for k, v in own[0]["metrics"].items() if isinstance(v, (int, float))]
        means.append({"arm": arm, "mode": mode, "cap": cap, "n": len(own),
                      "mean": {k: statistics.fmean(r["metrics"][k] for r in own) for k in keys},
                      "min": {k: min(r["metrics"][k] for r in own) for k in keys},
                      "max": {k: max(r["metrics"][k] for r in own) for k in keys}})
    loaded = [g for g in gpu_samples if g["utilization"] >= 80]
    return {"complete": complete, "stage": manifest["stage"], "manifest": manifest,
            "rows": rows, "means": means, "memory": memory, "correctness": correctness,
            "paired_contrasts": paired_contrasts(rows) if manifest["stage"] == "measure" else [],
            "server_log_audit": server_logs,
            "gpu_health": {"sample_count": len(gpu_samples),
                           "loaded_sm_mhz_range": [min(g["sm_mhz"] for g in loaded), max(g["sm_mhz"] for g in loaded)] if loaded else None,
                           "max_temperature_c": max((g["temperature_c"] for g in gpu_samples), default=None),
                           "max_memory_mib": max((g["memory_mib"] for g in gpu_samples), default=None)},
            "counts": {"runs": len(rows), "requests": COUNT * len(rows),
                       "warmups": len(warmups), "warmup_requests": COUNT * len(warmups)},
            "artifact_sha256": hashes,
            "scope": "Same-model workload-specific pool/cap test; inspect timing is diagnostic, not a formal performance comparison."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = analyze(args.directory, args.allow_partial)
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out / "analysis.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    lines = ["# M2 Pool Comparison", "", f"Stage: {result['stage']}; complete: {result['complete']}; counts: {result['counts']}", "",
             "Inspect times are diagnostic only.", "",
             "| Pool | Mode | Cap | N | Mean JCT | Queue | p95 JCT | Makespan | Peak active |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in result["means"]:
        m = row["mean"]
        lines.append(f"| {row['arm']} | {row['mode']} | {row['cap']} | {row['n']} | {m['mean_jct_s']:.3f} | "
                     f"{m['mean_queue_s']:.3f} | {m['p95_jct_s']:.3f} | {m['makespan_s']:.3f} | {m['peak_active']:.1f} |")
    if result["paired_contrasts"]:
        lines += ["", "## Paired Changes", "", "Positive gain is better. Two repeats do not give a precise population CI.", "",
                  "| Mode | Comparison | Mean JCT gain % | p95 gain % | Makespan gain % | Both repeats constrained benefit |",
                  "|---|---|---:|---:|---:|---|"]
        for row in result["paired_contrasts"]:
            gain = row["mean_gain_percent"]
            lines.append(f"| {row['mode']} | {row['comparison']} | {gain['mean_jct_s']:.2f} | {gain['p95_jct_s']:.2f} | "
                         f"{gain['makespan_s']:.2f} | {row['all_repeats_constrained_benefit']} |")
    with (args.out / "report.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
