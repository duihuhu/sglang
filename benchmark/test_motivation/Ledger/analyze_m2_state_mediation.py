#!/usr/bin/env python3
"""Report all calibration cells and the prespecified cap-mediated contrast."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_m2_hybrid_factors import lifecycle
from analyze_m2_progress_probe import intersect_intervals, metrics, union_intervals


def contrasts(rows):
    indexed = {}
    for row in rows:
        e = row["experiment"]
        key = (e["pool"], e["mode"], e["cap"], e["rep"])
        if key in indexed:
            raise ValueError("Duplicate cell")
        indexed[key] = row
    result = []
    for rep in (11, 12):
        def get(pool, mode, cap):
            return indexed[(pool, mode, cap, rep)]
        for key in ("all_mean_jct_s", "mean_queue_s", "mean_ttft_s", "makespan_s", "state_slot_seconds"):
            def gap(pool, cap):
                return get(pool, "mixed-4096", cap)["metrics"][key] - get(pool, "separate-4096", cap)["metrics"][key]
            changes = {}
            for mode in ("separate-4096", "mixed-4096"):
                a = get("natural", mode, 485)["metrics"][key]
                b = get("wide", mode, 485)["metrics"][key]
                c = get("wide", mode, 607)["metrics"][key]
                changes[mode] = {"A": a, "B": b, "C": c,
                                 "pool_change_B_minus_A": b-a,
                                 "cap_relaxation_B_minus_C": b-c}
            result.append({"rep": rep, "metric": key, "by_mode": changes,
                           "gap_natural_485": gap("natural", 485),
                           "gap_wide_485": gap("wide", 485),
                           "gap_wide_607": gap("wide", 607),
                           "delta_tight_minus_relaxed": gap("wide", 485) - gap("wide", 607)})
    return result


def extra_metrics(item):
    from analyze_m2_activation_pacing import nearest_rank

    resource = item["scheduler_resource_trace"]
    spans = lifecycle(item)
    requests = {r["rid"]: r for r in item["requests"]}
    offset = resource["event_clock_origin_s"] - item["experiment_clock_origin_s"]
    cap = item["experiment"]["cap"]
    cap_windows = union_intervals((offset + a["observed_s"], offset + b["observed_s"])
                                 for a, b in zip(resource["events"], resource["events"][1:])
                                 if a["active_requests"] == cap)
    pending = union_intervals((requests[rid]["submitted_s"], start) for rid, (start, _) in spans.items())
    simultaneous = intersect_intervals(cap_windows, pending)
    batch = item["scheduler_batch_trace"]
    modes = sorted({e["mode"] for e in batch["events"]})
    return {
        "cap_full_with_pending_seconds": sum(b-a for a, b in simultaneous),
        "kv_peak_fraction": resource["system"]["peak_kv_tokens"] / item["resolved_capacity"][1],
        "mean_tpot_s": statistics.fmean((r["latency_s"] - r["ttft_s"]) / (r["completion_tokens"] - 1)
                                       for r in requests.values()),
        "p95_jct_s": nearest_rank([r["latency_s"] for r in requests.values()]),
        "batch_counts": dict(Counter(e["mode"] for e in batch["events"])),
        "batch_sizes": {mode: {"mean": statistics.fmean(e["batch_size"] for e in batch["events"] if e["mode"] == mode),
                               "max": max(e["batch_size"] for e in batch["events"] if e["mode"] == mode)} for mode in modes},
        "no_token_blocks": sum(e["reason"] == "prefill_no_token" for e in batch["admission_blocks"]),
    }


def load_stage(directory, stage):
    from run_m2_state_mediation import COUNT, integrity_errors, progress_errors, session_plan, validate_info
    manifest = json.loads((directory / "manifest.json").read_text())
    done = json.loads((directory / "completed.json").read_text())
    if manifest["args"]["stage"] != stage or not done.get("completed") or done["source_hashes"] != manifest["source_hashes"]:
        raise ValueError("Incomplete or mismatched stage")
    if "recovery" in manifest:
        from recover_m2_state_mediation import validate_recovery
        validate_recovery(directory, manifest, done)
    expected = {(pool, mode, rep, cap) for pool, mode, rep, caps in session_plan(stage) for cap in caps}
    rows, hashes, seen, warmups, samples = [], {}, set(), set(), []
    for path in sorted(directory.glob("m2mediate-*.json")):
        item = json.loads(path.read_text())
        e = item["experiment"]
        timing = item.get("client_timing", {})
        if (timing.get("gc_disabled_during_replay") is not True
                or timing.get("health_monitor") != "persistent-nvidia-smi-500ms"
                or timing.get("http_preconnect_count") != COUNT
                or timing.get("connection_events", {}).get("created", {}).get("replay", 0)
                or timing.get("connection_events", {}).get("reused", {}).get("replay") != COUNT):
            raise ValueError("Require v3 isolated, preconnected client timing")
        key = (e["pool"], e["mode"], e["rep"], e["cap"])
        if key not in expected or e["stage"] not in (stage, "warmup"):
            raise ValueError(f"Unexpected artifact: {path}")
        own = warmups if e["stage"] == "warmup" else seen
        if key in own or item["integrity_errors"] or integrity_errors(item, COUNT, .05) or progress_errors(item):
            raise ValueError(f"Invalid or duplicate run: {path}")
        own.add(key)
        for field in ("server_initial", "server_final"):
            if list(validate_info(item[field], e["mode"], e["pool"], e["cap"], manifest["capacity_reference"])) != item["resolved_capacity"]:
                raise ValueError("Artifact capacity mismatch")
        hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        if e["stage"] == "warmup":
            continue
        value, details = metrics(item)
        value.update(extra_metrics(item))
        resource = item["scheduler_resource_trace"]
        offset = resource["event_clock_origin_s"] - item["experiment_clock_origin_s"]
        rows.append({"experiment": e, "payload_fingerprint": item["payload_fingerprint"],
                     "capacity": item["resolved_capacity"], "metrics": value,
                     "memory_usage": item["server_initial"]["internal_states"][0]["memory_usage"],
                     "request_progress": details,
                     "resource_timeline": [{"time_s": offset + event["observed_s"],
                                            **{k: event[k] for k in ("state_slots", "active_requests", "kv_tokens")}}
                                           for event in resource["events"]]})
        samples.extend(item["gpu_health_samples"])
    if seen != expected or warmups != expected or len(rows) != manifest["planned_measured_runs"]:
        raise ValueError("Incomplete measured or warmup matrix")
    loaded = [g for sample in samples for g in sample["gpus"] if g["utilization"] >= 80]
    return {"manifest": manifest, "rows": rows, "artifact_hashes": hashes,
            "audit": {"measured_runs": len(rows), "measured_requests": len(rows) * COUNT,
                      "warmup_runs": len(warmups), "warmup_requests": len(warmups) * COUNT,
                      "gpu_sample_pairs": len(samples),
                      "loaded_sm_mhz_range": [min(g["sm_mhz"] for g in loaded), max(g["sm_mhz"] for g in loaded)] if loaded else None,
                      "max_temperature_c": max(g["temperature_c"] for sample in samples for g in sample["gpus"]),
                      "max_arrival_lateness_ms": max(r["metrics"]["max_arrival_lateness_ms"] for r in rows)}}


def analyze(calibration, mediation=None):
    parts = {"calibration": load_stage(calibration, "calibrate")}
    if mediation is not None:
        parts["mediation"] = load_stage(mediation, "mediate")
        if parts["mediation"]["manifest"]["source_hashes"] != parts["calibration"]["manifest"]["source_hashes"]:
            raise ValueError("Different measurement versions")
        if parts["mediation"]["manifest"]["capacity_reference"] != parts["calibration"]["manifest"]["capacity_reference"]:
            raise ValueError("Different inspection references")
    rows = [r for part in parts.values() for r in part["rows"]]
    payloads = defaultdict(set)
    for row in rows:
        payloads[row["experiment"]["rep"]].add(row["payload_fingerprint"])
    if any(len(values) != 1 for values in payloads.values()):
        raise ValueError("Unpaired requests across modes, caps or pools")
    validity = []
    for row in rows:
        e, m = row["experiment"], row["metrics"]
        if e["pool"] == "wide" and e["cap"] == 607:
            validity.append({"mode": e["mode"], "rep": e["rep"],
                             "peak_exceeds_old_cap": m["peak_active"] > 485,
                             "peak_active": m["peak_active"], "kv_peak_fraction": m["kv_peak_fraction"],
                             "no_token_blocks": m["no_token_blocks"]})
    return {"complete": mediation is not None, "stages": {name: {k: v for k, v in part.items() if k != "rows"} for name, part in parts.items()},
            "rows": rows, "contrasts": contrasts(rows) if mediation is not None else [],
            "intervention_validity": validity,
            "analysis_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                (Path(__file__), Path(__file__).with_name("analyze_m2_progress_probe.py"))},
            "scope": "Exploratory native batching/cap controls, not hybrid exclusivity, fairness or held-out SOTA superiority."}


def report(data):
    lines = ["# M2 fixed-workload state-cap mediation", "", f"Both stages complete: {data['complete']}; measured runs: {len(data['rows'])}", "",
             "Every measured run uses 607 P512/O128 requests. All configured cells are retained.", "",
             "| Pool | Mode | Cap | Rep | Mean JCT s | Queue s | Makespan s | State slot-s | Peak active | Peak KV % |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["rows"]:
        e, m = row["experiment"], row["metrics"]
        lines.append(f"| {e['pool']} | {e['mode']} | {e['cap']} | {e['rep']} | {m['all_mean_jct_s']:.3f} | "
                     f"{m['mean_queue_s']:.3f} | {m['makespan_s']:.3f} | {m['state_slot_seconds']:.3f} | "
                     f"{m['peak_active']} | {100*m['kv_peak_fraction']:.2f} |")
    if data["complete"]:
        lines += ["", "## Prespecified JCT Contrast", "",
                  "D = Mixed-4K minus Separate-4K. Delta = D(wide,485) minus D(wide,607). Positive Delta supports amplification by the tighter cap; negative means the opposite.", "",
                  "| Rep | D natural485, s | D wide485, s | D wide607, s | Delta, s |",
                  "|---|---:|---:|---:|---:|"]
        for row in data["contrasts"]:
            if row["metric"] == "all_mean_jct_s":
                lines.append(f"| {row['rep']} | {row['gap_natural_485']:.3f} | {row['gap_wide_485']:.3f} | "
                             f"{row['gap_wide_607']:.3f} | {row['delta_tight_minus_relaxed']:.3f} |")
    lines += ["", "Only two independent starts per configuration. No confirmatory CI or post-hoc winner claim.",
              "A/B pool comparisons are cross-session; B/C cap interventions share the same physical pool. No GPU-kernel attribution or hybrid exclusivity is implied."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--mediation", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    data = analyze(args.calibration, args.mediation)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "analysis.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    (args.output / "report.md").write_text(report(data))
    print(report(data))


if __name__ == "__main__":
    main()
