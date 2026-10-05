"""Audit admission traces without treating scheduler timestamps as GPU durations."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def window_metrics(item, last_release):
    at = item["experiment"]["arrival"]
    if at is None or last_release <= at:
        return {"valid_window": False}
    trace = item["scheduler_batch_trace"]
    if trace.get("dropped_events") or trace.get("dropped_admission_blocks"):
        raise ValueError("Incomplete batch trace")
    offset = trace["event_clock_origin_s"] - item["experiment_clock_origin_s"]
    groups = {r["rid"]: r["group"] for r in item["requests"]}
    candidate_batches, candidate_tokens, mixed_batches = 0, 0, 0
    holder_launches = []
    modes = Counter()
    for batch in trace["events"]:
        timestamp = offset + batch["launch_s"]
        if not at <= timestamp <= last_release:
            continue
        modes[batch["mode"]] += 1
        members = batch["members"]
        candidates = [m for m in members if groups[m["rid"]] == "candidate" and m["phase"] == "prefill"]
        holders = [m for m in members if groups[m["rid"]] == "holder" and m["phase"] == "decode"]
        if candidates:
            candidate_batches += 1
            candidate_tokens += sum(m["work_tokens"] for m in candidates)
        if holders:
            holder_launches.append(timestamp)
        mixed_batches += int(bool(candidates) and bool(holders))
    times = [at, *sorted(set(holder_launches)), last_release]
    blocks = Counter(b["reason"] for b in trace["admission_blocks"]
                     if at <= offset + b["observed_s"] <= last_release)
    return {
        "valid_window": True,
        "arrival_to_last_holder_release_s": last_release-at,
        "candidate_prefill_batches_before_last_holder_release": candidate_batches,
        "candidate_prefill_tokens_before_last_holder_release": candidate_tokens,
        "holder_decode_launches_before_last_holder_release": len(holder_launches),
        "candidate_prefill_with_holder_decode_batches": mixed_batches,
        "first_holder_decode_launch_after_arrival_s": min(holder_launches)-at if holder_launches else None,
        "max_interval_without_holder_decode_launch_s": max(b-a for a, b in zip(times, times[1:])),
        "batch_modes": dict(modes), "admission_blocks": dict(blocks),
    }


def health_metrics(item):
    samples = [g for sample in item["gpu_health_samples"] for g in sample["gpus"]]
    loaded = [g for g in samples if g["utilization"] >= 80]
    return {"sample_count": len(samples),
            "loaded_sm_mhz_range": [min(g["sm_mhz"] for g in loaded), max(g["sm_mhz"] for g in loaded)] if loaded else None,
            "max_temperature_c": max((g["temperature_c"] for g in samples), default=None),
            "max_memory_mib": max((g["memory_mib"] for g in samples), default=None)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    analysis = json.loads(args.analysis.read_text())
    if not analysis["complete"] and not args.allow_partial:
        parser.error("Incomplete analysis requires --allow-partial")
    rows = []
    for row in analysis["rows"]:
        path = Path(row["source"])
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != analysis["artifact_sha256"][str(path)]:
            raise ValueError(f"Changed raw artifact: {path}")
        item = json.loads(raw)
        last_release = item["experiment"]["arrival"] + row["metrics"]["mechanism"]["holder_last_release_after_arrival_s"]
        rows.append({"experiment": row["experiment"], "scene_errors": row["scene_errors"],
                     "execution": window_metrics(item, last_release), "health": health_metrics(item),
                     "source": str(path)})
    result = {"analysis": str(args.analysis), "analysis_sha256": hashlib.sha256(args.analysis.read_bytes()).hexdigest(),
              "complete": analysis["complete"], "rows": rows,
              "scope": "CPU-observed scheduler ordering, not GPU execution times or proof of state-capacity mediation. "
                       "The no-holder-launch interval includes the arrival and last-release boundary intervals."}
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"audit.json").write_text(json.dumps(result, indent=2, allow_nan=False)+"\n")
    print(json.dumps({"complete": result["complete"], "runs": len(rows), "output": str(args.output)}))


if __name__ == "__main__":
    main()
