#!/usr/bin/env python3
"""Post-hoc launch-gap audit; never filters runs or attributes GPU kernel time."""

import argparse
import hashlib
import json
from pathlib import Path
import statistics


def launch_gaps(item):
    trace = item["scheduler_batch_trace"]
    events = trace["events"]
    origin = trace["event_clock_origin_s"]
    result = []
    for previous, current in zip(events, events[1:]):
        duration = current["launch_s"] - previous["launch_s"]
        if duration < 0:
            raise ValueError("Nonmonotonic launch times")
        if duration <= 1:
            continue
        samples = [s for s in item["gpu_health_samples"]
                   if origin + previous["launch_s"] <= s["observed_s"] < origin + current["launch_s"]]
        result.append({"duration_s": duration, "previous_mode": previous["mode"],
                       "next_mode": current["mode"], "previous_batch_size": previous["batch_size"],
                       "record_to_next_launch_s": current["launch_s"] - previous["recorded_s"],
                       "gpu_sample_pairs": len(samples),
                       "mean_utilization_during_gap": statistics.fmean(g["utilization"] for s in samples for g in s["gpus"]) if samples else None})
    return {"max_launch_interval_s": max((b["launch_s"] - a["launch_s"] for a, b in zip(events, events[1:])), default=0),
            "intervals_over_1s": sorted(result, key=lambda r: r["duration_s"], reverse=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.analysis.read_text())
    rows = []
    for stage in data["stages"].values():
        for filename, expected in stage["artifact_hashes"].items():
            if filename.endswith("-warmup.json"):
                continue
            raw = Path(filename).read_bytes()
            if hashlib.sha256(raw).hexdigest() != expected:
                raise ValueError("Measured artifact changed")
            item = json.loads(raw)
            rows.append({"experiment": item["experiment"], **launch_gaps(item)})
    result = {"scope": "Post-hoc timing diagnostic. Threshold is descriptive, not an exclusion rule. No root-cause or kernel-idleness attribution.",
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "analysis_sha256": hashlib.sha256(args.analysis.read_bytes()).hexdigest(), "rows": rows}
    with args.output.open("x") as output:
        json.dump(result, output, indent=2, allow_nan=False)
        output.write("\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
