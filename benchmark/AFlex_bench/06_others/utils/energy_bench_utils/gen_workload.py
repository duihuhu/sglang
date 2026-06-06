#!/usr/bin/env python3
"""Generate varying workload datasets for Tier2 DVFS evaluation.

Creates request traces with distinct phases to demonstrate DVFS frequency
adaptation under changing load conditions:
  Phase 1: Light load (low QPS, short sequences) → DVFS should lower freq
  Phase 2: Heavy load (high QPS, long sequences) → DVFS should raise freq
  Phase 3: Medium load (moderate QPS) → DVFS finds middle ground
  Phase 4: Bursty load (sudden spike then drop) → tests SLO_URGENT trigger

Output: JSONL file with {prompt, max_tokens, arrival_time_s} per request.
"""

import argparse
import json
import random
import sys
from pathlib import Path


def generate_phase_requests(
    phase_name: str,
    duration_s: float,
    qps: float,
    input_len_range: tuple,
    output_len_range: tuple,
    start_time: float = 0.0,
) -> list:
    """Generate requests for a single workload phase."""
    requests = []
    n_requests = int(duration_s * qps)
    for i in range(n_requests):
        arrival = start_time + i / qps
        il = random.randint(*input_len_range)
        ol = random.randint(*output_len_range)
        requests.append({
            "phase": phase_name,
            "arrival_time_s": round(arrival, 3),
            "input_len": il,
            "output_len": ol,
        })
    return requests


def generate_varying_workload(seed: int = 42) -> list:
    """Generate a multi-phase workload trace."""
    random.seed(seed)
    all_requests = []
    t = 0.0

    # Phase 1: Light load — low QPS, short sequences
    # DVFS should detect slack and lower frequency to save energy
    phase1 = generate_phase_requests(
        "light", duration_s=30, qps=1.0,
        input_len_range=(64, 256), output_len_range=(32, 64),
        start_time=t,
    )
    all_requests.extend(phase1)
    t += 30

    # Phase 2: Heavy load — high QPS, long sequences
    # DVFS should raise frequency to meet SLO
    phase2 = generate_phase_requests(
        "heavy", duration_s=30, qps=4.0,
        input_len_range=(256, 1024), output_len_range=(64, 128),
        start_time=t,
    )
    all_requests.extend(phase2)
    t += 30

    # Phase 3: Medium load — moderate QPS
    # DVFS should find energy-optimal frequency pair
    phase3 = generate_phase_requests(
        "medium", duration_s=30, qps=2.0,
        input_len_range=(128, 512), output_len_range=(32, 64),
        start_time=t,
    )
    all_requests.extend(phase3)
    t += 30

    # Phase 4: Bursty — sudden spike then quiet
    # Tests SLO_URGENT trigger and recovery
    burst = generate_phase_requests(
        "burst_peak", duration_s=10, qps=6.0,
        input_len_range=(256, 512), output_len_range=(64, 128),
        start_time=t,
    )
    all_requests.extend(burst)
    t += 10

    quiet = generate_phase_requests(
        "burst_quiet", duration_s=20, qps=0.5,
        input_len_range=(64, 128), output_len_range=(32, 64),
        start_time=t,
    )
    all_requests.extend(quiet)
    t += 20

    return all_requests


def generate_steady_workload(qps: float = 5.0, duration_s: float = 120,
                             seed: int = 42) -> list:
    """Generate a steady-state workload for baseline comparison."""
    random.seed(seed)
    return generate_phase_requests(
        "steady", duration_s=duration_s, qps=qps,
        input_len_range=(256, 1024), output_len_range=(64, 256),
        start_time=0.0,
    )


def main():
    parser = argparse.ArgumentParser(description="Generate workload traces for Tier2 DVFS eval")
    parser.add_argument("--output-dir", type=str, default="workloads",
                        help="Output directory for workload files")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["varying", "steady", "all"], default="all",
                        help="Which workload to generate")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ("varying", "all"):
        reqs = generate_varying_workload(seed=args.seed)
        path = out_dir / "workload_varying.jsonl"
        with open(path, "w") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
        print(f"[varying] {len(reqs)} requests, duration ~{reqs[-1]['arrival_time_s']:.0f}s → {path}")

        # Print phase summary
        from collections import Counter
        phases = Counter(r["phase"] for r in reqs)
        for phase, count in phases.items():
            subset = [r for r in reqs if r["phase"] == phase]
            avg_il = sum(r["input_len"] for r in subset) / len(subset)
            avg_ol = sum(r["output_len"] for r in subset) / len(subset)
            print(f"  {phase:15s}: {count:4d} reqs, avg_il={avg_il:.0f}, avg_ol={avg_ol:.0f}")

    if args.mode in ("steady", "all"):
        reqs = generate_steady_workload(seed=args.seed)
        path = out_dir / "workload_steady.jsonl"
        with open(path, "w") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
        print(f"[steady]  {len(reqs)} requests, duration ~{reqs[-1]['arrival_time_s']:.0f}s → {path}")


if __name__ == "__main__":
    main()
