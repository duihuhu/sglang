#!/usr/bin/env python3
"""Generate FIXED-LENGTH workloads where only QPS (arrival rate) varies.

Unlike gen_workload.py (which produces varying input/output lengths and
multi-phase traces), this generator fixes input_len and output_len for every
request so that the ONLY independent variable across datasets is the request
arrival rate (QPS).  This isolates the effect of load intensity on
TTFT / TPOT / throughput / energy / SLO-violation for the 3-mode comparison
(Tier1-freq-only vs max-freq vs auto-freq).

Arrival process: deterministic, evenly spaced (1/qps seconds apart). This keeps
the load perfectly steady so energy/latency differences come from the control
policy, not from arrival jitter.

Output: JSONL with {phase, arrival_time_s, input_len, output_len} per request.
"""

import argparse
import json
from pathlib import Path


def generate_fixed_workload(
    qps: float,
    duration_s: float,
    input_len: int,
    output_len: int,
) -> list:
    """Generate a steady fixed-length trace at a given QPS."""
    requests = []
    n_requests = int(round(duration_s * qps))
    interval = 1.0 / qps
    for i in range(n_requests):
        arrival = i * interval
        requests.append({
            "phase": f"qps{qps:g}",
            "arrival_time_s": round(arrival, 4),
            "input_len": input_len,
            "output_len": output_len,
        })
    return requests


def main():
    parser = argparse.ArgumentParser(
        description="Generate fixed-length, variable-QPS workloads")
    parser.add_argument("--output-dir", type=str, default="workloads",
                        help="Output directory for workload files")
    parser.add_argument("--qps", type=str, default="1,2,4,6,8",
                        help="Comma-separated QPS levels to generate")
    parser.add_argument("--duration-s", type=float, default=60.0,
                        help="Duration of each workload (seconds)")
    parser.add_argument("--input-len", type=int, default=512,
                        help="Fixed input length (tokens) for every request")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Fixed output length (tokens) for every request")
    parser.add_argument("--prefix", type=str, default="fixed",
                        help="Output filename prefix")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qps_levels = [float(x) for x in args.qps.split(",") if x.strip()]
    for qps in qps_levels:
        reqs = generate_fixed_workload(
            qps=qps, duration_s=args.duration_s,
            input_len=args.input_len, output_len=args.output_len,
        )
        # Filename encodes the fixed lengths and qps for traceability.
        qps_tag = f"{qps:g}".replace(".", "p")
        path = out_dir / (
            f"{args.prefix}_il{args.input_len}_ol{args.output_len}_qps{qps_tag}.jsonl"
        )
        with open(path, "w") as f:
            for r in reqs:
                f.write(json.dumps(r) + "\n")
        print(f"[qps={qps:g}] {len(reqs)} reqs, il={args.input_len}, "
              f"ol={args.output_len}, dur~{args.duration_s:.0f}s -> {path}")


if __name__ == "__main__":
    main()
