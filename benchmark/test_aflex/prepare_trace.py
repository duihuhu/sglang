#!/usr/bin/env python3
"""Azure CSV to macro workload using original sample/clip/Poisson semantics."""
import argparse
import csv
import json
import random
from pathlib import Path


def convert(source, n_requests=200, seed=42, qps=2):
    if n_requests <= 0 or qps <= 0:
        raise ValueError("n_requests and qps must be positive")
    rows = []
    with open(source) as f:
        for i, row in enumerate(csv.DictReader(f)):
            if i >= 500000:
                break
            inp, out = int(row["ContextTokens"]), int(row["GeneratedTokens"])
            if inp >= 1 and out >= 1:
                rows.append((min(inp, 4096), min(out, 1024)))
    rng = random.Random(seed)
    if len(rows) > n_requests:
        rows = rng.sample(rows, n_requests)
    rng.shuffle(rows)
    rng = random.Random(seed + qps)
    result, t = [], 0.0
    for inp, out in rows:
        t += rng.expovariate(qps)
        result.append(dict(input_len=inp, output_len=out, arrival_time_s=round(t, 4)))
    if not result:
        raise ValueError("no valid trace rows")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--qps", type=int, required=True)
    p.add_argument("--n-requests", type=int, default=200)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    rows = convert(args.input, args.n_requests, args.seed, args.qps)
    with Path(args.output).open("x") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
