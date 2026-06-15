#!/usr/bin/env python3
"""Generate controlled synthetic workloads to locate the energy crossover
between heterogeneous prefill-heavy PDAF (2PA4PF, P6+D2) and symmetric PDAF
(pdaf_8g_dyn, P4+D4).

Each workload = N requests with fixed (input_len, output_len) arriving at a
fixed Poisson-like rate (QPS). Controlling IL/OL/QPS independently lets us
isolate prefill-dominance vs decode-concurrency effects.
"""
import argparse
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent

def gen(il, ol, qps, n, seed=0):
    random.seed(seed)
    t = 0.0
    rows = []
    for _ in range(n):
        # exponential inter-arrival for target QPS
        t += random.expovariate(qps) if qps > 0 else 0.0
        rows.append({"input_len": il, "output_len": ol,
                     "arrival_time_s": round(t, 4)})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--specs", required=True,
                    help="semicolon list of name:il:ol:qps:n")
    args = ap.parse_args()
    for spec in args.specs.split(";"):
        if not spec.strip():
            continue
        name, il, ol, qps, n = spec.split(":")
        rows = gen(int(il), int(ol), float(qps), int(n))
        fp = OUT / f"workload_{name}.jsonl"
        with open(fp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        dur = rows[-1]["arrival_time_s"]
        print(f"{fp.name}: n={n} il={il} ol={ol} qps={qps} span={dur:.0f}s")

if __name__ == "__main__":
    main()
