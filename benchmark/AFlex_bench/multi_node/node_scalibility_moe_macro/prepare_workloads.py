#!/usr/bin/env python3
"""Prepare macro-benchmark workload files from Azure LLM Inference Trace CSVs.

Each CSV has columns: TIMESTAMP, ContextTokens, GeneratedTokens.
We sample N requests and assign arrival times based on target QPS (Poisson).
Output: JSONL files compatible with run_macro_benchmark.py.

Usage:
    python3 prepare_workloads.py --dataset conv --qps 1,2,3,4,5,6 --n-requests 200
    python3 prepare_workloads.py --dataset code --qps 1,2,3,4,5,6 --n-requests 200
"""
import argparse
import csv
import json
import os
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET_DIR = HERE / "dataset"
WORKLOAD_DIR = HERE / "workloads"

DATASETS = {
    "conv": DATASET_DIR / "AzureLLMInferenceTrace_conv_1week.csv",
    "code": DATASET_DIR / "AzureLLMInferenceTrace_code_1week.csv",
}

MAX_INPUT_LEN = 4096
MAX_OUTPUT_LEN = 1024


def load_and_sample(csv_path, n_requests, seed=42):
    """Load CSV, filter extremes, sample n_requests."""
    rows = []
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i >= 500000:
                break
            inp = int(row["ContextTokens"])
            out = int(row["GeneratedTokens"])
            if inp < 1 or out < 1:
                continue
            inp = min(inp, MAX_INPUT_LEN)
            out = min(out, MAX_OUTPUT_LEN)
            rows.append((inp, out))
    random.seed(seed)
    if len(rows) > n_requests:
        rows = random.sample(rows, n_requests)
    random.shuffle(rows)
    return rows


def make_workload(rows, qps):
    """Assign Poisson arrival times at target QPS."""
    reqs = []
    t = 0.0
    for inp, out in rows:
        t += random.expovariate(qps)
        reqs.append({
            "input_len": inp,
            "output_len": out,
            "arrival_time_s": round(t, 4),
        })
    return reqs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="all", help="conv, code, or all")
    parser.add_argument("--qps", default="1,2,3,4,5,6",
                        help="Comma-sep QPS values")
    parser.add_argument("--n-requests", type=int, default=200,
                        help="Number of requests per workload")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    datasets = list(DATASETS.keys()) if args.dataset == "all" else [args.dataset]
    qps_list = [int(q) for q in args.qps.split(",")]

    for ds_name in datasets:
        csv_path = DATASETS[ds_name]
        if not csv_path.exists():
            print(f"WARNING: {csv_path} not found, skipping")
            continue
        print(f"Loading {ds_name} from {csv_path}...")
        rows = load_and_sample(csv_path, args.n_requests, args.seed)
        print(f"  Sampled {len(rows)} requests "
              f"(avg input={sum(r[0] for r in rows)/len(rows):.0f}, "
              f"avg output={sum(r[1] for r in rows)/len(rows):.0f})")

        for qps in qps_list:
            random.seed(args.seed + qps)
            reqs = make_workload(rows, qps)
            out_file = WORKLOAD_DIR / f"macro_{ds_name}_qps{qps}.jsonl"
            with open(out_file, "w") as f:
                for r in reqs:
                    f.write(json.dumps(r) + "\n")
            duration = reqs[-1]["arrival_time_s"] if reqs else 0
            print(f"  {out_file.name}: {len(reqs)} reqs, "
                  f"duration={duration:.1f}s at QPS={qps}")

    print("Done.")


if __name__ == "__main__":
    main()
