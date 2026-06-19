#!/usr/bin/env python3
"""Generate fixed-length workloads for micro-benchmark.

4 scenarios:
  Chatbot:  IL=128,  OL=1024
  QA:       IL=512,  OL=256
  RAG:      IL=2048, OL=64
  Summary:  IL=4096, OL=64

Each scenario generates multiple QPS variants.
"""
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "workloads"
OUT.mkdir(parents=True, exist_ok=True)

SCENARIOS = {
    "chatbot":  {"il": 128,  "ol": 1024},
    "qa":       {"il": 512,  "ol": 256},
    "rag":      {"il": 2048, "ol": 64},
    "summary":  {"il": 4096, "ol": 64},
}

# 4GPU: QPS = 1, 2, 3
# 8GPU: QPS = 1, 2, 3, 4, 5, 6
QPS_4GPU = [1, 2, 3]
QPS_8GPU = [1, 2, 3, 4, 5, 6]

N_REQUESTS = 200
SEED = 42


def gen_workload(il, ol, qps, n, seed):
    random.seed(seed)
    t = 0.0
    rows = []
    for _ in range(n):
        t += random.expovariate(qps) if qps > 0 else 0.0
        rows.append({"input_len": il, "output_len": ol,
                     "arrival_time_s": round(t, 4)})
    return rows


def main():
    all_qps = sorted(set(QPS_4GPU + QPS_8GPU))
    generated = []

    for name, cfg in SCENARIOS.items():
        for qps in all_qps:
            rows = gen_workload(cfg["il"], cfg["ol"], qps, N_REQUESTS, SEED)
            fname = f"micro_{name}_qps{qps}.jsonl"
            fp = OUT / fname
            with open(fp, "w") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            dur = rows[-1]["arrival_time_s"]
            generated.append((fname, name, cfg["il"], cfg["ol"], qps, N_REQUESTS, dur))
            print(f"  {fname}: il={cfg['il']} ol={cfg['ol']} qps={qps} n={N_REQUESTS} span={dur:.0f}s")

    # Summary
    print(f"\nGenerated {len(generated)} workload files in {OUT}")
    print("\nScenario summary:")
    print(f"  {'Scenario':<10} {'IL':>5} {'OL':>5} {'QPS variants'}")
    for name, cfg in SCENARIOS.items():
        print(f"  {name:<10} {cfg['il']:>5} {cfg['ol']:>5} {all_qps}")


if __name__ == "__main__":
    main()
