#!/usr/bin/env python3
"""Merge PDAF affinity benchmark result files into affinity_bench_merged.json.

Combines per-QPS results across multiple runs (each deployment's qps_N entries
are unioned). The merged qps_list is recomputed from the union of all PASS keys.

Usage:
    python3 merge_affinity_results.py <result1.json> [<result2.json> ...]
    # Always merges into results/affinity_bench_merged.json
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MERGED = HERE / "results" / "affinity_bench_merged.json"


def main():
    if len(sys.argv) < 2:
        print("Usage: merge_affinity_results.py <result.json> [...]")
        sys.exit(1)

    if MERGED.exists():
        merged = json.load(open(MERGED))
    else:
        merged = {"meta": {}, "results": {}}

    for path in sys.argv[1:]:
        data = json.load(open(path))
        if not merged["meta"]:
            merged["meta"] = dict(data.get("meta", {}))
        for dep, dep_res in data.get("results", {}).items():
            if "__status__" in dep_res:
                continue
            tgt = merged["results"].setdefault(dep, {})
            for qkey, m in dep_res.items():
                if qkey.startswith("__"):
                    continue
                if isinstance(m, dict) and m.get("status") == "PASS":
                    tgt[qkey] = m

    all_qps = set()
    for dep_res in merged["results"].values():
        for qkey in dep_res:
            if qkey.startswith("qps_"):
                all_qps.add(int(qkey.replace("qps_", "")))
    merged["meta"]["qps_list"] = sorted(all_qps)

    with open(MERGED, "w") as f:
        json.dump(merged, f, indent=2)
    print(f"Merged into {MERGED}")
    print(f"qps_list: {merged['meta']['qps_list']}")
    for dep, dep_res in merged["results"].items():
        print(f"  {dep}: {sorted(dep_res.keys(), key=lambda k: int(k.replace('qps_','')))}")


if __name__ == "__main__":
    main()
