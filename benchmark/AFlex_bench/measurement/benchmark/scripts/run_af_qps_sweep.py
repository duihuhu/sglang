#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.config import load_bundle
from aflex_benchmark.deploy import build_plan
from aflex_benchmark.sweep import run_af_qps_sweep


def main():
    parser = argparse.ArgumentParser(description="Persistent AF QPS 1..16 sweep")
    parser.add_argument("--matrix", type=Path,
                        default=ROOT / "configs/sweep_af_qps1_16_node34.json")
    parser.add_argument("--cluster", type=Path)
    parser.add_argument("--results", type=Path,
                        default=ROOT / "results/sweep/run")
    parser.add_argument("--start-qps", type=int, default=1)
    parser.add_argument("--end-qps", type=int, default=16)
    parser.add_argument("--continue-on-partial", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="deploy once and execute the sweep")
    mode.add_argument("--plan", action="store_true",
                      help="compile and print the plan only (default)")
    args = parser.parse_args()
    if not (1 <= args.start_qps <= args.end_qps <= 16):
        parser.error("QPS range must satisfy 1 <= start <= end <= 16")
    bundle = load_bundle(ROOT / "configs", args.matrix, args.cluster)
    point = bundle["matrix"]["points"][0]
    selected = [qps for qps in point["qps"]
                if args.start_qps <= qps <= args.end_qps]
    plan = build_plan(bundle["cluster"],
                      bundle["models"]["models"][point["model"]]["path"], point,
                      run_tag="dryrun")
    plan.validate()
    inventory = {"mode": "execute" if args.execute else "plan", "point_id": point["id"],
                 "selected_qps": selected, "requests_per_qps": point["requests_per_qps"],
                 "replicas": len(plan.endpoints), "processes": len(plan.processes),
                 "endpoints": plan.endpoints, "results": str(args.results)}
    print(json.dumps(inventory, indent=2))
    if args.execute:
        result = run_af_qps_sweep(bundle, args.results, args.start_qps,
                                  args.end_qps, args.continue_on_partial)
        print(json.dumps(result, indent=2))
        return 0 if result["status"] == "complete" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
