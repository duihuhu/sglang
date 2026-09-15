#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.semantics import (
    build_semantic_plans,
    execute_semantic_validation,
    port_conflict_snapshot,
)


def main():
    parser = argparse.ArgumentParser(description="Validate Native versus AF generation semantics")
    parser.add_argument("--execute", action="store_true", help="deploy and issue GPU-backed requests")
    parser.add_argument("--results", type=Path, default=ROOT / "results/semantic_validation")
    parser.add_argument("--system", choices=("baseline", "af", "both"), default="both")
    parser.add_argument("--reuse-baseline", type=Path,
                        help="prior semantic_results.json or its results directory (AF only)")
    parser.add_argument("--run-tag", help="unique deployment tag; defaults to a timestamped tag")
    args = parser.parse_args()
    if args.reuse_baseline is not None and args.system != "af":
        parser.error("--reuse-baseline requires --system af")
    selected = ("baseline", "af") if args.system == "both" else (args.system,)
    if not args.execute:
        bundle, plans = build_semantic_plans(ROOT, selected, run_tag=args.run_tag)
        print(json.dumps({"mode": "plan", "no_side_effects": True, "model": bundle["model"],
                          "execution_order": list(selected),
                          "plans": {name: plan.manifest() for name, plan in plans.items()},
                          "port_conflict_snapshots": {
                              name: port_conflict_snapshot(plan) for name, plan in plans.items()
                          }}, indent=2))
        return 0
    try:
        summary = execute_semantic_validation(
            ROOT, args.results, selected, reuse_baseline=args.reuse_baseline,
            run_tag=args.run_tag,
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0 if all(item["all_api_pass"] and item["all_semantic_pass"]
                    for item in summary["systems"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
