#!/usr/bin/env python3
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import completed_ids, execute_item, expand_queue, filter_queue_by_point_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--cluster", type=Path, help="override the matrix cluster config")
    parser.add_argument("--results", default=ROOT / "results", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--priority", type=int)
    parser.add_argument("--point-id", action="append", default=[], help="point ID; repeat or comma-separate")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--allow-experimental", action="store_true",
        help="explicitly unlock experimental points that require preflight",
    )
    args = parser.parse_args()

    bundle = load_bundle(ROOT / "configs", args.matrix, args.cluster)
    queue = expand_queue(
        bundle, args.inventory, args.priority, args.smoke, args.allow_experimental
    )
    try:
        queue = filter_queue_by_point_ids(queue, args.point_id)
    except ValueError as exc:
        parser.error(str(exc))
    done = completed_ids(args.results)
    print(json.dumps({
        "points": len({item["point"]["id"] for item in queue}),
        "runs": len(queue),
        "ready": sum(item["state"] == "ready" for item in queue),
        "blocked": sum(item["state"] != "ready" for item in queue),
        "requires_preflight": sum(
            item["state"] == "requires_preflight" for item in queue
        ),
        "completed": len(done),
    }, indent=2))
    for item in queue:
        if args.inventory or args.dry_run or not args.execute:
            print(json.dumps({key: item[key] for key in ("run_id", "workload", "qps", "state")} | {"point_id": item["point"]["id"]}))
        elif item["run_id"] not in done:
            print(json.dumps(execute_item(item, bundle, args.results)))


if __name__ == "__main__":
    main()
