#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import execute_item, expand_queue, run_id


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def valid(summary, run_dir):
    if summary.get("status") != "complete" or summary.get("requests_total") != 64 or summary.get("requests_success") != 64 or summary.get("requests_failed") != 0:
        return False, "request_summary"
    requests = [json.loads(line) for line in (run_dir / "requests.jsonl").read_text().splitlines() if line.strip()]
    if len(requests) != 64 or any(not row.get("success") or row.get("completion_tokens") != row.get("expected_completion_tokens") for row in requests):
        return False, "completion_semantics"
    link = json.loads((run_dir / "link_validation.json").read_text())
    if link.get("status") != "pass" or not link.get("physical_verified"):
        return False, "physical_link"
    energy = json.loads((run_dir / "energy.json").read_text()).get("normalized", {})
    if not isinstance(energy.get("total_j"), (int, float)) or energy["total_j"] <= 0:
        return False, "energy"
    return True, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", type=Path, default=ROOT / "configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json")
    parser.add_argument("--results", type=Path, default=ROOT / "results/raw/measurement_rdma_qps4_20260825")
    parser.add_argument("--repeat", type=int, choices=(1, 2, 3))
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    bundle = load_bundle(ROOT / "configs", args.matrix, None)
    queue = expand_queue(bundle)
    assert len({item["point"]["id"] for item in queue}) == 6
    assert len(queue) == 36
    assert all(item["qps"] == 4 and item["point"]["expected_link"] == "rdma" for item in queue)
    repeats = [args.repeat] if args.repeat else [1, 2, 3]
    progress_path = args.results / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"schema_version": 1, "created_at": now(), "expected_valid_runs": 108, "runs": []}
    existing = {(row["repeat"], row["point_id"], row["workload"]): row for row in progress["runs"]}
    for repeat in repeats:
        repeat_root = args.results / f"repeat{repeat}"
        for base in queue:
            key = (repeat, base["point"]["id"], base["workload"])
            row = existing.setdefault(key, {"repeat": repeat, "point_id": key[1], "workload": key[2], "status": "pending", "attempts": []})
            if row["status"] == "valid" or not args.execute:
                continue
            initial_attempt = len(row["attempts"]) + 1
            for attempt in range(initial_attempt, initial_attempt + args.max_attempts):
                item = dict(base)
                item["run_id"] = run_id(item["point"], item["workload"], item["qps"], repeat * 1000 + attempt)
                attempt_root = repeat_root / f"attempt{attempt}"
                summary = execute_item(item, bundle, attempt_root)
                run_dir = attempt_root / item["run_id"]
                ok, reason = valid(summary, run_dir)
                row["attempts"].append({"attempt": attempt, "run_id": item["run_id"], "artifact": str(run_dir), "status": summary.get("status"), "valid": ok, "reason": reason, "finished_at": now()})
                row["status"] = "valid" if ok else "retry_needed"
                progress["runs"] = list(existing.values())
                progress["updated_at"] = now()
                write(progress_path, progress)
                if ok:
                    break
            if row["status"] != "valid":
                raise RuntimeError(f"exhausted retries: {key}")
    progress["runs"] = list(existing.values())
    progress["updated_at"] = now()
    write(progress_path, progress)
    print(json.dumps({"points": 6, "runs_per_repeat": 36, "valid": sum(row["status"] == "valid" for row in progress["runs"]), "progress": str(progress_path)}, indent=2))


if __name__ == "__main__":
    main()
