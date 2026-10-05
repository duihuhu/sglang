"""Prepare cache reductions from measured memory, then freeze a matched plan."""

import argparse
import json
import math
from pathlib import Path

from run_m2_batch_frontier import dump, hash_file
from run_m2_pool_comparison import ARMS, COUNT, MODES, source_hashes


TOKEN_BYTES = 8064
GIB = 1 << 30
ALLOC_TOLERANCE = 64 << 20
DEVICE_TOLERANCE = 256 << 20


def collect(directories):
    rows, evidence = {}, {}
    reference = None
    expected_hashes = source_hashes()
    for directory in directories:
        manifest_dir = directory if (directory / "manifest.json").exists() else directory.parent
        manifest = json.loads((manifest_dir / "manifest.json").read_text())
        if manifest["source_hashes"] != expected_hashes:
            raise ValueError("Inspection used different measurement sources")
        evidence[str(manifest_dir / "manifest.json")] = hash_file(manifest_dir / "manifest.json")
        sessions = [directory] if (directory / "session.json").exists() else sorted(directory.glob("session-*"))
        for session in sessions:
            artifacts = list(session.glob("m2pool-*-inspect.json"))
            if not artifacts:
                continue
            if len(artifacts) != 1:
                raise ValueError("Expected exactly one capacity diagnostic per session")
            path = artifacts[0]
            item = json.loads(path.read_text())
            arm = item["experiment"]["pool"]
            if arm in rows or item["integrity_errors"]:
                raise ValueError("Duplicate or invalid diagnostic")
            if arm.endswith("rebalanced") or arm == "unified-raised-cap":
                if item["scheduler_resource_trace"]["system"]["peak_active_requests"] != COUNT:
                    raise ValueError("Raised-cap configuration did not reach 607 concurrent requests")
            correctness_path = session / f"correctness-{arm}.json"
            correctness = json.loads(correctness_path.read_text())
            output_ids = [r["output_ids"] for r in correctness["records"]]
            if reference is None:
                reference = output_ids
            if output_ids != reference:
                raise ValueError("Cross-session/cross-pool output IDs differ")
            audit = item["server_initial"]["internal_states"][0]["m2_pool_audit"]
            suffix = Path(audit["path"]).name.split("-", 1)[1]
            memory_files = [session / "memory" / f"rank{rank}-{suffix}" for rank in (0, 1)]
            memory = [json.loads(p.read_text()) for p in memory_files]
            peaks = [json.loads(p.read_text())["memory"] for p in (session / "memory").glob("*.json")]
            config = json.loads((session / "session.json").read_text())["config"]
            rows[arm] = {"capacity": item["resolved_capacity"], "kv_tokens": item["resolved_capacity"][1],
                         "memory": [m["memory"] for m in memory], "config": config,
                         "peak_allocated": max(m["peak_allocated_since_previous_audit"] for m in peaks),
                         "peak_reserved": max(m["peak_reserved_since_previous_audit"] for m in peaks),
                         "log_allocation_retry_warning_count": (session / "server.log").read_text().count("memory allocation failed with OOM"),
                         "session": str(session)}
            for p in [path, correctness_path, *memory_files]:
                evidence[str(p)] = hash_file(p)
    if set(rows) != set(ARMS):
        raise ValueError(f"Incomplete pool inspection: {sorted(rows)}")
    return rows, evidence


def budget_plan(rows):
    base = rows["static-natural"]
    # Reserve an additional common 2 GiB after allocation retries in the .85 diagnostic.
    targets = {key: min(m[key] for m in base["memory"]) - 2 * GIB for key in ("allocated", "cuda_used")}
    sessions = []
    for arm in ARMS:
        row = rows[arm]
        reduction = max(0, *(m[key] - target for m in row["memory"] for key, target in targets.items()))
        tokens = row["kv_tokens"] - math.ceil(reduction / TOKEN_BYTES)
        if tokens < 2 * COUNT * (512 + 128):
            raise ValueError(f"Insufficient KV headroom after budget match: {arm}: {tokens}")
        sessions.append({"arm": arm, "mode": "mixed-4096", "reps": [22],
                         "kv_tokens": tokens, "reduction_bytes": reduction})
    return sessions, targets


def frozen_sessions(rows):
    allocated = [m["allocated"] for r in rows.values() for m in r["memory"]]
    used = [m["cuda_used"] for r in rows.values() for m in r["memory"]]
    if max(allocated) - min(allocated) > ALLOC_TOLERANCE:
        raise ValueError(f"Allocated memory mismatch {max(allocated) - min(allocated)} B")
    if max(used) - min(used) > DEVICE_TOLERANCE:
        raise ValueError(f"Device-used memory mismatch {max(used) - min(used)} B")
    natural_cap = rows["static-natural"]["capacity"][2]
    pairs = [(arm, mode) for arm in ("static-natural", "unified-raised-cap", "static-rebalanced", "unified-old-cap") for mode in MODES]
    result = []
    for rep in (31, 32):
        for arm, mode in pairs if rep == 31 else reversed(pairs):
            caps = [natural_cap, COUNT] if arm in ("static-rebalanced", "unified-raised-cap") else [natural_cap]
            if rep == 32:
                caps.reverse()
            result.append({"arm": arm, "mode": mode, "reps": [rep], "caps": caps,
                           "kv_tokens": rows[arm]["config"]["kv_tokens"]})
    return result


def refine_plan(rows):
    target = min(m["allocated"] for row in rows.values() for m in row["memory"]) + 3 * ALLOC_TOLERANCE // 4
    sessions = []
    for arm in ARMS:
        row = rows[arm]
        reduction = max(m["allocated"] for m in row["memory"]) - target
        if reduction <= 0:
            continue
        kv_tokens = row["kv_tokens"] - math.ceil(reduction / TOKEN_BYTES)
        if kv_tokens < 2 * COUNT * 640:
            raise ValueError("Refinement leaves insufficient KV headroom")
        sessions.append({"arm": arm, "mode": "mixed-4096", "reps": [23],
                         "kv_tokens": kv_tokens, "reduction_bytes": reduction})
    if not sessions:
        raise ValueError("No high-memory conditions need refinement")
    return sessions, {"allocated_ceiling": target}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("budget", "refine", "freeze"), required=True)
    parser.add_argument("--directories", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows, evidence = collect(args.directories)
    if args.mode == "freeze":
        sessions, targets = frozen_sessions(rows), None
    else:
        sessions, targets = (budget_plan if args.mode == "budget" else refine_plan)(rows)
    dump(args.out, {
        "source_hashes": source_hashes(), "sessions": sessions,
        "budget_targets": targets, "memory_evidence": rows, "evidence_sha256": evidence,
        "planner_sha256": hash_file(Path(__file__)),
        "allocated_tolerance_bytes": ALLOC_TOLERANCE, "device_used_tolerance_bytes": DEVICE_TOLERANCE,
        "allocation_retry_policy": "Recovered caching-allocator retries are retained and counted, not equivalent to failed requests; no allocator mode changes.",
        "planned_nonwarmup_runs": sum(len(s.get("caps", [0])) * len(s["reps"]) for s in sessions),
        "performance_interpretation": "Two fresh sessions per cell; exploratory paired estimates, not a precise CI or held-out claim.",
        "objective": "mean JCT; both p95 and makespan may regress by at most 5 percent to count as constrained benefit",
    })
    print(json.dumps({"sessions": sessions, "budget_targets": targets}, indent=2))


if __name__ == "__main__":
    main()
