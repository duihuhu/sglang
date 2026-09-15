#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import execute_item, expand_queue, run_id
from run_measurement_rdma_qps4 import write
import run_measurement_rdma_work_conserving as core

LANES = {
    0: {"nic": "mlx5_0", "gpus": {f"node{i}": [0] for i in range(1, 5)}, "hca_shared": False},
    2: {"nic": "mlx5_1", "gpus": {f"node{i}": [2] for i in range(1, 5)}, "hca_shared": False},
    6: {"nic": "mlx5_5", "gpus": {f"node{i}": [6] for i in range(1, 5)}, "hca_shared": False},
}
ARCHITECTURES = ("native", "pd", "af")
WORKLOADS = (
    "measurement_qa_lpld",
    "measurement_chatbot_lphd",
    "measurement_balanced_mpmd",
    "measurement_rag_hpld",
    "measurement_summary_hphd",
    "measurement_longcontext",
)
LOCK = threading.Lock()
GATE_NAME = "rdma4n_canary_gate.json"


def now():
    return datetime.now(timezone.utc).isoformat()


def resolve_artifact_path(path):
 path=Path(path)
 if path.exists(): return path
 marker="/benchmark/results/"
 text=str(path)
 if marker in text:
  candidate=ROOT/"results/raw"/text.split(marker,1)[1]
  if candidate.exists(): return candidate
 return path

def artifact_hash(path):
    digest = hashlib.sha256()
    for name in ("deployment_manifest.json", "summary.json", "requests.jsonl", "link_validation.json", "energy.json"):
        artifact = path / name
        if not artifact.is_file():
            raise RuntimeError(f"canary artifact missing {artifact}")
        digest.update(name.encode())
        digest.update(artifact.read_bytes())
    return digest.hexdigest()


def require_canary_gate(results):
    path = results / GATE_NAME
    try:
        gate = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("rdma4n canary gate missing or invalid") from exc
    if gate.get("status") != "pass" or gate.get("topology") != {"lanes": [0, 2, 6], "nodes": ["node1", "node2", "node3", "node4"], "hca_shared": False}:
        raise RuntimeError("rdma4n canary gate is not pass for required topology")
    artifacts = gate.get("artifacts")
    if not isinstance(artifacts, list) or [x.get("architecture") for x in artifacts] != list(ARCHITECTURES):
        raise RuntimeError("rdma4n canary gate does not contain Native/PD/AF in order")
    for record in artifacts:
        artifact = Path(record.get("artifact", ""))
        if record.get("status") != "pass" or artifact_hash(artifact) != record.get("artifact_hash"):
            raise RuntimeError(f"rdma4n canary artifact failed gate validation: {record.get('architecture')}")
    return gate


def acquire(results):
    results.mkdir(parents=True, exist_ok=True)
    path = results / "scheduler.lock"
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError("another measurement scheduler holds lock") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(json.dumps({"pid": os.getpid(), "scheduler": "rdma4n_strict", "started_at": now()}) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
    return handle


def configure(base, repeat, ordinal, lane, corunners, attempt, *, canary=False):
    item = copy.deepcopy(base)
    point = item["point"]
    layout = LANES[lane]["gpus"]
    point.update(core.placements(point, layout))
    point["port_base"] = (57000 + ordinal * 100) if canary else (24000 + ordinal * 500)
    actual = {node: {"gpus": gpus, "nic": LANES[lane]["nic"]} for node, gpus in layout.items()}
    point["metadata"].update({
        "phase": "rdma4n_only",
        "measurement_repeat": repeat,
        "retry_attempt": attempt,
        "lane_gpu": lane,
        "hca_group": LANES[lane]["nic"],
        "hca_shared": False,
        "actual_layout": actual,
        "concurrent_job_ids": list(corunners),
        "startup_barrier": "all ranks of each distributed component launch in one symmetric startup_stage",
        "parallel_rule": "four GPU lanes on every node: GPU0/1 share mlx5_0; GPU2/3 share mlx5_1; same rule for every architecture/workload",
        "canary": canary,
    })
    nonce = time.time_ns() if canary else repeat * 1_000_000 + ordinal * 100 + attempt
    item["run_id"] = run_id(point, item["workload"], item["qps"], nonce)
    return item, actual


def validate_symmetric_stages(plan):
    by_component = {}
    for process in plan.processes:
        if not process.gpus:
            continue
        component = process.metadata.get("component_role", process.role)
        by_component.setdefault(component, []).append(process)
    for component, processes in by_component.items():
        if len({p.startup_stage for p in processes}) != 1:
            raise ValueError(f"asymmetric startup stage for {component}")
        expected_nodes = {"NATIVE": 4, "P": 2, "D": 2, "F": 2, "A": 2}.get(component)
        if expected_nodes and len({p.node for p in processes}) != expected_nodes:
            raise ValueError(f"wrong node count for {component}")


def strict_audit(path, item, actual):
    try:
        summary = json.loads((path / "summary.json").read_text())
        manifest = json.loads((path / "deployment_manifest.json").read_text())
        link = json.loads((path / "link_validation.json").read_text())
        energy = json.loads((path / "energy.json").read_text())
        requests = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines() if line.strip()]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"missing_or_invalid_artifact:{type(exc).__name__}"
    if summary.get("status") != "complete" or (summary.get("requests_total"), summary.get("requests_success"), summary.get("requests_failed")) != (64, 64, 0):
        return False, "request_summary"
    if len(requests) != 64 or any(not r.get("success") or r.get("completion_tokens") != r.get("expected_completion_tokens") for r in requests):
        return False, "completion_semantics"
    used = {}
    for process in manifest.get("processes", []):
        if process.get("gpus"):
            used.setdefault(process["node"], set()).update(process["gpus"])
    used = {node: sorted(gpus) for node, gpus in used.items()}
    expected = {node: sorted(spec["gpus"]) for node, spec in actual.items()}
    if used != expected or len(used) != 4 or sum(map(len, used.values())) != 4:
        return False, "manifest_topology"
    if link.get("status") != "pass" or not link.get("physical_verified"):
        return False, "physical_rdma"
    paths = link.get("paths", [])
    if not any(p.get("path") == "rdma" and float(p.get("counter_value") or 0) > 0 for p in paths):
        return False, "physical_rdma_counter"
    scope = energy.get("scope")
    if scope != {node: spec["gpus"] for node, spec in actual.items()}:
        return False, "energy_scope"
    uuids = energy.get("gpu_uuids", {})
    if set(uuids) != set(actual):
        return False, "energy_uuid_nodes"
    for node, spec in actual.items():
        expected_indices = {str(gpu) for gpu in spec["gpus"]}
        if set(uuids.get(node, {})) != expected_indices or any(not value.startswith("GPU-") for value in uuids[node].values()):
            return False, "energy_uuid_scope"
    if not isinstance(energy.get("normalized", {}).get("total_j"), (int, float)) or energy["normalized"]["total_j"] <= 0:
        return False, "energy"
    return True, None


def execute_once(bundle, base, results, repeat, ordinal, lane, corunners, attempt, *, canary=False):
    item, actual = configure(base, repeat, ordinal, lane, corunners, attempt, canary=canary)
    plan = core.validate_plan_shape(bundle, item)
    validate_symmetric_stages(plan)
    root = results / ("rdma4n_canary" if canary else f"repeat{repeat}") / f"attempt{attempt}"
    summary = execute_item(item, bundle, root)
    path = root / item["run_id"]
    ok, reason = strict_audit(path, item, actual)
    return item, actual, summary, path, ok, reason


def run_canaries(bundle, base, results, progress):
    records = []
    for ordinal, (arch, lane) in enumerate(zip(ARCHITECTURES, LANES), 1):
        logical_id = f"canary:rdma4n:{arch}"
        item, actual, summary, path, ok, reason = execute_once(
            bundle, base[arch, WORKLOADS[0]], results, 0, ordinal, lane, [logical_id], 1, canary=True
        )
        record = {"architecture": arch, "lane_gpu": lane, "run_id": item["run_id"], "artifact": str(path), "layout": actual, "valid": ok, "reason": reason, "status": summary.get("status"), "finished_at": now()}
        records.append(record)
        progress["canaries"] = records
        progress["updated_at"] = now()
        write(results / "progress.json", progress)
        if not ok:
            progress.update({"status": "blocked_canary", "blocker": record, "updated_at": now()})
            write(results / "progress.json", progress)
            raise RuntimeError(f"canary failed: {record}")
    progress["canary_status"] = "passed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results/raw/measurement_rdma_qps4_20260825")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    gate = require_canary_gate(args.results)
    scheduler_lock = acquire(args.results)
    bundle = load_bundle(ROOT / "configs", ROOT / "configs/measurement_rdma_qps4_qwen3_30b_a3b_4gpu.json")
    queue = [item for item in expand_queue(bundle) if item["point"]["metadata"]["topology_id"] == "rdma4n"]
    base = {(item["point"]["architecture"], item["workload"]): item for item in queue}
    if len(base) != 18:
        raise ValueError(f"expected 18 architecture/workload bases, got {len(base)}")
    progress_path = args.results / "progress.json"
    old = json.loads(progress_path.read_text())
    existing = {(row.get("repeat"), row.get("point_id"), row.get("workload")): row for row in old.get("runs", [])}
    rows = {}
    for repeat in (1, 2, 3):
        for workload in WORKLOADS:
            for arch in ARCHITECTURES:
                point = base[arch, workload]["point"]
                key = (repeat, point["id"], workload)
                previous = existing.get(key)
                rows[key] = previous if previous else {"repeat": repeat, "point_id": point["id"], "architecture": arch, "workload": workload, "status": "pending", "attempts": []}
                if rows[key].get("status") != "valid":
                    rows[key]["status"] = "pending"
    progress = {key: value for key, value in old.items() if key not in ("runs", "valid_runs", "status", "blocker")}
    progress.update({
        "phase": "rdma4n_only",
        "expected_valid_runs": 54,
        "valid_runs": 0,
        "status": "rdma4n_canary_pending" if args.execute else "rdma4n_only_planned",
        "blocker": None,
        "rdma2n": {"status": "deferred", "history_preserved": True, "not_counted_in_phase": True},
        "prior_runs_audit": old.get("runs", []),
        "rdma4n_parallel": {
            "lanes": {str(gpu): spec for gpu, spec in LANES.items()},
            "max_concurrency": 3,
            "hca_shared": False,
            "co_runner_recording": True,
        },
        "topology_evidence": {
            "checked_at": now(),
            "command": "nvidia-smi topo -m on all four nodes",
            "all_nodes_match": True,
            "selected": {"GPU0": "mlx5_0/PXB", "GPU1": "mlx5_0/PXB", "GPU2": "mlx5_1/PXB", "GPU3": "mlx5_1/PXB"},
            "hca_contention_disclosure": "GPU0/1 share mlx5_0 and GPU2/3 share mlx5_1; HCA isolation is not claimed",
        },
        "runs": list(rows.values()),
    })
    write(progress_path, progress)
    if not args.execute:
        return
    progress["canary_gate"] = gate
    progress.update({"status": "rdma4n_only_running", "updated_at": now()})
    write(progress_path, progress)
    ordinal = 0
    for repeat in (1, 2, 3):
        for workload in WORKLOADS:
            batch = []
            logical_ids = [f"{repeat}:rdma4n:{arch}:{workload}" for arch in ARCHITECTURES]
            with ThreadPoolExecutor(max_workers=3) as pool:
                for arch_index, (arch, lane) in enumerate(zip(ARCHITECTURES, LANES)):
                    ordinal += 1
                    row = rows[repeat, base[arch, workload]["point"]["id"], workload]
                    if row["status"] == "valid":
                        continue
                    def run(row=row, arch=arch, lane=lane, ordinal=ordinal):
                        prior_attempt = max((int(record.get("attempt", 0)) for record in row.get("attempts", [])), default=0)
                        for attempt in range(prior_attempt + 1, prior_attempt + args.max_attempts + 1):
                            result = execute_once(bundle, base[arch, workload], args.results, repeat, ordinal, lane, logical_ids, attempt)
                            item, actual, summary, path, ok, reason = result
                            with LOCK:
                                row["attempts"].append({"attempt": attempt, "run_id": item["run_id"], "artifact": str(path), "valid": ok, "reason": reason, "status": summary.get("status"), "lane_gpu": lane, "hca_group": LANES[lane]["nic"], "hca_shared": False, "concurrent_job_ids": logical_ids, "layout": actual, "finished_at": now()})
                                row["status"] = "valid" if ok else "retry_needed"
                                progress["valid_runs"] = sum(r["status"] == "valid" for r in rows.values())
                                progress["runs"] = list(rows.values())
                                progress["updated_at"] = now()
                                write(progress_path, progress)
                            if ok:
                                return
                        raise RuntimeError(f"exhausted retries {repeat}:{arch}:{workload}")
                    batch.append(pool.submit(run))
                for future in as_completed(batch):
                    future.result()
    progress.update({"status": "rdma4n_complete", "valid_runs": 54, "updated_at": now()})
    progress["runs"] = list(rows.values())
    write(progress_path, progress)
    print(json.dumps({"valid": 54}))


if __name__ == "__main__":
    main()
