#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "benchmark" / "src"))
from aflex_benchmark.deploy import RemoteExecutor
from aflex_benchmark.runner import execute_item
from rq1lib_v2 import (
    ARCH_ORDER,
    atomic_json,
    canary_gate,
    configs,
    enrich_summary,
    exclusive_lock,
    expand,
    inventory_row,
    load,
    load_progress,
    metrics,
    result_status,
    select_node,
    sla_evaluation,
    utcnow,
    verify_model,
    workload_filename,
)


def csv_values(raw, cast=str):
    return tuple(cast(x.strip()) for value in raw for x in value.split(",") if x.strip())


def external_gpu_resource_conflict(summary):
    if summary.get("status") != "failed":
        return False
    error = str(summary.get("error") or "").lower()
    if any(marker in error for marker in ("out of memory", "cuda_error_out_of_memory", "cuda out of memory", "memory allocation failed", "oom")):
        return False
    explicit = ("planned gpu compute applications remain", "gpu ownership conflict", "gpu ownership barrier failed", "gpu resource conflict", "gpu is already in use", "gpu already in use", "gpu is in use by another process", "gpu occupied by another process")
    return any(marker in error for marker in explicit) or ("gpu" in error and any(marker in error for marker in ("device or resource busy", "gpu busy")))


def pause_pending(progress, row, reason):
    paused_at = utcnow()
    row["status"] = "pending"
    for field in ("finished_at", "sla_pass", "sla", "metrics", "artifact", "failure_class"):
        row.pop(field, None)
    progress.update(pause_reason=reason, paused_at=paused_at, updated_at=paused_at)
    return reason


def pause_for_external_conflict(progress, row, summary, artifact):
    reason = str(summary.get("error") or "external GPU resource conflict")
    row.setdefault("attempts", []).append({"at": utcnow(), "artifact": artifact, "summary_status": summary.get("status"), "failure_class": "external_resource_conflict", "error": reason})
    return pause_pending(progress, row, reason)


def read_request_rows(artifact):
    path = Path(artifact) / "requests.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description="Isolated RQ1-v2 runner (dry-run by default)")
    parser.add_argument("--node", required=True)
    parser.add_argument("--phase", choices=("canary_v2", "formal_v2"), default="formal_v2")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--architecture", action="append", default=[])
    parser.add_argument("--workload", action="append", default=[])
    parser.add_argument("--qps", action="append", default=[])
    parser.add_argument("--repeat", action="append", default=[])
    parser.add_argument("--results", type=Path, default=ROOT / "results/v2")
    parser.add_argument("--execute", action="store_true", help="perform SSH/GPU execution; absent means dry-run")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--check-gate", action="store_true")
    args = parser.parse_args()

    models, workloads, matrix, cluster = configs()
    cluster = select_node(cluster, args.node)
    queue = expand(args.node, args.phase, csv_values(args.model), csv_values(args.architecture), csv_values(args.workload), csv_values(args.qps, float), csv_values(args.repeat, int))
    progress_path = args.results / "progress.json"
    lock_path = args.results / "scheduler.lock"
    with exclusive_lock(lock_path):
        progress = load_progress(progress_path, args.node)
        gate = canary_gate(
            progress,
            matrix["canary_expected"],
            workload=matrix["canary_workload"],
            qps=matrix["canary_qps"],
        )
        if args.check_gate:
            print(json.dumps(gate, indent=2))
            return
        if args.phase == "formal_v2" and args.execute and not gate["open"]:
            raise RuntimeError("18/18 RQ1-v2 canary gate is not open; run --phase canary_v2 --execute first")

        host = cluster["nodes"][0]["host"]
        executor = RemoteExecutor(cluster["container"], False)
        availability = {}
        if args.execute:
            progress.pop("pause_reason", None); progress.pop("paused_at", None)
            for item in queue:
                path = item["model"]["path"]
                if path not in availability:
                    availability[path] = verify_model(executor, host, path)
        counts = {}
        paused = False
        pause_reason = None
        for item in queue:
            row = progress["runs"].setdefault(item["run_id"], inventory_row(item))
            if row["status"] in {"valid", "loadgen_invalid", "model_failure"}:
                continue
            if row["status"] == "environment_failure" and not args.retry_failed:
                continue
            if not args.execute:
                counts["planned"] = counts.get("planned", 0) + 1
                continue
            if not availability[item["model"]["path"]]:
                pause_reason = pause_pending(progress, row, f"model path temporarily unavailable on {host}: {item['model']['path']}")
                atomic_json(progress_path, progress)
                paused = True
                break

            execution = dict(item)
            execution["state"] = "ready"
            execution["point"] = dict(item["point"])
            execution["point"]["workload_path"] = str(ROOT / "data/workloads_v2" / workload_filename(item["workload"], item["qps"], item["repeat"]))
            bundle = {"cluster": cluster, "models": {"models": models}, "workloads": workloads, "matrix": {}, "cluster_source": str(ROOT / "configs/cluster.json")}
            summary = execute_item(execution, bundle, args.results / "artifacts")
            artifact_path = args.results / "artifacts" / item["run_id"]
            artifact = str(artifact_path)
            if external_gpu_resource_conflict(summary):
                pause_reason = pause_for_external_conflict(progress, row, summary, artifact)
                atomic_json(progress_path, progress)
                paused = True
                break
            if summary.get("status") in {"complete", "partial"}:
                enrich_summary(summary, read_request_rows(artifact_path), item["qps"], matrix["loadgen"]["min_send_qps_ratio"])
                atomic_json(artifact_path / "summary.json", summary)
            passed, sla = sla_evaluation(summary, matrix["sla"])
            status = result_status(summary)
            if status == "environment_failure":
                reason = str(summary.get("error") or "RQ1-v2 environment failure")
                row.setdefault("attempts", []).append(
                    {
                        "at": utcnow(),
                        "artifact": artifact,
                        "summary_status": summary.get("status"),
                        "failure_class": status,
                        "error": reason,
                    }
                )
                pause_reason = pause_pending(progress, row, reason)
                atomic_json(progress_path, progress)
                paused = True
                break
            failure_class = None if status in {"valid", "loadgen_invalid"} else status
            row["attempts"].append({"at": utcnow(), "artifact": artifact, "summary_status": summary.get("status"), "failure_class": failure_class})
            row.update(status=status, sla_pass=passed, sla=sla, metrics=metrics(summary, sla), artifact=artifact, finished_at=utcnow())
            if failure_class:
                row["failure_class"] = failure_class
            progress["updated_at"] = utcnow()
            atomic_json(progress_path, progress)
            counts[status] = counts.get(status, 0) + 1
        progress["updated_at"] = utcnow()
        atomic_json(progress_path, progress)
        print(json.dumps({"mode": "execute" if args.execute else "dry-run", "phase": args.phase, "selected": len(queue), "counts": counts, "gate": canary_gate(progress, matrix["canary_expected"], workload=matrix["canary_workload"], qps=matrix["canary_qps"]), "progress": str(progress_path), "paused": paused, "reason": pause_reason}, indent=2))


if __name__ == "__main__":
    main()
