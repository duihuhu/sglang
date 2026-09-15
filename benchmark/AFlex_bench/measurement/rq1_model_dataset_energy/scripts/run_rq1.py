#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[0]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT.parent / "benchmark" / "src"))
from rq1lib import *
from aflex_benchmark.deploy import RemoteExecutor
from aflex_benchmark.runner import execute_item


def csv_values(raw, cast=str):
    return tuple(
        cast(x.strip()) for value in raw for x in value.split(",") if x.strip()
    )


def run_status(summary, _sla_pass):
    return "valid" if summary.get("status") == "complete" else "failed"


def external_gpu_resource_conflict(summary):
    if summary.get("status") != "failed":
        return False
    error = str(summary.get("error") or "").lower()
    if any(
        marker in error
        for marker in (
            "out of memory",
            "cuda_error_out_of_memory",
            "cuda out of memory",
            "memory allocation failed",
            "oom",
        )
    ):
        return False
    explicit_markers = (
        "planned gpu compute applications remain",
        "gpu ownership conflict",
        "gpu ownership barrier failed",
        "gpu resource conflict",
        "gpu is already in use",
        "gpu already in use",
        "gpu is in use by another process",
        "gpu occupied by another process",
    )
    return any(marker in error for marker in explicit_markers) or (
        "gpu" in error
        and any(marker in error for marker in ("device or resource busy", "gpu busy"))
    )


def pause_pending(progress, row, reason):
    paused_at = utcnow()
    row["status"] = "pending"
    for field in (
        "finished_at",
        "sla_pass",
        "sla",
        "metrics",
        "artifact",
        "saturation_source",
    ):
        row.pop(field, None)
    progress["pause_reason"] = reason
    progress["paused_at"] = paused_at
    progress["updated_at"] = paused_at
    return reason


def pause_for_external_conflict(progress, row, summary, artifact):
    reason = str(summary.get("error") or "external GPU resource conflict")
    row.setdefault("attempts", []).append(
        {
            "at": utcnow(),
            "artifact": artifact,
            "summary_status": summary.get("status"),
            "failure_class": "external_resource_conflict",
            "error": reason,
        }
    )
    return pause_pending(progress, row, reason)


def pause_for_model_unavailable(progress, row, host, path):
    reason = f"model path temporarily unavailable on {host}: {path}"
    return pause_pending(progress, row, reason)


def main():
    p = argparse.ArgumentParser(
        description="Recoverable RQ1 energy matrix runner (dry-run by default)"
    )
    p.add_argument("--node", required=True, help="explicit cluster node name")
    p.add_argument("--phase", choices=("canary", "formal"), default="formal")
    p.add_argument("--model", action="append", default=[])
    p.add_argument("--architecture", action="append", default=[])
    p.add_argument("--workload", action="append", default=[])
    p.add_argument("--qps", action="append", default=[])
    p.add_argument("--repeat", action="append", default=[])
    p.add_argument("--results", type=Path, default=ROOT / "results/default")
    p.add_argument(
        "--execute",
        action="store_true",
        help="perform SSH/GPU execution; absent means dry-run",
    )
    p.add_argument("--retry-failed", action="store_true")
    p.add_argument("--check-gate", action="store_true")
    args = p.parse_args()
    models, workloads, matrix, cluster = configs()
    cluster = select_node(cluster, args.node)
    selected_models = csv_values(args.model) or tuple(models)
    gate_expected = len(selected_models) * len(ARCH_ORDER)
    progress_path = args.results / "progress.json"
    lock_path = args.results / "scheduler.lock"
    queue = expand(
        args.node,
        args.phase,
        csv_values(args.model),
        csv_values(args.architecture),
        csv_values(args.workload),
        csv_values(args.qps, int),
        csv_values(args.repeat, int),
    )
    with exclusive_lock(lock_path):
        progress = load_progress(progress_path, args.node)
        if args.check_gate:
            print(
                json.dumps(
                    canary_gate(progress, gate_expected, selected_models), indent=2
                )
            )
            return
        if (
            args.phase == "formal"
            and args.execute
            and not canary_gate(progress, gate_expected, selected_models)["open"]
        ):
            raise RuntimeError(
                "18/18 canary gate is not open; run --phase canary --execute first"
            )
        host = cluster["nodes"][0]["host"]
        executor = RemoteExecutor(cluster["container"], False)
        availability = {}
        if args.execute:
            progress.pop("pause_reason", None)
            progress.pop("paused_at", None)
            for item in queue:
                path = item["model"]["path"]
                availability.setdefault(path, verify_model(executor, host, path))
        counts = {}
        paused = False
        pause_reason = None
        for item in queue:
            row = progress["runs"].setdefault(item["run_id"], inventory_row(item))
            if row["status"] in {
                "valid",
                "blocked_model_missing",
                "skipped_saturated",
            }:
                continue
            if row["status"] == "failed" and not args.retry_failed:
                continue
            if not args.execute:
                counts["planned"] = counts.get("planned", 0) + 1
                continue
            if not availability[item["model"]["path"]]:
                pause_reason = pause_for_model_unavailable(
                    progress, row, host, item["model"]["path"]
                )
                atomic_json(progress_path, progress)
                paused = True
                break
            else:
                blocker = saturated(
                    progress, item, matrix["saturation"]["min_achieved_qps_ratio"]
                )
                if blocker and matrix["saturation"]["skip_higher_qps"]:
                    row.update(
                        status="skipped_saturated",
                        saturation_source=blocker,
                        finished_at=utcnow(),
                    )
                else:
                    execution = dict(item)
                    execution["state"] = "ready"
                    execution["point"] = dict(item["point"])
                    execution["point"]["workload_path"] = str(
                        ROOT
                        / "data/workloads"
                        / f"{item['workload']}_qps{item['qps']}.jsonl"
                    )
                    bundle = {
                        "cluster": cluster,
                        "models": {"models": models},
                        "workloads": workloads,
                        "matrix": {},
                        "cluster_source": str(ROOT / "configs/cluster.json"),
                    }
                    summary = execute_item(
                        execution, bundle, args.results / "artifacts"
                    )
                    artifact = str(args.results / "artifacts" / item["run_id"])
                    if args.phase == "formal" and external_gpu_resource_conflict(
                        summary
                    ):
                        pause_reason = pause_for_external_conflict(
                            progress, row, summary, artifact
                        )
                        atomic_json(progress_path, progress)
                        paused = True
                        break
                    passed, sla = sla_evaluation(summary, matrix["sla"])
                    status = run_status(summary, passed)
                    row["attempts"].append(
                        {
                            "at": utcnow(),
                            "artifact": artifact,
                            "summary_status": summary.get("status"),
                        }
                    )
                    row.update(
                        status=status,
                        sla_pass=passed,
                        sla=sla,
                        metrics=metrics(summary, sla),
                        artifact=artifact,
                        finished_at=utcnow(),
                    )
            progress["updated_at"] = utcnow()
            atomic_json(progress_path, progress)
            counts[row["status"]] = counts.get(row["status"], 0) + 1
        progress["updated_at"] = utcnow()
        atomic_json(progress_path, progress)
        print(
            json.dumps(
                {
                    "mode": "execute" if args.execute else "dry-run",
                    "phase": args.phase,
                    "selected": len(queue),
                    "counts": counts,
                    "gate": canary_gate(progress, gate_expected, selected_models),
                    "progress": str(progress_path),
                    "paused": paused,
                    "reason": pause_reason,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
