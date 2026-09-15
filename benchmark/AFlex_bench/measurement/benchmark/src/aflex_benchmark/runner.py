from __future__ import annotations

import hashlib
import json
import os
import shutil
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .collect import (
    append_jsonl,
    build_link_telemetry,
    collect_link_snapshot,
    collect_system,
    energy_delta,
    finish_link_telemetry,
    parse_comm_ledgers,
    read_backend_logs,
    read_cluster_energy,
    read_gpu_uuids,
    start_link_telemetry,
    stream_request,
    validate_link,
)
from .deploy import RemoteExecutor, build_plan, execute_lifecycle
from .deploy.base import DeploymentHandle, DeploymentPlan
from .stats import summarize_requests


def run_id(point, workload, qps, repeat=0):
    key = json.dumps(
        {"point": point, "workload": workload, "qps": qps, "repeat": repeat},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def completed_ids(results: Path) -> set[str]:
    if not results.exists():
        return set()
    completed = set()
    for path in results.glob("*/summary.json"):
        try:
            if json.loads(path.read_text()).get("status") == "complete":
                completed.add(path.parent.name)
        except (OSError, json.JSONDecodeError):
            continue
    return completed


def expand_queue(
    bundle, inventory=False, priority=None, smoke=False, allow_experimental=False
):
    models = bundle["models"]["models"]
    workloads = bundle["workloads"]
    queue = []
    for point in bundle["matrix"]["points"]:
        if smoke and not point.get("smoke"):
            continue
        if priority is not None and int(point.get("priority", 99)) > priority:
            continue
        model = models[point["model"]]
        state = "ready"
        is_experimental = bool(point.get("experimental") or model.get("experimental"))
        needs_preflight = bool(
            point.get("requires_preflight") or model.get("requires_preflight")
        )
        if (
            point["architecture"] in {"af", "pdaf"}
            and int(point["nodes"]) > 1
            and point.get("topology") != "per_node_replicas"
            and not point.get("multinode_preflight_complete")
        ):
            needs_preflight = True
        if model.get("blocked") or point.get("blocked"):
            state = "blocked"
        elif (is_experimental and not allow_experimental) or (
            needs_preflight and not (is_experimental and allow_experimental)
        ):
            state = "requires_preflight"
        for workload in point.get("workloads", workloads["classes"]):
            for qps in point.get("qps", workloads["qps"]):
                item = {
                    "point": point,
                    "model": model,
                    "workload": workload,
                    "qps": qps,
                    "state": state,
                }
                item["run_id"] = run_id(point, workload, qps)
                queue.append(item)
    return queue


def filter_queue_by_point_ids(queue, values):
    requested = {
        point_id.strip()
        for value in values
        for point_id in value.split(",")
        if point_id.strip()
    }
    if not requested:
        return queue
    available = {item["point"]["id"] for item in queue}
    unknown = requested - available
    if unknown:
        raise ValueError(
            f"point ID(s) not present in expanded queue: {', '.join(sorted(unknown))}"
        )
    return [item for item in queue if item["point"]["id"] in requested]


def request_endpoint(target, request_index):
    """Resolve a request target from a plan, handle, or legacy endpoint string."""
    if isinstance(target, DeploymentPlan):
        target = target.handle
    elif isinstance(target, str):
        target = DeploymentHandle((target,))
    return target.endpoint_for_request(request_index)


RUNTIME_OPTION_KEYS = (
    "debug_fast_fail",
    "skip_warmup",
    "health_timeout_s",
    "warmup_timeout_s",
    "warmup_curl_timeout_s",
    "request_timeout_s",
    "warmup_max_new_tokens",
    "warmup_input_len",
    "warmup_parallel",
    "wait_after_warmup_s",
    "max_inflight",
)


def runtime_options(point):
    return {key: point[key] for key in RUNTIME_OPTION_KEYS if key in point}


def normalize_requests(requests, workload, qps, timeout_s=None):
    """Return execution-ready request copies without mutating the source trace."""
    normalized = []
    for index, request in enumerate(requests):
        row = dict(request)
        row.setdefault("request_id", f"{workload}-{qps}-{index:05d}")
        if timeout_s is not None:
            row.setdefault("timeout_s", timeout_s)
        normalized.append(row)
    return normalized


def resolve_workload_path(generated_dir, workload, qps):
    """Resolve exact workload names before the legacy QPS-suffixed form."""
    root = Path(generated_dir)
    candidates = [
        root / f"{workload}.jsonl",
        root / f"{workload}_qps{qps}.jsonl",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    rendered = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(
        f"workload trace not found for {workload!r} at QPS {qps}; tried: {rendered}"
    )


def prepare_artifact_dir(results, rid):
    """Atomically install a fresh, run-scoped artifact directory."""
    results = Path(results)
    rid = str(rid)
    if not rid or Path(rid).name != rid or rid in {".", ".."}:
        raise ValueError(f"run_id must be a single safe path component: {rid!r}")
    results.mkdir(parents=True, exist_ok=True)
    out = results / rid
    nonce = f"{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}"
    staging = results / f".{rid}.{nonce}.fresh"
    backup = results / f".{rid}.{nonce}.stale"
    staging.mkdir()
    moved_old = False
    try:
        if os.path.lexists(out):
            os.replace(out, backup)
            moved_old = True
        os.replace(staging, out)
    except BaseException:
        if moved_old and not os.path.lexists(out) and os.path.lexists(backup):
            os.replace(backup, out)
        raise
    finally:
        if os.path.lexists(staging):
            shutil.rmtree(staging) if staging.is_dir() and not staging.is_symlink() else staging.unlink()
    if os.path.lexists(backup):
        shutil.rmtree(backup) if backup.is_dir() and not backup.is_symlink() else backup.unlink()
    return out


def execute_item(item, bundle, results, dry_run=False, runtime_overrides=None):
    rid = item["run_id"]
    if dry_run or item["state"] != "ready":
        return {"run_id": rid, "status": "dry-run" if dry_run else item["state"]}
    out = prepare_artifact_dir(results, rid)
    point = item["point"]
    plan = build_plan(bundle["cluster"], item["model"]["path"], point, run_tag=rid)
    options = runtime_options(point)
    options.update(runtime_overrides or {})
    plan.runtime_options.update(options)
    executor = RemoteExecutor(bundle["cluster"]["container"], False)
    raw = out / "requests.jsonl"
    workload = (
        Path(point["workload_path"])
        if point.get("workload_path")
        else resolve_workload_path(
            bundle["workloads"]["generated_dir"], item["workload"], item["qps"]
        )
    )
    loaded_requests = [
        json.loads(x) for x in workload.read_text().splitlines() if x.strip()
    ]
    requests = normalize_requests(
        loaded_requests,
        item["workload"],
        item["qps"],
        options.get("request_timeout_s"),
    )
    rows = []
    lifecycle_phase = "deployment"

    def body(target):
        nonlocal lifecycle_phase
        lifecycle_phase = "client_compatibility"
        system_before = collect_system(executor, bundle["cluster"], point["nodes"])
        link_before = (
            collect_link_snapshot(executor, plan)
            if point.get("expected_link")
            else None
        )
        energy_scope = point.get("energy_scope")
        if energy_scope in (None, "plan_used_gpus"):
            used_gpu_map = getattr(plan, "used_gpu_map", None)
            energy_scope = used_gpu_map() if used_gpu_map is not None else None
        elif not isinstance(energy_scope, dict):
            energy_scope = None
        energy_before = read_cluster_energy(
            executor, bundle["cluster"], point["nodes"], energy_scope
        )
        energy_gpu_uuids = read_gpu_uuids(executor, bundle["cluster"], energy_scope)
        link_handles = []
        link_window = {}
        start = time.monotonic()
        configured_inflight = options.get("max_inflight")
        admission = (
            threading.BoundedSemaphore(int(configured_inflight))
            if configured_inflight is not None
            else None
        )

        def admitted_request(index, request):
            # Wait for the open-loop arrival before admission. This avoids holding a
            # slot for a future request while preserving its original schedule.
            target_send = start + float(request.get("arrival_time_s", 0))
            delay = target_send - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            if admission is None:
                return stream_request(
                    request_endpoint(target, index),
                    request,
                    start,
                    options.get("request_timeout_s"),
                )
            with admission:
                return stream_request(
                    request_endpoint(target, index),
                    request,
                    start,
                    options.get("request_timeout_s"),
                )

        workers = (
            min(256, len(requests) or 1)
            if admission is not None
            else len(requests) or 1
        )
        link_handles = (
            start_link_telemetry(executor, plan, link_before)
            if link_before is not None
            else []
        )
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(admitted_request, index, request)
                    for index, request in enumerate(requests)
                ]
                for future in as_completed(futures):
                    row = future.result()
                    rows.append(row)
                    append_jsonl(raw, row)
        finally:
            link_window = finish_link_telemetry(executor, link_handles)
        energy_after = read_cluster_energy(
            executor, bundle["cluster"], point["nodes"], energy_scope
        )
        system_after = collect_system(executor, bundle["cluster"], point["nodes"])
        link_after = (
            collect_link_snapshot(executor, plan)
            if point.get("expected_link")
            else None
        )
        energy = energy_delta(energy_before, energy_after)
        (out / "energy.json").write_text(
            json.dumps(
                {
                    "before_mj": energy_before,
                    "after_mj": energy_after,
                    "normalized": energy,
                    "scope": energy_scope,
                    "gpu_uuids": energy_gpu_uuids,
                },
                indent=2,
            )
            + "\n"
        )
        (out / "system.json").write_text(
            json.dumps({"before": system_before, "after": system_after}, indent=2)
            + "\n"
        )
        if link_before is not None and link_after is not None:
            telemetry = build_link_telemetry(link_before, link_after, link_window)
            (out / "link_telemetry.json").write_text(
                json.dumps(telemetry, indent=2) + "\n"
            )
        return summarize_requests(rows, energy)

    try:
        summary = execute_lifecycle(
            plan, executor, body, artifact_dir=out, runtime_options=options
        )
        summary.update(
            {
                "run_id": rid,
                "point": point,
                "workload": item["workload"],
                "target_qps": item["qps"],
                "container": bundle["cluster"]["container"],
                "cluster_source": bundle.get("cluster_source"),
            }
        )
        if point.get("expected_link"):
            telemetry_path = out / "link_telemetry.json"
            telemetry = (
                json.loads(telemetry_path.read_text())
                if telemetry_path.exists()
                else {"nodes": []}
            )
            backend_logs = read_backend_logs(out / "logs")
            validation = validate_link(
                point,
                telemetry,
                backend_logs,
                semantic_comm_ledger=parse_comm_ledgers(backend_logs),
                request_succeeded=summary.get("status") == "complete",
            )
            (out / "link_validation.json").write_text(
                json.dumps(validation, indent=2) + "\n"
            )
            if validation["status"] not in {"pass", "verified_semantic"}:
                summary["request_status"] = summary.get("status")
                summary["status"] = "invalid_link"
                summary["link_validation"] = validation
    except Exception as exc:  # noqa: BLE001 - persist lifecycle failures as artifacts
        summary = {
            "run_id": rid,
            "status": "failed",
            "error": repr(exc),
            "point": point,
            "workload": item["workload"],
            "target_qps": item["qps"],
            "container": bundle["cluster"]["container"],
            "cluster_source": bundle.get("cluster_source"),
            "failure_stage": lifecycle_phase,
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
