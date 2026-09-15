from __future__ import annotations

import json
import random
import shlex
import threading
import time
import hashlib
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from .collect import append_jsonl, collect_system, energy_delta, read_cluster_energy, stream_request
from .deploy import RemoteExecutor, build_plan, execute_lifecycle
from .runner import request_endpoint, runtime_options
from .stats import describe, summarize_requests

FATAL_LOG_PATTERN = ("address already in use|bind(ing)? failed|failed to bind|"
                     "cannot assign requested address|traceback \(most recent call last\)")


def generate_poisson_workload(qps: int, count: int = 32, seed: int = 20260823,
                              input_len: int = 128, output_len: int = 64,
                              timeout_s: float = 45) -> list[dict]:
    """Generate one deterministic, independent Poisson trace for a QPS point."""
    rng = random.Random(f"{seed}:qps:{qps}")
    arrival = 0.0
    rows = []
    for index in range(count):
        arrival += rng.expovariate(float(qps))
        rows.append({
            "request_id": f"af-sweep-qps{qps}-{index:05d}",
            "arrival_time_s": arrival,
            "input_len": input_len,
            "output_len": output_len,
            "timeout_s": timeout_s,
        })
    return rows


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def probe_deployment(plan, executor) -> dict:
    """Check every A endpoint plus every planned process after a QPS point."""
    endpoint_rows = []
    for endpoint in plan.endpoints:
        host = endpoint.split("//", 1)[-1].rsplit(":", 1)[0]
        command = f"curl -fsS --max-time 3 {shlex.quote(endpoint.rstrip('/') + '/get_model_info')} >/dev/null"
        result = executor.run(host, command, check=False, quiet=True, timeout=5)
        endpoint_rows.append({"endpoint": endpoint, "ok": result.returncode == 0,
                              "returncode": result.returncode})
    process_rows = []
    fatal = False
    for process in plan.processes:
        status = executor._startup_status(process)
        detail = (status.stdout or status.stderr or "").strip()
        is_fatal = status.returncode == 22 or "fatal startup log:" in detail.lower()
        fatal = fatal or is_fatal
        process_rows.append({"role": process.role, "host": process.host,
                             "port": process.port, "alive": status.returncode == 0,
                             "fatal_log": is_fatal, "returncode": status.returncode,
                             "detail": detail})
    return {
        "endpoints": endpoint_rows,
        "processes": process_rows,
        "all_endpoints_ok": all(row["ok"] for row in endpoint_rows),
        "all_processes_alive": all(row["alive"] for row in process_rows),
        "fatal_log": fatal,
    }


def point_wall_limit(requests: list[dict], request_timeout_s: float = 45,
                     cap_s: float = 90) -> float:
    """Allow the full offered-arrival window plus one request timeout."""
    last_arrival = max((float(request.get("arrival_time_s", 0))
                        for request in requests), default=0.0)
    return min(float(cap_s), max(60.0, last_arrival + float(request_timeout_s) + 5.0))


def _run_point(target, qps: int, requests: list[dict], point_dir: Path,
               executor, bundle: dict, point: dict, wall_timeout_s: float = 60) -> dict:
    raw_path = point_dir / "requests.jsonl"
    system_before = collect_system(executor, bundle["cluster"], point["nodes"])
    energy_before = read_cluster_energy(executor, bundle["cluster"], point["nodes"])
    start = time.monotonic()
    deadline = start + wall_timeout_s
    max_inflight = int(point.get("sweep_max_inflight", point.get("max_inflight", 16)))
    admission = threading.BoundedSemaphore(max_inflight)
    rows = []

    def admitted(index, request):
        target_send = start + float(request["arrival_time_s"])
        delay = target_send - time.monotonic()
        if delay > 0:
            time.sleep(min(delay, max(0.0, deadline - time.monotonic())))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return {"request_id": request["request_id"], "success": False,
                    "error": "point wall timeout before send",
                    "scheduled_arrival_s": request["arrival_time_s"],
                    "sent_offset_s": time.monotonic() - start,
                    "arrival_lag_ms": max(0.0, (time.monotonic() - start - request["arrival_time_s"]) * 1000),
                    "completed_offset_s": time.monotonic() - start}
        with admission:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return {"request_id": request["request_id"], "success": False,
                        "error": "point wall timeout waiting for admission",
                        "scheduled_arrival_s": request["arrival_time_s"],
                        "sent_offset_s": time.monotonic() - start,
                        "arrival_lag_ms": max(0.0, (time.monotonic() - start - request["arrival_time_s"]) * 1000),
                        "completed_offset_s": time.monotonic() - start}
            return stream_request(request_endpoint(target, index), request, start,
                                  min(float(request.get("timeout_s", 45)), remaining))

    pool = ThreadPoolExecutor(max_workers=len(requests) or 1)
    pending = {pool.submit(admitted, index, request): request
               for index, request in enumerate(requests)}
    try:
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            done, pending_futures = wait(pending, timeout=remaining,
                                         return_when=FIRST_COMPLETED)
            if not done:
                break
            for future in done:
                request = pending.pop(future)
                try:
                    row = future.result()
                except BaseException as exc:
                    row = {"request_id": request["request_id"], "success": False,
                           "error": repr(exc), "scheduled_arrival_s": request["arrival_time_s"],
                           "sent_offset_s": time.monotonic() - start,
                           "arrival_lag_ms": 0.0,
                           "completed_offset_s": time.monotonic() - start}
                rows.append(row)
                append_jsonl(raw_path, row)
        for future, request in list(pending.items()):
            if future.done():
                try:
                    row = future.result()
                except BaseException as exc:
                    row = {"request_id": request["request_id"], "success": False,
                           "error": repr(exc),
                           "scheduled_arrival_s": request["arrival_time_s"],
                           "sent_offset_s": time.monotonic() - start,
                           "arrival_lag_ms": 0.0,
                           "completed_offset_s": time.monotonic() - start}
            else:
                future.cancel()
                now = min(time.monotonic(), deadline)
                row = {"request_id": request["request_id"], "success": False,
                       "error": "point wall timeout",
                       "scheduled_arrival_s": request["arrival_time_s"],
                       "sent_offset_s": now - start,
                       "arrival_lag_ms": max(
                           0.0, (now - start - request["arrival_time_s"]) * 1000),
                       "completed_offset_s": now - start}
            rows.append(row)
            append_jsonl(raw_path, row)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)

    energy_after = read_cluster_energy(executor, bundle["cluster"], point["nodes"])
    system_after = collect_system(executor, bundle["cluster"], point["nodes"])
    energy = energy_delta(energy_before, energy_after)
    write_json(point_dir / "energy.json", {"before_mj": energy_before,
                                            "after_mj": energy_after,
                                            "normalized": energy})
    write_json(point_dir / "system.json", {"before": system_before, "after": system_after})
    summary = summarize_requests(rows, energy)
    summary["arrival_lag_ms"] = describe([float(row.get("arrival_lag_ms", 0)) for row in rows])
    arrivals = [float(request.get("arrival_time_s", 0)) for request in requests]
    arrival_span = max(arrivals, default=0.0)
    summary.update({"target_qps": qps, "offered_qps": qps,
                    "max_inflight": max_inflight,
                    "arrival_span": arrival_span,
                    "wall_limit": wall_timeout_s,
                    "wall_time_s": min(time.monotonic() - start, wall_timeout_s)})
    return summary


def _core_metrics(summary: dict) -> dict:
    keys = ("status", "requests_total", "requests_success", "requests_failed",
            "achieved_qps", "output_throughput_tokens_s", "energy_per_output_token_j",
            "average_cluster_power_w", "arrival_lag_ms", "wall_time_s",
            "offered_qps", "max_inflight", "arrival_span", "wall_limit")
    return {key: summary.get(key) for key in keys}


def sweep_run_tag(point: dict) -> str:
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime())
    nonce = time.time_ns()
    digest = hashlib.sha256(f"{point.get('id', 'sweep')}:{nonce}".encode()).hexdigest()[:8]
    return f"sweep-{stamp}-{digest}"


def run_af_qps_sweep(bundle: dict, results: Path, start_qps: int = 1,
                     end_qps: int = 16, continue_on_partial: bool = False,
                     executor=None, lifecycle=execute_lifecycle,
                     point_runner=_run_point, probe=probe_deployment) -> dict:
    point = bundle["matrix"]["points"][0]
    configured = [int(value) for value in point["qps"]]
    selected = [qps for qps in configured if start_qps <= qps <= end_qps]
    if not selected or min(selected) < 1 or max(selected) > 16:
        raise ValueError("selected QPS range must be non-empty and within 1..16")
    if configured != list(range(1, 17)):
        raise ValueError("AF sweep config qps must be exactly 1..16")
    if int(point.get("requests_per_qps", 0)) != 32 or int(point.get("output_len", 0)) != 64:
        raise ValueError("AF sweep requires requests_per_qps=32 and output_len=64")

    model = bundle["models"]["models"][point["model"]]["path"]
    run_tag = sweep_run_tag(point)
    plan = build_plan(bundle["cluster"], model, point, run_tag=run_tag)
    plan.validate()
    options = runtime_options(point)
    plan.runtime_options.update(options)
    executor = executor or RemoteExecutor(bundle["cluster"]["container"], False)
    results.mkdir(parents=True, exist_ok=True)
    workloads_dir = results / "workloads"
    summary_path = results / "sweep_summary.json"
    overall = {"status": "running", "point_id": point["id"], "qps": selected,
               "last_stable_qps": None, "first_failed_qps": None,
               "stop_reason": None, "points": [], "run_tag": run_tag}
    write_json(summary_path, overall)

    def body(target):
        for qps in selected:
            point_dir = results / f"qps{qps}"
            point_dir.mkdir(parents=True, exist_ok=True)
            workload_path = workloads_dir / f"qps{qps}.jsonl"
            if workload_path.exists():
                requests = [json.loads(line) for line in workload_path.read_text().splitlines()
                            if line.strip()]
            else:
                requests = generate_poisson_workload(
                    qps, int(point["requests_per_qps"]), int(point.get("seed", 20260823)),
                    int(point.get("input_len", 128)), int(point["output_len"]),
                    float(point.get("request_timeout_s", 45)))
                write_jsonl(workload_path, requests)
            write_json(point_dir / "status.json", {"status": "running", "qps": qps})
            try:
                wall_limit = point_wall_limit(
                    requests, float(point.get("request_timeout_s", 45)),
                    float(point.get("point_wall_cap_s", 90)))
                item_summary = point_runner(target, qps, requests, point_dir,
                                            executor, bundle, point, wall_limit)
                health = probe(plan, executor)
                item_summary["post_run_probe"] = health
            except BaseException as exc:
                item_summary = {"status": "failed", "target_qps": qps,
                                "error": repr(exc), "failure_stage": "qps_body"}
                health = {"all_endpoints_ok": False, "all_processes_alive": False,
                          "fatal_log": False, "error": repr(exc)}
                item_summary["post_run_probe"] = health

            reason = None
            if item_summary.get("requests_success", 0) == 0:
                reason = "zero_successful_requests"
            elif not health.get("all_endpoints_ok", False):
                reason = "endpoint_probe_failed"
            elif not health.get("all_processes_alive", False):
                reason = "process_dead"
            elif health.get("fatal_log", False):
                reason = "fatal_log_pattern"
            elif item_summary.get("requests_failed", 0) and not continue_on_partial:
                reason = "partial_request_failure"
            stable = item_summary.get("status") == "complete" and reason is None
            if stable:
                overall["last_stable_qps"] = qps
            elif overall["first_failed_qps"] is None:
                overall["first_failed_qps"] = qps
            if reason:
                item_summary["stop_reason"] = reason
                overall["stop_reason"] = reason
                write_json(point_dir / "stop_reason.json", {"qps": qps, "stop_reason": reason})
            write_json(point_dir / "summary.json", item_summary)
            write_json(point_dir / "status.json", {"status": item_summary.get("status", "failed"),
                                                     "qps": qps, "stop_reason": reason})
            overall["points"].append({"qps": qps, **_core_metrics(item_summary),
                                      "stop_reason": reason})
            write_json(summary_path, overall)
            if reason:
                break
        return overall

    try:
        result = lifecycle(plan, executor, body, artifact_dir=results,
                           runtime_options=options)
        overall.update(result)
        overall["status"] = "failed" if overall["first_failed_qps"] is not None else "complete"
    except BaseException as exc:
        overall.update({"status": "failed", "error": repr(exc),
                        "stop_reason": overall["stop_reason"] or "deployment_or_cleanup_failure"})
    write_json(summary_path, overall)
    return overall
