from __future__ import annotations

import hashlib
import json
import math
import statistics
from pathlib import Path

from rq1lib import atomic_json, exclusive_lock, load, select_node, utcnow, verify_model

ROOT = Path(__file__).resolve().parents[1]
QPS_ORDER = (1.0, 2.0, 4.0, 8.0)
ARCH_ORDER = ("native", "pd", "af")
PHASES = ("canary_v2", "formal_v2")


def canonical_qps(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"QPS must be finite and positive: {value!r}")
    return format(value, ".15g")


def stable_id(phase, model, architecture, workload, qps, repeat):
    if phase not in PHASES:
        raise ValueError(f"invalid RQ1-v2 phase: {phase!r}")
    logical = {
        "schema": 2,
        "rq": "RQ1-v2",
        "phase": phase,
        "model": model,
        "architecture": architecture,
        "workload": workload,
        "qps": canonical_qps(qps),
        "repeat": int(repeat),
    }
    digest = hashlib.sha256(
        json.dumps(logical, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:16]
    return f"rq1-v2-{phase}-{digest}"


def workload_filename(workload, qps, repeat):
    return f"{workload}_qps{canonical_qps(qps)}_repeat{int(repeat)}.jsonl"


def configs():
    return (
        load(ROOT / "configs/models.json")["models"],
        load(ROOT / "configs/workloads_v2.json"),
        load(ROOT / "configs/matrix_v2.json"),
        load(ROOT / "configs/cluster.json"),
    )


def make_point(
    model,
    architecture,
    architecture_config,
    workloads,
    qps,
    node,
    request_timeout_s=180,
):
    point = dict(architecture_config["point"])
    point.update(
        {
            "id": f"rq1-v2-{model}-{architecture}-{node}",
            "rq": "RQ1-v2",
            "model": model,
            "workloads": list(workloads),
            "qps": list(qps),
            "energy_scope": "plan_used_gpus",
            "request_timeout_s": int(request_timeout_s),
            "health_timeout_s": 1800,
            "max_inflight": 256,
            "metadata": {
                "measurement": "RQ1-v2",
                "node": node,
                "deployment": architecture_config["label"],
            },
        }
    )
    return point


def expand(node, phase="formal_v2", model_filter=(), architecture_filter=(), workload_filter=(), qps_filter=(), repeat_filter=()):
    if phase not in PHASES:
        raise ValueError(f"invalid phase: {phase!r}")
    models, workloads, matrix, cluster = configs()
    select_node(cluster, node)
    wanted = lambda value, values: not values or value in values
    phase_workloads = [matrix["canary_workload"]] if phase == "canary_v2" else workloads["classes"]
    phase_qps = [float(matrix["canary_qps"])] if phase == "canary_v2" else QPS_ORDER
    phase_repeats = [1] if phase == "canary_v2" else matrix["repeats"]
    rows = []
    for model_id, model in models.items():
        if not wanted(model_id, model_filter):
            continue
        for architecture in ARCH_ORDER:
            if not wanted(architecture, architecture_filter):
                continue
            point = make_point(
                model_id,
                architecture,
                matrix["architectures"][architecture],
                phase_workloads,
                phase_qps,
                node,
                matrix.get("request_timeout_s", 180),
            )
            for workload in phase_workloads:
                if not wanted(workload, workload_filter):
                    continue
                for qps in phase_qps:
                    if not wanted(float(qps), tuple(float(x) for x in qps_filter)):
                        continue
                    for repeat in phase_repeats:
                        if not wanted(repeat, repeat_filter):
                            continue
                        rows.append({
                            "phase": phase, "model_id": model_id, "model": model,
                            "architecture": architecture, "point": point,
                            "workload": workload, "qps": float(qps), "repeat": repeat,
                            "run_id": stable_id(phase, model_id, architecture, workload, qps, repeat),
                        })
    return rows


def initial_progress(node):
    return {"schema_version": 2, "rq": "RQ1-v2", "node": node, "created_at": utcnow(), "updated_at": utcnow(), "runs": {}}


def load_progress(path, node):
    value = load(path) if Path(path).exists() else initial_progress(node)
    if value.get("rq") != "RQ1-v2":
        raise ValueError("progress is not an RQ1-v2 progress file")
    if value.get("node") != node:
        raise ValueError(f"progress belongs to node {value.get('node')!r}, not {node!r}")
    return value


def inventory_row(item, status="pending"):
    return {"run_id": item["run_id"], "phase": item["phase"], "model": item["model_id"], "model_path": item["model"]["path"], "architecture": item["architecture"], "workload": item["workload"], "qps": item["qps"], "repeat": item["repeat"], "status": status, "attempts": []}


def canary_gate(progress, expected=18, models=None, workload=None, qps=None):
    selected = set(models or ())
    rows = [
        row
        for row in progress["runs"].values()
        if row["phase"] == "canary_v2"
        and (not selected or row["model"] in selected)
        and (workload is None or row["workload"] == workload)
        and (qps is None or float(row["qps"]) == float(qps))
    ]
    valid = [r for r in rows if r["status"] == "valid"]
    return {"open": len(valid) == expected and len(rows) == expected, "valid": len(valid), "expected": expected, "observed": len(rows)}


def percentile(values, p):
    values = sorted(float(x) for x in values)
    if not values:
        return 0.0
    k = (len(values) - 1) * p / 100.0
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] if lo == hi else values[lo] * (hi - k) + values[hi] * (k - lo)


def calculate_load_metrics(rows, target_qps):
    if not rows:
        return {"target_qps": float(target_qps), "offered_qps": float(target_qps), "realized_offered_qps": 0.0, "send_qps": 0.0, "completion_qps": 0.0, "achieved_qps": 0.0, "arrival_lag_p50_ms": 0.0, "arrival_lag_p90_ms": 0.0, "arrival_lag_p99_ms": 0.0, "arrival_lag_max_ms": 0.0, "drain_time_s": 0.0, "drain_after_last_send_s": 0.0}
    scheduled = [float(r["scheduled_arrival_s"]) for r in rows]
    sent = [float(r["sent_offset_s"]) for r in rows]
    completed = [float(r["completed_offset_s"]) for r in rows]
    lags = [float(r.get("arrival_lag_ms", max(0.0, (s - a) * 1000))) for r, s, a in zip(rows, sent, scheduled)]
    ok = sum(bool(r.get("success")) for r in rows)
    span = lambda xs: max(xs) - min(xs)
    realized = (len(rows) - 1) / span(scheduled) if len(rows) > 1 and span(scheduled) > 0 else 0.0
    send_qps = (len(rows) - 1) / span(sent) if len(rows) > 1 and span(sent) > 0 else 0.0
    completion_qps = ok / (max(completed) - min(sent)) if ok and max(completed) > min(sent) else 0.0
    return {"target_qps": float(target_qps), "offered_qps": float(target_qps), "realized_offered_qps": realized, "send_qps": send_qps, "completion_qps": completion_qps, "achieved_qps": completion_qps, "arrival_lag_p50_ms": percentile(lags, 50), "arrival_lag_p90_ms": percentile(lags, 90), "arrival_lag_p99_ms": percentile(lags, 99), "arrival_lag_max_ms": max(lags), "drain_time_s": max(completed) - max(scheduled), "drain_after_last_send_s": max(completed) - max(sent)}


def enrich_summary(summary, request_rows, target_qps, min_send_ratio=0.98):
    values = calculate_load_metrics(request_rows, target_qps)
    ratio = values["send_qps"] / float(target_qps) if target_qps else 0.0
    values.update({"send_qps_ratio": ratio, "loadgen_healthy": ratio >= min_send_ratio})
    summary.update(values)
    summary.setdefault("metrics", {})["arrival_lag_ms"] = {
        "count": len(request_rows), "p50": values["arrival_lag_p50_ms"],
        "p90": values["arrival_lag_p90_ms"], "p99": values["arrival_lag_p99_ms"],
        "max": values["arrival_lag_max_ms"],
    }
    return summary


def sla_evaluation(summary, sla):
    total = int(summary.get("requests_total", 0)); success = int(summary.get("requests_success", 0))
    metrics = summary.get("metrics", {})
    values = {"success_rate": success / total if total else 0.0, "ttft_p90_ms": metrics.get("ttft_client_ms", {}).get("p90", float("inf")), "tpot_p90_ms": metrics.get("tpot_ms", {}).get("p90", float("inf")), "e2e_p90_ms": metrics.get("e2e_ms", {}).get("p90", float("inf"))}
    passed = values["success_rate"] >= sla["min_success_rate"] and all(values[k] <= sla[k] for k in ("ttft_p90_ms", "tpot_p90_ms", "e2e_p90_ms"))
    return passed, values


def classify_failure(summary):
    if summary.get("status") == "complete":
        return None
    stage = str(summary.get("failure_stage") or "").lower()
    error = str(summary.get("error") or "").lower()
    environment_markers = ("container", "connection refused", "no such file", "module not found", "modulenotfounderror", "health", "ssh", "router", "gpu ownership", "resource busy")
    return "environment_failure" if stage in {"deployment", "client_compatibility"} or any(x in error for x in environment_markers) else "model_failure"


def result_status(summary):
    if summary.get("status") in {"complete", "partial"}:
        return "valid" if summary.get("loadgen_healthy") else "loadgen_invalid"
    return classify_failure(summary)


def metrics(summary, sla_values):
    energy = summary.get("energy", {})
    names = ("target_qps", "offered_qps", "realized_offered_qps", "send_qps", "send_qps_ratio", "completion_qps", "achieved_qps", "arrival_lag_p50_ms", "arrival_lag_p90_ms", "arrival_lag_p99_ms", "arrival_lag_max_ms", "drain_time_s", "drain_after_last_send_s", "input_throughput_tokens_s", "output_throughput_tokens_s", "all_throughput_tokens_s", "average_cluster_power_w", "energy_per_input_token_j", "energy_per_output_token_j", "energy_per_successful_request_j", "input_tokens_per_j", "output_tokens_per_j")
    out = {name: summary.get(name) for name in names}
    out["total_energy_j"] = energy.get("total_j")
    out["loadgen_healthy"] = summary.get("loadgen_healthy")
    out.update(sla_values)
    return out


def aggregate(progress):
    groups = {}
    for row in progress["runs"].values():
        if row["phase"] != "formal_v2" or row["status"] != "valid":
            continue
        key = (row["model"], row["architecture"], row["workload"], row["qps"])
        groups.setdefault(key, []).append(row["metrics"])
    output = []
    for key, rows in sorted(groups.items()):
        numeric = set.intersection(*(set(k for k, v in row.items() if isinstance(v, (int, float)) and not isinstance(v, bool)) for row in rows))
        summary = {name: {"mean": statistics.fmean(r[name] for r in rows), "sample_std": statistics.stdev(r[name] for r in rows) if len(rows) > 1 else None} for name in sorted(numeric)}
        output.append({"model": key[0], "architecture": key[1], "workload": key[2], "qps": key[3], "repeats": len(rows), "metrics": summary})
    return output
