from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import statistics
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED_SRC = ROOT.parent / "benchmark" / "src"
QPS_ORDER = (2, 4, 8, 16)
ARCH_ORDER = ("native", "pd", "af")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def load(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as file:
        json.dump(value, file, indent=2, sort_keys=True); file.write("\n")
        file.flush(); os.fsync(file.fileno())
    os.replace(temporary, path)


@contextmanager
def exclusive_lock(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"another RQ1 runner owns {path}") from exc
        yield


def stable_id(phase, model, architecture, workload, qps, repeat):
    logical = {"schema": 1, "rq": "RQ1", "phase": phase, "model": model,
               "architecture": architecture, "workload": workload,
               "qps": qps, "repeat": repeat}
    digest = hashlib.sha256(json.dumps(logical, sort_keys=True,
                            separators=(",", ":")).encode()).hexdigest()[:16]
    return f"rq1-{phase}-{digest}"


def configs():
    models = load(ROOT / "configs/models.json")["models"]
    workloads = load(ROOT / "configs/workloads.json")
    matrix = load(ROOT / "configs/matrix.json")
    cluster = load(ROOT / "configs/cluster.json")
    return models, workloads, matrix, cluster


def select_node(cluster, node):
    matches = [item for item in cluster["nodes"] if item["name"] == node]
    if not matches:
        raise ValueError(f"unknown --node {node!r}; available: " +
                         ", ".join(x["name"] for x in cluster["nodes"]))
    selected = dict(cluster); selected["nodes"] = matches
    return selected


def make_point(model, architecture, architecture_config, workloads, qps, node):
    point = dict(architecture_config["point"])
    point.update({"id": f"rq1-{model}-{architecture}-{node}", "rq": "RQ1",
                  "model": model, "workloads": list(workloads), "qps": list(qps),
                  "energy_scope": "plan_used_gpus", "request_timeout_s": 1800,
                  "health_timeout_s": 1800, "max_inflight": 64,
                  "metadata": {"measurement": "RQ1", "node": node,
                               "deployment": architecture_config["label"]}})
    return point


def expand(node, phase="formal", model_filter=(), architecture_filter=(),
           workload_filter=(), qps_filter=(), repeat_filter=()):
    models, workloads, matrix, cluster = configs(); select_node(cluster, node)
    wanted = lambda value, values: not values or value in values
    phase_workloads = ([matrix["canary_workload"]] if phase == "canary"
                       else workloads["classes"])
    phase_qps = ([matrix["canary_qps"]] if phase == "canary" else QPS_ORDER)
    phase_repeats = ([1] if phase == "canary" else matrix["repeats"])
    rows = []
    for model in models:
        if not wanted(model, model_filter): continue
        for architecture in ARCH_ORDER:
            if not wanted(architecture, architecture_filter): continue
            point = make_point(model, architecture, matrix["architectures"][architecture],
                               phase_workloads, phase_qps, node)
            for workload in phase_workloads:
                if not wanted(workload, workload_filter): continue
                for qps in phase_qps:
                    if not wanted(qps, qps_filter): continue
                    for repeat in phase_repeats:
                        if not wanted(repeat, repeat_filter): continue
                        rows.append({"phase": phase, "model_id": model,
                                     "model": models[model], "architecture": architecture,
                                     "point": point, "workload": workload, "qps": qps,
                                     "repeat": repeat,
                                     "run_id": stable_id(phase, model, architecture,
                                                         workload, qps, repeat)})
    return rows


def initial_progress(node):
    return {"schema_version": 1, "rq": "RQ1", "node": node,
            "created_at": utcnow(), "updated_at": utcnow(), "runs": {}}


def load_progress(path, node):
    value = load(path) if Path(path).exists() else initial_progress(node)
    if value.get("node") != node:
        raise ValueError(f"progress belongs to node {value.get('node')!r}, not {node!r}")
    return value


def inventory_row(item, status="pending"):
    return {"run_id": item["run_id"], "phase": item["phase"],
            "model": item["model_id"], "model_path": item["model"]["path"],
            "architecture": item["architecture"], "workload": item["workload"],
            "qps": item["qps"], "repeat": item["repeat"], "status": status,
            "attempts": []}


def verify_model(executor, host, path):
    import shlex
    result = executor.run(host, f"test -e {shlex.quote(path)}", check=False,
                          quiet=True, timeout=20)
    return result.returncode == 0


def sla_evaluation(summary, sla):
    total = int(summary.get("requests_total", 0)); success = int(summary.get("requests_success", 0))
    ratio = success / total if total else 0.0
    metrics = summary.get("metrics", {})
    values = {"success_rate": ratio,
              "achieved_qps_ratio": float(summary.get("achieved_qps", 0)) /
                                    float(summary.get("target_qps", 1) or 1),
              "ttft_p90_ms": metrics.get("ttft_client_ms", {}).get("p90", float("inf")),
              "tpot_p90_ms": metrics.get("tpot_ms", {}).get("p90", float("inf")),
              "e2e_p90_ms": metrics.get("e2e_ms", {}).get("p90", float("inf"))}
    passed = (values["success_rate"] >= sla["min_success_rate"] and
              values["achieved_qps_ratio"] >= sla["min_achieved_qps_ratio"] and
              all(values[name] <= sla[name] for name in
                  ("ttft_p90_ms", "tpot_p90_ms", "e2e_p90_ms")))
    return passed, values


def canary_gate(progress, expected=18, models=None):
    selected = set(models or ())
    canaries = [r for r in progress["runs"].values() if r["phase"] == "canary" and (not selected or r["model"] in selected)]
    valid = [r for r in canaries if r["status"] == "valid"]
    return {"open": len(valid) == expected, "valid": len(valid),
            "expected": expected, "observed": len(canaries)}


def saturated(progress, item, threshold):
    for qps in QPS_ORDER:
        if qps >= item["qps"]: break
        rid = stable_id(item["phase"], item["model_id"], item["architecture"],
                        item["workload"], qps, item["repeat"])
        prior = progress["runs"].get(rid)
        if prior and prior.get("status") == "valid" and \
           prior.get("sla", {}).get("achieved_qps_ratio", 1) < threshold:
            return prior["run_id"]
    return None


def metrics(summary, sla_values):
    energy = summary.get("energy", {})
    return {"achieved_qps": summary.get("achieved_qps"),
            "input_tokens_s": summary.get("input_throughput_tokens_s"),
            "output_tokens_s": summary.get("output_throughput_tokens_s"),
            "all_tokens_s": summary.get("all_throughput_tokens_s"),
            "total_energy_j": energy.get("total_j"),
            "average_power_w": summary.get("average_cluster_power_w"),
            "energy_per_input_token_j": summary.get("energy_per_input_token_j"),
            "energy_per_output_token_j": summary.get("energy_per_output_token_j"),
            "energy_per_request_j": summary.get("energy_per_successful_request_j"),
            "input_tokens_per_j": summary.get("input_tokens_per_j"),
            "output_tokens_per_j": summary.get("output_tokens_per_j"), **sla_values}


def aggregate(progress):
    groups = {}
    for row in progress["runs"].values():
        if row["phase"] != "formal" or row["status"] != "valid": continue
        key = (row["model"], row["architecture"], row["workload"], row["qps"])
        groups.setdefault(key, []).append(row["metrics"])
    output = []
    for key, rows in sorted(groups.items()):
        numeric = set.intersection(*(set(k for k,v in row.items() if isinstance(v,(int,float))) for row in rows))
        summary = {name: {"mean": statistics.fmean(r[name] for r in rows),
                          "sample_std": statistics.stdev(r[name] for r in rows) if len(rows)>1 else None}
                   for name in sorted(numeric)}
        output.append({"model": key[0], "architecture": key[1], "workload": key[2],
                       "qps": key[3], "repeats": len(rows), "metrics": summary})
    return output
