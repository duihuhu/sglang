#!/usr/bin/env python3
"""Build and atomically update per-GPU-count measurement datasets (local I/O only)."""
import argparse
import copy
import hashlib
import json
import os
import tempfile
from pathlib import Path

RQ2_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = RQ2_ROOT / "data" / "summary"
MODEL = "Qwen3-30B-A3B"
WORKLOADS = [
    "measurement_qa_lpld", "measurement_chatbot_lphd",
    "measurement_balanced_mpmd", "measurement_rag_hpld",
    "measurement_summary_hphd", "measurement_longcontext",
]


def load_json(path):
    return json.loads(Path(path).read_text())


def atomic_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, indent=2, sort_keys=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, path)
        dir_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def config_summary(config_path, nodes, qps, total_gpus):
    config = load_json(config_path)
    points = [p for p in config["points"] if p.get("nodes") == nodes and p.get("gpus_total") == total_gpus]
    if not points:
        raise ValueError(f"no matching {nodes}-node/{total_gpus}-GPU points in {config_path}")
    if any(p.get("qps") != [qps] for p in points):
        raise ValueError(f"QPS mismatch in {config_path}")
    return {
        "nodes": nodes,
        "offered_qps": qps,
        "topology": f"rdma{nodes}n",
        "user_override": bool(config.get("metadata", {}).get("user_override_total_gpus_2", False)),
        "workloads": list(points[0]["workloads"]),
        "repeats": config.get("metadata", {}).get("repeats", 3),
        "architectures": sorted(p["architecture"] for p in points),
        "point_definitions": points,
        "config_artifact": str(Path(config_path)),
    }


def experiment_summary(config):
    deployments = {}
    for point in config["point_definitions"]:
        architecture = point["architecture"]
        metadata = point.get("metadata", {})
        if metadata.get("deployment"):
            deployments[architecture] = metadata["deployment"]
        elif architecture == "native":
            deployments[architecture] = {"native_placements": point.get("native_placements", [])}
        elif architecture == "pd":
            deployments[architecture] = {
                "prefill_placements": point.get("prefill_placements", []),
                "decode_placements": point.get("decode_placements", []),
            }
        elif architecture == "af":
            deployments[architecture] = {
                "ffn_placements": point.get("ffn_placements", []),
                "attention_placements": point.get("attention_placements", []),
            }
    return {
        "deployment_by_architecture": deployments,
        "metric_aggregation": "mean and sample_std across 3 independent repeats; sample_std uses n-1",
        "workloads": {"reference": "data_dictionary.workload_definitions", "ids": list(config["workloads"])},
    }


def validate_report(report, report_path):
    if report.get("complete") is not True:
        raise ValueError(f"report is not complete: {report_path}")
    groups = report.get("groups", [])
    if len(groups) != 18 or report.get("expected_valid_runs") != 54:
        raise ValueError(f"expected 18 groups and 54 valid runs: {report_path}")
    valid_runs = report.get("valid_runs", 54)
    if valid_runs != 54:
        raise ValueError(f"expected valid_runs=54: {report_path}")
    keys = {(g.get("architecture"), g.get("workload")) for g in groups}
    expected = {(a, w) for a in ("af", "native", "pd") for w in WORKLOADS}
    if keys != expected or any(g.get("repeats") != 3 or len(g.get("artifacts", [])) != 3 for g in groups):
        raise ValueError(f"group/repeat/artifact coverage mismatch: {report_path}")


def completed_experiment(experiment_id, report_path, config_path, nodes, qps, total_gpus):
    report_path = Path(report_path)
    report = load_json(report_path)
    validate_report(report, report_path)
    config = config_summary(config_path, nodes, qps, total_gpus)
    return {
        "status": "completed",
        "configuration": config,
        "validation": {
            "complete": True, "groups": 18, "repeats_per_group": 3,
            "valid_runs": 54, "expected_valid_runs": 54,
            "metrics_exactly_embedded_from_source_report": True,
        },
        "aggregate_results": report,
        "raw_runs": {
            "representation": "source artifact references",
            "groups": copy.deepcopy(report["groups"]),
        },
        "summary": experiment_summary(config),
        "comparisons": {},
        "key_findings": [],
        "limitations": ["This dataset preserves measured aggregates and artifact references; it does not infer cross-experiment conclusions."],
        "reproducibility": {
            "source_reports": [str(report_path)],
            "config": str(Path(config_path)),
            "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        },
    }


def pending_experiment(results_dir, config_path, nodes=2, qps=4, total_gpus=4):
    results_dir = Path(results_dir)
    progress_path = results_dir / "progress.json"
    progress = load_json(progress_path) if progress_path.is_file() else {}
    valid = progress.get("valid_runs", 0)
    expected = progress.get("expected_valid_runs", 54)
    state = progress.get("state", progress.get("status", "pending"))
    return {
        "status": "running" if state in {"formal", "canary", "preflight"} else "pending",
        "configuration": config_summary(config_path, nodes, qps, total_gpus),
        "validation": {"complete": False, "included_in_completed_aggregates": False},
        "aggregate_results": None,
        "raw_runs": {"representation": "withheld until final report is complete"},
        "comparisons": {}, "key_findings": [],
        "limitations": ["In-progress metrics are intentionally excluded."],
        "progress": {"state": state, "valid_runs": valid, "expected_valid_runs": expected, "updated_at": progress.get("updated_at")},
        "reproducibility": {
            "source_reports": [], "results_directory": str(results_dir),
            "progress_artifact": str(progress_path), "config": str(Path(config_path)),
        },
    }


def dataset(total_gpus, experiments, data_dictionary=None):
    value = {
        "schema": {"name": "aflex_measurement_by_communication_and_total_gpus", "version": 3},
        "communication": "rdma", "total_gpus": total_gpus, "model": MODEL,
        "experiments": experiments,
        "comparisons": {}, "key_findings": [],
        "limitations": ["Only completed experiments contribute aggregate_results; running experiments are status references only."],
        "reproducibility": {"source_reports": sorted({p for e in experiments.values() for p in e["reproducibility"]["source_reports"]})},
    }
    if data_dictionary is not None:
        value["data_dictionary"] = copy.deepcopy(data_dictionary)
    return value


def validate_dataset(value, expected_total_gpus):
    required = {"schema", "communication", "total_gpus", "model", "experiments", "comparisons", "key_findings", "limitations", "reproducibility"}
    if not required <= value.keys() or value["communication"] != "rdma" or value["total_gpus"] != expected_total_gpus:
        raise ValueError("dataset schema/identity validation failed")
    for experiment_id, experiment in value["experiments"].items():
        cfg = experiment["configuration"]
        if cfg["offered_qps"] not in (2, 4) or expected_total_gpus not in {p["gpus_total"] for p in cfg["point_definitions"]}:
            raise ValueError(f"GPU/QPS mismatch: {experiment_id}")
        if experiment["status"] == "completed":
            validate_report(experiment["aggregate_results"], experiment_id)
    return True


def append_completed(target, experiment_id, report, config, nodes, qps, total_gpus=4, dry_run=False):
    target = Path(target)
    current = load_json(target)
    validate_dataset(current, total_gpus)
    updated = copy.deepcopy(current)
    updated["experiments"][experiment_id] = completed_experiment(experiment_id, report, config, nodes, qps, total_gpus)
    updated["reproducibility"]["source_reports"] = sorted({p for e in updated["experiments"].values() for p in e["reproducibility"]["source_reports"]})
    validate_dataset(updated, total_gpus)
    if not dry_run:
        atomic_write(target, updated)
    return updated


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Atomically append a completed experiment to a per-GPU-count dataset")
    p.add_argument("--target", type=Path, default=DATA_DIR / "rdma_4gpu_qwen3_30b_a3b.json")
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--experiment-id", required=True)
    p.add_argument("--nodes", type=int, required=True)
    p.add_argument("--qps", type=int, required=True)
    p.add_argument("--total-gpus", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    a = parse_args(argv)
    value = append_completed(a.target, a.experiment_id, a.report, a.config, a.nodes, a.qps, a.total_gpus, a.dry_run)
    print(json.dumps({"target": str(a.target), "experiment": a.experiment_id, "status": "validated" if a.dry_run else "updated", "experiments": sorted(value["experiments"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
