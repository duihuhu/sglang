#!/usr/bin/env python3
"""Strictly serial orchestration for metadata-annotated communication matrices."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.config import load_bundle
from aflex_benchmark.runner import execute_item, expand_queue, run_id

DEFAULT_MATRIX = ROOT / "configs/comm_ablation_2gpu_qps2.json"
PHASES = ("smoke", "qps2")
SCHEMA_VERSION = 1


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _metadata(point: dict[str, Any]) -> dict[str, Any]:
    value = point.get("metadata", {})
    return value if isinstance(value, dict) else {}


def point_phase(point: dict[str, Any]) -> str:
    """Read phase only from point metadata; do not infer recipes or topology."""
    metadata = _metadata(point)
    values = [point.get("comm_ablation_phase"), point.get("ablation_phase"),
              metadata.get("comm_ablation_phase"), metadata.get("ablation_phase"),
              metadata.get("phase")]
    phases = {value.lower() for value in values if isinstance(value, str)}
    tags = metadata.get("tags", point.get("tags", []))
    if isinstance(tags, list):
        phases.update(str(tag).lower() for tag in tags if str(tag).lower() in PHASES)
    phases &= set(PHASES)
    if len(phases) != 1:
        raise ValueError(
            f"point {point.get('id', '<unknown>')} must declare exactly one "
            "smoke/qps2 phase in point metadata"
        )
    return phases.pop()


def point_key(point: dict[str, Any]) -> str:
    metadata = _metadata(point)
    value = (point.get("comm_ablation_id") or point.get("ablation_id") or
             metadata.get("comm_ablation_id") or metadata.get("ablation_id"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            f"point {point.get('id', '<unknown>')} lacks comm_ablation_id metadata"
        )
    return value.strip()


def validate_matrix(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    matrix = bundle.get("matrix", {})
    points = matrix.get("points")
    if not isinstance(points, list) or not points:
        raise ValueError("communication ablation matrix must contain at least one point")
    expected = matrix.get("metadata", {}).get("expected_points", len(points))
    if isinstance(expected, bool) or not isinstance(expected, int) or expected < 1:
        raise ValueError("matrix metadata.expected_points must be a positive integer")
    if len(points) != expected:
        raise ValueError(f"communication ablation matrix must contain exactly {expected} points")
    ids = [point.get("id") for point in points]
    if any(not isinstance(value, str) or not value for value in ids):
        raise ValueError("every communication ablation point must have a non-empty id")
    if len(set(ids)) != len(ids):
        raise ValueError("communication ablation point ids must be unique")
    for point in points:
        phase = point_phase(point)
        point_key(point)
        if phase == "qps2" and 2 not in point.get("qps", []):
            raise ValueError(f"qps2 point {point['id']} must include QPS 2")
        if phase == "smoke" and not point.get("smoke", False):
            raise ValueError(f"smoke point {point['id']} must set smoke=true")
    return points


def _contains_invalid_link(value: Any) -> bool:
    if isinstance(value, str):
        normalized = value.lower().replace("-", "_").replace(" ", "_")
        return "invalid_link" in normalized
    if isinstance(value, dict):
        return any(_contains_invalid_link(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_invalid_link(item) for item in value)
    return False


def classify_result(summary: dict[str, Any], phase: str) -> tuple[str, str | None]:
    if _contains_invalid_link(summary):
        return "invalid_link", "invalid_link"
    if summary.get("status") != "complete":
        return "failed", str(summary.get("error") or summary.get("status") or "failed")
    if phase == "smoke":
        if int(summary.get("requests_success", 0)) < 1:
            return "failed", "smoke requires at least one successful request"
        if int(summary.get("completion_tokens", 0)) < 1:
            return "failed", "smoke requires at least one complete output token"
    return "complete", None


def build_runs(bundle: dict[str, Any], phase: str, repeats: int) -> list[dict[str, Any]]:
    validate_matrix(bundle)
    requested = PHASES if phase == "all" else (phase,)
    queue = expand_queue(bundle, allow_experimental=True)
    runs: list[dict[str, Any]] = []
    seen: set[tuple[str, str, Any]] = set()
    for selected_phase in requested:
        matching = [item for item in queue if point_phase(item["point"]) == selected_phase]
        for item in matching:
            identity = (item["point"]["id"], item["workload"], item["qps"])
            if identity in seen:
                continue
            seen.add(identity)
            count = 1 if selected_phase == "smoke" else repeats
            for repeat in range(count):
                current = copy.deepcopy(item)
                current.update({"phase": selected_phase, "repeat": repeat,
                                "ablation_id": point_key(current["point"])})
                current["run_id"] = run_id(current["point"], current["workload"],
                                           current["qps"], repeat)
                runs.append(current)
    return runs


def _new_manifest(matrix: Path, phase: str, repeats: int,
                  runs: list[dict[str, Any]]) -> dict[str, Any]:
    now = _utc_now()
    return {"schema_version": SCHEMA_VERSION, "matrix": str(matrix), "phase": phase,
            "repeats": repeats, "created_at": now, "updated_at": now,
            "execution": "strictly_serial", "semantic_smoke": "extensible_via_semantics_module",
            "runs": [{"run_id": item["run_id"], "point_id": item["point"]["id"],
                      "ablation_id": item["ablation_id"], "phase": item["phase"],
                      "repeat": item["repeat"], "workload": item["workload"],
                      "qps": item["qps"], "status": "pending"} for item in runs]}


def _load_resume(path: Path, matrix: Path, phase: str, repeats: int,
                 runs: list[dict[str, Any]]) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    expected = [(item["run_id"], item["phase"]) for item in runs]
    actual = [(item.get("run_id"), item.get("phase")) for item in manifest.get("runs", [])]
    if (manifest.get("schema_version") != SCHEMA_VERSION or
            manifest.get("matrix") != str(matrix) or manifest.get("phase") != phase or
            manifest.get("repeats") != repeats or actual != expected):
        raise ValueError("existing run_manifest.json does not match this invocation")
    return manifest


def _status(manifest: dict[str, Any], running: bool = False) -> dict[str, Any]:
    counts: dict[str, int] = {}
    for row in manifest["runs"]:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    terminal = {"complete", "failed", "invalid_link", "skipped_invalid_link"}
    finished = all(row["status"] in terminal for row in manifest["runs"])
    return {"status": "running" if running else "complete" if finished else "planned",
            "updated_at": _utc_now(), "counts": counts,
            "runs_total": len(manifest["runs"]),
            "runs_finished": sum(row["status"] in terminal for row in manifest["runs"])}


def orchestrate(bundle: dict[str, Any], matrix: Path, results: Path, phase: str,
                repeats: int, execute: bool, resume: bool = False,
                executor: Callable[..., dict[str, Any]] = execute_item) -> dict[str, Any]:
    runs = build_runs(bundle, phase, repeats)
    results.mkdir(parents=True, exist_ok=True)
    manifest_path = results / "run_manifest.json"
    status_path = results / "status.json"
    if resume and manifest_path.exists():
        manifest = _load_resume(manifest_path, matrix, phase, repeats, runs)
    elif manifest_path.exists():
        raise FileExistsError(f"{manifest_path} exists; pass --resume or choose another results directory")
    else:
        manifest = _new_manifest(matrix, phase, repeats, runs)
        _write_json(manifest_path, manifest)
    if not execute:
        _write_json(status_path, _status(manifest))
        return manifest

    rows = {row["run_id"]: row for row in manifest["runs"]}
    invalid_keys = {row["ablation_id"] for row in manifest["runs"]
                    if row["status"] == "invalid_link"}
    terminal = {"complete", "failed", "invalid_link", "skipped_invalid_link"}
    _write_json(status_path, _status(manifest, running=True))
    for item in runs:
        row = rows[item["run_id"]]
        if row["status"] in terminal:
            continue
        if item["phase"] == "qps2" and item["ablation_id"] in invalid_keys:
            row.update({"status": "skipped_invalid_link", "finished_at": _utc_now()})
        else:
            row.update({"status": "running", "started_at": _utc_now(),
                        "result_dir": str(results / item["run_id"])})
            manifest["updated_at"] = _utc_now()
            _write_json(manifest_path, manifest)
            _write_json(status_path, _status(manifest, running=True))
            try:
                summary = executor(item, bundle, results)
            except Exception as exc:  # noqa: BLE001 - persist and continue every point
                summary = {"status": "failed", "error": repr(exc)}
            state, reason = classify_result(summary, item["phase"])
            row.update({"status": state, "reason": reason,
                        "summary_status": summary.get("status"), "finished_at": _utc_now()})
            if state == "invalid_link":
                invalid_keys.add(item["ablation_id"])
        manifest["updated_at"] = _utc_now()
        _write_json(manifest_path, manifest)
        _write_json(status_path, _status(manifest, running=True))
    _write_json(status_path, _status(manifest))
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--cluster", type=Path)
    parser.add_argument("--results", type=Path, default=ROOT / "results/comm_ablation")
    parser.add_argument("--phase", choices=("smoke", "qps2", "all"), default="all")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if not args.matrix.is_file():
        parser.error(f"matrix does not exist: {args.matrix}")
    try:
        bundle = load_bundle(ROOT / "configs", args.matrix, args.cluster)
        manifest = orchestrate(bundle, args.matrix, args.results, args.phase,
                               args.repeats, args.execute, args.resume)
    except (ValueError, FileExistsError) as exc:
        parser.error(str(exc))
    print(json.dumps({"manifest": str(args.results / "run_manifest.json"),
                      "runs": len(manifest["runs"]), "execute": args.execute}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
