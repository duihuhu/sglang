from __future__ import annotations

import json
import re
import time
import unicodedata
import urllib.request
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_json, validate_cluster
from .deploy import RemoteExecutor, build_plan, execute_lifecycle
from .deploy.base import DeploymentHandle, RoutingPolicy

QUESTIONS = (
    {"id": "arithmetic", "prompt": "Answer with only the short answer: What is 2 + 3?", "aliases": ("5", "five"), "regex": r"(?:^|\b)(?:5|five)(?:\b|$)"},
    {"id": "france_capital", "prompt": "Answer with only the short answer: What is the capital of France?", "aliases": ("paris",), "regex": r"\bparis\b"},
    {"id": "water_formula", "prompt": "Answer with only the short answer: What is the chemical formula for water?", "aliases": ("h2o", "h₂o"), "regex": r"\bh\s*2\s*o\b"},
    {"id": "red_planet", "prompt": "Answer with only the short answer: Which planet is known as the Red Planet?", "aliases": ("mars",), "regex": r"\bmars\b"},
    {"id": "days_week", "prompt": "Answer with only the short answer: How many days are in a week?", "aliases": ("7", "seven"), "regex": r"(?:^|\b)(?:7|seven)(?:\b|$)"},
    {"id": "largest_ocean", "prompt": "Answer with only the short answer: What is the largest ocean on Earth?", "aliases": ("pacific", "pacific ocean", "the pacific ocean"), "regex": r"\bpacific(?: ocean)?\b"},
    {"id": "author_1984", "prompt": "Answer with only the short answer: Who wrote the novel 1984?", "aliases": ("george orwell", "orwell"), "regex": r"\b(?:george\s+)?orwell\b"},
    {"id": "triangle_sides", "prompt": "Answer with only the short answer: How many sides does a triangle have?", "aliases": ("3", "three"), "regex": r"(?:^|\b)(?:3|three)(?:\b|$)"},
)
SAMPLING_PARAMS = {"temperature": 0.0, "max_new_tokens": 48}
SEMANTIC_AF_PORT_BASE = 12000
SEMANTIC_AF_COORDINATOR_PORT = 19308


def load_reused_baseline(path: Path, *, expected_model: str | None = None) -> list[dict[str, Any]]:
    source = Path(path)
    if source.is_dir():
        source = source / "semantic_results.json"
    try:
        payload = json.loads(source.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"reused baseline file does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"reused baseline is not valid JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("reused baseline must be a schema_version 1 semantic results object")
    if expected_model is not None and payload.get("model") != expected_model:
        raise ValueError(
            f"reused baseline model mismatch: expected {expected_model!r}, got {payload.get('model')!r}"
        )
    if payload.get("sampling_params") != SAMPLING_PARAMS:
        raise ValueError("reused baseline sampling_params do not match this semantic validation")
    systems = payload.get("systems")
    rows = systems.get("baseline") if isinstance(systems, dict) else None
    if not isinstance(rows, list):
        raise TypeError("reused baseline systems.baseline must be a list")
    expected_ids = [question["id"] for question in QUESTIONS]
    actual_ids = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise TypeError(f"reused baseline row {index} must be an object")
        question_id = row.get("question_id")
        if not isinstance(question_id, str):
            raise TypeError(f"reused baseline row {index} has invalid question_id")
        actual_ids.append(question_id)
        for key in ("api_success", "semantic_pass"):
            if not isinstance(row.get(key), bool):
                raise TypeError(f"reused baseline row {question_id!r} has invalid {key}")
        if row["api_success"] and not isinstance(row.get("normalized_text"), str):
            raise ValueError(f"reused baseline row {question_id!r} lacks normalized_text")
        output_ids = row.get("output_ids")
        if output_ids is not None and (not isinstance(output_ids, list) or
                                       any(isinstance(item, bool) or not isinstance(item, int)
                                           for item in output_ids)):
            raise ValueError(f"reused baseline row {question_id!r} has invalid output_ids")
    if len(actual_ids) != len(set(actual_ids)) or set(actual_ids) != set(expected_ids):
        raise ValueError(
            "reused baseline must contain exactly one row for every semantic question; "
            f"expected {expected_ids}, got {actual_ids}"
        )
    by_id = {row["question_id"]: row for row in rows}
    return [deepcopy(by_id[question_id]) for question_id in expected_ids]


def port_conflict_snapshot(plan) -> dict[str, Any]:
    entries = []
    for process in plan.processes:
        values = [(process.port, "http")]
        values += [(process.bootstrap_port, "bootstrap"), (process.nccl_port, "nccl")]
        values += [(port, "internal") for port in process.internal_ports]
        entries.extend({"host": process.host, "port": port, "kind": kind,
                        "role": process.role} for port, kind in values if port is not None)
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for entry in entries:
        grouped.setdefault((entry["host"], entry["port"]), []).append(entry)
    collisions = [items for items in grouped.values() if len(items) > 1]
    by_host: dict[str, list[int]] = {}
    for host, port in grouped:
        by_host.setdefault(host, []).append(port)
    checks = {host: ("busy=; for p in " + " ".join(map(str, sorted(ports))) +
                     "; do ss -ltnpH \"sport = :$p\" 2>/dev/null && busy=1; done; test -z \"$busy\"")
              for host, ports in by_host.items()}
    return {"unique": not collisions, "port_count": len(grouped),
            "ports_by_host": {host: sorted(ports) for host, ports in by_host.items()},
            "collisions": collisions, "free_check_commands": checks}


def normalize_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", "" if value is None else str(value)).casefold()
    text = re.sub(r"[^\w]+", " ", text, flags=re.UNICODE)
    return " ".join(text.split())


def expected_match(text: str, question: dict[str, Any]) -> bool:
    normalized = normalize_text(text)
    aliases = {normalize_text(alias) for alias in question.get("aliases", ())}
    return normalized in aliases or re.search(question["regex"], normalized, flags=re.IGNORECASE) is not None


def _first_text(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        pieces = [item for item in (_first_text(item) for item in value) if item is not None]
        return "".join(pieces) if pieces else None
    if isinstance(value, dict):
        for key in ("text", "generated_text", "output_text", "content", "response"):
            found = _first_text(value.get(key))
            if found is not None:
                return found
        for key in ("choices", "outputs", "data", "message"):
            found = _first_text(value.get(key))
            if found is not None:
                return found
    return None


def parse_generate_response(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("/generate response must be a JSON object")
    text = _first_text(payload)
    if text is None:
        raise ValueError("/generate response does not contain generated text")
    meta = payload.get("meta_info")
    if not isinstance(meta, dict):
        meta = {}
    output_ids = meta.get("output_ids", payload.get("output_ids"))
    if not isinstance(output_ids, list):
        output_ids = None
    token_count = None
    for key in ("completion_tokens", "output_token_count", "output_tokens", "num_output_tokens"):
        value = meta.get(key, payload.get(key))
        if isinstance(value, int) and not isinstance(value, bool):
            token_count = value
            break
    if token_count is None and output_ids is not None:
        token_count = len(output_ids)
    return {"text": text, "meta_info": meta, "output_ids": output_ids, "output_token_count": token_count}


def request_question(endpoint: str, question: dict[str, Any], timeout_s: float = 180) -> dict[str, Any]:
    payload = {"text": question["prompt"], "sampling_params": dict(SAMPLING_PARAMS), "stream": False}
    started = time.monotonic()
    row = {"question_id": question["id"], "prompt": question["prompt"],
           "expected_aliases": list(question["aliases"]), "expected_regex": question["regex"],
           "endpoint": endpoint, "request": payload, "api_success": False,
           "semantic_pass": False, "response_json": None, "text": None,
           "normalized_text": None, "meta_info": {}, "output_ids": None,
           "output_token_count": None, "error": None}
    try:
        request = urllib.request.Request(endpoint.rstrip("/") + "/generate",
            data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(request, timeout=float(timeout_s)) as response:
            raw = response.read().decode(errors="replace")
            status = getattr(response, "status", response.getcode())
        parsed_json = json.loads(raw)
        row["http_status"] = status
        row["response_json"] = parsed_json
        if status < 200 or status >= 300:
            raise RuntimeError(f"HTTP status {status}")
        parsed = parse_generate_response(parsed_json)
        row.update(parsed)
        row["normalized_text"] = normalize_text(parsed["text"])
        row["api_success"] = True
        row["semantic_pass"] = expected_match(parsed["text"], question)
    except Exception as exc:  # noqa: BLE001 - each question must be isolated
        row["error"] = repr(exc)
    row["latency_ms"] = (time.monotonic() - started) * 1000
    return row


def summarize(results: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"criteria": {
        "api_success": "HTTP/JSON response parsed successfully",
        "semantic_expected": "answer matches the question's aliases or regex",
        "deterministic_match": "Native and AF normalized text are exactly equal; token IDs are compared separately when both are available",
    }, "systems": {}}
    for name, rows in results.items():
        summary["systems"][name] = {
            "questions": len(rows), "api_pass": sum(bool(r.get("api_success")) for r in rows),
            "semantic_pass": sum(bool(r.get("semantic_pass")) for r in rows),
            "all_api_pass": bool(rows) and all(r.get("api_success") for r in rows),
            "all_semantic_pass": bool(rows) and all(r.get("semantic_pass") for r in rows),
        }
    baseline = {row["question_id"]: row for row in results.get("baseline", [])}
    af = {row["question_id"]: row for row in results.get("af", [])}
    comparisons = []
    for question in QUESTIONS:
        left, right = baseline.get(question["id"]), af.get(question["id"])
        if left is None or right is None:
            continue
        text_exact = (left.get("api_success") and right.get("api_success") and
                      left.get("normalized_text") == right.get("normalized_text"))
        left_ids, right_ids = left.get("output_ids"), right.get("output_ids")
        token_available = left_ids is not None and right_ids is not None
        comparisons.append({"question_id": question["id"], "normalized_text_exact": bool(text_exact),
                            "token_ids_available": token_available,
                            "token_ids_exact": left_ids == right_ids if token_available else None})
    summary["comparisons"] = comparisons
    summary["determinism"] = {
        "paired_questions": len(comparisons),
        "normalized_text_exact": sum(row["normalized_text_exact"] for row in comparisons),
        "all_normalized_text_exact": bool(comparisons) and all(row["normalized_text_exact"] for row in comparisons),
        "token_ids_comparable": sum(row["token_ids_available"] for row in comparisons),
        "token_ids_exact": sum(row["token_ids_exact"] is True for row in comparisons),
        "all_available_token_ids_exact": (all(row["token_ids_exact"] is True for row in comparisons if row["token_ids_available"])
                                          if any(row["token_ids_available"] for row in comparisons) else None),
    }
    return summary


def load_semantic_bundle(root: Path) -> dict[str, Any]:
    cluster = load_json(root / "configs/cluster_operator_node34.json")
    validate_cluster(cluster)
    models = load_json(root / "configs/models.json")["models"]
    af_matrix = load_json(root / "configs/node34_af_a8f8_tp1_shared_smoke_qps8.json")
    af_point = deepcopy(af_matrix["points"][0])
    af_point.update({"id": "node34-af-a8f8-tp1-semantic", "pf_capacity": 8,
                     "afd_port_base": SEMANTIC_AF_PORT_BASE,
                     "afd_coordinator_port": SEMANTIC_AF_COORDINATOR_PORT,
                     "warmup_parallel": True, "skip_warmup": False,
                     "wait_after_warmup_s": 0, "requires_preflight": False})
    baseline_point = {"id": "node34-native-pair-tp8-semantic", "architecture": "native",
                      "recipe": "legacy_native_pair_tp8", "nodes": 2, "tp": 8,
                      "warmup_parallel": True, "request_timeout_s": 180}
    return {"cluster": cluster, "model": models["dense_qwen3_32b"]["path"],
            "points": {"baseline": baseline_point, "af": af_point}}


def build_semantic_plans(root: Path, selected: tuple[str, ...], run_tag: str | None = None):
    bundle = load_semantic_bundle(root)
    plans = {}
    tag = run_tag or datetime.now(timezone.utc).strftime("semantic-%Y%m%dT%H%M%S%fZ")
    for name in selected:
        plan = build_plan(bundle["cluster"], bundle["model"], bundle["points"][name],
                          run_tag=f"{tag}-{name}")
        if name == "baseline":
            plan.endpoints = [f"http://{process.host}:{process.port}" for process in plan.processes
                              if process.role == "NATIVE"]
            plan.routing_policy = RoutingPolicy.CLIENT_ROUND_ROBIN
            plan.validate()
        plans[name] = plan
    return bundle, plans


def execute_semantic_validation(root: Path, results_dir: Path, selected: tuple[str, ...],
                                *, reuse_baseline: Path | None = None,
                                run_tag: str | None = None) -> dict[str, Any]:
    bundle, plans = build_semantic_plans(root, selected, run_tag=run_tag)
    executor = RemoteExecutor(bundle["cluster"]["container"], False)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    if reuse_baseline is not None:
        if selected != ("af",):
            raise ValueError("reuse_baseline is only valid when executing AF alone")
        all_rows["baseline"] = load_reused_baseline(
            reuse_baseline, expected_model=bundle["model"]
        )
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in selected:
        plan = plans[name]
        def body(target: DeploymentHandle, _name=name):
            return [request_question(target.endpoint_for_request(index), question,
                                     bundle["points"][_name].get("request_timeout_s", 180))
                    for index, question in enumerate(QUESTIONS)]
        try:
            all_rows[name] = execute_lifecycle(plan, executor, body,
                artifact_dir=results_dir / name, runtime_options=plan.runtime_options)
        except Exception as exc:  # noqa: BLE001 - persist deployment failures
            all_rows[name] = [{"question_id": q["id"], "prompt": q["prompt"],
                               "api_success": False, "semantic_pass": False,
                               "error": f"deployment: {exc!r}"} for q in QUESTIONS]
    artifact = {"schema_version": 1, "model": bundle["model"],
                "sampling_params": SAMPLING_PARAMS, "systems": all_rows,
                "execution_order": list(selected),
                "reused_baseline": str(reuse_baseline) if reuse_baseline else None,
                "port_conflict_snapshots": {
                    name: port_conflict_snapshot(plan) for name, plan in plans.items()
                }}
    summary = summarize(all_rows)
    (results_dir / "semantic_results.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (results_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary
