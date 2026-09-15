from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

EXPECTED_VALID_POINTS = (
    "native_nvlink",
    "native_pcie",
    "native_rdma",
    "pd_rdma",
    "af_nvlink",
)
EXPECTED_INVALID_POINTS = (
    ("pd_nvlink", "pd_nvlink_custom_pool"),
    ("pd_pcie", "pd_pcie"),
    ("af_pcie", "af_pcie"),
    ("af_rdma", "af_rdma_nic_bound"),
)
PERCENTILES = ("p50", "p90", "p99")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _single_run_dir(repeat_dir: Path) -> Path | None:
    candidates = sorted(p.parent for p in repeat_dir.glob("*/summary.json"))
    if len(candidates) > 1:
        raise ValueError(f"multiple runs found under {repeat_dir}")
    return candidates[0] if candidates else None


def _metric_values(summary: dict[str, Any], link: dict[str, Any]) -> dict[str, float]:
    metrics = summary["metrics"]
    values: dict[str, float] = {
        "qps": float(summary["achieved_qps"]),
        "output_throughput_tokens_s": float(summary["output_throughput_tokens_s"]),
        "total_energy_j": float(summary["energy"]["total_j"]),
        "energy_per_output_token_j": float(summary["energy_per_output_token_j"]),
        "average_cluster_power_w": float(summary["average_cluster_power_w"]),
    }
    for source, label in (
        ("ttft_client_ms", "ttft_ms"),
        ("tpot_ms", "tpot_ms"),
        ("e2e_ms", "e2e_ms"),
    ):
        for percentile in PERCENTILES:
            values[f"{label}_{percentile}"] = float(metrics[source][percentile])
    for path in link.get("paths", []):
        counter = path.get("counter_value")
        if isinstance(counter, (int, float)) and math.isfinite(counter):
            values[f"counter_{path['path']}"] = float(counter)
    return values


def _stats(rows: Iterable[dict[str, float]]) -> dict[str, dict[str, float]]:
    columns: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        for key, value in row.items():
            columns[key].append(value)
    return {
        key: {
            "mean": statistics.fmean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for key, values in sorted(columns.items())
    }


def _relative_to_nvlink(groups: list[dict[str, Any]]) -> None:
    baselines = {
        group["architecture"]: group for group in groups if group["link"] == "nvlink"
    }
    for group in groups:
        baseline = baselines.get(group["architecture"])
        if baseline is None:
            group["relative_to_architecture_nvlink"] = None
            continue
        relative: dict[str, float | None] = {}
        for metric, stat in group["metrics"].items():
            baseline_stat = baseline["metrics"].get(metric)
            denominator = baseline_stat["mean"] if baseline_stat else None
            relative[metric] = stat["mean"] / denominator if denominator else None
        group["relative_to_architecture_nvlink"] = relative


def _invalid_evidence(root: Path, logical_name: str, directory: str) -> dict[str, Any]:
    summaries = sorted((root / directory).rglob("summary.json"))
    if not summaries:
        return {
            "point": logical_name,
            "source": directory,
            "status": "missing",
            "reason": "missing_evidence",
            "backend": None,
            "counter": None,
        }
    summary = _read_json(summaries[-1])
    validation_path = summaries[-1].with_name("link_validation.json")
    validation = (
        _read_json(validation_path)
        if validation_path.exists()
        else summary.get("link_validation", {})
    )
    paths = validation.get("paths", [])
    evidence = []
    for path in paths:
        evidence.append(
            {
                "path": path.get("path"),
                "reason": path.get("reason") or validation.get("reason"),
                "backend": path.get("backend_log"),
                "counter": path.get("counter_value"),
                "counter_supported": path.get("counter_supported"),
                "noise_threshold": path.get("noise_threshold"),
            }
        )
    return {
        "point": logical_name,
        "source": directory,
        "status": summary.get("status"),
        "reason": validation.get("reason") or summary.get("error"),
        "backend": [row["backend"] for row in evidence],
        "counter": [row["counter"] for row in evidence],
        "paths": evidence,
    }


def build_report(
    qps2_root: Path, smoke_root: Path, *, strict_repeats: bool = True
) -> dict[str, Any]:
    groups: list[dict[str, Any]] = []
    repeat_checks: list[dict[str, Any]] = []
    failures: list[str] = []
    for point_name in EXPECTED_VALID_POINTS:
        valid_rows = []
        repeats = []
        architecture, link = point_name.split("_", 1)
        for repeat in range(1, 4):
            repeat_dir = qps2_root / point_name / f"repeat{repeat}"
            run_dir = _single_run_dir(repeat_dir) if repeat_dir.exists() else None
            status = "missing"
            link_status = "missing"
            run_id = None
            if run_dir:
                summary = _read_json(run_dir / "summary.json")
                validation = _read_json(run_dir / "link_validation.json")
                status = summary.get("status", "unknown")
                link_status = validation.get("status", "unknown")
                run_id = summary.get("run_id", run_dir.name)
                if status == "complete" and link_status in {
                    "pass",
                    "verified_semantic",
                }:
                    valid_rows.append(_metric_values(summary, validation))
            repeats.append(
                {
                    "repeat": repeat,
                    "run_id": run_id,
                    "status": status,
                    "link_status": link_status,
                    "valid": status == "complete"
                    and link_status in {"pass", "verified_semantic"},
                }
            )
        complete = len(valid_rows) == 3
        repeat_checks.append(
            {
                "point": point_name,
                "expected": 3,
                "valid_repeats": len(valid_rows),
                "complete": complete,
                "repeats": repeats,
            }
        )
        if not complete:
            failures.append(f"{point_name}: {len(valid_rows)}/3 valid repeats")
        if valid_rows:
            groups.append(
                {
                    "architecture": architecture,
                    "link": link,
                    "runs": len(valid_rows),
                    "metrics": _stats(valid_rows),
                }
            )
    if strict_repeats and failures:
        raise ValueError("incomplete comm ablation repeats: " + "; ".join(failures))
    _relative_to_nvlink(groups)
    invalid = [
        _invalid_evidence(
            smoke_root if logical in {"pd_nvlink", "af_rdma"} else qps2_root,
            logical,
            source,
        )
        for logical, source in EXPECTED_INVALID_POINTS
    ]
    return {
        "schema_version": 1,
        "inputs": {"qps2": str(qps2_root), "smoke": str(smoke_root)},
        "std_definition": "sample standard deviation across repeats (n-1)",
        "repeat_check": {
            "expected_per_point": 3,
            "all_complete": not failures,
            "points": repeat_checks,
        },
        "aggregates": groups,
        "invalid_points": invalid,
        "comparison_note": (
            "Groups are directly comparable across architectures. "
            "Relative values are emitted only when the same architecture "
            "has an observed NVLink baseline; missing baselines remain null."
        ),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Communication ablation report",
        "",
        "## Repeat completeness",
        "",
        "| Point | Valid repeats | Complete |",
        "|---|---:|:---:|",
    ]
    for point in report["repeat_check"]["points"]:
        lines.append(
            f"| {point['point']} | {point['valid_repeats']}/3 | "
            f"{'yes' if point['complete'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Valid points (mean ± sample std)",
            "",
            "Only runs with `status=complete` and link validation `status=pass` or `status=verified_semantic` are included. Semantic verification is reported separately and is not physical-link proof.",
            "",
        ]
    )
    for group in report["aggregates"]:
        lines.extend(
            [
                f"### {group['architecture']} / {group['link']}",
                "",
                "| Metric | Mean | Std | vs same-architecture NVLink |",
                "|---|---:|---:|---:|",
            ]
        )
        relative = group["relative_to_architecture_nvlink"]
        for metric, stat in group["metrics"].items():
            ratio = relative.get(metric) if relative is not None else None
            lines.append(
                f"| {metric} | {_fmt(stat['mean'])} | {_fmt(stat['std'])} | "
                f"{_fmt(ratio)} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Invalid-point evidence",
            "",
            "| Point | Status | Reason | Backend log | Counter |",
            "|---|---|---|---|---|",
        ]
    )
    for point in report["invalid_points"]:
        if point.get("paths"):
            reason = "; ".join(str(row["reason"]) for row in point["paths"])
            backend = "; ".join(_fmt(row["backend"]) for row in point["paths"])
            counter = "; ".join(_fmt(row["counter"]) for row in point["paths"])
        else:
            reason, backend, counter = (
                point["reason"],
                point["backend"],
                point["counter"],
            )
        lines.append(
            f"| {point['point']} | {point['status']} | {reason} | "
            f"{backend} | {counter} |"
        )
    lines.extend(["", "## Comparison policy", "", report["comparison_note"], ""])
    return "\n".join(lines)


def write_report(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comm_ablation_report.json"
    markdown_path = output_dir / "comm_ablation_report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_markdown(report))
    return json_path, markdown_path
