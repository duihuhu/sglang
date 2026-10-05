#!/usr/bin/env python3
"""Fit the preregistered nonnegative profiled-VTC cost model."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import nnls


PROMPTS = (1024, 4096, 16384)
OUTPUTS = (32, 128, 512)
REPETITIONS = 3


def load_profiles(result_dir: Path) -> list[dict[str, Any]]:
    artifacts = []
    for path in sorted(result_dir.glob("st-profile-*.json")):
        artifact = json.loads(path.read_text(encoding="utf-8"))
        artifact["_path"] = str(path)
        artifacts.append(artifact)
    return artifacts


def features(prompt: int, output: int) -> list[float]:
    p = prompt / 1024
    q = output / 32
    return [1.0, p, p * p, q, p * q, q * q]


def fit(result_dir: Path) -> dict[str, Any]:
    artifacts = load_profiles(result_dir)
    grouped: dict[tuple[int, int], list[float]] = defaultdict(list)
    errors = []
    for artifact in artifacts:
        exp = artifact.get("experiment", {})
        requests = artifact.get("requests", [])
        prompt = int(exp.get("profile_prompt", 0))
        output = int(exp.get("profile_output", 0))
        if (prompt, output) not in {
            (p, q) for p in PROMPTS for q in OUTPUTS
        }:
            errors.append(f"{artifact['_path']}: unexpected profile shape")
            continue
        if len(requests) != 1 or not requests[0].get("success", False):
            errors.append(f"{artifact['_path']}: profile request failed or missing")
            continue
        if int(requests[0].get("cached_tokens", 0)) != 0:
            errors.append(f"{artifact['_path']}: nonzero cached tokens")
        grouped[(prompt, output)].append(float(requests[0]["latency_s"]))

    for prompt in PROMPTS:
        for output in OUTPUTS:
            count = len(grouped[(prompt, output)])
            if count != REPETITIONS:
                errors.append(
                    f"P={prompt},O={output}: {count} repetitions, expected {REPETITIONS}"
                )
    if errors:
        raise ValueError("; ".join(errors))

    shapes = [(p, q) for p in PROMPTS for q in OUTPUTS]
    x = np.asarray([features(*shape) for shape in shapes], dtype=np.float64)
    measured = np.asarray(
        [statistics.median(grouped[shape]) for shape in shapes], dtype=np.float64
    )
    scaled_coefficients, residual_norm = nnls(x, measured)
    predicted = x @ scaled_coefficients
    relative_errors = np.abs(predicted - measured) / measured
    raw = {
        "request_weight": float(scaled_coefficients[0]),
        "input_weight": float(scaled_coefficients[1] / 1024),
        "input_quadratic_weight": float(scaled_coefficients[2] / (1024**2)),
        "output_weight": float(scaled_coefficients[3] / 32),
        "input_output_weight": float(
            scaled_coefficients[4] / (1024 * 32)
        ),
        "output_quadratic_weight": float(scaled_coefficients[5] / (32**2)),
    }
    rows = []
    for (prompt, output), actual, estimate, error in zip(
        shapes, measured, predicted, relative_errors
    ):
        rows.append(
            {
                "prompt": prompt,
                "output": output,
                "median_jct_s": float(actual),
                "predicted_cost": float(estimate),
                "relative_error": float(error),
            }
        )
    median_error = float(statistics.median(relative_errors.tolist()))
    max_error = float(max(relative_errors))
    return {
        "schema_version": 1,
        "method": "scipy.optimize.nnls",
        "features": [
            "1",
            "P/1024",
            "(P/1024)^2",
            "O/32",
            "(P/1024)*(O/32)",
            "(O/32)^2",
        ],
        "formal_artifacts": len(artifacts),
        "raw_scheduler_weights": raw,
        "scaled_coefficients": [float(value) for value in scaled_coefficients],
        "residual_norm": float(residual_norm),
        "median_relative_error": median_error,
        "max_relative_error": max_error,
        "gate": {
            "median_limit": 0.20,
            "max_limit": 0.35,
            "passed": median_error <= 0.20 and max_error <= 0.35,
        },
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = fit(args.result_dir)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if not result["gate"]["passed"]:
        raise SystemExit("profile fit gate failed")


if __name__ == "__main__":
    main()
