#!/usr/bin/env python3
"""Evaluate a preselected M2 pair using fresh paired confirmation runs."""

import argparse
import json
from pathlib import Path

from analyze_m2_batch_frontier import ENDPOINTS, paired_effect


def evaluate(data, selection, factor):
    pair = {}
    for side in ("baseline", "candidate"):
        matches = [r for r in data["rows"] if r["stage"] == "confirm"
                   and r["factor"] == factor and r["config"] == selection[side]]
        if len(matches) != 1:
            raise ValueError(f"Expected exactly one {side} confirmation group")
        pair[side] = {r["rep"]: r for r in matches[0]["runs"]}
    if set(pair["baseline"]) != set(pair["candidate"]):
        raise ValueError("Confirmation repetitions must match exactly")
    reps = sorted(pair["baseline"])
    if len(reps) < 5:
        raise ValueError("At least five fresh paired repetitions required")
    if min(reps) <= selection["last_screening_rep"]:
        raise ValueError("Screening repetitions cannot be reused as confirmation")
    primary = selection["primary_endpoint"]
    if primary not in ENDPOINTS:
        raise ValueError("Unknown primary endpoint")
    effects = {endpoint: paired_effect(
        [pair["baseline"][rep][endpoint] for rep in reps],
        [pair["candidate"][rep][endpoint] for rep in reps],
    ) for endpoint in ENDPOINTS}
    passed = (effects[primary]["mean_improvement"] >= selection["min_primary_improvement"]
              and effects[primary]["ci95"][0] > 0
              and all(effect["ci95"][0] >= -selection["noninferiority_margin"]
                      for endpoint, effect in effects.items() if endpoint != primary))
    return {"factor": factor, "repetitions": reps, "selection": selection,
            "effects": effects, "native_control_benefit_gate_passed": passed,
            "sota_residual_loss_demonstrated": False,
            "hybrid_specific_causality_demonstrated": False,
            "note": "A configuration control is not a new scheduler or a SOTA-wide result."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("analysis", type=Path)
    parser.add_argument("selection", type=Path)
    parser.add_argument("--factor", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.analysis.read_text()),
                      json.loads(args.selection.read_text()), args.factor)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
