#!/usr/bin/env python3
"""Build a provenance-preserving view after the documented screen01 interruption."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil


REPLACED_MODES = {"separate-4096", "mixed-1024"}
ALL_MODES = REPLACED_MODES | {"mixed-4096", "unchunked"}


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def cells(modes, repetitions):
    return {(mode, limit, order, rep) for mode in modes for limit in (0, 1, 4)
            for order in ("short-first", "long-first") for rep in repetitions}


def read_runs(directory):
    result = {}
    for path in sorted(directory.glob("*-screen.json")):
        item = read_json(path)
        exp = item["experiment"]
        key = tuple(exp[k] for k in ("mode", "activation_limit", "order", "rep"))
        if key in result or exp["stage"] != "screen" or exp["run_id"] != path.stem:
            raise ValueError("Duplicate or mislabeled measured artifact")
        result[key] = path
    return result


def assemble(original, recovery, output, current_hashes):
    old, new = [read_json(p / "manifest.json") for p in (original, recovery)]
    if (original / "completed.json").exists():
        raise ValueError("This recovery applies only to the interrupted original")
    done = read_json(recovery / "completed.json")
    if not done.get("completed") or not (
            old["source_hashes"] == new["source_hashes"] == done["source_hashes"] == current_hashes):
        raise ValueError("Source versions differ or recovery did not complete")
    a, b = old["args"], new["args"]
    changing = {"output", "modes", "sessions", "rep_start"}
    if {k: v for k, v in a.items() if k not in changing} != {
            k: v for k, v in b.items() if k not in changing}:
        raise ValueError("Recovery changed workload or measurement parameters")
    if (set(a["modes"]) != ALL_MODES or set(b["modes"]) != REPLACED_MODES
            or a["sessions"] != 2 or b["sessions"] != 1
            or a["rep_start"] != 1 or b["rep_start"] != 2
            or a["reps_per_session"] != 1 or a["stage"] != "screen"
            or a["regime"] != "state8" or a["waves"] != 4
            or a["short_per_wave"] != 16 or a["long_per_wave"] != 4
            or a["output_tokens"] != 64 or a["interval"] != 2
            or old["commands"] != {**old["commands"], **new["commands"]}
            or old["planned_measured_runs"] != 48 or new["planned_measured_runs"] != 12
            or old["planned_measured_requests"] != 3840 or new["planned_measured_requests"] != 960):
        raise ValueError("Unexpected recovery protocol")
    original_runs, recovery_runs = read_runs(original), read_runs(recovery)
    expected, replaced = cells(ALL_MODES, (1, 2)), cells(REPLACED_MODES, (2,))
    kept = {k: v for k, v in original_runs.items() if k not in replaced}
    excluded = {k: v for k, v in original_runs.items() if k in replaced}
    if (len(original_runs) != 39 or set(original_runs) - expected
            or set(kept) != expected - replaced or set(recovery_runs) != replaced
            or len(excluded) != 3
            or any(k[0] != "separate-4096" for k in excluded)):
        raise ValueError("Artifacts do not match the documented whole-session replacement")
    selected = {**kept, **recovery_runs}
    provenance = {
        "reason": "GPU health query timeout; replace the entire interrupted mode/repetition, not selected outcomes.",
        "original_completed": False,
        "original_manifest": old, "recovery_manifest": new,
        "selected": [{"cell": list(k), "source": str(v.resolve()), "sha256": sha256(v)}
                     for k, v in sorted(selected.items())],
        "excluded": [{"cell": list(k), "source": str(v.resolve()), "sha256": sha256(v),
                      "reason": "Whole-session replacement after interruption"}
                     for k, v in sorted(excluded.items())],
        "missing_original_cells": [list(k) for k in sorted(expected - set(original_runs))],
        "original_manifest_sha256": sha256(original / "manifest.json"),
        "recovery_manifest_sha256": sha256(recovery / "manifest.json"),
        "recovery_completed_sha256": sha256(recovery / "completed.json"),
    }
    output.mkdir(parents=True, exist_ok=False)
    for record in provenance["selected"]:
        source = Path(record["source"])
        destination = output / source.name
        shutil.copy2(source, destination)
        if sha256(destination) != record["sha256"]:
            raise ValueError("Artifact changed while assembling")
    manifest = {**old, "args": {**a, "output": str(output)}, "assembly": provenance}
    write_json(output / "manifest.json", manifest)
    write_json(output / "assembly_manifest.json", provenance)
    write_json(output / "completed.json", {
        "completed": True, "assembly_completed": True, "source_hashes": current_hashes,
        "note": "Completion refers to the assembled view, not uninterrupted execution of screen01.",
    })
    return {"selected": len(selected), "excluded": len(excluded), "original_completed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("recovery", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from run_m2_activation_pacing import source_hashes
    print(json.dumps(assemble(args.original, args.recovery, args.output, source_hashes())))


if __name__ == "__main__":
    main()
