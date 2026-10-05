"""Reuse frozen Mixed execution, with a separately validated near-only plan."""

import argparse
import json
from pathlib import Path
import sys

import run_m2_mixed_holdout as original


ORIGINAL_HASHES = original.hashes
ORIGINAL_SESSIONS = original.session_plan


def hashes():
    result = ORIGINAL_HASHES()
    for name in ("run_m2_mixed_near.py", "analyze_m2_mixed_near.py", "m2_mixed_near_protocol_20260917.md"):
        path = original.base.BASE / name
        result[str(path)] = original.base.hash_file(path)
    return result


def session_plan(stage):
    if stage != "measure":
        raise ValueError("Near adapter only measures a validated background calibration")
    return [{**s, "cases": [c for c in s["cases"] if c[2] == "near"]}
            for s in ORIGINAL_SESSIONS("measure")]


def prepare(source, output):
    manifest = json.loads((source / "manifest.json").read_text())
    plan = json.loads((source / "arrival-plan.json").read_text())
    expected_hashes = ORIGINAL_HASHES()
    if (manifest["stage"] != "calibrate" or manifest["source_hashes"] != expected_hashes
            or plan["source_hashes"] != expected_hashes
            or manifest["sessions"] != json.loads(json.dumps(ORIGINAL_SESSIONS("calibrate")))):
        raise ValueError("Changed background sources or matrix")
    items, artifacts = [], {}
    for index, session in enumerate(ORIGINAL_SESSIONS("calibrate")):
        parent = source / f"session-{index:02d}-native607-rep{session['rep']}"
        done = json.loads((parent / "completed.json").read_text())
        if not done["completed"] or done["formal_runs"] != 2:
            raise ValueError("Incomplete background session")
        actual = []
        for path in sorted(parent.glob("m2mh-*.json")):
            item = json.loads(path.read_text())
            e = item["experiment"]
            actual.append((e["profile"], e["stage"]))
            count = original.PROFILES[e["profile"]][0]
            if e["policy"] != "native607" or e["rep"] != session["rep"] or e["arrival"] is not None:
                raise ValueError("Wrong background identity")
            errors = item["integrity_errors"] + original.base.integrity_errors(item, count, .05) + original.base.progress_errors(item)
            if errors:
                raise ValueError(errors)
            for key in ("server_initial", "server_final"):
                if list(original.validate(item[key], "native607")) != item["resolved_capacity"]:
                    raise ValueError("Background capacity changed")
            connections = item["client_timing"]
            if connections["created"].get("replay", 0) or connections["reused"].get("replay") != count:
                raise ValueError("Background connections changed")
            artifacts[str(path.resolve())] = original.base.hash_file(path)
            if e["stage"] == "calibrate":
                items.append(item)
        expected = {(p, stage) for p in original.PROFILES for stage in ("warmup", "calibrate")}
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Background warmup/formal matrix differs")
    recomputed = original.freeze_arrivals(items)
    if any(plan[key] != recomputed[key] for key in ("arrivals", "evidence", "scenes_valid")):
        raise ValueError("Original calibration evidence changed")
    evidence = [e for e in recomputed["evidence"] if e["phase"] == "near"]
    if len(evidence) != 4 or any(e["errors"] for e in evidence):
        raise ValueError("Near calibration does not pass original scene criteria")
    result = {"arrivals": {k: v for k, v in plan["arrivals"].items() if k.endswith("-near")},
              "evidence": evidence, "phases": ["near"], "scenes_valid": True,
              "source_hashes": hashes(), "physical_gpu_ids": plan["physical_gpu_ids"], "gpus": plan["gpus"],
              "source_calibration": str(source.resolve()), "source_plan_sha256": original.base.hash_file(source / "arrival-plan.json"),
              "source_manifest_sha256": original.base.hash_file(source / "manifest.json"),
              "source_artifact_sha256": artifacts, "original_scenes_valid": plan["scenes_valid"],
              "excluded_scene_evidence": [e for e in plan["evidence"] if e["phase"] != "near"]}
    original.base.dump(output, result)
    print(json.dumps({"near_valid": True, "arrivals": result["arrivals"], "source_artifacts": len(artifacts)}))


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("prepare", "measure"):
        raise SystemExit("Use prepare --source DIR --output FILE, or measure with the original runner arguments")
    if sys.argv[1] == "prepare":
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--source", type=Path, required=True)
        parser.add_argument("--output", type=Path, required=True)
        args = parser.parse_args(sys.argv[2:])
        prepare(args.source, args.output)
        return
    original.hashes = hashes
    original.session_plan = session_plan
    sys.argv = [sys.argv[0], "--stage", "measure", *sys.argv[2:]]
    original.main()


if __name__ == "__main__":
    main()
