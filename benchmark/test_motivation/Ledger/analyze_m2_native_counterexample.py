"""Analyze fresh native-control calibration separately from historical screens."""

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics

from analyze_m2_unified_admission import metrics
from audit_m2_admission_execution import health_metrics, window_metrics
from run_m2_native_counterexample import POLICIES, REPS, base, session_plan, validate


def contrast(reference, treatment):
    keys = ("mean_jct_s", "p95_jct_s", "makespan_s", "state_slot_seconds")
    gain = {k: 100 * (reference[k] - treatment[k]) / reference[k] for k in keys}
    for name in ("mean_jct_s", "mean_ttft_s", "p95_ttft_s"):
        a, b = reference["candidate"][name], treatment["candidate"][name]
        gain[f"candidate_{name}"] = 100 * (a - b) / a
    return gain


def constrained(gain):
    return gain["mean_jct_s"] > 0 and gain["p95_jct_s"] >= -5 and gain["makespan_s"] >= -5


def analyze(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    completed = json.loads((directory / "completed.json").read_text())
    if (not completed["completed"] or completed["runs"] != 12
            or manifest["source_hashes"] != completed["source_hashes"]
            or manifest["sessions"] != json.loads(json.dumps(session_plan()))):
        raise ValueError("Incomplete or changed matrix")
    rows, artifacts, logs, warmups = [], {}, [], 0
    for index, session in enumerate(session_plan()):
        parent = directory / f"session-{index:02d}-{session['server_policy']}-rep{session['rep']}"
        done = json.loads((parent / "completed.json").read_text())
        if not done["completed"] or done["formal_runs"] != len(session["cases"]):
            raise ValueError("Incomplete session")
        paths = sorted(parent.glob("m2nc-*.json"))
        loaded = [(p, json.loads(p.read_text())) for p in paths]
        expected = {(p, s, session["rep"], stage) for p, s in session["cases"] for stage in ("warmup", "calibration")}
        actual = [(i["experiment"]["policy"], i["experiment"]["phase"], i["experiment"]["rep"], i["experiment"]["stage"]) for _, i in loaded]
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Session cells differ")
        log = (parent / "server.log").read_text()
        logs.append({"session": str(parent), "allocation_retry_warnings": log.count("memory allocation failed with OOM"),
                     "custom_all_reduce_fallback_warnings": log.count("To silence this warning, specify --disable-custom-all-reduce explicitly.")})
        for path, item in loaded:
            exp = item["experiment"]
            errors = item["integrity_errors"] + base.integrity_errors(item, 576, .05) + base.progress_errors(item)
            if errors:
                raise ValueError(f"Invalid artifact {path}: {errors}")
            for name in ("server_initial", "server_final"):
                if list(validate(item[name], exp["policy"])) != item["resolved_capacity"]:
                    raise ValueError("Capacity changed")
            connections = item["client_timing"]
            if connections["created"].get("replay", 0) or connections["reused"].get("replay") != 576:
                raise ValueError("Replay connection mismatch")
            snapshot = base.arrival_snapshot(item, exp["arrival"])
            errors = base.scene_errors(snapshot, "state", exp["phase"])
            if errors or errors != item["scene_errors"]:
                raise ValueError("Invalid scene")
            artifacts[str(path.resolve())] = base.hash_file(path)
            if exp["stage"] == "warmup":
                warmups += 1
                continue
            result = metrics(item)
            release = exp["arrival"] + result["mechanism"]["holder_last_release_after_arrival_s"]
            rows.append({"experiment": exp, "metrics": result, "snapshot": snapshot,
                "fingerprint": item["payload_fingerprint"], "capacity": item["resolved_capacity"],
                "control": item["server_final"]["internal_states"][0]["m2_admission"]["runs"][exp["run_id"]],
                "execution": window_metrics(item, release), "health": health_metrics(item),
                "source": str(path.resolve())})
    index = {(r["experiment"]["policy"], r["experiment"]["phase"], r["experiment"]["rep"]): r for r in rows}
    expected = {(p, s, r) for p in POLICIES for s in ("far", "near") for r in REPS}
    if len(index) != len(rows) or set(index) != expected or warmups != 12:
        raise ValueError("Wrong complete matrix")
    contrasts = []
    for phase in ("far", "near"):
        for policy in ("cap512", "slots96"):
            samples = []
            for rep in REPS:
                a, b = index["native607", phase, rep], index[policy, phase, rep]
                if a["fingerprint"] != b["fingerprint"]:
                    raise ValueError("Unpaired payload")
                gain = contrast(a["metrics"], b["metrics"])
                trigger = b["control"]["native_slot_delays"] > 0 if policy == "slots96" else b["metrics"]["blocks"].get("request_limit", 0) > 0
                samples.append({"rep": rep, "gain_percent": gain, "triggered": trigger, "constrained_gain": constrained(gain)})
            contrasts.append({"phase": phase, "policy": policy, "samples": samples,
                "mean_gain_percent": {k: statistics.fmean(s["gain_percent"][k] for s in samples) for k in samples[0]["gain_percent"]},
                "screen_positive": all(s["triggered"] and s["constrained_gain"] for s in samples)})
    return {"complete": True, "counts": {"runs": 12, "requests": 6912, "warmup_requests": 6912},
            "manifest": manifest, "artifact_sha256": artifacts, "rows": rows, "contrasts": contrasts, "server_logs": logs,
            "scope": "State-only native calibration, two repeats; not an independent state-cost ablation, held-out, or CI."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.output.mkdir(parents=True, exist_ok=False)
    base.dump(args.output / "analysis.json", result)
    lines = ["# M2 Native Calibration", "", f"Complete: True; {result['counts']}", "",
             "| Phase | Policy | Rep | Mean JCT | P95 JCT | Makespan | New TTFT | Slot delays | Request-limit blocks |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in result["rows"]:
        e, m = r["experiment"], r["metrics"]
        lines.append(f"| {e['phase']} | {e['policy']} | {e['rep']} | {m['mean_jct_s']:.3f} | {m['p95_jct_s']:.3f} | {m['makespan_s']:.3f} | {m['candidate']['mean_ttft_s']:.3f} | {r['control']['native_slot_delays']} | {m['blocks'].get('request_limit', 0)} |")
    lines += ["", "Positive percent means improvement; two exploratory repeats, not confidence intervals.", "",
              "| Phase | Policy | Mean gain % | P95 gain % | Makespan gain % | New mean TTFT gain % | Both constrained/triggered |", "|---|---|---:|---:|---:|---:|---|"]
    for c in result["contrasts"]:
        g = c["mean_gain_percent"]
        lines.append(f"| {c['phase']} | {c['policy']} | {g['mean_jct_s']:.2f} | {g['p95_jct_s']:.2f} | {g['makespan_s']:.2f} | {g['candidate_mean_ttft_s']:.2f} | {c['screen_positive']} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
