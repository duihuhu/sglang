"""Keep transfer, trigger evidence and capacity attribution separate."""

import argparse
import json
from pathlib import Path
import statistics

from analyze_m2_native_counterexample import constrained, contrast
from analyze_m2_unified_admission import metrics
from audit_m2_admission_execution import health_metrics, window_metrics
from run_m2_mixed_holdout import POLICIES, PROFILES, REPS, base, session_plan, validate


def triggered(row):
    policy, control = row["experiment"]["policy"], row["control"]
    if policy == "native607":
        return True
    if policy == "cap512":
        return row["metrics"]["blocks"].get("request_limit", 0) > 0
    return control["native_slot_delays" if policy == "slots96" else "delayed"] > 0


def scene_matched(a, b):
    x, y = a["snapshot"], b["snapshot"]
    return (not a["scene_errors"] and not b["scene_errors"]
            and x["alive"] == y["alive"]
            and x["median_remaining"] is not None and y["median_remaining"] is not None
            and abs(x["median_remaining"] - y["median_remaining"]) <= 8)


def pair(a, b):
    if a["fingerprint"] != b["fingerprint"]:
        raise ValueError("Unpaired payload/arrivals")
    gain = contrast(a["metrics"], b["metrics"])
    return {"rep": a["experiment"]["rep"], "gain_percent": gain,
            "scene_matched": scene_matched(a, b), "triggered": triggered(b),
            "constrained_gain": constrained(gain)}


def select_baseline(index):
    eligible = {"native607": 1.0}
    for policy in ("cap512", "slots96"):
        pairs = [pair(index["native607", "train512", phase, rep],
                      index[policy, "train512", phase, rep])
                 for phase in ("far", "near") for rep in REPS]
        if all(p["scene_matched"] and p["triggered"] and p["constrained_gain"] for p in pairs):
            eligible[policy] = statistics.geometric_mean(
                1 - p["gain_percent"]["mean_jct_s"] / 100 for p in pairs)
    return {"policy": min(eligible, key=lambda p: (eligible[p], POLICIES.index(p))),
            "training_scores": eligible, "selection_uses_held_out": False}


def analyze(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    completed = json.loads((directory / "completed.json").read_text())
    if (manifest["stage"] != "measure" or not completed["completed"] or completed["runs"] != 32
            or manifest["source_hashes"] != completed["source_hashes"]
            or manifest["sessions"] != json.loads(json.dumps(session_plan("measure")))):
        raise ValueError("Incomplete or changed matrix")
    rows, artifacts, logs, warmups, warmup_requests = [], {}, [], 0, 0
    for number, session in enumerate(session_plan("measure")):
        parent = directory / f"session-{number:02d}-{session['server_policy']}-rep{session['rep']}"
        done = json.loads((parent / "completed.json").read_text())
        if not done["completed"] or done["formal_runs"] != len(session["cases"]):
            raise ValueError("Incomplete session")
        expected = {(p, profile, phase, stage) for p, profile, phase in session["cases"]
                    for stage in ("warmup", "measure")}
        actual = []
        log = (parent / "server.log").read_text()
        logs.append({"session": str(parent), "allocation_retry_warnings": log.count("memory allocation failed with OOM"),
                     "custom_all_reduce_fallback_warnings": log.count("To silence this warning, specify --disable-custom-all-reduce explicitly.")})
        for path in sorted(parent.glob("m2mh-*.json")):
            item = json.loads(path.read_text())
            exp = item["experiment"]
            actual.append((exp["policy"], exp["profile"], exp["phase"], exp["stage"]))
            if exp["rep"] != session["rep"]:
                raise ValueError("Wrong repetition")
            count = PROFILES[exp["profile"]][0] + 64
            errors = item["integrity_errors"] + base.integrity_errors(item, count, .05) + base.progress_errors(item)
            if errors:
                raise ValueError(f"Invalid artifact {path}: {errors}")
            for name in ("server_initial", "server_final"):
                if list(validate(item[name], exp["policy"])) != item["resolved_capacity"]:
                    raise ValueError("Capacity changed")
            connections = item["client_timing"]
            if connections["created"].get("replay", 0) or connections["reused"].get("replay") != count:
                raise ValueError("Connection mismatch")
            snapshot = base.arrival_snapshot(item, exp["arrival"])
            scene = base.scene_errors(snapshot, "state", exp["phase"])
            if scene != item["scene_errors"]:
                raise ValueError("Scene check changed")
            artifacts[str(path.resolve())] = base.hash_file(path)
            if exp["stage"] == "warmup":
                warmups += 1
                warmup_requests += count
                continue
            result = metrics(item)
            release = exp["arrival"] + result["mechanism"]["holder_last_release_after_arrival_s"]
            rows.append({"experiment": exp, "metrics": result, "snapshot": snapshot, "scene_errors": scene,
                "fingerprint": item["payload_fingerprint"], "capacity": item["resolved_capacity"],
                "control": item["server_final"]["internal_states"][0]["m2_admission"]["runs"][exp["run_id"]],
                "execution": window_metrics(item, release), "health": health_metrics(item),
                "source": str(path.resolve()), "count": count})
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Session matrix differs")
    index = {(r["experiment"]["policy"], r["experiment"]["profile"], r["experiment"]["phase"], r["experiment"]["rep"]): r for r in rows}
    expected = {(p, profile, phase, rep) for p in POLICIES for profile in PROFILES
                for phase in ("far", "near") for rep in REPS}
    if len(index) != len(rows) or set(index) != expected or warmups != 32:
        raise ValueError("Wrong complete matrix")
    contrasts = []
    comparisons = [("native607", p) for p in POLICIES[1:]] + [(p, "decode32") for p in ("cap512", "slots96")]
    for profile in PROFILES:
        for phase in ("far", "near"):
            for reference, treatment in comparisons:
                samples = [pair(index[reference, profile, phase, rep], index[treatment, profile, phase, rep]) for rep in REPS]
                contrasts.append({"profile": profile, "phase": phase, "reference": reference, "treatment": treatment,
                    "samples": samples,
                    "mean_gain_percent": {k: statistics.fmean(s["gain_percent"][k] for s in samples) for k in samples[0]["gain_percent"]},
                    "screen_positive": all(s["scene_matched"] and s["triggered"] and s["constrained_gain"] for s in samples)})
    return {"complete": True, "counts": {"runs": len(rows), "requests": sum(r["count"] for r in rows),
            "warmups": warmups, "warmup_requests": warmup_requests}, "selected_baseline": select_baseline(index),
            "manifest": manifest, "artifact_sha256": artifacts, "rows": rows,
            "contrasts": contrasts, "server_logs": logs,
            "scope": "Mixed4K frozen-control transfer screen; two repeats, not state causality or SOTA superiority."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.output.mkdir(parents=True, exist_ok=False)
    base.dump(args.output / "analysis.json", result)
    lines = ["# M2 Mixed and Frozen-Control Transfer", "", f"Complete: True; {result['counts']}", "",
             f"Training-only baseline selection: {result['selected_baseline']}", "",
             "| Profile | Phase | Policy | Rep | Mean JCT | P95 JCT | Makespan | New TTFT | Triggered | Scene errors |",
             "|---|---|---|---:|---:|---:|---:|---:|---|---|"]
    for row in result["rows"]:
        e, m = row["experiment"], row["metrics"]
        lines.append(f"| {e['profile']} | {e['phase']} | {e['policy']} | {e['rep']} | {m['mean_jct_s']:.3f} | {m['p95_jct_s']:.3f} | {m['makespan_s']:.3f} | {m['candidate']['mean_ttft_s']:.3f} | {triggered(row)} | {row['scene_errors']} |")
    lines += ["", "Positive percentage means improvement; no population CI from two repeats.", "",
              "| Profile | Phase | Reference | Treatment | Mean gain % | P95 gain % | Makespan gain % | New TTFT gain % | Both valid/triggered/constrained |",
              "|---|---|---|---|---:|---:|---:|---:|---|"]
    for c in result["contrasts"]:
        g = c["mean_gain_percent"]
        lines.append(f"| {c['profile']} | {c['phase']} | {c['reference']} | {c['treatment']} | {g['mean_jct_s']:.2f} | {g['p95_jct_s']:.2f} | {g['makespan_s']:.2f} | {g['candidate_mean_ttft_s']:.2f} | {c['screen_positive']} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
