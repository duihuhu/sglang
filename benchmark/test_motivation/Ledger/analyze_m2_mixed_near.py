"""Analyze the amended near-only matrix without erasing failed far calibration."""

import argparse
import json
from pathlib import Path
import statistics

from analyze_m2_mixed_holdout import pair, triggered
from analyze_m2_unified_admission import metrics
from audit_m2_admission_execution import health_metrics, window_metrics
from run_m2_mixed_near import original, session_plan


def select_baseline(index):
    eligible = {"native607": 1.0}
    for policy in ("cap512", "slots96"):
        pairs = [pair(index["native607", "train512", rep], index[policy, "train512", rep])
                 for rep in original.REPS]
        if all(p["scene_matched"] and p["triggered"] and p["constrained_gain"] for p in pairs):
            eligible[policy] = statistics.geometric_mean(1 - p["gain_percent"]["mean_jct_s"] / 100 for p in pairs)
    return {"policy": min(eligible, key=lambda p: (eligible[p], original.POLICIES.index(p))),
            "training_scores": eligible, "selection_uses_held_out": False, "scope": "near only"}


def analyze(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    done = json.loads((directory / "completed.json").read_text())
    if (manifest["stage"] != "measure" or not done["completed"] or done["runs"] != 16
            or manifest["source_hashes"] != done["source_hashes"]
            or manifest["plan"]["phases"] != ["near"] or not manifest["plan"]["scenes_valid"]
            or manifest["sessions"] != json.loads(json.dumps(session_plan("measure")))):
        raise ValueError("Incomplete or changed near matrix")
    rows, artifacts, logs, warmups, warmup_requests = [], {}, [], 0, 0
    for number, session in enumerate(session_plan("measure")):
        parent = directory / f"session-{number:02d}-{session['server_policy']}-rep{session['rep']}"
        completed = json.loads((parent / "completed.json").read_text())
        if not completed["completed"] or completed["formal_runs"] != len(session["cases"]):
            raise ValueError("Incomplete session")
        expected = {(p, profile, phase, stage) for p, profile, phase in session["cases"]
                    for stage in ("warmup", "measure")}
        actual = []
        log = (parent / "server.log").read_text()
        logs.append({"session": str(parent), "allocation_retry_warnings": log.count("memory allocation failed with OOM"),
                     "custom_all_reduce_fallback_warnings": log.count("To silence this warning, specify --disable-custom-all-reduce explicitly.")})
        for path in sorted(parent.glob("m2mh-*.json")):
            item = json.loads(path.read_text())
            e = item["experiment"]
            actual.append((e["policy"], e["profile"], e["phase"], e["stage"]))
            count = original.PROFILES[e["profile"]][0] + 64
            errors = item["integrity_errors"] + original.base.integrity_errors(item, count, .05) + original.base.progress_errors(item)
            if errors or e["rep"] != session["rep"]:
                raise ValueError(f"Invalid artifact {path}: {errors}")
            for key in ("server_initial", "server_final"):
                if list(original.validate(item[key], e["policy"])) != item["resolved_capacity"]:
                    raise ValueError("Capacity changed")
            at = manifest["plan"]["arrivals"][f"{e['profile']}-near"]
            specs = original.specs_for(e["run_id"], e["profile"], -e["rep"] if e["stage"] == "warmup" else e["rep"], at)
            if e["arrival"] != at or original.base.fingerprint(specs) != item["payload_fingerprint"]:
                raise ValueError("Frozen payload/arrival changed")
            connections = item["client_timing"]
            if connections["created"].get("replay", 0) or connections["reused"].get("replay") != count:
                raise ValueError("Connection mismatch")
            snapshot = original.base.arrival_snapshot(item, at)
            scene = original.base.scene_errors(snapshot, "state", "near")
            if scene != item["scene_errors"]:
                raise ValueError("Scene check changed")
            artifacts[str(path.resolve())] = original.base.hash_file(path)
            if e["stage"] == "warmup":
                warmups += 1
                warmup_requests += count
                continue
            result = metrics(item)
            release = at + result["mechanism"]["holder_last_release_after_arrival_s"]
            rows.append({"experiment": e, "metrics": result, "snapshot": snapshot, "scene_errors": scene,
                "fingerprint": item["payload_fingerprint"], "capacity": item["resolved_capacity"],
                "control": item["server_final"]["internal_states"][0]["m2_admission"]["runs"][e["run_id"]],
                "execution": window_metrics(item, release), "health": health_metrics(item),
                "source": str(path.resolve()), "count": count})
        if len(actual) != len(expected) or set(actual) != expected:
            raise ValueError("Session matrix differs")
    index = {(r["experiment"]["policy"], r["experiment"]["profile"], r["experiment"]["rep"]): r for r in rows}
    expected = {(p, profile, rep) for p in original.POLICIES for profile in original.PROFILES for rep in original.REPS}
    if len(index) != len(rows) or set(index) != expected or warmups != 16:
        raise ValueError("Wrong complete matrix")
    contrasts = []
    comparisons = [("native607", p) for p in original.POLICIES[1:]] + [(p, "decode32") for p in ("cap512", "slots96")]
    for profile in original.PROFILES:
        for reference, treatment in comparisons:
            samples = [pair(index[reference, profile, rep], index[treatment, profile, rep]) for rep in original.REPS]
            contrasts.append({"profile": profile, "phase": "near", "reference": reference, "treatment": treatment,
                "samples": samples,
                "mean_gain_percent": {k: statistics.fmean(s["gain_percent"][k] for s in samples) for k in samples[0]["gain_percent"]},
                "screen_positive": all(s["scene_matched"] and s["triggered"] and s["constrained_gain"] for s in samples)})
    return {"complete": True, "phases": ["near"], "counts": {"runs": len(rows), "requests": sum(r["count"] for r in rows),
            "warmups": warmups, "warmup_requests": warmup_requests}, "selected_baseline": select_baseline(index),
            "manifest": manifest, "artifact_sha256": artifacts, "rows": rows,
            "contrasts": contrasts, "server_logs": logs,
            "scope": "Near-only amendment after failed far background calibration; two repeats, not full Mixed tuning, state causality or SOTA superiority."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    args.output.mkdir(parents=True, exist_ok=False)
    original.base.dump(args.output / "analysis.json", result)
    lines = ["# M2 Mixed Near-Only Screen", "", f"Complete: True; {result['counts']}", "",
             f"Training-only selection: {result['selected_baseline']}", "",
             "| Profile | Policy | Rep | Mean JCT | P95 | Makespan | New TTFT | Alive | Remaining | Triggered | Scene errors |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    for row in result["rows"]:
        e, m, s = row["experiment"], row["metrics"], row["snapshot"]
        lines.append(f"| {e['profile']} | {e['policy']} | {e['rep']} | {m['mean_jct_s']:.3f} | {m['p95_jct_s']:.3f} | {m['makespan_s']:.3f} | {m['candidate']['mean_ttft_s']:.3f} | {s['alive']} | {s['median_remaining']} | {triggered(row)} | {row['scene_errors']} |")
    lines += ["", "Positive percentage means improvement; two repeats are not confidence intervals.", "",
              "| Profile | Reference | Treatment | Mean gain % | P95 gain % | Makespan gain % | New TTFT gain % | Both valid/triggered/constrained |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for c in result["contrasts"]:
        g = c["mean_gain_percent"]
        lines.append(f"| {c['profile']} | {c['reference']} | {c['treatment']} | {g['mean_jct_s']:.2f} | {g['p95_jct_s']:.2f} | {g['makespan_s']:.2f} | {g['candidate_mean_ttft_s']:.2f} | {c['screen_positive']} |")
    (args.output / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
