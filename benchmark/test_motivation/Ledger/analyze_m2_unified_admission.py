"""Keep scene validation, mechanism observations and performance separate."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from analyze_m2_activation_pacing import nearest_rank
from analyze_m2_hybrid_factors import lifecycle
from run_m2_unified_admission import arrival_snapshot, integrity_errors, progress_errors, scene_errors, validate


def metrics(item):
    requests = item["requests"]
    spans = lifecycle(item)
    at = item["experiment"]["arrival"]
    result = {
        "mean_jct_s": statistics.fmean(r["latency_s"] for r in requests),
        "p95_jct_s": nearest_rank([r["latency_s"] for r in requests]),
        "mean_ttft_s": statistics.fmean(r["ttft_s"] for r in requests),
        "p95_ttft_s": nearest_rank([r["ttft_s"] for r in requests]),
        "makespan_s": max(r["finished_s"] for r in requests)-min(r["submitted_s"] for r in requests),
        "peak_active": item["scheduler_resource_trace"]["system"]["peak_active_requests"],
        "state_slot_seconds": item["scheduler_resource_trace"]["system"]["state_slot_seconds"],
        "blocks": dict(Counter(b["reason"] for b in item["scheduler_batch_trace"]["admission_blocks"])),
        "max_lateness_ms": 1000*max(r["submitted_s"]-r["scheduled_arrival_s"] for r in requests),
    }
    for group in ("holder", "candidate"):
        own = [r for r in requests if r["group"] == group]
        if not own:
            continue
        queues = [spans[r["rid"]][0]-r["submitted_s"] for r in own]
        mean_jct = statistics.fmean(r["latency_s"] for r in own)
        result[group] = {"count": len(own), "mean_jct_s": mean_jct,
                         "p95_jct_s": nearest_rank([r["latency_s"] for r in own]),
                         "mean_ttft_s": statistics.fmean(r["ttft_s"] for r in own),
                         "p95_ttft_s": nearest_rank([r["ttft_s"] for r in own]),
                         "mean_queue_s": statistics.fmean(queues), "p95_queue_s": nearest_rank(queues),
                         "mean_after_activation_s": mean_jct-statistics.fmean(queues),
                         "mean_holding_s": statistics.fmean(spans[r["rid"]][1]-spans[r["rid"]][0] for r in own)}
    if at is not None:
        candidates = [r for r in requests if r["group"] == "candidate"]
        holders = [r for r in requests if r["group"] == "holder"]
        first_candidate = min(spans[r["rid"]][0] for r in candidates)
        releases = [spans[r["rid"]][1] for r in holders]
        result["mechanism"] = {
            "first_candidate_activation_after_arrival_s": first_candidate-at,
            "holder_first_release_after_arrival_s": min(releases)-at,
            "holder_last_release_after_arrival_s": max(releases)-at,
            "holders_released_by_first_candidate": sum(t <= first_candidate for t in releases),
        }
    return result


def screen_status(contrast):
    if len(contrast["samples"]) < 2:
        return "PENDING: incomplete pairs"
    if not all(s["scene_valid"] for s in contrast["samples"]):
        return "INVALID SCENE"
    if not all(s["triggered"] for s in contrast["samples"]):
        return "NOT TRIGGERED"
    return "SCREEN POSITIVE" if contrast["two_rep_screen_positive"] else "NO CONSISTENT CONSTRAINED GAIN"


def control_delays(row):
    policy, control = row["experiment"]["policy"], row["control"]
    if policy == "slots64":
        return control["native_slot_delays"]
    if policy == "delayer32":
        return control["native_prefill_reasons"].get("delay", 0)
    return control["delayed"]


def paired(rows):
    index = {(r["experiment"]["policy"], r["experiment"]["profile"], r["experiment"]["phase"], r["experiment"]["rep"]): r for r in rows}
    if len(index) != len(rows):
        raise ValueError("Duplicate formal cell")
    contrasts = []
    for profile in ("state", "kv"):
        for phase in ("far", "near"):
            for policy in ("decode32", "slots64", "delayer32"):
                samples = []
                for rep in (51, 52):
                    a = index.get(("native", profile, phase, rep))
                    b = index.get((policy, profile, phase, rep))
                    if a is None or b is None:
                        continue
                    if a["fingerprint"] != b["fingerprint"]:
                        raise ValueError("Unpaired workload")
                    gain = {k: 100*(a["metrics"][k]-b["metrics"][k])/a["metrics"][k]
                            for k in ("mean_jct_s", "p95_jct_s", "makespan_s", "state_slot_seconds")}
                    valid_scene = not a["scene_errors"] and not b["scene_errors"]
                    triggered = b["control"]["delayed"] > 0 if policy == "decode32" else (
                        b["control"]["native_slot_delays"] > 0 if policy == "slots64" else
                        b["control"]["native_prefill_reasons"].get("delay", 0) > 0)
                    samples.append({"rep": rep, "gain_percent": gain, "scene_valid": valid_scene,
                                    "triggered": triggered,
                                    "constrained_gain": gain["mean_jct_s"] > 0 and gain["p95_jct_s"] >= -5 and gain["makespan_s"] >= -5})
                if samples:
                    contrasts.append({"profile": profile, "phase": phase, "policy": policy, "samples": samples,
                                      "mean_gain_percent": {k: statistics.fmean(s["gain_percent"][k] for s in samples) for k in samples[0]["gain_percent"]},
                                      "two_rep_screen_positive": len(samples) == 2 and all(s["scene_valid"] and s["triggered"] and s["constrained_gain"] for s in samples)})
    return contrasts


def analyze(directories, allow_partial=False):
    rows, warmups, sources, artifacts, logs, excluded = [], [], [], {}, [], []
    physical_gpus = None
    for directory in directories:
        manifest = json.loads((directory/"manifest.json").read_text())
        done_path = directory/"completed.json"
        done = json.loads(done_path.read_text()) if done_path.exists() else None
        if done is None and not allow_partial:
            raise ValueError("Missing completed marker; partial analysis must be explicit")
        if done is not None and (not done["completed"] or manifest["source_hashes"] != done["source_hashes"]):
            raise ValueError("Incomplete/source-mismatched input")
        current_gpus = [(g["index"], g["uuid"]) for g in manifest["gpus"]]
        if physical_gpus is not None and current_gpus != physical_gpus:
            raise ValueError("Cross-device comparison")
        physical_gpus = current_gpus
        sources.append({"directory": str(directory), "manifest": manifest, "completed": done})
        count = 0
        cells = set()
        for session in sorted(directory.glob("session-*")):
            loaded = [(p, json.loads(p.read_text())) for p in sorted(session.glob("m2ua-*.json"))]
            phases = ("background",) if manifest["stage"] == "calibrate" else ("far", "near")
            expected_session = {(p, phase, stage) for p in ("state", "kv") for phase in phases
                                for stage in ("warmup", manifest["stage"])}
            actual_session = {(d["experiment"]["profile"], d["experiment"]["phase"], d["experiment"]["stage"])
                              for _, d in loaded}
            if actual_session != expected_session or len(loaded) != len(expected_session):
                if not allow_partial:
                    raise ValueError(f"Incomplete session {session}")
                excluded.append({"session": str(session), "reason": "entire incomplete session excluded",
                                 "artifacts": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p, _ in loaded},
                                 "missing": sorted(expected_session-actual_session)})
                continue
            log = (session/"server.log").read_text()
            logs.append({"session": str(session), "allocation_retry_warnings": log.count("memory allocation failed with OOM"),
                         "custom_all_reduce_fallback_warnings": log.count("To silence this warning, specify --disable-custom-all-reduce explicitly.")})
            for path, item in loaded:
                exp = item["experiment"]
                expected_count = (512 if exp["profile"] == "state" else 96)+(64 if exp["arrival"] is not None else 0)
                errors = item["integrity_errors"]+integrity_errors(item, expected_count, .05)+progress_errors(item)
                if errors:
                    raise ValueError(f"Invalid run {path}: {errors}")
                for name in ("server_initial", "server_final"):
                    if list(validate(item[name], exp["policy"])) != item["resolved_capacity"]:
                        raise ValueError("Resolved capacity mismatch")
                connections = item["client_timing"]
                if connections["created"].get("replay", 0) or connections["reused"].get("replay") != expected_count:
                    raise ValueError("Connection isolation failed")
                control = item["server_final"]["internal_states"][0]["m2_admission"]["runs"][exp["run_id"]]
                snapshot = arrival_snapshot(item, exp["arrival"]) if exp["arrival"] is not None else None
                scene = scene_errors(snapshot, exp["profile"], exp["phase"]) if snapshot else []
                if scene != item.get("scene_errors", []):
                    raise ValueError("Scene check disagrees with runtime artifact")
                row = {"experiment": exp, "metrics": metrics(item), "scene_errors": scene,
                       "snapshot": snapshot, "control": control, "fingerprint": item["payload_fingerprint"],
                       "source": str(path), "count": expected_count}
                artifacts[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                if exp["stage"] == "warmup":
                    warmups.append(row)
                else:
                    rows.append(row)
                    count += 1
                    cells.add((exp["policy"], exp["profile"], exp["phase"], exp["rep"]))
        if done is not None and count != done["runs"]:
            raise ValueError("Completion count differs")
        stage = manifest["stage"]
        policies = {"calibrate": ("native",), "screen": ("native", "decode32"),
                    "native-controls": ("slots64", "delayer32")}[stage]
        expected = {(p, profile, phase, rep) for p in policies for profile in ("state", "kv")
                    for phase in (("background",) if stage == "calibrate" else ("far", "near"))
                    for rep in ((41, 42) if stage == "calibrate" else (51, 52))}
        if not allow_partial and (cells != expected or count != len(expected)):
            raise ValueError("Completed matrix differs from protocol")
        if not cells.issubset(expected):
            raise ValueError("Unregistered cells in partial input")
    if len(sources) > 1 and any(s["manifest"]["stage"] == "calibrate" for s in sources):
        raise ValueError("Do not mix scene calibration with performance")
    key = lambda r: (r["experiment"]["policy"], r["experiment"]["profile"], r["experiment"]["phase"], r["experiment"]["rep"])
    if Counter(map(key, rows)) != Counter(map(key, warmups)):
        raise ValueError("Warmup/formal matrix differs")
    if any(s["manifest"]["source_hashes"] != sources[0]["manifest"]["source_hashes"] for s in sources):
        raise ValueError("Do not combine source versions")
    return {"complete": all(s["completed"] is not None for s in sources) and not excluded,
            "sources": sources, "rows": rows, "warmups": len(warmups), "excluded": excluded,
            "counts": {"runs": len(rows), "requests": sum(r["count"] for r in rows),
                       "warmup_requests": sum(r["count"] for r in warmups)},
            "contrasts": paired(rows), "server_logs": logs, "artifact_sha256": artifacts,
            "scope": "Exploratory same-pool admission screen, not held-out, fairness, SOTA ranking or hybrid-exclusive attribution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    result = analyze(args.directories, args.allow_partial)
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output/"analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    lines = ["# M2 Unified Admission", "", f"Complete: {result['complete']}; {result['counts']}", "",
             "| Profile | Phase | Policy | Rep | Mean JCT | P95 | Makespan | Control delays | Scene errors |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for r in result["rows"]:
        e, m = r["experiment"], r["metrics"]
        lines.append(f"| {e['profile']} | {e['phase']} | {e['policy']} | {e['rep']} | {m['mean_jct_s']:.3f} | "
                     f"{m['p95_jct_s']:.3f} | {m['makespan_s']:.3f} | {control_delays(r)} | {r['scene_errors']} |")
    lines += ["", "Positive percent means improvement. At most two paired repeats; no precise CI. "
              "An incomplete pair matrix is pending, not a failed motivation.", "",
              "| Profile | Phase | Policy | Paired reps | Mean gain % | P95 gain % | Makespan gain % | Status |",
              "|---|---|---|---:|---:|---:|---:|---|"]
    for c in result["contrasts"]:
        g = c["mean_gain_percent"]
        lines.append(f"| {c['profile']} | {c['phase']} | {c['policy']} | {len(c['samples'])} | {g['mean_jct_s']:.2f} | "
                     f"{g['p95_jct_s']:.2f} | {g['makespan_s']:.2f} | {screen_status(c)} |")
    (args.output/"report.md").write_text("\n".join(lines)+"\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
