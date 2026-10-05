"""One fresh whole-session recovery after the native-control timing failure."""

import argparse
import asyncio
import contextlib
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time

from analyze_m2_unified_admission import analyze
from run_m2_unified_admission import (
    BASE, ROOT, command, dump, ensure_idle, hash_file, hashes, replay, stop_owned,
    wait_ready,
)


CASES = [("kv", "near"), ("kv", "far"), ("state", "near"), ("state", "far")]
POLICIES = ("delayer32", "slots64")


def check_adopted(rows):
    cells = {(r["experiment"]["policy"], r["experiment"]["profile"],
              r["experiment"]["phase"], r["experiment"]["rep"]) for r in rows}
    expected = {(p, profile, phase, 51) for p in POLICIES for profile, phase in CASES}
    if len(rows) != len(cells) or cells != expected:
        raise ValueError("Adopt exactly the two complete rep51 native-control sessions")
    return [(p, profile, phase, 52) for p in POLICIES for profile, phase in CASES]


def verify_gpu_identity(snapshots, expected):
    identity = {(g["index"], g["uuid"]) for g in expected}
    if any({(g["index"], g["uuid"]) for g in snapshot} != identity for snapshot in snapshots):
        raise ValueError("Physical GPU identity changed")


def prepare(original, output):
    original = original.resolve()
    result = analyze([original], allow_partial=True)
    missing = check_adopted(result["rows"])
    manifest = result["sources"][0]["manifest"]
    if manifest["stage"] != "native-controls" or hashes() != manifest["source_hashes"]:
        raise ValueError("Wrong stage or changed measured source")
    failures = []
    for excluded in result["excluded"]:
        for path in excluded["artifacts"]:
            item = json.loads(Path(path).read_text())
            if item["integrity_errors"]:
                failures.append({"source": path, "experiment": item["experiment"],
                                 "integrity_errors": item["integrity_errors"],
                                 "max_lateness_ms": 1000*max(r["submitted_s"]-r["scheduled_arrival_s"] for r in item["requests"])})
    record = {"original": str(original), "manifest": manifest, "missing": missing,
              "case_order": CASES, "policy_order": POLICIES,
              "adopted_sessions": sorted({str(Path(r["source"]).parent) for r in result["rows"]}),
              "adopted_artifact_sha256": result["artifact_sha256"],
              "excluded": result["excluded"], "failure_summaries": failures,
              "created_unix_s": time.time(), "port": 30144,
              "recovery_script_sha256": hash_file(Path(__file__)),
              "analysis_sha256": hash_file(BASE/"analyze_m2_unified_admission.py"),
              "retry_rule": "One fresh recovery attempt; stop for client/environment review if timing fails again."}
    dump(output, record)
    print(json.dumps({"plan": str(output), "adopted_runs": len(result["rows"]), "missing_runs": len(missing),
                      "failures": failures}), flush=True)


def verify_sources(plan):
    if hashes() != plan["manifest"]["source_hashes"] or hash_file(Path(__file__)) != plan["recovery_script_sha256"]:
        raise ValueError("Frozen measurement or recovery source changed")
    for path, expected in plan["adopted_artifact_sha256"].items():
        if hash_file(Path(path)) != expected:
            raise ValueError(f"Adopted artifact changed: {path}")


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    verify_sources(plan)
    manifest = plan["manifest"]
    args = argparse.Namespace(gpus=manifest["plan"]["physical_gpu_ids"], port=plan["port"])
    with contextlib.ExitStack() as stack:
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((BASE/f".m2_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = ensure_idle(args.port, args.gpus)
        verify_gpu_identity(initial, manifest["gpus"])
        output.mkdir(parents=True, exist_ok=False)
        dump(output/"recovery_manifest.json", {"plan": plan, "initial_gpus": initial, "created_unix_s": time.time()})
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        recovered = []
        for session, policy in enumerate(plan["policy_order"], 2):
            verify_sources(plan)
            verify_gpu_identity(ensure_idle(args.port, args.gpus), manifest["gpus"])
            directory = output/f"session-{session:02d}-{policy}-rep52"
            directory.mkdir()
            cmd = command(policy, args.port)
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
                   "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1",
                   "SGLANG_TENANT_PROGRESS_TRACE": "1", "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
                   "M2_ADMISSION_DECODE_PASSES": "0"}
            dump(directory/"session.json", {"command": cmd, "policy": policy, "rep": 52,
                 "http_keepalive_seconds": 1800, "diagnostic_passes": 0})
            with (directory/"server.log").open("x") as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    print(f"START native-control recovery {session} {policy} rep52 pid={process.pid}", flush=True)
                    wait_ready(process, f"http://127.0.0.1:{args.port}")
                    for profile, phase in plan["case_order"]:
                        arrival = manifest["plan"]["arrivals"][f"{profile}-{phase}"]
                        asyncio.run(replay(args, policy, profile, phase, 52, "warmup", arrival, directory))
                        asyncio.run(replay(args, policy, profile, phase, 52, "native-controls", arrival, directory))
                finally:
                    stop_owned(process)
            recovered.append(directory.resolve())
            print(f"STOP native-control recovery {session} {policy}", flush=True)
        verify_sources(plan)
        dump(output/"completed.json", {"completed": True, "runs": 8, "source_hashes": hashes(), "finished_unix_s": time.time()})
        assembled = output/"assembled"
        assembled.mkdir()
        for source in [*map(Path, plan["adopted_sessions"]), *recovered]:
            (assembled/source.name).symlink_to(source, target_is_directory=True)
        dump(assembled/"manifest.json", {**manifest, "recovery_provenance": plan,
             "kind": "assembled native-control matrix; not a continuous run"})
        dump(assembled/"completed.json", {"completed": True, "runs": 16, "source_hashes": hashes(), "finished_unix_s": time.time()})
        print(json.dumps({"assembled": str(assembled), "runs": 16}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("prepare", "run"), required=True)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "prepare":
        if args.original is None:
            parser.error("prepare needs --original")
        prepare(args.original, args.output)
    else:
        if args.plan is None:
            parser.error("run needs --plan")
        run(args.plan, args.output)


if __name__ == "__main__":
    main()
