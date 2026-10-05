"""Prepare or execute one whole missing session; never rewrite old results."""

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
    BASE, ROOT, command, dump, ensure_idle, gpu_snapshot, hash_file, hashes,
    replay, stop_owned, wait_ready,
)


CASES = [("kv", "near"), ("kv", "far"), ("state", "near"), ("state", "far")]


def check_missing(rows):
    expected = {(p, profile, phase, rep) for p in ("native", "decode32")
                for profile in ("state", "kv") for phase in ("far", "near") for rep in (51, 52)}
    actual = {(r["experiment"]["policy"], r["experiment"]["profile"], r["experiment"]["phase"], r["experiment"]["rep"])
              for r in rows}
    missing = {("native", p, phase, 52) for p, phase in CASES}
    if len(actual) != len(rows) or actual != expected-missing:
        raise ValueError("Recovery is only for the entire missing native/rep52 session")
    return sorted(missing)


def prepare(original, output):
    original = original.resolve()
    result = analyze([original], allow_partial=True)
    missing = check_missing(result["rows"])
    manifest = result["sources"][0]["manifest"]
    if hashes() != manifest["source_hashes"]:
        raise ValueError("Measured sources changed; cannot safely resume old matrix")
    sessions = sorted({str(Path(r["source"]).parent) for r in result["rows"]})
    record = {"original": str(original), "source_hashes": hashes(),
              "frozen_plan": manifest["plan"], "missing": missing, "case_order": CASES,
              "adopted_sessions": sessions, "adopted_artifact_sha256": result["artifact_sha256"],
              "excluded": result["excluded"], "physical_gpu_ids": manifest["plan"]["physical_gpu_ids"],
              "port": 30144, "created_unix_s": time.time(),
              "recovery_script_sha256": hash_file(Path(__file__)),
              "analysis_sha256": hash_file(BASE/"analyze_m2_unified_admission.py")}
    dump(output, record)
    print(json.dumps({"plan": str(output), "adopted_runs": len(result["rows"]), "missing": missing}), flush=True)


def run(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if hashes() != plan["source_hashes"] or hash_file(Path(__file__)) != plan["recovery_script_sha256"]:
        raise ValueError("Frozen measurement or recovery source changed")
    for path, expected in plan["adopted_artifact_sha256"].items():
        if hash_file(Path(path)) != expected:
            raise ValueError(f"Adopted artifact changed: {path}")
    args = argparse.Namespace(gpus=plan["physical_gpu_ids"], port=plan["port"])
    with contextlib.ExitStack() as stack:
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((BASE/f".m2_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = ensure_idle(args.port, args.gpus)
        output.mkdir(parents=True, exist_ok=False)
        dump(output/"recovery_manifest.json", {"plan": plan, "initial_gpus": initial,
             "created_unix_s": time.time(), "kind": "whole-session recovery"})
        directory = output/"session-03-native-rep52"
        directory.mkdir()
        cmd = command("native", args.port)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
               "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1",
               "SGLANG_TENANT_PROGRESS_TRACE": "1", "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
               "M2_ADMISSION_DECODE_PASSES": "0"}
        dump(directory/"session.json", {"command": cmd, "policy": "native", "rep": 52,
             "http_keepalive_seconds": 1800, "diagnostic_passes": 0})
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        with (directory/"server.log").open("x") as log:
            process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                print(f"START recovery native rep52 pid={process.pid}", flush=True)
                wait_ready(process, f"http://127.0.0.1:{args.port}")
                for profile, phase in plan["case_order"]:
                    arrival = plan["frozen_plan"]["arrivals"][f"{profile}-{phase}"]
                    asyncio.run(replay(args, "native", profile, phase, 52, "warmup", arrival, directory))
                    asyncio.run(replay(args, "native", profile, phase, 52, "screen", arrival, directory))
            finally:
                stop_owned(process)
        if hashes() != plan["source_hashes"] or hash_file(Path(__file__)) != plan["recovery_script_sha256"]:
            raise ValueError("Sources changed during recovery")
        dump(output/"completed.json", {"completed": True, "runs": 4, "source_hashes": hashes(),
             "finished_unix_s": time.time(), "recovery_script_sha256": plan["recovery_script_sha256"]})
        assembled = output/"assembled"
        assembled.mkdir()
        for source in [*map(Path, plan["adopted_sessions"]), directory.resolve()]:
            (assembled/source.name).symlink_to(source, target_is_directory=True)
        original_manifest = json.loads((Path(plan["original"])/"manifest.json").read_text())
        dump(assembled/"manifest.json", {**original_manifest, "recovery_provenance": plan,
             "kind": "assembled complete matrix; not a continuous run"})
        dump(assembled/"completed.json", {"completed": True, "runs": 16, "source_hashes": hashes(),
             "kind": "assembled complete matrix", "finished_unix_s": time.time()})
        print(json.dumps({"assembled": str(assembled), "gpus": gpu_snapshot(args.gpus)}), flush=True)


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
