#!/usr/bin/env python3
"""Recover whole calibration sessions without changing frozen v3 measurements."""

import argparse
import asyncio
import copy
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import run_m2_state_mediation as frozen


def session_files(session):
    pool, mode, rep, caps = session
    return [f"m2mediate-{pool}-{mode}-cap{cap}-rep{rep}-{stage}.json"
            for cap in caps for stage in ("warmup", "calibrate")]


def recovery_selection(plan, names, keep):
    if not 0 < keep < len(plan):
        raise ValueError("Recovery requires a completed prefix and remaining sessions")
    adopted = [name for session in plan[:keep] for name in session_files(session)]
    if not set(adopted) <= names:
        raise ValueError("Cannot adopt a partial session")
    # Reaching the next session demonstrates the previous session exited its finally block.
    for pool, mode, rep, _ in plan[1:keep + 1]:
        if f"server-{pool}-{mode}-rep{rep}.log" not in names:
            raise ValueError("Missing next-session evidence")
    expected = {name for session in plan for name in session_files(session)}
    artifacts = {name for name in names if name.startswith("m2mediate-") and name.endswith(".json")}
    if artifacts - expected:
        raise ValueError("Unexpected parent artifacts")
    return adopted, sorted(artifacts - set(adopted))


def recovery_hashes():
    return {str(path): frozen.hash_file(path) for path in
            (Path(__file__), frozen.BASE / "m2_state_mediation_interruption_20260916.md")}


def validate_recovery(directory, manifest, done):
    recovery = manifest.get("recovery")
    if recovery is None:
        return
    if done.get("recovery") != recovery:
        raise ValueError("Recovery completion provenance mismatch")
    if recovery["source_hashes"] != recovery_hashes():
        raise ValueError("Recovery driver/protocol changed")
    parent = Path(recovery["parent"])
    if frozen.hash_file(parent / "manifest.json") != recovery["parent_manifest_sha256"]:
        raise ValueError("Parent manifest changed")
    selected, excluded = recovery_selection(manifest["session_plan"], set(recovery["parent_files"]), recovery["keep_sessions"])
    if set(selected) != set(recovery["adopted_sha256"]) or excluded != recovery["excluded_artifacts"]:
        raise ValueError("Recovery selection mismatch")
    for name, expected in recovery["adopted_sha256"].items():
        if frozen.hash_file(directory / name) != expected or frozen.hash_file(parent / name) != expected:
            raise ValueError("Adopted artifact changed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    parent = args.parent.resolve()
    original = json.loads((parent / "manifest.json").read_text())
    hashes = frozen.measurement_hashes()
    plan = frozen.session_plan("calibrate")
    if ((parent / "completed.json").exists() or original["args"]["stage"] != "calibrate"
            or original["source_hashes"] != hashes
            or original["session_plan"] != json.loads(json.dumps(plan))):
        raise ValueError("Parent must be the interrupted, unchanged v3 calibration")
    reference = original["capacity_reference"]
    names = {p.name for p in parent.iterdir()}
    adopted, excluded = recovery_selection(plan, names, 4)
    manifest = copy.deepcopy(original)
    manifest["args"]["output"] = str(args.output)
    manifest["recovery"] = {
        "reason": "User turn interruption terminated original runner; retain complete sessions only",
        "parent": str(parent), "parent_manifest_sha256": frozen.hash_file(parent / "manifest.json"),
        "parent_files": sorted(names), "keep_sessions": 4,
        "adopted_sha256": {name: frozen.hash_file(parent / name) for name in adopted},
        "excluded_artifacts": excluded, "source_hashes": recovery_hashes(),
        "rerun_sessions": plan[4:],
    }
    for name in adopted:
        item = json.loads((parent / name).read_text())
        errors = frozen.integrity_errors(item, frozen.COUNT, .05) + frozen.progress_errors(item)
        if item["integrity_errors"] or errors:
            raise ValueError(f"Invalid adopted artifact: {name}: {errors}")
        e = item["experiment"]
        for field in ("server_initial", "server_final"):
            if list(frozen.validate_info(item[field], e["mode"], e["pool"], e["cap"], reference)) != item["resolved_capacity"]:
                raise ValueError("Adopted capacity mismatch")
    replay_args = argparse.Namespace(output=args.output, port=original["args"]["port"])
    with (frozen.BASE / ".m2_gpu67.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest["recovery"]["gpu_snapshot"] = frozen.ensure_available(replay_args.port)
        manifest["recovery"]["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.output.mkdir(parents=True, exist_ok=False)
        frozen.dump(args.output / "manifest.json", manifest)
        for name in adopted:
            shutil.copy2(parent / name, args.output / name)
        for pool, mode, rep, _ in plan[:4]:
            name = f"server-{pool}-{mode}-rep{rep}.log"
            shutil.copy2(parent / name, args.output / name)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "6,7", "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1",
               "SGLANG_TENANT_EVENT_TRACE": "1", "SGLANG_TENANT_PROGRESS_TRACE": "1"}

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"received {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        for pool, mode, rep, caps in plan[4:]:
            frozen.ensure_available(replay_args.port)
            if frozen.measurement_hashes() != hashes or recovery_hashes() != manifest["recovery"]["source_hashes"]:
                raise RuntimeError("Measurement/recovery source changed")
            with (args.output / f"server-{pool}-{mode}-rep{rep}.log").open("x") as log:
                process = subprocess.Popen(frozen.command(mode, pool, replay_args.port), cwd=frozen.ROOT,
                                           env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    print(f"RECOVERY START {pool}/{mode} rep={rep} pid={process.pid}", flush=True)
                    frozen.wait_ready(process, f"http://127.0.0.1:{replay_args.port}")
                    for cap in caps:
                        for stage in ("warmup", "calibrate"):
                            asyncio.run(frozen.replay(replay_args, pool, mode, rep, cap, stage, reference))
                finally:
                    frozen.stop_owned_server(process)
        if frozen.measurement_hashes() != hashes:
            raise RuntimeError("Measurement source changed")
        done = {"completed": True, "source_hashes": hashes, "recovery": manifest["recovery"]}
        validate_recovery(args.output, manifest, done)
        frozen.dump(args.output / "completed.json", done)


if __name__ == "__main__":
    main()
