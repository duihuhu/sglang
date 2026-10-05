"""Calibrate native controls without relabeling a request cap as a state ablation."""

import argparse
import asyncio
import contextlib
import dataclasses
import fcntl
import gc
import json
import os
from pathlib import Path
import signal
import subprocess
import time

import aiohttp

import run_m2_unified_admission as base
from run_m2_pool_comparison import capacity


POLICIES = ("native607", "cap512", "slots96")
RAW_BYTES = 19481072000
REPS = (61, 62)


def cap_for(policy):
    if policy not in POLICIES:
        raise ValueError("Unregistered policy")
    return 512 if policy == "cap512" else 607


def command(policy, port):
    cap_for(policy)
    cmd = base.command("native", port)
    return cmd + (["--min-free-slots-delay", "96"] if policy == "slots96" else [])


def validate(info, policy):
    adapted = json.loads(json.dumps(info))
    for state in adapted["internal_states"]:
        expected = 96 if policy == "slots96" else None
        if state["min_free_slots_delay"] != expected or state["enable_prefill_delayer"]:
            raise ValueError("Unexpected native control")
        audit = state["m2_admission"]
        if audit["diagnostic_limit"] != 0 or audit["shared_pool_bytes"] != RAW_BYTES:
            raise ValueError("Diagnostic action or physical pool changed")
        state["min_free_slots_delay"] = None
    resolved = capacity(adapted, base.MODE, base.ARM, cap_for(policy))
    if resolved[:3] != (607, base.KV_CAP, 607):
        raise ValueError(f"Bootstrap/request capacity changed: {resolved}")
    return resolved


def session_plan():
    return [
        {"rep": 61, "server_policy": "native607", "cases": [(p, s) for s in ("far", "near") for p in ("native607", "cap512")]},
        {"rep": 61, "server_policy": "slots96", "cases": [("slots96", s) for s in ("far", "near")]},
        {"rep": 62, "server_policy": "slots96", "cases": [("slots96", s) for s in ("near", "far")]},
        {"rep": 62, "server_policy": "native607", "cases": [(p, s) for s in ("near", "far") for p in ("cap512", "native607")]},
    ]


def hashes():
    result = base.hashes()
    for path in (Path(__file__), base.BASE / "m2_native_counterexample_protocol_20260917.md"):
        result[str(path)] = base.hash_file(path)
    return result


async def replay(args, policy, phase, rep, stage, arrival, directory):
    run_id = f"m2nc-{policy}-state-{phase}-rep{rep}-{stage}"
    specs = base.specs_for(run_id, "state", -rep if stage == "warmup" else rep, arrival)
    url = f"http://127.0.0.1:{args.port}"
    trace, connections = base.connection_audit()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=1800), trace_configs=[trace]) as client:
        await base.configure(client, url, run_id, rep, cap_for(policy))
        initial = await base.get_json(client, url + "/get_server_info")
        resolved = validate(initial, policy)
        connections["phase"] = "preconnect"
        await base.preconnect(client, url, len(specs))
        async with base.SelectedGpuMonitor(args.gpus) as health:
            gc.collect()
            enabled = gc.isenabled()
            gc.disable()
            start = time.perf_counter() + .5
            stop, samples = asyncio.Event(), []
            monitor = asyncio.create_task(health.monitor(stop, samples))
            connections["phase"] = "replay"
            try:
                results = await base.run_specs(client, url, specs, start, run_id)
            finally:
                stop.set()
                try:
                    await monitor
                finally:
                    if enabled:
                        gc.enable()
                    connections["phase"] = "after"
        final = await base.get_json(client, url + "/get_server_info")
        if validate(final, policy) != resolved:
            raise ValueError("Capacity changed within replay")
    item = {"experiment": {"run_id": run_id, "policy": policy, "profile": "state", "phase": phase,
                "rep": rep, "stage": stage, "arrival": arrival, "activation_limit": 0},
            "physical_gpu_ids": args.gpus, "resolved_capacity": resolved,
            "payload_fingerprint": base.fingerprint(specs), "client_timing": connections,
            "experiment_clock_origin_s": start, "server_initial": initial, "server_final": final,
            "scheduler_resource_trace": base.extract_trace(final, run_id),
            "scheduler_batch_trace": base.extract_batch_trace(final, run_id),
            "scheduler_vtc_trace": base.extract_vtc_trace(final, run_id), "gpu_health_samples": samples,
            "summary": base.summarize(results, 60),
            "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s} for r in results]}
    errors = base.integrity_errors(item, len(specs), .05) + base.progress_errors(item)
    if connections["created"].get("replay", 0) or connections["reused"].get("replay") != len(specs):
        errors.append("HTTP replay connection mismatch")
    item["integrity_errors"] = errors
    item["arrival_snapshot"] = base.arrival_snapshot(item, arrival)
    item["scene_errors"] = base.scene_errors(item["arrival_snapshot"], "state", phase)
    base.dump(directory / f"{run_id}.json", item)
    print(json.dumps({"run": run_id, "errors": errors, "scene_errors": item["scene_errors"],
                      "peak_active": item["scheduler_resource_trace"]["system"]["peak_active_requests"]}), flush=True)
    if errors or item["scene_errors"]:
        raise ValueError(errors + item["scene_errors"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-plan", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--port", type=int, default=30144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reference = json.loads(args.reference_plan.read_text())
    if (not reference["scenes_valid"] or reference["source_hashes"] != base.hashes()
            or reference["physical_gpu_ids"] != args.gpus or len(set(args.gpus)) != 2):
        raise ValueError("Need unchanged sources, valid reference scenes and same GPUs")
    expected_hashes, reference_hash = hashes(), base.hash_file(args.reference_plan)
    with contextlib.ExitStack() as stack:
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((base.BASE / f".m2_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = base.ensure_idle(args.port, args.gpus)
        identities = {(g["index"], g["uuid"]) for g in initial[0]}
        args.output.mkdir(parents=True, exist_ok=False)
        base.dump(args.output / "manifest.json", {"source_hashes": expected_hashes,
            "reference": reference, "reference_sha256": reference_hash, "sessions": session_plan(),
            "initial_gpus": initial, "created_unix_s": time.time(), "scope": "State-only native calibration"})
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        for index, session in enumerate(session_plan()):
            if hashes() != expected_hashes or base.hash_file(args.reference_plan) != reference_hash:
                raise ValueError("Frozen source/plan changed")
            snapshots = base.ensure_idle(args.port, args.gpus)
            if any({(g["index"], g["uuid"]) for g in s} != identities for s in snapshots):
                raise ValueError("Physical GPU identity changed")
            directory = args.output / f"session-{index:02d}-{session['server_policy']}-rep{session['rep']}"
            directory.mkdir()
            cmd = command(session["server_policy"], args.port)
            base.dump(directory / "session.json", {**session, "command": cmd})
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
                   "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1",
                   "SGLANG_TENANT_PROGRESS_TRACE": "1", "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
                   "M2_ADMISSION_DECODE_PASSES": "0"}
            with (directory / "server.log").open("x") as log:
                process = subprocess.Popen(cmd, cwd=base.ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    print(f"START native calibration {index} {session} pid={process.pid}", flush=True)
                    base.wait_ready(process, f"http://127.0.0.1:{args.port}")
                    for policy, phase in session["cases"]:
                        for stage in ("warmup", "calibration"):
                            asyncio.run(replay(args, policy, phase, session["rep"], stage,
                                reference["arrivals"][f"state-{phase}"], directory))
                finally:
                    base.stop_owned(process)
            base.dump(directory / "completed.json", {"completed": True, "formal_runs": len(session["cases"])})
            print(f"STOP native calibration {index}", flush=True)
        if hashes() != expected_hashes or base.hash_file(args.reference_plan) != reference_hash:
            raise ValueError("Sources changed at completion")
        base.dump(args.output / "completed.json", {"completed": True, "runs": 12,
            "source_hashes": expected_hashes, "finished_unix_s": time.time()})
        print(json.dumps({"completed": True, "runs": 12, "gpus": base.gpu_snapshot(args.gpus)}), flush=True)


if __name__ == "__main__":
    main()
