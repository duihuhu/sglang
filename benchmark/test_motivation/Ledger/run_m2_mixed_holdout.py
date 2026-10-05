"""Test frozen native controls and a bounded diagnostic in Mixed batching."""

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
import statistics
import subprocess
import time

import aiohttp

import run_m2_unified_admission as base
from run_m2_pool_comparison import capacity, command as pool_command


MODE = "mixed-4096"
RAW_BYTES = 19481072000
PROFILES = {"train512": (512, 512), "hold384": (384, 1024)}
POLICIES = ("native607", "cap512", "slots96", "decode32")
REPS = (81, 82)


def cap_for(policy):
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    return 512 if policy == "cap512" else 607


def command(policy, port):
    cap_for(policy)
    cmd = pool_command(MODE, base.ARM, port, kv_tokens=base.KV_CAP)
    cmd = [cmd[0], str(base.BASE / "launch_m2_unified_admission.py"), *cmd[3:]]
    return cmd + (["--min-free-slots-delay", "96"] if policy == "slots96" else [])


def validate(info, policy):
    adapted = json.loads(json.dumps(info))
    for state in adapted["internal_states"]:
        if (state["min_free_slots_delay"] != (96 if policy == "slots96" else None)
                or state["enable_prefill_delayer"]):
            raise ValueError("Unexpected native control")
        audit = state["m2_admission"]
        if (audit["diagnostic_limit"] != (32 if policy == "decode32" else 0)
                or audit["shared_pool_bytes"] != RAW_BYTES):
            raise ValueError("Diagnostic or pool mismatch")
        state["min_free_slots_delay"] = None
    resolved = capacity(adapted, MODE, base.ARM, cap_for(policy))
    if resolved[:3] != (607, base.KV_CAP, 607):
        raise ValueError("Bootstrap changed")
    return resolved


def specs_for(run_id, profile, rep, arrival):
    count, length = PROFILES[profile]
    specs = [base.RequestSpec(
        rid=f"{run_id}-holder-{i}", group="holder",
        input_ids=base.unique_input(length, 121000 + rep * 607 + i),
        output_len=128, arrival_offset_s=i * .001,
        cache_salt=f"{run_id}-h-{i}", tenant_key="shared",
    ) for i in range(count)]
    if arrival is not None:
        specs += [base.RequestSpec(
            rid=f"{run_id}-candidate-{i}", group="candidate",
            input_ids=base.unique_input(8192, 181000 + rep * 64 + i),
            output_len=64, arrival_offset_s=arrival + i * .001,
            cache_salt=f"{run_id}-c-{i}", tenant_key="shared",
        ) for i in range(64)]
    return specs


def session_plan(stage):
    if stage == "calibrate":
        return [{"rep": rep, "server_policy": "native607",
                 "cases": [("native607", p, "background") for p in
                           (tuple(PROFILES) if rep == 71 else tuple(reversed(PROFILES)))]}
                for rep in (71, 72)]
    if stage != "measure":
        raise ValueError("Unknown stage")
    cells = [(p, phase) for p in PROFILES for phase in ("far", "near")]
    first = [
        {"rep": 81, "server_policy": "native607",
         "cases": [(policy, p, phase) for p, phase in cells for policy in ("native607", "cap512")]},
        *[{"rep": 81, "server_policy": policy,
           "cases": [(policy, p, phase) for p, phase in cells]} for policy in ("slots96", "decode32")],
    ]
    return first + [{**s, "rep": 82, "cases": list(reversed(s["cases"]))} for s in reversed(first)]


def hashes():
    result = base.hashes()
    for name in ("run_m2_mixed_holdout.py", "analyze_m2_mixed_holdout.py",
                 "m2_mixed_holdout_protocol_20260917.md"):
        path = base.BASE / name
        result[str(path)] = base.hash_file(path)
    return result


async def replay(args, policy, profile, phase, rep, stage, arrival, directory):
    run_id = f"m2mh-{policy}-{profile}-{phase}-rep{rep}-{stage}"
    specs = specs_for(run_id, profile, -rep if stage == "warmup" else rep, arrival)
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
            raise ValueError("Capacity changed during replay")
    item = {"experiment": {"run_id": run_id, "policy": policy, "profile": profile,
                "phase": phase, "rep": rep, "stage": stage, "mode": MODE,
                "arrival": arrival, "activation_limit": 0},
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
    item["scene_errors"] = []
    if arrival is not None:
        item["arrival_snapshot"] = base.arrival_snapshot(item, arrival)
        item["scene_errors"] = base.scene_errors(item["arrival_snapshot"], "state", phase)
    base.dump(directory / f"{run_id}.json", item)
    print(json.dumps({"run": run_id, "errors": errors, "scene_errors": item["scene_errors"],
                      "peak_active": item["scheduler_resource_trace"]["system"]["peak_active_requests"]}), flush=True)
    if errors:
        raise ValueError(errors)
    return item


def freeze_arrivals(items):
    arrivals, evidence = {}, []
    for profile in PROFILES:
        own = [i for i in items if i["experiment"]["profile"] == profile]
        if len(own) != 2:
            raise ValueError("Need two background runs per profile")
        for phase, position in (("far", 16), ("near", 96)):
            anchors = [statistics.median(next(t for t, n in r["token_events"] if n >= position)
                                        for r in i["requests"]) for i in own]
            at = statistics.fmean(anchors)
            arrivals[f"{profile}-{phase}"] = at
            for item in own:
                snapshot = base.arrival_snapshot(item, at)
                errors = base.scene_errors(snapshot, "state", phase)
                evidence.append({"profile": profile, "phase": phase, "rep": item["experiment"]["rep"],
                                 "snapshot": snapshot, "errors": errors})
    return {"arrivals": arrivals, "evidence": evidence,
            "scenes_valid": not any(e["errors"] for e in evidence)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "measure"), required=True)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--port", type=int, default=30144)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("Need two distinct physical GPUs")
    expected_hashes = hashes()
    plan = json.loads(args.plan.read_text()) if args.plan else None
    if args.stage == "measure" and (not plan or not plan["scenes_valid"]
            or plan["source_hashes"] != expected_hashes or plan["physical_gpu_ids"] != args.gpus):
        parser.error("Need a valid unchanged calibration plan on the same GPUs")
    plan_hash = base.hash_file(args.plan) if args.plan else None
    sessions = session_plan(args.stage)
    items = []
    formal_count = 0
    with contextlib.ExitStack() as stack:
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((base.BASE / f".m2_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = base.ensure_idle(args.port, args.gpus)
        identities = {(g["index"], g["uuid"]) for g in initial[0]}
        if plan and identities != {(g["index"], g["uuid"]) for g in plan["gpus"]}:
            raise ValueError("Calibration GPU identity changed")
        args.output.mkdir(parents=True, exist_ok=False)
        base.dump(args.output / "manifest.json", {"stage": args.stage, "source_hashes": expected_hashes,
            "plan": plan, "plan_sha256": plan_hash, "sessions": sessions,
            "initial_gpus": initial, "created_unix_s": time.time()})
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        for index, session in enumerate(sessions):
            if hashes() != expected_hashes or (args.plan and base.hash_file(args.plan) != plan_hash):
                raise ValueError("Frozen sources/plan changed")
            snapshots = base.ensure_idle(args.port, args.gpus)
            if any({(g["index"], g["uuid"]) for g in s} != identities for s in snapshots):
                raise ValueError("Physical GPU identity changed")
            directory = args.output / f"session-{index:02d}-{session['server_policy']}-rep{session['rep']}"
            directory.mkdir()
            policy = session["server_policy"]
            cmd = command(policy, args.port)
            base.dump(directory / "session.json", {**session, "command": cmd})
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
                   "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1",
                   "SGLANG_TENANT_PROGRESS_TRACE": "1", "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
                   "M2_ADMISSION_DECODE_PASSES": "32" if policy == "decode32" else "0"}
            with (directory / "server.log").open("x") as log:
                process = subprocess.Popen(cmd, cwd=base.ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    print(f"START mixed holdout {index} {session} pid={process.pid}", flush=True)
                    base.wait_ready(process, f"http://127.0.0.1:{args.port}")
                    for selected, profile, phase in session["cases"]:
                        at = plan["arrivals"][f"{profile}-{phase}"] if plan else None
                        for stage in ("warmup", args.stage):
                            item = asyncio.run(replay(args, selected, profile, phase, session["rep"], stage, at, directory))
                            if stage != "warmup":
                                formal_count += 1
                                if args.stage == "calibrate":
                                    items.append(item)
                finally:
                    base.stop_owned(process)
            base.dump(directory / "completed.json", {"completed": True, "formal_runs": len(session["cases"])})
            print(f"STOP mixed holdout {index}", flush=True)
        if hashes() != expected_hashes or (args.plan and base.hash_file(args.plan) != plan_hash):
            raise ValueError("Frozen inputs changed at completion")
        if args.stage == "calibrate":
            frozen = {**freeze_arrivals(items), "source_hashes": expected_hashes,
                      "physical_gpu_ids": args.gpus, "gpus": initial[0]}
            base.dump(args.output / "arrival-plan.json", frozen)
            if not frozen["scenes_valid"]:
                raise ValueError("Background calibration scene failed; retain all evidence")
        base.dump(args.output / "completed.json", {"completed": True, "runs": formal_count,
            "source_hashes": expected_hashes, "finished_unix_s": time.time()})
        print(json.dumps({"completed": True, "runs": formal_count, "gpus": base.gpu_snapshot(args.gpus)}), flush=True)


if __name__ == "__main__":
    main()
