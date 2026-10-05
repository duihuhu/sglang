#!/usr/bin/env python3
"""Calibrate native batching and intervene on the state-derived cap."""

import argparse
import asyncio
import dataclasses
import fcntl
import gc
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import aiohttp

from m2_streaming_gpu_health import StreamingGpuMonitor
from run_m2_activation_pacing import command as base_command, integrity_errors
from run_m2_batch_frontier import BASE, ROOT, case_args, dump, ensure_available, hash_file, stop_owned_server, wait_ready
from run_m2_hybrid_factors import fingerprint
from run_m2_progress_probe import measurement_hashes as base_hashes, progress_errors, specs_for, validate_natural_info
from validate_admission_batch_motivation import extract_batch_trace
from validate_dynamic_cost_motivation import extract_trace, flush_cache, get_json, post_json, run_specs, summarize
from validate_vtc_state_tax_motivation import apply_runtime_policy, extract_vtc_trace


REPS = (11, 12)
CALIBRATION_MODES = ("separate-4096", "mixed-4096", "mixed-1024")
MEDIATION_MODES = ("separate-4096", "mixed-4096")
CAPS = {"natural": (128, 256, 485), "wide": (485, 607)}
COUNT = 607
MIN_KV = 2 * COUNT * (512 + 128)


async def preconnect(client, url, count):
    async def request():
        # This static endpoint performs no generation and does not touch caches.
        async with client.get(url + "/model_info") as response:
            response.raise_for_status()
            await response.read()
    await asyncio.gather(*(request() for _ in range(count)))


def connection_audit():
    state = {"phase": "setup", "created": {}, "reused": {}}
    trace = aiohttp.TraceConfig()
    async def created(session, context, params):
        phase = state["phase"]
        state["created"][phase] = state["created"].get(phase, 0) + 1
    async def reused(session, context, params):
        phase = state["phase"]
        state["reused"][phase] = state["reused"].get(phase, 0) + 1
    trace.on_connection_create_start.append(created)
    trace.on_connection_reuseconn.append(reused)
    return trace, state


def command(mode, pool, port):
    if pool not in CAPS or mode not in CALIBRATION_MODES:
        raise ValueError("Unknown configuration")
    args = base_command(mode, "natural", port)
    if pool == "wide":
        args += ["--max-mamba-cache-size", "728"]
    return args


def validate_info(info, mode, pool, cap, reference=None):
    capacity = validate_natural_info(info, mode)
    state, kv, requests, runtime = capacity
    expected_state = {"natural": 485, "wide": 728}[pool]
    if (state, requests, runtime) != (expected_state, expected_state, cap):
        raise ValueError(f"Wrong pool/cap: {capacity}, expected {pool}/{cap}")
    if kv < MIN_KV:
        raise ValueError(f"Insufficient KV headroom: {kv} < {MIN_KV}")
    for s in info["internal_states"]:
        for key, value in {"mem_fraction_static": .85, "max_prefill_tokens": 16384,
                           "max_running_requests": None, "enable_unified_memory": False,
                           "speculative_algorithm": None}.items():
            if s.get(key) != value or key not in s:
                raise ValueError(f"{key}: {s.get(key)!r} != {value!r}")
    if reference is not None and list(capacity[:3]) != reference[pool]:
        raise ValueError("Physical capacity changed from inspection")
    return capacity


def measurement_hashes():
    result = base_hashes()
    for path in (Path(__file__), BASE / "m2_state_mediation_protocol_20260916.md",
                 BASE / "m2_streaming_gpu_health.py",
                 ROOT / "python/sglang/srt/server_args.py",
                 ROOT / "python/sglang/srt/configs/mamba_utils.py",
                 ROOT / "python/sglang/srt/configs/kimi_linear.py"):
        result[str(path)] = hash_file(path)
    return result


def workload(run_id, rep, stage):
    # Warmups use the full shape, including the high-concurrency execution path.
    return specs_for(run_id, "inspect" if stage == "inspect" else "state-pressure",
                     -rep if stage == "warmup" else rep, COUNT)


def session_plan(stage):
    if stage == "inspect":
        return [("natural", "mixed-4096", 1, [485]), ("wide", "mixed-4096", 1, [607])]
    pool = "natural" if stage == "calibrate" else "wide"
    modes = list(CALIBRATION_MODES if pool == "natural" else MEDIATION_MODES)
    random.Random(20260916).shuffle(modes)
    result = []
    for index, rep in enumerate(REPS):
        for mode in modes if index == 0 else reversed(modes):
            caps = list(CAPS[pool])
            random.Random(20260916 + rep + CALIBRATION_MODES.index(mode)).shuffle(caps)
            result.append((pool, mode, rep, caps))
    return result


async def replay(args, pool, mode, rep, cap, stage, reference):
    run_id = f"m2mediate-{pool}-{mode}-cap{cap}-rep{rep}-{stage}"
    specs = workload(run_id, rep, stage)
    url = f"http://127.0.0.1:{args.port}"
    policy = case_args(run_id, rep, "mixed", "vtc", 16, 1, 2)
    connection_trace, connections = connection_audit()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
                                     connector=aiohttp.TCPConnector(limit=0),
                                     trace_configs=[connection_trace]) as client:
        await apply_runtime_policy(client, url, policy)
        changed = await post_json(client, url + "/set_internal_state", {"server_args": {
            "tenant_slot_gate_code": 1, "tenant_new_activation_limit": 0,
            "pp_max_micro_batch_size": cap}})
        if not changed or not all(changed):
            raise RuntimeError("Runtime policy update failed")
        await flush_cache(client, url)
        initial = await get_json(client, url + "/get_server_info")
        capacity = validate_info(initial, mode, pool, cap, reference)
        connections["phase"] = "preconnect"
        await preconnect(client, url, len(specs))
        async with StreamingGpuMonitor() as health:
            gc.collect()
            gc_was_enabled = gc.isenabled()
            gc.disable()
            start = time.perf_counter() + .5
            samples, stop = [], asyncio.Event()
            monitoring = asyncio.create_task(health.monitor(stop, samples))
            connections["phase"] = "replay"
            try:
                results = await run_specs(client, url, specs, start, run_id)
            finally:
                stop.set()
                try:
                    await monitoring
                finally:
                    if gc_was_enabled:
                        gc.enable()
                    connections["phase"] = "after"
        final = await get_json(client, url + "/get_server_info")
        if validate_info(final, mode, pool, cap, reference) != capacity:
            raise RuntimeError("Capacity changed within run")
    item = {
        "experiment": {"run_id": run_id, "mode": mode, "pool": pool, "rep": rep,
                       "cap": cap, "stage": stage, "activation_limit": 0},
        "physical_gpu_ids": [6, 7], "payload_fingerprint": fingerprint(specs),
        "client_timing": {"gc_disabled_during_replay": True, "health_monitor": "persistent-nvidia-smi-500ms",
                          "http_preconnect_count": len(specs), "connection_events": connections},
        "experiment_clock_origin_s": start, "resolved_capacity": capacity,
        "server_initial": initial, "server_final": final,
        "scheduler_resource_trace": extract_trace(final, run_id),
        "scheduler_batch_trace": extract_batch_trace(final, run_id),
        "scheduler_vtc_trace": extract_vtc_trace(final, run_id),
        "gpu_health_samples": samples, "summary": summarize(results, 60),
        "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s}
                     for r in results],
    }
    item["integrity_errors"] = integrity_errors(item, len(specs), .05) + progress_errors(item)
    if connections["created"].get("replay", 0):
        item["integrity_errors"].append("unprimed HTTP connections during replay")
    dump(args.output / f"{run_id}.json", item)
    print(json.dumps({"run": run_id, "requests": len(specs), "capacity": capacity,
                      "peak_active": item["scheduler_resource_trace"]["system"]["peak_active_requests"],
                      "errors": item["integrity_errors"]}), flush=True)
    if item["integrity_errors"]:
        raise RuntimeError(item["integrity_errors"])
    return list(capacity[:3])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "calibrate", "mediate"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-reference", type=Path)
    parser.add_argument("--calibration-directory", type=Path)
    parser.add_argument("--port", type=int, default=30133)
    args = parser.parse_args()
    hashes = measurement_hashes()
    reference = None
    if args.stage != "inspect":
        if args.capacity_reference is None:
            parser.error("Measurement requires inspection reference")
        loaded = json.loads(args.capacity_reference.read_text())
        done = json.loads((args.capacity_reference.parent / "completed.json").read_text())
        if not done.get("completed") or done["source_hashes"] != hashes or loaded["source_hashes"] != hashes:
            raise ValueError("Inspection/source mismatch")
        reference = loaded["pools"]
    if args.stage == "mediate":
        if args.calibration_directory is None:
            parser.error("Mediation follows completed calibration")
        done = json.loads((args.calibration_directory / "completed.json").read_text())
        if not done.get("completed") or done["source_hashes"] != hashes:
            raise ValueError("Calibration incomplete or source changed")
    plan = session_plan(args.stage)
    count = sum(len(caps) for _, _, _, caps in plan)
    manifest = {"args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                "source_hashes": hashes, "capacity_reference": reference,
                "session_plan": plan, "planned_measured_runs": count,
                "planned_measured_requests": count * (3 if args.stage == "inspect" else COUNT),
                "planned_warmup_runs": 0 if args.stage == "inspect" else count,
                "commands": {f"{p}-{m}": command(m, p, args.port) for p, m, _, _ in plan}}
    with (BASE / ".m2_gpu67.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest["gpu_snapshot"] = ensure_available(args.port)
        manifest["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output / "manifest.json", manifest)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": "6,7", "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1",
               "SGLANG_TENANT_EVENT_TRACE": "1", "SGLANG_TENANT_PROGRESS_TRACE": "1"}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"received {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        inspected = {}
        for pool, mode, rep, caps in plan:
            ensure_available(args.port)
            if measurement_hashes() != hashes:
                raise RuntimeError("Measurement sources changed")
            with (args.output / f"server-{pool}-{mode}-rep{rep}.log").open("x") as log:
                process = subprocess.Popen(command(mode, pool, args.port), cwd=ROOT, env=env,
                                           stdout=log, stderr=log, start_new_session=True)
                try:
                    print(f"START {pool}/{mode} rep={rep} pid={process.pid}", flush=True)
                    wait_ready(process, f"http://127.0.0.1:{args.port}")
                    for cap in caps:
                        if args.stage != "inspect":
                            asyncio.run(replay(args, pool, mode, rep, cap, "warmup", reference))
                        inspected[pool] = asyncio.run(replay(args, pool, mode, rep, cap, args.stage, reference))
                finally:
                    stop_owned_server(process)
        if measurement_hashes() != hashes:
            raise RuntimeError("Measurement sources changed")
        if args.stage == "inspect":
            dump(args.output / "capacity.json", {"pools": inspected, "source_hashes": hashes})
        dump(args.output / "completed.json", {"completed": True, "source_hashes": hashes})


if __name__ == "__main__":
    main()
