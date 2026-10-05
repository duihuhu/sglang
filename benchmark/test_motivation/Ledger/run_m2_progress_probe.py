#!/usr/bin/env python3
"""Observe per-request progress at natural capacity, without a new policy."""

import argparse
import asyncio
import dataclasses
import fcntl
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import aiohttp

from m2_gpu_health import monitor
from run_m2_activation_pacing import command, integrity_errors, source_hashes, validate_info as validate_activation_info
from run_m2_batch_frontier import BASE, ROOT, case_args, dump, ensure_available, hash_file, stop_owned_server, wait_ready
from run_m2_hybrid_factors import fingerprint
from validate_admission_batch_motivation import extract_batch_trace
from validate_dynamic_cost_motivation import extract_trace, flush_cache, get_json, make_group, post_json, run_specs, summarize
from validate_vtc_state_tax_motivation import apply_runtime_policy, extract_vtc_trace


MODES = ("separate-4096", "mixed-4096")
SHAPES = ("short", "medium", "long", "mixed", "state-pressure")


def validate_natural_info(info, mode):
    states = info.get("internal_states", [])
    if not states or any(s.get("max_total_tokens") is not None for s in states):
        raise ValueError("Expected automatic KV capacity, not an explicit token cap")
    # Preserve the raw server response. The old validator expects the configured
    # cap; natural mode must use the allocator's resolved capacity instead.
    adapted = {**info, "internal_states": [
        {**s, "max_total_tokens": s["memory_usage"]["token_capacity"]} for s in states]}
    return validate_activation_info(adapted, mode, "natural", 0)


def specs_for(run_id, shape, rep, pressure_count, warmup=False):
    groups = {
        "inspect": [("short", 2, 512), ("long", 1, 8192)],
        "short": [("short", 256, 512)],
        "medium": [("medium", 64, 2048)],
        "long": [("long", 16, 8192)],
        "mixed": [("short", 128, 512), ("long", 8, 8192)],
        "state-pressure": [("short", pressure_count, 512)],
    }[shape]
    canonical = f"m2progress-{shape}-rep{rep}"
    specs = []
    for group, count, length in groups:
        count = math.ceil(count / 8) if warmup else count
        specs.extend(make_group(canonical, group, count, length, 8 if shape == "inspect" else 128))
    random.Random(20260915 + rep).shuffle(specs)
    return [dataclasses.replace(s, rid=s.rid.replace(canonical, run_id),
                                cache_salt=s.cache_salt.replace(canonical, run_id),
                                arrival_offset_s=index * .001)
            for index, s in enumerate(specs)]


def measurement_hashes():
    hashes = source_hashes()
    for path in (Path(__file__), BASE / "m2_progress_probe_protocol_20260915.md",
                 ROOT / "python/sglang/srt/mem_cache/kv_cache_configurator.py"):
        hashes[str(path)] = hash_file(path)
    return hashes


def progress_errors(item):
    trace = item["scheduler_batch_trace"]
    errors, submitted = [], {}
    if not trace.get("progress_trace_enabled"):
        errors.append("progress trace disabled")
    expected = {r["rid"]: r["prompt_tokens"] for r in item["requests"]}
    for event in trace["events"]:
        members = event.get("members", [])
        if not members or len({m["rid"] for m in members}) != len(members):
            errors.append("missing or duplicate batch membership")
        for member in members:
            rid, work = member["rid"], member["work_tokens"]
            if rid not in expected or not isinstance(work, int) or work <= 0:
                errors.append("invalid progress record")
                continue
            if member["phase"] == "prefill":
                previous = submitted.get(rid, 0)
                if member["prefix_tokens"] != previous:
                    errors.append("prefill gap or repeated work")
                submitted[rid] = previous + work
    if submitted != expected:
        errors.append("prefill work does not match input")
    return sorted(set(errors))


async def replay(args, mode, rep, shape, pressure_count, stage, expected_capacity=None):
    run_id = f"m2progress-{mode}-{shape}-rep{rep}-{stage}"
    specs = specs_for(run_id, shape, rep, pressure_count, warmup=stage == "warmup")
    url = f"http://127.0.0.1:{args.port}"
    policy = case_args(run_id, rep, "mixed", "vtc", 16, 1, 2)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
                                     connector=aiohttp.TCPConnector(limit=0)) as client:
        await apply_runtime_policy(client, url, policy)
        updated = await post_json(client, url + "/set_internal_state", {"server_args": {
            "tenant_slot_gate_code": 1, "tenant_new_activation_limit": 0}})
        if not updated or not all(updated):
            raise RuntimeError("Policy update failed")
        await flush_cache(client, url)
        initial = await get_json(client, url + "/get_server_info")
        capacity = validate_natural_info(initial, mode)
        if expected_capacity is not None and list(capacity) != list(expected_capacity):
            raise RuntimeError(f"Natural capacity differs from frozen inspection: {capacity} vs {expected_capacity}")
        start = time.perf_counter() + .5
        samples, stop = [], asyncio.Event()
        monitoring = asyncio.create_task(monitor(stop, samples))
        try:
            results = await run_specs(client, url, specs, start, run_id)
        finally:
            stop.set()
            await monitoring
        final = await get_json(client, url + "/get_server_info")
        if validate_natural_info(final, mode) != capacity:
            raise RuntimeError("Capacity changed during replay")
    item = {
        "experiment": {"run_id": run_id, "mode": mode, "shape": shape, "rep": rep,
                       "stage": stage, "regime": "natural", "activation_limit": 0},
        "physical_gpu_ids": [6, 7], "payload_fingerprint": fingerprint(specs),
        "experiment_clock_origin_s": start, "resolved_capacity": capacity,
        "server_initial": initial, "server_final": final,
        "scheduler_resource_trace": extract_trace(final, run_id),
        "scheduler_batch_trace": extract_batch_trace(final, run_id),
        "scheduler_vtc_trace": extract_vtc_trace(final, run_id),
        "gpu_health_samples": samples, "summary": summarize(results, 60),
        "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s} for r in results],
    }
    item["integrity_errors"] = integrity_errors(item, len(specs), .05) + progress_errors(item)
    dump(args.output / f"{run_id}.json", item)
    print(json.dumps({"run": run_id, "requests": len(specs), "capacity": capacity,
                      "peak_active": item["scheduler_resource_trace"]["system"]["peak_active_requests"],
                      "errors": item["integrity_errors"]}), flush=True)
    if item["integrity_errors"]:
        raise RuntimeError(item["integrity_errors"])
    return capacity


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "observe"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--capacity-reference", type=Path)
    parser.add_argument("--port", type=int, default=30131)
    args = parser.parse_args()
    reference = None
    if args.stage == "observe":
        if args.capacity_reference is None:
            parser.error("observe requires completed inspection capacity")
        reference = json.loads(args.capacity_reference.read_text())
        done = json.loads((args.capacity_reference.parent / "completed.json").read_text())
        if not reference.get("inspection_passed") or not done.get("completed"):
            parser.error("inspection did not pass")
        if reference["source_hashes"] != done["source_hashes"] or reference["source_hashes"] != measurement_hashes():
            parser.error("measurement source changed since inspection")
    pressure_count = min(1024, math.ceil(1.25 * reference["resolved_capacity"][0])) if reference else 1
    modes = ["mixed-4096"] if args.stage == "inspect" else list(MODES)
    starts = 1 if args.stage == "inspect" else 2
    hashes = measurement_hashes()
    manifest = {"args": {**vars(args), "output": str(args.output),
                          "capacity_reference": str(args.capacity_reference) if args.capacity_reference else None},
                "source_hashes": hashes, "capacity_reference": reference,
                "pressure_count": pressure_count,
                "planned_measured_runs": 1 if args.stage == "inspect" else 20,
                "planned_measured_requests": 3 if args.stage == "inspect" else 1888 + 4 * pressure_count,
                "commands": {mode: command(mode, "natural", args.port) for mode in modes}}
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
        for rep in range(1, starts + 1):
            order = modes.copy()
            random.Random(20260915 + rep).shuffle(order)
            for mode in order:
                ensure_available(args.port)
                if measurement_hashes() != hashes:
                    raise RuntimeError("Measurement source changed")
                with (args.output / f"server-{mode}-rep{rep}.log").open("x") as log:
                    process = subprocess.Popen(command(mode, "natural", args.port), cwd=ROOT, env=env,
                                               stdout=log, stderr=log, start_new_session=True)
                    try:
                        print(f"START {mode} rep={rep} pid={process.pid}", flush=True)
                        wait_ready(process, f"http://127.0.0.1:{args.port}")
                        shapes = ["inspect"] if args.stage == "inspect" else list(SHAPES)
                        if args.stage == "observe":
                            for shape in shapes:
                                asyncio.run(replay(args, mode, -rep, shape, pressure_count, "warmup", reference["resolved_capacity"]))
                        random.Random(9150 + rep).shuffle(shapes)
                        for shape in shapes:
                            capacity = asyncio.run(replay(args, mode, rep, shape, pressure_count, args.stage,
                                                           reference["resolved_capacity"] if reference else None))
                            if args.stage == "inspect":
                                dump(args.output / "capacity.json", {"inspection_passed": True,
                                    "resolved_capacity": capacity, "source_hashes": hashes,
                                    "inspection_artifact": f"m2progress-{mode}-inspect-rep{rep}-inspect.json"})
                    finally:
                        stop_owned_server(process)
        if measurement_hashes() != hashes:
            raise RuntimeError("Measurement source changed during replay")
        dump(args.output / "completed.json", {"completed": True, "source_hashes": hashes})


if __name__ == "__main__":
    main()
