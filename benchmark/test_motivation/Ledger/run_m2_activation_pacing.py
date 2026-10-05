#!/usr/bin/env python3
"""Screen fresh-activation pacing against corrected native batching baselines."""

import argparse
import asyncio
import dataclasses
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import aiohttp

from m2_gpu_health import health_errors, monitor
from run_m2_batch_frontier import (
    BASE, ROOT, case_args, dump, ensure_available, hash_file, stop_owned_server, wait_ready,
)
from run_m2_hybrid_factors import command as factor_command, fingerprint
from validate_admission_batch_motivation import extract_batch_trace
from validate_dynamic_cost_motivation import (
    extract_trace, flush_cache, get_json, make_group, post_json, run_specs, summarize,
)
from validate_vtc_state_tax_motivation import apply_runtime_policy, extract_vtc_trace


MODES = {"separate-4096": (4096, False), "mixed-4096": (4096, True),
         "mixed-1024": (1024, True), "unchunked": (-1, False)}
REGIMES = ("state8", "wide-cap8", "natural")
ORDERS = ("short-first", "long-first")
LIMITS = (0, 1, 4)


def command(mode, regime, port):
    args = factor_command("state-low" if regime == "state8" else "wide", port)
    args[args.index("--chunked-prefill-size") + 1] = str(MODES[mode][0])
    if MODES[mode][1]:
        args.append("--enable-mixed-chunk")
    if regime == "natural":
        for flag in ("--max-mamba-cache-size", "--max-total-tokens", "--max-running-requests"):
            index = args.index(flag)
            del args[index:index + 2]
    return args


def build_specs(run_id, rep, order, waves, interval, short_count, long_count, output_tokens):
    if (order not in ORDERS or min(waves, short_count, long_count) < 1
            or output_tokens < 2 or interval <= .05 + .001 * max(short_count, long_count)):
        raise ValueError("Invalid workload or overlapping wave-order offsets")
    specs = []
    for wave in range(waves):
        canonical = f"m2p-rep{rep}-wave{wave}"
        for group, count, length in (("short", short_count, 512), ("long", long_count, 8192)):
            later = (group == "long") == (order == "short-first")
            own = make_group(canonical, group, count, length, output_tokens,
                             start_offset=wave * interval + .05 * later)
            specs.extend(dataclasses.replace(
                s, rid=s.rid.replace(canonical, f"{run_id}-w{wave}"),
                cache_salt=s.cache_salt.replace(canonical, f"{run_id}-w{wave}"),
            ) for s in own)
    return sorted(specs, key=lambda s: s.arrival_offset_s)


def validate_info(info, mode, regime, limit):
    states = info.get("internal_states", [])
    if len(states) != 1:
        raise ValueError("Expected one DP scheduler readback")
    capacities = set()
    for s in states:
        expected = {
            "tp_size": 2, "pp_size": 1,
            "tenant_slot_gate_code": 1, "tenant_new_activation_limit": limit,
            "schedule_policy": "tenant-vtc", "tenant_vtc_request_weight": 0,
            "tenant_vtc_input_weight": 1, "tenant_vtc_output_weight": 2,
            "tenant_vtc_input_quadratic_weight": 0,
            "tenant_vtc_input_output_weight": 0, "tenant_vtc_output_quadratic_weight": 0,
            "disable_radix_cache": True, "chunked_prefill_size": MODES[mode][0],
            "enable_mixed_chunk": MODES[mode][1], "enable_prefill_delayer": False,
            "min_free_slots_delay": None, "prefill_max_requests": None,
        }
        if regime != "natural":
            expected.update(max_mamba_cache_size=8 if regime == "state8" else 128,
                            max_total_tokens=131072, pp_max_micro_batch_size=8,
                            effective_max_running_requests_per_dp=8 if regime == "state8" else 64)
        for key, value in expected.items():
            if key not in s or s[key] != value:
                raise ValueError(f"{key}: {s.get(key)!r} != {value!r}")
        if any(s["cuda_graph_config"][p]["backend"] != "disabled" for p in ("prefill", "decode")):
            raise ValueError("Unexpected graph mode")
        if not s.get("tenant_resource_trace", {}).get("event_trace_enabled"):
            raise ValueError("Resource event capture disabled")
        capacity = tuple(s[k] for k in ("max_mamba_cache_size", "max_total_tokens",
                                        "effective_max_running_requests_per_dp", "pp_max_micro_batch_size"))
        if not all(isinstance(v, int) and v > 0 for v in capacity):
            raise ValueError(f"Unresolved pool capacities: {capacity}")
        capacities.add(capacity)
    if len(capacities) != 1:
        raise ValueError("Rank capacity mismatch")
    return capacities.pop()


def integrity_errors(item, expected_count, lateness_limit):
    errors = health_errors(item.get("gpu_health_samples", []), item.get("physical_gpu_ids", (6, 7)))
    requests = item["requests"]
    resource, batch = item.get("scheduler_resource_trace"), item.get("scheduler_batch_trace")
    if len(requests) != expected_count or len({r["rid"] for r in requests}) != expected_count:
        errors.append("request count or duplicate request")
    if any(not r["success"] or r["cached_tokens"] or r["num_retractions"]
           or r["prompt_tokens"] != r["input_len"] or r["completion_tokens"] != r["output_len"]
           or r["ttft_s"] is None or not r["token_events"] for r in requests):
        errors.append("request integrity")
    if any(r["submitted_s"] - r["scheduled_arrival_s"] > lateness_limit for r in requests):
        errors.append("arrival lateness")
    if not resource or resource.get("dropped_events") or not resource.get("events"):
        errors.append("resource trace")
    if not batch or batch.get("dropped_events") or batch.get("dropped_admission_blocks") or not batch.get("events"):
        errors.append("batch trace")
    if not item.get("scheduler_vtc_trace"):
        errors.append("VTC trace")
    if resource:
        cap = item["resolved_capacity"][3]
        system = resource["system"]
        if system["peak_active_requests"] > cap or system["current_active_requests"]:
            errors.append("concurrency or release")
        if any(e["active_requests"] != e["state_slots"] for e in resource["events"]):
            errors.append("one state per active request")
    if batch:
        limit = item["experiment"]["activation_limit"]
        for b in batch.get("admission_blocks", []):
            if b["reason"] == "activation_pacing":
                if not limit or b["new_activation_limit"] != limit or (
                    b["selected_requests"] - b["selected_reusing_request_slots"] != limit
                ):
                    errors.append("pacing count mismatch")
            if (b["reason"] == "request_limit" and b["candidate_rid"] is not None
                    and b["selected_reusing_request_slots"] > 0
                    and b["selected_requests"] < b["runtime_cap"] - b["running_requests"]
                    and b["selected_requests"] - b["selected_reusing_request_slots"] + 1 <= b["request_slots_free"]):
                errors.append("legacy slot defect reappeared")
    return sorted(set(errors))


async def run_case(args, mode, session_index, rep, limit, order, stage):
    run_id = f"m2p-{args.regime}-{mode}-k{limit}-{order}-rep{rep}-{stage}"
    waves = 1 if stage in ("smoke", "warmup") else args.waves
    specs = build_specs(run_id, rep, order, waves, args.interval,
                        args.short_per_wave, args.long_per_wave, args.output_tokens)
    policy = case_args(run_id, rep, "mixed", "vtc", 16, waves, args.interval)
    url = f"http://127.0.0.1:{args.port}"
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200),
                                     connector=aiohttp.TCPConnector(limit=0)) as client:
        await apply_runtime_policy(client, url, policy)
        settings = {"tenant_slot_gate_code": 1, "tenant_new_activation_limit": limit}
        if args.regime != "natural":
            settings["pp_max_micro_batch_size"] = 8
        updated = await post_json(client, url + "/set_internal_state", {"server_args": settings})
        if not updated or not all(updated):
            raise RuntimeError(f"Runtime update failed: {updated}")
        await flush_cache(client, url)
        initial = await get_json(client, url + "/get_server_info")
        capacity = validate_info(initial, mode, args.regime, limit)
        start = time.perf_counter() + .5
        gpu_samples, stop = [], asyncio.Event()
        monitoring = asyncio.create_task(monitor(stop, gpu_samples, args.gpus))
        try:
            results = await run_specs(client, url, specs, start, run_id)
        finally:
            stop.set()
            await monitoring
        final = await get_json(client, url + "/get_server_info")
        if validate_info(final, mode, args.regime, limit) != capacity:
            raise RuntimeError("Capacity changed during replay")
    item = {
        "experiment": {"run_id": run_id, "stage": stage, "mode": mode, "regime": args.regime,
                       "activation_limit": limit, "order": order, "rep": rep,
                       "session": session_index, "waves": waves, "interval_s": args.interval,
                       "short_per_wave": args.short_per_wave, "long_per_wave": args.long_per_wave,
                       "output_tokens": args.output_tokens},
        "payload_fingerprint": fingerprint(specs), "content_fingerprint": fingerprint(specs, arrivals=False),
        "physical_gpu_ids": args.gpus,
        "experiment_clock_origin_s": start, "resolved_capacity": capacity,
        "server_initial": initial, "server_final": final, "summary": summarize(results, 60),
        "scheduler_resource_trace": extract_trace(final, run_id),
        "scheduler_batch_trace": extract_batch_trace(final, run_id),
        "scheduler_vtc_trace": extract_vtc_trace(final, run_id),
        "gpu_health_samples": gpu_samples,
        "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s} for r in results],
    }
    item["integrity_errors"] = integrity_errors(item, len(specs), args.max_lateness)
    dump(args.output / f"{run_id}.json", item)
    print(f"{stage} {run_id}: requests={len(results)} errors={item['integrity_errors']}", flush=True)
    if item["integrity_errors"]:
        raise RuntimeError(item["integrity_errors"])


def source_hashes():
    paths = [Path(__file__), BASE / "m2_activation_pacing_protocol_20260910.md",
             BASE / "m2_gpu_health.py", BASE / "run_m2_batch_frontier.py",
             BASE / "run_m2_hybrid_factors.py", BASE / "validate_dynamic_cost_motivation.py",
             BASE / "validate_vtc_state_tax_motivation.py", BASE / "validate_admission_batch_motivation.py",
             ROOT / "python/sglang/srt/managers/scheduler.py",
             ROOT / "python/sglang/srt/managers/schedule_policy.py",
             ROOT / "python/sglang/srt/managers/schedule_batch.py",
             ROOT / "python/sglang/srt/mem_cache/memory_pool.py",
             ROOT / "python/sglang/srt/managers/scheduler_components/tenant_resource_tracker.py"]
    return {str(p): hash_file(p) for p in paths}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=("smoke", "screen"), required=True)
    parser.add_argument("--regime", choices=REGIMES, default="state8")
    parser.add_argument("--modes", nargs="+", choices=MODES)
    parser.add_argument("--sessions", type=int)
    parser.add_argument("--reps-per-session", type=int, default=1)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--waves", type=int, default=4)
    parser.add_argument("--interval", type=float, default=2)
    parser.add_argument("--short-per-wave", type=int, default=16)
    parser.add_argument("--long-per-wave", type=int, default=4)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--port", type=int, default=30129)
    parser.add_argument("--gpus", nargs=2, type=int, default=[6, 7])
    parser.add_argument("--max-lateness", type=float, default=.05)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.modes = args.modes or (["mixed-1024"] if args.stage == "smoke" else list(MODES))
    args.sessions = args.sessions if args.sessions is not None else (1 if args.stage == "smoke" else 2)
    if min(args.sessions, args.reps_per_session, args.rep_start) < 1 or args.max_lateness <= 0:
        parser.error("sessions, repetitions, seed start and lateness tolerance must be positive")
    if len(args.modes) != len(set(args.modes)):
        parser.error("duplicate modes")
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("two distinct nonnegative physical GPU indices are required")
    build_specs("check", 0, ORDERS[0], args.waves, args.interval,
                args.short_per_wave, args.long_per_wave, args.output_tokens)
    count = args.sessions * len(args.modes) * args.reps_per_session * len(LIMITS) * len(ORDERS)
    waves = 1 if args.stage == "smoke" else args.waves
    manifest = {
        "schema_version": 1, "args": {**vars(args), "output": str(args.output)},
        "planned_measured_runs": count,
        "planned_measured_requests": count * waves * (args.short_per_wave + args.long_per_wave),
        "commands": {mode: command(mode, args.regime, args.port) for mode in args.modes},
        "source_hashes": source_hashes(),
        "claim_scope": "Functional smoke or exploratory screening only; no confirmatory PASS.",
    }
    if args.dry_run:
        print(json.dumps(manifest, indent=2))
        return
    gpu_key = "".join(map(str, sorted(args.gpus)))
    with (BASE / f".m2_gpu{gpu_key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest["gpu_snapshot"] = ensure_available(args.port, args.gpus)
        manifest["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output / "manifest.json", manifest)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
               "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1"}
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"received {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        for session_index in range(args.sessions):
            modes = args.modes.copy()
            random.Random(20260910 + session_index).shuffle(modes)
            for mode in modes:
                ensure_available(args.port, args.gpus)
                if source_hashes() != manifest["source_hashes"]:
                    raise RuntimeError("Measurement sources changed")
                with (args.output / f"server-{mode}-session{session_index}.log").open("x") as log:
                    process = subprocess.Popen(command(mode, args.regime, args.port), cwd=ROOT, env=env,
                                               stdout=log, stderr=log, start_new_session=True)
                    try:
                        print(f"START {mode} session={session_index} pid={process.pid}", flush=True)
                        wait_ready(process, f"http://127.0.0.1:{args.port}")
                        variants = [(k, order) for k in LIMITS for order in ORDERS]
                        if args.stage != "smoke":
                            for k, order in variants:
                                awaitable = run_case(args, mode, session_index, -session_index, k, order, "warmup")
                                asyncio.run(awaitable)
                        for index in range(args.reps_per_session):
                            rep = args.rep_start + session_index * args.reps_per_session + index
                            cases = variants.copy()
                            random.Random(20260910 + rep).shuffle(cases)
                            for k, order in cases:
                                asyncio.run(run_case(args, mode, session_index, rep, k, order, args.stage))
                    finally:
                        stop_owned_server(process, args.gpus)
        ending_hashes = source_hashes()
        if ending_hashes != manifest["source_hashes"]:
            raise RuntimeError("Measurement sources changed during replay")
        dump(args.output / "completed.json", {"completed": True, "source_hashes": ending_hashes})


if __name__ == "__main__":
    main()
