#!/usr/bin/env python3
"""Paired slot-gate interventions with token-VTC fixed across native batching."""

import argparse
import asyncio
import dataclasses
import fcntl
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import aiohttp

from analyze_m2_hybrid_factors import metrics
from m2_gpu_health import health_errors, monitor
from run_m2_batch_frontier import (
    BASE, ROOT, case_args, dump, ensure_available, hash_file, stop_owned_server, wait_ready,
)
from run_m2_hybrid_factors import build_specs, command as factor_command, fingerprint
from validate_admission_batch_motivation import extract_batch_trace
from validate_dynamic_cost_motivation import extract_trace, flush_cache, get_json, post_json, run_specs, summarize
from validate_vtc_state_tax_motivation import apply_runtime_policy, extract_vtc_trace


CONFIGS = {
    "state8-prefill4096": ("state-low", 4096, False),
    "state8-mixed4096": ("state-low", 4096, True),
    "state8-mixed1024": ("state-low", 1024, True),
    "state8-unchunked": ("state-low", -1, False),
    "wide-cap8-prefill4096": ("wide", 4096, False),
}
GATES = {"legacy": 0, "reuse-aware": 1}


def command(config, port):
    regime, chunk, mixed = CONFIGS[config]
    args = factor_command(regime, port)
    args[args.index("--chunked-prefill-size") + 1] = str(chunk)
    if mixed:
        args.append("--enable-mixed-chunk")
    return args


def validate(info, config, gate):
    regime, chunk, mixed = CONFIGS[config]
    states = info.get("internal_states", [])
    if not states:
        raise RuntimeError("Missing server readback")
    for state in states:
        expected = {
            "max_mamba_cache_size": 8 if regime == "state-low" else 128,
            "effective_max_running_requests_per_dp": 8 if regime == "state-low" else 64,
            "max_total_tokens": 131072, "pp_max_micro_batch_size": 8,
            "chunked_prefill_size": chunk, "enable_mixed_chunk": mixed,
            "disable_radix_cache": True, "tenant_slot_gate_code": GATES[gate],
            "schedule_policy": "tenant-vtc", "tenant_vtc_request_weight": 0,
            "tenant_vtc_input_weight": 1, "tenant_vtc_output_weight": 2,
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"{key}: {state.get(key)!r} != {value!r}")
        if any(state["cuda_graph_config"][phase]["backend"] != "disabled"
               for phase in ("prefill", "decode")):
            raise RuntimeError("Graph confound")


async def run_case(url, output, config, gate, order, rep, stage):
    run_id = f"m2s-{config}-{gate}-{order}-rep{rep}"
    specs = build_specs(run_id, "E2", order, rep)
    args = case_args(run_id, rep, "mixed", "vtc", 16, 1, 1.5)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=240),
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        await apply_runtime_policy(session, url, args)
        updated = await post_json(session, url + "/set_internal_state", {"server_args": {
            "pp_max_micro_batch_size": 8, "tenant_slot_gate_code": GATES[gate]}})
        if not updated or not all(updated):
            raise RuntimeError(f"Gate update failed: {updated}")
        await flush_cache(session, url)
        initial = await get_json(session, url + "/get_server_info")
        validate(initial, config, gate)
        start = time.perf_counter() + .25
        gpu_samples, stop_monitor = [], asyncio.Event()
        monitoring = asyncio.create_task(monitor(stop_monitor, gpu_samples))
        try:
            results = await run_specs(session, url, specs, start, run_id)
        finally:
            stop_monitor.set()
            await monitoring
        final = await get_json(session, url + "/get_server_info")
        validate(final, config, gate)
    resource, batch = extract_trace(final, run_id), extract_batch_trace(final, run_id)
    errors = health_errors(gpu_samples)
    if any(not r.success or r.cached_tokens or r.num_retractions or r.prompt_tokens != r.input_len
           or r.completion_tokens != r.output_len for r in results):
        errors.append("request integrity")
    if not resource or resource.get("dropped_events") or not resource.get("events"):
        errors.append("resource trace")
    if not batch or batch.get("dropped_events") or batch.get("dropped_admission_blocks"):
        errors.append("batch trace")
    if resource:
        if resource["system"]["peak_active_requests"] > 8 or resource["system"]["current_active_requests"]:
            errors.append("concurrency or release")
        if any(e["active_requests"] != e["state_slots"] for e in resource["events"]):
            errors.append("one state per active request")
    item = {
        "experiment": {"run_id": run_id, "family": "E2", "variant": order, "rep": rep,
                       "stage": stage, "config": config, "gate": gate, "policy": "vtc"},
        "payload_fingerprint": fingerprint(specs), "content_fingerprint": fingerprint(specs, arrivals=False),
        "experiment_clock_origin_s": start, "server_initial": initial, "server_final": final,
        "summary": summarize(results, 60), "scheduler_resource_trace": resource,
        "scheduler_batch_trace": batch, "scheduler_vtc_trace": extract_vtc_trace(final, run_id),
        "integrity_errors": errors,
        "gpu_health_samples": gpu_samples,
        "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s} for r in results],
    }
    dump(output / f"{run_id}.json", item)
    if errors:
        raise RuntimeError(f"{run_id}: {errors}")
    observed = metrics(item)
    print(f"{stage} {run_id}: short={observed['short_p95']:.3f} "
          f"long={observed['long_p95']:.3f} makespan={observed['makespan']:.3f} "
          f"peak={observed['peak_state']} errors=0", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "confirm"), required=True)
    parser.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument("--port", type=int, default=30128)
    args = parser.parse_args()
    lock = (BASE / ".m2_gpu67.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = ensure_available(args.port)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), BASE / "m2_slot_batching_protocol_20260910.md",
               BASE / "m2_gpu_health.py",
               BASE / "run_m2_hybrid_factors.py", BASE / "run_m2_batch_frontier.py",
               BASE / "analyze_m2_hybrid_factors.py", BASE / "validate_dynamic_cost_motivation.py",
               BASE / "validate_vtc_state_tax_motivation.py",
               ROOT / "python/sglang/srt/managers/scheduler.py",
               ROOT / "python/sglang/srt/managers/schedule_policy.py",
               ROOT / "python/sglang/srt/mem_cache/memory_pool.py",
               ROOT / "python/sglang/srt/managers/scheduler_components/tenant_resource_tracker.py"]
    dump(args.output / "manifest.json", {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {**vars(args), "output": str(args.output)}, "gpu_snapshot": snapshot,
        "source_hashes": {str(p): hash_file(p) for p in sources},
        "commands": {c: command(c, args.port) for c in args.configs},
    })
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "6,7",
           "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1"}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    url = f"http://127.0.0.1:{args.port}"
    variants = [(gate, order) for gate in GATES for order in ("short-first", "long-first")]
    for config in args.configs:
        ensure_available(args.port)
        with (args.output / f"server-{config}.log").open("x") as log:
            process = subprocess.Popen(command(config, args.port), cwd=ROOT, env=env,
                                       stdout=log, stderr=log, start_new_session=True)
            try:
                print(f"START {config} pid={process.pid}", flush=True)
                wait_ready(process, url)
                for gate, order in variants:
                    asyncio.run(run_case(url, args.output, config, gate, order, 0,
                                         "smoke" if args.stage == "smoke" else "warmup"))
                if args.stage == "confirm":
                    for rep in range(20, 25):
                        shuffled = variants.copy()
                        random.Random(20260911 + rep).shuffle(shuffled)
                        for gate, order in shuffled:
                            asyncio.run(run_case(url, args.output, config, gate, order, rep, "confirm"))
            finally:
                stop_owned_server(process)
    dump(args.output / "completed.json", {"completed": True})


if __name__ == "__main__":
    main()
