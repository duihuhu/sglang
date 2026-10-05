#!/usr/bin/env python3
"""Controlled hybrid activation, pressure, and downstream-admission probes."""

import argparse
import asyncio
import dataclasses
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

import aiohttp

from run_m2_batch_frontier import (
    BASE, ROOT, case_args, dump, ensure_available, hash_file,
    stop_owned_server, wait_ready,
)
from validate_admission_batch_motivation import extract_batch_trace
from validate_dynamic_cost_motivation import (
    extract_trace, flush_cache, get_json, make_group, post_json, run_specs, summarize,
)
from validate_vtc_state_tax_motivation import apply_runtime_policy, extract_vtc_trace


REGIMES = {
    "wide": (128, 131072, 64, 64),
    "cap-control": (128, 131072, 64, 8),
    "state-low": (8, 131072, 8, 8),
    "kv-low": (128, 16384, 64, 64),
}


def command(regime, port):
    state, kv, _, _ = REGIMES[regime]
    return [str(ROOT / ".venv/bin/python"), "-m", "sglang.launch_server",
            "--model-path", "/models/Kimi-Linear-48B-A3B", "--tp-size", "2",
            "--trust-remote-code", "--host", "127.0.0.1", "--port", str(port),
            "--mem-fraction-static", "0.85", "--schedule-policy", "fcfs",
            "--max-mamba-cache-size", str(state), "--max-total-tokens", str(kv),
            "--max-running-requests", "64", "--disable-radix-cache",
            "--cuda-graph-backend-decode", "disabled",
            "--cuda-graph-backend-prefill", "disabled",
            "--chunked-prefill-size", "4096", "--max-prefill-tokens", "16384"]


def cases(regime):
    result = []
    if regime == "wide":
        result += [("E1", f"n{n}", "fcfs") for n in (1, 4, 16, 32)]
    result += [("E2", order, policy) for order in ("short-first", "long-first")
               for policy in ("vtc", "instant")]
    result += [("E3", f"o{o}", "fcfs") for o in (32, 256)]
    return result


def build_specs(run_id, family, variant, rep):
    canonical = f"m2-hybrid-{family}-rep{rep}"
    if family == "E1":
        count = int(variant[1:])
        specs = make_group(canonical, "activation", count, 16384 // count, 128)
    elif family == "E2":
        short_start, long_start = (0, .05) if variant == "short-first" else (.05, 0)
        specs = make_group(canonical, "short", 16, 512, 64, start_offset=short_start)
        specs += make_group(canonical, "long", 4, 8192, 64, start_offset=long_start)
        specs.sort(key=lambda spec: spec.arrival_offset_s)
    elif family == "E3":
        specs = make_group(canonical, "background", 8, 512, int(variant[1:]))
        specs += make_group(canonical, "probe", 1, 512, 16, start_offset=3.0)
    else:
        raise ValueError(family)
    return [dataclasses.replace(s, rid=s.rid.replace(canonical, run_id),
                                cache_salt=s.cache_salt.replace(canonical, run_id))
            for s in specs]


def fingerprint(specs, *, arrivals=True):
    payload = sorted((s.group, s.input_ids, s.output_len,
                      s.arrival_offset_s if arrivals else None) for s in specs)
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def validate_info(info, regime):
    state, kv, effective, cap = REGIMES[regime]
    states = info.get("internal_states", [])
    if not states:
        raise RuntimeError("Missing internal state")
    for s in states:
        for key, value in {
            "max_mamba_cache_size": state, "max_total_tokens": kv,
            "effective_max_running_requests_per_dp": effective,
            "pp_max_micro_batch_size": cap, "disable_radix_cache": True,
            "chunked_prefill_size": 4096, "enable_mixed_chunk": False,
        }.items():
            if s.get(key) != value:
                raise RuntimeError(f"{regime}: {key}={s.get(key)!r}, expected {value!r}")
        config = s.get("cuda_graph_config", {})
        if not isinstance(config, dict) or any(
            config.get(phase, {}).get("backend") != "disabled" for phase in ("decode", "prefill")
        ):
            raise RuntimeError(f"Graphs not disabled: {config}")
        if not s.get("tenant_resource_trace", {}).get("event_trace_enabled"):
            raise RuntimeError("Resource event trace not enabled")


async def run_case(url, directory, regime, family, variant, policy, rep, stage):
    run_id = f"m2h-{regime}-{family}-{variant}-{policy}-rep{rep}"
    path = directory / f"{run_id}.json"
    if path.exists():
        raise FileExistsError(path)
    specs = build_specs(run_id, family, variant, rep)
    args = case_args(run_id, rep, "mixed", policy, 16, 1, 1.5)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=180),
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        await apply_runtime_policy(session, url, args)
        changed = await post_json(session, url + "/set_internal_state", {
            "server_args": {"pp_max_micro_batch_size": REGIMES[regime][3]}})
        if not changed or not all(changed):
            raise RuntimeError(f"Failed runtime cap update: {changed}")
        await flush_cache(session, url)
        initial = await get_json(session, url + "/get_server_info")
        validate_info(initial, regime)
        start = time.perf_counter() + .25
        results = await run_specs(session, url, specs, start, run_id)
        final = await get_json(session, url + "/get_server_info")
    resource = extract_trace(final, run_id)
    batch = extract_batch_trace(final, run_id)
    summary = summarize(results, 60)
    errors = []
    if any(not r.success or r.cached_tokens or r.num_retractions
           or r.prompt_tokens != r.input_len for r in results):
        errors.append("request integrity")
    if not resource or not resource.get("events") or resource.get("dropped_events"):
        errors.append("resource event integrity")
    if not batch or batch.get("dropped_events") or batch.get("dropped_admission_blocks"):
        errors.append("batch/admission event integrity")
    if resource:
        system = resource["system"]
        if system["peak_active_requests"] > REGIMES[regime][3]:
            errors.append("runtime cap exceeded")
        if system["current_active_requests"]:
            errors.append("resources remain after request completion")
        if any(e["state_slots"] != e["active_requests"] for e in resource["events"]):
            errors.append("radix-off one-state-per-request invariant")
    resolved = {"fcfs": "fcfs", "vtc": "tenant-vtc", "instant": "tenant-instant-share"}[policy]
    if any(s.get("schedule_policy") != resolved for s in final["internal_states"]):
        errors.append("policy mismatch")
    dump(path, {
        "experiment": {"run_id": run_id, "family": family, "variant": variant,
                       "policy": policy, "regime": regime, "rep": rep, "stage": stage},
        "payload_fingerprint": fingerprint(specs),
        "content_fingerprint": fingerprint(specs, arrivals=False),
        "experiment_clock_origin_s": start,
        "server_initial": initial, "server_final": final,
        "summary": summary, "scheduler_resource_trace": resource,
        "scheduler_batch_trace": batch, "scheduler_vtc_trace": extract_vtc_trace(final, run_id),
        "integrity_errors": errors,
        "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s}
                     for r in results],
    })
    print(json.dumps({"case": run_id, "stage": stage, "p95": {
        k: round(v["p95_jct_s"], 3) for k, v in summary["groups"].items()},
        "peak": resource["system"]["peak_active_requests"] if resource else None,
        "blocks": len(batch.get("admission_blocks", [])) if batch else None,
        "errors": errors}), flush=True)
    if errors:
        raise RuntimeError(f"{run_id}: {errors}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "confirm"), required=True)
    parser.add_argument("--regimes", nargs="+", choices=("wide", "state-low", "kv-low"),
                        default=["wide", "state-low", "kv-low"])
    parser.add_argument("--port", type=int, default=30127)
    args = parser.parse_args()
    lock = (BASE / ".m2_gpu67.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = ensure_available(args.port)
    args.output.mkdir(parents=True, exist_ok=False)
    sources = [Path(__file__), BASE / "m2_hybrid_factors_protocol_20260910.md",
               BASE / "run_m2_batch_frontier.py", BASE / "validate_dynamic_cost_motivation.py",
               ROOT / "python/sglang/srt/managers/scheduler.py",
               ROOT / "python/sglang/srt/managers/schedule_policy.py",
               ROOT / "python/sglang/srt/managers/scheduler_components/tenant_resource_tracker.py"]
    dump(args.output / "manifest.json", {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {**vars(args), "output": str(args.output)}, "gpu_snapshot": snapshot,
        "source_hashes": {str(p): hash_file(p) for p in sources},
        "commands": {r: command(r, args.port) for r in args.regimes},
    })
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "6,7",
           "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1"}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    url = f"http://127.0.0.1:{args.port}"
    for server_regime in args.regimes:
        ensure_available(args.port)
        with (args.output / f"server-{server_regime}.log").open("x") as log:
            process = subprocess.Popen(command(server_regime, args.port), cwd=ROOT,
                                       env=env, stdout=log, stderr=log, start_new_session=True)
            try:
                print(f"START {server_regime} pid={process.pid}", flush=True)
                wait_ready(process, url)
                regimes = [server_regime]
                if server_regime == "wide" and args.stage == "confirm":
                    regimes += ["cap-control"]
                if args.stage == "smoke":
                    for family, variant, policy in [("E1", "n1", "fcfs"),
                                                     ("E2", "short-first", "vtc"),
                                                     ("E3", "o256", "fcfs")]:
                        asyncio.run(run_case(url, args.output, server_regime, family,
                                             variant, policy, 0, "smoke"))
                    continue
                for regime in regimes:
                    for family, variant, policy in cases(regime):
                        asyncio.run(run_case(url, args.output, regime, family,
                                             variant, policy, 0, "warmup"))
                for rep in range(10, 15):
                    work = [(regime, *case) for regime in regimes for case in cases(regime)
                            if case[0] != "E1" or rep < 13]
                    random.Random(20260910 + rep).shuffle(work)
                    for regime, family, variant, policy in work:
                        asyncio.run(run_case(url, args.output, regime, family,
                                             variant, policy, rep, "confirm"))
            finally:
                stop_owned_server(process)
    dump(args.output / "completed.json", {"completed": True})


if __name__ == "__main__":
    main()
