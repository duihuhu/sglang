"""Calibrate and screen admission decisions above a fixed unified pool."""

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

from run_m2_batch_frontier import BASE, ROOT, dump, hash_file, wait_ready
from run_m2_pool_comparison import (
    SelectedGpuMonitor, capacity, command as pool_command, configure,
    ensure_idle, gpu_snapshot, source_hashes as pool_hashes, stop_owned,
)
from run_m2_state_mediation import connection_audit, preconnect
from run_m2_activation_pacing import integrity_errors
from run_m2_progress_probe import progress_errors
from run_m2_hybrid_factors import fingerprint
from analyze_m2_hybrid_factors import lifecycle
from validate_dynamic_cost_motivation import (
    RequestSpec, extract_trace, get_json, run_specs, summarize, unique_input,
)
from validate_admission_batch_motivation import extract_batch_trace
from validate_vtc_state_tax_motivation import extract_vtc_trace


PROFILES = {"state": (512, 512), "kv": (96, 16384)}
POLICIES = ("native", "decode32", "slots64", "delayer32")
STATE_BYTES, KV_BYTES = 21708800, 8064
MODE, ARM, KV_CAP = "separate-4096", "unified-raised-cap", 781725


def hashes():
    result = pool_hashes()
    for path in (Path(__file__), BASE / "launch_m2_unified_admission.py",
                 BASE / "m2_unified_admission_protocol_20260916.md",
                 ROOT / "python/sglang/srt/managers/prefill_delayer.py",
                 ROOT / "python/sglang/srt/managers/min_free_slots_delayer.py"):
        result[str(path)] = hash_file(path)
    return result


def command(policy, port):
    if policy not in POLICIES:
        raise ValueError("Unknown policy")
    cmd = pool_command(MODE, ARM, port, kv_tokens=KV_CAP)
    cmd = [cmd[0], str(BASE / "launch_m2_unified_admission.py"), *cmd[3:]]
    if policy == "slots64":
        cmd += ["--min-free-slots-delay", "64"]
    if policy == "delayer32":
        cmd += ["--enable-prefill-delayer", "--prefill-delayer-max-delay-passes", "32"]
    return cmd


def specs_for(run_id, profile, rep, arrival=None):
    count, length = PROFILES[profile]
    specs = [RequestSpec(
        rid=f"{run_id}-holder-{i}", group="holder", input_ids=unique_input(length, 21000+rep*607+i),
        output_len=128, arrival_offset_s=i*.001, cache_salt=f"{run_id}-h-{i}", tenant_key="shared",
    ) for i in range(count)]
    if arrival is not None:
        specs += [RequestSpec(
            rid=f"{run_id}-candidate-{i}", group="candidate", input_ids=unique_input(8192, 81000+rep*64+i),
            output_len=64, arrival_offset_s=arrival+i*.001, cache_salt=f"{run_id}-c-{i}", tenant_key="shared",
        ) for i in range(64)]
    return specs


def validate(info, policy):
    # Reuse the frozen pool checks, adapting only the intentional native-control flags.
    adapted = json.loads(json.dumps(info))
    for state in adapted["internal_states"]:
        expected = {"min_free_slots_delay": 64 if policy == "slots64" else None,
                    "enable_prefill_delayer": policy == "delayer32"}
        for name, value in expected.items():
            if state.get(name) != value:
                raise ValueError(f"Native flag mismatch: {name}={state.get(name)}")
        if policy == "delayer32" and state.get("prefill_delayer_max_delay_passes") != 32:
            raise ValueError("Wrong native delay limit")
        audit = state["m2_admission"]
        if audit["diagnostic_limit"] != (32 if policy == "decode32" else 0):
            raise ValueError("Wrong diagnostic limit")
        if audit["shared_pool_bytes"] <= 0:
            raise ValueError("Missing actual shared-pool budget")
        state["min_free_slots_delay"] = None
        state["enable_prefill_delayer"] = False
    resolved = capacity(adapted, MODE, ARM, 607)
    if resolved[1] != KV_CAP:
        raise ValueError("KV bootstrap changed")
    return resolved


def arrival_snapshot(item, at):
    spans = lifecycle(item)
    holders = [r for r in item["requests"] if r["group"] == "holder"]
    alive = [r for r in holders if spans[r["rid"]][0] <= at < spans[r["rid"]][1]]
    progress = {r["rid"]: max((n for t, n in r["token_events"] if t <= at), default=0) for r in holders}
    remaining = [r["output_len"] - progress[r["rid"]] for r in alive]
    trace = item["scheduler_resource_trace"]
    offset = trace["event_clock_origin_s"] - item["experiment_clock_origin_s"]
    events = [e for e in trace["events"] if offset + e["observed_s"] <= at]
    event = events[-1] if events else {"state_slots": 0, "kv_tokens": 0}
    state, kv = event["state_slots"]*STATE_BYTES, event["kv_tokens"]*KV_BYTES
    return {"arrival_s": at, "holders": len(holders), "alive": len(alive),
            "prefilled": sum(n > 0 for n in progress.values()),
            "median_remaining": statistics.median(remaining) if remaining else None,
            "state_live_bytes_proxy": state, "kv_live_bytes_proxy": kv,
            "state_fraction": state/(state+kv) if state+kv else None,
            "shared_pool_bytes": item["server_initial"]["internal_states"][0]["m2_admission"]["shared_pool_bytes"]}


def scene_errors(snapshot, profile, phase):
    errors = []
    if snapshot["alive"] < .8*snapshot["holders"]:
        errors.append("fewer than 80% holders alive")
    if snapshot["prefilled"] < .95*snapshot["holders"]:
        errors.append("fewer than 95% holders prefilled")
    remaining = snapshot["median_remaining"]
    if remaining is None or (phase == "far" and remaining < 80) or (phase == "near" and not 16 <= remaining <= 48):
        errors.append("wrong remaining-work phase")
    fraction = snapshot["state_fraction"]
    if fraction is None or (profile == "state" and fraction < 2/3) or (profile == "kv" and fraction > 1/3):
        errors.append("wrong live-byte composition")
    return errors


async def replay(args, policy, profile, phase, rep, stage, arrival, directory):
    run_id = f"m2ua-{policy}-{profile}-{phase}-rep{rep}-{stage}"
    specs = specs_for(run_id, profile, -rep if stage == "warmup" else rep, arrival)
    url = f"http://127.0.0.1:{args.port}"
    audit, connections = connection_audit()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=1800), trace_configs=[audit]) as client:
        await configure(client, url, run_id, rep, 607)
        initial = await get_json(client, url+"/get_server_info")
        resolved = validate(initial, policy)
        connections["phase"] = "preconnect"
        await preconnect(client, url, len(specs))
        async with SelectedGpuMonitor(args.gpus) as health:
            gc.collect()
            enabled = gc.isenabled()
            gc.disable()
            start = time.perf_counter()+.5
            stop, samples = asyncio.Event(), []
            monitor = asyncio.create_task(health.monitor(stop, samples))
            connections["phase"] = "replay"
            try:
                results = await run_specs(client, url, specs, start, run_id)
            finally:
                stop.set()
                try:
                    await monitor
                finally:
                    if enabled:
                        gc.enable()
                    connections["phase"] = "after"
        final = await get_json(client, url+"/get_server_info")
        if validate(final, policy) != resolved:
            raise ValueError("Capacity changed within run")
    item = {"experiment": {"run_id": run_id, "policy": policy, "profile": profile, "phase": phase,
                "rep": rep, "stage": stage, "arrival": arrival, "activation_limit": 0},
            "physical_gpu_ids": args.gpus, "resolved_capacity": resolved,
            "payload_fingerprint": fingerprint(specs), "client_timing": connections,
            "experiment_clock_origin_s": start, "server_initial": initial, "server_final": final,
            "scheduler_resource_trace": extract_trace(final, run_id),
            "scheduler_batch_trace": extract_batch_trace(final, run_id),
            "scheduler_vtc_trace": extract_vtc_trace(final, run_id), "gpu_health_samples": samples,
            "summary": summarize(results, 60),
            "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s} for r in results]}
    errors = integrity_errors(item, len(specs), .05)+progress_errors(item)
    if connections["created"].get("replay", 0) or connections["reused"].get("replay") != len(specs):
        errors.append("HTTP replay connection mismatch")
    item["integrity_errors"] = errors
    if arrival is not None:
        item["arrival_snapshot"] = arrival_snapshot(item, arrival)
        item["scene_errors"] = scene_errors(item["arrival_snapshot"], profile, phase)
    dump(directory/f"{run_id}.json", item)
    print(json.dumps({"run": run_id, "errors": errors, "scene_errors": item.get("scene_errors"),
                      "peak_active": item["scheduler_resource_trace"]["system"]["peak_active_requests"]}), flush=True)
    if errors:
        raise ValueError(errors)
    return item


def freeze_arrivals(items):
    arrivals, evidence = {}, []
    for profile in PROFILES:
        own = [i for i in items if i["experiment"]["profile"] == profile]
        if len(own) != 2:
            raise ValueError("Need exactly two complete calibration runs per profile")
        for phase, position in (("far", 16), ("near", 96)):
            anchors = [statistics.median(next(t for t, n in r["token_events"] if n >= position)
                                        for r in i["requests"]) for i in own]
            at = statistics.fmean(anchors)
            arrivals[f"{profile}-{phase}"] = at
            for item in own:
                snapshot = arrival_snapshot(item, at)
                errors = scene_errors(snapshot, profile, phase)
                evidence.append({"profile": profile, "phase": phase, "rep": item["experiment"]["rep"],
                                 "snapshot": snapshot, "errors": errors})
    return {"arrivals": arrivals, "evidence": evidence,
            "scenes_valid": not any(e["errors"] for e in evidence)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("calibrate", "screen", "native-controls"), required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--port", type=int, default=30144)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("Need two distinct physical GPUs")
    plan = json.loads(args.plan.read_text()) if args.plan else None
    expected_hashes = hashes()
    if args.stage != "calibrate" and (not plan or not plan["scenes_valid"] or plan["source_hashes"] != expected_hashes):
        parser.error("Screening needs a valid, unchanged calibration plan")
    if plan and plan.get("physical_gpu_ids") != args.gpus:
        parser.error("Use the same physical GPUs as the scene calibration")
    policies = {"calibrate": ["native"], "screen": ["native", "decode32"],
                "native-controls": ["slots64", "delayer32"]}[args.stage]
    reps = (41, 42) if args.stage == "calibrate" else (51, 52)
    items = []
    with contextlib.ExitStack() as stack:
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((BASE/f".m2_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ensure_idle(args.port, args.gpus)
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output/"manifest.json", {"source_hashes": expected_hashes, "stage": args.stage,
             "plan": plan, "gpus": gpu_snapshot(args.gpus), "created_unix_s": time.time()})
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        session = 0
        for ri, rep in enumerate(reps):
            for policy in policies if ri == 0 else reversed(policies):
                ensure_idle(args.port, args.gpus)
                if hashes() != expected_hashes:
                    raise ValueError("Source changed during experiment")
                directory = args.output/f"session-{session:02d}-{policy}-rep{rep}"
                directory.mkdir()
                env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
                       "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1",
                       "SGLANG_TENANT_PROGRESS_TRACE": "1",
                       "SGLANG_TIMEOUT_KEEP_ALIVE": "1800",
                       "M2_ADMISSION_DECODE_PASSES": "32" if policy == "decode32" else "0"}
                cmd = command(policy, args.port)
                dump(directory/"session.json", {"command": cmd, "policy": policy, "rep": rep,
                     "http_keepalive_seconds": 1800, "diagnostic_passes": int(env["M2_ADMISSION_DECODE_PASSES"])})
                with (directory/"server.log").open("x") as log:
                    process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                    try:
                        print(f"START {session} {policy} rep{rep} pid={process.pid}", flush=True)
                        wait_ready(process, f"http://127.0.0.1:{args.port}")
                        cases = [(p, "background") for p in PROFILES] if args.stage == "calibrate" else [
                            (p, phase) for p in PROFILES for phase in ("far", "near")]
                        for profile, phase in cases if ri == 0 else reversed(cases):
                            arrival = None if args.stage == "calibrate" else plan["arrivals"][f"{profile}-{phase}"]
                            awaitable = replay(args, policy, profile, phase, rep, "warmup", arrival, directory)
                            asyncio.run(awaitable)
                            items.append(asyncio.run(replay(args, policy, profile, phase, rep, args.stage, arrival, directory)))
                    finally:
                        stop_owned(process)
                print(f"STOP {session} {policy}", flush=True)
                session += 1
        if hashes() != expected_hashes:
            raise ValueError("Source changed at completion")
        if args.stage == "calibrate":
            frozen = {**freeze_arrivals(items), "source_hashes": expected_hashes,
                      "parent_manifest": str(args.output/"manifest.json"),
                      "physical_gpu_ids": args.gpus,
                      "formal_reps": [51, 52], "primary": "mean JCT; p95/makespan regression <=5%"}
            dump(args.output/"screen-plan.json", frozen)
            print(json.dumps({"scenes_valid": frozen["scenes_valid"], "arrivals": frozen["arrivals"]}), flush=True)
        dump(args.output/"completed.json", {"completed": True, "source_hashes": expected_hashes,
             "finished_unix_s": time.time(), "runs": len(items), "gpus": gpu_snapshot(args.gpus)})


if __name__ == "__main__":
    main()
