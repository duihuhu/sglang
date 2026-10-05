#!/usr/bin/env python3
"""Run fresh pool controls on an explicitly selected, idle physical GPU pair."""

import argparse
import asyncio
import contextlib
import csv
import dataclasses
import fcntl
import gc
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import aiohttp

from m2_streaming_gpu_health import StreamingGpuMonitor
from run_m2_activation_pacing import command as base_command, integrity_errors, validate_info as base_validate
from run_m2_batch_frontier import BASE, ROOT, case_args, dump, hash_file, wait_ready
from run_m2_hybrid_factors import fingerprint
from run_m2_progress_probe import progress_errors, specs_for
from run_m2_state_mediation import connection_audit, preconnect, measurement_hashes as old_hashes
from validate_admission_batch_motivation import extract_batch_trace
from validate_dynamic_cost_motivation import extract_trace, flush_cache, get_json, post_json, run_specs, summarize, unique_input
from validate_vtc_state_tax_motivation import apply_runtime_policy, extract_vtc_trace


ARMS = ("static-natural", "unified-old-cap", "static-rebalanced", "unified-raised-cap")
MODES = ("separate-4096", "mixed-4096")
COUNT = 607


def gpu_snapshot(gpus):
    output = subprocess.check_output([
        "nvidia-smi", "-i", ",".join(map(str, gpus)),
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits"], text=True)
    return [{"index": int(r[0]), "uuid": r[1].strip(), "name": r[2].strip(),
             "memory_mib": int(r[3]), "total_mib": int(r[4]), "utilization": int(r[5])}
            for r in csv.reader(output.splitlines())]


def ensure_idle(port, gpus):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    snapshots = []
    for _ in range(2):
        state = gpu_snapshot(gpus)
        if {r["index"] for r in state} != set(gpus) or any(r["memory_mib"] > 128 or r["utilization"] for r in state):
            raise RuntimeError(f"Selected GPU pair is not idle: {state}")
        snapshots.append(state)
        time.sleep(1)
    return snapshots


def stop_owned(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    time.sleep(3)


class SelectedGpuMonitor(StreamingGpuMonitor):
    def __init__(self, gpus):
        self.gpus = tuple(gpus)

    async def __aenter__(self):
        self.process = await asyncio.create_subprocess_exec(
            "nvidia-smi", "-i", ",".join(map(str, self.gpus)),
            "--query-gpu=index,utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used",
            "--format=csv,noheader,nounits", "--loop-ms=500",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            self.first_sample = await self.read_sample()
        except BaseException:
            await self.close()
            raise
        return self

    async def read_sample(self):
        rows = []
        keys = ("index", "utilization", "sm_mhz", "temperature_c", "power_w", "memory_mib")
        for _ in self.gpus:
            line = await asyncio.wait_for(self.process.stdout.readline(), timeout=10)
            values = next(csv.reader([line.decode().strip()]))
            if len(values) != len(keys):
                raise RuntimeError("Incomplete GPU health sample")
            rows.append(dict(zip(keys, map(float, values))))
        if {r["index"] for r in rows} != set(self.gpus) or any(not math.isfinite(v) for r in rows for v in r.values()):
            raise RuntimeError("Wrong GPU health devices or invalid values")
        return {"observed_s": time.perf_counter(), "gpus": rows}


def command(mode, arm, port, *, natural_cap=485, kv_tokens=None, audited=False):
    args = base_command(mode, "natural", port)
    if arm != "static-natural":
        state = natural_cap if arm == "unified-old-cap" else COUNT
        args += ["--max-mamba-cache-size", str(state)]
    if arm.startswith("unified"):
        args += ["--enable-unified-memory"]
    if kv_tokens is not None:
        args += ["--max-total-tokens", str(kv_tokens)]
    if audited:
        args = [args[0], str(BASE / "launch_m2_pool_audit_server.py"), *args[3:]]
    return args


def source_hashes():
    result = old_hashes()
    for path in (Path(__file__), BASE / "launch_m2_pool_audit_server.py",
                 BASE / "m2_pool_comparison_live_protocol_20260916.md",
                 ROOT / "python/sglang/srt/mem_cache/unified_memory_pool.py",
                 ROOT / "python/sglang/srt/mem_cache/multi_ended_allocator.py",
                 ROOT / "python/sglang/srt/mem_cache/memory_pool.py"):
        result[str(path)] = hash_file(path)
    return result


def capacity(info, mode, arm, cap=None):
    states = info["internal_states"]
    adapted = {**info, "internal_states": [
        {**s, "max_total_tokens": s["memory_usage"]["token_capacity"]} for s in states]}
    resolved = base_validate(adapted, mode, "natural", 0)
    s = states[0]
    expected = {"enable_unified_memory": arm.startswith("unified"), "mem_fraction_static": .85,
                "max_prefill_tokens": 16384, "speculative_algorithm": None,
                "attention_backend": "triton", "linear_attn_backend": "triton",
                "mamba_backend": "triton", "page_size": 1, "context_length": None,
                "max_running_requests": None}
    for key, value in expected.items():
        if key not in s or s[key] != value:
            raise ValueError(f"{key} changed: {s.get(key)!r} != {value!r}")
    if cap is not None and resolved[3] != cap:
        raise ValueError(f"Runtime cap mismatch: {resolved} vs {cap}")
    if arm in ("static-rebalanced", "unified-raised-cap") and resolved[2] != COUNT:
        raise ValueError(f"The new request limit did not reach {COUNT}: {resolved}")
    return resolved


async def configure(client, url, run_id, rep, cap=None):
    await apply_runtime_policy(client, url, case_args(run_id, rep, "mixed", "vtc", 16, 1, 2))
    settings = {"tenant_slot_gate_code": 1, "tenant_new_activation_limit": 0}
    if cap is not None:
        settings["pp_max_micro_batch_size"] = cap
    result = await post_json(client, url + "/set_internal_state", {"server_args": settings})
    if not result or not all(result):
        raise RuntimeError("Failed to configure runtime policy")
    await flush_cache(client, url)


async def correctness(url, out, arm):
    records = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as client:
        for repeat in (1, 2):
            await flush_cache(client, url)
            for length in (512, 8192):
                payload = {"input_ids": unique_input(length, 821), "return_logprob": True,
                           "top_logprobs_num": 5, "stream": False,
                           "sampling_params": {"temperature": 0, "max_new_tokens": 16, "ignore_eos": True}}
                response = await post_json(client, url + "/generate", payload)
                meta = response["meta_info"]
                token_probs = meta.get("output_token_logprobs", [])
                if meta["completion_tokens"] != 16 or meta["prompt_tokens"] != length or len(token_probs) != 16:
                    raise ValueError(f"Missing output correctness evidence: {meta.keys()}")
                records.append({"repeat": repeat, "input_length": length,
                                "output_ids": [p[1] for p in token_probs], "response": response})
        info = await get_json(client, url + "/get_server_info")
    dump(out / f"correctness-{arm}.json", {"records": records, "info": info})
    return records


async def replay(args, arm, mode, rep, cap, stage, output):
    run_id = f"m2pool-{arm}-{mode}-cap{cap}-rep{rep}-{stage}"
    specs = specs_for(run_id, "state-pressure", -rep if stage == "warmup" else rep, COUNT)
    url = f"http://127.0.0.1:{args.port}"
    trace, connections = connection_audit()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
                                     connector=aiohttp.TCPConnector(limit=0), trace_configs=[trace]) as client:
        await configure(client, url, run_id, rep, cap)
        initial = await get_json(client, url + "/get_server_info")
        resolved = capacity(initial, mode, arm, cap)
        connections["phase"] = "preconnect"
        await preconnect(client, url, COUNT)
        async with SelectedGpuMonitor(args.gpus) as health:
            gc.collect()
            was_enabled = gc.isenabled()
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
                    if was_enabled:
                        gc.enable()
                    connections["phase"] = "after"
        final = await get_json(client, url + "/get_server_info")
        if capacity(final, mode, arm, cap) != resolved:
            raise ValueError("Capacity changed within replay")
    item = {
        "experiment": {"run_id": run_id, "pool": arm, "mode": mode, "rep": rep, "cap": cap,
                       "stage": stage, "activation_limit": 0},
        "physical_gpu_ids": args.gpus, "payload_fingerprint": fingerprint(specs),
        "client_timing": {"gc_disabled_during_replay": True, "http_preconnect_count": COUNT,
                          "health_monitor": "persistent-nvidia-smi-500ms", "connection_events": connections},
        "experiment_clock_origin_s": start, "resolved_capacity": resolved,
        "server_initial": initial, "server_final": final,
        "scheduler_resource_trace": extract_trace(final, run_id),
        "scheduler_batch_trace": extract_batch_trace(final, run_id),
        "scheduler_vtc_trace": extract_vtc_trace(final, run_id),
        "gpu_health_samples": samples, "summary": summarize(results, 60),
        "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s} for r in results],
    }
    errors = integrity_errors(item, COUNT, .05) + progress_errors(item)
    if connections["created"].get("replay", 0) or connections["reused"].get("replay") != COUNT:
        errors.append("HTTP connections were not all reused")
    item["integrity_errors"] = errors
    dump(output / f"{run_id}.json", item)
    print(json.dumps({"run": run_id, "capacity": resolved,
                      "peak_active": item["scheduler_resource_trace"]["system"]["peak_active_requests"],
                      "errors": errors}), flush=True)
    if errors:
        raise ValueError(errors)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("inspect", "measure"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--port", type=int, default=30143)
    parser.add_argument("--frozen-plan", type=Path)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("Select two distinct physical GPU IDs")
    if args.stage == "measure" and args.frozen_plan is None:
        parser.error("Measurements require a budget-matched, frozen plan")
    hashes = source_hashes()
    frozen = json.loads(args.frozen_plan.read_text()) if args.frozen_plan else None
    if frozen and frozen["source_hashes"] != hashes:
        raise ValueError("Frozen plan source hashes changed")
    with contextlib.ExitStack() as stack:
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((BASE / f".m2_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial_gpus = ensure_idle(args.port, args.gpus)
        args.output.mkdir(parents=True, exist_ok=False)
        dump(args.output / "manifest.json", {"stage": args.stage, "source_hashes": hashes,
             "gpus": initial_gpus, "frozen_plan": frozen, "created_unix_s": time.time()})
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        sessions = frozen["sessions"] if frozen else [
            {"arm": arm, "mode": "mixed-4096", "reps": [21]} for arm in ARMS]
        natural_cap, inspected, correctness_reference = 485, {}, None
        for index, session in enumerate(sessions):
            arm, mode = session["arm"], session["mode"]
            ensure_idle(args.port, args.gpus)
            if source_hashes() != hashes:
                raise ValueError("Sources changed during experiment")
            directory = args.output / f"session-{index:02d}-{arm}-{mode}"
            directory.mkdir()
            audit_dir = directory / "memory"
            if args.stage == "inspect":
                audit_dir.mkdir()
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
                   "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1",
                   "SGLANG_TENANT_PROGRESS_TRACE": "1"}
            env.pop("M2_POOL_AUDIT_DIR", None)
            if args.stage == "inspect":
                env["M2_POOL_AUDIT_DIR"] = str(audit_dir)
            cmd = command(mode, arm, args.port, natural_cap=natural_cap,
                          kv_tokens=session.get("kv_tokens"), audited=args.stage == "inspect")
            dump(directory / "session.json", {"config": session, "command": cmd, "gpus": gpu_snapshot(args.gpus)})
            with (directory / "server.log").open("x") as log:
                process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    print(f"START {index} {arm}/{mode} pid={process.pid}", flush=True)
                    wait_ready(process, f"http://127.0.0.1:{args.port}")
                    async def initial_info():
                        async with aiohttp.ClientSession() as client:
                            boot = await get_json(client, f"http://127.0.0.1:{args.port}/get_server_info")
                            boot_cap = boot["internal_states"][0]["effective_max_running_requests_per_dp"]
                            await configure(client, f"http://127.0.0.1:{args.port}", "pool-inspection", 21, boot_cap)
                            return await get_json(client, f"http://127.0.0.1:{args.port}/get_server_info")
                    info = asyncio.run(initial_info())
                    resolved = capacity(info, mode, arm)
                    dump(directory / "initial.json", info)
                    if arm == "static-natural":
                        natural_cap = resolved[2]
                    inspected[f"{index}-{arm}"] = resolved
                    if args.stage == "inspect":
                        records = asyncio.run(correctness(f"http://127.0.0.1:{args.port}", directory, arm))
                        ids = {(r["repeat"], r["input_length"]): r["output_ids"] for r in records}
                        if any(ids[(1, length)] != ids[(2, length)] for length in (512, 8192)):
                            raise ValueError("Single-request self-repeat output mismatch")
                        if correctness_reference is None:
                            correctness_reference = ids
                        elif ids != correctness_reference:
                            raise ValueError("Cross-pool output mismatch; inspect token logprobs before proceeding")
                    caps = session.get("caps", [resolved[2]])
                    for rep in session["reps"]:
                        for cap in caps:
                            asyncio.run(replay(args, arm, mode, rep, cap, "warmup", directory))
                            asyncio.run(replay(args, arm, mode, rep, cap, args.stage, directory))
                finally:
                    stop_owned(process)
            print(f"STOP {index} {arm}/{mode}", flush=True)
        if source_hashes() != hashes:
            raise ValueError("Sources changed at completion")
        dump(args.output / "completed.json", {"completed": True, "source_hashes": hashes,
             "inspected": inspected, "finished_unix_s": time.time(), "gpu_snapshot": gpu_snapshot(args.gpus)})


if __name__ == "__main__":
    main()
