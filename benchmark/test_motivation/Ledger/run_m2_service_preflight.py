"""Six fixed-target conditions, with observer-off/on correctness preflight."""

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

from audit_m2_service_preflight import inspect
from m2_service_workload import fingerprints, workload
import run_m2_unified_admission as base
from run_m2_pool_comparison import capacity, command as pool_command, correctness


MODE, ARM, RAW_BYTES = "mixed-4096", "unified-raised-cap", 19481072000
CASES = [(a, b) for a in ("packed", "split") for b in ("alone", "packed", "split")]


def hashes():
    result = base.pool_hashes()
    for name in ("run_m2_service_preflight.py", "m2_service_observer.py", "m2_service_workload.py",
                 "launch_m2_service_observer.py", "audit_m2_service_preflight.py",
                 "m2_service_preflight_protocol_20260917.md", "test_m2_service_observer.py",
                 "test_audit_m2_service_preflight.py"):
        path = base.BASE / name
        result[str(path)] = base.hash_file(path)
    return result


def validate(info, enabled):
    resolved = capacity(info, MODE, ARM, 607)
    if resolved != (607, 781725, 607, 607):
        raise ValueError(f"Unexpected effective capacity {resolved}")
    trace = info["internal_states"][0].get("m2_service_trace", {})
    if bool(trace.get("enabled")) != enabled:
        raise ValueError("Observer switch mismatch")
    if enabled and trace["shared_pool_bytes"] != RAW_BYTES:
        raise ValueError("Unified pool byte budget mismatch")
    return resolved


async def replay(args, enabled, target, neighbor, stage, directory):
    label = "on" if enabled else "off"
    run_id = f"m2sf-{label}-{target}-{neighbor}-{stage}"
    specs = workload(run_id, target, neighbor, rep=201 if stage == "preflight" else 200,
                     waves=2, interval=1.5, first="A")
    url = f"http://127.0.0.1:{args.port}"
    audit, connections = base.connection_audit()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
            connector=aiohttp.TCPConnector(limit=0, keepalive_timeout=1800), trace_configs=[audit]) as client:
        await base.configure(client, url, run_id, 201, 607)
        initial = await base.get_json(client, url + "/get_server_info")
        resolved = validate(initial, enabled)
        connections["phase"] = "preconnect"
        await base.preconnect(client, url, len(specs))
        async with base.SelectedGpuMonitor(args.gpus) as health:
            gc.collect()
            was_enabled = gc.isenabled()
            gc.disable()
            start = time.perf_counter() + .5
            stop, samples = asyncio.Event(), []
            monitoring = asyncio.create_task(health.monitor(stop, samples))
            connections["phase"] = "replay"
            try:
                results = await base.run_specs(client, url, specs, start, run_id)
            finally:
                stop.set()
                try:
                    await monitoring
                finally:
                    if was_enabled:
                        gc.enable()
                    connections["phase"] = "after"
        final = await base.get_json(client, url + "/get_server_info")
        if validate(final, enabled) != resolved:
            raise ValueError("Capacity changed during replay")
    item = {"experiment": {"run_id": run_id, "observer": label, "target": target, "neighbor": neighbor,
                "stage": stage, "rep": 201, "waves": 2, "interval": 1.5, "first": "A",
                "mode": MODE, "activation_limit": 0}, "physical_gpu_ids": args.gpus,
            "fingerprints": fingerprints(specs), "resolved_capacity": resolved,
            "client_timing": connections, "experiment_clock_origin_s": start,
            "client_clock": {"boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
                             "implementation": time.get_clock_info("perf_counter").implementation},
            "server_initial": initial, "server_final": final,
            "service_trace": final["internal_states"][0].get("m2_service_trace", {}),
            "scheduler_resource_trace": base.extract_trace(final, run_id),
            "scheduler_batch_trace": base.extract_batch_trace(final, run_id),
            "scheduler_vtc_trace": base.extract_vtc_trace(final, run_id),
            "gpu_health_samples": samples, "summary": base.summarize(results, 60),
            "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s, "ttft_s": r.ttft_s} for r in results]}
    errors = base.integrity_errors(item, len(specs), .05) + base.progress_errors(item)
    if connections["created"].get("replay", 0) or connections["reused"].get("replay") != len(specs):
        errors.append("HTTP connection reuse mismatch")
    if enabled:
        item["observation_audit"] = inspect(item)
        errors += item["observation_audit"]["errors"]
    item["integrity_errors"] = sorted(set(errors))
    base.dump(directory / f"{run_id}.json", item)
    print(json.dumps({"run": run_id, "requests": len(specs), "errors": errors,
                      "both_waiting_s": item.get("observation_audit", {}).get("both_waiting_seconds")}), flush=True)
    if errors:
        raise ValueError(errors)
    return item


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, nargs=2, required=True)
    parser.add_argument("--port", type=int, default=30148)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        parser.error("Need two distinct physical GPUs")
    frozen = hashes()
    items, correct = [], {}
    with contextlib.ExitStack() as stack:
        for gpu in sorted(args.gpus):
            lock = stack.enter_context((base.BASE / f".m2_gpu{gpu}.lock").open("a"))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        initial = base.ensure_idle(args.port, args.gpus)
        identities = {(r["index"], r["uuid"]) for r in initial[0]}
        args.output.mkdir(parents=True, exist_ok=False)
        base.dump(args.output / "manifest.json", {"source_hashes": frozen, "initial_gpus": initial,
            "protocol": "functional preflight, not a frozen F1 performance matrix", "cases": CASES,
            "created_unix_s": time.time(), "observer_order": [False, True]})
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        for enabled in (False, True):
            if hashes() != frozen:
                raise ValueError("Frozen sources changed")
            snapshots = base.ensure_idle(args.port, args.gpus)
            if any({(r["index"], r["uuid"]) for r in s} != identities for s in snapshots):
                raise ValueError("GPU identity changed")
            label = "on" if enabled else "off"
            directory = args.output / label
            directory.mkdir()
            cmd = pool_command(MODE, ARM, args.port, kv_tokens=781725)
            cmd = [cmd[0], str(base.BASE / "launch_m2_service_observer.py"), *cmd[3:]]
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ",".join(map(str, args.gpus)),
                   "M2_SERVICE_TRACE": "1" if enabled else "0",
                   "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1", "SGLANG_TENANT_EVENT_TRACE": "1",
                   "SGLANG_TENANT_PROGRESS_TRACE": "1", "SGLANG_TIMEOUT_KEEP_ALIVE": "1800"}
            base.dump(directory / "session.json", {"command": cmd, "observer": enabled})
            with (directory / "server.log").open("x") as log:
                process = subprocess.Popen(cmd, cwd=base.ROOT, env=env, stdout=log, stderr=log, start_new_session=True)
                try:
                    print(f"START service preflight observer={label} pid={process.pid}", flush=True)
                    base.wait_ready(process, f"http://127.0.0.1:{args.port}")
                    correct[label] = asyncio.run(correctness(f"http://127.0.0.1:{args.port}", directory, label))
                    for target, neighbor in CASES:
                        for stage in ("warmup", "preflight"):
                            item = asyncio.run(replay(args, enabled, target, neighbor, stage, directory))
                            if stage == "preflight":
                                items.append(item)
                finally:
                    base.stop_owned(process)
            print(f"STOP service preflight observer={label}", flush=True)
        if hashes() != frozen:
            raise ValueError("Source drift at completion")
        output_ids_equal = all(a["output_ids"] == b["output_ids"] for a, b in zip(correct["off"], correct["on"]))
        repeat_equal = all(rows[0]["output_ids"] == rows[2]["output_ids"] and rows[1]["output_ids"] == rows[3]["output_ids"]
                           for rows in correct.values())
        target_fixed = all(len({i["fingerprints"]["A"]["request_shape_arrivals"] for i in items
                               if i["experiment"]["target"] == shape}) == 1 for shape in ("packed", "split"))
        result = {"completed": True, "source_hashes": frozen, "runs": len(items),
                  "correctness_output_ids_equal": output_ids_equal, "correctness_repeat_equal": repeat_equal,
                  "target_fingerprints_fixed": target_fixed,
                  "scope": "one session per observer setting; no performance significance or hybrid causal claim",
                  "finished_unix_s": time.time(), "gpus": base.gpu_snapshot(args.gpus)}
        base.dump(args.output / "completed.json", result)
        if not all((output_ids_equal, repeat_equal, target_fixed)):
            raise ValueError("Output correctness or target matching gate failed")
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
