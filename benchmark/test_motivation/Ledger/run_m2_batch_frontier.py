#!/usr/bin/env python3
"""Run isolated M2 batch controls using existing SGLang experiment APIs."""

from __future__ import annotations

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
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import aiohttp

from validate_admission_batch_motivation import extract_batch_trace
from validate_dynamic_cost_motivation import (
    extract_trace, flush_cache, get_json, post_json, run_specs, summarize,
)
from validate_vtc_state_tax_motivation import (
    apply_runtime_policy, build_partition, extract_vtc_trace,
)


MODES = {"separate-4096": (False, 4096), "mixed-4096": (True, 4096),
         "mixed-1024": (True, 1024)}
ROOT = Path(__file__).resolve().parents[3]
BASE = Path(__file__).resolve().parent


def dump(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")


def hash_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def case_args(run_id: str, rep: int, composition: str, policy: str,
              factor: int, waves: int, interval: float) -> argparse.Namespace:
    return argparse.Namespace(
        run_id=run_id, composition=composition, runtime_policy=policy,
        request_weight=0, input_weight=1, input_quadratic_weight=0,
        output_weight=2, input_output_weight=0, output_quadratic_weight=0,
        packed_prompt=16384, packed_output=512, split_factor=factor, waves=waves,
        wave_interval_s=interval, arrival_gap_ms=1,
        wave_order="packed-first" if rep % 2 == 0 else "split-first",
    )


def paired_specs(args, rep: int):
    # Input contents depend on repetition/workload, never policy or server mode.
    canonical = argparse.Namespace(**vars(args))
    canonical.run_id = f"m2-paired-f{args.split_factor}-rep{rep}"
    canonical.composition = "mixed"
    specs = build_partition(canonical)
    if args.composition != "mixed":
        group = args.composition.removesuffix("-only")
        specs = [spec for spec in specs if spec.group == group]
    return [dataclasses.replace(
        spec, rid=spec.rid.replace(canonical.run_id, args.run_id),
        cache_salt=spec.cache_salt.replace(canonical.run_id, args.run_id),
    ) for spec in specs]


def payload_fingerprint(specs) -> str:
    records = [(s.group, s.input_ids, s.output_len, s.arrival_offset_s) for s in specs]
    return hashlib.sha256(json.dumps(records, separators=(",", ":")).encode()).hexdigest()


def launch_command(mode: str, port: int) -> list[str]:
    mixed, chunk = MODES[mode]
    command = [str(ROOT / ".venv/bin/python"), "-m", "sglang.launch_server",
               "--model-path", "/models/Kimi-Linear-48B-A3B",
               "--tp-size", "2", "--trust-remote-code", "--host", "127.0.0.1",
               "--port", str(port), "--mem-fraction-static", "0.85",
               "--schedule-policy", "fcfs", "--max-mamba-cache-size", "480",
               "--max-total-tokens", "196608", "--max-running-requests", "96",
               "--cuda-graph-max-bs-decode", "96", "--disable-prefill-cuda-graph",
               "--chunked-prefill-size", str(chunk), "--max-prefill-tokens", "16384"]
    if mixed:
        command.append("--enable-mixed-chunk")
    return command


def warmup_cases(stage: str, caps: list[int], waves: int):
    if stage != "confirm":
        return [(96, "mixed", "vtc", 1)]
    # Exercise isolated and mixed batch shapes before any confirmation sample.
    return [(cap, composition, policy, waves) for cap in caps
            for composition, policy in [("split-only", "fcfs"),
                                        ("packed-only", "fcfs"), ("mixed", "vtc")]]


def gpu_snapshot(gpu_ids=(6, 7)) -> str:
    return subprocess.check_output([
        "nvidia-smi", "-i", ",".join(map(str, gpu_ids)),
        "--query-gpu=index,uuid,name,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)


def ensure_available(port: int, gpu_ids=(6, 7)) -> str:
    with socket.socket() as sock:
        # Permit a recently closed listener's TIME_WAIT, not a live listener.
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    snapshot = gpu_snapshot(gpu_ids)
    for line in snapshot.strip().splitlines():
        if int(line.split(",")[3].strip()) > 1000:
            raise RuntimeError(f"GPU is not idle; will not displace another job: {line}")
    return snapshot


def validate_readback(info, mode: str, cap: int) -> None:
    states = info.get("internal_states", [])
    if not states:
        raise RuntimeError("Missing server internal state")
    for state in states:
        expected = {
            "max_mamba_cache_size": 480, "max_total_tokens": 196608,
            "effective_max_running_requests_per_dp": 96,
            "pp_max_micro_batch_size": cap, "chunked_prefill_size": MODES[mode][1],
            "enable_mixed_chunk": MODES[mode][0],
        }
        for key, value in expected.items():
            if state.get(key) != value:
                raise RuntimeError(f"Readback mismatch {key}: {state.get(key)!r} != {value!r}")


async def run_case(url: str, directory: Path, mode: str, cap: int, rep: int,
                   composition: str, policy: str, factor: int, waves: int,
                   interval: float, stage: str) -> None:
    run_id = f"m2-{mode}-n{cap}-f{factor}-{composition}-{policy}-rep{rep}"
    output = directory / f"{run_id}.json"
    if output.exists():
        raise FileExistsError(output)
    args = case_args(run_id, rep, composition, policy, factor, waves, interval)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=300),
                                     connector=aiohttp.TCPConnector(limit=0)) as session:
        await apply_runtime_policy(session, url, args)
        changed = await post_json(session, f"{url}/set_internal_state", {
            "server_args": {"pp_max_micro_batch_size": cap}})
        if not changed or not all(changed):
            raise RuntimeError(f"Failed to change admission cap: {changed}")
        await flush_cache(session, url)
        initial = await get_json(session, f"{url}/get_server_info")
        validate_readback(initial, mode, cap)
        specs = paired_specs(args, rep)
        results = await run_specs(session, url, specs, time.perf_counter() + 0.25, run_id)
        final = await get_json(session, f"{url}/get_server_info")
    resource = extract_trace(final, run_id)
    batch = extract_batch_trace(final, run_id)
    vtc = extract_vtc_trace(final, run_id)
    summary = summarize(results, 60)
    errors = []
    if any(not r.success or r.cached_tokens or r.num_retractions or
           r.prompt_tokens != r.input_len for r in results):
        errors.append("request integrity failure")
    if not resource or not batch or batch.get("dropped_events"):
        errors.append("resource/batch trace missing or dropped")
    if resource and resource["system"]["peak_active_requests"] > cap:
        errors.append("observed active concurrency exceeds requested cap")
    if policy == "vtc" and not vtc:
        errors.append("VTC trace missing")
    resolved = final.get("internal_states", [])
    expected_policy = {"fcfs": "fcfs", "vtc": "tenant-vtc",
                       "instant": "tenant-instant-share"}[policy]
    if any(s.get("schedule_policy") != expected_policy for s in resolved):
        errors.append("resolved policy mismatch")
    if resource:
        for group, usage in resource["tenants"].items():
            summary["groups"][group]["scheduler_active_resources"] = usage
    artifact = {
        "schema_version": 1,
        "experiment": {**vars(args), "stage": stage, "batch_mode": mode,
                       "admission_cap": cap, "repetition": rep},
        "payload_fingerprint": payload_fingerprint(specs),
        "server_initial": initial, "server_final": final,
        "summary": summary, "scheduler_resource_trace": resource,
        "scheduler_batch_trace": batch, "scheduler_vtc_trace": vtc,
        "integrity_errors": errors,
        "requests": [{**dataclasses.asdict(r), "latency_s": r.latency_s,
                      "ttft_s": r.ttft_s} for r in results],
    }
    dump(output, artifact)
    compact = {group: round(value["p95_jct_s"], 3)
               for group, value in summary["groups"].items()}
    print(json.dumps({"case": run_id, "p95": compact,
                      "makespan": round(max(r.finished_s for r in results), 3),
                      "integrity_errors": errors}), flush=True)
    if errors:
        raise RuntimeError(f"{run_id}: {errors}")


def stop_owned_server(process: subprocess.Popen, gpu_ids=(6, 7)) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=45)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=15)
    for _ in range(30):
        if all(int(line.split(",")[3]) <= 1000
               for line in gpu_snapshot(gpu_ids).strip().splitlines()):
            return
        time.sleep(1)
    raise RuntimeError("GPU memory did not drain after owned server stopped")


def wait_ready(process: subprocess.Popen, url: str) -> None:
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited {process.returncode}; inspect server log")
        try:
            with urllib.request.urlopen(f"{url}/health", timeout=2) as response:
                if response.status == 200:
                    return
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            pass
        time.sleep(1)
    raise TimeoutError("Server startup exceeded 600 seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--stage", choices=["smoke", "screen", "confirm"], required=True)
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--caps", nargs="+", type=int, default=[9, 24, 48, 96])
    parser.add_argument("--policies", nargs="+", choices=["vtc", "instant"], default=["vtc", "instant"])
    parser.add_argument("--repetitions", type=int, default=1)
    parser.add_argument("--rep-start", type=int, default=1)
    parser.add_argument("--factor", type=int, default=16)
    parser.add_argument("--waves", type=int, default=8)
    parser.add_argument("--interval", type=float, default=1.5)
    parser.add_argument("--port", type=int, default=30126)
    args = parser.parse_args()
    if any(cap < 1 or cap > 96 for cap in args.caps) or args.repetitions < 1:
        parser.error("caps must be in [1,96] and repetitions positive")
    if args.factor < 1 or 512 % args.factor:
        parser.error("factor must be a positive divisor of 512")
    args.output.mkdir(parents=True, exist_ok=False)
    lock = (BASE / ".m2_gpu67.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    snapshot = ensure_available(args.port)
    sources = [Path(__file__), BASE / "m2_batch_frontier_protocol_20260909.md",
               BASE / "validate_dynamic_cost_motivation.py",
               BASE / "validate_vtc_state_tax_motivation.py",
               BASE / "validate_admission_batch_motivation.py",
               ROOT / "python/sglang/srt/managers/scheduler.py",
               ROOT / "python/sglang/srt/managers/schedule_policy.py",
               ROOT / "python/sglang/srt/server_args.py"]
    dump(args.output / "manifest.json", {
        "args": {**vars(args), "output": str(args.output)},
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "gpu_snapshot": snapshot, "source_hashes": {str(p): hash_file(p) for p in sources},
        "launch_commands": {mode: launch_command(mode, args.port) for mode in args.modes},
    })
    url = f"http://127.0.0.1:{args.port}"
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "6,7",
                   "SGLANG_ENABLE_TENANT_RESOURCE_TRACKING": "1"}
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"received signal {signum}")
    signal.signal(signal.SIGTERM, interrupted)
    for mode in args.modes:
        ensure_available(args.port)
        with (args.output / f"server-{mode}.log").open("x") as log:
            process = subprocess.Popen(launch_command(mode, args.port), cwd=ROOT,
                                       env=environment, stdout=log, stderr=log,
                                       start_new_session=True)
            try:
                print(f"START {mode} pid={process.pid}", flush=True)
                wait_ready(process, url)
                for cap, composition, policy, waves in warmup_cases(
                        args.stage, args.caps, args.waves):
                    asyncio.run(run_case(url, args.output, mode, cap, 0,
                                         composition, policy, args.factor, waves,
                                         args.interval, "warmup"))
                for rep in range(args.rep_start, args.rep_start + args.repetitions):
                    caps = list(args.caps)
                    random.Random(20260909 + rep).shuffle(caps)
                    for cap in caps:
                        cases = [("split-only", "fcfs"), ("packed-only", "fcfs")]
                        cases += [("mixed", policy) for policy in args.policies]
                        random.Random(rep * 100 + cap).shuffle(cases)
                        for composition, policy in cases:
                            asyncio.run(run_case(url, args.output, mode, cap, rep,
                                                 composition, policy, args.factor,
                                                 args.waves, args.interval, args.stage))
            finally:
                stop_owned_server(process)
    dump(args.output / "completed.json", {"completed": True})


if __name__ == "__main__":
    main()
