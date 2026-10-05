#!/usr/bin/env python3
"""Check idle ownership and loaded SM clocks without changing clock settings."""

import argparse
import asyncio
import fcntl
import json
import os
from pathlib import Path
import statistics
import time

from m2_gpu_health import health_errors, monitor, snapshot
from run_m2_batch_frontier import BASE, dump, ensure_available


def load(seconds):
    import torch
    if torch.cuda.device_count() != 2:
        raise RuntimeError("Clock probe requires exactly two visible GPUs")
    matrices = []
    for device in range(2):
        a = torch.randn((4096, 4096), device=f"cuda:{device}", dtype=torch.float16)
        matrices.append((a, torch.empty_like(a)))
    deadline = time.perf_counter() + seconds
    iterations = 0
    while time.perf_counter() < deadline:
        for a, out in matrices:
            for _ in range(16):
                torch.mm(a, a, out=out)
        for device in range(2):
            torch.cuda.synchronize(device)
        iterations += 1
    return iterations


async def check(gpus, seconds):
    samples, stop = [], asyncio.Event()
    initial = snapshot(gpus)
    start = time.perf_counter()
    monitoring = asyncio.create_task(monitor(stop, samples, gpus))
    try:
        iterations = await asyncio.to_thread(load, seconds)
    finally:
        stop.set()
        await monitoring
    steady = [s for s in samples if s["observed_s"] - start >= 5]
    errors = health_errors(steady, gpus)
    ranges = {}
    for device in gpus:
        clocks = [g["sm_mhz"] for s in steady for g in s["gpus"]
                  if int(g["index"]) == device and g["utilization"] >= 80]
        if len(clocks) < 4:
            errors.append(f"GPU {device}: insufficient loaded samples")
        ranges[device] = {"loaded_samples": len(clocks),
                          "min_mhz": min(clocks) if clocks else None,
                          "median_mhz": statistics.median(clocks) if clocks else None,
                          "max_mhz": max(clocks) if clocks else None}
    return {"passed": not errors, "physical_gpu_ids": gpus, "initial": initial,
            "loaded_clocks": ranges, "errors": errors, "samples": samples,
            "iterations": iterations, "duration_s": seconds,
            "scope": "Read-only clock settings; synthetic GEMM preflight, not serving performance."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", nargs=2, type=int, required=True)
    parser.add_argument("--seconds", type=int, default=20)
    parser.add_argument("--port", type=int, default=30129)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0 or not 10 <= args.seconds <= 60:
        parser.error("select two distinct GPUs and a duration from 10 to 60 seconds")
    if args.output.exists():
        raise FileExistsError(args.output)
    key = "".join(map(str, sorted(args.gpus)))
    with (BASE / f".m2_gpu{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        before = ensure_available(args.port, args.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, args.gpus))
        result = asyncio.run(check(args.gpus, args.seconds))
        result["idle_snapshot"] = before
        result["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        dump(args.output, result)
        print(json.dumps({k: v for k, v in result.items() if k != "samples"}, indent=2))
        if not result["passed"]:
            raise RuntimeError("Loaded clock check failed; do not start timing experiments")


if __name__ == "__main__":
    main()
