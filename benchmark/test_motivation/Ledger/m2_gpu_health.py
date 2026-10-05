"""Read-only diagnostics for explicitly selected physical experiment GPUs."""

import asyncio
import csv
import io
import subprocess
import time


def snapshot(gpu_ids=(6, 7)):
    result = subprocess.run([
        "nvidia-smi", "-i", ",".join(map(str, gpu_ids)),
        "--query-gpu=index,utilization.gpu,clocks.sm,temperature.gpu,power.draw,memory.used",
        "--format=csv,noheader,nounits",
    ], capture_output=True, text=True, check=True, timeout=10)
    keys = ("index", "utilization", "sm_mhz", "temperature_c", "power_w", "memory_mib")
    rows = [{key: float(value.strip()) for key, value in zip(keys, row)}
            for row in csv.reader(io.StringIO(result.stdout))]
    if len(rows) != len(gpu_ids) or {r["index"] for r in rows} != set(gpu_ids):
        raise ValueError("Unexpected GPU health readback")
    return {"observed_s": time.perf_counter(), "gpus": rows}


async def monitor(stop, samples, gpu_ids=(6, 7)):
    while not stop.is_set():
        samples.append(await asyncio.to_thread(snapshot, gpu_ids))
        try:
            await asyncio.wait_for(stop.wait(), timeout=.5)
        except asyncio.TimeoutError:
            pass


def health_errors(samples, gpu_ids=(6, 7)):
    if not samples:
        return ["missing GPU health samples"]
    consecutive = dict.fromkeys(gpu_ids, 0)
    for sample in samples:
        if len(sample["gpus"]) != len(gpu_ids) or {g["index"] for g in sample["gpus"]} != set(gpu_ids):
            return ["GPU health samples do not match selected physical devices"]
        for gpu in sample["gpus"]:
            index = int(gpu["index"])
            low = gpu["utilization"] >= 80 and gpu["sm_mhz"] <= 300
            consecutive[index] = consecutive[index] + 1 if low else 0
            if consecutive[index] >= 3:
                return [f"GPU {index}: three loaded samples at <=300MHz; timing invalid"]
    return []
