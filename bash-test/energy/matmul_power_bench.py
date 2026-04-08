#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple

import torch


def _parse_gpus(raw: str) -> List[int]:
    out = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    if not out:
        raise ValueError("--gpu must not be empty, e.g. 0 or 0,1,2")
    return out


def _parse_sizes(raw: str) -> List[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("--sizes must not be empty")
    return out


def _parse_clocks(raw: str) -> List[int]:
    out = [int(x.strip()) for x in raw.split(",") if x.strip()]
    if not out:
        raise ValueError("--gpu-clocks must not be empty")
    return out


def _dtype_from_str(name: str) -> torch.dtype:
    m = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if name not in m:
        raise ValueError(f"Unsupported dtype: {name}")
    return m[name]


def _run_nvidia_smi_lgc(gpu: int, mhz: int) -> None:
    nv = shutil.which("nvidia-smi")
    if not nv:
        raise RuntimeError("nvidia-smi not found")
    subprocess.run(
        [nv, "-i", str(gpu), "-lgc", f"{mhz},{mhz}"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


def _run_nvidia_smi_rgc(gpu: int) -> None:
    nv = shutil.which("nvidia-smi")
    if not nv:
        return
    subprocess.run(
        [nv, "-i", str(gpu), "-rgc"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _init_nvml(gpu: int):
    import pynvml  # type: ignore

    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu)
    return pynvml, handle


def _read_power_w(pynvml_mod, handle) -> float:
    return float(pynvml_mod.nvmlDeviceGetPowerUsage(handle)) / 1000.0


def _read_total_energy_mj(pynvml_mod, handle):
    try:
        return float(pynvml_mod.nvmlDeviceGetTotalEnergyConsumption(handle))
    except Exception:
        return None


def bench_one_size(
    n: int,
    dtype: torch.dtype,
    warmup_iters: int,
    min_measure_s: float,
    gpu: int,
    pynvml_mod,
    handle,
    verbose_energy_log: bool = False,
):
    device = torch.device(f"cuda:{gpu}")
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)

    for _ in range(warmup_iters):
        _ = torch.matmul(a, b)
    torch.cuda.synchronize(device)

    start_energy_mj = _read_total_energy_mj(pynvml_mod, handle)
    start_power_w = _read_power_w(pynvml_mod, handle)
    t0 = time.perf_counter()
    iters = 0

    while True:
        _ = torch.matmul(a, b)
        iters += 1
        if iters % 4 == 0:
            torch.cuda.synchronize(device)
            if time.perf_counter() - t0 >= min_measure_s:
                break

    torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    end_power_w = _read_power_w(pynvml_mod, handle)
    end_energy_mj = _read_total_energy_mj(pynvml_mod, handle)

    elapsed_s = t1 - t0
    avg_latency_us = (elapsed_s * 1e6) / max(iters, 1)
    energy_by_power_j = 0.5 * (start_power_w + end_power_w) * elapsed_s
    energy_by_counter_j = None
    energy_source = "power_estimate"
    fallback_reason = ""
    if start_energy_mj is None or end_energy_mj is None:
        fallback_reason = "energy counter unavailable"
    elif end_energy_mj < start_energy_mj:
        fallback_reason = "energy counter decreased unexpectedly"
    else:
        energy_by_counter_j = (end_energy_mj - start_energy_mj) / 1000.0
        energy_source = "nvml_total_energy"

    total_energy_j = energy_by_counter_j if energy_by_counter_j is not None else energy_by_power_j

    if verbose_energy_log:
        counter_str = f"{energy_by_counter_j:.6f}" if energy_by_counter_j is not None else "NA"
        print(
            "[energy_log] "
            f"n={n} iters={iters} elapsed_s={elapsed_s:.6f} "
            f"counter_j={counter_str} power_j={energy_by_power_j:.6f} "
            f"selected={energy_source}"
            + (f" fallback_reason={fallback_reason}" if fallback_reason else "")
        )

    avg_power_w = total_energy_j / max(elapsed_s, 1e-9)
    energy_per_op_uj = (total_energy_j * 1e6) / max(iters, 1)
    return {
        "n": n,
        "iters": iters,
        "latency_us": avg_latency_us,
        "power_w": avg_power_w,
        "energy_uj": energy_per_op_uj,
    }


def run_one_gpu(
    gpu: int,
    tasks: List[Tuple[object, int]],
    dtype: torch.dtype,
    warmup_iters: int,
    measure_seconds: float,
    verbose_energy_log: bool = False,
):
    torch.cuda.set_device(gpu)
    pynvml_mod, handle = _init_nvml(gpu)
    rows = []
    current_clk = None
    try:
        for clk, n in tasks:
            if clk != "NA" and clk != current_clk:
                _run_nvidia_smi_lgc(gpu, int(clk))
                time.sleep(0.3)
                current_clk = clk
            r = bench_one_size(
                n=n,
                dtype=dtype,
                warmup_iters=warmup_iters,
                min_measure_s=measure_seconds,
                gpu=gpu,
                pynvml_mod=pynvml_mod,
                handle=handle,
                verbose_energy_log=verbose_energy_log,
            )
            rows.append(
                {
                    "gpu_clock_mhz": clk,
                    "n": r["n"],
                    "iters": r["iters"],
                    "latency_us": f"{r['latency_us']:.2f}",
                    "power_w": f"{r['power_w']:.2f}",
                    "energy_uj": f"{r['energy_uj']:.2f}",
                }
            )
    finally:
        pynvml_mod.nvmlShutdown()
        if current_clk not in (None, "NA"):
            _run_nvidia_smi_rgc(gpu)
    return rows


def main():
    parser = argparse.ArgumentParser(description="Compare matmul power/energy by matrix size.")
    parser.add_argument("--gpu", type=str, default="0", help="Single GPU id or comma list, e.g. 0 or 0,1,2")
    parser.add_argument("--sizes", type=str, default="128,8192")
    parser.add_argument("--gpu-clocks", type=str, default="", help="Comma-separated graphics clocks in MHz")
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--out-csv", type=str, default="", help="Optional output CSV path.")
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--measure-seconds", type=float, default=3.0)
    parser.add_argument(
        "--verbose-energy-log",
        action="store_true",
        help="Print per-case energy details for counter/power methods.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    sizes = _parse_sizes(args.sizes)
    gpus = _parse_gpus(args.gpu)
    clocks = _parse_clocks(args.gpu_clocks) if args.gpu_clocks.strip() else []
    dtype = _dtype_from_str(args.dtype)

    tasks: List[Tuple[object, int]] = []
    if clocks:
        for clk in clocks:
            for n in sizes:
                tasks.append((clk, n))
    else:
        for n in sizes:
            tasks.append(("NA", n))

    gpu_tasks = {g: [] for g in gpus}
    for i, t in enumerate(tasks):
        gpu_tasks[gpus[i % len(gpus)]].append(t)

    rows = []
    print("gpu_clock_mhz,n,iters,latency_us,power_w,energy_uj")
    with ThreadPoolExecutor(max_workers=len(gpus)) as ex:
        fut_map = {
            ex.submit(
                run_one_gpu,
                gpu,
                gpu_tasks[gpu],
                dtype,
                args.warmup_iters,
                args.measure_seconds,
                args.verbose_energy_log,
            ): gpu
            for gpu in gpus
        }
        for fut in as_completed(fut_map):
            rows.extend(fut.result())

    def _sort_key(r):
        clk_raw = str(r["gpu_clock_mhz"]).strip()
        clk = int(clk_raw) if clk_raw.isdigit() else 10**9
        return (clk, int(r["n"]))

    rows.sort(key=_sort_key)
    for r in rows:
        print(
            f"{r['gpu_clock_mhz']},{r['n']},{r['iters']},"
            f"{r['latency_us']},{r['power_w']},{r['energy_uj']}"
        )

    if args.out_csv:
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "gpu_clock_mhz",
                    "n",
                    "iters",
                    "latency_us",
                    "power_w",
                    "energy_uj",
                ],
            )
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
