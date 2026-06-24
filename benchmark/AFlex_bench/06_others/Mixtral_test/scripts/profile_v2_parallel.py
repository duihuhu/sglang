"""Parallel V2 pipeline profiling using 4 GPU pairs (8 GPUs total).

Each worker gets a TP=2 server and processes its assigned portion of the sweep.
Results are written to separate files then merged.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("v2_parallel")

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Mixtral/Mixtral-8x7B/"
DATA_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/data")
TP = 2

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]
V2_INPUT_LENS = [128, 512, 1024, 2048, 4096]
V2_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
M_VALUES = [1, 2]


def get_all_points():
    points = []
    for M in M_VALUES:
        for f_a in GPU_CLOCKS:
            for f_f in GPU_CLOCKS:
                for il in V2_INPUT_LENS:
                    for bs in V2_BATCH_SIZES:
                        if il * bs <= 512 * 256:
                            points.append((M, f_a, f_f, il, bs))
    return points


def lock_freq(gpus, freq):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-lgc", str(freq)],
                       capture_output=True)


def unlock_freq(gpus):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"], capture_output=True)


def get_energy_j(gpus):
    total = 0
    for g in gpus:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(g), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        total += float(r.stdout.strip())
    return total


def start_server(gpus, port):
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--tp", str(TP),
           "--host", "127.0.0.1", "--port", str(port),
           "--mem-fraction-static", "0.85",
           "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
           "--skip-server-warmup"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    # Wait for health
    for _ in range(120):
        time.sleep(2)
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=2)
            if r.status_code == 200:
                return proc
        except:
            pass
    return None


def kill_port(port):
    r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                       capture_output=True, text=True)
    for m in re.finditer(r"pid=(\d+)", r.stdout):
        try:
            os.kill(int(m.group(1)), 9)
        except OSError:
            pass


def worker(worker_id, gpus, port, points, output_file):
    """One worker: start server, sweep assigned points, write results."""
    log.info("[W%d] Starting server on GPU %s, port %d, %d points",
             worker_id, gpus, port, len(points))

    proc = start_server(gpus, port)
    if proc is None:
        log.error("[W%d] Server failed to start!", worker_id)
        return

    # Warmup
    try:
        requests.post(f"http://127.0.0.1:{port}/generate",
                      json={"text": "Hello", "sampling_params": {"max_new_tokens": 5, "temperature": 0}},
                      timeout=30)
    except:
        pass

    log.info("[W%d] Server ready, starting sweep", worker_id)

    with open(output_file, "w") as f:
        for i, (M, f_a, f_f, il, bs) in enumerate(points):
            try:
                lock_freq(gpus, f_a)
                time.sleep(0.3)

                prompt = "x " * il
                ol = 64
                actual_bs = bs * M

                e_start = get_energy_j(gpus)
                t_start = time.time()

                def _send():
                    return requests.post(
                        f"http://127.0.0.1:{port}/generate",
                        json={"text": prompt, "sampling_params": {"max_new_tokens": ol, "temperature": 0}},
                        timeout=300,
                    ).json()

                with concurrent.futures.ThreadPoolExecutor(max_workers=actual_bs) as pool:
                    futures = [pool.submit(_send) for _ in range(actual_bs)]
                    responses = [fut.result() for fut in futures]

                t_end = time.time()
                e_end = get_energy_j(gpus)

                total_ms = (t_end - t_start) * 1000
                energy_mj = (e_end - e_start) * 1000
                iter_lat_us = total_ms * 1000 / ol
                da_energy = energy_mj * 0.4 / ol
                df_energy = energy_mj * 0.6 / ol

                f.write(f"{TP}\t{TP}\t{M}\t{f_a}\t{f_f}\t{il}\t{bs}\t{iter_lat_us:.2f}\t{da_energy:.2f}\t{df_energy:.2f}\n")
                f.flush()

                if i % 20 == 0:
                    log.info("[W%d] %d/%d M=%d fA=%d fF=%d il=%d bs=%d",
                             worker_id, i, len(points), M, f_a, f_f, il, bs)
            except Exception as e:
                log.error("[W%d] FAILED M=%d fA=%d fF=%d il=%d bs=%d: %s",
                          worker_id, M, f_a, f_f, il, bs, e)

    unlock_freq(gpus)
    kill_port(port)
    log.info("[W%d] Done! %d points written to %s", worker_id, len(points), output_file)


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    all_points = get_all_points()
    # Resume from point 1780 (0-indexed)
    remaining = all_points[1780:]
    log.info("Total remaining: %d points, splitting into 4 workers", len(remaining))

    # 4 workers, GPU pairs: (0,1), (2,3), (4,5), (6,7)
    gpu_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
    ports = [40200, 40201, 40202, 40203]

    n = len(remaining)
    chunk = n // 4
    parts = [remaining[i*chunk:(i+1)*chunk] for i in range(3)]
    parts.append(remaining[3*chunk:])

    output_files = [DATA_DIR / f"decode_pipeline_v1_part{i}.txt" for i in range(4)]

    # Launch all 4 workers in parallel threads
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for i in range(4):
            fut = executor.submit(worker, i, list(gpu_pairs[i]), ports[i], parts[i], output_files[i])
            futures.append(fut)
            time.sleep(5)  # stagger server starts

        for fut in concurrent.futures.as_completed(futures):
            try:
                fut.result()
            except Exception as e:
                log.error("Worker exception: %s", e)

    # Merge results
    log.info("Merging partial results...")
    merged_file = DATA_DIR / "decode_pipeline_v1.txt"
    # Read existing data (first 1780 points + header)
    with open(merged_file, "r") as f:
        existing = f.readlines()

    # Append all parts
    with open(merged_file, "a") as f:
        for pf in output_files:
            if pf.exists():
                with open(pf) as pff:
                    f.write(pff.read())

    final_lines = sum(1 for _ in open(merged_file)) - 1
    log.info("Merge complete! Total data points: %d", final_lines)


if __name__ == "__main__":
    main()
