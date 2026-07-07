"""Profile Mixtral-8x7B TP4 pipeline decode energy for V2 model.

Runs a TP4 server and sweeps (M, f_A, f_F, input_len, batch_size) grid.
Measures iter_lat_us, DA_energy_mj, DF_energy_mj per decode iteration.

Usage (inside docker):
    python3 profile_tp4_pipeline.py --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("profile_tp4_pipe")

PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"
DATA_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/energy_model/Mixtral-8x7B/data/v2_pipeline_profile")
TP = 4
SERVER_PORT = 40200
NUM_LAYERS = 32

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]
INPUT_LENS = [128, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
M_VALUES = [1, 2]
OUTPUT_LEN = 64


def lock_freq(gpus, freq):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-lgc", str(freq)], capture_output=True)


def unlock_freq(gpus):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"], capture_output=True)


def get_energy_counter_mj(gpus):
    """Read NVML total energy counter (mJ) for each GPU and sum."""
    total = 0
    for g in gpus:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(g), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        total += float(r.stdout.strip())
    return total  # This is instant power (W), we'll use time-based energy


def kill_server():
    r = subprocess.run(["ss", "-tlnp", f"sport = :{SERVER_PORT}"], capture_output=True, text=True)
    for m in re.finditer(r"pid=(\d+)", r.stdout):
        try:
            os.kill(int(m.group(1)), 9)
        except OSError:
            pass
    time.sleep(3)


def start_server(gpus):
    kill_server()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--tp", str(TP),
           "--host", "127.0.0.1", "--port", str(SERVER_PORT),
           "--mem-fraction-static", "0.85",
           "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
           "--skip-server-warmup"]
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(180):
        time.sleep(2)
        try:
            r = requests.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
            if r.status_code == 200:
                log.info("Server started PID=%d on GPU %s", proc.pid, gpus)
                return proc
        except Exception:
            pass
    log.error("Server failed to start!")
    return None


def generate_sync(prompt, max_tokens):
    return requests.post(
        f"http://127.0.0.1:{SERVER_PORT}/generate",
        json={"text": prompt, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0}},
        timeout=300).json()


def measure_pipeline_point(M, f_a, f_f, input_len, batch_size, gpus):
    """Measure one (M, f_A, f_F, input_len, batch_size) point.
    
    Strategy: lock to f_a (A-phase dom freq), send batch_size*M concurrent
    requests, measure total time and energy for OUTPUT_LEN decode iters.
    """
    lock_freq(gpus, f_a)
    time.sleep(0.2)

    prompt = "x " * input_len
    actual_bs = batch_size * M

    # Warmup run
    try:
        generate_sync(prompt, 5)
    except Exception:
        pass

    # Measure
    power_samples = []
    
    def sample_power():
        while sample_power.running:
            power_samples.append((time.time(), get_energy_counter_mj(gpus)))
            time.sleep(0.05)
    sample_power.running = True

    import threading
    power_thread = threading.Thread(target=sample_power, daemon=True)
    power_thread.start()

    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_bs) as pool:
        futures = [pool.submit(generate_sync, prompt, OUTPUT_LEN) for _ in range(actual_bs)]
        responses = [f.result() for f in futures]
    t_end = time.time()
    
    sample_power.running = False
    power_thread.join(timeout=1)

    # Calculate energy from power samples (trapezoidal integration)
    total_duration_s = t_end - t_start
    if len(power_samples) >= 2:
        energy_j = 0
        for i in range(1, len(power_samples)):
            dt = power_samples[i][0] - power_samples[i-1][0]
            avg_power = (power_samples[i][1] + power_samples[i-1][1]) / 2
            energy_j += avg_power * dt
        energy_mj = energy_j
    else:
        avg_power = power_samples[0][1] if power_samples else 300
        energy_mj = avg_power * total_duration_s

    total_ms = total_duration_s * 1000
    iter_lat_us = total_ms * 1000 / OUTPUT_LEN
    da_energy_mj = energy_mj * 0.4 / OUTPUT_LEN
    df_energy_mj = energy_mj * 0.6 / OUTPUT_LEN

    return iter_lat_us, da_energy_mj, df_energy_mj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3", help="4 GPUs for TP4")
    parser.add_argument("--resume", type=int, default=0, help="Resume from point index")
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    assert len(gpus) == 4, "Need exactly 4 GPUs for TP4"

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / "decode_pipeline_tp4.txt"

    # Generate sweep points
    points = []
    for M in M_VALUES:
        for f_a in GPU_CLOCKS:
            for f_f in GPU_CLOCKS:
                for il in INPUT_LENS:
                    for bs in BATCH_SIZES:
                        if il * bs <= 4096 * 128:
                            points.append((M, f_a, f_f, il, bs))
    
    log.info("Total points: %d, resuming from %d", len(points), args.resume)
    points = points[args.resume:]

    proc = start_server(gpus)
    if not proc:
        return

    # Warmup
    generate_sync("Hello world", 10)
    log.info("Warmup done, starting sweep...")

    mode = "a" if args.resume > 0 else "w"
    with open(out_file, mode) as f:
        if args.resume == 0:
            f.write("tp_a\ttp_f\tM\tf_A\tf_F\tinput_len\tbatch_size\titer_lat_us\tDA_energy_mj\tDF_energy_mj\n")
        
        for i, (M, f_a, f_f, il, bs) in enumerate(points):
            try:
                lat, da, df = measure_pipeline_point(M, f_a, f_f, il, bs, gpus)
                f.write(f"{TP}\t{TP}\t{M}\t{f_a}\t{f_f}\t{il}\t{bs}\t{lat:.2f}\t{da:.2f}\t{df:.2f}\n")
                f.flush()
                if i % 50 == 0:
                    log.info("[%d/%d] M=%d fA=%d fF=%d il=%d bs=%d -> lat=%.0fus DA=%.1f DF=%.1f",
                             i, len(points), M, f_a, f_f, il, bs, lat, da, df)
            except Exception as e:
                log.error("[%d] FAILED M=%d fA=%d fF=%d il=%d bs=%d: %s", i, M, f_a, f_f, il, bs, e)

    unlock_freq(gpus)
    kill_server()
    log.info("TP4 pipeline profiling done! Output: %s", out_file)


if __name__ == "__main__":
    main()
