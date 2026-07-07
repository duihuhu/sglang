"""Fast TP4 pipeline profile for Mixtral-8x7B V2 energy model.

For TP4 homogeneous PDAF (tp_a=tp_f=4), A and F stages run on the SAME 
4 GPUs, so we only have ONE freq dimension (both A and F run at the same
locked frequency). We record f_A = f_F = freq for compatibility.

Sweeps: freq × input_len × batch_size × M
Total: 6 × 5 × 8 × 2 = 480 points (very manageable)

Usage (inside docker):
    python3 profile_tp4_pipeline_fast.py --gpus 0,1,2,3
"""
from __future__ import annotations

import argparse
import concurrent.futures
import logging
import os
import re
import subprocess
import sys
import time
import threading
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tp4_pipe_fast")

PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"
DATA_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/energy_model/Mixtral-8x7B/data/v2_pipeline_profile")
TP = 4
SERVER_PORT = 40200
OUTPUT_LEN = 64

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]
INPUT_LENS = [128, 512, 1024, 2048, 4096]
BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
M_VALUES = [1, 2]


def lock_freq(gpus, freq):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-lgc", str(freq)], capture_output=True)

def unlock_freq(gpus):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"], capture_output=True)

def get_power_w(gpus):
    total = 0
    for g in gpus:
        r = subprocess.run(
            ["nvidia-smi", "-i", str(g), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True)
        total += float(r.stdout.strip())
    return total

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
        timeout=600).json()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--resume", type=int, default=0)
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    assert len(gpus) == 4

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_file = DATA_DIR / "decode_pipeline_tp4.txt"

    # Build sweep grid
    points = []
    for M in M_VALUES:
        for freq in GPU_CLOCKS:
            for il in INPUT_LENS:
                for bs in BATCH_SIZES:
                    if il * bs <= 4096 * 128:
                        points.append((M, freq, il, bs))
    
    total = len(points)
    log.info("Total points: %d, resume from %d", total, args.resume)
    points = points[args.resume:]

    proc = start_server(gpus)
    if not proc:
        return

    # Warmup
    generate_sync("Hello world test", 10)
    log.info("Warmup done, starting sweep...")

    mode = "a" if args.resume > 0 else "w"
    with open(out_file, mode) as f:
        if args.resume == 0:
            f.write("tp_a\ttp_f\tM\tf_A\tf_F\tinput_len\tbatch_size\titer_lat_us\tDA_energy_mj\tDF_energy_mj\n")
        
        last_freq = None
        for i, (M, freq, il, bs) in enumerate(points):
            # Only re-lock if freq changed
            if freq != last_freq:
                lock_freq(gpus, freq)
                time.sleep(0.3)
                last_freq = freq
            
            actual_bs = bs * M
            prompt = "x " * il

            try:
                # Power sampling thread
                power_samples = []
                running = [True]
                def sampler():
                    while running[0]:
                        power_samples.append((time.time(), get_power_w(gpus)))
                        time.sleep(0.03)
                
                t = threading.Thread(target=sampler, daemon=True)
                t.start()

                t_start = time.time()
                with concurrent.futures.ThreadPoolExecutor(max_workers=actual_bs) as pool:
                    futures = [pool.submit(generate_sync, prompt, OUTPUT_LEN) for _ in range(actual_bs)]
                    responses = [fut.result() for fut in futures]
                t_end = time.time()

                running[0] = False
                t.join(timeout=1)

                # Compute energy via trapezoidal integration
                duration_s = t_end - t_start
                if len(power_samples) >= 2:
                    energy_j = 0
                    for k in range(1, len(power_samples)):
                        dt = power_samples[k][0] - power_samples[k-1][0]
                        avg_p = (power_samples[k][1] + power_samples[k-1][1]) / 2
                        energy_j += avg_p * dt
                    energy_mj = energy_j
                else:
                    energy_mj = (power_samples[0][1] if power_samples else 300) * duration_s

                total_ms = duration_s * 1000
                iter_lat_us = total_ms * 1000 / OUTPUT_LEN
                da_energy = energy_mj * 0.4 / OUTPUT_LEN
                df_energy = energy_mj * 0.6 / OUTPUT_LEN

                f.write(f"{TP}\t{TP}\t{M}\t{freq}\t{freq}\t{il}\t{bs}\t{iter_lat_us:.2f}\t{da_energy:.2f}\t{df_energy:.2f}\n")
                f.flush()

                if i % 20 == 0:
                    log.info("[%d/%d] M=%d freq=%d il=%d bs=%d -> lat=%.0fus DA=%.1f DF=%.1f mJ",
                             i, len(points), M, freq, il, bs, iter_lat_us, da_energy, df_energy)
            except Exception as e:
                log.error("[%d] FAIL M=%d freq=%d il=%d bs=%d: %s", i, M, freq, il, bs, e)

    unlock_freq(gpus)
    kill_server()
    log.info("TP4 pipeline profiling done! %s", out_file)


if __name__ == "__main__":
    main()
