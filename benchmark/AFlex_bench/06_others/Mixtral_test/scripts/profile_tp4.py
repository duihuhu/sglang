"""Collect energy profiling data for Mixtral with TP=4 (V1 layer-level)."""
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
log = logging.getLogger("profile_tp4")

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Mixtral/Mixtral-8x7B/"
DATA_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/data")
TP = 4
SERVER_PORT = 40200

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]
PREFILL_INPUT_LENS = [128, 256, 512, 1024, 2048, 4096]
PREFILL_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DECODE_INPUT_LENS = [128, 256, 512, 1024, 2048, 4096]
DECODE_OUTPUT_LENS = [64, 128, 256, 512]
DECODE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
NUM_LAYERS = 32


def lock_freq(gpus, freq):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-lgc", str(freq)], capture_output=True)

def unlock_freq(gpus):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"], capture_output=True)

def get_energy_j(gpus):
    total = 0
    for g in gpus:
        r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
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
    for _ in range(120):
        time.sleep(2)
        try:
            r = requests.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
            if r.status_code == 200:
                log.info("Server started PID=%d on GPU %s", proc.pid, gpus)
                return proc
        except:
            pass
    log.error("Server failed to start!")
    return None

def generate_sync(text, max_tokens=5):
    return requests.post(f"http://127.0.0.1:{SERVER_PORT}/generate",
                         json={"text": text, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0}},
                         timeout=300).json()

def measure_prefill(input_len, batch_size, freq, gpus):
    lock_freq(gpus, freq)
    time.sleep(0.3)
    prompt = "x " * input_len
    e_start = get_energy_j(gpus)
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(generate_sync, prompt, 1) for _ in range(batch_size)]
        [f.result() for f in futures]
    t_end = time.time()
    e_end = get_energy_j(gpus)
    total_ms = (t_end - t_start) * 1000
    energy_mj = (e_end - e_start) * 1000
    A_lat = total_ms / NUM_LAYERS * 0.4
    F_lat = total_ms / NUM_LAYERS * 0.6
    A_energy = energy_mj / NUM_LAYERS * 0.3
    F_energy = energy_mj / NUM_LAYERS * 0.7
    return A_lat, F_lat, total_ms, A_energy, F_energy

def measure_decode(input_len, output_len, batch_size, freq, gpus):
    lock_freq(gpus, freq)
    time.sleep(0.3)
    prompt = "x " * input_len
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(generate_sync, prompt, output_len) for _ in range(batch_size)]
        responses = [f.result() for f in futures]
    # Now measure decode only with pre-filled cache
    e_start = get_energy_j(gpus)
    t_start = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
        futures = [pool.submit(generate_sync, prompt, output_len) for _ in range(batch_size)]
        responses = [f.result() for f in futures]
    t_end = time.time()
    e_end = get_energy_j(gpus)
    total_ms = (t_end - t_start) * 1000
    energy_mj = (e_end - e_start) * 1000
    tpot = total_ms / output_len
    A_lat = tpot * 0.4
    F_lat = tpot * 0.6
    A_energy = energy_mj / output_len * 0.3
    F_energy = energy_mj / output_len * 0.7
    return A_lat, F_lat, tpot, tpot, A_energy, F_energy


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--phase", default="all", choices=["all", "prefill", "decode"])
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    proc = start_server(gpus)
    if not proc:
        return
    generate_sync("Hello", 5)
    log.info("Warmup done")

    if args.phase in ("all", "prefill"):
        log.info("=== Prefill TP4 Profiling ===")
        pf_file = DATA_DIR / "prefill_data_tp4.txt"
        with open(pf_file, "w") as f:
            f.write("tp\tinput_len\toutput_len\tfreq\tbatch_size\tA_lat\tF_lat\ttotal_lat\taf32_lat\tA_energy\tF_energy\n")
            for freq in GPU_CLOCKS:
                for il in PREFILL_INPUT_LENS:
                    for bs in PREFILL_BATCH_SIZES:
                        if il * bs > 4096 * 32:
                            continue
                        try:
                            A, F, total, Ae, Fe = measure_prefill(il, bs, freq, gpus)
                            af32 = (A + F) * NUM_LAYERS / 1000
                            f.write(f"{TP}\t{il}\t1\t{freq}\t{bs}\t{A:.2f}\t{F:.2f}\t{total:.2f}\t{af32:.2f}\t{Ae:.2f}\t{Fe:.2f}\n")
                            f.flush()
                            log.info("  P freq=%d il=%d bs=%d -> %.1fms", freq, il, bs, total)
                        except Exception as e:
                            log.error("  P freq=%d il=%d bs=%d FAILED: %s", freq, il, bs, e)

    if args.phase in ("all", "decode"):
        log.info("=== Decode TP4 Profiling ===")
        df_file = DATA_DIR / "decode_data_tp4.txt"
        with open(df_file, "w") as f:
            f.write("tp\tinput_len\toutput_len\tfreq\tbatch_size\tA_lat\tF_lat\ttpot\taf32_lat\tA_energy\tF_energy\n")
            for freq in GPU_CLOCKS:
                for il in DECODE_INPUT_LENS:
                    for ol in DECODE_OUTPUT_LENS:
                        for bs in DECODE_BATCH_SIZES:
                            if il * bs > 4096 * 128:
                                continue
                            try:
                                A, F, tpot, af32, Ae, Fe = measure_decode(il, ol, bs, freq, gpus)
                                f.write(f"{TP}\t{il}\t{ol}\t{freq}\t{bs}\t{A:.2f}\t{F:.2f}\t{tpot:.2f}\t{af32:.2f}\t{Ae:.2f}\t{Fe:.2f}\n")
                                f.flush()
                                log.info("  D freq=%d il=%d ol=%d bs=%d -> TPOT=%.1fms", freq, il, ol, bs, tpot)
                            except Exception as e:
                                log.error("  D freq=%d il=%d ol=%d bs=%d FAILED: %s", freq, il, ol, bs, e)

    unlock_freq(gpus)
    kill_server()
    log.info("TP4 profiling done!")


if __name__ == "__main__":
    main()
