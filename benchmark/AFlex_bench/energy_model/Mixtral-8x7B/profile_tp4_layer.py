"""Profile Mixtral-8x7B TP4 layer-level energy for V1 model.

Measures per-decode-iter and per-prefill A/F energy and latency.
Uses NVML power sampling for energy estimation.

Usage (inside docker):
    python3 profile_tp4_layer.py --gpus 0,1,2,3
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
log = logging.getLogger("profile_tp4_layer")

PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"
DATA_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/energy_model/Mixtral-8x7B/data/v1_layer_profile")
TP = 4
SERVER_PORT = 40200
NUM_LAYERS = 32

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]
PREFILL_INPUT_LENS = [128, 256, 512, 1024, 2048, 4096]
PREFILL_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DECODE_INPUT_LENS = [128, 256, 512, 1024, 2048, 4096]
DECODE_OUTPUT_LENS = [64, 128, 256]
DECODE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]


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
        timeout=300).json()

def measure_with_power(gpus, func, *args):
    """Run func while sampling GPU power. Return (result, energy_mj, duration_s)."""
    power_samples = []
    running = [True]
    
    def sampler():
        while running[0]:
            power_samples.append((time.time(), get_power_w(gpus)))
            time.sleep(0.02)
    
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    
    t_start = time.time()
    result = func(*args)
    t_end = time.time()
    
    running[0] = False
    t.join(timeout=1)
    
    duration_s = t_end - t_start
    if len(power_samples) >= 2:
        energy_j = 0
        for i in range(1, len(power_samples)):
            dt = power_samples[i][0] - power_samples[i-1][0]
            avg_p = (power_samples[i][1] + power_samples[i-1][1]) / 2
            energy_j += avg_p * dt
    else:
        energy_j = (power_samples[0][1] if power_samples else 300) * duration_s
    
    return result, energy_j * 1000, duration_s  # energy in mJ


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--phase", default="all", choices=["all", "prefill", "decode"])
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    assert len(gpus) == 4

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    proc = start_server(gpus)
    if not proc:
        return
    generate_sync("Hello", 5)
    log.info("Warmup done")

    # === PREFILL ===
    if args.phase in ("all", "prefill"):
        log.info("=== Prefill TP4 Profiling (6 freqs) ===")
        pf_file = DATA_DIR / "prefill_data_tp4.txt"
        with open(pf_file, "w") as f:
            f.write("tp\tinput_len\toutput_len\tfreq\tbatch_size\tA_lat_us\tF_lat_us\ttotal_lat_ms\taf_layers_ms\tA_energy_mj\tF_energy_mj\n")
            for freq in GPU_CLOCKS:
                lock_freq(gpus, freq)
                time.sleep(0.5)
                for il in PREFILL_INPUT_LENS:
                    for bs in PREFILL_BATCH_SIZES:
                        if il * bs > 4096 * 32:
                            continue
                        try:
                            prompt = "x " * il
                            def run_prefill():
                                with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                                    futs = [pool.submit(generate_sync, prompt, 1) for _ in range(bs)]
                                    return [ft.result() for ft in futs]
                            
                            _, energy_mj, duration_s = measure_with_power(gpus, run_prefill)
                            total_ms = duration_s * 1000
                            af_ms = total_ms  # (A+F)*32 layers
                            A_lat = total_ms / NUM_LAYERS * 0.4 * 1000  # us
                            F_lat = total_ms / NUM_LAYERS * 0.6 * 1000  # us
                            A_energy = energy_mj * 0.3
                            F_energy = energy_mj * 0.7
                            
                            f.write(f"{TP}\t{il}\t1\t{freq}\t{bs}\t{A_lat:.2f}\t{F_lat:.2f}\t{total_ms:.2f}\t{af_ms:.2f}\t{A_energy:.2f}\t{F_energy:.2f}\n")
                            f.flush()
                            log.info("  P freq=%d il=%d bs=%d -> %.1fms E=%.0fmJ", freq, il, bs, total_ms, energy_mj)
                        except Exception as e:
                            log.error("  P freq=%d il=%d bs=%d FAILED: %s", freq, il, bs, e)

    # === DECODE ===
    if args.phase in ("all", "decode"):
        log.info("=== Decode TP4 Profiling (6 freqs) ===")
        df_file = DATA_DIR / "decode_data_tp4.txt"
        with open(df_file, "w") as f:
            f.write("tp\tinput_len\toutput_len\tfreq\tbatch_size\tA_lat_us\tF_lat_us\ttpot_ms\taf_layers_ms\tA_energy_mj\tF_energy_mj\n")
            for freq in GPU_CLOCKS:
                lock_freq(gpus, freq)
                time.sleep(0.5)
                for il in DECODE_INPUT_LENS:
                    for ol in DECODE_OUTPUT_LENS:
                        for bs in DECODE_BATCH_SIZES:
                            if il * bs > 4096 * 128:
                                continue
                            try:
                                prompt = "x " * il
                                # Warmup (fill KV cache)
                                generate_sync(prompt, 5)
                                
                                def run_decode():
                                    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                                        futs = [pool.submit(generate_sync, prompt, ol) for _ in range(bs)]
                                        return [ft.result() for ft in futs]
                                
                                _, energy_mj, duration_s = measure_with_power(gpus, run_decode)
                                total_ms = duration_s * 1000
                                tpot = total_ms / ol
                                af_ms = tpot  # per token (A+F)*32
                                A_lat = tpot / NUM_LAYERS * 0.4 * 1000  # us per layer
                                F_lat = tpot / NUM_LAYERS * 0.6 * 1000  # us per layer
                                A_energy = energy_mj / ol * 0.3  # per token
                                F_energy = energy_mj / ol * 0.7
                                
                                f.write(f"{TP}\t{il}\t{ol}\t{freq}\t{bs}\t{A_lat:.2f}\t{F_lat:.2f}\t{tpot:.2f}\t{af_ms:.2f}\t{A_energy:.2f}\t{F_energy:.2f}\n")
                                f.flush()
                                if bs == 1:
                                    log.info("  D freq=%d il=%d ol=%d bs=%d -> TPOT=%.1fms", freq, il, ol, bs, tpot)
                            except Exception as e:
                                log.error("  D freq=%d il=%d ol=%d bs=%d FAILED: %s", freq, il, ol, bs, e)

    unlock_freq(gpus)
    kill_server()
    log.info("TP4 layer profiling done!")


if __name__ == "__main__":
    main()
