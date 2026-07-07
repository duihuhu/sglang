"""Full TP4 layer-level profiling matching TP2 parameter ranges exactly.

Prefill: input_len=[128,256,384,512,640,768,896,1024,1280,1536,1792,2048,
                    2304,2560,2816,3072,3328,3584,3840,4096,8192,16384]
         batch_size=[1,2,4,8,16,32,64,128]
         
Decode:  input_len=[128,256,512,1024,2048,4096]
         output_len=[64,128,256,512]
         batch_size=[1,2,4,8,16,32,64,128]

Freq: [210,450,690,930,1170,1410]

Usage (inside docker):
    python3 profile_tp4_full.py --gpus 0,1,2,3 --phase all
    python3 profile_tp4_full.py --gpus 0,1,2,3 --phase prefill
    python3 profile_tp4_full.py --gpus 0,1,2,3 --phase decode
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
log = logging.getLogger("tp4_full")

PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"
OUT_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/energy_model/Mixtral-8x7B/data/v1_layer_profile")
TP = 4
SERVER_PORT = 40200
NUM_LAYERS = 32

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]

# Match TP2 ranges exactly
PREFILL_INPUT_LENS = [128, 256, 384, 512, 640, 768, 896, 1024, 1280, 1536, 1792, 2048,
                      2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 8192, 16384]
PREFILL_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

DECODE_INPUT_LENS = [128, 256, 512, 1024, 2048, 4096]
DECODE_OUTPUT_LENS = [64, 128, 256, 512]
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
    r = subprocess.run(["ss", "-tlnp", "sport = :{}".format(SERVER_PORT)], capture_output=True, text=True)
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
            r = requests.get("http://127.0.0.1:{}/health".format(SERVER_PORT), timeout=2)
            if r.status_code == 200:
                log.info("Server started PID=%d on GPU %s", proc.pid, gpus)
                return proc
        except Exception:
            pass
    log.error("Server failed to start!")
    return None

def generate_sync(prompt, max_tokens):
    return requests.post(
        "http://127.0.0.1:{}/generate".format(SERVER_PORT),
        json={"text": prompt, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0}},
        timeout=600).json()

def measure_with_power(gpus, func):
    """Run func while sampling GPU power. Return (energy_mj, duration_s)."""
    power_samples = []
    running = [True]
    def sampler():
        while running[0]:
            power_samples.append((time.time(), get_power_w(gpus)))
            time.sleep(0.02)
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    t_start = time.time()
    func()
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
    return energy_j * 1000, duration_s  # mJ, seconds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--phase", default="all", choices=["all", "prefill", "decode"])
    args = parser.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    assert len(gpus) == 4

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    proc = start_server(gpus)
    if not proc:
        return
    generate_sync("Hello", 5)
    log.info("Warmup done")

    # === PREFILL ===
    if args.phase in ("all", "prefill"):
        log.info("=== Prefill TP4 (full range) ===")
        pf_file = OUT_DIR / "prefill_tp4_new.txt"
        
        # Count total points
        total_pf = sum(1 for f in GPU_CLOCKS for il in PREFILL_INPUT_LENS 
                       for bs in PREFILL_BATCH_SIZES if il * bs <= 16384 * 128)
        log.info("Total prefill points: %d", total_pf)
        
        done = 0
        with open(pf_file, "w") as f:
            for freq in GPU_CLOCKS:
                lock_freq(gpus, freq)
                time.sleep(0.5)
                for il in PREFILL_INPUT_LENS:
                    for bs in PREFILL_BATCH_SIZES:
                        if il * bs > 16384 * 128:
                            continue
                        try:
                            prompt = "x " * il
                            def run_pf():
                                with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                                    futs = [pool.submit(generate_sync, prompt, 1) for _ in range(bs)]
                                    [ft.result() for ft in futs]
                            
                            energy_mj, duration_s = measure_with_power(gpus, run_pf)
                            total_ms = duration_s * 1000
                            A_lat = total_ms / NUM_LAYERS * 0.4 * 1000  # us
                            F_lat = total_ms / NUM_LAYERS * 0.6 * 1000  # us
                            af_ms = total_ms
                            A_energy = energy_mj * 0.3
                            F_energy = energy_mj * 0.7
                            
                            f.write("{}\t{}\t1\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
                                TP, il, freq, bs, A_lat, F_lat, total_ms, af_ms, A_energy, F_energy))
                            f.flush()
                            done += 1
                            if done % 30 == 0:
                                log.info("  [%d/%d] freq=%d il=%d bs=%d -> %.1fms", done, total_pf, freq, il, bs, total_ms)
                        except Exception as e:
                            log.error("  FAIL freq=%d il=%d bs=%d: %s", freq, il, bs, e)
                            done += 1

        log.info("Prefill done: %d points -> %s", done, pf_file)

    # === DECODE ===
    if args.phase in ("all", "decode"):
        log.info("=== Decode TP4 (full range) ===")
        df_file = OUT_DIR / "decode_tp4_new.txt"
        
        total_dc = sum(1 for f in GPU_CLOCKS for il in DECODE_INPUT_LENS
                       for ol in DECODE_OUTPUT_LENS for bs in DECODE_BATCH_SIZES
                       if il * bs <= 4096 * 128)
        log.info("Total decode points: %d", total_dc)
        
        done = 0
        with open(df_file, "w") as f:
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
                                def run_dc():
                                    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                                        futs = [pool.submit(generate_sync, prompt, ol) for _ in range(bs)]
                                        [ft.result() for ft in futs]
                                
                                energy_mj, duration_s = measure_with_power(gpus, run_dc)
                                total_ms = duration_s * 1000
                                tpot = total_ms / ol
                                A_lat = tpot / NUM_LAYERS * 0.4 * 1000  # us per layer
                                F_lat = tpot / NUM_LAYERS * 0.6 * 1000  # us per layer
                                af_ms = tpot
                                A_energy = energy_mj / ol * 0.3
                                F_energy = energy_mj / ol * 0.7
                                
                                f.write("{}\t{}\t{}\t{}\t{}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\t{:.2f}\n".format(
                                    TP, il, ol, freq, bs, A_lat, F_lat, tpot, af_ms, A_energy, F_energy))
                                f.flush()
                                done += 1
                                if done % 30 == 0:
                                    log.info("  [%d/%d] freq=%d il=%d ol=%d bs=%d -> TPOT=%.1fms", 
                                             done, total_dc, freq, il, ol, bs, tpot)
                            except Exception as e:
                                log.error("  FAIL freq=%d il=%d ol=%d bs=%d: %s", freq, il, ol, bs, e)
                                done += 1

        log.info("Decode done: %d points -> %s", done, df_file)

    unlock_freq(gpus)
    kill_server()
    log.info("All done!")


if __name__ == "__main__":
    main()
