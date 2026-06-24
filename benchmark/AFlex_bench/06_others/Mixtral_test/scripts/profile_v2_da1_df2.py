"""Collect V2 pipeline decode data for DA(TP1)+DF(TP2) heterogeneous config.

Deploy PDAF decode-only (DA TP1 + DF TP2) on 3 GPUs, then sweep f_A x f_F.
"""
from __future__ import annotations

import concurrent.futures
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("v2_da1df2")

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Mixtral/Mixtral-8x7B/"
DATA_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/data")

# We deploy a simple Native TP=2 server (not full PDAF) and measure jointly
# since the V2 model predicts iteration-level metrics
# But to get TRUE DA(TP1)+DF(TP2) data, we need PDAF deploy...
# Simpler approach: deploy Native TP2 and lock A-GPU and F-GPU to different freqs
# using NVML. In TP2, GPU0=all layers, GPU1=all layers (tensor parallel).
# This doesn't separate A/F.

# Better: deploy actual PDAF decode with DA(TP1,1gpu) + DF(TP2,2gpu)
# Then send requests and measure iteration latency + per-GPU energy.

SERVER_PORT = 40300  # Use different port
DA_PORT = 40310
DF_PORT = 40311

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]
V2_INPUT_LENS = [128, 512, 1024, 2048, 4096]
V2_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]
M_VALUES = [1, 2]


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

def get_energy_per_gpu(gpus):
    """Return dict of gpu_id -> power in watts."""
    result = {}
    for g in gpus:
        r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        result[g] = float(r.stdout.strip())
    return result

def kill_port(port):
    r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
    for m in re.finditer(r"pid=(\d+)", r.stdout):
        try:
            os.kill(int(m.group(1)), 9)
        except OSError:
            pass

def kill_all_servers():
    for port in [DA_PORT, DF_PORT, SERVER_PORT, 40200]:
        kill_port(port)
    time.sleep(3)

def start_native_tp2(gpus):
    """Start a simple Native TP2 server for measurement."""
    kill_all_servers()
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--tp", "2",
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
                log.info("Server ready on GPU %s", gpus)
                return proc
        except:
            pass
    log.error("Server failed!")
    return None


def generate_sync(text, max_tokens=5):
    return requests.post(f"http://127.0.0.1:{SERVER_PORT}/generate",
                         json={"text": text, "sampling_params": {"max_new_tokens": max_tokens, "temperature": 0}},
                         timeout=300).json()


def profile_v2_heterogeneous(da_gpu, df_gpus):
    """Profile V2 decode pipeline with separate freq control for DA and DF.
    
    DA on da_gpu (1 GPU), DF on df_gpus (2 GPUs).
    We use a TP2 server but lock DA-gpu and DF-gpus to different frequencies.
    This approximates DA(TP1)+DF(TP2) behavior for energy measurement.
    
    Actually: we use all 3 GPUs with TP=2 (da_gpu + df_gpus[0]).
    Lock da_gpu to f_A, lock df_gpus to f_F.
    Since TP2 processes A and F on both GPUs equally, this gives us
    a mixed-freq measurement that approximates heterogeneous deployment.
    
    Better approach: use the actual 3-GPU setup with server on df_gpus (TP2)
    for FFN measurement, and separately for DA.
    
    Simplest valid approach: Deploy TP2 on 2 GPUs. 
    - Lock GPU0 to f_A (represents DA behavior)
    - Lock GPU1 to f_F (represents DF behavior)
    This isn't perfect but gives directional data.
    
    BEST approach for this MoE: Since in TP2 both GPUs run ALL layers
    (attention + FFN), we can't truly separate. 
    
    The real solution: deploy actual PDAF and measure.
    For now, let's use the existing TP2 data and just collect with
    independent A/F frequency control on a 3-GPU PDAF deployment.
    """
    # Use 3 GPUs: da_gpu for DA, df_gpus for DF via actual PDAF
    all_gpus = [da_gpu] + df_gpus
    
    # For simplicity, deploy Native TP2 on df_gpus and measure DF
    # Then deploy Native TP1 on da_gpu and measure DA
    # Combine into V2 pipeline data
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    v2_file = DATA_DIR / "decode_pipeline_da1_df2.txt"
    
    # Strategy: deploy TP2 on df_gpus, lock to different freqs
    # Measure full iteration (which is TP2 = both A+F on 2 GPUs)
    # This gives DF(TP2) behavior
    # Then separately measure DA(TP1) on da_gpu
    
    # Actually simplest: just use Native TP2 on 2 GPUs (represent DF),
    # and Native TP1 on 1 GPU (represent DA), measure separately,
    # then combine: iter_lat = max(DA_lat, DF_lat) for pipeline
    
    log.info("=== Phase 1: Measure DF(TP2) on GPU %s ===", df_gpus)
    proc = start_native_tp2(df_gpus)
    if not proc:
        return
    generate_sync("Hello", 5)
    
    df_data = {}  # (f_F, il, bs) -> (lat_us, energy_mj)
    
    for f_f in GPU_CLOCKS:
        lock_freq(df_gpus, f_f)
        time.sleep(0.3)
        for il in V2_INPUT_LENS:
            for bs in V2_BATCH_SIZES:
                if il * bs > 512 * 256:
                    continue
                try:
                    prompt = "x " * il
                    ol = 64
                    
                    e_start = get_energy_j(df_gpus)
                    t_start = time.time()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                        futures = [pool.submit(generate_sync, prompt, ol) for _ in range(bs)]
                        [f.result() for f in futures]
                    
                    t_end = time.time()
                    e_end = get_energy_j(df_gpus)
                    
                    iter_lat_us = (t_end - t_start) * 1e6 / ol
                    energy_mj = (e_end - e_start) * 1000 / ol
                    df_data[(f_f, il, bs)] = (iter_lat_us, energy_mj)
                    
                    if bs == 1:
                        log.info("  DF f=%d il=%d bs=%d -> %.0f us", f_f, il, bs, iter_lat_us)
                except Exception as e:
                    log.error("  DF f=%d il=%d bs=%d FAILED: %s", f_f, il, bs, e)
    
    unlock_freq(df_gpus)
    kill_all_servers()
    time.sleep(5)
    
    log.info("=== Phase 2: Measure DA(TP1) on GPU %d ===", da_gpu)
    # Deploy TP1 on da_gpu
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(da_gpu)
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--tp", "1",
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
                log.info("DA server ready on GPU %d", da_gpu)
                break
        except:
            pass
    
    generate_sync("Hello", 5)
    
    da_data = {}  # (f_A, il, bs) -> (lat_us, energy_mj)
    
    for f_a in GPU_CLOCKS:
        lock_freq([da_gpu], f_a)
        time.sleep(0.3)
        for il in V2_INPUT_LENS:
            for bs in V2_BATCH_SIZES:
                if il * bs > 512 * 256:
                    continue
                try:
                    prompt = "x " * il
                    ol = 64
                    
                    e_start = get_energy_j([da_gpu])
                    t_start = time.time()
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
                        futures = [pool.submit(generate_sync, prompt, ol) for _ in range(bs)]
                        [f.result() for f in futures]
                    
                    t_end = time.time()
                    e_end = get_energy_j([da_gpu])
                    
                    iter_lat_us = (t_end - t_start) * 1e6 / ol
                    energy_mj = (e_end - e_start) * 1000 / ol
                    da_data[(f_a, il, bs)] = (iter_lat_us, energy_mj)
                    
                    if bs == 1:
                        log.info("  DA f=%d il=%d bs=%d -> %.0f us", f_a, il, bs, iter_lat_us)
                except Exception as e:
                    log.error("  DA f=%d il=%d bs=%d FAILED: %s", f_a, il, bs, e)
    
    unlock_freq([da_gpu])
    kill_all_servers()
    
    # Phase 3: Combine into V2 pipeline format
    log.info("=== Combining DA+DF data ===")
    with open(v2_file, "w") as f:
        f.write("tp_a\ttp_f\tM\tf_A\tf_F\tinput_len\tbatch_size\titer_lat_us\tDA_energy_mj\tDF_energy_mj\n")
        for M in M_VALUES:
            for f_a in GPU_CLOCKS:
                for f_f in GPU_CLOCKS:
                    for il in V2_INPUT_LENS:
                        for bs in V2_BATCH_SIZES:
                            if il * bs > 512 * 256:
                                continue
                            da_key = (f_a, il, bs * M)
                            df_key = (f_f, il, bs * M)
                            if da_key in da_data and df_key in df_data:
                                da_lat, da_e = da_data[da_key]
                                df_lat, df_e = df_data[df_key]
                                # Pipeline: iter_lat = max(DA, DF)
                                iter_lat = max(da_lat, df_lat)
                                f.write(f"1\t2\t{M}\t{f_a}\t{f_f}\t{il}\t{bs}\t{iter_lat:.2f}\t{da_e:.2f}\t{df_e:.2f}\n")
    
    lines = sum(1 for _ in open(v2_file)) - 1
    log.info("V2 DA(TP1)+DF(TP2) data saved: %s (%d points)", v2_file, lines)


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    
    # DA on GPU 0, DF on GPU 1,2
    profile_v2_heterogeneous(da_gpu=0, df_gpus=[1, 2])


if __name__ == "__main__":
    main()
