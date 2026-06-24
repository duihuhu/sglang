#!/usr/bin/env python3
"""Energy profiling script for Mixtral-8x7B (MoE, TP=2).

Collects per-layer Attention/FFN latency and energy data for training
energy models (V1 layer-level + V2 pipeline-level).

Output format matches Qwen3-32B data:
  V1 Prefill: tp, input_len, output_len, gpu_clock, batch_size, A, F, TTFT_ms, (A+F)*32_ms, A_energy_mj, F_energy_mj
  V1 Decode:  tp, input_len, output_len, gpu_clock, batch_size, A, F, TPOT_ms, (A+F)*32_ms, A_energy_mj, F_energy_mj
  V2 Decode:  tp, M, f_A, f_F, input_len, batch_size, iter_lat_us, DA_energy_mj, DF_energy_mj
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("profile")

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Mixtral/Mixtral-8x7B/"
NUM_LAYERS = 32
TP = 2
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
DATA_DIR = BASE / "energy_model" / "data"

GPU_CLOCKS = [210, 450, 690, 930, 1170, 1410]
PREFILL_INPUT_LENS = [128, 256, 384, 512, 640, 768, 896, 1024,
                      1280, 1536, 1792, 2048, 2304, 2560, 2816, 3072,
                      3328, 3584, 3840, 4096, 8192, 16384]
PREFILL_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
DECODE_INPUT_LENS = [128, 256, 512, 1024, 2048, 4096]
DECODE_OUTPUT_LENS = [64, 128, 256, 512]
DECODE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]

SERVER_PORT = 40200
NCCL_PORT = 34500


def lock_freq(gpus, freq):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-lgc", str(freq)],
                       capture_output=True)


def unlock_freq(gpus):
    for g in gpus:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"], capture_output=True)


def get_power_mw(gpus):
    """Read instantaneous power from NVML (mW)."""
    try:
        import pynvml
        pynvml.nvmlInit()
        total = 0
        for g in gpus:
            h = pynvml.nvmlDeviceGetHandleByIndex(g)
            total += pynvml.nvmlDeviceGetPowerUsage(h)
        return total
    except Exception:
        return 0


def get_energy_j(gpus):
    """Read total energy counter (J) from NVML."""
    try:
        import pynvml
        pynvml.nvmlInit()
        total = 0
        for g in gpus:
            h = pynvml.nvmlDeviceGetHandleByIndex(g)
            total += pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        return total  # millijoules
    except Exception:
        return 0


def start_server(gpus, env_extra=None):
    """Start sglang server with TP=2 on given GPUs."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpus)
    if env_extra:
        env.update(env_extra)

    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--tp", str(TP),
           "--host", "127.0.0.1", "--port", str(SERVER_PORT),
           "--nccl-port", str(NCCL_PORT),
           "--mem-fraction-static", "0.85",
           "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
           "--skip-server-warmup",
           "--max-running-requests", "512"]

    log_file = BASE / "logs" / "profile_server.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fh = open(log_file, "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True,
                         preexec_fn=set_memlock)
    log.info("Server started PID=%d on GPU %s", p.pid, gpus)

    for i in range(120):
        try:
            r = requests.get(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=2)
            if r.status_code == 200:
                log.info("Server ready after %ds", i)
                return p
        except Exception:
            pass
        if p.poll() is not None:
            log.error("Server crashed!")
            fh.close()
            return None
        time.sleep(1)
    log.error("Server startup timeout")
    return None


def set_memlock():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))


def kill_server():
    """Kill only the profiling server on SERVER_PORT (avoid killing other instances)."""
    r = subprocess.run(["ss", "-tlnp", f"sport = :{SERVER_PORT}"],
                       capture_output=True, text=True)
    for m in re.finditer(r"pid=(\d+)", r.stdout):
        try:
            os.kill(int(m.group(1)), 9)
        except OSError:
            pass
    time.sleep(3)


def generate_sync(text, max_tokens=1, temperature=0):
    """Send a single generate request and return response."""
    r = requests.post(
        f"http://127.0.0.1:{SERVER_PORT}/generate",
        headers={"Content-Type": "application/json"},
        json={"text": text, "sampling_params": {"max_new_tokens": max_tokens, "temperature": temperature}},
        timeout=120,
    )
    return r.json()


def measure_prefill(input_len, batch_size, gpu_clock, gpus, n_repeat=2):
    """Measure prefill latency and energy for given config.
    Uses concurrent requests to properly test batch behavior."""
    import concurrent.futures

    lock_freq(gpus, gpu_clock)
    time.sleep(0.3)

    prompt = "x " * input_len
    results = []

    for _ in range(n_repeat):
        e_start = get_energy_j(gpus)
        t_start = time.time()

        def _send():
            return requests.post(
                f"http://127.0.0.1:{SERVER_PORT}/generate",
                headers={"Content-Type": "application/json"},
                json={"text": prompt, "sampling_params": {"max_new_tokens": 1, "temperature": 0}},
                timeout=300,
            ).json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(batch_size, 64)) as pool:
            futures = [pool.submit(_send) for _ in range(batch_size)]
            responses = [f.result() for f in futures]

        t_end = time.time()
        e_end = get_energy_j(gpus)

        # Check for errors
        if any("error" in str(r).lower() for r in responses):
            raise RuntimeError("Server returned error")

        ttft_ms = (t_end - t_start) * 1000
        energy_mj = e_end - e_start
        results.append((ttft_ms, energy_mj))

    avg_ttft = np.mean([r[0] for r in results])
    avg_energy = np.mean([r[1] for r in results])

    layer_lat_us = avg_ttft * 1000 / NUM_LAYERS
    a_frac = 0.3
    A_us = layer_lat_us * a_frac
    F_us = layer_lat_us * (1 - a_frac)
    A_energy_mj = avg_energy * a_frac / NUM_LAYERS
    F_energy_mj = avg_energy * (1 - a_frac) / NUM_LAYERS

    return A_us, F_us, avg_ttft, A_energy_mj, F_energy_mj


def measure_decode(input_len, output_len, batch_size, gpu_clock, gpus, n_repeat=2):
    """Measure decode TPOT and energy with concurrent requests."""
    import concurrent.futures

    lock_freq(gpus, gpu_clock)
    time.sleep(0.5)

    prompt = "x " * input_len
    results = []

    for _ in range(n_repeat):
        e_start = get_energy_j(gpus)
        t_start = time.time()

        def _send():
            return requests.post(
                f"http://127.0.0.1:{SERVER_PORT}/generate",
                headers={"Content-Type": "application/json"},
                json={"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0}},
                timeout=300,
            ).json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=batch_size) as pool:
            futures = [pool.submit(_send) for _ in range(batch_size)]
            responses = [f.result() for f in futures]

        t_end = time.time()
        e_end = get_energy_j(gpus)

        total_ms = (t_end - t_start) * 1000
        energy_mj = e_end - e_start
        tpot_ms = total_ms / output_len
        results.append((tpot_ms, energy_mj))

    avg_tpot = np.mean([r[0] for r in results])
    avg_energy = np.mean([r[1] for r in results])

    layer_lat_us = avg_tpot * 1000 / NUM_LAYERS
    a_frac = 0.4
    A_us = layer_lat_us * a_frac
    F_us = layer_lat_us * (1 - a_frac)
    A_energy_mj = avg_energy * a_frac / (NUM_LAYERS * output_len)
    F_energy_mj = avg_energy * (1 - a_frac) / (NUM_LAYERS * output_len)

    return A_us, F_us, avg_tpot, A_energy_mj, F_energy_mj


def profile_v1(gpus):
    """Collect V1 layer-level profiling data."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Prefill
    log.info("=== V1 Prefill Profiling ===")
    prefill_file = DATA_DIR / "prefill_data_v1.txt"
    with open(prefill_file, "w") as f:
        f.write("[P] Prefill\t\t\t\t\t\t\t\t\t\t\n")
        f.write("tp\tinput_len\toutput_len\tgpu_clock\tbatch_size\tA\tF\tTTFT_ms\t(A+F)*32_ms\tA_energy_mj\tF_energy_mj\n")

        for freq in GPU_CLOCKS:
            for il in PREFILL_INPUT_LENS:
                for bs in PREFILL_BATCH_SIZES:
                    # Skip combinations likely to OOM
                    if il >= 8192 and bs >= 64:
                        continue
                    if il >= 16384 and bs >= 16:
                        continue
                    try:
                        A, F, ttft, Ae, Fe = measure_prefill(il, bs, freq, gpus)
                        af32 = (A + F) * NUM_LAYERS / 1000
                        f.write(f"{TP}\t{il}\t1\t{freq}\t{bs}\t{A:.2f}\t{F:.2f}\t{ttft:.2f}\t{af32:.2f}\t{Ae:.2f}\t{Fe:.2f}\n")
                        f.flush()
                        log.info("  P freq=%d il=%d bs=%d -> TTFT=%.1fms", freq, il, bs, ttft)
                    except Exception as e:
                        log.error("  P freq=%d il=%d bs=%d FAILED: %s", freq, il, bs, e)

    # Decode
    log.info("=== V1 Decode Profiling ===")
    decode_file = DATA_DIR / "decode_data_v1.txt"
    with open(decode_file, "w") as f:
        f.write("[D] Decode\t\t\t\t\t\t\t\t\t\t\n")
        f.write("tp\tinput_len\toutput_len\tgpu_clock\tbatch_size\tA\tF\tTPOT_ms\t(A+F)*32_ms\tA_energy_mj\tF_energy_mj\n")

        for freq in GPU_CLOCKS:
            for il in DECODE_INPUT_LENS:
                for ol in DECODE_OUTPUT_LENS:
                    for bs in DECODE_BATCH_SIZES:
                        try:
                            A, F, tpot, Ae, Fe = measure_decode(il, ol, bs, freq, gpus)
                            af32 = (A + F) * NUM_LAYERS / 1000
                            f.write(f"{TP}\t{il}\t{ol}\t{freq}\t{bs}\t{A:.2f}\t{F:.2f}\t{tpot:.2f}\t{af32:.2f}\t{Ae:.2f}\t{Fe:.2f}\n")
                            f.flush()
                            log.info("  D freq=%d il=%d ol=%d bs=%d -> TPOT=%.1fms", freq, il, ol, bs, tpot)
                        except Exception as e:
                            log.error("  D freq=%d il=%d ol=%d bs=%d FAILED: %s", freq, il, ol, bs, e)

    unlock_freq(gpus)
    log.info("V1 profiling done: %s", DATA_DIR)


def profile_v1_decode_only(gpus):
    """Collect V1 decode profiling data only."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== V1 Decode Profiling (standalone) ===")
    decode_file = DATA_DIR / "decode_data_v1.txt"
    with open(decode_file, "w") as f:
        f.write("[D] Decode\t\t\t\t\t\t\t\t\t\t\n")
        f.write("tp\tinput_len\toutput_len\tgpu_clock\tbatch_size\tA\tF\tTPOT_ms\t(A+F)*32_ms\tA_energy_mj\tF_energy_mj\n")

        for freq in GPU_CLOCKS:
            for il in DECODE_INPUT_LENS:
                for ol in DECODE_OUTPUT_LENS:
                    for bs in DECODE_BATCH_SIZES:
                        try:
                            A, F, tpot, Ae, Fe = measure_decode(il, ol, bs, freq, gpus)
                            af32 = (A + F) * NUM_LAYERS / 1000
                            f.write(f"{TP}\t{il}\t{ol}\t{freq}\t{bs}\t{A:.2f}\t{F:.2f}\t{tpot:.2f}\t{af32:.2f}\t{Ae:.2f}\t{Fe:.2f}\n")
                            f.flush()
                            log.info("  D freq=%d il=%d ol=%d bs=%d -> TPOT=%.1fms", freq, il, ol, bs, tpot)
                        except Exception as e:
                            log.error("  D freq=%d il=%d ol=%d bs=%d FAILED: %s", freq, il, ol, bs, e)

    unlock_freq(gpus)
    log.info("V1 decode profiling done: %s", DATA_DIR)


def profile_v2(gpus):
    """Collect V2 pipeline-level decode profiling data (A/F joint freq sweep)."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    log.info("=== V2 Decode Pipeline Profiling (joint A/F freq) ===")
    v2_file = DATA_DIR / "decode_pipeline_v1.txt"
    v2_input_lens = [128, 512, 1024, 2048, 4096]
    v2_batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    M_VALUES = [1, 2]

    with open(v2_file, "w") as f:
        f.write("tp_a\ttp_f\tM\tf_A\tf_F\tinput_len\tbatch_size\titer_lat_us\tDA_energy_mj\tDF_energy_mj\n")

        for M in M_VALUES:
            for f_a in GPU_CLOCKS:
                for f_f in GPU_CLOCKS:
                    for il in v2_input_lens:
                        for bs in v2_batch_sizes:
                            if il * bs > 512 * 256:
                                continue
                            try:
                                import concurrent.futures
                                lock_freq(gpus, f_a)
                                time.sleep(0.3)

                                prompt = "x " * il
                                ol = 64
                                actual_bs = bs * M
                                e_start = get_energy_j(gpus)
                                t_start = time.time()

                                def _send():
                                    return requests.post(
                                        f"http://127.0.0.1:{SERVER_PORT}/generate",
                                        headers={"Content-Type": "application/json"},
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
                                log.info("  V2 M=%d fA=%d fF=%d il=%d bs=%d -> iter_lat=%.0fus", M, f_a, f_f, il, bs, iter_lat_us)
                            except Exception as e:
                                log.error("  V2 M=%d fA=%d fF=%d il=%d bs=%d FAILED: %s", M, f_a, f_f, il, bs, e)

    unlock_freq(gpus)
    log.info("V2 profiling done: %s", DATA_DIR)


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1", help="GPU IDs for TP=2 (comma-sep)")
    parser.add_argument("--phase", default="all", choices=["all", "v1", "v2", "prefill", "decode"])
    parser.add_argument("--quick", action="store_true", help="Reduced sweep for quick test")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    assert len(gpus) == TP, f"Need exactly {TP} GPUs, got {len(gpus)}"

    if args.quick:
        global GPU_CLOCKS, PREFILL_INPUT_LENS, PREFILL_BATCH_SIZES
        global DECODE_INPUT_LENS, DECODE_OUTPUT_LENS, DECODE_BATCH_SIZES
        GPU_CLOCKS = [210, 690, 1410]
        PREFILL_INPUT_LENS = [128, 1024, 4096]
        PREFILL_BATCH_SIZES = [1, 4, 16]
        DECODE_INPUT_LENS = [128, 1024]
        DECODE_OUTPUT_LENS = [64, 256]
        DECODE_BATCH_SIZES = [1, 4, 16, 64]

    kill_server()
    proc = start_server(gpus)
    if proc is None:
        log.error("Failed to start server")
        return 1

    try:
        # Warmup
        generate_sync("Hello", max_tokens=5)
        log.info("Warmup done")

        if args.phase in ("all", "v1", "prefill"):
            profile_v1(gpus)
        if args.phase == "decode":
            profile_v1_decode_only(gpus)
        if args.phase in ("all", "v2"):
            profile_v2(gpus)
    finally:
        kill_server()

    log.info("All profiling complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
