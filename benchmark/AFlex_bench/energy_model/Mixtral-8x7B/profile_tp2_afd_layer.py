#!/usr/bin/env python3
"""Profile Mixtral-8x7B TP2 AFD (A2/F2 same-node, per-rank IPC) layer-level latency & energy.

Runs from node3 host. Launches A2(GPU2,3) + F2(GPU0,1) inside container via docker exec.
Uses SGLANG_LAYER_PROFILE=2 to get per-forward AFD_FASTPATH_PROFILE log lines,
and NVML totalEnergyConsumption for energy measurement.

Output format matches data/v1_layer_profile/prefill_data_v1.txt and decode_data_v1.txt.

Usage:
    python3 profile_tp2_afd_layer.py --phase all --output-dir ./data/v1_layer_profile
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import logging
import os
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("profile_tp2_afd")

# ── Environment ──
NODE3_IP = os.environ.get("NODE3_IP", "10.252.129.34")
CONTAINER = os.environ.get("CONTAINER", "operator_test")
PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"
LOG_DIR = "/workspace/sglang/benchmark/AFlex_bench/energy_model/Mixtral-8x7B/logs"

TP = 2
NUM_LAYERS = 32

# GPU layout: F on GPU0,1; A on GPU2,3
F_GPUS = [0, 1]
A_GPUS = [2, 3]
ALL_GPUS = F_GPUS + A_GPUS

# Ports
F_PORT = 40100
A_PORT = 40101
UCX_BASE_PORT_F = 28100
UCX_BASE_PORT_A = 28200
SCHED_PORT_F = 68100
SCHED_PORT_A = 68200

# Sweep parameters
GPU_CLOCKS = [450, 690, 930, 1170, 1410]
PREFILL_INPUT_LENS = [128, 256, 512, 1024, 2048, 3072, 4096]
PREFILL_BATCH_SIZES = [1, 2, 4, 8, 16, 32]
DECODE_INPUT_LENS = [128, 512, 1024, 2048, 4096]
DECODE_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128]
DECODE_OUTPUT_LEN = 64

HEALTH_TIMEOUT = 600


# ═══════════════════════════════════════════════════════════════════════════════
# Docker / host helpers
# ═══════════════════════════════════════════════════════════════════════════════

def dexec(cmd: str, capture=False, timeout=60):
    """Execute shell command inside node3 container."""
    full = ["docker", "exec", CONTAINER, "bash", "-lc", cmd]
    if capture:
        r = subprocess.run(full, capture_output=True, text=True, timeout=timeout)
        return r.stdout, r.stderr, r.returncode
    subprocess.run(full, check=False, timeout=timeout)


def dexec_bg(cmd: str):
    """Execute command in background inside container (returns immediately)."""
    full_cmd = f"{cmd} &"
    dexec(full_cmd)


def lock_freq(freq: int):
    """Lock all 4 GPUs to given frequency inside container."""
    cmds = ";".join(
        f"nvidia-smi -i {g} --lock-gpu-clocks={freq},{freq}" for g in ALL_GPUS
    )
    dexec(cmds + ";true")
    log.info("Locked GPU %s to %d MHz", ALL_GPUS, freq)


def unlock_freq():
    """Reset GPU clocks."""
    cmds = ";".join(f"nvidia-smi -i {g} --reset-gpu-clocks" for g in ALL_GPUS)
    dexec(cmds + ";true")


def get_energy() -> dict[int, int]:
    """Read NVML totalEnergyConsumption (mJ) for GPU0-3 inside container."""
    idx_csv = ",".join(str(g) for g in ALL_GPUS)
    pycode = (
        "import pynvml,json;pynvml.nvmlInit();"
        f"idxs=[int(x) for x in '{idx_csv}'.split(',')];"
        "print(json.dumps({{i:pynvml.nvmlDeviceGetTotalEnergyConsumption("
        "pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idxs}}));"
        "pynvml.nvmlShutdown()"
    )
    out, _, rc = dexec(f"{PYTHON} -c {shlex.quote(pycode)}", capture=True, timeout=30)
    lines = [l for l in out.strip().splitlines() if l.startswith("{")]
    if lines:
        return {int(k): v for k, v in json.loads(lines[-1]).items()}
    return {g: 0 for g in ALL_GPUS}


def energy_diff_mj(before: dict, after: dict, gpus: list[int]) -> float:
    """Compute energy difference in millijoules for given GPU set."""
    return sum(after.get(g, 0) - before.get(g, 0) for g in gpus)


# ═══════════════════════════════════════════════════════════════════════════════
# Server lifecycle
# ═══════════════════════════════════════════════════════════════════════════════

def kill_servers():
    """Kill any existing sglang processes inside container."""
    dexec("pkill -9 -f 'sglang.launch_server' || true; sleep 2")
    time.sleep(3)


def start_af_pair():
    """Start AFD A2+F2 pair inside container. F on GPU0,1; A on GPU2,3."""
    kill_servers()
    dexec(f"mkdir -p {LOG_DIR}")

    # Common flags
    common = (
        f"--model-path {MODEL} --tp {TP} "
        "--afd-comm-backend ipc_cpp --afd-micro-batch 1 "
        "--mem-fraction-static 0.85 --max-running-requests 512 "
        "--skip-server-warmup --watchdog-timeout 600 "
        "--disable-cuda-graph --disable-piecewise-cuda-graph "
        "--disable-radix-cache "
    )

    # FFN server (GPU0,1) — must start first
    f_env = (
        "export CUDA_VISIBLE_DEVICES=0,1,2,3 "
        "SGLANG_DISABLE_REQUEST_LOGGING=true "
        f"SGLANG_LAYER_PROFILE=2 "
        f"AFD_UCX_BASE_PORT={UCX_BASE_PORT_F} AFD_SCHED_PORT={SCHED_PORT_F} "
        f"AFD_IPC_PEER_OFFSET=2 "
        f"AFD_NVML_DEVICE_INDICES=0,1 AFD_NVML_DEVICE_INDEX=0 "
        "AFD_IPC_SYNC_MODE=ipc_event "
        "AFD_ASYNC_PIPELINE=1;"
    )
    f_cmd = (
        f"{f_env} setsid {PYTHON} -m sglang.launch_server "
        f"--host 0.0.0.0 --port {F_PORT} --base-gpu-id 0 "
        f"--afd-perspective ffn {common}"
        f"> {LOG_DIR}/ffn.log 2>&1"
    )
    dexec_bg(f_cmd)
    log.info("FFN server launching on GPU 0,1...")
    time.sleep(8)

    # ATTN server (GPU2,3)
    a_env = (
        "export CUDA_VISIBLE_DEVICES=0,1,2,3 "
        "SGLANG_DISABLE_REQUEST_LOGGING=true "
        f"SGLANG_LAYER_PROFILE=2 "
        f"AFD_UCX_BASE_PORT={UCX_BASE_PORT_A} AFD_SCHED_PORT={SCHED_PORT_A} "
        f"AFD_IPC_PEER_OFFSET=-2 "
        f"AFD_NVML_DEVICE_INDICES=2,3 AFD_NVML_DEVICE_INDEX=2 "
        "AFD_IPC_SYNC_MODE=ipc_event "
        "AFD_ASYNC_PIPELINE=1 "
        "AFD_UCX_FFN_HOST=127.0.0.1;"
    )
    a_cmd = (
        f"{a_env} setsid {PYTHON} -m sglang.launch_server "
        f"--host 0.0.0.0 --port {A_PORT} --base-gpu-id 2 "
        f"--afd-perspective attn {common}"
        f"> {LOG_DIR}/attn.log 2>&1"
    )
    dexec_bg(a_cmd)
    log.info("ATTN server launching on GPU 2,3...")

    # Wait for health
    for name, port in [("FFN", F_PORT), ("ATTN", A_PORT)]:
        if not wait_health(port):
            log.error("%s server failed to start on port %d", name, port)
            return False
        log.info("  %s ready (port %d)", name, port)
    return True


def wait_health(port: int, timeout: int = HEALTH_TIMEOUT) -> bool:
    """Wait for server health endpoint via container's exposed port."""
    url = f"http://{NODE3_IP}:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(url, timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# Request & log parsing
# ═══════════════════════════════════════════════════════════════════════════════

def send_request(input_len: int, max_new_tokens: int, timeout: int = 300):
    """Send a single generate request to the ATTN server (entry point)."""
    payload = {
        "input_ids": [1000] * input_len,
        "sampling_params": {
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    url = f"http://{NODE3_IP}:{A_PORT}/generate"
    return requests.post(url, json=payload, timeout=timeout).json()


def send_batch(input_len: int, max_new_tokens: int, bs: int):
    """Send bs concurrent requests."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=bs) as pool:
        futs = [
            pool.submit(send_request, input_len, max_new_tokens)
            for _ in range(bs)
        ]
        return [f.result() for f in futs]


def parse_profile_log(log_path: str, last_n: int = 5) -> tuple[float, float] | None:
    """Parse last N AFD_FASTPATH_PROFILE lines from ATTN log, return avg (A_us, F_us).

    Log format: [AFD_FASTPATH_PROFILE] layers=32 bs=N A_total=XXms F_total=YYms ...
    Returns per-layer A and F in microseconds.
    """
    cmd = f"grep 'AFD_FASTPATH_PROFILE' {log_path} | tail -n {last_n}"
    out, _, rc = dexec(cmd, capture=True, timeout=15)
    if not out.strip():
        return None

    pattern = re.compile(
        r"AFD_FASTPATH_PROFILE\] layers=(\d+) bs=\d+ "
        r"A_total=([\d.]+)ms F_total=([\d.]+)ms"
    )
    a_totals, f_totals, layer_counts = [], [], []
    for line in out.strip().splitlines():
        m = pattern.search(line)
        if m:
            n_layers = int(m.group(1))
            a_totals.append(float(m.group(2)))
            f_totals.append(float(m.group(3)))
            layer_counts.append(n_layers)

    if not a_totals:
        return None

    avg_a_total_ms = sum(a_totals) / len(a_totals)
    avg_f_total_ms = sum(f_totals) / len(f_totals)
    n = layer_counts[0] if layer_counts else NUM_LAYERS
    a_per_layer_us = avg_a_total_ms * 1000.0 / n
    f_per_layer_us = avg_f_total_ms * 1000.0 / n
    return a_per_layer_us, f_per_layer_us


def clear_log():
    """Truncate ATTN log to avoid stale profile lines."""
    dexec(f"truncate -s 0 {LOG_DIR}/attn.log 2>/dev/null || true")


# ═══════════════════════════════════════════════════════════════════════════════
# Profiling loops
# ═══════════════════════════════════════════════════════════════════════════════

def warmup():
    """Send a warmup request to populate caches."""
    try:
        send_request(128, 2)
        time.sleep(1)
    except Exception as e:
        log.warning("Warmup failed: %s", e)


def profile_prefill(output_dir: Path):
    """Profile prefill phase: sweep freq x input_len x batch_size."""
    out_file = output_dir / "prefill_data_v1.txt"
    log.info("=== Prefill Profiling -> %s ===", out_file)

    with open(out_file, "w") as f:
        f.write("[P] Prefill\t\t\t\t\t\t\t\t\t\t\n")
        f.write("tp\tinput_len\toutput_len\tgpu_clock\tbatch_size\t"
                "A\tF\tTTFT_ms\t(A+F)*32_ms\tA_energy_mj\tF_energy_mj\n")

        for freq in GPU_CLOCKS:
            lock_freq(freq)
            time.sleep(1)
            warmup()

            for il in PREFILL_INPUT_LENS:
                for bs in PREFILL_BATCH_SIZES:
                    if il * bs > 4096 * 32:
                        continue
                    try:
                        clear_log()
                        time.sleep(0.3)

                        e_before = get_energy()
                        t0 = time.time()
                        send_batch(il, 1, bs)
                        t1 = time.time()
                        e_after = get_energy()

                        time.sleep(0.5)
                        result = parse_profile_log(
                            f"{LOG_DIR}/attn.log", last_n=3
                        )
                        if result is None:
                            log.warning("  P freq=%d il=%d bs=%d: no profile",
                                        freq, il, bs)
                            continue

                        a_us, f_us = result
                        ttft_ms = (t1 - t0) * 1000.0
                        af_32_ms = (a_us + f_us) * NUM_LAYERS / 1000.0
                        a_energy = energy_diff_mj(e_before, e_after, A_GPUS)
                        f_energy = energy_diff_mj(e_before, e_after, F_GPUS)

                        f.write(
                            f"{TP}\t{il}\t1\t{freq}\t{bs}\t"
                            f"{a_us:.2f}\t{f_us:.2f}\t{ttft_ms:.2f}\t"
                            f"{af_32_ms:.2f}\t{a_energy:.2f}\t{f_energy:.2f}\n"
                        )
                        f.flush()
                        log.info("  P freq=%d il=%d bs=%d -> A=%.0fus F=%.0fus "
                                 "TTFT=%.1fms", freq, il, bs, a_us, f_us, ttft_ms)
                    except Exception as e:
                        log.error("  P freq=%d il=%d bs=%d FAILED: %s",
                                  freq, il, bs, e)

    log.info("Prefill profiling complete: %s", out_file)


def profile_decode(output_dir: Path):
    """Profile decode phase: sweep freq x input_len x batch_size."""
    out_file = output_dir / "decode_data_v1.txt"
    log.info("=== Decode Profiling -> %s ===", out_file)

    with open(out_file, "w") as f:
        f.write("[D] Decode\t\t\t\t\t\t\t\t\t\t\n")
        f.write("tp\tinput_len\toutput_len\tgpu_clock\tbatch_size\t"
                "A\tF\tTPOT_ms\t(A+F)*32_ms\tA_energy_mj\tF_energy_mj\n")

        for freq in GPU_CLOCKS:
            lock_freq(freq)
            time.sleep(1)
            warmup()

            for il in DECODE_INPUT_LENS:
                for bs in DECODE_BATCH_SIZES:
                    try:
                        clear_log()
                        time.sleep(0.3)

                        e_before = get_energy()
                        t0 = time.time()
                        send_batch(il, DECODE_OUTPUT_LEN, bs)
                        t1 = time.time()
                        e_after = get_energy()

                        time.sleep(0.5)
                        result = parse_profile_log(
                            f"{LOG_DIR}/attn.log", last_n=10
                        )
                        if result is None:
                            log.warning("  D freq=%d il=%d bs=%d: no profile",
                                        freq, il, bs)
                            continue

                        a_us, f_us = result
                        total_ms = (t1 - t0) * 1000.0
                        tpot_ms = total_ms / DECODE_OUTPUT_LEN
                        af_32_ms = (a_us + f_us) * NUM_LAYERS / 1000.0
                        a_energy_total = energy_diff_mj(e_before, e_after, A_GPUS)
                        f_energy_total = energy_diff_mj(e_before, e_after, F_GPUS)
                        a_energy_per_tok = a_energy_total / DECODE_OUTPUT_LEN
                        f_energy_per_tok = f_energy_total / DECODE_OUTPUT_LEN

                        f.write(
                            f"{TP}\t{il}\t{DECODE_OUTPUT_LEN}\t{freq}\t{bs}\t"
                            f"{a_us:.2f}\t{f_us:.2f}\t{tpot_ms:.2f}\t"
                            f"{af_32_ms:.2f}\t{a_energy_per_tok:.2f}\t"
                            f"{f_energy_per_tok:.2f}\n"
                        )
                        f.flush()
                        log.info("  D freq=%d il=%d bs=%d -> A=%.0fus F=%.0fus "
                                 "TPOT=%.1fms", freq, il, bs, a_us, f_us, tpot_ms)
                    except Exception as e:
                        log.error("  D freq=%d il=%d bs=%d FAILED: %s",
                                  freq, il, bs, e)

    log.info("Decode profiling complete: %s", out_file)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Profile Mixtral TP2 AFD layer latency & energy on node3"
    )
    parser.add_argument("--phase", default="all", choices=["all", "prefill", "decode"])
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for TSV files (default: data/v1_layer_profile)")
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parent / "data" / "v1_layer_profile"
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("Starting TP2 AFD layer profiling on node3 (%s)", NODE3_IP)
    log.info("  Container: %s, Phase: %s, Output: %s", CONTAINER, args.phase, output_dir)

    if not start_af_pair():
        log.error("Failed to start AF pair. Aborting.")
        kill_servers()
        unlock_freq()
        sys.exit(1)

    try:
        warmup()
        log.info("Initial warmup done")

        if args.phase in ("all", "prefill"):
            profile_prefill(output_dir)

        if args.phase in ("all", "decode"):
            profile_decode(output_dir)

    finally:
        unlock_freq()
        kill_servers()
        log.info("Cleanup done. Profiling finished.")


if __name__ == "__main__":
    main()
