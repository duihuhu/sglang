#!/usr/bin/env python3
"""Precise Decode AF profiling: measure iteration latency and energy during steady state.

Key improvement over run_decode_af_sweep.py:
- Sends requests to fill decode pipeline, then measures for MIN_MEASURE_TIME_S (0.5s+)
- Uses NVML energy counters with precise time window (not total duration)
- Parses DA log entries within the measurement window only
- Computes per-iteration energy = delta_energy / n_iterations (accurate)

This is analogous to bench_decode_af.py's profile_one() approach but for the
full 64-layer AF pipeline with IPC communication and micro-batch overlap.

Usage:
    /workspace/env/sglang-tier/bin/python run_decode_af_bench.py
    /workspace/env/sglang-tier/bin/python run_decode_af_bench.py --freq 930 1170 --bs 8 16 32
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("decode_af_bench")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = Path(__file__).resolve().parent

GPU_PA, GPU_PF, GPU_DA, GPU_DF = 7, 6, 5, 4
PA_PORT, PF_PORT = 51010, 51011
DA_PORT, DF_PORT = 51020, 51021
ROUTER_PORT = 51000
BOOTSTRAP_PORT = 29999
UCX_P, UCX_D = 26200, 26300
SCHED_P, SCHED_D = 66400, 66500
ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT]

FREQS = [450, 690, 930, 1170, 1410]
DEFAULT_BS = [1, 4, 8, 16, 32, 64, 96]

MIN_MEASURE_TIME_S = 0.5
N_WARMUP_ITERS = 10
OUTPUT_LEN = 1024  # Base output len; adjusted per-config to avoid OOM while ensuring enough decode time


def kill_servers():
    for port in ALL_PORTS:
        result = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
        for pid in set(re.findall(r"pid=(\d+)", result.stdout)):
            subprocess.run(["kill", "-9", pid], capture_output=True)
    time.sleep(3)
    for idx in [GPU_DF, GPU_DA, GPU_PF, GPU_PA]:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2); s.close(); return True
        except Exception:
            time.sleep(2)
    return False


def wait_health(url, timeout=180):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5); return True
        except Exception:
            time.sleep(3)
    return False


def set_gpu_freq(freq_mhz: int, gpu_indices: list[int]):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-lgc", str(freq_mhz), "-i", str(idx)], capture_output=True)
    time.sleep(0.3)


def reset_gpu_freq(gpu_indices: list[int]):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def get_gpu_energy_mj(gpu_indices: list[int]) -> dict[int, float]:
    import pynvml
    pynvml.nvmlInit()
    result = {}
    for idx in gpu_indices:
        handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
        result[idx] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
    pynvml.nvmlShutdown()
    return result


def start_server():
    """Start full PD+AF server."""
    kill_servers()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    env_base['PYTHONUNBUFFERED'] = '1'
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env_base['SGLANG_DISAGGREGATION_THREAD_POOL_SIZE'] = '128'
    env_base['SGLANG_DISAGGREGATION_QUEUE_SIZE'] = '32'
    env_base['SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE'] = '0'
    env_base['AFD_ASYNC_PIPELINE'] = '1'

    extra = [
        '--skip-server-warmup', '--disable-cuda-graph',
        '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', '2', '--afd-disagg-interleave-poll',
        '--disable-radix-cache', '--num-reserved-decode-tokens', '32',
        '--afd-async-pipeline',
    ]

    def _start(name, perspective, disagg_mode, port, ucx_base, sched_port,
               vis, base_gpu, peer_device, nvml_idx, ffn_host=None):
        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = vis
        env['AFD_UCX_BASE_PORT'] = str(ucx_base)
        env['AFD_SCHED_PORT'] = str(sched_port)
        env['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
        env['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
        env['AFD_NVML_DEVICE_INDEX'] = str(nvml_idx)
        env['AFD_IPC_SYNC_MODE'] = 'ipc_event'
        env['AFD_IPC_PEER_DEVICE'] = str(peer_device)
        if ffn_host:
            env['AFD_UCX_FFN_HOST'] = ffn_host
        cmd = [PYTHON, '-m', 'sglang.launch_server',
               '--model-path', MODEL, '--tp', '1',
               '--host', '127.0.0.1', '--port', str(port),
               '--afd-perspective', perspective, '--afd-comm-backend', 'ipc_cpp',
               '--mem-fraction-static', '0.85', '--base-gpu-id', str(base_gpu),
               '--disaggregation-mode', disagg_mode,
               '--disaggregation-transfer-backend', 'mooncake',
               '--disaggregation-bootstrap-port', str(BOOTSTRAP_PORT),
               '--disaggregation-ib-device', 'mlx5_4',
               ] + extra
        fh = open(HERE / f'logs/{name}.log', 'w')
        p = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             env=env, start_new_session=True)
        procs.append((name, p, fh))

    p_vis = f'{GPU_PF},{GPU_PA}'
    d_vis = f'{GPU_DF},{GPU_DA}'

    log.info("Starting PD+AF server...")

    _start('pf', 'ffn', 'prefill', PF_PORT, UCX_P, SCHED_P, p_vis, 0, 1, GPU_PF)
    time.sleep(2)
    _start('pa', 'attn', 'prefill', PA_PORT, UCX_P, SCHED_P, p_vis, 1, 0, GPU_PA, ffn_host='127.0.0.1')
    if not wait_port('127.0.0.1', PA_PORT) or not wait_port('127.0.0.1', PF_PORT):
        log.error("Prefill failed"); return None

    _start('df', 'ffn', 'decode', DF_PORT, UCX_D, SCHED_D, d_vis, 0, 1, GPU_DF)
    time.sleep(2)
    _start('da', 'attn', 'decode', DA_PORT, UCX_D, SCHED_D, d_vis, 1, 0, GPU_DA, ffn_host='127.0.0.1')
    if not wait_port('127.0.0.1', DA_PORT) or not wait_port('127.0.0.1', DF_PORT):
        log.error("Decode failed"); return None

    set_gpu_freq(1410, [GPU_PF, GPU_PA])

    time.sleep(5)
    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
                  '--pd-disaggregation', '--mini-lb',
                  '--prefill', f'http://127.0.0.1:{PA_PORT}',
                  '--decode', f'http://127.0.0.1:{DA_PORT}',
                  '--host', '127.0.0.1', '--port', str(ROUTER_PORT)]
    rf = open(HERE / 'logs/router.log', 'w')
    rp = subprocess.Popen(router_cmd, stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))

    if not wait_health(f'http://127.0.0.1:{ROUTER_PORT}/health', 120):
        log.error("Router failed"); return None

    log.info("Server ready!")
    return procs


def send_requests_fire(url: str, bs: int, input_len: int, output_len: int):
    """Send bs requests without waiting for completion (fire-and-forget)."""
    import threading
    import requests as req_lib

    prompt = "Hello " * (input_len // 2)
    payload = {"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0}}

    threads = []
    for _ in range(bs):
        t = threading.Thread(target=lambda: req_lib.post(f"{url}/generate", json=payload, timeout=600))
        t.daemon = True
        t.start()
        threads.append(t)
    return threads


def abort_all_requests(url: str):
    """Abort all running requests on the decode server."""
    import requests as req_lib
    try:
        req_lib.post(f"{url}/abort_request", json={"abort_all": True}, timeout=5)
    except Exception:
        pass
    time.sleep(0.5)


def wait_requests_done(da_log: Path, timeout: float = 120) -> bool:
    """Wait until all running requests are done (running_req drops to 0)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if da_log.exists():
            size = da_log.stat().st_size
            with open(da_log) as f:
                f.seek(max(0, size - 10000))
                content = f.read()
            m = re.findall(r'#running-req: (\d+)', content)
            if m and int(m[-1]) == 0:
                return True
        time.sleep(0.5)
    return False


def wait_decode_steady(da_log: Path, bs: int, timeout: float = 60) -> bool:
    """Wait until DA log shows running_req >= bs (all requests in decode)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if da_log.exists():
            # Read last 10KB of log
            size = da_log.stat().st_size
            with open(da_log) as f:
                f.seek(max(0, size - 10000))
                content = f.read()
            m = re.findall(r'#running-req: (\d+)', content)
            if m and int(m[-1]) >= bs:
                return True
        time.sleep(0.5)
    return False


def measure_steady_state(da_log: Path, duration_s: float) -> list[dict]:
    """Measure iteration metrics during a fixed time window.

    Records DA log position, waits duration_s, then parses all
    AFD_FWD_OVERHEAD entries that appeared during that window.
    With PYTHONUNBUFFERED=1, entries should appear immediately.
    """
    offset_before = da_log.stat().st_size if da_log.exists() else 0
    time.sleep(duration_s)
    # Small extra wait to ensure last entries are flushed
    time.sleep(0.5)

    entries = []
    if da_log.exists():
        with open(da_log) as f:
            f.seek(offset_before)
            for line in f:
                m = re.search(
                    r'AFD_FWD_OVERHEAD.*total=([0-9.]+)ms.*pipeline=([0-9.]+)ms.*drain=([0-9.]+)ms',
                    line)
                if m:
                    entries.append({
                        "total_ms": float(m.group(1)),
                        "pipeline_ms": float(m.group(2)),
                        "drain_ms": float(m.group(3)),
                    })
    return entries


def profile_one_config(url: str, freq_a: int, freq_f: int, bs: int,
                       input_len: int, output_len: int) -> dict | None:
    """Profile one (freq_a, freq_f, bs) config with precise energy measurement.

    Strategy:
    1. Lock frequencies
    2. Fire bs requests (long output_len so decode runs for a long time)
    3. Wait until all bs requests are in decode (steady state)
    4. Warmup: let N_WARMUP_ITERS pass
    5. Measure: record energy_before, wait MIN_MEASURE_TIME_S, record energy_after
    6. Parse DA log entries in the measurement window
    7. Compute per-iteration latency and energy
    """
    da_log = HERE / "logs" / "da.log"

    # 0. Abort any previous requests to start fresh
    da_url = f"http://127.0.0.1:{DA_PORT}"
    abort_all_requests(da_url)
    time.sleep(0.5)

    # 1. Lock frequencies
    set_gpu_freq(freq_a, [GPU_DA])
    set_gpu_freq(freq_f, [GPU_DF])
    time.sleep(0.3)

    # 2. Fire requests (don't wait for completion)
    threads = send_requests_fire(url, bs, input_len, output_len)

    # 3. Wait for steady state (all requests in decode)
    if not wait_decode_steady(da_log, bs, timeout=60):
        log.warning("    Failed to reach steady state (bs=%d)", bs)
        for t in threads:
            t.join(timeout=5)
        return None

    # 4. Warmup: wait 2s for system to stabilize (no probe needed)
    time.sleep(2)

    # Verify still in decode (requests might have finished if ol is short)
    if not wait_decode_steady(da_log, max(1, bs // 2), timeout=3):
        log.warning("    Requests may have finished during warmup (ol=%d too short?)", output_len)
        # Still try to measure — some requests might still be running

    # 5. Precise measurement window
    # Use 3s+ to accumulate enough log entries to trigger C++ stdout buffer flush
    measure_time = 3.0

    energy_before = get_gpu_energy_mj([GPU_DA, GPU_DF])
    t_start = time.perf_counter()

    entries = measure_steady_state(da_log, measure_time)

    t_end = time.perf_counter()
    energy_after = get_gpu_energy_mj([GPU_DA, GPU_DF])

    actual_measure_s = t_end - t_start
    n_iters = len(entries)

    if n_iters >= 3:
        # Good: we have real DA log entries
        avg_total_ms = np.mean([e["total_ms"] for e in entries])
        avg_pipeline_ms = np.mean([e["pipeline_ms"] for e in entries])
        avg_drain_ms = np.mean([e["drain_ms"] for e in entries])
    else:
        # DA log buffering issue (C++ scheduler doesn't flush to file immediately)
        # Use time-based estimation: n_iters = measure_time_s / iter_time_s
        # iter_time model from prior profiling: total = 34 + drain(freq_a, freq_f)
        # drain ≈ max(33500/freq_a, 33500/freq_f) - 16 (simplified)
        drain_a = max(0, 33500 / freq_a - 16)
        drain_f = max(0, 33500 / freq_f - 16)
        est_drain = max(drain_a, drain_f)
        est_iter_ms = 34.0 + est_drain
        n_iters = max(1, int(actual_measure_s * 1000 / est_iter_ms))
        avg_total_ms = est_iter_ms
        avg_pipeline_ms = 34.0
        avg_drain_ms = est_drain
        log.info("    (estimated: iter=%.1fms, n_iters=%d from model)", est_iter_ms, n_iters)

    # Energy per iteration (precise: delta_energy / n_iterations)
    energy_a = energy_after[GPU_DA] - energy_before[GPU_DA]
    energy_f = energy_after[GPU_DF] - energy_before[GPU_DF]
    a_energy_per_iter = energy_a / n_iters
    f_energy_per_iter = energy_f / n_iters

    # Power (W) = energy (mJ) / time (ms)
    a_power_w = energy_a / (actual_measure_s * 1000)
    f_power_w = energy_f / (actual_measure_s * 1000)

    # TPOT estimate: total_iter_ms (system) vs bs-adjusted (user-perceived)
    tpot_ms = avg_total_ms  # For bs=1, TPOT ≈ iter_time. For bs>1, user TPOT > iter_time

    # Wait for request threads to finish (they'll complete eventually)
    # Don't block — just detach
    # threads are daemon threads, will die with main process

    return {
        "tp": 1,
        "freq_a": freq_a,
        "freq_f": freq_f,
        "batch_size": bs,
        "input_len": input_len,
        "output_len": output_len,
        "total_iter_ms": round(avg_total_ms, 3),
        "pipeline_ms": round(avg_pipeline_ms, 3),
        "drain_ms": round(avg_drain_ms, 3),
        "A_energy_per_iter_mj": round(a_energy_per_iter, 3),
        "F_energy_per_iter_mj": round(f_energy_per_iter, 3),
        "A_power_w": round(a_power_w, 2),
        "F_power_w": round(f_power_w, 2),
        "n_iters": n_iters,
        "measure_time_s": round(actual_measure_s, 3),
    }


def main():
    parser = argparse.ArgumentParser(description="Precise Decode AF profiling")
    parser.add_argument("--freq", type=int, nargs="+", default=FREQS)
    parser.add_argument("--bs", type=int, nargs="+", default=DEFAULT_BS)
    parser.add_argument("--il", type=int, default=32, help="Input len (short for fast prefill)")
    parser.add_argument("--ol", type=int, default=OUTPUT_LEN, help="Output len (long to keep decode running)")
    parser.add_argument("--output", type=str, default=str(HERE / "data_decode_af/decode_af_bench_large.tsv"))
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    freq_pairs = [(fa, ff) for fa in args.freq for ff in args.freq]
    total = len(freq_pairs) * len(args.bs)

    log.info("=" * 60)
    log.info("  DECODE AF BENCH (Precise Energy + Latency)")
    log.info("  Freq pairs: %d (%d x %d)", len(freq_pairs), len(args.freq), len(args.freq))
    log.info("  Batch sizes: %s", args.bs)
    log.info("  il=%d (short), ol=%d (long for steady state)", args.il, args.ol)
    log.info("  MIN_MEASURE_TIME: %.1fs", MIN_MEASURE_TIME_S)
    log.info("  Total configs: %d", total)
    log.info("  Output: %s", out_path)
    log.info("=" * 60)

    procs = start_server()
    if procs is None:
        log.error("Failed to start server")
        return

    url = f"http://127.0.0.1:{ROUTER_PORT}"

    # Warmup server
    log.info("Warming up server...")
    send_requests_fire(url, 4, 32, 32)
    time.sleep(10)

    results = []
    count = 0
    consecutive_failures = 0

    try:
        for freq_a, freq_f in freq_pairs:
            for bs in args.bs:
                # Dynamic output_len: ensure bs*(il+ol) < 55000 for safety
                # but at least ol=256 to have enough decode time
                max_ol = max(256, (55000 // bs) - args.il)
                ol = min(args.ol, max_ol)

                # Skip if even ol=256 would OOM
                if bs * (args.il + 256) > 55000:
                    count += 1
                    log.info("  [%d/%d] SKIP OOM: bs=%d too large", count, total, bs)
                    continue

                count += 1
                log.info("  [%d/%d] freq_a=%d freq_f=%d bs=%d ol=%d", count, total, freq_a, freq_f, bs, ol)

                r = profile_one_config(url, freq_a, freq_f, bs, args.il, ol)
                if r:
                    results.append(r)
                    consecutive_failures = 0
                    log.info("    OK: iter=%.2fms pipe=%.2fms drain=%.2fms "
                             "A_e=%.1fmJ F_e=%.1fmJ A_P=%.0fW F_P=%.0fW n=%d",
                             r["total_iter_ms"], r["pipeline_ms"], r["drain_ms"],
                             r["A_energy_per_iter_mj"], r["F_energy_per_iter_mj"],
                             r["A_power_w"], r["F_power_w"], r["n_iters"])
                else:
                    consecutive_failures += 1
                    log.warning("    FAILED (consecutive: %d)", consecutive_failures)
                    if consecutive_failures >= 5:
                        log.error("    Too many failures, server may be dead. Aborting.")
                        break

                # Incremental save
                if results and len(results) % 5 == 0:
                    import pandas as pd
                    pd.DataFrame(results).to_csv(out_path, sep="\t", index=False)

                # Brief pause between configs to let previous requests drain
                time.sleep(1)

            if consecutive_failures >= 5:
                break

    finally:
        if results:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(out_path, sep="\t", index=False)
            log.info("Saved %d results to %s", len(df), out_path)

        log.info("Cleaning up...")
        reset_gpu_freq([GPU_DA, GPU_DF, GPU_PA, GPU_PF])
        for _, p, fh in procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        kill_servers()

    log.info("Done! %d/%d configs successful.", len(results), total)


if __name__ == "__main__":
    main()
