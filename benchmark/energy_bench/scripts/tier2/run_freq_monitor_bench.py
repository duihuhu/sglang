#!/usr/bin/env python3
"""Run Tier2 DVFS benchmark with real-time GPU frequency monitoring.

Launches the server with DVFS enabled, runs workload, and simultaneously
samples GPU frequency every 200ms. Produces a time-series plot showing
frequency transitions on all 4 GPUs.

Usage:
    /workspace/env/sglang-tier/bin/python run_freq_monitor_bench.py \
        --workload workloads/workload_heavy.jsonl \
        --tpot-slo-us 300000 \
        --output-dir results/freq_monitor
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("freq_monitor")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = Path(__file__).resolve().parent
ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/test_motivation/energy_models"

GPU_PA, GPU_PF, GPU_DA, GPU_DF = 7, 6, 5, 4
GPU_INDICES = [GPU_DF, GPU_DA, GPU_PF, GPU_PA]
GPU_ROLES = {4: "DF", 5: "DA", 6: "PF", 7: "PA"}

PA_PORT, PF_PORT = 51010, 51011
DA_PORT, DF_PORT = 51020, 51021
ROUTER_PORT = 51000
BOOTSTRAP_PORT = 29999
UCX_P, UCX_D = 26200, 26300
SCHED_P, SCHED_D = 66400, 66500
ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT]


# ── GPU Monitoring Thread ────────────────────────────────────────────────

class FreqMonitor:
    """Background thread that samples GPU frequency and power."""

    def __init__(self, gpu_indices, interval_s=0.2):
        self.gpu_indices = gpu_indices
        self.interval_s = interval_s
        self.samples = []  # list of (time_s, {gpu_idx: freq_mhz}, {gpu_idx: power_w})
        self._stop = threading.Event()
        self._thread = None
        self._start_time = 0.0

    def start(self):
        self._start_time = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        try:
            import pynvml
            pynvml.nvmlInit()
            handles = {idx: pynvml.nvmlDeviceGetHandleByIndex(idx) for idx in self.gpu_indices}

            while not self._stop.is_set():
                t = time.monotonic() - self._start_time
                freqs = {}
                powers = {}
                for idx, handle in handles.items():
                    freqs[idx] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
                    powers[idx] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW → W
                self.samples.append((t, freqs, powers))
                time.sleep(self.interval_s)

            pynvml.nvmlShutdown()
        except Exception as e:
            log.warning("FreqMonitor error: %s", e)

    def to_csv(self, path):
        with open(path, 'w') as f:
            header = "time_s," + ",".join(f"freq_gpu{i}" for i in self.gpu_indices) + "," + \
                     ",".join(f"power_gpu{i}" for i in self.gpu_indices)
            f.write(header + "\n")
            for t, freqs, powers in self.samples:
                row = f"{t:.3f}," + ",".join(str(freqs.get(i, 0)) for i in self.gpu_indices) + "," + \
                      ",".join(f"{powers.get(i, 0):.1f}" for i in self.gpu_indices)
                f.write(row + "\n")


# ── Server Utils ─────────────────────────────────────────────────────────

def _get_pids_on_port(port):
    result = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
    return list(set(re.findall(r'pid=(\d+)', result.stdout)))

def reset_gpu_clocks():
    for idx in GPU_INDICES:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)

def kill_servers():
    for port in ALL_PORTS:
        for pid in _get_pids_on_port(port):
            subprocess.run(["kill", "-9", pid], capture_output=True)
    time.sleep(5)
    reset_gpu_clocks()

def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2); s.close(); return True
        except:
            time.sleep(2)
    return False

def wait_health(url, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5); return True
        except:
            time.sleep(3)
    return False

def warmup(url, n=8):
    import requests, concurrent.futures
    payloads = [{"text": f"Hello {i}", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}} for i in range(n)]
    def _send(p):
        try: requests.post(url + "/generate", json=p, timeout=120)
        except: pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_send, payloads))
    time.sleep(3)


def start_server(ttft_slo_ms, tpot_slo_us, dvfs_enabled, label):
    kill_servers()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env_base['SGLANG_DISAGGREGATION_THREAD_POOL_SIZE'] = '128'
    env_base['SGLANG_DISAGGREGATION_QUEUE_SIZE'] = '32'
    env_base['SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE'] = '0'

    extra = ['--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
             '--afd-micro-batch', '1', '--afd-disagg-interleave-poll',
             '--disable-radix-cache', '--num-reserved-decode-tokens', '32']

    dvfs_args = []
    if dvfs_enabled:
        dvfs_args = ['--afd-dvfs-enabled', '--afd-energy-model-dir', ENERGY_MODEL_DIR,
                     '--afd-ttft-slo-ms', str(ttft_slo_ms), '--afd-tpot-slo-us', str(tpot_slo_us)]

    def _start(name, gpu, perspective, disagg_mode, port, ucx_base, sched_port,
               ffn_host=None, visible_gpus=None, base_gpu_id=0):
        env = env_base.copy()
        env['CUDA_VISIBLE_DEVICES'] = visible_gpus if visible_gpus else str(gpu)
        env['AFD_UCX_BASE_PORT'] = str(ucx_base)
        env['AFD_SCHED_PORT'] = str(sched_port)
        env['SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT'] = '600'
        env['SGLANG_DISAGGREGATION_WAITING_TIMEOUT'] = '600'
        env['AFD_NVML_DEVICE_INDEX'] = str(gpu)
        env['AFD_IPC_SYNC_MODE'] = 'ipc_event'
        env['AFD_IPC_PEER_DEVICE'] = str(1 - base_gpu_id)
        if ffn_host:
            env['AFD_UCX_FFN_HOST'] = ffn_host
        cmd = [PYTHON, '-m', 'sglang.launch_server',
            '--model-path', MODEL, '--tp', '1',
            '--host', '127.0.0.1', '--port', str(port),
            '--afd-perspective', perspective, '--afd-comm-backend', 'ipc_cpp',
            '--mem-fraction-static', '0.85', '--base-gpu-id', str(base_gpu_id),
            '--disaggregation-mode', disagg_mode,
            '--disaggregation-transfer-backend', 'mooncake',
            '--disaggregation-bootstrap-port', str(BOOTSTRAP_PORT),
            '--disaggregation-ib-device', 'mlx5_4'] + extra + dvfs_args
        fh = open(HERE / f'{label}_{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    p_vis = f'{GPU_PF},{GPU_PA}'
    d_vis = f'{GPU_DF},{GPU_DA}'

    _start('pf', GPU_PF, 'ffn', 'prefill', PF_PORT, UCX_P, SCHED_P, visible_gpus=p_vis, base_gpu_id=0)
    time.sleep(2)
    _start('pa', GPU_PA, 'attn', 'prefill', PA_PORT, UCX_P, SCHED_P,
           ffn_host='127.0.0.1', visible_gpus=p_vis, base_gpu_id=1)
    if not wait_port('127.0.0.1', PA_PORT) or not wait_port('127.0.0.1', PF_PORT):
        return None

    _start('df', GPU_DF, 'ffn', 'decode', DF_PORT, UCX_D, SCHED_D, visible_gpus=d_vis, base_gpu_id=0)
    time.sleep(2)
    _start('da', GPU_DA, 'attn', 'decode', DA_PORT, UCX_D, SCHED_D,
           ffn_host='127.0.0.1', visible_gpus=d_vis, base_gpu_id=1)
    if not wait_port('127.0.0.1', DA_PORT) or not wait_port('127.0.0.1', DF_PORT):
        return None

    time.sleep(5)
    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', f'http://127.0.0.1:{PA_PORT}', '--decode', f'http://127.0.0.1:{DA_PORT}',
        '--host', '127.0.0.1', '--port', str(ROUTER_PORT)]
    rf = open(HERE / f'{label}_router.log', 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))

    if not wait_health(f'http://127.0.0.1:{ROUTER_PORT}/health', 120):
        return None

    warmup(f'http://127.0.0.1:{ROUTER_PORT}')
    return procs, f'http://127.0.0.1:{ROUTER_PORT}'


# ── Workload Runner ──────────────────────────────────────────────────────

async def run_workload(workload_path, url):
    with open(workload_path) as f:
        reqs = [json.loads(line) for line in f]

    log.info("Running workload: %d requests", len(reqs))
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        base_time = time.monotonic()
        for req in reqs:
            now = time.monotonic() - base_time
            if req["arrival_time_s"] > now:
                await asyncio.sleep(req["arrival_time_s"] - now)
            task = asyncio.create_task(_send(session, url, req))
            tasks.append(task)
        results = await asyncio.gather(*tasks)
    return results


async def _send(session, url, req):
    prompt = "Hello " * (req["input_len"] // 2)
    payload = {"text": prompt, "sampling_params": {"max_new_tokens": req["output_len"], "temperature": 0.0}}
    try:
        async with session.post(f"{url}/generate", json=payload) as resp:
            if resp.status == 200:
                return {"success": True}
    except:
        pass
    return {"success": False}


# ── Plotting ─────────────────────────────────────────────────────────────

def plot_freq_trace(monitor: FreqMonitor, workload_path: str, output_path: str, slo_label: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    if not monitor.samples:
        log.warning("No samples to plot")
        return

    # Load workload phases
    with open(workload_path) as f:
        reqs = [json.loads(line) for line in f]
    phases = {}
    for r in reqs:
        p = r['phase']
        if p not in phases:
            phases[p] = {'start': r['arrival_time_s'], 'end': r['arrival_time_s']}
        phases[p]['end'] = max(phases[p]['end'], r['arrival_time_s'])

    times = [s[0] for s in monitor.samples]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"GPU Frequency & Power Trace (Tier2 DVFS, SLO={slo_label})", fontsize=13, fontweight='bold')

    phase_colors = {"light": "#E8F5E9", "heavy": "#FFEBEE", "burst": "#FCE4EC",
                    "recovery": "#E3F2FD", "burst2": "#FFF3E0", "idle": "#F3E5F5",
                    "medium": "#FFF8E1", "sustained": "#FFEBEE", "cooldown": "#E8F5E9"}

    def add_phases(ax):
        for p, v in phases.items():
            color = phase_colors.get(p, "#F5F5F5")
            ax.axvspan(v['start'], v['end'], alpha=0.25, color=color, label=p)

    # Panel 1: Frequency per GPU
    ax = axes[0]
    add_phases(ax)
    colors = {4: 'blue', 5: 'red', 6: 'green', 7: 'orange'}
    for idx in GPU_INDICES:
        freqs = [s[1].get(idx, 0) for s in monitor.samples]
        ax.plot(times, freqs, color=colors[idx], linewidth=0.8, alpha=0.8,
                label=f"GPU{idx} ({GPU_ROLES[idx]})")
    ax.set_ylabel("SM Frequency (MHz)")
    ax.set_ylim(0, 1500)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_title("Panel 1: GPU SM Frequency (per GPU)")
    ax.grid(True, alpha=0.3)

    # Panel 2: Frequency grouped by role (P vs D)
    ax = axes[1]
    add_phases(ax)
    # Prefill pair average
    p_freqs = [(s[1].get(GPU_PA, 0) + s[1].get(GPU_PF, 0)) / 2 for s in monitor.samples]
    d_freqs = [(s[1].get(GPU_DA, 0) + s[1].get(GPU_DF, 0)) / 2 for s in monitor.samples]
    ax.plot(times, p_freqs, color='green', linewidth=1.2, label='Prefill (PA+PF avg)')
    ax.plot(times, d_freqs, color='red', linewidth=1.2, label='Decode (DA+DF avg)')
    ax.set_ylabel("Avg Frequency (MHz)")
    ax.set_ylim(0, 1500)
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title("Panel 2: Prefill vs Decode Frequency")
    ax.grid(True, alpha=0.3)

    # Panel 3: Power per GPU
    ax = axes[2]
    add_phases(ax)
    for idx in GPU_INDICES:
        powers = [s[2].get(idx, 0) for s in monitor.samples]
        ax.plot(times, powers, color=colors[idx], linewidth=0.8, alpha=0.8,
                label=f"GPU{idx} ({GPU_ROLES[idx]})")
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Time (s)")
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_title("Panel 3: GPU Power Draw")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log.info("Plot saved: %s", output_path)
    plt.close()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--tpot-slo-us", type=int, default=300000)
    parser.add_argument("--ttft-slo-ms", type=int, default=5000)
    parser.add_argument("--output-dir", type=str, default="results/freq_monitor")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    slo_label = f"TPOT≤{args.tpot_slo_us//1000}ms"

    log.info("Starting server with DVFS (SLO: TTFT≤%dms, TPOT≤%dms)...",
             args.ttft_slo_ms, args.tpot_slo_us // 1000)

    ret = start_server(args.ttft_slo_ms, args.tpot_slo_us, dvfs_enabled=True, label="freq_mon")
    if ret is None:
        log.error("Server failed to start")
        return
    procs, url = ret

    try:
        # Start frequency monitor
        monitor = FreqMonitor(GPU_INDICES, interval_s=0.2)
        monitor.start()
        log.info("Frequency monitor started (sampling every 200ms)")

        # Run workload
        results = asyncio.run(run_workload(args.workload, url))
        success = sum(1 for r in results if r["success"])
        log.info("Workload done: %d/%d successful", success, len(results))

        # Wait a bit for tail decode to finish
        time.sleep(10)
        monitor.stop()

        # Save data
        monitor.to_csv(str(out_dir / "freq_trace.csv"))
        log.info("Saved %d samples to freq_trace.csv", len(monitor.samples))

        # Plot
        plot_freq_trace(monitor, args.workload, str(out_dir / "freq_trace.png"), slo_label)

    finally:
        for _, p, f in procs:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except: pass
        kill_servers()


if __name__ == "__main__":
    main()
