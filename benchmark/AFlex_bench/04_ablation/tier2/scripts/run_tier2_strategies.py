#!/usr/bin/env python3
"""Compare Tier2 DVFS strategies with frequency monitoring.

Runs 3 configs on the same workload:
  1. Baseline DVFS (predictor only)
  2. DVFS + Feedback (force step-up on repeated SLO urgent)
  3. DVFS + Online Calibration (correct predictor bias with observed TPOT)

Each run monitors GPU frequency at 200ms intervals and produces a comparison plot.

Usage:
    /workspace/env/sglang-tier/bin/python run_tier2_strategies.py
"""

from __future__ import annotations

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
log = logging.getLogger("tier2_strategies")

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


class FreqMonitor:
    def __init__(self, gpu_indices, interval_s=0.2):
        self.gpu_indices = gpu_indices
        self.interval_s = interval_s
        self.samples = []
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
                    powers[idx] = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
                self.samples.append((t, freqs, powers))
                time.sleep(self.interval_s)
            pynvml.nvmlShutdown()
        except Exception as e:
            log.warning("FreqMonitor error: %s", e)

    def to_arrays(self):
        times = [s[0] for s in self.samples]
        freq_data = {idx: [s[1].get(idx, 0) for s in self.samples] for idx in self.gpu_indices}
        power_data = {idx: [s[2].get(idx, 0) for s in self.samples] for idx in self.gpu_indices}
        return times, freq_data, power_data


def _get_pids_on_port(port):
    result = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
    return list(set(re.findall(r'pid=(\d+)', result.stdout)))

def kill_servers():
    for port in ALL_PORTS:
        for pid in _get_pids_on_port(port):
            subprocess.run(["kill", "-9", pid], capture_output=True)
    time.sleep(5)
    for idx in GPU_INDICES:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)

def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2); s.close(); return True
        except: time.sleep(2)
    return False

def wait_health(url, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try: urllib.request.urlopen(url, timeout=5); return True
        except: time.sleep(3)
    return False

def warmup(url, n=8):
    import requests as req_lib
    import concurrent.futures
    payloads = [{"text": f"Hello {i}", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}} for i in range(n)]
    def _s(p):
        try: req_lib.post(url + "/generate", json=p, timeout=120)
        except: pass
    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        list(ex.map(_s, payloads))
    time.sleep(3)


def start_server(dvfs_extra_args: list, label: str):
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
    env_base['AFD_ASYNC_PIPELINE'] = '1'

    extra = ['--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
             '--afd-micro-batch', '2', '--afd-disagg-interleave-poll',
             '--disable-radix-cache', '--num-reserved-decode-tokens', '32',
             '--afd-async-pipeline']

    dvfs_args = ['--afd-dvfs-enabled', '--afd-energy-model-dir', ENERGY_MODEL_DIR,
                 '--afd-ttft-slo-ms', '5000', '--afd-tpot-slo-us', '60000'] + dvfs_extra_args

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
        fh = open(HERE / f'results/tier2_strategies/{label}_{name}.log', 'w')
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
    rf = open(HERE / f'results/tier2_strategies/{label}_router.log', 'w')
    rp = subprocess.Popen(router_cmd, stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))

    if not wait_health(f'http://127.0.0.1:{ROUTER_PORT}/health', 120):
        return None

    warmup(f'http://127.0.0.1:{ROUTER_PORT}')
    return procs


async def run_workload(workload_path, url):
    with open(workload_path) as f:
        reqs = [json.loads(line) for line in f]
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


def plot_comparison(all_data: dict, output_path: str, workload_path: str):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Load workload phases
    with open(workload_path) as f:
        reqs = [json.loads(line) for line in f]
    phases = {}
    for r in reqs:
        p = r['phase']
        if p not in phases:
            phases[p] = {'start': r['arrival_time_s'], 'end': r['arrival_time_s']}
        phases[p]['end'] = max(phases[p]['end'], r['arrival_time_s'])

    phase_colors = {"light": "#E8F5E9", "heavy": "#FFEBEE", "burst": "#FCE4EC",
                    "recovery": "#E3F2FD", "burst2": "#FFF3E0", "idle": "#F3E5F5",
                    "stable_light": "#E8F5E9", "overload": "#FFEBEE", "stable_after": "#E3F2FD"}

    n_configs = len(all_data)
    fig, axes = plt.subplots(n_configs, 1, figsize=(14, 4 * n_configs), sharex=True)
    if n_configs == 1:
        axes = [axes]

    fig.suptitle("Tier2 DVFS Strategies — GPU Frequency Comparison (TPOT SLO=60ms)", fontsize=13, fontweight='bold')

    colors = {4: 'blue', 5: 'red', 6: 'green', 7: 'orange'}

    for idx, (config_name, data) in enumerate(all_data.items()):
        ax = axes[idx]
        times, freq_data, _ = data

        # Phase backgrounds
        for p, v in phases.items():
            color = phase_colors.get(p, "#F5F5F5")
            ax.axvspan(v['start'], v['end'], alpha=0.2, color=color)

        for gpu_idx in GPU_INDICES:
            ax.plot(times, freq_data[gpu_idx], color=colors[gpu_idx], linewidth=0.8, alpha=0.8,
                    label=f"GPU{gpu_idx} ({GPU_ROLES[gpu_idx]})")

        ax.set_ylabel("Freq (MHz)")
        ax.set_ylim(0, 1550)
        ax.set_title(config_name)
        ax.legend(loc='upper right', fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    log.info("Plot saved: %s", output_path)
    plt.close()


def main():
    workload = str(HERE / "workloads" / "workload_heavy.jsonl")
    out_dir = HERE / "results" / "tier2_strategies"
    out_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        ("1_baseline_dvfs", []),
        ("2_dvfs_feedback", ["--afd-dvfs-feedback", "--afd-dvfs-feedback-threshold", "3", "--afd-dvfs-feedback-hold", "30"]),
        ("3_dvfs_calibration", ["--afd-dvfs-online-calibration", "--afd-dvfs-calibration-ema", "0.3"]),
    ]

    all_data = {}

    for config_name, dvfs_extra in configs:
        log.info("=" * 60)
        log.info("Running: %s", config_name)
        log.info("  Extra args: %s", dvfs_extra or "(none)")
        log.info("=" * 60)

        ret = start_server(dvfs_extra, config_name)
        if ret is None:
            log.error("Server failed for %s", config_name)
            continue
        procs, url = ret, f'http://127.0.0.1:{ROUTER_PORT}'

        try:
            monitor = FreqMonitor(GPU_INDICES, interval_s=0.2)
            monitor.start()

            results = asyncio.run(run_workload(workload, url))
            success = sum(1 for r in results if r.get("success"))
            log.info("%s: %d/%d successful", config_name, success, len(results))

            time.sleep(10)
            monitor.stop()

            times, freq_data, power_data = monitor.to_arrays()
            all_data[config_name] = (times, freq_data, power_data)

            # Save CSV
            with open(out_dir / f"{config_name}_freq.csv", 'w') as f:
                f.write("time_s," + ",".join(f"freq_gpu{i}" for i in GPU_INDICES) + "\n")
                for i, t in enumerate(times):
                    f.write(f"{t:.3f}," + ",".join(str(freq_data[g][i]) for g in GPU_INDICES) + "\n")

        finally:
            for _, p, fh in procs:
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except: pass
            kill_servers()
            time.sleep(5)

    # Plot comparison
    if all_data:
        plot_comparison(all_data, str(out_dir / "tier2_strategies_comparison.png"), workload)

    log.info("Done! Results in %s/", out_dir)


if __name__ == "__main__":
    main()
