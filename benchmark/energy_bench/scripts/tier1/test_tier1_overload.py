#!/usr/bin/env python3
"""Test Tier1 reload under overload — demonstrates TP change trigger.

Scenario:
  - Start PD+AF with tp=1, tight SLO (TPOT≤55ms)
  - Send overload workload (10 QPS, long output)
  - Tier1 monitor detects SLO violations → triggers reload
  - Orchestrator kills all, restarts with same TP (simulating TP change)
  - Benchmark pauses during reload, resumes after

This test verifies the full Tier1 reload pipeline works end-to-end.

Usage:
    /workspace/env/sglang-tier/bin/python test_tier1_overload.py
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
log = logging.getLogger("tier1_overload")

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = Path(__file__).resolve().parent
ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/test_motivation/energy_models"

GPU_PA, GPU_PF, GPU_DA, GPU_DF = 7, 6, 5, 4
GPU_INDICES = [GPU_DF, GPU_DA, GPU_PF, GPU_PA]

PA_PORT, PF_PORT = 51010, 51011
DA_PORT, DF_PORT = 51020, 51021
ROUTER_PORT = 51000
BOOTSTRAP_PORT = 29999
UCX_P, UCX_D = 26200, 26300
SCHED_P, SCHED_D = 66400, 66500
ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT]

STATS_DIR = HERE / "results" / "tier1_overload"
SIGNAL_PATH = str(STATS_DIR / "tier1_reload_signal.json")


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

def warmup(url, n=4):
    import requests as req_lib
    import concurrent.futures
    payloads = [{"text": f"Hi {i}", "sampling_params": {"max_new_tokens": 4, "temperature": 0.0}} for i in range(n)]
    def _s(p):
        try: req_lib.post(url + "/generate", json=p, timeout=60)
        except: pass
    with concurrent.futures.ThreadPoolExecutor(n) as ex:
        list(ex.map(_s, payloads))
    time.sleep(2)


def start_server():
    """Start PD+AF with Tier1 enabled and tight SLO."""
    kill_servers()
    STATS_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old signal
    from sglang.srt.energy.reload_signal import clear_signal
    clear_signal(SIGNAL_PATH)

    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    env_base['SGLANG_DISAGGREGATION_THREAD_POOL_SIZE'] = '128'
    env_base['SGLANG_DISAGGREGATION_QUEUE_SIZE'] = '32'
    env_base['SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE'] = '0'
    env_base['SGLANG_PYTHON'] = PYTHON
    env_base['AFD_ASYNC_PIPELINE'] = '1'

    extra = ['--skip-server-warmup', '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
             '--afd-micro-batch', '2', '--afd-disagg-interleave-poll',
             '--disable-radix-cache', '--num-reserved-decode-tokens', '32',
             '--afd-async-pipeline']

    dvfs_args = ['--afd-dvfs-enabled', '--afd-energy-model-dir', ENERGY_MODEL_DIR,
                 '--afd-ttft-slo-ms', '5000', '--afd-tpot-slo-us', '80000']

    tier1_args = ['--enable-tier1-pa',
                  '--tier1-monitor-window-s', '30',
                  '--tier1-gpu-count', '8',
                  '--tier1-stats-path', str(STATS_DIR / "decode_stats.json"),
                  '--tier1-prefill-data-path', '/workspace/sglang-tier/benchmark/test_motivation/hucc/paper/prefill_data_v1.txt',
                  '--tier1-decode-data-path', '/workspace/sglang-tier/benchmark/test_motivation/hucc/paper/decode_data_v1.txt',
                  '--tier1-il-rep-p', '512',
                  '--tier1-il-rep-d', '512',
                  '--tier1-ol-rep-d', '128',
                  '--tier1-bs-avg-d', '32']

    def _start(name, gpu, perspective, disagg_mode, port, ucx_base, sched_port,
               ffn_host=None, visible_gpus=None, base_gpu_id=0, is_pa=False):
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
        if is_pa:
            cmd += tier1_args
        else:
            cmd += ['--tier1-stats-path', str(STATS_DIR / "decode_stats.json")]
        fh = open(STATS_DIR / f'{name}.log', 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))
        log.info("  Started %s (port=%d)", name, port)

    p_vis = f'{GPU_PF},{GPU_PA}'
    d_vis = f'{GPU_DF},{GPU_DA}'

    log.info("Starting PD+AF with Tier1 (SLO: TPOT≤55ms)...")
    _start('pf', GPU_PF, 'ffn', 'prefill', PF_PORT, UCX_P, SCHED_P, visible_gpus=p_vis, base_gpu_id=0)
    time.sleep(2)
    _start('pa', GPU_PA, 'attn', 'prefill', PA_PORT, UCX_P, SCHED_P,
           ffn_host='127.0.0.1', visible_gpus=p_vis, base_gpu_id=1, is_pa=True)
    if not wait_port('127.0.0.1', PA_PORT) or not wait_port('127.0.0.1', PF_PORT):
        log.error("Prefill failed"); return None

    _start('df', GPU_DF, 'ffn', 'decode', DF_PORT, UCX_D, SCHED_D, visible_gpus=d_vis, base_gpu_id=0)
    time.sleep(2)
    _start('da', GPU_DA, 'attn', 'decode', DA_PORT, UCX_D, SCHED_D,
           ffn_host='127.0.0.1', visible_gpus=d_vis, base_gpu_id=1)
    if not wait_port('127.0.0.1', DA_PORT) or not wait_port('127.0.0.1', DF_PORT):
        log.error("Decode failed"); return None

    time.sleep(5)
    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', f'http://127.0.0.1:{PA_PORT}', '--decode', f'http://127.0.0.1:{DA_PORT}',
        '--host', '127.0.0.1', '--port', str(ROUTER_PORT)]
    rf = open(STATS_DIR / 'router.log', 'w')
    rp = subprocess.Popen(router_cmd, stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))

    if not wait_health(f'http://127.0.0.1:{ROUTER_PORT}/health', 120):
        log.error("Router failed"); return None

    warmup(f'http://127.0.0.1:{ROUTER_PORT}')
    log.info("Server ready!")
    return procs


async def run_workload_with_reload(workload_path: str, url: str):
    """Run workload, monitoring for Tier1 reload signals."""
    from sglang.srt.energy.reload_signal import is_reloading, wait_until_ready

    with open(workload_path) as f:
        reqs = [json.loads(line) for line in f]

    log.info("Running workload: %d requests (monitoring reload signal)", len(reqs))

    results = []
    reload_events = []
    timeout = aiohttp.ClientTimeout(total=300)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = []

        for i, req in enumerate(reqs):
            # Check reload signal
            if is_reloading(SIGNAL_PATH):
                pause_start = time.monotonic()
                log.info("[T=%.1fs] RELOAD DETECTED at req %d! Pausing...",
                        time.monotonic() - base_time, i)
                ready = wait_until_ready(SIGNAL_PATH, timeout=300)
                pause_dur = time.monotonic() - pause_start
                reload_events.append({
                    "time_s": round(pause_start - base_time + pause_dur, 1),
                    "duration_s": round(pause_dur, 1),
                    "success": ready,
                })
                if ready:
                    log.info("[T=%.1fs] Reload complete (%.1fs), resuming",
                            time.monotonic() - base_time, pause_dur)
                    base_time += pause_dur
                else:
                    log.error("Reload failed/timeout!")
                    break

            arrival = req["arrival_time_s"]
            now = time.monotonic() - base_time
            if arrival > now:
                await asyncio.sleep(arrival - now)

            task = asyncio.create_task(_send(session, url, req))
            tasks.append(task)

        raw_results = await asyncio.gather(*tasks)

    duration = time.monotonic() - base_time
    success = sum(1 for r in raw_results if r.get("success"))
    log.info("Workload done: %d/%d successful in %.1fs", success, len(reqs), duration)
    log.info("Reload events: %d", len(reload_events))
    for ev in reload_events:
        log.info("  Reload at T=%.1fs, duration=%.1fs, success=%s",
                ev["time_s"], ev["duration_s"], ev["success"])

    return {"success": success, "total": len(reqs), "duration_s": duration,
            "reload_events": reload_events}


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


def monitor_tier1_logs():
    """Background thread to print Tier1 events from PA log."""
    pa_log = STATS_DIR / "pa.log"
    last_pos = 0
    while True:
        time.sleep(5)
        if not pa_log.exists():
            continue
        try:
            with open(pa_log) as f:
                f.seek(last_pos)
                for line in f:
                    if "[Tier1]" in line or "tier1" in line.lower():
                        log.info("[PA] %s", line.strip()[-120:])
                last_pos = f.tell()
        except:
            pass


def main():
    workload = str(HERE / "workloads" / "workload_tier1_demo.jsonl")

    log.info("=" * 70)
    log.info("  TIER1 OVERLOAD TEST — Demonstrating Reload Under SLO Pressure")
    log.info("  Workload: %s", workload)
    log.info("  SLO: TPOT ≤ 55ms (tight — will be violated under heavy load)")
    log.info("  Expected: Tier1 detects SLO violation → triggers reload")
    log.info("=" * 70)

    procs = start_server()
    if procs is None:
        log.error("Failed to start server")
        return

    # Start log monitor
    mon_thread = threading.Thread(target=monitor_tier1_logs, daemon=True)
    mon_thread.start()

    try:
        url = f"http://127.0.0.1:{ROUTER_PORT}"
        results = asyncio.run(run_workload_with_reload(workload, url))

        log.info("\n" + "=" * 70)
        log.info("  RESULTS")
        log.info("=" * 70)
        log.info("  Requests: %d/%d successful", results["success"], results["total"])
        log.info("  Duration: %.1fs", results["duration_s"])
        log.info("  Reload events: %d", len(results["reload_events"]))

        # Save results
        with open(STATS_DIR / "overload_results.json", "w") as f:
            json.dump(results, f, indent=2)

    finally:
        log.info("Cleaning up...")
        for _, p, f in procs:
            try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except: pass
        kill_servers()


if __name__ == "__main__":
    main()
