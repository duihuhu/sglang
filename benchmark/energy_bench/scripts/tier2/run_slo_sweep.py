#!/usr/bin/env python3
"""Sweep different SLO settings to show energy-latency tradeoff.

Tests Tier2 DVFS under different TPOT SLO constraints:
  - Very tight: 60ms (aggressive, may violate)
  - Tight: 80ms
  - Medium: 120ms
  - Relaxed: 200ms
  - Very relaxed: 300ms (current default)

For each SLO, compares against baseline (no DVFS, max freq).
Reuses the same baseline result across all SLO settings.

Usage:
    /workspace/env/sglang-tier/bin/python run_slo_sweep.py --workload workloads/workload_varying.jsonl
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
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("slo_sweep")

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


def _get_pids_on_port(port):
    result = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
    return list(set(re.findall(r'pid=(\d+)', result.stdout)))


def _ports_busy():
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    return [p for p in ALL_PORTS if f":{p}" in out]


def reset_gpu_clocks():
    for idx in GPU_INDICES:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def kill_servers():
    for port in ALL_PORTS:
        for pid in _get_pids_on_port(port):
            subprocess.run(["kill", "-9", pid], capture_output=True)
    time.sleep(5)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        if not _ports_busy():
            break
        for port in _ports_busy():
            for pid in _get_pids_on_port(port):
                subprocess.run(["kill", "-9", pid], capture_output=True)
        time.sleep(3)
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


def get_gpu_energy_mj(gpu_indices):
    try:
        import pynvml
        pynvml.nvmlInit()
        result = {}
        for idx in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            result[idx] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        pynvml.nvmlShutdown()
        return result
    except:
        return {idx: 0 for idx in gpu_indices}


def start_server(ttft_slo_ms: float, tpot_slo_us: float, dvfs_enabled: bool, label: str):
    """Start PD+AF server with given SLO settings."""
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

    extra = [
        '--skip-server-warmup',
        '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', '1',
        '--afd-disagg-interleave-poll',
        '--disable-radix-cache',
        '--num-reserved-decode-tokens', '32',
    ]

    dvfs_args = []
    if dvfs_enabled:
        dvfs_args = [
            '--afd-dvfs-enabled',
            '--afd-energy-model-dir', ENERGY_MODEL_DIR,
            '--afd-ttft-slo-ms', str(ttft_slo_ms),
            '--afd-tpot-slo-us', str(tpot_slo_us),
        ]

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
            '--afd-perspective', perspective,
            '--afd-comm-backend', 'ipc_cpp',
            '--mem-fraction-static', '0.85',
            '--base-gpu-id', str(base_gpu_id),
            '--disaggregation-mode', disagg_mode,
            '--disaggregation-transfer-backend', 'mooncake',
            '--disaggregation-bootstrap-port', str(BOOTSTRAP_PORT),
            '--disaggregation-ib-device', 'mlx5_4',
        ] + extra + dvfs_args
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
        log.error('Prefill failed'); return None

    _start('df', GPU_DF, 'ffn', 'decode', DF_PORT, UCX_D, SCHED_D, visible_gpus=d_vis, base_gpu_id=0)
    time.sleep(2)
    _start('da', GPU_DA, 'attn', 'decode', DA_PORT, UCX_D, SCHED_D,
           ffn_host='127.0.0.1', visible_gpus=d_vis, base_gpu_id=1)

    if not wait_port('127.0.0.1', DA_PORT) or not wait_port('127.0.0.1', DF_PORT):
        log.error('Decode failed'); return None

    time.sleep(5)
    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', f'http://127.0.0.1:{PA_PORT}',
        '--decode', f'http://127.0.0.1:{DA_PORT}',
        '--host', '127.0.0.1', '--port', str(ROUTER_PORT)]
    rf = open(HERE / f'{label}_router.log', 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))

    if not wait_health(f'http://127.0.0.1:{ROUTER_PORT}/health', 120):
        log.error('Router failed'); return None

    url = f'http://127.0.0.1:{ROUTER_PORT}'
    warmup(url)
    return procs, url


async def run_workload(workload_path: str, url: str) -> dict:
    with open(workload_path) as f:
        requests_data = [json.loads(line) for line in f]

    energy_start = get_gpu_energy_mj(GPU_INDICES)
    timeout = aiohttp.ClientTimeout(total=600)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        base_time = time.monotonic()

        for i, req in enumerate(requests_data):
            arrival = req["arrival_time_s"]
            now = time.monotonic() - base_time
            if arrival > now:
                await asyncio.sleep(arrival - now)

            task = asyncio.create_task(_send_req(session, url, req))
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    duration_s = time.monotonic() - base_time
    energy_end = get_gpu_energy_mj(GPU_INDICES)
    total_energy_j = sum((energy_end[idx] - energy_start[idx]) / 1000.0 for idx in GPU_INDICES)

    successful = [r for r in results if r["success"]]
    ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in successful if r["tpot_ms"] > 0]
    total_tokens = sum(r["tokens"] for r in successful)

    return {
        "duration_s": round(duration_s, 1),
        "successful": len(successful),
        "total": len(results),
        "throughput_tok_s": round(total_tokens / duration_s, 1) if duration_s > 0 else 0,
        "ttft_avg": round(np.mean(ttfts), 1) if ttfts else 0,
        "ttft_p50": round(np.percentile(ttfts, 50), 1) if ttfts else 0,
        "ttft_p90": round(np.percentile(ttfts, 90), 1) if ttfts else 0,
        "ttft_p99": round(np.percentile(ttfts, 99), 1) if ttfts else 0,
        "tpot_avg": round(np.mean(tpots), 1) if tpots else 0,
        "tpot_p50": round(np.percentile(tpots, 50), 1) if tpots else 0,
        "tpot_p90": round(np.percentile(tpots, 90), 1) if tpots else 0,
        "tpot_p99": round(np.percentile(tpots, 99), 1) if tpots else 0,
        "energy_j": round(total_energy_j, 1),
        "power_w": round(total_energy_j / duration_s, 1) if duration_s > 0 else 0,
        "energy_per_tok_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens > 0 else 0,
    }


async def _send_req(session, url, req):
    prompt = "Hello " * (req["input_len"] // 2)
    payload = {"text": prompt, "sampling_params": {"max_new_tokens": req["output_len"], "temperature": 0.0}, "stream": True}
    result = {"success": False, "ttft_ms": 0, "tpot_ms": 0, "tokens": 0}
    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0
    try:
        async with session.post(f"{url}/generate", json=payload) as resp:
            if resp.status != 200:
                return result
            async for line in resp.content:
                text = line.decode().strip()
                if not text or text.startswith(":"): continue
                if text.startswith("data:"): text = text[5:].strip()
                if text == "[DONE]": break
                try:
                    json.loads(text)
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                    token_count += 1
                except: pass
    except:
        return result

    t_end = time.perf_counter()
    result["success"] = True
    result["tokens"] = token_count
    if first_token_time:
        result["ttft_ms"] = (first_token_time - t0) * 1000
    if token_count > 1 and first_token_time:
        result["tpot_ms"] = (t_end - first_token_time) * 1000 / (token_count - 1)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="results/slo_sweep")
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # SLO configurations to test (TTFT_ms, TPOT_us)
    slo_configs = [
        ("tight_60ms", 1000, 60000),
        ("medium_80ms", 2000, 80000),
        ("relaxed_120ms", 3000, 120000),
        ("loose_200ms", 5000, 200000),
        ("very_loose_300ms", 5000, 300000),
    ]

    all_results = {}

    # Run baseline once
    baseline_path = out_dir / "baseline.json"
    if args.skip_baseline and baseline_path.exists():
        all_results["baseline"] = json.loads(baseline_path.read_text())
        log.info("Loaded cached baseline")
    else:
        log.info("=" * 60)
        log.info("Running BASELINE (no DVFS, max freq)")
        log.info("=" * 60)
        ret = start_server(5000, 300000, dvfs_enabled=False, label="slo_baseline")
        if ret:
            procs, url = ret
            try:
                r = asyncio.run(run_workload(args.workload, url))
                all_results["baseline"] = r
                baseline_path.write_text(json.dumps(r, indent=2))
                log.info("Baseline: %d/%d ok, %.1f tok/s, %.1f J", r["successful"], r["total"], r["throughput_tok_s"], r["energy_j"])
            finally:
                for _, p, f in procs:
                    try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                    except: pass
                kill_servers()
                time.sleep(5)

    # Run each SLO config
    for slo_name, ttft_ms, tpot_us in slo_configs:
        cached = out_dir / f"{slo_name}.json"
        if cached.exists():
            all_results[slo_name] = json.loads(cached.read_text())
            log.info("Loaded cached %s", slo_name)
            continue

        log.info("=" * 60)
        log.info("Running SLO: %s (TTFT≤%dms, TPOT≤%dms)", slo_name, ttft_ms, tpot_us // 1000)
        log.info("=" * 60)

        ret = start_server(ttft_ms, tpot_us, dvfs_enabled=True, label=f"slo_{slo_name}")
        if ret is None:
            log.error("Server failed for %s, skipping", slo_name)
            continue
        procs, url = ret
        try:
            r = asyncio.run(run_workload(args.workload, url))
            all_results[slo_name] = r
            cached.write_text(json.dumps(r, indent=2))
            log.info("%s: %d/%d ok, %.1f tok/s, %.1f J", slo_name, r["successful"], r["total"], r["throughput_tok_s"], r["energy_j"])
        finally:
            for _, p, f in procs:
                try: os.killpg(os.getpgid(p.pid), signal.SIGKILL)
                except: pass
            kill_servers()
            time.sleep(5)

    # Print comparison table
    if "baseline" not in all_results:
        log.error("No baseline results")
        return

    base = all_results["baseline"]
    print(f"\n{'='*90}")
    print(f"  SLO SWEEP RESULTS — Energy vs Latency Tradeoff")
    print(f"{'='*90}")
    print(f"  {'SLO':>18s} {'Thpt':>7s} {'TTFT_avg':>9s} {'TTFT_p90':>9s} {'TPOT_avg':>9s} {'TPOT_p90':>9s} {'Energy':>8s} {'Save%':>7s} {'Power':>7s}")
    print(f"  {'─'*18} {'─'*7} {'─'*9} {'─'*9} {'─'*9} {'─'*9} {'─'*8} {'─'*7} {'─'*7}")

    print(f"  {'baseline':>18s} {base['throughput_tok_s']:>7.1f} {base['ttft_avg']:>9.1f} {base['ttft_p90']:>9.1f} "
          f"{base['tpot_avg']:>9.1f} {base['tpot_p90']:>9.1f} {base['energy_j']:>8.1f} {'—':>7s} {base['power_w']:>7.1f}")

    for slo_name, _, _ in slo_configs:
        if slo_name not in all_results:
            continue
        r = all_results[slo_name]
        save = (base["energy_j"] - r["energy_j"]) / base["energy_j"] * 100 if base["energy_j"] > 0 else 0
        print(f"  {slo_name:>18s} {r['throughput_tok_s']:>7.1f} {r['ttft_avg']:>9.1f} {r['ttft_p90']:>9.1f} "
              f"{r['tpot_avg']:>9.1f} {r['tpot_p90']:>9.1f} {r['energy_j']:>8.1f} {save:>6.1f}% {r['power_w']:>7.1f}")

    # Save summary
    summary = {"baseline": base, "slo_configs": {}}
    for slo_name, ttft_ms, tpot_us in slo_configs:
        if slo_name in all_results:
            r = all_results[slo_name]
            r["slo_ttft_ms"] = ttft_ms
            r["slo_tpot_us"] = tpot_us
            r["energy_saving_pct"] = round((base["energy_j"] - r["energy_j"]) / base["energy_j"] * 100, 1)
            summary["slo_configs"][slo_name] = r

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info("Results saved to %s/", out_dir)


if __name__ == "__main__":
    main()
