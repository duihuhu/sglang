#!/usr/bin/env python3
"""Tier1+Tier2 A/B/C benchmark — PD+AF with three control modes.

Three modes compared on real GPU (A800 SXM, 4 GPUs):
  A) Baseline: PD+AF at fixed max frequency (no DVFS, no Tier1)
  B) Tier2 only: PD+AF with per-batch energy-aware frequency selection
  C) Tier1+Tier2: Tier1 workload monitor + freq re-planning + Tier2 DVFS

Uses workloads from workloads/ directory (varying or steady).
Measures: TTFT, TPOT, throughput, GPU energy (NVML), freq distribution.

Usage:
    python run_tier1_tier2_bench.py --workload workloads/workload_varying.jsonl
    python run_tier1_tier2_bench.py --workload workloads/workload_steady.jsonl --modes baseline,tier1_tier2
    python run_tier1_tier2_bench.py --workload workloads/workload_varying.jsonl --skip-baseline
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("tier1_tier2_bench")

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


# ── Process Utils ────────────────────────────────────────────────────────


def _ports_busy():
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    return [p for p in ALL_PORTS if f":{p}" in out]


def _get_pids_on_port(port: int) -> list[str]:
    """Get PIDs of processes listening on a given port using ss."""
    import re
    result = subprocess.run(
        ["ss", "-tlnp", f"sport = :{port}"],
        capture_output=True, text=True,
    )
    pids = []
    for match in re.finditer(r'pid=(\d+)', result.stdout):
        pids.append(match.group(1))
    return list(set(pids))


def kill_our_servers():
    """Kill only servers launched by this script — identified by our specific ports.

    Uses ss to find PIDs listening on our ports, avoids killing unrelated
    processes on GPU 4-7.
    """
    # Kill by PID on our specific ports
    for port in ALL_PORTS:
        pids = _get_pids_on_port(port)
        for pid in pids:
            subprocess.run(["kill", "-9", pid], capture_output=True)

    time.sleep(5)

    # Verify ports are free
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        busy = _ports_busy()
        if not busy:
            break
        # Force kill anything still on our ports
        for port in busy:
            pids = _get_pids_on_port(port)
            for pid in pids:
                subprocess.run(["kill", "-9", pid], capture_output=True)
        log.info("Waiting for ports to free: %s", busy)
        time.sleep(3)
    reset_gpu_clocks()


def reset_gpu_clocks():
    """Reset GPU locked clocks on all GPUs to avoid interference between tests."""
    for idx in GPU_INDICES:
        subprocess.run(
            ["nvidia-smi", "-rgc", "-i", str(idx)],
            capture_output=True,
        )
    log.info("GPU clocks reset (unlocked) on GPUs %s", GPU_INDICES)


def cleanup_procs(procs):
    for name, p, f in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGKILL)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            f.close()
        except Exception:
            pass
    time.sleep(3)
    reset_gpu_clocks()


def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2)
            s.close()
            return True
        except Exception:
            time.sleep(2)
    return False


def wait_health(url, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def warmup(url, n=12):
    import requests
    import concurrent.futures
    payloads = [
        {"text": f"Hello world {i}, warmup:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(n)
    ]
    def _send(p):
        try:
            requests.post(url + "/generate", json=p, timeout=120)
        except Exception:
            pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_send, payloads))
    time.sleep(3)


# ── NVML Energy ──────────────────────────────────────────────────────────


def get_gpu_energy_mj(gpu_indices: list[int]) -> dict[int, int]:
    try:
        import pynvml
        pynvml.nvmlInit()
        result = {}
        for idx in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            result[idx] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        pynvml.nvmlShutdown()
        return result
    except Exception as e:
        log.warning("NVML energy read failed: %s", e)
        return {idx: 0 for idx in gpu_indices}


def get_gpu_freq_mhz(gpu_indices: list[int]) -> dict[int, int]:
    try:
        import pynvml
        pynvml.nvmlInit()
        result = {}
        for idx in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            result[idx] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        pynvml.nvmlShutdown()
        return result
    except Exception:
        return {idx: 0 for idx in gpu_indices}


# ── Server Launcher ──────────────────────────────────────────────────────


def start_pdaf(mode: str, label: str = "") -> Optional[tuple]:
    """Start PD+AF M=1 with C++ IPC backend on GPU 0-3.

    Args:
        mode: "baseline" | "tier2_only" | "tier1_tier2"
        label: Prefix for log files.
    """
    kill_our_servers()
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
    prefix = f'{label}_' if label else ''

    # Shared stats path for Tier 1 cross-process communication
    stats_dir = HERE / "results" / "tier1_shared"
    stats_dir.mkdir(parents=True, exist_ok=True)
    stats_path = str(stats_dir / "decode_stats.json")

    extra = [
        '--skip-server-warmup',
        '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', '2',
        '--afd-disagg-interleave-poll',
        '--disable-radix-cache',
        '--num-reserved-decode-tokens', '32',
        '--afd-async-pipeline',
    ]

    dvfs_args = []
    tier1_args = []
    if mode in ("tier2_only", "tier1_tier2"):
        dvfs_args = [
            '--afd-dvfs-enabled',
            '--afd-energy-model-dir', ENERGY_MODEL_DIR,
            '--afd-ttft-slo-ms', '5000',
            '--afd-tpot-slo-us', '300000',
        ]
    if mode == "tier1_tier2":
        tier1_args = [
            '--enable-tier1-pa',
            '--tier1-monitor-window-s', '15',
            '--tier1-gpu-count', '4',
            '--tier1-stats-path', stats_path,
            '--tier1-prefill-data-path', '/workspace/sglang-tier/benchmark/test_motivation/hucc/paper/prefill_data_v1.txt',
            '--tier1-decode-data-path', '/workspace/sglang-tier/benchmark/test_motivation/hucc/paper/decode_data_v1.txt',
        ]

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
        # peer_device: the other GPU in the same visible pair
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
        # Tier 1 args only on PA process
        if is_pa and tier1_args:
            cmd += tier1_args
        # Non-PA processes also need stats_path for polling
        elif mode == "tier1_tier2" and not is_pa:
            cmd += ['--tier1-stats-path', stats_path]
        log_path = HERE / f'{prefix}pdaf_{name}.log'
        fh = open(log_path, 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((name, p, fh))
        log.info("  Started %s (GPU=%s, port=%d, mode=%s)", name,
                 env['CUDA_VISIBLE_DEVICES'], port, mode)

    p_vis = f'{GPU_PF},{GPU_PA}'
    d_vis = f'{GPU_DF},{GPU_DA}'

    log.info("Starting PD+AF (mode=%s) on GPU %s...", mode, GPU_INDICES)
    _start('pf', GPU_PF, 'ffn', 'prefill', PF_PORT, UCX_P, SCHED_P,
           visible_gpus=p_vis, base_gpu_id=0)
    time.sleep(2)
    _start('pa', GPU_PA, 'attn', 'prefill', PA_PORT, UCX_P, SCHED_P,
           ffn_host='127.0.0.1', visible_gpus=p_vis, base_gpu_id=1, is_pa=True)

    if not wait_port('127.0.0.1', PA_PORT) or not wait_port('127.0.0.1', PF_PORT):
        log.error('Prefill servers failed to start')
        cleanup_procs(procs)
        return None

    _start('df', GPU_DF, 'ffn', 'decode', DF_PORT, UCX_D, SCHED_D,
           visible_gpus=d_vis, base_gpu_id=0)
    time.sleep(2)
    _start('da', GPU_DA, 'attn', 'decode', DA_PORT, UCX_D, SCHED_D,
           ffn_host='127.0.0.1', visible_gpus=d_vis, base_gpu_id=1)

    if not wait_port('127.0.0.1', DA_PORT) or not wait_port('127.0.0.1', DF_PORT):
        log.error('Decode servers failed to start')
        cleanup_procs(procs)
        return None

    time.sleep(5)

    router_cmd = [PYTHON, '-m', 'sglang_router.launch_router',
        '--pd-disaggregation', '--mini-lb',
        '--prefill', f'http://127.0.0.1:{PA_PORT}',
        '--decode', f'http://127.0.0.1:{DA_PORT}',
        '--host', '127.0.0.1', '--port', str(ROUTER_PORT),
    ]
    rf = open(HERE / f'{prefix}pdaf_router.log', 'w')
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(('router', rp, rf))

    if not wait_health(f'http://127.0.0.1:{ROUTER_PORT}/health', timeout=120):
        log.error('Router failed to start')
        cleanup_procs(procs)
        return None

    url = f'http://127.0.0.1:{ROUTER_PORT}'
    log.info("Server ready at %s (mode=%s), warming up...", url, mode)
    warmup(url)
    return procs, url


# ── Workload Execution ───────────────────────────────────────────────────


@dataclass
class RequestResult:
    success: bool = False
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    total_latency_ms: float = 0.0
    output_tokens: int = 0
    input_len: int = 0
    output_len: int = 0
    phase: str = ""


async def send_request(session, url, input_len, output_len, phase=""):
    prompt = "Hello " * (input_len // 2)
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0},
        "stream": True,
    }
    result = RequestResult(input_len=input_len, output_len=output_len, phase=phase)
    t0 = time.perf_counter()
    first_token_time = None
    token_count = 0

    try:
        async with session.post(f"{url}/generate", json=payload) as resp:
            if resp.status != 200:
                return result
            async for line in resp.content:
                now = time.perf_counter()
                text = line.decode().strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    json.loads(text)
                    if first_token_time is None:
                        first_token_time = now
                    token_count += 1
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        log.debug("Request failed: %s", e)
        return result

    t_end = time.perf_counter()
    result.success = True
    result.total_latency_ms = (t_end - t0) * 1000
    result.output_tokens = token_count
    if first_token_time is not None:
        result.ttft_ms = (first_token_time - t0) * 1000
    if token_count > 1 and first_token_time is not None:
        result.tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)
    return result


async def run_workload(workload_path: str, url: str, reload_signal_path: str = None) -> dict:
    """Execute workload trace, measure performance + energy.

    If reload_signal_path is provided, pauses sending when Tier1 reload is
    detected and resumes after reload completes.
    """
    with open(workload_path) as f:
        requests_data = [json.loads(line) for line in f]

    log.info("Running workload: %d requests from %s", len(requests_data), workload_path)

    energy_start = get_gpu_energy_mj(GPU_INDICES)
    freq_samples = []
    reload_pauses = []  # track reload pause durations

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        base_time = time.monotonic()

        for i, req in enumerate(requests_data):
            # Check for Tier1 reload signal
            if reload_signal_path:
                from sglang.srt.energy.reload_signal import is_reloading, wait_until_ready
                if is_reloading(reload_signal_path):
                    pause_start = time.monotonic()
                    log.info("[Reload] Tier1 reload detected at req %d, pausing...", i)
                    ready = wait_until_ready(reload_signal_path, timeout=300)
                    pause_dur = time.monotonic() - pause_start
                    reload_pauses.append(pause_dur)
                    if ready:
                        log.info("[Reload] Complete (%.1fs), resuming workload", pause_dur)
                        base_time += pause_dur  # adjust timeline
                    else:
                        log.error("[Reload] Timeout/error after %.1fs", pause_dur)
                        break

            arrival = req["arrival_time_s"]
            now = time.monotonic() - base_time
            if arrival > now:
                await asyncio.sleep(arrival - now)

            task = asyncio.create_task(send_request(
                session, url,
                input_len=req["input_len"],
                output_len=req["output_len"],
                phase=req.get("phase", ""),
            ))
            tasks.append(task)

            if i % 20 == 0:
                freq_samples.append({
                    "time_s": round(time.monotonic() - base_time, 1),
                    "freqs": get_gpu_freq_mhz(GPU_INDICES),
                })

        results = await asyncio.gather(*tasks)

    duration_s = time.monotonic() - base_time
    energy_end = get_gpu_energy_mj(GPU_INDICES)

    total_energy_j = sum(
        (energy_end[idx] - energy_start[idx]) / 1000.0
        for idx in GPU_INDICES
    )

    successful = [r for r in results if r.success]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms > 0]
    total_tokens = sum(r.output_tokens for r in successful)

    phase_stats = {}
    for r in successful:
        if r.phase not in phase_stats:
            phase_stats[r.phase] = {"ttft": [], "tpot": [], "tokens": 0, "count": 0}
        phase_stats[r.phase]["count"] += 1
        phase_stats[r.phase]["tokens"] += r.output_tokens
        if r.ttft_ms > 0:
            phase_stats[r.phase]["ttft"].append(r.ttft_ms)
        if r.tpot_ms > 0:
            phase_stats[r.phase]["tpot"].append(r.tpot_ms)

    return {
        "duration_s": round(duration_s, 1),
        "total_requests": len(results),
        "successful": len(successful),
        "failed": len(results) - len(successful),
        "total_output_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / duration_s, 1) if duration_s > 0 else 0,
        "ttft_avg_ms": round(float(np.mean(ttfts)), 1) if ttfts else 0,
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        "ttft_p90_ms": round(float(np.percentile(ttfts, 90)), 1) if ttfts else 0,
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p90_ms": round(float(np.percentile(tpots, 90)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "total_energy_j": round(total_energy_j, 1),
        "avg_power_w": round(total_energy_j / duration_s, 1) if duration_s > 0 else 0,
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens > 0 else 0,
        "freq_samples": freq_samples,
        "phase_stats": {
            phase: {
                "count": s["count"],
                "tokens": s["tokens"],
                "ttft_avg_ms": round(float(np.mean(s["ttft"])), 1) if s["ttft"] else 0,
                "tpot_avg_ms": round(float(np.mean(s["tpot"])), 1) if s["tpot"] else 0,
            }
            for phase, s in phase_stats.items()
        },
    }


# ── Output ───────────────────────────────────────────────────────────────


def print_results(label: str, m: dict):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"  Duration:       {m['duration_s']:.1f} s")
    print(f"  Requests:       {m['successful']}/{m['total_requests']} ok")
    print(f"  Throughput:     {m['throughput_tok_s']:.1f} tok/s")
    print(f"  TTFT (avg/p50/p90/p99): {m['ttft_avg_ms']:.1f} / {m['ttft_p50_ms']:.1f} / {m['ttft_p90_ms']:.1f} / {m['ttft_p99_ms']:.1f} ms")
    print(f"  TPOT (avg/p50/p90/p99): {m['tpot_avg_ms']:.1f} / {m['tpot_p50_ms']:.1f} / {m['tpot_p90_ms']:.1f} / {m['tpot_p99_ms']:.1f} ms")
    print(f"  Energy:         {m['total_energy_j']:.1f} J  ({m['avg_power_w']:.1f} W avg)")
    print(f"  Energy/token:   {m['energy_per_token_mj']:.2f} mJ/tok")
    if m.get("phase_stats"):
        print(f"  Per-phase:")
        for phase, s in m["phase_stats"].items():
            print(f"    {phase:15s}: {s['count']:4d} reqs, ttft={s['ttft_avg_ms']:.1f}ms, tpot={s['tpot_avg_ms']:.1f}ms")


def print_comparison(results: dict[str, dict]):
    """Print formatted 3-way comparison."""
    print(f"\n{'='*80}")
    print(f"  COMPARISON: Baseline vs Tier2-Only vs Tier1+Tier2")
    print(f"{'='*80}")

    baseline = results.get("baseline")
    if not baseline:
        log.warning("No baseline results for comparison")
        return

    def delta(b, d, lower_better=True):
        if b == 0:
            return "N/A"
        pct = (d - b) / b * 100
        sign = "+" if pct > 0 else ""
        good = (pct < 0) if lower_better else (pct > 0)
        mark = " ok" if good else (" BAD" if abs(pct) > 5 else "")
        return f"{sign}{pct:.1f}%{mark}"

    metrics = [
        ("Throughput (tok/s)", "throughput_tok_s", False),
        ("TTFT avg (ms)", "ttft_avg_ms", True),
        ("TTFT p90 (ms)", "ttft_p90_ms", True),
        ("TTFT p99 (ms)", "ttft_p99_ms", True),
        ("TPOT avg (ms)", "tpot_avg_ms", True),
        ("TPOT p90 (ms)", "tpot_p90_ms", True),
        ("TPOT p99 (ms)", "tpot_p99_ms", True),
        ("Total energy (J)", "total_energy_j", True),
        ("Avg power (W)", "avg_power_w", True),
        ("Energy/token (mJ)", "energy_per_token_mj", True),
    ]

    modes = [m for m in ["baseline", "tier2_only", "tier1_tier2"] if m in results]
    header = f"  {'Metric':<22s}"
    for m in modes:
        header += f" {m:>12s}"
    if len(modes) > 1:
        for m in modes[1:]:
            header += f" {'d_'+m:>12s}"
    print(header)
    print(f"  {'-'*22}" + f" {'-'*12}" * (len(modes) + len(modes) - 1))

    for name, key, lower_better in metrics:
        row = f"  {name:<22s}"
        for m in modes:
            val = results[m].get(key, 0)
            row += f" {val:>12.1f}"
        if len(modes) > 1:
            for m in modes[1:]:
                row += f" {delta(baseline[key], results[m][key], lower_better):>12s}"
        print(row)

    # Energy saving summary
    print(f"\n  --- Energy Savings ---")
    for m in modes[1:]:
        if baseline["total_energy_j"] > 0:
            saving = (baseline["total_energy_j"] - results[m]["total_energy_j"]) / baseline["total_energy_j"] * 100
            print(f"  {m}: {saving:.1f}% energy saved vs baseline")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Tier1+Tier2 A/B/C benchmark (PD+AF, 3 modes)")
    parser.add_argument("--workload", type=str, required=True,
                        help="Path to workload JSONL file")
    parser.add_argument("--modes", type=str, default="baseline,tier2_only,tier1_tier2",
                        help="Comma-separated modes to run")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-tier2", action="store_true")
    parser.add_argument("--skip-tier1-tier2", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results/tier1_tier2")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    modes_to_run = [m.strip() for m in args.modes.split(",")]
    if args.skip_baseline:
        modes_to_run = [m for m in modes_to_run if m != "baseline"]
    if args.skip_tier2:
        modes_to_run = [m for m in modes_to_run if m != "tier2_only"]
    if args.skip_tier1_tier2:
        modes_to_run = [m for m in modes_to_run if m != "tier1_tier2"]

    all_results = {}
    mode_labels = {
        "baseline": "BASELINE (PD+AF, No DVFS, Max Freq)",
        "tier2_only": "TIER2 ONLY (PD+AF, Per-Batch DVFS)",
        "tier1_tier2": "TIER1+TIER2 (Monitor + Freq Replan + DVFS)",
    }

    for mode in modes_to_run:
        log.info("=" * 65)
        log.info("PHASE: %s", mode_labels.get(mode, mode))
        log.info("=" * 65)

        # Try to load cached results
        cached = out_dir / f"{mode}_results.json"
        if mode not in modes_to_run and cached.exists():
            all_results[mode] = json.loads(cached.read_text())
            log.info("Loaded cached %s from %s", mode, cached)
            continue

        ret = start_pdaf(mode=mode, label=mode)
        if ret is None:
            log.error("%s server failed to start, skipping", mode)
            continue
        procs, url = ret

        try:
            signal_path = None
            if mode == "tier1_tier2":
                signal_path = str(HERE / "results" / "tier1_shared" / "tier1_reload_signal.json")
            results = asyncio.run(run_workload(args.workload, url, reload_signal_path=signal_path))
            all_results[mode] = results

            with open(cached, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print_results(mode_labels.get(mode, mode), results)
        finally:
            cleanup_procs(procs)
            kill_our_servers()
            time.sleep(5)

    # Load any cached results for modes we skipped
    for mode in ["baseline", "tier2_only", "tier1_tier2"]:
        if mode not in all_results:
            cached = out_dir / f"{mode}_results.json"
            if cached.exists():
                all_results[mode] = json.loads(cached.read_text())
                log.info("Loaded cached %s for comparison", mode)

    if len(all_results) >= 2:
        print_comparison(all_results)

    # Save combined summary
    summary = {
        "workload": args.workload,
        "modes": list(all_results.keys()),
        "results": all_results,
    }
    if "baseline" in all_results:
        for m in all_results:
            if m != "baseline" and all_results["baseline"]["total_energy_j"] > 0:
                saving = (
                    (all_results["baseline"]["total_energy_j"] - all_results[m]["total_energy_j"])
                    / all_results["baseline"]["total_energy_j"] * 100
                )
                summary[f"{m}_energy_saving_pct"] = round(saving, 1)

    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("All results saved to %s/", out_dir)


if __name__ == "__main__":
    main()
