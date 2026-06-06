#!/usr/bin/env python3
"""Fixed-length, variable-QPS benchmark — 3 frequency-control policies.

Compares three control policies on PD+AF (A800 SXM, 4 GPUs) across a sweep of
fixed-length workloads where only QPS (arrival rate) changes:

  1) tier1_freq : Tier1 (freq-only, NO model reload) + Tier2 per-batch DVFS.
                  Tier1 re-plans target frequency from the workload monitor but
                  never restarts servers (--tier1-disable-reload).
  2) max_freq   : No Tier. All GPUs locked to max SM clock (1410 MHz).
  3) auto_freq  : No Tier. GPU clocks unlocked → default hardware auto-boost.

Measures per workload: TTFT, TPOT, output throughput, total GPU energy (NVML),
and SLO-violation rate (fraction of requests exceeding TTFT/TPOT SLO).

Usage:
    python run_fixed_qps_bench.py \\
        --workloads workloads/fixed_il512_ol128_qps1.jsonl,workloads/fixed_il512_ol128_qps2.jsonl
    python run_fixed_qps_bench.py --workload-glob 'workloads/fixed_il512_ol128_qps*.jsonl'
    python run_fixed_qps_bench.py --workload-glob 'workloads/fixed_*.jsonl' --modes tier1_freq,max_freq
"""

from __future__ import annotations

import argparse
import asyncio
import glob as globlib
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("fixed_qps_bench")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = Path(__file__).resolve().parent
ENERGY_MODEL_DIR = os.environ.get(
    "SGLANG_ENERGY_MODEL_DIR",
    "/workspace/sglang/benchmark/test_motivation/energy_models"
)

MAX_SM_FREQ_MHZ = 1410

GPU_PA, GPU_PF, GPU_DA, GPU_DF = 7, 6, 5, 4
GPU_INDICES = [GPU_DF, GPU_DA, GPU_PF, GPU_PA]

PA_PORT, PF_PORT = 51010, 51011
DA_PORT, DF_PORT = 51020, 51021
ROUTER_PORT = 51000
BOOTSTRAP_PORT = 29999
UCX_P, UCX_D = 26200, 26300
SCHED_P, SCHED_D = 66400, 66500

ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT]


def _qps_sort_key(x):
    """Sort QPS labels numerically, falling back to string for var-len names."""
    try:
        return (0, float(x))
    except (TypeError, ValueError):
        return (1, str(x))


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
    # Critical: also wait for VRAM to be actually released before next launch,
    # otherwise the next mode starts with a degraded KV cache (run-order bias).
    wait_gpu_mem_free()


def _gpu_mem_used_mib() -> dict[int, int]:
    """Return per-GPU used memory (MiB) for the benchmark GPUs."""
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,memory.used",
         "--format=csv,noheader,nounits",
         "-i", ",".join(str(g) for g in GPU_INDICES)],
        capture_output=True, text=True,
    ).stdout
    used = {}
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) == 2 and parts[0].isdigit():
            used[int(parts[0])] = int(parts[1])
    return used


def wait_gpu_mem_free(threshold_mib: int = 1024, timeout: int = 120) -> bool:
    """Block until all benchmark GPUs drop below threshold memory usage.

    sglang worker processes can keep VRAM allocated for a short while *after*
    their listening port is freed.  Launching the next mode before VRAM is
    actually released causes a smaller KV cache / degraded run (the source of
    the run-order contamination seen at high QPS).  This guards against that.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        used = _gpu_mem_used_mib()
        busy = {g: m for g, m in used.items() if m > threshold_mib}
        if not busy:
            return True
        log.info("Waiting for GPU VRAM to free (>%dMiB): %s", threshold_mib, busy)
        time.sleep(3)
    log.warning("GPU VRAM still not free after %ds: %s",
                timeout, _gpu_mem_used_mib())
    return False


def reset_gpu_clocks():
    """Reset GPU locked clocks on all GPUs to avoid interference between tests."""
    for idx in GPU_INDICES:
        subprocess.run(
            ["nvidia-smi", "-rgc", "-i", str(idx)],
            capture_output=True,
        )
    log.info("GPU clocks reset (unlocked) on GPUs %s", GPU_INDICES)


def lock_gpu_clocks(freq_mhz: int = MAX_SM_FREQ_MHZ):
    """Lock all benchmark GPUs to a fixed SM clock (used by max_freq mode)."""
    for idx in GPU_INDICES:
        subprocess.run(
            ["nvidia-smi", "-lgc", f"{freq_mhz},{freq_mhz}", "-i", str(idx)],
            capture_output=True,
        )
    log.info("GPU clocks LOCKED to %d MHz on GPUs %s", freq_mhz, GPU_INDICES)


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


def start_pdaf(mode: str, label: str = "", log_dir: Optional[Path] = None,
               tpot_slo_ms: float = 300.0) -> Optional[tuple]:
    """Start PD+AF M=1 with C++ IPC backend on GPU 4-7.

    Args:
        mode: "tier1_freq" | "max_freq" | "auto_freq"
            - tier1_freq: Tier1 (freq-only, no reload) + Tier2 per-batch DVFS
            - max_freq:   no Tier, all GPUs locked to MAX_SM_FREQ_MHZ
            - auto_freq:  no Tier, GPU clocks unlocked (hardware auto-boost)
        label: Prefix for log files.
        log_dir: Directory for server + DVFS decision logs (kept separate from
            the bench scripts / workloads).
    """
    kill_our_servers()
    time.sleep(3)

    if log_dir is None:
        log_dir = HERE / "logs"
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Frequency policy for the no-Tier modes is applied at the hardware level
    # (Tier modes manage frequency themselves via NVML inside the scheduler).
    if mode == "max_freq":
        lock_gpu_clocks(MAX_SM_FREQ_MHZ)
    elif mode == "auto_freq":
        reset_gpu_clocks()

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

    # DVFS decision log: only meaningful when DVFS is active (tier1_freq).
    # Each server process writes its own JSONL (templated by perspective/gpu)
    # so there is no cross-process write contention.
    if mode == "tier1_freq":
        env_base['AFD_DVFS_DECISION_LOG'] = str(
            log_dir / f'{prefix}dvfs_decisions_{{persp}}_{{disagg}}_gpu{{gpu}}.jsonl'
        )

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
    if mode == "tier1_freq":
        # Tier2 per-batch DVFS
        dvfs_args = [
            '--afd-dvfs-enabled',
            '--afd-energy-model-dir', ENERGY_MODEL_DIR,
            '--afd-ttft-slo-ms', '5000',
            '--afd-tpot-slo-us', str(int(tpot_slo_ms * 1000)),
        ]
        # Tier1 freq-only: monitor + re-plan frequency, but NEVER reload model
        tier1_args = [
            '--enable-tier1-pa',
            '--tier1-disable-reload',
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
        elif mode == "tier1_freq" and not is_pa:
            cmd += ['--tier1-stats-path', stats_path]
        log_path = log_dir / f'{prefix}pdaf_{name}.log'
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
    rf = open(log_dir / f'{prefix}pdaf_router.log', 'w')
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
    ttft_processing_ms: float = 0.0  # TTFT excluding queue wait (server-side)
    tpot_ms: float = 0.0
    total_latency_ms: float = 0.0
    output_tokens: int = 0
    input_len: int = 0
    output_len: int = 0
    phase: str = ""
    slo_violated: bool = False
    ttft_violated: bool = False
    tpot_violated: bool = False
    # Per-token TPOT SLO tracking
    tpot_token_violations: int = 0  # number of tokens exceeding TPOT SLO
    tpot_token_total: int = 0       # total inter-token intervals measured
    tpot_per_token: list = None     # list of per-token latencies (ms)


async def send_request(session, url, input_len, output_len, phase="",
                       ttft_slo_ms=0.0, tpot_slo_ms=0.0):
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
    last_meta_info = None
    token_times = []  # timestamp of each token arrival

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
                    chunk = json.loads(text)
                    if first_token_time is None:
                        first_token_time = now
                    token_count += 1
                    token_times.append(now)
                    if isinstance(chunk, dict) and "meta_info" in chunk:
                        last_meta_info = chunk["meta_info"]
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

    # Per-token TPOT: compute inter-token latencies
    if len(token_times) > 1:
        per_token_ms = [(token_times[i] - token_times[i-1]) * 1000
                        for i in range(1, len(token_times))]
        result.tpot_per_token = per_token_ms
        result.tpot_token_total = len(per_token_ms)
        if tpot_slo_ms > 0:
            result.tpot_token_violations = sum(1 for t in per_token_ms if t > tpot_slo_ms)

    # Extract server-side processing TTFT (excludes queue wait)
    if last_meta_info and "ttft_pure_processing" in last_meta_info:
        result.ttft_processing_ms = last_meta_info["ttft_pure_processing"] * 1000
    elif last_meta_info and "time_to_first_token_processing" in last_meta_info:
        result.ttft_processing_ms = last_meta_info["time_to_first_token_processing"] * 1000

    # Per-request SLO check: use pure processing TTFT (no queue wait) if available
    ttft_for_slo = result.ttft_processing_ms if result.ttft_processing_ms > 0 else result.ttft_ms
    if ttft_slo_ms > 0 and ttft_for_slo > ttft_slo_ms:
        result.ttft_violated = True
    if tpot_slo_ms > 0 and result.tpot_ms > tpot_slo_ms:
        result.tpot_violated = True
    result.slo_violated = result.ttft_violated or result.tpot_violated
    return result


def procs_all_alive(procs) -> tuple:
    """Return (all_alive, dead_names) for the launched server processes."""
    dead = []
    for name, p, _f in procs:
        if p.poll() is not None:  # process exited
            dead.append(name)
    return (len(dead) == 0, dead)


async def run_workload(workload_path: str, url: str, reload_signal_path: str = None,
                       ttft_slo_ms: float = 0.0, tpot_slo_ms: float = 0.0,
                       procs=None, max_run_s: float = 0.0,
                       gpu_indices=None, prefill_gpus=None, decode_gpus=None) -> dict:
    """Execute workload trace, measure performance + energy + SLO violations.

    If reload_signal_path is provided, pauses sending when Tier1 reload is
    detected and resumes after reload completes.

    If ``procs`` is given, a watchdog aborts the run when any server process
    dies (e.g. OOM crash) so the whole sweep does not hang. ``max_run_s`` is a
    hard wall-clock cap for the request-gathering phase.
    """
    with open(workload_path) as f:
        requests_data = [json.loads(line) for line in f]

    log.info("Running workload: %d requests from %s", len(requests_data), workload_path)

    # Per-deployment GPU mapping (defaults to the module-level PD+AF 4-GPU map).
    _gpu_idx = gpu_indices if gpu_indices is not None else GPU_INDICES
    _pf_gpus = prefill_gpus if prefill_gpus is not None else [GPU_PA, GPU_PF]
    _df_gpus = decode_gpus if decode_gpus is not None else [GPU_DA, GPU_DF]

    energy_start = get_gpu_energy_mj(_gpu_idx)
    freq_samples = []
    reload_pauses = []  # track reload pause durations
    aborted = False
    abort_reason = ""

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        base_time = time.monotonic()

        for i, req in enumerate(requests_data):
            # Abort early if a server process has died (e.g. OOM).
            if procs is not None:
                alive, dead = procs_all_alive(procs)
                if not alive:
                    aborted = True
                    abort_reason = f"server process(es) died: {dead}"
                    log.error("[Watchdog] %s — aborting run at req %d", abort_reason, i)
                    break

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
                ttft_slo_ms=ttft_slo_ms,
                tpot_slo_ms=tpot_slo_ms,
            ))
            tasks.append(task)

            if i % 20 == 0:
                freq_samples.append({
                    "time_s": round(time.monotonic() - base_time, 1),
                    "freqs": get_gpu_freq_mhz(_gpu_idx),
                })

        # Gather remaining requests, but guard against a hung pipeline: if a
        # server dies or we exceed max_run_s, cancel outstanding tasks.
        if tasks:
            gather_task = asyncio.gather(*tasks, return_exceptions=True)
            cap = max_run_s if max_run_s > 0 else 600.0
            deadline = time.monotonic() + cap
            while True:
                done, pending = await asyncio.wait({gather_task}, timeout=3.0)
                if done:
                    raw = await gather_task
                    results = [r for r in raw if isinstance(r, RequestResult)]
                    break
                # still pending — check watchdog conditions
                if procs is not None:
                    alive, dead = procs_all_alive(procs)
                    if not alive:
                        aborted = True
                        abort_reason = f"server process(es) died mid-flight: {dead}"
                        log.error("[Watchdog] %s — cancelling outstanding requests", abort_reason)
                        gather_task.cancel()
                        try:
                            await gather_task
                        except (Exception, asyncio.CancelledError):
                            pass
                        results = []
                        break
                if time.monotonic() > deadline:
                    aborted = True
                    abort_reason = f"run exceeded max_run_s={cap:.0f}s (likely hung)"
                    log.error("[Watchdog] %s — cancelling outstanding requests", abort_reason)
                    gather_task.cancel()
                    try:
                        await gather_task
                    except (Exception, asyncio.CancelledError):
                        pass
                    results = []
                    break
        else:
            results = []

    duration_s = time.monotonic() - base_time
    energy_end = get_gpu_energy_mj(_gpu_idx)

    def _energy_j(indices):
        return sum((energy_end[idx] - energy_start[idx]) / 1000.0
                   for idx in indices)

    total_energy_j = _energy_j(_gpu_idx)
    # Per-stage split: prefill GPUs vs decode GPUs. NVML energy is read per-GPU,
    # so the P/D breakdown is exact given the deployment's GPU mapping.
    prefill_energy_j = _energy_j(_pf_gpus)
    decode_energy_j = _energy_j(_df_gpus)

    successful = [r for r in results if r.success]
    ttfts = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    ttfts_proc = [r.ttft_processing_ms for r in successful if r.ttft_processing_ms > 0]
    tpots = [r.tpot_ms for r in successful if r.tpot_ms > 0]
    total_tokens = sum(r.output_tokens for r in successful)

    # SLO violation rate over successful requests (failed reqs count as violations).
    n_ttft_viol = sum(1 for r in successful if r.ttft_violated)
    n_tpot_viol = sum(1 for r in successful if r.tpot_violated)
    n_slo_viol = sum(1 for r in successful if r.slo_violated)
    n_failed = len(results) - len(successful)
    # Overall violation: failed requests are also SLO violations.
    n_total_viol = n_slo_viol + n_failed
    slo_viol_rate = n_total_viol / len(results) if results else 0.0

    # Per-token TPOT SLO violation stats
    total_token_intervals = sum(r.tpot_token_total for r in successful)
    total_token_violations = sum(r.tpot_token_violations for r in successful)
    token_slo_viol_rate = (total_token_violations / total_token_intervals
                           if total_token_intervals > 0 else 0.0)
    # Per-token TPOT distribution (p50/p90/p99)
    all_per_token_tpots = []
    for r in successful:
        if r.tpot_per_token:
            all_per_token_tpots.extend(r.tpot_per_token)

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
        "aborted": aborted,
        "abort_reason": abort_reason,
        "total_requests": len(results),
        "successful": len(successful),
        "failed": n_failed,
        "total_output_tokens": total_tokens,
        "throughput_tok_s": round(total_tokens / duration_s, 1) if duration_s > 0 else 0,
        "ttft_avg_ms": round(float(np.mean(ttfts)), 1) if ttfts else 0,
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        "ttft_p90_ms": round(float(np.percentile(ttfts, 90)), 1) if ttfts else 0,
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        "ttft_proc_avg_ms": round(float(np.mean(ttfts_proc)), 1) if ttfts_proc else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(ttfts_proc, 50)), 1) if ttfts_proc else 0,
        "ttft_proc_p90_ms": round(float(np.percentile(ttfts_proc, 90)), 1) if ttfts_proc else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(ttfts_proc, 99)), 1) if ttfts_proc else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p90_ms": round(float(np.percentile(tpots, 90)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "total_energy_j": round(total_energy_j, 1),
        "prefill_energy_j": round(prefill_energy_j, 1),
        "decode_energy_j": round(decode_energy_j, 1),
        "avg_power_w": round(total_energy_j / duration_s, 1) if duration_s > 0 else 0,
        "prefill_power_w": round(prefill_energy_j / duration_s, 1) if duration_s > 0 else 0,
        "decode_power_w": round(decode_energy_j / duration_s, 1) if duration_s > 0 else 0,
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens > 0 else 0,
        "ttft_slo_ms": ttft_slo_ms,
        "tpot_slo_ms": tpot_slo_ms,
        "ttft_violations": n_ttft_viol,
        "tpot_violations": n_tpot_viol,
        "slo_violations": n_total_viol,
        "slo_violation_rate": round(slo_viol_rate * 100, 2),
        "tpot_per_token_total": total_token_intervals,
        "tpot_per_token_violations": total_token_violations,
        "tpot_per_token_viol_rate": round(token_slo_viol_rate * 100, 3),
        "tpot_per_token_p50_ms": round(float(np.percentile(all_per_token_tpots, 50)), 1) if all_per_token_tpots else 0,
        "tpot_per_token_p90_ms": round(float(np.percentile(all_per_token_tpots, 90)), 1) if all_per_token_tpots else 0,
        "tpot_per_token_p99_ms": round(float(np.percentile(all_per_token_tpots, 99)), 1) if all_per_token_tpots else 0,
        "tpot_per_token_max_ms": round(float(max(all_per_token_tpots)), 1) if all_per_token_tpots else 0,
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


MODE_LABELS = {
    "tier1_freq": "TIER1-FREQ (Tier1 freq-only, no reload + Tier2 DVFS)",
    "max_freq": "MAX-FREQ (No Tier, locked 1410 MHz)",
    "auto_freq": "AUTO-FREQ (No Tier, hardware auto-boost)",
}
ALL_MODES = ["tier1_freq", "max_freq", "auto_freq"]


def print_results(label: str, m: dict):
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    if m.get("aborted"):
        print(f"  *** ABORTED: {m.get('abort_reason', '')} ***")
    print(f"  Duration:       {m['duration_s']:.1f} s")
    print(f"  Requests:       {m['successful']}/{m['total_requests']} ok ({m['failed']} failed)")
    print(f"  Throughput:     {m['throughput_tok_s']:.1f} tok/s")
    print(f"  TTFT (avg/p50/p90/p99): {m['ttft_avg_ms']:.1f} / {m['ttft_p50_ms']:.1f} / {m['ttft_p90_ms']:.1f} / {m['ttft_p99_ms']:.1f} ms")
    print(f"  TPOT (avg/p50/p90/p99): {m['tpot_avg_ms']:.1f} / {m['tpot_p50_ms']:.1f} / {m['tpot_p90_ms']:.1f} / {m['tpot_p99_ms']:.1f} ms")
    print(f"  Energy:         {m['total_energy_j']:.1f} J  ({m['avg_power_w']:.1f} W avg)")
    print(f"  Energy/token:   {m['energy_per_token_mj']:.2f} mJ/tok")
    print(f"  SLO violation:  {m['slo_violation_rate']:.2f}%  "
          f"(ttft={m['ttft_violations']} tpot={m['tpot_violations']} "
          f"total={m['slo_violations']}/{m['total_requests']}; "
          f"SLO: ttft<={m['ttft_slo_ms']:.0f}ms tpot<={m['tpot_slo_ms']:.0f}ms)")


def print_comparison(results: dict[str, dict], qps_label: str = ""):
    """Print formatted N-way comparison across modes (ref = max_freq)."""
    title = "COMPARISON: tier1_freq vs max_freq vs auto_freq"
    if qps_label:
        title += f"  [{qps_label}]"
    print(f"\n{'='*88}")
    print(f"  {title}")
    print(f"{'='*88}")

    ref = results.get("max_freq")

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
        ("SLO violation (%)", "slo_violation_rate", True),
    ]

    modes = [m for m in ALL_MODES if m in results]
    header = f"  {'Metric':<22s}"
    for m in modes:
        header += f" {m:>12s}"
    if ref is not None:
        for m in modes:
            if m != "max_freq":
                header += f" {'d_'+m:>14s}"
    print(header)
    print(f"  {'-'*22}" + f" {'-'*12}" * len(modes))

    for name, key, lower_better in metrics:
        row = f"  {name:<22s}"
        for m in modes:
            val = results[m].get(key, 0)
            row += f" {val:>12.1f}"
        if ref is not None:
            for m in modes:
                if m != "max_freq":
                    row += f" {delta(ref[key], results[m][key], lower_better):>14s}"
        print(row)

    # Energy saving summary (vs max_freq reference)
    if ref is not None and ref["total_energy_j"] > 0:
        print(f"\n  --- Energy Savings (vs max_freq) ---")
        for m in modes:
            if m == "max_freq":
                continue
            saving = (ref["total_energy_j"] - results[m]["total_energy_j"]) / ref["total_energy_j"] * 100
            print(f"  {m}: {saving:.1f}% energy saved vs max_freq "
                  f"(SLO viol {results[m]['slo_violation_rate']:.1f}% vs {ref['slo_violation_rate']:.1f}%)")


def print_qps_sweep(sweep: dict):
    """Print a compact table of key metrics across QPS for every mode."""
    print(f"\n{'='*100}")
    print(f"  QPS SWEEP SUMMARY (fixed-length workloads)")
    print(f"{'='*100}")
    qps_keys = sorted(sweep.keys(), key=_qps_sort_key)
    cols = [
        ("Energy(J)", "total_energy_j"),
        ("Power(W)", "avg_power_w"),
        ("Thpt(tok/s)", "throughput_tok_s"),
        ("TTFTavg(ms)", "ttft_avg_ms"),
        ("TPOTavg(ms)", "tpot_avg_ms"),
        ("SLOviol(%)", "slo_violation_rate"),
    ]
    for col_label, key in cols:
        print(f"\n  --- {col_label} ---")
        header = f"  {'QPS':>6s}"
        for mode in ALL_MODES:
            header += f" {mode:>14s}"
        print(header)
        for qk in qps_keys:
            row = f"  {qk:>6s}"
            for mode in ALL_MODES:
                v = sweep[qk].get(mode, {}).get(key)
                row += f" {('-' if v is None else f'{v:.1f}'):>14s}"
            print(row)


# ── Main ─────────────────────────────────────────────────────────────────


def _cfg_from_path(path: str) -> dict:
    """Extract {il, ol, qps, tag, group} from a workload filename.

    Two filename conventions are supported:
      * fixed_il<I>_ol<O>_qps<N>.jsonl  → fixed-length sweep (il/ol/qps parsed)
      * any other name (e.g. workload_varying.jsonl) → variable-length trace;
        binned under group "var" with a sanitized tag so its results never
        collide with (or pollute) the fixed-length il<I>_ol<O> groups.
    """
    stem = Path(path).stem
    il = re.search(r"il(\d+)", stem)
    ol = re.search(r"ol(\d+)", stem)
    qps = re.search(r"qps([0-9p]+)", stem)

    # Variable-length trace: no il/ol/qps in the name → keep it out of the
    # fixed-length groups entirely (otherwise it shows up as "il?_ol?").
    if not (il and ol):
        name = re.sub(r"^workload_", "", stem)
        name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_") or stem
        return {
            "il": "var", "ol": "var", "qps": name,
            "group": f"var_{name}",
            "tag": f"var_{name}",
        }

    il_v = il.group(1)
    ol_v = ol.group(1)
    qps_v = qps.group(1).replace("p", ".") if qps else stem
    group = f"il{il_v}_ol{ol_v}"
    return {
        "il": il_v, "ol": ol_v, "qps": qps_v,
        "group": group,
        "tag": f"{group}_qps{(qps.group(1) if qps else stem)}",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fixed-length variable-QPS benchmark (PD+AF, 3 freq-control modes)")
    parser.add_argument("--workloads", type=str, default=None,
                        help="Comma-separated workload JSONL paths")
    parser.add_argument("--workload-glob", type=str, default=None,
                        help="Glob pattern for workload JSONL files")
    parser.add_argument("--modes", type=str, default="tier1_freq,max_freq,auto_freq",
                        help="Comma-separated modes: tier1_freq,max_freq,auto_freq")
    parser.add_argument("--ttft-slo-ms", type=float, default=5000.0,
                        help="TTFT SLO budget for violation accounting (ms)")
    parser.add_argument("--tpot-slo-ms", type=float, default=300.0,
                        help="TPOT SLO budget for violation accounting (ms)")
    parser.add_argument("--output-dir", type=str, default="results/fixed_qps/json")
    parser.add_argument("--log-dir", type=str, default="logs",
                        help="Directory for server + DVFS decision logs "
                        "(kept separate from bench scripts/workloads)")
    parser.add_argument("--force", action="store_true",
                        help="Re-run even if cached result exists")
    parser.add_argument("--max-run-s", type=float, default=300.0,
                        help="Hard wall-clock cap per run; abort (and skip) if "
                        "exceeded — guards against a hung pipeline. Default 300s.")
    args = parser.parse_args()

    # Resolve workload list
    workloads = []
    if args.workloads:
        workloads += [w.strip() for w in args.workloads.split(",") if w.strip()]
    if args.workload_glob:
        workloads += sorted(globlib.glob(args.workload_glob))
    if not workloads:
        parser.error("Provide --workloads or --workload-glob")
    # Dedup, keep order
    seen = set()
    workloads = [w for w in workloads if not (w in seen or seen.add(w))]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir)
    log_root.mkdir(parents=True, exist_ok=True)

    modes_to_run = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes_to_run:
        if m not in ALL_MODES:
            parser.error(f"Unknown mode '{m}'. Valid: {ALL_MODES}")

    log.info("Workloads (%d): %s", len(workloads), workloads)
    log.info("Modes: %s | TTFT_SLO=%.0fms TPOT_SLO=%.0fms | logs→%s",
             modes_to_run, args.ttft_slo_ms, args.tpot_slo_ms, log_root)

    # sweep[group][qps_label][mode] = result dict
    sweep: dict[str, dict] = {}

    for wl in workloads:
        cfg = _cfg_from_path(wl)
        group, qps_label, tag = cfg["group"], cfg["qps"], cfg["tag"]
        # Tighter-TPOT-SLO experiments tag results separately so they never
        # collide with the default 300ms-SLO cache/logs.
        if abs(args.tpot_slo_ms - 300.0) > 1e-6:
            tag = f"{tag}_tpot{int(args.tpot_slo_ms)}"
        sweep.setdefault(group, {}).setdefault(qps_label, {})
        log.info("#" * 70)
        log.info("WORKLOAD: %s  (il=%s ol=%s QPS=%s)", wl, cfg["il"], cfg["ol"], qps_label)
        log.info("#" * 70)

        for mode in modes_to_run:
            log.info("=" * 65)
            log.info("PHASE: %s | %s", tag, MODE_LABELS.get(mode, mode))
            log.info("=" * 65)

            cached = out_dir / f"{tag}_{mode}_results.json"
            if cached.exists() and not args.force:
                sweep[group][qps_label][mode] = json.loads(cached.read_text())
                log.info("Loaded cached result from %s", cached)
                continue

            run_log_dir = log_root / tag / mode
            ret = start_pdaf(mode=mode, label=f"{tag}_{mode}", log_dir=run_log_dir,
                             tpot_slo_ms=args.tpot_slo_ms)
            if ret is None:
                log.error("%s mode=%s server failed to start, skipping", tag, mode)
                continue
            procs, url = ret

            try:
                signal_path = None
                if mode == "tier1_freq":
                    # Tier1 still writes a freq config; reload signal should
                    # never fire because --tier1-disable-reload is set, but we
                    # keep watching it so the harness pauses if it somehow does.
                    signal_path = str(HERE / "results" / "tier1_shared" / "tier1_reload_signal.json")
                results = asyncio.run(run_workload(
                    wl, url, reload_signal_path=signal_path,
                    ttft_slo_ms=args.ttft_slo_ms, tpot_slo_ms=args.tpot_slo_ms,
                    procs=procs, max_run_s=args.max_run_s,
                ))
                results["workload"] = wl
                results["il"] = cfg["il"]
                results["ol"] = cfg["ol"]
                results["qps"] = qps_label
                results["mode"] = mode
                results["log_dir"] = str(run_log_dir)
                sweep[group][qps_label][mode] = results

                # Don't cache aborted runs (crash/hang) so they re-run later.
                if results.get("aborted"):
                    log.error("%s mode=%s ABORTED: %s (not cached)",
                              tag, mode, results.get("abort_reason"))
                else:
                    with open(cached, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                print_results(f"{tag} | {MODE_LABELS.get(mode, mode)}", results)
            except (KeyboardInterrupt, SystemExit):
                raise
            except (Exception, asyncio.CancelledError) as e:
                # A single run blowing up (e.g. watchdog cancellation, network
                # error, OOM mid-flight) must NOT kill the whole sweep. Log it,
                # leave it uncached so it can be re-run, and move on.
                import traceback
                log.error("%s mode=%s crashed: %r — skipping (not cached)\n%s",
                          tag, mode, e, traceback.format_exc())
            finally:
                cleanup_procs(procs)
                kill_our_servers()
                time.sleep(5)

        # Per-(group,QPS) comparison
        if len(sweep[group][qps_label]) >= 2:
            print_comparison(sweep[group][qps_label], qps_label=f"{group} QPS={qps_label}")

    # Cross-QPS sweep table, per il/ol group
    for group, qps_map in sweep.items():
        print(f"\n\n##### GROUP: {group} #####")
        print_qps_sweep(qps_map)

    # Save combined summary
    summary = {
        "workloads": workloads,
        "modes": modes_to_run,
        "ttft_slo_ms": args.ttft_slo_ms,
        "tpot_slo_ms": args.tpot_slo_ms,
        "sweep": sweep,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info("All results saved to %s/ (logs in %s/)", out_dir, log_root)


if __name__ == "__main__":
    main()
