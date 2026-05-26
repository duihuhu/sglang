#!/usr/bin/env python3
"""Tier2 DVFS A/B benchmark — PD+AF M=1 (C++ IPC) with/without DVFS.

Adopts the proven launch pattern from run_pdaf_vs_pd.py (C++ IPC backend,
mooncake transfer, disable-cuda-graph). Uses GPU 0-3.

Two modes:
  A) Baseline: PD+AF at fixed max frequency (no DVFS)
  B) Tier2 DVFS: PD+AF with energy-aware frequency selection

Measures: TTFT, TPOT, throughput, GPU energy (NVML).

Usage:
    python run_tier2_bench.py --workload workloads/workload_varying.jsonl
    python run_tier2_bench.py --workload workloads/workload_varying.jsonl --skip-baseline
    python run_tier2_bench.py --workload workloads/workload_varying.jsonl --skip-dvfs
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
log = logging.getLogger("tier2_bench")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = Path(__file__).resolve().parent
ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/test_motivation/energy_models"

# GPU allocation: front 4 cards
GPU_PA, GPU_PF, GPU_DA, GPU_DF = 3, 2, 1, 0
GPU_INDICES = [GPU_DF, GPU_DA, GPU_PF, GPU_PA]

# Ports (avoid conflict with other tests on GPU 4-7)
PA_PORT, PF_PORT = 51010, 51011
DA_PORT, DF_PORT = 51020, 51021
ROUTER_PORT = 51000
BOOTSTRAP_PORT = 19999
UCX_P, UCX_D = 26200, 26300
SCHED_P, SCHED_D = 66400, 66500

ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT]


# ── Process Utils ────────────────────────────────────────────────────────


def _ports_busy():
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    return [p for p in ALL_PORTS if f":{p}" in out]


def kill_our_servers():
    """Kill only servers launched by this script (sglang-tier env, our ports)."""
    for t in ["sglang-tier/bin/python.*sglang.launch_server",
              "sglang-tier/bin/python.*sglang_router",
              "sglang::router",
              "sglang::scheduler"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(5)
    for t in ["sglang-tier/bin/python.*sglang.launch_server",
              "sglang-tier/bin/python.*sglang_router",
              "sglang::router",
              "sglang::scheduler"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        busy = _ports_busy()
        if not busy:
            break
        log.info("Waiting for ports to free: %s", busy)
        time.sleep(3)


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
    """Read total energy consumption (millijoules) for each GPU."""
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


def start_pdaf(dvfs_enabled: bool, label: str = "") -> Optional[tuple]:
    """Start PD+AF M=1 with C++ IPC backend on GPU 0-3.

    Args:
        dvfs_enabled: If True, add --afd-dvfs-enabled and energy model args.
        label: Prefix for log files.
    """
    kill_our_servers()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
    env_base['UCX_LOG_LEVEL'] = 'fatal'
    env_base['AFD_UCX_TLS'] = 'rc,tcp,cuda_copy,cuda_ipc'
    prefix = f'{label}_' if label else ''

    extra = [
        '--skip-server-warmup',
        '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
        '--afd-micro-batch', '1',
        '--max-running-requests', '96',
        '--afd-disagg-interleave-poll',
        '--disable-radix-cache',
    ]

    dvfs_args = []
    if dvfs_enabled:
        dvfs_args = [
            '--afd-dvfs-enabled',
            '--afd-energy-model-dir', ENERGY_MODEL_DIR,
            '--afd-ttft-slo-ms', '5000',
            '--afd-tpot-slo-us', '300000',
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
        if ffn_host:
            env['AFD_UCX_FFN_HOST'] = ffn_host
        cmd = [PYTHON, '-m', 'sglang.launch_server',
            '--model-path', MODEL, '--tp', '1',
            '--host', '127.0.0.1', '--port', str(port),
            '--afd-perspective', perspective,
            '--afd-comm-backend', 'ipc_cpp',
            '--mem-fraction-static', '0.70',
            '--base-gpu-id', str(base_gpu_id),
            '--disaggregation-mode', disagg_mode,
            '--disaggregation-transfer-backend', 'mooncake',
            '--disaggregation-bootstrap-port', str(BOOTSTRAP_PORT),
            '--disaggregation-ib-device', 'mlx5_4',
        ] + extra + dvfs_args
        log_path = HERE / f'{prefix}pdaf_{name}.log'
        fh = open(log_path, 'w')
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((name, p, fh))
        log.info("  Started %s (GPU=%s, port=%d)", name, env['CUDA_VISIBLE_DEVICES'], port)

    # Prefill pair: PF first, then PA
    p_vis = f'{GPU_PF},{GPU_PA}'
    d_vis = f'{GPU_DF},{GPU_DA}'

    log.info("Starting PD+AF (dvfs=%s) on GPU %s...", dvfs_enabled, GPU_INDICES)
    _start('pf', GPU_PF, 'ffn', 'prefill', PF_PORT, UCX_P, SCHED_P,
           visible_gpus=p_vis, base_gpu_id=0)
    time.sleep(2)
    _start('pa', GPU_PA, 'attn', 'prefill', PA_PORT, UCX_P, SCHED_P,
           ffn_host='127.0.0.1', visible_gpus=p_vis, base_gpu_id=1)

    if not wait_port('127.0.0.1', PA_PORT) or not wait_port('127.0.0.1', PF_PORT):
        log.error('Prefill servers failed to start')
        cleanup_procs(procs)
        return None

    # Decode pair: DF first, then DA
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

    # Router
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
    log.info("Server ready at %s, warming up...", url)
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


async def run_workload(workload_path: str, url: str) -> dict:
    """Execute workload trace, measure performance + energy."""
    with open(workload_path) as f:
        requests_data = [json.loads(line) for line in f]

    log.info("Running workload: %d requests from %s", len(requests_data), workload_path)

    energy_start = get_gpu_energy_mj(GPU_INDICES)
    freq_samples = []

    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        base_time = time.monotonic()

        for i, req in enumerate(requests_data):
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
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Duration:       {m['duration_s']:.1f} s")
    print(f"  Requests:       {m['successful']}/{m['total_requests']} ok")
    print(f"  Throughput:     {m['throughput_tok_s']:.1f} tok/s")
    print(f"  TTFT (avg/p50/p90/p99): {m['ttft_avg_ms']:.1f} / {m['ttft_p50_ms']:.1f} / {m['ttft_p90_ms']:.1f} / {m['ttft_p99_ms']:.1f} ms")
    print(f"  TPOT (avg/p50/p90/p99): {m['tpot_avg_ms']:.1f} / {m['tpot_p50_ms']:.1f} / {m['tpot_p90_ms']:.1f} / {m['tpot_p99_ms']:.1f} ms")
    print(f"  Energy:         {m['total_energy_j']:.1f} J  ({m['avg_power_w']:.1f} W avg)")
    print(f"  Energy/token:   {m['energy_per_token_mj']:.2f} mJ/tok")
    print(f"  Per-phase:")
    for phase, s in m.get("phase_stats", {}).items():
        print(f"    {phase:15s}: {s['count']:4d} reqs, ttft={s['ttft_avg_ms']:.1f}ms, tpot={s['tpot_avg_ms']:.1f}ms")


def print_comparison(baseline: dict, dvfs: dict):
    print(f"\n{'='*70}")
    print(f"  COMPARISON: Baseline (max freq) vs Tier2 DVFS")
    print(f"{'='*70}")

    def delta(b, d, lower_better=True):
        if b == 0: return "N/A"
        pct = (d - b) / b * 100
        sign = "+" if pct > 0 else ""
        good = (pct < 0) if lower_better else (pct > 0)
        mark = " ✓" if good else (" ✗" if abs(pct) > 5 else "")
        return f"{sign}{pct:.1f}%{mark}"

    rows = [
        ("Throughput (tok/s)", baseline['throughput_tok_s'], dvfs['throughput_tok_s'], False),
        ("TTFT avg (ms)", baseline['ttft_avg_ms'], dvfs['ttft_avg_ms'], True),
        ("TTFT p90 (ms)", baseline['ttft_p90_ms'], dvfs['ttft_p90_ms'], True),
        ("TPOT avg (ms)", baseline['tpot_avg_ms'], dvfs['tpot_avg_ms'], True),
        ("TPOT p90 (ms)", baseline['tpot_p90_ms'], dvfs['tpot_p90_ms'], True),
        ("Total energy (J)", baseline['total_energy_j'], dvfs['total_energy_j'], True),
        ("Avg power (W)", baseline['avg_power_w'], dvfs['avg_power_w'], True),
        ("Energy/token (mJ)", baseline['energy_per_token_mj'], dvfs['energy_per_token_mj'], True),
    ]
    print(f"  {'Metric':<22s} {'Baseline':>10s} {'DVFS':>10s} {'Delta':>12s}")
    print(f"  {'-'*22} {'-'*10} {'-'*10} {'-'*12}")
    for name, b, d, lb in rows:
        print(f"  {name:<22s} {b:>10.1f} {d:>10.1f} {delta(b, d, lb):>12s}")


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Tier2 DVFS A/B benchmark (PD+AF C++ IPC)")
    parser.add_argument("--workload", type=str, required=True)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-dvfs", action="store_true")
    parser.add_argument("--output-dir", type=str, default="results")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline_results = None
    dvfs_results = None

    # ── Baseline ─────────────────────────────────────────────────────
    if not args.skip_baseline:
        log.info("=" * 60)
        log.info("PHASE A: Baseline (PD+AF, no DVFS, max frequency)")
        log.info("=" * 60)

        ret = start_pdaf(dvfs_enabled=False, label="baseline")
        if ret is None:
            log.error("Baseline server failed to start")
            sys.exit(1)
        procs, url = ret

        baseline_results = asyncio.run(run_workload(args.workload, url))

        cleanup_procs(procs)
        kill_our_servers()
        time.sleep(5)

        with open(out_dir / "baseline_results.json", "w") as f:
            json.dump(baseline_results, f, indent=2, default=str)
        print_results("BASELINE (PD+AF, No DVFS)", baseline_results)
    else:
        cached = out_dir / "baseline_results.json"
        if cached.exists():
            baseline_results = json.loads(cached.read_text())
            log.info("Loaded cached baseline from %s", cached)

    # ── DVFS ─────────────────────────────────────────────────────────
    if not args.skip_dvfs:
        log.info("=" * 60)
        log.info("PHASE B: Tier2 DVFS (PD+AF, energy-aware frequency)")
        log.info("=" * 60)

        ret = start_pdaf(dvfs_enabled=True, label="dvfs")
        if ret is None:
            log.error("DVFS server failed to start")
            sys.exit(1)
        procs, url = ret

        dvfs_results = asyncio.run(run_workload(args.workload, url))

        cleanup_procs(procs)
        kill_our_servers()

        with open(out_dir / "dvfs_results.json", "w") as f:
            json.dump(dvfs_results, f, indent=2, default=str)
        print_results("TIER2 DVFS (PD+AF, Energy-Aware)", dvfs_results)
    else:
        cached = out_dir / "dvfs_results.json"
        if cached.exists():
            dvfs_results = json.loads(cached.read_text())
            log.info("Loaded cached DVFS from %s", cached)

    # ── Comparison ───────────────────────────────────────────────────
    if baseline_results and dvfs_results:
        print_comparison(baseline_results, dvfs_results)
        summary = {
            "baseline": baseline_results,
            "dvfs": dvfs_results,
            "energy_saving_pct": round(
                (baseline_results["total_energy_j"] - dvfs_results["total_energy_j"])
                / baseline_results["total_energy_j"] * 100, 1
            ) if baseline_results["total_energy_j"] > 0 else 0,
        }
        with open(out_dir / "comparison.json", "w") as f:
            json.dump(summary, f, indent=2, default=str)
        log.info("Results saved to %s/", out_dir)


if __name__ == "__main__":
    main()
