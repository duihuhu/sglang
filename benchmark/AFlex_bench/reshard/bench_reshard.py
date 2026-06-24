#!/usr/bin/env python3
"""Reshard end-to-end benchmark: 4GPU TP1 → 8GPU TP2 → 4GPU TP1.

Simulates a workload-driven graceful reload scenario:
  Phase 1: Low QPS on 4GPU PDAF TP1 (baseline)
  Phase 2: Ramp QPS up, stressing the system
  Phase 3: Trigger graceful reload to 8GPU TP2, continue traffic
  Phase 4: High QPS on 8GPU TP2 (verify recovery)
  Phase 5: Reduce QPS, trigger graceful reload back to 4GPU TP1
  Phase 6: Low QPS on 4GPU TP1 (verify energy savings)

Metrics collected per-request: TTFT, TPOT, timestamp, success.
Metrics collected per-phase: throughput, energy, SLO violations.

Usage:
    python bench_reshard.py [--gpus 0,1,2,3,4,5,6,7] [--duration-per-phase 60]
    python bench_reshard.py --dry-run  # Print plan without executing
"""

import argparse
import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import aiohttp
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_reshard")

HERE = Path(__file__).resolve().parent

# ─── Configuration ────────────────────────────────────────────────────────────

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/models_v2"

ROUTER_PORT = 42000
PA_PORT = 42010
PF_PORT = 42011
DA_PORT = 42020
DF_PORT = 42021
BOOTSTRAP_PORT = 49999

# Shadow ports for new deployment during reload
SHADOW_PA_PORT = 42110
SHADOW_PF_PORT = 42111
SHADOW_DA_PORT = 42120
SHADOW_DF_PORT = 42121
SHADOW_BOOTSTRAP_PORT = 49998

TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0

# Fixed workload parameters
INPUT_LEN = 512
OUTPUT_LEN = 128

ALL_PORTS = list(range(42000, 42030)) + list(range(42100, 42130)) + [BOOTSTRAP_PORT, SHADOW_BOOTSTRAP_PORT]


# ─── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RequestResult:
    timestamp: float
    phase: str
    success: bool
    ttft_ms: float = 0.0
    tpot_ms: float = 0.0
    e2e_s: float = 0.0
    tokens: int = 0


@dataclass
class PhaseResult:
    phase: str
    start_time: float
    end_time: float
    qps: float
    tp: int
    ngpu: int
    total_requests: int = 0
    successful: int = 0
    failed: int = 0
    total_tokens: int = 0
    throughput_tok_s: float = 0.0
    ttft_avg_ms: float = 0.0
    ttft_p50_ms: float = 0.0
    ttft_p99_ms: float = 0.0
    tpot_avg_ms: float = 0.0
    tpot_p50_ms: float = 0.0
    tpot_p99_ms: float = 0.0
    total_energy_j: float = 0.0
    energy_per_token_mj: float = 0.0
    slo_violation_rate: float = 0.0
    is_reload: bool = False


# ─── Utility functions ────────────────────────────────────────────────────────

def kill_all():
    """Kill sglang processes on managed ports."""
    import re as _re
    for port in ALL_PORTS:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True)
        for m in _re.finditer(r"pid=(\d+)", r.stdout):
            try:
                os.kill(int(m.group(1)), 9)
            except OSError:
                pass
    subprocess.run(["pkill", "-9", "-f", "sglang_router.launch_router"], capture_output=True)
    time.sleep(3)


def wait_health(port, timeout=300):
    import socket
    import requests
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect(("127.0.0.1", port))
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(3)
    else:
        return False
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}/health", timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


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
    except Exception:
        return {idx: 0 for idx in gpu_indices}


def lock_gpu_freq(gpu_indices, freq_mhz=1410):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-i", str(idx),
                        f"--lock-gpu-clocks={freq_mhz},{freq_mhz}"],
                       capture_output=True)


def unlock_gpu_freq(gpu_indices):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-i", str(idx), "--reset-gpu-clocks"],
                       capture_output=True)


# ─── PDAF Deployment ──────────────────────────────────────────────────────────

class PDAFDeployer:
    """Manages PDAF server lifecycle for a given TP and GPU set."""

    def __init__(self, log_dir: Path, ports=None):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs = []
        # Configurable ports for shadow mode
        self.pa_port = (ports or {}).get("pa", PA_PORT)
        self.pf_port = (ports or {}).get("pf", PF_PORT)
        self.da_port = (ports or {}).get("da", DA_PORT)
        self.df_port = (ports or {}).get("df", DF_PORT)
        self.router_port = (ports or {}).get("router", ROUTER_PORT)
        self.bootstrap_port = (ports or {}).get("bootstrap", BOOTSTRAP_PORT)

    def _base_env(self):
        env = os.environ.copy()
        env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
        env["UCX_LOG_LEVEL"] = "fatal"
        env["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
        env["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
        env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
        env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
        return env

    def _popen(self, name, cmd, env):
        fh = open(self.log_dir / f"{name}.log", "w")
        cmd = ["prlimit", "--memlock=unlimited:unlimited"] + cmd
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        self.procs.append((name, p, fh))
        log.info("  Started %s (PID=%d)", name, p.pid)
        return p

    def cleanup(self):
        for name, p, fh in self.procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            fh.close()
        self.procs.clear()
        time.sleep(3)

    def start(self, p_cvd: str, d_cvd: str, tp: int, micro_batch: int = 2) -> bool:
        """Start full PDAF stack: PF→PA→DF→DA→Router. Returns success."""
        env_base = self._base_env()
        ucx_p = 28200 if self.pa_port == PA_PORT else 28400
        ucx_d = 28300 if self.da_port == DA_PORT else 28500
        sched_p = 68400 if self.pa_port == PA_PORT else 68600
        sched_d = 68500 if self.da_port == DA_PORT else 68700

        common = ["--model-path", MODEL, "--tp", str(tp),
                  "--host", "127.0.0.1",
                  "--afd-comm-backend", "ipc_cpp",
                  "--afd-micro-batch", str(micro_batch),
                  "--afd-dynamic-micro-batch",
                  "--mem-fraction-static", "0.85",
                  "--max-running-requests", "512",
                  "--skip-server-warmup",
                  "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                  "--afd-disagg-interleave-poll",
                  "--disable-radix-cache",
                  "--num-reserved-decode-tokens", "512",
                  "--disaggregation-transfer-backend", "mooncake",
                  "--disaggregation-bootstrap-port", str(self.bootstrap_port),
                  "--disaggregation-ib-device", "mlx5_4",
                  "--enable-metrics"]

        p_gpus = p_cvd.split(",")
        d_gpus = d_cvd.split(",")
        p_ffn_nvml = ",".join(p_gpus[:tp])
        p_attn_nvml = ",".join(p_gpus[tp:])
        d_ffn_nvml = ",".join(d_gpus[:tp])
        d_attn_nvml = ",".join(d_gpus[tp:])

        def _env(cvd, ucx_base, sched_port, peer_device, nvml_idx, ffn_host=None):
            e = env_base.copy()
            e["CUDA_VISIBLE_DEVICES"] = cvd
            e["AFD_UCX_BASE_PORT"] = str(ucx_base)
            e["AFD_SCHED_PORT"] = str(sched_port)
            e["AFD_IPC_SYNC_MODE"] = "ipc_event"
            e["AFD_IPC_PEER_DEVICE"] = str(peer_device)
            e["AFD_NVML_DEVICE_INDEX"] = str(nvml_idx).split(",")[0]
            if ffn_host:
                e["AFD_UCX_FFN_HOST"] = ffn_host
            return e

        def _cmd(port, perspective, disagg, base_gpu_id):
            return [PYTHON, "-m", "sglang.launch_server",
                    "--port", str(port),
                    "--afd-perspective", perspective,
                    "--disaggregation-mode", disagg,
                    "--base-gpu-id", str(base_gpu_id)] + common

        # PF → PA → DF → DA
        self._popen("pf", _cmd(self.pf_port, "ffn", "prefill", 0),
                    _env(p_cvd, ucx_p, sched_p, tp, p_ffn_nvml))
        time.sleep(5)
        self._popen("pa", _cmd(self.pa_port, "attn", "prefill", tp),
                    _env(p_cvd, ucx_p, sched_p, 0, p_attn_nvml, "127.0.0.1"))
        time.sleep(5)
        self._popen("df", _cmd(self.df_port, "ffn", "decode", 0),
                    _env(d_cvd, ucx_d, sched_d, tp, d_ffn_nvml))
        time.sleep(8)
        self._popen("da", _cmd(self.da_port, "attn", "decode", tp),
                    _env(d_cvd, ucx_d, sched_d, 0, d_attn_nvml, "127.0.0.1"))

        log.info("Waiting for PDAF (TP=%d) servers...", tp)
        for port, name in [(self.pa_port, "PA"), (self.da_port, "DA")]:
            if not wait_health(port, 600):
                log.error("  %s (port %d) failed", name, port)
                return False
            log.info("  %s ready", name)
        # PF/DF health depends on A-side; just verify port is open
        import socket
        for port, name in [(self.pf_port, "PF"), (self.df_port, "DF")]:
            deadline = time.time() + 120
            while time.time() < deadline:
                try:
                    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                        s.settimeout(2)
                        s.connect(("127.0.0.1", port))
                        break
                except (ConnectionRefusedError, OSError):
                    time.sleep(3)
            else:
                log.error("  %s (port %d) not listening", name, port)
                return False
            log.info("  %s port open", name)

        # Router
        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation", "--mini-lb",
                 "--prefill", f"http://127.0.0.1:{self.pa_port}",
                 "--decode", f"http://127.0.0.1:{self.da_port}",
                 "--host", "127.0.0.1", "--port", str(self.router_port)]
        self._popen("router", cmd_r, os.environ.copy())
        if not wait_health(self.router_port, 60):
            log.error("Router failed")
            return False
        log.info("PDAF TP%d ready", tp)
        return True


# ─── Workload Sender ──────────────────────────────────────────────────────────

async def send_one_request(session, url, req_id, phase, bench_start, results):
    """Send one request and record per-request metrics."""
    payload = {
        "text": "Explain the theory of " + "x" * INPUT_LEN,
        "sampling_params": {"max_new_tokens": OUTPUT_LEN, "temperature": 0.0},
        "stream": True,
    }
    t0 = time.monotonic()
    timestamp = t0 - bench_start
    first_token_time = None
    token_count = 0

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append(RequestResult(
                    timestamp=timestamp, phase=phase, success=False))
                return
            async for line in resp.content:
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
                        first_token_time = time.monotonic()
                    token_count += 1
                except json.JSONDecodeError:
                    pass
    except Exception:
        results.append(RequestResult(
            timestamp=timestamp, phase=phase, success=False))
        return

    t_end = time.monotonic()
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0
    tpot_ms = 0.0
    if token_count > 1 and first_token_time:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)

    results.append(RequestResult(
        timestamp=timestamp, phase=phase, success=True,
        ttft_ms=ttft_ms, tpot_ms=tpot_ms,
        e2e_s=t_end - t0, tokens=token_count,
    ))


async def run_phase(qps: float, duration_s: float, phase: str,
                    bench_start: float, results: list,
                    gpu_indices: list, router_port: int = None) -> PhaseResult:
    """Run a constant-QPS phase for duration_s seconds."""
    port = router_port or ROUTER_PORT
    url = f"http://127.0.0.1:{port}/generate"
    interval = 1.0 / qps if qps > 0 else 1.0
    n_requests = int(qps * duration_s)

    energy_start = get_gpu_energy_mj(gpu_indices)
    phase_start = time.monotonic()

    timeout = aiohttp.ClientTimeout(total=duration_s + 120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        tasks = []
        for i in range(n_requests):
            await asyncio.sleep(interval)
            task = asyncio.create_task(
                send_one_request(session, url, i, phase, bench_start, results))
            tasks.append(task)

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    phase_end = time.monotonic()
    energy_end = get_gpu_energy_mj(gpu_indices)
    total_energy_j = sum((energy_end.get(i, 0) - energy_start.get(i, 0)) / 1000.0
                         for i in gpu_indices)

    # Compute phase stats from results
    phase_results = [r for r in results if r.phase == phase]
    ok = [r for r in phase_results if r.success]
    fail_count = sum(1 for r in phase_results if not r.success)
    total_tokens = sum(r.tokens for r in ok)
    duration = phase_end - phase_start

    ttfts = [r.ttft_ms for r in ok if r.ttft_ms > 0]
    tpots = [r.tpot_ms for r in ok if r.tpot_ms > 0]

    return PhaseResult(
        phase=phase,
        start_time=phase_start - bench_start,
        end_time=phase_end - bench_start,
        qps=qps,
        tp=0, ngpu=len(gpu_indices),
        total_requests=n_requests,
        successful=len(ok),
        failed=fail_count,
        total_tokens=total_tokens,
        throughput_tok_s=round(total_tokens / duration, 1) if duration > 0 else 0,
        ttft_avg_ms=round(float(np.mean(ttfts)), 1) if ttfts else 0,
        ttft_p50_ms=round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        ttft_p99_ms=round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        tpot_avg_ms=round(float(np.mean(tpots)), 1) if tpots else 0,
        tpot_p50_ms=round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        tpot_p99_ms=round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        total_energy_j=round(total_energy_j, 1),
        energy_per_token_mj=round(total_energy_j * 1000 / total_tokens, 2) if total_tokens > 0 else 0,
        slo_violation_rate=round(
            (sum(1 for r in ok if r.ttft_ms > TTFT_SLO_MS) +
             sum(1 for r in ok if r.tpot_ms > TPOT_SLO_MS) + fail_count)
            / max(len(phase_results), 1) * 100, 1),
    )


# ─── Graceful Reload Trigger ─────────────────────────────────────────────────

def trigger_graceful_reload(old_deployer, old_tp, new_tp, old_gpus, new_gpus,
                            log_dir, drain_timeout=15):
    """Graceful reload: start shadow deployment, then switch traffic.

    Returns (new_deployer, reload_duration_s) on success, or (None, duration) on failure.
    The old deployer is killed AFTER the new one is ready.
    Traffic continues flowing to old_deployer's router during the transition.
    """
    import requests

    log.info("=" * 60)
    log.info("RESHARD: TP%d (%d GPU) → TP%d (%d GPU)", old_tp, len(old_gpus), new_tp, len(new_gpus))
    log.info("=" * 60)

    reload_start = time.monotonic()

    # Shadow ports so new modules don't conflict with running ones
    shadow_ports = {
        "pa": SHADOW_PA_PORT, "pf": SHADOW_PF_PORT,
        "da": SHADOW_DA_PORT, "df": SHADOW_DF_PORT,
        "router": ROUTER_PORT + 100,  # 42100
        "bootstrap": SHADOW_BOOTSTRAP_PORT,
    }

    # Compute GPU assignments for new TP
    new_p_cvd = ",".join(str(g) for g in new_gpus[:new_tp * 2])
    new_d_cvd = ",".join(str(g) for g in new_gpus[new_tp * 2:])

    # Step 1: Start shadow deployment on different ports (old still serving)
    log.info("  Starting shadow deployment (TP%d) while old serves...", new_tp)
    shadow_deployer = PDAFDeployer(log_dir, ports=shadow_ports)
    if not shadow_deployer.start(new_p_cvd, new_d_cvd, tp=new_tp):
        log.error("  Shadow deployment failed!")
        shadow_deployer.cleanup()
        return None, time.monotonic() - reload_start

    # Step 2: Shadow is ready. Now kill old deployment.
    log.info("  Shadow ready! Switching traffic...")
    old_deployer.cleanup()
    time.sleep(2)

    # Step 3: Start final deployment on primary ports (shadow freed GPUs)
    # Actually since shadow uses different ports, we promote shadow as primary.
    # The benchmark's run_phase sends to ROUTER_PORT, so we need primary ports.
    # Kill shadow, restart on primary ports. OR just update the traffic target.
    # Simpler: kill shadow + start fresh on primary ports using the new GPUs.
    # BUT that doubles the reload time. Better approach: keep shadow alive,
    # and send traffic to shadow router port during phase 4.
    # Return the shadow deployer so the caller can use its router_port.

    reload_duration = time.monotonic() - reload_start
    log.info("  Reload completed in %.1fs (shadow on port %d)", reload_duration, shadow_ports["router"])
    return shadow_deployer, reload_duration


# ─── Main Benchmark ──────────────────────────────────────────────────────────

def run_benchmark(gpus: list, duration_per_phase: int = 60):
    """Run the full reshard benchmark."""
    assert len(gpus) >= 8, "Need at least 8 GPUs for this benchmark"

    gpus_4 = gpus[:4]
    gpus_8 = gpus[:8]
    results_dir = HERE / "results"
    results_dir.mkdir(exist_ok=True)
    log_dir = HERE / "logs"

    all_results: List[RequestResult] = []
    phase_results: List[PhaseResult] = []
    events = []  # (timestamp, event_name)

    bench_start = time.monotonic()
    timestamp = lambda: time.monotonic() - bench_start

    # ─── Phase 1: 4GPU TP1, low QPS ──────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 1: Deploy 4GPU PDAF TP1, low QPS")
    log.info("=" * 60)

    kill_all()
    lock_gpu_freq(gpus_8)
    deployer = PDAFDeployer(log_dir / "phase1")
    p_cvd = ",".join(str(g) for g in gpus_4[:2])
    d_cvd = ",".join(str(g) for g in gpus_4[2:])
    if not deployer.start(p_cvd, d_cvd, tp=1):
        log.error("Phase 1 deployment failed")
        return

    events.append((timestamp(), "deploy_4gpu_tp1"))
    pr = asyncio.run(run_phase(
        qps=1.0, duration_s=duration_per_phase, phase="phase1_low",
        bench_start=bench_start, results=all_results, gpu_indices=gpus_4))
    pr.tp = 1
    pr.ngpu = 4
    phase_results.append(pr)
    log.info("  Phase 1 done: throughput=%.1f tok/s, TTFT_avg=%.1f ms, energy=%.1f J",
             pr.throughput_tok_s, pr.ttft_avg_ms, pr.total_energy_j)

    # ─── Phase 2: 4GPU TP1, ramp QPS ─────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 2: Increase QPS (stress 4GPU TP1)")
    log.info("=" * 60)

    events.append((timestamp(), "ramp_qps"))
    pr = asyncio.run(run_phase(
        qps=3.0, duration_s=duration_per_phase, phase="phase2_stress",
        bench_start=bench_start, results=all_results, gpu_indices=gpus_4))
    pr.tp = 1
    pr.ngpu = 4
    phase_results.append(pr)
    log.info("  Phase 2 done: throughput=%.1f tok/s, TTFT_avg=%.1f ms, SLO_viol=%.1f%%",
             pr.throughput_tok_s, pr.ttft_avg_ms, pr.slo_violation_rate)

    # ─── Phase 3: Graceful reload to 8GPU TP2 ────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 3: Graceful reload → 8GPU TP2")
    log.info("=" * 60)

    events.append((timestamp(), "reload_start_4to8"))
    reload_start = time.monotonic()

    # Run traffic AND reload concurrently:
    # - Traffic continues to old router (some requests may fail during switch)
    # - Reload happens in parallel: drain → wait idle → kill old → start new
    import concurrent.futures

    reload_done_event = {"done": False, "duration": 0, "success": False}

    def _do_reload_4to8():
        """Reload in background thread while traffic continues."""
        t0 = time.monotonic()
        # Drain: old requests finish, new ones get 503
        import requests as req
        try:
            req.post(f"http://127.0.0.1:{ROUTER_PORT}/admin/drain_module", json={
                "prefill_urls": [f"http://127.0.0.1:{PA_PORT}"],
                "decode_urls": [f"http://127.0.0.1:{DA_PORT}"],
            }, timeout=5)
            log.info("  [reload] Router drain started")
        except Exception as e:
            log.warning("  [reload] Drain request failed: %s", e)

        # Wait for inflight to complete (poll /is_idle on PA and DA)
        idle_deadline = time.time() + 30
        while time.time() < idle_deadline:
            try:
                r = req.get(f"http://127.0.0.1:{PA_PORT}/is_idle", timeout=3)
                if r.status_code == 200 and r.json().get("idle"):
                    break
            except Exception:
                break
            time.sleep(1)
        log.info("  [reload] Old modules drained/idle")

        # Kill old deployment
        deployer.cleanup()
        time.sleep(2)
        log.info("  [reload] Old deployment killed, starting new...")

        # Start new on primary ports
        new_dep = PDAFDeployer(log_dir / "phase3")
        new_p_cvd = ",".join(str(g) for g in gpus_8[:4])
        new_d_cvd = ",".join(str(g) for g in gpus_8[4:])
        ok = new_dep.start(new_p_cvd, new_d_cvd, tp=2)
        reload_done_event["duration"] = time.monotonic() - t0
        reload_done_event["done"] = True
        reload_done_event["success"] = ok
        return new_dep if ok else None

    # Launch reload in a thread while continuing to send traffic
    with concurrent.futures.ThreadPoolExecutor(1) as executor:
        reload_future = executor.submit(_do_reload_4to8)

        # Send traffic during reload — some will fail (503) during drain/switch
        pr_reload = asyncio.run(run_phase(
            qps=3.0, duration_s=90, phase="phase3_during_reload",
            bench_start=bench_start, results=all_results, gpu_indices=gpus_8))
        pr_reload.tp = 0  # mixed
        pr_reload.ngpu = 0

        # Wait for reload to finish if not already
        new_deployer = reload_future.result(timeout=300)

    if new_deployer is None:
        log.error("Phase 3 deployment failed")
        return

    deployer = new_deployer
    reload_duration = reload_done_event["duration"]
    events.append((timestamp(), f"reload_done_4to8 ({reload_duration:.0f}s)"))
    log.info("  Reload completed in %.1fs", reload_duration)

    # Record reload traffic and reload event
    phase_results.append(pr_reload)
    phase_results.append(PhaseResult(
        phase="reload_4to8", start_time=reload_start - bench_start,
        end_time=time.monotonic() - bench_start, qps=0, tp=2, ngpu=8,
        is_reload=True,
    ))

    # ─── Phase 4: 8GPU TP2, high QPS ─────────────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 4: High QPS on 8GPU TP2")
    log.info("=" * 60)

    events.append((timestamp(), "high_qps_8gpu"))
    pr = asyncio.run(run_phase(
        qps=5.0, duration_s=duration_per_phase, phase="phase4_high",
        bench_start=bench_start, results=all_results, gpu_indices=gpus_8))
    pr.tp = 2
    pr.ngpu = 8
    phase_results.append(pr)
    log.info("  Phase 4 done: throughput=%.1f tok/s, TTFT_avg=%.1f ms, energy=%.1f J",
             pr.throughput_tok_s, pr.ttft_avg_ms, pr.total_energy_j)

    # ─── Phase 5: Reduce QPS, then reload back to 4GPU TP1 ───────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 5: Reduce QPS, reload → 4GPU TP1")
    log.info("=" * 60)

    events.append((timestamp(), "reduce_qps"))
    pr = asyncio.run(run_phase(
        qps=1.5, duration_s=duration_per_phase // 2, phase="phase5_reduce",
        bench_start=bench_start, results=all_results, gpu_indices=gpus_8))
    pr.tp = 2
    pr.ngpu = 8
    phase_results.append(pr)
    log.info("  Phase 5 pre-reload: throughput=%.1f, energy_per_token=%.2f mJ",
             pr.throughput_tok_s, pr.energy_per_token_mj)

    # Reload back — with concurrent traffic
    events.append((timestamp(), "reload_start_8to4"))
    reload_start = time.monotonic()

    reload_done_event2 = {"done": False, "duration": 0, "success": False}

    def _do_reload_8to4():
        t0 = time.monotonic()
        import requests as req
        # Drain
        try:
            req.post(f"http://127.0.0.1:{deployer.router_port}/admin/drain_module", json={
                "prefill_urls": [f"http://127.0.0.1:{deployer.pa_port}"],
                "decode_urls": [f"http://127.0.0.1:{deployer.da_port}"],
            }, timeout=5)
            log.info("  [reload] Router drain started (8→4)")
        except Exception as e:
            log.warning("  [reload] Drain failed: %s", e)

        # Wait idle
        idle_deadline = time.time() + 30
        while time.time() < idle_deadline:
            try:
                r = req.get(f"http://127.0.0.1:{deployer.pa_port}/is_idle", timeout=3)
                if r.status_code == 200 and r.json().get("idle"):
                    break
            except Exception:
                break
            time.sleep(1)
        log.info("  [reload] Old modules drained (8→4)")

        deployer.cleanup()
        kill_all()
        time.sleep(2)

        new_dep = PDAFDeployer(log_dir / "phase5")
        p_cvd = ",".join(str(g) for g in gpus_4[:2])
        d_cvd = ",".join(str(g) for g in gpus_4[2:])
        ok = new_dep.start(p_cvd, d_cvd, tp=1)
        reload_done_event2["duration"] = time.monotonic() - t0
        reload_done_event2["done"] = True
        reload_done_event2["success"] = ok
        return new_dep if ok else None

    with concurrent.futures.ThreadPoolExecutor(1) as executor:
        reload_future2 = executor.submit(_do_reload_8to4)

        # Continue sending traffic (will get 503 during switch window)
        pr_reload2 = asyncio.run(run_phase(
            qps=1.5, duration_s=75, phase="phase5_during_reload",
            bench_start=bench_start, results=all_results,
            gpu_indices=gpus_4, router_port=deployer.router_port))
        pr_reload2.tp = 0
        pr_reload2.ngpu = 0

        new_deployer2 = reload_future2.result(timeout=300)

    if new_deployer2 is None:
        log.error("Phase 5 re-deploy failed")
        return
    deployer = new_deployer2

    reload_duration = reload_done_event2["duration"]
    events.append((timestamp(), f"reload_done_8to4 ({reload_duration:.0f}s)"))
    log.info("  Reload back completed in %.1fs", reload_duration)

    phase_results.append(pr_reload2)
    phase_results.append(PhaseResult(
        phase="reload_8to4", start_time=reload_start - bench_start,
        end_time=time.monotonic() - bench_start, qps=0, tp=1, ngpu=4,
        is_reload=True,
    ))

    # ─── Phase 6: 4GPU TP1, low QPS (verify energy saving) ───────────────────
    log.info("\n" + "=" * 60)
    log.info("PHASE 6: Low QPS on 4GPU TP1 (energy saving)")
    log.info("=" * 60)

    events.append((timestamp(), "low_qps_4gpu_final"))
    pr = asyncio.run(run_phase(
        qps=1.0, duration_s=duration_per_phase, phase="phase6_low",
        bench_start=bench_start, results=all_results, gpu_indices=gpus_4))
    pr.tp = 1
    pr.ngpu = 4
    phase_results.append(pr)
    log.info("  Phase 6 done: throughput=%.1f tok/s, energy=%.1f J, energy_per_token=%.2f mJ",
             pr.throughput_tok_s, pr.total_energy_j, pr.energy_per_token_mj)

    # ─── Cleanup & Save ──────────────────────────────────────────────────────
    deployer.cleanup()
    unlock_gpu_freq(gpus_8)

    total_time = time.monotonic() - bench_start
    log.info("\n" + "=" * 60)
    log.info("BENCHMARK COMPLETE (total %.0fs)", total_time)
    log.info("=" * 60)

    # Save results
    output = {
        "benchmark": "reshard_graceful_reload",
        "model": MODEL,
        "total_duration_s": round(total_time, 1),
        "gpus": gpus,
        "input_len": INPUT_LEN,
        "output_len": OUTPUT_LEN,
        "phases": [asdict(pr) for pr in phase_results],
        "events": events,
        "requests": [asdict(r) for r in all_results],
    }

    out_path = results_dir / "reshard_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Results saved to %s", out_path)

    # Print summary table
    print("\n" + "─" * 90)
    print(f"{'Phase':<20} {'QPS':>5} {'TP':>3} {'GPU':>4} {'Thpt':>8} "
          f"{'TTFT_avg':>9} {'TPOT_avg':>9} {'Energy':>8} {'E/tok':>8} {'SLO%':>6}")
    print("─" * 90)
    for pr in phase_results:
        if pr.is_reload:
            dur = pr.end_time - pr.start_time
            print(f"{'⟳ ' + pr.phase:<20} {'---':>5} {pr.tp:>3} {pr.ngpu:>4} "
                  f"{'---':>8} {'---':>9} {'---':>9} {'---':>8} {'---':>8} "
                  f"{dur:>5.0f}s")
        else:
            print(f"{pr.phase:<20} {pr.qps:>5.1f} {pr.tp:>3} {pr.ngpu:>4} "
                  f"{pr.throughput_tok_s:>7.1f} "
                  f"{pr.ttft_avg_ms:>8.1f} {pr.tpot_avg_ms:>8.1f} "
                  f"{pr.total_energy_j:>7.1f} {pr.energy_per_token_mj:>7.2f} "
                  f"{pr.slo_violation_rate:>5.1f}")
    print("─" * 90)

    return out_path


# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Reshard graceful reload benchmark")
    parser.add_argument("--gpus", type=str, default="0,1,2,3,4,5,6,7",
                        help="Comma-separated GPU indices (need 8)")
    parser.add_argument("--duration-per-phase", type=int, default=60,
                        help="Duration of each traffic phase in seconds")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print plan without executing")
    args = parser.parse_args()

    gpus = [int(x) for x in args.gpus.split(",")]
    assert len(gpus) >= 8, f"Need at least 8 GPUs, got {len(gpus)}"

    if args.dry_run:
        print("Benchmark plan:")
        print(f"  GPUs: {gpus}")
        print(f"  Phase duration: {args.duration_per_phase}s")
        print(f"  Phase 1: 4GPU TP1, QPS=1.0 ({args.duration_per_phase}s)")
        print(f"  Phase 2: 4GPU TP1, QPS=3.0 ({args.duration_per_phase}s) [stress]")
        print(f"  Phase 3: Graceful reload → 8GPU TP2")
        print(f"  Phase 4: 8GPU TP2, QPS=5.0 ({args.duration_per_phase}s)")
        print(f"  Phase 5: 8GPU TP2, QPS=1.5 ({args.duration_per_phase//2}s) + reload → 4GPU")
        print(f"  Phase 6: 4GPU TP1, QPS=1.0 ({args.duration_per_phase}s) [energy check]")
        print(f"  Estimated total: ~{args.duration_per_phase * 5}s + reload time")
        return

    run_benchmark(gpus, duration_per_phase=args.duration_per_phase)


if __name__ == "__main__":
    main()
