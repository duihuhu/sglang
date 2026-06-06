#!/usr/bin/env python3
"""Run M=1 and M=3 AFD benchmarks (UCX only, GPUs 4,5) and generate Gantt chart.

Standalone AFD (no PD): DA on GPU 5 receives requests, sends FFN work to DF on GPU 4.
AFD_TIMELINE captured via AFD_DETAILED_TIMING=1.
"""

import concurrent.futures
import json
import logging
import os
import re
import signal
import socket
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("gantt_bench")

PYTHON = "/workspace/env/af-test/bin/python3"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))

DA_PORT = 50020
DF_PORT = 50021

# Use different UCX ports for M=1 and M=3 to avoid "Device is busy"
UCX_BASE_PORT_M1 = 25200
UCX_BASE_PORT_M3 = 25300
SCHED_PORT_M1 = 65400
SCHED_PORT_M3 = 65500

GPU_DF = 4  # FFN
GPU_DA = 5  # Attn

_log_procs = []


def kill_all():
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)


def wait_port(port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2)
            s.close()
            return True
        except Exception:
            time.sleep(2)
    return False


def run_benchmark(m_stage):
    """Start DA+DF, run requests, capture AFD_TIMELINE."""
    global _log_procs
    _log_procs = []

    kill_all()

    da_log = f"/tmp/gantt_DA_m{m_stage}_detailed.log"
    df_log = f"/tmp/gantt_DF_m{m_stage}_detailed.log"
    for f in [da_log, df_log]:
        if os.path.exists(f):
            os.remove(f)

    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_DETAILED_TIMING"] = "1"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["UCX_WARN_UNUSED_ENV_VARS"] = "n"
    env_base["SGLANG_LOG_LEVEL"] = "error"

    extra = [
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", str(m_stage),
        "--max-running-requests", "64",
        "--mem-fraction-static", "0.85",
    ]

    # ── Start DF ──
    df_env = env_base.copy()
    df_env["CUDA_VISIBLE_DEVICES"] = str(GPU_DF)
    df_env["AFD_UCX_BASE_PORT"] = str(UCX_BASE_PORT_M1 if m_stage == 1 else UCX_BASE_PORT_M3)
    df_env["AFD_SCHED_PORT"] = str(SCHED_PORT_M1 if m_stage == 1 else SCHED_PORT_M3)
    df_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(DF_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
    ] + extra

    log.info("M=%d: Starting DF on GPU %d", m_stage, GPU_DF)
    df_fh = open(df_log, "w")
    df_proc = subprocess.Popen(df_cmd, env=df_env,
                               stdout=df_fh, stderr=subprocess.STDOUT,
                               start_new_session=True)
    _log_procs.append(("DF", df_proc, df_fh))

    if not wait_port(DF_PORT, timeout=360):
        log.error("M=%d: DF failed to start", m_stage)
        return False
    log.info("M=%d: DF ready", m_stage)

    # ── Start DA ──
    da_env = env_base.copy()
    da_env["CUDA_VISIBLE_DEVICES"] = str(GPU_DA)
    da_env["AFD_UCX_BASE_PORT"] = str(UCX_BASE_PORT_M1 if m_stage == 1 else UCX_BASE_PORT_M3)
    da_env["AFD_SCHED_PORT"] = str(SCHED_PORT_M1 if m_stage == 1 else SCHED_PORT_M3)
    da_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    da_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(DA_PORT),
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--enable-metrics",
    ] + extra

    log.info("M=%d: Starting DA on GPU %d", m_stage, GPU_DA)
    da_fh = open(da_log, "w")
    da_proc = subprocess.Popen(da_cmd, env=da_env,
                               stdout=da_fh, stderr=subprocess.STDOUT,
                               start_new_session=True)
    _log_procs.append(("DA", da_proc, da_fh))

    if not wait_port(DA_PORT, timeout=360):
        log.error("M=%d: DA failed to start", m_stage)
        return False
    log.info("M=%d: DA ready", m_stage)

    # Wait for AFD UCX connection to be established
    time.sleep(15)

    # ── Warmup ──
    import requests
    warm_payloads = [
        {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(16)
    ]
    log.info("M=%d: Warmup (%d requests)...", m_stage, len(warm_payloads))

    def _send(payload):
        try:
            requests.post(f"http://127.0.0.1:{DA_PORT}/generate",
                         json=payload, timeout=120)
        except Exception:
            pass

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(5)
    log.info("M=%d: Warmup done", m_stage)

    # ── Benchmark requests to generate decode steps ──
    bench_payloads = [
        {"text": f"Tell me something interesting about the number {i}:",
         "sampling_params": {"max_new_tokens": 64, "temperature": 0.0}}
        for i in range(32)
    ]
    log.info("M=%d: Running benchmark (%d requests)...", m_stage, len(bench_payloads))
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_send, bench_payloads))
    time.sleep(15)
    log.info("M=%d: Benchmark done", m_stage)

    # ── Shutdown ──
    for name, proc, fh in _log_procs:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except Exception:
            try:
                proc.terminate()
            except Exception:
                pass
    time.sleep(8)
    for name, proc, fh in _log_procs:
        try:
            proc.kill()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
    _log_procs = []
    # Additional cooldown for UCX resource release
    time.sleep(15)
    kill_all()
    time.sleep(5)

    # ── Verify timeline data ──
    for name, fpath in [("DA", da_log), ("DF", df_log)]:
        try:
            with open(fpath) as f:
                content = f.read()
        except Exception:
            content = ""
        count = len(re.findall(r"\[AFD_TIMELINE\]", content))
        log.info("M=%d: %s log has %d AFD_TIMELINE entries (%d bytes)",
                 m_stage, name, count, len(content))
        if count == 0:
            log.warning("M=%d: No AFD_TIMELINE in %s — check server startup", m_stage, name)

    return True


def main():
    kill_all()

    # ── M=1 ──
    log.info("=" * 60)
    log.info("PHASE 1: M=1 benchmark")
    log.info("=" * 60)
    if not run_benchmark(1):
        log.error("M=1 failed")
        return 1

    # ── M=3 ──
    log.info("=" * 60)
    log.info("PHASE 2: M=3 benchmark")
    log.info("=" * 60)
    if not run_benchmark(3):
        log.error("M=3 failed")
        return 1

    # ── Generate Gantt chart ──
    log.info("=" * 60)
    log.info("Generating Gantt chart...")
    log.info("=" * 60)

    draw_script = os.path.join(HERE, "draw_gantt_actual.py")
    result = subprocess.run(
        [PYTHON, draw_script],
        capture_output=True, text=True, timeout=120,
    )
    log.info("draw_gantt_actual.py stdout:\n%s", result.stdout)
    if result.stderr:
        log.info("draw_gantt_actual.py stderr:\n%s", result.stderr)

    out = os.path.join(HERE, "gantt_m1_vs_m3_combined.png")
    if os.path.exists(out):
        log.info("Chart saved: %s", out)
    else:
        log.error("Chart not found at %s", out)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
