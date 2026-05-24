#!/usr/bin/env python3
"""
Test PD+AF with C++ IPC backend (ipc_cpp).
Launches ATTN + FFN servers on GPU 4,5 with ipc_cpp backend,
sends a few requests, measures TPOT.

Usage:
    /workspace/env/sglang-tier/bin/python benchmark/af_bench/test_cpp_ipc_e2e.py
"""
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("test_cpp_ipc")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs_cpp_ipc")
os.makedirs(LOG_DIR, exist_ok=True)


def kill_all():
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
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
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def run_af_ipc_cpp():
    """Launch AF M=1 with ipc_cpp backend on GPU 4 (ATTN) + GPU 5 (FFN)."""
    kill_all()
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

    a_port, f_port = 30000, 30001
    sched_port = 67500

    extra = [
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", "1",
        "--max-running-requests", "64",
    ]

    # FFN GPU 5 (start first - IPC server side)
    f_env = env_base.copy()
    f_env["CUDA_VISIBLE_DEVICES"] = "4,5"  # Both GPUs visible for IPC
    f_env["AFD_SCHED_PORT"] = str(sched_port)
    f_env["AFD_IPC_PEER_DEVICE"] = "0"  # peer is cuda:0 (physical GPU 4)
    f_env["AFD_IPC_SYNC_MODE"] = "cpu_flag"
    f_env["SGLANG_SET_CUDA_DEVICE"] = "1"  # Use cuda:1 (physical GPU 5)
    f_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(f_port),
        "--afd-perspective", "ffn",
        "--afd-comm-backend", "ipc_cpp",
        "--mem-fraction-static", "0.85",
        "--base-gpu-id", "1",
    ] + extra
    log.info("Starting FFN server on GPU 5...")
    fh = open(os.path.join(LOG_DIR, "ffn.log"), "w")
    pf = subprocess.Popen(f_cmd, env=f_env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("FFN", pf, fh))

    time.sleep(3)

    # ATTN GPU 4 (connects to FFN)
    a_env = env_base.copy()
    a_env["CUDA_VISIBLE_DEVICES"] = "4,5"  # Both GPUs visible for IPC
    a_env["AFD_SCHED_PORT"] = str(sched_port)
    a_env["AFD_IPC_PEER_DEVICE"] = "1"  # peer is cuda:1 (physical GPU 5)
    a_env["AFD_IPC_SYNC_MODE"] = "cpu_flag"
    a_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(a_port),
        "--afd-perspective", "attn",
        "--afd-comm-backend", "ipc_cpp",
        "--mem-fraction-static", "0.85",
        "--base-gpu-id", "0",
    ] + extra
    log.info("Starting ATTN server on GPU 4...")
    fh2 = open(os.path.join(LOG_DIR, "attn.log"), "w")
    pa = subprocess.Popen(a_cmd, env=a_env, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("ATTN", pa, fh2))

    # Wait for servers
    log.info("Waiting for servers to start...")
    if not wait_port("127.0.0.1", a_port, timeout=300):
        log.error("ATTN server failed to start")
        # Print last 50 lines of log
        os.system(f"tail -50 {LOG_DIR}/attn.log")
        return None, procs
    if not wait_port("127.0.0.1", f_port, timeout=300):
        log.error("FFN server failed to start")
        os.system(f"tail -50 {LOG_DIR}/ffn.log")
        return None, procs

    if not wait_health(f"http://127.0.0.1:{a_port}/health", timeout=120):
        log.error("ATTN health check failed")
        return None, procs

    log.info("Both servers ready!")
    return f"http://127.0.0.1:{a_port}", procs


def benchmark_requests(url, n_requests=10, input_len=512, output_len=32):
    """Send requests and measure TTFT/TPOT."""
    import requests
    import concurrent.futures

    results = []

    def send_one(i):
        prompt = "Hello " * (input_len // 2)
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": output_len,
                "temperature": 0.0,
            },
            "stream": True,
        }
        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0

        try:
            resp = requests.post(f"{url}/generate", json=payload, stream=True, timeout=120)
            for chunk in resp.iter_lines():
                if chunk:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
            t_end = time.perf_counter()

            ttft = (first_token_time - t0) * 1000 if first_token_time else None
            if token_count > 1 and first_token_time:
                tpot = (t_end - first_token_time) * 1000 / (token_count - 1)
            else:
                tpot = None

            return {"ttft_ms": ttft, "tpot_ms": tpot, "tokens": token_count, "success": True}
        except Exception as e:
            return {"error": str(e), "success": False}

    # Warmup
    log.info("Warming up (3 requests)...")
    for i in range(3):
        send_one(i)

    # Benchmark (sequential for accurate TPOT)
    log.info(f"Benchmarking ({n_requests} requests, input~{input_len}, output={output_len})...")
    for i in range(n_requests):
        r = send_one(i)
        results.append(r)
        if r["success"]:
            log.info(f"  req {i}: TTFT={r['ttft_ms']:.1f}ms, TPOT={r['tpot_ms']:.1f}ms, tokens={r['tokens']}")

    return results


def main():
    log.info("=== PD+AF with C++ IPC Backend Test ===")

    url, procs = run_af_ipc_cpp()
    if url is None:
        log.error("Failed to start servers")
        for _, p, f in procs:
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:
                pass
        return

    try:
        results = benchmark_requests(url, n_requests=10, input_len=512, output_len=32)

        ok = [r for r in results if r["success"]]
        if ok:
            ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms")]
            tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms")]

            print("\n" + "=" * 60)
            print("  AF M=1 + C++ IPC (cpu_flag) Results")
            print("=" * 60)
            if ttfts:
                print(f"  TTFT: mean={sum(ttfts)/len(ttfts):.1f}ms, "
                      f"min={min(ttfts):.1f}ms, max={max(ttfts):.1f}ms")
            if tpots:
                print(f"  TPOT: mean={sum(tpots)/len(tpots):.1f}ms, "
                      f"min={min(tpots):.1f}ms, max={max(tpots):.1f}ms")
            print(f"  Success: {len(ok)}/{len(results)}")
            print("=" * 60)

            # Compare with known baselines
            print("\n  Comparison (single request, Qwen3-32B, 512 input):")
            print(f"    PD TP=1 baseline:     TPOT = 45.2 ms")
            print(f"    Python IPC (pre-opt): TPOT = 92.2 ms (+104%)")
            print(f"    Python IPC (opt):     TPOT = 80.8 ms (+79%)")
            if tpots:
                mean_tpot = sum(tpots) / len(tpots)
                overhead = (mean_tpot - 45.2) / 45.2 * 100
                print(f"    C++ IPC (this test):  TPOT = {mean_tpot:.1f} ms (+{overhead:.0f}%)")
        else:
            log.error("All requests failed!")
            for r in results:
                log.error(f"  {r}")

    finally:
        log.info("Cleaning up...")
        for _, p, f in procs:
            try:
                os.killpg(os.getpgid(p.pid), 9)
            except Exception:
                pass
            try:
                f.close()
            except Exception:
                pass
        time.sleep(3)


if __name__ == "__main__":
    main()
