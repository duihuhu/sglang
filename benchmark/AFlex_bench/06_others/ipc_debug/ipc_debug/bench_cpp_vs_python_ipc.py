#!/usr/bin/env python3
"""
Multi-config benchmark: C++ IPC vs Python IPC across different input lengths.
Tests single-request TPOT/TTFT with varying input_len.

Usage:
    /workspace/env/sglang-tier/bin/python benchmark/af_bench/bench_cpp_vs_python_ipc.py
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
log = logging.getLogger("bench")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs_bench_comparison")
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


def start_af_servers(comm_backend, label_suffix=""):
    """Start AF M=1 servers. Returns (url, procs) or (None, procs) on failure."""
    kill_all()
    time.sleep(2)
    # Clean IPC files
    os.system("rm -f /tmp/afd_ipc_cpp_* /dev/shm/afd_ipc_cpp_* /tmp/afd_ipc_50* /dev/shm/afd_ipc_flags_*")

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

    if comm_backend == "ipc_cpp":
        f_env = env_base.copy()
        f_env["CUDA_VISIBLE_DEVICES"] = "4,5"
        f_env["AFD_SCHED_PORT"] = str(sched_port)
        f_env["AFD_IPC_PEER_DEVICE"] = "0"
        f_env["AFD_IPC_SYNC_MODE"] = "cpu_flag"
        f_cmd = [PYTHON, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--tp", "1",
            "--host", "127.0.0.1", "--port", str(f_port),
            "--afd-perspective", "ffn",
            "--afd-comm-backend", "ipc_cpp",
            "--mem-fraction-static", "0.85",
            "--base-gpu-id", "1",
        ] + extra

        a_env = env_base.copy()
        a_env["CUDA_VISIBLE_DEVICES"] = "4,5"
        a_env["AFD_SCHED_PORT"] = str(sched_port)
        a_env["AFD_IPC_PEER_DEVICE"] = "1"
        a_env["AFD_IPC_SYNC_MODE"] = "cpu_flag"
        a_cmd = [PYTHON, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--tp", "1",
            "--host", "127.0.0.1", "--port", str(a_port),
            "--afd-perspective", "attn",
            "--afd-comm-backend", "ipc_cpp",
            "--mem-fraction-static", "0.85",
            "--base-gpu-id", "0",
        ] + extra

    elif comm_backend == "ipc":
        f_env = env_base.copy()
        f_env["CUDA_VISIBLE_DEVICES"] = "4,5"
        f_env["AFD_SCHED_PORT"] = str(sched_port)
        f_env["AFD_IPC_PEER_DEVICE"] = "0"
        f_cmd = [PYTHON, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--tp", "1",
            "--host", "127.0.0.1", "--port", str(f_port),
            "--afd-perspective", "ffn",
            "--afd-comm-backend", "ipc",
            "--mem-fraction-static", "0.85",
            "--base-gpu-id", "1",
        ] + extra

        a_env = env_base.copy()
        a_env["CUDA_VISIBLE_DEVICES"] = "4,5"
        a_env["AFD_SCHED_PORT"] = str(sched_port)
        a_env["AFD_IPC_PEER_DEVICE"] = "1"
        a_cmd = [PYTHON, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--tp", "1",
            "--host", "127.0.0.1", "--port", str(a_port),
            "--afd-perspective", "attn",
            "--afd-comm-backend", "ipc",
            "--mem-fraction-static", "0.85",
            "--base-gpu-id", "0",
        ] + extra

    # Start FFN first
    tag = f"{comm_backend}{label_suffix}"
    fh = open(os.path.join(LOG_DIR, f"{tag}_ffn.log"), "w")
    pf = subprocess.Popen(f_cmd, env=f_env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("FFN", pf, fh))
    time.sleep(3)

    # Start ATTN
    fh2 = open(os.path.join(LOG_DIR, f"{tag}_attn.log"), "w")
    pa = subprocess.Popen(a_cmd, env=a_env, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("ATTN", pa, fh2))

    if not wait_port("127.0.0.1", a_port, timeout=300):
        log.error("%s: ATTN failed to start", tag)
        return None, procs
    if not wait_port("127.0.0.1", f_port, timeout=300):
        log.error("%s: FFN failed to start", tag)
        return None, procs
    if not wait_health(f"http://127.0.0.1:{a_port}/health", timeout=120):
        log.error("%s: health check failed", tag)
        return None, procs

    log.info("%s: servers ready", tag)
    return f"http://127.0.0.1:{a_port}", procs


def cleanup_procs(procs):
    for _, p, f in procs:
        try:
            os.killpg(os.getpgid(p.pid), 9)
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


def benchmark_single_request(url, input_len, output_len=64, n_requests=8, warmup=3):
    """Send sequential requests, measure TTFT/TPOT."""
    import requests as req_lib

    prompt = "Hello world. " * (input_len // 3)

    # Warmup
    for _ in range(warmup):
        try:
            payload = {"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0}}
            req_lib.post(f"{url}/generate", json=payload, timeout=120)
        except Exception:
            pass

    # Benchmark
    ttfts, tpots = [], []
    for i in range(n_requests):
        payload = {"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0}, "stream": True}
        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0
        try:
            resp = req_lib.post(f"{url}/generate", json=payload, stream=True, timeout=120)
            for chunk in resp.iter_lines():
                if chunk:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
            t_end = time.perf_counter()
            if first_token_time and token_count > 1:
                ttfts.append((first_token_time - t0) * 1000)
                tpots.append((t_end - first_token_time) * 1000 / (token_count - 1))
        except Exception as e:
            log.warning("Request %d failed: %s", i, e)

    if not tpots:
        return None

    import numpy as np
    return {
        "ttft_mean": round(np.mean(ttfts), 1),
        "ttft_min": round(np.min(ttfts), 1),
        "tpot_mean": round(np.mean(tpots), 1),
        "tpot_min": round(np.min(tpots), 1),
        "tpot_max": round(np.max(tpots), 1),
        "n_ok": len(tpots),
    }


def main():
    import numpy as np

    print("=" * 100)
    print("  C++ IPC vs Python IPC: Multi-Config Benchmark")
    print("  Model: Qwen3-32B, GPU 4+5 (A800-SXM4-80GB), AF M=1, single request")
    print("=" * 100)
    print()

    # Test configurations: (input_len, output_len)
    configs = [
        (128, 64),
        (256, 64),
        (512, 64),
        (1024, 64),
        (2048, 64),
    ]

    results = {}

    for backend in ["ipc_cpp", "ipc"]:
        log.info(">>> Starting %s backend <<<", backend)
        url, procs = start_af_servers(backend)
        if url is None:
            log.error("Failed to start %s servers", backend)
            cleanup_procs(procs)
            continue

        results[backend] = {}
        for input_len, output_len in configs:
            log.info("  Testing input=%d output=%d ...", input_len, output_len)
            r = benchmark_single_request(url, input_len, output_len, n_requests=8, warmup=3)
            if r:
                results[backend][(input_len, output_len)] = r
                log.info("    TTFT=%.1fms TPOT=%.1fms (n=%d)",
                         r["ttft_mean"], r["tpot_mean"], r["n_ok"])
            else:
                log.warning("    FAILED")

        cleanup_procs(procs)

    # Print comparison table
    print()
    print("=" * 100)
    print("  RESULTS COMPARISON")
    print("=" * 100)
    print()
    print(f"{'Input':<8} {'Output':<8} | {'C++ IPC TTFT':>14} {'C++ IPC TPOT':>14} | {'Py IPC TTFT':>14} {'Py IPC TPOT':>14} | {'TPOT Δ':>10}")
    print("-" * 100)

    for input_len, output_len in configs:
        cpp = results.get("ipc_cpp", {}).get((input_len, output_len))
        py = results.get("ipc", {}).get((input_len, output_len))

        cpp_ttft = f"{cpp['ttft_mean']:.1f}ms" if cpp else "N/A"
        cpp_tpot = f"{cpp['tpot_mean']:.1f}ms" if cpp else "N/A"
        py_ttft = f"{py['ttft_mean']:.1f}ms" if py else "N/A"
        py_tpot = f"{py['tpot_mean']:.1f}ms" if py else "N/A"

        if cpp and py:
            delta = cpp['tpot_mean'] - py['tpot_mean']
            delta_pct = delta / py['tpot_mean'] * 100
            delta_str = f"{delta:+.1f}ms ({delta_pct:+.1f}%)"
        else:
            delta_str = "N/A"

        print(f"{input_len:<8} {output_len:<8} | {cpp_ttft:>14} {cpp_tpot:>14} | {py_ttft:>14} {py_tpot:>14} | {delta_str:>10}")

    print()
    print("  Baseline reference: PD TP=1 TPOT ≈ 45.2ms (decode, single request)")
    print("=" * 100)

    # Save results
    with open(os.path.join(LOG_DIR, "comparison_results.json"), "w") as f:
        json.dump({k: {str(kk): v for kk, v in vv.items()} for k, vv in results.items()}, f, indent=2)
    log.info("Results saved to %s/comparison_results.json", LOG_DIR)


if __name__ == "__main__":
    main()
