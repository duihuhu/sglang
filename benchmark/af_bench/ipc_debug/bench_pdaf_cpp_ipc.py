#!/usr/bin/env python3
"""
4-GPU PD+AF benchmark with C++ IPC backend.

Architecture (same as run_comparison.py pdaf_m1):
  GPU 4: Decode-FFN (DF)
  GPU 5: Decode-ATTN (DA)
  GPU 6: Prefill-FFN (PF)
  GPU 7: Prefill-ATTN (PA)

PD transfer: Mooncake (RDMA) for KV cache
AF transfer: ipc_cpp (NVLink) for hidden states

Comparison: ipc_cpp vs ipc (Python) vs ucx

Usage:
    /workspace/env/sglang-tier/bin/python benchmark/af_bench/bench_pdaf_cpp_ipc.py
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
log = logging.getLogger("pdaf_bench")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs_pdaf_cpp_ipc")
os.makedirs(LOG_DIR, exist_ok=True)
BOOTSTRAP = 18999


def kill_all():
    for t in ["sglang.launch_server", "sglang_router", "sglang::router"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(5)
    for t in ["sglang.launch_server", "sglang_router", "sglang::router"]:
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


def warmup_server(url, n=8):
    import requests
    import concurrent.futures
    payloads = [
        {"text": f"Hello world {i}, warmup:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(n)
    ]
    def _send(p):
        try:
            requests.post(f"{url}/generate", json=p, timeout=120)
        except Exception:
            pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_send, payloads))
    time.sleep(3)


def run_pdaf(comm_backend, label):
    """Launch 4-GPU PD+AF system.

    GPU layout:
      Prefill pair: PF=GPU6, PA=GPU7 (IPC between 6↔7)
      Decode pair:  DF=GPU4, DA=GPU5 (IPC between 4↔5)
      PD transfer:  Mooncake RDMA (PA→DA KV cache)
    """
    kill_all()
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["UCX_LOG_LEVEL"] = "fatal"

    da_port, df_port = 50020, 50021
    pa_port, pf_port = 50010, 50011
    router_port = 50000

    sched_p = 65400
    sched_d = 65500

    extra = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", "1",
        "--max-running-requests", "96",
    ]

    def _start(name, gpus, perspective, disagg_mode, port, sched_port,
               base_gpu_id=0, peer_device=None):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["AFD_SCHED_PORT"] = str(sched_port)
        env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
        env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"

        if comm_backend == "ipc_cpp":
            env["AFD_IPC_PEER_DEVICE"] = str(peer_device) if peer_device is not None else "0"
            env["AFD_IPC_SYNC_MODE"] = "cpu_flag"

        cmd = [PYTHON, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--tp", "1",
            "--host", "127.0.0.1", "--port", str(port),
            "--afd-perspective", perspective,
            "--afd-comm-backend", comm_backend,
            "--mem-fraction-static", "0.85",
            "--disaggregation-mode", disagg_mode,
            "--disaggregation-transfer-backend", "mooncake",
            "--disaggregation-bootstrap-port", str(BOOTSTRAP),
            "--disaggregation-ib-device", "mlx5_4",
            "--base-gpu-id", str(base_gpu_id),
        ] + extra

        fh = open(os.path.join(LOG_DIR, f"{label}_{name}.log"), "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))
        return p

    # Prefill pair: PF=GPU6, PA=GPU7 (both see GPUs 6,7)
    # PF is FFN on gpu_id=0 (physical 6), peer=1 (physical 7)
    # PA is ATTN on gpu_id=1 (physical 7), peer=0 (physical 6)
    _start("pf", "6,7", "ffn", "prefill", pf_port, sched_p,
           base_gpu_id=0, peer_device=1)
    time.sleep(2)
    _start("pa", "6,7", "attn", "prefill", pa_port, sched_p,
           base_gpu_id=1, peer_device=0)

    log.info("Waiting for prefill servers...")
    if not wait_port("127.0.0.1", pa_port, timeout=300):
        log.error("PA failed to start")
        return None, procs
    if not wait_port("127.0.0.1", pf_port, timeout=300):
        log.error("PF failed to start")
        return None, procs

    # Decode pair: DF=GPU4, DA=GPU5 (both see GPUs 4,5)
    # DF is FFN on gpu_id=0 (physical 4), peer=1 (physical 5)
    # DA is ATTN on gpu_id=1 (physical 5), peer=0 (physical 4)
    _start("df", "4,5", "ffn", "decode", df_port, sched_d,
           base_gpu_id=0, peer_device=1)
    time.sleep(2)
    _start("da", "4,5", "attn", "decode", da_port, sched_d,
           base_gpu_id=1, peer_device=0)

    log.info("Waiting for decode servers...")
    if not wait_port("127.0.0.1", da_port, timeout=300):
        log.error("DA failed to start")
        return None, procs
    if not wait_port("127.0.0.1", df_port, timeout=300):
        log.error("DF failed to start")
        return None, procs

    # Health checks
    if not wait_health(f"http://127.0.0.1:{pa_port}/health", timeout=120):
        log.error("PA health failed")
        return None, procs
    if not wait_health(f"http://127.0.0.1:{da_port}/health", timeout=120):
        log.error("DA health failed")
        return None, procs

    time.sleep(5)  # Mooncake bootstrap

    # Router
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", f"http://127.0.0.1:{pa_port}",
        "--decode", f"http://127.0.0.1:{da_port}",
        "--host", "127.0.0.1", "--port", str(router_port),
    ]
    rf = open(os.path.join(LOG_DIR, f"{label}_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))

    if not wait_health(f"http://127.0.0.1:{router_port}/health", timeout=120):
        log.error("Router health failed")
        return None, procs

    log.info("%s: all servers + router ready!", label)
    return f"http://127.0.0.1:{router_port}", procs


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


def benchmark(url, input_len, output_len, n_requests=10, warmup=3):
    """Sequential single-request benchmark."""
    import requests as req_lib

    prompt = "Hello world. " * (input_len // 3)

    # Warmup
    for _ in range(warmup):
        try:
            payload = {"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0}}
            req_lib.post(f"{url}/generate", json=payload, timeout=180)
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
            resp = req_lib.post(f"{url}/generate", json=payload, stream=True, timeout=180)
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
        "ttft_p50": round(np.median(ttfts), 1),
        "tpot_mean": round(np.mean(tpots), 1),
        "tpot_p50": round(np.median(tpots), 1),
        "tpot_min": round(np.min(tpots), 1),
        "n_ok": len(tpots),
    }


def main():
    import numpy as np

    print("=" * 110)
    print("  PD+AF 4-GPU Benchmark: C++ IPC vs Python IPC")
    print("  Model: Qwen3-32B, A800-SXM4-80GB × 4")
    print("  Layout: PF(GPU6) + PA(GPU7) | DF(GPU4) + DA(GPU5) | Router")
    print("  PD: Mooncake RDMA | AF: IPC (NVLink)")
    print("=" * 110)
    print()

    # Test configs: (input_len, output_len)
    configs = [
        (128, 128),
        (512, 128),
        (1024, 128),
        (2048, 128),
    ]

    backends = ["ipc_cpp", "ipc"]
    results = {}

    for backend in backends:
        label = f"pdaf_{backend}"
        log.info(">>> Starting PD+AF with %s backend <<<", backend)

        url, procs = run_pdaf(backend, label)
        if url is None:
            log.error("Failed to start %s", label)
            cleanup_procs(procs)
            continue

        warmup_server(url, n=8)
        results[backend] = {}

        for input_len, output_len in configs:
            log.info("  [%s] input=%d output=%d ...", backend, input_len, output_len)
            r = benchmark(url, input_len, output_len, n_requests=8, warmup=3)
            if r:
                results[backend][(input_len, output_len)] = r
                log.info("    TTFT=%.1fms TPOT=%.1fms (n=%d)",
                         r["ttft_mean"], r["tpot_mean"], r["n_ok"])
            else:
                log.warning("    FAILED")

        cleanup_procs(procs)

    # Print comparison
    print()
    print("=" * 110)
    print("  PD+AF 4-GPU RESULTS (single request, sequential)")
    print("=" * 110)
    print()
    print(f"{'Input':<8} {'Output':<8} | {'C++ IPC TTFT':>14} {'C++ IPC TPOT':>14} | {'Py IPC TTFT':>14} {'Py IPC TPOT':>14} | {'TPOT Δ':>16}")
    print("-" * 110)

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

        print(f"{input_len:<8} {output_len:<8} | {cpp_ttft:>14} {cpp_tpot:>14} | {py_ttft:>14} {py_tpot:>14} | {delta_str:>16}")

    print()
    print("  Reference: PD-only TP=1 (2 GPU) TPOT ≈ 67ms, AF-only M=1 (2 GPU) TPOT ≈ 80.8ms (Python IPC)")
    print("=" * 110)

    # Save
    with open(os.path.join(LOG_DIR, "pdaf_results.json"), "w") as f:
        json.dump({k: {str(kk): v for kk, v in vv.items()} for k, vv in results.items()}, f, indent=2)
    log.info("Results saved to %s/pdaf_results.json", LOG_DIR)


if __name__ == "__main__":
    main()
