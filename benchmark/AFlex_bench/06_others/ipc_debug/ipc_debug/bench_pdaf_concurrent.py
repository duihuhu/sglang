#!/usr/bin/env python3
"""
4-GPU PD+AF multi-concurrency benchmark: C++ IPC vs Python IPC.

Tests throughput and latency under varying concurrency levels.
GPU layout: PF(GPU4) + PA(GPU5) | DF(GPU6) + DA(GPU7)

Usage:
    /workspace/env/sglang-tier/bin/python benchmark/af_bench/bench_pdaf_concurrent.py
"""
import json
import logging
import os
import socket
import subprocess
import sys
import time
import concurrent.futures

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pdaf_conc")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs_pdaf_concurrent")
os.makedirs(LOG_DIR, exist_ok=True)
BOOTSTRAP = 18999

INPUT_LEN = 512
OUTPUT_LEN = 256


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


def warmup_server(url, n=12):
    import requests
    payloads = [
        {"text": f"Warmup {i} " * 30, "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
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
    """Launch 4-GPU PD+AF on GPUs 4-7."""
    kill_all()
    os.system("rm -f /tmp/afd_ipc_cpp_* /dev/shm/afd_ipc_cpp_* /dev/shm/afd_ipc_flags_*")
    time.sleep(2)

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
        "--max-running-requests", "128",
    ]

    def _start(name, gpus, perspective, disagg_mode, port, sched_port,
               base_gpu_id=0, peer_device=None):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        env["AFD_SCHED_PORT"] = str(sched_port)
        env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
        env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
        if comm_backend in ("ipc_cpp", "ipc"):
            if peer_device is not None:
                env["AFD_IPC_PEER_DEVICE"] = str(peer_device)
        if comm_backend == "ipc_cpp":
            env["AFD_IPC_SYNC_MODE"] = "ipc_event"

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

    # Prefill: PF=GPU4(id=0), PA=GPU5(id=1), both see "4,5"
    _start("pf", "4,5", "ffn", "prefill", pf_port, sched_p, base_gpu_id=0, peer_device=1)
    time.sleep(2)
    _start("pa", "4,5", "attn", "prefill", pa_port, sched_p, base_gpu_id=1, peer_device=0)

    log.info("Waiting for prefill servers...")
    if not wait_port("127.0.0.1", pa_port, timeout=300) or not wait_port("127.0.0.1", pf_port, timeout=300):
        log.error("Prefill servers failed")
        return None, procs

    # Decode: DF=GPU6(id=0), DA=GPU7(id=1), both see "6,7"
    _start("df", "6,7", "ffn", "decode", df_port, sched_d, base_gpu_id=0, peer_device=1)
    time.sleep(2)
    _start("da", "6,7", "attn", "decode", da_port, sched_d, base_gpu_id=1, peer_device=0)

    log.info("Waiting for decode servers...")
    if not wait_port("127.0.0.1", da_port, timeout=300) or not wait_port("127.0.0.1", df_port, timeout=300):
        log.error("Decode servers failed")
        return None, procs

    if not wait_health(f"http://127.0.0.1:{pa_port}/health", timeout=120):
        log.error("PA health failed")
        return None, procs
    if not wait_health(f"http://127.0.0.1:{da_port}/health", timeout=120):
        log.error("DA health failed")
        return None, procs
    time.sleep(5)

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

    log.info("%s: system ready!", label)
    return f"http://127.0.0.1:{router_port}", procs


def cleanup_procs(procs):
    for _, p, f in procs:
        try:
            os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            try: p.kill()
            except: pass
        try: f.close()
        except: pass
    time.sleep(3)


def bench_concurrent(url, concurrency, n_requests, input_len=INPUT_LEN, output_len=OUTPUT_LEN):
    """Run concurrent requests, measure real TPOT via streaming."""
    import requests as req_lib
    import numpy as np

    prompt = "Hello world. " * (input_len // 3)

    def send_one(i):
        payload = {"text": prompt, "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0}, "stream": True}
        t0 = time.perf_counter()
        first_token_time = None
        token_count = 0
        try:
            resp = req_lib.post(f"{url}/generate", json=payload, stream=True, timeout=300)
            for chunk in resp.iter_lines():
                if chunk:
                    token_count += 1
                    if first_token_time is None:
                        first_token_time = time.perf_counter()
            t_end = time.perf_counter()
            if first_token_time and token_count > 1:
                ttft = (first_token_time - t0) * 1000
                tpot = (t_end - first_token_time) * 1000 / (token_count - 1)
                return {
                    "success": True,
                    "latency_ms": (t_end - t0) * 1000,
                    "ttft_ms": ttft,
                    "tpot_ms": tpot,
                    "output_tokens": token_count,
                }
            else:
                return {"success": False, "error": "no tokens"}
        except Exception as e:
            return {"success": False, "error": str(e)}

    wall_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(send_one, i) for i in range(n_requests)]
        results = [f.result() for f in futures]
    wall_end = time.perf_counter()
    wall_s = wall_end - wall_start

    ok = [r for r in results if r["success"]]
    if not ok:
        return None

    tpots = np.array([r["tpot_ms"] for r in ok])
    ttfts = np.array([r["ttft_ms"] for r in ok])
    total_output_tokens = sum(r["output_tokens"] for r in ok)
    throughput = total_output_tokens / wall_s

    return {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_ok": len(ok),
        "throughput_tok_s": round(throughput, 1),
        "tpot_mean_ms": round(np.mean(tpots), 1),
        "tpot_p50_ms": round(np.median(tpots), 1),
        "tpot_p95_ms": round(np.percentile(tpots, 95), 1),
        "ttft_mean_ms": round(np.mean(ttfts), 1),
        "ttft_p50_ms": round(np.median(ttfts), 1),
        "wall_s": round(wall_s, 1),
    }


def main():
    import numpy as np

    print("=" * 120)
    print("  PD+AF 4-GPU Multi-Concurrency Benchmark: C++ IPC vs Python IPC")
    print(f"  Model: Qwen3-32B | GPUs 4-7 (A800-SXM4-80GB) | Input={INPUT_LEN} Output={OUTPUT_LEN}")
    print("=" * 120)
    print()

    concurrency_levels = [1, 4, 16, 32, 64]
    n_requests_map = {1: 10, 4: 40, 16: 80, 32: 128, 64: 192}

    backends = ["ipc_cpp"]
    all_results = {}

    for backend in backends:
        label = f"pdaf_conc_{backend}"
        log.info(">>> Starting PD+AF with %s <<<", backend)

        url, procs = run_pdaf(backend, label)
        if url is None:
            log.error("Failed to start %s", backend)
            cleanup_procs(procs)
            continue

        warmup_server(url, n=16)
        all_results[backend] = {}

        for conc in concurrency_levels:
            n_req = n_requests_map[conc]
            log.info("  [%s] concurrency=%d n_requests=%d ...", backend, conc, n_req)
            r = bench_concurrent(url, conc, n_req)
            if r:
                all_results[backend][conc] = r
                log.info("    TPOT_p50=%.1fms  TTFT_p50=%.1fms  throughput=%.1f tok/s  ok=%d/%d",
                         r["tpot_p50_ms"], r["ttft_p50_ms"], r["throughput_tok_s"], r["n_ok"], r["n_requests"])
            else:
                log.warning("    FAILED")

        cleanup_procs(procs)

    # Print comparison
    print()
    print("=" * 120)
    print("  RESULTS: PD+AF 4-GPU Multi-Concurrency (ipc_event mode)")
    print(f"  Input={INPUT_LEN}, Output={OUTPUT_LEN}")
    print("=" * 120)
    print()
    print(f"{'Conc':<6} | {'Thru(tok/s)':>12} {'TPOT_mean':>12} {'TPOT_p50':>12} {'TPOT_p95':>12} {'TTFT_mean':>12} {'TTFT_p50':>12} | {'ok':>6}")
    print("-" * 120)

    for conc in concurrency_levels:
        cpp = all_results.get("ipc_cpp", {}).get(conc)

        if cpp:
            print(f"{conc:<6} | {cpp['throughput_tok_s']:>12.1f} {cpp['tpot_mean_ms']:>10.1f}ms {cpp['tpot_p50_ms']:>10.1f}ms "
                  f"{cpp['tpot_p95_ms']:>10.1f}ms {cpp['ttft_mean_ms']:>10.1f}ms {cpp['ttft_p50_ms']:>10.1f}ms | "
                  f"{cpp['n_ok']:>4}/{cpp['n_requests']}")
        else:
            print(f"{conc:<6} | {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12} | {'N/A':>6}")

    print("=" * 120)

    # Save
    out_path = os.path.join(LOG_DIR, "concurrent_results.json")
    with open(out_path, "w") as f:
        json.dump({k: {str(kk): v for kk, v in vv.items()} for k, vv in all_results.items()}, f, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
