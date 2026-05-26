#!/usr/bin/env python3
"""
PD+AF Concurrency Sweep: M=1 vs M=2 async pipeline.

Tests TTFT, TPOT, and output throughput across concurrency levels 1..1024.
Uses GPUs 4-7 (DF=4, DA=5, PF=6, PA=7).
"""
import asyncio
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("conc_sweep")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(HERE, "..", "benchmark_replay.py")
LOG_DIR = os.path.join(HERE, "concurrency_sweep_logs")
os.makedirs(LOG_DIR, exist_ok=True)

BOOTSTRAP = 18999
ALL_PORTS = [50000, 50010, 50011, 50020, 50021]

INPUT_LEN = 128
OUTPUT_LEN = 200
# DF KV cache capacity ~64751 tokens. Max safe decode batch = 64751 / (INPUT+OUTPUT).
# For output=200: 64751/328 ≈ 197. For output=128: 64751/256 ≈ 252.
# Strategy: use output=200 for conc<=128, output=100 for conc>=256 (safe batch ~360).
CONCURRENCY_LEVELS = [64, 128, 256]


def kill_all():
    for t in ["sglang.launch_server", "sglang_router", "sglang::router"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(5)
    for t in ["sglang.launch_server", "sglang_router", "sglang::router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
        if not any(f":{p}" in out for p in ALL_PORTS):
            break
        time.sleep(3)


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


def warmup(url, n=8):
    import requests
    import concurrent.futures

    payloads = [
        {"text": f"Hello {i}", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
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


def start_pdaf(micro_batch: int, async_pipeline: bool = False):
    """Start PD+AF system with 4 GPUs (4,5,6,7)."""
    kill_all()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["SGLANG_DISAGGREGATION_QUEUE_SIZE"] = "32"
    env_base["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"
    if async_pipeline:
        env_base["AFD_ASYNC_PIPELINE"] = "1"

    extra = [
        "--skip-server-warmup",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", str(micro_batch),
        "--afd-disagg-interleave-poll",
        "--disable-radix-cache",
        "--num-reserved-decode-tokens", "64",
        "--max-running-requests", "200",
    ]
    if async_pipeline:
        extra.append("--afd-async-pipeline")

    def _start(name, perspective, disagg_mode, port, ucx_base, sched_port,
               visible_gpus, base_gpu_id, peer_device, ffn_host=None):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = visible_gpus
        env["AFD_UCX_BASE_PORT"] = str(ucx_base)
        env["AFD_SCHED_PORT"] = str(sched_port)
        env["AFD_IPC_SYNC_MODE"] = "ipc_event"
        env["AFD_IPC_PEER_DEVICE"] = str(peer_device)
        env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
        env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
        if ffn_host:
            env["AFD_UCX_FFN_HOST"] = ffn_host
        cmd = [
            PYTHON, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--tp", "1",
            "--host", "127.0.0.1", "--port", str(port),
            "--afd-perspective", perspective,
            "--afd-comm-backend", "ipc_cpp",
            "--mem-fraction-static", "0.85",
            "--disaggregation-mode", disagg_mode,
            "--disaggregation-transfer-backend", "mooncake",
            "--disaggregation-bootstrap-port", str(BOOTSTRAP),
            "--disaggregation-ib-device", "mlx5_4",
            "--base-gpu-id", str(base_gpu_id),
        ] + extra
        tag = f"m{micro_batch}_{'async' if async_pipeline else 'sync'}"
        fh = open(os.path.join(LOG_DIR, f"sweep_{tag}_{name}.log"), "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    # Prefill: GPU 6 (FFN), GPU 7 (Attn)
    # Decode:  GPU 4 (FFN), GPU 5 (Attn)
    p_vis, d_vis = "6,7", "4,5"
    _start("pf", "ffn", "prefill", 50011, 25200, 65400, p_vis, 0, 1)
    time.sleep(2)
    _start("pa", "attn", "prefill", 50010, 25200, 65400, p_vis, 1, 0, "127.0.0.1")
    if not wait_port("127.0.0.1", 50010, 300) or not wait_port("127.0.0.1", 50011, 300):
        log.error("Prefill servers failed to start")
        cleanup_procs(procs)
        return None

    _start("df", "ffn", "decode", 50021, 25300, 65500, d_vis, 0, 1)
    time.sleep(2)
    _start("da", "attn", "decode", 50020, 25300, 65500, d_vis, 1, 0, "127.0.0.1")
    if not wait_port("127.0.0.1", 50020, 300) or not wait_port("127.0.0.1", 50021, 300):
        log.error("Decode servers failed to start")
        cleanup_procs(procs)
        return None

    time.sleep(5)

    # Router
    router_cmd = [
        PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", "http://127.0.0.1:50010",
        "--decode", "http://127.0.0.1:50020",
        "--host", "127.0.0.1", "--port", "50000",
    ]
    rf = open(os.path.join(LOG_DIR, f"sweep_m{micro_batch}_router.log"), "w")
    rp = subprocess.Popen(
        router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True
    )
    procs.append(("router", rp, rf))
    if not wait_health("http://127.0.0.1:50000/health", 180):
        log.error("Router failed to start")
        cleanup_procs(procs)
        return None

    warmup("http://127.0.0.1:50000")
    return procs, "http://127.0.0.1:50000"


def run_benchmark_at_concurrency(url, concurrency, label):
    """Run benchmark_replay.py at a specific concurrency level."""
    # Use enough requests to reach steady state
    n_requests = concurrency * 3

    # DF capacity ~64751 tokens. With max_running_requests=200:
    # 200*(128+100)=45600 < 64751, safe.
    output_len = 100

    # QPS: ramp up gradually to avoid Mooncake transfer burst.
    # Fill concurrency window over ~4 seconds.
    qps = max(concurrency // 4, 16)

    dump_file = os.path.join(LOG_DIR, f"result_{label}_conc{concurrency}.json")
    cmd = [
        PYTHON, BENCHMARK,
        "--dataset", "sample",
        "--url", url,
        "--max-requests", str(n_requests),
        "--concurrency", str(concurrency),
        "--sample-input-len", str(INPUT_LEN),
        "--sample-output-len", str(output_len),
        "--sample-qps", str(qps),
        "--timeout", "600",
        "--dump", dump_file,
        "--scenario-label", f"{label}_conc{concurrency}",
    ]
    log.info("  Running conc=%d, n_requests=%d, output=%d, qps=%d", concurrency, n_requests, output_len, qps)
    subprocess.run(cmd, cwd=HERE, capture_output=True)

    if not os.path.exists(dump_file):
        log.error("  Result file missing: %s", dump_file)
        return None

    with open(dump_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    ok = [r for r in results if r.get("success")]
    if not ok:
        return None

    wall = data.get("wall_duration_s", 1) or 1
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    out_thru = total_out / wall

    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]

    record = {
        "concurrency": concurrency,
        "n_requests": n_requests,
        "n_ok": len(ok),
        "output_thru_tok_s": round(out_thru, 1),
        "wall_s": round(wall, 1),
    }

    if ttfts:
        ttfts_sorted = sorted(ttfts)
        m = len(ttfts_sorted)
        record["mean_ttft_ms"] = round(sum(ttfts) / m, 2)
        record["p50_ttft_ms"] = round(ttfts_sorted[m // 2], 2)
        record["p95_ttft_ms"] = round(ttfts_sorted[int(m * 0.95)], 2)
        record["p99_ttft_ms"] = round(ttfts_sorted[int(m * 0.99)], 2)

    if tpots:
        tpots_sorted = sorted(tpots)
        k = len(tpots_sorted)
        record["mean_tpot_ms"] = round(sum(tpots) / k, 2)
        record["p50_tpot_ms"] = round(tpots_sorted[k // 2], 2)
        record["p95_tpot_ms"] = round(tpots_sorted[int(k * 0.95)], 2)
        record["p99_tpot_ms"] = round(tpots_sorted[int(k * 0.99)], 2)

    return record


def main():
    configs = [
        ("M2_async", 2, True),
        ("M1", 1, False),
    ]

    all_results = {}

    for label, m, async_pipe in configs:
        log.info("\n" + "=" * 80)
        log.info(">>> Starting PD+AF %s <<<", label)
        log.info("=" * 80)

        sweep_results = []

        for conc in CONCURRENCY_LEVELS:
            log.info("[%s] Starting server for concurrency=%d ...", label, conc)
            ret = start_pdaf(m, async_pipeline=async_pipe)
            if not ret:
                log.error("%s conc=%d: Failed to start system", label, conc)
                kill_all()
                time.sleep(10)
                sweep_results.append({"concurrency": conc, "error": True})
                continue

            procs, url = ret
            log.info("[%s] Testing concurrency=%d ...", label, conc)
            r = run_benchmark_at_concurrency(url, conc, label)
            if r:
                sweep_results.append(r)
                log.info(
                    "  [%s] conc=%d: TTFT=%.1fms TPOT=%.1fms Thru=%.1f tok/s (%d ok)",
                    label, conc,
                    r.get("mean_ttft_ms", 0),
                    r.get("mean_tpot_ms", 0),
                    r["output_thru_tok_s"],
                    r["n_ok"],
                )
            else:
                log.warning("  [%s] conc=%d: FAILED", label, conc)
                sweep_results.append({"concurrency": conc, "error": True})

            cleanup_procs(procs)
            kill_all()
            time.sleep(10)

        all_results[label] = sweep_results

    # Print summary table
    print("\n" + "=" * 120)
    print("  PD+AF Concurrency Sweep: M=1 vs M=2 async pipeline")
    print(f"  Model: {MODEL}")
    print(f"  Input={INPUT_LEN}, Output={OUTPUT_LEN}")
    print("=" * 120)

    header = (
        f"  {'Conc':>6} | {'Config':<12} | {'TTFT(ms)':>10} | {'TPOT(ms)':>10} | "
        f"{'OutThru(tok/s)':>15} | {'P95_TTFT':>10} | {'P95_TPOT':>10} | {'OK':>6}"
    )
    print(header)
    print("  " + "-" * 110)

    for conc in CONCURRENCY_LEVELS:
        for label, _, _ in configs:
            results = all_results.get(label, [])
            r = next((x for x in results if x.get("concurrency") == conc), None)
            if r and not r.get("error"):
                print(
                    f"  {conc:>6} | {label:<12} | "
                    f"{r.get('mean_ttft_ms', 0):>10.1f} | "
                    f"{r.get('mean_tpot_ms', 0):>10.1f} | "
                    f"{r['output_thru_tok_s']:>15.1f} | "
                    f"{r.get('p95_ttft_ms', 0):>10.1f} | "
                    f"{r.get('p95_tpot_ms', 0):>10.1f} | "
                    f"{r['n_ok']:>6}"
                )
            elif r:
                print(f"  {conc:>6} | {label:<12} | {'FAILED':^70}")
        if conc < CONCURRENCY_LEVELS[-1]:
            print("  " + "-" * 110)

    # M=2 vs M=1 comparison
    print("\n  --- M=2 vs M=1 Improvement ---")
    print(f"  {'Conc':>6} | {'TTFT':>12} | {'TPOT':>12} | {'Throughput':>12}")
    print("  " + "-" * 50)
    m1_results = {r["concurrency"]: r for r in all_results.get("M1", []) if not r.get("error")}
    m2_results = {r["concurrency"]: r for r in all_results.get("M2_async", []) if not r.get("error")}
    for conc in CONCURRENCY_LEVELS:
        m1 = m1_results.get(conc)
        m2 = m2_results.get(conc)
        if m1 and m2:
            ttft_d = ((m2.get("mean_ttft_ms", 0) / m1.get("mean_ttft_ms", 1)) - 1) * 100 if m1.get("mean_ttft_ms") else 0
            tpot_d = ((m2.get("mean_tpot_ms", 0) / m1.get("mean_tpot_ms", 1)) - 1) * 100 if m1.get("mean_tpot_ms") else 0
            thru_d = ((m2["output_thru_tok_s"] / m1["output_thru_tok_s"]) - 1) * 100 if m1["output_thru_tok_s"] else 0
            print(f"  {conc:>6} | {ttft_d:>+10.1f}% | {tpot_d:>+10.1f}% | {thru_d:>+10.1f}%")

    print("=" * 120)

    # Save results
    output_file = os.path.join(LOG_DIR, "concurrency_sweep_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
