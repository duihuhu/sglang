#!/usr/bin/env python3
"""
PD TP=1 and PD TP=2 concurrency sweep (64, 128, 256).
Uses GPUs 4-7. Comparable to PD+AF sweep in run_concurrency_sweep.py.

PD TP=1: P=GPU6, D=GPU4 (2 GPU only, same as AF uses per-perspective)
PD TP=2: P=GPU6,7, D=GPU4,5 (4 GPU total)
"""
import json
import logging
import os
import socket
import subprocess
import sys
import time
import urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pd_sweep")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(HERE, "..", "benchmark_replay.py")
LOG_DIR = os.path.join(HERE, "pd_concurrency_sweep_logs")
os.makedirs(LOG_DIR, exist_ok=True)

BOOTSTRAP = 18999
ALL_PORTS = [50000, 50010, 50020]

INPUT_LEN = 128
OUTPUT_LEN = 100
CONCURRENCY_LEVELS = [64, 128, 256]


def kill_all():
    for t in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(5)
    for t in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
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


def start_pd_tp1():
    """PD TP=1: P=GPU6, D=GPU4."""
    kill_all()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"

    # Prefill: GPU 6
    env_p = env_base.copy()
    env_p["CUDA_VISIBLE_DEVICES"] = "6"
    p_cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", "50010",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
    ]
    fh = open(os.path.join(LOG_DIR, "pd_tp1_p.log"), "w")
    p = subprocess.Popen(p_cmd, env=env_p, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("prefill", p, fh))

    # Decode: GPU 4
    env_d = env_base.copy()
    env_d["CUDA_VISIBLE_DEVICES"] = "4"
    d_cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", "50020",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
    ]
    fh2 = open(os.path.join(LOG_DIR, "pd_tp1_d.log"), "w")
    p2 = subprocess.Popen(d_cmd, env=env_d, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("decode", p2, fh2))

    if not wait_port("127.0.0.1", 50010, 300) or not wait_port("127.0.0.1", 50020, 300):
        log.error("PD TP=1: servers failed to start")
        cleanup_procs(procs)
        return None

    # Router
    router_cmd = [
        PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", "http://127.0.0.1:50010",
        "--decode", "http://127.0.0.1:50020",
        "--host", "127.0.0.1", "--port", "50000",
    ]
    rf = open(os.path.join(LOG_DIR, "pd_tp1_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health("http://127.0.0.1:50000/health", 180):
        log.error("PD TP=1: router failed")
        cleanup_procs(procs)
        return None

    warmup("http://127.0.0.1:50000")
    return procs, "http://127.0.0.1:50000"


def start_pd_tp2():
    """PD TP=2: P=GPU6,7, D=GPU4,5."""
    kill_all()
    time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"

    # Prefill: TP=2 on GPU 6,7
    env_p = env_base.copy()
    env_p["CUDA_VISIBLE_DEVICES"] = "6,7"
    p_cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "2",
        "--host", "127.0.0.1", "--port", "50010",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
    ]
    fh = open(os.path.join(LOG_DIR, "pd_tp2_p.log"), "w")
    p = subprocess.Popen(p_cmd, env=env_p, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("prefill", p, fh))

    # Decode: TP=2 on GPU 4,5
    env_d = env_base.copy()
    env_d["CUDA_VISIBLE_DEVICES"] = "4,5"
    d_cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "2",
        "--host", "127.0.0.1", "--port", "50020",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-radix-cache",
    ]
    fh2 = open(os.path.join(LOG_DIR, "pd_tp2_d.log"), "w")
    p2 = subprocess.Popen(d_cmd, env=env_d, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("decode", p2, fh2))

    if not wait_port("127.0.0.1", 50010, 300) or not wait_port("127.0.0.1", 50020, 300):
        log.error("PD TP=2: servers failed to start")
        cleanup_procs(procs)
        return None

    # Router
    router_cmd = [
        PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", "http://127.0.0.1:50010",
        "--decode", "http://127.0.0.1:50020",
        "--host", "127.0.0.1", "--port", "50000",
    ]
    rf = open(os.path.join(LOG_DIR, "pd_tp2_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health("http://127.0.0.1:50000/health", 180):
        log.error("PD TP=2: router failed")
        cleanup_procs(procs)
        return None

    warmup("http://127.0.0.1:50000")
    return procs, "http://127.0.0.1:50000"


def run_benchmark_at_concurrency(url, concurrency, label):
    """Run benchmark_replay.py at a specific concurrency level."""
    n_requests = concurrency * 3
    output_len = OUTPUT_LEN
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
        ("PD_TP2", start_pd_tp2),
        ("PD_TP1", start_pd_tp1),
    ]

    all_results = {}

    for label, start_fn in configs:
        log.info("\n" + "=" * 80)
        log.info(">>> Starting %s <<<", label)
        log.info("=" * 80)

        sweep_results = []

        for conc in CONCURRENCY_LEVELS:
            log.info("[%s] Starting server for concurrency=%d ...", label, conc)
            ret = start_fn()
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
    print("  PD Concurrency Sweep: TP=1 vs TP=2")
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
        for label, _ in configs:
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

    print("=" * 120)

    # Save results
    output_file = os.path.join(LOG_DIR, "pd_concurrency_sweep_results.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", output_file)


if __name__ == "__main__":
    main()
