#!/usr/bin/env python3
"""
Multi input/output length sweep for all 4 architectures.
Tests different (input_len, output_len) combinations at increasing concurrency
until throughput saturates or requests fail.

Architectures: PD_TP2, AF_M2, AF_M1, PD_TP1
GPU: 4-7 (same as other sweeps)
Logs go to logs/len_sweep_logs/
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
log = logging.getLogger("len_sweep")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(HERE, "..", "benchmark_replay.py")
LOG_DIR = os.path.join(HERE, "logs", "len_sweep_logs")
os.makedirs(LOG_DIR, exist_ok=True)

BOOTSTRAP = 18999
ALL_PORTS = [50000, 50010, 50011, 50020, 50021]

# (input_len, output_len) combinations to test
LEN_CONFIGS = [
    (128, 128),
    (256, 256),
    (512, 128),
    (512, 512),
    (1024, 256),
    (1024, 1024),
    (2048, 512),
]

# Concurrency levels: start low, double until failure
CONC_LEVELS = [1, 4, 16, 64, 128, 256, 512, 1024]


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
        {"text": f"Hello {i}", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}}
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


# ── Server launchers ──────────────────────────────────────────────────────

def start_pd_tp2():
    kill_all(); time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"

    env_p = env_base.copy(); env_p["CUDA_VISIBLE_DEVICES"] = "6,7"
    p_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "2",
        "--host", "127.0.0.1", "--port", "50010",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-radix-cache"]
    fh = open(os.path.join(LOG_DIR, "pd_tp2_p.log"), "w")
    p = subprocess.Popen(p_cmd, env=env_p, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("prefill", p, fh))

    env_d = env_base.copy(); env_d["CUDA_VISIBLE_DEVICES"] = "4,5"
    d_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "2",
        "--host", "127.0.0.1", "--port", "50020",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-radix-cache"]
    fh2 = open(os.path.join(LOG_DIR, "pd_tp2_d.log"), "w")
    p2 = subprocess.Popen(d_cmd, env=env_d, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("decode", p2, fh2))

    if not wait_port("127.0.0.1", 50010, 300) or not wait_port("127.0.0.1", 50020, 300):
        cleanup_procs(procs); return None
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", "http://127.0.0.1:50010", "--decode", "http://127.0.0.1:50020",
        "--host", "127.0.0.1", "--port", "50000"]
    rf = open(os.path.join(LOG_DIR, "pd_tp2_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health("http://127.0.0.1:50000/health", 180):
        cleanup_procs(procs); return None
    warmup("http://127.0.0.1:50000")
    return procs, "http://127.0.0.1:50000"


def start_pd_tp1():
    kill_all(); time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"

    env_p = env_base.copy(); env_p["CUDA_VISIBLE_DEVICES"] = "6"
    p_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", "50010",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-radix-cache"]
    fh = open(os.path.join(LOG_DIR, "pd_tp1_p.log"), "w")
    p = subprocess.Popen(p_cmd, env=env_p, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("prefill", p, fh))

    env_d = env_base.copy(); env_d["CUDA_VISIBLE_DEVICES"] = "4"
    d_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", "50020",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--disable-radix-cache"]
    fh2 = open(os.path.join(LOG_DIR, "pd_tp1_d.log"), "w")
    p2 = subprocess.Popen(d_cmd, env=env_d, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("decode", p2, fh2))

    if not wait_port("127.0.0.1", 50010, 300) or not wait_port("127.0.0.1", 50020, 300):
        cleanup_procs(procs); return None
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", "http://127.0.0.1:50010", "--decode", "http://127.0.0.1:50020",
        "--host", "127.0.0.1", "--port", "50000"]
    rf = open(os.path.join(LOG_DIR, "pd_tp1_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health("http://127.0.0.1:50000/health", 180):
        cleanup_procs(procs); return None
    warmup("http://127.0.0.1:50000")
    return procs, "http://127.0.0.1:50000"


def start_pdaf(micro_batch, async_pipeline=False):
    kill_all(); time.sleep(3)
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

    extra = ["--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
             "--afd-micro-batch", str(micro_batch), "--afd-disagg-interleave-poll",
             "--disable-radix-cache", "--num-reserved-decode-tokens", "64",
             "--max-running-requests", "200"]
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
        cmd = [PYTHON, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--tp", "1",
            "--host", "127.0.0.1", "--port", str(port),
            "--afd-perspective", perspective, "--afd-comm-backend", "ipc_cpp",
            "--mem-fraction-static", "0.85",
            "--disaggregation-mode", disagg_mode,
            "--disaggregation-transfer-backend", "mooncake",
            "--disaggregation-bootstrap-port", str(BOOTSTRAP),
            "--disaggregation-ib-device", "mlx5_4",
            "--base-gpu-id", str(base_gpu_id)] + extra
        tag = f"m{micro_batch}_{'async' if async_pipeline else 'sync'}"
        fh = open(os.path.join(LOG_DIR, f"af_{tag}_{name}.log"), "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    _start("pf", "ffn", "prefill", 50011, 25200, 65400, "6,7", 0, 1)
    time.sleep(2)
    _start("pa", "attn", "prefill", 50010, 25200, 65400, "6,7", 1, 0, "127.0.0.1")
    if not wait_port("127.0.0.1", 50010, 300) or not wait_port("127.0.0.1", 50011, 300):
        cleanup_procs(procs); return None
    _start("df", "ffn", "decode", 50021, 25300, 65500, "4,5", 0, 1)
    time.sleep(2)
    _start("da", "attn", "decode", 50020, 25300, 65500, "4,5", 1, 0, "127.0.0.1")
    if not wait_port("127.0.0.1", 50020, 300) or not wait_port("127.0.0.1", 50021, 300):
        cleanup_procs(procs); return None
    time.sleep(5)
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", "http://127.0.0.1:50010", "--decode", "http://127.0.0.1:50020",
        "--host", "127.0.0.1", "--port", "50000"]
    rf = open(os.path.join(LOG_DIR, f"af_m{micro_batch}_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health("http://127.0.0.1:50000/health", 180):
        cleanup_procs(procs); return None
    warmup("http://127.0.0.1:50000")
    return procs, "http://127.0.0.1:50000"


# ── Benchmark runner ──────────────────────────────────────────────────────

def run_benchmark(url, concurrency, input_len, output_len, label):
    """Run benchmark and return metrics dict or None on failure."""
    # Adaptive request count: fewer requests for long sequences
    est_time_per_req = (input_len + output_len) * 0.045  # ~45ms per token
    if concurrency <= 4:
        n_requests = max(concurrency * 3, 8)
    else:
        n_requests = max(concurrency * 2, 20)
    # Cap to avoid extremely long runs
    max_wall_est = n_requests / max(concurrency, 1) * est_time_per_req
    if max_wall_est > 300:
        n_requests = max(int(300 * concurrency / est_time_per_req), concurrency)

    qps = max(concurrency // 4, 8)
    timeout_s = max(int(max_wall_est * 2), 600)

    dump_file = os.path.join(LOG_DIR, f"result_{label}_in{input_len}_out{output_len}_conc{concurrency}.json")
    cmd = [
        PYTHON, BENCHMARK,
        "--dataset", "sample", "--url", url,
        "--max-requests", str(n_requests),
        "--concurrency", str(concurrency),
        "--sample-input-len", str(input_len),
        "--sample-output-len", str(output_len),
        "--sample-qps", str(qps),
        "--timeout", str(timeout_s),
        "--dump", dump_file,
        "--scenario-label", f"{label}_in{input_len}_out{output_len}_conc{concurrency}",
    ]
    try:
        subprocess.run(cmd, cwd=HERE, capture_output=True, timeout=timeout_s + 120)
    except subprocess.TimeoutExpired:
        return None

    if not os.path.exists(dump_file):
        return None
    with open(dump_file) as f:
        data = json.load(f)

    results = data.get("results", [])
    ok = [r for r in results if r.get("success")]
    if not ok or len(ok) < n_requests * 0.3:
        return None

    wall = data.get("wall_duration_s", 1) or 1
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    out_thru = total_out / wall

    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]

    record = {
        "config": label, "input_len": input_len, "output_len": output_len,
        "concurrency": concurrency, "n_requests": n_requests, "n_ok": len(ok),
        "output_thru_tok_s": round(out_thru, 1), "wall_s": round(wall, 1),
    }
    if ttfts:
        s = sorted(ttfts); m = len(s)
        record["mean_ttft_ms"] = round(sum(ttfts)/m, 2)
        record["p50_ttft_ms"] = round(s[m//2], 2)
        record["p95_ttft_ms"] = round(s[int(m*0.95)], 2)
    if tpots:
        s = sorted(tpots); k = len(s)
        record["mean_tpot_ms"] = round(sum(tpots)/k, 2)
        record["p50_tpot_ms"] = round(s[k//2], 2)
        record["p95_tpot_ms"] = round(s[int(k*0.95)], 2)
    return record


def compute_max_conc(input_len, output_len, kv_capacity=64751):
    """Estimate max safe concurrency based on KV cache capacity."""
    tokens_per_req = input_len + output_len
    # TP=2 has 2x capacity; AF uses single GPU decode (same as TP=1 capacity)
    max_batch_tp2 = kv_capacity * 2 // tokens_per_req
    max_batch_tp1 = kv_capacity // tokens_per_req
    return max_batch_tp2, max_batch_tp1


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    configs = [
        ("PD_TP2", lambda: start_pd_tp2()),
        ("AF_M2", lambda: start_pdaf(2, async_pipeline=True)),
        ("AF_M1", lambda: start_pdaf(1, async_pipeline=False)),
        ("PD_TP1", lambda: start_pd_tp1()),
    ]

    all_results = []

    for label, start_fn in configs:
        log.info("\n" + "=" * 80)
        log.info(">>> %s <<<", label)
        log.info("=" * 80)

        for input_len, output_len in LEN_CONFIGS:
            max_tp2, max_tp1 = compute_max_conc(input_len, output_len)
            if "TP2" in label:
                cap = min(max_tp2, 1024)
            else:
                cap = min(max_tp1, 1024)

            # Filter concurrency levels to those within capacity
            concs = [c for c in CONC_LEVELS if c <= cap]
            if not concs:
                log.warning("[%s] in=%d out=%d: no feasible concurrency (cap=%d)",
                           label, input_len, output_len, cap)
                continue

            log.info("[%s] in=%d out=%d (cap≈%d), testing conc=%s",
                     label, input_len, output_len, cap, concs)

            # Start server once per (config, len_combo)
            log.info("[%s] Starting server...", label)
            ret = start_fn()
            if not ret:
                log.error("[%s] Failed to start for in=%d out=%d", label, input_len, output_len)
                kill_all(); time.sleep(10)
                continue

            procs, url = ret
            prev_thru = 0

            for conc in concs:
                log.info("  [%s] in=%d out=%d conc=%d ...", label, input_len, output_len, conc)
                r = run_benchmark(url, conc, input_len, output_len, label)
                if r:
                    all_results.append(r)
                    thru = r["output_thru_tok_s"]
                    log.info("    TTFT=%.1f TPOT=%.1f Thru=%.1f (%d/%d)",
                             r.get("mean_ttft_ms", 0), r.get("mean_tpot_ms", 0),
                             thru, r["n_ok"], r["n_requests"])
                    # Stop if throughput saturated (< 5% gain) or declining
                    if prev_thru > 0 and thru < prev_thru * 0.9:
                        log.info("    Throughput declining, stopping this config.")
                        break
                    prev_thru = thru
                else:
                    log.warning("    FAILED at conc=%d, stopping.", conc)
                    break
                time.sleep(3)

            cleanup_procs(procs)
            kill_all()
            time.sleep(10)

    # Save results
    import csv
    output_json = os.path.join(LOG_DIR, "len_sweep_results.json")
    with open(output_json, "w") as f:
        json.dump(all_results, f, indent=2)

    output_csv = os.path.join(HERE, "results", "len_sweep_results.csv")
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    fields = ["config", "input_len", "output_len", "concurrency", "n_requests", "n_ok",
              "mean_ttft_ms", "p50_ttft_ms", "p95_ttft_ms",
              "mean_tpot_ms", "p50_tpot_ms", "p95_tpot_ms",
              "output_thru_tok_s", "wall_s"]
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for r in all_results:
            writer.writerow(r)

    log.info("Done! %d records saved.", len(all_results))
    log.info("  JSON: %s", output_json)
    log.info("  CSV:  %s", output_csv)

    # Print summary table
    print("\n" + "=" * 130)
    print(f"  Length Sweep Results ({len(all_results)} records)")
    print("=" * 130)
    print(f"  {'Config':<8} | {'In':>5} | {'Out':>5} | {'Conc':>5} | {'TTFT':>8} | {'TPOT':>8} | {'Thru':>10} | {'OK':>8}")
    print("  " + "-" * 115)
    for r in all_results:
        print(f"  {r['config']:<8} | {r['input_len']:>5} | {r['output_len']:>5} | "
              f"{r['concurrency']:>5} | {r.get('mean_ttft_ms',0):>7.1f}ms | "
              f"{r.get('mean_tpot_ms',0):>7.1f}ms | {r['output_thru_tok_s']:>8.1f} t/s | "
              f"{r['n_ok']:>4}/{r['n_requests']}")
    print("=" * 130)


if __name__ == "__main__":
    main()
