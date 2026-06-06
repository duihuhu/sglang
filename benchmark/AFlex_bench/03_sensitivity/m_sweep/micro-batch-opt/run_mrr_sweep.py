#!/usr/bin/env python3
"""
Test AF M=1 and M=2 with different --max-running-requests limits.
in=32, out=32, conc=1024/2048, max_running=200/500/1000.
"""
import json, logging, os, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("mrr_sweep")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(HERE, "..", "benchmark_replay.py")
LOG_DIR = os.path.join(HERE, "logs", "mrr_sweep_logs")
os.makedirs(LOG_DIR, exist_ok=True)

BOOTSTRAP = 18999
ALL_PORTS = [50000, 50010, 50011, 50020, 50021]
INPUT_LEN = 32
OUTPUT_LEN = 32


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
        try: os.killpg(os.getpgid(p.pid), 9)
        except:
            try: p.kill()
            except: pass
        try: f.close()
        except: pass
    time.sleep(3)


def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2); s.close(); return True
        except: time.sleep(2)
    return False


def wait_health(url, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try: urllib.request.urlopen(url, timeout=5); return True
        except: time.sleep(3)
    return False


def warmup(url, n=8):
    import requests, concurrent.futures
    payloads = [{"text": f"Hello {i}", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}} for i in range(n)]
    def _send(p):
        try: requests.post(url + "/generate", json=p, timeout=120)
        except: pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(_send, payloads))
    time.sleep(3)


def start_pdaf(micro_batch, async_pipeline, max_running):
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
             "--disable-radix-cache", "--num-reserved-decode-tokens", "32",
             "--max-running-requests", str(max_running)]
    if async_pipeline:
        extra.append("--afd-async-pipeline")

    tag = f"m{micro_batch}_mrr{max_running}"

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
        if ffn_host: env["AFD_UCX_FFN_HOST"] = ffn_host
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
        fh = open(os.path.join(LOG_DIR, f"{tag}_{name}.log"), "w")
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
    rf = open(os.path.join(LOG_DIR, f"{tag}_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health("http://127.0.0.1:50000/health", 180):
        cleanup_procs(procs); return None
    warmup("http://127.0.0.1:50000")
    return procs, "http://127.0.0.1:50000"


def run_bench(url, concurrency, label):
    n_requests = concurrency * 2
    qps = concurrency // 4
    dump_file = os.path.join(LOG_DIR, f"result_{label}_conc{concurrency}.json")
    cmd = [PYTHON, BENCHMARK, "--dataset", "sample", "--url", url,
           "--max-requests", str(n_requests), "--concurrency", str(concurrency),
           "--sample-input-len", str(INPUT_LEN), "--sample-output-len", str(OUTPUT_LEN),
           "--sample-qps", str(qps), "--timeout", "300",
           "--dump", dump_file, "--scenario-label", f"{label}_conc{concurrency}"]
    log.info("  conc=%d, n=%d, qps=%d", concurrency, n_requests, qps)
    try:
        subprocess.run(cmd, cwd=HERE, capture_output=True, timeout=360)
    except subprocess.TimeoutExpired:
        return None

    if not os.path.exists(dump_file):
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
    record = {"label": label, "concurrency": concurrency,
              "n_requests": n_requests, "n_ok": len(ok),
              "output_thru_tok_s": round(out_thru, 1), "wall_s": round(wall, 1)}
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


def main():
    # Test matrix: M × max_running × concurrency
    test_matrix = [
        # (label_prefix, micro_batch, async_pipeline)
        ("M2", 2, True),
        ("M1", 1, False),
    ]
    max_running_values = [200, 500, 1000]
    conc_values = [1024, 2048]

    all_results = []

    for m_label, m_batch, async_pipe in test_matrix:
        for mrr in max_running_values:
            label = f"AF_{m_label}_mrr{mrr}"
            log.info("\n" + "=" * 80)
            log.info(">>> %s (micro_batch=%d, max_running=%d) <<<", label, m_batch, mrr)
            log.info("=" * 80)

            # Start server once per (M, max_running) combo
            ret = start_pdaf(m_batch, async_pipe, mrr)
            if not ret:
                log.error("%s: Failed to start", label)
                kill_all(); time.sleep(10)
                continue

            procs, url = ret

            for conc in conc_values:
                log.info("[%s] conc=%d ...", label, conc)
                r = run_bench(url, conc, label)
                if r:
                    all_results.append(r)
                    log.info("  %s conc=%d: TTFT=%.1f TPOT=%.1f Thru=%.1f (%d/%d)",
                             label, conc, r.get("mean_ttft_ms", 0), r.get("mean_tpot_ms", 0),
                             r["output_thru_tok_s"], r["n_ok"], r["n_requests"])
                else:
                    log.warning("  %s conc=%d: FAILED", label, conc)
                    all_results.append({"label": label, "concurrency": conc, "error": True})
                time.sleep(5)

            cleanup_procs(procs)
            kill_all()
            time.sleep(10)

    # Print summary
    print("\n" + "=" * 110)
    print(f"  AF max_running_requests Sweep (in={INPUT_LEN}, out={OUTPUT_LEN})")
    print("=" * 110)
    print(f"  {'Label':<20} | {'Conc':>5} | {'TTFT':>8} | {'TPOT':>8} | {'Thru':>10} | {'OK':>10}")
    print("  " + "-" * 95)
    for r in all_results:
        if r.get("error"):
            print(f"  {r['label']:<20} | {r['concurrency']:>5} | {'FAILED':^40}")
        else:
            print(f"  {r['label']:<20} | {r['concurrency']:>5} | "
                  f"{r.get('mean_ttft_ms',0):>7.1f}ms | {r.get('mean_tpot_ms',0):>7.1f}ms | "
                  f"{r['output_thru_tok_s']:>8.1f} t/s | {r['n_ok']:>4}/{r['n_requests']}")
    print("=" * 110)

    # Save
    with open(os.path.join(LOG_DIR, "mrr_sweep_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved.")


if __name__ == "__main__":
    main()
