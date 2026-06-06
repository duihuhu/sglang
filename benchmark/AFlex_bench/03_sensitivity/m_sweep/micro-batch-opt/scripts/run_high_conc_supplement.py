#!/usr/bin/env python3
"""Supplement: AF_M1 conc=512,768 and PD_TP1 conc=256-2048."""
import json, logging, os, socket, subprocess, sys, time, urllib.request

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("supp")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(HERE, "..", "benchmark_replay.py")
LOG_DIR = os.path.join(HERE, "high_conc_sweep_logs")

BOOTSTRAP = 18999
ALL_PORTS = [50000, 50010, 50011, 50020, 50021]
INPUT_LEN = 32
OUTPUT_LEN = 32


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


def start_pdaf_m1():
    kill_all(); time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["SGLANG_DISAGGREGATION_QUEUE_SIZE"] = "32"
    env_base["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"

    extra = ["--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
             "--afd-micro-batch", "1", "--afd-disagg-interleave-poll", "--disable-radix-cache",
             "--num-reserved-decode-tokens", "32"]

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
        fh = open(os.path.join(LOG_DIR, f"supp_af_m1_{name}.log"), "w")
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
    rf = open(os.path.join(LOG_DIR, "supp_af_m1_router.log"), "w")
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
    fh = open(os.path.join(LOG_DIR, "supp_pd_tp1_p.log"), "w")
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
    fh2 = open(os.path.join(LOG_DIR, "supp_pd_tp1_d.log"), "w")
    p2 = subprocess.Popen(d_cmd, env=env_d, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("decode", p2, fh2))

    if not wait_port("127.0.0.1", 50010, 300) or not wait_port("127.0.0.1", 50020, 300):
        cleanup_procs(procs); return None
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", "http://127.0.0.1:50010", "--decode", "http://127.0.0.1:50020",
        "--host", "127.0.0.1", "--port", "50000"]
    rf = open(os.path.join(LOG_DIR, "supp_pd_tp1_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health("http://127.0.0.1:50000/health", 180):
        cleanup_procs(procs); return None
    warmup("http://127.0.0.1:50000")
    return procs, "http://127.0.0.1:50000"


def run_bench(url, concurrency, label):
    n_requests = concurrency * 2
    qps = max(concurrency // 4, 32)
    dump_file = os.path.join(LOG_DIR, f"result_{label}_conc{concurrency}.json")
    cmd = [PYTHON, BENCHMARK, "--dataset", "sample", "--url", url,
           "--max-requests", str(n_requests), "--concurrency", str(concurrency),
           "--sample-input-len", str(INPUT_LEN), "--sample-output-len", str(OUTPUT_LEN),
           "--sample-qps", str(qps), "--timeout", "600",
           "--dump", dump_file, "--scenario-label", f"{label}_conc{concurrency}"]
    log.info("  conc=%d, n=%d, qps=%d", concurrency, n_requests, qps)
    try:
        subprocess.run(cmd, cwd=HERE, capture_output=True, timeout=900)
    except subprocess.TimeoutExpired:
        log.warning("  TIMEOUT at conc=%d", concurrency)
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
    record = {"concurrency": concurrency, "n_requests": n_requests, "n_ok": len(ok),
              "output_thru_tok_s": round(out_thru, 1), "wall_s": round(wall, 1)}
    if ttfts:
        s = sorted(ttfts); m = len(s)
        record.update({"mean_ttft_ms": round(sum(ttfts)/m, 2), "p50_ttft_ms": round(s[m//2], 2),
                       "p95_ttft_ms": round(s[int(m*0.95)], 2), "p99_ttft_ms": round(s[int(m*0.99)], 2)})
    if tpots:
        s = sorted(tpots); k = len(s)
        record.update({"mean_tpot_ms": round(sum(tpots)/k, 2), "p50_tpot_ms": round(s[k//2], 2),
                       "p95_tpot_ms": round(s[int(k*0.95)], 2), "p99_tpot_ms": round(s[int(k*0.99)], 2)})
    return record


def main():
    all_results = {}

    # AF_M1: supplement conc=512, 768
    log.info("=== AF_M1 supplement (conc=512, 768) ===")
    af_m1_results = []
    for conc in [512, 768]:
        log.info("[AF_M1] Starting for conc=%d", conc)
        ret = start_pdaf_m1()
        if not ret:
            log.error("AF_M1 conc=%d: start failed", conc)
            kill_all(); time.sleep(10)
            af_m1_results.append({"concurrency": conc, "error": True})
            continue
        procs, url = ret
        r = run_bench(url, conc, "AF_M1")
        if r:
            af_m1_results.append(r)
            log.info("  AF_M1 conc=%d: TTFT=%.1f TPOT=%.1f Thru=%.1f (%d/%d)",
                     conc, r.get("mean_ttft_ms",0), r.get("mean_tpot_ms",0),
                     r["output_thru_tok_s"], r["n_ok"], r["n_requests"])
        else:
            af_m1_results.append({"concurrency": conc, "error": True})
            log.warning("  AF_M1 conc=%d: FAILED", conc)
        cleanup_procs(procs); kill_all(); time.sleep(10)
    all_results["AF_M1_supp"] = af_m1_results

    # PD_TP1: full sweep 256-2048
    log.info("\n=== PD_TP1 full sweep (256-2048) ===")
    pd_results = []
    for conc in [256, 512, 768, 1024, 1536, 2048]:
        log.info("[PD_TP1] Starting for conc=%d", conc)
        ret = start_pd_tp1()
        if not ret:
            log.error("PD_TP1 conc=%d: start failed", conc)
            kill_all(); time.sleep(10)
            pd_results.append({"concurrency": conc, "error": True})
            continue
        procs, url = ret
        r = run_bench(url, conc, "PD_TP1")
        if r:
            pd_results.append(r)
            log.info("  PD_TP1 conc=%d: TTFT=%.1f TPOT=%.1f Thru=%.1f (%d/%d)",
                     conc, r.get("mean_ttft_ms",0), r.get("mean_tpot_ms",0),
                     r["output_thru_tok_s"], r["n_ok"], r["n_requests"])
        else:
            pd_results.append({"concurrency": conc, "error": True})
            log.warning("  PD_TP1 conc=%d: FAILED", conc)
        cleanup_procs(procs); kill_all(); time.sleep(10)
    all_results["PD_TP1"] = pd_results

    with open(os.path.join(LOG_DIR, "supplement_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Done. Results in supplement_results.json")


if __name__ == "__main__":
    main()
