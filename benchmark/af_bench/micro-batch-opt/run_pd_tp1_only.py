#!/usr/bin/env python3
"""PD_TP1 only: high-concurrency sweep (2048-6144). Uses GPU 4 (decode) + GPU 6 (prefill)."""
import json, logging, os, socket, subprocess, sys, time, urllib.request, threading, re

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pd_tp1")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
BENCHMARK = os.path.join(HERE, "..", "benchmark_replay.py")
LOG_DIR = os.path.join(HERE, "logs", "high_conc_v2_logs")
os.makedirs(LOG_DIR, exist_ok=True)

BOOTSTRAP = 18999
ALL_PORTS = [50000, 50010, 50020]
INPUT_LEN = 32
OUTPUT_LEN = 32
CONCURRENCY_LEVELS = [2048, 3072, 4096, 5120, 6144]


def kill_our_servers():
    for port in ALL_PORTS:
        out = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True).stdout
        for line in out.splitlines():
            if f":{port}" in line and "pid=" in line:
                for m in re.finditer(r"pid=(\d+)", line):
                    try: os.kill(int(m.group(1)), 9)
                    except: pass
    time.sleep(5)


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


class DecodeBatchMonitor:
    def __init__(self, decode_url, interval=0.3):
        self.url = decode_url.rstrip("/") + "/v1/loads"
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear(); self.samples = []
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread: self._thread.join(timeout=5)

    def _poll(self):
        while not self._stop.is_set():
            try:
                resp = urllib.request.urlopen(self.url, timeout=2)
                data = json.loads(resp.read())
                agg = data.get("aggregate", {})
                running = agg.get("total_running_reqs", 0)
                if running == 0:
                    loads = data.get("loads", [])
                    running = sum(w.get("num_running_reqs", 0) for w in loads)
                self.samples.append({"t": time.time(), "running": running})
            except: pass
            self._stop.wait(self.interval)

    def summary(self):
        if not self.samples: return {}
        vals = [s["running"] for s in self.samples]
        vals_nz = [v for v in vals if v > 0]
        if not vals_nz: return {"max_running": 0, "mean_running": 0, "samples": len(vals)}
        return {"max_running": max(vals_nz), "mean_running": round(sum(vals_nz)/len(vals_nz), 1),
                "p50_running": sorted(vals_nz)[len(vals_nz)//2],
                "samples": len(vals), "nonzero_samples": len(vals_nz)}


def start_pd_tp1():
    kill_our_servers(); time.sleep(3)
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE"] = "0"

    env_p = env_base.copy()
    env_p["CUDA_VISIBLE_DEVICES"] = "6"
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

    env_d = env_base.copy()
    env_d["CUDA_VISIBLE_DEVICES"] = "4"
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
    return procs, "http://127.0.0.1:50000", "http://127.0.0.1:50020"


def run_benchmark(url, decode_url, concurrency, label):
    n_requests = concurrency * 2
    qps = max(concurrency // 4, 32)
    monitor = DecodeBatchMonitor(decode_url, interval=0.3)
    monitor.start()

    dump_file = os.path.join(LOG_DIR, f"result_{label}_conc{concurrency}.json")
    cmd = [PYTHON, BENCHMARK, "--dataset", "sample", "--url", url,
           "--max-requests", str(n_requests), "--concurrency", str(concurrency),
           "--sample-input-len", str(INPUT_LEN), "--sample-output-len", str(OUTPUT_LEN),
           "--sample-qps", str(qps), "--timeout", "600",
           "--dump", dump_file, "--scenario-label", f"{label}_conc{concurrency}"]
    log.info("  conc=%d, n=%d, qps=%d", concurrency, n_requests, qps)
    try: subprocess.run(cmd, cwd=HERE, capture_output=True, timeout=720)
    except subprocess.TimeoutExpired: pass

    monitor.stop()
    batch_stats = monitor.summary()

    if not os.path.exists(dump_file): return None, batch_stats
    with open(dump_file) as f: data = json.load(f)
    results = data.get("results", [])
    ok = [r for r in results if r.get("success")]
    if not ok: return None, batch_stats

    wall = data.get("wall_duration_s", 1) or 1
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    out_thru = total_out / wall
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]

    record = {"concurrency": concurrency, "n_requests": n_requests, "n_ok": len(ok),
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
    record["decode_batch"] = batch_stats
    return record, batch_stats


def main():
    all_results = []
    for conc in CONCURRENCY_LEVELS:
        log.info("[PD_TP1] Starting server for conc=%d ...", conc)
        ret = start_pd_tp1()
        if not ret:
            log.error("PD_TP1 conc=%d: Failed to start", conc)
            kill_our_servers(); time.sleep(10)
            all_results.append({"concurrency": conc, "error": True})
            continue

        procs, url, decode_url = ret
        log.info("[PD_TP1] Testing concurrency=%d ...", conc)
        r, batch_stats = run_benchmark(url, decode_url, conc, "PD_TP1")
        if r:
            all_results.append(r)
            log.info("  conc=%d: TTFT=%.1fms TPOT=%.1fms Thru=%.1f tok/s (%d/%d) | D_batch: max=%s mean=%s",
                     conc, r.get("mean_ttft_ms", 0), r.get("mean_tpot_ms", 0),
                     r["output_thru_tok_s"], r["n_ok"], r["n_requests"],
                     batch_stats.get("max_running", "?"), batch_stats.get("mean_running", "?"))
        else:
            log.warning("  conc=%d: FAILED | D_batch: %s", conc, batch_stats)
            all_results.append({"concurrency": conc, "error": True, "decode_batch": batch_stats})

        cleanup_procs(procs)
        kill_our_servers()
        time.sleep(10)

    print("\n" + "=" * 100)
    print(f"  PD_TP1 High-Concurrency (in={INPUT_LEN}, out={OUTPUT_LEN})")
    print("=" * 100)
    for r in all_results:
        if r.get("error"):
            print(f"  conc={r['concurrency']}: FAILED")
        else:
            db = r.get("decode_batch", {})
            print(f"  conc={r['concurrency']}: Thru={r['output_thru_tok_s']} t/s, "
                  f"TTFT={r.get('mean_ttft_ms',0):.1f}ms, TPOT={r.get('mean_tpot_ms',0):.1f}ms, "
                  f"D_max={db.get('max_running','?')}, D_mean={db.get('mean_running','?')}, "
                  f"OK={r['n_ok']}/{r['n_requests']}")
    print("=" * 100)

    with open(os.path.join(LOG_DIR, "pd_tp1_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Done.")


if __name__ == "__main__":
    main()
