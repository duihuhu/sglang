#!/usr/bin/env python3
"""Run PD+AF M=1 and M=3 benchmarks simultaneously across 8 GPUs.

M=3: GPUs 0,1,2,3  (DF=0, DA=1, PF=2, PA=3)
M=1: GPUs 4,5,6,7  (DF=4, DA=5, PF=6, PA=7)

Both run QPS=2, 500 requests each, in parallel.
"""
import json, logging, os, socket, subprocess, sys, time, threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qps_sweep_par")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "throughput_logs")
BENCHMARK = os.path.join(os.path.dirname(_HERE), "benchmark_replay.py")

QPS = 2
MAX_REQUESTS = 500
OUTPUT_LEN = 128
INPUT_LEN = 512
CONCURRENCY = 500

# ── Per-variant config ──────────────────────────────────────────────────────

VARIANTS = {
    "m3": {
        "micro_batch": 3,
        "gpus": {"DF": "0", "DA": "1", "PF": "2", "PA": "3"},
        "http": {"PA": 50010, "PF": 50011, "DA": 50020, "DF": 50021},
        "ucx": {"A": 25100, "F": 25100, "DA": 25200, "DF": 25200},
        "sched": {"P": 65300, "D": 65400},
        "bootstrap": 18999,
        "router_port": 50001,
    },
    "m1": {
        "micro_batch": 1,
        "gpus": {"DF": "4", "DA": "5", "PF": "6", "PA": "7"},
        "http": {"PA": 50030, "PF": 50031, "DA": 50040, "DF": 50041},
        "ucx": {"A": 25300, "F": 25300, "DA": 25400, "DF": 25400},
        "sched": {"P": 65500, "D": 65600},
        "bootstrap": 18998,
        "router_port": 50002,
    },
}


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
        except:
            time.sleep(2)
    return False


def wait_health(url, timeout=180):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except:
            time.sleep(3)
    return False


def launch_variant(name, cfg):
    """Launch all 4 servers + router for one variant. Returns (procs, router_port)."""
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"

    extra = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", str(cfg["micro_batch"]),
        "--max-running-requests", "500",
    ]

    procs = []

    def start_srv(label, cmd, env, log_suffix):
        fh = open(os.path.join(LOG_DIR, f"qps_sweep_{name}_{log_suffix}"), "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((label, p, fh))
        log.info("[%s] Starting %s on GPU %s", name, label, env.get("CUDA_VISIBLE_DEVICES", "?"))

    g = cfg["gpus"]
    h = cfg["http"]
    u = cfg["ucx"]
    s = cfg["sched"]
    bootstrap = cfg["bootstrap"]

    # --- PF ---
    pf_env = env_base.copy()
    pf_env["CUDA_VISIBLE_DEVICES"] = g["PF"]
    pf_env["AFD_UCX_BASE_PORT"] = str(u["F"])
    pf_env["AFD_SCHED_PORT"] = str(s["P"])
    pf_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(h["PF"]),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.88",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(bootstrap),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_srv("PF", pf_cmd, pf_env, "pf.log")

    # --- PA ---
    pa_env = env_base.copy()
    pa_env["CUDA_VISIBLE_DEVICES"] = g["PA"]
    pa_env["AFD_UCX_BASE_PORT"] = str(u["A"])
    pa_env["AFD_SCHED_PORT"] = str(s["P"])
    pa_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    pa_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(h["PA"]),
        "--enable-metrics",
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.88",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(bootstrap),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_srv("PA", pa_cmd, pa_env, "pa.log")

    if not wait_port("127.0.0.1", h["PA"], timeout=300) or \
       not wait_port("127.0.0.1", h["PF"], timeout=300):
        log.error("[%s] PA/PF failed to start", name)
        return procs, None

    # --- DF ---
    df_env = env_base.copy()
    df_env["CUDA_VISIBLE_DEVICES"] = g["DF"]
    df_env["AFD_UCX_BASE_PORT"] = str(u["DF"])
    df_env["AFD_SCHED_PORT"] = str(s["D"])
    df_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(h["DF"]),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.88",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(bootstrap),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_srv("DF", df_cmd, df_env, "df.log")

    # --- DA ---
    da_env = env_base.copy()
    da_env["CUDA_VISIBLE_DEVICES"] = g["DA"]
    da_env["AFD_UCX_BASE_PORT"] = str(u["DA"])
    da_env["AFD_SCHED_PORT"] = str(s["D"])
    da_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    da_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(h["DA"]),
        "--enable-metrics",
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.88",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(bootstrap),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_srv("DA", da_cmd, da_env, "da.log")

    if not wait_port("127.0.0.1", h["DA"], timeout=300) or \
       not wait_port("127.0.0.1", h["DF"], timeout=300):
        log.error("[%s] DA/DF failed to start", name)
        return procs, None

    # --- Router ---
    rport = cfg["router_port"]
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", f"http://127.0.0.1:{h['PA']}",
        "--decode", f"http://127.0.0.1:{h['DA']}",
        "--host", "127.0.0.1", "--port", str(rport),
    ]
    router_fh = open(os.path.join(LOG_DIR, f"qps_sweep_{name}_router.log"), "w")
    router_proc = subprocess.Popen(router_cmd, env=os.environ.copy(),
                                   stdout=router_fh, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    procs.append(("router", router_proc, router_fh))

    if not wait_health(f"http://127.0.0.1:{rport}/health", timeout=120):
        log.error("[%s] Router health check failed", name)
        return procs, None

    log.info("[%s] All servers + router ready (router=%d)", name, rport)
    return procs, rport


def run_benchmark(name, router_port, result_event):
    """Run benchmark against router, save results."""
    import requests, concurrent.futures

    # Warmup
    warm_payloads = [
        {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(16)
    ]
    if name == "m1":
        log.info("[%s] Warmup...", name)
    def _send(p):
        requests.post(f"http://127.0.0.1:{router_port}/generate", json=p, timeout=120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(5)

    label = f"pdaf_{name}_qps{QPS}"
    dump_file = os.path.join(LOG_DIR, f"results_{label}.json")
    log.info("[%s] Running benchmark QPS=%d, %d requests", name, QPS, MAX_REQUESTS)

    cmd = [
        PYTHON, BENCHMARK,
        "--url", f"http://127.0.0.1:{router_port}",
        "--dataset", "sample",
        "--max-requests", str(MAX_REQUESTS),
        "--concurrency", str(CONCURRENCY),
        "--sample-input-len", str(INPUT_LEN),
        "--sample-output-len", str(OUTPUT_LEN),
        "--sample-qps", str(QPS),
        "--timeout", "600",
        "--dump", dump_file,
        "--scenario-label", label,
    ]
    log.info("[%s] %s", name, " ".join(cmd))
    subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK))

    # Parse results
    result = None
    if os.path.exists(dump_file):
        with open(dump_file) as f:
            data = json.load(f)
        results = data.get("results", [])
        ok = [r for r in results if r.get("success")]
        total_in = sum(r.get("input_tokens", 0) for r in ok)
        total_out = sum(r.get("output_tokens", 0) for r in ok)
        wall = data.get("wall_duration_s", 1)
        out_thru = total_out / wall if wall > 0 else 0
        total_thru = (total_in + total_out) / wall if wall > 0 else 0
        ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
        tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
        avg_ttft = sum(ttft)/len(ttft) if ttft else 0
        avg_tpot = sum(tpot)/len(tpot) if tpot else 0
        result = {
            "label": label, "qps": QPS,
            "out_tok_s": round(out_thru, 1),
            "total_tok_s": round(total_thru, 1),
            "ttft_ms": round(avg_ttft, 1),
            "tpot_ms": round(avg_tpot, 1),
            "ok": f"{len(ok)}/{len(results)}",
            "wall_s": round(wall, 1),
        }
        log.info("[%s] out=%.1f tok/s, TPOT=%.1f ms, TTFT=%.1f ms, wall=%.1fs",
                 name, out_thru, avg_tpot, avg_ttft, wall)
    else:
        log.error("[%s] No results file!", name)

    result_event.set()
    return result


def cleanup(procs_list):
    """Terminate all processes across all variants."""
    for name, p, fh in procs_list:
        try: os.killpg(os.getpgid(p.pid), 2)
        except: p.terminate()
    time.sleep(3)
    for name, p, fh in procs_list:
        try: p.kill()
        except: pass
        try: fh.close()
        except: pass


def main():
    kill_all()

    all_procs = []
    all_results = {}
    result_events = {}

    # Launch both variants in sequence (servers start quickly)
    for name in ["m3", "m1"]:
        log.info("=" * 60)
        log.info("Launching variant: %s", name)
        log.info("=" * 60)
        procs, rport = launch_variant(name, VARIANTS[name])
        all_procs.extend(procs)
        if rport is None:
            log.error("Failed to launch %s, aborting", name)
            cleanup(all_procs)
            sys.exit(1)

        # Run benchmark in a thread so both can run concurrently
        ev = threading.Event()
        result_events[name] = ev

        def _runner(n=name, p=rport, e=ev):
            r = run_benchmark(n, p, e)
            all_results[n] = r

        t = threading.Thread(target=_runner)
        t.start()

    # Wait for ALL benchmarks to finish
    start = time.time()
    while not all(e.is_set() for e in result_events.values()):
        time.sleep(5)
        elapsed = time.time() - start
        done = [n for n, e in result_events.items() if e.is_set()]
        pending = [n for n, e in result_events.items() if not e.is_set()]
        if int(elapsed) % 30 == 0:
            log.info("Elapsed %ds, done=%s, pending=%s", int(elapsed), done, pending)

    log.info("All benchmarks complete in %.1fs", time.time() - start)

    # ── Print results table ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  PD+AF Parallel Benchmark: M=3 + M=1 @ QPS=2")
    print(f"  Model: Qwen3-32B, Input=512, Output=128, {MAX_REQUESTS} requests")
    print("=" * 90)
    print(f"  {'Label':<22} {'QPS':<6} {'Out(tok/s)':<14} {'Total(tok/s)':<14} {'TPOT(ms)':<10} {'TTFT(ms)':<10} {'OK':<8} {'Wall(s)':<8}")
    print("  " + "-" * 92)
    for name in ["m3", "m1"]:
        r = all_results.get(name)
        if r:
            print(f"  {r['label']:<22} {r['qps']:<6} {r['out_tok_s']:<14.1f} {r['total_tok_s']:<14.1f} {r['tpot_ms']:<10.1f} {r['ttft_ms']:<10.1f} {r['ok']:<8} {r['wall_s']:<8.1f}")
    print("=" * 90)

    # Save combined results
    with open(os.path.join(LOG_DIR, "qps_sweep_parallel_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    # Cleanup
    cleanup(all_procs)
    log.info("All done")


if __name__ == "__main__":
    main()
