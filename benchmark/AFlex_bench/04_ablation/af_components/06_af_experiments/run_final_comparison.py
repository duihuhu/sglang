#!/usr/bin/env python3
"""Final comprehensive test: M=3 vs M=1 with unique UCX ports per config.
Uses incrementing UCX base ports to avoid RDMA resource conflicts.
"""
import json, logging, os, socket, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("final_test")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "throughput_logs")
BENCHMARK = os.path.join(os.path.dirname(_HERE), "benchmark_replay.py")

CONCURRENCY = 64
MAX_REQUESTS = 128
OUTPUT_LEN = 128
INPUT_LEN = 512
ROUTER_PORT = 50000
BOOTSTRAP = 18999

import requests, concurrent.futures

def kill_all():
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(5)

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

def run_config(label, afd_args, ucx_base_p, ucx_base_d, sched_p, sched_d):
    """Run a single PD+AF config with unique UCX ports."""
    kill_all()
    # Extra wait for RDMA cleanup
    time.sleep(15)
    procs = []
    
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"

    PA_PORT, PF_PORT = 50010, 50011
    DA_PORT, DF_PORT = 50020, 50021

    extra = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--max-running-requests", "64",
    ] + afd_args

    def start_server(name, cmd, env, log_name):
        fh = open(os.path.join(LOG_DIR, log_name), "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))
        return p

    # PF GPU 2
    pf_env = env_base.copy()
    pf_env["CUDA_VISIBLE_DEVICES"] = "2"
    pf_env["AFD_UCX_BASE_PORT"] = str(ucx_base_p)
    pf_env["AFD_SCHED_PORT"] = str(sched_p)
    pf_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(PF_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_server("PF", pf_cmd, pf_env, f"{label}_pf.log")

    # PA GPU 3
    pa_env = env_base.copy()
    pa_env["CUDA_VISIBLE_DEVICES"] = "3"
    pa_env["AFD_UCX_BASE_PORT"] = str(ucx_base_p)
    pa_env["AFD_SCHED_PORT"] = str(sched_p)
    pa_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    pa_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(PA_PORT),
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_server("PA", pa_cmd, pa_env, f"{label}_pa.log")

    if not wait_port("127.0.0.1", PA_PORT, timeout=300) or not wait_port("127.0.0.1", PF_PORT, timeout=300):
        log.error("%s: PA/PF failed to start", label)
        for n, p, fh in procs:
            try: os.killpg(os.getpgid(p.pid), 2)
            except: p.terminate()
        return None
    log.info("%s: PA and PF ready", label)

    # DF GPU 0
    df_env = env_base.copy()
    df_env["CUDA_VISIBLE_DEVICES"] = "0"
    df_env["AFD_UCX_BASE_PORT"] = str(ucx_base_d)
    df_env["AFD_SCHED_PORT"] = str(sched_d)
    df_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(DF_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_server("DF", df_cmd, df_env, f"{label}_df.log")

    # DA GPU 1
    da_env = env_base.copy()
    da_env["CUDA_VISIBLE_DEVICES"] = "1"
    da_env["AFD_UCX_BASE_PORT"] = str(ucx_base_d)
    da_env["AFD_SCHED_PORT"] = str(sched_d)
    da_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    da_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(DA_PORT),
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_server("DA", da_cmd, da_env, f"{label}_da.log")

    if not wait_port("127.0.0.1", DA_PORT, timeout=300) or not wait_port("127.0.0.1", DF_PORT, timeout=300):
        log.error("%s: DA/DF failed to start", label)
        for n, p, fh in procs:
            try: os.killpg(os.getpgid(p.pid), 2)
            except: p.terminate()
        return None
    log.info("%s: DA and DF ready", label)

    # Router
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", f"http://127.0.0.1:{PA_PORT}",
        "--decode", f"http://127.0.0.1:{DA_PORT}",
        "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
    ]
    router_fh = open(os.path.join(LOG_DIR, f"{label}_router.log"), "w")
    router_proc = subprocess.Popen(router_cmd, env=os.environ.copy(),
                                   stdout=router_fh, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    procs.append(("router", router_proc, router_fh))

    if not wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=120):
        log.error("%s: Router health check failed", label)
        for n, p, fh in procs:
            try: os.killpg(os.getpgid(p.pid), 2)
            except: p.terminate()
        return None
    log.info("%s: All servers ready", label)

    # Warmup
    warm_payloads = [
        {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(8)
    ]
    def _send(payload):
        requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=payload, timeout=120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(5)
    log.info("%s: Warmup done", label)

    # Benchmark
    cmd = [
        PYTHON, BENCHMARK,
        "--url", f"http://127.0.0.1:{ROUTER_PORT}",
        "--dataset", "sample",
        "--max-requests", str(MAX_REQUESTS),
        "--concurrency", str(CONCURRENCY),
        "--sample-input-len", str(INPUT_LEN),
        "--sample-output-len", str(OUTPUT_LEN),
        "--sample-qps", "9999",
        "--timeout", "600",
        "--dump", os.path.join(LOG_DIR, f"results_{label}.json"),
        "--scenario-label", label,
    ]
    log.info("%s: Benchmark", label)
    subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK))

    result = None
    dump_file = os.path.join(LOG_DIR, f"results_{label}.json")
    if os.path.exists(dump_file):
        with open(dump_file) as f:
            data = json.load(f)
        results_list = data.get("results", [])
        ok = [r for r in results_list if r.get("success")]
        total_in = sum(r.get("input_tokens", 0) for r in ok)
        total_out = sum(r.get("output_tokens", 0) for r in ok)
        wall = data.get("wall_duration_s", 1)
        out_thru = total_out / wall if wall > 0 else 0
        total_thru = (total_in + total_out) / wall if wall > 0 else 0
        ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
        tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]

        result = {
            "label": label,
            "output_throughput": out_thru,
            "total_throughput": total_thru,
            "mean_ttft_ms": sum(ttft)/len(ttft) if ttft else 0,
            "mean_tpot_ms": sum(tpot)/len(tpot) if tpot else 0,
            "succeeded": len(ok),
            "total": len(results_list),
        }
        log.info("=== %s: Out=%.1f tok/s, TPOT=%.1f ms, OK=%d/%d ===",
                 label, out_thru, result["mean_tpot_ms"], len(ok), len(results_list))

    # Cleanup
    for n, p, fh in procs:
        try: os.killpg(os.getpgid(p.pid), 2)
        except: p.terminate()
    time.sleep(3)
    for n, p, fh in procs:
        try: p.kill()
        except: pass
        try: fh.close()
        except: pass

    return result

# Each config gets unique UCX ports to avoid RDMA resource conflicts
# Format: (label, afd_args, ucx_base_p, ucx_base_d, sched_p, sched_d)
configs = [
    ("final_m3", ["--afd-micro-batch", "3"], 25100, 25200, 65300, 65400),
    ("final_m1", ["--afd-micro-batch", "1"], 25300, 25400, 65500, 65600),
]

all_results = []
for label, afd_args, ucx_p, ucx_d, sp, sd in configs:
    log.info("=" * 60)
    log.info("Starting: %s", label)
    log.info("=" * 60)
    r = run_config(label, afd_args, ucx_p, ucx_d, sp, sd)
    if r:
        all_results.append(r)

# Summary
log.info("\n" + "=" * 80)
log.info("FINAL COMPARISON: M=3 vs M=1 (unique UCX ports)")
log.info("=" * 80)
log.info(f"{'Label':<15} {'Out tok/s':<12} {'TPOT ms':<10} {'TTFT ms':<10} {'OK':<10}")
log.info("-" * 80)
for r in all_results:
    log.info(f"  {r['label']:<15} {r['output_throughput']:<12.1f} {r['mean_tpot_ms']:<10.1f} {r['mean_ttft_ms']:<10.1f} {r['succeeded']}/{r['total']}")
log.info("=" * 80)

with open(os.path.join(LOG_DIR, "final_comparison.json"), "w") as f:
    json.dump(all_results, f, indent=2)
log.info("Done")
