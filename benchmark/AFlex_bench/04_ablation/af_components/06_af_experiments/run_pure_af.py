#!/usr/bin/env python3
"""Run pure AF (no PD) baseline: Attn TP=2 (GPU 0,1) + FFN TP=2 (GPU 2,3).
4 GPUs total for fair comparison with PD+AF (4 GPUs).
"""
import json, logging, os, socket, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pure_af")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "throughput_logs")
BENCHMARK = os.path.join(os.path.dirname(_HERE), "benchmark_replay.py")

CONCURRENCY = 32
MAX_REQUESTS = 128
OUTPUT_LEN = 128
INPUT_LEN = 512

import requests, concurrent.futures

def kill_all():
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(5)
    subprocess.run(["pkill", "-9", "-f", "ucx"], capture_output=True)
    time.sleep(15)

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

def run_pure_af(label, m_stage, ucx_base, sched_port, max_running=64):
    """Run pure AF (no PD) with Attn TP=2 (GPU 0,1) + FFN TP=2 (GPU 2,3)."""
    kill_all()
    procs = []

    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"

    A_PORT = 30000
    F_PORT = 30001

    extra = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--max-running-requests", str(max_running),
        "--afd-micro-batch", str(m_stage),
    ]

    # FFN server TP=2 (GPU 6,7) - must start first (it listens)
    f_env = env_base.copy()
    f_env["CUDA_VISIBLE_DEVICES"] = "6,7"
    f_env["AFD_UCX_BASE_PORT"] = str(ucx_base)
    f_env["AFD_SCHED_PORT"] = str(sched_port)
    f_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "2",
        "--host", "127.0.0.1", "--port", str(F_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
    ] + extra
    fh = open(os.path.join(LOG_DIR, f"{label}_f.log"), "w")
    p = subprocess.Popen(f_cmd, env=f_env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("FFN", p, fh))

    # Attn server TP=2 (GPU 4,5) - connects to FFN
    a_env = env_base.copy()
    a_env["CUDA_VISIBLE_DEVICES"] = "4,5"
    a_env["AFD_UCX_BASE_PORT"] = str(ucx_base)
    a_env["AFD_SCHED_PORT"] = str(sched_port)
    a_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    a_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "2",
        "--host", "127.0.0.1", "--port", str(A_PORT),
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
    ] + extra
    fh2 = open(os.path.join(LOG_DIR, f"{label}_a.log"), "w")
    p2 = subprocess.Popen(a_cmd, env=a_env, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("Attn", p2, fh2))

    if not wait_port("127.0.0.1", A_PORT, timeout=300):
        log.error("%s: Attn server failed to start", label)
        for n, p, f in procs:
            try: os.killpg(os.getpgid(p.pid), 2)
            except: p.terminate()
        return None
    if not wait_port("127.0.0.1", F_PORT, timeout=300):
        log.error("%s: FFN server failed to start", label)
        for n, p, f in procs:
            try: os.killpg(os.getpgid(p.pid), 2)
            except: p.terminate()
        return None
    log.info("%s: Both servers ready (Attn TP=2 GPU 4,5 + FFN TP=2 GPU 6,7)", label)

    if not wait_health(f"http://127.0.0.1:{A_PORT}/health", timeout=60):
        log.error("%s: Attn health check failed", label)
        for n, p, f in procs:
            try: os.killpg(os.getpgid(p.pid), 2)
            except: p.terminate()
        return None

    # Warmup
    warm_payloads = [
        {"text": f"Hello world {i}, this is a warmup request for pure AF test:",
         "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(8)
    ]
    log.info("%s: Warmup...", label)
    def _send(payload):
        try:
            requests.post(f"http://127.0.0.1:{A_PORT}/generate", json=payload, timeout=120)
        except Exception as e:
            log.warning("Warmup failed: %s", e)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(5)
    log.info("%s: Warmup done", label)

    # Benchmark (send to Attn server directly)
    cmd = [
        PYTHON, BENCHMARK,
        "--url", f"http://127.0.0.1:{A_PORT}",
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
    log.info("%s: Benchmark: %s", label, " ".join(cmd))
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
        log.info("=== %s ===", label)
        log.info("  Succeeded: %d/%d", len(ok), len(results_list))
        log.info("  Output throughput: %.1f tok/s", out_thru)
        log.info("  Total throughput: %.1f tok/s", total_thru)
        log.info("  Mean TTFT: %.1f ms", result["mean_ttft_ms"])
        log.info("  Mean TPOT: %.1f ms", result["mean_tpot_ms"])

    # Cleanup
    for name, p, fh in procs:
        try: os.killpg(os.getpgid(p.pid), 2)
        except: p.terminate()
    time.sleep(3)
    for name, p, fh in procs:
        try: p.kill()
        except: pass
        try: fh.close()
        except: pass

    return result

# Run pure AF tests
all_results = []

# M=1 pure AF (Attn TP=2 GPU 0,1 + FFN TP=2 GPU 2,3)
log.info("=" * 60)
log.info("Pure AF M=1 (Attn TP=2 GPU 4,5 + FFN TP=2 GPU 6,7)")
log.info("=" * 60)
r = run_pure_af("pure_af_m1_tp2", m_stage=1, ucx_base=26000, sched_port=66000, max_running=32)
if r:
    all_results.append(r)

# M=3 pure AF (same GPUs, different UCX ports)
log.info("=" * 60)
log.info("Pure AF M=3 (Attn TP=2 GPU 4,5 + FFN TP=2 GPU 6,7)")
log.info("=" * 60)
r = run_pure_af("pure_af_m3_tp2", m_stage=3, ucx_base=27000, sched_port=67000, max_running=32)
if r:
    all_results.append(r)

# Summary
log.info("\n" + "=" * 80)
log.info("PURE AF COMPARISON (4 GPUs 4567, Attn TP=2 + FFN TP=2)")
log.info("=" * 80)
log.info(f"{'Label':<25} {'Out tok/s':<12} {'Total tok/s':<12} {'TTFT ms':<10} {'TPOT ms':<10} {'OK':<10}")
log.info("-" * 80)
for r in all_results:
    log.info(f"  {r['label']:<25} {r['output_throughput']:<12.1f} {r['total_throughput']:<12.1f} {r['mean_ttft_ms']:<10.1f} {r['mean_tpot_ms']:<10.1f} {r['succeeded']}/{r['total']}")
# Reference from previous tests
log.info("-" * 80)
log.info(f"  {'PD+AF M=1 (ref)':<25} {'302':<12} {'1512':<12} {'575':<10} {'191':<10} {'128/128':<10}")
log.info(f"  {'Pure PD TP=2 (ref)':<25} {'1068':<12} {'5350':<12} {'149':<10} {'50':<10} {'128/128':<10}")
log.info("=" * 80)

with open(os.path.join(LOG_DIR, "pure_af_tp2_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)
log.info("Done")
