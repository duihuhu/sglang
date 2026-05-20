#!/usr/bin/env python3
"""Run 4-card baselines: pure AF (TP=2 each) and pure PD (TP=2 each).
Both use 4 GPUs total for fair comparison with PD+AF (4 GPUs, TP=1 each).
"""
import json, logging, os, socket, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("baselines")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "throughput_logs")
BENCHMARK = os.path.join(os.path.dirname(_HERE), "benchmark_replay.py")

CONCURRENCY = 64
MAX_REQUESTS = 128
OUTPUT_LEN = 128
INPUT_LEN = 512

def kill_all():
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(5)
    subprocess.run(["pkill", "-9", "-f", "ucx"], capture_output=True)
    time.sleep(10)

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

import requests, concurrent.futures

all_results = []

# ============================================================
# Test 1: Pure PD (Prefill TP=2 on GPU 0,1 + Decode TP=2 on GPU 2,3)
# ============================================================
log.info("=" * 60)
log.info("Test 1: Pure PD (TP=2 prefill + TP=2 decode)")
log.info("=" * 60)

kill_all()
procs = []

env_base = os.environ.copy()
env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

PREFILL_PORT = 50010
DECODE_PORT = 50020
ROUTER_PORT = 50000
BOOTSTRAP = 18999

# Prefill server (GPU 0,1, TP=2)
p_env = env_base.copy()
p_env["CUDA_VISIBLE_DEVICES"] = "0,1"
p_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "2",
    "--host", "127.0.0.1", "--port", str(PREFILL_PORT),
    "--disaggregation-mode", "prefill",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
    "--max-running-requests", "64",
    "--mem-fraction-static", "0.85",
]
fh = open(os.path.join(LOG_DIR, "pure_pd_prefill.log"), "w")
p = subprocess.Popen(p_cmd, env=p_env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
procs.append(("prefill", p, fh))

# Decode server (GPU 2,3, TP=2)
d_env = env_base.copy()
d_env["CUDA_VISIBLE_DEVICES"] = "2,3"
d_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "2",
    "--host", "127.0.0.1", "--port", str(DECODE_PORT),
    "--disaggregation-mode", "decode",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
    "--max-running-requests", "64",
    "--mem-fraction-static", "0.85",
]
fh2 = open(os.path.join(LOG_DIR, "pure_pd_decode.log"), "w")
p2 = subprocess.Popen(d_cmd, env=d_env, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
procs.append(("decode", p2, fh2))

if not wait_port("127.0.0.1", PREFILL_PORT, timeout=300):
    log.error("Pure PD: Prefill server failed to start")
else:
    log.info("Pure PD: Prefill server ready")
if not wait_port("127.0.0.1", DECODE_PORT, timeout=300):
    log.error("Pure PD: Decode server failed to start")
else:
    log.info("Pure PD: Decode server ready")

# Router
router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
    "--pd-disaggregation", "--mini-lb",
    "--prefill", f"http://127.0.0.1:{PREFILL_PORT}",
    "--decode", f"http://127.0.0.1:{DECODE_PORT}",
    "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
]
rfh = open(os.path.join(LOG_DIR, "pure_pd_router.log"), "w")
rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rfh, stderr=subprocess.STDOUT, start_new_session=True)
procs.append(("router", rp, rfh))

if wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=120):
    log.info("Pure PD: Router ready")

    # Warmup
    warm_payloads = [{"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}} for i in range(8)]
    def _send(payload):
        requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=payload, timeout=120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(5)
    log.info("Pure PD: Warmup done")

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
        "--dump", os.path.join(LOG_DIR, "results_pure_pd.json"),
        "--scenario-label", "pure_pd",
    ]
    subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK))

    dump_file = os.path.join(LOG_DIR, "results_pure_pd.json")
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
            "label": "pure_pd_tp2",
            "output_throughput": out_thru,
            "total_throughput": total_thru,
            "mean_ttft_ms": sum(ttft)/len(ttft) if ttft else 0,
            "mean_tpot_ms": sum(tpot)/len(tpot) if tpot else 0,
            "succeeded": len(ok),
            "total": len(results_list),
        }
        all_results.append(result)
        log.info("=== pure_pd_tp2 ===")
        log.info("  Output throughput: %.1f tok/s", out_thru)
        log.info("  Total throughput: %.1f tok/s", total_thru)
        log.info("  Mean TTFT: %.1f ms", result["mean_ttft_ms"])
        log.info("  Mean TPOT: %.1f ms", result["mean_tpot_ms"])
else:
    log.error("Pure PD: Router failed")

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

# ============================================================
# Test 2: Pure AF (Attn TP=2 on GPU 0,1 + FFN TP=2 on GPU 2,3)
# ============================================================
log.info("=" * 60)
log.info("Test 2: Pure AF (TP=2 attn + TP=2 ffn, no PD)")
log.info("=" * 60)

kill_all()
procs = []

env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
env_base["UCX_LOG_LEVEL"] = "fatal"

A_PORT = 30000
F_PORT = 30001
UCX_BASE = 25000
SCHED_PORT = 65000

# FFN server (GPU 2,3, TP=2)
f_env = env_base.copy()
f_env["CUDA_VISIBLE_DEVICES"] = "2,3"
f_env["AFD_UCX_BASE_PORT"] = str(UCX_BASE)
f_env["AFD_SCHED_PORT"] = str(SCHED_PORT)
f_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "2",
    "--host", "127.0.0.1", "--port", str(F_PORT),
    "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
    "--afd-micro-batch", "1",
    "--max-running-requests", "64",
    "--mem-fraction-static", "0.85",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
]
fh = open(os.path.join(LOG_DIR, "pure_af_m1_f.log"), "w")
p = subprocess.Popen(f_cmd, env=f_env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
procs.append(("FFN", p, fh))

# Attn server (GPU 0,1, TP=2)
a_env = env_base.copy()
a_env["CUDA_VISIBLE_DEVICES"] = "0,1"
a_env["AFD_UCX_BASE_PORT"] = str(UCX_BASE)
a_env["AFD_SCHED_PORT"] = str(SCHED_PORT)
a_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
a_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "2",
    "--host", "127.0.0.1", "--port", str(A_PORT),
    "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
    "--afd-micro-batch", "1",
    "--max-running-requests", "64",
    "--mem-fraction-static", "0.85",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
]
fh2 = open(os.path.join(LOG_DIR, "pure_af_m1_a.log"), "w")
p2 = subprocess.Popen(a_cmd, env=a_env, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
procs.append(("Attn", p2, fh2))

if not wait_port("127.0.0.1", A_PORT, timeout=300) or not wait_port("127.0.0.1", F_PORT, timeout=300):
    log.error("Pure AF M=1: Failed to start")
else:
    log.info("Pure AF M=1: Servers ready")
    if wait_health(f"http://127.0.0.1:{A_PORT}/health", timeout=60):
        # Warmup
        warm_payloads = [{"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}} for i in range(8)]
        def _send2(payload):
            requests.post(f"http://127.0.0.1:{A_PORT}/generate", json=payload, timeout=120)
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(_send2, warm_payloads))
        time.sleep(5)
        log.info("Pure AF M=1: Warmup done")

        # Benchmark
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
            "--dump", os.path.join(LOG_DIR, "results_pure_af_m1.json"),
            "--scenario-label", "pure_af_m1",
        ]
        subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK))

        dump_file = os.path.join(LOG_DIR, "results_pure_af_m1.json")
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
                "label": "pure_af_m1_tp2",
                "output_throughput": out_thru,
                "total_throughput": total_thru,
                "mean_ttft_ms": sum(ttft)/len(ttft) if ttft else 0,
                "mean_tpot_ms": sum(tpot)/len(tpot) if tpot else 0,
                "succeeded": len(ok),
                "total": len(results_list),
            }
            all_results.append(result)
            log.info("=== pure_af_m1_tp2 ===")
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

# ============================================================
# Summary
# ============================================================
log.info("\n" + "=" * 100)
log.info("BASELINE SUMMARY (4 GPUs)")
log.info("=" * 100)
log.info(f"{'Label':<25} {'Out tok/s':<12} {'Total tok/s':<12} {'TTFT ms':<10} {'TPOT ms':<10}")
log.info("-" * 100)
for r in all_results:
    log.info(f"  {r['label']:<25} {r['output_throughput']:<12.1f} {r['total_throughput']:<12.1f} {r['mean_ttft_ms']:<10.1f} {r['mean_tpot_ms']:<10.1f}")
# Add PD+AF M=1 from previous test
log.info(f"  {'pdaf_m1 (prev)':<25} {'301.9':<12} {'1512.1':<12} {'3708.7':<10} {'165.7':<10}")
log.info("=" * 100)

with open(os.path.join(LOG_DIR, "baselines_4gpu.json"), "w") as f:
    json.dump(all_results, f, indent=2)
log.info("Done")
