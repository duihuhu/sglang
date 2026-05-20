#!/usr/bin/env python3
"""Benchmark Native and PD+AF M=1 at multiple concurrency levels."""
import json, logging, os, socket, subprocess, sys, time, shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("stress2")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "throughput_logs")
BENCHMARK = os.path.join(os.path.dirname(_HERE), "benchmark_replay.py")

OUTPUT_LEN = 128
INPUT_LEN = 512
ROUTER_PORT = 50000

ALL_RESULTS = []

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

def run_bench(url, label):
    cmd = [
        PYTHON, BENCHMARK,
        "--url", url, "--dataset", "sample",
        "--max-requests", str(MAX_REQ), "--concurrency", str(CONC),
        "--sample-input-len", str(INPUT_LEN), "--sample-output-len", str(OUTPUT_LEN),
        "--sample-qps", "9999", "--timeout", "600",
        "--dump", os.path.join(LOG_DIR, f"results_{label}.json"),
        "--scenario-label", label,
    ]
    log.info("Benchmark: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK))
    dump_file = os.path.join(LOG_DIR, f"results_{label}.json")
    if os.path.exists(dump_file):
        with open(dump_file) as f:
            data = json.load(f)
        results = data.get("results", [])
        ok = [r for r in results if r.get("success")]
        total_in = sum(r.get("input_tokens", 0) for r in ok)
        total_out = sum(r.get("output_tokens", 0) for r in ok)
        wall = data.get("wall_duration_s", 1)
        ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
        tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
        out_thru = total_out / wall if wall > 0 else 0
        total_thru = (total_in + total_out) / wall if wall > 0 else 0
        avg_ttft = sum(ttft)/len(ttft) if ttft else 0
        avg_tpot = sum(tpot)/len(tpot) if tpot else 0
        return {
            "label": label, "out_tok_s": round(out_thru, 1),
            "total_tok_s": round(total_thru, 1),
            "ttft_ms": round(avg_ttft, 1), "tpot_ms": round(avg_tpot, 1),
            "ok": f"{len(ok)}/{len(results)}", "wall_s": round(wall, 1),
            "concurrency": CONC,
        }
    return None

# ─── Native at C=128 ───────────────────────────────────────────────────────

CONC = 128
MAX_REQ = 256

kill_all()
log.info("=" * 60)
log.info("  NATIVE tp=4, C=%d", CONC)
log.info("=" * 60)

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "4",
    "--host", "127.0.0.1", "--port", "30000",
    "--mem-fraction-static", "0.90", "--enable-metrics",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--skip-server-warmup", "--max-running-requests", str(CONC),
]
fh = open(os.path.join(LOG_DIR, "native_c128.log"), "w")
proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)

if not wait_port("127.0.0.1", 30000) or not wait_health("http://127.0.0.1:30000/health"):
    log.error("Native C=%d failed", CONC)
else:
    r = run_bench("http://127.0.0.1:30000", "native_c128")
    if r: ALL_RESULTS.append(r)
proc.terminate()
try: proc.wait(timeout=10)
except: pass
time.sleep(3)

# ─── Native at C=256 ───────────────────────────────────────────────────────

CONC = 256
MAX_REQ = 512

kill_all()
log.info("=" * 60)
log.info("  NATIVE tp=4, C=%d", CONC)
log.info("=" * 60)

cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "4",
    "--host", "127.0.0.1", "--port", "30000",
    "--mem-fraction-static", "0.90", "--enable-metrics",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--skip-server-warmup", "--max-running-requests", str(CONC),
]
fh = open(os.path.join(LOG_DIR, "native_c256.log"), "w")
proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)

if not wait_port("127.0.0.1", 30000) or not wait_health("http://127.0.0.1:30000/health"):
    log.error("Native C=%d failed", CONC)
else:
    r = run_bench("http://127.0.0.1:30000", "native_c256")
    if r: ALL_RESULTS.append(r)
proc.terminate()
try: proc.wait(timeout=10)
except: pass
time.sleep(3)

# ─── PD+AF M=1 at C=128 ────────────────────────────────────────────────────

for CONC, MAX_REQ, max_run in [(128, 256, 128), (256, 512, 256)]:
    label = f"pdaf_m1_c{CONC}"
    kill_all()
    log.info("=" * 60)
    log.info("  PD+AF M=1, C=%d, max_running=%d", CONC, max_run)
    log.info("=" * 60)

    PA_PORT, PF_PORT = 50010, 50011
    DA_PORT, DF_PORT = 50020, 50021
    BOOTSTRAP = 18999

    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"

    extra = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", "1", "--max-running-requests", str(max_run),
    ]

    procs = []
    def start_server(name, cmd, env, log_name):
        fh = open(os.path.join(LOG_DIR, log_name), "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))

    # PF GPU 2
    pf_env = env_base.copy()
    pf_env["CUDA_VISIBLE_DEVICES"] = "2"
    pf_env["AFD_UCX_BASE_PORT"] = "25100"
    pf_env["AFD_SCHED_PORT"] = "65300"
    pf_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(PF_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.90",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_server("PF", pf_cmd, pf_env, f"{label}_pf.log")

    # PA GPU 3
    pa_env = env_base.copy()
    pa_env["CUDA_VISIBLE_DEVICES"] = "3"
    pa_env["AFD_UCX_BASE_PORT"] = "25100"
    pa_env["AFD_SCHED_PORT"] = "65300"
    pa_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    pa_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(PA_PORT),
        "--enable-metrics",
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.90",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_server("PA", pa_cmd, pa_env, f"{label}_pa.log")

    if not wait_port("127.0.0.1", PA_PORT) or not wait_port("127.0.0.1", PF_PORT):
        log.error("PA/PF failed"); break

    # DF GPU 0
    df_env = env_base.copy()
    df_env["CUDA_VISIBLE_DEVICES"] = "0"
    df_env["AFD_UCX_BASE_PORT"] = "25200"
    df_env["AFD_SCHED_PORT"] = "65400"
    df_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(DF_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.90",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_server("DF", df_cmd, df_env, f"{label}_df.log")

    # DA GPU 1
    da_env = env_base.copy()
    da_env["CUDA_VISIBLE_DEVICES"] = "1"
    da_env["AFD_UCX_BASE_PORT"] = "25200"
    da_env["AFD_SCHED_PORT"] = "65400"
    da_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    da_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(DA_PORT),
        "--enable-metrics",
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.90",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra
    start_server("DA", da_cmd, da_env, f"{label}_da.log")

    if not wait_port("127.0.0.1", DA_PORT) or not wait_port("127.0.0.1", DF_PORT):
        log.error("DA/DF failed"); break

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
        log.error("Router failed"); break

    # Warmup
    import requests, concurrent.futures
    warm_payloads = [
        {"text": f"Warmup {i}", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(16)
    ]
    def _send(p):
        try: requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=p, timeout=120)
        except: pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(3)
    log.info("Warmup done, starting benchmark...")

    r = run_bench(f"http://127.0.0.1:{ROUTER_PORT}", label)
    if r: ALL_RESULTS.append(r)

    for name, p, fh in procs:
        try: os.killpg(os.getpgid(p.pid), 2)
        except: p.terminate()
    time.sleep(3)
    for name, p, fh in procs:
        try: p.kill()
        except: pass
        try: fh.close()
        except: pass

# ─── Final comparison ──────────────────────────────────────────────────────

print("\n" + "=" * 100)
print("  STRESS TEST RESULTS — Native vs PD+AF M=1 at multiple concurrency levels")
print("  Model: Qwen3-32B, Input=512, Output=128")
print("=" * 100)
print(f"  {'Config':<22} {'C':<6} {'Out(tok/s)':<14} {'Total(tok/s)':<14} {'TPOT(ms)':<12} {'TTFT(ms)':<12} {'OK':<10} {'Wall(s)':<10}")
print("  " + "-" * 104)
for r in ALL_RESULTS:
    print(f"  {r['label']:<22} {r['concurrency']:<6} {r['out_tok_s']:<14.1f} {r['total_tok_s']:<14.1f} {r['tpot_ms']:<12.1f} {r['ttft_ms']:<12.1f} {r['ok']:<10} {r['wall_s']:<10.1f}")
print("=" * 100)

# Save
with open(os.path.join(LOG_DIR, "stress_comparison.json"), "w") as f:
    json.dump(ALL_RESULTS, f, indent=2)
