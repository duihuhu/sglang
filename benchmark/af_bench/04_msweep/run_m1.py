#!/usr/bin/env python3
"""Run PD+AF M=1 benchmark (no overlap)."""
import json, logging, os, socket, subprocess, sys, time, shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("m1")

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

def kill_all():
    for t in ["sglang.launch_server", "sglang_router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)

def wait_port(host, port, timeout=180):
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

kill_all()

PA_PORT, PF_PORT = 50010, 50011
DA_PORT, DF_PORT = 50020, 50021
UCX_PA, UCX_PF = 25100, 25100
UCX_DA, UCX_DF = 25200, 25200
SCHED_P = 65300
SCHED_D = 65400
BOOTSTRAP = 18999
label = "pdaf_m1"

env_base = os.environ.copy()
env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
env_base["UCX_LOG_LEVEL"] = "fatal"

extra_decode = [
    "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--afd-micro-batch", "1", "--max-running-requests", "64",
]
extra_prefill = [
    "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--afd-micro-batch", "1", "--max-running-requests", "64",
]

procs = []

def start_server(name, cmd, env, log_name):
    fh = open(os.path.join(LOG_DIR, log_name), "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append((name, p, fh))
    return p

# PF GPU 2
pf_env = env_base.copy()
pf_env["CUDA_VISIBLE_DEVICES"] = "2"
pf_env["AFD_UCX_BASE_PORT"] = str(UCX_PF)
pf_env["AFD_SCHED_PORT"] = str(SCHED_P)
pf_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "1",
    "--host", "127.0.0.1", "--port", str(PF_PORT),
    "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "prefill",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
] + extra_prefill
log.info("Starting PF on GPU 2")
start_server("PF", pf_cmd, pf_env, f"{label}_pf.log")

# PA GPU 3
pa_env = env_base.copy()
pa_env["CUDA_VISIBLE_DEVICES"] = "3"
pa_env["AFD_UCX_BASE_PORT"] = str(UCX_PA)
pa_env["AFD_SCHED_PORT"] = str(SCHED_P)
pa_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
pa_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "1",
    "--host", "127.0.0.1", "--port", str(PA_PORT),
    "--enable-metrics",
    "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "prefill",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
] + extra_prefill
log.info("Starting PA on GPU 3")
start_server("PA", pa_cmd, pa_env, f"{label}_pa.log")

if not wait_port("127.0.0.1", PA_PORT, timeout=300) or not wait_port("127.0.0.1", PF_PORT, timeout=300):
    log.error("PA/PF failed to start")
    sys.exit(1)
log.info("PA and PF ready")

# DF GPU 0
df_env = env_base.copy()
df_env["CUDA_VISIBLE_DEVICES"] = "0"
df_env["AFD_UCX_BASE_PORT"] = str(UCX_DF)
df_env["AFD_SCHED_PORT"] = str(SCHED_D)
df_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "1",
    "--host", "127.0.0.1", "--port", str(DF_PORT),
    "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "decode",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
] + extra_decode
log.info("Starting DF on GPU 0")
start_server("DF", df_cmd, df_env, f"{label}_df.log")

# DA GPU 1
da_env = env_base.copy()
da_env["CUDA_VISIBLE_DEVICES"] = "1"
da_env["AFD_UCX_BASE_PORT"] = str(UCX_DA)
da_env["AFD_SCHED_PORT"] = str(SCHED_D)
da_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
da_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "1",
    "--host", "127.0.0.1", "--port", str(DA_PORT),
    "--enable-metrics",
    "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "decode",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
] + extra_decode
log.info("Starting DA on GPU 1")
start_server("DA", da_cmd, da_env, f"{label}_da.log")

if not wait_port("127.0.0.1", DA_PORT, timeout=300) or not wait_port("127.0.0.1", DF_PORT, timeout=300):
    log.error("DA/DF failed to start")
    sys.exit(1)
log.info("DA and DF ready")

# Router
router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
    "--pd-disaggregation", "--mini-lb",
    "--prefill", f"http://127.0.0.1:{PA_PORT}",
    "--decode", f"http://127.0.0.1:{DA_PORT}",
    "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
]
log.info("Starting router")
router_fh = open(os.path.join(LOG_DIR, f"{label}_router.log"), "w")
router_proc = subprocess.Popen(router_cmd, env=os.environ.copy(),
                               stdout=router_fh, stderr=subprocess.STDOUT,
                               start_new_session=True)
procs.append(("router", router_proc, router_fh))

if not wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=120):
    log.error("Router health check failed")
    sys.exit(1)
log.info("All PD+AF M=1 servers ready")

# Warmup
import requests, concurrent.futures
warm_payloads = [
    {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
    for i in range(16)
]
log.info("Warmup...")
def _send(payload):
    requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=payload, timeout=120)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(_send, warm_payloads))
time.sleep(3)
log.info("Warmup done")

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
log.info("Benchmark: %s", " ".join(cmd))
result = subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK))

dump_file = os.path.join(LOG_DIR, f"results_{label}.json")
if os.path.exists(dump_file):
    with open(dump_file) as f:
        data = json.load(f)
    results = data.get("results", [])
    ok = [r for r in results if r.get("success")]
    total_in = sum(r.get("input_tokens", 0) for r in ok)
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    wall = data.get("wall_duration_s", 1)
    in_thru = total_in / wall if wall > 0 else 0
    out_thru = total_out / wall if wall > 0 else 0
    total_thru = (total_in + total_out) / wall if wall > 0 else 0
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]

    log.info("=== %s ===", label)
    log.info("  Succeeded: %d/%d", len(ok), len(results))
    log.info("  Input throughput: %.1f tok/s", in_thru)
    log.info("  Output throughput: %.1f tok/s", out_thru)
    log.info("  Total throughput: %.1f tok/s", total_thru)
    log.info("  Mean TTFT: %.1f ms", sum(ttft)/len(ttft) if ttft else 0)
    log.info("  Mean TPOT: %.1f ms", sum(tpot)/len(tpot) if tpot else 0)

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

log.info("Done")

# Load existing comparison
with open(os.path.join(LOG_DIR, "comparison.json")) as f:
    all_results = json.load(f)

new_entry = {
    "label": label,
    "throughput_tok_s": round(out_thru, 1),
    "input_throughput_tok_s": round(in_thru, 1),
    "total_throughput_tok_s": round(total_thru, 1),
    "mean_ttft_ms": round(sum(ttft)/len(ttft), 1) if ttft else None,
    "mean_tpot_ms": round(sum(tpot)/len(tpot), 1) if tpot else None,
    "succeeded": len(ok), "total": len(results),
    "wall_s": round(wall, 1), "output_tokens": total_out
}
all_results.append(new_entry)
with open(os.path.join(LOG_DIR, "comparison.json"), "w") as f:
    json.dump(all_results, f, indent=2)

print("\n" + "=" * 100)
print("  THROUGHPUT COMPARISON (No shared prefix)")
print("=" * 100)
print(f"  {'Config':<30} {'Out(tok/s)':<12} {'Total(tok/s)':<14} {'TTFT(ms)':<10} {'TPOT(ms)':<10}")
print("  " + "-" * 76)
for r in all_results:
    total_s = f"{r.get('total_throughput_tok_s', 'N/A'):<14}" if isinstance(r.get('total_throughput_tok_s'), (int, float)) else f"{'N/A':<14}"
    ttft_s = f"{r['mean_ttft_ms']:<10.1f}" if r.get('mean_ttft_ms') else f"{'N/A':<10}"
    tpot_s = f"{r['mean_tpot_ms']:<10.1f}" if r.get('mean_tpot_ms') else f"{'N/A':<10}"
    print(f"  {r['label']:<30} {r['throughput_tok_s']:<12.1f} {total_s} {ttft_s} {tpot_s}")
print("=" * 100)
