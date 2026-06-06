#!/usr/bin/env python3
"""Run Native TP4 at multiple QPS levels: 2, 4, 6, 8 req/s, 500 requests each."""
import json, logging, os, socket, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("qps_sweep_native")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "throughput_logs")
BENCHMARK = os.path.join(os.path.dirname(_HERE), "benchmark_replay.py")

QPS_VALUES = [2, 4, 6, 8]
MAX_REQUESTS = 500
OUTPUT_LEN = 128
INPUT_LEN = 512
CONCURRENCY = 500
SERVER_PORT = 30000


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


# ── Start Native TP4 server ──────────────────────────────────────────────

kill_all()

env = os.environ.copy()
env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

server_cmd = [
    PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "4",
    "--host", "127.0.0.1", "--port", str(SERVER_PORT),
    "--mem-fraction-static", "0.90",
    "--enable-metrics",
    "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--max-running-requests", str(CONCURRENCY),
]
log.info("Starting Native TP4 server: %s", " ".join(server_cmd))
fh = open(os.path.join(LOG_DIR, "qps_sweep_native.log"), "w")
proc = subprocess.Popen(server_cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                        start_new_session=True)

if not wait_port("127.0.0.1", SERVER_PORT, timeout=300):
    log.error("Native server failed to start")
    sys.exit(1)
if not wait_health(f"http://127.0.0.1:{SERVER_PORT}/health", timeout=120):
    log.error("Native health check failed")
    sys.exit(1)
log.info("Native TP4 server ready")

# ── Warmup ────────────────────────────────────────────────────────────────
import requests, concurrent.futures
warm_payloads = [
    {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
    for i in range(16)
]
log.info("Warmup...")
def _send(payload):
    requests.post(f"http://127.0.0.1:{SERVER_PORT}/generate", json=payload, timeout=120)
with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
    list(ex.map(_send, warm_payloads))
time.sleep(5)
log.info("Warmup done")

# ── Run benchmarks at each QPS ────────────────────────────────────────────
all_results = []

for qps in QPS_VALUES:
    label = f"native_tp4_qps{qps}"
    dump_file = os.path.join(LOG_DIR, f"results_{label}.json")
    log.info("=" * 60)
    log.info("  Benchmark: QPS=%d, requests=%d, ol=%d", qps, MAX_REQUESTS, OUTPUT_LEN)
    log.info("=" * 60)

    cmd = [
        PYTHON, BENCHMARK,
        "--url", f"http://127.0.0.1:{SERVER_PORT}",
        "--dataset", "sample",
        "--max-requests", str(MAX_REQUESTS),
        "--concurrency", str(CONCURRENCY),
        "--sample-input-len", str(INPUT_LEN),
        "--sample-output-len", str(OUTPUT_LEN),
        "--sample-qps", str(qps),
        "--timeout", "600",
        "--dump", dump_file,
        "--scenario-label", label,
    ]
    log.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK))

    # Parse results
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

        entry = {
            "label": label,
            "qps": qps,
            "out_tok_s": round(out_thru, 1),
            "total_tok_s": round(total_thru, 1),
            "ttft_ms": round(avg_ttft, 1),
            "tpot_ms": round(avg_tpot, 1),
            "ok": f"{len(ok)}/{len(results)}",
            "wall_s": round(wall, 1),
        }
        all_results.append(entry)
        log.info("  Results: out=%.1f tok/s, TPOT=%.1f ms, TTFT=%.1f ms, wall=%.1fs",
                 out_thru, avg_tpot, avg_ttft, wall)
    else:
        log.error("  No results file found!")

# ── Final table ───────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  Native TP4 QPS Sweep — QPS 2,4,6,8")
print("  Model: Qwen3-32B, Input=512, Output=128, 500 requests each")
print("=" * 90)
print(f"  {'Label':<26} {'QPS':<6} {'Out(tok/s)':<14} {'Total(tok/s)':<14} {'TPOT(ms)':<10} {'TTFT(ms)':<10} {'OK':<8} {'Wall(s)':<8}")
print("  " + "-" * 96)
for r in all_results:
    print(f"  {r['label']:<26} {r['qps']:<6} {r['out_tok_s']:<14.1f} {r['total_tok_s']:<14.1f} {r['tpot_ms']:<10.1f} {r['ttft_ms']:<10.1f} {r['ok']:<8} {r['wall_s']:<8.1f}")
print("=" * 90)

# Save
with open(os.path.join(LOG_DIR, "qps_sweep_native_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

# ── Cleanup ───────────────────────────────────────────────────────────────
proc.terminate()
time.sleep(3)
try:
    proc.kill()
except:
    pass
fh.close()

log.info("All done")
