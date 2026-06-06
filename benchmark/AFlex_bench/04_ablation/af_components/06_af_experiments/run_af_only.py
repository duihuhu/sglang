#!/usr/bin/env python3
"""Run AF-only (no PD) M=1 and M=3 at QPS=2, TP=2 each."""
import json, logging, os, socket, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("af_only")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "throughput_logs")
BENCHMARK = os.path.join(os.path.dirname(_HERE), "benchmark_replay.py")

MAX_REQUESTS = 500
OUTPUT_LEN = 128
INPUT_LEN = 512
CONCURRENCY = 500
A_PORT = 30000
F_PORT = 30001  # F also has HTTP but unused


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


def run_bench(label, qps, url):
    dump_file = os.path.join(LOG_DIR, f"results_{label}.json")
    cmd = [
        PYTHON, BENCHMARK,
        "--url", url,
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
        return {
            "label": label,
            "out_tok_s": round(out_thru, 1),
            "total_tok_s": round(total_thru, 1),
            "ttft_ms": round(avg_ttft, 1),
            "tpot_ms": round(avg_tpot, 1),
            "ok": f"{len(ok)}/{len(results)}",
            "wall_s": round(wall, 1),
        }
    return None


# ── Start AF-only servers ──────────────────────────────────────────────────

kill_all()

env_base = os.environ.copy()
env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
env_base["UCX_LOG_LEVEL"] = "fatal"

procs = []


def start_server(name, cmd, env, log_name):
    fh = open(os.path.join(LOG_DIR, log_name), "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append((name, p, fh))
    return p


all_results = []

for micro_batch in [1, 3]:
    label_m = f"afonly_m{micro_batch}_tp2"

    # F server: GPUs 2,3, TP=2, perspective=ffn
    f_env = env_base.copy()
    f_env["CUDA_VISIBLE_DEVICES"] = "2,3"
    f_env["AFD_UCX_BASE_PORT"] = "25100"
    f_env["AFD_SCHED_PORT"] = "65300"
    f_cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "2",
        "--host", "127.0.0.1", "--port", str(F_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", str(micro_batch), "--max-running-requests", "500",
    ]
    start_server(f"F_m{micro_batch}", f_cmd, f_env, f"{label_m}_f.log")
    log.info("F server (TP=2, M=%d) starting on GPUs 2,3", micro_batch)

    # A server: GPUs 0,1, TP=2, perspective=attn
    a_env = env_base.copy()
    a_env["CUDA_VISIBLE_DEVICES"] = "0,1"
    a_env["AFD_UCX_BASE_PORT"] = "25100"
    a_env["AFD_SCHED_PORT"] = "65300"
    a_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    a_cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "2",
        "--host", "127.0.0.1", "--port", str(A_PORT),
        "--enable-metrics",
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", str(micro_batch), "--max-running-requests", "500",
    ]
    start_server(f"A_m{micro_batch}", a_cmd, a_env, f"{label_m}_a.log")
    log.info("A server (TP=2, M=%d) starting on GPUs 0,1", micro_batch)

    # Wait for A to be ready
    if not wait_port("127.0.0.1", A_PORT, timeout=300):
        log.error("A server failed to start")
        sys.exit(1)
    if not wait_health(f"http://127.0.0.1:{A_PORT}/health", timeout=120):
        log.error("A health check failed")
        sys.exit(1)
    log.info("AF-only M=%d servers ready, benchmarking QPS=2...", micro_batch)

    # Warmup
    import requests, concurrent.futures
    warm_payloads = [
        {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(16)
    ]
    log.info("Warmup...")
    def _send(payload):
        requests.post(f"http://127.0.0.1:{A_PORT}/generate", json=payload, timeout=120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(5)
    log.info("Warmup done")

    # Benchmark at QPS=2
    result = run_bench(f"{label_m}_qps2", 2, f"http://127.0.0.1:{A_PORT}")
    if result:
        all_results.append(result)
        log.info("  Results: out=%.1f tok/s, TPOT=%.1f ms, TTFT=%.1f ms, wall=%.1fs",
                 result["out_tok_s"], result["tpot_ms"], result["ttft_ms"], result["wall_s"])

    # Cleanup servers for next M value
    for name, p, fh in procs:
        try: os.killpg(os.getpgid(p.pid), 2)
        except: p.terminate()
    time.sleep(3)
    for name, p, fh in procs:
        try: p.kill()
        except: pass
        try: fh.close()
        except: pass
    procs.clear()


# ── Final table ───────────────────────────────────────────────────────────
print("\n" + "=" * 90)
print("  AF-only (no PD) QPS=2 — Model: Qwen3-32B, Input=512, Output=128")
print("=" * 90)
print(f"  {'Label':<24} {'Out(tok/s)':<14} {'Total(tok/s)':<14} {'TPOT(ms)':<10} {'TTFT(ms)':<10} {'OK':<8} {'Wall(s)':<8}")
print("  " + "-" * 88)
for r in all_results:
    print(f"  {r['label']:<24} {r['out_tok_s']:<14.1f} {r['total_tok_s']:<14.1f} {r['tpot_ms']:<10.1f} {r['ttft_ms']:<10.1f} {r['ok']:<8} {r['wall_s']:<8.1f}")
print("=" * 90)

with open(os.path.join(LOG_DIR, "af_only_results.json"), "w") as f:
    json.dump(all_results, f, indent=2)

log.info("All done")
