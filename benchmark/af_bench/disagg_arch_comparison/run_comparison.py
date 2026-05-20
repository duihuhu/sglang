#!/usr/bin/env python3
"""
Compare 6 disaggregation architectures on GPUs 4-7.
Quick single-run comparison: one concurrency per config, all metrics.
"""
import json, logging, os, socket, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("comparison")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "logs")
os.makedirs(LOG_DIR, exist_ok=True)
BENCHMARK = os.path.join(os.path.dirname(HERE), "benchmark_replay.py")

# ---- unified params ----
# P-light / D-heavy: short input = fast prefill → requests quickly enter decode;
# long output = many decode steps → large decode batch for M=3 pipeline overlap.
INPUT_LEN = 128
OUTPUT_LEN = 512
MAX_REQUESTS = 400
# max_running_requests differs per architecture:
# - native/pd: full model ~64 GB, short input → more space for KV cache
# - af: attn-only GPU (~8 GB weights) → ~60 GB for KV cache → can handle very large batches
# - pdaf: decode-FFN GPU activation-limited, but 128 input ≈ 8× less activation than 1024
MAX_RUNNING = 64             # native_tp1, pd_only
AF_MAX_RUNNING = 256         # af_m1, af_m3_opt (KV cache ~32MB/req at start, 60GB = 1800+ reqs)
PD_AF_MAX_RUNNING = 96       # pdaf_m1, pdaf_m3_opt
BOOTSTRAP = 18999
UCX_TLS = "rc,tcp,cuda_copy,cuda_ipc"

# ---- per-architecture QPS (scaled by GPU count) ----
Q = 16
QPS = {
    "native_tp1":         Q,       # 16  (1 GPU)
    "pd_only":            2 * Q,   # 32  (2 GPU)
    "af_m1":              2 * Q,   # 32  (2 GPU)
    "af_m3_opt":          2 * Q,   # 32  (2 GPU)
    "pdaf_m1":            4 * Q,   # 64  (4 GPU)
    "pdaf_m3_opt":        4 * Q,   # 64  (4 GPU)
}
# All configs share the same client concurrency
CLIENT_CONCURRENCY = 256

GPU_COUNT = {
    "native_tp1": 1, "pd_only": 2, "af_m1": 2,
    "af_m3_opt": 2, "pdaf_m1": 4, "pdaf_m3_opt": 4,
}

# Per-arch max_running_requests
MAX_RUNNING_MAP = {
    "native_tp1": MAX_RUNNING,
    "pd_only": MAX_RUNNING,
    "af_m1": AF_MAX_RUNNING,
    "af_m3_opt": AF_MAX_RUNNING,
    "pdaf_m1": PD_AF_MAX_RUNNING,
    "pdaf_m3_opt": PD_AF_MAX_RUNNING,
}

# Global batch stats collected per config
ALL_RESULTS = []
BATCH_STATS_ALL = {}


def collect_batch_stats(log_files, label):
    """Parse AFD_DBG lines from server logs to extract per-iteration batch sizes.

    Returns a dict with batch size summary per perspective (ATTN/FFN).
    """
    import re, glob as _glob
    batch_re = re.compile(r'\[AFD_DBG\] === ITER \d+ (\w+) batch=(\d+)')
    layer_batch_re = re.compile(r'\[AFD_DBG\] L\s*\d+ batch=(\d+)')

    stats = {}
    for logf in log_files:
        if not os.path.exists(logf):
            continue
        batches = []
        with open(logf) as f:
            for line in f:
                m = batch_re.search(line)
                if m:
                    perspective = m.group(1)
                    bsz = int(m.group(2))
                    batches.append(bsz)
        if batches:
            key = os.path.basename(logf).replace(f"{label}_", "").replace(".log", "")
            s = sorted(batches)
            n = len(s)
            stats[key] = {
                "n_iters": n,
                "min": s[0],
                "max": s[-1],
                "mean": round(sum(s) / n, 1),
                "p50": s[n // 2],
                "p95": s[int(n * 0.95)],
                "p99": s[int(n * 0.99)],
            }
    return stats


def collect_and_store(label):
    """Find server logs for 'label' and collect batch size stats."""
    import glob as _glob
    log_files = _glob.glob(os.path.join(LOG_DIR, f"{label}*.log"))
    bstat = collect_batch_stats(log_files, label)
    if bstat:
        BATCH_STATS_ALL[label] = bstat
    return bstat


ALL_PORTS = [30000, 30001, 50000, 50010, 50011, 50020, 50021]

def _ports_free():
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    busy = [p for p in ALL_PORTS if f":{p}" in out]
    return busy

def kill_all():
    # Graceful shutdown first (SIGTERM) to let UCX clean up listeners
    for t in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(8)
    # Force kill (SIGKILL)
    for t in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)
    subprocess.run(["pkill", "-9", "-f", "ucx"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "ucp"], capture_output=True)
    subprocess.run(["sync"], capture_output=True)
    # Wait for ports to be released (up to 60s)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        busy = _ports_free()
        if not busy:
            break
        log.info("Waiting for ports to be freed: %s", busy)
        time.sleep(5)
    else:
        log.error("Ports still busy after 60s: %s", busy)


def cleanup_procs(procs):
    """Kill all processes in a list of (name, Popen, filehandle) tuples."""
    for _, p, f in procs:
        try:
            os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        try:
            f.close()
        except Exception:
            pass
    time.sleep(3)


def wait_port(host, port, timeout=300):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=2)
            s.close()
            return True
        except Exception:
            time.sleep(2)
    return False


def wait_health(url, timeout=180):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except Exception:
            time.sleep(3)
    return False


def warmup(url):
    import requests, concurrent.futures
    payloads = [
        {"text": f"Hello world {i}, warmup req:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(12)
    ]
    def _send(p):
        try:
            requests.post(url, json=p, timeout=120)
        except Exception:
            pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:
        list(ex.map(_send, payloads))
    time.sleep(3)


def run_benchmark(label, url, concurrency, qps):
    log.info("Benchmark: %s concurrency=%d qps=%d", label, concurrency, qps)
    dump_file = os.path.join(LOG_DIR, f"result_{label}.json")
    cmd = [
        PYTHON, BENCHMARK, "--dataset", "sample",
        "--url", url,
        "--max-requests", str(MAX_REQUESTS),
        "--concurrency", str(concurrency),
        "--sample-input-len", str(INPUT_LEN),
        "--sample-output-len", str(OUTPUT_LEN),
        "--sample-qps", str(qps),
        "--timeout", "900",
        "--dump", dump_file,
        "--scenario-label", label,
    ]
    result = subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK),
                            capture_output=False)
    if not os.path.exists(dump_file):
        log.error("Result file missing: %s", dump_file)
        return None
    with open(dump_file) as f:
        data = json.load(f)
    results = data.get("results", [])
    ok = [r for r in results if r.get("success")]
    total_in = sum(r.get("input_tokens", 0) for r in ok)
    total_out = sum(r.get("output_tokens", 0) for r in ok)
    wall = data.get("wall_duration_s", 1) or 1
    out_thru = total_out / wall
    total_thru = (total_in + total_out) / wall
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    gpus = GPU_COUNT[label]
    record = {
        "label": label, "gpus": gpus,
        "concurrency": concurrency, "qps": qps,
        "output_thru_tok_s": round(out_thru, 1),
        "total_thru_tok_s": round(total_thru, 1),
        "mean_ttft_ms": round(sum(ttft) / len(ttft), 1) if ttft else None,
        "mean_tpot_ms": round(sum(tpot) / len(tpot), 1) if tpot else None,
        "p50_ttft_ms": round(sorted(ttft)[len(ttft)//2], 1) if ttft else None,
        "p95_ttft_ms": round(sorted(ttft)[int(len(ttft)*0.95)], 1) if ttft else None,
        "p50_tpot_ms": round(sorted(tpot)[len(tpot)//2], 1) if tpot else None,
        "p95_tpot_ms": round(sorted(tpot)[int(len(tpot)*0.95)], 1) if tpot else None,
        "succeeded": f"{len(ok)}/{len(results)}",
        "wall_s": round(wall, 1),
    }
    log.info("  %s: out=%.1f tok/s  TTFT=%.1f ms  TPOT=%.1f ms  ok=%d/%d",
             label, out_thru, record["mean_ttft_ms"] or 0,
             record["mean_tpot_ms"] or 0, len(ok), len(results))
    return record



# =============================================================
# Config 1: Native TP=1  (GPU 4)
# =============================================================
def run_native_tp1():
    label = "native_tp1"
    kill_all()
    procs = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "4"
    env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    port = 30000
    mr = MAX_RUNNING_MAP[label]
    cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(port),
        "--mem-fraction-static", "0.85",
        "--max-running-requests", str(mr),
        "--skip-server-warmup",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    ]
    fh = open(os.path.join(LOG_DIR, f"{label}.log"), "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("native", p, fh))
    if not wait_port("127.0.0.1", port, timeout=300):
        log.error("native_tp1 failed to start")
        return None
    url = f"http://127.0.0.1:{port}"
    warmup(url)
    r = run_benchmark(label, url, CLIENT_CONCURRENCY, QPS[label])
    collect_and_store(label)
    cleanup_procs(procs)
    return r


# =============================================================
# Config 2: PD-only 2-GPU  (GPU 4,5)
# =============================================================
def run_pd_only():
    label = "pd_only"
    kill_all()
    procs = []
    env = os.environ.copy()
    env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    p_port, d_port, r_port = 50010, 50020, 50000
    mr = MAX_RUNNING_MAP[label]

    # Prefill GPU 4
    env_p = env.copy()
    env_p["CUDA_VISIBLE_DEVICES"] = "4"
    p_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(p_port),
        "--mem-fraction-static", "0.85",
        "--max-running-requests", str(mr),
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup",
    ]
    fh = open(os.path.join(LOG_DIR, f"{label}_prefill.log"), "w")
    p = subprocess.Popen(p_cmd, env=env_p, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("prefill", p, fh))

    # Decode GPU 5
    env_d = env.copy()
    env_d["CUDA_VISIBLE_DEVICES"] = "5"
    d_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(d_port),
        "--mem-fraction-static", "0.85",
        "--max-running-requests", str(mr),
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
        "--skip-server-warmup",
    ]
    fh2 = open(os.path.join(LOG_DIR, f"{label}_decode.log"), "w")
    p2 = subprocess.Popen(d_cmd, env=env_d, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("decode", p2, fh2))

    if not wait_port("127.0.0.1", p_port) or not wait_port("127.0.0.1", d_port):
        log.error("pd_only: servers failed to start")
        return None

    # Router
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", f"http://127.0.0.1:{p_port}",
        "--decode", f"http://127.0.0.1:{d_port}",
        "--host", "127.0.0.1", "--port", str(r_port),
    ]
    rf = open(os.path.join(LOG_DIR, f"{label}_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health(f"http://127.0.0.1:{r_port}/health", timeout=120):
        log.error("pd_only: router failed")
        return None

    url = f"http://127.0.0.1:{r_port}"
    warmup(url)
    r = run_benchmark(label, url, CLIENT_CONCURRENCY, QPS[label])
    collect_and_store(label)
    cleanup_procs(procs)
    return r


# =============================================================
# Config 3: AF-only  (GPU 4=Attn, 5=FFN)  — supports M=1 and M=3-opt
# =============================================================
def run_af_only(m_stage, async_sched, label):
    kill_all()
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = UCX_TLS
    env_base["UCX_LOG_LEVEL"] = "fatal"
    a_port, f_port = 30000, 30001
    # Use different UCX ports per M-stage to avoid "Device is busy" between configs
    ucx_base = 27000 + m_stage * 100  # M=1→27100, M=3→27300
    sched = 67000 + m_stage * 100        # M=1→67100, M=3→67300
    mr = MAX_RUNNING_MAP[label]

    extra = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", str(m_stage),
        "--max-running-requests", str(mr),
    ]
    if async_sched:
        extra.append("--afd-async-schedule")

    # FFN GPU 5 (start first - listener)
    f_env = env_base.copy()
    f_env["CUDA_VISIBLE_DEVICES"] = "5"
    f_env["AFD_UCX_BASE_PORT"] = str(ucx_base)
    f_env["AFD_SCHED_PORT"] = str(sched)
    f_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(f_port),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
    ] + extra
    fh = open(os.path.join(LOG_DIR, f"{label}_f.log"), "w")
    pf = subprocess.Popen(f_cmd, env=f_env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("FFN", pf, fh))

    time.sleep(2)

    # Attn GPU 4 (connects to FFN)
    a_env = env_base.copy()
    a_env["CUDA_VISIBLE_DEVICES"] = "4"
    a_env["AFD_UCX_BASE_PORT"] = str(ucx_base)
    a_env["AFD_SCHED_PORT"] = str(sched)
    a_env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    a_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(a_port),
        "--afd-perspective", "attn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.85",
    ] + extra
    fh2 = open(os.path.join(LOG_DIR, f"{label}_a.log"), "w")
    pa = subprocess.Popen(a_cmd, env=a_env, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("Attn", pa, fh2))

    if not wait_port("127.0.0.1", a_port) or not wait_port("127.0.0.1", f_port):
        log.error("%s: servers failed", label)
        return None
    if not wait_health(f"http://127.0.0.1:{a_port}/health", timeout=60):
        log.error("%s: health check failed", label)
        return None

    url = f"http://127.0.0.1:{a_port}"
    warmup(url)
    r = run_benchmark(label, url, CLIENT_CONCURRENCY, QPS[label])
    collect_and_store(label)
    cleanup_procs(procs)
    return r


# =============================================================
# Config 4: PD+AF 4-GPU  (DF=4, DA=5, PF=6, PA=7)
# =============================================================
def run_pd_af(m_stage, async_sched, label):
    kill_all()
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = UCX_TLS
    env_base["UCX_LOG_LEVEL"] = "fatal"
    da_port, df_port = 50020, 50021
    pa_port, pf_port = 50010, 50011
    router_port = 50000
    # Use different UCX ports per M-stage to avoid "Device is busy" between configs
    ucx_p = 25100 + m_stage * 100  # M=1→25200, M=3→25400
    ucx_d = 25200 + m_stage * 100  # M=1→25300, M=3→25500
    sched_p = 65300 + m_stage * 100  # M=1→65400, M=3→65600
    sched_d = 65400 + m_stage * 100  # M=1→65500, M=3→65700

    extra = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", str(m_stage),
        "--max-running-requests", str(MAX_RUNNING_MAP[label]),
        "--afd-disagg-interleave-poll",
    ]
    if async_sched:
        extra.append("--afd-async-schedule")

    def _start(name, gpu, perspective, disagg_mode, port, ucx_base, sched_port,
               ffn_host=None, log_suffix=""):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["AFD_UCX_BASE_PORT"] = str(ucx_base)
        env["AFD_SCHED_PORT"] = str(sched_port)
        env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
        env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
        if ffn_host:
            env["AFD_UCX_FFN_HOST"] = ffn_host
        cmd = [PYTHON, "-m", "sglang.launch_server",
            "--model-path", MODEL, "--tp", "1",
            "--host", "127.0.0.1", "--port", str(port),
            "--afd-perspective", perspective, "--afd-comm-backend", "ucx",
            "--mem-fraction-static", "0.85",
            "--disaggregation-mode", disagg_mode,
            "--disaggregation-transfer-backend", "mooncake",
            "--disaggregation-bootstrap-port", str(BOOTSTRAP),
            "--disaggregation-ib-device", "mlx5_4",
        ] + extra
        fh = open(os.path.join(LOG_DIR, f"{label}_{name}.log"), "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        procs.append((name, p, fh))
        return p

    # PF GPU 6
    _start("pf", 6, "ffn", "prefill", pf_port, ucx_p, sched_p)
    # PA GPU 7
    _start("pa", 7, "attn", "prefill", pa_port, ucx_p, sched_p, ffn_host="127.0.0.1")

    if not wait_port("127.0.0.1", pa_port, timeout=300) or not wait_port("127.0.0.1", pf_port, timeout=300):
        log.error("%s: prefill servers failed", label)
        return None

    # DF GPU 4
    _start("df", 4, "ffn", "decode", df_port, ucx_d, sched_d)
    # DA GPU 5
    _start("da", 5, "attn", "decode", da_port, ucx_d, sched_d, ffn_host="127.0.0.1")

    if not wait_port("127.0.0.1", da_port) or not wait_port("127.0.0.1", df_port):
        log.error("%s: decode servers failed", label)
        return None

    # Health-check prefill and decode Attn servers before starting router
    if not wait_health(f"http://127.0.0.1:{pa_port}/health", timeout=120):
        log.error("%s: PA health check failed", label)
        return None
    if not wait_health(f"http://127.0.0.1:{da_port}/health", timeout=120):
        log.error("%s: DA health check failed", label)
        return None
    # Extra delay for Mooncake bootstrap to fully initialize
    time.sleep(5)

    # Router
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", f"http://127.0.0.1:{pa_port}",
        "--decode", f"http://127.0.0.1:{da_port}",
        "--host", "127.0.0.1", "--port", str(router_port),
    ]
    rf = open(os.path.join(LOG_DIR, f"{label}_router.log"), "w")
    rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not wait_health(f"http://127.0.0.1:{router_port}/health", timeout=120):
        log.error("%s: router failed", label)
        return None

    url = f"http://127.0.0.1:{router_port}"
    warmup(url)
    r = run_benchmark(label, url, CLIENT_CONCURRENCY, QPS[label])
    collect_and_store(label)
    cleanup_procs(procs)
    return r


# =============================================================
# Main
# =============================================================
def main():
    print("=" * 110)
    print("  DISAGG ARCHITECTURE COMPARISON  (GPUs 4-7, Qwen3-32B, Q=%d, input=%d, output=%d, N=%d)" % (Q, INPUT_LEN, OUTPUT_LEN, MAX_REQUESTS))
    print("=" * 110)

    # Resume: native_tp1 already completed (output=767.4 tok/s, 400/400)
    ALL_RESULTS.append({
        "label": "native_tp1", "gpus": 1,
        "concurrency": CLIENT_CONCURRENCY, "qps": QPS["native_tp1"],
        "output_thru_tok_s": 767.4, "total_thru_tok_s": 961.0,
        "mean_ttft_ms": 733.0, "mean_tpot_ms": 243.9,
        "p50_ttft_ms": 854.7, "p95_ttft_ms": 1192.4,
        "p50_tpot_ms": 284.4, "p95_tpot_ms": 396.7,
        "succeeded": "400/400", "wall_s": 266.9,
    })
    # Also pre-populate pd_only (already completed)
    ALL_RESULTS.append({
        "label": "pd_only", "gpus": 2,
        "concurrency": CLIENT_CONCURRENCY, "qps": QPS["pd_only"],
        "output_thru_tok_s": 731.7, "total_thru_tok_s": 917.0,
        "mean_ttft_ms": 734.4, "mean_tpot_ms": 244.4,
        "succeeded": "400/400", "wall_s": 279.9,
    })

    configs = [
        # ("native_tp1",     lambda: run_native_tp1()),  # already done
        # ("pd_only",        lambda: run_pd_only()),     # already done
        ("af_m1",          lambda: run_af_only(1, False, "af_m1")),
        ("af_m3_opt",      lambda: run_af_only(3, True, "af_m3_opt")),
        ("pdaf_m1",        lambda: run_pd_af(1, False, "pdaf_m1")),
        ("pdaf_m3_opt",    lambda: run_pd_af(3, True, "pdaf_m3_opt")),
    ]

    for name, fn in configs:
        log.info(">>> Running %s <<<", name)
        try:
            r = fn()
        except Exception as e:
            log.exception("Config %s crashed: %s", name, e)
            r = None
        if r:
            ALL_RESULTS.append(r)
        # Small cooldown between configs
        time.sleep(5)

    # ---- Summary ----
    print("\n" + "=" * 110)
    print("  COMPARISON SUMMARY")
    print("=" * 110)
    header = f"  {'Config':<18} {'GPU':>4} {'Conc':>5} {'QPS':>5} {'Out(tok/s)':>12} {'Total(tok/s)':>12} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'P50TTFT':>10} {'P95TTFT':>10} {'P50TPOT':>10} {'P95TPOT':>10} {'OK':>10}"
    print(header)
    print("  " + "-" * 114)
    for r in ALL_RESULTS:
        print(f"  {r['label']:<18} {str(r['gpus']):>4} {r['concurrency']:>5} {r['qps']:>5} "
              f"{r['output_thru_tok_s']:>12.1f} {r['total_thru_tok_s']:>12.1f} "
              f"{str(r['mean_ttft_ms']):>10} {str(r['mean_tpot_ms']):>10} "
              f"{str(r.get('p50_ttft_ms','')):>10} {str(r.get('p95_ttft_ms','')):>10} "
              f"{str(r.get('p50_tpot_ms','')):>10} {str(r.get('p95_tpot_ms','')):>10} "
              f"{r['succeeded']:>10}")
    print("=" * 110)

    # Batch size statistics
    if BATCH_STATS_ALL:
        print("\n--- Batch Size Per Iteration ---")
        for label, bstats in sorted(BATCH_STATS_ALL.items()):
            for log_key, bs in sorted(bstats.items()):
                print(f"  {label:<18} [{log_key:<20}] "
                      f"iters={bs['n_iters']:>5}  "
                      f"min={bs['min']:>4}  max={bs['max']:>4}  "
                      f"mean={bs['mean']:>6.1f}  p50={bs['p50']:>4}  "
                      f"p95={bs['p95']:>4}  p99={bs['p99']:>4}")

    # Per-GPU efficiency
    print("\n--- Per-GPU Efficiency ---")
    for r in ALL_RESULTS:
        gpu_count = r['gpus']
        per_gpu = r['output_thru_tok_s'] / gpu_count if isinstance(gpu_count, int) else 0
        print(f"  {r['label']:<18} {r['output_thru_tok_s']:>8.1f} tok/s  / {gpu_count} GPU = {per_gpu:>8.1f} tok/s/GPU")

    with open(os.path.join(HERE, "results.json"), "w") as f:
        json.dump({"results": ALL_RESULTS, "batch_stats": BATCH_STATS_ALL}, f, indent=2)
    log.info("Results saved to %s", os.path.join(HERE, "results.json"))


if __name__ == "__main__":
    main()
