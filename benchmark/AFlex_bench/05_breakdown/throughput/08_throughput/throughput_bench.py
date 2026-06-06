#!/usr/bin/env python3
"""Throughput benchmark: Native(tp=4) vs PD+AF w/ and w/o overlap schedule.

All use 4 GPUs (0,1,2,3).
"""

import json, logging, os, socket, subprocess, sys, time, shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("throughput")

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
    for t in ["sglang.launch_server", "sglang_router", "sglang::"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)


def wait_port(host, port, timeout=180):
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


def run_benchmark(url, label):
    cmd = [
        PYTHON, BENCHMARK,
        "--url", url,
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
        total_out = sum(r.get("output_tokens", 0) for r in ok)
        wall = data.get("wall_duration_s", 1)
        throughput = total_out / wall if wall > 0 else 0
        ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
        tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
        log.info("=== %s ===", label)
        log.info("  Succeeded: %d/%d", len(ok), len(results))
        log.info("  Output throughput: %.1f tok/s", throughput)
        log.info("  Total output tokens: %d", total_out)
        log.info("  Wall duration: %.1f s", wall)
        if ttft:
            log.info("  Mean TTFT: %.1f ms", sum(ttft) / len(ttft))
        if tpot:
            log.info("  Mean TPOT: %.1f ms", sum(tpot) / len(tpot))
        return {
            "label": label,
            "throughput_tok_s": round(throughput, 1),
            "mean_ttft_ms": round(sum(ttft) / len(ttft), 1) if ttft else None,
            "mean_tpot_ms": round(sum(tpot) / len(tpot), 1) if tpot else None,
            "succeeded": len(ok),
            "total": len(results),
            "wall_s": round(wall, 1),
            "output_tokens": total_out,
        }
    return None


# ─── Config 1: Native tp=4 ───────────────────────────────────────────────────

def run_native_tp4():
    kill_all()
    log.info("=" * 60)
    log.info("  CONFIG 1: Native (tp=4) on GPUs 0,1,2,3")
    log.info("=" * 60)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

    cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--tp", "4",
        "--host", "127.0.0.1",
        "--port", "30000",
        "--mem-fraction-static", "0.90",
        "--enable-metrics",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--skip-server-warmup",
        "--max-running-requests", "128",
    ]
    log.info("Launch: %s", " ".join(cmd))
    fh = open(os.path.join(LOG_DIR, "native_tp4.log"), "w")
    proc = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                            start_new_session=True)

    if not wait_port("127.0.0.1", 30000):
        log.error("Native server failed to start")
        proc.terminate()
        return None
    if not wait_health("http://127.0.0.1:30000/health", timeout=120):
        log.error("Native health check failed")
        proc.terminate()
        return None
    log.info("Native tp=4 ready")

    result = run_benchmark("http://127.0.0.1:30000", "native_tp4")

    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
    time.sleep(3)
    return result


# ─── Config 2+3: PD+AF ───────────────────────────────────────────────────────

def run_pdaf(enable_overlap, label):
    kill_all()

    if enable_overlap:
        log.info("=" * 60)
        log.info("  CONFIG: PD+AF WITH overlap schedule on GPUs 0,1,2,3")
        log.info("=" * 60)
    else:
        log.info("=" * 60)
        log.info("  CONFIG: PD+AF WITHOUT overlap schedule on GPUs 0,1,2,3")
        log.info("=" * 60)

    PA_PORT, PF_PORT = 50010, 50011
    DA_PORT, DF_PORT = 50020, 50021
    UCX_PA, UCX_PF = 25100, 25100
    UCX_DA, UCX_DF = 25200, 25200
    SCHED_P = 65300
    SCHED_D = 65400
    BOOTSTRAP = 18999

    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"

    # Overlap schedule only for decode — prefill crashes with CUDA index OOB under
    # parallel CPU scheduling due to data races in the prefill batch preparation.
    overlap_decode = ["--afd-enable-overlap-schedule"] if enable_overlap else []
    overlap_prefill = []  # prefill stays serial for stability

    extra_decode = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", "3", "--max-running-requests", "64",
    ] + overlap_decode
    extra_prefill = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", "3", "--max-running-requests", "64",
    ] + overlap_prefill

    procs = []

    def start_server(name, cmd, env, log_name):
        fh = open(os.path.join(LOG_DIR, log_name), "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
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
    log.info("Starting PF (Prefill FFN) on GPU 2")
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
    log.info("Starting PA (Prefill Attn) on GPU 3")
    start_server("PA", pa_cmd, pa_env, f"{label}_pa.log")

    if not wait_port("127.0.0.1", PA_PORT) or not wait_port("127.0.0.1", PF_PORT):
        log.error("PA/PF failed to start")
        return cleanup_and_return(procs)

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
    log.info("Starting DF (Decode FFN) on GPU 0")
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
    log.info("Starting DA (Decode Attn) on GPU 1")
    start_server("DA", da_cmd, da_env, f"{label}_da.log")

    if not wait_port("127.0.0.1", DA_PORT) or not wait_port("127.0.0.1", DF_PORT):
        log.error("DA/DF failed to start")
        return cleanup_and_return(procs)

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
        return cleanup_and_return(procs)
    log.info("All PD+AF servers ready")

    # Warmup
    import requests
    import concurrent.futures
    warm_payloads = [
        {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(16)
    ]
    log.info("Warmup (%d concurrent requests)...", len(warm_payloads))
    def _send(payload):
        requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=payload,
                      timeout=120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(3)
    log.info("Warmup done")

    result = run_benchmark(f"http://127.0.0.1:{ROUTER_PORT}", label)

    return cleanup_and_return(procs, result)


def cleanup_and_return(procs, result=None):
    for name, p, fh in procs:
        try:
            os.killpg(os.getpgid(p.pid), 2)  # SIGINT
        except Exception:
            p.terminate()
    time.sleep(3)
    for name, p, fh in procs:
        try:
            p.kill()
        except Exception:
            pass
        try:
            fh.close()
        except Exception:
            pass
    return result


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    shutil.rmtree(LOG_DIR, ignore_errors=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    all_results = []

    # 1. Native tp=4
    r = run_native_tp4()
    if r:
        all_results.append(r)

    # 2. PD+AF without overlap
    r = run_pdaf(enable_overlap=False, label="pdaf_no_overlap")
    if r:
        all_results.append(r)

    # 3. PD+AF with overlap
    r = run_pdaf(enable_overlap=True, label="pdaf_with_overlap")
    if r:
        all_results.append(r)

    # Summary
    print("\n" + "=" * 100)
    print("  THROUGHPUT COMPARISON: 4-GPU Configs")
    print("  Model: Qwen3-32B, Concurrency=%d, MaxReq=%d, OutputLen=%d" %
          (CONCURRENCY, MAX_REQUESTS, OUTPUT_LEN))
    print("=" * 100)
    header = f"  {'Config':<35} {'Thru(tok/s)':<14} {'TTFT(ms)':<12} {'TPOT(ms)':<12} {'Success':<10}"
    print(header)
    print("  " + "-" * 83)
    for r in all_results:
        line = f"  {r['label']:<35} {r['throughput_tok_s']:<14.1f} "
        if r['mean_ttft_ms']:
            line += f"{r['mean_ttft_ms']:<12.1f} "
        else:
            line += f"{'N/A':<12} "
        if r['mean_tpot_ms']:
            line += f"{r['mean_tpot_ms']:<12.1f} "
        else:
            line += f"{'N/A':<12} "
        line += f"{r['succeeded']}/{r['total']}"
        print(line)
    print("=" * 100)

    # Save results
    out_path = os.path.join(LOG_DIR, "comparison.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
