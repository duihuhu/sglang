#!/usr/bin/env python3
"""
Quick comparison: PD+AF M=1 vs PD+AF M=3 (with interleave poll).
Uses GPUs 4-7: DF=4, DA=5, PF=6, PA=7.
"""
import json, logging, os, socket, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pdaf_cmp")

PYTHON = "/workspace/env/af-test/bin/python3"
MODEL = "/models/Qwen/Qwen3-32B/"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "pdaf_m1_vs_m3_logs")
os.makedirs(LOG_DIR, exist_ok=True)
BENCHMARK = os.path.join(HERE, "benchmark_replay.py")

INPUT_LEN = 128
OUTPUT_LEN = 512
MAX_REQUESTS = 200
PD_AF_MAX_RUNNING = 96
BOOTSTRAP = 18999
UCX_TLS = "rc,tcp,cuda_copy,cuda_ipc"
CLIENT_CONCURRENCY = 128
QPS = 32

ALL_PORTS = [50000, 50010, 50011, 50020, 50021]


def _ports_free():
    out = subprocess.run(["ss", "-tlnp"], capture_output=True, text=True).stdout
    return [p for p in ALL_PORTS if f":{p}" in out]


def kill_all():
    for t in ["sglang.launch_server", "sglang_router", "sglang::router"]:
        subprocess.run(["pkill", "-f", t], capture_output=True)
    time.sleep(8)
    for t in ["sglang.launch_server", "sglang_router", "sglang::router"]:
        subprocess.run(["pkill", "-9", "-f", t], capture_output=True)
    time.sleep(3)
    subprocess.run(["pkill", "-9", "-f", "ucx"], capture_output=True)
    subprocess.run(["sync"], capture_output=True)
    deadline = time.monotonic() + 60
    while time.monotonic() < deadline:
        busy = _ports_free()
        if not busy:
            break
        log.info("Waiting for ports: %s", busy)
        time.sleep(5)


def cleanup_procs(procs):
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
        {"text": f"Hello world {i}, warmup:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
        for i in range(8)
    ]
    def _send(p):
        try:
            requests.post(url, json=p, timeout=120)
        except Exception:
            pass
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(_send, payloads))
    time.sleep(3)


def run_benchmark(label, url):
    log.info("Benchmark: %s concurrency=%d qps=%d", label, CLIENT_CONCURRENCY, QPS)
    dump_file = os.path.join(LOG_DIR, f"result_{label}.json")
    cmd = [
        PYTHON, BENCHMARK, "--dataset", "sample",
        "--url", url,
        "--max-requests", str(MAX_REQUESTS),
        "--concurrency", str(CLIENT_CONCURRENCY),
        "--sample-input-len", str(INPUT_LEN),
        "--sample-output-len", str(OUTPUT_LEN),
        "--sample-qps", str(QPS),
        "--timeout", "900",
        "--dump", dump_file,
        "--scenario-label", label,
    ]
    subprocess.run(cmd, cwd=HERE)
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
    record = {
        "label": label,
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


def run_pd_af(m_stage, async_sched, interleave_poll, label):
    kill_all()
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["AFD_UCX_TLS"] = UCX_TLS
    env_base["UCX_LOG_LEVEL"] = "fatal"
    da_port, df_port = 50020, 50021
    pa_port, pf_port = 50010, 50011
    router_port = 50000
    ucx_p = 25100 + m_stage * 100
    ucx_d = 25200 + m_stage * 100
    sched_p = 65300 + m_stage * 100
    sched_d = 65400 + m_stage * 100

    extra = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", str(m_stage),
        "--max-running-requests", str(PD_AF_MAX_RUNNING),
    ]
    if async_sched:
        extra.append("--afd-async-schedule")
    if interleave_poll:
        extra.append("--afd-disagg-interleave-poll")

    def _start(name, gpu, perspective, disagg_mode, port, ucx_base, sched_port, ffn_host=None):
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

    # PF GPU 6 (FFN listener first)
    _start("pf", 6, "ffn", "prefill", pf_port, ucx_p, sched_p)
    time.sleep(2)
    # PA GPU 7
    _start("pa", 7, "attn", "prefill", pa_port, ucx_p, sched_p, ffn_host="127.0.0.1")

    if not wait_port("127.0.0.1", pa_port, timeout=300) or not wait_port("127.0.0.1", pf_port, timeout=300):
        log.error("%s: prefill servers failed", label)
        cleanup_procs(procs)
        return None

    # DF GPU 4 (FFN listener first)
    _start("df", 4, "ffn", "decode", df_port, ucx_d, sched_d)
    time.sleep(2)
    # DA GPU 5
    _start("da", 5, "attn", "decode", da_port, ucx_d, sched_d, ffn_host="127.0.0.1")

    if not wait_port("127.0.0.1", da_port) or not wait_port("127.0.0.1", df_port):
        log.error("%s: decode servers failed", label)
        cleanup_procs(procs)
        return None

    if not wait_health(f"http://127.0.0.1:{pa_port}/health", timeout=120):
        log.error("%s: PA health failed", label)
        cleanup_procs(procs)
        return None
    if not wait_health(f"http://127.0.0.1:{da_port}/health", timeout=120):
        log.error("%s: DA health failed", label)
        cleanup_procs(procs)
        return None
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
        cleanup_procs(procs)
        return None

    url = f"http://127.0.0.1:{router_port}"
    warmup(url)
    r = run_benchmark(label, url)
    cleanup_procs(procs)
    return r


def main():
    print("=" * 80)
    print("  PD+AF M=1 vs M=3 Comparison")
    print(f"  Model: {MODEL}")
    print(f"  Input={INPUT_LEN}, Output={OUTPUT_LEN}, N={MAX_REQUESTS}, Conc={CLIENT_CONCURRENCY}, QPS={QPS}")
    print("=" * 80)

    results = []

    # Config 1: PD+AF M=1
    log.info(">>> Running pdaf_m1 <<<")
    r1 = run_pd_af(m_stage=1, async_sched=False, interleave_poll=False, label="pdaf_m1")
    if r1:
        results.append(r1)
    time.sleep(5)

    # Config 2: PD+AF M=3 + interleave poll
    log.info(">>> Running pdaf_m3_opt <<<")
    r2 = run_pd_af(m_stage=3, async_sched=True, interleave_poll=True, label="pdaf_m3_opt")
    if r2:
        results.append(r2)

    # Summary
    print("\n" + "=" * 80)
    print("  RESULTS")
    print("=" * 80)
    header = f"  {'Config':<16} {'Out(tok/s)':>12} {'Total(tok/s)':>12} {'TTFT(ms)':>10} {'TPOT(ms)':>10} {'P95TTFT':>10} {'P95TPOT':>10} {'OK':>10}"
    print(header)
    print("  " + "-" * 78)
    for r in results:
        print(f"  {r['label']:<16} {r['output_thru_tok_s']:>12.1f} {r['total_thru_tok_s']:>12.1f} "
              f"{str(r['mean_ttft_ms']):>10} {str(r['mean_tpot_ms']):>10} "
              f"{str(r.get('p95_ttft_ms','')):>10} {str(r.get('p95_tpot_ms','')):>10} "
              f"{r['succeeded']:>10}")
    print("=" * 80)

    # Comparison
    if len(results) == 2:
        m1, m3 = results[0], results[1]
        print("\n--- M=3 vs M=1 Improvement ---")
        if m1['output_thru_tok_s'] > 0:
            thru_gain = (m3['output_thru_tok_s'] - m1['output_thru_tok_s']) / m1['output_thru_tok_s'] * 100
            print(f"  Throughput: {thru_gain:+.1f}%")
        if m1['mean_tpot_ms'] and m3['mean_tpot_ms']:
            tpot_gain = (m3['mean_tpot_ms'] - m1['mean_tpot_ms']) / m1['mean_tpot_ms'] * 100
            print(f"  TPOT:       {tpot_gain:+.1f}% (lower is better)")
        if m1['mean_ttft_ms'] and m3['mean_ttft_ms']:
            ttft_gain = (m3['mean_ttft_ms'] - m1['mean_ttft_ms']) / m1['mean_ttft_ms'] * 100
            print(f"  TTFT:       {ttft_gain:+.1f}% (lower is better)")

    with open(os.path.join(LOG_DIR, "comparison_results.json"), "w") as f:
        json.dump({"results": results}, f, indent=2)
    log.info("Results saved to %s/comparison_results.json", LOG_DIR)


if __name__ == "__main__":
    main()
