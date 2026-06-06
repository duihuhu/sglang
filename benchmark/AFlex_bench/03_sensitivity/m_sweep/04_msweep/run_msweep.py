#!/usr/bin/env python3
"""Sequential benchmark of M=1,2,3,4,5 for pipeline breakdown analysis.
Each variant uses 4 GPUs (0-3), runs 50 requests at QPS=2.
Logs saved with prefix like msweep_m1_, msweep_m2_, etc.
"""

import json, os, signal, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "throughput_logs")
PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
BENCHMARK = os.path.join(os.path.dirname(HERE), "benchmark_replay.py")

N_REQUESTS = 50
QPS = 2
OUTPUT_LEN = 64
INPUT_LEN = 512
CONCURRENCY = 128

EXTRA = [
    "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--max-running-requests", "128", "--mem-fraction-static", "0.88",
]

# Ports (same for all variants since we run sequentially)
PF_PORT, PA_PORT = 50011, 50010
DF_PORT, DA_PORT = 50021, 50020
ROUTER_PORT = 50001
BOOTSTRAP = 18999
P_SCHED, D_SCHED = 65300, 65400
UCX_A, UCX_F, UCX_DA, UCX_DF = 25100, 25100, 25200, 25200

ENV_BASE = {
    "SGLANG_DISABLE_REQUEST_LOGGING": "true",
    "AFD_UCX_TLS": "rc,tcp,cuda_copy,cuda_ipc",
    "UCX_LOG_LEVEL": "fatal",
}


def kill_all():
    for sig in [signal.SIGTERM, signal.SIGKILL]:
        for name in ["sglang.launch_server", "sglang_router"]:
            subprocess.run(["pkill", f"-{sig.value}", "-f", name], capture_output=True)
    time.sleep(3)


def wait_port(port, timeout=180):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            s = socket.create_connection(("127.0.0.1", port), timeout=2)
            s.close()
            return True
        except:
            time.sleep(2)
    return False


def wait_health(url, timeout=120):
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=5)
            return True
        except:
            time.sleep(3)
    return False


def launch_servers(m_value):
    """Launch PA, PF, DA, DF, Router for given M value. Returns list of (label, Popen, filehandle)."""
    procs = []
    prefix = f"msweep_m{m_value}"

    def start(label, gpu, env_extra, cmd):
        fh = open(os.path.join(LOG_DIR, f"{prefix}_{label.lower()}.log"), "w")
        env = os.environ.copy()
        env.update(ENV_BASE)
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env.update(env_extra)
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((label, p, fh))
        print(f"  [{label}] GPU={gpu} started (pid={p.pid})")

    cmd_base = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1",
        "--afd-comm-backend", "ucx",
        "--afd-micro-batch", str(m_value),
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + EXTRA

    # PF (GPU 2)
    pf_cmd = cmd_base + [
        "--port", str(PF_PORT), "--afd-perspective", "ffn",
        "--disaggregation-mode", "prefill",
    ]
    start("PF", 2, {"AFD_UCX_BASE_PORT": str(UCX_F), "AFD_SCHED_PORT": str(P_SCHED)}, pf_cmd)

    # PA (GPU 3)
    pa_cmd = cmd_base + [
        "--port", str(PA_PORT), "--afd-perspective", "attn", "--enable-metrics",
        "--disaggregation-mode", "prefill",
    ]
    start("PA", 3, {"AFD_UCX_BASE_PORT": str(UCX_A), "AFD_SCHED_PORT": str(P_SCHED),
                    "AFD_UCX_FFN_HOST": "127.0.0.1"}, pa_cmd)

    if not wait_port(PA_PORT) or not wait_port(PF_PORT):
        print("  ERROR: PA/PF failed to start!")
        return None

    # DF (GPU 0)
    df_cmd = cmd_base + [
        "--port", str(DF_PORT), "--afd-perspective", "ffn",
        "--disaggregation-mode", "decode",
    ]
    start("DF", 0, {"AFD_UCX_BASE_PORT": str(UCX_DF), "AFD_SCHED_PORT": str(D_SCHED)}, df_cmd)

    # DA (GPU 1)
    da_cmd = cmd_base + [
        "--port", str(DA_PORT), "--afd-perspective", "attn", "--enable-metrics",
        "--disaggregation-mode", "decode",
    ]
    start("DA", 1, {"AFD_UCX_BASE_PORT": str(UCX_DA), "AFD_SCHED_PORT": str(D_SCHED),
                    "AFD_UCX_FFN_HOST": "127.0.0.1"}, da_cmd)

    if not wait_port(DA_PORT) or not wait_port(DF_PORT):
        print("  ERROR: DA/DF failed to start!")
        return None

    # Router
    router_cmd = [
        PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", f"http://127.0.0.1:{PA_PORT}",
        "--decode", f"http://127.0.0.1:{DA_PORT}",
        "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
    ]
    router_fh = open(os.path.join(LOG_DIR, f"{prefix}_router.log"), "w")
    router = subprocess.Popen(router_cmd, stdout=router_fh, stderr=subprocess.STDOUT,
                               start_new_session=True)
    procs.append(("router", router, router_fh))

    if not wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health"):
        print("  ERROR: Router health check failed!")
        return None

    print(f"  All servers ready for M={m_value}")
    return procs


def run_benchmark(m_value):
    """Run benchmark against router, return results dict."""
    import concurrent.futures, requests

    prefix = f"msweep_m{m_value}"

    # Warmup
    warm = [{"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
            for i in range(16)]
    def send(p):
        requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=p, timeout=120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(send, warm))
    time.sleep(5)

    print(f"  Running M={m_value} benchmark ({N_REQUESTS} reqs, QPS={QPS})...")
    dump_file = os.path.join(LOG_DIR, f"{prefix}_results.json")
    cmd = [
        PYTHON, BENCHMARK,
        "--url", f"http://127.0.0.1:{ROUTER_PORT}",
        "--dataset", "sample",
        "--max-requests", str(N_REQUESTS),
        "--concurrency", str(CONCURRENCY),
        "--sample-input-len", str(INPUT_LEN),
        "--sample-output-len", str(OUTPUT_LEN),
        "--sample-qps", str(QPS),
        "--timeout", "600",
        "--dump", dump_file,
        "--scenario-label", f"pdaf_m{m_value}_qps{QPS}",
    ]
    subprocess.run(cmd, cwd=os.path.dirname(BENCHMARK))

    result = {}
    if os.path.exists(dump_file):
        with open(dump_file) as f:
            data = json.load(f)
        results = data.get("results", [])
        ok = [r for r in results if r.get("success")]
        total_in = sum(r.get("input_tokens", 0) for r in ok)
        total_out = sum(r.get("output_tokens", 0) for r in ok)
        wall = data.get("wall_duration_s", 1)
        tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
        ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
        result = {
            "M": m_value,
            "out_tok_s": round(total_out / wall, 1) if wall > 0 else 0,
            "tpot_ms": round(sum(tpot) / len(tpot), 1) if tpot else 0,
            "ttft_ms": round(sum(ttft) / len(ttft), 1) if ttft else 0,
            "ok": f"{len(ok)}/{len(results)}",
        }
        print(f"  M={m_value}: out={result['out_tok_s']} tok/s, TPOT={result['tpot_ms']}ms, TTFT={result['ttft_ms']}ms")
    return result


def cleanup(procs):
    for label, p, fh in procs:
        try:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except:
            p.terminate()
    time.sleep(3)
    for label, p, fh in procs:
        try: p.kill()
        except: pass
        try: fh.close()
        except: pass


def main():
    kill_all()
    all_results = {}

    for m in [1, 2, 3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"  M = {m}")
        print(f"{'='*60}")

        procs = launch_servers(m)
        if procs is None:
            print(f"  FAILED to start M={m}, skipping")
            continue

        result = run_benchmark(m)
        all_results[f"M={m}"] = result
        cleanup(procs)
        time.sleep(5)  # Let ports free

    # Print summary
    print("\n" + "=" * 70)
    print("  M-Sweep Summary")
    print("=" * 70)
    print(f"  {'M':<6} {'Out(tok/s)':<14} {'TPOT(ms)':<10} {'TTFT(ms)':<10} {'OK':<8}")
    print("  " + "-" * 48)
    for m in [1, 2, 3, 4, 5]:
        r = all_results.get(f"M={m}", {})
        if r:
            print(f"  {m:<6} {r['out_tok_s']:<14.1f} {r['tpot_ms']:<10.1f} {r['ttft_ms']:<10.1f} {r['ok']:<8}")

    with open(os.path.join(LOG_DIR, "msweep_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to msweep_all_results.json")
    print(f"Logs in {LOG_DIR}/msweep_m*_*.log")


if __name__ == "__main__":
    main()
