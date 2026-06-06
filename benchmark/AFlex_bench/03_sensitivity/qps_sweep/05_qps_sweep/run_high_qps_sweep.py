#!/usr/bin/env python3
"""Sweep M=1..5 at high QPS (inf) to test pipeline efficiency under full load.
All requests sent at once — system runs at max capacity.
"""

import json, os, signal, socket, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "throughput_logs")
PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
BENCHMARK = os.path.join(os.path.dirname(HERE), "benchmark_replay.py")

N_REQUESTS = 200
# "inf" QPS — send all requests as fast as possible
QPS = 9999
OUTPUT_LEN = 64
INPUT_LEN = 512
CONCURRENCY = 256

EXTRA = [
    "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--max-running-requests", "256", "--mem-fraction-static", "0.88",
]

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
        for name in ["sglang.launch_server", "sglang_router", "sglang::router", "sglang::scheduler"]:
            subprocess.run(["pkill", f"-{sig.value}", "-f", name], capture_output=True)
    time.sleep(10)


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


def launch_servers(m_value, prefix):
    procs = []

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
        return None, procs

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
        return None, procs

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
        return None, procs

    print(f"  All servers ready for M={m_value}")
    return procs, []


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


def run_benchmark(m_value, prefix):
    import concurrent.futures, requests

    # Warmup
    warm = [{"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}
            for i in range(16)]
    def send(p):
        requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=p, timeout=120)
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as ex:
        list(ex.map(send, warm))
    time.sleep(5)

    print(f"  Running M={m_value} HIGH-QPS benchmark ({N_REQUESTS} reqs, QPS=inf)...")
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
        "--scenario-label", f"pdaf_hq_m{m_value}",
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
            "total_tok_s": round((total_in + total_out) / wall, 1) if wall > 0 else 0,
            "tpot_ms": round(sum(tpot) / len(tpot), 1) if tpot else 0,
            "ttft_ms": round(sum(ttft) / len(ttft), 1) if ttft else 0,
            "wall_s": round(wall, 1),
            "ok": f"{len(ok)}/{len(results)}",
            "total_out": total_out,
            "total_in": total_in,
        }
        print(f"  M={m_value}: out={result['out_tok_s']} tok/s, total={result['total_tok_s']} tok/s, "
              f"TPOT={result['tpot_ms']}ms, TTFT={result['ttft_ms']}ms, wall={result['wall_s']}s")
    return result


def main():
    kill_all()
    all_results = {}

    for m in [1, 2, 3, 4, 5]:
        print(f"\n{'='*60}")
        print(f"  HIGH-QPS M = {m}")
        print(f"{'='*60}")

        prefix = f"hqsweep_m{m}"
        procs, partial = launch_servers(m, prefix)
        if procs is None:
            print(f"  FAILED to start M={m}, cleaning up")
            cleanup(partial)
            kill_all()
            time.sleep(5)
            continue

        result = run_benchmark(m, prefix)
        all_results[f"M={m}"] = result
        cleanup(procs)
        kill_all()
        time.sleep(5)

    # Summary
    print("\n" + "=" * 80)
    print("  HIGH-QPS M-Sweep Summary (QPS=inf, 200 requests, il=512, ol=64)")
    print("=" * 80)
    print(f"  {'M':<6} {'Out(tok/s)':<14} {'Total(tok/s)':<14} {'TPOT(ms)':<10} {'TTFT(ms)':<10} {'Wall(s)':<10} {'OK':<8}")
    print("  " + "-" * 72)
    for m in [1, 2, 3, 4, 5]:
        r = all_results.get(f"M={m}", {})
        if r:
            print(f"  {m:<6} {r['out_tok_s']:<14.1f} {r['total_tok_s']:<14.1f} "
                  f"{r['tpot_ms']:<10.1f} {r['ttft_ms']:<10.1f} {r['wall_s']:<10.1f} {r['ok']:<8}")

    # Find best M for throughput
    best = max(all_results.items(), key=lambda x: x[1].get("total_tok_s", 0))
    print(f"\n  Best throughput: M={best[0]} with {best[1]['total_tok_s']} total tok/s")

    with open(os.path.join(LOG_DIR, "hqsweep_all_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to hqsweep_all_results.json")


if __name__ == "__main__":
    main()
