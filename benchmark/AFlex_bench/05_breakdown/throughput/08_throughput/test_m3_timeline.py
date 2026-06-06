#!/usr/bin/env python3
"""Single-request M=3 pipeline timing test for PD+AF with detailed per-stage data."""
import subprocess, time, json, os, sys, signal, logging, uuid, socket, shutil

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("m3_timeline")

PYTHON = sys.executable
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "raw_logs")
procs = []

def _wait_port(host, port, timeout=180):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            s = socket.create_connection((host, port), timeout=1)
            s.close(); return True
        except: time.sleep(1)
    return False

def _wait_health(url, timeout=180):
    import urllib.request
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=3)
            return True
        except: time.sleep(2)
    return False

def _start_server(cmd, env, log_path):
    fh = open(log_path, "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append((p, fh))
    return p

shutil.rmtree(LOG_DIR, ignore_errors=True)
os.makedirs(LOG_DIR, exist_ok=True)

try:
    PA_PORT, PF_PORT = 50010, 50011
    DA_PORT, DF_PORT = 50020, 50021
    ROUTER_PORT = 50000
    UCX_PA, UCX_PF = 25100, 25100
    UCX_DA, UCX_DF = 25200, 25200
    SCHED_P = 65300
    SCHED_D = 65400
    BOOTSTRAP = 18999

    extra_decode = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", "3", "--max-running-requests", "16",
        "--afd-enable-overlap-schedule",
    ]
    extra_prefill = [
        "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-micro-batch", "3", "--max-running-requests", "16",
        "--afd-enable-overlap-schedule",
    ]

    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["CUDA_LAUNCH_BLOCKING"] = "1"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["AFD_DETAILED_TIMING"] = "1"

    # PF GPU 2
    pf_env = env_base.copy()
    pf_env["CUDA_VISIBLE_DEVICES"] = "2"
    pf_env["AFD_UCX_BASE_PORT"] = str(UCX_PF)
    pf_env["AFD_SCHED_PORT"] = str(SCHED_P)
    pf_cmd = [PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL, "--tp", "1",
        "--host", "127.0.0.1", "--port", str(PF_PORT),
        "--afd-perspective", "ffn", "--afd-comm-backend", "ucx",
        "--mem-fraction-static", "0.7",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra_prefill
    log.info("Starting PF (Prefill FFN) on GPU 2 port %d", PF_PORT)
    _start_server(pf_cmd, pf_env, os.path.join(LOG_DIR, "pf.log"))

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
        "--mem-fraction-static", "0.7",
        "--disaggregation-mode", "prefill",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra_prefill
    log.info("Starting PA (Prefill Attn) on GPU 3 port %d", PA_PORT)
    _start_server(pa_cmd, pa_env, os.path.join(LOG_DIR, "pa.log"))

    if not _wait_port("127.0.0.1", PA_PORT) or not _wait_port("127.0.0.1", PF_PORT):
        log.error("PA/PF failed to start"); sys.exit(1)
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
        "--mem-fraction-static", "0.7",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra_decode
    log.info("Starting DF (Decode FFN) on GPU 0 port %d", DF_PORT)
    _start_server(df_cmd, df_env, os.path.join(LOG_DIR, "df.log"))

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
        "--mem-fraction-static", "0.7",
        "--disaggregation-mode", "decode",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-bootstrap-port", str(BOOTSTRAP),
        "--disaggregation-ib-device", "mlx5_4",
    ] + extra_decode
    log.info("Starting DA (Decode Attn) on GPU 1 port %d", DA_PORT)
    _start_server(da_cmd, da_env, os.path.join(LOG_DIR, "da.log"))

    if not _wait_port("127.0.0.1", DA_PORT) or not _wait_port("127.0.0.1", DF_PORT):
        log.error("DA/DF failed to start"); sys.exit(1)
    log.info("DA and DF ready")

    # Router
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
        "--pd-disaggregation", "--mini-lb",
        "--prefill", f"http://127.0.0.1:{PA_PORT}",
        "--decode", f"http://127.0.0.1:{DA_PORT}",
        "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
    ]
    log.info("Starting router on port %d", ROUTER_PORT)
    router_fh = open(os.path.join(LOG_DIR, "router.log"), "w")
    router_proc = subprocess.Popen(router_cmd, env=os.environ.copy(),
        stdout=router_fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append((router_proc, router_fh))

    if not _wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=120):
        log.error("Router health check failed"); sys.exit(1)
    log.info("Router ready")

    import requests as http_req

    # Warmup
    log.info("--- Warmup with concurrent requests to populate decode batch ---")
    import concurrent.futures
    warm_payloads = [
        {"text": f"Hi {uuid.uuid4().hex[:8]}:", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}}
        for _ in range(6)
    ]
    def _send(payload):
        return http_req.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=payload, timeout=300).json()

    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        list(ex.map(_send, warm_payloads))
    time.sleep(3)
    log.info("Warmup done — decode batch should now have multiple sequences for M=3")

    # Send concurrent requests to trigger M=3 decode pipeline
    log.info("--- Concurrent request pipeline capture (M=3) ---")
    test_payloads = [
        {"text": f"Explain quantum computing. Req {i} {uuid.uuid4().hex[:8]}:",
         "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}}
        for i in range(6)
    ]
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        results = list(ex.map(_send, test_payloads))
    t1 = time.time()
    wall = t1 - t0
    total_tokens = sum(r.get("meta_info", {}).get("completion_tokens", 0) for r in results)
    log.info(f"Concurrent test complete: {len(results)} requests, total_tokens={total_tokens}, wall={wall:.1f}s")

finally:
    log.info("Cleaning up...")
    for p, fh in procs:
        try: os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        except: p.terminate()
    time.sleep(3)
    for p, fh in procs:
        try: p.kill()
        except: pass
        try: fh.close()
        except: pass
    log.info("Done")
