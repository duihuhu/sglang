#!/usr/bin/env python3
"""Run PD+AF M=1 benchmark with IPC backend to capture IPC_INNER breakdown data."""
import json, logging, os, socket, subprocess, sys, time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ipc_brk")

PYTHON = "/workspace/env/af-test/bin/python3"
MODEL = "/models/Qwen/Qwen3-32B/"
_HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(_HERE, "ipc_breakdown_logs")
os.makedirs(LOG_DIR, exist_ok=True)

MAX_REQUESTS = 8
CONCURRENCY = 4
OUTPUT_LEN = 64
INPUT_LEN = 128
ROUTER_PORT = 50000

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

kill_all()

PA_PORT, PF_PORT = 50010, 50011
DA_PORT, DF_PORT = 50020, 50021
SCHED_P = 65300
SCHED_D = 65400
BOOTSTRAP = 18999

env_base = os.environ.copy()
env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

extra_decode = [
    "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--afd-micro-batch", "1", "--max-running-requests", "16",
]
extra_prefill = [
    "--skip-server-warmup", "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--afd-micro-batch", "1", "--max-running-requests", "16",
]

procs = []

def start_server(name, cmd, env, log_name):
    fh = open(os.path.join(LOG_DIR, log_name), "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    procs.append((name, p, fh))
    return p

# PF GPU 2 (IPC backend)
pf_env = env_base.copy()
pf_env["CUDA_VISIBLE_DEVICES"] = "2"
pf_env["AFD_SCHED_PORT"] = str(SCHED_P)
pf_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "1",
    "--host", "127.0.0.1", "--port", str(PF_PORT),
    "--afd-perspective", "ffn", "--afd-comm-backend", "ipc",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "prefill",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
] + extra_prefill
log.info("Starting PF on GPU 2 (IPC)")
start_server("PF", pf_cmd, pf_env, "pf.log")

# PA GPU 3 (IPC backend)
pa_env = env_base.copy()
pa_env["CUDA_VISIBLE_DEVICES"] = "3"
pa_env["AFD_SCHED_PORT"] = str(SCHED_P)
pa_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "1",
    "--host", "127.0.0.1", "--port", str(PA_PORT),
    "--enable-metrics",
    "--afd-perspective", "attn", "--afd-comm-backend", "ipc",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "prefill",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
] + extra_prefill
log.info("Starting PA on GPU 3 (IPC)")
start_server("PA", pa_cmd, pa_env, "pa.log")

if not wait_port("127.0.0.1", PA_PORT, timeout=300) or not wait_port("127.0.0.1", PF_PORT, timeout=300):
    log.error("PA/PF failed to start")
    kill_all()
    sys.exit(1)
log.info("PA and PF ready")

# DF GPU 0 (IPC backend)
df_env = env_base.copy()
df_env["CUDA_VISIBLE_DEVICES"] = "0"
df_env["AFD_SCHED_PORT"] = str(SCHED_D)
df_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "1",
    "--host", "127.0.0.1", "--port", str(DF_PORT),
    "--afd-perspective", "ffn", "--afd-comm-backend", "ipc",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "decode",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
] + extra_decode
log.info("Starting DF on GPU 0 (IPC)")
start_server("DF", df_cmd, df_env, "df.log")

# DA GPU 1 (IPC backend)
da_env = env_base.copy()
da_env["CUDA_VISIBLE_DEVICES"] = "1"
da_env["AFD_SCHED_PORT"] = str(SCHED_D)
da_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "1",
    "--host", "127.0.0.1", "--port", str(DA_PORT),
    "--enable-metrics",
    "--afd-perspective", "attn", "--afd-comm-backend", "ipc",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "decode",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
] + extra_decode
log.info("Starting DA on GPU 1 (IPC)")
start_server("DA", da_cmd, da_env, "da.log")

if not wait_port("127.0.0.1", DA_PORT, timeout=300) or not wait_port("127.0.0.1", DF_PORT, timeout=300):
    log.error("DA/DF failed to start")
    kill_all()
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
router_fh = open(os.path.join(LOG_DIR, "router.log"), "w")
router_proc = subprocess.Popen(router_cmd, env=os.environ.copy(),
                               stdout=router_fh, stderr=subprocess.STDOUT,
                               start_new_session=True)
procs.append(("router", router_proc, router_fh))

if not wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=120):
    log.error("Router health check failed")
    kill_all()
    sys.exit(1)
log.info("All servers ready")

# Warmup
import requests, concurrent.futures
warm_payloads = [
    {"text": f"Hi {i}:", "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}}
    for i in range(4)
]
log.info("Warmup...")
def _send(payload):
    requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=payload, timeout=120)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    list(ex.map(_send, warm_payloads))
time.sleep(3)
log.info("Warmup done")

# Benchmark
payloads = [
    {"text": f"The meaning of life is {i} " * 16, "sampling_params": {"max_new_tokens": OUTPUT_LEN, "temperature": 0.0}}
    for i in range(MAX_REQUESTS)
]
log.info("Sending %d requests...", MAX_REQUESTS)
def _bench(payload):
    r = requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate", json=payload, timeout=300)
    return r.status_code
with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
    results = list(ex.map(_bench, payloads))
ok = sum(1 for r in results if r == 200)
log.info("Done: %d/%d succeeded", ok, MAX_REQUESTS)
time.sleep(2)

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

log.info("Logs saved to %s", LOG_DIR)
