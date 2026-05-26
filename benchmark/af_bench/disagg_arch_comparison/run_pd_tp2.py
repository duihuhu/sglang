#!/usr/bin/env python3
"""PD TP=2 concurrency sweep (4 GPU: P=GPU6,7, D=GPU4,5)."""
import sys, os, time, subprocess, json, asyncio, aiohttp, logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
from run_pdaf_vs_pd import (kill_all, wait_port, wait_health, warmup,
                            cleanup_procs, measure_concurrent,
                            CONCURRENCIES, OUTPUT_TOKENS)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pd_tp2")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"
BOOTSTRAP = 18999

kill_all()
time.sleep(3)

procs = []
env_base = os.environ.copy()
env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

# Prefill: TP=2 on GPU 6,7
env_p = env_base.copy()
env_p["CUDA_VISIBLE_DEVICES"] = "6,7"
p_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "2",
    "--host", "127.0.0.1", "--port", "50010",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "prefill",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--disable-radix-cache",
]
fh = open("pd_tp2_p.log", "w")
p = subprocess.Popen(p_cmd, env=env_p, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
procs.append(("prefill", p, fh))

# Decode: TP=2 on GPU 4,5
env_d = env_base.copy()
env_d["CUDA_VISIBLE_DEVICES"] = "4,5"
d_cmd = [PYTHON, "-m", "sglang.launch_server",
    "--model-path", MODEL, "--tp", "2",
    "--host", "127.0.0.1", "--port", "50020",
    "--mem-fraction-static", "0.85",
    "--disaggregation-mode", "decode",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP),
    "--disaggregation-ib-device", "mlx5_4",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--disable-radix-cache",
]
fh2 = open("pd_tp2_d.log", "w")
p2 = subprocess.Popen(d_cmd, env=env_d, stdout=fh2, stderr=subprocess.STDOUT, start_new_session=True)
procs.append(("decode", p2, fh2))

log.info("Waiting for PD TP=2 servers...")
if not wait_port("127.0.0.1", 50010, timeout=300) or not wait_port("127.0.0.1", 50020, timeout=300):
    log.error("PD TP=2: servers failed to start")
    cleanup_procs(procs)
    sys.exit(1)

# Router
router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
    "--pd-disaggregation", "--mini-lb",
    "--prefill", "http://127.0.0.1:50010",
    "--decode", "http://127.0.0.1:50020",
    "--host", "127.0.0.1", "--port", "50000",
]
rf = open("pd_tp2_router.log", "w")
rp = subprocess.Popen(router_cmd, env=os.environ.copy(), stdout=rf, stderr=subprocess.STDOUT, start_new_session=True)
procs.append(("router", rp, rf))

if not wait_health("http://127.0.0.1:50000/health", timeout=180):
    log.error("PD TP=2: router failed")
    cleanup_procs(procs)
    sys.exit(1)

url = "http://127.0.0.1:50000"
log.info("PD TP=2 ready! Warming up...")
warmup(url)

log.info("Running concurrency sweep to 1024...")
data = {}
for conc in CONCURRENCIES:
    n_req = max(conc, 40)
    r = asyncio.run(measure_concurrent(url, 16, OUTPUT_TOKENS, conc, n_req))
    if r:
        data[conc] = r
        log.info(f'  conc={conc}: TPOT={r["mean_tpot_ms"]}ms TTFT={r["mean_ttft_ms"]}ms '
                 f'Thru={r["throughput_tok_s"]}tok/s running_max={r["max_running"]} '
                 f'running_avg={r["avg_running"]} (n={r["n_ok"]}/{n_req})')
    else:
        log.warning(f"  conc={conc}: FAILED")

cleanup_procs(procs)
kill_all()

print()
print("=" * 95)
print("  PD TP=2 (4 GPU: P=GPU6,7 D=GPU4,5) Concurrency Sweep (in=512, out=128)")
print("=" * 95)
print(f"{'Conc':<6} {'TPOT':<12} {'TTFT':<12} {'Throughput':<14} {'MaxRun':<8} {'AvgRun':<8}")
print("-" * 65)
for conc in CONCURRENCIES:
    r = data.get(conc)
    if r:
        print(f"{conc:<6} {r['mean_tpot_ms']:<10.1f}ms {r['mean_ttft_ms']:<10.1f}ms "
              f"{r['throughput_tok_s']:<12.1f}tok/s {r['max_running']:<8} {r['avg_running']:<8}")
    else:
        print(f"{conc:<6} {'FAILED':<12} {'FAILED':<12} {'FAILED':<14} {'N/A':<8} {'N/A':<8}")
print("=" * 95)

with open("pd_tp2_conc_1024.json", "w") as f:
    json.dump(data, f, indent=2, default=str)
log.info("Done")
