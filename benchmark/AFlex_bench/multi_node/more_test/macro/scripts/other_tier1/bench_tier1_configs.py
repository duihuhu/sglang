#!/usr/bin/env python3
"""Benchmark two Tier1-optimized AFD configs on QPS=16 code dataset, node3+node4.

Config A (14 GPU, solver Rank 1): k_P=4, tp=(2,1)@(930,1170) + k_D=1, tp=(1,1)@930
Config B (10 GPU, solver sweet-spot): k_P=4, tp=(1,1)@930 + k_D=1, tp=(1,1)@930

Deploys on node3 (prefill/decode) + node4 (remaining prefill pairs).
Captures per-GPU frequency timeline via nvidia-smi polling.
"""
from __future__ import annotations

import argparse, asyncio, json, logging, os, shlex, subprocess, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench_tier1")

HERE = Path(__file__).resolve().parent
HERE.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Node config (node3 + node4) ──
NODE1_IP = os.environ.get("MN_NODE3_IP", "10.252.129.34")  # node3 = prefill primary
NODE2_IP = os.environ.get("MN_NODE4_IP", "10.252.129.33")  # node4
CONTAINER = "operator_test"
PYTHON = "python3"
MODEL = "/models/Qwen3-32B"
LOG_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/multi_node/logs")
IB_JSON_FILE = "/tmp/ib_scal_map.json"
GPUS_PER_NODE = 8

# SLO
TTFT_SLO_MS = 2000.0
TPOT_SLO_MS = 100.0

# GPU/NIC affinity
GPU_NIC = {0: "mlx5_0", 1: "mlx5_0", 2: "mlx5_1", 3: "mlx5_1",
           4: "mlx5_4", 5: "mlx5_4", 6: "mlx5_5", 7: "mlx5_5"}

CLEANUP_SCRIPT = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"

# AFD ports
BS_PORT = 49999
ROUTER_PORT = 44000
SUB_ROUTER_BASE = 45000
DECODE_PORT_BASE = 43020
PREFILL_PORT_BASE = 43200

MAX_GPU_FREQ = 1410
ENERGY_MODEL_DIR_V1 = "/workspace/sglang/benchmark/AFlex_bench/energy_model/Qwen3-32B/models_v1"
FREQ_TIMELINE_DIR = HERE / "freq_timelines"
FREQ_TIMELINE_DIR.mkdir(parents=True, exist_ok=True)

# ── Tier1 configs to test ──
@dataclass
class Tier1TestConfig:
    name: str
    k_p: int; k_d: int
    tp_pa: int; tp_pf: int
    tp_da: int; tp_df: int
    f_pa: int; f_pf: int
    f_da: int; f_df: int
    tier: bool = True

CONFIGS = [
    Tier1TestConfig("tier1_14g", k_p=4, k_d=1, tp_pa=2, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=1170, f_da=930, f_df=930, tier=True),
    Tier1TestConfig("tier1_10g", k_p=4, k_d=1, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930, tier=True),
]

QPS = 16
DATASET = "code"
MAX_RUN_S = 400
WORKLOAD_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/multi_node/more_test/macro/data/workloads")


# ── SSH helpers ──
def _ssh(host, cmd):
    return ["ssh", "-o", "StrictHostKeyChecking=no", f"root@{host}", cmd]

def _dexec(host, cmd):
    """docker exec on a remote node."""
    inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(cmd)}"
    return subprocess.run(_ssh(host, inner), capture_output=True, text=True)

def _is_local(host):
    import socket
    try:
        return socket.gethostbyname(socket.gethostname()) == socket.gethostbyname(host)
    except Exception:
        return False

def write_ib_json():
    ib_map = {str(g): GPU_NIC[g] for g in range(GPUS_PER_NODE)}
    Path(IB_JSON_FILE).write_text(json.dumps(ib_map))

def cleanup_all():
    log.info("Cleaning up both nodes...")
    subprocess.run(_ssh(NODE1_IP, f"bash {CLEANUP_SCRIPT}"), capture_output=True, check=False)
    subprocess.run(_ssh(NODE2_IP, f"bash {CLEANUP_SCRIPT}"), capture_output=True, check=False)
    time.sleep(3)

def wait_health(host, port, timeout=300):
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _dexec(host, f"curl -s http://localhost:{port}/health")
            if "ok" in (r.stdout or "").lower() or r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False

def test_generate(url):
    import requests
    for attempt in range(3):
        try:
            r = requests.post(url + "/generate", json={
                "text": "Hello, explain quantum computing:",
                "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}, timeout=120)
            if r.status_code == 200 and "text" in r.json():
                return True
        except Exception as e:
            log.warning("warmup attempt %d: %s", attempt + 1, e)
        time.sleep(10)
    return False

def lock_freq_map(host, gpu_freq):
    if not gpu_freq: return
    lines = []
    for g, f in sorted(gpu_freq.items()):
        lines.append(f"from sglang.srt.layers.dvfs import lock_gpus; lock_gpus([{g}], {f})")
    py = "; ".join(lines)
    _dexec(host, f'{PYTHON} -c "{py}"')
    log.info("  Locked %s: %s", host, {g: f"{f}MHz" for g, f in sorted(gpu_freq.items())})

def unlock_freq(host):
    py = "from sglang.srt.layers.dvfs import unlock_gpus; unlock_gpus(list(range(8)))"
    _dexec(host, f'{PYTHON} -c "{py}"')


# ── GPU allocation ──
def alloc_af_gpus(tp_a, tp_f, base_gpus):
    """Interleaved: attn=even slots, ffn=odd slots."""
    attn, ffn = [], []
    idx = 0
    while len(attn) < tp_a or len(ffn) < tp_f:
        if len(attn) < tp_a:
            attn.append(base_gpus[idx]); idx += 1
        if len(ffn) < tp_f and idx < len(base_gpus):
            ffn.append(base_gpus[idx]); idx += 1
    return attn, ffn


def plan_allocation(cfg):
    """Plan GPU placement across node3+node4."""
    n1_free = list(range(GPUS_PER_NODE))
    n2_free = list(range(GPUS_PER_NODE))

    # Decode on node3
    d_a, d_f = alloc_af_gpus(cfg.tp_da, cfg.tp_df, n1_free)
    n1_free = [g for g in n1_free if g not in set(d_a) | set(d_f)]

    # Prefill pairs: fill node3 first, then node4
    p_pairs = []
    for _ in range(cfg.k_p):
        for label, free in [("n3", n1_free), ("n4", n2_free)]:
            need = cfg.tp_pa + cfg.tp_pf
            if need > len(free): continue
            a, f = alloc_af_gpus(cfg.tp_pa, cfg.tp_pf, free[:need])
            p_pairs.append((label, a, f))
            if label == "n3":
                n1_free = [g for g in n1_free if g not in set(a) | set(f)]
            else:
                n2_free = [g for g in n2_free if g not in set(a) | set(f)]
            break

    # Build freq map
    freq_map = {NODE1_IP: {}, NODE2_IP: {}}
    for label, a, f in p_pairs:
        host = NODE1_IP if label == "n3" else NODE2_IP
        for g in a: freq_map[host][g] = cfg.f_pa
        for g in f: freq_map[host][g] = cfg.f_pf
    for g in d_a: freq_map[NODE1_IP][g] = cfg.f_da
    for g in d_f: freq_map[NODE1_IP][g] = cfg.f_df

    total_gpu = cfg.k_p * (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)
    return {
        "p_pairs": [(NODE1_IP if l == "n3" else NODE2_IP, a, f) for l, a, f in p_pairs],
        "decode": {"host": NODE1_IP, "attn": d_a, "ffn": d_f},
        "freq_map": freq_map,
        "total_gpu": total_gpu,
        "n3_used": GPUS_PER_NODE - len(n1_free),
        "n4_used": GPUS_PER_NODE - len(n2_free),
    }


# ── AFD launch ──
def _afd_env(persp, attn_gpus, ffn_gpus, ucx_port, sched_port):
    cvd = "0,1,2,3,4,5,6,7"
    base = (
        f"CUDA_VISIBLE_DEVICES={cvd} "
        "SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
        "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
        "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
        "AFD_IPC_SYNC_MODE=ipc_event "
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
    )
    if persp == "ffn":
        nvml = ",".join(str(g) for g in ffn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
                f"AFD_IPC_PEER_OFFSET=-1 AFD_NVML_DEVICE_INDICES={nvml} "
                f"AFD_NVML_DEVICE_INDEX={ffn_gpus[0]}")
    nvml = ",".join(str(g) for g in attn_gpus)
    return (base + f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
            f"AFD_IPC_PEER_OFFSET=1 AFD_NVML_DEVICE_INDICES={nvml} "
            f"AFD_NVML_DEVICE_INDEX={attn_gpus[0]} AFD_UCX_FFN_HOST=127.0.0.1")


def _afd_flags(tp, tier):
    f = (
        f"--model-path {MODEL} --tp {tp} --gpu-id-step 2 "
        "--afd-comm-backend ipc_cpp --afd-micro-batch 1 "
        "--mem-fraction-static 0.85 --max-running-requests 512 "
        "--skip-server-warmup --watchdog-timeout 600 "
        "--disable-cuda-graph --disable-piecewise-cuda-graph "
        "--afd-disagg-interleave-poll --disable-radix-cache "
        "--num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-bootstrap-port {BS_PORT} "
        f"--disaggregation-ib-device {IB_JSON_FILE} --enable-metrics"
    )
    if tier:
        f += (" --afd-dvfs-enabled "
              f"--afd-energy-model-dir {ENERGY_MODEL_DIR_V1} "
              f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)} "
              f"--afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)} "
              "--afd-dvfs-decode-compositional --afd-dvfs-idle-lock")
    return f


def _launch(host, persp, mode, attn, ffn, tp, port, base_gpu, flags, tag, ucx, sched):
    """Launch via setsid so process survives docker exec exit (same as run_macro_benchmark)."""
    env = _afd_env(persp, attn, ffn, ucx, sched)
    prefix = "setsid prlimit --memlock=unlimited:unlimited "
    cmd = (f"{env} {prefix}{PYTHON} -m sglang.launch_server --host {host} "
           f"--port {port} --afd-perspective {persp} "
           f"--disaggregation-mode {mode} --base-gpu-id {base_gpu} {flags}")
    full_cmd = f"{cmd} > /tmp/{tag}.log 2>&1 < /dev/null &"
    # Use same pattern as run_macro_benchmark: dexec_local / dexec_remote
    if host == NODE1_IP:
        inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(full_cmd)}"
        if _is_local(host):
            subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", full_cmd], check=False)
        else:
            subprocess.run(_ssh(host, inner), check=False)
    else:
        inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(full_cmd)}"
        subprocess.run(_ssh(host, inner), check=False)
    log.info("  Launched %s on %s:%d", tag, host, port)


def deploy(cfg):
    alloc = plan_allocation(cfg)
    p_pairs = alloc["p_pairs"]
    d_info = alloc["decode"]

    log.info("=" * 60)
    log.info("DEPLOY %s: %d GPU (n3:%d/8 n4:%d/8)", cfg.name, alloc["total_gpu"],
             alloc["n3_used"], alloc["n4_used"])
    for i, (host, a, f) in enumerate(p_pairs):
        log.info("  P%d: %s attn=%s ffn=%s", i, host.split(".")[-1], a, f)
    log.info("  D:  %s attn=%s ffn=%s", d_info["host"].split(".")[-1], d_info["attn"], d_info["ffn"])

    write_ib_json()

    # Lock frequencies
    for host, fmap in alloc["freq_map"].items():
        if fmap and cfg.tier:
            lock_freq_map(host, fmap)

    # Per-perspective flags (heterogeneous TP support)
    p_cf_a = _afd_flags(cfg.tp_pa, cfg.tier)  # PA uses tp_pa
    p_cf_f = _afd_flags(cfg.tp_pf, cfg.tier)  # PF uses tp_pf
    d_cf_a = _afd_flags(cfg.tp_da, cfg.tier)  # DA uses tp_da
    d_cf_f = _afd_flags(cfg.tp_df, cfg.tier)  # DF uses tp_df

    # Launch prefill pairs
    prefill_eps = []
    for i, (host, attn_gpus, ffn_gpus) in enumerate(p_pairs):
        pa_port = PREFILL_PORT_BASE + i * 10
        pf_port = PREFILL_PORT_BASE + i * 10 + 1
        ucx = 28200 + i * 100
        sched = 68400 + i * 100
        time.sleep(2)
        _launch(host, "ffn", "prefill", attn_gpus, ffn_gpus, cfg.tp_pf, pf_port,
                ffn_gpus[0], p_cf_f, f"{cfg.name}_p{i}f", ucx, sched)
        time.sleep(5)
        _launch(host, "attn", "prefill", attn_gpus, ffn_gpus, cfg.tp_pa, pa_port,
                attn_gpus[0], p_cf_a, f"{cfg.name}_p{i}a", ucx, sched)
        prefill_eps.append((host, pa_port))

    # Launch decode
    da_port = DECODE_PORT_BASE
    df_port = DECODE_PORT_BASE + 1
    time.sleep(5)
    _launch(d_info["host"], "ffn", "decode", d_info["attn"], d_info["ffn"],
            cfg.tp_df, df_port, d_info["ffn"][0], d_cf_f, f"{cfg.name}_df", 28300, 68500)
    time.sleep(5)
    _launch(d_info["host"], "attn", "decode", d_info["attn"], d_info["ffn"],
            cfg.tp_da, da_port, d_info["attn"][0], d_cf_a, f"{cfg.name}_da", 28300, 68500)

    # Health checks
    for i, (host, _) in enumerate(prefill_eps):
        if not wait_health(host, PREFILL_PORT_BASE + i * 10 + 1, 300):
            log.error("PF%d health failed", i); return None
        if not wait_health(host, PREFILL_PORT_BASE + i * 10, 300):
            log.error("PA%d health failed", i); return None
    if not wait_health(d_info["host"], da_port, 300):
        log.error("DA health failed"); return None

    # Sub-routers (one per prefill pair → decode)
    sub_ports = []
    for i, (p_host, pa_port) in enumerate(prefill_eps):
        sp = SUB_ROUTER_BASE + i
        rc = (f"nohup {PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
              f"--prefill http://{p_host}:{pa_port} --decode http://{d_info['host']}:{da_port} "
              f"--host {NODE1_IP} --port {sp} "
              f"> {LOG_DIR}/{cfg.name}_sub{i}.log 2>&1 < /dev/null &")
        _dexec(NODE1_IP, rc)
        if not wait_health(NODE1_IP, sp, 60):
            log.error("sub-router %d failed", i); return None
        sub_ports.append(sp)

    # Top-level round-robin
    workers = " ".join(f"http://{NODE1_IP}:{p}" for p in sub_ports)
    rc = (f"nohup {PYTHON} -m sglang_router.launch_router "
          f"--host {NODE1_IP} --port {ROUTER_PORT} --policy round_robin "
          f"--worker-urls {workers} > {LOG_DIR}/{cfg.name}_router.log 2>&1 < /dev/null &")
    _dexec(NODE1_IP, rc)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        return None

    return f"http://{NODE1_IP}:{ROUTER_PORT}"


# ── Frequency timeline ──
class FreqMonitor:
    def __init__(self, host, gpus, interval=0.15):
        self.host = host; self.gpus = gpus; self.interval = interval
        self.samples = []; self._stop = False
        import threading
        self._t = threading.Thread(target=self._poll, daemon=True)

    def _poll(self):
        import threading as th
        gpu_csv = ",".join(str(g) for g in self.gpus)
        while not self._stop:
            t0 = time.time()
            try:
                r = _dexec(self.host,
                    f"nvidia-smi --query-gpu=index,clocks.current.sm "
                    f"--format=csv,noheader,nounits -i {gpu_csv}")
                for line in (r.stdout or "").strip().splitlines():
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        self.samples.append({"t": round(t0, 3), "gpu": int(parts[0].strip()),
                                             "freq_mhz": int(parts[1].strip())})
            except Exception:
                pass
            th.Event().wait(max(0, self.interval - (time.time() - t0)))

    def start(self): self._t.start()
    def stop(self): self._stop = True; self._t.join(timeout=5)


# ── Workload runner ──
def get_energy(host, gpus):
    py = ("import pynvml,json;pynvml.nvmlInit();"
          f"print(json.dumps({{i:pynvml.nvmlDeviceGetTotalEnergyConsumption("
          f"pynvml.nvmlDeviceGetHandleByIndex(i)) for i in {gpus}}}));"
          "pynvml.nvmlShutdown()")
    r = _dexec(host, f'{PYTHON} -c "{py}"')
    try:
        for line in (r.stdout or "").strip().splitlines():
            if line.startswith("{"):
                return {int(k): v for k, v in json.loads(line).items()}
    except Exception:
        pass
    return {i: 0 for i in gpus}


async def send_one(session, url, req, base_time, results):
    import aiohttp
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0: await asyncio.sleep(delay)
    payload = {"text": "x" * req["input_len"],
               "sampling_params": {"max_new_tokens": req["output_len"],
                                   "temperature": 0.0, "ignore_eos": True},
               "stream": True}
    t0 = time.monotonic(); first_token_time = None; token_count = 0; last_meta = {}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"success": False}); return
            async for line in resp.content:
                now = time.monotonic()
                text = line.decode().strip()
                if not text or text.startswith(":"): continue
                if text.startswith("data:"): text = text[5:].strip()
                if text == "[DONE]": break
                try:
                    chunk = json.loads(text)
                    if first_token_time is None: first_token_time = now
                    token_count += 1
                    if isinstance(chunk, dict) and "meta_info" in chunk:
                        last_meta = chunk["meta_info"]
                except json.JSONDecodeError: pass
    except Exception:
        results.append({"success": False}); return
    t_end = time.monotonic()
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0
    ttft_proc_ms = 0.0
    if last_meta.get("ttft_pure_processing"):
        ttft_proc_ms = last_meta["ttft_pure_processing"] * 1000
    elif last_meta.get("time_to_first_token_processing"):
        ttft_proc_ms = last_meta["time_to_first_token_processing"] * 1000
    tpot_ms = 0.0
    if token_count > 1 and first_token_time:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)
    results.append({"success": True, "completion_tokens": token_count,
                    "ttft_ms": ttft_ms, "ttft_proc_ms": ttft_proc_ms, "tpot_ms": tpot_ms})


async def run_workload(reqs, url, run_s):
    import aiohttp, numpy as np
    gpus = list(range(GPUS_PER_NODE))
    e1s = get_energy(NODE1_IP, gpus); e2s = get_energy(NODE2_IP, gpus)
    results = []
    timeout = aiohttp.ClientTimeout(total=run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [asyncio.create_task(send_one(session, url, r, base_time, results)) for r in reqs]
        if tasks:
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), timeout=run_s)
            except asyncio.TimeoutError:
                log.warning("Timed out after %ds", run_s)
    duration_s = time.monotonic() - base_time
    e1e = get_energy(NODE1_IP, gpus); e2e = get_energy(NODE2_IP, gpus)
    energy_n1 = sum((e1e.get(i, 0) - e1s.get(i, 0)) / 1000.0 for i in gpus)
    energy_n2 = sum((e2e.get(i, 0) - e2s.get(i, 0)) / 1000.0 for i in gpus)
    total_energy = energy_n1 + energy_n2

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    if not ok: return {"status": "FAIL", "failed": len(fail)}

    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0
    src = ttfts_proc if ttfts_proc else ttfts
    n_ttft_viol = sum(1 for v in src if v > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for r in ok if r["tpot_ms"] > TPOT_SLO_MS)
    slo_rate = (n_ttft_viol + n_tpot_viol + len(fail)) / len(results) * 100 if results else 0

    return {
        "status": "PASS", "duration_s": round(duration_s, 1),
        "total_requests": len(reqs), "successful": len(ok), "failed": len(fail),
        "total_tokens": total_tokens, "throughput_tok_s": round(throughput, 1),
        "ttft_proc_avg_ms": round(float(np.mean(src)), 1) if src else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(src, 50)), 1) if src else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(src, 99)), 1) if src else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "energy_n3_j": round(energy_n1, 1), "energy_n4_j": round(energy_n2, 1),
        "total_energy_j": round(total_energy, 1),
        "energy_per_token_mj": round(total_energy * 1000 / total_tokens, 2) if total_tokens else 0,
        "slo_violation_rate": round(slo_rate, 1),
        "ttft_violations": n_ttft_viol, "tpot_violations": n_tpot_viol,
    }


# ── Main ──
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", default=None,
                        choices=[c.name for c in CONFIGS])
    parser.add_argument("--skip-deploy", action="store_true")
    parser.add_argument("--url", type=str, default=None)
    args = parser.parse_args()

    targets = [c for c in CONFIGS if args.configs is None or c.name in args.configs]
    results = {}

    for cfg in targets:
        log.info("\n" + "#" * 72)
        log.info("TESTING: %s", cfg.name)
        log.info("#" * 72)

        if args.skip_deploy:
            url = args.url
            if url is None: raise ValueError("--url required")
        else:
            cleanup_all()
            time.sleep(5)
            url = deploy(cfg)
            if url is None:
                log.error("DEPLOY FAILED"); results[cfg.name] = {"status": "DEPLOY_FAILED"}; continue
            if not test_generate(url):
                log.error("WARMUP FAILED"); results[cfg.name] = {"status": "WARMUP_FAILED"}
                cleanup_all(); continue
            time.sleep(3)

        # Load workload
        wl_file = WORKLOAD_DIR / f"macro_{DATASET}_qps{QPS}.jsonl"
        if not wl_file.exists():
            log.error("Workload not found: %s", wl_file)
            results[cfg.name] = {"status": "NO_WORKLOAD"}; continue
        with open(wl_file) as f:
            reqs = [json.loads(line) for line in f]
        last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
        run_s = int(min(max(MAX_RUN_S, last_arrival + 150), 900))
        log.info("Workload: %d reqs, run_window=%ds", len(reqs), run_s)

        # Start freq monitors on both nodes (all 8 GPUs)
        mon_n3 = FreqMonitor(NODE1_IP, list(range(8)))
        mon_n4 = FreqMonitor(NODE2_IP, list(range(8)))
        mon_n3.start(); mon_n4.start()
        t_start = time.time()

        try:
            summary = asyncio.run(run_workload(reqs, url + "/generate", run_s))
        finally:
            mon_n3.stop(); mon_n4.stop()
            t_end = time.time()
            # Collect freq samples
            freq_samples = []
            for s in mon_n3.samples: freq_samples.append({"node": "n3", **s})
            for s in mon_n4.samples: freq_samples.append({"node": "n4", **s})
            freq_samples.sort(key=lambda x: x["t"])

        if isinstance(summary, dict) and summary.get("status") == "PASS":
            summary["config"] = {f.name: getattr(cfg, f.name) for f in cfg.__dataclass_fields__}
            summary["freq_timeline"] = {
                "samples": len(freq_samples), "duration_s": round(t_end - t_start, 1),
            }
            # Save freq timeline separately
            ft_path = FREQ_TIMELINE_DIR / f"{cfg.name}_{DATASET}_qps{QPS}.json"
            ft_path.write_text(json.dumps({
                "config": cfg.name, "dataset": DATASET, "qps": QPS,
                "t_start": t_start, "t_end": t_end,
                "nvidia_smi_timeline": freq_samples,
            }, indent=2))
            summary["freq_timeline_path"] = str(ft_path.relative_to(HERE))
            log.info("PASS: thpt=%.1f TTFT=%.1fms TPOT=%.1fms E/tok=%.1fmJ SLO=%.1f%% freq_n=%d",
                     summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                     summary["tpot_avg_ms"], summary["energy_per_token_mj"],
                     summary["slo_violation_rate"], len(freq_samples))
        else:
            log.error("FAIL: %s", summary)

        results[cfg.name] = summary

        if not args.skip_deploy:
            unlock_freq(NODE1_IP); unlock_freq(NODE2_IP)
            cleanup_all()
            time.sleep(5)

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"tier1_bench_{ts}.json"
    meta = {"benchmark": "tier1_config_comparison", "qps": QPS, "dataset": DATASET,
            "ttft_slo_ms": TTFT_SLO_MS, "tpot_slo_ms": TPOT_SLO_MS,
            "node3": NODE1_IP, "node4": NODE2_IP}
    out_path.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    log.info("Saved %s", out_path)

    print(f"\n{'='*72}")
    print("RESULTS")
    print(f"{'='*72}")
    for name, r in results.items():
        if r.get("status") == "PASS":
            print(f"  {name}: thpt={r['throughput_tok_s']:.1f} TTFT={r['ttft_proc_avg_ms']:.1f}ms "
                  f"TPOT={r['tpot_avg_ms']:.1f}ms E/tok={r['energy_per_token_mj']:.1f}mJ "
                  f"SLO={r['slo_violation_rate']:.1f}%")
        else:
            print(f"  {name}: {r.get('status')}")


if __name__ == "__main__":
    main()
