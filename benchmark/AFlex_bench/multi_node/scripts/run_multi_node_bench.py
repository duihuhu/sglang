#!/usr/bin/env python3
"""Multi-node (16-GPU) micro-benchmark for Qwen3-32B across 4 architectures.

Architectures (all 16 GPU = node1 8 + node2 8):
  - native_dp : 16x TP=1 instances (node1 GPU0-7 + node2 GPU0-7), round_robin router.
  - pd_dp_xnode : 8 PD pairs, prefill 8 inst on node1, decode 8 inst on node2 (cross-node KV).
  - pd_dp_intra : 8 PD pairs, 4P+4D on each node (KV mostly intra-node).
  - pdaf : 4PA+4PF on node1 + 4DA+4DF on node2 (tp=4), AF operator disaggregation.

Each architecture runs in two power modes:
  - baseline : GPU clocks locked to max (1410 MHz).
  - tier     : per-instance DVFS via energy model (--dvfs-* / --afd-dvfs-*).

Topology notes (pdaf):
  node1 (Prefill): PF base_gpu_id=0 -> CUDA 0..3; PA base_gpu_id=tp -> CUDA 4..7.
  node2 (Decode):  DF base_gpu_id=0 -> CUDA 0..3; DA base_gpu_id=tp -> CUDA 4..7.
  PA<->PF / DA<->DF via intra-node CUDA IPC (ipc_cpp). P->D KV via mooncake RoCE.

Both nodes run an `operator_test` container with /workspace/sglang synced and
the model at /models/Qwen3-32B. The ORCHESTRATOR runs on the HOST (containers
have no `docker` cmd); node1 host orchestrates node2 over
`ssh <node2> docker exec operator_test ...`.

Usage (run on node1 HOST):
    python3 run_multi_node_bench.py --arch all --mode all --scenario all --qps 1,2,4
    python3 run_multi_node_bench.py --arch pdaf,native_dp --mode baseline --qps 1,2
"""
import argparse
import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

import aiohttp
import numpy as np
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mn_bench")

# ---- Topology / environment ----
NODE1_IP = os.environ.get("MN_NODE1_IP", "10.252.129.36")  # prefill side (local)
NODE2_IP = os.environ.get("MN_NODE2_IP", "10.252.129.35")  # decode side (remote)
CONTAINER = os.environ.get("MN_CONTAINER", "operator_test")
PYTHON = "/usr/bin/python3"
MODEL = "/models/Qwen3-32B/"
IB_DEV = os.environ.get("MN_IB_DEV", "mlx5_bond_0")

HERE = Path(__file__).resolve().parent
BASE = HERE.parent                       # .../multi_node (host path)
LOG_C = "/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"  # container path (for server stdout)
# Orchestrator runs on the HOST, so workloads use the host-mounted path.
WORKLOAD_DIR = Path("/mnt/workspace/lt/sglang/benchmark/AFlex_bench/retesting/workloads")
CLEANUP = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"  # container path

MAX_GPU_FREQ = 1410

# DVFS / energy model (Tier mode)
ENERGY_MODEL_DIR = ("/workspace/sglang/benchmark/AFlex_bench/03_sensitivity/"
                    "slo_sweep/retrain/models_v2")

# Ports
ROUTER_PORT = 42000
PA_PORT, PF_PORT = 42010, 42011
DA_PORT, DF_PORT = 42020, 42021
BS_PORT = 49999

# SLO defaults
TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0


# ============================================================
# Remote / container orchestration helpers
# ============================================================

def _ssh(host, cmd):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", host, cmd]


def dexec_local_bg(shell_cmd):
    """Launch a backgrounded command inside the local node1 container."""
    subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", shell_cmd],
                   check=False)


def dexec_remote_bg(shell_cmd):
    """Launch a backgrounded command inside the node2 container via ssh."""
    inner = f"docker exec {CONTAINER} bash -lc {json_quote(shell_cmd)}"
    subprocess.run(_ssh(NODE2_IP, inner), check=False)


def json_quote(s):
    """Quote a shell command so it survives ssh + bash -lc nesting."""
    import shlex
    return shlex.quote(s)


def cleanup_all():
    dexec_local_bg(f"bash {CLEANUP}")
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc 'bash {CLEANUP}'"),
                   check=False)
    time.sleep(5)


def wait_health(host, port, timeout=600, check_model_info=False):
    deadline = time.time() + timeout
    ep = "get_model_info" if check_model_info else "health"
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{host}:{port}/{ep}", timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def get_gpu_energy_mj_local(gpu_indices):
    """Read NVML total energy (mJ) for local GPUs."""
    try:
        import pynvml
        pynvml.nvmlInit()
        result = {}
        for idx in gpu_indices:
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            result[idx] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        pynvml.nvmlShutdown()
        return result
    except Exception as e:
        log.warning("local NVML energy read failed: %s", e)
        return {idx: 0 for idx in gpu_indices}


def get_gpu_energy_mj_remote(gpu_indices):
    """Read NVML total energy (mJ) for node2 GPUs via host-level ssh python."""
    idx_csv = ",".join(str(i) for i in gpu_indices)
    pycode = (
        "import pynvml,json;pynvml.nvmlInit();"
        f"idxs=[int(x) for x in '{idx_csv}'.split(',')];"
        "print(json.dumps({i:pynvml.nvmlDeviceGetTotalEnergyConsumption("
        "pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idxs}));"
        "pynvml.nvmlShutdown()"
    )
    inner = f"{PYTHON} -c {json_quote(pycode)}"
    try:
        out = subprocess.run(_ssh(NODE2_IP, inner), capture_output=True,
                             text=True, timeout=30)
        line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
        return {int(k): v for k, v in json.loads(line[-1]).items()} if line else \
               {i: 0 for i in gpu_indices}
    except Exception as e:
        log.warning("remote NVML energy read failed: %s", e)
        return {i: 0 for i in gpu_indices}


def lock_freq_both(freq=MAX_GPU_FREQ):
    cmd = ";".join(f"nvidia-smi -i {i} --lock-gpu-clocks={freq},{freq}"
                   for i in range(8)) + ";true"
    dexec_local_bg(cmd)
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc {json_quote(cmd)}"),
                   check=False)
    log.info("  Locked both nodes' GPUs 0-7 to %d MHz", freq)


def unlock_freq_both():
    cmd = ";".join(f"nvidia-smi -i {i} --reset-gpu-clocks" for i in range(8)) + ";true"
    dexec_local_bg(cmd)
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc {json_quote(cmd)}"),
                   check=False)
    log.info("  Unlocked both nodes' GPUs")


# ============================================================
# PDAF deployment (16 GPU, 4PA+4PF+4DA+4DF)
# ============================================================

def _afd_env_exports(tp, role):
    """Build the `export ...;` prefix for one AFD server.

    role in {PF, PA, DF, DA}.
    """
    base = ("export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            "AFD_IPC_SYNC_MODE=ipc_event "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 ")
    ffn_nvml = ",".join(str(i) for i in range(tp))          # 0..tp-1
    attn_nvml = ",".join(str(i) for i in range(tp, 2 * tp))  # tp..2tp-1
    ucx = "28200" if role in ("PF", "PA") else "28300"
    sched = "68400" if role in ("PF", "PA") else "68500"
    if role in ("PF", "DF"):  # FFN side
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET={tp} "
                f"AFD_NVML_DEVICE_INDICES={ffn_nvml} AFD_NVML_DEVICE_INDEX=0;")
    else:  # ATTN side
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET=-{tp} "
                f"AFD_NVML_DEVICE_INDICES={attn_nvml} AFD_NVML_DEVICE_INDEX={tp} "
                f"AFD_UCX_FFN_HOST=127.0.0.1;")


def _common_flags(tp, tier=False):
    base = (f"--model-path {MODEL} --tp {tp} --afd-comm-backend ipc_cpp "
            "--afd-micro-batch 2 --afd-dynamic-micro-batch --mem-fraction-static 0.85 "
            "--max-running-requests 512 --skip-server-warmup "
            "--disable-cuda-graph --disable-piecewise-cuda-graph "
            "--afd-disagg-interleave-poll --disable-radix-cache "
            "--num-reserved-decode-tokens 512 "
            "--disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {BS_PORT} "
            f"--disaggregation-ib-device {IB_DEV} --enable-metrics")
    if tier:
        base += (" --afd-dvfs-enabled "
                 f"--afd-energy-model-dir {ENERGY_MODEL_DIR} "
                 f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)} "
                 f"--afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)} "
                 "--afd-dvfs-idle-lock")
    return base


def _launch(role, host, port, perspective, disagg, base_gpu, tp, logname, nccl_port,
            tier=False):
    env = _afd_env_exports(tp, role)
    cmd = (f"{env} setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
           f"-m sglang.launch_server --host {host} --port {port} "
           f"--nccl-port {nccl_port} "
           f"--afd-perspective {perspective} --disaggregation-mode {disagg} "
           f"--base-gpu-id {base_gpu} {_common_flags(tp, tier)} "
           f"> {LOG_C}/{logname}.log 2>&1 < /dev/null &")
    if host == NODE1_IP:
        dexec_local_bg(cmd)
    else:
        dexec_remote_bg(cmd)


def start_pdaf(tp, tier=False):
    """Launch 4PA+4PF (node1) + 4DA+4DF (node2) + router. Returns router URL."""
    log.info("Launching PDAF: P=node1 8GPU (PF+PA tp%d), D=node2 8GPU (DF+DA tp%d) tier=%s",
             tp, tp, tier)
    # Distinct nccl ports per server to avoid EADDRINUSE when PF/PA (and DF/DA)
    # share the same node + CUDA_VISIBLE_DEVICES.
    # node1 prefill side
    _launch("PF", NODE1_IP, PF_PORT, "ffn", "prefill", 0, tp, "pf", 39411, tier)
    time.sleep(6)
    _launch("PA", NODE1_IP, PA_PORT, "attn", "prefill", tp, tp, "pa", 39421, tier)
    # node2 decode side
    _launch("DF", NODE2_IP, DF_PORT, "ffn", "decode", 0, tp, "df", 39431, tier)
    time.sleep(8)
    _launch("DA", NODE2_IP, DA_PORT, "attn", "decode", tp, tp, "da", 39441, tier)

    log.info("Waiting for PDAF servers (timeout 600s each)...")
    checks = [(NODE1_IP, PF_PORT, "PF", True), (NODE1_IP, PA_PORT, "PA", False),
              (NODE2_IP, DF_PORT, "DF", True), (NODE2_IP, DA_PORT, "DA", False)]
    for host, port, name, mi in checks:
        if not wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed to start", name, host, port)
            return None
        log.info("  %s ready (%s:%d)", name, host, port)

    # router on node1: prefill=PA(node1), decode=DA(node2)
    router_cmd = (f"setsid {PYTHON} -m sglang_router.launch_router "
                  "--pd-disaggregation --mini-lb "
                  f"--prefill http://{NODE1_IP}:{PA_PORT} "
                  f"--decode http://{NODE2_IP}:{DA_PORT} "
                  f"--host {NODE1_IP} --port {ROUTER_PORT} "
                  f"> {LOG_C}/router.log 2>&1 < /dev/null &")
    dexec_local_bg(router_cmd)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  router failed")
        return None
    log.info("  router ready (%s:%d)", NODE1_IP, ROUTER_PORT)
    return f"http://{NODE1_IP}:{ROUTER_PORT}"


# ============================================================
# Native DP16 / PD DP8 deployments (TP=1 instances)
# ============================================================

def _dp_dvfs_flags(tier):
    """Non-AFD DVFS flags (Native DP / PD DP use --dvfs-*)."""
    if not tier:
        return ""
    return (" --dvfs-enabled "
            f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR} "
            f"--dvfs-ttft-slo-ms {int(TTFT_SLO_MS)} "
            f"--dvfs-tpot-slo-us {int(TPOT_SLO_MS * 1000)}")


def _dp_env(gpu):
    return (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={gpu} AFD_NVML_DEVICE_INDEX={gpu} "
            f"AFD_NVML_DEVICE_INDICES={gpu};")


def _launch_plain(host, gpu, port, nccl_port, logname, extra, tier):
    """Launch one TP=1 sglang server pinned to a single GPU."""
    cmd = (f"{_dp_env(gpu)} setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
           f"-m sglang.launch_server --model-path {MODEL} --tp 1 "
           f"--host {host} --port {port} --nccl-port {nccl_port} "
           "--mem-fraction-static 0.85 --disable-cuda-graph "
           "--disable-piecewise-cuda-graph --skip-server-warmup "
           f"{extra}{_dp_dvfs_flags(tier)} "
           f"> {LOG_C}/{logname}.log 2>&1 < /dev/null &")
    if host == NODE1_IP:
        dexec_local_bg(cmd)
    else:
        dexec_remote_bg(cmd)


def start_native_dp(tier=False):
    """16x TP=1 instances: node1 GPU0-7 + node2 GPU0-7, round_robin router."""
    log.info("Launching Native DP16: 8 inst/node x 2 nodes, tier=%s", tier)
    insts = []  # (host, gpu, port)
    for gpu in range(8):
        insts.append((NODE1_IP, gpu, 53200 + gpu * 10))
    for gpu in range(8):
        insts.append((NODE2_IP, gpu, 53200 + gpu * 10))
    for i, (host, gpu, port) in enumerate(insts):
        _launch_plain(host, gpu, port, 33300 + gpu * 10,
                      f"dp_{'n1' if host == NODE1_IP else 'n2'}_{gpu}", "", tier)
        if i % 4 == 3:
            time.sleep(2)

    for host, gpu, port in insts:
        if not wait_health(host, port, 400):
            log.error("  DP inst %s:%d (gpu%d) failed", host, port, gpu)
            return None
    log.info("  all 16 DP instances ready")

    worker_urls = " ".join(f"http://{h}:{p}" for h, _, p in insts)
    router_cmd = (f"setsid {PYTHON} -m sglang_router.launch_router "
                  f"--host {NODE1_IP} --port {ROUTER_PORT} --policy round_robin "
                  f"--worker-urls {worker_urls} "
                  f"> {LOG_C}/router.log 2>&1 < /dev/null &")
    dexec_local_bg(router_cmd)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  router failed")
        return None
    log.info("  router ready (%s:%d)", NODE1_IP, ROUTER_PORT)
    return f"http://{NODE1_IP}:{ROUTER_PORT}"


def _start_pd_dp(pairs, tier=False):
    """Launch PD DP given pairs of ((p_host,p_gpu),(d_host,d_gpu)). TP=1 each.

    Returns router URL or None.
    """
    instances = []
    for i, ((p_host, p_gpu), (d_host, d_gpu)) in enumerate(pairs):
        instances.append({
            "idx": i,
            "p_host": p_host, "p_gpu": p_gpu, "p_port": 53100 + i * 10,
            "d_host": d_host, "d_gpu": d_gpu, "d_port": 53101 + i * 10,
            "bs_port": 49100 + i * 10,
        })
    for inst in instances:
        p_extra = ("--disaggregation-mode prefill "
                   "--disaggregation-transfer-backend mooncake "
                   f"--disaggregation-bootstrap-port {inst['bs_port']} "
                   f"--disaggregation-ib-device {IB_DEV}")
        _launch_plain(inst["p_host"], inst["p_gpu"], inst["p_port"],
                      34000 + inst["idx"] * 10, f"pd{inst['idx']}_p", p_extra, tier)
        time.sleep(2)
        d_extra = ("--disaggregation-mode decode "
                   "--disaggregation-transfer-backend mooncake "
                   f"--disaggregation-bootstrap-port {inst['bs_port']} "
                   f"--disaggregation-ib-device {IB_DEV}")
        _launch_plain(inst["d_host"], inst["d_gpu"], inst["d_port"],
                      34001 + inst["idx"] * 10, f"pd{inst['idx']}_d", d_extra, tier)
        time.sleep(2)

    for inst in instances:
        for role, host, port in [("P", inst["p_host"], inst["p_port"]),
                                 ("D", inst["d_host"], inst["d_port"])]:
            if not wait_health(host, port, 400):
                log.error("  PD%d %s (%s:%d) failed", inst["idx"], role, host, port)
                return None
    log.info("  all %d PD pairs ready", len(instances))

    router_cmd = [f"setsid {PYTHON} -m sglang_router.launch_router",
                  "--pd-disaggregation",
                  f"--host {NODE1_IP} --port {ROUTER_PORT}"]
    for inst in instances:
        router_cmd.append(f"--prefill http://{inst['p_host']}:{inst['p_port']} "
                          f"{inst['bs_port']}")
    for inst in instances:
        router_cmd.append(f"--decode http://{inst['d_host']}:{inst['d_port']}")
    rc = " ".join(router_cmd) + f" > {LOG_C}/router.log 2>&1 < /dev/null &"
    dexec_local_bg(rc)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  PD router failed")
        return None
    log.info("  router ready (%s:%d)", NODE1_IP, ROUTER_PORT)
    return f"http://{NODE1_IP}:{ROUTER_PORT}"


def start_pd_dp_xnode(tier=False):
    """8 PD pairs: prefill 8 inst on node1, decode 8 inst on node2 (cross-node KV)."""
    log.info("Launching PD DP8 (cross-node): P=node1 GPU0-7, D=node2 GPU0-7, tier=%s", tier)
    pairs = [((NODE1_IP, g), (NODE2_IP, g)) for g in range(8)]
    return _start_pd_dp(pairs, tier)


def start_pd_dp_intra(tier=False):
    """8 PD pairs: 4P+4D on each node (KV mostly intra-node).

    node1: P on GPU0-3, D on GPU4-7. node2: P on GPU0-3, D on GPU4-7.
    """
    log.info("Launching PD DP8 (intra-node): 4P+4D per node, tier=%s", tier)
    pairs = []
    for g in range(4):  # node1 P(g) -> node1 D(g+4)
        pairs.append(((NODE1_IP, g), (NODE1_IP, g + 4)))
    for g in range(4):  # node2 P(g) -> node2 D(g+4)
        pairs.append(((NODE2_IP, g), (NODE2_IP, g + 4)))
    return _start_pd_dp(pairs, tier)


# Architecture registry: name -> (start_fn, label)
ARCHS = {
    "native_dp":   (lambda tier: start_native_dp(tier), "Native DP16"),
    "pd_dp_xnode": (lambda tier: start_pd_dp_xnode(tier), "PD DP8 (x-node)"),
    "pd_dp_intra": (lambda tier: start_pd_dp_intra(tier), "PD DP8 (intra)"),
    "pdaf":        (lambda tier: start_pdaf(4, tier), "PDAF 4PA4PF4DA4DF"),
}

async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"text": "x" * req["input_len"],
               "sampling_params": {"max_new_tokens": req["output_len"],
                                   "temperature": 0.0, "ignore_eos": True},
               "stream": True}
    t0 = time.monotonic()
    first_token_time = None
    token_count = 0
    last_meta = {}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"success": False})
                return
            async for line in resp.content:
                now = time.monotonic()
                text = line.decode().strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    chunk = json.loads(text)
                    if first_token_time is None:
                        first_token_time = now
                    token_count += 1
                    if isinstance(chunk, dict) and "meta_info" in chunk:
                        last_meta = chunk["meta_info"]
                except json.JSONDecodeError:
                    pass
    except Exception:
        results.append({"success": False})
        return

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

    results.append({
        "success": True, "completion_tokens": token_count,
        "ttft_ms": ttft_ms, "ttft_proc_ms": ttft_proc_ms,
        "tpot_ms": tpot_ms, "e2e_s": t_end - t0,
    })


async def run_workload(reqs, url, n1_gpus, n2_gpus, max_run_s=400):
    e1_start = get_gpu_energy_mj_local(n1_gpus)
    e2_start = get_gpu_energy_mj_remote(n2_gpus)
    results = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [asyncio.create_task(send_one(session, url, r, base_time, results))
                 for r in reqs]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=max_run_s)
            except asyncio.TimeoutError:
                log.warning("Timed out after %ds", max_run_s)
    duration_s = time.monotonic() - base_time
    e1_end = get_gpu_energy_mj_local(n1_gpus)
    e2_end = get_gpu_energy_mj_remote(n2_gpus)
    energy_n1_j = sum((e1_end.get(i, 0) - e1_start.get(i, 0)) / 1000.0 for i in n1_gpus)
    energy_n2_j = sum((e2_end.get(i, 0) - e2_start.get(i, 0)) / 1000.0 for i in n2_gpus)
    total_energy_j = energy_n1_j + energy_n2_j

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    if not ok:
        return {"status": "FAIL", "failed": len(fail)}

    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0
    n_ttft_viol = sum(1 for r in ok if (r.get("ttft_proc_ms") or r["ttft_ms"]) > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for r in ok if r["tpot_ms"] > TPOT_SLO_MS)
    n_slo_viol = n_ttft_viol + n_tpot_viol + len(fail)
    slo_rate = n_slo_viol / len(results) * 100 if results else 0

    return {
        "status": "PASS", "duration_s": round(duration_s, 1),
        "total_requests": len(reqs), "successful": len(ok), "failed": len(fail),
        "total_tokens": total_tokens, "throughput_tok_s": round(throughput, 1),
        "ttft_proc_avg_ms": round(float(np.mean(ttfts_proc)), 1) if ttfts_proc else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(ttfts_proc, 99)), 1) if ttfts_proc else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "energy_node1_j": round(energy_n1_j, 1),
        "energy_node2_j": round(energy_n2_j, 1),
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens else 0,
        "slo_violation_rate": round(slo_rate, 1),
        "ttft_violations": n_ttft_viol, "tpot_violations": n_tpot_viol,
    }


SCENARIOS = ["chatbot", "qa", "rag", "summary"]


# ============================================================
# Main
# ============================================================

def test_generate(url):
    try:
        r = requests.post(url + "/generate", json={
            "text": "Hello, explain quantum computing:",
            "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}, timeout=120)
        return "text" in r.json()
    except Exception as e:
        log.error("warmup generate failed: %s", e)
        return False


def run_one_workload(url, scenario, qps, n1_gpus, n2_gpus, max_run_s):
    wl_file = WORKLOAD_DIR / f"micro_{scenario}_qps{qps}.jsonl"
    if not wl_file.exists():
        log.warning("  workload not found: %s", wl_file)
        return None
    with open(wl_file) as f:
        reqs = [json.loads(l) for l in f]
    wl_key = f"{scenario}_qps{qps}"
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(max_run_s, last_arrival + 150), 900))
    log.info("  %s (%d reqs, last_arrival=%.0fs, run_window=%ds)",
             wl_key, len(reqs), last_arrival, run_s)
    summary = asyncio.run(run_workload(reqs, url + "/generate", n1_gpus, n2_gpus, run_s))
    if summary.get("status") == "PASS":
        log.info("  Thpt=%.1f tok/s | TTFT=%.1fms | TPOT=%.1fms | "
                 "E=%.0fJ (n1=%.0f n2=%.0f, %.1f mJ/tok) | SLO=%.1f%%",
                 summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                 summary["tpot_avg_ms"], summary["total_energy_j"],
                 summary["energy_node1_j"], summary["energy_node2_j"],
                 summary["energy_per_token_mj"], summary["slo_violation_rate"])
    else:
        log.error("  FAIL: %s", summary)
    return wl_key, summary


def run_deploy(arch, tier, scenarios, qps_list, max_run_s):
    """Deploy one architecture in one power mode, run all workloads. Returns dict."""
    start_fn, label = ARCHS[arch]
    mode = "tier" if tier else "baseline"
    full_name = f"{arch}_{mode}"
    log.info("=" * 70)
    log.info("DEPLOY: %s [%s]  (16 GPU, Qwen3-32B)", label, mode)
    log.info("=" * 70)

    cleanup_all()
    url = start_fn(tier)
    if url is None:
        log.error("  %s deployment FAILED", full_name)
        cleanup_all()
        return full_name, {"__status__": "DEPLOY_FAILED"}

    n1_gpus = list(range(8))
    n2_gpus = list(range(8))
    # Baseline: lock freq to max. Tier: let DVFS control clocks.
    if not tier:
        lock_freq_both(MAX_GPU_FREQ)

    log.info("Warmup...")
    if not test_generate(url):
        log.error("  warmup failed")
        if not tier:
            unlock_freq_both()
        cleanup_all()
        return full_name, {"__status__": "WARMUP_FAILED"}
    time.sleep(3)

    deploy_results = {}
    for scenario in scenarios:
        for qps in qps_list:
            log.info("-" * 50)
            res = run_one_workload(url, scenario, qps, n1_gpus, n2_gpus, max_run_s)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

    if not tier:
        unlock_freq_both()
    cleanup_all()
    return full_name, deploy_results


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="all",
                        help="Comma-sep: native_dp,pd_dp_xnode,pd_dp_intra,pdaf or 'all'")
    parser.add_argument("--mode", default="all",
                        help="Comma-sep: baseline,tier or 'all'")
    parser.add_argument("--scenario", default="all",
                        help="Comma-sep: chatbot,qa,rag,summary or 'all'")
    parser.add_argument("--qps", default="1,2,4", help="Comma-sep QPS list")
    parser.add_argument("--max-run-s", type=int, default=400)
    args = parser.parse_args()

    archs = list(ARCHS.keys()) if args.arch == "all" else args.arch.split(",")
    modes = ["baseline", "tier"] if args.mode == "all" else args.mode.split(",")
    scenarios = SCENARIOS if args.scenario == "all" else args.scenario.split(",")
    qps_list = [int(q) for q in args.qps.split(",")]

    result_dir = BASE / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for arch in archs:
        if arch not in ARCHS:
            log.error("unknown arch: %s", arch)
            continue
        for mode in modes:
            tier = (mode == "tier")
            full_name, res = run_deploy(arch, tier, scenarios, qps_list, args.max_run_s)
            all_results[full_name] = res
            # Persist incrementally so a crash mid-run keeps prior data.
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_file = result_dir / f"multi_arch_16gpu_{ts}.json"
            payload = {"meta": {"node1": NODE1_IP, "node2": NODE2_IP, "ib_dev": IB_DEV,
                                "model": MODEL, "archs": archs, "modes": modes,
                                "scenarios": scenarios, "qps": qps_list},
                       "results": all_results}
            with open(out_file, "w") as f:
                json.dump(payload, f, indent=2)
            log.info("Results saved (incremental): %s", out_file)

    # Final summary table
    print("\n" + "=" * 104)
    print("  MULTI-NODE 16-GPU BENCHMARK (Qwen3-32B)")
    print("=" * 104)
    hdr = (f"{'Deploy':<24} {'Workload':<16} {'Thpt':>8} {'TTFT':>8} {'TPOT':>8} "
           f"{'E_tot':>9} {'mJ/tok':>8} {'SLO%':>6}")
    print(hdr)
    print("-" * 104)
    for dep, wl_results in all_results.items():
        if "__status__" in wl_results:
            print(f"{dep:<24} {wl_results['__status__']}")
            continue
        for wl, m in wl_results.items():
            if m.get("status") != "PASS":
                print(f"{dep:<24} {wl:<16} FAIL")
                continue
            print(f"{dep:<24} {wl:<16} {m['throughput_tok_s']:>8.1f} "
                  f"{m['ttft_proc_avg_ms']:>8.1f} {m['tpot_avg_ms']:>8.1f} "
                  f"{m['total_energy_j']:>9.0f} {m['energy_per_token_mj']:>8.1f} "
                  f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
