#!/usr/bin/env python3
"""Cross-node scalability benchmark for Mixtral-8x7B (MoE).

Tests 3 architectures x 2 power modes (= 6 schemes) across 4 datasets, on
cross-node (2-node) deployments. MoE requires TP=2 minimum (model ~90GB).

Card configs (TP=2 per instance):
  8-card: each node uses GPU [4,5,6,7] → 2 TP=2 instances per node
  16-card: each node uses GPU [0..7] → 4 TP=2 instances per node

Schemes:
  native_dp : N x TP=2 instances, round_robin router.
  pd_dp     : N/2 PD pairs (P-TP2+D-TP2) per node.
  pdaf      : PA(TP4)+PF(TP4) on node1, DA(TP4)+DF(TP4) on node2.
  pdaf_dp2  : 2x [PA(TP2)+PF(TP2)] on node1, 2x [DA(TP2)+DF(TP2)] on node2.

Usage:
    python3 run_moe_scalability.py --ngpu 16 --deploy all --mode all
    python3 run_moe_scalability.py --ngpu 8 --deploy native_dp --mode baseline --qps 1,2,3,4,5,6
"""
import argparse
import asyncio
import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import aiohttp
import numpy as np
import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("moe_bench")

# ---- Topology ----
NODE1_IP = os.environ.get("MN_NODE1_IP", "10.252.129.36")  # prefill side (local)
NODE2_IP = os.environ.get("MN_NODE2_IP", "10.252.129.35")  # decode side (remote)
CONTAINER = os.environ.get("MN_CONTAINER", "operator_test")
PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
LOG_C = "/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"
WORKLOAD_DIR = Path("/mnt/workspace/lt/sglang/benchmark/AFlex_bench/retesting/workloads")
CLEANUP = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"

MAX_GPU_FREQ = 1410
# Native/PD use V1 (per-layer A/F split) models via UnifiedDVFSController.
# PDAF uses V2 (coupled decode-iteration) models via AFD DVFS controller.
ENERGY_MODEL_DIR_V1 = ("/workspace/sglang/benchmark/AFlex_bench/06_others/"
                       "Mixtral_test/energy_model/models_v1")
ENERGY_MODEL_DIR_V2 = ("/workspace/sglang/benchmark/AFlex_bench/06_others/"
                       "Mixtral_test/energy_model/models_v2")

# Ports
ROUTER_PORT = 42000
PA_PORT, PF_PORT = 42010, 42011
DA_PORT, DF_PORT = 42020, 42021
BS_PORT = 49999
IB_JSON_FILE = "/tmp/ib_scal_map.json"

TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0

# GPU/NIC affinity map (full 8-GPU topology; 4/8-card configs use the tail)
GPU_NIC = {0: "mlx5_0", 1: "mlx5_0", 2: "mlx5_1", 3: "mlx5_1",
           4: "mlx5_4", 5: "mlx5_4", 6: "mlx5_5", 7: "mlx5_5"}

SCENARIOS = ["qa_lpld", "chatbot_lphd", "balanced_mpmd", "rag_hpld", "summary_hphd"]


def card_gpus(ngpu):
    """Per-node GPU list for the given total card count (last cards, affine)."""
    if ngpu == 4:
        return [6, 7]                    # 2 per node x 2 nodes = 4 total
    if ngpu == 8:
        return [4, 5, 6, 7]              # 4 per node x 2 nodes = 8 total
    return [0, 1, 2, 3, 4, 5, 6, 7]      # 8 per node x 2 nodes = 16 total


# ============================================================
# Orchestration helpers
# ============================================================

def _ssh(host, cmd):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", host, cmd]


def dexec_local(shell_cmd):
    subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", shell_cmd], check=False)


def dexec_remote(shell_cmd):
    inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(shell_cmd)}"
    subprocess.run(_ssh(NODE2_IP, inner), check=False)


def cleanup_all():
    dexec_local(f"bash {CLEANUP}")
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


def get_energy_local(gpus):
    try:
        import pynvml
        pynvml.nvmlInit()
        res = {i: pynvml.nvmlDeviceGetTotalEnergyConsumption(
            pynvml.nvmlDeviceGetHandleByIndex(i)) for i in gpus}
        pynvml.nvmlShutdown()
        return res
    except Exception as e:
        log.warning("local NVML energy read failed: %s", e)
        return {i: 0 for i in gpus}


def get_energy_remote(gpus):
    idx_csv = ",".join(str(i) for i in gpus)
    pycode = (
        "import pynvml,json;pynvml.nvmlInit();"
        f"idxs=[int(x) for x in '{idx_csv}'.split(',')];"
        "print(json.dumps({i:pynvml.nvmlDeviceGetTotalEnergyConsumption("
        "pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idxs}));"
        "pynvml.nvmlShutdown()"
    )
    inner = f"{PYTHON} -c {shlex.quote(pycode)}"
    try:
        out = subprocess.run(_ssh(NODE2_IP, inner), capture_output=True,
                             text=True, timeout=30)
        line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
        return {int(k): v for k, v in json.loads(line[-1]).items()} if line else \
               {i: 0 for i in gpus}
    except Exception as e:
        log.warning("remote NVML energy read failed: %s", e)
        return {i: 0 for i in gpus}


def lock_freq_both(gpus, freq=MAX_GPU_FREQ):
    cmd = ";".join(f"nvidia-smi -i {i} --lock-gpu-clocks={freq},{freq}"
                   for i in gpus) + ";true"
    dexec_local(cmd)
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc {shlex.quote(cmd)}"),
                   check=False)
    log.info("  Locked GPUs %s to %d MHz on both nodes", gpus, freq)


def unlock_freq_both(gpus):
    cmd = ";".join(f"nvidia-smi -i {i} --reset-gpu-clocks" for i in gpus) + ";true"
    dexec_local(cmd)
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc {shlex.quote(cmd)}"),
                   check=False)


def write_ib_json(mapping):
    """Write per-GPU NIC JSON mapping to both containers via docker cp."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(mapping, f)
        tmp = f.name
    subprocess.run(["docker", "cp", tmp, f"{CONTAINER}:{IB_JSON_FILE}"], check=False)
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-q",
                    tmp, f"{NODE2_IP}:{IB_JSON_FILE}"], check=False)
    subprocess.run(_ssh(NODE2_IP, f"docker cp {IB_JSON_FILE} {CONTAINER}:{IB_JSON_FILE}"),
                   check=False)
    os.unlink(tmp)


# ============================================================
# Deployment: launchers
# ============================================================

def _launch(host, cmd, logname):
    # cmd = "export VARS...; PYTHON -m sglang..." — the env exports MUST run
    # before python. Insert setsid+prlimit right before PYTHON (after exports)
    # so the env vars (CUDA_VISIBLE_DEVICES, AFD_NVML_DEVICE_INDICES, ...) apply
    # to the python process. Prepending setsid before the exports would make
    # prlimit try to exec "export" and drop all env vars.
    prefix = "setsid prlimit --memlock=unlimited:unlimited "
    full = cmd.replace(f"{PYTHON} -m", f"{prefix}{PYTHON} -m", 1)
    full = f"{full} > {LOG_C}/{logname}.log 2>&1 < /dev/null &"
    if host == NODE1_IP:
        dexec_local(full)
    else:
        dexec_remote(full)


def _dvfs_flags(tier):
    if not tier:
        return ""
    return (" --dvfs-enabled "
            f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR_V1} "
            f"--dvfs-ttft-slo-ms {int(TTFT_SLO_MS)} "
            f"--dvfs-tpot-slo-us {int(TPOT_SLO_MS * 1000)}")


def _afd_dvfs_flags(tier):
    if not tier:
        return ""
    return (" --afd-dvfs-enabled "
            f"--afd-energy-model-dir {ENERGY_MODEL_DIR_V2} "
            f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)} "
            f"--afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)} "
            "--afd-dvfs-idle-lock")


def _plain_env(gpu):
    return (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={gpu} AFD_NVML_DEVICE_INDEX={gpu} "
            f"AFD_NVML_DEVICE_INDICES={gpu};")


def _launch_plain(host, gpu, port, nccl_port, logname, extra, tier):
    cmd = (f"{_plain_env(gpu)} setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
           f"-m sglang.launch_server --model-path {MODEL} --tp 1 "
           f"--host {host} --port {port} --nccl-port {nccl_port} "
           "--mem-fraction-static 0.85 --disable-cuda-graph "
           "--disable-piecewise-cuda-graph --skip-server-warmup "
           f"{extra}{_dvfs_flags(tier)} "
           f"> {LOG_C}/{logname}.log 2>&1 < /dev/null &")
    if host == NODE1_IP:
        dexec_local(cmd)
    else:
        dexec_remote(cmd)


def start_native_dp(ngpu, tier=False):
    """N TP=2 instances: paired GPUs on each node."""
    gpus = card_gpus(ngpu)
    # Pair consecutive GPUs for TP=2
    gpu_pairs = [(gpus[i], gpus[i+1]) for i in range(0, len(gpus), 2)]
    log.info("Native DP (MoE TP=2): %d instances per node, tier=%s", len(gpu_pairs), tier)
    insts = []
    for pair in gpu_pairs:
        insts.append((NODE1_IP, pair, 53200 + pair[0] * 10))
    for pair in gpu_pairs:
        insts.append((NODE2_IP, pair, 53200 + pair[0] * 10))
    for host, pair, port in insts:
        cvd = f"{pair[0]},{pair[1]}"
        nvml = cvd
        cmd = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
               f"CUDA_VISIBLE_DEVICES={cvd} AFD_NVML_DEVICE_INDEX={pair[0]} "
               f"AFD_NVML_DEVICE_INDICES={nvml}; "
               f"setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
               f"-m sglang.launch_server --model-path {MODEL} --tp 2 "
               f"--host {host} --port {port} --nccl-port {33300 + pair[0] * 10} "
               "--mem-fraction-static 0.85 --disable-cuda-graph "
               "--disable-piecewise-cuda-graph --skip-server-warmup "
               f"{_dvfs_flags(tier)} "
               f"> {LOG_C}/dp_{'n1' if host == NODE1_IP else 'n2'}_{pair[0]}.log 2>&1 < /dev/null &")
        if host == NODE1_IP:
            dexec_local(cmd)
        else:
            dexec_remote(cmd)
        time.sleep(2)
    for host, pair, port in insts:
        if not wait_health(host, port, 400):
            log.error("  DP inst %s:%d (gpu%s) failed", host, port, pair)
            return None
    log.info("  all %d DP instances ready", len(insts))
    worker_urls = " ".join(f"http://{h}:{p}" for h, _, p in insts)
    rc = (f"setsid {PYTHON} -m sglang_router.launch_router "
          f"--host {NODE1_IP} --port {ROUTER_PORT} --policy round_robin "
          f"--worker-urls {worker_urls} > {LOG_C}/router.log 2>&1 < /dev/null &")
    dexec_local(rc)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  router failed")
        return None
    log.info("  router ready")
    return f"http://{NODE1_IP}:{ROUTER_PORT}"

def start_pd_dp(ngpu, tier=False):
    """PD pairs with TP=2 each, distributed across both nodes."""
    gpus = card_gpus(ngpu)
    # Each PD pair needs 4 GPUs: P(TP=2) + D(TP=2)
    # Group into quads: (p_gpu0, p_gpu1, d_gpu0, d_gpu1)
    instances = []
    idx = 0
    for host in (NODE1_IP, NODE2_IP):
        for i in range(0, len(gpus), 4):
            if i + 3 >= len(gpus):
                break
            instances.append({
                "idx": idx, "host": host,
                "p_gpus": f"{gpus[i]},{gpus[i+1]}", "d_gpus": f"{gpus[i+2]},{gpus[i+3]}",
                "p_port": 53100 + idx * 10, "d_port": 53101 + idx * 10,
                "bs_port": 49100 + idx * 10,
                "p_nic": GPU_NIC[gpus[i]], "d_nic": GPU_NIC[gpus[i+2]],
            })
            idx += 1
    log.info("PD DP (MoE TP=2): %d pairs, tier=%s", len(instances), tier)
    for inst in instances:
        host = inst["host"]
        # Prefill
        p_cmd = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
                 f"CUDA_VISIBLE_DEVICES={inst['p_gpus']} "
                 f"AFD_NVML_DEVICE_INDICES={inst['p_gpus']}; "
                 f"setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
                 f"-m sglang.launch_server --model-path {MODEL} --tp 2 "
                 f"--host {host} --port {inst['p_port']} --nccl-port {34000+inst['idx']*10} "
                 "--mem-fraction-static 0.85 --disable-cuda-graph "
                 "--disable-piecewise-cuda-graph --skip-server-warmup "
                 "--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {inst['bs_port']} "
                 f"--disaggregation-ib-device {inst['p_nic']} "
                 f"{_dvfs_flags(tier)} "
                 f"> {LOG_C}/pd{inst['idx']}_p.log 2>&1 < /dev/null &")
        if host == NODE1_IP:
            dexec_local(p_cmd)
        else:
            dexec_remote(p_cmd)
        time.sleep(2)
        # Decode
        d_cmd = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
                 f"CUDA_VISIBLE_DEVICES={inst['d_gpus']} "
                 f"AFD_NVML_DEVICE_INDICES={inst['d_gpus']}; "
                 f"setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
                 f"-m sglang.launch_server --model-path {MODEL} --tp 2 "
                 f"--host {host} --port {inst['d_port']} --nccl-port {34001+inst['idx']*10} "
                 "--mem-fraction-static 0.85 --disable-cuda-graph "
                 "--disable-piecewise-cuda-graph --skip-server-warmup "
                 "--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {inst['bs_port']} "
                 f"--disaggregation-ib-device {inst['d_nic']} "
                 f"{_dvfs_flags(tier)} "
                 f"> {LOG_C}/pd{inst['idx']}_d.log 2>&1 < /dev/null &")
        if host == NODE1_IP:
            dexec_local(d_cmd)
        else:
            dexec_remote(d_cmd)
        time.sleep(2)
    for inst in instances:
        for role, port in [("P", inst["p_port"]), ("D", inst["d_port"])]:
            if not wait_health(inst["host"], port, 400):
                log.error("  PD%d %s (%s:%d) failed", inst["idx"], role, inst["host"], port)
                return None
    log.info("  all %d PD pairs ready", len(instances))
    rc_parts = [f"setsid {PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {NODE1_IP} --port {ROUTER_PORT}"]
    for inst in instances:
        rc_parts.append(f"--prefill http://{inst['host']}:{inst['p_port']} {inst['bs_port']}")
    for inst in instances:
        rc_parts.append(f"--decode http://{inst['host']}:{inst['d_port']}")
    rc = " ".join(rc_parts) + f" > {LOG_C}/router.log 2>&1 < /dev/null &"
    dexec_local(rc)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  PD router failed")
        return None
    log.info("  router ready")
    return f"http://{NODE1_IP}:{ROUTER_PORT}"


def _afd_env(role, gpus, tp, attn_gpus, ffn_gpus):
    """Build export prefix for one AFD server.

    role in {PF,PA,DF,DA}. attn_gpus/ffn_gpus = physical GPU id lists (interleaved).
    Uses AFD_IPC_PEER_OFFSET so each attn rank pairs with the ffn rank on its
    neighbor GPU.
    """
    cvd = "0,1,2,3,4,5,6,7"
    base = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            "AFD_IPC_SYNC_MODE=ipc_event "
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            f"CUDA_VISIBLE_DEVICES={cvd} ")
    ucx = "28200" if role in ("PF", "PA") else "28300"
    sched = "68400" if role in ("PF", "PA") else "68500"
    # interleaved: attn on even-index gpus, ffn on odd-index; peer offset = +/-1
    if role in ("PF", "DF"):  # FFN
        nvml = ",".join(str(g) for g in ffn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET=-1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={ffn_gpus[0]};")
    else:  # ATTN
        nvml = ",".join(str(g) for g in attn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET=1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={attn_gpus[0]} "
                f"AFD_UCX_FFN_HOST=127.0.0.1;")


def _afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    max_running = 48 if ngpu == 4 else 512
    mem_frac = "0.88" if ngpu == 4 else "0.85"
    flags = (f"--model-path {MODEL} --tp {tp} --gpu-id-step {gpu_step} "
             "--afd-comm-backend ipc_cpp "
             f"--afd-micro-batch 2 --afd-dynamic-micro-batch --mem-fraction-static {mem_frac} "
             f"--max-running-requests {max_running} --skip-server-warmup "
             "--watchdog-timeout 600 "
             "--disable-cuda-graph --disable-piecewise-cuda-graph "
             "--afd-disagg-interleave-poll --disable-radix-cache "
             "--num-reserved-decode-tokens 512 "
             "--disaggregation-transfer-backend mooncake "
             f"--disaggregation-bootstrap-port {BS_PORT} "
             f"--disaggregation-ib-device {ib_dev} --enable-metrics")
    return flags + _afd_dvfs_flags(tier)


def start_pdaf(ngpu, tier=False):
    """PDAF with affine interleaved layout.

    4-card: tp=1, PA=GPU6(mlx5_5), PF=GPU7(mlx5_5) per node, step=1.
    8-card: tp=2, PA=GPU4,6(mlx5_4,mlx5_5), PF=GPU5,7 per node, step=2 (affine,
            PA uses 2 distinct NICs).
    """
    gpus = card_gpus(ngpu)
    if ngpu == 4:
        tp, step = 1, 1
        attn_gpus, ffn_gpus = [6], [7]
        attn_base, ffn_base = 6, 7
        ib_map = {"6": GPU_NIC[6], "7": GPU_NIC[7]}
    elif ngpu == 8:
        tp, step = 2, 2
        attn_gpus, ffn_gpus = [4, 6], [5, 7]   # interleaved, each attn rank own NIC
        attn_base, ffn_base = 4, 5
        ib_map = {str(g): GPU_NIC[g] for g in gpus}
    else:  # 16-card: tp=4, attn on even-index GPUs (4 NICs), ffn on odd-index
        tp, step = 4, 2
        attn_gpus, ffn_gpus = [0, 2, 4, 6], [1, 3, 5, 7]
        attn_base, ffn_base = 0, 1
        ib_map = {str(g): GPU_NIC[g] for g in gpus}
    write_ib_json(ib_map)
    ib_dev = IB_JSON_FILE
    log.info("PDAF affine %d-card: PA=%s PF=%s tp=%d step=%d tier=%s",
             ngpu, attn_gpus, ffn_gpus, tp, step, tier)
    cf = _afd_common(tp, ib_dev, step, tier, ngpu=ngpu)

    # node1 PF
    env = _afd_env("PF", gpus, tp, attn_gpus, ffn_gpus)
    _launch(NODE1_IP, f"{env} {PYTHON} -m sglang.launch_server --host {NODE1_IP} "
            f"--port {PF_PORT} --afd-perspective ffn --disaggregation-mode prefill "
            f"--base-gpu-id {ffn_base} {cf}", "pf")
    time.sleep(6)
    # node1 PA
    env = _afd_env("PA", gpus, tp, attn_gpus, ffn_gpus)
    _launch(NODE1_IP, f"{env} {PYTHON} -m sglang.launch_server --host {NODE1_IP} "
            f"--port {PA_PORT} --afd-perspective attn --disaggregation-mode prefill "
            f"--base-gpu-id {attn_base} {cf}", "pa")
    # node2 DF
    env = _afd_env("DF", gpus, tp, attn_gpus, ffn_gpus)
    _launch(NODE2_IP, f"{env} {PYTHON} -m sglang.launch_server --host {NODE2_IP} "
            f"--port {DF_PORT} --afd-perspective ffn --disaggregation-mode decode "
            f"--base-gpu-id {ffn_base} {cf}", "df")
    time.sleep(8)
    # node2 DA
    env = _afd_env("DA", gpus, tp, attn_gpus, ffn_gpus)
    _launch(NODE2_IP, f"{env} {PYTHON} -m sglang.launch_server --host {NODE2_IP} "
            f"--port {DA_PORT} --afd-perspective attn --disaggregation-mode decode "
            f"--base-gpu-id {attn_base} {cf}", "da")

    log.info("Waiting for PDAF servers...")
    for host, port, name, mi in [(NODE1_IP, PF_PORT, "PF", True),
                                 (NODE1_IP, PA_PORT, "PA", False),
                                 (NODE2_IP, DF_PORT, "DF", True),
                                 (NODE2_IP, DA_PORT, "DA", False)]:
        if not wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed", name, host, port)
            return None
        log.info("  %s ready", name)
    rc = (f"setsid {PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
          f"--prefill http://{NODE1_IP}:{PA_PORT} --decode http://{NODE2_IP}:{DA_PORT} "
          f"--host {NODE1_IP} --port {ROUTER_PORT} > {LOG_C}/router.log 2>&1 < /dev/null &")
    dexec_local(rc)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  router failed")
        return None
    log.info("  router ready")
    return f"http://{NODE1_IP}:{ROUTER_PORT}"


def start_pdaf_dp2(ngpu, tier=False):
    """PDAF with TP2×2DP: two independent AF-split pipelines in parallel.

    16-card layout (8 GPUs/node):
      Group0: PA0=GPU0,2(TP2,step=2) ↔ PF0=GPU1,3(TP2,step=2) on Node1
              DA0=GPU0,2(TP2,step=2) ↔ DF0=GPU1,3(TP2,step=2) on Node2
      Group1: PA1=GPU4,6(TP2,step=2) ↔ PF1=GPU5,7(TP2,step=2) on Node1
              DA1=GPU4,6(TP2,step=2) ↔ DF1=GPU5,7(TP2,step=2) on Node2
    Router round-robins across 2 prefill endpoints (PA0, PA1).
    """
    if ngpu != 16:
        log.error("pdaf_dp2 only supports 16-card (need 8 GPUs/node)")
        return None

    tp, step = 2, 2
    groups = [
        {"attn_gpus": [0, 2], "ffn_gpus": [1, 3],
         "attn_base": 0, "ffn_base": 1,
         "pa_port": 42010, "pf_port": 42110,
         "da_port": 42210, "df_port": 42310,
         "bs_port": 49999,
         "ucx_base_pf": "28200", "ucx_base_da": "28300",
         "sched_pf": "68400", "sched_da": "68500"},
        {"attn_gpus": [4, 6], "ffn_gpus": [5, 7],
         "attn_base": 4, "ffn_base": 5,
         "pa_port": 42410, "pf_port": 42510,
         "da_port": 42610, "df_port": 42710,
         "bs_port": 49899,
         "ucx_base_pf": "28400", "ucx_base_da": "28500",
         "sched_pf": "68600", "sched_da": "68700"},
    ]

    gpus = card_gpus(ngpu)
    ib_map = {str(g): GPU_NIC[g] for g in gpus}
    write_ib_json(ib_map)
    ib_dev = IB_JSON_FILE

    log.info("PDAF_DP2 16-card: 2 groups × TP2 AF-split, tier=%s", tier)

    for gi, grp in enumerate(groups):
        cf = _afd_common_dp2(tp, ib_dev, step, tier, grp["bs_port"])
        cvd = "0,1,2,3,4,5,6,7"

        # PF (FFN prefill)
        nvml_f = ",".join(str(g) for g in grp["ffn_gpus"])
        env_pf = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
                  "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
                  "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
                  "AFD_IPC_SYNC_MODE=ipc_event "
                  "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
                  "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
                  "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
                  f"CUDA_VISIBLE_DEVICES={cvd} "
                  f"AFD_UCX_BASE_PORT={grp['ucx_base_pf']} "
                  f"AFD_SCHED_PORT={grp['sched_pf']} "
                  f"AFD_IPC_PEER_OFFSET=-1 "
                  f"AFD_NVML_DEVICE_INDICES={nvml_f} "
                  f"AFD_NVML_DEVICE_INDEX={grp['ffn_gpus'][0]};")
        _launch(NODE1_IP,
                f"{env_pf} {PYTHON} -m sglang.launch_server --host {NODE1_IP} "
                f"--port {grp['pf_port']} --afd-perspective ffn "
                f"--disaggregation-mode prefill --base-gpu-id {grp['ffn_base']} {cf}",
                f"pf{gi}")
        time.sleep(4)

        # PA (Attention prefill)
        nvml_a = ",".join(str(g) for g in grp["attn_gpus"])
        env_pa = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
                  "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
                  "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
                  "AFD_IPC_SYNC_MODE=ipc_event "
                  "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
                  "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
                  "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
                  f"CUDA_VISIBLE_DEVICES={cvd} "
                  f"AFD_UCX_BASE_PORT={grp['ucx_base_pf']} "
                  f"AFD_SCHED_PORT={grp['sched_pf']} "
                  f"AFD_IPC_PEER_OFFSET=1 "
                  f"AFD_NVML_DEVICE_INDICES={nvml_a} "
                  f"AFD_NVML_DEVICE_INDEX={grp['attn_gpus'][0]} "
                  f"AFD_UCX_FFN_HOST=127.0.0.1;")
        _launch(NODE1_IP,
                f"{env_pa} {PYTHON} -m sglang.launch_server --host {NODE1_IP} "
                f"--port {grp['pa_port']} --afd-perspective attn "
                f"--disaggregation-mode prefill --base-gpu-id {grp['attn_base']} {cf}",
                f"pa{gi}")
        time.sleep(4)

        # DF (FFN decode)
        env_df = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
                  "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
                  "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
                  "AFD_IPC_SYNC_MODE=ipc_event "
                  "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
                  "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
                  "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
                  f"CUDA_VISIBLE_DEVICES={cvd} "
                  f"AFD_UCX_BASE_PORT={grp['ucx_base_da']} "
                  f"AFD_SCHED_PORT={grp['sched_da']} "
                  f"AFD_IPC_PEER_OFFSET=-1 "
                  f"AFD_NVML_DEVICE_INDICES={nvml_f} "
                  f"AFD_NVML_DEVICE_INDEX={grp['ffn_gpus'][0]};")
        _launch(NODE2_IP,
                f"{env_df} {PYTHON} -m sglang.launch_server --host {NODE2_IP} "
                f"--port {grp['df_port']} --afd-perspective ffn "
                f"--disaggregation-mode decode --base-gpu-id {grp['ffn_base']} {cf}",
                f"df{gi}")
        time.sleep(4)

        # DA (Attention decode)
        env_da = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
                  "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
                  "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
                  "AFD_IPC_SYNC_MODE=ipc_event "
                  "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
                  "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
                  "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
                  f"CUDA_VISIBLE_DEVICES={cvd} "
                  f"AFD_UCX_BASE_PORT={grp['ucx_base_da']} "
                  f"AFD_SCHED_PORT={grp['sched_da']} "
                  f"AFD_IPC_PEER_OFFSET=1 "
                  f"AFD_NVML_DEVICE_INDICES={nvml_a} "
                  f"AFD_NVML_DEVICE_INDEX={grp['attn_gpus'][0]} "
                  f"AFD_UCX_FFN_HOST=127.0.0.1;")
        _launch(NODE2_IP,
                f"{env_da} {PYTHON} -m sglang.launch_server --host {NODE2_IP} "
                f"--port {grp['da_port']} --afd-perspective attn "
                f"--disaggregation-mode decode --base-gpu-id {grp['attn_base']} {cf}",
                f"da{gi}")

    log.info("Waiting for PDAF_DP2 servers (8 processes)...")
    health_checks = []
    for gi, grp in enumerate(groups):
        health_checks += [
            (NODE1_IP, grp["pf_port"], f"PF{gi}", True),
            (NODE1_IP, grp["pa_port"], f"PA{gi}", False),
            (NODE2_IP, grp["df_port"], f"DF{gi}", True),
            (NODE2_IP, grp["da_port"], f"DA{gi}", False),
        ]
    for host, port, name, mi in health_checks:
        if not wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed", name, host, port)
            return None
        log.info("  %s ready", name)

    # Two-tier router: sub-routers (PD per group) + top-level round-robin
    sub_router_ports = [43500, 43600]
    for gi, grp in enumerate(groups):
        sub_rc = (f"setsid {PYTHON} -m sglang_router.launch_router "
                  f"--pd-disaggregation --mini-lb "
                  f"--prefill http://{NODE1_IP}:{grp['pa_port']} "
                  f"--decode http://{NODE2_IP}:{grp['da_port']} "
                  f"--host {NODE1_IP} --port {sub_router_ports[gi]} "
                  f"> {LOG_C}/router_g{gi}.log 2>&1 < /dev/null &")
        dexec_local(sub_rc)
    for gi in range(len(groups)):
        if not wait_health(NODE1_IP, sub_router_ports[gi], 60):
            log.error("  sub-router %d failed", gi)
            return None
        log.info("  sub-router %d ready (port %d)", gi, sub_router_ports[gi])

    # Top-level round-robin router
    worker_urls = " ".join(f"http://{NODE1_IP}:{p}" for p in sub_router_ports)
    top_rc = (f"setsid {PYTHON} -m sglang_router.launch_router "
              f"--host {NODE1_IP} --port {ROUTER_PORT} --policy round_robin "
              f"--worker-urls {worker_urls} > {LOG_C}/router.log 2>&1 < /dev/null &")
    dexec_local(top_rc)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  top router failed")
        return None
    log.info("  router ready (DP2, 2-tier)")
    return f"http://{NODE1_IP}:{ROUTER_PORT}"


def _afd_common_dp2(tp, ib_dev, gpu_step, tier, bs_port):
    """Common flags for pdaf_dp2 (same as _afd_common but with custom bs_port)."""
    max_running = 256
    mem_frac = "0.85"
    flags = (f"--model-path {MODEL} --tp {tp} --gpu-id-step {gpu_step} "
             "--afd-comm-backend ipc_cpp "
             f"--afd-micro-batch 2 --afd-dynamic-micro-batch --mem-fraction-static {mem_frac} "
             f"--max-running-requests {max_running} --skip-server-warmup "
             "--watchdog-timeout 600 "
             "--disable-cuda-graph --disable-piecewise-cuda-graph "
             "--afd-disagg-interleave-poll --disable-radix-cache "
             "--num-reserved-decode-tokens 512 "
             "--disaggregation-transfer-backend mooncake "
             f"--disaggregation-bootstrap-port {bs_port} "
             f"--disaggregation-ib-device {ib_dev} --enable-metrics")
    return flags + _afd_dvfs_flags(tier)


SCHEMES = {
    "native_dp": start_native_dp,
    "pd_dp": start_pd_dp,
    "pdaf": start_pdaf,
    "pdaf_dp2": start_pdaf_dp2,
}

# ============================================================
# Workload runner
# ============================================================

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
    e1s = get_energy_local(n1_gpus)
    e2s = get_energy_remote(n2_gpus)
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
    e1e = get_energy_local(n1_gpus)
    e2e = get_energy_remote(n2_gpus)
    energy_n1_j = sum((e1e.get(i, 0) - e1s.get(i, 0)) / 1000.0 for i in n1_gpus)
    energy_n2_j = sum((e2e.get(i, 0) - e2s.get(i, 0)) / 1000.0 for i in n2_gpus)
    total_energy_j = energy_n1_j + energy_n2_j

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    if not ok:
        return {"status": "FAIL", "failed": len(fail)}

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
        "energy_node1_j": round(energy_n1_j, 1),
        "energy_node2_j": round(energy_n2_j, 1),
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens else 0,
        "slo_violation_rate": round(slo_rate, 1),
        "ttft_violations": n_ttft_viol, "tpot_violations": n_tpot_viol,
    }


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
    log.info("  %s (%d reqs, run_window=%ds)", wl_key, len(reqs), run_s)
    summary = asyncio.run(run_workload(reqs, url + "/generate", n1_gpus, n2_gpus, run_s))
    if summary.get("status") == "PASS":
        log.info("  Thpt=%.1f tok/s | TTFT=%.1fms | TPOT=%.1fms | E=%.0fJ (%.1f mJ/tok) | SLO=%.1f%%",
                 summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                 summary["tpot_avg_ms"], summary["total_energy_j"],
                 summary["energy_per_token_mj"], summary["slo_violation_rate"])
    else:
        log.error("  FAIL: %s", summary)
    return wl_key, summary

# ============================================================
# Main
# ============================================================

def run_deploy(scheme, ngpu, tier, scenarios, qps_list, max_run_s):
    gpus = card_gpus(ngpu)
    mode = "tier" if tier else "baseline"
    full_name = f"{scheme}_{mode}"
    log.info("=" * 70)
    log.info("DEPLOY: %s [%d-card cross-node, GPUs %s/node]", full_name, ngpu, gpus)
    log.info("=" * 70)

    cleanup_all()
    url = SCHEMES[scheme](ngpu, tier)
    if url is None:
        log.error("  %s deployment FAILED", full_name)
        cleanup_all()
        return full_name, {"__status__": "DEPLOY_FAILED"}

    # All schemes (Native/PD/PDAF) start from a uniform 1410MHz baseline.
    # baseline mode: stays locked at 1410 for the whole run.
    # tier mode: DVFS adjusts frequency down from this 1410 starting point
    #            (instead of the GPU hardware-default 1140MHz applications clock).
    lock_freq_both(gpus, MAX_GPU_FREQ)

    log.info("Warmup...")
    if not test_generate(url):
        log.error("  warmup failed")
        unlock_freq_both(gpus)
        cleanup_all()
        return full_name, {"__status__": "WARMUP_FAILED"}
    time.sleep(3)

    deploy_results = {}
    for scenario in scenarios:
        for qps in qps_list:
            log.info("-" * 50)
            res = run_one_workload(url, scenario, qps, gpus, gpus, max_run_s)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

    unlock_freq_both(gpus)
    cleanup_all()
    return full_name, deploy_results


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpu", type=int, required=True, choices=[4, 8, 16])
    parser.add_argument("--deploy", default="all",
                        help="Comma-sep: native_dp,pd_dp,pdaf or 'all'")
    parser.add_argument("--mode", default="all",
                        help="Comma-sep: baseline,tier or 'all'")
    parser.add_argument("--scenario", default="all",
                        help="Comma-sep: chatbot,qa,rag,summary or 'all'")
    parser.add_argument("--qps", default=None, help="Comma-sep QPS list")
    parser.add_argument("--max-run-s", type=int, default=400)
    args = parser.parse_args()

    schemes = list(SCHEMES.keys()) if args.deploy == "all" else args.deploy.split(",")
    modes = ["baseline", "tier"] if args.mode == "all" else args.mode.split(",")
    scenarios = SCENARIOS if args.scenario == "all" else args.scenario.split(",")
    if args.qps:
        qps_list = [int(q) for q in args.qps.split(",")]
    elif args.ngpu == 4:
        qps_list = [1, 2, 3]
    else:
        qps_list = [2, 4, 6, 8, 12, 16]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for scheme in schemes:
        if scheme not in SCHEMES:
            log.error("unknown scheme: %s", scheme)
            continue
        for mode in modes:
            tier = (mode == "tier")
            full_name, res = run_deploy(scheme, args.ngpu, tier, scenarios,
                                        qps_list, args.max_run_s)
            all_results[full_name] = res
            ts = time.strftime("%Y%m%d_%H%M%S")
            out_file = RESULTS_DIR / f"scal_{args.ngpu}card_{ts}.json"
            payload = {"meta": {"node1": NODE1_IP, "node2": NODE2_IP, "model": MODEL,
                                "ngpu_total": args.ngpu, "gpus_per_node": card_gpus(args.ngpu),
                                "schemes": schemes, "modes": modes,
                                "scenarios": scenarios, "qps": qps_list},
                       "results": all_results}
            with open(out_file, "w") as f:
                json.dump(payload, f, indent=2)
            log.info("Results saved (incremental): %s", out_file)

    print("\n" + "=" * 104)
    print(f"  CROSS-NODE SCALABILITY ({args.ngpu}-card, Qwen3-32B)")
    print("=" * 104)
    hdr = (f"{'Deploy':<22} {'Workload':<16} {'Thpt':>8} {'TTFT':>8} {'TPOT':>8} "
           f"{'E_tot':>9} {'mJ/tok':>8} {'SLO%':>6}")
    print(hdr)
    print("-" * 104)
    for dep, wl in all_results.items():
        if "__status__" in wl:
            print(f"{dep:<22} {wl['__status__']}")
            continue
        for w, m in wl.items():
            if m.get("status") != "PASS":
                print(f"{dep:<22} {w:<16} FAIL")
                continue
            print(f"{dep:<22} {w:<16} {m['throughput_tok_s']:>8.1f} "
                  f"{m['ttft_proc_avg_ms']:>8.1f} {m['tpot_avg_ms']:>8.1f} "
                  f"{m['total_energy_j']:>9.0f} {m['energy_per_token_mj']:>8.1f} "
                  f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
