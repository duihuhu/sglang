#!/usr/bin/env python3
"""PDAF with P(TP=2) + 3x decode instances D(TP=2), conv benchmark.

Idea (from the observation that decode is the bottleneck): shrink prefill to
4 cards (TP=2) and spend the freed 4 cards on a third decode instance, giving
50% more decode concurrency than the 2Decode layout.

Layout (16 cards):
  node1: P  = PA[0,2] + PF[1,3]  (TP=2, prefill)
         D0 = DA[4,6] + DF[5,7]  (TP=2, decode)
  node2: D1 = DA[0,2] + DF[1,3]  (TP=2, decode)
         D2 = DA[4,6] + DF[5,7]  (TP=2, decode)

Prefill drops TP4->TP2, but conv TTFT (~280ms) has huge headroom under the
5000ms SLO, so trading prefill TP for a 3rd decode instance is favorable.

Router: three PD MiniLB sub-routers (each = prefill PA + one decode DA) behind a
top-level round-robin router, so decode traffic splits deterministically 3 ways.

SLO: TTFT=5000ms, TPOT=300ms, M=1.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pdaf_3d")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_macro_benchmark as RMB

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

# Ports. P + D0 live on node1; D1 + D2 live on node2.
PA_PORT = 42010
PF_PORT = 42011
D0A_PORT = 42020
D0F_PORT = 42021
D1A_PORT = 42030
D1F_PORT = 42031
D2A_PORT = 42040
D2F_PORT = 42041
ROUTER_PORT = 42000
SUB_ROUTER_PORTS = [42050, 42051, 42052]
BS_PORT = 49999


def _afd_comm_opt_env():
    parts = []
    for key in ("AFD_FUSED_PIPELINE", "AFD_FUSED_COMM_STREAM", "AFD_GPU_ONLY_IPC"):
        val = os.environ.get(key)
        if val is not None:
            parts.append(f"{key}={val}")
    return (" ".join(parts) + " ") if parts else ""


def _afd_env(role, attn_gpus, ffn_gpus, ucx_port, sched_port):
    """Build env for one PDAF server. attn+ffn of an instance are co-located."""
    cvd = "0,1,2,3,4,5,6,7"
    ipc_sync = os.environ.get("AFD_IPC_SYNC_MODE", "ipc_event")
    dvfs_log = os.environ.get("AFD_DVFS_DECISION_LOG")
    dvfs_log_env = f"AFD_DVFS_DECISION_LOG={dvfs_log} " if dvfs_log else ""
    base = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            f"AFD_IPC_SYNC_MODE={ipc_sync} "
            f"{_afd_comm_opt_env().strip()} "
            f"{dvfs_log_env}"
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            f"CUDA_VISIBLE_DEVICES={cvd} ")
    if role == "ffn":
        nvml = ",".join(str(g) for g in ffn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
                f"AFD_IPC_PEER_OFFSET=-1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={ffn_gpus[0]};")
    else:
        nvml = ",".join(str(g) for g in attn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
                f"AFD_IPC_PEER_OFFSET=1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={attn_gpus[0]} "
                f"AFD_UCX_FFN_HOST=127.0.0.1;")


def _afd_common(tp, ib_dev, gpu_step, tier, bs_port):
    flags = (f"--model-path {RMB.MODEL} --tp {tp} --gpu-id-step {gpu_step} "
             "--afd-comm-backend ipc_cpp "
             "--afd-micro-batch 1 --mem-fraction-static 0.85 "
             "--max-running-requests 512 --skip-server-warmup "
             "--watchdog-timeout 600 "
             "--disable-cuda-graph --disable-piecewise-cuda-graph "
             "--afd-disagg-interleave-poll --disable-radix-cache "
             "--num-reserved-decode-tokens 512 "
             "--disaggregation-transfer-backend mooncake "
             f"--disaggregation-bootstrap-port {bs_port} "
             f"--disaggregation-ib-device {ib_dev} --enable-metrics")
    if tier:
        flags += (" --afd-dvfs-enabled "
                  f"--afd-energy-model-dir {RMB.ENERGY_MODEL_DIR_V2} "
                  f"--afd-ttft-slo-ms {int(RMB.TTFT_SLO_MS)} "
                  f"--afd-tpot-slo-us {int(RMB.TPOT_SLO_MS * 1000)} "
                  "--afd-dvfs-idle-lock")
        if os.environ.get("AFD_DVFS_ONLINE_CALIBRATION") == "1":
            flags += " --afd-dvfs-online-calibration"
        ema = os.environ.get("AFD_DVFS_CALIBRATION_EMA")
        if ema:
            flags += f" --afd-dvfs-calibration-ema {ema}"
    return flags


def start_pdaf_3decode(tier=False):
    """Start PDAF: P(TP=2) on node1 + 3 decode instances (TP=2 each)."""
    tp = 2
    step = 2

    # node1: prefill + decode0
    p_attn, p_ffn = [0, 2], [1, 3]
    d0_attn, d0_ffn = [4, 6], [5, 7]
    # node2: decode1 + decode2
    d1_attn, d1_ffn = [0, 2], [1, 3]
    d2_attn, d2_ffn = [4, 6], [5, 7]

    log.info("PDAF 3xDecode: node1 P[A:%s F:%s]+D0[A:%s F:%s] | node2 D1[A:%s F:%s]+D2[A:%s F:%s] | TP=%d tier=%s",
             p_attn, p_ffn, d0_attn, d0_ffn, d1_attn, d1_ffn, d2_attn, d2_ffn, tp, tier)

    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    ib_map = {str(g): RMB.GPU_NIC[g] for g in all_gpus}
    RMB.write_ib_json(ib_map)
    ib_dev = RMB.IB_JSON_FILE

    cf = _afd_common(tp, ib_dev, step, tier, BS_PORT)

    # --- node1: Prefill (TP=2, GPU0-3) ---
    env = _afd_env("ffn", p_attn, p_ffn, ucx_port=28200, sched_port=68400)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {PF_PORT} --afd-perspective ffn --disaggregation-mode prefill "
                f"--base-gpu-id {p_ffn[0]} {cf}", "pf")
    time.sleep(6)
    env = _afd_env("attn", p_attn, p_ffn, ucx_port=28200, sched_port=68400)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {PA_PORT} --afd-perspective attn --disaggregation-mode prefill "
                f"--base-gpu-id {p_attn[0]} {cf}", "pa")

    # --- node1: Decode0 (TP=2, GPU4-7) ---
    time.sleep(5)
    env = _afd_env("ffn", d0_attn, d0_ffn, ucx_port=28300, sched_port=68500)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {D0F_PORT} --afd-perspective ffn --disaggregation-mode decode "
                f"--base-gpu-id {d0_ffn[0]} {cf}", "d0f")
    time.sleep(8)
    env = _afd_env("attn", d0_attn, d0_ffn, ucx_port=28300, sched_port=68500)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {D0A_PORT} --afd-perspective attn --disaggregation-mode decode "
                f"--base-gpu-id {d0_attn[0]} {cf}", "d0a")

    # --- node2: Decode1 (TP=2, GPU0-3) ---
    time.sleep(5)
    env = _afd_env("ffn", d1_attn, d1_ffn, ucx_port=28200, sched_port=68400)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {D1F_PORT} --afd-perspective ffn --disaggregation-mode decode "
                f"--base-gpu-id {d1_ffn[0]} {cf}", "d1f")
    time.sleep(8)
    env = _afd_env("attn", d1_attn, d1_ffn, ucx_port=28200, sched_port=68400)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {D1A_PORT} --afd-perspective attn --disaggregation-mode decode "
                f"--base-gpu-id {d1_attn[0]} {cf}", "d1a")

    # --- node2: Decode2 (TP=2, GPU4-7) ---
    time.sleep(5)
    env = _afd_env("ffn", d2_attn, d2_ffn, ucx_port=28300, sched_port=68500)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {D2F_PORT} --afd-perspective ffn --disaggregation-mode decode "
                f"--base-gpu-id {d2_ffn[0]} {cf}", "d2f")
    time.sleep(8)
    env = _afd_env("attn", d2_attn, d2_ffn, ucx_port=28300, sched_port=68500)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {D2A_PORT} --afd-perspective attn --disaggregation-mode decode "
                f"--base-gpu-id {d2_attn[0]} {cf}", "d2a")

    log.info("Waiting for all PDAF servers...")
    for host, port, name, mi in [
        (RMB.NODE1_IP, PF_PORT, "PF", True),
        (RMB.NODE1_IP, PA_PORT, "PA", False),
        (RMB.NODE1_IP, D0F_PORT, "D0F", True),
        (RMB.NODE1_IP, D0A_PORT, "D0A", False),
        (RMB.NODE2_IP, D1F_PORT, "D1F", True),
        (RMB.NODE2_IP, D1A_PORT, "D1A", False),
        (RMB.NODE2_IP, D2F_PORT, "D2F", True),
        (RMB.NODE2_IP, D2A_PORT, "D2A", False),
    ]:
        if not RMB.wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed", name, host, port)
            return None
        log.info("  %s ready", name)

    # Router: three PD sub-routers (prefill PA + one decode DA each) behind a
    # top-level round-robin router, for deterministic 3-way decode split.
    sub_specs = [
        (SUB_ROUTER_PORTS[0], RMB.NODE1_IP, D0A_PORT, "router_d0"),
        (SUB_ROUTER_PORTS[1], RMB.NODE2_IP, D1A_PORT, "router_d1"),
        (SUB_ROUTER_PORTS[2], RMB.NODE2_IP, D2A_PORT, "router_d2"),
    ]
    for sub_port, da_host, da_port, log_name in sub_specs:
        rc = (f"setsid {RMB.PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
              f"--prefill http://{RMB.NODE1_IP}:{PA_PORT} "
              f"--decode http://{da_host}:{da_port} "
              f"--host {RMB.NODE1_IP} --port {sub_port} "
              f"> {RMB.LOG_C}/{log_name}.log 2>&1 < /dev/null &")
        RMB.dexec_local(rc)
        if not RMB.wait_health(RMB.NODE1_IP, sub_port, 60):
            log.error("  sub-router %s failed", log_name)
            return None
        log.info("  sub-router %s ready", log_name)

    worker_urls = " ".join(f"http://{RMB.NODE1_IP}:{p}" for p, *_ in sub_specs)
    rc = (f"setsid {RMB.PYTHON} -m sglang_router.launch_router "
          f"--host {RMB.NODE1_IP} --port {ROUTER_PORT} --policy round_robin "
          f"--worker-urls {worker_urls} > {RMB.LOG_C}/router.log 2>&1 < /dev/null &")
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, ROUTER_PORT, 60):
        log.error("  top-level router failed")
        return None
    log.info("  top-level round-robin router ready")
    return f"http://{RMB.NODE1_IP}:{ROUTER_PORT}"
