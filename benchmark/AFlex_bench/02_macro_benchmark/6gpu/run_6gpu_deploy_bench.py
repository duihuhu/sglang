#!/usr/bin/env python3
"""6-GPU deployment benchmark for heterogeneous PDAF topology comparison.

Tests 4 deployment topologies (each with/without Tier DVFS) on 6 A800 GPUs:
  1) native_dp6      : 6x TP=1 independent instances, round-robin
  2) pd_dp3          : PD disagg DP=3, 3x (P-TP1 + D-TP1)
  3) pdaf_6g_1p2d    : PDAF decode-heavy: 1PA(TP1)+1PF(TP1) on 2 GPU, 2DA(TP1)+2DF(TP1) on 4 GPU (M=2)
  4) pdaf_6g_2p1d    : PDAF prefill-heavy: 2PA(TP1)+2PF(TP1) on 4 GPU, 1DA(TP1)+1DF(TP1) on 2 GPU (M=2)

Usage:
    python run_6gpu_deploy_bench.py --deploys native_dp6,native_dp6_tier,pd_dp3,pd_dp3_tier,pdaf_6g_1p2d,pdaf_6g_1p2d_tier,pdaf_6g_2p1d,pdaf_6g_2p1d_tier
    python run_6gpu_deploy_bench.py  # runs all 8
"""
import argparse
import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import run_fixed_qps_bench as B

log = logging.getLogger("6gpu_bench")

PYTHON = B.PYTHON
MODEL = B.MODEL
HERE = Path(__file__).resolve().parent
SCRIPTS = HERE.parent / "scripts"

ROUTER_PORT = 54000
PA_PORT, PF_PORT = 54010, 54011
DA_PORT, DF_PORT = 54020, 54021
BOOTSTRAP_PORT = 49800
UCX_P, UCX_D = 27200, 27300
SCHED_P, SCHED_D = 67400, 67500

# Use GPU 0-5 (leave 6,7 free)
GPUS_6 = list(range(6))

DVFS_TTFT_SLO_MS = 5000
DVFS_TPOT_SLO_US = 200000

WORKLOAD_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads")
WORKLOADS = [
    ("Code Medium", WORKLOAD_DIR / "workload_azure_code_medium_real.jsonl"),
    ("Conv Light", WORKLOAD_DIR / "workload_azure_conv_light_real.jsonl"),
    ("Conv Medium", WORKLOAD_DIR / "workload_azure_conv_medium_real.jsonl"),
    ("Conv Heavy", WORKLOAD_DIR / "workload_azure_conv_heavy_real.jsonl"),
]

ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT] + \
    [54200 + i * 10 for i in range(6)] + \
    [PF_PORT + 100 + i * 2 for i in range(3)] + \
    [DF_PORT + 100 + i * 2 for i in range(3)]


def kill_servers():
    for port in ALL_PORTS:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True)
        for m in re.finditer(r'pid=(\d+)', r.stdout):
            subprocess.run(["kill", "-9", m.group(1)], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang_router"], capture_output=True)
    time.sleep(5)


def reset_freq():
    for idx in GPUS_6:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def lock_max_freq():
    for idx in GPUS_6:
        subprocess.run(["nvidia-smi", "-lgc", "1410,1410", "-i", str(idx)],
                       capture_output=True)


def _popen(name, cmd, env, log_dir, procs):
    fh = open(log_dir / f"{name}.log", "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    procs.append((name, p, fh))
    log.info("  Started %s (CVD=%s)", name, env.get("CUDA_VISIBLE_DEVICES", "?"))
    return p


def _dvfs_args():
    return ["--afd-dvfs-enabled",
            "--afd-energy-model-dir", B.ENERGY_MODEL_DIR,
            "--afd-energy-model-v3-dir", B.ENERGY_MODEL_V3_DIR,
            "--afd-ttft-slo-ms", str(DVFS_TTFT_SLO_MS),
            "--afd-tpot-slo-us", str(DVFS_TPOT_SLO_US)]


def _unified_dvfs_args():
    return ["--dvfs-enabled",
            "--dvfs-energy-model-dir", B.ENERGY_MODEL_DIR,
            "--dvfs-ttft-slo-ms", str(DVFS_TTFT_SLO_MS),
            "--dvfs-tpot-slo-us", str(DVFS_TPOT_SLO_US)]


_PD_COMMON_6G = [
    "--mem-fraction-static", "0.85",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-ib-device", "mlx5_4",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--disable-radix-cache",
]

_AFD_EXTRA_6G = [
    "--afd-comm-backend", "ipc_cpp",
    "--mem-fraction-static", "0.85",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-ib-device", "mlx5_4",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--afd-disagg-interleave-poll",
    "--disable-radix-cache",
]


# ======== NATIVE DP6 ========
def start_native_dp6(tier, log_dir):
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    worker_urls = []

    for i in range(6):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(i)
        port = 54200 + i * 10
        dvfs = []
        if tier:
            env["AFD_NVML_DEVICE_INDICES"] = str(i)
            env["AFD_NVML_DEVICE_INDEX"] = str(i)
            dvfs = _unified_dvfs_args()
        cmd = [PYTHON, "-m", "sglang.launch_server",
               "--model-path", MODEL, "--tp", "1",
               "--host", "127.0.0.1", "--port", str(port),
               "--mem-fraction-static", "0.85",
               "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
               "--disable-radix-cache"] + dvfs
        _popen(f"inst{i}", cmd, env, log_dir, procs)
        worker_urls.append(f"http://127.0.0.1:{port}")

    for i in range(6):
        if not B.wait_port("127.0.0.1", 54200 + i * 10, 300):
            log.error("DP6 instance %d failed", i)
            B.cleanup_procs(procs)
            return None
    # Router
    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
           "--policy", "round_robin",
           "--worker-urls"] + worker_urls
    rf = open(log_dir / "router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("DP6 router failed")
        B.cleanup_procs(procs)
        return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("Native DP6 %s ready, warming up...", "(Tier)" if tier else "")
    B.warmup(url)
    return procs, url


# ======== PD DP3 ========
def start_pd_dp3(tier, log_dir):
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    prefill_specs, decode_urls = [], []
    instances = [
        {"p_cvd": "0", "d_cvd": "1"},
        {"p_cvd": "2", "d_cvd": "3"},
        {"p_cvd": "4", "d_cvd": "5"},
    ]

    for i, inst in enumerate(instances):
        pf_port = PF_PORT + 100 + i * 2
        df_port = DF_PORT + 100 + i * 2
        bport = BOOTSTRAP_PORT + 1 + i
        common = _PD_COMMON_6G + ["--disaggregation-bootstrap-port", str(bport)]
        dvfs = _unified_dvfs_args() if tier else []

        env_p = env_base.copy()
        env_p["CUDA_VISIBLE_DEVICES"] = inst["p_cvd"]
        if tier:
            env_p["AFD_NVML_DEVICE_INDICES"] = inst["p_cvd"]
            env_p["AFD_NVML_DEVICE_INDEX"] = inst["p_cvd"]
        p_cmd = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", "1",
                 "--host", "127.0.0.1", "--port", str(pf_port),
                 "--disaggregation-mode", "prefill"] + common + dvfs
        _popen(f"prefill{i}", p_cmd, env_p, log_dir, procs)

        env_d = env_base.copy()
        env_d["CUDA_VISIBLE_DEVICES"] = inst["d_cvd"]
        if tier:
            env_d["AFD_NVML_DEVICE_INDICES"] = inst["d_cvd"]
            env_d["AFD_NVML_DEVICE_INDEX"] = inst["d_cvd"]
        d_cmd = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", "1",
                 "--host", "127.0.0.1", "--port", str(df_port),
                 "--disaggregation-mode", "decode"] + common + dvfs
        _popen(f"decode{i}", d_cmd, env_d, log_dir, procs)

        prefill_specs.append((f"http://127.0.0.1:{pf_port}", bport))
        decode_urls.append(f"http://127.0.0.1:{df_port}")

    for url, _ in prefill_specs:
        if not B.wait_port("127.0.0.1", int(url.rsplit(":", 1)[1]), timeout=300):
            log.error("PD-DP3 prefill failed"); B.cleanup_procs(procs); return None
    for url in decode_urls:
        if not B.wait_port("127.0.0.1", int(url.rsplit(":", 1)[1]), timeout=300):
            log.error("PD-DP3 decode failed"); B.cleanup_procs(procs); return None

    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--pd-disaggregation",
           "--policy", "round_robin",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    for url, bport in prefill_specs:
        cmd += ["--prefill", url, str(bport)]
    for url in decode_urls:
        cmd += ["--decode", url]
    rf = open(log_dir / "router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("PD-DP3 router failed"); B.cleanup_procs(procs); return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PD DP3 %s ready, warming up...", "(Tier)" if tier else "")
    B.warmup(url)
    return procs, url


# ======== PDAF 6G ========
def _afd_env(env_base, cvd, ucx_base, sched_port, peer_device, nvml_indices=None,
             ffn_host=None):
    env = env_base.copy()
    env["CUDA_VISIBLE_DEVICES"] = cvd
    env["AFD_UCX_BASE_PORT"] = str(ucx_base)
    env["AFD_SCHED_PORT"] = str(sched_port)
    env["AFD_IPC_SYNC_MODE"] = "ipc_event"
    env["AFD_IPC_PEER_DEVICE"] = str(peer_device)
    env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
    env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
    if nvml_indices:
        env["AFD_NVML_DEVICE_INDICES"] = ",".join(str(g) for g in nvml_indices)
        env["AFD_NVML_DEVICE_INDEX"] = str(nvml_indices[0])
    if ffn_host:
        env["AFD_UCX_FFN_HOST"] = ffn_host
    return env


def start_pdaf_6g_1p2d(tier, log_dir):
    """PDAF decode-heavy: 1PA(TP1)+1PF(TP1) on GPU 0,1; 2DA(TP1)+2DF(TP1) on GPU 2,3,4,5.

    Layout:
      Prefill: PF(ffn, TP=1, base=0) + PA(attn, TP=1, base=1) sharing CVD=0,1
      Decode:  DF(ffn, TP=2, base=0) + DA(attn, TP=2, base=2) sharing CVD=2,3,4,5
    """
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"
    if tier:
        env_base["AFD_DVFS_DECISION_LOG"] = str(log_dir / "dvfs_{persp}_{disagg}.jsonl")

    micro_batch = 2
    dvfs = _dvfs_args() if tier else []

    # Prefill: PF(ffn, TP1, base=0) + PA(attn, TP1, base=1) on CVD=0,1
    env_pf = _afd_env(env_base, "0,1", UCX_P, SCHED_P, peer_device=1, nvml_indices=[0])
    pf_cmd = [PYTHON, "-m", "sglang.launch_server",
              "--model-path", MODEL, "--tp", "1",
              "--host", "127.0.0.1", "--port", str(PF_PORT),
              "--afd-perspective", "ffn", "--disaggregation-mode", "prefill",
              "--base-gpu-id", "0", "--afd-micro-batch", str(micro_batch),
              "--afd-dynamic-micro-batch",
              "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)] + _AFD_EXTRA_6G + dvfs
    _popen("pf", pf_cmd, env_pf, log_dir, procs)
    time.sleep(2)

    env_pa = _afd_env(env_base, "0,1", UCX_P, SCHED_P, peer_device=0,
                      ffn_host="127.0.0.1", nvml_indices=[1])
    pa_cmd = [PYTHON, "-m", "sglang.launch_server",
              "--model-path", MODEL, "--tp", "1",
              "--host", "127.0.0.1", "--port", str(PA_PORT),
              "--afd-perspective", "attn", "--disaggregation-mode", "prefill",
              "--base-gpu-id", "1", "--afd-micro-batch", str(micro_batch),
              "--afd-dynamic-micro-batch",
              "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)] + _AFD_EXTRA_6G + dvfs
    _popen("pa", pa_cmd, env_pa, log_dir, procs)

    if not B.wait_port("127.0.0.1", PF_PORT, 300) or \
       not B.wait_port("127.0.0.1", PA_PORT, 300):
        log.error("PDAF 1P2D prefill failed"); B.cleanup_procs(procs); return None

    # Decode: DF(ffn, TP2, base=0) + DA(attn, TP2, base=2) on CVD=2,3,4,5
    env_df = _afd_env(env_base, "2,3,4,5", UCX_D, SCHED_D, peer_device=2, nvml_indices=[2, 3])
    df_cmd = [PYTHON, "-m", "sglang.launch_server",
              "--model-path", MODEL, "--tp", "2",
              "--host", "127.0.0.1", "--port", str(DF_PORT),
              "--afd-perspective", "ffn", "--disaggregation-mode", "decode",
              "--base-gpu-id", "0", "--afd-micro-batch", str(micro_batch),
              "--afd-dynamic-micro-batch",
              "--afd-attn-tp", "2",
              "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)] + _AFD_EXTRA_6G + dvfs
    _popen("df", df_cmd, env_df, log_dir, procs)
    time.sleep(5)

    env_da = _afd_env(env_base, "2,3,4,5", UCX_D, SCHED_D, peer_device=0,
                      ffn_host="127.0.0.1", nvml_indices=[4, 5])
    da_cmd = [PYTHON, "-m", "sglang.launch_server",
              "--model-path", MODEL, "--tp", "2",
              "--host", "127.0.0.1", "--port", str(DA_PORT),
              "--afd-perspective", "attn", "--disaggregation-mode", "decode",
              "--base-gpu-id", "2", "--afd-micro-batch", str(micro_batch),
              "--afd-dynamic-micro-batch",
              "--afd-ffn-tp", "2",
              "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)] + _AFD_EXTRA_6G + dvfs
    _popen("da", da_cmd, env_da, log_dir, procs)

    if not B.wait_port("127.0.0.1", DA_PORT, 300) or \
       not B.wait_port("127.0.0.1", DF_PORT, 300):
        log.error("PDAF 1P2D decode failed"); B.cleanup_procs(procs); return None

    time.sleep(5)
    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--pd-disaggregation", "--mini-lb",
           "--prefill", f"http://127.0.0.1:{PA_PORT}",
           "--decode", f"http://127.0.0.1:{DA_PORT}",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    rf = open(log_dir / "router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("PDAF 1P2D router failed"); B.cleanup_procs(procs); return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PDAF 1P+2D %s ready, warming up...", "(Tier)" if tier else "")
    B.warmup(url)
    return procs, url


def start_pdaf_6g_2p1d(tier, log_dir):
    """PDAF prefill-heavy: 2PA(TP2)+2PF(TP2) on GPU 0,1,2,3; 1DA(TP1)+1DF(TP1) on GPU 4,5.

    Layout:
      Prefill: PF(ffn, TP=2, base=0) + PA(attn, TP=2, base=2) sharing CVD=0,1,2,3
      Decode:  DF(ffn, TP=1, base=0) + DA(attn, TP=1, base=1) sharing CVD=4,5
    """
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"
    if tier:
        env_base["AFD_DVFS_DECISION_LOG"] = str(log_dir / "dvfs_{persp}_{disagg}.jsonl")

    micro_batch = 2
    dvfs = _dvfs_args() if tier else []

    # Prefill: PF(ffn, TP2, base=0) + PA(attn, TP2, base=2) on CVD=0,1,2,3
    env_pf = _afd_env(env_base, "0,1,2,3", UCX_P, SCHED_P, peer_device=2,
                      nvml_indices=[0, 1])
    pf_cmd = [PYTHON, "-m", "sglang.launch_server",
              "--model-path", MODEL, "--tp", "2",
              "--host", "127.0.0.1", "--port", str(PF_PORT),
              "--afd-perspective", "ffn", "--disaggregation-mode", "prefill",
              "--base-gpu-id", "0", "--afd-micro-batch", str(micro_batch),
              "--afd-dynamic-micro-batch",
              "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)] + _AFD_EXTRA_6G + dvfs
    _popen("pf", pf_cmd, env_pf, log_dir, procs)
    time.sleep(2)

    env_pa = _afd_env(env_base, "0,1,2,3", UCX_P, SCHED_P, peer_device=0,
                      ffn_host="127.0.0.1", nvml_indices=[2, 3])
    pa_cmd = [PYTHON, "-m", "sglang.launch_server",
              "--model-path", MODEL, "--tp", "2",
              "--host", "127.0.0.1", "--port", str(PA_PORT),
              "--afd-perspective", "attn", "--disaggregation-mode", "prefill",
              "--base-gpu-id", "2", "--afd-micro-batch", str(micro_batch),
              "--afd-dynamic-micro-batch",
              "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)] + _AFD_EXTRA_6G + dvfs
    _popen("pa", pa_cmd, env_pa, log_dir, procs)

    if not B.wait_port("127.0.0.1", PF_PORT, 300) or \
       not B.wait_port("127.0.0.1", PA_PORT, 300):
        log.error("PDAF 2P1D prefill failed"); B.cleanup_procs(procs); return None

    # Decode: DF(ffn, TP1, base=0) + DA(attn, TP1, base=1) on CVD=4,5
    env_df = _afd_env(env_base, "4,5", UCX_D, SCHED_D, peer_device=1, nvml_indices=[4])
    df_cmd = [PYTHON, "-m", "sglang.launch_server",
              "--model-path", MODEL, "--tp", "1",
              "--host", "127.0.0.1", "--port", str(DF_PORT),
              "--afd-perspective", "ffn", "--disaggregation-mode", "decode",
              "--base-gpu-id", "0", "--afd-micro-batch", str(micro_batch),
              "--afd-dynamic-micro-batch",
              "--afd-attn-tp", "1",
              "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)] + _AFD_EXTRA_6G + dvfs
    _popen("df", df_cmd, env_df, log_dir, procs)
    time.sleep(5)

    env_da = _afd_env(env_base, "4,5", UCX_D, SCHED_D, peer_device=0,
                      ffn_host="127.0.0.1", nvml_indices=[5])
    da_cmd = [PYTHON, "-m", "sglang.launch_server",
              "--model-path", MODEL, "--tp", "1",
              "--host", "127.0.0.1", "--port", str(DA_PORT),
              "--afd-perspective", "attn", "--disaggregation-mode", "decode",
              "--base-gpu-id", "1", "--afd-micro-batch", str(micro_batch),
              "--afd-dynamic-micro-batch",
              "--afd-ffn-tp", "1",
              "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT)] + _AFD_EXTRA_6G + dvfs
    _popen("da", da_cmd, env_da, log_dir, procs)

    if not B.wait_port("127.0.0.1", DA_PORT, 300) or \
       not B.wait_port("127.0.0.1", DF_PORT, 300):
        log.error("PDAF 2P1D decode failed"); B.cleanup_procs(procs); return None

    time.sleep(5)
    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--pd-disaggregation", "--mini-lb",
           "--prefill", f"http://127.0.0.1:{PA_PORT}",
           "--decode", f"http://127.0.0.1:{DA_PORT}",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    rf = open(log_dir / "router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("PDAF 2P1D router failed"); B.cleanup_procs(procs); return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PDAF 2P+1D %s ready, warming up...", "(Tier)" if tier else "")
    B.warmup(url)
    return procs, url


# ======== Dispatcher ========
DEPLOY_MAP = {
    "native_dp6":       lambda tier, ld: start_native_dp6(False, ld),
    "native_dp6_tier":  lambda tier, ld: start_native_dp6(True, ld),
    "pd_dp3":           lambda tier, ld: start_pd_dp3(False, ld),
    "pd_dp3_tier":      lambda tier, ld: start_pd_dp3(True, ld),
    "pdaf_6g_1p2d":     lambda tier, ld: start_pdaf_6g_1p2d(False, ld),
    "pdaf_6g_1p2d_tier":lambda tier, ld: start_pdaf_6g_1p2d(True, ld),
    "pdaf_6g_2p1d":     lambda tier, ld: start_pdaf_6g_2p1d(False, ld),
    "pdaf_6g_2p1d_tier":lambda tier, ld: start_pdaf_6g_2p1d(True, ld),
}

DEPLOY_LABELS = {
    "native_dp6": "Native DP6 (6×TP1)",
    "native_dp6_tier": "Native DP6 + Tier",
    "pd_dp3": "PD DP3 (3×P1D1)",
    "pd_dp3_tier": "PD DP3 + Tier",
    "pdaf_6g_1p2d": "PDAF 1P+2D (decode-heavy)",
    "pdaf_6g_1p2d_tier": "PDAF 1P+2D + Tier",
    "pdaf_6g_2p1d": "PDAF 2P+1D (prefill-heavy)",
    "pdaf_6g_2p1d_tier": "PDAF 2P+1D + Tier",
}

ALL_DEPLOYS = list(DEPLOY_MAP.keys())


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="6-GPU deployment benchmark")
    ap.add_argument("--deploys", type=str, default=",".join(ALL_DEPLOYS))
    ap.add_argument("--ttft-slo-ms", type=float, default=5000.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=200.0)
    ap.add_argument("--max-run-s", type=float, default=600.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    global DVFS_TTFT_SLO_MS, DVFS_TPOT_SLO_US
    DVFS_TTFT_SLO_MS = int(args.ttft_slo_ms)
    DVFS_TPOT_SLO_US = int(args.tpot_slo_ms * 1000)

    deploys = [d.strip() for d in args.deploys.split(",") if d.strip()]
    for d in deploys:
        if d not in DEPLOY_MAP:
            ap.error(f"Unknown deploy '{d}'. Valid: {ALL_DEPLOYS}")

    out_dir = HERE / "results_6gpu_azure" / "json"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_root = HERE / "logs_6gpu"
    log_root.mkdir(parents=True, exist_ok=True)

    log.info("6-GPU Bench | Deploys: %s | %d workloads", deploys, len(WORKLOADS))

    all_results = {}

    for deploy in deploys:
        all_results[deploy] = {}
        for wl_label, wl_path in WORKLOADS:
            tag = f"{deploy}_{wl_label.replace(' ', '_').lower()}"
            cached = out_dir / f"{tag}_results.json"

            if cached.exists() and not args.force:
                results = json.loads(cached.read_text())
                all_results[deploy][wl_label] = results
                log.info("Loaded cached %s", cached)
                continue

            log.info("=" * 70)
            log.info("DEPLOY: %s | Workload: %s", deploy, wl_label)
            log.info("=" * 70)

            kill_servers()
            reset_freq()
            time.sleep(2)
            if not deploy.endswith("_tier"):
                lock_max_freq()

            run_log_dir = log_root / tag
            run_log_dir.mkdir(parents=True, exist_ok=True)

            tier = deploy.endswith("_tier")
            ret = DEPLOY_MAP[deploy](tier, run_log_dir)
            if ret is None:
                log.error("%s failed to start, skipping", deploy)
                reset_freq()
                continue
            procs, url = ret

            try:
                results = asyncio.run(B.run_workload(
                    str(wl_path), url,
                    ttft_slo_ms=args.ttft_slo_ms,
                    tpot_slo_ms=args.tpot_slo_ms,
                    procs=procs, max_run_s=args.max_run_s,
                    gpu_indices=GPUS_6,
                    prefill_gpus=GPUS_6,
                    decode_gpus=GPUS_6,
                ))
                results.update({"deploy": deploy, "workload": str(wl_path),
                                "wl_label": wl_label, "ngpu": 6})
                all_results[deploy][wl_label] = results
                if not results.get("aborted"):
                    with open(cached, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                log.info("  %s | thpt=%.1f TTFT=%.0f TPOT=%.0f energy=%.0fJ SLO=%.1f%%",
                         tag, results["throughput_tok_s"], results["ttft_avg_ms"],
                         results["tpot_avg_ms"], results["total_energy_j"],
                         results["slo_violation_rate"])
            except Exception as e:
                import traceback
                log.error("%s crashed: %r\n%s", tag, e, traceback.format_exc())
            finally:
                B.cleanup_procs(procs)
                kill_servers()
                reset_freq()
                time.sleep(5)

    # Save summary
    with open(out_dir / "6gpu_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Print table
    print("\n" + "=" * 110)
    print("  6-GPU DEPLOYMENT BENCHMARK SUMMARY")
    print("=" * 110)
    print(f"  {'Deploy':<25} {'Workload':<15} {'Thpt':>8} {'TTFT':>8} {'TPOT':>8} "
          f"{'Energy':>9} {'mJ/tok':>8} {'SLO%':>6}")
    print("-" * 110)
    for deploy in deploys:
        for wl_label in [w[0] for w in WORKLOADS]:
            m = all_results.get(deploy, {}).get(wl_label)
            if not m:
                continue
            print(f"  {deploy:<25} {wl_label:<15} {m['throughput_tok_s']:>8.1f} "
                  f"{m['ttft_avg_ms']:>8.0f} {m['tpot_avg_ms']:>8.0f} "
                  f"{m['total_energy_j']:>9.0f} {m.get('energy_per_token_mj',0):>8.1f} "
                  f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 110)
    log.info("All results saved to %s/", out_dir)


if __name__ == "__main__":
    main()
