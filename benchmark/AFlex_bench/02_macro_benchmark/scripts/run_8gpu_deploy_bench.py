#!/usr/bin/env python3
"""8-GPU full-scale deployment benchmark.

Tests multiple deployment topologies on all 8 A800 GPUs with various workloads
at increasing QPS to find throughput ceilings and SLO collapse points.

Topologies:
  - native_tp8      : Single instance, TP=8, no disaggregation (8 GPU)
  - pd_p4d4         : PD disagg, Prefill TP=4 (GPU 4-7) + Decode TP=4 (GPU 0-3)
  - pd_dp4          : PD disagg DP=4, 4x (P-TP1 + D-TP1) instances
  - pdaf_8g_m1      : PD+AF, M=1, each component TP=2 (PA/PF/DA/DF)
  - pdaf_8g_m2      : PD+AF, M=2, each component TP=2 (PA/PF/DA/DF)
  - pdaf_8g_m1_tier : PD+AF M=1 + DVFS, each component TP=2
  - pdaf_8g_m2_tier : PD+AF M=2 + DVFS, each component TP=2
  - pdaf_8g_dyn     : PD+AF DynM, each component TP=2
  - pdaf_8g_dyn_tier: PD+AF DynM + DVFS, each component TP=2
  - native_dp8      : Native DP=8, 8x independent TP=1 instances, round-robin

Usage:
    python run_8gpu_deploy_bench.py --deploys pd_p4d4,pdaf_8g_dyn,pdaf_8g_dyn_tier \
        --workloads /path/to/wl1.jsonl,/path/to/wl2.jsonl \
        --freq auto --max-run-s 600
"""
import argparse
import asyncio
import json
import logging
import os
import re
import signal
import socket
import subprocess
import time
from pathlib import Path

import run_fixed_qps_bench as B

log = logging.getLogger("8gpu_bench")

PYTHON = B.PYTHON
MODEL = B.MODEL
HERE = Path(__file__).resolve().parent

ROUTER_PORT = 53000
PA_PORT, PF_PORT = 53010, 53011
DA_PORT, DF_PORT = 53020, 53021
BOOTSTRAP_PORT = 49999
UCX_P, UCX_D = 28200, 28300
SCHED_P, SCHED_D = 68400, 68500

# DP4 ports: 4 instances, each with prefill + decode + bootstrap
DP4_PORTS = [
    {"p": 53100, "d": 53101, "bs": 49100},
    {"p": 53110, "d": 53111, "bs": 49110},
    {"p": 53120, "d": 53121, "bs": 49120},
    {"p": 53130, "d": 53131, "bs": 49130},
]

# PD-DP ports: PF_PORT+100+i*2 and DF_PORT+100+i*2 for i in 0..3
_PD_DP_PORTS = [PF_PORT + 100 + i * 2 for i in range(4)] + \
               [DF_PORT + 100 + i * 2 for i in range(4)]

ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT] + \
    [p for inst in DP4_PORTS for p in [inst["p"], inst["d"]]] + \
    _PD_DP_PORTS + [53100] + \
    [53200 + i * 10 for i in range(8)]  # native_dp8 ports

DEPLOYMENTS = {
    "native_tp8": {
        "label": "Native TP=8 (single instance, no disaggregation, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": list(range(8)),
        "decode_gpus": list(range(8)),
        "ngpu": 8,
        "native": {"cvd": "0,1,2,3,4,5,6,7", "tp": 8, "port": 53100},
    },
    "pd_p4d4": {
        "label": "PD Disagg (Prefill TP=4 + Decode TP=4, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [4, 5, 6, 7],
        "decode_gpus": [0, 1, 2, 3],
        "ngpu": 8,
        "pd": {"p_cvd": "4,5,6,7", "p_tp": 4, "d_cvd": "0,1,2,3", "d_tp": 4},
    },
    "pd_dp4": {
        "label": "PD Disagg DP=4 (4x P-TP1/D-TP1 instances, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 2, 4, 6],
        "decode_gpus": [1, 3, 5, 7],
        "ngpu": 8,
        "dp": {"instances": [{"p_cvd": "0", "d_cvd": "1"},
                             {"p_cvd": "2", "d_cvd": "3"},
                             {"p_cvd": "4", "d_cvd": "5"},
                             {"p_cvd": "6", "d_cvd": "7"}], "tp": 1},
    },
    "pd_dp4_tier": {
        "label": "PD Disagg DP=4 + Tier DVFS (4x P-TP1/D-TP1, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 2, 4, 6],
        "decode_gpus": [1, 3, 5, 7],
        "ngpu": 8,
        "dp": {"instances": [{"p_cvd": "0", "d_cvd": "1"},
                             {"p_cvd": "2", "d_cvd": "3"},
                             {"p_cvd": "4", "d_cvd": "5"},
                             {"p_cvd": "6", "d_cvd": "7"}], "tp": 1,
               "tier": True},
    },
    "pdaf_8g_m1": {
        "label": "PD+AF M=1 (each component TP=2, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3],
        "decode_gpus": [4, 5, 6, 7],
        "ngpu": 8,
        "af_p_vis": "0,1,2,3",
        "af_d_cvd": "4,5,6,7",
        "micro_batch": 1,
        "af_tp": 2,
    },
    "pdaf_8g_m2": {
        "label": "PD+AF M=2 (each component TP=2, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3],
        "decode_gpus": [4, 5, 6, 7],
        "ngpu": 8,
        "af_p_vis": "0,1,2,3",
        "af_d_cvd": "4,5,6,7",
        "micro_batch": 2,
        "af_tp": 2,
    },
    "pdaf_8g_m1_tier": {
        "label": "PD+AF M=1 + DVFS (each component TP=2, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3],
        "decode_gpus": [4, 5, 6, 7],
        "ngpu": 8,
        "af_p_vis": "0,1,2,3",
        "af_d_cvd": "4,5,6,7",
        "micro_batch": 1,
        "af_tp": 2,
    },
    "pdaf_8g_m2_tier": {
        "label": "PD+AF M=2 + DVFS (each component TP=2, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3],
        "decode_gpus": [4, 5, 6, 7],
        "ngpu": 8,
        "af_p_vis": "0,1,2,3",
        "af_d_cvd": "4,5,6,7",
        "micro_batch": 2,
        "af_tp": 2,
    },
    "pdaf_8g_dyn": {
        "label": "PD+AF DynM (each component TP=2, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3],
        "decode_gpus": [4, 5, 6, 7],
        "ngpu": 8,
        "af_p_vis": "0,1,2,3",
        "af_d_cvd": "4,5,6,7",
        "micro_batch": 2,
        "af_tp": 2,
        "dynamic_mb": True,
    },
    "pdaf_8g_dyn_tier": {
        "label": "PD+AF DynM + DVFS (each component TP=2, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3],
        "decode_gpus": [4, 5, 6, 7],
        "ngpu": 8,
        "af_p_vis": "0,1,2,3",
        "af_d_cvd": "4,5,6,7",
        "micro_batch": 2,
        "af_tp": 2,
        "dynamic_mb": True,
    },
    "pdaf_8g_asym_1p6d_tier": {
        "label": "PD+AF Asym 1PA1PF+2DA4DF + DVFS (P:TP1x2, D:DA-TP2+DF-TP4, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1],
        "decode_gpus": [2, 3, 4, 5, 6, 7],
        "ngpu": 8,
        "asym_pdaf": {
            "p_cvd": "0,1",
            "p_tp": 1,
            "d_cvd": "2,3,4,5,6,7",
            "d_attn_tp": 2,
            "d_ffn_tp": 4,
            "micro_batch": 2,
            "dynamic_mb": True,
        },
    },
    "pdaf_8g_asym_1p6d": {
        "label": "PD+AF Asym 1PA1PF+2DA4DF (P:TP1x2, D:DA-TP2+DF-TP4, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1],
        "decode_gpus": [2, 3, 4, 5, 6, 7],
        "ngpu": 8,
        "asym_pdaf": {
            "p_cvd": "0,1",
            "p_tp": 1,
            "d_cvd": "2,3,4,5,6,7",
            "d_attn_tp": 2,
            "d_ffn_tp": 4,
            "micro_batch": 2,
            "dynamic_mb": True,
        },
    },
    "pdaf_8g_2pa4pf": {
        "label": "PD+AF Hetero 2PA4PF+1DA1DF (P:PF-TP4+PA-TP2, D:TP1+TP1, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3, 4, 5],
        "decode_gpus": [6, 7],
        "ngpu": 8,
        "hetero_pdaf": {
            "p_cvd": "0,1,2,3,4,5",
            "p_attn_tp": 2,
            "p_ffn_tp": 4,
            "p_attn_base": 4,
            "p_ffn_base": 0,
            "d_cvd": "6,7",
            "d_attn_tp": 1,
            "d_ffn_tp": 1,
            "d_attn_base": 1,
            "d_ffn_base": 0,
            "micro_batch": 2,
            "dynamic_mb": True,
            "d_max_running": 32,
        },
    },
    "pdaf_8g_2pa4pf_tier": {
        "label": "PD+AF Hetero 2PA4PF+1DA1DF + DVFS (P:PF-TP4+PA-TP2, D:TP1+TP1, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3, 4, 5],
        "decode_gpus": [6, 7],
        "ngpu": 8,
        "hetero_pdaf": {
            "p_cvd": "0,1,2,3,4,5",
            "p_attn_tp": 2,
            "p_ffn_tp": 4,
            "p_attn_base": 4,
            "p_ffn_base": 0,
            "d_cvd": "6,7",
            "d_attn_tp": 1,
            "d_ffn_tp": 1,
            "d_attn_base": 1,
            "d_ffn_base": 0,
            "micro_batch": 2,
            "dynamic_mb": True,
            "d_max_running": 32,
        },
    },
    "pdaf_8g_4pa2pf": {
        "label": "PD+AF Hetero 4PA2PF+1DA1DF (P:PF-TP2+PA-TP4, D:TP1+TP1, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3, 4, 5],
        "decode_gpus": [6, 7],
        "ngpu": 8,
        "hetero_pdaf": {
            "p_cvd": "0,1,2,3,4,5",
            "p_attn_tp": 4,
            "p_ffn_tp": 2,
            "p_attn_base": 2,
            "p_ffn_base": 0,
            "d_cvd": "6,7",
            "d_attn_tp": 1,
            "d_ffn_tp": 1,
            "d_attn_base": 1,
            "d_ffn_base": 0,
            "micro_batch": 2,
            "dynamic_mb": True,
            "d_max_running": 32,
        },
    },
    "pdaf_8g_4pa2pf_tier": {
        "label": "PD+AF Hetero 4PA2PF+1DA1DF + DVFS (P:PF-TP2+PA-TP4, D:TP1+TP1, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": [0, 1, 2, 3, 4, 5],
        "decode_gpus": [6, 7],
        "ngpu": 8,
        "hetero_pdaf": {
            "p_cvd": "0,1,2,3,4,5",
            "p_attn_tp": 4,
            "p_ffn_tp": 2,
            "p_attn_base": 2,
            "p_ffn_base": 0,
            "d_cvd": "6,7",
            "d_attn_tp": 1,
            "d_ffn_tp": 1,
            "d_attn_base": 1,
            "d_ffn_base": 0,
            "micro_batch": 2,
            "dynamic_mb": True,
            "d_max_running": 32,
        },
    },
    "native_dp8": {
        "label": "Native DP=8 (8x TP=1 independent instances, round-robin, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": list(range(8)),
        "decode_gpus": list(range(8)),
        "ngpu": 8,
        "dp_full": {
            "instances": [
                {"cvd": "0", "tp": 1, "port": 53200},
                {"cvd": "1", "tp": 1, "port": 53210},
                {"cvd": "2", "tp": 1, "port": 53220},
                {"cvd": "3", "tp": 1, "port": 53230},
                {"cvd": "4", "tp": 1, "port": 53240},
                {"cvd": "5", "tp": 1, "port": 53250},
                {"cvd": "6", "tp": 1, "port": 53260},
                {"cvd": "7", "tp": 1, "port": 53270},
            ],
        },
    },
    "native_dp8_tier": {
        "label": "Native DP=8 + Tier DVFS (8x TP=1 instances, round-robin, 8 GPU)",
        "gpus": list(range(8)),
        "prefill_gpus": list(range(8)),
        "decode_gpus": list(range(8)),
        "ngpu": 8,
        "dp_full": {
            "tier": True,
            "instances": [
                {"cvd": "0", "tp": 1, "port": 53200},
                {"cvd": "1", "tp": 1, "port": 53210},
                {"cvd": "2", "tp": 1, "port": 53220},
                {"cvd": "3", "tp": 1, "port": 53230},
                {"cvd": "4", "tp": 1, "port": 53240},
                {"cvd": "5", "tp": 1, "port": 53250},
                {"cvd": "6", "tp": 1, "port": 53260},
                {"cvd": "7", "tp": 1, "port": 53270},
            ],
        },
    },
}


def apply_freq(gpus, freq: str):
    for idx in gpus:
        if freq == "max":
            subprocess.run(["nvidia-smi", "-lgc", f"{B.MAX_SM_FREQ_MHZ},{B.MAX_SM_FREQ_MHZ}",
                            "-i", str(idx)], capture_output=True)
        else:
            subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)
    log.info("Freq policy '%s' applied to GPUs %s", freq, gpus)


def reset_freq(gpus):
    for idx in gpus:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


def kill_servers():
    for port in ALL_PORTS:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True)
        for m in re.finditer(r'pid=(\d+)', r.stdout):
            subprocess.run(["kill", "-9", m.group(1)], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang_router"], capture_output=True)
    time.sleep(5)
    # Verify router port is free
    for _ in range(10):
        r = subprocess.run(["ss", "-tlnp", f"sport = :{ROUTER_PORT}"],
                           capture_output=True, text=True)
        if "LISTEN" not in r.stdout:
            break
        time.sleep(2)


def _popen(name, cmd, env, log_dir, prefix, procs):
    fh = open(log_dir / f"{prefix}{name}.log", "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    procs.append((name, p, fh))
    log.info("  Started %s (CVD=%s, port via cmd)", name, env.get("CUDA_VISIBLE_DEVICES"))
    return p


def start_router_multi(procs, log_dir, prefix, prefill_specs, decode_urls,
                       policy="round_robin"):
    """PD router across multiple instances (DP).

    prefill_specs: list of (url, bootstrap_port) tuples.
    decode_urls:   list of decode urls.
    """
    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--pd-disaggregation",
           "--policy", policy,
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    for url, bport in prefill_specs:
        cmd += ["--prefill", url, str(bport)]
    for url in decode_urls:
        cmd += ["--decode", url]
    rf = open(log_dir / f"{prefix}router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("Multi-instance router failed to start")
        return False
    return True


def start_router(procs, log_dir, prefix, prefill_port, decode_port):
    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--pd-disaggregation", "--mini-lb",
           "--prefill", f"http://127.0.0.1:{prefill_port}",
           "--decode", f"http://127.0.0.1:{decode_port}",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    rf = open(log_dir / f"{prefix}router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("Router failed to start")
        return False
    return True


_PD_COMMON = [
    "--mem-fraction-static", "0.85",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT),
    "--disaggregation-ib-device", "mlx5_4",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--disable-radix-cache",
]


def start_native_tp8(log_dir, prefix, spec):
    """Start a single native TP=8 instance (no PD, no AF)."""
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    native = spec["native"]
    env = env_base.copy()
    env["CUDA_VISIBLE_DEVICES"] = native["cvd"]
    port = native["port"]
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--tp", str(native["tp"]),
           "--host", "127.0.0.1", "--port", str(port),
           "--mem-fraction-static", "0.85",
           "--disable-cuda-graph",
           "--disable-piecewise-cuda-graph",
           "--disable-radix-cache"]
    _popen("native", cmd, env, log_dir, prefix, procs)
    if not B.wait_port("127.0.0.1", port, 360):
        log.error("Native TP=8 failed to start")
        B.cleanup_procs(procs)
        return None
    url = f"http://127.0.0.1:{port}"
    log.info("Native TP=8 ready at %s, warming up...", url)
    B.warmup(url)
    return procs, url


def _pd_common_bootstrap(bootstrap_port):
    """_PD_COMMON with a per-instance bootstrap port."""
    out = []
    skip_next = False
    for i, tok in enumerate(_PD_COMMON):
        if skip_next:
            out.append(str(bootstrap_port)); skip_next = False; continue
        out.append(tok)
        if tok == "--disaggregation-bootstrap-port":
            skip_next = True
    return out


def start_pd_dp(log_dir, prefix, instances, tp, tier=False):
    """PD DP: N independent 1P1D instances with router load-balancing.

    When tier=True, each prefill/decode server runs the unified single-knob
    DVFS controller, locked to its own physical GPU.
    """
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    prefill_specs, decode_urls = [], []

    for i, inst in enumerate(instances):
        pf_port = PF_PORT + 100 + i * 2
        df_port = DF_PORT + 100 + i * 2
        bport = BOOTSTRAP_PORT + 1 + i
        common = _pd_common_bootstrap(bport)
        dvfs = _unified_dvfs_args() if tier else []

        env_p = env_base.copy()
        env_p["CUDA_VISIBLE_DEVICES"] = inst["p_cvd"]
        if tier:
            env_p["AFD_NVML_DEVICE_INDICES"] = inst["p_cvd"]
            env_p["AFD_NVML_DEVICE_INDEX"] = inst["p_cvd"].split(",")[0]
        p_cmd = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", str(tp),
                 "--host", "127.0.0.1", "--port", str(pf_port),
                 "--disaggregation-mode", "prefill"] + common + dvfs
        _popen(f"prefill{i}", p_cmd, env_p, log_dir, prefix, procs)

        env_d = env_base.copy()
        env_d["CUDA_VISIBLE_DEVICES"] = inst["d_cvd"]
        if tier:
            env_d["AFD_NVML_DEVICE_INDICES"] = inst["d_cvd"]
            env_d["AFD_NVML_DEVICE_INDEX"] = inst["d_cvd"].split(",")[0]
        d_cmd = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", str(tp),
                 "--host", "127.0.0.1", "--port", str(df_port),
                 "--disaggregation-mode", "decode"] + common + dvfs
        _popen(f"decode{i}", d_cmd, env_d, log_dir, prefix, procs)

        prefill_specs.append((f"http://127.0.0.1:{pf_port}", bport))
        decode_urls.append(f"http://127.0.0.1:{df_port}")

    for url, _ in prefill_specs:
        if not B.wait_port("127.0.0.1", int(url.rsplit(":", 1)[1]), timeout=300):
            log.error("PD-DP prefill %s failed to start", url)
            B.cleanup_procs(procs); return None
    for url in decode_urls:
        if not B.wait_port("127.0.0.1", int(url.rsplit(":", 1)[1]), timeout=300):
            log.error("PD-DP decode %s failed to start", url)
            B.cleanup_procs(procs); return None

    if not start_router_multi(procs, log_dir, prefix, prefill_specs, decode_urls):
        B.cleanup_procs(procs); return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PD DP=%d (TP=%d) ready at %s, warming up...",
             len(instances), tp, url)
    B.warmup(url)
    return procs, url


def start_pd(log_dir, prefix, p_cvd, p_tp, d_cvd, d_tp):
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

    env_p = env_base.copy()
    env_p["CUDA_VISIBLE_DEVICES"] = p_cvd
    p_cmd = [PYTHON, "-m", "sglang.launch_server",
             "--model-path", MODEL, "--tp", str(p_tp),
             "--host", "127.0.0.1", "--port", str(PF_PORT),
             "--disaggregation-mode", "prefill"] + _PD_COMMON
    _popen("prefill", p_cmd, env_p, log_dir, prefix, procs)

    env_d = env_base.copy()
    env_d["CUDA_VISIBLE_DEVICES"] = d_cvd
    d_cmd = [PYTHON, "-m", "sglang.launch_server",
             "--model-path", MODEL, "--tp", str(d_tp),
             "--host", "127.0.0.1", "--port", str(DF_PORT),
             "--disaggregation-mode", "decode"] + _PD_COMMON
    _popen("decode", d_cmd, env_d, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", PF_PORT, timeout=300) or \
       not B.wait_port("127.0.0.1", DF_PORT, timeout=300):
        log.error("PD servers failed to start")
        B.cleanup_procs(procs)
        return None
    if not start_router(procs, log_dir, prefix, PF_PORT, DF_PORT):
        B.cleanup_procs(procs)
        return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PD (P-tp%d/D-tp%d) ready at %s, warming up...", p_tp, d_tp, url)
    B.warmup(url)
    return procs, url


_AFD_EXTRA_BASE = [
    "--afd-comm-backend", "ipc_cpp",
    "--mem-fraction-static", "0.85",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT),
    "--disaggregation-ib-device", "mlx5_4",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--afd-disagg-interleave-poll",
    "--disable-radix-cache",
]


def _dvfs_args():
    return ["--afd-dvfs-enabled",
            "--afd-energy-model-dir", B.ENERGY_MODEL_DIR,
            "--afd-energy-model-v3-dir", B.ENERGY_MODEL_V3_DIR,
            "--afd-ttft-slo-ms", str(DVFS_TTFT_SLO_MS),
            "--afd-tpot-slo-us", str(DVFS_TPOT_SLO_US)]


def _unified_dvfs_args():
    """Unified single-knob DVFS args for PD / Native (non-AF) instances."""
    return ["--dvfs-enabled",
            "--dvfs-energy-model-dir", B.ENERGY_MODEL_DIR,
            "--dvfs-ttft-slo-ms", str(DVFS_TTFT_SLO_MS),
            "--dvfs-tpot-slo-us", str(DVFS_TPOT_SLO_US)]

DVFS_TTFT_SLO_MS = 5000
DVFS_TPOT_SLO_US = 300000


def _tier1_stats_path():
    d = HERE / "results" / "tier1_shared"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "8gpu_decode_stats.json")


def _tier1_pa_args(stats_path):
    return ["--enable-tier1-pa", "--tier1-disable-reload",
            "--tier1-monitor-window-s", "15", "--tier1-gpu-count", "8",
            "--tier1-stats-path", stats_path,
            "--tier1-prefill-data-path",
            "/workspace/sglang/benchmark/test_motivation/hucc/paper/prefill_data_v1.txt",
            "--tier1-decode-data-path",
            "/workspace/sglang/benchmark/test_motivation/hucc/paper/decode_data_v1.txt"]


def _owned_gpus(cvd, base_gpu_id, tp):
    """Physical NVML indices this process owns = CVD[base : base+tp]."""
    cvd_list = [x.strip() for x in str(cvd).split(",") if x.strip()]
    return cvd_list[base_gpu_id:base_gpu_id + tp]


def _afd_env(env_base, cvd, ucx_base, sched_port, peer_device, ffn_host=None,
             nvml_indices=None):
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


def _afd_cmd(port, perspective, disagg, tp, base_gpu_id, micro_batch=2,
             attn_tp=None, ffn_tp=None, tier=False, is_pa=False, stats_path=None,
             dynamic_mb=False, max_running=None):
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--tp", str(tp),
           "--host", "127.0.0.1", "--port", str(port),
           "--afd-perspective", perspective,
           "--disaggregation-mode", disagg,
           "--base-gpu-id", str(base_gpu_id)] + _AFD_EXTRA_BASE + [
           "--afd-micro-batch", str(micro_batch)]
    if dynamic_mb:
        cmd += ["--afd-dynamic-micro-batch"]
    if attn_tp is not None:
        cmd += ["--afd-attn-tp", str(attn_tp)]
    if ffn_tp is not None:
        cmd += ["--afd-ffn-tp", str(ffn_tp)]
    if max_running is not None:
        cmd += ["--max-running-requests", str(max_running)]
    if tier:
        cmd += _dvfs_args()
        if is_pa:
            cmd += _tier1_pa_args(stats_path)
        elif stats_path:
            cmd += ["--tier1-stats-path", stats_path]
    return cmd


def start_pdaf(deploy, log_dir, prefix):
    """Start 8-GPU PD+AF: PF(TP2)+PA(TP2) on 4 prefill GPUs, DF(TP2)+DA(TP2) on 4 decode GPUs.

    GPU layout:
      Prefill: PF(ffn, TP2, base=0) + PA(attn, TP2, base=2) sharing CVD=0,1,2,3
      Decode:  DF(ffn, TP2, base=0) + DA(attn, TP2, base=2) sharing CVD=4,5,6,7
               peer_device: PF→2 (PA's base), PA→0 (PF's base)
                            DF→2 (DA's base), DA→0 (DF's base)
    """
    tier = deploy.endswith("_tier")
    stats_path = _tier1_stats_path() if tier else None
    spec = DEPLOYMENTS[deploy]
    micro_batch = spec["micro_batch"]
    tp = spec["af_tp"]
    p_vis = spec["af_p_vis"]
    d_cvd = spec["af_d_cvd"]
    dynamic_mb = spec.get("dynamic_mb", False)

    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"
    if tier:
        env_base["AFD_DVFS_DECISION_LOG"] = str(
            log_dir / f"{prefix}dvfs_decisions_{{persp}}_{{disagg}}_gpu{{gpu}}.jsonl")

    # Prefill: PF(ffn, TP2, base=0) + PA(attn, TP2, base=2) on CVD=0,1,2,3
    env_pf = _afd_env(env_base, p_vis, UCX_P, SCHED_P, peer_device=2,
                      nvml_indices=_owned_gpus(p_vis, 0, tp))
    _popen("pf", _afd_cmd(PF_PORT, "ffn", "prefill", tp=tp, base_gpu_id=0,
                          micro_batch=micro_batch, tier=tier, stats_path=stats_path,
                          dynamic_mb=dynamic_mb),
           env_pf, log_dir, prefix, procs)
    time.sleep(2)

    env_pa = _afd_env(env_base, p_vis, UCX_P, SCHED_P, peer_device=0, ffn_host="127.0.0.1",
                      nvml_indices=_owned_gpus(p_vis, 2, tp))
    _popen("pa", _afd_cmd(PA_PORT, "attn", "prefill", tp=tp, base_gpu_id=2,
                          micro_batch=micro_batch, tier=tier, is_pa=True,
                          stats_path=stats_path, dynamic_mb=dynamic_mb),
           env_pa, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", PF_PORT, 300) or \
       not B.wait_port("127.0.0.1", PA_PORT, 300):
        log.error("AF prefill failed to start")
        B.cleanup_procs(procs)
        return None

    # Decode: DF(ffn, TP2, base=0) + DA(attn, TP2, base=2) on CVD=4,5,6,7
    env_df = _afd_env(env_base, d_cvd, UCX_D, SCHED_D, peer_device=2,
                      nvml_indices=_owned_gpus(d_cvd, 0, tp))
    _popen("df", _afd_cmd(DF_PORT, "ffn", "decode", tp=tp, base_gpu_id=0,
                          micro_batch=micro_batch, attn_tp=tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb),
           env_df, log_dir, prefix, procs)
    time.sleep(5)

    env_da = _afd_env(env_base, d_cvd, UCX_D, SCHED_D, peer_device=0, ffn_host="127.0.0.1",
                      nvml_indices=_owned_gpus(d_cvd, 2, tp))
    _popen("da", _afd_cmd(DA_PORT, "attn", "decode", tp=tp, base_gpu_id=2,
                          micro_batch=micro_batch, ffn_tp=tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb),
           env_da, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", DA_PORT, 300) or \
       not B.wait_port("127.0.0.1", DF_PORT, 300):
        log.error("AF decode failed to start")
        B.cleanup_procs(procs)
        return None

    time.sleep(5)
    if not start_router(procs, log_dir, prefix, PA_PORT, DA_PORT):
        B.cleanup_procs(procs)
        return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("%s ready at %s, warming up...", deploy, url)
    B.warmup(url)
    return procs, url


def start_pdaf_asym(deploy, log_dir, prefix):
    """Start asymmetric PDAF: 1PA(TP1)+1PF(TP1) on 2 prefill GPUs,
    2DA(TP2)+4DF(TP4) on 6 decode GPUs.

    GPU layout:
      Prefill: PF(ffn, TP=1, base=0) + PA(attn, TP=1, base=1) on CVD=0,1
      Decode:  DF(ffn, TP=4, base=0) + DA(attn, TP=2, base=4) on CVD=2,3,4,5,6,7
    """
    tier = deploy.endswith("_tier")
    stats_path = _tier1_stats_path() if tier else None
    spec = DEPLOYMENTS[deploy]
    asym = spec["asym_pdaf"]
    p_cvd = asym["p_cvd"]
    d_cvd = asym["d_cvd"]
    p_tp = asym["p_tp"]
    d_attn_tp = asym["d_attn_tp"]
    d_ffn_tp = asym["d_ffn_tp"]
    micro_batch = asym["micro_batch"]
    dynamic_mb = asym.get("dynamic_mb", False)

    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"
    if tier:
        env_base["AFD_DVFS_DECISION_LOG"] = str(
            log_dir / f"{prefix}dvfs_decisions_{{persp}}_{{disagg}}_gpu{{gpu}}.jsonl")

    # Prefill: PF(ffn, TP=1, base=0) + PA(attn, TP=1, base=1) on CVD=0,1
    env_pf = _afd_env(env_base, p_cvd, UCX_P, SCHED_P, peer_device=1,
                      nvml_indices=_owned_gpus(p_cvd, 0, p_tp))
    _popen("pf", _afd_cmd(PF_PORT, "ffn", "prefill", tp=p_tp, base_gpu_id=0,
                          micro_batch=micro_batch, tier=tier, stats_path=stats_path,
                          dynamic_mb=dynamic_mb),
           env_pf, log_dir, prefix, procs)
    time.sleep(2)

    env_pa = _afd_env(env_base, p_cvd, UCX_P, SCHED_P, peer_device=0,
                      ffn_host="127.0.0.1",
                      nvml_indices=_owned_gpus(p_cvd, 1, p_tp))
    _popen("pa", _afd_cmd(PA_PORT, "attn", "prefill", tp=p_tp, base_gpu_id=1,
                          micro_batch=micro_batch, tier=tier, is_pa=True,
                          stats_path=stats_path, dynamic_mb=dynamic_mb),
           env_pa, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", PF_PORT, 300) or \
       not B.wait_port("127.0.0.1", PA_PORT, 300):
        log.error("AF prefill (asym) failed to start")
        B.cleanup_procs(procs)
        return None

    # Decode: DF(ffn, TP=4, base=0) + DA(attn, TP=2, base=4) on CVD=2,3,4,5,6,7
    env_df = _afd_env(env_base, d_cvd, UCX_D, SCHED_D, peer_device=4,
                      nvml_indices=_owned_gpus(d_cvd, 0, d_ffn_tp))
    _popen("df", _afd_cmd(DF_PORT, "ffn", "decode", tp=d_ffn_tp, base_gpu_id=0,
                          micro_batch=micro_batch, attn_tp=d_attn_tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb),
           env_df, log_dir, prefix, procs)
    time.sleep(5)

    env_da = _afd_env(env_base, d_cvd, UCX_D, SCHED_D, peer_device=0,
                      ffn_host="127.0.0.1",
                      nvml_indices=_owned_gpus(d_cvd, 4, d_attn_tp))
    _popen("da", _afd_cmd(DA_PORT, "attn", "decode", tp=d_attn_tp, base_gpu_id=4,
                          micro_batch=micro_batch, ffn_tp=d_ffn_tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb),
           env_da, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", DA_PORT, 300) or \
       not B.wait_port("127.0.0.1", DF_PORT, 300):
        log.error("AF decode (asym) failed to start")
        B.cleanup_procs(procs)
        return None

    time.sleep(5)
    if not start_router(procs, log_dir, prefix, PA_PORT, DA_PORT):
        B.cleanup_procs(procs)
        return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("%s ready at %s, warming up...", deploy, url)
    B.warmup(url)
    return procs, url


def start_pdaf_hetero(deploy, log_dir, prefix):
    """Start heterogeneous PDAF where prefill itself uses attn_tp != ffn_tp.

    Layout (general, driven by spec['hetero_pdaf']):
      Prefill: PF(ffn, tp=p_ffn_tp, base=p_ffn_base) +
               PA(attn, tp=p_attn_tp, base=p_attn_base) sharing p_cvd
      Decode:  DF(ffn, tp=d_ffn_tp, base=d_ffn_base) +
               DA(attn, tp=d_attn_tp, base=d_attn_base) sharing d_cvd
    """
    tier = deploy.endswith("_tier")
    stats_path = _tier1_stats_path() if tier else None
    spec = DEPLOYMENTS[deploy]
    h = spec["hetero_pdaf"]
    p_cvd, d_cvd = h["p_cvd"], h["d_cvd"]
    p_attn_tp, p_ffn_tp = h["p_attn_tp"], h["p_ffn_tp"]
    d_attn_tp, d_ffn_tp = h["d_attn_tp"], h["d_ffn_tp"]
    micro_batch = h["micro_batch"]
    dynamic_mb = h.get("dynamic_mb", False)
    d_max_running = h.get("d_max_running")

    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"
    if tier:
        env_base["AFD_DVFS_DECISION_LOG"] = str(
            log_dir / f"{prefix}dvfs_decisions_{{persp}}_{{disagg}}_gpu{{gpu}}.jsonl")

    # Prefill: PF(ffn) + PA(attn) sharing p_cvd
    env_pf = _afd_env(env_base, p_cvd, UCX_P, SCHED_P, peer_device=h["p_attn_base"],
                      nvml_indices=_owned_gpus(p_cvd, h["p_ffn_base"], p_ffn_tp))
    _popen("pf", _afd_cmd(PF_PORT, "ffn", "prefill", tp=p_ffn_tp,
                          base_gpu_id=h["p_ffn_base"], micro_batch=micro_batch,
                          attn_tp=p_attn_tp, ffn_tp=p_ffn_tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb),
           env_pf, log_dir, prefix, procs)
    time.sleep(2)

    env_pa = _afd_env(env_base, p_cvd, UCX_P, SCHED_P, peer_device=h["p_ffn_base"],
                      ffn_host="127.0.0.1",
                      nvml_indices=_owned_gpus(p_cvd, h["p_attn_base"], p_attn_tp))
    _popen("pa", _afd_cmd(PA_PORT, "attn", "prefill", tp=p_attn_tp,
                          base_gpu_id=h["p_attn_base"], micro_batch=micro_batch,
                          attn_tp=p_attn_tp, ffn_tp=p_ffn_tp, tier=tier, is_pa=True,
                          stats_path=stats_path, dynamic_mb=dynamic_mb),
           env_pa, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", PF_PORT, 300) or \
       not B.wait_port("127.0.0.1", PA_PORT, 300):
        log.error("AF prefill (hetero) failed to start")
        B.cleanup_procs(procs)
        return None

    # Decode: DF(ffn) + DA(attn) sharing d_cvd
    env_df = _afd_env(env_base, d_cvd, UCX_D, SCHED_D, peer_device=h["d_attn_base"],
                      nvml_indices=_owned_gpus(d_cvd, h["d_ffn_base"], d_ffn_tp))
    _popen("df", _afd_cmd(DF_PORT, "ffn", "decode", tp=d_ffn_tp,
                          base_gpu_id=h["d_ffn_base"], micro_batch=micro_batch,
                          attn_tp=d_attn_tp, ffn_tp=d_ffn_tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb,
                          max_running=d_max_running),
           env_df, log_dir, prefix, procs)
    time.sleep(5)

    env_da = _afd_env(env_base, d_cvd, UCX_D, SCHED_D, peer_device=h["d_ffn_base"],
                      ffn_host="127.0.0.1",
                      nvml_indices=_owned_gpus(d_cvd, h["d_attn_base"], d_attn_tp))
    _popen("da", _afd_cmd(DA_PORT, "attn", "decode", tp=d_attn_tp,
                          base_gpu_id=h["d_attn_base"], micro_batch=micro_batch,
                          attn_tp=d_attn_tp, ffn_tp=d_ffn_tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb,
                          max_running=d_max_running),
           env_da, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", DA_PORT, 300) or \
       not B.wait_port("127.0.0.1", DF_PORT, 300):
        log.error("AF decode (hetero) failed to start")
        B.cleanup_procs(procs)
        return None

    time.sleep(5)
    if not start_router(procs, log_dir, prefix, PA_PORT, DA_PORT):
        B.cleanup_procs(procs)
        return None
    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("%s ready at %s, warming up...", deploy, url)
    B.warmup(url)
    return procs, url


def start_dp_full(log_dir, prefix, spec):
    """Start DP=N: independent full SGLang instances with round-robin router.

    Used by native_dp8 / native_dp4. When the dp_full spec sets tier=True each
    single-card instance runs the unified single-knob DVFS controller.
    """
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"

    instances = spec["dp_full"]["instances"]
    tier = spec["dp_full"].get("tier", False)
    worker_urls = []

    for i, inst in enumerate(instances):
        env = env_base.copy()
        env["CUDA_VISIBLE_DEVICES"] = inst["cvd"]
        dvfs = []
        if tier:
            env["AFD_NVML_DEVICE_INDICES"] = inst["cvd"]
            env["AFD_NVML_DEVICE_INDEX"] = inst["cvd"].split(",")[0]
            dvfs = _unified_dvfs_args()
        port = inst["port"]
        cmd = [PYTHON, "-m", "sglang.launch_server",
               "--model-path", MODEL, "--tp", str(inst["tp"]),
               "--host", "127.0.0.1", "--port", str(port),
               "--mem-fraction-static", "0.85",
               "--disable-cuda-graph",
               "--disable-piecewise-cuda-graph",
               "--disable-radix-cache"] + dvfs
        _popen(f"inst{i}", cmd, env, log_dir, prefix, procs)
        worker_urls.append(f"http://127.0.0.1:{port}")

    for i, inst in enumerate(instances):
        if not B.wait_port("127.0.0.1", inst["port"], 300):
            log.error("DP4 instance %d failed to start", i)
            B.cleanup_procs(procs)
            return None

    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
           "--policy", "round_robin",
           "--worker-urls"] + worker_urls
    rf = open(log_dir / f"{prefix}router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health",
                         timeout=180):
        log.error("DP4 router failed to start")
        B.cleanup_procs(procs)
        return None

    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("DP=4 (4x TP=2) ready at %s, warming up...", url)
    B.warmup(url)
    return procs, url


def start_deploy(deploy, log_dir, prefix):
    spec = DEPLOYMENTS[deploy]
    if "native" in spec:
        return start_native_tp8(log_dir, prefix, spec)
    if "dp_full" in spec:
        return start_dp_full(log_dir, prefix, spec)
    if "dp" in spec:
        c = spec["dp"]
        return start_pd_dp(log_dir, prefix, c["instances"], c["tp"],
                           tier=c.get("tier", False))
    if "pd" in spec:
        c = spec["pd"]
        return start_pd(log_dir, prefix, c["p_cvd"], c["p_tp"], c["d_cvd"], c["d_tp"])
    if "asym_pdaf" in spec:
        return start_pdaf_asym(deploy, log_dir, prefix)
    if "hetero_pdaf" in spec:
        return start_pdaf_hetero(deploy, log_dir, prefix)
    return start_pdaf(deploy, log_dir, prefix)


def _print_one(tag, spec, m):
    proc_ttft = m.get("ttft_proc_avg_ms", 0)
    ttft_str = f"TTFT={m['ttft_avg_ms']:.0f}ms(proc={proc_ttft:.0f}ms)" if proc_ttft > 0 else f"TTFT={m['ttft_avg_ms']:.0f}ms"
    log.info("  %s: thpt=%.1f tok/s %s TPOT=%.0fms energy=%.0fJ "
             "(P=%.0f D=%.0f) SLOviol=%.1f%%  [%d GPU, %.0fJ/GPU]",
             tag, m["throughput_tok_s"], ttft_str, m["tpot_avg_ms"],
             m["total_energy_j"], m["prefill_energy_j"], m["decode_energy_j"],
             m["slo_violation_rate"], spec["ngpu"], m["total_energy_j"] / spec["ngpu"])


def _print_summary(sweep):
    print("\n" + "=" * 100)
    print("  8-GPU DEPLOYMENT COMPARISON SUMMARY")
    print("=" * 100)
    for deploy, groups in sweep.items():
        print(f"\n##### {deploy} — {DEPLOYMENTS[deploy]['label']} #####")
        print(f"  {'grp/qps':<25} {'thpt':>9} {'TTFT':>9} {'TPOT':>8} "
              f"{'energy':>9} {'J/tok':>8} {'SLO%':>6}")
        for group, qmap in sorted(groups.items()):
            for qps in sorted(qmap, key=B._qps_sort_key):
                m = qmap[qps]
                print(f"  {group+' q'+str(qps):<25} {m['throughput_tok_s']:>9.1f} "
                      f"{m['ttft_avg_ms']:>9.0f} {m['tpot_avg_ms']:>8.0f} "
                      f"{m['total_energy_j']:>9.0f} {m.get('energy_per_token_mj',0):>8.1f} "
                      f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 100)


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="8-GPU deployment benchmark")
    ap.add_argument("--deploys", type=str,
                    default="pd_p4d4,pdaf_8g_m1,pdaf_8g_m2,pdaf_8g_m1_tier,pdaf_8g_m2_tier",
                    help="Comma-separated topologies")
    ap.add_argument("--workloads", type=str, required=True,
                    help="Comma-separated absolute workload JSONL paths")
    ap.add_argument("--freq", type=str, default="auto", choices=["auto", "max"])
    ap.add_argument("--ttft-slo-ms", type=float, default=5000.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=300.0)
    ap.add_argument("--output-dir", type=str, default="results/8gpu/json")
    ap.add_argument("--log-dir", type=str, default="logs/8gpu")
    ap.add_argument("--max-run-s", type=float, default=600.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-collapse", action="store_true", default=True,
                    help="Stop QPS sweep for a deploy+workload when SLO violation >50%%")
    args = ap.parse_args()

    deploys = [d.strip() for d in args.deploys.split(",") if d.strip()]
    for d in deploys:
        if d not in DEPLOYMENTS:
            ap.error(f"Unknown deploy '{d}'. Valid: {list(DEPLOYMENTS)}")
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir); log_root.mkdir(parents=True, exist_ok=True)

    log.info("8-GPU Bench | Deploys: %s | freq=%s | %d workloads",
             deploys, args.freq, len(workloads))

    global DVFS_TTFT_SLO_MS, DVFS_TPOT_SLO_US
    DVFS_TTFT_SLO_MS = int(args.ttft_slo_ms)
    DVFS_TPOT_SLO_US = int(args.tpot_slo_ms * 1000)

    # Group workloads by (il, ol) so we can stop early per group
    from collections import defaultdict
    wl_groups = defaultdict(list)
    for wl in workloads:
        cfg = B._cfg_from_path(wl)
        wl_groups[cfg["group"]].append((wl, cfg))
    for g in wl_groups:
        wl_groups[g].sort(key=lambda x: B._qps_sort_key(x[1]["qps"]))

    sweep = {}
    collapsed = set()

    for deploy in deploys:
        spec = DEPLOYMENTS[deploy]
        sweep.setdefault(deploy, {})

        for group, wl_list in sorted(wl_groups.items()):
            for wl, cfg in wl_list:
                qps_label = cfg["qps"]
                tag = f"{deploy}_{cfg['tag']}"

                if args.stop_on_collapse and (deploy, group) in collapsed:
                    log.info("Skipping %s (group %s collapsed)", tag, group)
                    continue

                cached = out_dir / f"{tag}_results.json"
                if cached.exists() and not args.force:
                    results = json.loads(cached.read_text())
                    sweep[deploy].setdefault(group, {})[qps_label] = results
                    log.info("Loaded cached %s", cached)
                    if results.get("slo_violation_rate", 0) > 50:
                        collapsed.add((deploy, group))
                    continue

                log.info("=" * 70)
                log.info("DEPLOY %s | %s | il=%s ol=%s QPS=%s",
                         deploy, spec["label"], cfg["il"], cfg["ol"], qps_label)
                log.info("=" * 70)

                kill_servers()
                reset_freq(spec["gpus"])
                time.sleep(2)
                # Non-tier deploys lock to max freq; tier deploys use auto (DVFS controls freq)
                effective_freq = "auto" if deploy.endswith("_tier") else "max"
                apply_freq(spec["gpus"], effective_freq)

                run_log_dir = log_root / tag
                run_log_dir.mkdir(parents=True, exist_ok=True)
                ret = start_deploy(deploy, run_log_dir, f"{tag}_")
                if ret is None:
                    log.error("%s failed to start, skipping", tag)
                    reset_freq(spec["gpus"])
                    continue
                procs, url = ret
                try:
                    results = asyncio.run(B.run_workload(
                        wl, url, ttft_slo_ms=args.ttft_slo_ms,
                        tpot_slo_ms=args.tpot_slo_ms,
                        procs=procs, max_run_s=args.max_run_s,
                        gpu_indices=spec["gpus"],
                        prefill_gpus=spec["prefill_gpus"],
                        decode_gpus=spec["decode_gpus"],
                    ))
                    results.update({"deploy": deploy, "workload": wl,
                                    "il": cfg["il"], "ol": cfg["ol"],
                                    "qps": qps_label, "ngpu": spec["ngpu"]})
                    sweep[deploy].setdefault(group, {})[qps_label] = results
                    if not results.get("aborted"):
                        with open(cached, "w") as f:
                            json.dump(results, f, indent=2, default=str)
                    _print_one(tag, spec, results)
                    if results.get("slo_violation_rate", 0) > 50:
                        collapsed.add((deploy, group))
                        log.info(">>> %s collapsed at QPS=%s (SLO viol %.1f%%), "
                                 "stopping this group", deploy, qps_label,
                                 results["slo_violation_rate"])
                except (KeyboardInterrupt, SystemExit):
                    raise
                except (Exception, asyncio.CancelledError) as e:
                    import traceback
                    log.error("%s crashed: %r\n%s", tag, e, traceback.format_exc())
                finally:
                    B.cleanup_procs(procs)
                    kill_servers()
                    reset_freq(spec["gpus"])
                    time.sleep(5)

    with open(out_dir / "8gpu_summary.json", "w") as f:
        json.dump(sweep, f, indent=2, default=str)
    _print_summary(sweep)
    log.info("All 8-GPU results saved to %s/", out_dir)


if __name__ == "__main__":
    main()
