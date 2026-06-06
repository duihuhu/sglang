#!/usr/bin/env python3
"""4-GPU deployment benchmark: PD TP2, PD DP2, PDAF DynM, PDAF Tier+DynM.
Uses GPUs 4-7. Adapted from run_8gpu_deploy_bench.py."""
import argparse
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fixed_qps_bench as B

HERE = Path(__file__).resolve().parent
PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"

log = logging.getLogger("4gpu_bench")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

# Ports (avoid conflict with 8GPU)
ROUTER_PORT = 54000
PF_PORT = 54011
PA_PORT = 54010
DF_PORT = 54021
DA_PORT = 54020
BOOTSTRAP_PORT = 48999
UCX_P = 18100
UCX_D = 18200
SCHED_P = 18301
SCHED_D = 18302
ALL_PORTS = [ROUTER_PORT, PF_PORT, PA_PORT, DF_PORT, DA_PORT]

# GPU allocation: always use GPUs 4-7
BASE_GPUS = [4, 5, 6, 7]

DEPLOYMENTS = {
    "pd_tp2": {
        "label": "PD P-TP2/D-TP2 (P=4,5 D=6,7, 4 GPU)",
        "gpus": BASE_GPUS,
        "prefill_gpus": [4, 5],
        "decode_gpus": [6, 7],
        "ngpu": 4,
        "pd": {"p_cvd": "4,5", "p_tp": 2, "d_cvd": "6,7", "d_tp": 2},
    },
    "pd_dp2": {
        "label": "DP=2 (2x TP=2 full instances, round-robin, 4 GPU)",
        "gpus": BASE_GPUS,
        "prefill_gpus": BASE_GPUS,
        "decode_gpus": BASE_GPUS,
        "ngpu": 4,
        "dp_full": {
            "instances": [
                {"cvd": "4,5", "tp": 2, "port": 54100},
                {"cvd": "6,7", "tp": 2, "port": 54110},
            ],
        },
    },
    "pdaf_4g_dyn": {
        "label": "PD+AF DynM (PA/PF TP1 + DA/DF TP1, P=4,5 D=6,7, 4 GPU)",
        "gpus": BASE_GPUS,
        "prefill_gpus": [4, 5],
        "decode_gpus": [6, 7],
        "ngpu": 4,
        "af_p_vis": "4,5",
        "af_d_cvd": "6,7",
        "micro_batch": 2,
        "af_tp": 1,
        "dynamic_mb": True,
    },
    "pdaf_4g_dyn_tier": {
        "label": "PD+AF DynM + Tier1/DVFS (PA/PF TP1 + DA/DF TP1, P=4,5 D=6,7, 4 GPU)",
        "gpus": BASE_GPUS,
        "prefill_gpus": [4, 5],
        "decode_gpus": [6, 7],
        "ngpu": 4,
        "af_p_vis": "4,5",
        "af_d_cvd": "6,7",
        "micro_batch": 2,
        "af_tp": 1,
        "dynamic_mb": True,
    },
    "pd_dp2_disagg": {
        "label": "PD DP=2 (2x 1P1D: P=4/D=5, P=6/D=7, TP=1, 4 GPU)",
        "gpus": BASE_GPUS,
        "prefill_gpus": [4, 6],
        "decode_gpus": [5, 7],
        "ngpu": 4,
        "dp": {"instances": [{"p_cvd": "4", "d_cvd": "5"},
                             {"p_cvd": "6", "d_cvd": "7"}], "tp": 1},
    },
    "pd_dp2_disagg_tier": {
        "label": "PD DP=2 + Tier DVFS (2x 1P1D: P=4/D=5, P=6/D=7, TP=1, 4 GPU)",
        "gpus": BASE_GPUS,
        "prefill_gpus": [4, 6],
        "decode_gpus": [5, 7],
        "ngpu": 4,
        "dp": {"instances": [{"p_cvd": "4", "d_cvd": "5"},
                             {"p_cvd": "6", "d_cvd": "7"}], "tp": 1,
               "tier": True},
    },
    "native_dp4": {
        "label": "Native DP=4 (4x TP=1 full instances, round-robin, 4 GPU)",
        "gpus": BASE_GPUS,
        "prefill_gpus": BASE_GPUS,
        "decode_gpus": BASE_GPUS,
        "ngpu": 4,
        "dp_full": {
            "instances": [
                {"cvd": "4", "tp": 1, "port": 54120},
                {"cvd": "5", "tp": 1, "port": 54130},
                {"cvd": "6", "tp": 1, "port": 54140},
                {"cvd": "7", "tp": 1, "port": 54150},
            ],
        },
    },
    "native_dp4_tier": {
        "label": "Native DP=4 + Tier DVFS (4x TP=1 instances, round-robin, 4 GPU)",
        "gpus": BASE_GPUS,
        "prefill_gpus": BASE_GPUS,
        "decode_gpus": BASE_GPUS,
        "ngpu": 4,
        "dp_full": {
            "tier": True,
            "instances": [
                {"cvd": "4", "tp": 1, "port": 54120},
                {"cvd": "5", "tp": 1, "port": 54130},
                {"cvd": "6", "tp": 1, "port": 54140},
                {"cvd": "7", "tp": 1, "port": 54150},
            ],
        },
    },
}


def apply_freq(gpus, freq: str):
    for idx in gpus:
        if freq == "max":
            subprocess.run(["nvidia-smi", "-lgc",
                            f"{B.MAX_SM_FREQ_MHZ},{B.MAX_SM_FREQ_MHZ}",
                            "-i", str(idx)], capture_output=True)
        else:
            subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)],
                           capture_output=True)
    log.info("Freq policy '%s' applied to GPUs %s", freq, gpus)


def reset_freq(gpus):
    for idx in gpus:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)],
                       capture_output=True)


def kill_servers():
    for port in ALL_PORTS + [54100, 54110]:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True)
        for m in re.finditer(r'pid=(\d+)', r.stdout):
            subprocess.run(["kill", "-9", m.group(1)], capture_output=True)
    time.sleep(3)


def _popen(name, cmd, env, log_dir, prefix, procs):
    fh = open(log_dir / f"{prefix}{name}.log", "w")
    p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                         start_new_session=True)
    procs.append((name, p, fh))
    log.info("  Started %s (CVD=%s, port via cmd)",
             name, env.get("CUDA_VISIBLE_DEVICES"))
    return p


# ─── PD TP2 ───────────────────────────────────────────────────────────
_PD_COMMON = [
    "--mem-fraction-static", "0.85",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", str(BOOTSTRAP_PORT),
    "--disaggregation-ib-device", "mlx5_4",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--disable-radix-cache",
]


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

    # Router
    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--pd-disaggregation", "--mini-lb",
           "--prefill", f"http://127.0.0.1:{PF_PORT}",
           "--decode", f"http://127.0.0.1:{DF_PORT}",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    rf = open(log_dir / f"{prefix}router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("Router failed to start")
        B.cleanup_procs(procs)
        return None

    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PD (P-tp%d/D-tp%d) ready at %s, warming up...", p_tp, d_tp, url)
    B.warmup(url)
    return procs, url


# ─── PD DP2 ───────────────────────────────────────────────────────────
def start_pd_dp2(log_dir, prefix, spec, tpot_slo_ms=None, ttft_slo_ms=None):
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
            dvfs = _unified_dvfs_args(tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms)
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
            log.error("DP2 instance %d failed to start", i)
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
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("DP2 router failed to start")
        B.cleanup_procs(procs)
        return None

    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("DP=2 (2x TP=2) ready at %s, warming up...", url)
    B.warmup(url)
    return procs, url


# ─── PD DP2 Disagg (2x 1P1D, TP=1) ────────────────────────────────────
_PD_DISAGG_BOOTSTRAP_BASE = 24100
_PD_DISAGG_PORTS = [(54200, 54201), (54210, 54211)]  # (prefill_port, decode_port) per inst

_PD_COMMON_ARGS = [
    "--model-path", MODEL,
    "--mem-fraction-static", "0.8",
    "--chunked-prefill-size", "8192",
    "--max-running-requests", "256",
    "--disable-cuda-graph",
    "--enable-metrics",
]


def start_pd_dp_disagg(log_dir, prefix, instances, tp, tier=False,
                       tpot_slo_ms=None, ttft_slo_ms=None):
    """PD DP: N independent 1P1D instances with router load-balancing.

    When tier=True, each prefill/decode server runs the unified single-knob
    DVFS controller locked to its own physical GPU.
    """
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    prefill_specs, decode_urls = [], []

    for i, inst in enumerate(instances):
        pf_port, df_port = _PD_DISAGG_PORTS[i]
        bport = _PD_DISAGG_BOOTSTRAP_BASE + i

        common_pd = [
            "--disaggregation-ib-device", "mlx5_0",
            "--disaggregation-bootstrap-port", str(bport),
        ]
        dvfs = (_unified_dvfs_args(tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms)
                if tier else [])

        env_p = env_base.copy()
        env_p["CUDA_VISIBLE_DEVICES"] = inst["p_cvd"]
        if tier:
            env_p["AFD_NVML_DEVICE_INDICES"] = inst["p_cvd"]
            env_p["AFD_NVML_DEVICE_INDEX"] = inst["p_cvd"].split(",")[0]
        p_cmd = ([PYTHON, "-m", "sglang.launch_server"] + _PD_COMMON_ARGS + [
            "--tp", str(tp),
            "--host", "127.0.0.1", "--port", str(pf_port),
            "--disaggregation-mode", "prefill",
        ] + common_pd + dvfs)
        log.info("  Started prefill%d (CVD=%s, port=%d)", i, inst["p_cvd"], pf_port)
        pf_log = os.path.join(log_dir, f"{prefix}_prefill{i}.log")
        fp = open(pf_log, "w")
        pp = subprocess.Popen(p_cmd, env=env_p, stdout=fp, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((f"prefill{i}", pp, fp))

        env_d = env_base.copy()
        env_d["CUDA_VISIBLE_DEVICES"] = inst["d_cvd"]
        if tier:
            env_d["AFD_NVML_DEVICE_INDICES"] = inst["d_cvd"]
            env_d["AFD_NVML_DEVICE_INDEX"] = inst["d_cvd"].split(",")[0]
        d_cmd = ([PYTHON, "-m", "sglang.launch_server"] + _PD_COMMON_ARGS + [
            "--tp", str(tp),
            "--host", "127.0.0.1", "--port", str(df_port),
            "--disaggregation-mode", "decode",
        ] + common_pd + dvfs)
        log.info("  Started decode%d (CVD=%s, port=%d)", i, inst["d_cvd"], df_port)
        df_log = os.path.join(log_dir, f"{prefix}_decode{i}.log")
        fd = open(df_log, "w")
        dp = subprocess.Popen(d_cmd, env=env_d, stdout=fd, stderr=subprocess.STDOUT,
                             start_new_session=True)
        procs.append((f"decode{i}", dp, fd))

        prefill_specs.append((f"http://127.0.0.1:{pf_port}", bport))
        decode_urls.append(f"http://127.0.0.1:{df_port}")

    # Wait for all servers
    for url, _ in prefill_specs:
        port = int(url.rsplit(":", 1)[1])
        if not B.wait_port("127.0.0.1", port, timeout=300):
            log.error("PD-DP prefill %s failed to start", url)
            B.cleanup_procs(procs)
            return None
    for url in decode_urls:
        port = int(url.rsplit(":", 1)[1])
        if not B.wait_port("127.0.0.1", port, timeout=300):
            log.error("PD-DP decode %s failed to start", url)
            B.cleanup_procs(procs)
            return None

    # Start router (PD-disaggregation mode: separate prefill/decode URLs)
    router_cmd = [PYTHON, "-m", "sglang_router.launch_router",
                  "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
                  "--pd-disaggregation", "--mini-lb"]
    for purl, bport in prefill_specs:
        router_cmd += ["--prefill", purl, str(bport)]
    for durl in decode_urls:
        router_cmd += ["--decode", durl]
    rl = os.path.join(log_dir, f"{prefix}_router.log")
    rf = open(rl, "w")
    rp = subprocess.Popen(router_cmd, env=env_base, stdout=rf, stderr=subprocess.STDOUT,
                          start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("PD-DP router failed to start")
        B.cleanup_procs(procs)
        return None

    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PD DP=%d (each tp%d) ready at %s, warming up...", len(instances), tp, url)
    B.warmup(url)
    return procs, url


# ─── PDAF (4GPU) ──────────────────────────────────────────────────────
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


def _dvfs_args(tpot_slo_us=None, ttft_slo_ms=None):
    slo = str(int(tpot_slo_us)) if tpot_slo_us else "300000"
    ttft = str(int(ttft_slo_ms)) if ttft_slo_ms else "5000"
    return ["--afd-dvfs-enabled",
            "--afd-energy-model-dir", B.ENERGY_MODEL_DIR,
            "--afd-ttft-slo-ms", ttft,
            "--afd-tpot-slo-us", slo]


def _unified_dvfs_args(tpot_slo_ms=None, ttft_slo_ms=None):
    """Unified single-knob DVFS args for PD / Native (non-AF) instances."""
    slo = str(int(tpot_slo_ms * 1000)) if tpot_slo_ms else "300000"
    ttft = str(int(ttft_slo_ms)) if ttft_slo_ms else "5000"
    return ["--dvfs-enabled",
            "--dvfs-energy-model-dir", B.ENERGY_MODEL_DIR,
            "--dvfs-ttft-slo-ms", ttft,
            "--dvfs-tpot-slo-us", slo]


def _tier1_stats_path():
    d = HERE / "results" / "tier1_shared"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "4gpu_decode_stats.json")


def _tier1_pa_args(stats_path):
    return ["--enable-tier1-pa", "--tier1-disable-reload",
            "--tier1-monitor-window-s", "15", "--tier1-gpu-count", "4",
            "--tier1-stats-path", stats_path,
            "--tier1-prefill-data-path",
            "/workspace/sglang/benchmark/test_motivation/hucc/paper/prefill_data_v1.txt",
            "--tier1-decode-data-path",
            "/workspace/sglang/benchmark/test_motivation/hucc/paper/decode_data_v1.txt"]


def _owned_gpus(cvd, base_gpu_id, tp):
    cvd_list = [x.strip() for x in str(cvd).split(",") if x.strip()]
    return cvd_list[base_gpu_id:base_gpu_id + tp]


def _afd_env(env_base, cvd, ucx_base, sched_port, peer_device,
             ffn_host=None, nvml_indices=None):
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
             attn_tp=None, ffn_tp=None, tier=False, is_pa=False,
             stats_path=None, dynamic_mb=False, tpot_slo_ms=None,
             ttft_slo_ms=None):
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
    if tier:
        slo_us = int(tpot_slo_ms * 1000) if tpot_slo_ms else None
        cmd += _dvfs_args(tpot_slo_us=slo_us, ttft_slo_ms=ttft_slo_ms)
        if is_pa:
            cmd += _tier1_pa_args(stats_path)
        elif stats_path:
            cmd += ["--tier1-stats-path", stats_path]
    return cmd


def start_pdaf(deploy, log_dir, prefix, tpot_slo_ms=None, ttft_slo_ms=None):
    """Start 4-GPU PD+AF: PF(TP1)+PA(TP1) on GPUs 4,5; DF(TP1)+DA(TP1) on GPUs 6,7.

    GPU layout:
      Prefill: PF(ffn, TP1, base=0) + PA(attn, TP1, base=1) sharing CVD=4,5
      Decode:  DF(ffn, TP1, base=0) + DA(attn, TP1, base=1) sharing CVD=6,7
    """
    tier = deploy.endswith("_tier")
    stats_path = _tier1_stats_path() if tier else None
    spec = DEPLOYMENTS[deploy]
    micro_batch = spec["micro_batch"]
    tp = spec["af_tp"]  # 1
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

    # Prefill: PF(ffn, TP1, base=0) + PA(attn, TP1, base=1) on CVD=4,5
    env_pf = _afd_env(env_base, p_vis, UCX_P, SCHED_P, peer_device=1,
                      nvml_indices=_owned_gpus(p_vis, 0, tp))
    _popen("pf", _afd_cmd(PF_PORT, "ffn", "prefill", tp=tp, base_gpu_id=0,
                          micro_batch=micro_batch, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb,
                          tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms),
           env_pf, log_dir, prefix, procs)
    time.sleep(2)

    env_pa = _afd_env(env_base, p_vis, UCX_P, SCHED_P, peer_device=0,
                      ffn_host="127.0.0.1",
                      nvml_indices=_owned_gpus(p_vis, 1, tp))
    _popen("pa", _afd_cmd(PA_PORT, "attn", "prefill", tp=tp, base_gpu_id=1,
                          micro_batch=micro_batch, tier=tier, is_pa=True,
                          stats_path=stats_path, dynamic_mb=dynamic_mb,
                          tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms),
           env_pa, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", PF_PORT, 300) or \
       not B.wait_port("127.0.0.1", PA_PORT, 300):
        log.error("AF prefill failed to start")
        B.cleanup_procs(procs)
        return None

    # Decode: DF(ffn, TP1, base=0) + DA(attn, TP1, base=1) on CVD=6,7
    env_df = _afd_env(env_base, d_cvd, UCX_D, SCHED_D, peer_device=1,
                      nvml_indices=_owned_gpus(d_cvd, 0, tp))
    _popen("df", _afd_cmd(DF_PORT, "ffn", "decode", tp=tp, base_gpu_id=0,
                          micro_batch=micro_batch, attn_tp=tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb,
                          tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms),
           env_df, log_dir, prefix, procs)
    time.sleep(5)

    env_da = _afd_env(env_base, d_cvd, UCX_D, SCHED_D, peer_device=0,
                      ffn_host="127.0.0.1",
                      nvml_indices=_owned_gpus(d_cvd, 1, tp))
    _popen("da", _afd_cmd(DA_PORT, "attn", "decode", tp=tp, base_gpu_id=1,
                          micro_batch=micro_batch, ffn_tp=tp, tier=tier,
                          stats_path=stats_path, dynamic_mb=dynamic_mb,
                          tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms),
           env_da, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", DA_PORT, 300) or \
       not B.wait_port("127.0.0.1", DF_PORT, 300):
        log.error("AF decode failed to start")
        B.cleanup_procs(procs)
        return None

    # PD router for PA (prefill attn serves as PD prefill endpoint)
    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--pd-disaggregation", "--mini-lb",
           "--prefill", f"http://127.0.0.1:{PA_PORT}",
           "--decode", f"http://127.0.0.1:{DA_PORT}",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    rf = open(log_dir / f"{prefix}router.log", "w")
    rp = subprocess.Popen(cmd, env=os.environ.copy(), stdout=rf,
                          stderr=subprocess.STDOUT, start_new_session=True)
    procs.append(("router", rp, rf))
    if not B.wait_health(f"http://127.0.0.1:{ROUTER_PORT}/health", timeout=180):
        log.error("PDAF router failed to start")
        B.cleanup_procs(procs)
        return None

    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PDAF 4GPU ready at %s, warming up...", url)
    B.warmup(url)
    return procs, url


# ─── Dispatch ─────────────────────────────────────────────────────────
def start_deploy(deploy, log_dir, prefix, tpot_slo_ms=None, ttft_slo_ms=None):
    spec = DEPLOYMENTS[deploy]
    if "dp" in spec:
        dp = spec["dp"]
        return start_pd_dp_disagg(log_dir, prefix, dp["instances"], dp["tp"],
                                  tier=dp.get("tier", False),
                                  tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms)
    elif "pd" in spec:
        pd = spec["pd"]
        return start_pd(log_dir, prefix, pd["p_cvd"], pd["p_tp"],
                        pd["d_cvd"], pd["d_tp"])
    elif "dp_full" in spec:
        return start_pd_dp2(log_dir, prefix, spec,
                            tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms)
    else:
        return start_pdaf(deploy, log_dir, prefix, tpot_slo_ms=tpot_slo_ms, ttft_slo_ms=ttft_slo_ms)


def effective_freq(deploy, user_freq):
    if deploy.endswith("_tier"):
        return "auto"
    return "max"


def _print_one(tag, spec, m):
    ngpu = spec["ngpu"]
    thpt = m.get("throughput_tok_s", 0)
    ttft = m.get("ttft_avg_ms", 0)
    tpot = m.get("tpot_avg_ms", 0)
    energy = m.get("total_energy_j", 0)
    pe = m.get("prefill_energy_j", 0)
    de = m.get("decode_energy_j", 0)
    slo = m.get("slo_violation_rate", 0)
    per_gpu = energy / ngpu if ngpu else 0
    log.info("  %s: thpt=%.1f tok/s TTFT=%dms TPOT=%dms energy=%dJ "
             "(P=%d D=%d) SLOviol=%.1f%%  [%d GPU, %dJ/GPU]",
             tag, thpt, ttft, tpot, energy, pe, de, slo, ngpu, per_gpu)


def _apply_gpu_base(base: int):
    """Reconfigure module-level GPU and port constants for a different 4-GPU group."""
    global BASE_GPUS, ROUTER_PORT, PF_PORT, PA_PORT, DF_PORT, DA_PORT
    global BOOTSTRAP_PORT, UCX_P, UCX_D, SCHED_P, SCHED_D, ALL_PORTS, DEPLOYMENTS

    gpus = [base, base + 1, base + 2, base + 3]
    BASE_GPUS = gpus

    if base == 0:
        ROUTER_PORT = 55000
        PF_PORT = 55011
        PA_PORT = 55010
        DF_PORT = 55021
        DA_PORT = 55020
        BOOTSTRAP_PORT = 49999
        UCX_P = 19100
        UCX_D = 19200
        SCHED_P = 19301
        SCHED_D = 19302
    else:
        ROUTER_PORT = 54000
        PF_PORT = 54011
        PA_PORT = 54010
        DF_PORT = 54021
        DA_PORT = 54020
        BOOTSTRAP_PORT = 48999
        UCX_P = 18100
        UCX_D = 18200
        SCHED_P = 18301
        SCHED_D = 18302

    ALL_PORTS = [ROUTER_PORT, PF_PORT, PA_PORT, DF_PORT, DA_PORT]

    g0, g1, g2, g3 = gpus
    vis_p = f"{g0},{g1}"
    vis_d = f"{g2},{g3}"

    DEPLOYMENTS["pd_tp2"]["gpus"] = gpus
    DEPLOYMENTS["pd_tp2"]["prefill_gpus"] = [g0, g1]
    DEPLOYMENTS["pd_tp2"]["decode_gpus"] = [g2, g3]
    DEPLOYMENTS["pd_tp2"]["pd"] = {"p_cvd": vis_p, "p_tp": 2, "d_cvd": vis_d, "d_tp": 2}
    DEPLOYMENTS["pd_tp2"]["label"] = f"PD P-TP2/D-TP2 (P={g0},{g1} D={g2},{g3}, 4 GPU)"

    DEPLOYMENTS["pd_dp2"]["gpus"] = gpus
    DEPLOYMENTS["pd_dp2"]["prefill_gpus"] = gpus
    DEPLOYMENTS["pd_dp2"]["decode_gpus"] = gpus

    DEPLOYMENTS["pdaf_4g_dyn"]["gpus"] = gpus
    DEPLOYMENTS["pdaf_4g_dyn"]["prefill_gpus"] = [g0, g1]
    DEPLOYMENTS["pdaf_4g_dyn"]["decode_gpus"] = [g2, g3]
    DEPLOYMENTS["pdaf_4g_dyn"]["af_p_vis"] = vis_p
    DEPLOYMENTS["pdaf_4g_dyn"]["af_d_cvd"] = vis_d
    DEPLOYMENTS["pdaf_4g_dyn"]["label"] = f"PD+AF DynM (PA/PF TP1 + DA/DF TP1, P={g0},{g1} D={g2},{g3}, 4 GPU)"

    DEPLOYMENTS["pdaf_4g_dyn_tier"]["gpus"] = gpus
    DEPLOYMENTS["pdaf_4g_dyn_tier"]["prefill_gpus"] = [g0, g1]
    DEPLOYMENTS["pdaf_4g_dyn_tier"]["decode_gpus"] = [g2, g3]
    DEPLOYMENTS["pdaf_4g_dyn_tier"]["af_p_vis"] = vis_p
    DEPLOYMENTS["pdaf_4g_dyn_tier"]["af_d_cvd"] = vis_d
    DEPLOYMENTS["pdaf_4g_dyn_tier"]["label"] = f"PD+AF DynM + Tier1/DVFS (PA/PF TP1 + DA/DF TP1, P={g0},{g1} D={g2},{g3}, 4 GPU)"


# ─── Main sweep logic ─────────────────────────────────────────────────
def main():
    import asyncio
    from collections import defaultdict

    ap = argparse.ArgumentParser(description="4-GPU deployment benchmark")
    ap.add_argument("--deploys", type=str, required=True)
    ap.add_argument("--workloads", type=str, required=True)
    ap.add_argument("--freq", type=str, default="max")
    ap.add_argument("--ttft-slo-ms", type=float, default=5000.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=300.0)
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--log-dir", type=str, required=True)
    ap.add_argument("--max-run-s", type=float, default=600.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--stop-on-collapse", action="store_true", default=False)
    ap.add_argument("--no-stop-on-collapse", dest="stop_on_collapse", action="store_false")
    ap.add_argument("--gpu-base", type=int, default=None,
                    help="Override GPU base index (0 or 4). Adjusts all GPU/port configs.")
    args = ap.parse_args()

    if args.gpu_base is not None:
        _apply_gpu_base(args.gpu_base)

    deploys = [d.strip() for d in args.deploys.split(",") if d.strip()]
    for d in deploys:
        if d not in DEPLOYMENTS:
            ap.error(f"Unknown deploy '{d}'. Valid: {list(DEPLOYMENTS)}")
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir); log_root.mkdir(parents=True, exist_ok=True)

    log.info("4-GPU Bench | Deploys: %s | freq=%s | %d workloads",
             deploys, args.freq, len(workloads))

    wl_groups = defaultdict(list)
    for wl in workloads:
        cfg = B._cfg_from_path(wl)
        wl_groups[cfg["group"]].append((wl, cfg))
    for g in wl_groups:
        def _sort_key(x):
            try:
                return float(x[1]["qps"])
            except (ValueError, TypeError):
                return 0.0
        wl_groups[g].sort(key=_sort_key)

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
                freq_policy = "auto" if deploy.endswith("_tier") else "max"
                apply_freq(spec["gpus"], freq_policy)

                run_log_dir = log_root / tag
                run_log_dir.mkdir(parents=True, exist_ok=True)
                ret = start_deploy(deploy, run_log_dir, f"{tag}_",
                                   tpot_slo_ms=args.tpot_slo_ms,
                                   ttft_slo_ms=args.ttft_slo_ms)
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
                    reset_freq(spec.get("decode_gpus", spec["gpus"]))
                    time.sleep(5)

    # Save summary
    summary_file = out_dir / "4gpu_summary.json"
    with open(summary_file, "w") as f:
        json.dump(sweep, f, indent=2, default=str)

    # Print comparison table
    print("\n" + "=" * 100)
    print("  4-GPU DEPLOYMENT COMPARISON SUMMARY")
    print("=" * 100)
    for deploy in deploys:
        print(f"\n##### {deploy} — {DEPLOYMENTS[deploy]['label']} #####")
        print(f"  {'grp/qps':<30} {'thpt':>8} {'TTFT':>8} {'TPOT':>8}"
              f" {'energy':>10} {'J/tok':>8} {'SLO%':>6}")
        for group, qps_data in sorted(sweep.get(deploy, {}).items()):
            def _qps_sort(x):
                try:
                    return float(x[0])
                except (ValueError, TypeError):
                    return x[0]
            for qps, m in sorted(qps_data.items(), key=_qps_sort):
                label = f"{group} q{qps}"
                print(f"  {label:<30} {m.get('throughput_tok_s',0):>8.1f}"
                      f" {m.get('ttft_avg_ms',0):>8.0f}"
                      f" {m.get('tpot_avg_ms',0):>8.0f}"
                      f" {m.get('total_energy_j',0):>10.0f}"
                      f" {m.get('energy_per_token_mj',0):>8.1f}"
                      f" {m.get('slo_violation_rate',0):>6.1f}")
    print("=" * 100)
    log.info("All 4-GPU results saved to %s/", out_dir)


if __name__ == "__main__":
    main()
