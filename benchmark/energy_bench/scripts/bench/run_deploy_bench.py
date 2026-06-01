#!/usr/bin/env python3
"""Deployment-topology comparison benchmark (fixed-length, variable-QPS).

Reuses the validated workload / energy / SLO machinery from
``run_fixed_qps_bench.py`` but parameterizes the *deployment topology*:

    - pd_tp2     : standard PD disaggregation, Prefill TP=2 + Decode TP=2 (4 GPU)
    - pdaf_m2    : PD+AF baseline, PA+PF+DA+DF each TP=1, M=2 micro-batch (4 GPU)
    - pdaf_df2   : PD+AF, DF TP=2 (heterogeneous, 5 GPU)
    - pdaf_da2   : PD+AF, DA TP=2 (heterogeneous, 5 GPU)
    - pdaf_d_tp2 : PD+AF, DA TP=2 + DF TP=2 (6 GPU)

All topologies are benchmarked at a fixed hardware frequency policy
(auto-boost by default) so the comparison isolates the *deployment* variable,
not DVFS. Each topology declares its own GPU map so the per-GPU NVML energy is
attributed to the prefill / decode stages correctly.

Usage:
    python run_deploy_bench.py --deploys pd_tp2,pdaf_m2 \
        --workloads /abs/path/fixed_il512_ol128_qps2.jsonl,... \
        --freq auto --max-run-s 360
"""
import argparse
import asyncio
import json
import logging
import os
import subprocess
import time
from pathlib import Path

# Reuse the validated helpers from the main benchmark module.
import run_fixed_qps_bench as B

log = logging.getLogger("deploy_bench")

PYTHON = B.PYTHON
MODEL = B.MODEL
HERE = Path(__file__).resolve().parent

# Ports (kept distinct from run_fixed_qps_bench defaults to avoid collisions).
ROUTER_PORT = 52000
PA_PORT, PF_PORT = 52010, 52011
DA_PORT, DF_PORT = 52020, 52021
BOOTSTRAP_PORT = 39999
UCX_P, UCX_D = 27200, 27300
SCHED_P, SCHED_D = 67400, 67500

# ── Deployment registry ──────────────────────────────────────────────────
# Each entry declares which physical GPUs it uses and how to split them into
# prefill vs decode for per-stage energy attribution. `start` is filled in
# after the start functions are defined.
DEPLOYMENTS = {
    # ── Standard PD disaggregation (stable, no AF) — card-count variants ──
    "pd_p1d1": {
        "label": "PD P-tp1/D-tp1 (P=GPU7 D=GPU4, 2 GPU)",
        "gpus": [4, 7], "prefill_gpus": [7], "decode_gpus": [4], "ngpu": 2,
        "pd": {"p_cvd": "7", "p_tp": 1, "d_cvd": "4", "d_tp": 1},
    },
    "pd_p1d2": {
        "label": "PD P-tp1/D-tp2 (P=GPU7 D=GPU4,5, 3 GPU)",
        "gpus": [4, 5, 7], "prefill_gpus": [7], "decode_gpus": [4, 5], "ngpu": 3,
        "pd": {"p_cvd": "7", "p_tp": 1, "d_cvd": "4,5", "d_tp": 2},
    },
    "pd_p2d1": {
        "label": "PD P-tp2/D-tp1 (P=GPU6,7 D=GPU4, 3 GPU)",
        "gpus": [4, 6, 7], "prefill_gpus": [6, 7], "decode_gpus": [4], "ngpu": 3,
        "pd": {"p_cvd": "6,7", "p_tp": 2, "d_cvd": "4", "d_tp": 1},
    },
    "pd_p2d2": {
        "label": "PD P-tp2/D-tp2 (P=GPU6,7 D=GPU4,5, 4 GPU)",
        "gpus": [4, 5, 6, 7], "prefill_gpus": [6, 7], "decode_gpus": [4, 5], "ngpu": 4,
        "pd": {"p_cvd": "6,7", "p_tp": 2, "d_cvd": "4,5", "d_tp": 2},
    },
    "pd_dp2": {
        "label": "PD DP=2 (2x 1P1D: inst0 P=0/D=1, inst1 P=2/D=3, 4 GPU)",
        "gpus": [0, 1, 2, 3], "prefill_gpus": [0, 2], "decode_gpus": [1, 3], "ngpu": 4,
        "dp": {"instances": [{"p_cvd": "0", "d_cvd": "1"},
                             {"p_cvd": "2", "d_cvd": "3"}], "tp": 1},
    },
    "pd_p2d4": {
        "label": "PD P-tp2/D-tp4 (P=GPU6,7 D=GPU0,1,2,3, 6 GPU)",
        "gpus": [0, 1, 2, 3, 6, 7], "prefill_gpus": [6, 7],
        "decode_gpus": [0, 1, 2, 3], "ngpu": 6,
        "pd": {"p_cvd": "6,7", "p_tp": 2, "d_cvd": "0,1,2,3", "d_tp": 4},
    },
    # ── PD+AF (operator-level disagg) — stable homogeneous variants ──
    "pdaf_m2": {
        "label": "PD+AF M=2 (PA7 PF6 DA5 DF4, each TP=1, 4 GPU)",
        "gpus": [4, 5, 6, 7], "prefill_gpus": [6, 7], "decode_gpus": [4, 5], "ngpu": 4,
    },
    "pdaf_m2_tier": {
        "label": "PD+AF M=2 + Tier1/DVFS (PA1 PF0 DA3 DF2, each TP=1, 4 GPU)",
        "gpus": [0, 1, 2, 3], "prefill_gpus": [0, 1], "decode_gpus": [2, 3], "ngpu": 4,
        "af_p_vis": "0,1", "af_d_cvd": "2,3",
    },
    "pdaf_d_tp2": {
        "label": "PD+AF Decode TP=2 (PA7 PF6 DA=2,3 DF=4,5, 6 GPU)",
        "gpus": [2, 3, 4, 5, 6, 7], "prefill_gpus": [6, 7],
        "decode_gpus": [2, 3, 4, 5], "ngpu": 6,
    },
    # ── PD+AF M=1 variants (no micro-batch pipeline) ──
    "pdaf_m1": {
        "label": "PD+AF M=1 (PA1 PF0 DA3 DF2, each TP=1, 4 GPU)",
        "gpus": [0, 1, 2, 3], "prefill_gpus": [0, 1], "decode_gpus": [2, 3], "ngpu": 4,
        "af_p_vis": "0,1", "af_d_cvd": "2,3", "micro_batch": 1,
    },
    "pdaf_m1_tier": {
        "label": "PD+AF M=1 + Tier1/DVFS (PA1 PF0 DA3 DF2, each TP=1, 4 GPU)",
        "gpus": [0, 1, 2, 3], "prefill_gpus": [0, 1], "decode_gpus": [2, 3], "ngpu": 4,
        "af_p_vis": "0,1", "af_d_cvd": "2,3", "micro_batch": 1,
    },
}

# ── Frequency + process helpers ──────────────────────────────────────────


def apply_freq(gpus, freq: str):
    """Apply a hardware frequency policy to the deployment's GPUs."""
    for idx in gpus:
        if freq == "max":
            subprocess.run(["nvidia-smi", "-lgc", f"{B.MAX_SM_FREQ_MHZ},{B.MAX_SM_FREQ_MHZ}",
                            "-i", str(idx)], capture_output=True)
        else:  # auto
            subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)
    log.info("Freq policy '%s' applied to GPUs %s", freq, gpus)


def reset_freq(gpus):
    for idx in gpus:
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(idx)], capture_output=True)


ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT,
             PF_PORT + 100, PF_PORT + 102, DF_PORT + 100, DF_PORT + 102]


def kill_servers():
    """Kill any processes listening on our ports."""
    import re
    for port in ALL_PORTS:
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
    log.info("  Started %s (CVD=%s, port via cmd)", name, env.get("CUDA_VISIBLE_DEVICES"))
    return p


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


def start_router_multi(procs, log_dir, prefix, prefill_specs, decode_urls,
                       policy="round_robin"):
    """PD router across multiple instances (DP).

    prefill_specs: list of (url, bootstrap_port) tuples.
    decode_urls:   list of decode urls.
    Uses the full router (not mini-lb) so multi-endpoint load balancing works.
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


# [START_FUNCS]

# ── Start functions (one per topology family) ────────────────────────────

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
    """Standard PD: Prefill (p_tp on p_cvd) + Decode (d_tp on d_cvd)."""
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


def _pd_common_bootstrap(bootstrap_port):
    """_PD_COMMON with a per-instance bootstrap port (for DP multi-instance)."""
    out = []
    skip_next = False
    for i, tok in enumerate(_PD_COMMON):
        if skip_next:
            out.append(str(bootstrap_port)); skip_next = False; continue
        out.append(tok)
        if tok == "--disaggregation-bootstrap-port":
            skip_next = True
    return out


def start_pd_dp(log_dir, prefix, instances, tp):
    """PD DP: N independent 1P1D instances, router load-balances across them.

    Each instance is a self-contained prefill+decode disaggregation pair with
    its own ports and bootstrap port. The full router (round_robin) fans
    requests across instances.
    """
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    prefill_specs, decode_urls = [], []

    for i, inst in enumerate(instances):
        pf_port, df_port = PF_PORT + 100 + i * 2, DF_PORT + 100 + i * 2
        bport = BOOTSTRAP_PORT + 1 + i
        common = _pd_common_bootstrap(bport)

        env_p = env_base.copy(); env_p["CUDA_VISIBLE_DEVICES"] = inst["p_cvd"]
        p_cmd = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", str(tp),
                 "--host", "127.0.0.1", "--port", str(pf_port),
                 "--disaggregation-mode", "prefill"] + common
        _popen(f"prefill{i}", p_cmd, env_p, log_dir, prefix, procs)

        env_d = env_base.copy(); env_d["CUDA_VISIBLE_DEVICES"] = inst["d_cvd"]
        d_cmd = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", str(tp),
                 "--host", "127.0.0.1", "--port", str(df_port),
                 "--disaggregation-mode", "decode"] + common
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
    log.info("PD DP=%d (each tp%d) ready at %s, warming up...",
             len(instances), tp, url)
    B.warmup(url)
    return procs, url


# ── PD+AF start (parameterized for homo/hetero decode TP) ────────────────


def _afd_env(env_base, cvd, ucx_base, sched_port, peer_device, ffn_host=None):
    env = env_base.copy()
    env["CUDA_VISIBLE_DEVICES"] = cvd
    env["AFD_UCX_BASE_PORT"] = str(ucx_base)
    env["AFD_SCHED_PORT"] = str(sched_port)
    env["AFD_IPC_SYNC_MODE"] = "ipc_event"
    env["AFD_IPC_PEER_DEVICE"] = str(peer_device)
    env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
    env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
    if ffn_host:
        env["AFD_UCX_FFN_HOST"] = ffn_host
    return env


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

# Legacy alias (M=2 default) — kept for backward compat in case external scripts reference it.
_AFD_EXTRA = _AFD_EXTRA_BASE + ["--afd-micro-batch", "2"]


def _dvfs_args():
    """Tier2 per-batch DVFS args (added to every AF process when tier=True)."""
    return ["--afd-dvfs-enabled",
            "--afd-energy-model-dir", B.ENERGY_MODEL_DIR,
            "--afd-ttft-slo-ms", "5000",
            "--afd-tpot-slo-us", "300000"]


def _tier1_stats_path():
    d = HERE / "results" / "tier1_shared"
    d.mkdir(parents=True, exist_ok=True)
    return str(d / "deploy_decode_stats.json")


def _tier1_pa_args(stats_path):
    """Tier1 freq-only orchestration args (PA process only)."""
    return ["--enable-tier1-pa", "--tier1-disable-reload",
            "--tier1-monitor-window-s", "15", "--tier1-gpu-count", "4",
            "--tier1-stats-path", stats_path,
            "--tier1-prefill-data-path",
            "/workspace/sglang-tier/benchmark/test_motivation/hucc/paper/prefill_data_v1.txt",
            "--tier1-decode-data-path",
            "/workspace/sglang-tier/benchmark/test_motivation/hucc/paper/decode_data_v1.txt"]


def _afd_cmd(port, perspective, disagg, tp, base_gpu_id, attn_tp=None, ffn_tp=None,
             tier=False, is_pa=False, stats_path=None, micro_batch=2):
    cmd = [PYTHON, "-m", "sglang.launch_server",
           "--model-path", MODEL, "--tp", str(tp),
           "--host", "127.0.0.1", "--port", str(port),
           "--afd-perspective", perspective,
           "--disaggregation-mode", disagg,
           "--base-gpu-id", str(base_gpu_id)] + _AFD_EXTRA_BASE + [
           "--afd-micro-batch", str(micro_batch)]
    if attn_tp is not None:
        cmd += ["--afd-attn-tp", str(attn_tp)]
    if ffn_tp is not None:
        cmd += ["--afd-ffn-tp", str(ffn_tp)]
    if tier:
        cmd += _dvfs_args()
        if is_pa:
            cmd += _tier1_pa_args(stats_path)
        elif stats_path:
            cmd += ["--tier1-stats-path", stats_path]
    return cmd


def _start_af_prefill(env_base, log_dir, prefix, procs, tier=False, stats_path=None,
                      p_vis="6,7", micro_batch=2):
    """Prefill side: PF(ffn,TP1) + PA(attn,TP1) on the two GPUs in p_vis."""
    env_pf = _afd_env(env_base, p_vis, UCX_P, SCHED_P, peer_device=1)
    _popen("pf", _afd_cmd(PF_PORT, "ffn", "prefill", tp=1, base_gpu_id=0,
                          tier=tier, stats_path=stats_path, micro_batch=micro_batch),
           env_pf, log_dir, prefix, procs)
    time.sleep(2)
    env_pa = _afd_env(env_base, p_vis, UCX_P, SCHED_P, peer_device=0, ffn_host="127.0.0.1")
    pa_cmd = _afd_cmd(PA_PORT, "attn", "prefill", tp=1, base_gpu_id=1,
                      tier=tier, is_pa=True, stats_path=stats_path, micro_batch=micro_batch)
    _popen("pa", pa_cmd, env_pa, log_dir, prefix, procs)
    return B.wait_port("127.0.0.1", PA_PORT, 300) and B.wait_port("127.0.0.1", PF_PORT, 300)


# Decode-side layout per topology: (cvd, df_tp, df_base, da_tp, da_base,
#                                   df_attn_tp, da_ffn_tp, df_peer, da_peer)
_DECODE_LAYOUT = {
    "pdaf_m2":    ("4,5",     1, 0, 1, 1, None, None, 1, 0),
    "pdaf_m1":    ("2,3",     1, 0, 1, 1, None, None, 1, 0),
    "pdaf_d_tp2": ("2,3,4,5", 2, 0, 2, 2, 2,    2,    2, 0),
}


def start_pdaf(deploy, log_dir, prefix):
    """Generic PD+AF launcher; decode topology selected by `deploy`."""
    # Tier variants reuse a base layout but enable DVFS + Tier1.
    tier = deploy.endswith("_tier")
    base = deploy[:-5] if tier else deploy
    stats_path = _tier1_stats_path() if tier else None
    spec = DEPLOYMENTS[deploy]
    p_vis = spec.get("af_p_vis", "6,7")
    micro_batch = spec.get("micro_batch", 2)

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

    if not _start_af_prefill(env_base, log_dir, prefix, procs, tier=tier,
                             stats_path=stats_path, p_vis=p_vis, micro_batch=micro_batch):
        log.error("AF prefill failed to start")
        B.cleanup_procs(procs)
        return None

    cvd, df_tp, df_base, da_tp, da_base, df_attn, da_ffn, df_peer, da_peer = \
        _DECODE_LAYOUT[base]
    cvd = spec.get("af_d_cvd", cvd)

    env_df = _afd_env(env_base, cvd, UCX_D, SCHED_D, peer_device=df_peer)
    _popen("df", _afd_cmd(DF_PORT, "ffn", "decode", tp=df_tp, base_gpu_id=df_base,
                          attn_tp=df_attn, tier=tier, stats_path=stats_path,
                          micro_batch=micro_batch),
           env_df, log_dir, prefix, procs)
    time.sleep(5)
    env_da = _afd_env(env_base, cvd, UCX_D, SCHED_D, peer_device=da_peer,
                      ffn_host="127.0.0.1")
    _popen("da", _afd_cmd(DA_PORT, "attn", "decode", tp=da_tp, base_gpu_id=da_base,
                          ffn_tp=da_ffn, tier=tier, stats_path=stats_path,
                          micro_batch=micro_batch),
           env_da, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", DA_PORT, 300) or not B.wait_port("127.0.0.1", DF_PORT, 300):
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


def start_deploy(deploy, log_dir, prefix):
    spec = DEPLOYMENTS[deploy]
    if "dp" in spec:
        c = spec["dp"]
        return start_pd_dp(log_dir, prefix, c["instances"], c["tp"])
    if "pd" in spec:
        c = spec["pd"]
        return start_pd(log_dir, prefix, c["p_cvd"], c["p_tp"], c["d_cvd"], c["d_tp"])
    return start_pdaf(deploy, log_dir, prefix)


# ── Main ─────────────────────────────────────────────────────────────────


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser(description="Deployment-topology comparison benchmark")
    ap.add_argument("--deploys", type=str, default="pd_p2d2,pdaf_m2",
                    help="Comma-separated topologies: " + ",".join(DEPLOYMENTS))
    ap.add_argument("--workloads", type=str, required=True,
                    help="Comma-separated absolute workload JSONL paths")
    ap.add_argument("--freq", type=str, default="auto", choices=["auto", "max"],
                    help="Hardware frequency policy for all topologies")
    ap.add_argument("--ttft-slo-ms", type=float, default=5000.0)
    ap.add_argument("--tpot-slo-ms", type=float, default=300.0)
    ap.add_argument("--output-dir", type=str, default="results/deploy/json")
    ap.add_argument("--log-dir", type=str, default="logs/deploy")
    ap.add_argument("--max-run-s", type=float, default=360.0)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    deploys = [d.strip() for d in args.deploys.split(",") if d.strip()]
    for d in deploys:
        if d not in DEPLOYMENTS:
            ap.error(f"Unknown deploy '{d}'. Valid: {list(DEPLOYMENTS)}")
    workloads = [w.strip() for w in args.workloads.split(",") if w.strip()]

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    log_root = Path(args.log_dir); log_root.mkdir(parents=True, exist_ok=True)

    log.info("Deploys: %s | freq=%s | %d workloads", deploys, args.freq, len(workloads))

    # sweep[deploy][group][qps] = result
    sweep = {}
    for deploy in deploys:
        spec = DEPLOYMENTS[deploy]
        sweep.setdefault(deploy, {})
        for wl in workloads:
            cfg = B._cfg_from_path(wl)
            group, qps_label, base_tag = cfg["group"], cfg["qps"], cfg["tag"]
            tag = f"{deploy}_{base_tag}"
            cached = out_dir / f"{tag}_results.json"
            if cached.exists() and not args.force:
                sweep[deploy].setdefault(group, {})[qps_label] = json.loads(cached.read_text())
                log.info("Loaded cached %s", cached)
                continue

            log.info("=" * 70)
            log.info("DEPLOY %s | %s | il=%s ol=%s QPS=%s", deploy, spec["label"],
                     cfg["il"], cfg["ol"], qps_label)
            log.info("=" * 70)

            kill_servers()
            reset_freq(spec["gpus"])
            time.sleep(2)
            apply_freq(spec["gpus"], args.freq)
            # Point the shared cleanup/reset helper at THIS deploy's GPUs so
            # cleanup_procs() doesn't touch cards owned by other users.
            B.GPU_INDICES = list(spec["gpus"])

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
                    wl, url, ttft_slo_ms=args.ttft_slo_ms, tpot_slo_ms=args.tpot_slo_ms,
                    procs=procs, max_run_s=args.max_run_s,
                    gpu_indices=spec["gpus"], prefill_gpus=spec["prefill_gpus"],
                    decode_gpus=spec["decode_gpus"],
                ))
                results.update({"deploy": deploy, "workload": wl, "il": cfg["il"],
                                "ol": cfg["ol"], "qps": qps_label, "ngpu": spec["ngpu"]})
                sweep[deploy].setdefault(group, {})[qps_label] = results
                if not results.get("aborted"):
                    with open(cached, "w") as f:
                        json.dump(results, f, indent=2, default=str)
                _print_one(tag, spec, results)
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

    with open(out_dir / "deploy_summary.json", "w") as f:
        json.dump(sweep, f, indent=2, default=str)
    _print_summary(sweep)
    log.info("All deploy results saved to %s/", out_dir)


def _print_one(tag, spec, m):
    log.info("  %s: thpt=%.1f tok/s TTFT=%.0fms TPOT=%.0fms energy=%.0fJ "
             "(P=%.0f D=%.0f) SLOviol=%.1f%%  [%d GPU, %.0fJ/GPU]",
             tag, m["throughput_tok_s"], m["ttft_avg_ms"], m["tpot_avg_ms"],
             m["total_energy_j"], m["prefill_energy_j"], m["decode_energy_j"],
             m["slo_violation_rate"], spec["ngpu"], m["total_energy_j"] / spec["ngpu"])


def _print_summary(sweep):
    print("\n" + "=" * 100)
    print("  DEPLOYMENT COMPARISON SUMMARY")
    print("=" * 100)
    for deploy, groups in sweep.items():
        print(f"\n##### {deploy} — {DEPLOYMENTS[deploy]['label']} #####")
        print(f"  {'grp/qps':<22} {'thpt':>9} {'TTFT':>9} {'TPOT':>8} "
              f"{'energy':>9} {'J/tok':>8} {'SLO%':>6}")
        for group, qmap in sorted(groups.items()):
            for qps in sorted(qmap, key=B._qps_sort_key):
                m = qmap[qps]
                print(f"  {group+' q'+str(qps):<22} {m['throughput_tok_s']:>9.1f} "
                      f"{m['ttft_avg_ms']:>9.0f} {m['tpot_avg_ms']:>8.0f} "
                      f"{m['total_energy_j']:>9.0f} {m.get('energy_per_token_mj',0):>8.1f} "
                      f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
