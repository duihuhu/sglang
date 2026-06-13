#!/usr/bin/env python3
"""Test PDAF with asymmetric layout: 4PA + 2PF + 1DA + 1DF on Conv Heavy.

GPU layout:
  Prefill: CVD=0,1,2,3,4,5 -> PF(TP=2, base=0) + PA(TP=4, base=2)
  Decode:  CVD=6,7         -> DF(TP=1, base=0) + DA(TP=1, base=1)
"""
import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_fixed_qps_bench as B

log = logging.getLogger("pdaf_4pa2pf")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")

PYTHON = B.PYTHON
MODEL = B.MODEL
HERE = Path(__file__).resolve().parent

ROUTER_PORT = 53000
PA_PORT, PF_PORT = 53010, 53011
DA_PORT, DF_PORT = 53020, 53021

UCX_P, UCX_D = 28200, 28300
SCHED_P, SCHED_D = 68400, 68500

# GPU layout: 4PA + 2PF + 1DA + 1DF
P_CVD = "0,1,2,3,4,5"  # Prefill: PF(0,1) + PA(2,3,4,5)
D_CVD = "6,7"           # Decode:  DF(6) + DA(7)

PF_TP = 2  # PF uses 2 GPUs
PA_TP = 4  # PA uses 4 GPUs
DF_TP = 1  # DF uses 1 GPU
DA_TP = 1  # DA uses 1 GPU

ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT]

_AFD_EXTRA_BASE = [
    "--afd-comm-backend", "ipc_cpp",
    "--mem-fraction-static", "0.85",
    "--disaggregation-transfer-backend", "mooncake",
    "--disaggregation-bootstrap-port", "49999",
    "--disaggregation-ib-device", "mlx5_4",
    "--skip-server-warmup",
    "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
    "--afd-disagg-interleave-poll",
    "--disable-radix-cache",
]

# DVFS args
DVFS_TTFT_SLO_MS = 4000
DVFS_TPOT_SLO_US = 250000

ENERGY_MODEL_DIR = "/workspace/sglang-tier/benchmark/AFlex_bench/03_sensitivity/slo_sweep/retrain/models_v2"


def _dvfs_args():
    return [
        "--afd-dvfs-enabled",
        "--afd-energy-model-dir", ENERGY_MODEL_DIR,
        "--afd-ttft-slo-ms", str(DVFS_TTFT_SLO_MS),
        "--afd-tpot-slo-us", str(DVFS_TPOT_SLO_US),
    ]


def kill_servers():
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang_router"], capture_output=True)
    time.sleep(2)
    for port in ALL_PORTS:
        for _ in range(10):
            r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                               capture_output=True, text=True)
            if "LISTEN" not in r.stdout:
                break
            import re
            for m in re.finditer(r'pid=(\d+)', r.stdout):
                subprocess.run(["kill", "-9", m.group(1)], capture_output=True)
            time.sleep(1)


def _popen(name, cmd, env, log_dir, prefix, procs):
    log_file = log_dir / f"{prefix}{name}.log"
    fp = open(log_file, "w")
    p = subprocess.Popen(cmd, env=env, stdout=fp, stderr=fp, preexec_fn=os.setsid)
    procs.append((name, p, fp))
    log.info("  Started %s (pid=%d, CVD=%s)", name, p.pid, env.get("CUDA_VISIBLE_DEVICES"))
    return p


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


def _afd_cmd(port, perspective, disagg, tp, base_gpu_id, micro_batch=2,
             attn_tp=None, ffn_tp=None, tier=False, is_pa=False, dynamic_mb=True):
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
        cmd += _dvfs_args()
    return cmd


def start_router(procs, log_dir, prefix, p_port, d_port):
    cmd = [PYTHON, "-m", "sglang_router.launch_router",
           "--pd-disaggregation", "--mini-lb",
           "--prefill", f"http://127.0.0.1:{p_port}",
           "--decode", f"http://127.0.0.1:{d_port}",
           "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    env = os.environ.copy()
    _popen("router", cmd, env, log_dir, prefix, procs)
    if not B.wait_port("127.0.0.1", ROUTER_PORT, 60):
        log.error("Router failed to start")
        return False
    return True


def start_pdaf_4pa2pf(log_dir, prefix, tier=True):
    """Start 8-GPU PDAF: 4PA(TP=4) + 2PF(TP=2) + 1DA(TP=1) + 1DF(TP=1)."""
    procs = []
    env_base = os.environ.copy()
    env_base["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
    env_base["UCX_LOG_LEVEL"] = "fatal"
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"
    if tier:
        env_base["SGLANG_ENERGY_MODEL_DIR"] = ENERGY_MODEL_DIR

    # Prefill: PF(TP=2, base=0, GPU 0,1) + PA(TP=4, base=2, GPU 2,3,4,5) on CVD=0,1,2,3,4,5
    env_pf = _afd_env(env_base, P_CVD, UCX_P, SCHED_P, peer_device=2,
                      nvml_indices=[0, 1])
    _popen("pf", _afd_cmd(PF_PORT, "ffn", "prefill", tp=PF_TP, base_gpu_id=0,
                           micro_batch=2, attn_tp=PA_TP, ffn_tp=PF_TP,
                           tier=tier, dynamic_mb=True),
           env_pf, log_dir, prefix, procs)
    time.sleep(2)

    env_pa = _afd_env(env_base, P_CVD, UCX_P, SCHED_P, peer_device=0,
                      ffn_host="127.0.0.1", nvml_indices=[2, 3, 4, 5])
    _popen("pa", _afd_cmd(PA_PORT, "attn", "prefill", tp=PA_TP, base_gpu_id=2,
                           micro_batch=2, attn_tp=PA_TP, ffn_tp=PF_TP,
                           tier=tier, is_pa=True, dynamic_mb=True),
           env_pa, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", PF_PORT, 300) or \
       not B.wait_port("127.0.0.1", PA_PORT, 300):
        log.error("AF prefill (4PA+2PF) failed to start")
        B.cleanup_procs(procs)
        return None

    # Decode: DF(TP=1, base=0, GPU 6) + DA(TP=1, base=1, GPU 7) on CVD=6,7
    env_df = _afd_env(env_base, D_CVD, UCX_D, SCHED_D, peer_device=1,
                      nvml_indices=[6])
    _popen("df", _afd_cmd(DF_PORT, "ffn", "decode", tp=DF_TP, base_gpu_id=0,
                           micro_batch=2, attn_tp=DA_TP, ffn_tp=DF_TP,
                           tier=tier, dynamic_mb=True),
           env_df, log_dir, prefix, procs)
    time.sleep(5)

    env_da = _afd_env(env_base, D_CVD, UCX_D, SCHED_D, peer_device=0,
                      ffn_host="127.0.0.1", nvml_indices=[7])
    _popen("da", _afd_cmd(DA_PORT, "attn", "decode", tp=DA_TP, base_gpu_id=1,
                           micro_batch=2, attn_tp=DA_TP, ffn_tp=DF_TP,
                           tier=tier, dynamic_mb=True),
           env_da, log_dir, prefix, procs)

    if not B.wait_port("127.0.0.1", DA_PORT, 300) or \
       not B.wait_port("127.0.0.1", DF_PORT, 300):
        log.error("AF decode (1DA+1DF) failed to start")
        B.cleanup_procs(procs)
        return None

    time.sleep(5)
    if not start_router(procs, log_dir, prefix, PA_PORT, DA_PORT):
        B.cleanup_procs(procs)
        return None

    url = f"http://127.0.0.1:{ROUTER_PORT}"
    log.info("PDAF 4PA+2PF+1DA+1DF ready at %s, warming up...", url)
    B.warmup(url)
    return procs, url


def main():
    workload = "/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads/workload_azure_code_heavy_real.jsonl"
    out_dir = Path("/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/8gpu/results_8gpu_azure_slo2/json")
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path("/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/8gpu/logs_pdaf_4pa2pf")
    log_dir.mkdir(parents=True, exist_ok=True)

    ttft_slo = 4000.0
    tpot_slo = 200.0

    log.info("=" * 70)
    log.info("PDAF 4PA+2PF+1DA+1DF — Azure Code Heavy (Tier)")
    log.info("SLO: TTFT<%dms TPOT<%dms", ttft_slo, tpot_slo)
    log.info("=" * 70)

    kill_servers()
    time.sleep(3)

    # Reset all GPU frequencies
    for i in range(8):
        subprocess.run(["nvidia-smi", "-rgc", "-i", str(i)], capture_output=True)

    ret = start_pdaf_4pa2pf(log_dir, "pdaf_4pa2pf_conv_heavy_", tier=True)
    if ret is None:
        log.error("Failed to start PDAF 4PA+2PF+1DA+1DF")
        return

    procs, url = ret
    try:
        results = asyncio.run(B.run_workload(
            workload, url, ttft_slo_ms=ttft_slo, tpot_slo_ms=tpot_slo,
            procs=procs, max_run_s=800,
            gpu_indices=list(range(8)),
            prefill_gpus=[0, 1, 2, 3, 4, 5],
            decode_gpus=[6, 7],
        ))
        results.update({
            "deploy": "pdaf_4pa2pf_tier",
            "workload": workload,
            "ngpu": 8,
        })

        tag = "pdaf_4pa2pf_tier_var_azure_code_heavy_real"
        out_file = out_dir / f"{tag}_results.json"
        with open(out_file, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info("Results saved: %s", out_file)

        log.info("  Throughput: %.1f tok/s", results.get("throughput_tok_s", 0))
        log.info("  TTFT avg: %.1f ms", results.get("ttft_avg_ms", 0))
        log.info("  TPOT avg: %.1f ms", results.get("tpot_avg_ms", 0))
        log.info("  Total energy: %.0f J", results.get("total_energy_j", 0))
        log.info("  SLO violation: %.1f%%", results.get("slo_violation_rate", 0))

    except Exception as e:
        import traceback
        log.error("Test crashed: %r\n%s", e, traceback.format_exc())
    finally:
        B.cleanup_procs(procs)
        kill_servers()
        for i in range(8):
            subprocess.run(["nvidia-smi", "-rgc", "-i", str(i)], capture_output=True)


if __name__ == "__main__":
    main()
