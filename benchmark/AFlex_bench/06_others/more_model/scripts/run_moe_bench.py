#!/usr/bin/env python3
"""MoE Model (Qwen3-30B-A3B) Deployment Benchmark.

Tests three deployment topologies on 8x A800 GPUs:
  - native_dp8      : Native DP=8, 8x TP=1 instances (8 GPU)
  - pd_dp4          : PD disagg DP=4, 4x (P-TP1 + D-TP1) (8 GPU)
  - pdaf_dp2        : PD+AF DP=2, 2x (PA+PF+DA+DF) (8 GPU)

Each topology can optionally enable Tier DVFS (suffix _tier).

Metrics collected:
  - Throughput (tok/s)
  - TTFT processing time (ms) - excludes queuing delay
  - TPOT (ms) - time per output token
  - Total energy (J) via NVML energy counters
  - Energy efficiency (mJ/tok)
  - SLO violation rate (%)
  - GPU frequency samples over time

Usage:
    # Run all deployments on all workloads
    python run_moe_bench.py --deploy all

    # Test specific deployments
    python run_moe_bench.py --deploy native_dp8,native_dp8_tier --test-generate

    # Custom SLO and workloads
    python run_moe_bench.py --deploy pdaf_dp2_tier --ttft-slo-ms 3000 --tpot-slo-ms 200
"""
import argparse
import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
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
log = logging.getLogger("moe_bench")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen3-30B-A3B/"
HERE = Path(__file__).resolve().parent
ENERGY_MODEL_DIR = "/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/models"
LOG_DIR = HERE.parent / "logs"
RESULTS_DIR = HERE.parent / "results"

# Ports
ROUTER_PORT = 42000
BS_PORT = 49999

# PDAF ports
PA_PORT = 42010
PF_PORT = 42011
DA_PORT = 42020
DF_PORT = 42021
UCX_P = 28200
UCX_D = 28300
SCHED_P = 68400
SCHED_D = 68500

# PD ports
PD_P_PORT = 41000
PD_D_PORT = 41001

# Native port
NATIVE_PORT = 40000

# GPU allocation: use all 8 GPUs (0-7)
GPUS = [0, 1, 2, 3, 4, 5, 6, 7]

ALL_PORTS = [ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT,
             PD_P_PORT, PD_D_PORT, NATIVE_PORT, BS_PORT,
             53100, 53101, 53110, 53111,  # PD DP2 ports
             53120, 53121, 53130, 53131,  # PD DP4 extra ports
             53200, 53210, 53220, 53230,  # Native DP4 ports
             53240, 53250, 53260, 53270]  # Native DP8 extra ports


def kill_all():
    """Kill all sglang processes and free ports."""
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"], capture_output=True)
    subprocess.run(["pkill", "-9", "-f", "sglang_router"], capture_output=True)
    time.sleep(3)
    for port in ALL_PORTS:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True)
        for m in re.finditer(r'pid=(\d+)', r.stdout):
            subprocess.run(["kill", "-9", m.group(1)], capture_output=True)
    time.sleep(2)


def wait_health(port, timeout=300, check_model_info=False):
    """Wait for server health endpoint.
    For FFN perspective servers, use /get_model_info instead of /health.
    """
    import socket
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(2)
                s.connect(("127.0.0.1", port))
                break
        except (ConnectionRefusedError, OSError):
            time.sleep(3)
    else:
        return False

    endpoint = "/get_model_info" if check_model_info else "/health"
    while time.time() < deadline:
        try:
            r = requests.get(f"http://127.0.0.1:{port}{endpoint}", timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def get_gpu_energy_mj(gpu_indices):
    """Read cumulative energy (millijoules) via pynvml for given GPUs."""
    try:
        import pynvml
        pynvml.nvmlInit()
        result = {}
        for idx in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            result[idx] = pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)
        pynvml.nvmlShutdown()
        return result
    except Exception as e:
        log.warning("NVML energy read failed: %s", e)
        return {idx: 0 for idx in gpu_indices}


def get_gpu_freq_mhz(gpu_indices):
    """Read current SM clock (MHz) via pynvml."""
    try:
        import pynvml
        pynvml.nvmlInit()
        result = {}
        for idx in gpu_indices:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            result[idx] = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        pynvml.nvmlShutdown()
        return result
    except Exception:
        return {idx: 0 for idx in gpu_indices}


MAX_GPU_FREQ = 1410  # MHz, A800 max graphics clock


def lock_gpu_freq(gpu_indices, freq_mhz=MAX_GPU_FREQ):
    """Lock GPU clocks to specified frequency."""
    for idx in gpu_indices:
        subprocess.run(
            ["nvidia-smi", "-i", str(idx),
             "--lock-gpu-clocks=" + f"{freq_mhz},{freq_mhz}"],
            capture_output=True)
    log.info("  Locked GPU %s to %d MHz", gpu_indices, freq_mhz)


def unlock_gpu_freq(gpu_indices):
    """Reset GPU clocks to default."""
    for idx in gpu_indices:
        subprocess.run(
            ["nvidia-smi", "-i", str(idx), "--reset-gpu-clocks"],
            capture_output=True)
    log.info("  Unlocked GPU %s clocks", gpu_indices)


class DeploymentManager:
    def __init__(self, log_dir: Path, tier: bool = False):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs = []
        self.tier = tier

    def _popen(self, name, cmd, env):
        fh = open(self.log_dir / f"{name}.log", "w")
        p = subprocess.Popen(cmd, env=env, stdout=fh, stderr=subprocess.STDOUT,
                             start_new_session=True)
        self.procs.append((name, p, fh))
        log.info("  Started %s (PID=%d)", name, p.pid)
        return p

    def cleanup(self):
        for name, p, fh in self.procs:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except (ProcessLookupError, OSError):
                pass
            fh.close()
        self.procs.clear()
        time.sleep(3)

    def _common_env(self):
        env = os.environ.copy()
        env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
        env["UCX_LOG_LEVEL"] = "fatal"
        return env

    def start_native(self, gpus="0", tp=1):
        """Start native single-instance server."""
        env = self._common_env()
        env["CUDA_VISIBLE_DEVICES"] = gpus
        cmd = [PYTHON, "-m", "sglang.launch_server",
               "--model-path", MODEL, "--tp", str(tp),
               "--host", "127.0.0.1", "--port", str(NATIVE_PORT),
               "--mem-fraction-static", "0.75",
               "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
               "--skip-server-warmup"]
        if self.tier:
            cmd += ["--dvfs-enabled",
                    "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                    "--dvfs-ttft-slo-ms", "2000",
                    "--dvfs-tpot-slo-us", "250000"]
        self._popen("native", cmd, env)
        if not wait_health(NATIVE_PORT, 300):
            log.error("Native server failed to start")
            return None
        log.info("Native TP=%d ready on port %d", tp, NATIVE_PORT)
        return NATIVE_PORT

    def start_native_dp(self, n_instances=4):
        """Start Native DP (n x TP=1 independent instances, round-robin)."""
        env_base = self._common_env()
        ports = []
        for i in range(n_instances):
            port = 53200 + i * 10
            nccl_port = 33300 + i * 10
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(GPUS[i])
            cmd = [PYTHON, "-m", "sglang.launch_server",
                   "--model-path", MODEL, "--tp", "1",
                   "--host", "127.0.0.1", "--port", str(port),
                   "--nccl-port", str(nccl_port),
                   "--mem-fraction-static", "0.75",
                   "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                   "--skip-server-warmup"]
            if self.tier:
                cmd += ["--dvfs-enabled",
                        "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                        "--dvfs-ttft-slo-ms", "2000",
                        "--dvfs-tpot-slo-us", "250000"]
            self._popen(f"dp_{i}", cmd, env)
            ports.append(port)
            time.sleep(2)

        for i, port in enumerate(ports):
            if not wait_health(port, 300):
                log.error("  DP instance %d (port %d) failed", i, port)
                return None
            log.info("  DP instance %d ready (port %d)", i, port)

        # Launch router for round-robin load balancing
        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
                 "--policy", "round_robin",
                 "--worker-urls"] + [f"http://127.0.0.1:{p}" for p in ports]
        self._popen("dp_router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("Native DP router failed")
            return None

        log.info("Native DP=%d ready (tier=%s, router port %d)",
                 n_instances, self.tier, ROUTER_PORT)
        return ROUTER_PORT

    def start_pd_dp(self, n_pairs=4):
        """Start PD DP: n_pairs x (P-TP1 + D-TP1) instances.
        DP4: 4 pairs on 8 GPUs.
        """
        env_base = self._common_env()
        instances = []
        for i in range(n_pairs):
            instances.append({
                "p_cvd": str(GPUS[i * 2]),
                "d_cvd": str(GPUS[i * 2 + 1]),
                "p_port": 53100 + i * 10,
                "d_port": 53101 + i * 10,
                "bs_port": 49100 + i * 10,
            })

        for idx, inst in enumerate(instances):
            # Prefill
            env_p = env_base.copy()
            env_p["CUDA_VISIBLE_DEVICES"] = inst["p_cvd"]
            cmd_p = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "1",
                     "--host", "127.0.0.1", "--port", str(inst["p_port"]),
                     "--nccl-port", str(34000 + idx * 10),
                     "--mem-fraction-static", "0.75",
                     "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup",
                     "--disaggregation-mode", "prefill",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(inst["bs_port"]),
                     "--disaggregation-ib-device", "mlx5_4"]
            self._popen(f"pd{idx}_prefill", cmd_p, env_p)
            time.sleep(3)

            # Decode
            env_d = env_base.copy()
            env_d["CUDA_VISIBLE_DEVICES"] = inst["d_cvd"]
            cmd_d = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "1",
                     "--host", "127.0.0.1", "--port", str(inst["d_port"]),
                     "--nccl-port", str(34001 + idx * 10),
                     "--mem-fraction-static", "0.75",
                     "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup",
                     "--disaggregation-mode", "decode",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(inst["bs_port"]),
                     "--disaggregation-ib-device", "mlx5_4"]
            if self.tier:
                cmd_d += ["--dvfs-enabled",
                          "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                          "--dvfs-ttft-slo-ms", "2000",
                          "--dvfs-tpot-slo-us", "250000"]
            self._popen(f"pd{idx}_decode", cmd_d, env_d)
            time.sleep(3)

        # Wait for all instances
        all_ports = []
        for idx, inst in enumerate(instances):
            for role, port in [("P", inst["p_port"]), ("D", inst["d_port"])]:
                if not wait_health(port, 300):
                    log.error("  PD%d %s (port %d) failed", idx, role, port)
                    return None
                log.info("  PD%d %s ready (port %d)", idx, role, port)
            all_ports.append(inst)

        # Router with multiple prefill/decode endpoints (with bootstrap ports)
        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        for inst in instances:
            cmd_r += ["--prefill", f"http://127.0.0.1:{inst['p_port']}", str(inst["bs_port"])]
        for inst in instances:
            cmd_r += ["--decode", f"http://127.0.0.1:{inst['d_port']}"]
        self._popen("pd_router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PD DP%d router failed", n_pairs)
            return None
        log.info("PD DP%d ready (tier=%s, router port %d)", n_pairs, self.tier, ROUTER_PORT)
        return ROUTER_PORT

    def start_pd(self, p_gpus="0", d_gpus="1", tp=1):
        """Start PD disaggregation (P-TP + D-TP)."""
        env_base = self._common_env()

        # Prefill
        env_p = env_base.copy()
        env_p["CUDA_VISIBLE_DEVICES"] = p_gpus
        cmd_p = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", str(tp),
                 "--host", "127.0.0.1", "--port", str(PD_P_PORT),
                 "--mem-fraction-static", "0.75",
                 "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                 "--skip-server-warmup",
                 "--disaggregation-mode", "prefill",
                 "--disaggregation-transfer-backend", "mooncake",
                 "--disaggregation-bootstrap-port", str(BS_PORT),
                 "--disaggregation-ib-device", "mlx5_4"]
        self._popen("pd_prefill", cmd_p, env_p)
        time.sleep(5)

        # Decode
        env_d = env_base.copy()
        env_d["CUDA_VISIBLE_DEVICES"] = d_gpus
        cmd_d = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", str(tp),
                 "--host", "127.0.0.1", "--port", str(PD_D_PORT),
                 "--mem-fraction-static", "0.75",
                 "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                 "--skip-server-warmup",
                 "--disaggregation-mode", "decode",
                 "--disaggregation-transfer-backend", "mooncake",
                 "--disaggregation-bootstrap-port", str(BS_PORT),
                 "--disaggregation-ib-device", "mlx5_4"]
        if self.tier:
            cmd_d += ["--dvfs-enabled",
                      "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                      "--dvfs-ttft-slo-ms", "2000",
                      "--dvfs-tpot-slo-us", "250000"]
        self._popen("pd_decode", cmd_d, env_d)

        for port, name in [(PD_P_PORT, "Prefill"), (PD_D_PORT, "Decode")]:
            if not wait_health(port, 300):
                log.error("PD %s failed to start", name)
                return None
            log.info("  PD %s ready (port %d)", name, port)

        # Router
        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation", "--mini-lb",
                 "--prefill", f"http://127.0.0.1:{PD_P_PORT}",
                 "--decode", f"http://127.0.0.1:{PD_D_PORT}",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        self._popen("pd_router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PD router failed")
            return None
        log.info("PD Disagg ready (router port %d)", ROUTER_PORT)
        return ROUTER_PORT

    def start_pdaf(self, micro_batch=2):
        """Start PDAF disaggregation (PA+PF+DA+DF, each TP=2, 8 GPU).
        Layout: PF(GPU0,1) + PA(GPU2,3) + DF(GPU4,5) + DA(GPU6,7)
        Each component uses TP=2 for MoE model.
        """
        env_base = self._common_env()
        env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
        env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
        env_base["AFD_ASYNC_PIPELINE"] = "1"

        common = ["--model-path", MODEL, "--tp", "2",
                  "--host", "127.0.0.1",
                  "--afd-comm-backend", "ipc_cpp",
                  "--afd-micro-batch", str(micro_batch),
                  "--afd-dynamic-micro-batch",
                  "--mem-fraction-static", "0.75",
                  "--max-running-requests", "512",
                  "--skip-server-warmup",
                  "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                  "--afd-disagg-interleave-poll",
                  "--disable-radix-cache",
                  "--num-reserved-decode-tokens", "512",
                  "--disaggregation-transfer-backend", "mooncake",
                  "--disaggregation-bootstrap-port", str(BS_PORT),
                  "--disaggregation-ib-device", "mlx5_4",
                  "--enable-return-routed-experts",
                  "--enable-metrics"]

        dvfs_args = []
        if self.tier:
            dvfs_args = ["--afd-dvfs-enabled",
                         "--afd-energy-model-dir", ENERGY_MODEL_DIR,
                         "--afd-ttft-slo-ms", "2000",
                         "--afd-tpot-slo-us", "250000",
                         "--afd-dvfs-idle-lock"]

        def _env(cvd, ucx_base, sched_port, peer_device, nvml_idx, ffn_host=None):
            e = env_base.copy()
            e["CUDA_VISIBLE_DEVICES"] = cvd
            e["AFD_UCX_BASE_PORT"] = str(ucx_base)
            e["AFD_SCHED_PORT"] = str(sched_port)
            e["AFD_IPC_SYNC_MODE"] = "ipc_event"
            e["AFD_IPC_PEER_DEVICE"] = str(peer_device)
            e["AFD_NVML_DEVICE_INDICES"] = str(nvml_idx)
            e["AFD_NVML_DEVICE_INDEX"] = str(nvml_idx)
            e["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
            e["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
            e["AFD_LIF_SHARED_PATH"] = "/tmp/afd_lif_shared.txt"
            if ffn_host:
                e["AFD_UCX_FFN_HOST"] = ffn_host
            return e

        def _cmd(port, perspective, disagg, base_gpu_id):
            cmd = [PYTHON, "-m", "sglang.launch_server",
                   "--port", str(port),
                   "--afd-perspective", perspective,
                   "--disaggregation-mode", disagg,
                   "--base-gpu-id", str(base_gpu_id)] + common + dvfs_args
            return cmd

        # PF (FFN, TP=2, GPU0,1) - base_gpu_id=0, peer=2
        pf_cvd = "0,1,2,3"
        self._popen("pf", _cmd(PF_PORT, "ffn", "prefill", 0),
                    _env(pf_cvd, UCX_P, SCHED_P, peer_device=2, nvml_idx="0,1"))
        time.sleep(5)

        # PA (Attn, TP=2, GPU2,3) - base_gpu_id=2, peer=0
        pa_cvd = "0,1,2,3"
        self._popen("pa", _cmd(PA_PORT, "attn", "prefill", 2),
                    _env(pa_cvd, UCX_P, SCHED_P, peer_device=0, nvml_idx="2,3",
                         ffn_host="127.0.0.1"))
        time.sleep(5)

        # DF (FFN, TP=2, GPU4,5) - base_gpu_id=0, peer=2
        df_cvd = "4,5,6,7"
        self._popen("df", _cmd(DF_PORT, "ffn", "decode", 0),
                    _env(df_cvd, UCX_D, SCHED_D, peer_device=2, nvml_idx="4,5"))
        time.sleep(8)

        # DA (Attn, TP=2, GPU6,7) - base_gpu_id=2, peer=0
        da_cvd = "4,5,6,7"
        self._popen("da", _cmd(DA_PORT, "attn", "decode", 2),
                    _env(da_cvd, UCX_D, SCHED_D, peer_device=0, nvml_idx="6,7",
                         ffn_host="127.0.0.1"))

        log.info("Waiting for PDAF servers...")
        # PA/DA use /health, PF/DF (FFN perspective) use /get_model_info
        # because FFN event loop doesn't send detokenizer heartbeats
        checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
                  (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
        for port, name, use_model_info in checks:
            if not wait_health(port, 600, check_model_info=use_model_info):
                log.error("  %s (port %d) failed to start", name, port)
                return None
            log.info("  %s ready (port %d)", name, port)

        # Router
        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation", "--mini-lb",
                 "--prefill", f"http://127.0.0.1:{PA_PORT}",
                 "--decode", f"http://127.0.0.1:{DA_PORT}",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        self._popen("router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PDAF router failed")
            return None
        log.info("PDAF ready (M=%d, tier=%s, router port %d)",
                 micro_batch, self.tier, ROUTER_PORT)
        return ROUTER_PORT

    def start_pdaf_asym(self, micro_batch=2):
        """Start asymmetric PDAF: PA(TP1)+PF(TP2) on 3GPU + DA(TP1)+DF(TP4) on 5GPU.
        MoE-friendly layout: FFN needs at least TP=2, Attn can use TP=1.
        Layout:
          Prefill: CVD="0,1,2" -> PF(TP=2, base=0) + PA(TP=1, base=2)
          Decode:  CVD="3,4,5,6,7" -> DF(TP=4, base=0) + DA(TP=1, base=4)
        Gives decode side more GPU for KV cache capacity.
        """
        return self._start_pdaf_asym_impl(micro_batch)

    def _start_pdaf_asym_impl(self, micro_batch):
        """Implementation of asymmetric PDAF deployment."""
        env_base = self._common_env()
        env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
        env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
        env_base["AFD_ASYNC_PIPELINE"] = "1"

        dvfs_args = []
        if self.tier:
            dvfs_args = ["--afd-dvfs-enabled",
                         "--afd-energy-model-dir", ENERGY_MODEL_DIR,
                         "--afd-ttft-slo-ms", "2000",
                         "--afd-tpot-slo-us", "250000",
                         "--afd-dvfs-idle-lock"]

        # Prefill side: PF TP=2 (2GPU) + PA TP=1 (1GPU) = 3 GPU
        p_cvd = "0,1,2"
        p_ffn_tp = 2
        p_attn_tp = 1
        # Decode side: DF TP=4 (4GPU) + DA TP=1 (1GPU) = 5 GPU
        d_cvd = "3,4,5,6,7"
        d_ffn_tp = 4
        d_attn_tp = 1

        common_base = ["--model-path", MODEL,
                       "--host", "127.0.0.1",
                       "--afd-comm-backend", "ipc_cpp",
                       "--afd-micro-batch", str(micro_batch),
                       "--afd-dynamic-micro-batch",
                       "--max-running-requests", "512",
                       "--skip-server-warmup",
                       "--disable-cuda-graph",
                       "--disable-piecewise-cuda-graph",
                       "--afd-disagg-interleave-poll",
                       "--disable-radix-cache",
                       "--num-reserved-decode-tokens", "512",
                       "--disaggregation-transfer-backend", "mooncake",
                       "--disaggregation-bootstrap-port", str(BS_PORT),
                       "--disaggregation-ib-device", "mlx5_4",
                       "--enable-return-routed-experts",
                       "--enable-metrics"]

        # Prefill TP=1(PA)/TP=2(PF): PA is small, PF needs TP=2
        # Decode TP=1(DA)/TP=4(DF): more GPU = more KV cache
        p_mem_frac = "0.70"
        d_mem_frac = "0.75"

        def _env(cvd, ucx_base, sched_port, peer_device, nvml_idx,
                 ffn_host=None):
            e = env_base.copy()
            e["CUDA_VISIBLE_DEVICES"] = cvd
            e["AFD_UCX_BASE_PORT"] = str(ucx_base)
            e["AFD_SCHED_PORT"] = str(sched_port)
            e["AFD_IPC_SYNC_MODE"] = "ipc_event"
            e["AFD_IPC_PEER_DEVICE"] = str(peer_device)
            e["AFD_NVML_DEVICE_INDICES"] = str(nvml_idx)
            e["AFD_NVML_DEVICE_INDEX"] = str(nvml_idx)
            e["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
            e["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
            e["AFD_LIF_SHARED_PATH"] = "/tmp/afd_lif_shared.txt"
            if ffn_host:
                e["AFD_UCX_FFN_HOST"] = ffn_host
            return e

        def _cmd(port, perspective, disagg, tp, base_gpu_id,
                 attn_tp=None, ffn_tp=None, mem_frac="0.75"):
            cmd = [PYTHON, "-m", "sglang.launch_server",
                   "--port", str(port),
                   "--tp", str(tp),
                   "--afd-perspective", perspective,
                   "--disaggregation-mode", disagg,
                   "--base-gpu-id", str(base_gpu_id),
                   "--mem-fraction-static", mem_frac]
            cmd += common_base + dvfs_args
            if attn_tp is not None:
                cmd += ["--afd-attn-tp", str(attn_tp)]
            if ffn_tp is not None:
                cmd += ["--afd-ffn-tp", str(ffn_tp)]
            return cmd

        # --- Prefill side (3 GPU: PF TP=2, PA TP=1) ---
        # PF (FFN, TP=2, GPU0,1, base=0)
        self._popen("pf",
                    _cmd(PF_PORT, "ffn", "prefill", tp=p_ffn_tp,
                         base_gpu_id=0, attn_tp=p_attn_tp,
                         ffn_tp=p_ffn_tp, mem_frac=p_mem_frac),
                    _env(p_cvd, UCX_P, SCHED_P, peer_device=2,
                         nvml_idx="0,1"))
        time.sleep(5)

        # PA (Attn, TP=1, GPU2, base=2)
        self._popen("pa",
                    _cmd(PA_PORT, "attn", "prefill", tp=p_attn_tp,
                         base_gpu_id=2, attn_tp=p_attn_tp,
                         ffn_tp=p_ffn_tp, mem_frac=p_mem_frac),
                    _env(p_cvd, UCX_P, SCHED_P, peer_device=0,
                         nvml_idx="2", ffn_host="127.0.0.1"))
        time.sleep(5)

        # --- Decode side (5 GPU: DF TP=4, DA TP=1) ---
        # DF (FFN, TP=4, GPU3-6, base=0)
        self._popen("df",
                    _cmd(DF_PORT, "ffn", "decode", tp=d_ffn_tp,
                         base_gpu_id=0, attn_tp=d_attn_tp,
                         ffn_tp=d_ffn_tp, mem_frac=d_mem_frac),
                    _env(d_cvd, UCX_D, SCHED_D, peer_device=4,
                         nvml_idx="3,4,5,6"))
        time.sleep(8)

        # DA (Attn, TP=1, GPU7, base=4)
        self._popen("da",
                    _cmd(DA_PORT, "attn", "decode", tp=d_attn_tp,
                         base_gpu_id=4, attn_tp=d_attn_tp,
                         ffn_tp=d_ffn_tp, mem_frac=d_mem_frac),
                    _env(d_cvd, UCX_D, SCHED_D, peer_device=0,
                         nvml_idx="7", ffn_host="127.0.0.1"))

        log.info("Waiting for PDAF Asym servers...")
        checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
                  (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
        for port, name, use_model_info in checks:
            if not wait_health(port, 600,
                               check_model_info=use_model_info):
                log.error("  %s (port %d) failed to start", name, port)
                return None
            log.info("  %s ready (port %d)", name, port)

        # Router
        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation", "--mini-lb",
                 "--prefill", f"http://127.0.0.1:{PA_PORT}",
                 "--decode", f"http://127.0.0.1:{DA_PORT}",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        self._popen("router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PDAF Asym router failed")
            return None
        log.info("PDAF Asym ready (M=%d, tier=%s, 3P5D: PA-TP1+PF-TP2 | DA-TP1+DF-TP4)",
                 micro_batch, self.tier)
        return ROUTER_PORT

    def start_pdaf_asym_4p4d(self, micro_batch=2):
        """Start 4P+4D: PA(TP2)+PF(TP2) on 4GPU + DA(TP2)+DF(TP2) on 4GPU.
        Balanced layout with higher Decode Attn parallelism.
        """
        return self._start_pdaf_asym_4p4d_impl(micro_batch)

    def _start_pdaf_asym_4p4d_impl(self, micro_batch):
        """4P+4D: PA(TP2,GPU2,3)+PF(TP2,GPU0,1) | DA(TP2,GPU6,7)+DF(TP2,GPU4,5)."""
        env_base = self._common_env()
        env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
        env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
        env_base["AFD_ASYNC_PIPELINE"] = "1"

        dvfs_args = []
        if self.tier:
            dvfs_args = ["--afd-dvfs-enabled",
                         "--afd-energy-model-dir", ENERGY_MODEL_DIR,
                         "--afd-ttft-slo-ms", "2000",
                         "--afd-tpot-slo-us", "250000",
                         "--afd-dvfs-idle-lock"]

        p_cvd = "0,1,2,3"
        p_ffn_tp = 2
        p_attn_tp = 2
        d_cvd = "4,5,6,7"
        d_ffn_tp = 2
        d_attn_tp = 2

        common_base = ["--model-path", MODEL,
                       "--host", "127.0.0.1",
                       "--afd-comm-backend", "ipc_cpp",
                       "--afd-micro-batch", str(micro_batch),
                       "--afd-dynamic-micro-batch",
                       "--max-running-requests", "512",
                       "--skip-server-warmup",
                       "--disable-cuda-graph",
                       "--disable-piecewise-cuda-graph",
                       "--afd-disagg-interleave-poll",
                       "--disable-radix-cache",
                       "--num-reserved-decode-tokens", "512",
                       "--disaggregation-transfer-backend", "mooncake",
                       "--disaggregation-bootstrap-port", str(BS_PORT),
                       "--disaggregation-ib-device", "mlx5_4",
                       "--enable-return-routed-experts",
                       "--enable-metrics"]

        p_mem_frac = "0.80"
        d_mem_frac = "0.80"

        def _env(cvd, ucx_base, sched_port, peer_device, nvml_idx,
                 ffn_host=None):
            e = env_base.copy()
            e["CUDA_VISIBLE_DEVICES"] = cvd
            e["AFD_UCX_BASE_PORT"] = str(ucx_base)
            e["AFD_SCHED_PORT"] = str(sched_port)
            e["AFD_IPC_SYNC_MODE"] = "ipc_event"
            e["AFD_IPC_PEER_DEVICE"] = str(peer_device)
            e["AFD_NVML_DEVICE_INDICES"] = str(nvml_idx)
            e["AFD_NVML_DEVICE_INDEX"] = str(nvml_idx).split(",")[0]
            e["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
            e["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
            e["AFD_LIF_SHARED_PATH"] = "/tmp/afd_lif_shared.txt"
            if ffn_host:
                e["AFD_UCX_FFN_HOST"] = ffn_host
            return e

        def _cmd(port, perspective, disagg, tp, base_gpu_id,
                 attn_tp=None, ffn_tp=None, mem_frac="0.80"):
            cmd = [PYTHON, "-m", "sglang.launch_server",
                   "--port", str(port),
                   "--tp", str(tp),
                   "--afd-perspective", perspective,
                   "--disaggregation-mode", disagg,
                   "--base-gpu-id", str(base_gpu_id),
                   "--mem-fraction-static", mem_frac]
            cmd += common_base + dvfs_args
            if attn_tp is not None:
                cmd += ["--afd-attn-tp", str(attn_tp)]
            if ffn_tp is not None:
                cmd += ["--afd-ffn-tp", str(ffn_tp)]
            return cmd

        # PF (FFN, TP=2, GPU0,1, base=0)
        self._popen("pf",
                    _cmd(PF_PORT, "ffn", "prefill", tp=p_ffn_tp,
                         base_gpu_id=0, attn_tp=p_attn_tp,
                         ffn_tp=p_ffn_tp, mem_frac=p_mem_frac),
                    _env(p_cvd, UCX_P, SCHED_P, peer_device=2,
                         nvml_idx="0,1"))
        time.sleep(5)

        # PA (Attn, TP=2, GPU2,3, base=2)
        self._popen("pa",
                    _cmd(PA_PORT, "attn", "prefill", tp=p_attn_tp,
                         base_gpu_id=2, attn_tp=p_attn_tp,
                         ffn_tp=p_ffn_tp, mem_frac=p_mem_frac),
                    _env(p_cvd, UCX_P, SCHED_P, peer_device=0,
                         nvml_idx="2,3", ffn_host="127.0.0.1"))
        time.sleep(5)

        # DF (FFN, TP=2, GPU4,5, base=0)
        self._popen("df",
                    _cmd(DF_PORT, "ffn", "decode", tp=d_ffn_tp,
                         base_gpu_id=0, attn_tp=d_attn_tp,
                         ffn_tp=d_ffn_tp, mem_frac=d_mem_frac),
                    _env(d_cvd, UCX_D, SCHED_D, peer_device=2,
                         nvml_idx="4,5"))
        time.sleep(8)

        # DA (Attn, TP=2, GPU6,7, base=2)
        self._popen("da",
                    _cmd(DA_PORT, "attn", "decode", tp=d_attn_tp,
                         base_gpu_id=2, attn_tp=d_attn_tp,
                         ffn_tp=d_ffn_tp, mem_frac=d_mem_frac),
                    _env(d_cvd, UCX_D, SCHED_D, peer_device=0,
                         nvml_idx="6,7", ffn_host="127.0.0.1"))

        log.info("Waiting for PDAF 4P4D servers...")
        checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
                  (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
        for port, name, use_model_info in checks:
            if not wait_health(port, 600,
                               check_model_info=use_model_info):
                log.error("  %s (port %d) failed to start", name, port)
                return None
            log.info("  %s ready (port %d)", name, port)

        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation", "--mini-lb",
                 "--prefill", f"http://127.0.0.1:{PA_PORT}",
                 "--decode", f"http://127.0.0.1:{DA_PORT}",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        self._popen("router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PDAF 4P4D router failed")
            return None
        log.info("PDAF 4P4D ready (M=%d, tier=%s, PA-TP2+PF-TP2 | DA-TP2+DF-TP2)",
                 micro_batch, self.tier)
        return ROUTER_PORT

    def start_pdaf_dp(self, micro_batch=2, n_instances=2):
        """Start PDAF DP2: 2x (PA+PF+DA+DF) on 8 GPUs.
        Instance 0: PF(GPU0)+PA(GPU1)+DF(GPU2)+DA(GPU3)
        Instance 1: PF(GPU4)+PA(GPU5)+DF(GPU6)+DA(GPU7)
        """
        env_base = self._common_env()
        env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
        env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
        env_base["AFD_ASYNC_PIPELINE"] = "1"

        dvfs_args = []
        if self.tier:
            dvfs_args = ["--afd-dvfs-enabled",
                         "--afd-energy-model-dir", ENERGY_MODEL_DIR,
                         "--afd-ttft-slo-ms", "2000",
                         "--afd-tpot-slo-us", "250000",
                         "--afd-dvfs-idle-lock"]

        all_pa_ports = []
        all_da_ports = []

        for inst_idx in range(n_instances):
            g = GPUS[inst_idx * 4: inst_idx * 4 + 4]
            port_offset = inst_idx * 100
            pa_port = PA_PORT + port_offset
            pf_port = PF_PORT + port_offset
            da_port = DA_PORT + port_offset
            df_port = DF_PORT + port_offset
            ucx_p = UCX_P + inst_idx * 1000
            ucx_d = UCX_D + inst_idx * 1000
            sched_p = SCHED_P + inst_idx * 1000
            sched_d = SCHED_D + inst_idx * 1000
            bs_port = BS_PORT + inst_idx

            common = ["--model-path", MODEL, "--tp", "1",
                      "--host", "127.0.0.1",
                      "--afd-comm-backend", "ipc_cpp",
                      "--afd-micro-batch", str(micro_batch),
                      "--mem-fraction-static", "0.75",
                      "--max-running-requests", "512",
                      "--skip-server-warmup",
                      "--disable-cuda-graph",
                      "--disable-piecewise-cuda-graph",
                      "--afd-disagg-interleave-poll",
                      "--disable-radix-cache",
                      "--num-reserved-decode-tokens", "512",
                      "--disaggregation-transfer-backend", "mooncake",
                      "--disaggregation-bootstrap-port", str(bs_port),
                      "--disaggregation-ib-device", "mlx5_4"]

            def _env(cvd, ucx_base, sched_port, peer_device, nvml_idx,
                     ffn_host=None):
                e = env_base.copy()
                e["CUDA_VISIBLE_DEVICES"] = cvd
                e["AFD_UCX_BASE_PORT"] = str(ucx_base)
                e["AFD_SCHED_PORT"] = str(sched_port)
                e["AFD_IPC_SYNC_MODE"] = "ipc_event"
                e["AFD_IPC_PEER_DEVICE"] = str(peer_device)
                e["AFD_NVML_DEVICE_INDICES"] = str(nvml_idx)
                e["AFD_NVML_DEVICE_INDEX"] = str(nvml_idx)
                e["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
                e["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
                if ffn_host:
                    e["AFD_UCX_FFN_HOST"] = ffn_host
                return e

            def _cmd(port, perspective, disagg, base_gpu_id):
                return [PYTHON, "-m", "sglang.launch_server",
                        "--port", str(port),
                        "--afd-perspective", perspective,
                        "--disaggregation-mode", disagg,
                        "--base-gpu-id",
                        str(base_gpu_id)] + common + dvfs_args

            pf_cvd = f"{g[0]},{g[1]}"
            pa_cvd = f"{g[0]},{g[1]}"
            df_cvd = f"{g[2]},{g[3]}"
            da_cvd = f"{g[2]},{g[3]}"

            self._popen(f"pf_{inst_idx}",
                        _cmd(pf_port, "ffn", "prefill", 0),
                        _env(pf_cvd, ucx_p, sched_p, 1, g[0]))
            time.sleep(5)
            self._popen(f"pa_{inst_idx}",
                        _cmd(pa_port, "attn", "prefill", 1),
                        _env(pa_cvd, ucx_p, sched_p, 0, g[1],
                             ffn_host="127.0.0.1"))
            time.sleep(5)
            self._popen(f"df_{inst_idx}",
                        _cmd(df_port, "ffn", "decode", 0),
                        _env(df_cvd, ucx_d, sched_d, 1, g[2]))
            time.sleep(8)
            self._popen(f"da_{inst_idx}",
                        _cmd(da_port, "attn", "decode", 1),
                        _env(da_cvd, ucx_d, sched_d, 0, g[3],
                             ffn_host="127.0.0.1"))
            time.sleep(5)

            all_pa_ports.append(pa_port)
            all_da_ports.append(da_port)

            log.info("  PDAF inst %d launched (GPUs %s)", inst_idx, g)

        log.info("Waiting for PDAF DP%d servers...", n_instances)
        for inst_idx in range(n_instances):
            port_offset = inst_idx * 100
            checks = [
                (PA_PORT + port_offset, f"PA_{inst_idx}", False),
                (PF_PORT + port_offset, f"PF_{inst_idx}", True),
                (DA_PORT + port_offset, f"DA_{inst_idx}", False),
                (DF_PORT + port_offset, f"DF_{inst_idx}", True),
            ]
            for port, name, use_model_info in checks:
                if not wait_health(port, 600, check_model_info=use_model_info):
                    log.error("  %s (port %d) failed", name, port)
                    return None
                log.info("  %s ready (port %d)", name, port)

        # Router with multiple prefill/decode endpoints
        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation", "--mini-lb",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        for p in all_pa_ports:
            cmd_r += ["--prefill", f"http://127.0.0.1:{p}"]
        for p in all_da_ports:
            cmd_r += ["--decode", f"http://127.0.0.1:{p}"]
        self._popen("pdaf_router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PDAF DP%d router failed", n_instances)
            return None
        log.info("PDAF DP%d ready (M=%d, tier=%s, router port %d)",
                 n_instances, micro_batch, self.tier, ROUTER_PORT)
        return ROUTER_PORT


def test_generate(port, n_tokens=32):
    """Quick generation test."""
    url = f"http://127.0.0.1:{port}/generate"
    payload = {
        "text": "Explain the concept of Mixture of Experts in machine learning:",
        "sampling_params": {"max_new_tokens": n_tokens, "temperature": 0.0}
    }
    try:
        r = requests.post(url, json=payload, timeout=60,
                          headers={"Content-Type": "application/json"})
        data = r.json()
        if "text" in data:
            meta = data.get("meta_info", {})
            log.info("  Generated %d tokens, TTFT=%.1fms, E2E=%.2fs",
                     meta.get("completion_tokens", "?"),
                     meta.get("ttft_pure_processing", 0) * 1000,
                     meta.get("e2e_latency", 0))
            return True
        else:
            log.error("  Generate failed: %s", data.get("message", str(data)[:200]))
            return False
    except Exception as e:
        log.error("  Generate request failed: %s", e)
        return False



WORKLOAD_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads")
WORKLOADS = [
    WORKLOAD_DIR / "workload_azure_code_medium_real.jsonl",
    WORKLOAD_DIR / "workload_azure_conv_light_real.jsonl",
    WORKLOAD_DIR / "workload_azure_conv_medium_real.jsonl",
    WORKLOAD_DIR / "workload_azure_conv_heavy_real.jsonl",
]

# SLO thresholds
DEFAULT_TTFT_SLO_MS = 2000.0
DEFAULT_TPOT_SLO_MS = 250.0


async def _send_request(session, url, req, base_time, results_list,
                        ttft_slo_ms, tpot_slo_ms):
    """Send one streaming request and record latency + SLO violations.

    TPOT = (last_token_time - first_token_time) / (tokens - 1)
    This excludes queue wait, consistent with Dense benchmark.
    """
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)

    payload = {
        "text": "x" * req["input_len"],
        "sampling_params": {"max_new_tokens": req["output_len"], "temperature": 0.0},
        "stream": True,
    }
    t0 = time.monotonic()
    first_token_time = None
    token_count = 0
    token_times = []
    last_meta_info = {}

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results_list.append({
                    "success": False,
                    "input_len": req["input_len"],
                    "output_len": req["output_len"],
                })
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
                    token_times.append(now)
                    if isinstance(chunk, dict) and "meta_info" in chunk:
                        last_meta_info = chunk["meta_info"]
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        results_list.append({
            "success": False,
            "input_len": req["input_len"],
            "output_len": req["output_len"],
            "e2e_s": time.monotonic() - t0,
            "error": str(e),
        })
        return

    t_end = time.monotonic()
    e2e_s = t_end - t0

    # TTFT: client-side time to first token (includes queue wait)
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0.0
    # TTFT pure processing (server-side, excludes queue)
    ttft_proc_ms = 0.0
    if last_meta_info.get("ttft_pure_processing"):
        ttft_proc_ms = last_meta_info["ttft_pure_processing"] * 1000
    elif last_meta_info.get("time_to_first_token_processing"):
        ttft_proc_ms = last_meta_info["time_to_first_token_processing"] * 1000

    # TPOT: (last_token - first_token) / (tokens - 1), excludes queue
    tpot_ms = 0.0
    if token_count > 1 and first_token_time is not None:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)

    # Per-token inter-token latencies
    tpot_per_token_ms = None
    if len(token_times) > 1:
        tpot_per_token_ms = [(token_times[i] - token_times[i-1]) * 1000
                             for i in range(1, len(token_times))]

    # SLO: use ttft_proc (no queue) if available, else ttft_ms
    ttft_for_slo = ttft_proc_ms if ttft_proc_ms > 0 else ttft_ms
    ttft_violated = ttft_for_slo > ttft_slo_ms if ttft_for_slo > 0 else False
    tpot_violated = tpot_ms > tpot_slo_ms if tpot_ms > 0 else False

    results_list.append({
        "success": True,
        "input_len": req["input_len"],
        "output_len": req["output_len"],
        "completion_tokens": token_count,
        "ttft_ms": ttft_ms,
        "ttft_proc_ms": ttft_proc_ms,
        "tpot_ms": tpot_ms,
        "tpot_per_token_ms": tpot_per_token_ms,
        "e2e_s": e2e_s,
        "ttft_violated": ttft_violated,
        "tpot_violated": tpot_violated,
        "slo_violated": ttft_violated or tpot_violated,
    })


async def run_workload_async(workload_path, url, max_run_s=600,
                             ttft_slo_ms=DEFAULT_TTFT_SLO_MS,
                             tpot_slo_ms=DEFAULT_TPOT_SLO_MS):
    """Run a workload trace, collect per-request metrics + energy + SLO."""
    with open(workload_path) as f:
        requests_data = [json.loads(line) for line in f]

    log.info("  Workload: %s (%d reqs, SLO: TTFT<%dms TPOT<%dms)",
             Path(workload_path).stem, len(requests_data),
             int(ttft_slo_ms), int(tpot_slo_ms))

    # Energy measurement start
    energy_start = get_gpu_energy_mj(GPUS)
    freq_samples = []
    results_list = []

    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = []
        for i, req in enumerate(requests_data):
            arrival = req["arrival_time_s"]
            now = time.monotonic() - base_time
            if arrival > now:
                await asyncio.sleep(arrival - now)

            tasks.append(asyncio.create_task(
                _send_request(session, url, req, base_time, results_list,
                              ttft_slo_ms, tpot_slo_ms)))

            # Sample GPU frequency every 20 requests
            if i % 20 == 0:
                freq_samples.append({
                    "time_s": round(time.monotonic() - base_time, 1),
                    "freqs": get_gpu_freq_mhz(GPUS),
                })

        # Wait for all with timeout
        if tasks:
            gather_task = asyncio.gather(*tasks, return_exceptions=True)
            try:
                await asyncio.wait_for(gather_task, timeout=max_run_s)
            except asyncio.TimeoutError:
                log.warning("  Workload timed out after %ds", max_run_s)
                gather_task.cancel()
                try:
                    await gather_task
                except (Exception, asyncio.CancelledError):
                    pass

    duration_s = time.monotonic() - base_time
    energy_end = get_gpu_energy_mj(GPUS)

    # Energy calculation (millijoules → joules)
    total_energy_j = sum((energy_end.get(idx, 0) - energy_start.get(idx, 0)) / 1000.0
                         for idx in GPUS)

    # Result aggregation
    successful = [r for r in results_list if r.get("success")]
    failed = [r for r in results_list if not r.get("success")]

    if not successful:
        return {"status": "FAIL", "total_requests": len(requests_data),
                "successful": 0, "failed": len(failed)}

    ttfts_proc = [r["ttft_proc_ms"] for r in successful if r.get("ttft_proc_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in successful if r.get("ttft_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in successful if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in successful)
    throughput = total_tokens / duration_s if duration_s > 0 else 0

    # SLO violations
    n_ttft_viol = sum(1 for r in successful if r.get("ttft_violated"))
    n_tpot_viol = sum(1 for r in successful if r.get("tpot_violated"))
    n_slo_viol = sum(1 for r in successful if r.get("slo_violated"))
    n_total_viol = n_slo_viol + len(failed)
    slo_viol_rate = n_total_viol / len(results_list) * 100 if results_list else 0.0

    summary = {
        "status": "PASS",
        "duration_s": round(duration_s, 1),
        "total_requests": len(requests_data),
        "successful": len(successful),
        "failed": len(failed),
        "total_output_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 1),
        # TTFT with queue (client-side: first_token - request_sent)
        "ttft_avg_ms": round(float(np.mean(ttfts)), 1) if ttfts else 0,
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        "ttft_p90_ms": round(float(np.percentile(ttfts, 90)), 1) if ttfts else 0,
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        # TTFT pure processing (server-side, no queue)
        "ttft_proc_avg_ms": round(float(np.mean(ttfts_proc)), 1) if ttfts_proc else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(ttfts_proc, 50)), 1) if ttfts_proc else 0,
        "ttft_proc_p90_ms": round(float(np.percentile(ttfts_proc, 90)), 1) if ttfts_proc else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(ttfts_proc, 99)), 1) if ttfts_proc else 0,
        # TPOT (from first_token to last_token, no queue wait)
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p90_ms": round(float(np.percentile(tpots, 90)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        # Energy
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens > 0 else 0,
        # SLO
        "ttft_slo_ms": ttft_slo_ms,
        "tpot_slo_ms": tpot_slo_ms,
        "ttft_violations": n_ttft_viol,
        "tpot_violations": n_tpot_viol,
        "slo_violations": n_slo_viol,
        "slo_violation_rate": round(slo_viol_rate, 1),
        # Freq samples
        "freq_samples": freq_samples,
        # Per-request TPOT values for CDF plotting
        "tpot_values_ms": [round(t, 1) for t in tpots],
        "ttft_values_ms": [round(t, 1) for t in ttfts],
        "ttft_proc_values_ms": [round(t, 1) for t in ttfts_proc],
        # Per-request detailed breakdown
        "per_request_details": [
            {k: v for k, v in r.items()
             if k in ("input_len", "output_len", "completion_tokens",
                      "tpot_ms", "tpot_per_token_ms", "e2e_s",
                      "ttft_ms", "ttft_proc_ms")}
            for r in successful
        ],
    }
    return summary


DEPLOY_CONFIGS = {
    "native_dp8": {"method": "native_dp", "n_instances": 8, "tier": False},
    "native_dp8_tier": {"method": "native_dp", "n_instances": 8, "tier": True},
    "pd_dp4": {"method": "pd_dp", "n_pairs": 4, "tier": False},
    "pd_dp4_tier": {"method": "pd_dp", "n_pairs": 4, "tier": True},
    "pdaf_tp2": {"method": "pdaf", "micro_batch": 2, "tier": False},
    "pdaf_tp2_tier": {"method": "pdaf", "micro_batch": 2, "tier": True},
    "pdaf_asym_1p6d": {"method": "pdaf_asym", "tier": False},
    "pdaf_asym_1p6d_tier": {"method": "pdaf_asym", "tier": True},
    "pdaf_asym_4p4d": {"method": "pdaf_asym_4p4d", "tier": False},
    "pdaf_asym_4p4d_tier": {"method": "pdaf_asym_4p4d", "tier": True},
}


def main():
    parser = argparse.ArgumentParser(description="MoE model deployment benchmark")
    parser.add_argument("--deploy", default="all",
                        help="Deployment to test (comma-sep or 'all')")
    parser.add_argument("--test-generate", action="store_true",
                        help="Run quick generation test")
    parser.add_argument("--workloads", type=str, default=None,
                        help="Comma-separated paths to workload JSONLs (default: light,medium,heavy)")
    parser.add_argument("--ttft-slo-ms", type=float, default=DEFAULT_TTFT_SLO_MS,
                        help="TTFT SLO threshold (ms)")
    parser.add_argument("--tpot-slo-ms", type=float, default=DEFAULT_TPOT_SLO_MS,
                        help="TPOT SLO threshold (ms)")
    parser.add_argument("--max-run-s", type=int, default=600,
                        help="Max seconds per benchmark run")
    parser.add_argument("--keep-alive", action="store_true",
                        help="Don't kill servers after test")
    args = parser.parse_args()

    deploys = list(DEPLOY_CONFIGS.keys()) if args.deploy == "all" else args.deploy.split(",")

    # Determine workloads
    if args.workloads:
        workload_list = [Path(p.strip()) for p in args.workloads.split(",")]
    else:
        workload_list = WORKLOADS

    all_results = {}

    for deploy_name in deploys:
        if deploy_name not in DEPLOY_CONFIGS:
            log.error("Unknown deployment: %s", deploy_name)
            continue

        cfg = DEPLOY_CONFIGS[deploy_name]
        log.info("=" * 70)
        log.info("DEPLOYMENT: %s (tier=%s)", deploy_name, cfg["tier"])
        log.info("=" * 70)

        kill_all()
        time.sleep(5)
        mgr = DeploymentManager(LOG_DIR / deploy_name, tier=cfg["tier"])

        port = None
        if cfg["method"] == "native":
            port = mgr.start_native(gpus=cfg["gpus"], tp=cfg["tp"])
        elif cfg["method"] == "native_dp":
            port = mgr.start_native_dp(n_instances=cfg.get("n_instances", 8))
        elif cfg["method"] == "pd":
            port = mgr.start_pd(p_gpus=cfg["p_gpus"], d_gpus=cfg["d_gpus"], tp=cfg["tp"])
        elif cfg["method"] == "pd_dp":
            port = mgr.start_pd_dp(n_pairs=cfg.get("n_pairs", 4))
        elif cfg["method"] == "pdaf":
            port = mgr.start_pdaf(micro_batch=cfg["micro_batch"])
        elif cfg["method"] == "pdaf_asym":
            port = mgr.start_pdaf_asym(micro_batch=cfg.get("micro_batch", 2))
        elif cfg["method"] == "pdaf_asym_4p4d":
            port = mgr.start_pdaf_asym_4p4d(micro_batch=cfg.get("micro_batch", 2))
        elif cfg["method"] == "pdaf_dp":
            port = mgr.start_pdaf_dp(micro_batch=cfg["micro_batch"])

        if port is None:
            log.error("Deployment %s FAILED to start!", deploy_name)
            mgr.cleanup()
            continue

        # Lock GPU frequency: non-Tier → max freq; Tier → let DVFS manage
        if not cfg["tier"]:
            lock_gpu_freq(GPUS, MAX_GPU_FREQ)

        if args.test_generate:
            log.info("Running generation test...")
            success = test_generate(port)
            log.info("  Result: %s", "PASS" if success else "FAIL")
            if not success:
                log.error("  Generation test failed, skipping workloads")
                unlock_gpu_freq(GPUS)
                mgr.cleanup()
                kill_all()
                continue

        # Run all workloads
        url = f"http://127.0.0.1:{port}/generate"
        deploy_results = {}

        for wl_path in workload_list:
            if not wl_path.exists():
                log.warning("  Workload not found: %s", wl_path)
                continue

            wl_name = wl_path.stem
            log.info("-" * 50)
            log.info("Workload: %s (max_run=%ds)", wl_name, args.max_run_s)
            log.info("-" * 50)

            summary = asyncio.run(run_workload_async(
                str(wl_path), url, args.max_run_s,
                ttft_slo_ms=args.ttft_slo_ms,
                tpot_slo_ms=args.tpot_slo_ms))

            # Log results
            if summary["status"] == "PASS":
                log.info("  PASS: %d/%d requests",
                         summary["successful"], summary["total_requests"])
                log.info("  Throughput: %.1f tok/s", summary["throughput_tok_s"])
                log.info("  TTFT proc: avg=%.1fms p50=%.1fms p99=%.1fms",
                         summary["ttft_proc_avg_ms"],
                         summary["ttft_proc_p50_ms"],
                         summary["ttft_proc_p99_ms"])
                log.info("  TPOT: avg=%.1fms p50=%.1fms p99=%.1fms",
                         summary["tpot_avg_ms"],
                         summary["tpot_p50_ms"],
                         summary["tpot_p99_ms"])
                log.info("  Energy: %.1f J (%.2f mJ/tok)",
                         summary["total_energy_j"],
                         summary["energy_per_token_mj"])
                log.info("  SLO violation: %.1f%% (TTFT=%d, TPOT=%d, failed=%d)",
                         summary["slo_violation_rate"],
                         summary["ttft_violations"],
                         summary["tpot_violations"],
                         summary["failed"])
            else:
                log.error("  FAIL: %s", summary)

            # Save individual result
            out_dir = RESULTS_DIR / "json"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_file = out_dir / f"{deploy_name}_{wl_name}.json"
            result_data = {
                "deploy": deploy_name,
                "workload": wl_name,
                "tier": cfg["tier"],
                "model": MODEL,
                "ttft_slo_ms": args.ttft_slo_ms,
                "tpot_slo_ms": args.tpot_slo_ms,
                **summary,
            }
            with open(out_file, "w") as f:
                json.dump(result_data, f, indent=2)
            log.info("  Saved: %s", out_file)

            deploy_results[wl_name] = summary
            time.sleep(5)

        all_results[deploy_name] = deploy_results

        # Unlock GPU frequency after each deployment test
        unlock_gpu_freq(GPUS)

        if not args.keep_alive:
            mgr.cleanup()
            kill_all()
        else:
            log.info("Servers kept alive. Router at port %d", port)
            break

    # Print summary table
    _print_summary(all_results)
    log.info("All benchmarks complete.")


def _print_summary(all_results):
    """Print a final comparison table."""
    print("\n" + "=" * 100)
    print("  MoE BENCHMARK SUMMARY (Qwen3-30B-A3B)")
    print("=" * 100)
    header = f"{'Deploy':<22} {'Workload':<30} {'Thpt':>7} {'TTFT':>7} {'TPOT':>7} {'Energy':>8} {'mJ/tok':>7} {'SLO%':>6}"
    print(header)
    print("-" * 100)
    for deploy, wl_results in all_results.items():
        for wl, m in wl_results.items():
            if m.get("status") != "PASS":
                print(f"  {deploy:<22} {wl:<30} {'FAIL':>7}")
                continue
            print(f"  {deploy:<22} {wl:<30} "
                  f"{m['throughput_tok_s']:>7.1f} "
                  f"{m['ttft_proc_avg_ms']:>7.0f} "
                  f"{m['tpot_avg_ms']:>7.0f} "
                  f"{m['total_energy_j']:>8.0f} "
                  f"{m['energy_per_token_mj']:>7.1f} "
                  f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
