#!/usr/bin/env python3
"""Macro-benchmark runner for Mixtral-8x7B on 8 GPUs.

Tests 8 deployment schemes across 6 Azure workloads:
  - native_dp4 / native_dp4_tier
  - pd_2p2d / pd_2p2d_tier
  - pdaf_tp2 / pdaf_tp2_tier
  - pdaf_hetero / pdaf_hetero_tier

Usage:
    python run_macro_bench.py
    python run_macro_bench.py --deploys native_dp4,pdaf_tp2
    python run_macro_bench.py --deploys pdaf_hetero_tier --workloads workload_azure_conv_light_real.jsonl
"""
from __future__ import annotations

import resource
resource.setrlimit(resource.RLIMIT_MEMLOCK,
                   (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

import argparse
import asyncio
import json
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import aiohttp
import numpy as np
import requests

# ============================================================
# Constants
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("macro_bench")

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Mixtral/Mixtral-8x7B/"
HERE = Path(__file__).resolve().parent
WORKLOAD_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/macro-dataset")
RESULT_DIR = HERE / "results"
LOG_DIR = HERE / "logs"
ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/models"
AFD_ENERGY_MODEL_DIR = ENERGY_MODEL_DIR

MAX_GPU_FREQ = 1410
ROUTER_PORT = 42000
PA_PORT = 42010
PF_PORT = 42011
DA_PORT = 42020
DF_PORT = 42021
TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0
WORKLOAD_TIMEOUT_S = 600
HEALTH_TIMEOUT_S = 600
POWER_SAMPLE_INTERVAL_S = 0.5

ALL_GPUS = list(range(8))
ALL_PORTS = (list(range(42000, 42030)) + list(range(53100, 53180)) +
             list(range(53200, 53280)) + [49999] + list(range(49100, 49140)))

WORKLOAD_FILES = [
    "workload_azure_code_heavy_real.jsonl",
    "workload_azure_code_light_real.jsonl",
    "workload_azure_code_medium_real.jsonl",
    "workload_azure_conv_heavy_real.jsonl",
    "workload_azure_conv_light_real.jsonl",
    "workload_azure_conv_medium.jsonl",
]

ALL_DEPLOYS = [
    "native_dp4", "native_dp4_tier",
    "pd_2p2d", "pd_2p2d_tier",
    "pdaf_tp2", "pdaf_tp2_tier",
    "pdaf_hetero", "pdaf_hetero_tier",
]

# ============================================================
# Utility functions
# ============================================================

def kill_all():
    """Kill sglang processes on managed ports via ss -tlnp."""
    for port in ALL_PORTS:
        r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"],
                           capture_output=True, text=True)
        if "pid=" in r.stdout:
            for m in re.finditer(r"pid=(\d+)", r.stdout):
                pid = int(m.group(1))
                try:
                    os.kill(pid, 9)
                except OSError:
                    pass
    subprocess.run(["pkill", "-9", "-f", "sglang_router.launch_router"],
                   capture_output=True)
    time.sleep(3)


def wait_health(port, timeout=HEALTH_TIMEOUT_S, check_model_info=False):
    """Wait for a server endpoint to become healthy."""
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


def lock_gpu_freq(gpu_indices, freq_mhz=MAX_GPU_FREQ):
    """Lock GPU frequency using DVFSController (NVML C library)."""
    from sglang.srt.layers.dvfs import DVFSController
    for idx in gpu_indices:
        ctrl = DVFSController(device_index=idx)
        ret = ctrl.lock_sm_clock(freq_mhz)
        if ret != 0:
            log.warning("  Failed to lock GPU %d to %d MHz (ret=%d), fallback to nvidia-smi", idx, freq_mhz, ret)
            subprocess.run(["nvidia-smi", "-i", str(idx),
                            f"--lock-gpu-clocks={freq_mhz},{freq_mhz}"],
                           capture_output=True)
    log.info("  Locked GPU %s to %d MHz (NVML)", gpu_indices, freq_mhz)


def unlock_gpu_freq(gpu_indices):
    """Unlock GPU frequency using DVFSController (NVML C library)."""
    from sglang.srt.layers.dvfs import DVFSController
    for idx in gpu_indices:
        ctrl = DVFSController(device_index=idx)
        ret = ctrl.unlock_sm_clock()
        if ret != 0:
            subprocess.run(["nvidia-smi", "-i", str(idx), "--reset-gpu-clocks"],
                           capture_output=True)
    log.info("  Unlocked GPU %s (NVML)", gpu_indices)


def test_generate(port, n_tokens=8):
    url = f"http://127.0.0.1:{port}/generate"
    payload = {"text": "Hello, explain quantum computing:",
               "sampling_params": {"max_new_tokens": n_tokens, "temperature": 0.0}}
    try:
        r = requests.post(url, json=payload, timeout=120)
        data = r.json()
        return "text" in data
    except Exception as e:
        log.error("Generate test failed: %s", e)
        return False


# ============================================================
# Power Monitor (background thread)
# ============================================================

class PowerMonitor:
    """Sample GPU power via nvidia-smi in background thread."""

    def __init__(self, gpu_indices, interval=POWER_SAMPLE_INTERVAL_S):
        self.gpu_indices = gpu_indices
        self.interval = interval
        self.samples = []
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        return self._compute_energy()

    def _run(self):
        gpu_str = ",".join(str(g) for g in self.gpu_indices)
        while not self._stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi",
                     f"--id={gpu_str}",
                     "--query-gpu=power.draw",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                if r.returncode == 0:
                    powers = [float(x.strip()) for x in r.stdout.strip().split("\n") if x.strip()]
                    self.samples.append((time.time(), sum(powers)))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def _compute_energy(self):
        if len(self.samples) < 2:
            return 0.0
        energy_j = 0.0
        for i in range(1, len(self.samples)):
            dt = self.samples[i][0] - self.samples[i - 1][0]
            avg_power = (self.samples[i][1] + self.samples[i - 1][1]) / 2.0
            energy_j += avg_power * dt
        return energy_j
# ============================================================
# Deploy Manager
# ============================================================

class DeployManager:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs = []

    def _popen(self, name, cmd, env):
        fh = open(self.log_dir / f"{name}.log", "w")
        cmd = ["prlimit", "--memlock=unlimited:unlimited"] + cmd
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

    def _base_env(self):
        env = os.environ.copy()
        env["SGLANG_DISABLE_REQUEST_LOGGING"] = "true"
        env["UCX_LOG_LEVEL"] = "fatal"
        return env
    # --- Native DP4 (4 instances of TP=2) ---
    def start_native_dp4(self, tier=False):
        """4 instances: (0,1), (2,3), (4,5), (6,7). Round-robin router."""
        env_base = self._base_env()
        gpu_pairs = [(0, 1), (2, 3), (4, 5), (6, 7)]
        ports = []
        for i, (g0, g1) in enumerate(gpu_pairs):
            port = 53200 + i * 10
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = f"{g0},{g1}"
            if tier:
                env["AFD_NVML_DEVICE_INDICES"] = f"{g0},{g1}"
                env["AFD_NVML_DEVICE_INDEX"] = str(g0)
            cmd = [PYTHON, "-m", "sglang.launch_server",
                   "--model-path", MODEL, "--tp", "2",
                   "--host", "127.0.0.1", "--port", str(port),
                   "--nccl-port", str(33300 + i * 10),
                   "--mem-fraction-static", "0.85",
                   "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                   "--skip-server-warmup", "--disable-radix-cache"]
            if tier:
                cmd += ["--dvfs-enabled",
                        "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                        "--dvfs-ttft-slo-ms", str(int(TTFT_SLO_MS)),
                        "--dvfs-tpot-slo-us", str(int(TPOT_SLO_MS * 1000))]
            self._popen(f"dp_{i}", cmd, env)
            ports.append(port)
            time.sleep(2)

        for i, port in enumerate(ports):
            if not wait_health(port, HEALTH_TIMEOUT_S):
                log.error("DP instance %d failed (port %d)", i, port)
                return None
            log.info("  DP instance %d ready (port %d)", i, port)

        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
                 "--policy", "round_robin",
                 "--worker-urls"] + [f"http://127.0.0.1:{p}" for p in ports]
        self._popen("router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("Native DP4 router failed")
            return None
        log.info("Native DP4 ready (tier=%s)", tier)
        return ROUTER_PORT
    # --- PD 2P+2D (PD disaggregation) ---
    def start_pd_2p2d(self, tier=False):
        """2 PD pairs: P(0,1)+D(4,5), P(2,3)+D(6,7). DP=2."""
        env_base = self._base_env()
        instances = [
            {"p_cvd": "0,1", "d_cvd": "4,5", "p_port": 53100, "d_port": 53101, "bs_port": 49100},
            {"p_cvd": "2,3", "d_cvd": "6,7", "p_port": 53110, "d_port": 53111, "bs_port": 49110},
        ]
        for idx, inst in enumerate(instances):
            env_p = env_base.copy()
            env_p["CUDA_VISIBLE_DEVICES"] = inst["p_cvd"]
            cmd_p = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "2",
                     "--host", "127.0.0.1", "--port", str(inst["p_port"]),
                     "--nccl-port", str(34000 + idx * 10),
                     "--mem-fraction-static", "0.85",
                     "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup", "--disable-radix-cache",
                     "--disaggregation-mode", "prefill",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(inst["bs_port"]),
                     "--disaggregation-ib-device", "mlx5_4"]
            self._popen(f"pd{idx}_p", cmd_p, env_p)
            time.sleep(3)

            env_d = env_base.copy()
            env_d["CUDA_VISIBLE_DEVICES"] = inst["d_cvd"]
            if tier:
                env_d["AFD_NVML_DEVICE_INDICES"] = inst["d_cvd"]
                env_d["AFD_NVML_DEVICE_INDEX"] = inst["d_cvd"].split(",")[0]
            cmd_d = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "2",
                     "--host", "127.0.0.1", "--port", str(inst["d_port"]),
                     "--nccl-port", str(34001 + idx * 10),
                     "--mem-fraction-static", "0.85",
                     "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup", "--disable-radix-cache",
                     "--disaggregation-mode", "decode",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(inst["bs_port"]),
                     "--disaggregation-ib-device", "mlx5_4"]
            if tier:
                cmd_d += ["--dvfs-enabled",
                          "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                          "--dvfs-ttft-slo-ms", str(int(TTFT_SLO_MS)),
                          "--dvfs-tpot-slo-us", str(int(TPOT_SLO_MS * 1000))]
            self._popen(f"pd{idx}_d", cmd_d, env_d)
            time.sleep(3)

        for idx, inst in enumerate(instances):
            for role, port in [("P", inst["p_port"]), ("D", inst["d_port"])]:
                if not wait_health(port, HEALTH_TIMEOUT_S):
                    log.error("PD%d %s failed (port %d)", idx, role, port)
                    return None
                log.info("  PD%d %s ready (port %d)", idx, role, port)

        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        for inst in instances:
            cmd_r += ["--prefill", f"http://127.0.0.1:{inst['p_port']}",
                      str(inst["bs_port"])]
        for inst in instances:
            cmd_r += ["--decode", f"http://127.0.0.1:{inst['d_port']}"]
        self._popen("pd_router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PD router failed")
            return None
        log.info("PD 2P2D ready (tier=%s)", tier)
        return ROUTER_PORT
    # --- PDAF Symmetric TP2 ---
    def _pdaf_env(self, cvd, ucx_base, sched_port, peer_device, nvml_idx, ffn_host=None):
        env = self._base_env()
        env["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
        env["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
        env["AFD_ASYNC_PIPELINE"] = "1"
        env["CUDA_VISIBLE_DEVICES"] = cvd
        env["AFD_UCX_BASE_PORT"] = str(ucx_base)
        env["AFD_SCHED_PORT"] = str(sched_port)
        env["AFD_IPC_SYNC_MODE"] = "ipc_event"
        env["AFD_IPC_PEER_DEVICE"] = str(peer_device)
        env["AFD_NVML_DEVICE_INDICES"] = str(nvml_idx)
        env["AFD_NVML_DEVICE_INDEX"] = str(nvml_idx).split(",")[0]
        env["SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT"] = "600"
        env["SGLANG_DISAGGREGATION_WAITING_TIMEOUT"] = "600"
        if ffn_host:
            env["AFD_UCX_FFN_HOST"] = ffn_host
        return env

    def _pdaf_common_args(self, tier=False):
        common = ["--model-path", MODEL,
                  "--host", "127.0.0.1",
                  "--afd-comm-backend", "ipc_cpp",
                  "--afd-micro-batch", "2",
                  "--afd-dynamic-micro-batch",
                  "--mem-fraction-static", "0.85",
                  "--max-running-requests", "512",
                  "--skip-server-warmup",
                  "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                  "--afd-disagg-interleave-poll",
                  "--disable-radix-cache",
                  "--num-reserved-decode-tokens", "512",
                  "--disaggregation-transfer-backend", "mooncake",
                  "--disaggregation-bootstrap-port", "49999",
                  "--disaggregation-ib-device", "mlx5_4",
                  "--enable-metrics"]
        if tier:
            common += ["--afd-dvfs-enabled",
                       "--afd-energy-model-dir", AFD_ENERGY_MODEL_DIR,
                       "--afd-ttft-slo-ms", str(int(TTFT_SLO_MS)),
                       "--afd-tpot-slo-us", str(int(TPOT_SLO_MS * 1000)),
                       "--afd-dvfs-idle-lock"]
        return common
    def start_pdaf_tp2(self, tier=False):
        """Symmetric PDAF: PA(TP2)+PF(TP2) on CVD=0,1,2,3, DA(TP2)+DF(TP2) on CVD=4,5,6,7."""
        p_cvd = "0,1,2,3"
        d_cvd = "4,5,6,7"
        tp_ffn = 2
        tp_attn = 2
        common = self._pdaf_common_args(tier)
        ucx_p, ucx_d = 28200, 28300
        sched_p, sched_d = 68400, 68500

        def _cmd(port, tp, perspective, disagg, base_gpu_id):
            return [PYTHON, "-m", "sglang.launch_server",
                    "--port", str(port), "--tp", str(tp),
                    "--afd-perspective", perspective,
                    "--disaggregation-mode", disagg,
                    "--base-gpu-id", str(base_gpu_id)] + common

        # PF (FFN, base=0, TP2)
        self._popen("pf", _cmd(PF_PORT, tp_ffn, "ffn", "prefill", 0),
                    self._pdaf_env(p_cvd, ucx_p, sched_p, peer_device=tp_ffn,
                                   nvml_idx="0,1"))
        time.sleep(8)
        # PA (Attn, base=2, TP2)
        self._popen("pa", _cmd(PA_PORT, tp_attn, "attn", "prefill", tp_ffn),
                    self._pdaf_env(p_cvd, ucx_p, sched_p, peer_device=0,
                                   nvml_idx="2,3", ffn_host="127.0.0.1"))
        time.sleep(8)
        # DF (FFN, base=0, TP2)
        self._popen("df", _cmd(DF_PORT, tp_ffn, "ffn", "decode", 0),
                    self._pdaf_env(d_cvd, ucx_d, sched_d, peer_device=tp_ffn,
                                   nvml_idx="4,5"))
        time.sleep(10)
        # DA (Attn, base=2, TP2)
        self._popen("da", _cmd(DA_PORT, tp_attn, "attn", "decode", tp_ffn),
                    self._pdaf_env(d_cvd, ucx_d, sched_d, peer_device=0,
                                   nvml_idx="6,7", ffn_host="127.0.0.1"))

        log.info("Waiting for PDAF TP2 servers...")
        checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
                  (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
        for port, name, use_model_info in checks:
            if not wait_health(port, HEALTH_TIMEOUT_S, check_model_info=use_model_info):
                log.error("  %s (port %d) failed", name, port)
                return None
            log.info("  %s ready (port %d)", name, port)

        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation", "--mini-lb",
                 "--prefill", f"http://127.0.0.1:{PA_PORT}",
                 "--decode", f"http://127.0.0.1:{DA_PORT}",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        self._popen("router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PDAF TP2 router failed")
            return None
        log.info("PDAF TP2 ready (tier=%s)", tier)
        return ROUTER_PORT

    def start_pdaf_hetero(self, tier=False):
        """Heterogeneous PDAF: PF(TP4)+PA(TP1) on CVD=0,1,2,3,4, DF(TP2)+DA(TP1) on CVD=5,6,7."""
        p_cvd = "0,1,2,3,4"
        d_cvd = "5,6,7"
        tp_ffn_p, tp_attn_p = 4, 1
        tp_ffn_d, tp_attn_d = 2, 1
        common = self._pdaf_common_args(tier)
        ucx_p, ucx_d = 28200, 28300
        sched_p, sched_d = 68400, 68500

        def _cmd(port, tp, perspective, disagg, base_gpu_id):
            return [PYTHON, "-m", "sglang.launch_server",
                    "--port", str(port), "--tp", str(tp),
                    "--afd-perspective", perspective,
                    "--disaggregation-mode", disagg,
                    "--base-gpu-id", str(base_gpu_id)] + common

        # PF (FFN TP4, base=0, logical GPUs 0-3)
        self._popen("pf", _cmd(PF_PORT, tp_ffn_p, "ffn", "prefill", 0),
                    self._pdaf_env(p_cvd, ucx_p, sched_p,
                                   peer_device=tp_ffn_p, nvml_idx="0,1,2,3"))
        time.sleep(10)
        # PA (Attn TP1, base=4, logical GPU 4)
        self._popen("pa", _cmd(PA_PORT, tp_attn_p, "attn", "prefill", tp_ffn_p),
                    self._pdaf_env(p_cvd, ucx_p, sched_p,
                                   peer_device=0, nvml_idx="4",
                                   ffn_host="127.0.0.1"))
        time.sleep(8)
        # DF (FFN TP2, base=0, logical GPUs 0-1)
        self._popen("df", _cmd(DF_PORT, tp_ffn_d, "ffn", "decode", 0),
                    self._pdaf_env(d_cvd, ucx_d, sched_d,
                                   peer_device=tp_ffn_d, nvml_idx="5,6"))
        time.sleep(10)
        # DA (Attn TP1, base=2, logical GPU 2)
        self._popen("da", _cmd(DA_PORT, tp_attn_d, "attn", "decode", tp_ffn_d),
                    self._pdaf_env(d_cvd, ucx_d, sched_d,
                                   peer_device=0, nvml_idx="7",
                                   ffn_host="127.0.0.1"))

        log.info("Waiting for PDAF Hetero servers...")
        checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
                  (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
        for port, name, use_model_info in checks:
            if not wait_health(port, HEALTH_TIMEOUT_S, check_model_info=use_model_info):
                log.error("  %s (port %d) failed", name, port)
                return None
            log.info("  %s ready (port %d)", name, port)

        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation", "--mini-lb",
                 "--prefill", f"http://127.0.0.1:{PA_PORT}",
                 "--decode", f"http://127.0.0.1:{DA_PORT}",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        self._popen("router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PDAF Hetero router failed")
            return None
        log.info("PDAF Hetero ready (tier=%s)", tier)
        return ROUTER_PORT


# ============================================================
# Workload runner (async streaming client)
# ============================================================

async def send_one(session, url, req, base_time, results):
    """Send a single request following arrival_time_s."""
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"text": "x" * req["input_len"],
               "sampling_params": {"max_new_tokens": req["output_len"],
                                   "temperature": 0.0,
                                   "ignore_eos": True},
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
    tpot_ms = 0.0
    if token_count > 1 and first_token_time:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)

    results.append({
        "success": True,
        "completion_tokens": token_count,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
        "e2e_s": t_end - t0,
    })


async def run_workload(reqs, url, gpu_indices, timeout_s=WORKLOAD_TIMEOUT_S):
    """Run all requests and collect metrics with power monitoring."""
    power_mon = PowerMonitor(gpu_indices)
    power_mon.start()
    results = []
    client_timeout = aiohttp.ClientTimeout(total=timeout_s + 60)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        base_time = time.monotonic()
        tasks = [asyncio.create_task(send_one(session, url, r, base_time, results))
                 for r in reqs]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=timeout_s)
            except asyncio.TimeoutError:
                log.warning("Workload timed out after %ds", timeout_s)

    duration_s = time.monotonic() - base_time
    total_energy_j = power_mon.stop()

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]

    fail_rate = len(fail) / len(reqs) if reqs else 0
    if fail_rate > 0.3:
        return {"status": "CRASHED", "total_requests": len(reqs),
                "successful": len(ok), "failed": len(fail),
                "fail_rate": round(fail_rate * 100, 1)}
    if not ok:
        return {"status": "FAIL", "failed": len(fail)}

    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0

    n_ttft_viol = sum(1 for t in ttfts if t > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for t in tpots if t > TPOT_SLO_MS)
    slo_violations = n_ttft_viol + n_tpot_viol + len(fail)
    slo_rate = slo_violations / len(reqs) * 100 if reqs else 0
    success_rate = len(ok) / len(reqs) * 100 if reqs else 0

    return {
        "status": "PASS",
        "duration_s": round(duration_s, 1),
        "total_requests": len(reqs),
        "successful": len(ok),
        "failed": len(fail),
        "success_rate": round(success_rate, 1),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 1),
        "ttft_avg_ms": round(float(np.mean(ttfts)), 1) if ttfts else 0,
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens > 0 else 0,
        "slo_violation_rate": round(slo_rate, 1),
        "ttft_violations": n_ttft_viol,
        "tpot_violations": n_tpot_viol,
    }


# ============================================================
# Deploy dispatcher
# ============================================================

def deploy_scheme(mgr, deploy_name):
    """Start the given deployment scheme. Returns router port or None."""
    tier = deploy_name.endswith("_tier")
    base = deploy_name.replace("_tier", "")
    if base == "native_dp4":
        return mgr.start_native_dp4(tier=tier)
    elif base == "pd_2p2d":
        return mgr.start_pd_2p2d(tier=tier)
    elif base == "pdaf_tp2":
        return mgr.start_pdaf_tp2(tier=tier)
    elif base == "pdaf_hetero":
        return mgr.start_pdaf_hetero(tier=tier)
    else:
        log.error("Unknown deployment: %s", deploy_name)
        return None


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Macro benchmark for Mixtral-8x7B")
    parser.add_argument("--deploys", default=None,
                        help="Comma-sep deploy names (default: all 8)")
    parser.add_argument("--workloads", default=None,
                        help="Comma-sep workload filenames (override default)")
    parser.add_argument("--timeout", type=int, default=WORKLOAD_TIMEOUT_S,
                        help="Per-workload timeout in seconds")
    args = parser.parse_args()

    # Check ulimit
    soft, hard = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    log.info("ulimit -l: soft=%s hard=%s", soft, hard)

    deploys = ALL_DEPLOYS if args.deploys is None else args.deploys.split(",")
    if args.workloads:
        wl_files = [f.strip() for f in args.workloads.split(",")]
    else:
        wl_files = WORKLOAD_FILES

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for deploy_name in deploys:
        if deploy_name not in ALL_DEPLOYS:
            log.error("Unknown deploy: %s (valid: %s)", deploy_name, ALL_DEPLOYS)
            continue

        log.info("=" * 70)
        log.info("DEPLOY: %s", deploy_name)
        log.info("=" * 70)

        kill_all()
        time.sleep(5)
        tier = deploy_name.endswith("_tier")
        mgr = DeployManager(LOG_DIR / deploy_name)
        port = deploy_scheme(mgr, deploy_name)

        if port is None:
            log.error("Deploy %s FAILED to start", deploy_name)
            mgr.cleanup()
            all_results[deploy_name] = {"_status": "DEPLOY_FAILED"}
            continue

        if not tier:
            lock_gpu_freq(ALL_GPUS, MAX_GPU_FREQ)
        else:
            unlock_gpu_freq(ALL_GPUS)

        log.info("Warmup...")
        if not test_generate(port):
            log.error("Warmup generate failed for %s", deploy_name)
            if not tier:
                unlock_gpu_freq(ALL_GPUS)
            mgr.cleanup()
            kill_all()
            all_results[deploy_name] = {"_status": "WARMUP_FAILED"}
            continue
        time.sleep(3)

        url = f"http://127.0.0.1:{port}/generate"
        deploy_results = {}

        for wl_fname in wl_files:
            wl_path = WORKLOAD_DIR / wl_fname
            if not wl_path.exists():
                log.warning("Workload not found: %s", wl_path)
                continue

            with open(wl_path) as f:
                reqs = [json.loads(l) for l in f if l.strip()]

            wl_key = wl_fname.replace(".jsonl", "")
            log.info("-" * 50)
            log.info("  %s (%d requests)", wl_key, len(reqs))
            log.info("-" * 50)

            summary = asyncio.run(
                run_workload(reqs, url, ALL_GPUS, args.timeout))

            if summary.get("status") == "PASS":
                log.info("  Thpt=%.1f tok/s | TTFT=%.1f/%.1f/%.1fms | "
                         "TPOT=%.1f/%.1f/%.1fms | E=%.0fJ | SLO=%.1f%%",
                         summary["throughput_tok_s"],
                         summary["ttft_avg_ms"], summary["ttft_p50_ms"],
                         summary["ttft_p99_ms"],
                         summary["tpot_avg_ms"], summary["tpot_p50_ms"],
                         summary["tpot_p99_ms"],
                         summary["total_energy_j"],
                         summary["slo_violation_rate"])
            elif summary.get("status") == "CRASHED":
                log.error("  CRASHED: >30%% requests failed (%d/%d)",
                          summary["failed"], summary["total_requests"])
            else:
                log.error("  FAIL: %s", summary)

            deploy_results[wl_key] = summary
            time.sleep(5)

        all_results[deploy_name] = deploy_results

        if not tier:
            unlock_gpu_freq(ALL_GPUS)
        mgr.cleanup()
        kill_all()

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULT_DIR / f"macro_mixtral_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved: %s", out_file)

    # Print summary table
    print("\n" + "=" * 130)
    print("  MACRO BENCHMARK (Mixtral-8x7B, 8 GPU)")
    print("=" * 130)
    hdr = (f"{'Deploy':<20} {'Workload':<38} {'Thpt':>7} "
           f"{'TTFT_avg':>8} {'TTFT_p99':>8} {'TPOT_avg':>8} {'TPOT_p99':>8} "
           f"{'Energy':>8} {'SLO%':>6} {'OK':>5}")
    print(hdr)
    print("-" * 130)
    for dep, wl_results in all_results.items():
        if isinstance(wl_results, dict) and "_status" in wl_results:
            print(f"  {dep:<20} ** {wl_results['_status']} **")
            continue
        for wl, m in wl_results.items():
            if m.get("status") not in ("PASS",):
                status = m.get("status", "FAIL")
                print(f"  {dep:<20} {wl:<38} {status}")
                continue
            print(f"  {dep:<20} {wl:<38} "
                  f"{m['throughput_tok_s']:>7.1f} "
                  f"{m['ttft_avg_ms']:>8.1f} "
                  f"{m['ttft_p99_ms']:>8.1f} "
                  f"{m['tpot_avg_ms']:>8.1f} "
                  f"{m['tpot_p99_ms']:>8.1f} "
                  f"{m['total_energy_j']:>8.0f} "
                  f"{m['slo_violation_rate']:>6.1f} "
                  f"{m['successful']:>5}")
    print("=" * 130)


if __name__ == "__main__":
    main()
