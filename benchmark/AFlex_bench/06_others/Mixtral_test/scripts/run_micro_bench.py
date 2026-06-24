#!/usr/bin/env python3
"""Micro-benchmark runner for Mixtral-8x7B (MoE).

Tests 2 architectures (Native/PD) on 4 GPU with TP=2.
PDAF requires 8 GPUs for MoE models (TP=2 per component).

  4GPU: Native DP2(TP=2) / PD(P-TP2+D-TP2), QPS=1/2/3

Usage:
    python run_micro_bench.py --ngpu 4 --deploy all
    python run_micro_bench.py --ngpu 4 --deploy native_dp,pd_dp --scenario chatbot
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
log = logging.getLogger("micro_bench")

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Mixtral/Mixtral-8x7B/"
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
WORKLOAD_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/retesting/workloads")
ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/models_v2"
AFD_ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/models_v2"

MAX_GPU_FREQ = 1410
ROUTER_PORT = 42000

# PDAF ports
PA_PORT = 42010
PF_PORT = 42011
DA_PORT = 42020
DF_PORT = 42021

# SLO defaults
TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0


# ============================================================
# Utility functions
# ============================================================

ALL_PORTS = list(range(42000, 42030)) + list(range(53100, 53180)) + \
            list(range(53200, 53280)) + [49999] + list(range(49100, 49140))


def kill_all():
    """Kill sglang processes on managed ports only (avoid killing other GPU workloads)."""
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
    subprocess.run(["pkill", "-9", "-f", "sglang_router.launch_router"], capture_output=True)
    time.sleep(3)


def wait_health(port, timeout=300, check_model_info=False):
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


def lock_gpu_freq(gpu_indices, freq_mhz=MAX_GPU_FREQ):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-i", str(idx),
                        f"--lock-gpu-clocks={freq_mhz},{freq_mhz}"],
                       capture_output=True)
    log.info("  Locked GPU %s to %d MHz", gpu_indices, freq_mhz)


def unlock_gpu_freq(gpu_indices):
    for idx in gpu_indices:
        subprocess.run(["nvidia-smi", "-i", str(idx), "--reset-gpu-clocks"],
                       capture_output=True)
    log.info("  Unlocked GPU %s", gpu_indices)


def test_generate(port, n_tokens=16):
    url = f"http://127.0.0.1:{port}/generate"
    payload = {"text": "Hello, explain quantum computing:",
               "sampling_params": {"max_new_tokens": n_tokens, "temperature": 0.0}}
    try:
        r = requests.post(url, json=payload, timeout=60)
        data = r.json()
        return "text" in data
    except Exception as e:
        log.error("Generate test failed: %s", e)
        return False


# ============================================================
# Deployment Manager
# ============================================================

class DeployManager:
    def __init__(self, log_dir, tier=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs = []
        self.tier = tier

    @staticmethod
    def _set_memlock_unlimited():
        import resource
        resource.setrlimit(resource.RLIMIT_MEMLOCK,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

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

    # --- Native DP ---
    def start_native_dp(self, gpus):
        """Start DP instances with TP=2 (Mixtral needs 2 GPUs per instance)."""
        n = len(gpus) // 2  # DP parallelism
        env_base = self._base_env()
        ports = []
        for i in range(n):
            port = 53200 + i * 10
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = f"{gpus[i*2]},{gpus[i*2+1]}"
            cmd = [PYTHON, "-m", "sglang.launch_server",
                   "--model-path", MODEL, "--tp", "2",
                   "--host", "127.0.0.1", "--port", str(port),
                   "--nccl-port", str(33300 + i * 10),
                   "--mem-fraction-static", "0.85",
                   "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                   "--skip-server-warmup"]
            if self.tier:
                cmd += ["--dvfs-enabled",
                        "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                        "--dvfs-ttft-slo-ms", str(int(TTFT_SLO_MS)),
                        "--dvfs-tpot-slo-us", str(int(TPOT_SLO_MS * 1000))]
            self._popen(f"dp_{i}", cmd, env)
            ports.append(port)
            time.sleep(2)

        for i, port in enumerate(ports):
            if not wait_health(port, 300):
                log.error("DP instance %d failed (port %d)", i, port)
                return None
            log.info("  DP instance %d ready (port %d)", i, port)

        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
                 "--policy", "round_robin",
                 "--worker-urls"] + [f"http://127.0.0.1:{p}" for p in ports]
        self._popen("router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("Router failed")
            return None
        log.info("Native DP%d ready (tier=%s)", n, self.tier)
        return ROUTER_PORT

    # --- PD DP ---
    def start_pd_dp(self, gpu_pairs):
        """Start PD: each pair = (p_gpus, d_gpus) as comma-sep strings, TP=2."""
        env_base = self._base_env()
        instances = []
        for i, (p_gpus, d_gpus) in enumerate(gpu_pairs):
            instances.append({
                "p_cvd": p_gpus, "d_cvd": d_gpus,
                "p_port": 53100 + i * 10, "d_port": 53101 + i * 10,
                "bs_port": 49100 + i * 10,
            })

        for idx, inst in enumerate(instances):
            env_p = env_base.copy()
            env_p["CUDA_VISIBLE_DEVICES"] = inst["p_cvd"]
            cmd_p = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "2",
                     "--host", "127.0.0.1", "--port", str(inst["p_port"]),
                     "--nccl-port", str(34000 + idx * 10),
                     "--mem-fraction-static", "0.85",
                     "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup",
                     "--disaggregation-mode", "prefill",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(inst["bs_port"]),
                     "--disaggregation-ib-device", "mlx5_4"]
            self._popen(f"pd{idx}_p", cmd_p, env_p)
            time.sleep(3)

            env_d = env_base.copy()
            env_d["CUDA_VISIBLE_DEVICES"] = inst["d_cvd"]
            cmd_d = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "2",
                     "--host", "127.0.0.1", "--port", str(inst["d_port"]),
                     "--nccl-port", str(34001 + idx * 10),
                     "--mem-fraction-static", "0.85",
                     "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup",
                     "--disaggregation-mode", "decode",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(inst["bs_port"]),
                     "--disaggregation-ib-device", "mlx5_4"]
            if self.tier:
                cmd_d += ["--dvfs-enabled",
                          "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                          "--dvfs-ttft-slo-ms", str(int(TTFT_SLO_MS)),
                          "--dvfs-tpot-slo-us", str(int(TPOT_SLO_MS * 1000))]
            self._popen(f"pd{idx}_d", cmd_d, env_d)
            time.sleep(3)

        for idx, inst in enumerate(instances):
            for role, port in [("P", inst["p_port"]), ("D", inst["d_port"])]:
                if not wait_health(port, 300):
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
        log.info("PD DP%d ready (tier=%s)", len(gpu_pairs), self.tier)
        return ROUTER_PORT

    # --- PD Asymmetric (1P + nD) ---
    def start_pd_1pnd(self, p_gpus, d_gpu_list):
        """Start PD with 1 Prefill (TP=2) + N Decode (TP=2 each).
        
        Args:
            p_gpus: comma-sep GPU IDs for prefill, e.g. "2,3"
            d_gpu_list: list of comma-sep GPU IDs for each decode instance,
                       e.g. ["4,5", "6,7"]
        """
        env_base = self._base_env()
        bs_port = 49100

        # Start Prefill
        env_p = env_base.copy()
        env_p["CUDA_VISIBLE_DEVICES"] = p_gpus
        p_port = 53100
        cmd_p = [PYTHON, "-m", "sglang.launch_server",
                 "--model-path", MODEL, "--tp", "2",
                 "--host", "127.0.0.1", "--port", str(p_port),
                 "--nccl-port", "34000",
                 "--mem-fraction-static", "0.85",
                 "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                 "--skip-server-warmup",
                 "--disaggregation-mode", "prefill",
                 "--disaggregation-transfer-backend", "mooncake",
                 "--disaggregation-bootstrap-port", str(bs_port),
                 "--disaggregation-ib-device", "mlx5_4"]
        self._popen("pd_p", cmd_p, env_p)
        time.sleep(3)

        # Start Decode instances
        d_ports = []
        for i, d_gpus in enumerate(d_gpu_list):
            env_d = env_base.copy()
            env_d["CUDA_VISIBLE_DEVICES"] = d_gpus
            d_port = 53110 + i * 10
            cmd_d = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "2",
                     "--host", "127.0.0.1", "--port", str(d_port),
                     "--nccl-port", str(34010 + i * 10),
                     "--mem-fraction-static", "0.85",
                     "--disable-cuda-graph",
                     "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup",
                     "--disaggregation-mode", "decode",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(bs_port),
                     "--disaggregation-ib-device", "mlx5_4"]
            if self.tier:
                cmd_d += ["--dvfs-enabled",
                          "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
                          "--dvfs-ttft-slo-ms", str(int(TTFT_SLO_MS)),
                          "--dvfs-tpot-slo-us",
                          str(int(TPOT_SLO_MS * 1000))]
            self._popen(f"pd_d{i}", cmd_d, env_d)
            d_ports.append(d_port)
            time.sleep(3)

        # Wait for health
        if not wait_health(p_port, 300):
            log.error("PD P failed (port %d)", p_port)
            return None
        log.info("  P ready (port %d)", p_port)
        for i, dp in enumerate(d_ports):
            if not wait_health(dp, 300):
                log.error("PD D%d failed (port %d)", i, dp)
                return None
            log.info("  D%d ready (port %d)", i, dp)

        # Router
        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
                 "--prefill", f"http://127.0.0.1:{p_port}",
                 str(bs_port)]
        for dp in d_ports:
            cmd_r += ["--decode", f"http://127.0.0.1:{dp}"]
        self._popen("pd_router", cmd_r, os.environ.copy())
        if not wait_health(ROUTER_PORT, 60):
            log.error("PD 1P%dD router failed", len(d_gpu_list))
            return None
        log.info("PD 1P+%dD ready (tier=%s)", len(d_gpu_list), self.tier)
        return ROUTER_PORT

    # --- PDAF ---
    def start_pdaf(self, p_cvd, d_cvd, tp, micro_batch=2,
                   tp_attn=None, tp_ffn=None):
        """Start PDAF: PA+PF on p_cvd, DA+DF on d_cvd.
        
        Supports heterogeneous TP: tp_attn/tp_ffn override tp for A/F.
        E.g. tp_attn=1, tp_ffn=2 for MoE where FFN is larger.
        """
        if tp_attn is None:
            tp_attn = tp
        if tp_ffn is None:
            tp_ffn = tp
        env_base = self._base_env()
        env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
        env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
        env_base["AFD_ASYNC_PIPELINE"] = "1"

        common = ["--model-path", MODEL,
                  "--host", "127.0.0.1",
                  "--afd-comm-backend", "ipc_cpp",
                  "--afd-micro-batch", str(micro_batch),
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

        dvfs_args = []
        if self.tier:
            dvfs_args = ["--afd-dvfs-enabled",
                         "--afd-energy-model-dir", AFD_ENERGY_MODEL_DIR,
                         "--afd-ttft-slo-ms", str(int(TTFT_SLO_MS)),
                         "--afd-tpot-slo-us", str(int(TPOT_SLO_MS * 1000)),
                         "--afd-dvfs-idle-lock"]

        ucx_p, ucx_d = 28200, 28300
        sched_p, sched_d = 68400, 68500

        def _env(cvd, ucx_base, sched_port, peer_device, nvml_idx, ffn_host=None):
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
            if ffn_host:
                e["AFD_UCX_FFN_HOST"] = ffn_host
            return e

        def _cmd(port, perspective, disagg, base_gpu_id, comp_tp):
            return [PYTHON, "-m", "sglang.launch_server",
                    "--port", str(port),
                    "--tp", str(comp_tp),
                    "--afd-perspective", perspective,
                    "--disaggregation-mode", disagg,
                    "--base-gpu-id", str(base_gpu_id)] + common + dvfs_args

        # GPU layout: [FFN GPUs ... | Attn GPUs ...]
        # For heterogeneous TP: FFN uses first tp_ffn GPUs, Attn uses next tp_attn
        p_gpus = p_cvd.split(",")
        d_gpus = d_cvd.split(",")
        p_ffn_nvml = ",".join(p_gpus[:tp_ffn])
        p_attn_nvml = ",".join(p_gpus[tp_ffn:tp_ffn + tp_attn])
        d_ffn_nvml = ",".join(d_gpus[:tp_ffn])
        d_attn_nvml = ",".join(d_gpus[tp_ffn:tp_ffn + tp_attn])

        # PF (FFN, base=0, TP=tp_ffn)
        # IPC peer: Attn rank0 is at logical index tp_ffn
        self._popen("pf", _cmd(PF_PORT, "ffn", "prefill", 0, tp_ffn),
                    _env(p_cvd, ucx_p, sched_p, peer_device=tp_ffn,
                         nvml_idx=p_ffn_nvml))
        time.sleep(8)
        # PA (Attn, base=tp_ffn, TP=tp_attn)
        # IPC peer: FFN rank0 is at logical index 0
        self._popen("pa", _cmd(PA_PORT, "attn", "prefill", tp_ffn, tp_attn),
                    _env(p_cvd, ucx_p, sched_p, peer_device=0,
                         nvml_idx=p_attn_nvml, ffn_host="127.0.0.1"))
        time.sleep(8)
        # DF (FFN, base=0, TP=tp_ffn)
        self._popen("df", _cmd(DF_PORT, "ffn", "decode", 0, tp_ffn),
                    _env(d_cvd, ucx_d, sched_d, peer_device=tp_ffn,
                         nvml_idx=d_ffn_nvml))
        time.sleep(10)
        # DA (Attn, base=tp_ffn, TP=tp_attn)
        self._popen("da", _cmd(DA_PORT, "attn", "decode", tp_ffn, tp_attn),
                    _env(d_cvd, ucx_d, sched_d, peer_device=0,
                         nvml_idx=d_attn_nvml, ffn_host="127.0.0.1"))

        log.info("Waiting for PDAF servers...")
        checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
                  (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
        for port, name, use_model_info in checks:
            if not wait_health(port, 600, check_model_info=use_model_info):
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
            log.error("PDAF router failed")
            return None
        log.info("PDAF A-TP%d F-TP%d ready (tier=%s)", tp_attn, tp_ffn, self.tier)
        return ROUTER_PORT


# ============================================================
# Workload runner
# ============================================================

async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"text": "x" * req["input_len"],
               "sampling_params": {"max_new_tokens": req["output_len"],
                                   "temperature": 0.0},
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
        "success": True,
        "completion_tokens": token_count,
        "ttft_ms": ttft_ms,
        "ttft_proc_ms": ttft_proc_ms,
        "tpot_ms": tpot_ms,
        "e2e_s": t_end - t0,
    })


async def run_workload(reqs, url, gpu_indices, max_run_s=400):
    energy_start = get_gpu_energy_mj(gpu_indices)
    results = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [asyncio.create_task(send_one(session, url, r, base_time, results))
                 for r in reqs]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=max_run_s)
            except asyncio.TimeoutError:
                log.warning("Timed out after %ds", max_run_s)

    duration_s = time.monotonic() - base_time
    energy_end = get_gpu_energy_mj(gpu_indices)
    total_energy_j = sum((energy_end.get(i, 0) - energy_start.get(i, 0)) / 1000.0
                         for i in gpu_indices)

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    if not ok:
        return {"status": "FAIL", "failed": len(fail)}

    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0

    # SLO violations (use proc TTFT if available)
    n_ttft_viol = sum(1 for r in ok
                      if (r.get("ttft_proc_ms") or r["ttft_ms"]) > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for r in ok if r["tpot_ms"] > TPOT_SLO_MS)
    n_slo_viol = n_ttft_viol + n_tpot_viol + len(fail)
    slo_rate = n_slo_viol / len(results) * 100 if results else 0

    return {
        "status": "PASS",
        "duration_s": round(duration_s, 1),
        "total_requests": len(reqs),
        "successful": len(ok),
        "failed": len(fail),
        "total_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 1),
        "ttft_proc_avg_ms": round(float(np.mean(ttfts_proc)), 1) if ttfts_proc else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(ttfts_proc, 50)), 1) if ttfts_proc else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(ttfts_proc, 99)), 1) if ttfts_proc else 0,
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
# GPU configs per ngpu
# ============================================================

def get_deploy_configs(ngpu, gpus=None):
    """Return deploy configs for given GPU count.
    
    Args:
        ngpu: number of GPUs (4 or 8)
        gpus: explicit GPU id list, e.g. [1,2,5,6]. If None, uses 0..ngpu-1.
    """
    if gpus is None:
        gpus = list(range(ngpu))
    assert len(gpus) == ngpu, f"Expected {ngpu} GPUs, got {len(gpus)}: {gpus}"

    if ngpu == 4:
        g = gpus  # e.g. [0,1,2,3]
        p_cvd = f"{g[0]},{g[1]}"
        d_cvd = f"{g[2]},{g[3]}"
        return {
            "native_dp": {
                "start": lambda mgr: mgr.start_native_dp(g),
                "gpus": g,
            },
            "pd_dp": {
                "start": lambda mgr: mgr.start_pd_dp([(p_cvd, d_cvd)]),
                "gpus": g,
            },
        }
    elif ngpu == 6:
        # 6-GPU configs for Mixtral:
        # Native: TP2 × DP3 (all 6 GPUs)
        # PD: 1P(TP2) + 2D(TP2) (all 6 GPUs)
        # PDAF: PA(TP1)+PF(TP2) | DA(TP1)+DF(TP2) (all 6 GPUs)
        g = gpus
        p_cvd = f"{g[0]},{g[1]},{g[2]}"
        d_cvd = f"{g[3]},{g[4]},{g[5]}"
        return {
            "native_dp": {
                "start": lambda mgr: mgr.start_native_dp(g),
                "gpus": g,
            },
            "pd_dp": {
                "start": lambda mgr: mgr.start_pd_1pnd(
                    f"{g[0]},{g[1]}",
                    [f"{g[2]},{g[3]}", f"{g[4]},{g[5]}"]),
                "gpus": g,
            },
            "pdaf": {
                "start": lambda mgr, _p=p_cvd, _d=d_cvd: mgr.start_pdaf(
                    _p, _d, tp=2, tp_attn=1, tp_ffn=2),
                "gpus": g,
            },
        }
    else:  # 8 GPU
        g = gpus
        p_cvd = ",".join(str(x) for x in g[:4])
        d_cvd = ",".join(str(x) for x in g[4:])
        return {
            "native_dp": {
                "start": lambda mgr: mgr.start_native_dp(g),
                "gpus": g,
            },
            "pd_dp": {
                "start": lambda mgr: mgr.start_pd_dp(
                    [(f"{g[0]},{g[1]}", f"{g[2]},{g[3]}"),
                     (f"{g[4]},{g[5]}", f"{g[6]},{g[7]}")]),
                "gpus": g,
            },
            "pdaf": {
                "start": lambda mgr, _p=p_cvd, _d=d_cvd: mgr.start_pdaf(_p, _d, tp=2),
                "gpus": g,
            },
        }


SCENARIOS = ["chatbot", "qa", "rag", "summary"]


# ============================================================
# Main
# ============================================================

def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpu", type=int, required=True, choices=[4, 6, 8])
    parser.add_argument("--gpus", default=None,
                        help="Explicit GPU IDs (comma-sep), e.g. '1,2,5,6'")
    parser.add_argument("--deploy", default="all",
                        help="Comma-sep: native_dp,pd_dp,pdaf or 'all'")
    parser.add_argument("--scenario", default="all",
                        help="Comma-sep: chatbot,qa,rag,summary or 'all'")
    parser.add_argument("--qps", default=None,
                        help="Override QPS list (comma-sep integers)")
    parser.add_argument("--tier", action="store_true",
                        help="Enable Tier DVFS")
    parser.add_argument("--max-run-s", type=int, default=400)
    args = parser.parse_args()

    gpu_list = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    configs = get_deploy_configs(args.ngpu, gpus=gpu_list)
    deploys = list(configs.keys()) if args.deploy == "all" else args.deploy.split(",")
    scenarios = SCENARIOS if args.scenario == "all" else args.scenario.split(",")

    if args.qps:
        qps_list = [int(q) for q in args.qps.split(",")]
    elif args.ngpu == 4:
        qps_list = [1, 2, 3]
    else:
        qps_list = [1, 2, 3, 4, 5, 6]

    tier_suffix = "_tier" if args.tier else ""
    result_dir = BASE / f"micro_benchmark/{args.ngpu}gpu/results"
    log_dir = BASE / f"micro_benchmark/{args.ngpu}gpu/logs"
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for deploy_name in deploys:
        if deploy_name not in configs:
            log.error("Unknown deploy: %s", deploy_name)
            continue

        cfg = configs[deploy_name]
        full_name = f"{deploy_name}{tier_suffix}"
        log.info("=" * 70)
        log.info("DEPLOY: %s (%d GPU, tier=%s)", full_name, args.ngpu, args.tier)
        log.info("=" * 70)

        kill_all()
        time.sleep(5)
        mgr = DeployManager(log_dir / full_name, tier=args.tier)
        port = cfg["start"](mgr)

        if port is None:
            log.error("Deploy %s FAILED", full_name)
            mgr.cleanup()
            continue

        # Lock freq if no tier
        if not args.tier:
            lock_gpu_freq(cfg["gpus"], MAX_GPU_FREQ)

        # Warmup
        log.info("Warmup...")
        if not test_generate(port):
            log.error("Warmup failed")
            unlock_gpu_freq(cfg["gpus"])
            mgr.cleanup()
            kill_all()
            continue
        time.sleep(3)

        url = f"http://127.0.0.1:{port}/generate"
        deploy_results = {}

        for scenario in scenarios:
            for qps in qps_list:
                wl_file = WORKLOAD_DIR / f"micro_{scenario}_qps{qps}.jsonl"
                if not wl_file.exists():
                    log.warning("  Workload not found: %s", wl_file)
                    continue

                with open(wl_file) as f:
                    reqs = [json.loads(l) for l in f]

                wl_key = f"{scenario}_qps{qps}"
                log.info("-" * 50)
                log.info("  %s (%d reqs)", wl_key, len(reqs))
                log.info("-" * 50)

                summary = asyncio.run(
                    run_workload(reqs, url, cfg["gpus"], args.max_run_s))

                if summary.get("status") == "PASS":
                    log.info("  Thpt=%.1f tok/s | TTFT=%.1fms | TPOT=%.1fms | "
                             "E=%.0fJ (%.1f mJ/tok) | SLO=%.1f%%",
                             summary["throughput_tok_s"],
                             summary["ttft_proc_avg_ms"],
                             summary["tpot_avg_ms"],
                             summary["total_energy_j"],
                             summary["energy_per_token_mj"],
                             summary["slo_violation_rate"])
                else:
                    log.error("  FAIL: %s", summary)

                deploy_results[wl_key] = summary
                time.sleep(5)

        all_results[full_name] = deploy_results

        if not args.tier:
            unlock_gpu_freq(cfg["gpus"])
        mgr.cleanup()
        kill_all()

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = result_dir / f"micro_{args.ngpu}gpu{tier_suffix}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved: %s", out_file)

    # Print summary table
    print("\n" + "=" * 100)
    print(f"  MICRO BENCHMARK ({args.ngpu} GPU, Qwen3-32B, tier={args.tier})")
    print("=" * 100)
    hdr = f"{'Deploy':<16} {'Workload':<18} {'Thpt':>7} {'TTFT':>7} {'TPOT':>7} {'Energy':>8} {'mJ/tok':>7} {'SLO%':>6}"
    print(hdr)
    print("-" * 100)
    for dep, wl_results in all_results.items():
        for wl, m in wl_results.items():
            if m.get("status") != "PASS":
                print(f"  {dep:<16} {wl:<18} FAIL")
                continue
            print(f"  {dep:<16} {wl:<18} "
                  f"{m['throughput_tok_s']:>7.1f} "
                  f"{m['ttft_proc_avg_ms']:>7.1f} "
                  f"{m['tpot_avg_ms']:>7.1f} "
                  f"{m['total_energy_j']:>8.0f} "
                  f"{m['energy_per_token_mj']:>7.1f} "
                  f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
