#!/usr/bin/env python3
"""4-GPU static benchmark for MoE (Qwen3-30B-A3B) on GPU 4,5,6,7.

Compares: Native DP4 / PD DP2 / PDAF TP1, with and without Tier.
Uses same workloads as the 8GPU static bench.

Usage:
    python run_static_bench_4gpu.py --deploy all --sweep
    python run_static_bench_4gpu.py --deploy native_dp4,pd_dp2 --sweep
"""
import argparse
import asyncio
import json
import logging
import os
import random
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
log = logging.getLogger("moe_4gpu_static")

PYTHON = "/workspace/env/sglang-tier/bin/python"
MODEL = "/models/Qwen3-30B-A3B/"
HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs_4gpu"
RESULTS_DIR = HERE / "results_4gpu"
WORKLOAD_DIR = HERE / "workloads"

GPUS = [4, 5, 6, 7]
MAX_GPU_FREQ = 1410
ROUTER_PORT = 43000

PA_PORT = 43010
PF_PORT = 43011
DA_PORT = 43020
DF_PORT = 43021

ENERGY_MODEL_DIR = "/workspace/sglang-tier/benchmark/AFlex_bench/06_others/more_model/energy_model/models"
TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0


def kill_all():
    os.system("pkill -9 -f 'sglang.*4320' 2>/dev/null")
    os.system("pkill -9 -f 'sglang.*4321' 2>/dev/null")
    os.system("pkill -9 -f 'sglang.*4322' 2>/dev/null")
    os.system("pkill -9 -f 'sglang.*4323' 2>/dev/null")
    os.system("pkill -9 -f 'sglang.*4301' 2>/dev/null")
    os.system("pkill -9 -f 'sglang.*4302' 2>/dev/null")
    os.system("pkill -9 -f 'sglang_router.*43000' 2>/dev/null")
    time.sleep(3)


def wait_health(port, timeout=180, check_model_info=False):
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


def get_gpu_energy_mj(gpus):
    import pynvml
    pynvml.nvmlInit()
    result = {}
    for g in gpus:
        h = pynvml.nvmlDeviceGetHandleByIndex(g)
        result[g] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
    return result


def lock_gpu_freq(gpus, freq):
    for g in gpus:
        os.system(f"nvidia-smi -i {g} -lgc {freq},{freq} >/dev/null 2>&1")
    log.info("  Locked GPU %s to %d MHz", gpus, freq)


def unlock_gpu_freq(gpus):
    for g in gpus:
        os.system(f"nvidia-smi -i {g} -rgc >/dev/null 2>&1")


def test_generate(port, timeout=30, retries=5):
    for i in range(retries):
        try:
            r = requests.post(
                f"http://127.0.0.1:{port}/generate",
                json={"text": "Hello", "sampling_params": {"max_new_tokens": 8, "temperature": 0}},
                timeout=timeout)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


class DeployManager:
    def __init__(self, log_dir, tier=False):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.procs = []
        self.tier = tier

    def _popen(self, cmd, name, env=None):
        log_file = self.log_dir / f"{name}.log"
        fp = open(log_file, "w")
        p = subprocess.Popen(cmd, stdout=fp, stderr=subprocess.STDOUT,
                             env=env or os.environ.copy())
        self.procs.append((p, fp))
        log.info("  Started %s (PID=%d)", name, p.pid)
        return p

    def cleanup(self):
        for p, fp in self.procs:
            try:
                os.kill(p.pid, signal.SIGKILL)
            except Exception:
                pass
            fp.close()
        self.procs = []

    def _base_env(self):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in GPUS)
        return env

    def _tier_args(self):
        if not self.tier:
            return []
        return [
            "--dvfs-enabled",
            "--dvfs-energy-model-dir", ENERGY_MODEL_DIR,
            "--dvfs-ttft-slo-ms", "2000",
            "--dvfs-tpot-slo-us", "250000",
        ]

    def start_native_dp(self):
        """4x TP1 on GPU 4,5,6,7"""
        ports = []
        for i, gpu in enumerate(GPUS):
            port = 43200 + i * 10
            nccl_port = 34300 + i * 10
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            cmd = [PYTHON, "-m", "sglang.launch_server",
                   "--model-path", MODEL, "--tp", "1",
                   "--host", "127.0.0.1", "--port", str(port),
                   "--nccl-port", str(nccl_port),
                   "--mem-fraction-static", "0.85",
                   "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                   "--skip-server-warmup"] + self._tier_args()
            self._popen(cmd, f"dp_{i}", env=env)
            ports.append(port)

        log.info("Waiting for %d Native DP instances...", len(GPUS))
        for port in ports:
            if not wait_health(port):
                log.error("  DP instance port %d failed", port)
                return None
            log.info("  DP instance ready (port %d)", port)

        # Round-robin router
        cmd = [PYTHON, "-m", "sglang_router.launch_router",
               "--host", "127.0.0.1", "--port", str(ROUTER_PORT),
               "--prometheus-port", "9100",
               "--policy", "round_robin",
               "--worker-urls"] + [f"http://127.0.0.1:{p}" for p in ports]
        self._popen(cmd, "router")
        time.sleep(5)
        if not wait_health(ROUTER_PORT, timeout=30):
            log.warning("Router health check timed out, proceeding anyway")
        log.info("Native DP4 ready (tier=%s)", self.tier)
        return ROUTER_PORT

    def start_pd_dp(self):
        """2 PD pairs: (4,5) and (6,7) with mooncake transfer"""
        pairs = [(4, 5), (6, 7)]
        instances = []

        for i, (p_gpu, d_gpu) in enumerate(pairs):
            instances.append({
                "p_cvd": str(p_gpu),
                "d_cvd": str(d_gpu),
                "p_port": 43200 + i * 20,
                "d_port": 43200 + i * 20 + 10,
                "bs_port": 44100 + i * 10,
                "p_nccl": 34400 + i * 20,
                "d_nccl": 34400 + i * 20 + 10,
            })

        tier_args = self._tier_args()
        for idx, inst in enumerate(instances):
            env_p = os.environ.copy()
            env_p["CUDA_VISIBLE_DEVICES"] = inst["p_cvd"]
            cmd_p = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "1",
                     "--host", "127.0.0.1", "--port", str(inst["p_port"]),
                     "--nccl-port", str(inst["p_nccl"]),
                     "--mem-fraction-static", "0.85",
                     "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup",
                     "--disaggregation-mode", "prefill",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(inst["bs_port"]),
                     "--disaggregation-ib-device", "mlx5_4"] + tier_args
            self._popen(cmd_p, f"pd{idx}_prefill", env=env_p)
            time.sleep(3)

            env_d = os.environ.copy()
            env_d["CUDA_VISIBLE_DEVICES"] = inst["d_cvd"]
            cmd_d = [PYTHON, "-m", "sglang.launch_server",
                     "--model-path", MODEL, "--tp", "1",
                     "--host", "127.0.0.1", "--port", str(inst["d_port"]),
                     "--nccl-port", str(inst["d_nccl"]),
                     "--mem-fraction-static", "0.85",
                     "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                     "--skip-server-warmup",
                     "--disaggregation-mode", "decode",
                     "--disaggregation-transfer-backend", "mooncake",
                     "--disaggregation-bootstrap-port", str(inst["bs_port"]),
                     "--disaggregation-ib-device", "mlx5_4"] + tier_args
            self._popen(cmd_d, f"pd{idx}_decode", env=env_d)
            time.sleep(3)

        log.info("Waiting for PD DP2 pairs...")
        for idx, inst in enumerate(instances):
            for role, port in [("P", inst["p_port"]), ("D", inst["d_port"])]:
                if not wait_health(port, timeout=240):
                    log.error("  PD%d %s (port %d) failed", idx, role, port)
                    return None
                log.info("  PD%d %s ready (port %d)", idx, role, port)

        cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
                 "--pd-disaggregation",
                 "--prometheus-port", "9100",
                 "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        for inst in instances:
            cmd_r += ["--prefill", f"http://127.0.0.1:{inst['p_port']}",
                      str(inst["bs_port"])]
        for inst in instances:
            cmd_r += ["--decode", f"http://127.0.0.1:{inst['d_port']}"]
        self._popen(cmd_r, "router")
        time.sleep(5)
        if not wait_health(ROUTER_PORT, timeout=30):
            log.warning("PD Router health check timed out")
        log.info("PD DP2 ready (tier=%s)", self.tier)
        return ROUTER_PORT

    def start_pdaf(self, micro_batch=2):
        """PDAF TP1: PF(4)+PA(5)+DF(6)+DA(7)"""
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
        env["AFD_COMM_BACKEND"] = "ipc_cpp"
        env["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
        env["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
        env["AFD_ASYNC_PIPELINE"] = "1"

        tier_args = []
        if self.tier:
            tier_args = ["--afd-dvfs-enabled",
                         "--afd-energy-model-dir", ENERGY_MODEL_DIR,
                         "--afd-ttft-slo-ms", "2000",
                         "--afd-tpot-slo-us", "250000",
                         "--afd-dvfs-idle-lock"]

        BS_PORT = 44200
        common = ["--model-path", MODEL, "--tp", "1",
                  "--host", "127.0.0.1",
                  "--afd-comm-backend", "ipc_cpp",
                  "--afd-micro-batch", str(micro_batch),
                  "--mem-fraction-static", "0.75",
                  "--max-running-requests", "512",
                  "--skip-server-warmup",
                  "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
                  "--disable-radix-cache",
                  "--disaggregation-transfer-backend", "mooncake",
                  "--disaggregation-bootstrap-port", str(BS_PORT),
                  "--disaggregation-ib-device", "mlx5_4"]

        # PF on GPU4 (base_gpu_id=0, peer=1)
        env_pf = env.copy()
        env_pf["AFD_IPC_SYNC_MODE"] = "ipc_event"
        env_pf["AFD_IPC_PEER_DEVICE"] = "1"
        env_pf["AFD_NVML_DEVICE_INDICES"] = "4"
        env_pf["AFD_NVML_DEVICE_INDEX"] = "4"
        cmd_pf = [PYTHON, "-m", "sglang.launch_server",
                  "--port", str(PF_PORT),
                  "--afd-perspective", "ffn",
                  "--disaggregation-mode", "prefill",
                  "--base-gpu-id", "0"] + common + tier_args
        self._popen(cmd_pf, "pf", env=env_pf)
        time.sleep(5)

        # PA on GPU5 (base_gpu_id=1, peer=0)
        env_pa = env.copy()
        env_pa["AFD_IPC_SYNC_MODE"] = "ipc_event"
        env_pa["AFD_IPC_PEER_DEVICE"] = "0"
        env_pa["AFD_NVML_DEVICE_INDICES"] = "5"
        env_pa["AFD_NVML_DEVICE_INDEX"] = "5"
        env_pa["AFD_UCX_FFN_HOST"] = "127.0.0.1"
        cmd_pa = [PYTHON, "-m", "sglang.launch_server",
                  "--port", str(PA_PORT),
                  "--afd-perspective", "attn",
                  "--disaggregation-mode", "prefill",
                  "--base-gpu-id", "1"] + common + tier_args
        self._popen(cmd_pa, "pa", env=env_pa)
        time.sleep(5)

        # DF on GPU6 (base_gpu_id=2, peer=3)
        env_df = env.copy()
        env_df["AFD_IPC_SYNC_MODE"] = "ipc_event"
        env_df["AFD_IPC_PEER_DEVICE"] = "3"
        env_df["AFD_NVML_DEVICE_INDICES"] = "6"
        env_df["AFD_NVML_DEVICE_INDEX"] = "6"
        cmd_df = [PYTHON, "-m", "sglang.launch_server",
                  "--port", str(DF_PORT),
                  "--afd-perspective", "ffn",
                  "--disaggregation-mode", "decode",
                  "--base-gpu-id", "2"] + common + tier_args
        self._popen(cmd_df, "df", env=env_df)
        time.sleep(8)

        # DA on GPU7 (base_gpu_id=3, peer=2)
        env_da = env.copy()
        env_da["AFD_IPC_SYNC_MODE"] = "ipc_event"
        env_da["AFD_IPC_PEER_DEVICE"] = "2"
        env_da["AFD_NVML_DEVICE_INDICES"] = "7"
        env_da["AFD_NVML_DEVICE_INDEX"] = "7"
        env_da["AFD_UCX_FFN_HOST"] = "127.0.0.1"
        cmd_da = [PYTHON, "-m", "sglang.launch_server",
                  "--port", str(DA_PORT),
                  "--afd-perspective", "attn",
                  "--disaggregation-mode", "decode",
                  "--base-gpu-id", "3"] + common + tier_args
        self._popen(cmd_da, "da", env=env_da)

        log.info("Waiting for PDAF servers...")
        checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
                  (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
        for port, name, use_model_info in checks:
            if not wait_health(port, timeout=300, check_model_info=use_model_info):
                log.error("  %s (port %d) failed", name, port)
                return None
            log.info("  %s ready (port %d)", name, port)

        prefill_url = f"http://127.0.0.1:{PA_PORT}"
        decode_url = f"http://127.0.0.1:{DA_PORT}"
        cmd_router = [PYTHON, "-m", "sglang_router.launch_router",
                      "--pd-disaggregation", "--mini-lb",
                      "--prometheus-port", "9100",
                      "--prefill", prefill_url,
                      "--decode", decode_url,
                      "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
        self._popen(cmd_router, "router")
        time.sleep(5)
        if not wait_health(ROUTER_PORT, timeout=60):
            log.error("PDAF router failed")
            return None
        log.info("PDAF TP1 ready (tier=%s)", self.tier)
        return ROUTER_PORT


# ---------- Workload + metrics ----------

def gen_static_workload(il, ol, qps, n, seed=42):
    random.seed(seed)
    t = 0.0
    rows = []
    for _ in range(n):
        t += random.expovariate(qps) if qps > 0 else 0.0
        rows.append({"input_len": il, "output_len": ol,
                     "arrival_time_s": round(t, 4)})
    return rows


async def send_one(session, url, req, base_time, results):
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
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"success": False})
                return
            async for line in resp.content:
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
                        first_token_time = time.monotonic()
                    token_count += 1
                except json.JSONDecodeError:
                    pass
    except Exception:
        results.append({"success": False})
        return

    t_end = time.monotonic()
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0.0
    tpot_ms = 0.0
    if token_count > 1 and first_token_time:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)
    results.append({
        "success": True, "completion_tokens": token_count,
        "ttft_ms": ttft_ms, "tpot_ms": tpot_ms, "e2e_s": t_end - t0,
    })


async def run_workload(reqs, url, max_run_s=300):
    energy_start = get_gpu_energy_mj(GPUS)
    results = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [asyncio.create_task(send_one(session, url, r, base_time, results))
                 for r in reqs]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                   timeout=max_run_s)
        except asyncio.TimeoutError:
            log.warning("Timed out after %ds", max_run_s)
    duration_s = time.monotonic() - base_time
    energy_end = get_gpu_energy_mj(GPUS)
    total_energy_j = sum((energy_end[i] - energy_start[i]) / 1000.0 for i in GPUS)

    ok = [r for r in results if r.get("success")]
    if not ok:
        return {"status": "FAIL", "successful": 0, "failed": len(results)}

    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0

    return {
        "status": "PASS",
        "duration_s": round(duration_s, 1),
        "successful": len(ok),
        "failed": len(results) - len(ok),
        "total_output_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 1),
        "ttft_avg_ms": round(float(np.mean(ttfts)), 1) if ttfts else 0,
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens > 0 else 0,
    }


DEPLOY_CONFIGS = {
    "native_dp4": {"method": "native_dp", "tier": False},
    "native_dp4_tier": {"method": "native_dp", "tier": True},
    "pd_dp2": {"method": "pd_dp", "tier": False},
    "pd_dp2_tier": {"method": "pd_dp", "tier": True},
    "pdaf_tp1": {"method": "pdaf", "tier": False},
    "pdaf_tp1_tier": {"method": "pdaf", "tier": True},
}


def run_one_deploy(deploy_name, workloads, max_run_s):
    cfg = DEPLOY_CONFIGS[deploy_name]
    log.info("=" * 70)
    log.info("DEPLOY: %s", deploy_name)
    log.info("=" * 70)

    kill_all()
    time.sleep(5)
    mgr = DeployManager(LOG_DIR / deploy_name, tier=cfg["tier"])

    if cfg["method"] == "native_dp":
        port = mgr.start_native_dp()
    elif cfg["method"] == "pd_dp":
        port = mgr.start_pd_dp()
    elif cfg["method"] == "pdaf":
        port = mgr.start_pdaf()
    else:
        port = None

    if port is None:
        log.error("Deploy %s FAILED", deploy_name)
        mgr.cleanup()
        return None

    if not cfg["tier"]:
        lock_gpu_freq(GPUS, MAX_GPU_FREQ)

    log.info("Warmup...")
    if not test_generate(port):
        log.error("Warmup failed")
        unlock_gpu_freq(GPUS)
        mgr.cleanup()
        kill_all()
        return None

    time.sleep(3)
    url = f"http://127.0.0.1:{port}/generate"
    deploy_results = {}

    for wl_name, reqs in workloads.items():
        log.info("-" * 50)
        log.info("Workload: %s (%d reqs)", wl_name, len(reqs))
        log.info("-" * 50)
        summary = asyncio.run(run_workload(reqs, url, max_run_s))
        if summary["status"] == "PASS":
            log.info("  Thpt=%.1f | TTFT=%.1fms | TPOT=%.1fms | E=%.0fJ (%.2f mJ/tok)",
                     summary["throughput_tok_s"], summary["ttft_avg_ms"],
                     summary["tpot_avg_ms"], summary["total_energy_j"],
                     summary["energy_per_token_mj"])
        else:
            log.error("  FAIL: %s", summary)
        deploy_results[wl_name] = summary
        time.sleep(5)

    unlock_gpu_freq(GPUS)
    mgr.cleanup()
    kill_all()
    return deploy_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", default="all")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--il", type=int, default=512)
    parser.add_argument("--ol", type=int, default=256)
    parser.add_argument("--qps", type=float, default=4.0)
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--max-run-s", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    deploys = (list(DEPLOY_CONFIGS.keys()) if args.deploy == "all"
               else args.deploy.split(","))

    if args.sweep:
        sweep_configs = [
            ("il512_ol128_qps2", 512, 128, 2.0, 200),
            ("il512_ol256_qps4", 512, 256, 4.0, 200),
            ("il1024_ol256_qps2", 1024, 256, 2.0, 200),
            ("il1024_ol512_qps4", 1024, 512, 4.0, 200),
            ("il2048_ol256_qps2", 2048, 256, 2.0, 150),
            ("il256_ol512_qps6", 256, 512, 6.0, 300),
        ]
    else:
        sweep_configs = [
            (f"il{args.il}_ol{args.ol}_qps{args.qps:.0f}",
             args.il, args.ol, args.qps, args.n),
        ]

    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    workloads = {}
    for name, il, ol, qps, n in sweep_configs:
        rows = gen_static_workload(il, ol, qps, n, seed=args.seed)
        fp = WORKLOAD_DIR / f"{name}.jsonl"
        with open(fp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        workloads[name] = rows
        log.info("Workload '%s': il=%d ol=%d qps=%.1f n=%d span=%.0fs",
                 name, il, ol, qps, n, rows[-1]["arrival_time_s"])

    all_results = {}
    for deploy_name in deploys:
        if deploy_name not in DEPLOY_CONFIGS:
            log.error("Unknown: %s", deploy_name)
            continue
        result = run_one_deploy(deploy_name, workloads, args.max_run_s)
        if result:
            all_results[deploy_name] = result

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"static_4gpu_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved: %s", out_file)

    # Summary table
    print("\n" + "=" * 100)
    print("  MoE 4-GPU STATIC BENCH (Qwen3-30B-A3B, GPU 4,5,6,7)")
    print("=" * 100)
    hdr = f"{'Deploy':<18} {'Workload':<22} {'Thpt':>7} {'TTFT':>7} {'TPOT':>7} {'Energy':>8} {'mJ/tok':>7}"
    print(hdr)
    print("-" * 100)
    for dep, wl_results in all_results.items():
        for wl, m in wl_results.items():
            if m.get("status") != "PASS":
                print(f"  {dep:<18} {wl:<22} FAIL")
                continue
            print(f"  {dep:<18} {wl:<22} "
                  f"{m['throughput_tok_s']:>7.1f} "
                  f"{m['ttft_avg_ms']:>7.1f} "
                  f"{m['tpot_avg_ms']:>7.1f} "
                  f"{m['total_energy_j']:>8.0f} "
                  f"{m['energy_per_token_mj']:>7.2f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
