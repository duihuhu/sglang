"""Run heterogeneous PDAF benchmark: 1PA(TP1)+4PF(TP4)+1DA(TP1)+2DF(TP2) on 8 GPUs.

Tests baseline and Tier on rag/summary datasets at QPS 1-9.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("hetero_bench")

PYTHON = "/workspace/env/sglang-test/bin/python"
MODEL = "/models/Mixtral/Mixtral-8x7B/"
WORKLOAD_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/retesting/workloads")
RESULT_DIR = Path("/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/micro_benchmark/8gpu_hetero")
ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/AFlex_bench/06_others/Mixtral_test/energy_model/models"

# Heterogeneous layout: 1PA(TP1)+4PF(TP4)+1DA(TP1)+2DF(TP2)
# GPU allocation:
#   P side: PF on GPU 0,1,2,3 (TP4), PA on GPU 4 (TP1) -> P_CVD = "0,1,2,3,4"
#   D side: DF on GPU 5,6 (TP2), DA on GPU 7 (TP1)     -> D_CVD = "5,6,7"
# Wait... PA TP1 needs only Attention weights (2.7GB), should be fine on 1 GPU
# PF TP4 needs FFN weights / 4 = ~22.5GB per GPU, fine
# DA TP1 needs Attention weights (2.7GB), fine
# DF TP2 needs FFN weights / 2 = ~45GB per GPU, fine

P_CVD = "0,1,2,3,4"
D_CVD = "5,6,7"
TP_FFN_P = 4  # PF uses TP4
TP_ATTN_P = 1  # PA uses TP1
TP_FFN_D = 2  # DF uses TP2
TP_ATTN_D = 1  # DA uses TP1

# Physical GPU mapping
PF_PHYS = [0, 1, 2, 3]
PA_PHYS = [4]
DF_PHYS = [5, 6]
DA_PHYS = [7]
ALL_PHYS = list(range(8))

PA_PORT = 42010
PF_PORT = 42011
DA_PORT = 42020
DF_PORT = 42021
ROUTER_PORT = 42000

TTFT_SLO_MS = 2000
TPOT_SLO_MS = 100

DATASETS = ["rag", "summary", "chatbot", "qa"]
QPS_LIST = list(range(1, 10))

procs = []


def kill_port(port):
    r = subprocess.run(["ss", "-tlnp", f"sport = :{port}"], capture_output=True, text=True)
    for m in re.finditer(r"pid=(\d+)", r.stdout):
        try:
            os.kill(int(m.group(1)), 9)
        except OSError:
            pass

def kill_all():
    for port in [PA_PORT, PF_PORT, DA_PORT, DF_PORT, ROUTER_PORT]:
        kill_port(port)
    for p in procs:
        try:
            p.kill()
        except:
            pass
    procs.clear()
    time.sleep(5)

def wait_health(port, timeout=600, check_model_info=False):
    for _ in range(timeout // 2):
        try:
            if check_model_info:
                r = requests.get(f"http://127.0.0.1:{port}/get_model_info", timeout=3)
            else:
                r = requests.get(f"http://127.0.0.1:{port}/health", timeout=3)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False

def unlock_freq():
    for g in ALL_PHYS:
        subprocess.run(["nvidia-smi", "-i", str(g), "-rgc"], capture_output=True)

def lock_freq_max():
    for g in ALL_PHYS:
        subprocess.run(["nvidia-smi", "-i", str(g), "--lock-gpu-clocks=1410,1410"], capture_output=True)

def start_pdaf(tier=False):
    """Deploy heterogeneous PDAF: 1PA(TP1)+4PF(TP4)+1DA(TP1)+2DF(TP2)."""
    kill_all()
    unlock_freq()
    if not tier:
        lock_freq_max()

    env_base = os.environ.copy()
    env_base["AFD_UCX_TLS"] = "rc,tcp,cuda_copy,cuda_ipc"
    env_base["SGLANG_DISAGGREGATION_THREAD_POOL_SIZE"] = "128"
    env_base["AFD_ASYNC_PIPELINE"] = "1"

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

    dvfs_args = []
    if tier:
        dvfs_args = ["--afd-dvfs-enabled",
                     "--afd-energy-model-dir", ENERGY_MODEL_DIR,
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

    def _cmd(port, tp, perspective, disagg, base_gpu_id):
        return [PYTHON, "-m", "sglang.launch_server",
                "--port", str(port), "--tp", str(tp),
                "--afd-perspective", perspective,
                "--disaggregation-mode", disagg,
                "--base-gpu-id", str(base_gpu_id)] + common + dvfs_args

    # P side: CVD = "0,1,2,3,4"
    # Layout: PF(TP4) uses logical GPU 0-3, PA(TP1) uses logical GPU 4
    p_ffn_nvml = ",".join(str(g) for g in PF_PHYS)
    p_attn_nvml = ",".join(str(g) for g in PA_PHYS)

    # D side: CVD = "5,6,7"
    # Layout: DF(TP2) uses logical GPU 0-1, DA(TP1) uses logical GPU 2
    d_ffn_nvml = ",".join(str(g) for g in DF_PHYS)
    d_attn_nvml = ",".join(str(g) for g in DA_PHYS)

    # PF (FFN TP4, base=0, logical GPUs 0-3 in P_CVD)
    log.info("Starting PF (TP4, phys GPU %s)...", PF_PHYS)
    p = subprocess.Popen(_cmd(PF_PORT, TP_FFN_P, "ffn", "prefill", 0),
                         env=_env(P_CVD, ucx_p, sched_p,
                                  peer_device=TP_FFN_P,
                                  nvml_idx=p_ffn_nvml),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    time.sleep(10)

    # PA (Attn TP1, base=4, logical GPU 4 in P_CVD)
    log.info("Starting PA (TP1, phys GPU %s)...", PA_PHYS)
    p = subprocess.Popen(_cmd(PA_PORT, TP_ATTN_P, "attn", "prefill", TP_FFN_P),
                         env=_env(P_CVD, ucx_p, sched_p,
                                  peer_device=0,
                                  nvml_idx=p_attn_nvml, ffn_host="127.0.0.1"),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    time.sleep(8)

    # DF (FFN TP2, base=0, logical GPUs 0-1 in D_CVD)
    log.info("Starting DF (TP2, phys GPU %s)...", DF_PHYS)
    p = subprocess.Popen(_cmd(DF_PORT, TP_FFN_D, "ffn", "decode", 0),
                         env=_env(D_CVD, ucx_d, sched_d,
                                  peer_device=TP_FFN_D,
                                  nvml_idx=d_ffn_nvml),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    time.sleep(10)

    # DA (Attn TP1, base=2, logical GPU 2 in D_CVD)
    log.info("Starting DA (TP1, phys GPU %s)...", DA_PHYS)
    p = subprocess.Popen(_cmd(DA_PORT, TP_ATTN_D, "attn", "decode", TP_FFN_D),
                         env=_env(D_CVD, ucx_d, sched_d,
                                  peer_device=0,
                                  nvml_idx=d_attn_nvml, ffn_host="127.0.0.1"),
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)

    # Wait health
    checks = [(PA_PORT, "PA", False), (PF_PORT, "PF", True),
              (DA_PORT, "DA", False), (DF_PORT, "DF", True)]
    for port, name, use_model in checks:
        if not wait_health(port, 600, check_model_info=use_model):
            log.error("%s failed (port %d)!", name, port)
            return False
        log.info("  %s ready", name)

    # Router
    cmd_r = [PYTHON, "-m", "sglang_router.launch_router",
             "--pd-disaggregation", "--mini-lb",
             "--prefill", f"http://127.0.0.1:{PA_PORT}",
             "--decode", f"http://127.0.0.1:{DA_PORT}",
             "--host", "127.0.0.1", "--port", str(ROUTER_PORT)]
    p = subprocess.Popen(cmd_r, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    procs.append(p)
    if not wait_health(ROUTER_PORT, 60):
        log.error("Router failed!")
        return False

    mode = "Tier" if tier else "Baseline"
    log.info("Hetero PDAF [%s] ready: 1PA(TP1)+4PF(TP4)+1DA(TP1)+2DF(TP2)", mode)
    return True


async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (asyncio.get_event_loop().time() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"text": "x" * req["input_len"],
               "sampling_params": {"max_new_tokens": req["output_len"], "temperature": 0.0},
               "stream": True}
    t0 = asyncio.get_event_loop().time()
    first_token_time = None
    token_count = 0
    last_meta = {}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"success": False})
                return
            async for line in resp.content:
                now = asyncio.get_event_loop().time()
                text = line.decode().strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    d = json.loads(text)
                    last_meta = d.get("meta_info", last_meta)
                    if first_token_time is None:
                        first_token_time = now
                    token_count += 1
                except:
                    pass
        t_end = asyncio.get_event_loop().time()
        ttft = (first_token_time - t0) if first_token_time else None
        tpot = (t_end - first_token_time) / max(token_count - 1, 1) if first_token_time and token_count > 1 else None

        e2e_lat = t_end - t0
        prompt_tokens = last_meta.get("prompt_tokens", req["input_len"])
        completion_tokens = last_meta.get("completion_tokens", token_count)

        results.append({
            "success": True,
            "ttft_s": ttft,
            "tpot_s": tpot,
            "e2e_lat_s": e2e_lat,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "ttft_proc_s": ttft,
        })
    except Exception as e:
        results.append({"success": False, "error": str(e)})


async def run_workload_reqs(reqs):
    """Run workload from list of request dicts with arrival_time_s."""
    url = f"http://127.0.0.1:{ROUTER_PORT}/generate"
    results = []
    timeout = aiohttp.ClientTimeout(total=600)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = asyncio.get_event_loop().time()
        tasks = [send_one(session, url, req, base_time, results) for req in reqs]
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


def compute_metrics(results):
    ok = [r for r in results if r.get("success")]
    if not ok:
        return None
    ttfts = [r["ttft_s"] * 1000 for r in ok if r.get("ttft_s")]
    tpots = [r["tpot_s"] * 1000 for r in ok if r.get("tpot_s")]
    e2es = [r["e2e_lat_s"] * 1000 for r in ok if r.get("e2e_lat_s")]

    import numpy as np
    return {
        "num_requests": len(results),
        "success_rate": len(ok) / len(results),
        "ttft_avg_ms": float(np.mean(ttfts)) if ttfts else 0,
        "ttft_p50_ms": float(np.median(ttfts)) if ttfts else 0,
        "ttft_p99_ms": float(np.percentile(ttfts, 99)) if ttfts else 0,
        "tpot_avg_ms": float(np.mean(tpots)) if tpots else 0,
        "tpot_p50_ms": float(np.median(tpots)) if tpots else 0,
        "tpot_p99_ms": float(np.percentile(tpots, 99)) if tpots else 0,
        "e2e_avg_ms": float(np.mean(e2es)) if e2es else 0,
    }


def get_total_power():
    total = 0
    for g in ALL_PHYS:
        r = subprocess.run(["nvidia-smi", "-i", str(g), "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        total += float(r.stdout.strip())
    return total


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for tier in [False, True]:
        mode = "tier" if tier else "baseline"
        log.info("=== Starting %s tests ===", mode.upper())

        if not start_pdaf(tier=tier):
            log.error("Failed to deploy %s, skipping", mode)
            continue

        # Warmup
        requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate",
                      json={"text": "Hello", "sampling_params": {"max_new_tokens": 10, "temperature": 0}},
                      timeout=60)
        log.info("Warmup done")

        for ds in DATASETS:
            for qps in QPS_LIST:
                wl_file = WORKLOAD_DIR / f"micro_{ds}_qps{qps}.jsonl"
                if not wl_file.exists():
                    log.warning("Workload %s not found, skipping", wl_file)
                    continue

                key = f"{mode}_{ds}_qps{qps}"
                log.info("  Running %s ...", key)

                try:
                    with open(wl_file) as wf:
                        reqs = [json.loads(line) for line in wf]

                    power_before = get_total_power()
                    t0 = time.time()

                    results = asyncio.run(run_workload_reqs(reqs))

                    t1 = time.time()
                    power_after = get_total_power()
                    avg_power = (power_before + power_after) / 2
                    energy_j = avg_power * (t1 - t0)

                    metrics = compute_metrics(results)
                    if metrics:
                        metrics["qps"] = qps
                        metrics["dataset"] = ds
                        metrics["mode"] = mode
                        metrics["energy_j"] = energy_j
                        metrics["avg_power_w"] = avg_power
                        metrics["duration_s"] = t1 - t0
                        all_results[key] = metrics

                        log.info("    TTFT=%.1fms TPOT=%.2fms E=%.0fJ (%.0f%% ok)",
                                 metrics["ttft_avg_ms"], metrics["tpot_avg_ms"],
                                 energy_j, metrics["success_rate"] * 100)

                        # Check if service crashed
                        if metrics["success_rate"] < 0.5:
                            log.warning("    Low success rate, checking health...")
                            if not wait_health(ROUTER_PORT, 10):
                                log.error("    Service crashed at QPS=%d, restarting...", qps)
                                if not start_pdaf(tier=tier):
                                    break
                                requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate",
                                              json={"text": "Hello", "sampling_params": {"max_new_tokens": 5, "temperature": 0}},
                                              timeout=60)
                    else:
                        log.warning("    No valid results for %s", key)

                except Exception as e:
                    log.error("    %s FAILED: %s", key, e)
                    if not wait_health(ROUTER_PORT, 10):
                        log.error("    Service dead, restarting...")
                        if not start_pdaf(tier=tier):
                            break
                        requests.post(f"http://127.0.0.1:{ROUTER_PORT}/generate",
                                      json={"text": "Hello", "sampling_params": {"max_new_tokens": 5, "temperature": 0}},
                                      timeout=60)

        kill_all()
        time.sleep(5)

    # Save all results
    out_file = RESULT_DIR / "hetero_pdaf_results.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("All results saved to %s", out_file)

    # Generate summary
    log.info("\n=== SUMMARY ===")
    for ds in DATASETS:
        log.info("  Dataset: %s", ds)
        for qps in QPS_LIST:
            bl_key = f"baseline_{ds}_qps{qps}"
            ti_key = f"tier_{ds}_qps{qps}"
            bl = all_results.get(bl_key, {})
            ti = all_results.get(ti_key, {})
            if bl and ti:
                e_save = (1 - ti.get("energy_j", 0) / bl.get("energy_j", 1)) * 100
                log.info("    QPS=%d: BL TTFT=%.0f TPOT=%.1f | Tier TTFT=%.0f TPOT=%.1f | E-save=%.1f%%",
                         qps,
                         bl.get("ttft_avg_ms", 0), bl.get("tpot_avg_ms", 0),
                         ti.get("ttft_avg_ms", 0), ti.get("tpot_avg_ms", 0),
                         e_save)


if __name__ == "__main__":
    main()
