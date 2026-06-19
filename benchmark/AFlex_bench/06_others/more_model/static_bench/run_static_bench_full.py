#!/usr/bin/env python3
"""Full Cartesian-product static benchmark.

Sweeps: QPS={1,2,3,4,5,6} × IL={128,256,512,1024} × OL={128,256,512,1024}
Total: 96 workloads per deploy config, 6 deploys = 576 runs.

Results saved incrementally so we don't lose progress on failure.

Usage:
    python run_static_bench_full.py --deploy all
    python run_static_bench_full.py --deploy native_dp8,pdaf_tp2
    python run_static_bench_full.py --resume  # skip already-done workloads
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import argparse
import asyncio
import json
import logging
import random
import time

import numpy as np
import requests as requests_lib

from run_moe_bench import (
    GPUS,
    MAX_GPU_FREQ,
    DeploymentManager,
    get_gpu_energy_mj,
    get_gpu_freq_mhz,
    kill_all,
    lock_gpu_freq,
    test_generate,
    unlock_gpu_freq,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("static_full")

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs_full"
RESULTS_DIR = HERE / "results_full"
WORKLOAD_DIR = HERE / "workloads_full"

QPS_LIST = [1, 2, 3, 4, 5, 6]
IL_LIST = [128, 256, 512, 1024]
OL_LIST = [128, 256, 512, 1024]
N_REQUESTS = 150  # requests per workload

DEPLOY_CONFIGS = {
    "native_dp8": {"method": "native_dp", "n_instances": 8, "tier": False},
    "native_dp8_tier": {"method": "native_dp", "n_instances": 8, "tier": True},
    "pd_dp4": {"method": "pd_dp", "n_pairs": 4, "tier": False},
    "pd_dp4_tier": {"method": "pd_dp", "n_pairs": 4, "tier": True},
    "pdaf_tp2": {"method": "pdaf", "micro_batch": 2, "tier": False},
    "pdaf_tp2_tier": {"method": "pdaf", "micro_batch": 2, "tier": True},
}


def gen_static_workload(il, ol, qps, n, seed=42):
    random.seed(seed)
    t = 0.0
    rows = []
    for _ in range(n):
        t += random.expovariate(qps) if qps > 0 else 0.0
        rows.append({"input_len": il, "output_len": ol,
                     "arrival_time_s": round(t, 4)})
    return rows


def workload_name(il, ol, qps):
    return f"il{il}_ol{ol}_qps{qps}"


import aiohttp


async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {
        "text": "x" * req["input_len"],
        "sampling_params": {"max_new_tokens": req["output_len"],
                            "temperature": 0.0},
        "stream": True,
    }
    # Per-request timeout: generous but bounded
    per_req_timeout = max(120, req["output_len"] * 0.5)  # at least 120s
    t0 = time.monotonic()
    first_token_time = None
    token_count = 0
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"success": False})
                return
            async for line in resp.content.iter_any():
                if time.monotonic() - t0 > per_req_timeout:
                    raise asyncio.TimeoutError("per-request timeout")
                text = line.decode().strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    json.loads(text)
                    if first_token_time is None:
                        first_token_time = time.monotonic()
                    token_count += 1
                except json.JSONDecodeError:
                    pass
    except (asyncio.TimeoutError, Exception):
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


async def run_workload(reqs, url, max_run_s=600):
    energy_start = get_gpu_energy_mj(GPUS)
    results = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60, sock_read=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [asyncio.create_task(send_one(session, url, r, base_time, results))
                 for r in reqs]
        try:
            await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True),
                                   timeout=max_run_s)
        except asyncio.TimeoutError:
            log.warning("Timed out after %ds, cancelling remaining tasks", max_run_s)
            for t in tasks:
                if not t.done():
                    t.cancel()
            await asyncio.sleep(2)
    duration_s = time.monotonic() - base_time
    energy_end = get_gpu_energy_mj(GPUS)
    total_energy_j = sum(
        (energy_end.get(i, 0) - energy_start.get(i, 0)) / 1000.0 for i in GPUS)

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


def load_existing_results(results_file):
    if results_file.exists():
        with open(results_file) as f:
            return json.load(f)
    return {}


def save_results(results_file, data):
    results_file.parent.mkdir(parents=True, exist_ok=True)
    with open(results_file, "w") as f:
        json.dump(data, f, indent=2)


def check_server_health(port, timeout=10):
    """Quick health check: returns True if server responds."""
    try:
        r = requests_lib.get(f"http://127.0.0.1:{port}/health", timeout=timeout)
        return r.status_code == 200
    except Exception:
        return False


def deploy_server(deploy_name, cfg):
    """Start deployment, return (mgr, port) or (None, None) on failure."""
    kill_all()
    time.sleep(5)
    mgr = DeploymentManager(LOG_DIR / deploy_name, tier=cfg["tier"])

    port = None
    if cfg["method"] == "native_dp":
        port = mgr.start_native_dp(n_instances=cfg["n_instances"])
    elif cfg["method"] == "pd_dp":
        port = mgr.start_pd_dp(n_pairs=cfg["n_pairs"])
    elif cfg["method"] == "pdaf":
        port = mgr.start_pdaf(micro_batch=cfg["micro_batch"])

    if port is None:
        log.error("Deploy %s FAILED to start", deploy_name)
        mgr.cleanup()
        return None, None

    if not cfg["tier"]:
        lock_gpu_freq(GPUS, MAX_GPU_FREQ)

    log.info("Warmup...")
    if not test_generate(port):
        log.error("Warmup generation test failed")
        unlock_gpu_freq(GPUS)
        mgr.cleanup()
        kill_all()
        return None, None

    time.sleep(3)
    return mgr, port


MAX_RESTART_ATTEMPTS = 3


def run_deploy_full(deploy_name, results_file, max_run_s, resume=False):
    """Deploy and run all 96 workloads with crash detection + auto-recovery."""
    cfg = DEPLOY_CONFIGS[deploy_name]
    all_results = load_existing_results(results_file)
    if deploy_name not in all_results:
        all_results[deploy_name] = {}

    all_wl_names = []
    for qps in QPS_LIST:
        for il in IL_LIST:
            for ol in OL_LIST:
                all_wl_names.append((workload_name(il, ol, qps), il, ol, qps))

    if resume:
        done = set(all_results[deploy_name].keys())
        todo = [(n, il, ol, qps) for n, il, ol, qps in all_wl_names if n not in done]
        log.info("Resume: %d/%d done, %d remaining for %s",
                 len(done), len(all_wl_names), len(todo), deploy_name)
    else:
        todo = all_wl_names

    if not todo:
        log.info("All workloads already done for %s", deploy_name)
        return all_results

    log.info("=" * 70)
    log.info("DEPLOY: %s (%d workloads to run)", deploy_name, len(todo))
    log.info("=" * 70)

    mgr, port = deploy_server(deploy_name, cfg)
    if port is None:
        return all_results

    url = f"http://127.0.0.1:{port}/generate"
    consecutive_fails = 0

    for idx, (wl_name, il, ol, qps) in enumerate(todo):
        log.info("[%d/%d] Workload: %s (IL=%d OL=%d QPS=%d)",
                 idx + 1, len(todo), wl_name, il, ol, qps)

        # Pre-check: is server still alive?
        if not check_server_health(port):
            log.warning("  Server NOT healthy before workload, restarting...")
            unlock_gpu_freq(GPUS)
            mgr.cleanup()
            restarted = False
            for attempt in range(1, MAX_RESTART_ATTEMPTS + 1):
                log.info("  Restart attempt %d/%d...", attempt, MAX_RESTART_ATTEMPTS)
                mgr, port = deploy_server(deploy_name, cfg)
                if port is not None:
                    url = f"http://127.0.0.1:{port}/generate"
                    restarted = True
                    log.info("  Server restarted successfully")
                    break
                time.sleep(10)
            if not restarted:
                log.error("  Failed to restart after %d attempts, skipping remaining workloads",
                          MAX_RESTART_ATTEMPTS)
                all_results[deploy_name][wl_name] = {"status": "CRASH", "note": "server_unrecoverable"}
                save_results(results_file, all_results)
                break

        reqs = gen_static_workload(il, ol, qps, N_REQUESTS, seed=42)
        summary = asyncio.run(run_workload(reqs, url, max_run_s))

        if summary["status"] == "PASS":
            log.info("  OK: thpt=%.1f tok/s, TPOT=%.1fms, Energy=%.0fJ (%.2f mJ/tok)",
                     summary["throughput_tok_s"], summary["tpot_avg_ms"],
                     summary["total_energy_j"], summary["energy_per_token_mj"])
            consecutive_fails = 0
        else:
            log.warning("  FAIL: %s", summary)
            consecutive_fails += 1

            # Post-check: did this workload crash the server?
            if not check_server_health(port):
                log.warning("  Server CRASHED during workload %s!", wl_name)
                summary["note"] = "server_crashed_during_run"
                # Try to restart for subsequent workloads
                unlock_gpu_freq(GPUS)
                mgr.cleanup()
                restarted = False
                for attempt in range(1, MAX_RESTART_ATTEMPTS + 1):
                    log.info("  Restart attempt %d/%d...", attempt, MAX_RESTART_ATTEMPTS)
                    mgr, port = deploy_server(deploy_name, cfg)
                    if port is not None:
                        url = f"http://127.0.0.1:{port}/generate"
                        restarted = True
                        log.info("  Server restarted after crash")
                        break
                    time.sleep(10)
                if not restarted:
                    log.error("  Cannot recover server, skipping rest for %s", deploy_name)
                    all_results[deploy_name][wl_name] = summary
                    save_results(results_file, all_results)
                    break
            elif consecutive_fails >= 5:
                log.warning("  5 consecutive FAIL (server alive), might be overloaded. Waiting 30s...")
                time.sleep(30)
                consecutive_fails = 0

        all_results[deploy_name][wl_name] = summary
        save_results(results_file, all_results)
        time.sleep(3)

    unlock_gpu_freq(GPUS)
    mgr.cleanup()
    kill_all()
    log.info("Completed all workloads for %s", deploy_name)
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Full Cartesian-product static benchmark (QPS×IL×OL)")
    parser.add_argument("--deploy", default="all",
                        help="Comma-sep deploy names or 'all'")
    parser.add_argument("--max-run-s", type=int, default=600,
                        help="Max seconds per workload")
    parser.add_argument("--resume", action="store_true",
                        help="Skip already-done workloads")
    args = parser.parse_args()

    deploys = (list(DEPLOY_CONFIGS.keys()) if args.deploy == "all"
               else args.deploy.split(","))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_file = RESULTS_DIR / "static_full_results.json"

    total_wl = len(QPS_LIST) * len(IL_LIST) * len(OL_LIST)
    log.info("Full sweep: QPS=%s × IL=%s × OL=%s = %d workloads per deploy",
             QPS_LIST, IL_LIST, OL_LIST, total_wl)
    log.info("Deploys: %s", deploys)
    log.info("Results file: %s", results_file)

    for deploy_name in deploys:
        if deploy_name not in DEPLOY_CONFIGS:
            log.error("Unknown deploy: %s", deploy_name)
            continue
        run_deploy_full(deploy_name, results_file, args.max_run_s,
                        resume=args.resume)

    log.info("ALL DONE. Results: %s", results_file)


if __name__ == "__main__":
    main()
