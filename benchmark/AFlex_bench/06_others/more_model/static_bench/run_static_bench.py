#!/usr/bin/env python3
"""Static (fixed-length) benchmark: Native DP8 vs PD DP4 vs PDAF TP2.

Uses fixed input_len / output_len / QPS workloads for clean comparison.
Measures: throughput, TTFT, TPOT, total energy, energy efficiency.

Usage:
    python run_static_bench.py --deploy native_dp8,pd_dp4,pdaf_tp2
    python run_static_bench.py --deploy all --il 512 --ol 256 --qps 4 --n 200
    python run_static_bench.py --deploy all --sweep  # run multiple configs
"""
import sys
from pathlib import Path

# Add parent scripts dir to path for reusing DeploymentManager
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import argparse
import asyncio
import json
import logging
import random
import time

import numpy as np

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
log = logging.getLogger("static_bench")

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE / "logs"
RESULTS_DIR = HERE / "results"
WORKLOAD_DIR = HERE / "workloads"


def gen_static_workload(il, ol, qps, n, seed=42):
    """Generate fixed-length workload with Poisson arrivals."""
    random.seed(seed)
    t = 0.0
    rows = []
    for _ in range(n):
        t += random.expovariate(qps) if qps > 0 else 0.0
        rows.append({"input_len": il, "output_len": ol,
                     "arrival_time_s": round(t, 4)})
    return rows


def save_workload(rows, name):
    """Save workload to JSONL file."""
    WORKLOAD_DIR.mkdir(parents=True, exist_ok=True)
    fp = WORKLOAD_DIR / f"{name}.jsonl"
    with open(fp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return fp


# Deployment configs
DEPLOY_CONFIGS = {
    "native_dp8": {"method": "native_dp", "n_instances": 8, "tier": False},
    "native_dp8_tier": {"method": "native_dp", "n_instances": 8, "tier": True},
    "pd_dp4": {"method": "pd_dp", "n_pairs": 4, "tier": False},
    "pd_dp4_tier": {"method": "pd_dp", "n_pairs": 4, "tier": True},
    "pdaf_tp2": {"method": "pdaf", "micro_batch": 2, "tier": False},
    "pdaf_tp2_tier": {"method": "pdaf", "micro_batch": 2, "tier": True},
}


import aiohttp


async def send_one(session, url, req, base_time, results):
    """Send one request with streaming, measure TTFT/TPOT."""
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)

    payload = {
        "text": "x" * req["input_len"],
        "sampling_params": {"max_new_tokens": req["output_len"],
                            "temperature": 0.0},
        "stream": True,
    }
    t0 = time.monotonic()
    first_token_time = None
    token_count = 0
    last_meta = {}

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"success": False, "input_len": req["input_len"],
                                "output_len": req["output_len"]})
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
    except Exception as e:
        results.append({"success": False, "input_len": req["input_len"],
                        "output_len": req["output_len"], "error": str(e)})
        return

    t_end = time.monotonic()
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0.0
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
        "input_len": req["input_len"],
        "output_len": req["output_len"],
        "completion_tokens": token_count,
        "ttft_ms": ttft_ms,
        "ttft_proc_ms": ttft_proc_ms,
        "tpot_ms": tpot_ms,
        "e2e_s": t_end - t0,
    })


async def run_workload(reqs, url, max_run_s=300):
    """Run workload and collect metrics + energy."""
    energy_start = get_gpu_energy_mj(GPUS)
    results = []

    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = []
        for req in reqs:
            tasks.append(asyncio.create_task(
                send_one(session, url, req, base_time, results)))

        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=max_run_s)
            except asyncio.TimeoutError:
                log.warning("Workload timed out after %ds", max_run_s)

    duration_s = time.monotonic() - base_time
    energy_end = get_gpu_energy_mj(GPUS)
    total_energy_j = sum(
        (energy_end.get(i, 0) - energy_start.get(i, 0)) / 1000.0
        for i in GPUS)

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]

    if not ok:
        return {"status": "FAIL", "successful": 0, "failed": len(fail)}

    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0

    return {
        "status": "PASS",
        "duration_s": round(duration_s, 1),
        "total_requests": len(reqs),
        "successful": len(ok),
        "failed": len(fail),
        "total_output_tokens": total_tokens,
        "throughput_tok_s": round(throughput, 1),
        "ttft_avg_ms": round(float(np.mean(ttfts)), 1) if ttfts else 0,
        "ttft_p50_ms": round(float(np.percentile(ttfts, 50)), 1) if ttfts else 0,
        "ttft_p99_ms": round(float(np.percentile(ttfts, 99)), 1) if ttfts else 0,
        "ttft_proc_avg_ms": round(float(np.mean(ttfts_proc)), 1) if ttfts_proc else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens > 0 else 0,
    }


def run_one_deploy(deploy_name, workloads, max_run_s):
    """Deploy one config, run all workloads, return results dict."""
    cfg = DEPLOY_CONFIGS[deploy_name]
    log.info("=" * 70)
    log.info("DEPLOY: %s", deploy_name)
    log.info("=" * 70)

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
        log.error("Deploy %s FAILED", deploy_name)
        mgr.cleanup()
        return None

    # Lock freq only for non-Tier configs; Tier uses DVFS
    if not cfg["tier"]:
        lock_gpu_freq(GPUS, MAX_GPU_FREQ)

    # Warmup
    log.info("Warmup generation test...")
    if not test_generate(port):
        log.error("Generation test failed")
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
            log.info("  Throughput: %.1f tok/s", summary["throughput_tok_s"])
            log.info("  TTFT: avg=%.1fms p50=%.1fms p99=%.1fms",
                     summary["ttft_avg_ms"], summary["ttft_p50_ms"],
                     summary["ttft_p99_ms"])
            log.info("  TPOT: avg=%.1fms p50=%.1fms p99=%.1fms",
                     summary["tpot_avg_ms"], summary["tpot_p50_ms"],
                     summary["tpot_p99_ms"])
            log.info("  Energy: %.1f J (%.2f mJ/tok)",
                     summary["total_energy_j"], summary["energy_per_token_mj"])
        else:
            log.error("  FAIL: %s", summary)

        deploy_results[wl_name] = summary
        time.sleep(5)

    unlock_gpu_freq(GPUS)
    mgr.cleanup()
    kill_all()
    return deploy_results


def print_comparison(all_results):
    """Print comparison table."""
    print("\n" + "=" * 110)
    print("  STATIC BENCHMARK: Native DP8 vs PD DP4 vs PDAF TP2 (Qwen3-30B-A3B)")
    print("=" * 110)
    hdr = (f"{'Deploy':<14} {'Workload':<24} {'Thpt':>7} {'TTFT':>7} "
           f"{'TPOT':>7} {'Energy':>8} {'mJ/tok':>7} {'Req':>5} {'Fail':>5}")
    print(hdr)
    print("-" * 110)
    for deploy, wl_results in all_results.items():
        for wl, m in wl_results.items():
            if m.get("status") != "PASS":
                print(f"  {deploy:<14} {wl:<24} FAIL")
                continue
            print(f"  {deploy:<14} {wl:<24} "
                  f"{m['throughput_tok_s']:>7.1f} "
                  f"{m['ttft_avg_ms']:>7.1f} "
                  f"{m['tpot_avg_ms']:>7.1f} "
                  f"{m['total_energy_j']:>8.1f} "
                  f"{m['energy_per_token_mj']:>7.2f} "
                  f"{m['successful']:>5} "
                  f"{m['failed']:>5}")
    print("=" * 110)

    # Energy comparison vs native_dp8
    if "native_dp8" in all_results:
        print("\n  Energy savings vs native_dp8:")
        native = all_results["native_dp8"]
        for deploy, wl_results in all_results.items():
            if deploy == "native_dp8":
                continue
            for wl, m in wl_results.items():
                if m.get("status") != "PASS":
                    continue
                native_e = native.get(wl, {}).get("total_energy_j", 0)
                if native_e > 0:
                    saving = (native_e - m["total_energy_j"]) / native_e * 100
                    print(f"    {deploy:<14} {wl:<24} {saving:+.1f}%")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Static fixed-length benchmark for MoE model")
    parser.add_argument("--deploy", default="all",
                        help="Comma-sep deploy names or 'all'")
    parser.add_argument("--il", type=int, default=512,
                        help="Input length (tokens)")
    parser.add_argument("--ol", type=int, default=256,
                        help="Output length (tokens)")
    parser.add_argument("--qps", type=float, default=4.0,
                        help="Requests per second")
    parser.add_argument("--n", type=int, default=200,
                        help="Number of requests")
    parser.add_argument("--max-run-s", type=int, default=300,
                        help="Max seconds per workload run")
    parser.add_argument("--sweep", action="store_true",
                        help="Run multiple (il, ol, qps) combinations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    deploys = (list(DEPLOY_CONFIGS.keys()) if args.deploy == "all"
               else args.deploy.split(","))

    # Generate workloads
    if args.sweep:
        sweep_configs = [
            # (name, il, ol, qps, n)
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

    workloads = {}
    for name, il, ol, qps, n in sweep_configs:
        rows = gen_static_workload(il, ol, qps, n, seed=args.seed)
        save_workload(rows, name)
        workloads[name] = rows
        dur = rows[-1]["arrival_time_s"]
        log.info("Workload '%s': il=%d ol=%d qps=%.1f n=%d span=%.0fs",
                 name, il, ol, qps, n, dur)

    # Run benchmarks
    all_results = {}
    for deploy_name in deploys:
        if deploy_name not in DEPLOY_CONFIGS:
            log.error("Unknown deploy: %s (available: %s)",
                      deploy_name, list(DEPLOY_CONFIGS.keys()))
            continue
        result = run_one_deploy(deploy_name, workloads, args.max_run_s)
        if result:
            all_results[deploy_name] = result

    # Save full results
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"static_bench_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Full results saved to: %s", out_file)

    # Print comparison
    print_comparison(all_results)


if __name__ == "__main__":
    main()
