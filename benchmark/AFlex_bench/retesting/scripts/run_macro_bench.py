#!/usr/bin/env python3
"""Macro-benchmark runner: Azure workloads on 4GPU and 8GPU.

Usage:
    python run_macro_bench.py --ngpu 8 --deploy pdaf
    python run_macro_bench.py --ngpu 8 --deploy all
    python run_macro_bench.py --ngpu 4 --deploy native_dp,pd_dp
"""
import sys
from pathlib import Path

# Reuse infrastructure from micro bench
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_micro_bench import (
    ENERGY_MODEL_DIR,
    PYTHON,
    MODEL,
    ROUTER_PORT,
    TTFT_SLO_MS,
    TPOT_SLO_MS,
    DeployManager,
    get_gpu_energy_mj,
    kill_all,
    lock_gpu_freq,
    unlock_gpu_freq,
    test_generate,
    wait_health,
    run_workload,
    MAX_GPU_FREQ,
)

import argparse
import asyncio
import json
import logging
import time
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("macro_bench")

HERE = Path(__file__).resolve().parent
BASE = HERE.parent
AZURE_WL_DIR = Path("/workspace/sglang-tier/benchmark/AFlex_bench/02_macro_benchmark/workloads")

# Azure workloads to test
AZURE_WORKLOADS_8GPU = [
    "workload_azure_conv_light_real.jsonl",
    "workload_azure_conv_medium_real.jsonl",
    "workload_azure_conv_heavy_real.jsonl",
    "workload_azure_code_light_real.jsonl",
    "workload_azure_code_medium_real.jsonl",
    "workload_azure_code_heavy_real.jsonl",
]

AZURE_WORKLOADS_4GPU = [
    "workload_azure_conv_light_real.jsonl",
    "workload_azure_conv_medium_real.jsonl",
    "workload_azure_code_light_real.jsonl",
    "workload_azure_code_medium_real.jsonl",
]


def get_deploy_configs(ngpu):
    if ngpu == 4:
        gpus = [0, 1, 2, 3]
        return {
            "pdaf": {
                "start": lambda mgr: mgr.start_pdaf("0,1", "2,3", tp=1),
                "gpus": gpus,
            },
            "native_dp": {
                "start": lambda mgr: mgr.start_native_dp(gpus),
                "gpus": gpus,
            },
            "pd_dp": {
                "start": lambda mgr: mgr.start_pd_dp([(0, 1), (2, 3)]),
                "gpus": gpus,
            },
        }
    else:
        gpus = [0, 1, 2, 3, 4, 5, 6, 7]
        return {
            "pdaf": {
                "start": lambda mgr: mgr.start_pdaf("0,1,2,3", "4,5,6,7", tp=2),
                "gpus": gpus,
            },
            "native_dp": {
                "start": lambda mgr: mgr.start_native_dp(gpus),
                "gpus": gpus,
            },
            "pd_dp": {
                "start": lambda mgr: mgr.start_pd_dp(
                    [(0, 1), (2, 3), (4, 5), (6, 7)]),
                "gpus": gpus,
            },
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ngpu", type=int, required=True, choices=[4, 8])
    parser.add_argument("--deploy", default="all",
                        help="Comma-sep: pdaf,native_dp,pd_dp or 'all'")
    parser.add_argument("--tier", action="store_true")
    parser.add_argument("--max-run-s", type=int, default=600)
    parser.add_argument("--workloads", default=None,
                        help="Comma-sep workload filenames (override default)")
    args = parser.parse_args()

    configs = get_deploy_configs(args.ngpu)
    deploys = list(configs.keys()) if args.deploy == "all" else args.deploy.split(",")

    if args.workloads:
        wl_files = [f.strip() for f in args.workloads.split(",")]
    elif args.ngpu == 8:
        wl_files = AZURE_WORKLOADS_8GPU
    else:
        wl_files = AZURE_WORKLOADS_4GPU

    tier_suffix = "_tier" if args.tier else ""
    result_dir = BASE / f"macro_benchmark/{args.ngpu}gpu/results"
    log_dir = BASE / f"macro_benchmark/{args.ngpu}gpu/logs"
    result_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for deploy_name in deploys:
        if deploy_name not in configs:
            log.error("Unknown: %s", deploy_name)
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

        if not args.tier:
            lock_gpu_freq(cfg["gpus"], MAX_GPU_FREQ)

        log.info("Warmup...")
        if not test_generate(port):
            log.error("Warmup failed")
            if not args.tier:
                unlock_gpu_freq(cfg["gpus"])
            mgr.cleanup()
            kill_all()
            continue
        time.sleep(3)

        url = f"http://127.0.0.1:{port}/generate"
        deploy_results = {}

        for wl_fname in wl_files:
            wl_path = AZURE_WL_DIR / wl_fname
            if not wl_path.exists():
                log.warning("Not found: %s", wl_path)
                continue

            with open(wl_path) as f:
                reqs = [json.loads(l) for l in f]

            wl_key = wl_fname.replace("workload_azure_", "").replace("_real.jsonl", "")
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

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = result_dir / f"macro_{args.ngpu}gpu{tier_suffix}_{ts}.json"
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)
    log.info("Results saved: %s", out_file)

    # Summary
    print("\n" + "=" * 110)
    print(f"  MACRO BENCHMARK ({args.ngpu} GPU, Qwen3-32B, tier={args.tier})")
    print("=" * 110)
    hdr = f"{'Deploy':<16} {'Workload':<20} {'Thpt':>7} {'TTFT':>7} {'TPOT':>7} {'Energy':>8} {'mJ/tok':>7} {'SLO%':>6} {'Req':>5}"
    print(hdr)
    print("-" * 110)
    for dep, wl_results in all_results.items():
        for wl, m in wl_results.items():
            if m.get("status") != "PASS":
                print(f"  {dep:<16} {wl:<20} FAIL")
                continue
            print(f"  {dep:<16} {wl:<20} "
                  f"{m['throughput_tok_s']:>7.1f} "
                  f"{m['ttft_proc_avg_ms']:>7.1f} "
                  f"{m['tpot_avg_ms']:>7.1f} "
                  f"{m['total_energy_j']:>8.0f} "
                  f"{m['energy_per_token_mj']:>7.1f} "
                  f"{m['slo_violation_rate']:>6.1f} "
                  f"{m['successful']:>5}")
    print("=" * 110)


if __name__ == "__main__":
    main()
