#!/usr/bin/env python3
"""Full 6-scheme x 7-dataset benchmark on 16-card (node3=34, node4=33).

Schemes:
  1. SGLang      = native_tp_baseline (TP=2, locked 1410MHz)
  2. DynamoLLM   = native_tp_tier     (TP=2, DVFS)
  3. DistServe   = pd_hetero_baseline  (P=4xTP2, D=2xTP4, locked 1410MHz)
  4. BiScale     = pd_hetero_tier      (same topology, DVFS)
  5. MegaScale   = pdaf_baseline       (AF disagg M=1, locked 1410MHz)
  6. AFlex       = pdaf_tier           (AF disagg M=1, DVFS)

Datasets (7 total):
  Variable-length: code, conv
  Fixed-length: qa_lpld, chatbot_lphd, balanced_mpmd, rag_hpld, summary_hphd

QPS sweep: 2, 4, 6, 8, 12, 16

Usage:
    MN_NODE1_IP=10.252.129.34 MN_NODE2_IP=10.252.129.33 \
    python3 -u run_full_7dataset.py 2>&1 | tee /tmp/full_7ds_bench.log
"""
import importlib
import os
import sys
import json
import time
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("full_7ds")

# Force node3/node4
os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

# Import the main benchmark module (same directory)
sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_macro_benchmark as RMB

# --- Patch PDAF to use M=1 ---
_orig_afd_common = RMB._afd_common

def _patched_afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    result = _orig_afd_common(tp, ib_dev, gpu_step, tier, ngpu)
    # Replace --afd-micro-batch 2 with --afd-micro-batch 1
    result = result.replace("--afd-micro-batch 2", "--afd-micro-batch 1")
    return result

RMB._afd_common = _patched_afd_common

# --- Configuration ---
SCHEMES_ORDER = [
    ("native_tp", "baseline"),   # SGLang
    ("native_tp", "tier"),       # DynamoLLM
    ("pd_hetero", "baseline"),   # DistServe
    ("pd_hetero", "tier"),       # BiScale
    ("pdaf", "baseline"),        # MegaScale
    ("pdaf", "tier"),            # AFlex
]

VARIABLE_DATASETS = ["code", "conv"]
FIXED_DATASETS = ["qa_lpld", "chatbot_lphd", "balanced_mpmd", "rag_hpld", "summary_hphd"]
ALL_DATASETS = VARIABLE_DATASETS + FIXED_DATASETS

QPS_LIST = [2, 4, 6, 8, 12, 16]
MAX_RUN_S = 400
NGPU = 16

# Workload paths
MACRO_WL_DIR = Path(__file__).resolve().parent / "workloads"
MICRO_WL_DIR = Path("/mnt/workspace/lt/sglang/benchmark/AFlex_bench/retesting/workloads")

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def get_workload_file(dataset, qps):
    """Get workload file path for a given dataset and QPS."""
    if dataset in VARIABLE_DATASETS:
        return MACRO_WL_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    else:
        return MICRO_WL_DIR / f"micro_{dataset}_qps{qps}.jsonl"


def run_one_dataset_qps(url, dataset, qps, gpus):
    """Run one workload; return (wl_key, metrics) or None."""
    wl_file = get_workload_file(dataset, qps)
    if not wl_file.exists():
        log.warning("  workload not found: %s", wl_file)
        return None

    # Temporarily patch WORKLOAD_DIR in main module so run_one_workload works
    orig_wl_dir = RMB.WORKLOAD_DIR
    if dataset in VARIABLE_DATASETS:
        RMB.WORKLOAD_DIR = MACRO_WL_DIR
        scenario = dataset
        prefix = "macro"
    else:
        RMB.WORKLOAD_DIR = MICRO_WL_DIR
        scenario = dataset
        prefix = "micro"

    # The workload file must match pattern: {prefix}_{scenario}_qps{qps}.jsonl
    # run_one_workload uses: WORKLOAD_DIR / f"macro_{scenario}_qps{qps}.jsonl"
    # For fixed datasets we need to handle the "micro_" prefix
    import asyncio
    wl_key = f"{dataset}_qps{qps}"

    with open(wl_file) as f:
        reqs = [json.loads(l) for l in f]

    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    import numpy as np
    run_s = int(min(max(MAX_RUN_S, last_arrival + 150), 900))
    log.info("  %s (%d reqs, run_window=%ds)", wl_key, len(reqs), run_s)

    summary = asyncio.run(RMB.run_workload(
        reqs, url + "/generate", gpus, gpus, run_s))

    if summary.get("status") == "PASS":
        log.info("  Thpt=%.1f tok/s | TTFT=%.1fms | TPOT=%.1fms | "
                 "E=%.0fJ (%.1f mJ/tok) | SLO=%.1f%%",
                 summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                 summary["tpot_avg_ms"], summary["total_energy_j"],
                 summary["energy_per_token_mj"], summary["slo_violation_rate"])
    else:
        log.error("  FAIL: %s", summary)

    RMB.WORKLOAD_DIR = orig_wl_dir
    return wl_key, summary


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    log.info("=" * 70)
    log.info("FULL 7-DATASET BENCHMARK: 6 schemes x 7 datasets x QPS %s", QPS_LIST)
    log.info("Nodes: node1=%s (prefill), node2=%s (decode)", RMB.NODE1_IP, RMB.NODE2_IP)
    log.info("PDAF micro-batch = 1 (M=1)")
    log.info("=" * 70)

    all_results = {}
    gpus = RMB.card_gpus(NGPU)

    for scheme, mode in SCHEMES_ORDER:
        tier = (mode == "tier")
        full_name = f"{scheme}_{mode}"
        log.info("\n" + "=" * 70)
        log.info("DEPLOY: %s [16-card, GPUs %s/node]", full_name, gpus)
        log.info("=" * 70)

        RMB.cleanup_all()
        url = RMB.SCHEMES[scheme](NGPU, tier)
        if url is None:
            log.error("  %s deployment FAILED", full_name)
            RMB.cleanup_all()
            all_results[full_name] = {"__status__": "DEPLOY_FAILED"}
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)

        log.info("Warmup...")
        if not RMB.test_generate(url):
            log.error("  warmup failed for %s", full_name)
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            all_results[full_name] = {"__status__": "WARMUP_FAILED"}
            continue
        time.sleep(3)

        deploy_results = {}
        for dataset in ALL_DATASETS:
            for qps in QPS_LIST:
                log.info("-" * 50)
                res = run_one_dataset_qps(url, dataset, qps, gpus)
                if res is not None:
                    deploy_results[res[0]] = res[1]
                time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        all_results[full_name] = deploy_results

        # Save incremental results
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = RESULTS_DIR / f"full_7ds_{ts}.json"
        payload = {
            "meta": {
                "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP,
                "model": RMB.MODEL, "ngpu_total": NGPU,
                "gpus_per_node": gpus,
                "schemes": [f"{s}_{m}" for s, m in SCHEMES_ORDER],
                "datasets": ALL_DATASETS,
                "qps": QPS_LIST,
                "pdaf_micro_batch": 1,
            },
            "results": all_results
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Results saved: %s", out_file)

    # Final summary
    print("\n" + "=" * 110)
    print("  FULL 7-DATASET RESULTS (16-card, Qwen3-32B, M=1)")
    print("=" * 110)
    hdr = (f"{'Deploy':<22} {'Workload':<20} {'Thpt':>7} {'TTFT':>7} "
           f"{'TPOT':>7} {'E_tot':>8} {'mJ/tok':>8} {'SLO%':>5}")
    print(hdr)
    print("-" * 110)
    for dep, wl in all_results.items():
        if "__status__" in wl:
            print(f"{dep:<22} {wl['__status__']}")
            continue
        for w, m in wl.items():
            if not isinstance(m, dict) or m.get("status") != "PASS":
                print(f"{dep:<22} {w:<20} FAIL")
                continue
            print(f"{dep:<22} {w:<20} {m['throughput_tok_s']:>7.1f} "
                  f"{m['ttft_proc_avg_ms']:>7.1f} {m['tpot_avg_ms']:>7.1f} "
                  f"{m['total_energy_j']:>8.0f} {m['energy_per_token_mj']:>8.1f} "
                  f"{m['slo_violation_rate']:>5.1f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
