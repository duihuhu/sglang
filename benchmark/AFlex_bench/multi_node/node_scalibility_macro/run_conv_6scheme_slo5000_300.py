#!/usr/bin/env python3
"""Conv-only 6-scheme benchmark with relaxed SLO.

SLO:
  TTFT = 5000 ms
  TPOT = 300 ms

Schemes:
  SGLang      = native_tp_baseline (TP=2)
  DynamoLLM   = native_tp_tier (TP=2 + DVFS)
  DistServe   = pd_hetero_baseline (P=4xTP2, D=2xTP4)
  BiScale     = pd_hetero_tier (same topology + DVFS)
  MegaScale   = pdaf_baseline (M=1)
  AFlex       = pdaf_tier (M=1 + V1 compositional DVFS)
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("conv_slo5000_300")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_macro_benchmark as RMB

# Relax SLO globally for both metric calculation and DVFS launch flags.
RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

# Force PDAF M=1 and use the validated V1 compositional DVFS path for AFlex.
_orig_afd_common = RMB._afd_common

def _patched_afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    result = _orig_afd_common(tp, ib_dev, gpu_step, tier, ngpu)
    result = result.replace("--afd-micro-batch 2", "--afd-micro-batch 1")
    # Ensure no dynamic M is accidentally enabled if inherited from another script.
    result = result.replace("--afd-dynamic-micro-batch", "")
    if tier:
        # Conv sweep showed V1 compositional DVFS is consistently better than
        # the V2 coupled path for AFlex under this loose SLO.
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
    return result

RMB._afd_common = _patched_afd_common

SCHEMES_ORDER = [
    ("native_tp", "baseline"),
    ("native_tp", "tier"),
    ("pd_hetero", "baseline"),
    ("pd_hetero", "tier"),
    ("pdaf", "baseline"),
    ("pdaf", "tier"),
]
QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    gpus = RMB.card_gpus(NGPU)
    all_results = {}

    log.info("=" * 72)
    log.info("CONV-ONLY 6-SCHEME BENCHMARK | SLO TTFT=5000ms TPOT=300ms")
    log.info("Nodes: node1=%s, node2=%s | 16-card | PDAF M=1", RMB.NODE1_IP, RMB.NODE2_IP)
    log.info("=" * 72)

    for scheme, mode in SCHEMES_ORDER:
        tier = mode == "tier"
        full_name = f"{scheme}_{mode}"
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s", full_name)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = RMB.SCHEMES[scheme](NGPU, tier)
        if url is None:
            log.error("%s deployment FAILED", full_name)
            RMB.cleanup_all()
            all_results[full_name] = {"__status__": "DEPLOY_FAILED"}
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        log.info("Warmup...")
        if not RMB.test_generate(url):
            log.error("%s warmup FAILED", full_name)
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            all_results[full_name] = {"__status__": "WARMUP_FAILED"}
            continue
        time.sleep(3)

        deploy_results = {}
        for qps in QPS_LIST:
            log.info("-" * 50)
            res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        all_results[full_name] = deploy_results

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = RESULTS_DIR / f"conv_6scheme_slo5000_300_{ts}.json"
        payload = {
            "meta": {
                "node1": RMB.NODE1_IP,
                "node2": RMB.NODE2_IP,
                "model": RMB.MODEL,
                "ngpu_total": NGPU,
                "gpus_per_node": gpus,
                "dataset": DATASET,
                "qps": QPS_LIST,
                "schemes": [f"{s}_{m}" for s, m in SCHEMES_ORDER],
                "ttft_slo_ms": RMB.TTFT_SLO_MS,
                "tpot_slo_ms": RMB.TPOT_SLO_MS,
                "pdaf_micro_batch": 1,
            },
            "results": all_results,
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Results saved: %s", out_file)

    print("\n" + "=" * 104)
    print("  CONV 6-SCHEME RESULTS | SLO TTFT=5000ms TPOT=300ms")
    print("=" * 104)
    print(f"{'Deploy':<22} {'Workload':<14} {'Thpt':>8} {'TTFT':>8} {'TPOT':>8} {'E_tot':>9} {'mJ/tok':>8} {'SLO%':>6}")
    print("-" * 104)
    for dep, wl in all_results.items():
        if "__status__" in wl:
            print(f"{dep:<22} {wl['__status__']}")
            continue
        for w, m in wl.items():
            if m.get("status") != "PASS":
                print(f"{dep:<22} {w:<14} FAIL")
                continue
            print(f"{dep:<22} {w:<14} {m['throughput_tok_s']:>8.1f} "
                  f"{m['ttft_proc_avg_ms']:>8.1f} {m['tpot_avg_ms']:>8.1f} "
                  f"{m['total_energy_j']:>9.0f} {m['energy_per_token_mj']:>8.1f} "
                  f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
