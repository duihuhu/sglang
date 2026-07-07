#!/usr/bin/env python3
"""Retest DynamoLLM + BiScale after Unified DVFS min-energy fix.

Runs all native_tp*_tier and pd_*_tier configs on conv dataset.
Compare against prior results in results/dvfs_min_energy_retest_*.json.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("dvfs_retest")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

TIER_CONFIGS = []
for tp in (1, 2, 4, 8):
    TIER_CONFIGS.append(
        (f"native_tp{tp}_tier", lambda t=tp: MTS.start_native_tp_variant(t, True)))
for tp in (1, 2, 4, 8):
    TIER_CONFIGS.append(
        (f"pd_xnode_tp{tp}_tier", lambda t=tp: MTS.start_pd_xnode(t, True)))
for tp in (1, 2, 4):
    TIER_CONFIGS.append(
        (f"pd_intra_tp{tp}_tier", lambda t=tp: MTS.start_pd_intra(t, True)))


def _save(all_results, tag: str):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"dvfs_min_energy_retest_{tag}_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": DATASET,
            "qps": QPS_LIST,
            "dvfs_policy": "min_energy_under_slo",
            "configs": [n for n, _ in TIER_CONFIGS],
        },
        "results": all_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)
    return out


def main():
    import resource

    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    gpus = RMB.card_gpus(NGPU)
    all_results = {}

    log.info("=" * 72)
    log.info("DVFS MIN-ENERGY RETEST | %d tier configs", len(TIER_CONFIGS))
    log.info("=" * 72)

    for name, deploy_fn in TIER_CONFIGS:
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s", name)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            all_results[name] = {"__status__": "DEPLOY_FAILED"}
            _save(all_results, "partial")
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            all_results[name] = {"__status__": "WARMUP_FAILED"}
            _save(all_results, "partial")
            continue
        time.sleep(3)

        deploy_results = {}
        for qps in QPS_LIST:
            res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        all_results[name] = deploy_results
        _save(all_results, "partial")

    print("\n" + "=" * 100)
    print("  DVFS MIN-ENERGY RETEST DONE")
    print("=" * 100)
    for name, wl in all_results.items():
        if "__status__" in wl:
            print(f"{name:<28} {wl['__status__']}")
            continue
        passed = sum(1 for m in wl.values() if m.get("status") == "PASS")
        q16 = wl.get("conv_qps16", {})
        e = q16.get("energy_per_token_mj", 0) if q16.get("status") == "PASS" else 0
        print(f"{name:<28} {passed}/6 PASS  QPS16={e:.0f} mJ/tok")


if __name__ == "__main__":
    main()
