#!/usr/bin/env python3
"""Re-benchmark DynamoLLM + BiScale with min-frequency DVFS (classic policy).

Only runs:
  native_tp1_tier   (DynamoLLM)
  pd_xnode_tp1_tier (BiScale)

All 7 datasets x 6 QPS. Uses --dvfs-objective freq.
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("minfreq_tier")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB
from run_fixed_6scheme_7dataset import (
    ALL_DATASETS,
    QPS_LIST,
    SCHEMES,
    LABELS,
    MAX_RUN_S,
    NGPU,
    RESULTS_DIR,
    run_one_dataset_qps,
)

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

_orig_dvfs_flags = RMB._dvfs_flags


def _minfreq_dvfs_flags(tier):
    return _orig_dvfs_flags(tier, objective="freq")


RMB._dvfs_flags = _minfreq_dvfs_flags

TIER_SCHEMES = [
    ("native_tp1_tier", lambda: MTS.start_native_tp_variant(1, True)),
    ("pd_xnode_tp1_tier", lambda: MTS.start_pd_xnode(1, True)),
]

SCHEME_OUT_KEYS = {
    "native_tp1_tier": "native_tp1_tier_minfreq",
    "pd_xnode_tp1_tier": "pd_xnode_tp1_tier_minfreq",
}


def _save(all_results: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"fixed_6scheme_7dataset_minfreq_tier_{tag}_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": ALL_DATASETS,
            "qps": QPS_LIST,
            "dvfs_policy": "min_freq_under_slo",
            "schemes": list(SCHEME_OUT_KEYS.values()),
            "scheme_labels": {
                "native_tp1_tier_minfreq": "DynamoLLM (min-freq)",
                "pd_xnode_tp1_tier_minfreq": "BiScale (min-freq)",
            },
        },
        "results": all_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)
    return out


def main():
    import resource

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default=",".join(ALL_DATASETS),
    )
    args = parser.parse_args()

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    all_results: dict = {}
    gpus = RMB.card_gpus(NGPU)

    log.info("=" * 72)
    log.info("MIN-FREQ TIER BENCHMARK | DynamoLLM + BiScale | datasets=%s", datasets)
    log.info("=" * 72)

    for name, deploy_fn in TIER_SCHEMES:
        out_key = SCHEME_OUT_KEYS[name]
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s -> %s (%s)", name, out_key, LABELS[name])
        log.info("=" * 72)

        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            all_results[out_key] = {"__status__": "DEPLOY_FAILED"}
            _save(all_results)
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            all_results[out_key] = {"__status__": "WARMUP_FAILED"}
            _save(all_results)
            continue
        time.sleep(3)

        deploy_results = {}
        for dataset in datasets:
            for qps in QPS_LIST:
                log.info("-" * 50)
                res = run_one_dataset_qps(url, dataset, qps, gpus)
                if res is not None:
                    deploy_results[res[0]] = res[1]
                time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        all_results[out_key] = deploy_results
        _save(all_results)

    out = _save(all_results, tag="final")
    log.info("Done -> %s", out)


if __name__ == "__main__":
    main()
