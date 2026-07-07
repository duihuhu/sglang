#!/usr/bin/env python3
"""SGLang / DynamoLLM / DistServe on code dataset with --disable-radix-cache.

Uses COMMON_BENCH_SERVER_FLAGS from run_macro_benchmark (radix cache off).
SLO: TTFT=5s, TPOT=300ms. QPS: 2,4,6,8,12,16.
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
log = logging.getLogger("noradix_3scheme_code")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB
from run_fixed_6scheme_7dataset import (
    QPS_LIST,
    LABELS,
    MAX_RUN_S,
    NGPU,
    RESULTS_DIR,
    run_one_dataset_qps,
)

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

DATASET = "code"
SCHEMES = [
    ("native_tp1_baseline", lambda: MTS.start_native_tp_variant(1, False)),
    ("native_tp1_tier", lambda: MTS.start_native_tp_variant(1, True)),
    ("pd_xnode_tp1_baseline", lambda: MTS.start_pd_xnode(1, False)),
]


def _save(all_results: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"noradix_3scheme_code_{tag}_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": DATASET,
            "qps": QPS_LIST,
            "disable_radix_cache": True,
            "schemes": [s[0] for s in SCHEMES],
            "scheme_labels": {k: LABELS[k] for k, _ in SCHEMES},
            "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS,
        },
        "results": all_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)
    return out


def main():
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--schemes",
        nargs="*",
        choices=[s[0] for s in SCHEMES],
        help="Limit to specific schemes",
    )
    args = parser.parse_args()

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    schemes = SCHEMES
    if args.schemes:
        wanted = set(args.schemes)
        schemes = [(n, fn) for n, fn in SCHEMES if n in wanted]

    all_results: dict = {}
    gpus = RMB.card_gpus(NGPU)

    log.info("=" * 72)
    log.info(
        "NO-RADIX 3-SCHEME CODE | %s + %s | schemes=%s",
        RMB.NODE1_IP,
        RMB.NODE2_IP,
        [n for n, _ in schemes],
    )
    log.info("=" * 72)

    for name, deploy_fn in schemes:
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s (%s)", name, LABELS[name])
        log.info("=" * 72)

        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            all_results[name] = {"__status__": "DEPLOY_FAILED"}
            _save(all_results)
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            all_results[name] = {"__status__": "WARMUP_FAILED"}
            _save(all_results)
            continue
        time.sleep(3)

        deploy_results = {}
        for qps in QPS_LIST:
            log.info("-" * 50)
            res = run_one_dataset_qps(url, DATASET, qps, gpus)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        all_results[name] = deploy_results
        _save(all_results)

    out = _save(all_results, tag="final")
    log.info("All done -> %s", out)


if __name__ == "__main__":
    main()
