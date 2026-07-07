#!/usr/bin/env python3
"""SGLang / DynamoLLM native TP sweep on code dataset (no radix cache).

TP2, TP4, TP8 on 16-card (8 per node). Baseline = SGLang, tier = DynamoLLM.
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
log = logging.getLogger("native_tp_sweep_code")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB
from run_fixed_6scheme_7dataset import QPS_LIST, MAX_RUN_S, NGPU, RESULTS_DIR, run_one_dataset_qps

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

DATASET = "code"
TP_LIST = [2, 4, 8]

LABELS = {
    "native_tp2_baseline": "SGLang TP2",
    "native_tp4_baseline": "SGLang TP4",
    "native_tp8_baseline": "SGLang TP8",
    "native_tp2_tier": "DynamoLLM TP2",
    "native_tp4_tier": "DynamoLLM TP4",
    "native_tp8_tier": "DynamoLLM TP8",
}


def _schemes(tps: list[int]) -> list[tuple[str, callable]]:
    out = []
    for tp in tps:
        out.append((f"native_tp{tp}_baseline", lambda t=tp: MTS.start_native_tp_variant(t, False)))
        out.append((f"native_tp{tp}_tier", lambda t=tp: MTS.start_native_tp_variant(t, True)))
    return out


def _save(all_results: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"native_tp_sweep_code_{tag}_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": DATASET,
            "qps": QPS_LIST,
            "tp_list": TP_LIST,
            "disable_radix_cache": True,
            "scheme_labels": LABELS,
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
    parser.add_argument("--tp", type=str, default="2,4,8",
                        help="Comma-separated TP values (default: 2,4,8)")
    parser.add_argument(
        "--schemes",
        nargs="*",
        help="Limit scheme keys (e.g. native_tp2_baseline native_tp2_tier)",
    )
    args = parser.parse_args()

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    tps = [int(x) for x in args.tp.split(",") if x.strip()]
    schemes = _schemes(tps)
    if args.schemes:
        wanted = set(args.schemes)
        schemes = [(n, fn) for n, fn in schemes if n in wanted]

    all_results: dict = {}
    gpus = RMB.card_gpus(NGPU)

    log.info("=" * 72)
    log.info("NATIVE TP SWEEP CODE | TP=%s | SGLang(baseline) + DynamoLLM(tier)", tps)
    log.info("Nodes: %s + %s", RMB.NODE1_IP, RMB.NODE2_IP)
    log.info("=" * 72)

    for name, deploy_fn in schemes:
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s (%s)", name, LABELS.get(name, name))
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
