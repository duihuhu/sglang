#!/usr/bin/env python3
"""Retest summary_hphd QPS12 for DistServe/BiScale boundary point."""
from __future__ import annotations

import asyncio
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
log = logging.getLogger("retest_boundary")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB
from run_fixed_6scheme_7dataset import (
    MICRO_WL_DIR,
    SCHEMES,
    LABELS,
    run_one_dataset_qps,
)

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

DATASET = "summary_hphd"
QPS = 12
TARGET_SCHEMES = [
    ("pd_xnode_tp1_baseline", lambda: MTS.start_pd_xnode(1, False)),
    ("pd_xnode_tp1_tier", lambda: MTS.start_pd_xnode(1, True)),
]
NGPU = 16
RESULTS_DIR = HERE / "results"
BASE_JSON = RESULTS_DIR / "fixed_6scheme_7dataset_final_20260706_143908.json"


def main():
    import resource

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    if not BASE_JSON.exists():
        files = sorted(RESULTS_DIR.glob("fixed_6scheme_7dataset_final_*.json"))
        if not files:
            raise FileNotFoundError("No fixed_6scheme_7dataset_final_*.json")
        base_path = files[-1]
    else:
        base_path = BASE_JSON

    payload = json.loads(base_path.read_text())
    results = payload["results"]
    gpus = RMB.card_gpus(NGPU)
    wl_key = f"{DATASET}_qps{QPS}"

    log.info("Retest %s | base=%s", wl_key, base_path.name)

    for name, deploy_fn in TARGET_SCHEMES:
        old = results.get(name, {}).get(wl_key, {})
        log.info(
            "Old %s (%s): %.1f mJ/tok",
            name,
            LABELS[name],
            old.get("energy_per_token_mj", float("nan")),
        )

        log.info("=" * 60)
        log.info("DEPLOY %s (%s)", name, LABELS[name])
        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            log.error("Deploy failed")
            continue
        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            log.error("Warmup failed")
            continue
        time.sleep(3)

        res = run_one_dataset_qps(url, DATASET, QPS, gpus)
        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()

        if res is not None:
            results.setdefault(name, {})[res[0]] = res[1]

    b = results["pd_xnode_tp1_baseline"][wl_key]["energy_per_token_mj"]
    t = results["pd_xnode_tp1_tier"][wl_key]["energy_per_token_mj"]
    log.info("New DistServe=%.1f  BiScale=%.1f  BiScale<DistServe=%s", b, t, t < b)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"fixed_6scheme_7dataset_final_{ts}.json"
    payload["results"] = results
    payload["meta"]["retest"] = {
        "workload": wl_key,
        "reason": "BiScale vs DistServe boundary at QPS12",
        "base_file": base_path.name,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)


if __name__ == "__main__":
    main()
