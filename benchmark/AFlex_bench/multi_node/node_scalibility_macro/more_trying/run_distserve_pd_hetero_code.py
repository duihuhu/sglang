#!/usr/bin/env python3
"""DistServe on heterogeneous PD: P=4×TP2 + D=2×TP4, code dataset.

Same topology as BiScale hetero, baseline (locked 1410 MHz, no DVFS).
SLO: TTFT=5s, TPOT=300ms. QPS: 2,4,6,8,12,16.
"""
from __future__ import annotations

import argparse
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
log = logging.getLogger("distserve_pd_hetero_code")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
import run_macro_benchmark as RMB

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "code"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEME_KEY = "pd_hetero_baseline"
MACRO_WL_DIR = MACRO_DIR / "workloads"


def _wl_key(qps: int) -> str:
    return f"{DATASET}_qps{qps}"


def _workload_file(qps: int) -> Path | None:
    path = MACRO_WL_DIR / f"macro_{DATASET}_qps{qps}.jsonl"
    return path if path.exists() else None


def run_one_qps(url: str, qps: int, gpus: list[int]):
    wl_file = _workload_file(qps)
    if wl_file is None:
        log.error("Workload missing: %s", _wl_key(qps))
        return None
    with open(wl_file) as f:
        reqs = [json.loads(line) for line in f]
    key = _wl_key(qps)
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(MAX_RUN_S, last_arrival + 150), 900))
    log.info("  %s (%d reqs, run_window=%ds)", key, len(reqs), run_s)
    summary = asyncio.run(
        RMB.run_workload(reqs, url + "/generate", gpus, gpus, run_s)
    )
    if summary.get("status") == "PASS":
        log.info(
            "  Thpt=%.1f tok/s | TTFT=%.1fms | TPOT=%.1fms | "
            "E=%.0fJ (%.1f mJ/tok) | SLO=%.1f%%",
            summary["throughput_tok_s"],
            summary["ttft_proc_avg_ms"],
            summary["tpot_avg_ms"],
            summary["total_energy_j"],
            summary["energy_per_token_mj"],
            summary["slo_violation_rate"],
        )
    else:
        log.error("  FAIL: %s", summary)
    return key, summary


def _save(results: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"distserve_pd_hetero_code_{tag}_{ts}.json"
    payload = {
        "meta": {
            "scheme": SCHEME_KEY,
            "label": "DistServe (P=4×TP2+D=2×TP4)",
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": DATASET,
            "qps": QPS_LIST,
            "topology": "P=4×TP2 prefill @node1, D=2×TP4 decode @node2",
            "dvfs_policy": "baseline_locked",
            "disable_radix_cache": True,
            "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS,
        },
        "results": {SCHEME_KEY: results},
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)
    return out


def main():
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", type=str, default=None,
                        help="Comma-separated QPS list (default: all)")
    args = parser.parse_args()

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    qps_list = QPS_LIST
    if args.qps:
        qps_list = [int(x) for x in args.qps.split(",")]

    gpus = RMB.card_gpus(NGPU)
    results: dict = {}

    log.info("=" * 72)
    log.info("DistServe PD-HETERO CODE | P=4×TP2 + D=2×TP4 | baseline (no DVFS)")
    log.info("Nodes: %s (prefill) + %s (decode)", RMB.NODE1_IP, RMB.NODE2_IP)
    log.info("=" * 72)

    RMB.cleanup_all()
    url = RMB.start_pd_hetero(NGPU, tier=False)
    if url is None:
        results["__status__"] = "DEPLOY_FAILED"
        _save(results)
        raise SystemExit(1)

    RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
    if not RMB.test_generate(url):
        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        results["__status__"] = "WARMUP_FAILED"
        _save(results)
        raise SystemExit(1)
    time.sleep(3)

    for qps in qps_list:
        log.info("-" * 50)
        res = run_one_qps(url, qps, gpus)
        if res is not None:
            results[res[0]] = res[1]
        _save(results)
        time.sleep(5)

    RMB.unlock_freq_both(gpus)
    RMB.cleanup_all()
    results.pop("__status__", None)
    out = _save(results, tag="final")
    log.info("All done -> %s", out)


if __name__ == "__main__":
    main()
