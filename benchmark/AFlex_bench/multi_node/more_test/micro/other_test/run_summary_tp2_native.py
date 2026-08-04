#!/usr/bin/env python3
"""Run summary_hphd × QPS 2/4/8/16 for SGLang/DynamoLLM native TP2 (micro benchmark口径).

Follows micro/run_benchmark.py: per-request recording, restart_per_point deploy,
energy over all 16 GPUs (input+output token denominator).

Usage (on node1 host):
  python3 run_summary_tp2_native.py
  MN_NODE2_IP=10.252.129.34 python3 run_summary_tp2_native.py --force
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
log = logging.getLogger("summary_tp2")

MICRO_ROOT = Path(__file__).resolve().parent.parent
MORE_TRYING = MICRO_ROOT.parents[1] / "more_test/macro/scripts" / "more_trying"
MACRO_DIR = MICRO_ROOT.parents[1] / "more_test/macro/scripts"

sys.path.insert(0, str(MICRO_ROOT))
sys.path.insert(0, str(MORE_TRYING))
sys.path.insert(0, str(MACRO_DIR))

os.environ.setdefault("MN_NODE1_IP", "10.252.129.36")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.34")

import bench_common as BC
import deploy_schemes as DS
import run_macro_benchmark as RMB
import run_more_trying_sweep as MTS

import run_micro_6scheme_sweep as MS  # noqa: E402

DATASET = "summary_hphd"
QPS_LIST = [2, 4, 8, 16]
GPUS = list(range(8))
ALL_GPUS = GPUS
MAX_FREQ = RMB.MAX_GPU_FREQ

SCHEMES = [
    ("native_tp2_baseline", "sglang", False),
    ("native_tp2_tier", "dynamollm", True),
]

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE = OUT_DIR / "summary_tp2_native.json"


def _sync_nodes() -> None:
    n1 = os.environ["MN_NODE1_IP"]
    n2 = os.environ["MN_NODE2_IP"]
    RMB.NODE1_IP = n1
    RMB.NODE2_IP = n2
    MTS.RMB.NODE1_IP = n1
    MTS.RMB.NODE2_IP = n2
    MS.RMB.NODE1_IP = n1
    MS.RMB.NODE2_IP = n2
    MS.NODE1 = n1
    MS.NODE2 = n2
    DS._sync_nodes()


def _is_pass(entry: dict | None) -> bool:
    return isinstance(entry, dict) and entry.get("status") in ("PASS", "PARTIAL_TIMEOUT")


def deploy_native_tp2(scheme_key: str, is_dynamo: bool) -> str | None:
    log.info("=" * 60)
    log.info("Deploying %s: 8xTP2 (4/node), round-robin", scheme_key)
    log.info("=" * 60)

    RMB.cleanup_all()
    time.sleep(8)

    if is_dynamo:
        RMB.unlock_freq_both(ALL_GPUS)
    else:
        RMB.lock_freq_both(ALL_GPUS, MAX_FREQ)

    url = MTS.start_native_tp_variant(2, is_dynamo)
    if url is None:
        return None
    log.info("  Deploy OK: %s", url)
    return url


def unlock_after_run(scheme_key: str) -> None:
    if scheme_key == "native_tp2_baseline":
        return
    RMB.unlock_freq_both(ALL_GPUS)


def _run_point(url: str, qps: int) -> dict:
    reqs = BC.load_workload(DATASET, qps)
    run_s = BC.run_window_s(reqs)
    log.info("  %s (%d reqs, run_window=%ds)", BC.wl_key(DATASET, qps), len(reqs), run_s)
    summary = asyncio.run(
        BC.run_workload_with_requests(
            reqs,
            url + "/generate",
            RMB.get_energy_local,
            RMB.get_energy_remote,
            GPUS,
            GPUS,
            run_s,
        )
    )
    if summary.get("status") in ("PASS", "PARTIAL_TIMEOUT"):
        log.info(
            "  %s: thpt=%.1f TTFT p50/p90=%.1f/%.1f TPOT p50/p90=%.1f/%.1f E/tok=%.1fmJ",
            summary["status"],
            summary["throughput_tok_s"],
            summary.get("ttft_proc_p50_ms", 0),
            summary.get("ttft_proc_p90_ms", 0),
            summary.get("tpot_p50_ms", 0),
            summary.get("tpot_p90_ms", 0),
            summary.get("energy_per_token_mj", 0),
        )
    else:
        log.error("  FAIL: %s", summary.get("status"))
    return summary


def _save(results: dict, meta: dict) -> None:
    payload = {
        "meta": meta,
        "deploy": {
            "native_tp2_baseline": {
                "label": "SGLang",
                "topology": "native_tp2",
                "tier": False,
                "gpus": 16,
                "nodes": 2,
                "tp": 2,
                "instances": 8,
                "freq_policy": "locked_1410mhz",
                "deploy_policy": "restart_per_point",
            },
            "native_tp2_tier": {
                "label": "DynamoLLM",
                "topology": "native_tp2",
                "tier": True,
                "gpus": 16,
                "nodes": 2,
                "tp": 2,
                "instances": 8,
                "freq_policy": "unified_dvfs",
                "deploy_policy": "restart_per_point",
            },
        },
        "results": results,
    }
    RESULT_FILE.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("Saved %s", RESULT_FILE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", default="2,4,8,16")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    qps_list = [int(x) for x in args.qps.split(",") if x.strip()]
    _sync_nodes()

    if RESULT_FILE.exists() and not args.force:
        results = json.loads(RESULT_FILE.read_text()).get("results", {})
    else:
        results = {}

    meta = {
        "benchmark": "micro_summary_tp2_native",
        "node1": RMB.NODE1_IP,
        "node2": RMB.NODE2_IP,
        "dataset": DATASET,
        "qps": qps_list,
        "schemes": [s[0] for s in SCHEMES],
        "energy_denominator": "input_plus_output",
        "ttft_slo_ms": BC.TTFT_SLO_MS,
        "tpot_slo_ms": BC.TPOT_SLO_MS,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    for scheme_key, _scheme_name, is_dynamo in SCHEMES:
        scheme_bucket = results.setdefault(scheme_key, {})
        for qps in qps_list:
            key = BC.wl_key(DATASET, qps)
            if not args.force and _is_pass(scheme_bucket.get(key)):
                log.info("Skip PASS %s %s", scheme_key, key)
                continue

            label = "SGLang TP2" if not is_dynamo else "DynamoLLM TP2"
            log.info("\n" + "#" * 72)
            log.info("RUN %s | %s | QPS=%d", label, DATASET, qps)
            log.info("#" * 72)

            url = deploy_native_tp2(scheme_key, is_dynamo)
            if url is None:
                scheme_bucket[key] = {"status": "DEPLOY_FAILED"}
                _save(results, meta)
                continue

            if not DS.warmup_url(url, scheme_key):
                scheme_bucket[key] = {"status": "WARMUP_FAILED"}
                unlock_after_run(scheme_key)
                RMB.cleanup_all()
                _save(results, meta)
                continue

            time.sleep(3)
            entry = _run_point(url, qps)
            entry["deploy"] = {
                "scheme": scheme_key,
                "topology": "native_tp2",
                "tp": 2,
                "tier": is_dynamo,
                "node1": RMB.NODE1_IP,
                "node2": RMB.NODE2_IP,
                "router_url": url,
            }
            scheme_bucket[key] = entry
            unlock_after_run(scheme_key)
            RMB.cleanup_all()
            _save(results, meta)
            time.sleep(5)

    meta["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    _save(results, meta)
    log.info("Done -> %s", RESULT_FILE)


if __name__ == "__main__":
    main()
