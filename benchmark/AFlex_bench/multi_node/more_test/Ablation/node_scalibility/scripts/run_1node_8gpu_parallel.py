#!/usr/bin/env python3
"""Run 8GPU single-node benchmarks in parallel: node3=code, node4=conv.

Five schemes (MegaScale skipped): sglang, dynamollm, distserve, biscale, aflex.
Each node runs all 5 schemes sequentially for its assigned dataset at QPS=8.

Usage (from node1):
  python3 run_1node_8gpu_parallel.py
  python3 run_1node_8gpu_parallel.py --force
"""
from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import logging
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_DIR = ROOT / "data"
LOG_DIR = ROOT / "logs"

NODE3 = os.environ.get("BENCH_NODE3", "10.252.129.34")
NODE4 = os.environ.get("BENCH_NODE4", "10.252.129.33")

SCHEMES = ("sglang", "dynamollm", "distserve", "biscale", "aflex")
QPS = 8

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("1node_8gpu_parallel")


def _load_one_node_bench():
    path = HERE / "run_node_scalability_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_node_scalability_benchmark", str(path))
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod.OneNodeBench, mod.save_results


def _is_pass(entry: dict | None) -> bool:
    return isinstance(entry, dict) and entry.get("status") in ("PASS", "PARTIAL")


def run_node_dataset(host: str, dataset: str, force: bool) -> dict:
    OneNodeBench, _ = _load_one_node_bench()
    bench = OneNodeBench()
    key = f"{dataset}_qps{QPS}"
    out_file = DATA_DIR / f"8gpu_six_schemes_{dataset}_qps{QPS}.json"

    existing: dict = {}
    if out_file.exists():
        existing = json.loads(out_file.read_text()).get("results", {})

    results: dict = {}
    for scheme in SCHEMES:
        if not force and _is_pass((existing.get(scheme) or {}).get(key)):
            log.info("Skip PASS %s on %s %s", scheme, host, key)
            results[scheme] = existing[scheme][key]
            continue
        log.info("=" * 72)
        log.info("RUN %s | host=%s | %s | qps=%d", scheme, host, dataset, QPS)
        log.info("=" * 72)
        results[scheme] = bench.run_scheme(host, scheme, dataset, QPS)
        merged = dict(existing)
        for s, data in results.items():
            merged[s] = {key: data}
        payload = {
            "meta": {
                "benchmark": "8gpu_1node_six_schemes",
                "dataset": dataset,
                "qps": QPS,
                "gpu_count": 8,
                "nodes": 1,
                "host": host,
                "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "results": merged,
        }
        out_file.write_text(json.dumps(payload, indent=2) + "\n")
        log.info("Saved partial %s", out_file)
        time.sleep(5)

    merged = dict(existing)
    for s, data in results.items():
        merged[s] = {key: data}
    payload = {
        "meta": {
            "benchmark": "8gpu_1node_six_schemes",
            "dataset": dataset,
            "qps": QPS,
            "gpu_count": 8,
            "nodes": 1,
            "host": host,
            "schemes": list(SCHEMES),
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "results": merged,
    }
    out_file.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("Done %s on %s -> %s", dataset, host, out_file)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log.info("Parallel 8GPU 1-node: node3(%s)=code, node4(%s)=conv, qps=%d", NODE3, NODE4, QPS)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        f_code = pool.submit(run_node_dataset, NODE3, "code", args.force)
        f_conv = pool.submit(run_node_dataset, NODE4, "conv", args.force)
        code_res = f_code.result()
        conv_res = f_conv.result()

    log.info("All parallel runs finished")
    for scheme in SCHEMES:
        cr = code_res.get(scheme, {})
        cv = conv_res.get(scheme, {})
        log.info(
            "  %s: code=%s E/tok=%s | conv=%s E/tok=%s",
            scheme,
            cr.get("status"), cr.get("energy_per_token_mj"),
            cv.get("status"), cv.get("energy_per_token_mj"),
        )


if __name__ == "__main__":
    main()
