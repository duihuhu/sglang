#!/usr/bin/env python3
"""Run the PDAF 3-Decode layout on conv qps8/12/16 and save metrics.

Uses V1 compositional DVFS (best single-decode config) so results are directly
comparable to the 2Decode V1 numbers.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_conv_pdaf_3decode as P3D  # noqa: E402
import run_macro_benchmark as RMB  # noqa: E402

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [8, 12, 16]
DATASET = "conv"
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"

# Use V1 compositional DVFS (matches the 2Decode V1 comparison baseline).
_ORIG_COMMON = P3D._afd_common


def _v1_common(tp, ib_dev, gpu_step, tier, bs_port):
    result = _ORIG_COMMON(tp, ib_dev, gpu_step, tier, bs_port)
    if tier:
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
    return result


P3D._afd_common = _v1_common


def save(payload, out_file, status="partial"):
    payload["meta"]["status"] = status
    payload["meta"]["updated_at"] = time.strftime("%Y%m%d_%H%M%S")
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_file)
    print(f"[save] {out_file}", flush=True)


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"conv_3decode_v1_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP, "model": RMB.MODEL,
            "layout": "PDAF 3xDecode: P(TP2 node1) + 3x D(TP2), M=1, V1 compositional DVFS",
            "dataset": DATASET, "qps": QPS_LIST,
            "ttft_slo_ms": RMB.TTFT_SLO_MS, "tpot_slo_ms": RMB.TPOT_SLO_MS,
            "status": "partial",
        },
        "results": {"aflex_3decode_v1": {}},
    }
    save(payload, out_file)
    dr = payload["results"]["aflex_3decode_v1"]

    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    RMB.cleanup_all()
    url = P3D.start_pdaf_3decode(True)
    if url is None:
        dr["__status__"] = "DEPLOY_FAILED"
        save(payload, out_file)
        RMB.cleanup_all()
        return

    RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
    if not RMB.test_generate(url):
        dr["__status__"] = "WARMUP_FAILED"
        save(payload, out_file)
        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        return
    time.sleep(3)

    try:
        for qps in QPS_LIST:
            res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
            if res is not None:
                dr[res[0]] = res[1]
                save(payload, out_file)
            time.sleep(5)
    finally:
        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        save(payload, out_file, status="completed")
    print(f"3DECODE SWEEP COMPLETED: {out_file}", flush=True)


if __name__ == "__main__":
    main()
