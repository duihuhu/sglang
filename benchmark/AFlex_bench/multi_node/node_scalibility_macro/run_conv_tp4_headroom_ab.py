#!/usr/bin/env python3
"""A/B test headroom-aggressive DVFS on the real AFlex baseline.

Baseline = single decode, DA/DF TP=4 (start_pdaf 16-card), M=1, V1 compositional
DVFS. Compares three modes on conv qps8/12/16:
  - v1_base:    V1 compositional DVFS (current best single-decode config)
  - v1_hr06:    + headroom-aggressive engages when latency < 60% of TPOT SLO
  - v1_hr04:    + headroom-aggressive engages when latency < 40% of TPOT SLO

Headroom-aggressive only lowers the attn (DA) frequency when the predicted
latency still meets the SLO, so it should save energy/token on loose-SLO conv
without raising SLO violations. Results saved incrementally.
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

import run_macro_benchmark as RMB  # noqa: E402

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [8, 12, 16]
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Force M=1 + V1 compositional DVFS (the validated best single-decode config).
_ORIG_AFD_COMMON = RMB._afd_common


def _v1_afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    result = _ORIG_AFD_COMMON(tp, ib_dev, gpu_step, tier, ngpu)
    result = result.replace("--afd-micro-batch 2", "--afd-micro-batch 1")
    result = result.replace("--afd-dynamic-micro-batch", "")
    if tier:
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
    return result


RMB._afd_common = _v1_afd_common

# (name, headroom env value or None)
MODES = [
    ("v1_base", None),
    ("v1_hr06", "0.6"),
    ("v1_hr04", "0.4"),
]


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
    out_file = RESULTS_DIR / f"conv_tp4_headroom_ab_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP, "model": RMB.MODEL,
            "layout": "single decode DA/DF TP4 (start_pdaf 16), M=1, V1 compositional DVFS",
            "dataset": DATASET, "qps": QPS_LIST,
            "ttft_slo_ms": RMB.TTFT_SLO_MS, "tpot_slo_ms": RMB.TPOT_SLO_MS,
            "modes": [m for m, _ in MODES],
            "status": "partial",
        },
        "results": {},
    }
    save(payload, out_file)

    gpus = RMB.card_gpus(NGPU)
    for mode_name, hr in MODES:
        if hr is None:
            os.environ.pop("AFD_DVFS_HEADROOM_AGGRESSIVE", None)
        else:
            os.environ["AFD_DVFS_HEADROOM_AGGRESSIVE"] = hr
        deploy_results = payload["results"].setdefault(f"aflex_tp4_{mode_name}", {})
        print("=" * 80, flush=True)
        print(f"DEPLOY mode={mode_name} headroom={hr}", flush=True)
        print("=" * 80, flush=True)

        RMB.cleanup_all()
        url = RMB.SCHEMES["pdaf"](NGPU, True)
        if url is None:
            deploy_results["__status__"] = "DEPLOY_FAILED"
            save(payload, out_file)
            RMB.cleanup_all()
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            deploy_results["__status__"] = "WARMUP_FAILED"
            save(payload, out_file)
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            continue
        time.sleep(3)

        try:
            for qps in QPS_LIST:
                res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
                if res is not None:
                    deploy_results[res[0]] = res[1]
                    save(payload, out_file)
                time.sleep(5)
        finally:
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            save(payload, out_file)

    save(payload, out_file, status="completed")
    print(f"AB COMPLETED: {out_file}", flush=True)


if __name__ == "__main__":
    main()
