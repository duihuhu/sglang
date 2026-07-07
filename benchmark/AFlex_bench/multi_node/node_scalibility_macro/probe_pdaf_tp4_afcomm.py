#!/usr/bin/env python3
"""Deploy the real AFlex baseline (single decode, DA/DF TP=4, M=1) on node3/node4
and send a few requests, so we can inspect per-layer AF comm vs compute in the
DA rank0 log. Optional AF-comm toggles are forwarded via host env.

This does NOT run a full sweep; it is a targeted probe to (a) confirm whether
AFD_FUSED_PIPELINE / AFD_GPU_ONLY_IPC actually take effect in the TP4 path
(where the rank-0 communicator is wrapped by BroadcastTensorCommunicator), and
(b) capture the AFD_FWD_OVERHEAD breakdown.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import run_macro_benchmark as RMB  # noqa: E402

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

# Force M=1 (AFlex dashboard config) and V1 compositional DVFS.
_orig_afd_common = RMB._afd_common


def _patched_afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    result = _orig_afd_common(tp, ib_dev, gpu_step, tier, ngpu)
    result = result.replace("--afd-micro-batch 2", "--afd-micro-batch 1")
    result = result.replace("--afd-dynamic-micro-batch", "")
    if tier:
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
    return result


RMB._afd_common = _patched_afd_common

NGPU = 16


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    gpus = RMB.card_gpus(NGPU)
    RMB.cleanup_all()
    url = RMB.SCHEMES["pdaf"](NGPU, True)
    out = {"url": url, "requests": []}
    if url is None:
        out["status"] = "DEPLOY_FAILED"
    else:
        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        payload = {"text": "Hello", "sampling_params": {"max_new_tokens": 32, "temperature": 0}}
        for i in range(6):
            t0 = time.time()
            try:
                r = requests.post(url + "/generate", json=payload, timeout=120)
                dt = time.time() - t0
                out["requests"].append({"i": i, "status": r.status_code,
                                        "latency_s": round(dt, 3),
                                        "body_prefix": r.text[:160]})
                print(f"REQ {i} status={r.status_code} latency={dt:.3f}s {r.text[:80]!r}",
                      flush=True)
            except Exception as e:
                out["requests"].append({"i": i, "error": repr(e),
                                        "latency_s": round(time.time() - t0, 3)})
                print(f"REQ {i} ERROR {e!r}", flush=True)
            time.sleep(1)
        RMB.unlock_freq_both(gpus)

    (HERE / "results" / "probe_pdaf_tp4_afcomm.json").write_text(json.dumps(out, indent=2))
    print("SAVED results/probe_pdaf_tp4_afcomm.json", flush=True)
    RMB.cleanup_all()


if __name__ == "__main__":
    main()
