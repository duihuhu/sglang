#!/usr/bin/env python3
"""A/B test AF-comm optimizations on the 2Decode V1 AFlex config (node3/node4).

Compares three engine-level AF communication modes for the M=1 hot path:
  - baseline: per-layer send_tensor + recv_tensor (2 Python->C++ calls)
  - fused:    C++ FusedPipeline.send_recv (1 Python->C++ call per layer)
  - fused_cs: fused + dedicated high-priority comm stream for overlap

All modes reuse run_conv_pdaf_2decode.start_pdaf_2decode (fixed bootstrap port)
and the V1 compositional DVFS path. Results are saved incrementally.
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

import run_conv_pdaf_2decode as P2D  # noqa: E402
import run_macro_benchmark as RMB  # noqa: E402

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0
P2D.RMB.TTFT_SLO_MS = RMB.TTFT_SLO_MS
P2D.RMB.TPOT_SLO_MS = RMB.TPOT_SLO_MS

QPS_LIST = [8, 12, 16]
DATASET = "conv"
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# (name, env overrides). V1 compositional DVFS via models_v1 + decode_compositional.
# gpu_only removes the CPU busy-poll from the M=1 critical path: recv is a GPU
# wait-kernel on device memory, so the CPU races ahead enqueuing later layers.
DEFAULT_MODES = "baseline,fused,gpu_only"
ALL_MODES = {
    "baseline": {
        "AFD_FUSED_PIPELINE": "0", "AFD_FUSED_COMM_STREAM": "0",
        "AFD_GPU_ONLY_IPC": "0", "AFD_IPC_SYNC_MODE": "ipc_event",
    },
    "fused": {
        "AFD_FUSED_PIPELINE": "1", "AFD_FUSED_COMM_STREAM": "0",
        "AFD_GPU_ONLY_IPC": "0", "AFD_IPC_SYNC_MODE": "ipc_event",
    },
    "fused_cs": {
        "AFD_FUSED_PIPELINE": "1", "AFD_FUSED_COMM_STREAM": "1",
        "AFD_GPU_ONLY_IPC": "0", "AFD_IPC_SYNC_MODE": "ipc_event",
    },
    "gpu_only": {
        "AFD_FUSED_PIPELINE": "0", "AFD_FUSED_COMM_STREAM": "0",
        "AFD_GPU_ONLY_IPC": "1", "AFD_IPC_SYNC_MODE": "gpu_signal",
    },
}


def patch_v1_common():
    """Patch P2D._afd_common to use V1 compositional DVFS (best single-decode DVFS)."""
    orig = P2D._afd_common

    def _patched(tp, ib_dev, gpu_step, tier, bs_port):
        result = orig(tp, ib_dev, gpu_step, tier, bs_port)
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
        return result

    P2D._afd_common = _patched
    return orig


def save(payload, out_file, status="partial"):
    payload["meta"]["status"] = status
    payload["meta"]["updated_at"] = time.strftime("%Y%m%d_%H%M%S")
    tmp = out_file.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(out_file)
    print(f"[save] {out_file}", flush=True)


def main():
    import argparse
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", default=DEFAULT_MODES)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    mode_names = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in mode_names:
        if m not in ALL_MODES:
            raise ValueError(f"Unknown mode {m}; available={sorted(ALL_MODES)}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.out) if args.out else RESULTS_DIR / f"conv_2decode_afcomm_ab_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP, "model": RMB.MODEL,
            "layout": "2Decode V1 (P TP4 + 2x D TP2), M=1, V1 compositional DVFS",
            "dataset": DATASET, "qps": QPS_LIST,
            "ttft_slo_ms": RMB.TTFT_SLO_MS, "tpot_slo_ms": RMB.TPOT_SLO_MS,
            "modes": mode_names,
            "status": "partial",
        },
        "results": {},
    }
    save(payload, out_file)

    gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    orig_common = patch_v1_common()
    try:
        for mode_name in mode_names:
            env = ALL_MODES[mode_name]
            for k, v in env.items():
                os.environ[k] = v
            deploy_results = payload["results"].setdefault(f"aflex_2decode_v1_{mode_name}", {})
            print("=" * 80, flush=True)
            print(f"DEPLOY mode={mode_name} env={env}", flush=True)
            print("=" * 80, flush=True)

            RMB.cleanup_all()
            url = P2D.start_pdaf_2decode(True)
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
    finally:
        P2D._afd_common = orig_common

    save(payload, out_file, status="completed")
    print(f"AB COMPLETED: {out_file}", flush=True)


if __name__ == "__main__":
    main()
