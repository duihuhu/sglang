#!/usr/bin/env python3
"""Conv-only 6-scheme benchmark with relaxed SLO — run on node3+node4.

SLO:
  TTFT = 5000 ms
  TPOT = 300 ms

Schemes:
  SGLang      = native_tp_baseline (TP=2)
  DynamoLLM   = native_tp_tier (TP=2 + DVFS)
  DistServe   = pd_hetero_baseline (P=4xTP2, D=4xTP2)  <-- modified: decode 4xTP2
  BiScale     = pd_hetero_tier (same topology + DVFS)
  MegaScale   = pdaf_baseline (M=1)
  AFlex       = pdaf_tier (M=1 + V1 compositional DVFS)

Differences from the node1+node2 version:
  - Runs on node3 (10.252.129.34) + node4 (10.252.129.33)
  - pd_hetero decode side: 4xTP2 instead of 2xTP4
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("conv_slo5000_300_node34")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")  # node3 as prefill
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")  # node4 as decode

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_macro_benchmark as RMB

# Relax SLO globally.
RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

# Override pd_hetero to use 4xTP2 decode instead of 2xTP4.
_orig_start_pd_hetero = RMB.start_pd_hetero


def start_pd_hetero_4x2(ngpu, tier=False):
    """Homogeneous PD: P-node 4xTP2 prefill + D-node 4xTP2 decode.

    16-card layout:
        node3 (Prefill):  4 TP2 instances [0,1] [2,3] [4,5] [6,7]
        node4 (Decode):   4 TP2 instances [0,1] [2,3] [4,5] [6,7]
    """
    log.info("PD Hetero(4x2) %d-card: P=4xTP2@node3, D=4xTP2@node4, tier=%s", ngpu, tier)
    p_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    d_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    p_insts = []
    for i, gpus in enumerate(p_groups):
        port = 53100 + i * 10
        bs_port = 49100 + i
        p_insts.append({"gpus": gpus, "port": port, "bs_port": bs_port, "idx": i})
    d_insts = []
    for i, gpus in enumerate(d_groups):
        port = 53150 + i * 10
        d_insts.append({"gpus": gpus, "port": port, "idx": i})

    # Launch prefill instances on node3 (NODE1_IP)
    for inst in p_insts:
        nic = RMB.GPU_NIC[inst["gpus"][0]]
        extra = ("--disaggregation-mode prefill "
                 "--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {inst['bs_port']} "
                 f"--disaggregation-ib-device {nic}")
        cmd = (f"{RMB._plain_env_multi(inst['gpus'])} setsid prlimit "
               f"--memlock=unlimited:unlimited {RMB.PYTHON} -m sglang.launch_server "
               f"--model-path {RMB.MODEL} --tp 2 --host {RMB.NODE1_IP} "
               f"--port {inst['port']} --nccl-port {34000 + inst['idx'] * 10} "
               f"{RMB.COMMON_BENCH_SERVER_FLAGS}"
               f"{extra}{RMB._dvfs_flags(tier)} "
               f"> {RMB.LOG_C}/pd_het_p{inst['idx']}.log 2>&1 < /dev/null &")
        RMB.dexec_local(cmd)
        time.sleep(3)

    # Launch decode instances on node4 (NODE2_IP) — 4xTP2
    for inst in d_insts:
        nic = RMB.GPU_NIC[inst["gpus"][0]]
        extra = ("--disaggregation-mode decode "
                 "--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {p_insts[0]['bs_port']} "
                 f"--disaggregation-ib-device {nic}")
        cmd = (f"{RMB._plain_env_multi(inst['gpus'])} setsid prlimit "
               f"--memlock=unlimited:unlimited {RMB.PYTHON} -m sglang.launch_server "
               f"--model-path {RMB.MODEL} --tp 2 --host {RMB.NODE2_IP} "
               f"--port {inst['port']} --nccl-port {34050 + inst['idx'] * 10} "
               f"{RMB.COMMON_BENCH_SERVER_FLAGS}"
               f"{extra}{RMB._dvfs_flags(tier)} "
               f"> {RMB.LOG_C}/pd_het_d{inst['idx']}.log 2>&1 < /dev/null &")
        RMB.dexec_remote(cmd)
        time.sleep(3)

    # Wait for all to be healthy
    for inst in p_insts:
        if not RMB.wait_health(RMB.NODE1_IP, inst["port"], 500):
            log.error("  PD-Het P%d (%s:%d) failed", inst["idx"], RMB.NODE1_IP, inst["port"])
            return None
    for inst in d_insts:
        if not RMB.wait_health(RMB.NODE2_IP, inst["port"], 500):
            log.error("  PD-Het D%d (%s:%d) failed", inst["idx"], RMB.NODE2_IP, inst["port"])
            return None
    log.info("  all PD-Het instances ready (4P-TP2 + 4D-TP2)")

    # Router
    rc_parts = [f"setsid {RMB.PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT}"]
    for inst in p_insts:
        rc_parts.append(f"--prefill http://{RMB.NODE1_IP}:{inst['port']} {inst['bs_port']}")
    for inst in d_insts:
        rc_parts.append(f"--decode http://{RMB.NODE2_IP}:{inst['port']}")
    rc = " ".join(rc_parts) + f" > {RMB.LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        log.error("  PD-Het router failed")
        return None
    log.info("  router ready")
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


# Monkey-patch pd_hetero in SCHEMES dict.
RMB.SCHEMES["pd_hetero"] = start_pd_hetero_4x2

# Force PDAF M=1 and use the validated V1 compositional DVFS path for AFlex.
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

SCHEMES_ORDER = [
    ("native_tp", "baseline"),
    ("native_tp", "tier"),
    ("pd_hetero", "baseline"),
    ("pd_hetero", "tier"),
    ("pdaf", "baseline"),
    ("pdaf", "tier"),
]
QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    gpus = RMB.card_gpus(NGPU)
    all_results = {}

    log.info("=" * 72)
    log.info("CONV-ONLY 6-SCHEME BENCHMARK (node3+node4) | SLO TTFT=5000ms TPOT=300ms")
    log.info("Nodes: node3=%s, node4=%s | 16-card | PDAF M=1", RMB.NODE1_IP, RMB.NODE2_IP)
    log.info("pd_hetero decode: 4xTP2 (not 2xTP4)")
    log.info("=" * 72)

    for scheme, mode in SCHEMES_ORDER:
        tier = mode == "tier"
        full_name = f"{scheme}_{mode}"
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s", full_name)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = RMB.SCHEMES[scheme](NGPU, tier)
        if url is None:
            log.error("%s deployment FAILED", full_name)
            RMB.cleanup_all()
            all_results[full_name] = {"__status__": "DEPLOY_FAILED"}
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        log.info("Warmup...")
        if not RMB.test_generate(url):
            log.error("%s warmup FAILED", full_name)
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            all_results[full_name] = {"__status__": "WARMUP_FAILED"}
            continue
        time.sleep(3)

        deploy_results = {}
        for qps in QPS_LIST:
            log.info("-" * 50)
            res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        all_results[full_name] = deploy_results

        ts = time.strftime("%Y%m%d_%H%M%S")
        out_file = RESULTS_DIR / f"conv_6scheme_slo5000_300_node34_{ts}.json"
        payload = {
            "meta": {
                "node3": RMB.NODE1_IP,
                "node4": RMB.NODE2_IP,
                "model": RMB.MODEL,
                "ngpu_total": NGPU,
                "gpus_per_node": gpus,
                "dataset": DATASET,
                "qps": QPS_LIST,
                "schemes": [f"{s}_{m}" for s, m in SCHEMES_ORDER],
                "ttft_slo_ms": RMB.TTFT_SLO_MS,
                "tpot_slo_ms": RMB.TPOT_SLO_MS,
                "pdaf_micro_batch": 1,
                "pd_hetero_decode": "4xTP2",
            },
            "results": all_results,
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Results saved: %s", out_file)

    print("\n" + "=" * 104)
    print("  CONV 6-SCHEME RESULTS (node3+node4) | SLO TTFT=5000ms TPOT=300ms | pd_hetero D=4xTP2")
    print("=" * 104)
    print(f"{'Deploy':<22} {'Workload':<14} {'Thpt':>8} {'TTFT':>8} {'TPOT':>8} {'E_tot':>9} {'mJ/tok':>8} {'SLO%':>6}")
    print("-" * 104)
    for dep, wl in all_results.items():
        if "__status__" in wl:
            print(f"{dep:<22} {wl['__status__']}")
            continue
        for w, m in wl.items():
            if m.get("status") != "PASS":
                print(f"{dep:<22} {w:<14} FAIL")
                continue
            print(f"{dep:<22} {w:<14} {m['throughput_tok_s']:>8.1f} "
                  f"{m['ttft_proc_avg_ms']:>8.1f} {m['tpot_avg_ms']:>8.1f} "
                  f"{m['total_energy_j']:>9.0f} {m['energy_per_token_mj']:>8.1f} "
                  f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
