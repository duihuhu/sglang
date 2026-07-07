#!/usr/bin/env python3
"""Conv test: DistServe/BiScale with P=3xTP2, D=2xTP4 layout.

Layout:
  node1 (Prefill): 3 x TP2 instances on GPU [0,1] [2,3] [4,5]
  node2 (Decode):  2 x TP4 instances on GPU [0,1,2,3] [4,5,6,7]

SLO: TTFT=5000ms, TPOT=300ms
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
log = logging.getLogger("conv_3p2d")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_macro_benchmark as RMB

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def start_pd_3p2d(tier=False):
    """PD with P=3xTP2 on node1, D=2xTP4 on node2."""
    log.info("PD 3P2D: P=3xTP2@node1[0:5], D=2xTP4@node2[0:7], tier=%s", tier)
    p_groups = [[0, 1], [2, 3], [4, 5]]
    d_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]

    p_insts = []
    for i, gpus in enumerate(p_groups):
        port = 53100 + i * 10
        bs_port = 49100 + i
        p_insts.append({"gpus": gpus, "port": port, "bs_port": bs_port, "idx": i})

    d_insts = []
    for i, gpus in enumerate(d_groups):
        port = 53150 + i * 10
        d_insts.append({"gpus": gpus, "port": port, "idx": i})

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
               f"> {RMB.LOG_C}/pd_3p2d_p{inst['idx']}.log 2>&1 < /dev/null &")
        RMB.dexec_local(cmd)
        time.sleep(3)

    for inst in d_insts:
        nic = RMB.GPU_NIC[inst["gpus"][0]]
        extra = ("--disaggregation-mode decode "
                 "--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {p_insts[0]['bs_port']} "
                 f"--disaggregation-ib-device {nic}")
        cmd = (f"{RMB._plain_env_multi(inst['gpus'])} setsid prlimit "
               f"--memlock=unlimited:unlimited {RMB.PYTHON} -m sglang.launch_server "
               f"--model-path {RMB.MODEL} --tp 4 --host {RMB.NODE2_IP} "
               f"--port {inst['port']} --nccl-port {34050 + inst['idx'] * 10} "
               f"{RMB.COMMON_BENCH_SERVER_FLAGS}"
               f"{extra}{RMB._dvfs_flags(tier)} "
               f"> {RMB.LOG_C}/pd_3p2d_d{inst['idx']}.log 2>&1 < /dev/null &")
        RMB.dexec_remote(cmd)
        time.sleep(3)

    for inst in p_insts:
        if not RMB.wait_health(RMB.NODE1_IP, inst["port"], 500):
            log.error("  P%d failed", inst["idx"])
            return None
    for inst in d_insts:
        if not RMB.wait_health(RMB.NODE2_IP, inst["port"], 500):
            log.error("  D%d failed", inst["idx"])
            return None
    log.info("  all instances ready (3P + 2D)")

    rc_parts = [f"setsid {RMB.PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT}"]
    for inst in p_insts:
        rc_parts.append(f"--prefill http://{RMB.NODE1_IP}:{inst['port']} {inst['bs_port']}")
    for inst in d_insts:
        rc_parts.append(f"--decode http://{RMB.NODE2_IP}:{inst['port']}")
    rc = " ".join(rc_parts) + f" > {RMB.LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        log.error("  router failed")
        return None
    log.info("  router ready")
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    gpus = RMB.card_gpus(NGPU)
    all_results = {}

    log.info("=" * 72)
    log.info("CONV P=3xTP2 D=2xTP4 | SLO TTFT=5000ms TPOT=300ms")
    log.info("=" * 72)

    for mode in ["baseline", "tier"]:
        tier = mode == "tier"
        label = "DistServe_3P2D" if not tier else "BiScale_3P2D"
        full_name = f"pd_3p2d_{mode}"
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s (%s)", label, full_name)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = start_pd_3p2d(tier)
        if url is None:
            log.error("%s deployment FAILED", label)
            RMB.cleanup_all()
            all_results[full_name] = {"__status__": "DEPLOY_FAILED"}
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        log.info("Warmup...")
        if not RMB.test_generate(url):
            log.error("%s warmup FAILED", label)
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
        out_file = RESULTS_DIR / f"conv_pd_3p2d_{ts}.json"
        payload = {
            "meta": {
                "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP,
                "model": RMB.MODEL, "ngpu_total": NGPU,
                "layout": "P=3xTP2@node1, D=2xTP4@node2",
                "dataset": DATASET, "qps": QPS_LIST,
                "ttft_slo_ms": RMB.TTFT_SLO_MS,
                "tpot_slo_ms": RMB.TPOT_SLO_MS,
            },
            "results": all_results,
        }
        with open(out_file, "w") as f:
            json.dump(payload, f, indent=2)
        log.info("Results saved: %s", out_file)

    print("\n" + "=" * 104)
    print("  CONV P=3xTP2 D=2xTP4 | SLO TTFT=5000ms TPOT=300ms")
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
