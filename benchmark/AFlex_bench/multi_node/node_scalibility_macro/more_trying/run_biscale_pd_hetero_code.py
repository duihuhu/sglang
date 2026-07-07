#!/usr/bin/env python3
"""BiScale (paper DVFS) on heterogeneous PD: P=4×TP2 + D=2×TP4, code dataset.

Topology (16-card, node3 prefill + node4 decode):
  Prefill:  4 × TP2 on node1  [0,1] [2,3] [4,5] [6,7]
  Decode:   2 × TP4 on node2  [0,1,2,3] [4,5,6,7]

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
log = logging.getLogger("biscale_pd_hetero_code")

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
LOG_DIR = Path(RMB.LOG_C)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

SCHEME_KEY = "pd_hetero_tier_biscale"
MACRO_WL_DIR = MACRO_DIR / "workloads"


def _wl_key(qps: int) -> str:
    return f"{DATASET}_qps{qps}"


def _workload_file(qps: int) -> Path | None:
    path = MACRO_WL_DIR / f"macro_{DATASET}_qps{qps}.jsonl"
    return path if path.exists() else None


_orig_start_pd_hetero = RMB.start_pd_hetero


def start_pd_hetero_biscale(ngpu: int, tier: bool = True):
    """P=4×TP2 + D=2×TP4 with paper BiScale DVFS + decision logs."""
    log.info("PD Hetero biscale: P=4×TP2@node1, D=2×TP4@node2")
    p_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
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
        dvfs_log = (f"{RMB.DVFS_LOG_DIR}/biscale_het_p{inst['idx']}_"
                    f"gpu{inst['gpus'][0]}.jsonl")
        cmd = (f"{RMB._plain_env_multi(inst['gpus'], dvfs_log)} setsid prlimit "
               f"--memlock=unlimited:unlimited {RMB.PYTHON} -m sglang.launch_server "
               f"--model-path {RMB.MODEL} --tp 2 --host {RMB.NODE1_IP} "
               f"--port {inst['port']} --nccl-port {34000 + inst['idx'] * 10} "
               f"{RMB.COMMON_BENCH_SERVER_FLAGS}"
               f"{extra}{RMB._pd_dvfs_flags(tier)} "
               f"> {RMB.LOG_C}/pd_het_p{inst['idx']}.log 2>&1 < /dev/null &")
        RMB.dexec_local(cmd)
        time.sleep(3)

    for inst in d_insts:
        nic = RMB.GPU_NIC[inst["gpus"][0]]
        extra = ("--disaggregation-mode decode "
                 "--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {p_insts[0]['bs_port']} "
                 f"--disaggregation-ib-device {nic}")
        dvfs_log = (f"{RMB.DVFS_LOG_DIR}/biscale_het_d{inst['idx']}_"
                    f"gpu{inst['gpus'][0]}.jsonl")
        cmd = (f"{RMB._plain_env_multi(inst['gpus'], dvfs_log)} setsid prlimit "
               f"--memlock=unlimited:unlimited {RMB.PYTHON} -m sglang.launch_server "
               f"--model-path {RMB.MODEL} --tp 4 --host {RMB.NODE2_IP} "
               f"--port {inst['port']} --nccl-port {34050 + inst['idx'] * 10} "
               f"{RMB.COMMON_BENCH_SERVER_FLAGS}"
               f"{extra}{RMB._pd_dvfs_flags(tier)} "
               f"> {RMB.LOG_C}/pd_het_d{inst['idx']}.log 2>&1 < /dev/null &")
        RMB.dexec_remote(cmd)
        time.sleep(3)

    for inst in p_insts:
        if not RMB.wait_health(RMB.NODE1_IP, inst["port"], 500):
            log.error("PD-Het P%d failed", inst["idx"])
            return None
    for inst in d_insts:
        if not RMB.wait_health(RMB.NODE2_IP, inst["port"], 500):
            log.error("PD-Het D%d failed", inst["idx"])
            return None

    rc_parts = [f"setsid {RMB.PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT}"]
    for inst in p_insts:
        rc_parts.append(f"--prefill http://{RMB.NODE1_IP}:{inst['port']} {inst['bs_port']}")
    for inst in d_insts:
        rc_parts.append(f"--decode http://{RMB.NODE2_IP}:{inst['port']}")
    rc = " ".join(rc_parts) + f" > {RMB.LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        return None
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


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
    out = RESULTS_DIR / f"biscale_pd_hetero_code_{tag}_{ts}.json"
    payload = {
        "meta": {
            "scheme": SCHEME_KEY,
            "label": "BiScale (paper, P=4×TP2+D=2×TP4)",
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": DATASET,
            "qps": QPS_LIST,
            "topology": "P=4×TP2 prefill @node1, D=2×TP4 decode @node2",
            "dvfs_policy": "biscale",
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
    log.info("BiScale PD-HETERO CODE | P=4×TP2 + D=2×TP4 | dvfs-policy=biscale")
    log.info("Nodes: %s (prefill) + %s (decode)", RMB.NODE1_IP, RMB.NODE2_IP)
    log.info("=" * 72)

    RMB.cleanup_all()
    url = start_pd_hetero_biscale(NGPU, tier=True)
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
