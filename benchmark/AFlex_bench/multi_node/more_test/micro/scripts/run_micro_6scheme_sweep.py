#!/usr/bin/env python3
"""Run 6 schemes x 4 micro datasets (QA/Chatbot/RAG/Summary) on node1+node2.

Schemes:
  1. SGLang     - 16xTP1, no DVFS, locked 1410MHz
  2. DynamoLLM  - 16xTP1, DVFS unified
  3. DistServe  - 2P(TP4)+4D(TP2), no DVFS, locked 1410MHz
  4. BiScale    - 2P(TP4)+4D(TP2), DVFS BiScale
  5. MegaScale  - AFD solver max-throughput (16GPU, locked 1410MHz)
  6. AFlex      - AFD solver min-energy (per-component DVFS)

Micro datasets (fixed-length):
  qa_lpld:       il=128,  ol=64   (Low Prefill, Low Decode)
  chatbot_lphd:  il=128,  ol=1024 (Low Prefill, High Decode)
  rag_hpld:      il=4096, ol=64   (High Prefill, Low Decode)
  summary_hphd:  il=4096, ol=1024 (High Prefill, High Decode)

Usage (on node1 HOST, NOT inside container):
  python3 run_micro_6scheme_sweep.py --datasets all --qps-list 2,4,8,16
  python3 run_micro_6scheme_sweep.py --datasets qa_lpld --schemes sglang,aflex --qps-list 2,4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("micro_6scheme")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent.parent / "more_test/macro/scripts"
sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(MACRO_DIR / "more_trying"))
sys.path.insert(0, str(MACRO_DIR / "more_trying" / "other_tier1"))

import run_macro_benchmark as RMB

# Use node1 + node2
RMB.NODE1_IP = os.environ.get("MN_NODE1_IP", "10.252.129.36")
RMB.NODE2_IP = os.environ.get("MN_NODE2_IP", "10.252.129.35")
RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

NODE1 = RMB.NODE1_IP
NODE2 = RMB.NODE2_IP
MODEL = RMB.MODEL
PYTHON = RMB.PYTHON
LOG_C = RMB.LOG_C
FLAGS = RMB.COMMON_BENCH_SERVER_FLAGS
GPU_NIC = RMB.GPU_NIC
ROUTER_PORT = RMB.ROUTER_PORT

ENERGY_MODEL_DIR_V1 = (
    "/workspace/sglang/benchmark/AFlex_bench/energy_model/"
    "Qwen3-32B/models_v1"
)

WORKLOAD_DIR = HERE.parent / "exp-all" / "workloads"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ALL_GPUS = list(range(8))
MAX_FREQ = 1410

MICRO_DATASETS = ["qa_lpld", "chatbot_lphd", "rag_hpld", "summary_hphd"]
QPS_DEFAULT = [2, 4, 8, 16]

NATIVE_DVFS_FLAGS = (
    f" --dvfs-enabled "
    f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR_V1} "
    f"--dvfs-ttft-slo-ms 5000 "
    f"--dvfs-tpot-slo-us 300000"
)

BISCALE_DVFS_FLAGS = (
    f" --dvfs-enabled "
    f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR_V1} "
    f"--dvfs-ttft-slo-ms 5000 "
    f"--dvfs-tpot-slo-us 300000 "
    f"--dvfs-policy biscale"
)


# ============================================================
# Scheme 1 & 2: SGLang / DynamoLLM (16 x TP1)
# ============================================================
def deploy_native_tp1(scheme: str):
    is_dynamo = scheme == "dynamollm"
    log.info("=" * 60)
    log.info("Deploying %s: 16xTP1 (8/node), round-robin", scheme.upper())
    log.info("=" * 60)

    RMB.cleanup_all()
    time.sleep(8)

    if is_dynamo:
        RMB.unlock_freq_both(ALL_GPUS)
    else:
        RMB.lock_freq_both(ALL_GPUS, MAX_FREQ)

    insts = []
    idx = 0
    for host in (NODE1, NODE2):
        for gpu in ALL_GPUS:
            port = 53200 + idx * 10
            insts.append({"host": host, "gpu": gpu, "port": port, "idx": idx})
            idx += 1

    for inst in insts:
        gpu = inst["gpu"]
        host = inst["host"]
        port = inst["port"]
        i = inst["idx"]
        dvfs_flags = NATIVE_DVFS_FLAGS if is_dynamo else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={gpu} AFD_NVML_DEVICE_INDEX={gpu} "
            f"AFD_NVML_DEVICE_INDICES={gpu}; "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp 1 --host {host} --port {port} "
            f"--nccl-port {33300 + i * 10} {FLAGS}{dvfs_flags} "
            f"> {LOG_C}/{scheme}_g{gpu}_n{'1' if host == NODE1 else '2'}.log 2>&1 < /dev/null &"
        )
        if host == NODE1:
            RMB.dexec_local(cmd)
        else:
            RMB.dexec_remote(cmd)
        if i % 4 == 3:
            time.sleep(2)

    log.info("  Waiting for all 16 instances...")
    time.sleep(10)
    for inst in insts:
        if not RMB.wait_health(inst["host"], inst["port"], 400):
            log.error("  Instance %s:%d FAILED!", inst["host"], inst["port"])
            return None
    log.info("  All 16 instances healthy!")

    worker_urls = " ".join(f"http://{inst['host']}:{inst['port']}" for inst in insts)
    rc = (
        f"setsid {PYTHON} -m sglang_router.launch_router "
        f"--host {NODE1} --port {ROUTER_PORT} --policy round_robin "
        f"--worker-urls {worker_urls} > {LOG_C}/router.log 2>&1 < /dev/null &"
    )
    RMB.dexec_local(rc)
    if not RMB.wait_health(NODE1, ROUTER_PORT, 60):
        log.error("  Router FAILED!")
        return None
    url = f"http://{NODE1}:{ROUTER_PORT}"
    if not RMB.test_generate(url):
        log.error("  Warmup FAILED!")
        return None
    log.info("  Deploy OK: %s", url)
    return url


# ============================================================
# Scheme 3 & 4: DistServe / BiScale (2P(TP4)+4D(TP2))
# ============================================================
P_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]
D_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]


def deploy_pd_hetero(scheme: str):
    is_biscale = scheme == "biscale"
    log.info("=" * 60)
    log.info("Deploying %s: 2P(TP4)+4D(TP2)=16GPU", scheme.upper())
    log.info("=" * 60)

    RMB.cleanup_all()
    time.sleep(8)

    if is_biscale:
        RMB.unlock_freq_both(ALL_GPUS)
    else:
        RMB.lock_freq_both(ALL_GPUS, MAX_FREQ)

    p_insts = []
    d_insts = []

    for i, gpus in enumerate(P_GROUPS):
        p_port = 53100 + i * 10
        bs_port = 49100 + i
        nic = GPU_NIC[gpus[0]]
        csv = ",".join(str(g) for g in gpus)
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={NODE1}; "
        )
        extra = (
            f"--disaggregation-mode prefill "
            f"--disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {bs_port} "
            f"--disaggregation-ib-device {nic} "
        )
        dvfs_part = BISCALE_DVFS_FLAGS if is_biscale else ""
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp 4 --host {NODE1} --port {p_port} "
            f"--nccl-port {34000 + i * 10} {FLAGS}{extra}{dvfs_part} "
            f"> {LOG_C}/{scheme}_p{i}.log 2>&1 < /dev/null &"
        )
        RMB.dexec_local(cmd)
        p_insts.append({"port": p_port, "bs_port": bs_port, "gpus": gpus, "idx": i})
        time.sleep(3)

    for i, gpus in enumerate(D_GROUPS):
        d_port = 53150 + i * 10
        nic = GPU_NIC[gpus[0]]
        csv = ",".join(str(g) for g in gpus)
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={NODE2}; "
        )
        extra = (
            f"--disaggregation-mode decode "
            f"--disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {p_insts[0]['bs_port']} "
            f"--disaggregation-ib-device {nic} "
        )
        dvfs_part = BISCALE_DVFS_FLAGS if is_biscale else ""
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp 2 --host {NODE2} --port {d_port} "
            f"--nccl-port {34050 + i * 10} {FLAGS}{extra}{dvfs_part} "
            f"> {LOG_C}/{scheme}_d{i}.log 2>&1 < /dev/null &"
        )
        RMB.dexec_remote(cmd)
        d_insts.append({"port": d_port, "gpus": gpus, "idx": i})
        time.sleep(3)

    log.info("  Waiting for servers...")
    for inst in p_insts:
        if not RMB.wait_health(NODE1, inst["port"], 400):
            log.error("  P%d FAILED!", inst["idx"])
            return None
    for inst in d_insts:
        if not RMB.wait_health(NODE2, inst["port"], 400):
            log.error("  D%d FAILED!", inst["idx"])
            return None
    log.info("  All P/D servers healthy!")

    rc_parts = [
        f"setsid {PYTHON} -m sglang_router.launch_router",
        "--pd-disaggregation",
        f"--host {NODE1} --port {ROUTER_PORT}",
    ]
    for inst in p_insts:
        rc_parts.append(f"--prefill http://{NODE1}:{inst['port']} {inst['bs_port']}")
    for inst in d_insts:
        rc_parts.append(f"--decode http://{NODE2}:{inst['port']}")
    rc = " ".join(rc_parts) + f" > {LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)

    if not RMB.wait_health(NODE1, ROUTER_PORT, 60):
        log.error("  Router FAILED!")
        return None
    url = f"http://{NODE1}:{ROUTER_PORT}"
    if not RMB.test_generate(url):
        log.error("  Warmup FAILED!")
        return None
    log.info("  Deploy OK: %s", url)
    return url


# ============================================================
# Scheme 5 & 6: MegaScale / AFlex (AFD with Tier1 Solver configs)
# ============================================================
@dataclass
class AFDConfig:
    name: str
    qps: int
    k_p: int
    k_d: int
    tp_pa: int
    tp_pf: int
    tp_da: int
    tp_df: int
    f_pa: int
    f_pf: int
    f_da: int
    f_df: int
    tier: bool = True


import bench_tier1_v2 as BT2
from bench_tier1_v2 import deploy as deploy_afd, run_benchmark as run_benchmark_afd

# bench_tier1_v2 overrides RMB.NODE1_IP to node3/4 on import; restore node1/2
RMB.NODE1_IP = NODE1
RMB.NODE2_IP = NODE2
RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0


def _get_aflex_configs(dataset: str, qps_list: list[int]) -> list[AFDConfig]:
    """Load AFlex configs from solver results."""
    sol_file = (
        MACRO_DIR
        / "more_trying"
        / "results"
        / f"tier1_{dataset}_solutions.json"
    )
    if sol_file.exists():
        sols = json.loads(sol_file.read_text())
        configs = []
        for sol in sols:
            q = sol["qps"]
            if q not in qps_list:
                continue
            s = sol["solution"]
            configs.append(
                AFDConfig(
                    name=f"aflex_{dataset}_q{q}",
                    qps=q,
                    k_p=s["k_p"],
                    k_d=s["k_d"],
                    tp_pa=s["pa"][0],
                    tp_pf=s["pf"][0],
                    tp_da=s["da"][0],
                    tp_df=s["df"][0],
                    f_pa=s["pa"][1],
                    f_pf=s["pf"][1],
                    f_da=s["da"][1],
                    f_df=s["df"][1],
                    tier=True,
                )
            )
        return configs
    # Fallback: use conv-like config (1P(TP4+4)+1D(TP4+4), DVFS)
    log.warning("No solver result for %s, using conv-like fallback", dataset)
    return [
        AFDConfig(
            f"aflex_{dataset}_q{q}",
            qps=q,
            k_p=1, k_d=1,
            tp_pa=4, tp_pf=4, tp_da=4, tp_df=4,
            f_pa=930, f_pf=930, f_da=930, f_df=930,
            tier=True,
        )
        for q in qps_list
    ]


def _get_megascale_configs(dataset: str, qps_list: list[int]) -> list[AFDConfig]:
    """MegaScale: use conv-derived solver configs (all 16GPU, locked 1410MHz).

    For micro datasets we reuse the conv topology since the solver's
    solve_max_throughput gives the same layout for short/long inputs
    (it's throughput-driven, not SLO-driven).
    """
    # Conv MegaScale configs from test.md
    mega_topo = {
        2: (1, 1, 4, 4, 4, 4),
        4: (2, 1, 2, 2, 4, 4),
        8: (4, 1, 1, 1, 4, 4),
        16: (6, 1, 1, 1, 2, 2),
    }
    configs = []
    for q in qps_list:
        if q in mega_topo:
            kp, kd, tpa, tpf, tda, tdf = mega_topo[q]
        else:
            kp, kd, tpa, tpf, tda, tdf = 1, 1, 4, 4, 4, 4
        configs.append(
            AFDConfig(
                name=f"mega_{dataset}_q{q}",
                qps=q,
                k_p=kp, k_d=kd,
                tp_pa=tpa, tp_pf=tpf,
                tp_da=tda, tp_df=tdf,
                f_pa=MAX_FREQ, f_pf=MAX_FREQ,
                f_da=MAX_FREQ, f_df=MAX_FREQ,
                tier=False,
            )
        )
    return configs


def run_afd_scheme(scheme: str, dataset: str, qps_list: list[int]) -> dict:
    """Run MegaScale or AFlex for one dataset across all QPS."""
    if scheme == "megascale":
        configs = _get_megascale_configs(dataset, qps_list)
    else:
        configs = _get_aflex_configs(dataset, qps_list)

    results = {}
    for cfg in configs:
        gpu_count = cfg.k_p * (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)
        log.info("\n" + "#" * 72)
        log.info("%s: %s (QPS=%d, %dGPU)", scheme.upper(), cfg.name, cfg.qps, gpu_count)
        log.info("#" * 72)

        BT2.QPS = cfg.qps
        BT2.DATASET = dataset
        RMB.cleanup_all()
        time.sleep(8)

        url = deploy_afd(cfg)
        if url is None:
            results[cfg.name] = {"status": "DEPLOY_FAILED"}
            continue

        if not cfg.tier:
            RMB.lock_freq_both(ALL_GPUS, MAX_FREQ)
            log.info("  Locked all GPUs to %dMHz", MAX_FREQ)

        if not RMB.test_generate(url[0]):
            results[cfg.name] = {"status": "WARMUP_FAILED"}
            RMB.cleanup_all()
            continue

        time.sleep(3)
        summary = run_benchmark_afd(url, cfg)
        results[cfg.name] = summary

        if isinstance(summary, dict) and summary.get("status") == "PASS":
            log.info(
                "PASS: QPS=%d GPU=%d thpt=%.1f TTFT=%.1fms TPOT=%.1fms E/tok=%.1fmJ",
                cfg.qps, gpu_count,
                summary["throughput_tok_s"],
                summary["ttft_proc_avg_ms"],
                summary["tpot_avg_ms"],
                summary.get("energy_per_token_mj", 0),
            )
        else:
            log.error("FAIL: %s -> %s", cfg.name, summary)

        if not cfg.tier:
            RMB.unlock_freq_both(ALL_GPUS)
        RMB.cleanup_all()
        time.sleep(5)

    return results


# ============================================================
# Workload runner (for non-AFD schemes)
# ============================================================
def run_benchmark_simple(url: str, dataset: str, qps: int) -> dict:
    """Run a micro workload against a simple URL endpoint."""
    wl_file = WORKLOAD_DIR / f"micro_{dataset}_qps{qps}.jsonl"
    if not wl_file.exists():
        log.error("  Workload not found: %s", wl_file)
        return {"status": "NO_WORKLOAD"}

    with open(wl_file) as f:
        reqs = [json.loads(l) for l in f]

    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(400, last_arrival + 150), 900))
    log.info("  %s QPS=%d (%d reqs, run_window=%ds)", dataset, qps, len(reqs), run_s)

    summary = asyncio.run(
        RMB.run_workload(reqs, url + "/generate", ALL_GPUS, ALL_GPUS, run_s)
    )
    if summary.get("status") == "PASS":
        log.info(
            "  PASS: Thpt=%.1f | TTFT=%.1f | TPOT=%.1f | E/tok=%.1fmJ",
            summary["throughput_tok_s"],
            summary["ttft_proc_p50_ms"],
            summary["tpot_p50_ms"],
            summary.get("energy_per_token_mj", 0),
        )
    else:
        log.warning("  FAIL: %s", summary.get("status", "UNKNOWN"))
    return summary


# ============================================================
# Main orchestration
# ============================================================
SCHEME_ORDER = ["sglang", "dynamollm", "distserve", "biscale", "megascale", "aflex"]


def run_simple_scheme(scheme: str, dataset: str, qps_list: list[int]) -> dict:
    """Run SGLang/DynamoLLM/DistServe/BiScale for one dataset across QPS."""
    results = {}
    for qps in qps_list:
        log.info("\n" + "#" * 70)
        log.info("## %s | %s QPS=%d", scheme.upper(), dataset, qps)
        log.info("#" * 70)

        if scheme in ("sglang", "dynamollm"):
            url = deploy_native_tp1(scheme)
        else:
            url = deploy_pd_hetero(scheme)

        if url is None:
            results[f"q{qps}"] = {"status": "DEPLOY_FAILED"}
            continue

        summary = run_benchmark_simple(url, dataset, qps)
        results[f"q{qps}"] = summary

    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets", default="all",
        help="Comma-sep: qa_lpld,chatbot_lphd,rag_hpld,summary_hphd or 'all'",
    )
    parser.add_argument(
        "--schemes", default="all",
        help="Comma-sep: sglang,dynamollm,distserve,biscale,megascale,aflex or 'all'",
    )
    parser.add_argument("--qps-list", default="2,4,8,16")
    parser.add_argument(
        "--skip-redeploy", action="store_true",
        help="For SGLang/DynamoLLM/DistServe/BiScale: deploy once, run all datasets (faster)",
    )
    args = parser.parse_args()

    datasets = MICRO_DATASETS if args.datasets == "all" else args.datasets.split(",")
    schemes = SCHEME_ORDER if args.schemes == "all" else args.schemes.split(",")
    qps_list = [int(x) for x in args.qps_list.split(",")]

    all_results = {}
    start_time = time.time()

    for scheme in schemes:
        if scheme not in SCHEME_ORDER:
            log.error("Unknown scheme: %s", scheme)
            continue

        log.info("\n" + "=" * 72)
        log.info("  SCHEME: %s", scheme.upper())
        log.info("=" * 72)

        if scheme in ("megascale", "aflex"):
            for dataset in datasets:
                key = f"{scheme}_{dataset}"
                log.info("\n>>> %s / %s <<<", scheme, dataset)
                all_results[key] = run_afd_scheme(scheme, dataset, qps_list)
        else:
            if args.skip_redeploy:
                # Deploy once, run all datasets x QPS
                if scheme in ("sglang", "dynamollm"):
                    url = deploy_native_tp1(scheme)
                else:
                    url = deploy_pd_hetero(scheme)
                if url is None:
                    for dataset in datasets:
                        all_results[f"{scheme}_{dataset}"] = {"status": "DEPLOY_FAILED"}
                    continue
                for dataset in datasets:
                    key = f"{scheme}_{dataset}"
                    ds_results = {}
                    for qps in qps_list:
                        log.info("--- %s | %s QPS=%d ---", scheme, dataset, qps)
                        summary = run_benchmark_simple(url, dataset, qps)
                        ds_results[f"q{qps}"] = summary
                        time.sleep(3)
                    all_results[key] = ds_results
                RMB.cleanup_all()
                time.sleep(5)
            else:
                for dataset in datasets:
                    key = f"{scheme}_{dataset}"
                    all_results[key] = run_simple_scheme(scheme, dataset, qps_list)

    # Save results
    elapsed = time.time() - start_time
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"micro_6scheme_4ds_{ts}.json"
    payload = {
        "meta": {
            "benchmark": "micro_6scheme_4datasets",
            "datasets": datasets,
            "schemes": schemes,
            "qps_list": qps_list,
            "node1": NODE1,
            "node2": NODE2,
            "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS,
            "elapsed_s": round(elapsed, 1),
        },
        "results": all_results,
    }
    out_file.write_text(json.dumps(payload, indent=2, default=str))
    log.info("\n" + "=" * 72)
    log.info("ALL DONE! Elapsed: %.1f min", elapsed / 60)
    log.info("Results saved to: %s", out_file)
    log.info("=" * 72)

    # Print summary table
    print(f"\n{'='*80}")
    print("MICRO BENCHMARK RESULTS (6 schemes x 4 datasets)")
    print(f"{'='*80}")
    for scheme in schemes:
        print(f"\n  [{scheme.upper()}]")
        for dataset in datasets:
            key = f"{scheme}_{dataset}"
            res = all_results.get(key, {})
            if not res:
                continue
            print(f"    {dataset}:")
            items = res.items() if isinstance(res, dict) else []
            for k, v in sorted(items):
                if isinstance(v, dict) and v.get("status") == "PASS":
                    print(
                        f"      {k}: thpt={v['throughput_tok_s']:.1f} "
                        f"TTFT={v.get('ttft_proc_avg_ms', v.get('ttft_proc_p50_ms', 0)):.1f}ms "
                        f"TPOT={v.get('tpot_avg_ms', v.get('tpot_p50_ms', 0)):.1f}ms "
                        f"E/tok={v.get('energy_per_token_mj', 0):.1f}mJ"
                    )
                elif isinstance(v, dict):
                    print(f"      {k}: {v.get('status', 'ERROR')}")


if __name__ == "__main__":
    main()
