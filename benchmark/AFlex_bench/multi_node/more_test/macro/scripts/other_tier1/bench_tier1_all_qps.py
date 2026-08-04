#!/usr/bin/env python3
"""Benchmark Tier1-optimal configs across QPS=2/4/6/8/12/16, code dataset, node3+node4.

For each QPS, uses the Tier1SolverV2-derived optimal configuration.
Captures freq timeline + energy metrics for each run.
"""
from __future__ import annotations

import argparse, asyncio, json, logging, os, sys, time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench_tier1_allqps")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import run_macro_benchmark as RMB
from freq_timeline_utils import FreqTimelineSession, ensure_container_log_dir
from run_fixed_6scheme_7dataset import _wl_key, _workload_file, MAX_RUN_S as F67_MAX_RUN_S

# Override to node3+node4
RMB.NODE1_IP = os.environ.get("MN_NODE3_IP", "10.252.129.34")
RMB.NODE2_IP = os.environ.get("MN_NODE4_IP", "10.252.129.33")
RMB.TTFT_SLO_MS = 2000.0
RMB.TPOT_SLO_MS = 100.0

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GPUS_PER_NODE = 8
SUB_ROUTER_BASE = 45000
DECODE_PORT_BASE = 43020
PREFILL_PORT_BASE = 43200
NCCL_PORT_BASE = 37300
DATASET = "code"


@dataclass
class Tier1TestConfig:
    name: str
    qps: int
    k_p: int; k_d: int
    tp_pa: int; tp_pf: int
    tp_da: int; tp_df: int
    f_pa: int; f_pf: int
    f_da: int; f_df: int
    tier: bool = True


# Solver-derived optimal configs for code dataset, SLO=(2000ms, 100ms)
CONFIGS = [
    Tier1TestConfig("tier1_q2", qps=2, k_p=2, k_d=1, tp_pa=2, tp_pf=2, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
    Tier1TestConfig("tier1_q4", qps=4, k_p=4, k_d=1, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
    Tier1TestConfig("tier1_q6", qps=6, k_p=4, k_d=1, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
    Tier1TestConfig("tier1_q8", qps=8, k_p=4, k_d=1, tp_pa=2, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=1170, f_da=930, f_df=930),
    Tier1TestConfig("tier1_q12", qps=12, k_p=4, k_d=1, tp_pa=2, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=1170, f_da=930, f_df=930),
    Tier1TestConfig("tier1_q16", qps=16, k_p=4, k_d=1, tp_pa=2, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=1170, f_da=930, f_df=930),
]


# ── Reuse deploy logic from bench_tier1_v2.py ──
import bench_tier1_v2 as BT2
from bench_tier1_v2 import (
    plan_allocation, deploy, _run_workload_rr, run_benchmark,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps-list", default="2,4,6,8,12,16")
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()

    qps_targets = [int(x) for x in args.qps_list.split(",")]
    targets = [c for c in CONFIGS if c.qps in qps_targets and c.name not in args.skip]

    all_results = {}
    for cfg in targets:
        gpu_count = cfg.k_p * (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)
        log.info("\n" + "#" * 72)
        log.info("TESTING: %s (QPS=%d, %dGPU)", cfg.name, cfg.qps, gpu_count)
        log.info("#" * 72)

        # Update global QPS in bench_tier1_v2 module
        BT2.QPS = cfg.qps

        RMB.cleanup_all()
        time.sleep(8)

        url = deploy(cfg)
        if url is None:
            all_results[cfg.name] = {"status": "DEPLOY_FAILED"}
            continue
        if any(not RMB.test_generate(u) for u in url):
            all_results[cfg.name] = {"status": "WARMUP_FAILED"}
            RMB.cleanup_all()
            continue

        time.sleep(3)
        summary = run_benchmark(url, cfg)
        all_results[cfg.name] = summary

        if isinstance(summary, dict) and summary.get("status") == "PASS":
            log.info("PASS: QPS=%d GPU=%d thpt=%.1f TTFT=%.1fms TPOT=%.1fms E/tok=%.1fmJ SLO=%.1f%%",
                     cfg.qps, gpu_count,
                     summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                     summary["tpot_avg_ms"], summary.get("energy_per_token_mj", 0),
                     summary["slo_violation_rate"])
        else:
            log.error("FAIL: %s -> %s", cfg.name, summary)

        RMB.cleanup_all()
        time.sleep(5)

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"tier1_allqps_{ts}.json"
    meta = {
        "benchmark": "tier1_all_qps_sweep",
        "dataset": DATASET,
        "ttft_slo_ms": RMB.TTFT_SLO_MS,
        "tpot_slo_ms": RMB.TPOT_SLO_MS,
        "n3": RMB.NODE1_IP, "n4": RMB.NODE2_IP,
    }
    out.write_text(json.dumps({"meta": meta, "results": all_results}, indent=2))
    log.info("Saved %s", out)

    print(f"\n{'='*72}")
    print("TIER1 ALL-QPS RESULTS (code dataset)")
    print(f"{'='*72}")
    for cfg in targets:
        r = all_results.get(cfg.name, {})
        gpu_count = cfg.k_p * (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)
        if r.get("status") == "PASS":
            print(f"  QPS={cfg.qps:2d} [{cfg.name}] GPU={gpu_count:2d}: "
                  f"thpt={r['throughput_tok_s']:.1f} TTFT={r['ttft_proc_avg_ms']:.1f}ms "
                  f"TPOT={r['tpot_avg_ms']:.1f}ms E/tok={r.get('energy_per_token_mj',0):.1f}mJ "
                  f"SLO={r['slo_violation_rate']:.1f}%")
        else:
            print(f"  QPS={cfg.qps:2d} [{cfg.name}] GPU={gpu_count:2d}: {r.get('status', 'UNKNOWN')}")


if __name__ == "__main__":
    main()
