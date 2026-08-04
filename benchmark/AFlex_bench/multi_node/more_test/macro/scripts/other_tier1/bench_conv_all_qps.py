#!/usr/bin/env python3
"""Benchmark AFlex + MegaScale Tier1-optimal configs for CONV dataset, node3+node4.

For each QPS: solver-derived optimal config (AFlex with DVFS, MegaScale locked 1410MHz).
"""
from __future__ import annotations

import argparse, asyncio, json, logging, os, sys, time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench_conv_allqps")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import run_macro_benchmark as RMB
from freq_timeline_utils import FreqTimelineSession, ensure_container_log_dir
from run_fixed_6scheme_7dataset import _wl_key, _workload_file, MAX_RUN_S as F67_MAX_RUN_S

RMB.NODE1_IP = os.environ.get("MN_NODE3_IP", "10.252.129.34")
RMB.NODE2_IP = os.environ.get("MN_NODE4_IP", "10.252.129.33")
RMB.TTFT_SLO_MS = 2000.0
RMB.TPOT_SLO_MS = 100.0

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET = "conv"
MAX_FREQ = 1410


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


# Solver-V2-derived optimal configs for conv dataset, SLO=(2000ms, 100ms)
AFLEX_CONFIGS = [
    Tier1TestConfig("aflex_q2", qps=2, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
    Tier1TestConfig("aflex_q4", qps=4, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
    Tier1TestConfig("aflex_q6", qps=6, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
    Tier1TestConfig("aflex_q8", qps=8, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=2, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
    Tier1TestConfig("aflex_q12", qps=12, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=2, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
    Tier1TestConfig("aflex_q16", qps=16, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=4, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930),
]

# MegaScale: same topology, locked max freq, no DVFS
MEGA_CONFIGS = [
    Tier1TestConfig("mega_q2", qps=2, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=1, tp_df=1,
                    f_pa=MAX_FREQ, f_pf=MAX_FREQ, f_da=MAX_FREQ, f_df=MAX_FREQ, tier=False),
    Tier1TestConfig("mega_q4", qps=4, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=1, tp_df=1,
                    f_pa=MAX_FREQ, f_pf=MAX_FREQ, f_da=MAX_FREQ, f_df=MAX_FREQ, tier=False),
    Tier1TestConfig("mega_q6", qps=6, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=1, tp_df=1,
                    f_pa=MAX_FREQ, f_pf=MAX_FREQ, f_da=MAX_FREQ, f_df=MAX_FREQ, tier=False),
    Tier1TestConfig("mega_q8", qps=8, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=2, tp_df=1,
                    f_pa=MAX_FREQ, f_pf=MAX_FREQ, f_da=MAX_FREQ, f_df=MAX_FREQ, tier=False),
    Tier1TestConfig("mega_q12", qps=12, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=2, tp_df=1,
                    f_pa=MAX_FREQ, f_pf=MAX_FREQ, f_da=MAX_FREQ, f_df=MAX_FREQ, tier=False),
    Tier1TestConfig("mega_q16", qps=16, k_p=1, k_d=1, tp_pa=4, tp_pf=4, tp_da=4, tp_df=1,
                    f_pa=MAX_FREQ, f_pf=MAX_FREQ, f_da=MAX_FREQ, f_df=MAX_FREQ, tier=False),
]


import bench_tier1_v2 as BT2
from bench_tier1_v2 import deploy, run_benchmark


def run_configs(configs, scheme_name):
    all_results = {}
    for cfg in configs:
        gpu_count = cfg.k_p * (cfg.tp_pa + cfg.tp_pf) + cfg.k_d * (cfg.tp_da + cfg.tp_df)
        log.info("\n" + "#" * 72)
        log.info("%s: %s (QPS=%d, %dGPU)", scheme_name, cfg.name, cfg.qps, gpu_count)
        log.info("#" * 72)

        BT2.QPS = cfg.qps
        BT2.DATASET = DATASET

        RMB.cleanup_all()
        time.sleep(8)

        url = deploy(cfg)
        if url is None:
            all_results[cfg.name] = {"status": "DEPLOY_FAILED"}
            continue

        # For MegaScale (tier=False): lock freq after deploy
        if not cfg.tier:
            gpus = RMB.card_gpus(16)
            RMB.lock_freq_both(gpus, MAX_FREQ)
            log.info("  Locked all GPUs to %dMHz", MAX_FREQ)

        if not RMB.test_generate(url[0]):
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

        if not cfg.tier:
            gpus = RMB.card_gpus(16)
            RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        time.sleep(5)

    return all_results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps-list", default="2,4,6,8,12,16")
    parser.add_argument("--scheme", default="both", choices=["aflex", "megascale", "both"])
    args = parser.parse_args()

    qps_targets = [int(x) for x in args.qps_list.split(",")]

    all_out = {}
    if args.scheme in ("aflex", "both"):
        targets = [c for c in AFLEX_CONFIGS if c.qps in qps_targets]
        all_out["aflex"] = run_configs(targets, "AFlex")

    if args.scheme in ("megascale", "both"):
        targets = [c for c in MEGA_CONFIGS if c.qps in qps_targets]
        all_out["megascale"] = run_configs(targets, "MegaScale")

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"conv_allqps_{ts}.json"
    meta = {
        "benchmark": "conv_aflex_megascale_sweep",
        "dataset": DATASET,
        "ttft_slo_ms": RMB.TTFT_SLO_MS,
        "tpot_slo_ms": RMB.TPOT_SLO_MS,
        "n3": RMB.NODE1_IP, "n4": RMB.NODE2_IP,
    }
    out.write_text(json.dumps({"meta": meta, "results": all_out}, indent=2))
    log.info("Saved %s", out)

    # Print summary
    print(f"\n{'='*72}")
    print(f"CONV DATASET RESULTS")
    print(f"{'='*72}")
    for scheme, results in all_out.items():
        print(f"\n  [{scheme}]")
        for name, r in sorted(results.items()):
            if isinstance(r, dict) and r.get("status") == "PASS":
                print(f"    {name}: thpt={r['throughput_tok_s']:.1f} TTFT={r['ttft_proc_avg_ms']:.1f}ms "
                      f"TPOT={r['tpot_avg_ms']:.1f}ms E/tok={r.get('energy_per_token_mj',0):.1f}mJ")
            else:
                print(f"    {name}: {r.get('status','?') if isinstance(r,dict) else 'ERROR'}")


if __name__ == "__main__":
    main()
