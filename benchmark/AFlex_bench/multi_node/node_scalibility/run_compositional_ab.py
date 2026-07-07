#!/usr/bin/env python3
"""A/B benchmark: PDAF+Tier (V2 coupled) vs PDAF+Tier (compositional).

Runs chatbot_lphd across QPS sweep on 16-card PDAF to compare
the old V2 pipeline model vs the new compositional decode DVFS.
"""
from __future__ import annotations
import json, time, sys, os, subprocess, copy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import run_node_scalability as RS

SCENARIO = "chatbot_lphd"
QPS_LIST = [2, 4, 6, 8, 12, 16]
MAX_RUN_S = 400
NGPU = 16
OUT_DIR = Path(__file__).parent / "results" / "compositional_ab"
OUT_DIR.mkdir(parents=True, exist_ok=True)

log = RS.log


def patch_dvfs_flags(compositional: bool):
    """Monkey-patch _afd_dvfs_flags to toggle compositional mode."""
    def _flags(tier):
        if not tier:
            return ""
        base = (" --afd-dvfs-enabled "
                f"--afd-energy-model-dir {RS.ENERGY_MODEL_DIR_V2} "
                f"--afd-ttft-slo-ms {int(RS.TTFT_SLO_MS)} "
                f"--afd-tpot-slo-us {int(RS.TPOT_SLO_MS * 1000)} "
                "--afd-dvfs-idle-lock ")
        if compositional:
            base += ("--afd-dvfs-decode-compositional "
                     "--afd-dvfs-feedback "
                     "--afd-dvfs-comm-us 3300.0")
        return base
    RS._afd_dvfs_flags = _flags


def run_sweep(label: str, compositional: bool, qps_filter: list = None) -> dict:
    """Run full QPS sweep for one variant."""
    patch_dvfs_flags(compositional)
    results = {}
    sweep_qps = qps_filter if qps_filter else QPS_LIST

    for qps in sweep_qps:
        log.info("=== %s | %s QPS=%d ===", label, SCENARIO, qps)
        RS.cleanup_all()
        time.sleep(5)

        url = RS.start_pdaf(NGPU, tier=True)
        if url is None:
            log.error("Deploy failed for %s qps=%d", label, qps)
            results[f"qps{qps}"] = {"status": "DEPLOY_FAILED"}
            continue

        gpus = RS.card_gpus(NGPU)
        RS.lock_freq_both(gpus, 1410)

        if not RS.test_generate(url):
            log.error("Warmup failed for %s qps=%d", label, qps)
            RS.cleanup_all()
            results[f"qps{qps}"] = {"status": "WARMUP_FAILED"}
            continue

        result = RS.run_one_workload(url, SCENARIO, qps, gpus, gpus, MAX_RUN_S)
        if result is None:
            log.error("Workload failed for %s qps=%d", label, qps)
            RS.cleanup_all()
            results[f"qps{qps}"] = {"status": "WORKLOAD_FAILED"}
            continue
        wl_key, metrics = result
        log.info("  DEBUG metrics keys: %s", list(metrics.keys()) if isinstance(metrics, dict) else type(metrics))
        if metrics.get("status") == "FAIL":
            log.error("  ALL REQUESTS FAILED for %s qps=%d: %s", label, qps, metrics)
        results[f"qps{qps}"] = metrics
        log.info("  %s qps=%d: TPOT_p50=%.1fms E/tok=%.0fmJ tput=%.0ftok/s",
                 label, qps,
                 metrics.get("tpot_p50_ms", 0),
                 metrics.get("energy_per_token_mj", 0),
                 metrics.get("throughput_tok_s", 0))

        RS.cleanup_all()
        time.sleep(3)

    return results


def main():
    log.info("Starting compositional A/B benchmark on chatbot_lphd 16-card")

    # V2 results from previous partial run
    v2_results = {
        "qps2": {"tpot_p50_ms": 131.2, "energy_per_token_mj": 2876, "throughput_tok_s": 411, "status": "PASS"},
        "qps4": {"tpot_p50_ms": 131.1, "energy_per_token_mj": 2319, "throughput_tok_s": 512, "status": "PASS"},
        "qps6": {"tpot_p50_ms": 132.3, "energy_per_token_mj": 2306, "throughput_tok_s": 512, "status": "PASS"},
        "qps8": {"tpot_p50_ms": 130.3, "energy_per_token_mj": 2292, "throughput_tok_s": 512, "status": "PASS"},
        "qps12": {"tpot_p50_ms": 130.9, "energy_per_token_mj": 2282, "throughput_tok_s": 512, "status": "PASS"},
        "qps16": {"tpot_p50_ms": 131.3, "energy_per_token_mj": 2274, "throughput_tok_s": 512, "status": "PASS"},
    }

    # Skip V2 remaining since we already have all points
    # Run compositional (new) - full sweep
    log.info("========== Phase 2: PDAF+Tier (compositional) ==========")
    comp_results = run_sweep("Compositional", compositional=True)

    # Save
    out = {
        "scenario": SCENARIO,
        "ngpu": NGPU,
        "qps_list": QPS_LIST,
        "timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "v2_coupled": v2_results,
        "compositional": comp_results,
    }
    out_file = OUT_DIR / f"ab_{SCENARIO}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_file, "w") as f:
        json.dump(out, f, indent=2)
    log.info("Saved results to %s", out_file)

    # Print summary table
    print("\n" + "="*80)
    print(f"{'QPS':>4} | {'V2 TPOT_p50':>12} {'V2 E/tok':>10} {'V2 tput':>8} | "
          f"{'Comp TPOT_p50':>14} {'Comp E/tok':>10} {'Comp tput':>10}")
    print("-"*80)
    for qps in QPS_LIST:
        k = f"qps{qps}"
        v2 = v2_results.get(k, {})
        co = comp_results.get(k, {})
        print(f"{qps:>4} | "
              f"{v2.get('tpot_p50_ms',0):>10.1f}ms {v2.get('energy_per_token_mj',0):>8.0f}mJ {v2.get('throughput_tok_s',0):>7.0f} | "
              f"{co.get('tpot_p50_ms',0):>12.1f}ms {co.get('energy_per_token_mj',0):>8.0f}mJ {co.get('throughput_tok_s',0):>9.0f}")
    print("="*80)


if __name__ == "__main__":
    main()
