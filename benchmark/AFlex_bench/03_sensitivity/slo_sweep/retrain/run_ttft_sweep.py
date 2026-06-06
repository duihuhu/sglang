#!/usr/bin/env python3
"""TTFT SLO sweep: vary TTFT SLO with fixed TPOT SLO, measure processing TTFT.

Uses 4 GPUs (4-7) with PDAF DynM Tier + steady workload.
Tests both V2 (DVFS) and Baseline (max freq) for each TTFT SLO.
"""
import json
import subprocess
import sys
import time
from pathlib import Path

PYTHON = "/workspace/env/sglang-tier/bin/python"
BENCH_SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "bench" / "run_4gpu_deploy_bench.py")
WL_FILE = str(Path(__file__).resolve().parent.parent / "workloads" / "workload_steady.jsonl")
V2_MODEL_DIR = str(Path(__file__).resolve().parent / "models_v2")

TTFT_SLOS = [5000, 2000, 1000, 500, 300, 200]
TPOT_SLO = 150  # fixed

OUT_BASE = Path(__file__).resolve().parent / "results_ttft_sweep"
LOG_BASE = Path(__file__).resolve().parent / "logs_ttft_sweep"


def run_one(ttft_slo: int, scheme: str):
    """Run a single test point."""
    if scheme == "baseline":
        deploy = "pdaf_4g_dyn"
        freq = "max"
    else:
        deploy = "pdaf_4g_dyn_tier"
        freq = "auto"

    tag = f"{scheme}_ttft{ttft_slo}"
    out_dir = OUT_BASE / tag / "json"
    log_dir = LOG_BASE / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = dict(**subprocess.os.environ)
    env["SGLANG_ENERGY_MODEL_DIR"] = V2_MODEL_DIR

    cmd = [
        PYTHON, BENCH_SCRIPT,
        "--deploys", deploy,
        "--workloads", WL_FILE,
        "--freq", freq,
        "--tpot-slo-ms", str(TPOT_SLO),
        "--ttft-slo-ms", str(ttft_slo),
        "--output-dir", str(out_dir),
        "--log-dir", str(log_dir),
        "--max-run-s", "300",
        "--force",
        "--gpu-base", "0",
    ]

    print(f"\n{'='*60}")
    print(f"  {scheme.upper()} | TTFT SLO={ttft_slo}ms | TPOT SLO={TPOT_SLO}ms")
    print(f"{'='*60}")

    run_log = log_dir / f"{tag}_run.log"
    with open(run_log, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
                                timeout=900)

    if result.returncode != 0:
        print(f"  FAILED (exit={result.returncode})")
        return None

    # Find result JSON
    json_files = list(out_dir.glob("*_results.json"))
    if not json_files:
        print(f"  No result JSON found")
        return None

    with open(json_files[0]) as f:
        data = json.load(f)

    print(f"  Thpt={data.get('throughput_tok_s', 0):.1f} tok/s  "
          f"TTFT={data.get('ttft_avg_ms', 0):.0f}ms (proc={data.get('ttft_proc_avg_ms', 0):.0f}ms)  "
          f"TPOT={data.get('tpot_avg_ms', 0):.0f}ms  "
          f"Energy={data.get('total_energy_j', 0):.0f}J  "
          f"SLO={data.get('slo_violation_rate', 0):.1f}%")
    return data


def main():
    print("TTFT SLO Sweep (TPOT fixed at %dms)" % TPOT_SLO)
    print(f"TTFT SLOs: {TTFT_SLOS}")

    all_results = {}

    for ttft_slo in TTFT_SLOS:
        for scheme in ["baseline", "v2"]:
            result = run_one(ttft_slo, scheme)
            if result:
                all_results[f"{scheme}_ttft{ttft_slo}"] = result

    # Save combined results
    summary_path = OUT_BASE / "ttft_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_path}")

    # Print table
    print(f"\n{'TTFT SLO':>10} | {'Scheme':>10} | {'Thpt':>7} | {'TTFT(total)':>11} | {'TTFT(proc)':>10} | {'TPOT':>6} | {'Energy':>8} | {'SLO%':>5}")
    print("-" * 90)
    for ttft_slo in TTFT_SLOS:
        for scheme in ["baseline", "v2"]:
            key = f"{scheme}_ttft{ttft_slo}"
            r = all_results.get(key)
            if r:
                print(f"{ttft_slo:>10} | {scheme:>10} | {r.get('throughput_tok_s',0):>7.1f} | "
                      f"{r.get('ttft_avg_ms',0):>10.0f}ms | {r.get('ttft_proc_avg_ms',0):>9.0f}ms | "
                      f"{r.get('tpot_avg_ms',0):>5.0f}ms | {r.get('total_energy_j',0):>7.0f}J | "
                      f"{r.get('slo_violation_rate',0):>5.1f}")


if __name__ == "__main__":
    main()
