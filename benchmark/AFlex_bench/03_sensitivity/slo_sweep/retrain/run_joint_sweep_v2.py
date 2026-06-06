#!/usr/bin/env python3
"""Joint TTFT x TPOT SLO sweep (single worker, GPU 4-7).

Reruns the full grid to capture ttft_proc_avg_ms (excluded queuing).
"""
import json
import subprocess
import sys
import time
from itertools import product
from pathlib import Path

PYTHON = "/workspace/env/sglang-tier/bin/python"
BENCH_SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "bench" / "run_4gpu_deploy_bench.py")
WL_FILE = str(Path(__file__).resolve().parent.parent / "workloads" / "workload_steady.jsonl")
V2_MODEL_DIR = str(Path(__file__).resolve().parent / "models_v2")

TTFT_SLOS = [2000, 1000, 500, 300, 200, 150]
TPOT_SLOS = [200, 150, 120, 100, 90, 80]

GPU_BASE = 4

OUT_BASE = Path(__file__).resolve().parent / "results_joint_sweep_v3"
LOG_BASE = Path(__file__).resolve().parent / "logs_joint_sweep_v3"


def run_one(ttft: int, tpot: int, scheme: str):
    """Run a single experiment."""
    if scheme == "baseline":
        deploy = "pdaf_4g_dyn"
        freq = "max"
    else:
        deploy = "pdaf_4g_dyn_tier"
        freq = "auto"

    tag = f"{scheme}_ttft{ttft}_tpot{tpot}"
    out_dir = OUT_BASE / f"ttft_{ttft}_tpot_{tpot}" / scheme / "json"
    log_dir = LOG_BASE / f"ttft_{ttft}_tpot_{tpot}" / scheme
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # Skip if result already exists
    existing = list(out_dir.glob("*_results.json"))
    if existing:
        with open(existing[0]) as f:
            data = json.load(f)
        if data.get("ttft_proc_avg_ms", 0) > 0 or scheme == "baseline":
            print(f"  [SKIP] {tag} (already done)", flush=True)
            return True

    env = dict(**subprocess.os.environ)
    env["SGLANG_ENERGY_MODEL_DIR"] = V2_MODEL_DIR

    cmd = [
        PYTHON, BENCH_SCRIPT,
        "--deploys", deploy,
        "--workloads", WL_FILE,
        "--freq", freq,
        "--tpot-slo-ms", str(tpot),
        "--ttft-slo-ms", str(ttft),
        "--output-dir", str(out_dir),
        "--log-dir", str(log_dir),
        "--max-run-s", "300",
        "--force",
        "--gpu-base", str(GPU_BASE),
    ]

    run_log = log_dir / f"{tag}_run.log"
    with open(run_log, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=env, timeout=600)
    return result.returncode == 0


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    LOG_BASE.mkdir(parents=True, exist_ok=True)

    combos = list(product(TTFT_SLOS, TPOT_SLOS))
    schemes = ["v2", "baseline"]
    total = len(combos) * len(schemes)

    print(f"Joint TTFT x TPOT SLO Sweep (GPU {GPU_BASE}-{GPU_BASE+3})")
    print(f"Grid: {len(TTFT_SLOS)} TTFT x {len(TPOT_SLOS)} TPOT x {len(schemes)} schemes = {total} tasks")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60, flush=True)

    done = 0
    fails = 0
    t0 = time.time()

    for ttft, tpot in combos:
        for scheme in schemes:
            done += 1
            tag = f"{scheme}_ttft{ttft}_tpot{tpot}"
            elapsed_min = (time.time() - t0) / 60
            print(f"\n[{done}/{total}] {tag} (elapsed={elapsed_min:.0f}m)", flush=True)

            try:
                ok = run_one(ttft, tpot, scheme)
                if not ok:
                    fails += 1
                    print(f"  FAILED: {tag}", flush=True)
            except subprocess.TimeoutExpired:
                fails += 1
                print(f"  TIMEOUT: {tag}", flush=True)
            except Exception as e:
                fails += 1
                print(f"  ERROR: {tag}: {e}", flush=True)

    elapsed_total = (time.time() - t0) / 60
    print(f"\n{'='*60}")
    print(f"Done: {done}/{total} | Fails: {fails} | Time: {elapsed_total:.1f} min")
    print(f"Results: {OUT_BASE}/")


if __name__ == "__main__":
    main()
