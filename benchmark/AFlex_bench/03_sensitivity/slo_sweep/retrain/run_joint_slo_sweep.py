#!/usr/bin/env python3
"""Joint TTFT x TPOT SLO sweep: parallel 2-worker execution on 8 GPUs.

Worker A: GPU 0-3 (ports 55xxx)
Worker B: GPU 4-7 (ports 54xxx)

Each worker pulls tasks from a shared queue and runs them sequentially.
"""
import argparse
import json
import subprocess
import sys
import threading
import time
from itertools import product
from pathlib import Path
from queue import Empty, Queue

PYTHON = "/workspace/env/sglang-tier/bin/python"
BENCH_SCRIPT = str(Path(__file__).resolve().parent.parent / "scripts" / "bench" / "run_4gpu_deploy_bench.py")
WL_FILE = str(Path(__file__).resolve().parent.parent / "workloads" / "workload_steady.jsonl")
V2_MODEL_DIR = str(Path(__file__).resolve().parent / "models_v2")

TTFT_SLOS = [5000, 2000, 1000, 500, 300, 200]
TPOT_SLOS = [300, 250, 200, 150, 100, 90, 80, 70]

OUT_BASE = Path(__file__).resolve().parent / "results_joint_sweep"
LOG_BASE = Path(__file__).resolve().parent / "logs_joint_sweep"


def run_one(ttft: int, tpot: int, scheme: str, gpu_base: int, out_base: Path, log_base: Path):
    """Run a single experiment. Returns (success, tag)."""
    if scheme == "baseline":
        deploy = "pdaf_4g_dyn"
        freq = "max"
        tag = f"baseline_ttft{ttft}_tpot{tpot}"
    else:
        deploy = "pdaf_4g_dyn_tier"
        freq = "auto"
        tag = f"v2_ttft{ttft}_tpot{tpot}"

    out_dir = out_base / f"ttft_{ttft}_tpot_{tpot}" / scheme / "json"
    log_dir = log_base / f"ttft_{ttft}_tpot_{tpot}" / scheme
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    env = dict(
        PATH="/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
        HOME="/root",
        SGLANG_ENERGY_MODEL_DIR=V2_MODEL_DIR,
    )

    cmd = [
        PYTHON, BENCH_SCRIPT,
        "--deploys", deploy,
        "--workloads", WL_FILE,
        "--freq", freq,
        "--tpot-slo-ms", str(tpot),
        "--ttft-slo-ms", str(ttft),
        "--output-dir", str(out_dir),
        "--log-dir", str(log_dir),
        "--max-run-s", "600",
        "--force",
        "--gpu-base", str(gpu_base),
    ]

    run_log = log_dir / f"{tag}_run.log"
    with open(run_log, "w") as fh:
        result = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                env={**subprocess.os.environ, **env},
                                timeout=900)
    return result.returncode == 0, tag


def worker(name: str, gpu_base: int, queue: Queue, results: list, lock: threading.Lock):
    """Worker thread: pull tasks from queue and execute sequentially."""
    while True:
        try:
            task = queue.get_nowait()
        except Empty:
            break

        ttft, tpot, scheme = task
        print(f"[{name}] Starting: TTFT={ttft}ms TPOT={tpot}ms scheme={scheme}", flush=True)
        t0 = time.time()

        try:
            ok, tag = run_one(ttft, tpot, scheme, gpu_base, OUT_BASE, LOG_BASE)
            elapsed = time.time() - t0
            status = "OK" if ok else "FAIL"
            print(f"[{name}] {status}: {tag} ({elapsed:.0f}s)", flush=True)
            with lock:
                results.append({"ttft": ttft, "tpot": tpot, "scheme": scheme,
                                "status": status, "elapsed": elapsed})
        except subprocess.TimeoutExpired:
            elapsed = time.time() - t0
            print(f"[{name}] TIMEOUT: TTFT={ttft} TPOT={tpot} {scheme} ({elapsed:.0f}s)", flush=True)
            with lock:
                results.append({"ttft": ttft, "tpot": tpot, "scheme": scheme,
                                "status": "TIMEOUT", "elapsed": elapsed})
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[{name}] ERROR: TTFT={ttft} TPOT={tpot} {scheme}: {e}", flush=True)
            with lock:
                results.append({"ttft": ttft, "tpot": tpot, "scheme": scheme,
                                "status": "ERROR", "elapsed": elapsed})

        queue.task_done()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-only", action="store_true",
                    help="Only run baseline experiments")
    ap.add_argument("--v2-only", action="store_true",
                    help="Only run V2 experiments")
    args = ap.parse_args()

    OUT_BASE.mkdir(parents=True, exist_ok=True)
    LOG_BASE.mkdir(parents=True, exist_ok=True)

    tasks = Queue()
    combos = list(product(TTFT_SLOS, TPOT_SLOS))
    print(f"Total combinations: {len(combos)} (TTFT x TPOT)")

    if not args.baseline_only:
        for ttft, tpot in combos:
            tasks.put((ttft, tpot, "v2"))

    if not args.v2_only:
        for ttft, tpot in combos:
            tasks.put((ttft, tpot, "baseline"))

    total = tasks.qsize()
    print(f"Total tasks: {total}")
    print(f"Worker A: GPU 0-3 | Worker B: GPU 4-7")
    print(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []
    lock = threading.Lock()

    t_a = threading.Thread(target=worker, args=("WorkerA", 0, tasks, results, lock))
    t_b = threading.Thread(target=worker, args=("WorkerB", 4, tasks, results, lock))

    t0 = time.time()
    t_a.start()
    t_b.start()
    t_a.join()
    t_b.join()

    elapsed_total = time.time() - t0
    print("=" * 60)
    print(f"All {total} tasks completed in {elapsed_total/60:.1f} min")
    print(f"Success: {sum(1 for r in results if r['status']=='OK')}/{total}")
    print(f"Results: {OUT_BASE}/")

    summary_path = OUT_BASE / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
