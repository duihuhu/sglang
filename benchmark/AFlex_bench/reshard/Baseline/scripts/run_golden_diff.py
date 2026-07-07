#!/usr/bin/env python3
"""Run golden-diff diagnostic: normal TP2 vs in-place TP1->TP2 reshard.

Steps:
  1. Launch golden TP2, dump scheduler state at event_loop entry.
  2. Launch TP1 + inplace-reshard-max-tp=2, dump at entry, trigger reshard,
     dump again at post_reshard.
  3. Structured diff of golden vs reshard state JSON.

Usage (inside operator_test container):
  python3 benchmark/AFlex_bench/reshard/Baseline/scripts/run_golden_diff.py
"""
from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path("/workspace/sglang")
if not ROOT.exists():
    ROOT = Path(__file__).resolve().parents[4]  # repo root from scripts/

MODEL = "/models/Qwen3-32B"
PORT = 31700
NCCL_PORT = 32800
BASE = f"http://127.0.0.1:{PORT}"
OUT_BASE = ROOT / "benchmark/AFlex_bench/reshard/Baseline/results/golden_diff"
LOG_DIR = ROOT / "benchmark/AFlex_bench/reshard/Baseline/logs"

COMMON_ARGS = [
    "python3", "-m", "sglang.launch_server",
    "--model-path", MODEL,
    "--host", "127.0.0.1",
    "--port", str(PORT),
    "--nccl-port", str(NCCL_PORT),
    "--mem-fraction-static", "0.85",
    "--disable-cuda-graph",
    "--disable-piecewise-cuda-graph",
    "--skip-server-warmup",
    "--base-gpu-id", "0",
    "--attention-backend", "triton",
]


def pkill_server():
    subprocess.run(
        "pkill -TERM -f 'sglang.launch_server.*31700' || true; "
        "sleep 2; pkill -KILL -f 'sglang.launch_server.*31700' || true",
        shell=True,
        check=False,
    )
    time.sleep(2)


def wait_ready(timeout=600):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = requests.get(BASE + "/health", timeout=5)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def launch(cmd: list[str], log_name: str, env: dict) -> subprocess.Popen:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / log_name
    f = open(log_path, "w")
    print(f"Launch: {' '.join(cmd)}")
    print(f"  log -> {log_path}")
    return subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env, cwd=str(ROOT))


def trigger_reshard(new_tp: int = 2):
    r = requests.post(
        BASE + "/reshard_tp",
        json={"new_tp_size": new_tp},
        timeout=30,
    )
    print(f"reshard_tp -> {r.status_code} {r.text[:200]}")
    return r.status_code == 200


def wait_post_reshard(timeout=300):
    """Wait until post_reshard dump files exist for both ranks."""
    dump_dir = OUT_BASE / "inplace_tp2"
    for _ in range(timeout):
        ok = all(
            (dump_dir / f"state_inplace_tp2_post_reshard_rank{r}.json").exists()
            for r in (0, 1)
        )
        if ok:
            return True
        # also probe generate
        try:
            r = requests.post(
                BASE + "/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"max_new_tokens": 4, "temperature": 0},
                },
                timeout=30,
            )
            if r.status_code == 200:
                pass
        except Exception:
            pass
        time.sleep(2)
    return False


def main():
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    golden_dir = OUT_BASE / "golden_tp2"
    inplace_dir = OUT_BASE / "inplace_tp2"
    for d in (golden_dir, inplace_dir):
        d.mkdir(parents=True, exist_ok=True)
        for f in d.glob("*.json"):
            f.unlink()

    pkill_server()

    # --- Phase 1: Golden TP2 ---
    print("\n=== Phase 1: Golden TP2 ===")
    env = os.environ.copy()
    env["SGLANG_RESHARD_STATE_DUMP"] = str(golden_dir)
    env["SGLANG_RESHARD_STATE_SCENARIO"] = "golden_tp2"
    cmd = COMMON_ARGS + ["--tp", "2"]
    proc = launch(cmd, "golden_diff_tp2.log", env)
    if not wait_ready(900):
        proc.kill()
        raise RuntimeError("Golden TP2 failed to become ready")
    time.sleep(3)
    for r in (0, 1):
        p = golden_dir / f"state_golden_tp2_event_loop_entry_rank{r}.json"
        print(f"  golden rank{r} dump: {'OK' if p.exists() else 'MISSING'}")
    proc.terminate()
    proc.wait(timeout=30)
    pkill_server()
    time.sleep(5)

    # --- Phase 2: In-place TP1 -> TP2 ---
    print("\n=== Phase 2: In-place TP1 -> TP2 ===")
    env = os.environ.copy()
    env["SGLANG_RESHARD_STATE_DUMP"] = str(inplace_dir)
    env["SGLANG_RESHARD_STATE_SCENARIO"] = "inplace_tp2"
    cmd = COMMON_ARGS + ["--tp", "1", "--inplace-reshard-max-tp", "2"]
    proc = launch(cmd, "golden_diff_inplace_tp1.log", env)
    if not wait_ready(900):
        proc.kill()
        raise RuntimeError("In-place TP1 failed to become ready")
    time.sleep(3)
    for r in (0, 1):
        p = inplace_dir / f"state_inplace_tp2_event_loop_entry_rank{r}.json"
        print(f"  pre-reshard rank{r} dump: {'OK' if p.exists() else 'MISSING'}")

    trigger_reshard(2)
    if not wait_post_reshard(300):
        print("WARNING: post_reshard dumps not found within timeout")
    for r in (0, 1):
        p = inplace_dir / f"state_inplace_tp2_post_reshard_rank{r}.json"
        print(f"  post-reshard rank{r} dump: {'OK' if p.exists() else 'MISSING'}")

    proc.terminate()
    proc.wait(timeout=30)
    pkill_server()

    # --- Phase 3: Diff ---
    print("\n=== Phase 3: Diff ===")
    diff_out = OUT_BASE / "diff_report.json"
    diff_script = ROOT / "benchmark/AFlex_bench/reshard/Baseline/scripts/diff_reshard_state.py"
    subprocess.run(
        [
            "python3", str(diff_script),
            "--golden-dir", str(golden_dir),
            "--reshard-dir", str(inplace_dir),
            "--golden-tag", "event_loop_entry",
            "--reshard-tag", "post_reshard",
            "--golden-scenario", "golden_tp2",
            "--reshard-scenario", "inplace_tp2",
            "--out", str(diff_out),
        ],
        check=True,
    )
    with open(diff_out) as f:
        report = json.load(f)
    summary_path = OUT_BASE / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"total_diffs={report['summary']['total_diffs']}\n")
        f.write(f"critical_diffs={len(report['summary']['critical_diffs'])}\n")
        for d in report["summary"]["critical_diffs"]:
            f.write(f"rank{d['rank']} {d['field']}: golden={d['golden']!r} reshard={d['reshard']!r}\n")
    print(f"\nDone. Report: {diff_out}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
