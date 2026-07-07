#!/usr/bin/env python3
"""Capture GPU frequency timelines for 3 tier schemes on code dataset.

Launches each tier scheme (DynamoLLM, BiScale, AFlex), runs the code workload,
and records actual GPU SM clock frequencies over time.

For AFlex (PDAF tier): uses AFD_DVFS_DECISION_LOG for precise per-decision logs.
For DynamoLLM/BiScale: polls nvidia-smi at ~100ms intervals for actual SM freq.

Usage (run on node1 host):
    python3 capture_freq_timeline.py --qps 8 --n-requests 100
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import aiohttp
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_macro_benchmark as MN

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("freq_capture")

WORKLOAD_DIR = HERE / "workloads"
OUT_DIR = HERE / "results" / "freq_timeline_code"
NGPU = 16

CONTAINER_LOG_DIR = (
    "/workspace/sglang/benchmark/AFlex_bench/multi_node/"
    "node_scalibility_macro/results/freq_timeline_code"
)


def load_workload(dataset: str, qps: int, n_requests: int | None) -> list[dict]:
    path = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if n_requests and n_requests < len(rows):
        rows = rows[:n_requests]
    return rows


class FreqMonitor:
    """Poll nvidia-smi for actual SM clock on specified GPUs."""

    def __init__(self, host: str, gpus: list[int], interval_s: float = 0.1):
        self.host = host
        self.gpus = gpus
        self.interval = interval_s
        self.samples: list[dict] = []
        self._stop = threading.Event()
        self._thread = None

    def _poll_loop(self):
        gpu_csv = ",".join(str(g) for g in self.gpus)
        while not self._stop.is_set():
            t = time.time()
            try:
                if self.host == MN.NODE1_IP:
                    cmd = (f"docker exec {MN.CONTAINER} nvidia-smi "
                           f"--query-gpu=index,clocks.current.sm "
                           f"--format=csv,noheader,nounits -i {gpu_csv}")
                    out = subprocess.check_output(cmd, shell=True,
                                                 timeout=2).decode().strip()
                else:
                    cmd = (f"ssh -o StrictHostKeyChecking=no root@{self.host} "
                           f"\"docker exec {MN.CONTAINER} nvidia-smi "
                           f"--query-gpu=index,clocks.current.sm "
                           f"--format=csv,noheader,nounits -i {gpu_csv}\"")
                    out = subprocess.check_output(cmd, shell=True,
                                                 timeout=3).decode().strip()
                for line in out.splitlines():
                    parts = line.strip().split(",")
                    if len(parts) == 2:
                        gpu_idx = int(parts[0].strip())
                        freq = int(parts[1].strip())
                        self.samples.append({"t": round(t, 3),
                                             "gpu": gpu_idx, "freq_mhz": freq})
            except Exception as e:
                log.warning("nvidia-smi poll failed: %s", e)
            elapsed = time.time() - t
            sleep_t = max(0, self.interval - elapsed)
            self._stop.wait(sleep_t)

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)


async def run_workload(reqs, url, max_run_s=300):
    results = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [
            asyncio.create_task(
                MN.send_one(session, url + "/generate", r, base_time, results))
            for r in reqs
        ]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=max_run_s,
                )
            except asyncio.TimeoutError:
                log.warning("Timed out after %ds", max_run_s)
    return results


def patch_afd_env_for_logging(log_dir: Path):
    """Monkey-patch run_macro_benchmark._afd_env to add DVFS decision logging."""
    orig = MN._afd_env

    def _patched(role, gpus, tp, attn_gpus, ffn_gpus):
        base = orig(role, gpus, tp, attn_gpus, ffn_gpus)
        cpath = (f"{CONTAINER_LOG_DIR}/{log_dir.name}/"
                 f"dvfs_decisions_{{persp}}_{{disagg}}_gpu{{gpu}}.jsonl")
        return base.replace(";", f" AFD_DVFS_DECISION_LOG='{cpath}';", 1)

    MN._afd_env = _patched
    return orig


def run_scheme(scheme: str, qps: int, n_requests: int, max_run_s: int):
    """Run one tier scheme, capture frequency timeline."""
    log.info("=" * 60)
    log.info("Starting scheme: %s, qps=%d, n_req=%d", scheme, qps, n_requests)
    log.info("=" * 60)

    scheme_dir = OUT_DIR / scheme
    scheme_dir.mkdir(parents=True, exist_ok=True)

    # Cleanup any previous servers
    MN.cleanup_all()
    time.sleep(5)

    # Launch the tier server
    orig_afd_env = None
    if scheme == "pdaf_tier":
        orig_afd_env = patch_afd_env_for_logging(scheme_dir)
        url = MN.start_pdaf(NGPU, tier=True)
    elif scheme == "native_tp2_tier":
        url = MN.start_native_tp(NGPU, tier=True)
    elif scheme == "pd_dp_tier":
        url = MN.start_pd_dp(NGPU, tier=True)
    else:
        raise ValueError(f"Unknown scheme: {scheme}")

    if url is None:
        log.error("Failed to start %s", scheme)
        if orig_afd_env:
            MN._afd_env = orig_afd_env
        return None

    log.info("Server ready at %s", url)

    # Start frequency monitor (poll both nodes)
    gpus_n1 = list(range(8))
    gpus_n2 = list(range(8))
    monitor_n1 = FreqMonitor(MN.NODE1_IP, gpus_n1, interval_s=0.2)
    monitor_n2 = FreqMonitor(MN.NODE2_IP, gpus_n2, interval_s=0.2)

    # Load workload
    reqs = load_workload("code", qps, n_requests)
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(max_run_s, last_arrival + 60), 600))

    log.info("Running workload: %d reqs, max_time=%ds", len(reqs), run_s)

    # Start monitoring
    monitor_n1.start()
    monitor_n2.start()
    t_start = time.time()

    # Run workload
    results = asyncio.run(run_workload(reqs, url, run_s))

    t_end = time.time()
    monitor_n1.stop()
    monitor_n2.stop()

    # Restore patched env
    if orig_afd_env:
        MN._afd_env = orig_afd_env

    # Cleanup
    MN.cleanup_all()

    # Save results
    ok = [r for r in results if r.get("success")]
    log.info("Completed: %d/%d ok, duration=%.1fs", len(ok), len(reqs),
             t_end - t_start)

    # Merge monitor samples with time offset
    timeline_n1 = [{"node": 1, **s} for s in monitor_n1.samples]
    timeline_n2 = [{"node": 2, **s} for s in monitor_n2.samples]
    timeline = sorted(timeline_n1 + timeline_n2, key=lambda x: x["t"])

    output = {
        "scheme": scheme,
        "dataset": "code",
        "qps": qps,
        "n_requests": len(reqs),
        "n_ok": len(ok),
        "t_start": t_start,
        "t_end": t_end,
        "duration_s": round(t_end - t_start, 1),
        "freq_samples_count": len(timeline),
        "nvidia_smi_timeline": timeline,
    }

    out_path = scheme_dir / "freq_timeline.json"
    out_path.write_text(json.dumps(output, indent=2))
    log.info("Saved %d freq samples to %s", len(timeline), out_path)

    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qps", type=int, default=8,
                        help="QPS for the code workload")
    parser.add_argument("--n-requests", type=int, default=100,
                        help="Number of requests to send")
    parser.add_argument("--max-run-s", type=int, default=300,
                        help="Max runtime per scheme")
    parser.add_argument("--schemes", default="native_tp2_tier,pd_dp_tier,pdaf_tier",
                        help="Comma-separated list of tier schemes to test")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    schemes = [s.strip() for s in args.schemes.split(",")]

    all_results = {}
    for scheme in schemes:
        try:
            result = run_scheme(scheme, args.qps, args.n_requests, args.max_run_s)
            if result:
                all_results[scheme] = result
        except Exception as e:
            log.error("Scheme %s failed: %s", scheme, e)
            import traceback
            traceback.print_exc()

    # Save combined summary
    summary_path = OUT_DIR / "capture_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2, default=str))
    log.info("\nAll done. Summary: %s", summary_path)


if __name__ == "__main__":
    main()
