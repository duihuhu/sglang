#!/usr/bin/env python3
"""Benchmark SGLang TP=8 instance cold-start breakdown.

Measures the time spent in each phase of launching a fresh SGLang instance:
  1. Build TP communication group (NCCL init)
  2. Materialize weights & KV cache (model load + memory pool)
  3. Initialize inference engine (CUDA graph capture)

Designed to run inside an 8-GPU Docker container (e.g. node2 operator_test).
Parses SGLang server logs to extract per-phase timing.

Usage (inside container):
    python3 bench_instance_startup.py \
        --model-path /models/Qwen3-32B \
        --tp 8 \
        --port 39900 \
        --nccl-port 39910 \
        --repeats 3 \
        --output results/startup_breakdown.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional


@dataclass
class StartupBreakdown:
    """Timing breakdown for a single startup trial."""
    trial: int
    tp_size: int
    tp_comm_group_s: float
    weight_load_s: float
    memory_pool_s: float
    cuda_graph_s: float
    total_ready_s: float
    weight_mem_gb: float
    graph_mem_gb: float
    nccl_warmup_s: Optional[float] = None


def _parse_timestamp(line: str) -> Optional[float]:
    """Extract epoch seconds from SGLang log line timestamp.

    Handles formats:
      [2026-07-05 16:41:23] ...
      [2026-07-05 16:41:23.456] ...
      [2026-07-05 16:41:23.456 TP0] ...
    """
    m = re.match(
        r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})(?:\.(\d{3}))?(?:\s+TP\d+)?\]",
        line,
    )
    if not m:
        return None
    from datetime import datetime
    dt = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
    ts = dt.timestamp()
    if m.group(2):
        ts += int(m.group(2)) / 1000.0
    return ts


def parse_startup_log(log_text: str) -> dict:
    """Extract timing from SGLang server log output."""
    result = {}

    # Phase 1: TP comm group (torch distributed init) — has self-reported elapsed
    m = re.search(
        r"Init torch distributed ends\. elapsed=([\d.]+)\s*s", log_text
    )
    if m:
        result["tp_comm_group_s"] = float(m.group(1))

    # NCCL warmup (subset of distributed init)
    m = re.search(
        r"NCCL/RCCL warmup completed in ([\d.]+)s", log_text
    )
    if m:
        result["nccl_warmup_s"] = float(m.group(1))

    # Phase 2a: Weight loading — has self-reported elapsed
    m = re.search(
        r"Load weight end\.\s*elapsed=([\d.]+)\s*s.*?mem usage=([\d.]+)\s*GB",
        log_text,
    )
    if m:
        result["weight_load_s"] = float(m.group(1))
        result["weight_mem_gb"] = float(m.group(2))

    # Phase 2b: Memory pool (KV cache) — compute from timestamps
    lines = log_text.split("\n")
    init_dist_begin_ts = None
    init_dist_end_ts = None
    load_weight_begin_ts = None
    load_weight_end_ts = None
    memory_pool_end_ts = None
    cuda_graph_begin_ts = None
    cuda_graph_end_ts = None
    piecewise_begin_ts = None
    piecewise_end_ts = None
    server_ready_ts = None
    first_ts = None

    for line in lines:
        ts = _parse_timestamp(line)
        if ts is not None and first_ts is None:
            first_ts = ts
        if "Init torch distributed begin" in line and ts and init_dist_begin_ts is None:
            init_dist_begin_ts = ts
        if "Init torch distributed ends" in line and ts:
            init_dist_end_ts = ts
        if "Load weight begin" in line and ts and load_weight_begin_ts is None:
            load_weight_begin_ts = ts
        if "Load weight end" in line and ts:
            load_weight_end_ts = ts
        if "Memory pool end" in line and ts:
            memory_pool_end_ts = ts
        if "Capture cuda graph begin" in line and ts and cuda_graph_begin_ts is None:
            cuda_graph_begin_ts = ts
        if "Capture cuda graph end" in line and ts:
            cuda_graph_end_ts = ts
        if "Capture piecewise CUDA graph begin" in line and ts and piecewise_begin_ts is None:
            piecewise_begin_ts = ts
        if "Capture piecewise CUDA graph end" in line and ts:
            piecewise_end_ts = ts
        if "The server is fired up and ready to roll!" in line and ts:
            server_ready_ts = ts

    if load_weight_end_ts and memory_pool_end_ts:
        result["memory_pool_s"] = memory_pool_end_ts - load_weight_end_ts

    if first_ts and server_ready_ts:
        result["first_to_ready_s"] = server_ready_ts - first_ts

    # Timestamp-based precise breakdown
    if init_dist_begin_ts and init_dist_end_ts:
        result["ts_tp_comm_s"] = init_dist_end_ts - init_dist_begin_ts
    if load_weight_begin_ts and load_weight_end_ts:
        result["ts_weight_load_s"] = load_weight_end_ts - load_weight_begin_ts
    if load_weight_end_ts and memory_pool_end_ts:
        result["ts_memory_pool_s"] = memory_pool_end_ts - load_weight_end_ts
    if cuda_graph_begin_ts and (piecewise_end_ts or cuda_graph_end_ts):
        end = piecewise_end_ts or cuda_graph_end_ts
        result["ts_cuda_graph_total_s"] = end - cuda_graph_begin_ts
    if cuda_graph_begin_ts and cuda_graph_end_ts:
        result["ts_cuda_graph_standard_s"] = cuda_graph_end_ts - cuda_graph_begin_ts
    if piecewise_begin_ts and piecewise_end_ts:
        result["ts_piecewise_cuda_graph_s"] = piecewise_end_ts - piecewise_begin_ts

    # Process spawn overhead: from wall clock t0 to first log line
    if first_ts:
        result["first_log_ts"] = first_ts
    if server_ready_ts:
        result["ready_log_ts"] = server_ready_ts

    # Phase 3: CUDA graph capture — has self-reported elapsed
    m = re.search(
        r"Capture cuda graph end\. Time elapsed: ([\d.]+)\s*s.*?"
        r"mem usage=([\d.]+)\s*GB",
        log_text,
    )
    if m:
        result["cuda_graph_s"] = float(m.group(1))
        result["graph_mem_gb"] = float(m.group(2))

    # Piecewise CUDA graph (additive)
    m_pw = re.search(
        r"Capture piecewise CUDA graph end\. Time elapsed: ([\d.]+)\s*s",
        log_text,
    )
    if m_pw:
        pw_time = float(m_pw.group(1))
        result["piecewise_cuda_graph_s"] = pw_time
        result["cuda_graph_s"] = result.get("cuda_graph_s", 0) + pw_time

    # Server ready marker
    if server_ready_ts:
        result["server_ready"] = True

    return result


def launch_and_measure(
    model_path: str,
    tp_size: int,
    port: int,
    nccl_port: int,
    extra_args: List[str],
    timeout_s: int = 300,
) -> tuple[str, float]:
    """Launch SGLang server, wait for ready, return (log_text, total_seconds)."""

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--tp", str(tp_size),
        "--port", str(port),
        "--nccl-port", str(nccl_port),
        "--host", "127.0.0.1",
        "--mem-fraction-static", "0.85",
        "--disable-radix-cache",
        "--log-level", "info",
    ] + extra_args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(tp_size))
    env["SGLANG_LOG_MS"] = "1"

    t0 = time.perf_counter()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        preexec_fn=os.setsid,
    )

    log_lines = []
    ready = False
    try:
        while True:
            elapsed = time.perf_counter() - t0
            if elapsed > timeout_s:
                raise TimeoutError(
                    f"Server did not become ready within {timeout_s}s"
                )

            line = proc.stdout.readline()
            if not line:
                if proc.poll() is not None:
                    break
                time.sleep(0.1)
                continue

            log_lines.append(line)
            if "The server is fired up and ready to roll!" in line:
                ready = True
                break

        total_s = time.perf_counter() - t0
    finally:
        # Kill the entire process group
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (OSError, ProcessLookupError):
            pass
        time.sleep(2)
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (OSError, ProcessLookupError):
            pass
        try:
            proc.wait(timeout=10)
        except Exception:
            pass

    if not ready:
        log_text = "".join(log_lines)
        raise RuntimeError(
            f"Server did not reach ready state. Last 50 lines:\n"
            + "\n".join(log_lines[-50:])
        )

    return "".join(log_lines), total_s


def kill_existing_sglang(port: int):
    """Kill any existing SGLang processes on the given port."""
    subprocess.run(
        f"pkill -f 'sglang.launch_server.*--port {port}' || true",
        shell=True, capture_output=True,
    )
    subprocess.run(
        f"fuser -k {port}/tcp 2>/dev/null || true",
        shell=True, capture_output=True,
    )
    time.sleep(2)


def run_trial(
    trial: int,
    model_path: str,
    tp_size: int,
    port: int,
    nccl_port: int,
    extra_args: List[str],
    log_dir: Path,
    timeout_s: int,
) -> StartupBreakdown:
    """Run a single startup trial."""
    print(f"\n{'='*60}")
    print(f"  Trial {trial}: TP={tp_size}, model={model_path}")
    print(f"{'='*60}")

    kill_existing_sglang(port)

    log_text, total_s = launch_and_measure(
        model_path=model_path,
        tp_size=tp_size,
        port=port,
        nccl_port=nccl_port,
        extra_args=extra_args,
        timeout_s=timeout_s,
    )

    # Save raw log
    log_file = log_dir / f"trial_{trial}_tp{tp_size}.log"
    log_file.write_text(log_text, encoding="utf-8")
    print(f"  Log saved to {log_file}")

    # Parse timing
    parsed = parse_startup_log(log_text)
    print(f"  Parsed: {json.dumps(parsed, indent=2)}")

    breakdown = StartupBreakdown(
        trial=trial,
        tp_size=tp_size,
        tp_comm_group_s=parsed.get("tp_comm_group_s", 0.0),
        weight_load_s=parsed.get("weight_load_s", 0.0),
        memory_pool_s=parsed.get("memory_pool_s", parsed.get("ts_memory_pool_s", 0.0)),
        cuda_graph_s=parsed.get("cuda_graph_s", 0.0),
        total_ready_s=total_s,
        weight_mem_gb=parsed.get("weight_mem_gb", 0.0),
        graph_mem_gb=parsed.get("graph_mem_gb", 0.0),
        nccl_warmup_s=parsed.get("nccl_warmup_s"),
    )

    # Compute process spawn overhead
    spawn_overhead = total_s - parsed.get("first_to_ready_s", total_s)

    print(f"\n  Breakdown (self-reported):")
    print(f"    TP comm group:      {breakdown.tp_comm_group_s:.2f} s")
    print(f"    Weight load:        {breakdown.weight_load_s:.2f} s")
    print(f"    Memory pool/KV:     {breakdown.memory_pool_s:.3f} s")
    print(f"    CUDA graph total:   {breakdown.cuda_graph_s:.2f} s")
    print(f"      - standard:       {parsed.get('ts_cuda_graph_standard_s', parsed.get('cuda_graph_s', 0) - parsed.get('piecewise_cuda_graph_s', 0)):.2f} s")
    print(f"      - piecewise:      {parsed.get('piecewise_cuda_graph_s', 0):.2f} s")
    print(f"    Process spawn:      {spawn_overhead:.2f} s")
    print(f"    Total ready:        {breakdown.total_ready_s:.2f} s")
    print(f"    Weight mem:         {breakdown.weight_mem_gb:.2f} GB")
    print(f"    Graph mem:          {breakdown.graph_mem_gb:.2f} GB")

    return breakdown


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", default="/models/Qwen3-32B")
    p.add_argument("--tp", type=int, default=8)
    p.add_argument("--port", type=int, default=39900)
    p.add_argument("--nccl-port", type=int, default=39910)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--timeout-s", type=int, default=300)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/startup_breakdown.json"),
    )
    p.add_argument("--extra-args", nargs=argparse.REMAINDER, default=[])
    return p.parse_args()


def main():
    args = parse_args()
    log_dir = args.output.parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    results: List[StartupBreakdown] = []

    for trial in range(1, args.repeats + 1):
        breakdown = run_trial(
            trial=trial,
            model_path=args.model_path,
            tp_size=args.tp,
            port=args.port,
            nccl_port=args.nccl_port,
            extra_args=args.extra_args,
            log_dir=log_dir,
            timeout_s=args.timeout_s,
        )
        results.append(breakdown)

    # Compute summary statistics
    import statistics

    tp_comm_vals = [r.tp_comm_group_s for r in results]
    weight_vals = [r.weight_load_s for r in results]
    pool_vals = [r.memory_pool_s for r in results]
    cuda_graph_vals = [r.cuda_graph_s for r in results]
    total_vals = [r.total_ready_s for r in results]

    summary = {
        "model": args.model_path,
        "tp_size": args.tp,
        "num_trials": args.repeats,
        "breakdown_avg": {
            "tp_comm_group_s": statistics.mean(tp_comm_vals),
            "weight_load_s": statistics.mean(weight_vals),
            "memory_pool_kv_cache_s": statistics.mean(pool_vals),
            "cuda_graph_capture_s": statistics.mean(cuda_graph_vals),
            "total_ready_s": statistics.mean(total_vals),
        },
        "breakdown_std": {
            "tp_comm_group_s": statistics.stdev(tp_comm_vals) if len(tp_comm_vals) > 1 else 0,
            "weight_load_s": statistics.stdev(weight_vals) if len(weight_vals) > 1 else 0,
            "memory_pool_kv_cache_s": statistics.stdev(pool_vals) if len(pool_vals) > 1 else 0,
            "cuda_graph_capture_s": statistics.stdev(cuda_graph_vals) if len(cuda_graph_vals) > 1 else 0,
            "total_ready_s": statistics.stdev(total_vals) if len(total_vals) > 1 else 0,
        },
        "trials": [asdict(r) for r in results],
    }

    # Paper-ready output
    print(f"\n{'='*60}")
    print(f"  SUMMARY: {args.model_path} TP={args.tp} ({args.repeats} trials)")
    print(f"{'='*60}")
    print(f"  Build TP comm group:       {summary['breakdown_avg']['tp_comm_group_s']:.2f} s "
          f"(±{summary['breakdown_std']['tp_comm_group_s']:.2f})")
    print(f"  Materialize weight+KV$:    {summary['breakdown_avg']['weight_load_s']:.2f} s "
          f"(±{summary['breakdown_std']['weight_load_s']:.2f})")
    print(f"  Init inference engine:     {summary['breakdown_avg']['cuda_graph_capture_s']:.2f} s "
          f"(±{summary['breakdown_std']['cuda_graph_capture_s']:.2f})")
    print(f"  Other (pool+http+misc):    {summary['breakdown_avg']['memory_pool_kv_cache_s']:.2f} s "
          f"(±{summary['breakdown_std']['memory_pool_kv_cache_s']:.2f})")
    print(f"  ─────────────────────────────────")
    print(f"  Total cold-start:          {summary['breakdown_avg']['total_ready_s']:.2f} s "
          f"(±{summary['breakdown_std']['total_ready_s']:.2f})")

    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Results saved to {args.output}")


if __name__ == "__main__":
    main()
