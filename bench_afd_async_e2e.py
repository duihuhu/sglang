#!/usr/bin/env python3
"""End-to-end AF-only benchmark for --afd-async-schedule (IPC, M=3).

Compares against the v4 baseline reported in
``python/sglang/srt/energy/versions/version4.md``:
    M=3 UCX → 118.6 output tok/s

Run params match v4 §2.1 (Test Configuration) as closely as the AF-only
setup allows:
    Model            Qwen3-32B (BF16)
    DA TP=2 on GPU 0,1   DF TP=2 on GPU 2,3
    --afd-comm-backend ipc
    max_running 64, qps 8, input 1024, output 128, 100 prompts

Two phases per backend:
    1. Boot DA + DF, wait for /health
    2. Run sglang.bench_serving --dataset-name random ...
       Capture output_tps / TTFT / TPOT.

Usage:
    /workspace/env/af-test/bin/python bench_afd_async_e2e.py \
        --variant async-on   # uses --afd-async-schedule
    /workspace/env/af-test/bin/python bench_afd_async_e2e.py \
        --variant baseline   # legacy schedule (for IPC-baseline comparison)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path("/workspace/sglang")
PYTHON = "/workspace/env/af-test/bin/python"
MODEL = "/models/Qwen/Qwen3-32B/"

SHARED_GPUS = "0,1,2,3"
DA_BASE_GPU = 0
DF_BASE_GPU = 2
DA_PEER_OFFSET = +2
DF_PEER_OFFSET = -2
TP = 2

DA_PORT = 30100
DF_PORT = 30101
SCHED_PORT = 65300

DEFAULT_LOG_BASE = Path("/tmp/afd_async_e2e")


def free_port(p: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", p))
        except OSError:
            return False
    return True


def cleanup_old():
    subprocess.call(
        ["pkill", "-9", "-f", "sglang.launch_server"],
        stderr=subprocess.DEVNULL,
    )
    time.sleep(2)
    for d in ("/tmp", "/dev/shm"):
        try:
            for f in os.listdir(d):
                if f.startswith("afd_ipc"):
                    try:
                        os.unlink(os.path.join(d, f))
                    except Exception:
                        pass
        except Exception:
            pass


def build_cmd(role: str, base_gpu: int, port: int,
              micro_batch: int, async_sched: bool,
              max_running: int) -> list:
    cmd = [
        PYTHON, "-m", "sglang.launch_server",
        "--model-path", MODEL,
        "--tp", str(TP),
        "--base-gpu-id", str(base_gpu),
        "--host", "0.0.0.0",
        "--port", str(port),
        "--afd-perspective", role,
        "--afd-comm-backend", "ipc",
        "--afd-micro-batch", str(micro_batch),
        "--mem-fraction-static", "0.85",
        "--max-running-requests", str(max_running),
        "--trust-remote-code",
        "--skip-server-warmup",
    ]
    if async_sched:
        cmd.append("--afd-async-schedule")
    return cmd


def launch(role: str, base_gpu: int, peer_offset: int, port: int,
           log_dir: Path, micro_batch: int, async_sched: bool,
           max_running: int) -> subprocess.Popen:
    log = log_dir / f"{role}.log"
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": SHARED_GPUS,
        "AFD_SCHED_HOST": "127.0.0.1",
        "AFD_SCHED_PORT": str(SCHED_PORT),
        "AFD_IPC_PEER_OFFSET": str(peer_offset),
    }
    cmd = build_cmd(role, base_gpu, port, micro_batch, async_sched, max_running)
    print(f"[bench] launching {role}: base_gpu={base_gpu} "
          f"peer_offset={peer_offset:+d} port={port} "
          f"M={micro_batch} async={async_sched}", flush=True)
    print(f"  log: {log}", flush=True)
    fh = open(log, "w")
    proc = subprocess.Popen(
        cmd, stdout=fh, stderr=subprocess.STDOUT, env=env,
        preexec_fn=os.setsid,
    )
    return proc


def wait_ready(url: str, proc: subprocess.Popen, timeout: float) -> bool:
    deadline = time.time() + timeout
    last = 0.0
    while time.time() < deadline:
        if proc.poll() is not None:
            print(f"  [bench] {url} process exited code={proc.returncode}",
                  flush=True)
            return False
        try:
            urllib.request.urlopen(f"{url}/health", timeout=3)
            return True
        except Exception:
            pass
        if time.time() - last > 15:
            print(f"  [bench] still waiting for {url} "
                  f"({int(time.time() - (deadline - timeout))}s)",
                  flush=True)
            last = time.time()
        time.sleep(2)
    return False


def kill_process(p: subprocess.Popen):
    try:
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:
        pass


def run_bench(num_prompts: int, max_concurrency: int,
              input_len: int, output_len: int,
              qps: float, da_url: str, log_dir: Path) -> dict:
    """Run benchmark_replay.py --dataset sample against the DA endpoint.

    Returns a dict with throughput / latency metrics.
    """
    dump = log_dir / "bench_dump.json"
    cmd = [
        PYTHON,
        str(REPO / "benchmark/test_motivation/AzurePublicDataset/benchmark_replay.py"),
        "--dataset", "sample",
        "--url", da_url,
        "--max-requests", str(num_prompts),
        "--sample-input-len", str(input_len),
        "--sample-output-len", str(output_len),
        "--sample-qps", str(qps),
        "--concurrency", str(max_concurrency),
        "--timeout", "300",
        "--dump", str(dump),
        "--scenario-label", "afd-async",
        "--seed", "42",
    ]
    print("[bench] running:", " ".join(cmd), flush=True)
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
    print(out.stdout[-3000:], flush=True)
    if out.returncode != 0:
        print("[bench] stderr:", out.stderr[-1000:], flush=True)
        return {"ok": False, "stdout": out.stdout, "stderr": out.stderr}

    text = out.stdout

    def grab(label):
        m = re.search(rf"{re.escape(label)}\s*[:|]?\s*([0-9.]+)", text)
        return float(m.group(1)) if m else None

    return {
        "ok": True,
        "output_tps": grab("Output throughput (tok/s)") or grab("Output tps"),
        "input_tps": grab("Input throughput (tok/s)") or grab("Input tps"),
        "ttft_ms_mean": grab("Mean TTFT (ms)"),
        "tpot_ms_mean": grab("Mean TPOT (ms)"),
        "request_throughput": grab("Request throughput (req/s)"),
        "completed": grab("Total requests"),
        "raw_tail": text[-4000:],
        "dump_path": str(dump),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["async-on", "baseline"],
                    default="async-on")
    ap.add_argument("--micro-batch", type=int, default=3)
    ap.add_argument("--max-running", type=int, default=64)
    ap.add_argument("--num-prompts", type=int, default=100)
    ap.add_argument("--max-concurrency", type=int, default=64)
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--qps", type=float, default=8.0)
    ap.add_argument("--server-timeout", type=float, default=600)
    args = ap.parse_args()

    async_sched = args.variant == "async-on"
    log_dir = DEFAULT_LOG_BASE / args.variant
    log_dir.mkdir(parents=True, exist_ok=True)

    cleanup_old()
    if not (free_port(DA_PORT) and free_port(DF_PORT)):
        print("[bench] ports busy, retry cleanup", flush=True)
        cleanup_old()

    da_url = f"http://127.0.0.1:{DA_PORT}"
    df = launch("ffn", DF_BASE_GPU, DF_PEER_OFFSET, DF_PORT, log_dir,
                args.micro_batch, async_sched, args.max_running)
    time.sleep(3)
    da = launch("attn", DA_BASE_GPU, DA_PEER_OFFSET, DA_PORT, log_dir,
                args.micro_batch, async_sched, args.max_running)

    metrics: dict = {"variant": args.variant, "args": vars(args)}
    try:
        if not wait_ready(da_url, da, timeout=args.server_timeout):
            raise RuntimeError("DA did not come up")
        if df.poll() is not None:
            raise RuntimeError(f"DF exited (code {df.returncode})")
        time.sleep(5)  # give IPC handshakes time to settle

        result = run_bench(
            num_prompts=args.num_prompts,
            max_concurrency=args.max_concurrency,
            input_len=args.input_len,
            output_len=args.output_len,
            qps=args.qps,
            da_url=da_url,
            log_dir=log_dir,
        )
        metrics.update(result)
    finally:
        print("[bench] tearing down", flush=True)
        for p in [da, df]:
            kill_process(p)
        time.sleep(3)
        for p in [da, df]:
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGKILL)
            except Exception:
                pass
        cleanup_old()

    out_path = log_dir / "metrics.json"
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n[bench] results saved to {out_path}", flush=True)
    print(json.dumps({k: v for k, v in metrics.items()
                      if k not in ("raw_tail",)},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
