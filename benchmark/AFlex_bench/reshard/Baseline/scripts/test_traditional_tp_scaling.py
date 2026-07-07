#!/usr/bin/env python3
"""Traditional SGLang tensor-parallel scaling baseline.

This script measures the classic whole-instance TP migration flow used by
systems such as DynamoLLM: keep the old TP instance serving, start a new
SGLang instance with a larger TP degree, wait until the new instance is ready,
drain/cut over traffic, then terminate the old instance.

It intentionally does not use AFlex component-level reconnect or partial
resharding. The unit of scaling is the whole SGLang server process group.

Example:
    python3 benchmark/AFlex_bench/reshard/Baseline/scripts/test_traditional_tp_scaling.py \
        --model-path /models/Qwen3-32B \
        --source-tp 2 --source-gpus 0,1 \
        --target-tp 4 --target-gpus 0,1,2,3 \
        --output benchmark/AFlex_bench/reshard/Baseline/results/tp2_to_tp4.jsonl

For CI or machines without GPUs, validate the harness with:
    python3 .../test_traditional_tp_scaling.py --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    import requests
except ImportError as exc:  # pragma: no cover - environment issue
    raise SystemExit("requests is required to run this benchmark") from exc


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_RESULTS_DIR = REPO_ROOT / "benchmark/AFlex_bench/reshard/Baseline/results"
DEFAULT_LOG_DIR = REPO_ROOT / "benchmark/AFlex_bench/reshard/Baseline/logs"


@dataclass
class RequestMetric:
    success: bool
    url: str
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int = 0
    error: str | None = None


@dataclass
class PhaseMetric:
    name: str
    start_s: float
    end_s: float
    duration_s: float


class ManagedServer:
    def __init__(
        self,
        name: str,
        model_path: str,
        tp: int,
        gpus: list[int],
        host: str,
        port: int,
        nccl_port: int,
        log_dir: Path,
        extra_args: list[str],
        python_bin: str,
        dry_run: bool,
    ) -> None:
        self.name = name
        self.model_path = model_path
        self.tp = tp
        self.gpus = gpus
        self.host = host
        self.port = port
        self.nccl_port = nccl_port
        self.log_dir = log_dir
        self.extra_args = extra_args
        self.python_bin = python_bin
        self.dry_run = dry_run
        self.proc: subprocess.Popen[str] | None = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self) -> None:
        if self.dry_run:
            print(f"[dry-run] start {self.name}: TP{self.tp} on GPUs {self.gpus} port={self.port}")
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"{self.name}_tp{self.tp}_p{self.port}.log"
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in self.gpus)
        cmd = [
            self.python_bin,
            "-m",
            "sglang.launch_server",
            "--model-path",
            self.model_path,
            "--tp",
            str(self.tp),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--nccl-port",
            str(self.nccl_port),
            "--mem-fraction-static",
            "0.85",
            "--disable-cuda-graph",
            "--disable-piecewise-cuda-graph",
            "--skip-server-warmup",
            *self.extra_args,
        ]
        log = log_file.open("w")
        self.proc = subprocess.Popen(
            cmd,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        print(f"started {self.name} pid={self.proc.pid} log={log_file}")

    def wait_ready(self, timeout_s: float, poll_s: float = 2.0) -> float:
        if self.dry_run:
            simulated = 0.15 if self.name == "source" else 0.30
            time.sleep(simulated)
            return simulated
        start = time.monotonic()
        deadline = start + timeout_s
        last_error = ""
        while time.monotonic() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited before ready with code {self.proc.returncode}")
            try:
                resp = requests.get(f"{self.base_url}/health", timeout=2)
                if resp.status_code == 200:
                    return time.monotonic() - start
                last_error = f"HTTP {resp.status_code}"
            except Exception as exc:  # noqa: BLE001 - diagnostic only
                last_error = str(exc)
            time.sleep(poll_s)
        raise TimeoutError(f"{self.name} did not become healthy within {timeout_s}s: {last_error}")

    def stop(self, grace_s: float = 10.0) -> None:
        if self.dry_run:
            print(f"[dry-run] stop {self.name}")
            return
        if self.proc is None or self.proc.poll() is not None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
            self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=5)


def now_phase(name: str, t0: float, fn) -> tuple[Any, PhaseMetric]:
    start = time.monotonic()
    result = fn()
    end = time.monotonic()
    return result, PhaseMetric(name=name, start_s=start - t0, end_s=end - t0, duration_s=end - start)


def make_prompt(prompt_len: int) -> str:
    # Character length is sufficient for a stable synthetic benchmark prompt.
    return "x" * prompt_len


def send_generate(url: str, prompt: str, output_len: int, timeout_s: float, dry_run: bool) -> RequestMetric:
    if dry_run:
        base = 450.0 + len(prompt) * 0.02 + output_len * 1.5
        return RequestMetric(success=True, url=url, ttft_ms=base, e2e_ms=base + output_len * 12, output_tokens=output_len)
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0, "ignore_eos": True},
        "stream": True,
    }
    t_start = time.monotonic()
    first_token_t: float | None = None
    token_count = 0
    try:
        with requests.post(f"{url}/generate", json=payload, stream=True, timeout=timeout_s) as resp:
            if resp.status_code != 200:
                return RequestMetric(False, url=url, error=f"HTTP {resp.status_code}: {resp.text[:200]}")
            for raw in resp.iter_lines():
                if not raw:
                    continue
                line = raw.decode("utf-8", errors="replace").strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                if first_token_t is None:
                    first_token_t = time.monotonic()
                token_count += 1
    except Exception as exc:  # noqa: BLE001 - benchmark should record failures
        return RequestMetric(False, url=url, error=repr(exc))
    t_end = time.monotonic()
    return RequestMetric(
        success=True,
        url=url,
        ttft_ms=(first_token_t - t_start) * 1000 if first_token_t else None,
        e2e_ms=(t_end - t_start) * 1000,
        output_tokens=token_count,
    )


def run_probe_batch(url: str, num_requests: int, concurrency: int, prompt_len: int, output_len: int, timeout_s: float, dry_run: bool) -> list[RequestMetric]:
    prompt = make_prompt(prompt_len)
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(send_generate, url, prompt, output_len, timeout_s, dry_run) for _ in range(num_requests)]
        return [f.result() for f in concurrent.futures.as_completed(futures)]


def summarize_requests(metrics: list[RequestMetric]) -> dict[str, Any]:
    ok = [m for m in metrics if m.success]
    ttfts = [m.ttft_ms for m in ok if m.ttft_ms is not None]
    e2es = [m.e2e_ms for m in ok if m.e2e_ms is not None]
    return {
        "requests": len(metrics),
        "success": len(ok),
        "failures": len(metrics) - len(ok),
        "ttft_avg_ms": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else None,
        "e2e_avg_ms": statistics.mean(e2es) if e2es else None,
    }


def parse_gpus(value: str) -> list[int]:
    gpus = [int(x) for x in value.split(",") if x.strip()]
    if not gpus:
        raise argparse.ArgumentTypeError("GPU list cannot be empty")
    return gpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", default=os.environ.get("SGLANG_MODEL", "/models/Qwen3-32B"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--source-tp", type=int, default=2)
    parser.add_argument("--target-tp", type=int, default=4)
    parser.add_argument("--source-gpus", type=parse_gpus, default=parse_gpus("0,1"))
    parser.add_argument("--target-gpus", type=parse_gpus, default=parse_gpus("0,1,2,3"))
    parser.add_argument("--source-port", type=int, default=31000)
    parser.add_argument("--target-port", type=int, default=31010)
    parser.add_argument("--source-nccl-port", type=int, default=32100)
    parser.add_argument("--target-nccl-port", type=int, default=32110)
    parser.add_argument("--ready-timeout-s", type=float, default=600.0)
    parser.add_argument("--drain-s", type=float, default=5.0, help="Router drain window before terminating the old TP instance")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--server-arg", action="append", default=[], help="Extra argument forwarded to sglang.launch_server; repeatable")
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "traditional_tp_scaling.jsonl")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--keep-servers", action="store_true", help="Leave launched servers running after the benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Validate the harness without launching SGLang")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.source_gpus) != args.source_tp:
        raise SystemExit("--source-gpus length must match --source-tp")
    if len(args.target_gpus) != args.target_tp:
        raise SystemExit("--target-gpus length must match --target-tp")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    phases: list[PhaseMetric] = []
    source = ManagedServer("source", args.model_path, args.source_tp, args.source_gpus, args.host, args.source_port, args.source_nccl_port, args.log_dir, args.server_arg, args.python, args.dry_run)
    target = ManagedServer("target", args.model_path, args.target_tp, args.target_gpus, args.host, args.target_port, args.target_nccl_port, args.log_dir, args.server_arg, args.python, args.dry_run)
    before_metrics: list[RequestMetric] = []
    after_metrics: list[RequestMetric] = []

    try:
        _, phase = now_phase("start_source_tp", t0, lambda: (source.start(), source.wait_ready(args.ready_timeout_s)))
        phases.append(phase)
        before_metrics, phase = now_phase(
            "probe_source_before_scale",
            t0,
            lambda: run_probe_batch(source.base_url, args.num_requests, args.concurrency, args.prompt_len, args.output_len, args.request_timeout_s, args.dry_run),
        )
        phases.append(phase)

        _, phase = now_phase("start_target_tp_and_load_model", t0, lambda: (target.start(), target.wait_ready(args.ready_timeout_s)))
        phases.append(phase)

        _, phase = now_phase("drain_old_instance", t0, lambda: time.sleep(args.drain_s if not args.dry_run else min(args.drain_s, 0.1)))
        phases.append(phase)

        cutover_at = time.monotonic() - t0
        after_metrics, phase = now_phase(
            "probe_target_after_cutover",
            t0,
            lambda: run_probe_batch(target.base_url, args.num_requests, args.concurrency, args.prompt_len, args.output_len, args.request_timeout_s, args.dry_run),
        )
        phases.append(phase)

        _, phase = now_phase("stop_source_tp", t0, source.stop)
        phases.append(phase)
        total_s = time.monotonic() - t0

        record = {
            "scheme": "traditional_sglang_tp_whole_instance",
            "scenario": f"TP{args.source_tp}->TP{args.target_tp}",
            "model_path": args.model_path,
            "source_tp": args.source_tp,
            "target_tp": args.target_tp,
            "source_gpus": args.source_gpus,
            "target_gpus": args.target_gpus,
            "prompt_len": args.prompt_len,
            "output_len": args.output_len,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "dry_run": args.dry_run,
            "cutover_at_s": cutover_at,
            "total_s": total_s,
            "phases": [asdict(p) for p in phases],
            "before": summarize_requests(before_metrics),
            "after": summarize_requests(after_metrics),
            "request_details": {
                "before": [asdict(m) for m in before_metrics],
                "after": [asdict(m) for m in after_metrics],
            },
        }
        with args.output.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(json.dumps({k: record[k] for k in ("scenario", "total_s", "cutover_at_s", "before", "after")}, indent=2))
        print(f"wrote {args.output}")
        return 0
    finally:
        if not args.keep_servers:
            target.stop()
            source.stop()


if __name__ == "__main__":
    raise SystemExit(main())
