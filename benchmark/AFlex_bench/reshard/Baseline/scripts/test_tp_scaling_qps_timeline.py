#!/usr/bin/env python3
"""QPS workload timeline for optimized TP scaling.

Runs a continuous QPS workload while expanding from source TP to target TP via
IPC/NVLink shadow loading. It records every request's timing and whether it
landed on source, target, or the cutover gap.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import csv
import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import requests

from test_optimized_tp_scaling_ipc import EnvManagedServer, post_json
from test_traditional_tp_scaling import DEFAULT_LOG_DIR, DEFAULT_RESULTS_DIR, PhaseMetric, now_phase, parse_gpus


@dataclass
class TimelineRequest:
    request_id: int
    scheduled_s: float
    start_s: float
    end_s: float
    route: str
    success: bool
    status_code: int | None = None
    ttft_ms: float | None = None
    e2e_ms: float | None = None
    output_tokens: int = 0
    error: str | None = None


@dataclass
class GpuSample:
    t_s: float
    index: int
    memory_used_mib: int
    memory_total_mib: int
    utilization_gpu_pct: int


def query_gpu_samples(t_s: float) -> list[GpuSample]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,memory.total,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    out = subprocess.check_output(cmd, text=True)
    samples: list[GpuSample] = []
    for row in csv.reader(out.splitlines()):
        if len(row) < 4:
            continue
        samples.append(
            GpuSample(
                t_s=t_s,
                index=int(row[0].strip()),
                memory_used_mib=int(row[1].strip()),
                memory_total_mib=int(row[2].strip()),
                utilization_gpu_pct=int(row[3].strip()),
            )
        )
    return samples


def start_gpu_sampler(base_t: float, interval_s: float, stop: threading.Event, sink: list[GpuSample]) -> threading.Thread:
    def loop() -> None:
        while not stop.is_set():
            t_s = time.monotonic() - base_t
            try:
                sink.extend(query_gpu_samples(t_s))
            except Exception:
                pass
            stop.wait(interval_s)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th


class RouteState:
    def __init__(self, source_url: str) -> None:
        self._lock = threading.Lock()
        self.route = "source"
        self.url = source_url

    def set(self, route: str, url: str | None) -> None:
        with self._lock:
            self.route = route
            self.url = url

    def get(self) -> tuple[str, str | None]:
        with self._lock:
            return self.route, self.url


def send_one(req_id: int, scheduled_s: float, origin_t: float, router: RouteState, prompt: str, output_len: int, timeout_s: float) -> TimelineRequest:
    delay = origin_t + scheduled_s - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    start_s = time.monotonic() - origin_t
    route, url = router.get()
    if route == "cutover" or url is None:
        end_s = time.monotonic() - origin_t
        return TimelineRequest(req_id, scheduled_s, start_s, end_s, route, False, error="route_unavailable_during_cutover")

    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": output_len, "temperature": 0.0, "ignore_eos": True},
        "stream": True,
    }
    first_token_t: float | None = None
    token_count = 0
    status_code: int | None = None
    try:
        with requests.post(f"{url}/generate", json=payload, stream=True, timeout=timeout_s) as resp:
            status_code = resp.status_code
            if resp.status_code != 200:
                end_s = time.monotonic() - origin_t
                return TimelineRequest(req_id, scheduled_s, start_s, end_s, route, False, status_code=status_code, error=resp.text[:200])
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
        end_s = time.monotonic() - origin_t
        return TimelineRequest(
            req_id,
            scheduled_s,
            start_s,
            end_s,
            route,
            True,
            status_code=status_code,
            ttft_ms=(first_token_t - (origin_t + start_s)) * 1000 if first_token_t else None,
            e2e_ms=(end_s - start_s) * 1000,
            output_tokens=token_count,
        )
    except Exception as exc:  # noqa: BLE001
        end_s = time.monotonic() - origin_t
        return TimelineRequest(req_id, scheduled_s, start_s, end_s, route, False, status_code=status_code, error=repr(exc))


def start_workload(router: RouteState, qps: float, duration_s: float, prompt_len: int, output_len: int, timeout_s: float, origin_t: float, start_offset_s: float) -> tuple[concurrent.futures.ThreadPoolExecutor, list[concurrent.futures.Future[TimelineRequest]]]:
    interval = 1.0 / qps
    count = int(duration_s * qps)
    prompt = "x" * prompt_len
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=max(4, min(64, count)))
    futures = [
        executor.submit(send_one, i, start_offset_s + i * interval, origin_t, router, prompt, output_len, timeout_s)
        for i in range(count)
    ]
    return executor, futures


def summarize_requests(requests_: list[TimelineRequest]) -> dict[str, Any]:
    ok = [r for r in requests_ if r.success]
    failed = [r for r in requests_ if not r.success]
    ttfts = [r.ttft_ms for r in ok if r.ttft_ms is not None]
    e2es = [r.e2e_ms for r in ok if r.e2e_ms is not None]
    by_route: dict[str, dict[str, int]] = {}
    for r in requests_:
        slot = by_route.setdefault(r.route, {"success": 0, "fail": 0})
        slot["success" if r.success else "fail"] += 1
    return {
        "total": len(requests_),
        "success": len(ok),
        "failed": len(failed),
        "ttft_avg_ms": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50_ms": statistics.median(ttfts) if ttfts else None,
        "e2e_avg_ms": statistics.mean(e2es) if e2es else None,
        "by_route": by_route,
        "failures": [asdict(r) for r in failed],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", default=os.environ.get("SGLANG_MODEL", "/models/Qwen3-32B"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--source-tp", type=int, default=2)
    parser.add_argument("--target-tp", type=int, default=4)
    parser.add_argument("--source-gpus", type=parse_gpus, default=parse_gpus("0,1"))
    parser.add_argument("--target-gpus", type=parse_gpus, default=parse_gpus("2,3,4,5"))
    parser.add_argument("--visible-gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5"))
    parser.add_argument("--source-port", type=int, default=31300)
    parser.add_argument("--target-port", type=int, default=31310)
    parser.add_argument("--source-nccl-port", type=int, default=32400)
    parser.add_argument("--target-nccl-port", type=int, default=32410)
    parser.add_argument("--ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--qps", type=float, default=1.0)
    parser.add_argument("--workload-duration-s", type=float, default=90.0)
    parser.add_argument("--scale-at-s", type=float, default=15.0, help="When to start IPC export + target shadow launch relative to workload start")
    parser.add_argument("--cutover-s", type=float, default=2.0, help="Simulated visible route reset window after target is ready")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--export-timeout-s", type=float, default=120.0)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--ipc-dir", type=Path, default=Path("/tmp/sglang_reshard_qps"))
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "node1_tp2_to_tp4_ipc_qps1_timeline.json")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--keep-servers", action="store_true")
    parser.add_argument("--sample-gpu", action="store_true", help="Sample nvidia-smi memory/utilization into the output JSON")
    parser.add_argument("--gpu-sample-interval-s", type=float, default=0.5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(args.ipc_dir, ignore_errors=True)
    args.ipc_dir.mkdir(parents=True, exist_ok=True)

    target_base = args.visible_gpus.index(args.target_gpus[0])
    source_base = args.visible_gpus.index(args.source_gpus[0])
    common_kwargs = dict(
        model_path=args.model_path,
        host=args.host,
        log_dir=args.log_dir,
        extra_args=args.server_arg,
        python_bin=args.python,
        dry_run=False,
        visible_gpus=args.visible_gpus,
    )
    source = EnvManagedServer("qps_source", tp=args.source_tp, gpus=args.source_gpus, port=args.source_port, nccl_port=args.source_nccl_port, base_gpu_id=source_base, **common_kwargs)
    target_env = {
        "SGLANG_RESHARD_IPC_DIR": str(args.ipc_dir),
        "SGLANG_RESHARD_OLD_TP": str(args.source_tp),
        "SGLANG_RESHARD_MODULE_TYPE": "prefill",
        "SGLANG_RESHARD_PERSPECTIVE": "full",
        "SGLANG_RESHARD_IPC_AUTO_CLEANUP": "0",
    }
    target = EnvManagedServer("qps_target_ipc", tp=args.target_tp, gpus=args.target_gpus, port=args.target_port, nccl_port=args.target_nccl_port, base_gpu_id=target_base, extra_env=target_env, **common_kwargs)

    phases: list[PhaseMetric] = []
    events: list[dict[str, Any]] = []
    base_t = time.monotonic()
    gpu_samples: list[GpuSample] = []
    sampler_stop = threading.Event()
    sampler_thread = start_gpu_sampler(base_t, args.gpu_sample_interval_s, sampler_stop, gpu_samples) if args.sample_gpu else None
    router = RouteState(source.base_url)
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    futures: list[concurrent.futures.Future[TimelineRequest]] = []

    try:
        _, phase = now_phase("start_source_tp", base_t, lambda: (source.start(), source.wait_ready(args.ready_timeout_s)))
        phases.append(phase)
        events.append({"name": "source_ready", "time_s": phase.end_s})

        workload_start_s = time.monotonic() - base_t
        executor, futures = start_workload(
            router,
            args.qps,
            args.workload_duration_s,
            args.prompt_len,
            args.output_len,
            args.request_timeout_s,
            base_t,
            workload_start_s,
        )
        events.append({"name": "workload_start", "time_s": workload_start_s, "qps": args.qps})

        scale_delay = base_t + args.scale_at_s - time.monotonic()
        if scale_delay > 0:
            time.sleep(scale_delay)
        events.append({"name": "scale_start", "time_s": time.monotonic() - base_t})

        def export_ipc() -> dict[str, Any]:
            return post_json(
                f"{source.base_url}/admin/export_weights_ipc",
                {"ipc_dir": str(args.ipc_dir), "module_type": "prefill", "perspective": "full"},
                args.export_timeout_s,
            )

        export_response, phase = now_phase("export_source_ipc_handles_online", base_t, export_ipc)
        phases.append(phase)
        events.append({"name": "ipc_export_done", "time_s": phase.end_s, "response": export_response})

        _, phase = now_phase("start_target_tp_from_ipc_nvlink", base_t, lambda: (target.start(), target.wait_ready(args.ready_timeout_s)))
        phases.append(phase)
        events.append({"name": "target_ready", "time_s": phase.end_s})

        cutover_start = time.monotonic() - base_t
        router.set("cutover", None)
        events.append({"name": "cutover_start", "time_s": cutover_start})
        time.sleep(args.cutover_s)
        cutover_end = time.monotonic() - base_t
        router.set("target", target.base_url)
        events.append({"name": "cutover_end", "time_s": cutover_end})
        phases.append(PhaseMetric("visible_cutover_gap", cutover_start, cutover_end, cutover_end - cutover_start))

        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        results.sort(key=lambda r: r.request_id)
        workload_end_s = max((r.end_s for r in results), default=time.monotonic() - base_t)
        events.append({"name": "workload_done", "time_s": workload_end_s})

        _, phase = now_phase("stop_source_tp", base_t, source.stop)
        phases.append(phase)
        total_s = time.monotonic() - base_t

        record = {
            "scheme": "optimized_sglang_tp_ipc_nvlink_qps_timeline",
            "scenario": f"TP{args.source_tp}->TP{args.target_tp}",
            "model_path": args.model_path,
            "source_tp": args.source_tp,
            "target_tp": args.target_tp,
            "source_gpus": args.source_gpus,
            "target_gpus": args.target_gpus,
            "visible_gpus": args.visible_gpus,
            "qps": args.qps,
            "workload_duration_s": args.workload_duration_s,
            "scale_at_s": args.scale_at_s,
            "cutover_s": args.cutover_s,
            "prompt_len": args.prompt_len,
            "output_len": args.output_len,
            "total_s": total_s,
            "phases": [asdict(p) for p in phases],
            "events": events,
            "requests": [asdict(r) for r in results],
            "summary": summarize_requests(results),
        }
        args.output.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps({"output": str(args.output), "summary": record["summary"], "events": events}, indent=2, ensure_ascii=False))
        return 0
    finally:
        if args.sample_gpu:
            sampler_stop.set()
            if sampler_thread is not None:
                sampler_thread.join(timeout=2)
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=False)
        if not args.keep_servers:
            target.stop()
            source.stop()
        shutil.rmtree(args.ipc_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
