#!/usr/bin/env python3
"""Optimized SGLang TP scaling baseline with IPC/NVLink weight inheritance.

The old TP instance keeps serving while it exports CUDA IPC handles for its
resident GPU weights. The target TP instance starts in the background, skips
safetensors disk IO with the dummy loader, opens the old GPU tensors through
CUDA IPC, re-slices them for the new TP degree, and copies them to the target
GPUs over NVLink/P2P. Only the final drain/cutover is counted as visible
interruption.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import requests

from test_traditional_tp_scaling import (
    DEFAULT_LOG_DIR,
    DEFAULT_RESULTS_DIR,
    ManagedServer,
    PhaseMetric,
    RequestMetric,
    now_phase,
    parse_gpus,
    run_probe_batch,
    summarize_requests,
)


def post_json(url: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    resp = requests.post(url, json=payload, timeout=timeout_s)
    try:
        body = resp.json()
    except Exception:
        body = {"text": resp.text}
    if resp.status_code != 200 or body.get("success") is False:
        raise RuntimeError(f"POST {url} failed: status={resp.status_code} body={body}")
    return body


class EnvManagedServer(ManagedServer):
    def __init__(self, *args, extra_env: dict[str, str] | None = None, base_gpu_id: int | None = None, visible_gpus: list[int] | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.extra_env = extra_env or {}
        self.base_gpu_id = base_gpu_id
        self.visible_gpus = visible_gpus

    def start(self) -> None:
        if self.dry_run:
            print(f"[dry-run] start {self.name}: TP{self.tp} on GPUs {self.gpus} port={self.port}")
            return
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_file = self.log_dir / f"{self.name}_tp{self.tp}_p{self.port}.log"
        env = os.environ.copy()
        env.update(self.extra_env)
        cvd = self.visible_gpus if self.visible_gpus is not None else self.gpus
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in cvd)
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
        ]
        if self.base_gpu_id is not None:
            cmd.extend(["--base-gpu-id", str(self.base_gpu_id)])
        cmd.extend(self.extra_args)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--model-path", default=os.environ.get("SGLANG_MODEL", "/models/Qwen3-32B"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--source-tp", type=int, default=2)
    parser.add_argument("--target-tp", type=int, default=4)
    parser.add_argument("--source-gpus", type=parse_gpus, default=parse_gpus("0,1"))
    parser.add_argument("--target-gpus", type=parse_gpus, default=parse_gpus("2,3,4,5"))
    parser.add_argument("--visible-gpus", type=parse_gpus, default=parse_gpus("0,1,2,3,4,5"), help="GPUs visible to both processes so CUDA IPC device indices stay valid")
    parser.add_argument("--source-port", type=int, default=31200)
    parser.add_argument("--target-port", type=int, default=31210)
    parser.add_argument("--source-nccl-port", type=int, default=32300)
    parser.add_argument("--target-nccl-port", type=int, default=32310)
    parser.add_argument("--ready-timeout-s", type=float, default=900.0)
    parser.add_argument("--drain-s", type=float, default=1.0, help="Visible drain/cutover window after target is ready")
    parser.add_argument("--prompt-len", type=int, default=2048)
    parser.add_argument("--output-len", type=int, default=32)
    parser.add_argument("--num-requests", type=int, default=4)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--request-timeout-s", type=float, default=300.0)
    parser.add_argument("--export-timeout-s", type=float, default=120.0)
    parser.add_argument("--python", default=os.environ.get("PYTHON", "python3"))
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument("--ipc-dir", type=Path, default=Path("/tmp/sglang_reshard_baseline"))
    parser.add_argument("--output", type=Path, default=DEFAULT_RESULTS_DIR / "optimized_tp_scaling_ipc.jsonl")
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--keep-servers", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if len(args.source_gpus) != args.source_tp:
        raise SystemExit("--source-gpus length must match --source-tp")
    if len(args.target_gpus) != args.target_tp:
        raise SystemExit("--target-gpus length must match --target-tp")
    if args.target_gpus[0] not in args.visible_gpus:
        raise SystemExit("--visible-gpus must include target GPUs")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    if not args.dry_run:
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
        dry_run=args.dry_run,
        visible_gpus=args.visible_gpus,
    )
    source = EnvManagedServer(
        "source",
        tp=args.source_tp,
        gpus=args.source_gpus,
        port=args.source_port,
        nccl_port=args.source_nccl_port,
        base_gpu_id=source_base,
        **common_kwargs,
    )
    target_env = {
        "SGLANG_RESHARD_IPC_DIR": str(args.ipc_dir),
        "SGLANG_RESHARD_OLD_TP": str(args.source_tp),
        "SGLANG_RESHARD_MODULE_TYPE": "prefill",
        "SGLANG_RESHARD_PERSPECTIVE": "full",
        "SGLANG_RESHARD_IPC_AUTO_CLEANUP": "0",
    }
    target = EnvManagedServer(
        "target_ipc",
        tp=args.target_tp,
        gpus=args.target_gpus,
        port=args.target_port,
        nccl_port=args.target_nccl_port,
        base_gpu_id=target_base,
        extra_env=target_env,
        **common_kwargs,
    )

    t0 = time.monotonic()
    phases: list[PhaseMetric] = []
    before_metrics: list[RequestMetric] = []
    after_metrics: list[RequestMetric] = []
    export_response: dict[str, Any] | None = None

    try:
        _, phase = now_phase("start_source_tp", t0, lambda: (source.start(), source.wait_ready(args.ready_timeout_s)))
        phases.append(phase)
        before_metrics, phase = now_phase(
            "probe_source_before_scale",
            t0,
            lambda: run_probe_batch(source.base_url, args.num_requests, args.concurrency, args.prompt_len, args.output_len, args.request_timeout_s, args.dry_run),
        )
        phases.append(phase)

        def export_ipc() -> dict[str, Any]:
            if args.dry_run:
                time.sleep(0.05)
                return {"success": True, "elapsed_s": 0.05, "message": "dry-run"}
            return post_json(
                f"{source.base_url}/admin/export_weights_ipc",
                {"ipc_dir": str(args.ipc_dir), "module_type": "prefill", "perspective": "full"},
                args.export_timeout_s,
            )

        export_response, phase = now_phase("export_source_ipc_handles_online", t0, export_ipc)
        phases.append(phase)

        _, phase = now_phase("start_target_tp_from_ipc_nvlink", t0, lambda: (target.start(), target.wait_ready(args.ready_timeout_s)))
        phases.append(phase)

        # This is the visible interruption window in the optimized baseline: new
        # requests stop going to the old endpoint and are switched to target.
        _, phase = now_phase("visible_drain_and_cutover", t0, lambda: time.sleep(args.drain_s if not args.dry_run else min(args.drain_s, 0.05)))
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
        phase_map = {p.name: p.duration_s for p in phases}
        visible_downtime_s = phase_map.get("visible_drain_and_cutover", 0.0)
        async_prepare_s = phase_map.get("export_source_ipc_handles_online", 0.0) + phase_map.get("start_target_tp_from_ipc_nvlink", 0.0)

        record = {
            "scheme": "optimized_sglang_tp_ipc_nvlink_shadow",
            "scenario": f"TP{args.source_tp}->TP{args.target_tp}",
            "model_path": args.model_path,
            "source_tp": args.source_tp,
            "target_tp": args.target_tp,
            "source_gpus": args.source_gpus,
            "target_gpus": args.target_gpus,
            "visible_gpus": args.visible_gpus,
            "ipc_dir": str(args.ipc_dir),
            "prompt_len": args.prompt_len,
            "output_len": args.output_len,
            "num_requests": args.num_requests,
            "concurrency": args.concurrency,
            "dry_run": args.dry_run,
            "cutover_at_s": cutover_at,
            "visible_downtime_s": visible_downtime_s,
            "async_prepare_s": async_prepare_s,
            "total_s": total_s,
            "export_response": export_response,
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
        print(json.dumps({k: record[k] for k in ("scenario", "total_s", "async_prepare_s", "visible_downtime_s", "before", "after")}, indent=2))
        print(f"wrote {args.output}")
        return 0
    finally:
        if not args.keep_servers:
            target.stop()
            source.stop()
        if not args.dry_run:
            shutil.rmtree(args.ipc_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
