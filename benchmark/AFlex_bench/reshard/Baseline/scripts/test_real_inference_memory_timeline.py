#!/usr/bin/env python3
"""Real SGLang inference workload + GPU memory timeline sampler.

This helper does not implement in-place TP reshard by itself. It is a verifier
for any real serving path: start/reshard servers externally, point this script at
the active HTTP endpoint(s), and it will continuously send /generate requests
while sampling nvidia-smi memory/utilization. Use it to validate that requests
really go through prefill/decode and that weights/KV cache are resident.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import requests


@dataclass
class RequestRecord:
    request_id: int
    scheduled_s: float
    start_s: float
    end_s: float
    url: str
    status: str
    http_status: Optional[int]
    error: Optional[str]
    ttft_s: Optional[float]
    e2e_s: Optional[float]
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]


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
    rows: list[GpuSample] = []
    for row in csv.reader(out.splitlines()):
        if len(row) < 4:
            continue
        rows.append(
            GpuSample(
                t_s=t_s,
                index=int(row[0].strip()),
                memory_used_mib=int(row[1].strip()),
                memory_total_mib=int(row[2].strip()),
                utilization_gpu_pct=int(row[3].strip()),
            )
        )
    return rows


def start_sampler(origin_t: float, interval_s: float, stop: threading.Event, sink: list[GpuSample]) -> threading.Thread:
    def loop() -> None:
        while not stop.is_set():
            t_s = time.monotonic() - origin_t
            try:
                sink.extend(query_gpu_samples(t_s))
            except Exception:
                pass
            stop.wait(interval_s)

    th = threading.Thread(target=loop, daemon=True)
    th.start()
    return th


def send_generate(req_id: int, scheduled_s: float, origin_t: float, url: str, prompt: str, output_len: int, timeout_s: float) -> RequestRecord:
    delay = origin_t + scheduled_s - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    start_s = time.monotonic() - origin_t
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": output_len, "temperature": 0},
    }
    try:
        r = requests.post(url.rstrip("/") + "/generate", json=payload, timeout=timeout_s)
        end_s = time.monotonic() - origin_t
        meta = {}
        try:
            meta = r.json().get("meta_info", {})
        except Exception:
            pass
        status = "ok" if r.status_code == 200 else "http_error"
        return RequestRecord(
            request_id=req_id,
            scheduled_s=scheduled_s,
            start_s=start_s,
            end_s=end_s,
            url=url,
            status=status,
            http_status=r.status_code,
            error=None if status == "ok" else r.text[:500],
            ttft_s=meta.get("ttft_pure_processing"),
            e2e_s=meta.get("e2e_latency") or (end_s - start_s),
            prompt_tokens=meta.get("prompt_tokens"),
            completion_tokens=meta.get("completion_tokens"),
        )
    except Exception as exc:
        end_s = time.monotonic() - origin_t
        return RequestRecord(req_id, scheduled_s, start_s, end_s, url, "exception", None, repr(exc), None, None, None, None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", action="append", required=True, help="Active SGLang endpoint. Can be repeated for manual cutover windows.")
    p.add_argument("--qps", type=float, default=1.0)
    p.add_argument("--duration-s", type=float, default=120.0)
    p.add_argument("--prompt-len", type=int, default=2048)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--timeout-s", type=float, default=120.0)
    p.add_argument("--sample-interval-s", type=float, default=0.5)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    prompt = "x " * max(1, args.prompt_len // 2)
    origin_t = time.monotonic()
    stop = threading.Event()
    gpu_samples: list[GpuSample] = []
    sampler = start_sampler(origin_t, args.sample_interval_s, stop, gpu_samples)

    total = int(args.duration_s * args.qps)
    interval = 1.0 / args.qps
    records: list[RequestRecord] = []
    with ThreadPoolExecutor(max_workers=max(4, min(64, total))) as ex:
        futs = []
        for i in range(total):
            # If multiple URLs are provided, split the run into equal phases.
            url = args.url[min(len(args.url) - 1, int(i * len(args.url) / max(1, total)))]
            futs.append(ex.submit(send_generate, i, i * interval, origin_t, url, prompt, args.output_len, args.timeout_s))
        for fut in as_completed(futs):
            records.append(fut.result())

    stop.set()
    sampler.join(timeout=2)
    records.sort(key=lambda r: r.request_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "summary": {
                    "total_requests": len(records),
                    "successful_requests": sum(1 for r in records if r.status == "ok"),
                    "failed_requests": sum(1 for r in records if r.status != "ok"),
                    "max_memory_used_mib_by_gpu": {
                        str(i): max((s.memory_used_mib for s in gpu_samples if s.index == i), default=0)
                        for i in sorted({s.index for s in gpu_samples})
                    },
                },
                "requests": [asdict(r) for r in records],
                "gpu_samples": [asdict(s) for s in gpu_samples],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
