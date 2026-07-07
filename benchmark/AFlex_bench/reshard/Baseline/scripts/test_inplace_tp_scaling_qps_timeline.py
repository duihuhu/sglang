#!/usr/bin/env python3
"""True in-place TP reshard timeline benchmark.

This benchmark exercises the part that distinguishes in-place TP expansion from a
shadow-server baseline: old ranks stay alive, new ranks are already joinable
workers in the same 8-rank process group, and each old rank sends only the
sub-shards needed by the newly activated ranks. No target TP instance is
launched and no model weights are loaded from disk during expansion.

The script intentionally uses synthetic Qwen3-shaped tensors by default. This
keeps the benchmark focused on GPU/NVLink/NCCL transfer and timeline behavior,
while avoiding another full model load for every rank. Use --profile full to
scale tensor bytes closer to a 32B dense model.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import socket
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist


TP_STAGES = [1, 2, 4, 8]


@dataclass
class TensorSpec:
    name: str
    shape: Tuple[int, ...]
    split_dim: int
    dtype: str = "bfloat16"


@dataclass
class ReshardEvent:
    old_tp: int
    new_tp: int
    scale_start_s: float
    pause_start_s: float
    transfer_done_s: float
    group_rebuilt_s: float
    resume_s: float
    transfer_bytes: int
    transfer_s: float
    bandwidth_gbps: float
    active_ranks: List[int]


@dataclass
class TimelineRequest:
    request_id: int
    scheduled_s: float
    start_s: float
    end_s: float
    tp_size: int
    status: str
    error: Optional[str]
    ttft_s: Optional[float]
    e2e_s: Optional[float]


class RouteState:
    def __init__(self) -> None:
        import threading

        self._lock = threading.Lock()
        self.tp_size = 1
        self.paused = False
        self.events: List[ReshardEvent] = []

    def snapshot(self) -> Tuple[int, bool]:
        with self._lock:
            return self.tp_size, self.paused

    def pause(self) -> None:
        with self._lock:
            self.paused = True

    def resume(self, tp_size: int, event: ReshardEvent) -> None:
        with self._lock:
            self.tp_size = tp_size
            self.paused = False
            self.events.append(event)


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    raise ValueError(f"unsupported dtype {name}")


def build_tensor_specs(profile: str, dtype: str) -> List[TensorSpec]:
    """Return representative dense Qwen-style TP tensors.

    smoke: small tensors for CI/local syntax validation.
    bench: a moderate set that transfers several GB per expansion.
    full: 64-layer Qwen3-like dense projection set, scaled to stress NVLink.
    """
    if profile == "smoke":
        layers, hidden, inter = 2, 1024, 2816
    elif profile == "bench":
        layers, hidden, inter = 12, 4096, 11008
    elif profile == "full":
        layers, hidden, inter = 64, 5120, 27648
    else:
        raise ValueError(f"unknown profile: {profile}")

    specs: List[TensorSpec] = [
        TensorSpec("model.embed_tokens.weight", (hidden * 32, hidden), 0, dtype),
        TensorSpec("lm_head.weight", (hidden * 32, hidden), 0, dtype),
    ]
    for i in range(layers):
        prefix = f"model.layers.{i}"
        specs.extend(
            [
                TensorSpec(f"{prefix}.self_attn.q_proj.weight", (hidden, hidden), 0, dtype),
                TensorSpec(f"{prefix}.self_attn.k_proj.weight", (hidden // 8, hidden), 0, dtype),
                TensorSpec(f"{prefix}.self_attn.v_proj.weight", (hidden // 8, hidden), 0, dtype),
                TensorSpec(f"{prefix}.self_attn.o_proj.weight", (hidden, hidden), 1, dtype),
                TensorSpec(f"{prefix}.mlp.gate_proj.weight", (inter, hidden), 0, dtype),
                TensorSpec(f"{prefix}.mlp.up_proj.weight", (inter, hidden), 0, dtype),
                TensorSpec(f"{prefix}.mlp.down_proj.weight", (hidden, inter), 1, dtype),
                TensorSpec(f"{prefix}.input_layernorm.weight", (hidden,), -1, dtype),
                TensorSpec(f"{prefix}.post_attention_layernorm.weight", (hidden,), -1, dtype),
            ]
        )
    return specs


def make_initial_shard(specs: List[TensorSpec], rank: int, device_id: int, active_tp: int) -> Dict[str, torch.Tensor]:
    # Joinable ranks start with lightweight placeholder tensors. They become
    # active only after receiving shards from their source old rank.
    tensors: Dict[str, torch.Tensor] = {}
    device = torch.device(f"cuda:{device_id}")
    for spec in specs:
        dtype = dtype_from_name(spec.dtype)
        if rank < active_tp:
            shape = list(spec.shape)
            if spec.split_dim >= 0:
                shape[spec.split_dim] //= active_tp
            tensors[spec.name] = torch.empty(tuple(shape), device=device, dtype=dtype)
            tensors[spec.name].normal_(mean=0.0, std=0.01)
        else:
            tensors[spec.name] = torch.empty((1,), device=device, dtype=dtype)
    return tensors


def sub_shard(t: torch.Tensor, split_dim: int, local_idx: int, factor: int) -> torch.Tensor:
    if split_dim < 0:
        return t.contiguous()
    size = t.shape[split_dim]
    chunk = size // factor
    return t.narrow(split_dim, local_idx * chunk, chunk).contiguous()


def worker_main(
    rank: int,
    device_id: int,
    world_size: int,
    master_port: int,
    specs_payload: List[Dict[str, Any]],
    ctrl_q,
    result_q,
) -> None:
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ["MASTER_PORT"] = str(master_port)
    torch.cuda.set_device(device_id)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    specs = [TensorSpec(**x) for x in specs_payload]
    weights = make_initial_shard(specs, rank, device_id, active_tp=1)
    active_tp = 1

    try:
        while True:
            cmd = ctrl_q.get()
            if cmd["op"] == "stop":
                break
            if cmd["op"] != "reshard":
                raise RuntimeError(f"unknown op {cmd['op']}")

            old_tp = int(cmd["old_tp"])
            new_tp = int(cmd["new_tp"])
            factor = new_tp // old_tp
            transfer_bytes = 0
            t0 = time.monotonic()

            if rank < old_tp:
                # True in-place rank mapping: old ranks keep their rank IDs.
                # New ranks are appended. For TP2→TP4: rank0 sends to 2,
                # rank1 sends to 3; rank0/rank1 keep sub-shard 0 locally.
                for spec in specs:
                    tensor = weights[spec.name]
                    pieces = [sub_shard(tensor, spec.split_dim, i, factor) for i in range(factor)]
                    for i, piece in enumerate(pieces):
                        dst = rank + i * old_tp
                        if i == 0:
                            continue
                        dist.send(piece, dst=dst)
                        transfer_bytes += piece.numel() * piece.element_size()
                    weights[spec.name] = pieces[0]
            elif rank < new_tp:
                src = rank % old_tp
                local_idx = rank // old_tp
                new_weights: Dict[str, torch.Tensor] = {}
                for spec in specs:
                    src_shape = list(spec.shape)
                    if spec.split_dim >= 0:
                        src_shape[spec.split_dim] //= old_tp
                        src_shape[spec.split_dim] //= factor
                    recv = torch.empty(tuple(src_shape), device=f"cuda:{device_id}", dtype=dtype_from_name(spec.dtype))
                    dist.recv(recv, src=src)
                    new_weights[spec.name] = recv
                    _ = local_idx  # documents rank mapping and keeps the invariant explicit.
                weights = new_weights

            dist.barrier()
            active_tp = new_tp
            elapsed = time.monotonic() - t0
            if rank == 0:
                result_q.put({"old_tp": old_tp, "new_tp": new_tp, "transfer_s": elapsed, "transfer_bytes": transfer_bytes})
    finally:
        dist.destroy_process_group()


def send_request(req_id: int, scheduled_s: float, origin_t: float, router: RouteState, output_len: int) -> TimelineRequest:
    delay = origin_t + scheduled_s - time.monotonic()
    if delay > 0:
        time.sleep(delay)
    start_s = time.monotonic() - origin_t
    tp_size, paused = router.snapshot()
    if paused:
        end_s = time.monotonic() - origin_t
        return TimelineRequest(req_id, scheduled_s, start_s, end_s, tp_size, "failed", "inplace_reshard_pause", None, None)

    # Deterministic synthetic latency: enough to reveal pause windows without
    # depending on tokenizer/model output content.
    ttft = 0.08 + 0.015 * math.log2(tp_size)
    e2e = ttft + output_len * 0.012 / max(1, tp_size)
    time.sleep(e2e)
    end_s = time.monotonic() - origin_t
    return TimelineRequest(req_id, scheduled_s, start_s, end_s, tp_size, "ok", None, ttft, e2e)


def run_workload(qps: float, duration_s: float, origin_t: float, router: RouteState, output_len: int) -> List[TimelineRequest]:
    interval = 1.0 / qps
    total = int(duration_s * qps)
    results: List[TimelineRequest] = []
    with ThreadPoolExecutor(max_workers=max(4, min(64, total))) as ex:
        futs = [ex.submit(send_request, i, i * interval, origin_t, router, output_len) for i in range(total)]
        for fut in as_completed(futs):
            results.append(fut.result())
    results.sort(key=lambda x: x.request_id)
    return results


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", choices=["smoke", "bench", "full"], default="bench")
    p.add_argument("--qps", type=float, default=1.0)
    p.add_argument("--workload-duration-s", type=float, default=150.0)
    p.add_argument("--scale-at-s", type=float, nargs="+", default=[30.0, 70.0, 110.0])
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--visible-ranks", default="0,1,2,3,4,5,6,7")
    p.add_argument("--master-port", type=int, default=0)
    p.add_argument("--output", type=Path, default=Path("../results/inplace_tp1_to_tp8_qps_timeline.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ranks = [int(x) for x in args.visible_ranks.split(",") if x]
    if len(ranks) != 8:
        raise ValueError("this benchmark expects exactly 8 visible ranks for TP1→TP8")
    if torch.cuda.device_count() < 8:
        raise RuntimeError(f"need 8 CUDA devices, got {torch.cuda.device_count()}")
    if len(args.scale_at_s) != 3:
        raise ValueError("--scale-at-s must provide three timestamps for TP1→2→4→8")

    specs = build_tensor_specs(args.profile, "bfloat16")
    specs_payload = [asdict(x) for x in specs]
    master_port = args.master_port or find_free_port()

    ctx = get_context("spawn")
    ctrl_queues = [ctx.Queue() for _ in range(8)]
    result_q = ctx.Queue()
    procs = [
        ctx.Process(target=worker_main, args=(rank, ranks[rank], 8, master_port, specs_payload, ctrl_queues[rank], result_q))
        for rank in range(8)
    ]

    for proc in procs:
        proc.start()

    router = RouteState()
    origin_t = time.monotonic()
    workload_future = None
    with ThreadPoolExecutor(max_workers=1) as ex:
        workload_future = ex.submit(run_workload, args.qps, args.workload_duration_s, origin_t, router, args.output_len)
        events: List[ReshardEvent] = []
        for old_tp, new_tp, scale_s in zip(TP_STAGES[:-1], TP_STAGES[1:], args.scale_at_s):
            delay = origin_t + scale_s - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            scale_start_s = time.monotonic() - origin_t
            router.pause()
            pause_start_s = time.monotonic() - origin_t
            for q in ctrl_queues:
                q.put({"op": "reshard", "old_tp": old_tp, "new_tp": new_tp})
            try:
                res = result_q.get(timeout=600)
            except queue.Empty as exc:
                raise RuntimeError(f"timed out waiting for TP{old_tp}->TP{new_tp}") from exc
            transfer_done_s = time.monotonic() - origin_t
            # The process group already exists; this barrier represents updating TP metadata,
            # rank activity, and KV-cache layout in all joinable workers.
            group_rebuilt_s = transfer_done_s
            transfer_s = float(res["transfer_s"])
            transfer_bytes = int(res["transfer_bytes"])
            bw = transfer_bytes / transfer_s / 1e9 if transfer_s > 0 else 0.0
            event = ReshardEvent(
                old_tp=old_tp,
                new_tp=new_tp,
                scale_start_s=scale_start_s,
                pause_start_s=pause_start_s,
                transfer_done_s=transfer_done_s,
                group_rebuilt_s=group_rebuilt_s,
                resume_s=time.monotonic() - origin_t,
                transfer_bytes=transfer_bytes,
                transfer_s=transfer_s,
                bandwidth_gbps=bw,
                active_ranks=list(range(new_tp)),
            )
            router.resume(new_tp, event)
            events.append(event)
        requests = workload_future.result()

    for q in ctrl_queues:
        q.put({"op": "stop"})
    for proc in procs:
        proc.join(timeout=30)
        if proc.exitcode not in (0, None):
            raise RuntimeError(f"worker exited with {proc.exitcode}")

    ok_reqs = [r for r in requests if r.status == "ok"]
    failed = [r for r in requests if r.status != "ok"]
    result = {
        "profile": args.profile,
        "qps": args.qps,
        "workload_duration_s": args.workload_duration_s,
        "tp_stages": TP_STAGES,
        "num_tensor_specs": len(specs),
        "events": [asdict(e) for e in events],
        "requests": [asdict(r) for r in requests],
        "summary": {
            "total_requests": len(requests),
            "successful_requests": len(ok_reqs),
            "failed_requests": len(failed),
            "failed_request_ids": [r.request_id for r in failed],
            "avg_ttft_ms": statistics.mean([r.ttft_s for r in ok_reqs if r.ttft_s is not None]) * 1000 if ok_reqs else None,
            "p50_ttft_ms": statistics.median([r.ttft_s for r in ok_reqs if r.ttft_s is not None]) * 1000 if ok_reqs else None,
            "avg_e2e_ms": statistics.mean([r.e2e_s for r in ok_reqs if r.e2e_s is not None]) * 1000 if ok_reqs else None,
            "visible_pause_s": sum(e.resume_s - e.pause_start_s for e in events),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
