#!/usr/bin/env python3
"""Manual 4-GPU integration test for AFD component NCCL staging transport.

Run directly; the parent launches four torch.distributed workers, enforces a
wall-clock timeout, reports every worker exit code, and cleans up on failure.
"""
from __future__ import annotations

import argparse
import os
import socket
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "python"))

from sglang.srt.reshard.afd_component_weight_staging import (  # noqa: E402
    TorchDistributedNCCLMaxWorldTransport,
)
from sglang.srt.reshard.afd_weight_reshard import SplitRule  # noqa: E402

WORLD_SIZE = 4


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _source_shards(full: torch.Tensor, rule: SplitRule, tp: int):
    if rule.kind == "replicated":
        return tuple(full.clone() for _ in range(tp))
    if rule.kind == "column_fused":
        shards = []
        for rank in range(tp):
            offset, pieces = 0, []
            for segment in rule.segments:
                local_size = segment // tp
                pieces.append(full.narrow(rule.dim, offset + rank * local_size, local_size))
                offset += segment
            shards.append(torch.cat(pieces, dim=rule.dim).contiguous())
        return tuple(shards)
    local_size = full.shape[rule.dim] // tp
    return tuple(
        full.narrow(rule.dim, rank * local_size, local_size).contiguous()
        for rank in range(tp)
    )


def _expected_target(full: torch.Tensor, rule: SplitRule, tp: int, rank: int):
    if rule.kind == "replicated":
        return full
    if rule.kind == "column_fused":
        offset, pieces = 0, []
        for segment in rule.segments:
            local_size = segment // tp
            pieces.append(full.narrow(rule.dim, offset + rank * local_size, local_size))
            offset += segment
        return torch.cat(pieces, dim=rule.dim).contiguous()
    local_size = full.shape[rule.dim] // tp
    return full.narrow(rule.dim, rank * local_size, local_size).contiguous()


def _run_case(
    transport: TorchDistributedNCCLMaxWorldTransport,
    *,
    name: str,
    full: torch.Tensor,
    rule: SplitRule,
    source_tp: int,
    target_tp: int,
) -> None:
    rank = transport.rank
    source_shards = _source_shards(full, rule, source_tp)
    local: Optional[torch.Tensor] = None
    if rank < source_tp:
        local = source_shards[rank].to(transport.device)
    result = transport.stage_target_shard(
        name, local, rule, source_tp, target_tp, rank
    )
    if target_tp > source_tp and rank < source_tp:
        assert result is None, f"{name}: active rank {rank} retained prepare shadow"
    elif rank < target_tp:
        assert result is not None, f"{name}: joining rank {rank} got no target"
        expected = _expected_target(full, rule, target_tp, rank).to(result.device)
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        # Shrink path returns CPU tensor on rank 0 via Gloo fallback.
        if target_tp < source_tp:
            assert result.device.type == "cpu", (
                f"{name}: shrink rank {rank} result should be CPU"
            )
    else:
        assert result is None, f"{name}: non-target rank {rank} retained output"
    dist.barrier(group=transport.control_group)
    if target_tp > source_tp:
        deferred = transport.materialize_deferred_target(
            name, local, rule, source_tp, target_tp, rank
        )
        if rank < source_tp:
            assert deferred is not None
            expected = _expected_target(full, rule, target_tp, rank).to(
                deferred.device
            )
            torch.testing.assert_close(deferred, expected, rtol=0, atol=0)
        else:
            assert deferred is None
    # Every case has the same control-group completion boundary on every rank.
    dist.barrier(group=transport.control_group)
    if rank == 0:
        print(f"PASS {name}: TP{source_tp}->TP{target_tp}", flush=True)


def _worker(rank: int, port: int, collective_timeout_s: float) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        RANK=str(rank),
        LOCAL_RANK=str(rank),
        WORLD_SIZE=str(WORLD_SIZE),
    )
    torch.cuda.set_device(rank)
    timeout = timedelta(seconds=collective_timeout_s)
    dist.init_process_group("gloo", rank=rank, world_size=WORLD_SIZE, timeout=timeout)
    control = data = None
    try:
        ranks = list(range(WORLD_SIZE))
        # Creation order is deliberately identical on all ranks. Neither group
        # is the default world group, and data/control use different backends.
        control = dist.new_group(ranks=ranks, backend="gloo", timeout=timeout)
        data = dist.new_group(ranks=ranks, backend="nccl", timeout=timeout)
        transport = TorchDistributedNCCLMaxWorldTransport(
            rank=rank,
            world_size=WORLD_SIZE,
            data_group=data,
            control_group=control,
            device=f"cuda:{rank}",
        )
        cases = (
            dict(
                name="row_tp2_to_tp4",
                full=torch.arange(4 * 16, dtype=torch.float32).reshape(4, 16),
                rule=SplitRule("row", 1), source_tp=2, target_tp=4,
            ),
            dict(
                name="fused_tp2_to_tp4",
                full=torch.arange(16 * 5, dtype=torch.float32).reshape(16, 5),
                rule=SplitRule("column_fused", 0, (8, 4, 4)),
                source_tp=2, target_tp=4,
            ),
            dict(
                name="replicated_tp2_to_tp4",
                full=torch.arange(3 * 7, dtype=torch.float32).reshape(3, 7),
                rule=SplitRule("replicated"), source_tp=2, target_tp=4,
            ),
        )
        for case in cases:
            _run_case(transport, **case)
        shrink_cases = (
            dict(
                name="row_tp4_to_tp1",
                full=torch.arange(3 * 16, dtype=torch.float32).reshape(3, 16),
                rule=SplitRule("row", 1), source_tp=4, target_tp=1,
            ),
            dict(
                name="fused_tp4_to_tp1",
                full=torch.arange(16 * 3, dtype=torch.float32).reshape(16, 3),
                rule=SplitRule("column_fused", 0, (8, 4, 4)),
                source_tp=4, target_tp=1,
            ),
            dict(
                name="replicated_tp4_to_tp1",
                full=torch.arange(3 * 7, dtype=torch.float32).reshape(3, 7),
                rule=SplitRule("replicated"), source_tp=4, target_tp=1,
            ),
        )
        for case in shrink_cases:
            _run_case(transport, **case)
    finally:
        # Explicitly release process groups and cached buffers before worker exit.
        if dist.is_initialized():
            if data is not None:
                dist.destroy_process_group(data)
            if control is not None:
                dist.destroy_process_group(control)
            dist.destroy_process_group()
        torch.cuda.synchronize(rank)
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--collective-timeout-s", type=float, default=30.0)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.device_count() < WORLD_SIZE:
        print("ERROR: four visible CUDA GPUs are required", file=sys.stderr)
        return 2

    ctx = mp.get_context("spawn")
    port = _free_port()
    processes = [
        ctx.Process(target=_worker, args=(rank, port, args.collective_timeout_s))
        for rank in range(WORLD_SIZE)
    ]
    started = time.monotonic()
    for process in processes:
        process.start()
    deadline = started + args.timeout_s
    for process in processes:
        process.join(max(0.0, deadline - time.monotonic()))
    timed_out = [process for process in processes if process.is_alive()]
    if timed_out:
        for process in timed_out:
            process.terminate()
        for process in timed_out:
            process.join(10)
        for process in timed_out:
            if process.is_alive():
                process.kill()
                process.join(5)
    exit_codes = [process.exitcode for process in processes]
    print(f"worker_exit_codes={exit_codes}", flush=True)
    if timed_out:
        print(f"ERROR: timed out after {args.timeout_s:.1f}s", file=sys.stderr)
        return 124
    if any(code != 0 for code in exit_codes):
        print("ERROR: one or more NCCL workers failed", file=sys.stderr)
        return 1
    print(f"PASS all 4-GPU NCCL cases in {time.monotonic() - started:.3f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
