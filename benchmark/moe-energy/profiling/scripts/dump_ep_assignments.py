#!/usr/bin/env python3
"""Dump per-rank local expert assignment counts for forced routing modes."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import sys
import time
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(SCRIPT_DIR))

from profile_utils import (  # noqa: E402
    DEFAULT_MODEL,
    build_forward_batch,
    forced_routing,
    gather_object,
    load_runner,
    make_reqs,
    NvmlController,
)


def worker(server_args, port_args, args, gpu_id, rank):
    from sglang.srt.layers.moe import initialize_moe_config

    initialize_moe_config(server_args)
    runner = load_runner(server_args, port_args, gpu_id, rank)
    world = server_args.tp_size
    ctrl = NvmlController(gpu_id)
    ctrl.lock(args.freq)
    time.sleep(args.clock_settle_seconds)

    layer = runner.model.model.layers[args.layer_id]
    moe = layer.mlp
    reqs = make_reqs(args.batch_size, args.length - 1, np.random.default_rng(args.seed))
    fb = build_forward_batch(reqs, runner, "decode")
    gen = torch.Generator(device=runner.device).manual_seed(args.seed)
    hidden = torch.randn(
        args.batch_size,
        runner.model_config.hidden_size,
        device=runner.device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    residual = hidden.clone()
    forced_routing(moe, args.routing, world)

    hs, _ = layer.post_attention_layernorm(hidden, residual)
    hs_flat = hs.view(-1, hs.shape[-1])
    router_logits, _ = moe.gate(hs_flat)
    topk_output = moe.topk(hs_flat, router_logits)
    dispatch_output = moe.experts.dispatcher.dispatch(hs_flat, topk_output)

    local_ids = dispatch_output.topk_output.topk_ids.reshape(-1)
    valid = int((local_ids >= 0).sum())
    dropped = int((local_ids < 0).sum())
    per_local = torch.bincount(
        local_ids[local_ids >= 0], minlength=moe.experts.num_local_experts
    ).cpu().tolist()
    active = [x for x in per_local if x > 0]

    out = {
        "rank": rank,
        "routing": args.routing,
        "valid_local_assignments": valid,
        "dropped_assignments": dropped,
        "active_local_experts": len(active),
        "per_expert_min": min(active) if active else 0,
        "per_expert_max": max(active) if active else 0,
        "per_expert_std": float(np.std(active)) if active else 0.0,
        "num_tokens": args.batch_size,
        "top_k": moe.topk.topk_config.top_k,
    }

    ctrl.unlock()
    ctrl.close()

    if rank == 0:
        print(json.dumps(gather_object(out, world), indent=2))
    else:
        gather_object(out, world)

    if world > 1:
        from sglang.srt.distributed.parallel_state import destroy_distributed_environment

        destroy_distributed_environment()


def main():
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import maybe_reindex_device_id

    parser = argparse.ArgumentParser(description=__doc__)
    ServerArgs.add_cli_args(parser)
    parser.set_defaults(
        model_path=DEFAULT_MODEL,
        moe_runner_backend="triton",
        moe_a2a_backend="none",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        tp_size=4,
        ep_size=4,
        max_total_tokens=1200000,
    )
    parser.add_argument("--routing", choices=["balanced", "skewed_rank0"], default="balanced")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--freq", type=int, default=930)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--clock-settle-seconds", type=float, default=0.05)
    parser.add_argument("--local-world-size", type=int, default=4)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)

    if args.local_world_size != server_args.tp_size:
        raise SystemExit("--local-world-size must equal --tp-size")

    procs = []
    for rank in range(args.local_world_size):
        with maybe_reindex_device_id(rank) as gpu_id:
            proc = multiprocessing.Process(
                target=worker, args=(server_args, port_args, args, gpu_id, rank)
            )
            proc.start()
            procs.append(proc)
    for proc in procs:
        proc.join()


if __name__ == "__main__":
    main()
