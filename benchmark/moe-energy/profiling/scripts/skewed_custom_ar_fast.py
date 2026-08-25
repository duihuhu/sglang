#!/usr/bin/env python3
"""Fast skewed_rank0: custom AR on/off per-rank breakdown (wall + CUDA segments)."""
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
    dist_barrier,
    forced_routing,
    gather_object,
    load_runner,
    make_reqs,
    NvmlController,
    stable_seed,
)

def worker(server_args, port_args, args, gpu_id, rank):
    from sglang.srt.distributed.communication_op import moe_expert_parallel_all_reduce
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    initialize_moe_config(server_args)
    runner = load_runner(server_args, port_args, gpu_id, rank)
    world = server_args.tp_size
    ctrl = NvmlController(gpu_id)
    layer = runner.model.model.layers[args.layer_id]
    moe = layer.mlp

    runner.req_to_token_pool.clear()
    runner.token_to_kv_pool_allocator.clear()
    reqs = make_reqs(
        args.batch_size,
        args.length - 1,
        np.random.default_rng(
            stable_seed(args.seed, "decode", args.length, args.batch_size, args.layer_id, "req")
        ),
    )
    fb = build_forward_batch(reqs, runner, "decode")
    gen = torch.Generator(device=runner.device).manual_seed(
        stable_seed(args.seed, "decode", args.length, args.batch_size, args.layer_id, "hidden")
    )
    hidden = torch.randn(
        args.batch_size,
        runner.model_config.hidden_size,
        device=runner.device,
        dtype=torch.bfloat16,
        generator=gen,
    )
    residual = hidden.clone()
    original = forced_routing(moe, "skewed_rank0", world)

    def target():
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
            hs, _ = layer.post_attention_layernorm(hidden, residual)
            return layer.mlp(hs, fb)

    def cuda_segment(name, fn):
        acc = []
        for _ in range(args.warmup):
            fn(acc)
        dist_barrier(world)
        for _ in range(args.repeat):
            fn(acc)
        return float(np.median(acc)) * 1000.0

    try:
        ctrl.lock(args.freq)
        time.sleep(args.clock_settle_seconds)

        # Per-rank wall time (profile_world only returns rank_latency on rank 0).
        for _ in range(args.warmup):
            target()
        torch.cuda.synchronize()
        dist_barrier(world)
        probe_start = time.perf_counter()
        target()
        torch.cuda.synchronize()
        probe_s = max(time.perf_counter() - probe_start, 1e-6)
        actual_repeat = max(args.repeat, int(0.5 / probe_s) + 1)
        repeat_tensor = torch.tensor(actual_repeat, device="cuda", dtype=torch.int64)
        if world > 1:
            torch.distributed.all_reduce(repeat_tensor, op=torch.distributed.ReduceOp.MAX)
        actual_repeat = int(repeat_tensor.item())
        dist_barrier(world)
        start_ns = time.perf_counter_ns()
        for _ in range(actual_repeat):
            target()
        torch.cuda.synchronize()
        wall_us = (time.perf_counter_ns() - start_ns) / 1000.0 / actual_repeat

        hs, _ = layer.post_attention_layernorm(hidden, residual)
        hf = hs.view(-1, hs.shape[-1])

        def full_mlp(acc):
            with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                layer.mlp(hs, fb)
                e.record()
            torch.cuda.synchronize()
            acc.append(s.elapsed_time(e))

        def ep_ar_seg(acc):
            rl, _ = moe.gate(hf)
            tk = moe.topk(hf, rl)
            disp = moe.experts.dispatcher.dispatch(hf, tk)
            ci = moe.experts.run_moe_core(disp)
            eo = moe.experts.dispatcher.combine(ci)
            buf = eo[..., :hf.shape[-1]].contiguous()
            with torch.no_grad():
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                moe_expert_parallel_all_reduce(buf)
                e.record()
            torch.cuda.synchronize()
            acc.append(s.elapsed_time(e))

        def moe_core_seg(acc):
            rl, _ = moe.gate(hf)
            tk = moe.topk(hf, rl)
            disp = moe.experts.dispatcher.dispatch(hf, tk)
            with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                moe.experts.run_moe_core(disp)
                e.record()
            torch.cuda.synchronize()
            acc.append(s.elapsed_time(e))

        local = {
            "rank": rank,
            "disable_custom_all_reduce": server_args.disable_custom_all_reduce,
            "wall_us": float(wall_us),
            "cuda_full_mlp_us": cuda_segment("full", full_mlp),
            "cuda_moe_core_us": cuda_segment("core", moe_core_seg),
            "cuda_ep_ar_us": cuda_segment("ar", ep_ar_seg),
            "actual_repeat": int(actual_repeat),
        }
        ctrl.unlock()
    finally:
        if original is not None:
            moe.topk.forward = original
        ctrl.close()
        if rank == 0:
            rows = gather_object(local, world)
            print(json.dumps(rows, indent=2))
        else:
            gather_object(local, world)
        if world > 1:
            from sglang.srt.distributed.parallel_state import destroy_distributed_environment

            destroy_distributed_environment()


def main():
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import maybe_reindex_device_id

    parser = argparse.ArgumentParser()
    ServerArgs.add_cli_args(parser)
    parser.set_defaults(
        model_path=DEFAULT_MODEL,
        moe_runner_backend="triton",
        moe_a2a_backend="none",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
        tp_size=4,
        ep_size=4,
        max_total_tokens=600000,
        disable_custom_all_reduce=True,
    )
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freq", type=int, default=930)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--clock-settle-seconds", type=float, default=0.05)
    parser.add_argument("--local-world-size", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    server_args = ServerArgs.from_cli_args(args)
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    procs = []
    for rank in range(args.local_world_size):
        with maybe_reindex_device_id(rank) as gpu_id:
            p = multiprocessing.Process(target=worker, args=(server_args, port_args, args, gpu_id, rank))
            p.start()
            procs.append(p)
    for p in procs:
        p.join()


if __name__ == "__main__":
    main()
