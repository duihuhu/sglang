#!/usr/bin/env python3
"""Compare breakdown CUDA-event timing vs e2e wall-clock on identical MoE forward."""
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
    profile_world,
    stable_seed,
    sync_any,
)

COMPONENTS = ("gate", "topk", "dispatch", "moe_core", "combine", "ep_allreduce", "total", "sum_parts")


def breakdown_cuda_events(runner, layer, hidden, residual, warmup: int, repeat: int, world: int) -> dict[str, float]:
    from sglang.srt.distributed.communication_op import moe_expert_parallel_all_reduce
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    moe = layer.mlp
    acc = {name: [] for name in COMPONENTS}

    def one_forward():
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
            hs, _ = layer.post_attention_layernorm(hidden, residual)
            hs_flat = hs.view(-1, hs.shape[-1])

            total_s = torch.cuda.Event(enable_timing=True)
            total_e = torch.cuda.Event(enable_timing=True)
            total_s.record()

            gate_s = torch.cuda.Event(enable_timing=True)
            gate_e = torch.cuda.Event(enable_timing=True)
            gate_s.record()
            router_logits, _ = moe.gate(hs_flat)
            gate_e.record()

            topk_s = torch.cuda.Event(enable_timing=True)
            topk_e = torch.cuda.Event(enable_timing=True)
            topk_s.record()
            topk_output = moe.topk(hs_flat, router_logits)
            topk_e.record()

            dispatch_s = torch.cuda.Event(enable_timing=True)
            dispatch_e = torch.cuda.Event(enable_timing=True)
            dispatch_s.record()
            dispatch_output = moe.experts.dispatcher.dispatch(hs_flat, topk_output)
            dispatch_e.record()

            core_s = torch.cuda.Event(enable_timing=True)
            core_e = torch.cuda.Event(enable_timing=True)
            core_s.record()
            combine_input = moe.experts.run_moe_core(dispatch_output)
            core_e.record()

            combine_s = torch.cuda.Event(enable_timing=True)
            combine_e = torch.cuda.Event(enable_timing=True)
            combine_s.record()
            expert_out = moe.experts.dispatcher.combine(combine_input)
            combine_e.record()

            ar_s = torch.cuda.Event(enable_timing=True)
            ar_e = torch.cuda.Event(enable_timing=True)
            ar_s.record()
            if moe.ep_size > 1:
                expert_out = moe_expert_parallel_all_reduce(expert_out)
            ar_e.record()

            total_e.record()

        torch.cuda.synchronize()
        parts = {
            "gate": gate_s.elapsed_time(gate_e),
            "topk": topk_s.elapsed_time(topk_e),
            "dispatch": dispatch_s.elapsed_time(dispatch_e),
            "moe_core": core_s.elapsed_time(core_e),
            "combine": combine_s.elapsed_time(combine_e),
            "ep_allreduce": ar_s.elapsed_time(ar_e),
            "total": total_s.elapsed_time(total_e),
        }
        parts["sum_parts"] = sum(parts[k] for k in ("gate", "topk", "dispatch", "moe_core", "combine", "ep_allreduce"))
        for k, v in parts.items():
            acc[k].append(v)

    for _ in range(warmup):
        one_forward()
    torch.cuda.synchronize()
    dist_barrier(world)
    for _ in range(repeat):
        one_forward()
    torch.cuda.synchronize()
    return {name: float(np.median(values)) for name, values in acc.items()}


def cuda_event_full_mlp(runner, layer, hidden, residual, fb, warmup: int, repeat: int, world: int) -> float:
    acc = []

    def one_forward():
        with torch.no_grad():
            from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

            with forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                hs, _ = layer.post_attention_layernorm(hidden, residual)
                layer.mlp(hs, fb)
                e.record()
        torch.cuda.synchronize()
        acc.append(s.elapsed_time(e))

    for _ in range(warmup):
        one_forward()
    torch.cuda.synchronize()
    dist_barrier(world)
    for _ in range(repeat):
        one_forward()
    torch.cuda.synchronize()
    return float(np.median(acc))


def worker(server_args, port_args, args, gpu_id, rank):
    from sglang.srt.layers.moe import initialize_moe_config

    initialize_moe_config(server_args)
    runner = load_runner(server_args, port_args, gpu_id, rank)
    world = server_args.tp_size
    ctrl = NvmlController(gpu_id)
    layer = runner.model.model.layers[args.layer_id]

    out = {}
    try:
        for routing in args.routing_modes:
            runner.req_to_token_pool.clear()
            runner.token_to_kv_pool_allocator.clear()
            # Match profile_utils seed scheme (routing NOT in seed — same as e2e TSV)
            request_seed = stable_seed(args.seed, "decode", args.length, args.batch_size, args.layer_id, "req")
            hidden_seed = stable_seed(args.seed, "decode", args.length, args.batch_size, args.layer_id, "hidden")
            reqs = make_reqs(args.batch_size, args.length - 1, np.random.default_rng(request_seed))
            fb = build_forward_batch(reqs, runner, "decode")
            gen = torch.Generator(device=runner.device).manual_seed(hidden_seed)
            hidden = torch.randn(
                args.batch_size,
                runner.model_config.hidden_size,
                device=runner.device,
                dtype=torch.bfloat16,
                generator=gen,
            )
            residual = hidden.clone()

            original_topk = forced_routing(layer.mlp, routing, world)
            try:
                ctrl.lock(args.freq)
                time.sleep(args.clock_settle_seconds)

                def target():
                    hs, _ = layer.post_attention_layernorm(hidden, residual)
                    return layer.mlp(hs, fb)

                max_lat, rank_lat, _, _, actual_repeat = profile_world(
                    target, ctrl, args.warmup, args.repeat, world, "decode"
                )
                breakdown = breakdown_cuda_events(
                    runner, layer, hidden, residual, args.warmup, args.repeat, world
                )
                cuda_full = cuda_event_full_mlp(
                    runner, layer, hidden, residual, fb, args.warmup, args.repeat, world
                )

                local = {
                    "rank": rank,
                    "routing": routing,
                    "wall_max_us": max_lat,
                    "wall_per_rank_us": rank_lat,
                    "cuda_full_mlp_ms": cuda_full,
                    "breakdown_ms": breakdown,
                    "actual_repeat": actual_repeat,
                    "disable_custom_ar": server_args.disable_custom_all_reduce,
                    "max_total_tokens": server_args.max_total_tokens,
                }
                ctrl.unlock()
            finally:
                if original_topk is not None:
                    layer.mlp.topk.forward = original_topk

            gathered = gather_object(local, world) if rank == 0 else gather_object(local, world)
            if rank == 0:
                out[routing] = gathered
        if rank == 0:
            print(json.dumps(out, indent=2))
    finally:
        ctrl.close()
        if world > 1:
            from sglang.srt.distributed.parallel_state import destroy_distributed_environment

            destroy_distributed_environment()


def parse_args():
    from sglang.srt.server_args import ServerArgs

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
        max_total_tokens=600000,
        disable_custom_all_reduce=False,
    )
    parser.add_argument("--routing-modes", nargs="+", default=["balanced", "skewed_rank0"])
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freq", type=int, default=930)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--clock-settle-seconds", type=float, default=0.05)
    parser.add_argument("--local-world-size", type=int, default=4)
    return parser.parse_args()


def main():
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import maybe_reindex_device_id

    args = parse_args()
    server_args = ServerArgs.from_cli_args(args)
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
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
