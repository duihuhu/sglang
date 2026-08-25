#!/usr/bin/env python3
"""Decode-stage MoE EP breakdown: balanced vs skewed_rank0 with CUDA event timing."""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
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
    sync_any,
)

MOE_PARTS = ("gate", "topk", "dispatch", "moe_core", "combine", "ep_allreduce")
# Profile keys exported per rank (total kept as alias for moe_cuda_total).
PROFILE_KEYS = (
    "layernorm",
    *MOE_PARTS,
    "moe_cuda_total",
    "cuda_full_mlp",
    "moe_parts_sum",
    "gap_cuda_mlp_vs_moe_total",
    "gap_wall_vs_cuda_mlp",
)
COMPONENTS = PROFILE_KEYS + ("total",)  # "total" == moe_cuda_total for legacy readers


def _event_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return start.elapsed_time(end)


def _to_us(ms: float) -> float:
    return ms * 1000.0


def profile_moe_unified_session(
    runner,
    layer,
    hidden,
    residual,
    fb,
    warmup: int,
    repeat: int,
    world: int,
) -> dict[str, float]:
    """Single forward_context session: shared warmup, paired wall+cuda_full per iter, then breakdown."""
    import time

    from sglang.srt.distributed.communication_op import moe_expert_parallel_all_reduce
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    moe = layer.mlp
    wall_acc: list[float] = []
    cuda_full_acc: list[float] = []
    breakdown_acc = {name: [] for name in ("layernorm", *MOE_PARTS, "moe_cuda_total")}

    def one_breakdown_forward():
        ln_s = torch.cuda.Event(enable_timing=True)
        ln_e = torch.cuda.Event(enable_timing=True)
        ln_s.record()
        hs, _ = layer.post_attention_layernorm(hidden, residual)
        ln_e.record()
        hs_flat = hs.view(-1, hs.shape[-1])

        moe_s = torch.cuda.Event(enable_timing=True)
        moe_e = torch.cuda.Event(enable_timing=True)
        moe_s.record()

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

        moe_e.record()
        torch.cuda.synchronize()

        breakdown_acc["layernorm"].append(_event_ms(ln_s, ln_e))
        breakdown_acc["gate"].append(_event_ms(gate_s, gate_e))
        breakdown_acc["topk"].append(_event_ms(topk_s, topk_e))
        breakdown_acc["dispatch"].append(_event_ms(dispatch_s, dispatch_e))
        breakdown_acc["moe_core"].append(_event_ms(core_s, core_e))
        breakdown_acc["combine"].append(_event_ms(combine_s, combine_e))
        breakdown_acc["ep_allreduce"].append(_event_ms(ar_s, ar_e))
        breakdown_acc["moe_cuda_total"].append(_event_ms(moe_s, moe_e))

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
        for _ in range(warmup):
            hs, _ = layer.post_attention_layernorm(hidden, residual)
            layer.mlp(hs, fb)
        torch.cuda.synchronize()
        dist_barrier(world)

        torch.cuda.synchronize()
        probe_start = time.perf_counter()
        hs, _ = layer.post_attention_layernorm(hidden, residual)
        layer.mlp(hs, fb)
        torch.cuda.synchronize()
        probe_s = max(time.perf_counter() - probe_start, 1e-6)
        actual_repeat = max(repeat, int(0.5 / probe_s) + 1)
        repeat_tensor = torch.tensor(actual_repeat, device="cuda", dtype=torch.int64)
        if world > 1:
            torch.distributed.all_reduce(repeat_tensor, op=torch.distributed.ReduceOp.MAX)
        actual_repeat = int(repeat_tensor.item())
        dist_barrier(world)

        for _ in range(actual_repeat):
            torch.cuda.synchronize()
            start_ns = time.perf_counter_ns()
            cuda_s = torch.cuda.Event(enable_timing=True)
            cuda_e = torch.cuda.Event(enable_timing=True)
            cuda_s.record()
            hs, _ = layer.post_attention_layernorm(hidden, residual)
            layer.mlp(hs, fb)
            cuda_e.record()
            torch.cuda.synchronize()
            wall_acc.append((time.perf_counter_ns() - start_ns) / 1e6)
            cuda_full_acc.append(cuda_s.elapsed_time(cuda_e))

        dist_barrier(world)
        for _ in range(warmup):
            one_breakdown_forward()
        dist_barrier(world)
        for _ in range(repeat):
            one_breakdown_forward()

    out = {
        "wall_us": _to_us(float(np.median(wall_acc))),
        "wall_actual_repeat": actual_repeat,
        "cuda_full_mlp": _to_us(float(np.median(cuda_full_acc))),
    }
    for name, values in breakdown_acc.items():
        out[name] = _to_us(float(np.median(values)))
    out["total"] = out["moe_cuda_total"]
    return out


def profile_moe_breakdown(runner, layer, hidden, residual, warmup: int, repeat: int, world: int) -> dict[str, float]:
    from sglang.srt.distributed.communication_op import moe_expert_parallel_all_reduce
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    moe = layer.mlp
    acc = {name: [] for name in ("layernorm", *MOE_PARTS, "moe_cuda_total")}

    def one_forward():
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
            ln_s = torch.cuda.Event(enable_timing=True)
            ln_e = torch.cuda.Event(enable_timing=True)
            ln_s.record()
            hs, _ = layer.post_attention_layernorm(hidden, residual)
            ln_e.record()
            hs_flat = hs.view(-1, hs.shape[-1])

            moe_s = torch.cuda.Event(enable_timing=True)
            moe_e = torch.cuda.Event(enable_timing=True)
            moe_s.record()

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

            moe_e.record()

        torch.cuda.synchronize()
        acc["layernorm"].append(_event_ms(ln_s, ln_e))
        acc["gate"].append(_event_ms(gate_s, gate_e))
        acc["topk"].append(_event_ms(topk_s, topk_e))
        acc["dispatch"].append(_event_ms(dispatch_s, dispatch_e))
        acc["moe_core"].append(_event_ms(core_s, core_e))
        acc["combine"].append(_event_ms(combine_s, combine_e))
        acc["ep_allreduce"].append(_event_ms(ar_s, ar_e))
        acc["moe_cuda_total"].append(_event_ms(moe_s, moe_e))

    for _ in range(warmup):
        one_forward()
    torch.cuda.synchronize()
    dist_barrier(world)

    for _ in range(repeat):
        one_forward()
    torch.cuda.synchronize()

    out = {name: _to_us(float(np.median(values))) for name, values in acc.items()}
    out["total"] = out["moe_cuda_total"]
    return out


def profile_cuda_full_mlp(runner, layer, hidden, residual, fb, warmup: int, repeat: int, world: int) -> float:
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    acc = []

    def one_forward():
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
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
    return _to_us(float(np.median(acc)))


def _enrich_profile(local: dict[str, float]) -> dict[str, float]:
    local["moe_parts_sum"] = sum(local[p] for p in MOE_PARTS)
    local["gap_cuda_mlp_vs_moe_total"] = local["cuda_full_mlp"] - local["moe_cuda_total"]
    local["gap_wall_vs_cuda_mlp"] = local["wall_us"] - local["cuda_full_mlp"]
    local["total"] = local["moe_cuda_total"]
    return local


def profile_moe_wall_clock(runner, layer, hidden, residual, fb, warmup: int, repeat: int, world: int) -> dict[str, float]:
    import time

    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
        def target():
            hs, _ = layer.post_attention_layernorm(hidden, residual)
            return layer.mlp(hs, fb)

        for _ in range(warmup):
            target()
        torch.cuda.synchronize()
        dist_barrier(world)
        probe_start = time.perf_counter()
        target()
        torch.cuda.synchronize()
        probe_s = max(time.perf_counter() - probe_start, 1e-6)
        actual_repeat = max(repeat, int(0.5 / probe_s) + 1)
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

    gathered = gather_object({"rank": torch.distributed.get_rank() if world > 1 else 0, "wall_us": wall_us}, world)
    if gathered is None:
        return {"wall_us": wall_us, "wall_max_us": wall_us, "wall_per_rank_us": [wall_us], "actual_repeat": actual_repeat}
    walls = [float(x["wall_us"]) for x in sorted(gathered, key=lambda r: r["rank"])]
    return {
        "wall_us": float(wall_us),
        "wall_max_us": float(max(walls)),
        "wall_per_rank_us": walls,
        "actual_repeat": int(actual_repeat),
    }


def worker(server_args, port_args, args, gpu_id, rank):
    from sglang.srt.layers.moe import initialize_moe_config

    initialize_moe_config(server_args)
    runner = load_runner(server_args, port_args, gpu_id, rank)
    world = server_args.tp_size
    ctrl = NvmlController(gpu_id)
    layer = runner.model.model.layers[args.layer_id]
    output = Path(args.output)
    results = []

    try:
        for routing in args.routing_modes:
            for batch_size in args.batch_sizes:
                runner.req_to_token_pool.clear()
                runner.token_to_kv_pool_allocator.clear()
                failed = False
                fb = hidden = residual = None
                request_seed = stable_seed(args.seed, "decode", args.length, batch_size, args.layer_id, "req")
                hidden_seed = stable_seed(args.seed, "decode", args.length, batch_size, args.layer_id, "hidden")

                try:
                    reqs = make_reqs(batch_size, args.length - 1, np.random.default_rng(request_seed))
                    fb = build_forward_batch(reqs, runner, "decode")
                    gen = torch.Generator(device=runner.device).manual_seed(hidden_seed)
                    hidden = torch.randn(
                        batch_size,
                        runner.model_config.hidden_size,
                        device=runner.device,
                        dtype=torch.bfloat16,
                        generator=gen,
                    )
                    residual = hidden.clone()
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    failed = True
                    if rank == 0:
                        print(f"[skip setup] routing={routing} batch={batch_size}: {exc}")
                    torch.cuda.empty_cache()

                if sync_any(failed, world):
                    continue

                original_topk = forced_routing(layer.mlp, routing, world)
                try:
                    ctrl.lock(args.freq)
                    time.sleep(args.clock_settle_seconds)

                    local = profile_moe_unified_session(
                        runner, layer, hidden, residual, fb, args.warmup, args.repeat, world
                    )
                    wall_us = local["wall_us"]
                    wall_gathered = gather_object(
                        {
                            "rank": torch.distributed.get_rank() if world > 1 else 0,
                            "wall_us": wall_us,
                        },
                        world,
                    )
                    if wall_gathered is not None:
                        walls = [float(x["wall_us"]) for x in sorted(wall_gathered, key=lambda r: r["rank"])]
                        local["wall_max_us"] = float(max(walls))
                        local["wall_per_rank_us"] = walls
                    else:
                        local["wall_max_us"] = wall_us
                        local["wall_per_rank_us"] = [wall_us]
                    local = _enrich_profile(local)
                    local["session_mode"] = "unified"
                    gathered = gather_object(local, world)
                    if rank != 0:
                        continue

                    per_rank = gathered
                    max_profile = {name: max(row[name] for row in per_rank) for name in PROFILE_KEYS}
                    rank0_profile = per_rank[0]

                    row = {
                        "routing": routing,
                        "batch_size": batch_size,
                        "length": args.length,
                        "freq_mhz": args.freq,
                        "world_size": world,
                        "session_mode": "unified",
                        "per_rank_us": per_rank,
                        "max_rank_us": max_profile,
                        "rank0_us": rank0_profile,
                    }
                    results.append(row)
                    print(
                        f"routing={routing} batch={batch_size} "
                        f"wall={local['wall_max_us']:.1f}us ln+cuda_mlp={max_profile['cuda_full_mlp']:.1f}us "
                        f"ln={max_profile['layernorm']:.1f} moe_cuda={max_profile['moe_cuda_total']:.1f}us "
                        f"gap_wall_cuda={max_profile['gap_wall_vs_cuda_mlp']:.1f}us "
                        f"gap_cuda_path={max_profile['gap_cuda_mlp_vs_moe_total']:.1f}us "
                        f"topk={max_profile['topk']:.1f} core={max_profile['moe_core']:.1f}"
                    )
                finally:
                    if original_topk is not None:
                        layer.mlp.topk.forward = original_topk
                    ctrl.unlock()

                del fb, hidden, residual
                torch.cuda.empty_cache()

        if rank == 0:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(results, indent=2) + "\n")
            print(f"Wrote {output}")
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
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[32, 2048])
    parser.add_argument("--freq", type=int, default=930)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--clock-settle-seconds", type=float, default=0.05)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-world-size", type=int, default=4)
    return parser.parse_args()


def main():
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import maybe_reindex_device_id

    args = parse_args()
    server_args = ServerArgs.from_cli_args(args)
    if server_args.ep_size != server_args.tp_size:
        raise SystemExit("moe_ep requires ep_size == tp_size")
    _set_envs_and_config(server_args)
    port_args = PortArgs.init_new(server_args)
    local_world = args.local_world_size
    if local_world != server_args.tp_size:
        raise SystemExit("--local-world-size must equal --tp-size")

    if local_world == 1:
        worker(server_args, port_args, args, 0, 0)
        return

    procs = []
    for rank in range(local_world):
        with maybe_reindex_device_id(rank) as gpu_id:
            proc = multiprocessing.Process(target=worker, args=(server_args, port_args, args, gpu_id, rank))
            proc.start()
            procs.append(proc)
    for proc in procs:
        proc.join()
    failed = [(p.pid, p.exitcode) for p in procs if p.exitcode]
    if failed:
        raise SystemExit(f"workers failed: {failed}")


if __name__ == "__main__":
    main()
