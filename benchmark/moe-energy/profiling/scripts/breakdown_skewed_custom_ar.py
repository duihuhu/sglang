#!/usr/bin/env python3
"""Breakdown how disable_custom_all_reduce affects skewed_rank0 MoE forward."""
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
)


def _median_event(run_fn, warmup: int, repeat: int) -> float:
    acc = []
    for _ in range(warmup):
        run_fn(acc)
    for _ in range(repeat):
        run_fn(acc)
    return float(np.median(acc))


def profile_skewed(runner, layer, hidden, residual, fb, ctrl, warmup: int, repeat: int, world: int) -> dict:
    from sglang.srt.distributed.communication_op import moe_expert_parallel_all_reduce
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    moe = layer.mlp
    out: dict = {}

    def wall_full():
        def target():
            with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                hs, _ = layer.post_attention_layernorm(hidden, residual)
                return layer.mlp(hs, fb)
        return profile_world(target, ctrl, warmup, repeat, world, "decode")

    max_lat, rank_lat, _, _, actual_repeat = wall_full()
    out["wall_max_us"] = float(max_lat)
    out["wall_per_rank_us"] = [float(x) for x in rank_lat]
    out["wall_actual_repeat"] = int(actual_repeat)

    # CUDA: full layer.mlp (matches e2e path)
    def cuda_full_mlp():
        def run(acc):
            with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                hs, _ = layer.post_attention_layernorm(hidden, residual)
                layer.mlp(hs, fb)
                e.record()
            torch.cuda.synchronize()
            acc.append(s.elapsed_time(e))
        return _median_event(run, warmup, repeat)

    out["cuda_full_mlp_us"] = cuda_full_mlp() * 1000.0

    # CUDA: experts() only (gate/topk outside)
    hs_prep, _ = layer.post_attention_layernorm(hidden, residual)
    hs_flat = hs_prep.view(-1, hs_prep.shape[-1])
    router_logits, _ = moe.gate(hs_flat)
    topk_output = moe.topk(hs_flat, router_logits)

    def cuda_experts_only():
        def run(acc):
            with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                moe.experts(hs_flat, topk_output)
                e.record()
            torch.cuda.synchronize()
            acc.append(s.elapsed_time(e))
        return _median_event(run, warmup, repeat)

    out["cuda_experts_only_us"] = cuda_experts_only() * 1000.0

    # CUDA: outer EP allreduce only (on real expert output)
    dispatch_output = moe.experts.dispatcher.dispatch(hs_flat, topk_output)
    combine_input = moe.experts.run_moe_core(dispatch_output)
    expert_out = moe.experts.dispatcher.combine(combine_input)
    expert_out = expert_out[..., :hs_flat.shape[-1]].contiguous()

    def cuda_ep_ar_only():
        def run(acc):
            with torch.no_grad():
                buf = expert_out.clone()
                s, e = torch.cuda.Event(True), torch.cuda.Event(True)
                s.record()
                moe_expert_parallel_all_reduce(buf)
                e.record()
            torch.cuda.synchronize()
            acc.append(s.elapsed_time(e))
        return _median_event(run, warmup, repeat)

    out["cuda_ep_ar_only_us"] = cuda_ep_ar_only() * 1000.0

    # Standard breakdown segments (CUDA events, ms -> us)
    segments = {}

    def one_bd(acc):
        with torch.no_grad(), forward_context(ForwardContext(attn_backend=runner.attn_backend)):
            hs, _ = layer.post_attention_layernorm(hidden, residual)
            hf = hs.view(-1, hs.shape[-1])
            total_s, total_e = torch.cuda.Event(True), torch.cuda.Event(True)
            total_s.record()
            gs, ge = torch.cuda.Event(True), torch.cuda.Event(True)
            gs.record()
            rl, _ = moe.gate(hf)
            ge.record()
            ts, te = torch.cuda.Event(True), torch.cuda.Event(True)
            ts.record()
            tk = moe.topk(hf, rl)
            te.record()
            ds, de = torch.cuda.Event(True), torch.cuda.Event(True)
            ds.record()
            disp = moe.experts.dispatcher.dispatch(hf, tk)
            de.record()
            cs, ce = torch.cuda.Event(True), torch.cuda.Event(True)
            cs.record()
            ci = moe.experts.run_moe_core(disp)
            ce.record()
            cbs, cbe = torch.cuda.Event(True), torch.cuda.Event(True)
            cbs.record()
            eo = moe.experts.dispatcher.combine(ci)
            cbe.record()
            ars, are = torch.cuda.Event(True), torch.cuda.Event(True)
            ars.record()
            if moe.ep_size > 1:
                eo = moe_expert_parallel_all_reduce(eo)
            are.record()
            total_e.record()
        torch.cuda.synchronize()
        acc.append(
            {
                "gate": gs.elapsed_time(ge),
                "topk": ts.elapsed_time(te),
                "dispatch": ds.elapsed_time(de),
                "moe_core": cs.elapsed_time(ce),
                "combine": cbs.elapsed_time(cbe),
                "ep_allreduce": ars.elapsed_time(are),
                "total": total_s.elapsed_time(total_e),
            }
        )

    bd_acc = []
    for _ in range(warmup):
        one_bd(bd_acc)
    dist_barrier(world)
    for _ in range(repeat):
        one_bd(bd_acc)
    med = {k: float(np.median([x[k] for x in bd_acc])) * 1000.0 for k in bd_acc[0]}
    med["sum_parts"] = sum(med[k] for k in ("gate", "topk", "dispatch", "moe_core", "combine", "ep_allreduce"))
    out["breakdown_us"] = med

    # Gap accounting
    out["gap_cuda_full_minus_bd_total_us"] = out["cuda_full_mlp_us"] - med["total"]
    out["gap_wall_minus_cuda_full_us"] = out["wall_max_us"] - out["cuda_full_mlp_us"]
    out["experts_reduce_results"] = bool(moe.experts.reduce_results)
    return out


def worker(server_args, port_args, args, gpu_id, rank):
    from sglang.srt.layers.moe import initialize_moe_config

    initialize_moe_config(server_args)
    runner = load_runner(server_args, port_args, gpu_id, rank)
    world = server_args.tp_size
    ctrl = NvmlController(gpu_id)
    layer = runner.model.model.layers[args.layer_id]

    try:
        runner.req_to_token_pool.clear()
        runner.token_to_kv_pool_allocator.clear()
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

        original = forced_routing(layer.mlp, "skewed_rank0", world)
        try:
            ctrl.lock(args.freq)
            time.sleep(args.clock_settle_seconds)
            local = profile_skewed(runner, layer, hidden, residual, fb, ctrl, args.warmup, args.repeat, world)
            local["rank"] = rank
            local["disable_custom_all_reduce"] = server_args.disable_custom_all_reduce
            ctrl.unlock()
        finally:
            if original is not None:
                layer.mlp.topk.forward = original

        if rank == 0:
            print(json.dumps(gather_object(local, world), indent=2))
        else:
            gather_object(local, world)
    finally:
        ctrl.close()
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
        max_total_tokens=600000,
        disable_custom_all_reduce=True,
    )
    parser.add_argument("--length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--freq", type=int, default=930)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeat", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--clock-settle-seconds", type=float, default=0.05)
    parser.add_argument("--local-world-size", type=int, default=4)
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
