#!/usr/bin/env python3
"""MoE EP kernel-only profiler: CUDA time + energy for run_moe_core only."""
from __future__ import annotations

import argparse
import multiprocessing
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
sys.path.insert(0, str(REPO_ROOT / "python"))
sys.path.insert(0, str(SCRIPT_DIR))

from profile_utils import (  # noqa: E402
    DEFAULT_MODEL,
    FORCED_ROUTING_CHOICES,
    KERNEL_EP_BATCHES,
    KERNEL_EP_FREQS,
    KERNEL_EP_LENGTHS,
    SCHEMA_VERSION,
    append_row,
    build_forward_batch,
    dist_barrier,
    forced_routing,
    kernel_shape_excluded,
    load_runner,
    make_reqs,
    NvmlController,
    profile_moe_kernel_core,
    read_completed,
    routing_summary,
    row_key,
    stable_seed,
    sync_any,
    topology,
)


def kernel_worker(server_args, port_args, args, gpu_id, rank):
    from sglang.srt.layers.moe import initialize_moe_config
    from sglang.srt.model_executor.forward_context import ForwardContext, forward_context

    initialize_moe_config(server_args)
    runner = load_runner(server_args, port_args, gpu_id, rank)
    world = server_args.tp_size
    ctrl = NvmlController(gpu_id)
    layer = runner.model.model.layers[args.layer_id]
    layout = topology(args, server_args)
    output = Path(args.output)
    completed = read_completed(output)
    out_f = None
    if rank == 0:
        output.parent.mkdir(parents=True, exist_ok=True)
        out_f = output.open("a", buffering=1)
        print(f"{SCHEMA_VERSION} kernel: resumed {len(completed)} rows from {output}")

    try:
        freqs = sorted({ctrl.snap(f) for f in args.freqs})
        for length in args.lengths:
            for batch_size in args.batch_sizes:
                base = {
                    "phase": args.phase,
                    "component": "K",
                    **layout,
                    "length": length,
                    "batch": batch_size,
                }
                pending = [
                    f for f in freqs if row_key({**base, "freq_mhz": f}) not in completed
                ]
                if not pending:
                    continue

                shape_tokens = length * batch_size
                default_safe_shape_tokens = (
                    524288 if args.phase == "prefill" else 131072
                )
                configured_shape_limit = (
                    default_safe_shape_tokens
                    if args.shape_token_limit is None
                    else args.shape_token_limit
                )
                shape_limit = (
                    runner.max_total_num_tokens
                    if configured_shape_limit == 0
                    else min(runner.max_total_num_tokens, configured_shape_limit)
                )
                if shape_tokens > shape_limit:
                    if rank == 0:
                        print(
                            f"[skip capacity] length={length} batch={batch_size} "
                            f"tokens={shape_tokens} limit={shape_limit}"
                        )
                    continue

                if kernel_shape_excluded(args.phase, world, length, batch_size):
                    if rank == 0:
                        print(
                            f"[skip matrix] length={length} batch={batch_size} "
                            f"(see kernel-EP/doc.md)"
                        )
                    continue

                runner.req_to_token_pool.clear()
                runner.token_to_kv_pool_allocator.clear()
                failed = False
                fb = hidden = residual = dispatch_output = routing = None
                request_seed = stable_seed(
                    args.seed, args.phase, length, batch_size, args.layer_id, "req"
                )
                hidden_seed = stable_seed(
                    args.seed, args.phase, length, batch_size, args.layer_id, "hidden"
                )
                try:
                    reqs = make_reqs(
                        batch_size,
                        length if args.phase == "prefill" else length - 1,
                        np.random.default_rng(request_seed),
                    )
                    fb = build_forward_batch(reqs, runner, args.phase)
                    n_tokens = (
                        int(fb.seq_lens_sum)
                        if args.phase == "prefill"
                        else batch_size
                    )
                    gen = torch.Generator(device=runner.device).manual_seed(hidden_seed)
                    hidden = torch.randn(
                        n_tokens,
                        runner.model_config.hidden_size,
                        device=runner.device,
                        dtype=torch.bfloat16,
                        generator=gen,
                    )
                    residual = hidden.clone()
                    with torch.no_grad(), forward_context(
                        ForwardContext(attn_backend=runner.attn_backend)
                    ):
                        forced_routing(layer.mlp, args.forced_routing, world)
                        hs, _ = layer.post_attention_layernorm(hidden, residual)
                        hs_flat = hs.view(-1, hs.shape[-1])
                        router_logits, _ = layer.mlp.gate(hs_flat)
                        topk_output = layer.mlp.topk(hs_flat, router_logits)
                        dispatch_output = layer.mlp.experts.dispatcher.dispatch(
                            hs_flat, topk_output
                        )
                        routing = routing_summary(layer.mlp, hs_flat, world)
                        if routing is not None:
                            routing["forced_mode"] = args.forced_routing
                except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                    failed = True
                    if rank == 0:
                        print(f"[skip setup] length={length} batch={batch_size}: {exc}")
                        traceback.print_exc()
                    torch.cuda.empty_cache()

                if sync_any(failed, world):
                    fb = hidden = residual = dispatch_output = routing = None
                    runner.req_to_token_pool.clear()
                    runner.token_to_kv_pool_allocator.clear()
                    torch.cuda.empty_cache()
                    if args.stop_on_shape_failure:
                        if rank == 0:
                            print(
                                f"[stop length] setup failed at length={length} "
                                f"batch={batch_size}"
                            )
                        break
                    continue

                experts = layer.mlp.experts
                for freq in pending:
                    ctrl.lock(freq)
                    time.sleep(args.clock_settle_seconds)
                    (
                        latency,
                        rank_latency,
                        rank_energy,
                        total_energy,
                        actual_repeat,
                        rank_cuda_latency,
                        rank_sync_wall_latency,
                    ) = profile_moe_kernel_core(
                        ctrl,
                        experts,
                        dispatch_output,
                        args.warmup,
                        args.repeat,
                        world,
                        args.phase,
                    )
                    ctrl.unlock()
                    if latency is None:
                        continue
                    if rank == 0:
                        row = {
                            "schema_version": SCHEMA_VERSION,
                            "status": "ok",
                            **base,
                            "freq_mhz": freq,
                            "latency_us": latency,
                            "latency_semantics": "max_rank_local_wall_excluding_barrier",
                            "latency_per_rank_us": rank_latency,
                            "latency_cuda_per_rank_us": rank_cuda_latency,
                            "latency_sync_us": max(rank_sync_wall_latency),
                            "latency_sync_per_rank_us": rank_sync_wall_latency,
                            "energy_per_rank_mj": rank_energy,
                            "energy_total_mj": total_energy,
                            "energy_sync_barrier": True,
                            "energy_measurement_semantics": (
                                "block_nvml_global_sync_amortized_per_repeat"
                            ),
                            "warmup": args.warmup,
                            "requested_repeat": args.repeat,
                            "actual_repeat": actual_repeat,
                            "layer_id": args.layer_id,
                            "model": server_args.model_path,
                            "request_seed": request_seed,
                            "hidden_seed": hidden_seed,
                            "routing": routing,
                            "kernel_segment": "moe_core",
                        }
                        append_row(out_f, row)
                        completed.add(row_key(row))
                        print(
                            f"K {args.phase} {args.forced_routing} l={length} "
                            f"b={batch_size} f={freq}: {latency:.1f} us, "
                            f"{total_energy:.3f} mJ"
                        )

                del fb, hidden, residual, dispatch_output
                torch.cuda.empty_cache()
    except KeyboardInterrupt:
        if rank == 0:
            print("Interrupted; completed JSONL rows are durable")
    finally:
        if out_f:
            out_f.close()
        ctrl.close()
        if world > 1:
            from sglang.srt.distributed.parallel_state import destroy_distributed_environment

            destroy_distributed_environment()


def add_kernel_args(parser: argparse.ArgumentParser, phase: str) -> None:
    from sglang.srt.server_args import ServerArgs

    ServerArgs.add_cli_args(parser)
    parser.set_defaults(
        model_path=DEFAULT_MODEL,
        moe_runner_backend="triton",
        moe_a2a_backend="none",
        cuda_graph_backend_decode="disabled",
        cuda_graph_backend_prefill="disabled",
    )
    parser.add_argument(
        "--forced-routing",
        choices=[c for c in FORCED_ROUTING_CHOICES if c != "natural"],
        required=True,
    )
    parser.add_argument(
        "--parallel-mode",
        choices=["moe_ep"],
        default="moe_ep",
    )
    parser.add_argument(
        "--freqs",
        type=int,
        nargs="+",
        default=KERNEL_EP_FREQS,
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=KERNEL_EP_LENGTHS,
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=KERNEL_EP_BATCHES,
    )
    parser.add_argument("--repeat", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--clock-settle-seconds", type=float, default=0.05)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local-world-size", type=int)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--shape-token-limit",
        type=int,
        default=None,
        help="per-shape token safety limit; 0 uses only runner.max_total_num_tokens",
    )
    parser.add_argument(
        "--stop-on-shape-failure",
        action="store_true",
    )


def validate_kernel_topology(parser, args, server_args) -> None:
    world = server_args.tp_size
    if args.parallel_mode != "moe_ep":
        parser.error("kernel profiler requires --parallel-mode moe_ep")
    if server_args.ep_size != world:
        parser.error("moe_ep requires --ep-size == --tp-size")
    if world % server_args.ep_size:
        parser.error("--tp-size must be divisible by --ep-size")


def profile_kernel_main(phase: str) -> None:
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.server_args import PortArgs, ServerArgs
    from sglang.srt.utils import maybe_reindex_device_id

    parser = argparse.ArgumentParser(
        description=f"Qwen3 MoE kernel-only profiler ({phase})"
    )
    add_kernel_args(parser, phase)
    args = parser.parse_args()
    args.phase = phase
    if phase == "decode" and any(x < 2 for x in args.lengths):
        parser.error("decode lengths must be >= 2")
    if args.shape_token_limit is not None and args.shape_token_limit < 0:
        parser.error("--shape-token-limit must be >= 0")
    if args.quick:
        args.lengths = [KERNEL_EP_LENGTHS[0]]
        args.batch_sizes = [1]
        args.freqs = [args.freqs[-1]]
        args.warmup = min(args.warmup, 2)
        args.repeat = min(args.repeat, 5)

    server_args = ServerArgs.from_cli_args(args)
    validate_kernel_topology(parser, args, server_args)
    _set_envs_and_config(server_args)
    if server_args.nnodes != 1:
        parser.error("this profiler currently supports one node only")
    local_world = args.local_world_size or server_args.tp_size
    if local_world != server_args.tp_size:
        parser.error("--local-world-size must equal --tp-size on one node")
    port_args = PortArgs.init_new(server_args)

    if local_world == 1:
        kernel_worker(server_args, port_args, args, 0, 0)
        return

    procs = []
    for rank in range(local_world):
        with maybe_reindex_device_id(rank) as gpu_id:
            proc = multiprocessing.Process(
                target=kernel_worker,
                args=(server_args, port_args, args, gpu_id, rank),
            )
            proc.start()
            procs.append(proc)
    for proc in procs:
        proc.join()
    failed = [(p.pid, p.exitcode) for p in procs if p.exitcode]
    if failed:
        raise SystemExit(f"kernel profiling workers failed: {failed}")


if __name__ == "__main__":
    raise SystemExit(profile_kernel_main(sys.argv[1] if len(sys.argv) > 1 else "decode"))
