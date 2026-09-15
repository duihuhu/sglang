#!/usr/bin/env python3
"""Single- or multi-node Mixtral EP smoke using a real ModelRunner."""
import argparse
import logging
import multiprocessing

import torch

from ep_profiling_utils import parallel_identity, stable_shape_seed
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.entrypoints.engine import _set_envs_and_config
from sglang.srt.model_executor.model_runner import ModelRunner
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import maybe_reindex_device_id, suppress_other_loggers
from ep_profiling_utils import initialize_worker_configs


def load_model(server_args, port_args, gpu_id, tp_rank):
    suppress_other_loggers()
    initialize_worker_configs(server_args)
    model_config = ModelConfig.from_server_args(server_args)
    moe_ep_rank = tp_rank // (server_args.tp_size // server_args.ep_size)
    runner = ModelRunner(
        model_config=model_config,
        mem_fraction_static=server_args.mem_fraction_static,
        gpu_id=gpu_id,
        tp_rank=tp_rank,
        tp_size=server_args.tp_size,
        moe_ep_rank=moe_ep_rank,
        moe_ep_size=server_args.ep_size,
        pp_rank=0,
        pp_size=1,
        nccl_port=port_args.nccl_port,
        server_args=server_args,
    )
    if server_args.tp_size > 1:
        if server_args.moe_a2a_backend == "mooncake":
            from sglang.srt.distributed.parallel_state import get_tp_group

            torch.distributed.barrier(group=get_tp_group().cpu_group)
        else:
            torch.distributed.barrier()
    return runner


def worker(server_args, port_args, args, gpu_id, rank):
    runner = load_model(server_args, port_args, gpu_id, rank)
    layer = runner.model.model.layers[args.layer_id]
    moe = layer.mlp
    experts = moe.experts
    layout = parallel_identity(server_args, args.layer_id)
    local_start = experts.moe_ep_rank * (experts.num_experts // experts.moe_ep_size)
    local_ids = list(range(local_start, local_start + experts.num_local_experts))
    print({"rank": rank, "placement": "contiguous", "layout": layout,
           "dispatcher": type(experts.dispatcher).__name__,
           "local_experts": local_ids,
           "expert_module": type(experts).__name__}, flush=True)
    seed = stable_shape_seed(args.seed, "smoke", layout, args.tokens, 1, "hidden_states")
    gen = torch.Generator(device=runner.device); gen.manual_seed(seed)
    hidden = torch.randn(args.tokens, runner.model_config.hidden_size,
                         device=runner.device, dtype=torch.bfloat16, generator=gen)
    residual = hidden.clone()
    from sglang.srt.layers.dp_attention import set_is_extend_in_batch
    set_is_extend_in_batch(args.phase == "prefill")
    with torch.no_grad():
        normalized, _ = layer.post_attention_layernorm(hidden, residual)
        output = layer.mlp(normalized)
    torch.cuda.synchronize()
    print({"rank": rank, "f_forward": "ok", "input_shape": list(hidden.shape),
           "output_shape": list(output.shape), "output_dtype": str(output.dtype)}, flush=True)
    if server_args.tp_size > 1:
        from sglang.srt.distributed.parallel_state import destroy_distributed_environment
        destroy_distributed_environment()


def main():
    parser = argparse.ArgumentParser(description="Load real Mixtral ModelRunner and run one EP F forward")
    ServerArgs.add_cli_args(parser)
    parser.add_argument("--tokens", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layer-id", type=int, default=0)
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--local-world-size", type=int, default=None,
                        help="workers to spawn on this node (default: tp_size / nnodes)")
    args = parser.parse_args()
    server_args = ServerArgs.from_cli_args(args)
    _set_envs_and_config(server_args)
    server_args.disable_cuda_graph = True
    server_args.disable_cuda_graph_padding = True
    server_args.disable_piecewise_cuda_graph = True
    if server_args.tp_size % server_args.nnodes != 0:
        parser.error("--tp-size must be divisible by --nnodes")
    expected_local_world_size = server_args.tp_size // server_args.nnodes
    local_world_size = (expected_local_world_size if args.local_world_size is None
                        else args.local_world_size)
    if local_world_size < 1:
        parser.error("--local-world-size must be positive")
    if local_world_size != expected_local_world_size:
        parser.error("--local-world-size must equal tp_size / nnodes")
    if server_args.nnodes > 1 and not server_args.dist_init_addr:
        parser.error("multi-node launch requires --dist-init-addr HOST:PORT")
    port_args = PortArgs.init_new(server_args)
    if server_args.nnodes == 1 and local_world_size == 1:
        worker(server_args, port_args, args, 0, 0)
        return
    workers = []
    for local_rank in range(local_world_size):
        tp_rank = server_args.node_rank * local_world_size + local_rank
        if server_args.nnodes == 1:
            with maybe_reindex_device_id(local_rank) as gpu_id:
                proc = multiprocessing.Process(
                    target=worker,
                    args=(server_args, port_args, args, gpu_id, tp_rank),
                )
                proc.start()
        else:
            proc = multiprocessing.Process(
                target=worker,
                args=(server_args, port_args, args, local_rank, tp_rank),
            )
            proc.start()
        workers.append(proc)
    for proc in workers:
        proc.join()
    failed = [(proc.pid, proc.exitcode) for proc in workers if proc.exitcode != 0]
    if failed:
        raise SystemExit(f"smoke workers failed (pid, exitcode): {failed}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING, format="%(message)s"); main()
