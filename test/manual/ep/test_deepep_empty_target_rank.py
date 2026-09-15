"""DeepEP normal empty-target rank protocol test for Ampere (A800).

Exercises dispatch/combine when one or more EP ranks receive zero tokens.
This is the decode-path failure mode observed with Qwen3 on A800.
"""

import argparse
import sys

import deep_ep
import torch
import torch.distributed as dist

from sglang.test.test_deepep_utils import calc_diff, init_dist


def _make_buffer(group, hidden: int, num_experts: int, num_max_tokens: int):
    num_rdma_bytes = 0
    if group.size() > 8:
        num_rdma_bytes = deep_ep.Buffer.get_combine_config(
            group.size()
        ).get_rdma_buffer_size_hint(hidden * 2, group.size())
    return deep_ep.Buffer(
        group,
        int(1e9),
        num_rdma_bytes,
        low_latency_mode=False,
        num_qps_per_rank=1,
        allow_mnnvl=False,
    )


def _route_all_to_rank(
    num_tokens: int,
    num_experts: int,
    num_ranks: int,
    target_rank: int,
    num_topk: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Route every token to experts owned by ``target_rank``."""
    experts_per_rank = num_experts // num_ranks
    expert_start = target_rank * experts_per_rank
    topk_ids = (
        torch.arange(
            expert_start, expert_start + num_topk, dtype=torch.int64, device=device
        )
        .unsqueeze(0)
        .repeat(num_tokens, 1)
        .contiguous()
    )
    topk_weights = torch.ones(
        (num_tokens, num_topk), dtype=torch.float32, device=device
    )
    return topk_ids, topk_weights


def _run_step(
    buffer: deep_ep.Buffer,
    group: dist.ProcessGroup,
    x: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    num_experts: int,
    async_finish: bool,
) -> tuple[bool, str]:
    num_ranks = group.size()
    config = deep_ep.Buffer.get_dispatch_config(num_ranks)
    previous_event = buffer.capture() if async_finish else None
    (
        num_tokens_per_rank,
        num_tokens_per_rdma_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        previous_event,
    ) = buffer.get_dispatch_layout(
        topk_ids,
        num_experts,
        previous_event=previous_event,
        async_finish=async_finish,
        allocate_on_comm_stream=previous_event is not None,
    )
    recv_x, recv_topk_ids, recv_topk_weights, recv_counts, handle, event = (
        buffer.dispatch(
            x,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            num_tokens_per_rank=num_tokens_per_rank,
            num_tokens_per_rdma_rank=num_tokens_per_rdma_rank,
            is_token_in_rank=is_token_in_rank,
            num_tokens_per_expert=num_tokens_per_expert,
            previous_event=previous_event,
            async_finish=async_finish,
            allocate_on_comm_stream=previous_event is not None,
            expert_alignment=1,
            config=config,
        )
    )
    event.current_stream_wait() if async_finish else ()
    recv_x = recv_x[0] if isinstance(recv_x, tuple) else recv_x

    combine_args = {
        "x": recv_x,
        "handle": handle,
        "topk_weights": recv_topk_weights,
        "config": deep_ep.Buffer.get_combine_config(num_ranks),
        "async_finish": async_finish,
    }
    if async_finish:
        combine_args["previous_event"] = buffer.capture()
        combine_args["allocate_on_comm_stream"] = True
    combined_x, _, combine_event = buffer.combine(**combine_args)
    combine_event.current_stream_wait() if async_finish else ()

    denom = is_token_in_rank.sum(dim=1).clamp_min(1).unsqueeze(1).float()
    diff = calc_diff(combined_x.float() / denom, x.float())
    meta = (
        f"recv_tokens={recv_x.size(0)} recv_counts={recv_counts} "
        f"gbl_recv={num_tokens_per_rank.tolist()} diff={diff:.3e}"
    )
    return diff < 5e-6, meta


def _worker(local_rank: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, args.num_local_ranks)
    num_ranks = world_size
    assert args.num_experts % num_ranks == 0

    buffer = _make_buffer(group, args.hidden, args.num_experts, 128)
    device = "cuda"

    scenarios = []
    for num_tokens in (1, 2, 4, 8):
        for target_rank in range(num_ranks):
            scenarios.append((num_tokens, target_rank))

    failures = []
    for round_idx in range(args.rounds):
        for num_tokens, target_rank in scenarios:
            x = torch.randn(
                (num_tokens, args.hidden), dtype=torch.bfloat16, device=device
            )
            topk_ids, topk_weights = _route_all_to_rank(
                num_tokens,
                args.num_experts,
                num_ranks,
                target_rank,
                args.num_topk,
                device,
            )
            for async_finish in (True,):
                try:
                    ok, meta = _run_step(
                        buffer,
                        group,
                        x,
                        topk_ids,
                        topk_weights,
                        args.num_experts,
                        async_finish,
                    )
                    if not ok:
                        failures.append(
                            f"round={round_idx} tokens={num_tokens} "
                            f"target={target_rank} async={async_finish} {meta}"
                        )
                except Exception as exc:
                    failures.append(
                        f"round={round_idx} tokens={num_tokens} "
                        f"target={target_rank} async={async_finish} "
                        f"error={type(exc).__name__}: {exc}"
                    )
        dist.barrier(group=group)

    if local_rank == 0:
        if failures:
            print(f"FAILED: {len(failures)} cases", flush=True)
            for item in failures[:20]:
                print(item, flush=True)
            sys.exit(1)
        print(
            f"PASSED: {args.rounds * len(scenarios)} empty-target cases "
            f"on {num_ranks} ranks",
            flush=True,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-local-ranks", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--num-experts", type=int, default=128)
    parser.add_argument("--num-topk", type=int, default=8)
    parser.add_argument("--rounds", type=int, default=50)
    cli_args = parser.parse_args()

    torch.multiprocessing.spawn(
        _worker,
        args=(cli_args,),
        nprocs=cli_args.num_local_ranks,
    )
