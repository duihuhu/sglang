#!/usr/bin/env python3
"""
Benchmark: AF communication overhead for AFD (Attention-FFN Disaggregation).

Measures three components of AFD inter-node communication latency:

  Mode 1 (p2p):       Single rank-to-rank transfer (simulates ZMQ/UCX cross-node)
                       Tensor = (bs, seq_len, H) — full hidden_states after all-reduce
                       Uses 2 GPUs: GPU 0 → GPU 1

  Mode 2 (broadcast):  NVLink broadcast from rank 0 to all other ranks
                       (simulates intra-node distribution after cross-node recv)
                       Uses N GPUs: GPU 0 → GPU 1..N-1

  Mode 3 (e2e):       End-to-end AFD communication = P2P + broadcast
                       First half GPUs = "Attn node", second half = "FFN node"
                       Attn rank 0 → FFN rank 0 → NVLink broadcast to FFN rank 1..
                       Uses N GPUs (even): GPU 0 sends, GPU N/2 recvs + broadcasts

In all modes the transferred tensor is the FULL hidden_states (bs, seq_len, H),
because AFD cross-node traffic is always the complete all-reduced tensor
regardless of TP degree. TP only affects intra-node parallelism.

Launch examples:
    # Mode 1: P2P baseline (2 GPUs)
    torchrun --nproc_per_node=2 bench_af_comm.py --mode p2p

    # Mode 2: NVLink broadcast (4 GPUs, rank 0 → rank 1,2,3)
    torchrun --nproc_per_node=4 bench_af_comm.py --mode broadcast

    # Mode 3: end-to-end (8 GPUs, Attn[0:4] → FFN[4:8])
    torchrun --nproc_per_node=8 bench_af_comm.py --mode e2e

    # All modes quick test
    torchrun --nproc_per_node=2 bench_af_comm.py --mode p2p --quick
    torchrun --nproc_per_node=4 bench_af_comm.py --mode broadcast --quick
    torchrun --nproc_per_node=8 bench_af_comm.py --mode e2e --quick

Output columns:
    mode  n_gpus  batch_size  seq_len  data_bytes  latency_us  bandwidth_gbps
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

HIDDEN_SIZE = 5120  # Qwen3-32B, full hidden (not divided by TP)

# Aligned with profiling data configs
# seq_len=1 → Decode (1 token), others → Prefill (= input_len)
DEFAULT_BATCH_SIZES = [1, 4, 16, 64, 128, 256]
DEFAULT_SEQ_LENS = [1, 128, 512, 1024, 2048, 4096, 8192]


def setup():
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    return rank, world_size


# ── Mode 1: Point-to-point (simulates ZMQ/UCX cross-node) ──────────────

def bench_p2p(rank, tensor, n_warmup, n_repeat):
    """Rank 0 sends full tensor to rank 1. Returns latency in us."""
    for _ in range(n_warmup):
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)
        dist.barrier()

    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        if rank == 0:
            dist.send(tensor, dst=1)
        else:
            dist.recv(tensor, src=0)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / n_repeat * 1e6


# ── Mode 2: NVLink broadcast (simulates intra-node distribution) ───────

def bench_broadcast(rank, tensor, n_warmup, n_repeat):
    """Rank 0 broadcasts full tensor to all ranks via NCCL broadcast."""
    for _ in range(n_warmup):
        dist.broadcast(tensor, src=0)
        dist.barrier()

    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        dist.broadcast(tensor, src=0)
    torch.cuda.synchronize()
    dist.barrier()
    return (time.perf_counter() - t0) / n_repeat * 1e6


# ── Mode 3: End-to-end (P2P cross-node + broadcast intra-node) ────────

def bench_e2e(rank, world_size, tensor, n_warmup, n_repeat):
    """Attn rank 0 → FFN rank 0 (P2P) → FFN broadcast to FFN ranks.

    GPU layout: [0 .. half-1] = Attn node, [half .. world_size-1] = FFN node
    Only Attn rank 0 sends; FFN rank 0 receives then broadcasts within FFN group.
    """
    half = world_size // 2
    is_attn = rank < half
    ffn_ranks = list(range(half, world_size))
    ffn_group = dist.new_group(ffn_ranks)

    for _ in range(n_warmup):
        if rank == 0:
            dist.send(tensor, dst=half)
        elif rank == half:
            dist.recv(tensor, src=0)
        dist.barrier()
        if not is_attn:
            dist.broadcast(tensor, src=half, group=ffn_group)
        dist.barrier()

    torch.cuda.synchronize()
    dist.barrier()
    t0 = time.perf_counter()
    for _ in range(n_repeat):
        if rank == 0:
            dist.send(tensor, dst=half)
        elif rank == half:
            dist.recv(tensor, src=0)
        # Attn ranks wait at barrier while FFN does broadcast
        dist.barrier()
        if not is_attn:
            dist.broadcast(tensor, src=half, group=ffn_group)
    torch.cuda.synchronize()
    dist.barrier()
    lat = (time.perf_counter() - t0) / n_repeat * 1e6

    dist.destroy_process_group(ffn_group)
    return lat


def main():
    parser = argparse.ArgumentParser(
        description="AFD communication overhead benchmark")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["p2p", "broadcast", "e2e"],
                        help="p2p: rank0→rank1 | broadcast: rank0→all | "
                             "e2e: P2P + broadcast")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default=None,
                        help="Output file (default: af_comm_{mode}.txt)")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    rank, world_size = setup()
    mode = args.mode

    if mode == "e2e":
        assert world_size >= 4 and world_size % 2 == 0, \
            f"e2e mode needs >=4 even GPUs, got {world_size}"
    if mode == "p2p":
        assert world_size >= 2, f"p2p mode needs >=2 GPUs, got {world_size}"

    batch_sizes = args.batch_sizes or DEFAULT_BATCH_SIZES
    seq_lens = args.seq_lens or DEFAULT_SEQ_LENS

    if args.quick:
        batch_sizes = [1, 16, 128]
        seq_lens = [1, 1024, 4096]
        args.repeat = 50
        args.warmup = 5

    output_file = args.output or f"af_comm_{mode}.txt"

    if rank == 0:
        half = world_size // 2
        print(f"{'=' * 70}")
        print(f" AFD Communication Benchmark — mode: {mode}")
        print(f"{'=' * 70}")
        print(f"  World size:   {world_size} GPUs")
        print(f"  Hidden size:  {args.hidden_size} (Qwen3-32B, full H)")
        print(f"  Tensor shape: (bs, seq_len, {args.hidden_size})")
        print(f"  Batch sizes:  {batch_sizes}")
        print(f"  Seq lens:     {seq_lens}")
        print(f"  Repeat:       {args.repeat}")
        if mode == "p2p":
            print(f"  Flow:         GPU 0 → GPU 1 (single P2P)")
        elif mode == "broadcast":
            print(f"  Flow:         GPU 0 → GPU 0..{world_size - 1} "
                  f"(NCCL broadcast)")
        elif mode == "e2e":
            print(f"  Flow:         Attn[0] → FFN[{half}] (P2P) → "
                  f"FFN[{half}..{world_size - 1}] (broadcast)")
        print(f"{'=' * 70}\n")

    results = []

    for bs in batch_sizes:
        for sl in seq_lens:
            data_bytes = bs * sl * args.hidden_size * 2  # bf16
            mem_gb = data_bytes / 1e9
            if mem_gb > 70:
                if rank == 0:
                    print(f"  bs={bs:>3} seq={sl:>5} | SKIPPED (>{70:.0f} GB)")
                dist.barrier()
                continue

            try:
                tensor = torch.randn(
                    bs, sl, args.hidden_size,
                    dtype=torch.bfloat16, device=f"cuda:{rank}")

                if mode == "p2p":
                    lat_us = bench_p2p(rank, tensor, args.warmup, args.repeat)
                elif mode == "broadcast":
                    lat_us = bench_broadcast(
                        rank, tensor, args.warmup, args.repeat)
                elif mode == "e2e":
                    lat_us = bench_e2e(
                        rank, world_size, tensor, args.warmup, args.repeat)

                del tensor
                torch.cuda.empty_cache()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if rank == 0:
                    print(f"  bs={bs:>3} seq={sl:>5} | OOM — skipped")
                dist.barrier()
                continue

            if rank == 0:
                bw_gbps = (data_bytes / (lat_us * 1e-6) / 1e9
                           if lat_us > 0 else 0)
                results.append({
                    "mode": mode,
                    "n_gpus": world_size,
                    "batch_size": bs,
                    "seq_len": sl,
                    "data_bytes": data_bytes,
                    "latency_us": round(lat_us, 2),
                    "bandwidth_gbps": round(bw_gbps, 2),
                })
                data_mb = data_bytes / 1e6
                print(f"  bs={bs:>3} seq={sl:>5} | "
                      f"{data_mb:>8.2f} MB | "
                      f"{lat_us:>10.1f} us | "
                      f"{bw_gbps:>6.1f} GB/s")

    if rank == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, output_file)
        with open(out_path, "w") as f:
            f.write("mode\tn_gpus\tbatch_size\tseq_len\t"
                    "data_bytes\tlatency_us\tbandwidth_gbps\n")
            for r in results:
                f.write(f"{r['mode']}\t{r['n_gpus']}\t{r['batch_size']}\t"
                        f"{r['seq_len']}\t{r['data_bytes']}\t"
                        f"{r['latency_us']}\t{r['bandwidth_gbps']}\n")
        print(f"\nResults saved to {out_path}")
        print(f"Total measurements: {len(results)}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
