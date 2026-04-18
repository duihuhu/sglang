#!/usr/bin/env python3
"""
Benchmark: AF communication overhead — GPU-to-GPU activation transfer latency.

Measures the time to send intermediate hidden states (the tensor transferred
between Attention GPU and FFN GPU in AF disaggregation) using NCCL.

Launch with torchrun (requires 2 GPUs):
    torchrun --nproc_per_node=2 hucc_comm.py
    torchrun --nproc_per_node=2 hucc_comm.py --quick
    torchrun --nproc_per_node=2 hucc_comm.py --output af_comm_overhead.txt
"""

import argparse
import os
import sys
import time

import torch
import torch.distributed as dist

HIDDEN_SIZE = 5120  # Qwen3-32B

DEFAULT_BATCH_SIZES = [1, 4, 16, 64, 256]
DEFAULT_SEQ_LENS = [1, 128, 1024, 8192]


def setup():
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        print(
            "错误：未检测到分布式环境变量（RANK / WORLD_SIZE）。\n"
            "本脚本需用 torchrun 启动（2 张 GPU，rank 0→1 的 send/recv）：\n"
            "  cd bash-test/comm_test && torchrun --nproc_per_node=2 hucc_comm.py\n"
            "若单机多卡，通常还需设置 MASTER_ADDR 与 MASTER_PORT，例如：\n"
            "  export MASTER_ADDR=127.0.0.1\n"
            "  export MASTER_PORT=29500",
            file=sys.stderr,
        )
        sys.exit(1)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    return rank


def bench_send_recv(rank, bs, seq_len, hidden_size, n_warmup, n_repeat):
    """Measure unidirectional send latency from rank 0 to rank 1."""
    tensor = torch.randn(bs, seq_len, hidden_size,
                         dtype=torch.bfloat16, device=f"cuda:{rank}")
    data_bytes = bs * seq_len * hidden_size * 2

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
    t1 = time.perf_counter()

    lat_us = (t1 - t0) / n_repeat * 1e6
    bw_gbps = data_bytes / (lat_us * 1e-6) / 1e9 if lat_us > 0 else 0
    return lat_us, bw_gbps, data_bytes


def main():
    parser = argparse.ArgumentParser(description="AF communication overhead benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--seq-lens", type=int, nargs="+", default=None)
    parser.add_argument("--hidden-size", type=int, default=HIDDEN_SIZE)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", type=str, default="af_comm_overhead.txt")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    rank = setup()

    batch_sizes = args.batch_sizes or DEFAULT_BATCH_SIZES
    seq_lens = args.seq_lens or DEFAULT_SEQ_LENS

    if args.quick:
        batch_sizes = [1, 16]
        seq_lens = [1, 1024]
        args.repeat = 50
        args.warmup = 5

    if rank == 0:
        print(f"{'='*60}")
        print(f" AF Communication Overhead Benchmark")
        print(f"{'='*60}")
        print(f"  GPUs:        rank 0 (cuda:0) -> rank 1 (cuda:1)")
        print(f"  Hidden size: {args.hidden_size}")
        print(f"  Batch sizes: {batch_sizes}")
        print(f"  Seq lens:    {seq_lens}")
        print(f"  Repeat:      {args.repeat}")
        print(f"{'='*60}\n")

    results = []

    for bs in batch_sizes:
        for sl in seq_lens:
            mem_gb = bs * sl * args.hidden_size * 2 / 1e9
            if mem_gb > 70:
                if rank == 0:
                    print(f"  bs={bs:>3} seq={sl:>5} | "
                          f"{mem_gb*1000:>8.2f} MB | SKIPPED (>{70:.0f} GB)")
                dist.barrier()
                continue

            try:
                lat_us, bw_gbps, data_bytes = bench_send_recv(
                    rank, bs, sl, args.hidden_size, args.warmup, args.repeat)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if rank == 0:
                    print(f"  bs={bs:>3} seq={sl:>5} | OOM — skipped")
                dist.barrier()
                continue

            if rank == 0:
                results.append({
                    "batch_size": bs,
                    "seq_len": sl,
                    "data_bytes": data_bytes,
                    "latency_us": round(lat_us, 2),
                    "bandwidth_gbps": round(bw_gbps, 2),
                })
                data_mb = data_bytes / 1e6
                print(f"  bs={bs:>3} seq={sl:>5} | "
                      f"{data_mb:>8.2f} MB | "
                      f"{lat_us:>10.2f} us | "
                      f"{bw_gbps:>7.2f} GB/s")

    if rank == 0:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.join(script_dir, args.output)
        with open(out_path, "w") as f:
            f.write("batch_size\tseq_len\tdata_bytes\tlatency_us\tbandwidth_gbps\n")
            for r in results:
                f.write(f"{r['batch_size']}\t{r['seq_len']}\t{r['data_bytes']}\t"
                        f"{r['latency_us']}\t{r['bandwidth_gbps']}\n")
        print(f"\nResults saved to {out_path}")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()