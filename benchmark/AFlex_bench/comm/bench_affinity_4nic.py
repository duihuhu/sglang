#!/usr/bin/env python3
"""Test 4-NIC GPUDirect RDMA with proper PCIe affinity.

Affinity mapping (from nvidia-smi topo -m):
  GPU0 <-> mlx5_0 (PXB, same PCIe switch)
  GPU2 <-> mlx5_1 (PXB)
  GPU4 <-> mlx5_4 (PXB)
  GPU6 <-> mlx5_5 (PXB)

Usage:
  # node2 (receiver):
  python3 bench_affinity_4nic.py receiver

  # node1 (sender):
  python3 bench_affinity_4nic.py sender --remote-host 10.252.129.35
"""
import argparse
import json
import multiprocessing as mp
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

AFFINITY_MAP = [
    (0, "mlx5_0"),
    (2, "mlx5_1"),
    (4, "mlx5_4"),
    (6, "mlx5_5"),
]

SIZES = [
    1024, 4096, 8192, 16384, 32768, 65536,
    131072, 262144, 524288, 1048576, 2097152,
    4194304, 8388608, 16777216, 33554432, 67108864,
]


def worker_sender(gpu_id, ib_device, remote_host, port_offset, result_slot):
    from bench_mooncake_rdma_write import run_sender, BENCH_ITERS, WARMUP_ITERS
    r = run_sender(gpu_id, remote_host, SIZES, ib_device=ib_device,
                   port_offset=port_offset, use_batch=False, batch_count=1)
    result_slot[gpu_id] = dict(r)


def worker_receiver(gpu_id, ib_device, port_offset):
    from bench_mooncake_rdma_write import run_receiver
    run_receiver(gpu_id, SIZES, ib_device=ib_device, port_offset=port_offset)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("role", choices=["sender", "receiver"])
    parser.add_argument("--remote-host", default="10.252.129.35")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--outfile", default="results/rdma_4nic_affinity.json")
    args = parser.parse_args()

    import bench_mooncake_rdma_write as _mod
    _mod.BENCH_ITERS = args.iters
    _mod.WARMUP_ITERS = args.warmup

    if args.role == "receiver":
        procs = []
        for idx, (gpu_id, ib_dev) in enumerate(AFFINITY_MAP):
            p = mp.Process(target=worker_receiver,
                           args=(gpu_id, ib_dev, idx * 10))
            procs.append(p)
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=600)
        print("All receivers done.")

    elif args.role == "sender":
        manager = mp.Manager()
        all_results = manager.dict()

        procs = []
        for idx, (gpu_id, ib_dev) in enumerate(AFFINITY_MAP):
            p = mp.Process(target=worker_sender,
                           args=(gpu_id, ib_dev, args.remote_host, idx * 10, all_results))
            procs.append(p)
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=600)

        # Aggregate
        print("\n" + "=" * 90)
        print("  4-NIC GPUDirect RDMA Write (PCIe-affine: GPU0/mlx5_0, GPU2/mlx5_1, GPU4/mlx5_4, GPU6/mlx5_5)")
        print("=" * 90)
        print(f"  {'Size':>12} | {'GPU0/mlx5_0':>12} | {'GPU2/mlx5_1':>12} | "
              f"{'GPU4/mlx5_4':>12} | {'GPU6/mlx5_5':>12} | {'Aggregate':>12}")
        print(f"  {'-' * 80}")

        aggregate = {}
        for size in SIZES:
            bws = []
            parts = []
            for gpu_id, _ in AFFINITY_MAP:
                if gpu_id in all_results and size in all_results[gpu_id]:
                    bw = all_results[gpu_id][size]["bw_GBs"]
                    bws.append(bw)
                    parts.append(f"{bw:>9.2f}GB/s")
                else:
                    parts.append(f"{'N/A':>12}")
            total_bw = sum(bws)
            aggregate[size] = {
                "size_bytes": size,
                "per_nic": {str(gpu_id): all_results[gpu_id][size]
                            for gpu_id, _ in AFFINITY_MAP
                            if gpu_id in all_results and size in all_results[gpu_id]},
                "aggregate_bw_GBs": round(total_bw, 3),
            }
            print(f"  {size:>12} | {' | '.join(parts)} | {total_bw:>9.2f}GB/s")

        print()

        outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.outfile)
        os.makedirs(os.path.dirname(outpath), exist_ok=True)
        with open(outpath, "w") as f:
            json.dump({"aggregate": {str(k): v for k, v in aggregate.items()}}, f, indent=2)
        print(f"Results saved to {outpath}")


if __name__ == "__main__":
    main()
