#!/usr/bin/env python3
"""Mooncake Transfer Engine RDMA Write benchmark — matches sglang PD KV transfer path.

SGLang's cross-node KV cache transfer uses mooncake's `transfer_sync_write` /
`batch_transfer_sync_write` (one-sided RDMA Write into pre-registered remote GPU memory).
This benchmark measures that exact primitive at various message sizes.

Supports:
  - Single NIC (1x RDMA device)
  - Multi-NIC (up to 4x RDMA devices, one engine per GPU)
  - NVLink baseline (same-node GPU-to-GPU cudaMemcpyPeerAsync)

Usage (cross-node, single NIC):
    # node2 (receiver, 10.252.129.35):
    python3 bench_mooncake_rdma_write.py receiver --gpu 0

    # node1 (sender, 10.252.129.36):
    python3 bench_mooncake_rdma_write.py sender --gpu 0 --remote-host 10.252.129.35

Multi-NIC (4 GPUs, 4 NICs):
    # node2:
    python3 bench_mooncake_rdma_write.py receiver --multi-nic 4

    # node1:
    python3 bench_mooncake_rdma_write.py sender --multi-nic 4 --remote-host 10.252.129.35

NVLink baseline (same node):
    python3 bench_mooncake_rdma_write.py nvlink --src-gpu 0 --dst-gpu 4
"""

import argparse
import json
import os
import socket
import struct
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

# Benchmark parameters
WARMUP_ITERS = 50
BENCH_ITERS = 200
SYNC_PORT_BASE = 29500

# Message sizes: match KV cache transfer granularity
# Qwen3-32B: hidden=5120, num_kv_heads=8, head_dim=128, bf16
# Per-token KV per layer: 2 * 8 * 128 * 2 = 4096 bytes (K+V)
# Per-token all 64 layers: 64 * 4096 = 262144 bytes = 256KB
DEFAULT_SIZES = [
    1024,         # 1KB
    4096,         # 4KB - 1 token KV per layer
    8192,         # 8KB
    16384,        # 16KB
    32768,        # 32KB
    65536,        # 64KB
    131072,       # 128KB
    262144,       # 256KB - 1 token all layers
    524288,       # 512KB
    1048576,      # 1MB - ~4 tokens all layers
    2097152,      # 2MB
    4194304,      # 4MB - ~16 tokens all layers
    8388608,      # 8MB
    16777216,     # 16MB - ~64 tokens all layers
    33554432,     # 32MB
    67108864,     # 64MB - ~256 tokens all layers
]

# AFD-specific sizes (Qwen3-32B, hidden=5120, bf16=10240 bytes/token for activation)
AFD_SIZES = [
    10240,        # 1 token activation
    40960,        # 4 tokens
    81920,        # 8 tokens
    163840,       # 16 tokens
    327680,       # 32 tokens
    655360,       # 64 tokens
    1310720,      # 128 tokens
    2621440,      # 256 tokens
    5242880,      # 512 tokens
    10485760,     # 1024 tokens
]


def get_local_ip():
    """Get local IP address."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.252.129.35", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def sync_barrier(role: str, remote_host: str, port: int):
    """Simple TCP barrier for synchronization between sender and receiver."""
    if role == "receiver":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        conn, _ = srv.accept()
        conn.recv(4)
        conn.send(b"ACK!")
        conn.close()
        srv.close()
    else:
        for _ in range(300):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((remote_host, port))
                s.send(b"SYN!")
                s.recv(4)
                s.close()
                return
            except ConnectionRefusedError:
                time.sleep(0.1)
        raise RuntimeError(f"Cannot connect to {remote_host}:{port}")


def exchange_metadata(role: str, remote_host: str, port: int, local_data: bytes) -> bytes:
    """Exchange metadata (session_id, buffer pointers) between sender and receiver."""
    if role == "receiver":
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", port))
        srv.listen(1)
        conn, _ = srv.accept()
        # Send our data
        conn.send(struct.pack("!I", len(local_data)))
        conn.send(local_data)
        # Recv peer data
        sz = struct.unpack("!I", conn.recv(4))[0]
        peer_data = b""
        while len(peer_data) < sz:
            peer_data += conn.recv(sz - len(peer_data))
        conn.close()
        srv.close()
        return peer_data
    else:
        for _ in range(300):
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.connect((remote_host, port))
                # Recv peer data
                sz = struct.unpack("!I", s.recv(4))[0]
                peer_data = b""
                while len(peer_data) < sz:
                    peer_data += s.recv(sz - len(peer_data))
                # Send our data
                s.send(struct.pack("!I", len(local_data)))
                s.send(local_data)
                s.close()
                return peer_data
            except ConnectionRefusedError:
                time.sleep(0.1)
        raise RuntimeError(f"Cannot connect to {remote_host}:{port}")


class MooncakeBenchmark:
    """Benchmark using mooncake transfer engine (RDMA Write) — same as sglang PD transfer."""

    def __init__(self, gpu_id: int, ib_device: Optional[str] = None):
        self.gpu_id = gpu_id
        self.ib_device = ib_device

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        torch.cuda.set_device(0)

        from mooncake.engine import TransferEngine
        self.engine = TransferEngine()

        local_ip = get_local_ip()
        hostname = local_ip
        device_name = ib_device if ib_device else ""

        ret = self.engine.initialize(hostname, "P2PHANDSHAKE", "rdma", device_name)
        if ret != 0:
            raise RuntimeError(f"Mooncake init failed (ret={ret}), gpu={gpu_id}, ib={ib_device}")

        self.session_id = f"{local_ip}:{self.engine.get_rpc_port()}"
        print(f"[GPU{gpu_id}] Mooncake engine initialized: session={self.session_id}, ib={ib_device}")

    def allocate_and_register(self, max_size: int) -> Tuple[torch.Tensor, int]:
        """Allocate GPU buffer and register with mooncake."""
        buf = torch.empty(max_size // 2, dtype=torch.bfloat16, device="cuda:0")
        ptr = buf.data_ptr()
        ret = self.engine.register_memory(ptr, max_size)
        if ret != 0:
            raise RuntimeError(f"register_memory failed: ptr={ptr}, size={max_size}")
        return buf, ptr

    def transfer_sync_write(self, remote_session: str, src_ptr: int, dst_ptr: int, length: int) -> int:
        """Single RDMA write — matches sglang's engine.transfer_sync_write()."""
        return self.engine.transfer_sync_write(remote_session, src_ptr, dst_ptr, length)

    def batch_transfer_sync_write(self, remote_session: str,
                                   src_ptrs: List[int], dst_ptrs: List[int],
                                   lengths: List[int]) -> int:
        """Batch RDMA write — matches sglang's engine.batch_transfer_sync_write()."""
        return self.engine.batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)


def run_sender(gpu_id: int, remote_host: str, sizes: List[int],
               ib_device: Optional[str] = None, port_offset: int = 0,
               use_batch: bool = False, batch_count: int = 1) -> Dict:
    """Run sender (writer) side benchmark."""
    bench = MooncakeBenchmark(gpu_id, ib_device)
    max_size = max(sizes) * batch_count
    buf, src_ptr = bench.allocate_and_register(max_size)

    # Fill with pattern for verification
    buf.fill_(1.0)
    torch.cuda.synchronize()

    meta_port = SYNC_PORT_BASE + port_offset
    sync_port = SYNC_PORT_BASE + 100 + port_offset

    # Exchange session IDs and buffer pointers
    local_meta = json.dumps({
        "session_id": bench.session_id,
        "ptr": src_ptr,
    }).encode()
    peer_meta_bytes = exchange_metadata("sender", remote_host, meta_port, local_meta)
    peer_meta = json.loads(peer_meta_bytes.decode())
    remote_session = peer_meta["session_id"]
    remote_ptr = peer_meta["ptr"]

    print(f"[Sender GPU{gpu_id}] Remote: session={remote_session}, ptr={remote_ptr}")
    print(f"[Sender GPU{gpu_id}] {'Batch' if use_batch else 'Single'} transfer mode, "
          f"batch_count={batch_count}")

    # Wait for both sides ready
    sync_barrier("sender", remote_host, sync_port)

    results = {}
    for size in sizes:
        torch.cuda.synchronize()

        # Warmup
        for _ in range(WARMUP_ITERS):
            if use_batch and batch_count > 1:
                src_ptrs = [src_ptr + i * size for i in range(batch_count)]
                dst_ptrs = [remote_ptr + i * size for i in range(batch_count)]
                lengths = [size] * batch_count
                bench.batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)
            else:
                bench.transfer_sync_write(remote_session, src_ptr, remote_ptr, size)

        # Benchmark
        latencies = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            if use_batch and batch_count > 1:
                src_ptrs = [src_ptr + i * size for i in range(batch_count)]
                dst_ptrs = [remote_ptr + i * size for i in range(batch_count)]
                lengths = [size] * batch_count
                ret = bench.batch_transfer_sync_write(remote_session, src_ptrs, dst_ptrs, lengths)
            else:
                ret = bench.transfer_sync_write(remote_session, src_ptr, remote_ptr, size)
            t1 = time.perf_counter()
            if ret != 0:
                print(f"  WARNING: transfer failed ret={ret} at size={size}")
                break
            latencies.append((t1 - t0) * 1e6)  # us

        if latencies:
            total_bytes = size * batch_count
            lat_sorted = sorted(latencies)
            n = len(lat_sorted)
            p50 = lat_sorted[n // 2]
            p95 = lat_sorted[int(n * 0.95)]
            p99 = lat_sorted[int(n * 0.99)]
            bw = total_bytes / (p50 * 1e-6) / 1e9  # GB/s

            results[size] = {
                "size_bytes": size,
                "total_bytes": total_bytes,
                "batch_count": batch_count,
                "p50_us": round(p50, 2),
                "p95_us": round(p95, 2),
                "p99_us": round(p99, 2),
                "bw_GBs": round(bw, 3),
                "latencies_us": [round(x, 2) for x in latencies],
            }
            print(f"  {total_bytes:>10} B ({size:>10}×{batch_count}): "
                  f"p50={p50:>8.1f} us  p95={p95:>8.1f} us  BW={bw:>7.2f} GB/s")

    # Signal completion
    sync_barrier("sender", remote_host, sync_port + 1)
    return results


def run_receiver(gpu_id: int, sizes: List[int],
                 ib_device: Optional[str] = None, port_offset: int = 0,
                 batch_count: int = 1) -> None:
    """Run receiver side — just registers memory and waits."""
    bench = MooncakeBenchmark(gpu_id, ib_device)
    max_size = max(sizes) * batch_count
    buf, dst_ptr = bench.allocate_and_register(max_size)
    buf.zero_()
    torch.cuda.synchronize()

    meta_port = SYNC_PORT_BASE + port_offset
    sync_port = SYNC_PORT_BASE + 100 + port_offset

    # Exchange metadata
    local_meta = json.dumps({
        "session_id": bench.session_id,
        "ptr": dst_ptr,
    }).encode()
    peer_meta_bytes = exchange_metadata("receiver", "", meta_port, local_meta)
    peer_meta = json.loads(peer_meta_bytes.decode())
    print(f"[Receiver GPU{gpu_id}] Peer: session={peer_meta['session_id']}")

    # Signal ready
    sync_barrier("receiver", "", sync_port)

    # Wait for sender to finish
    sync_barrier("receiver", "", sync_port + 1)
    print(f"[Receiver GPU{gpu_id}] Done.")


def run_nvlink_benchmark(src_gpu: int, dst_gpu: int, sizes: List[int]) -> Dict:
    """NVLink (cudaMemcpyPeerAsync) benchmark for same-node comparison."""
    print(f"[NVLink] Benchmarking GPU{src_gpu} -> GPU{dst_gpu}")

    torch.cuda.set_device(src_gpu)
    results = {}

    for size in sizes:
        num_elem = max(1, size // 2)
        src_buf = torch.randn(num_elem, dtype=torch.bfloat16, device=f"cuda:{src_gpu}")
        dst_buf = torch.empty(num_elem, dtype=torch.bfloat16, device=f"cuda:{dst_gpu}")

        # Warmup
        for _ in range(WARMUP_ITERS):
            dst_buf.copy_(src_buf, non_blocking=True)
            torch.cuda.synchronize()

        # Benchmark
        latencies = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            dst_buf.copy_(src_buf, non_blocking=True)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1e6)

        lat_sorted = sorted(latencies)
        n = len(lat_sorted)
        p50 = lat_sorted[n // 2]
        p95 = lat_sorted[int(n * 0.95)]
        p99 = lat_sorted[int(n * 0.99)]
        bw = size / (p50 * 1e-6) / 1e9

        results[size] = {
            "size_bytes": size,
            "p50_us": round(p50, 2),
            "p95_us": round(p95, 2),
            "p99_us": round(p99, 2),
            "bw_GBs": round(bw, 3),
            "latencies_us": [round(x, 2) for x in latencies],
        }
        print(f"  {size:>10} B: p50={p50:>8.1f} us  p95={p95:>8.1f} us  BW={bw:>7.2f} GB/s")

        del src_buf, dst_buf
        torch.cuda.empty_cache()

    return results


def run_multi_nic_sender(num_nics: int, remote_host: str, sizes: List[int],
                         ib_devices: List[str], use_batch: bool = False,
                         batch_count: int = 1) -> Dict:
    """Run multi-NIC sender: one process per GPU/NIC, aggregate bandwidth."""
    import multiprocessing as mp

    manager = mp.Manager()
    all_results = manager.dict()

    def _worker(gpu_id, ib_dev, results_slot):
        r = run_sender(gpu_id, remote_host, sizes, ib_device=ib_dev,
                       port_offset=gpu_id * 10, use_batch=use_batch,
                       batch_count=batch_count)
        results_slot[gpu_id] = r

    procs = []
    for i in range(num_nics):
        p = mp.Process(target=_worker, args=(i, ib_devices[i], all_results))
        procs.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=600)

    # Aggregate results
    aggregated = {}
    for size in sizes:
        total_bw = 0
        p50_max = 0
        for gpu_id in range(num_nics):
            if gpu_id in all_results and size in all_results[gpu_id]:
                r = all_results[gpu_id][size]
                total_bw += r["bw_GBs"]
                p50_max = max(p50_max, r["p50_us"])

        if total_bw > 0:
            aggregated[size] = {
                "size_bytes": size,
                "num_nics": num_nics,
                "aggregate_bw_GBs": round(total_bw, 3),
                "max_p50_us": round(p50_max, 2),
            }

    return {"per_nic": dict(all_results), "aggregate": aggregated}


def run_multi_nic_receiver(num_nics: int, sizes: List[int],
                           ib_devices: List[str], batch_count: int = 1):
    """Run multi-NIC receiver: one process per GPU/NIC."""
    import multiprocessing as mp

    procs = []
    for i in range(num_nics):
        p = mp.Process(target=run_receiver,
                       args=(i, sizes),
                       kwargs={"ib_device": ib_devices[i], "port_offset": i * 10,
                               "batch_count": batch_count})
        procs.append(p)

    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=600)


def print_summary(results: Dict, label: str):
    """Print formatted summary table."""
    print(f"\n{'=' * 90}")
    print(f"  {label}")
    print(f"{'=' * 90}")
    print(f"  {'Size':>12} | {'p50(us)':>10} | {'p95(us)':>10} | {'p99(us)':>10} | {'BW(GB/s)':>10}")
    print(f"  {'-' * 70}")

    for size in sorted(results.keys()):
        r = results[size]
        if "bw_GBs" in r:
            print(f"  {size:>12} | {r['p50_us']:>10.1f} | {r['p95_us']:>10.1f} | "
                  f"{r['p99_us']:>10.1f} | {r['bw_GBs']:>10.2f}")
        elif "aggregate_bw_GBs" in r:
            print(f"  {size:>12} | {r['max_p50_us']:>10.1f} | {'—':>10} | "
                  f"{'—':>10} | {r['aggregate_bw_GBs']:>10.2f}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Mooncake RDMA Write benchmark (matches sglang PD KV transfer)")
    parser.add_argument("role", choices=["sender", "receiver", "nvlink", "all"],
                        help="sender/receiver for cross-node, nvlink for same-node, "
                             "all for full suite (requires both nodes)")
    parser.add_argument("--remote-host", default="10.252.129.35",
                        help="Remote host IP (for sender)")
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID for single-NIC mode")
    parser.add_argument("--src-gpu", type=int, default=0, help="Source GPU for NVLink test")
    parser.add_argument("--dst-gpu", type=int, default=4, help="Dest GPU for NVLink test")
    parser.add_argument("--ib-device", default=None,
                        help="IB device name (e.g. mlx5_bond_0). Auto-detect if not set.")
    parser.add_argument("--multi-nic", type=int, default=0,
                        help="Number of NICs for multi-NIC test (0=single NIC)")
    parser.add_argument("--ib-devices", default="mlx5_0,mlx5_1,mlx5_4,mlx5_5",
                        help="Comma-separated IB device list for multi-NIC")
    parser.add_argument("--sizes", default="default",
                        choices=["default", "afd", "small", "large", "full"],
                        help="Message size set")
    parser.add_argument("--use-batch", action="store_true",
                        help="Use batch_transfer_sync_write (multi-layer simulation)")
    parser.add_argument("--batch-count", type=int, default=1,
                        help="Number of transfers per batch (simulates #layers)")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--outfile", default=None, help="Output JSON file")
    args = parser.parse_args()

    # Override module-level iteration counts
    _mod = sys.modules[__name__]
    _mod.BENCH_ITERS = args.iters
    _mod.WARMUP_ITERS = args.warmup

    if args.sizes == "default":
        sizes = DEFAULT_SIZES
    elif args.sizes == "afd":
        sizes = AFD_SIZES
    elif args.sizes == "small":
        sizes = [s for s in DEFAULT_SIZES if s <= 262144]
    elif args.sizes == "large":
        sizes = [s for s in DEFAULT_SIZES if s >= 262144]
    elif args.sizes == "full":
        sizes = sorted(set(DEFAULT_SIZES + AFD_SIZES))

    ib_devices_list = args.ib_devices.split(",")

    if args.role == "nvlink":
        print(f"=== NVLink P2P Benchmark: GPU{args.src_gpu} -> GPU{args.dst_gpu} ===")
        results = run_nvlink_benchmark(args.src_gpu, args.dst_gpu, sizes)
        print_summary(results, f"NVLink GPU{args.src_gpu}->GPU{args.dst_gpu}")
        if args.outfile:
            with open(args.outfile, "w") as f:
                json.dump({"nvlink": {str(k): v for k, v in results.items()}}, f, indent=2)
            print(f"Results saved to {args.outfile}")

    elif args.role == "receiver":
        if args.multi_nic > 0:
            print(f"=== Multi-NIC Receiver ({args.multi_nic} NICs) ===")
            run_multi_nic_receiver(args.multi_nic, sizes, ib_devices_list[:args.multi_nic],
                                   batch_count=args.batch_count)
        else:
            print(f"=== Single-NIC Receiver (GPU{args.gpu}) ===")
            run_receiver(args.gpu, sizes, ib_device=args.ib_device,
                         batch_count=args.batch_count)

    elif args.role == "sender":
        if args.multi_nic > 0:
            print(f"=== Multi-NIC Sender ({args.multi_nic} NICs) -> {args.remote_host} ===")
            results = run_multi_nic_sender(
                args.multi_nic, args.remote_host, sizes,
                ib_devices_list[:args.multi_nic],
                use_batch=args.use_batch, batch_count=args.batch_count)
            print_summary(results["aggregate"],
                          f"Aggregate RDMA Write ({args.multi_nic} NICs)")
            if args.outfile:
                serializable = {
                    "aggregate": {str(k): v for k, v in results["aggregate"].items()},
                }
                with open(args.outfile, "w") as f:
                    json.dump(serializable, f, indent=2)
                print(f"Results saved to {args.outfile}")
        else:
            print(f"=== Single-NIC Sender (GPU{args.gpu}) -> {args.remote_host} ===")
            results = run_sender(args.gpu, args.remote_host, sizes,
                                 ib_device=args.ib_device,
                                 use_batch=args.use_batch,
                                 batch_count=args.batch_count)
            print_summary(results, "Mooncake RDMA Write (single NIC)")
            if args.outfile:
                with open(args.outfile, "w") as f:
                    json.dump({str(k): v for k, v in results.items()}, f, indent=2)
                print(f"Results saved to {args.outfile}")

    elif args.role == "all":
        print("Use 'sender'/'receiver'/'nvlink' roles separately on respective nodes.")
        print("See run_comm_bench.sh for orchestrated execution.")


if __name__ == "__main__":
    main()
