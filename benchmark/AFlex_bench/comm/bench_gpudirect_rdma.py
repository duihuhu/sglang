#!/usr/bin/env python3
"""GPUDirect RDMA benchmark via UCX — matches sglang AFD communication path.

Tests GPU tensor send/recv over InfiniBand using UCX rendezvous (put_zcopy),
measuring GPUDirect RDMA throughput and latency.

Usage (cross-node):
    # .35 (server/receiver):
    ulimit -l unlimited
    CUDA_VISIBLE_DEVICES=0 python3 bench_gpudirect_rdma.py server

    # .36 (client/sender):
    ulimit -l unlimited
    CUDA_VISIBLE_DEVICES=0 python3 bench_gpudirect_rdma.py client --host 10.252.129.35

Same-node (two GPUs):
    python3 bench_gpudirect_rdma.py both --gpu-server 0 --gpu-client 1
"""
import argparse
import asyncio
import json
import multiprocessing
import os
import sys
import time
import threading
from typing import Dict, List, Optional

import numpy as np
import torch

# ---------- Config ----------
UCX_TLS = os.environ.get("AFD_UCX_TLS", "rc,tcp,cuda_copy,cuda_ipc")
BASE_PORT = int(os.environ.get("BENCH_PORT", "19876"))
WARMUP_ITERS = 30
BENCH_ITERS = 200

# Message sizes: 2B → 8MB (matches perftest -a range + AFD workload sizes)
DEFAULT_SIZES = [
    2, 4, 8, 16, 32, 64, 128, 256, 512,
    1024, 2048, 4096, 8192, 16384, 32768,
    65536, 131072, 262144, 524288,
    1048576, 2097152, 4194304, 8388608,
]

# AFD-specific sizes (Qwen3-32B hidden=5120, bf16=10KB/token)
AFD_SIZES = [
    10240,       # 1 token
    40960,       # 4 tokens
    81920,       # 8 tokens
    163840,      # 16 tokens
    327680,      # 32 tokens
    655360,      # 64 tokens
    1310720,     # 128 tokens
    2621440,     # 256 tokens
    5242880,     # 512 tokens
]


def _setup_ucx():
    os.environ.setdefault("UCX_TLS", UCX_TLS)
    os.environ.setdefault("UCX_LOG_LEVEL", "error")
    os.environ.setdefault("UCX_RNDV_THRESH", "8192")
    os.environ.setdefault("UCX_RNDV_SCHEME", "put_zcopy")
    os.environ.setdefault("UCX_ZCOPY_THRESH", "8192")
    os.environ.setdefault("UCX_MEMTYPE_CACHE", "n")
    import ucp
    return ucp


class AsyncBridge:
    def __init__(self):
        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=10)

    def _run(self):
        self._loop = asyncio.SelectorEventLoop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def run(self, coro):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result()

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


def _make_gpu_buf(size_bytes: int, device: str = "cuda:0") -> torch.Tensor:
    num_elements = max(1, size_bytes // 2)
    return torch.empty(num_elements, dtype=torch.bfloat16, device=device)


def run_server(host: str, port: int, gpu: int, sizes: List[int],
               results_dict: Dict, outfile: Optional[str] = None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.cuda.set_device(0)
    ucp = _setup_ucx()
    bridge = AsyncBridge()

    async def _main():
        connected = asyncio.get_event_loop().create_future()

        async def _on_connect(ep):
            connected.set_result(ep)

        listener = ucp.create_listener(_on_connect, port)
        print(f"[Server] Listening on port {port}, GPU={gpu}, UCX_TLS={UCX_TLS}")
        print(f"[Server] Waiting for client...")

        ep = await connected
        print(f"[Server] Client connected")

        for size_bytes in sizes:
            buf = _make_gpu_buf(size_bytes)

            for _ in range(WARMUP_ITERS):
                await ep.recv(buf)
                await ep.send(np.array([1], dtype=np.uint8))

            recv_times = []
            for _ in range(BENCH_ITERS):
                t0 = time.perf_counter()
                await ep.recv(buf)
                t1 = time.perf_counter()
                await ep.send(np.array([1], dtype=np.uint8))
                recv_times.append((t1 - t0) * 1e6)

            results_dict[size_bytes] = {"recv_us": recv_times}
            p50 = sorted(recv_times)[len(recv_times) // 2]
            bw = size_bytes / (p50 * 1e-6) / 1e9
            print(f"[Server] {size_bytes:>8} B: recv p50={p50:>8.1f} us  BW={bw:>6.2f} GB/s")

        await ep.close()
        listener.close()

    bridge.run(_main())
    bridge.stop()


def run_client(host: str, port: int, gpu: int, sizes: List[int],
               results_dict: Dict, outfile: Optional[str] = None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    torch.cuda.set_device(0)
    ucp = _setup_ucx()
    bridge = AsyncBridge()

    async def _main():
        print(f"[Client] Connecting to {host}:{port}, GPU={gpu}")
        ep = await ucp.create_endpoint(host, port)
        print(f"[Client] Connected")

        for size_bytes in sizes:
            x = torch.randn(max(1, size_bytes // 2), dtype=torch.bfloat16, device="cuda:0")
            torch.cuda.synchronize()

            for _ in range(WARMUP_ITERS):
                await ep.send(x)
                await ep.recv(np.empty(1, dtype=np.uint8))

            send_times = []
            rtt_times = []
            for _ in range(BENCH_ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                await ep.send(x)
                t1 = time.perf_counter()
                await ep.recv(np.empty(1, dtype=np.uint8))
                t2 = time.perf_counter()
                send_times.append((t1 - t0) * 1e6)
                rtt_times.append((t2 - t0) * 1e6)

            results_dict[size_bytes] = {"send_us": send_times, "rtt_us": rtt_times}
            p50_send = sorted(send_times)[len(send_times) // 2]
            p50_rtt = sorted(rtt_times)[len(rtt_times) // 2]
            bw = size_bytes / (p50_send * 1e-6) / 1e9
            print(f"[Client] {size_bytes:>8} B: send p50={p50_send:>8.1f} us  "
                  f"RTT p50={p50_rtt:>8.1f} us  BW={bw:>6.2f} GB/s")

        await ep.close()

    bridge.run(_main())
    bridge.stop()


def print_summary(sizes, client_results, server_results, outfile=None):
    lines = []
    lines.append("")
    lines.append("=" * 110)
    lines.append("  GPUDirect RDMA Benchmark (UCX send/recv, GPU tensor)")
    lines.append("=" * 110)
    lines.append(f"  UCX_TLS: {UCX_TLS}")
    lines.append(f"  Warmup: {WARMUP_ITERS}, Bench: {BENCH_ITERS} iters")
    lines.append("")
    lines.append(f"  {'Size':>10} | {'send p50':>10} | {'send p95':>10} | "
                 f"{'recv p50':>10} | {'recv p95':>10} | "
                 f"{'RTT p50':>10} | {'BW(send)':>10} | {'BW(recv)':>10}")
    lines.append("  " + "-" * 100)

    json_data = []
    for size in sizes:
        c = dict(client_results.get(size, {}))
        s = dict(server_results.get(size, {}))

        if 'send_us' in c and 'recv_us' in s:
            send = sorted(c['send_us'])
            recv = sorted(s['recv_us'])
            rtt = sorted(c['rtt_us'])
            n = len(send)
            p50_send = send[n // 2]
            p95_send = send[int(n * 0.95)]
            p50_recv = recv[n // 2]
            p95_recv = recv[int(n * 0.95)]
            p50_rtt = rtt[n // 2]
            bw_send = size / (p50_send * 1e-6) / 1e9
            bw_recv = size / (p50_recv * 1e-6) / 1e9

            lines.append(
                f"  {size:>10} | {p50_send:>8.1f}us | {p95_send:>8.1f}us | "
                f"{p50_recv:>8.1f}us | {p95_recv:>8.1f}us | "
                f"{p50_rtt:>8.1f}us | {bw_send:>7.2f}GB/s | {bw_recv:>7.2f}GB/s"
            )
            json_data.append({
                "size_bytes": size,
                "send_p50_us": round(p50_send, 2),
                "send_p95_us": round(p95_send, 2),
                "recv_p50_us": round(p50_recv, 2),
                "recv_p95_us": round(p95_recv, 2),
                "rtt_p50_us": round(p50_rtt, 2),
                "bw_send_GBs": round(bw_send, 3),
                "bw_recv_GBs": round(bw_recv, 3),
            })
        else:
            lines.append(f"  {size:>10} | {'N/A':>10} | {'N/A':>10} | "
                         f"{'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")

    lines.append("")
    lines.append("  Path: GPU mem → NIC DMA (GPUDirect) → IB wire → NIC DMA → GPU mem")
    lines.append("")

    text = "\n".join(lines)
    print(text)

    if outfile:
        with open(outfile, "w") as f:
            f.write(text + "\n")
        json_path = outfile.replace(".log", ".json")
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)
        print(f"  Results saved to: {outfile}, {json_path}")


def run_both(host, port, gpu_server, gpu_client, sizes, outfile):
    manager = multiprocessing.Manager()
    server_results = manager.dict()
    client_results = manager.dict()

    server_proc = multiprocessing.Process(
        target=run_server, args=(host, port, gpu_server, sizes, server_results)
    )
    client_proc = multiprocessing.Process(
        target=run_client, args=(host, port, gpu_client, sizes, client_results)
    )

    server_proc.start()
    time.sleep(3)
    client_proc.start()

    client_proc.join(timeout=300)
    server_proc.join(timeout=10)

    print_summary(sizes, client_results, server_results, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="GPUDirect RDMA benchmark (UCX, GPU tensor send/recv)")
    parser.add_argument("role", choices=["server", "client", "both"],
                        help="server=receiver, client=sender, both=same-node 2-GPU")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Server listen addr (server) or remote addr (client)")
    parser.add_argument("--port", type=int, default=BASE_PORT)
    parser.add_argument("--gpu-server", type=int, default=0)
    parser.add_argument("--gpu-client", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU id for single-role mode")
    parser.add_argument("--sizes", default="all",
                        choices=["all", "afd", "small", "large"],
                        help="all=2B-8MB, afd=AFD workload sizes, small=2B-64KB, large=64KB-8MB")
    parser.add_argument("--outfile", default=None,
                        help="Output file for results")
    parser.add_argument("--iters", type=int, default=BENCH_ITERS)
    args = parser.parse_args()

    BENCH_ITERS = args.iters

    if args.sizes == "all":
        sizes = DEFAULT_SIZES
    elif args.sizes == "afd":
        sizes = AFD_SIZES
    elif args.sizes == "small":
        sizes = [s for s in DEFAULT_SIZES if s <= 65536]
    elif args.sizes == "large":
        sizes = [s for s in DEFAULT_SIZES if s >= 65536]

    if args.role == "both":
        run_both(args.host, args.port, args.gpu_server, args.gpu_client, sizes, args.outfile)
    elif args.role == "server":
        results = {}
        run_server(args.host, args.port, args.gpu, sizes, results, args.outfile)
    elif args.role == "client":
        results = {}
        run_client(args.host, args.port, args.gpu, sizes, results, args.outfile)
