#!/usr/bin/env python3
"""Standalone UCX RDMA tensor transfer latency benchmark.

Measures the pure UCX send/recv latency for GPU tensors of various sizes,
isolating the `await endpoint.send(x)` / `await endpoint.recv(buf)` cost
without any model inference overhead.

Usage:
    # Terminal 1 (receiver/server):
    CUDA_VISIBLE_DEVICES=0 python3 bench_ucx_transfer.py --role server

    # Terminal 2 (sender/client):
    CUDA_VISIBLE_DEVICES=1 python3 bench_ucx_transfer.py --role client --host 127.0.0.1

Or run both in one shot (recommended):
    python3 bench_ucx_transfer.py --both
"""
import argparse
import asyncio
import multiprocessing
import os
import time
import threading
from typing import Optional

import numpy as np
import torch


# UCX transport config
UCX_TLS = os.environ.get("AFD_UCX_TLS", "rc,tcp,cuda_copy,cuda_ipc")
BASE_PORT = 19876
WARMUP_ITERS = 20
BENCH_ITERS = 100


def _setup_ucx():
    """Import and configure UCX."""
    os.environ.setdefault("UCX_TLS", UCX_TLS)
    os.environ.setdefault("UCX_LOG_LEVEL", "error")
    import ucp
    return ucp


class AsyncBridge:
    """Persistent event loop in a background thread."""
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


def run_server(host: str, port: int, device: int, sizes: list, results_dict: dict):
    """Server (receiver) process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.cuda.set_device(0)

    ucp = _setup_ucx()
    bridge = AsyncBridge()

    async def _server_main():
        connected = asyncio.get_event_loop().create_future()

        async def _on_connect(ep):
            connected.set_result(ep)

        listener = ucp.create_listener(_on_connect, port)
        print(f"[Server] Listening on {host}:{port}, GPU={device}")

        ep = await connected
        print(f"[Server] Client connected")

        for size_bytes in sizes:
            num_elements = size_bytes // 2  # bf16
            buf = torch.empty(num_elements, dtype=torch.bfloat16, device="cuda:0")

            # Warmup
            for _ in range(WARMUP_ITERS):
                await ep.recv(buf)
                # Send ack (1 byte)
                ack = np.array([1], dtype=np.uint8)
                await ep.send(ack)

            # Benchmark
            recv_times = []
            for _ in range(BENCH_ITERS):
                t0 = time.perf_counter()
                await ep.recv(buf)
                t1 = time.perf_counter()
                # Send ack
                ack = np.array([1], dtype=np.uint8)
                await ep.send(ack)
                recv_times.append((t1 - t0) * 1e6)  # us

            results_dict[size_bytes] = {
                "recv_us": recv_times,
            }
            print(f"[Server] size={size_bytes:>8} B: recv p50={sorted(recv_times)[len(recv_times)//2]:.1f} us")

        await ep.close()
        listener.close()

    bridge.run(_server_main())
    bridge.stop()


def run_client(host: str, port: int, device: int, sizes: list, results_dict: dict):
    """Client (sender) process."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.cuda.set_device(0)

    ucp = _setup_ucx()
    bridge = AsyncBridge()

    async def _client_main():
        print(f"[Client] Connecting to {host}:{port}, GPU={device}")
        ep = await ucp.create_endpoint(host, port)
        print(f"[Client] Connected")

        for size_bytes in sizes:
            num_elements = size_bytes // 2  # bf16
            x = torch.randn(num_elements, dtype=torch.bfloat16, device="cuda:0")
            torch.cuda.synchronize()

            # Warmup
            for _ in range(WARMUP_ITERS):
                await ep.send(x)
                ack = np.empty(1, dtype=np.uint8)
                await ep.recv(ack)

            # Benchmark
            send_times = []
            rtt_times = []
            for _ in range(BENCH_ITERS):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                await ep.send(x)
                t1 = time.perf_counter()
                ack = np.empty(1, dtype=np.uint8)
                await ep.recv(ack)
                t2 = time.perf_counter()
                send_times.append((t1 - t0) * 1e6)
                rtt_times.append((t2 - t0) * 1e6)

            results_dict[size_bytes] = {
                "send_us": send_times,
                "rtt_us": rtt_times,
            }
            p50_send = sorted(send_times)[len(send_times)//2]
            p50_rtt = sorted(rtt_times)[len(rtt_times)//2]
            print(f"[Client] size={size_bytes:>8} B: send p50={p50_send:.1f} us, RTT p50={p50_rtt:.1f} us")

        await ep.close()

    bridge.run(_client_main())
    bridge.stop()


def run_both(host: str, port: int, gpu_server: int, gpu_client: int):
    """Run server and client in separate processes."""
    # Tensor sizes to test (matching real workload)
    # Qwen3-32B: hidden_size=5120, bf16 -> 10KB per token
    # Typical batch sizes: 1, 4, 8, 16, 32, 64
    sizes = [
        10240,       # 10KB  = 1 token  (5120 * 2)
        40960,       # 40KB  = 4 tokens
        81920,       # 80KB  = 8 tokens
        163840,      # 160KB = 16 tokens
        327680,      # 320KB = 32 tokens
        655360,      # 640KB = 64 tokens
        1310720,     # 1.3MB = 128 tokens
    ]

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
    time.sleep(3)  # Let server start listening
    client_proc.start()

    client_proc.join(timeout=120)
    server_proc.join(timeout=10)

    # Print summary
    print("\n" + "=" * 100)
    print("  UCX RDMA GPU Tensor Transfer Latency Benchmark")
    print("  (Pure endpoint.send/recv, no model overhead)")
    print("=" * 100)
    print(f"\n  Server GPU: {gpu_server}, Client GPU: {gpu_client}")
    print(f"  UCX_TLS: {UCX_TLS}")
    print(f"  Warmup: {WARMUP_ITERS} iters, Bench: {BENCH_ITERS} iters")
    print()

    header = f"  {'Size':>10} | {'Tokens':>6} | {'send p50':>10} | {'send p95':>10} | {'recv p50':>10} | {'recv p95':>10} | {'RTT p50':>10} | {'RTT p95':>10}"
    print(header)
    print("  " + "-" * 95)

    for size in sizes:
        tokens = size // 10240
        c = dict(client_results.get(size, {}))
        s = dict(server_results.get(size, {}))

        if 'send_us' in c and 'recv_us' in s:
            send = sorted(c['send_us'])
            recv = sorted(s['recv_us'])
            rtt = sorted(c['rtt_us'])
            n = len(send)

            print(f"  {size:>10} | {tokens:>6} | "
                  f"{send[n//2]:>8.1f}us | {send[int(n*0.95)]:>8.1f}us | "
                  f"{recv[n//2]:>8.1f}us | {recv[int(n*0.95)]:>8.1f}us | "
                  f"{rtt[n//2]:>8.1f}us | {rtt[int(n*0.95)]:>8.1f}us")
        else:
            print(f"  {size:>10} | {tokens:>6} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10} | {'N/A':>10}")

    print()
    print("  Column definitions:")
    print("    send p50/p95 = await endpoint.send(tensor) on sender GPU")
    print("    recv p50/p95 = await endpoint.recv(buf) on receiver GPU")
    print("    RTT p50/p95  = send + wait_for_ack (full round-trip)")
    print()
    print("  This measures the PURE UCX transfer cost:")
    print("    - NIC DMA read from sender GPU (GPUDirect RDMA)")
    print("    - Network wire transfer (InfiniBand/RoCE)")
    print("    - NIC DMA write to receiver GPU")
    print("    - UCX protocol overhead (tag matching, completion)")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UCX RDMA transfer latency benchmark")
    parser.add_argument("--role", choices=["server", "client", "both"], default="both")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BASE_PORT)
    parser.add_argument("--gpu-server", type=int, default=0)
    parser.add_argument("--gpu-client", type=int, default=1)
    args = parser.parse_args()

    if args.role == "both":
        run_both(args.host, args.port, args.gpu_server, args.gpu_client)
    elif args.role == "server":
        sizes = [10240, 40960, 81920, 163840, 327680, 655360, 1310720]
        results = {}
        run_server(args.host, args.port, int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")), sizes, results)
    else:
        sizes = [10240, 40960, 81920, 163840, 327680, 655360, 1310720]
        results = {}
        run_client(args.host, args.port, int(os.environ.get("CUDA_VISIBLE_DEVICES", "1")), sizes, results)
