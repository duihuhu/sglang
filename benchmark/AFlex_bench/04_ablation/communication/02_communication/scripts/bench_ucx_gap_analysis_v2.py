#!/usr/bin/env python3
"""Measure run_coroutine_threadsafe cross-thread dispatch latency.

This is the suspected source of the ~193us unexplained gap.
In inference, the daemon thread calls:
    asyncio.run_coroutine_threadsafe(_async_send(...), bridge._loop)
and the coroutine executes on the bridge thread's event loop.

We measure:
1. Pure cross-thread dispatch (no UCX)
2. Cross-thread dispatch + UCX send
3. With GIL contention (simulated GPU ops on main thread)
4. With actual CUDA operations running concurrently
"""
import argparse
import asyncio
import multiprocessing
import os
import time
import threading
from typing import Optional
from concurrent.futures import Future

import numpy as np
import torch


UCX_TLS = os.environ.get("AFD_UCX_TLS", "rc,tcp,cuda_copy,cuda_ipc")
BASE_PORT = 19878
WARMUP = 30
ITERS = 200


def _setup_ucx():
    os.environ.setdefault("UCX_TLS", UCX_TLS)
    os.environ.setdefault("UCX_LOG_LEVEL", "error")
    import ucp
    return ucp


class AsyncBridge:
    def __init__(self):
        self._ready = threading.Event()
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread = threading.Thread(target=self._run, daemon=True, name="ucx-bridge")
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


def run_server(host, port, device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.cuda.set_device(0)
    ucp = _setup_ucx()
    bridge = AsyncBridge()

    async def _main():
        connected = asyncio.get_event_loop().create_future()
        async def _on_connect(ep):
            connected.set_result(ep)
        listener = ucp.create_listener(_on_connect, port)
        print(f"[Server] Listening on port {port}")
        ep = await connected
        print(f"[Server] Connected")

        buf = torch.empty(5120, dtype=torch.bfloat16, device="cuda:0")
        ack = np.array([1], dtype=np.uint8)

        # Receive until done signal
        ctrl = np.empty(1, dtype=np.int64)
        while True:
            await ep.recv(ctrl)
            if ctrl[0] < 0:
                break
            for _ in range(int(ctrl[0])):
                await ep.recv(buf)
                await ep.send(ack)

        await ep.close()
        listener.close()

    bridge.run(_main())
    bridge.stop()


def run_client(host, port, device):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.cuda.set_device(0)
    ucp = _setup_ucx()
    bridge = AsyncBridge()

    x = torch.randn(5120, dtype=torch.bfloat16, device="cuda:0")
    torch.cuda.synchronize()

    # Connect
    async def _connect():
        ep = await ucp.create_endpoint(host, port)
        return ep
    ep = bridge.run(_connect())
    print(f"[Client] Connected to {host}:{port}")

    ack_buf = np.empty(1, dtype=np.uint8)
    send_lock = asyncio.Lock()

    def p50(data):
        s = sorted(data)
        return s[len(s)//2]
    def p95(data):
        s = sorted(data)
        return s[int(len(s)*0.95)]
    def mean(data):
        return sum(data)/len(data)

    # ================================================================
    # TEST 1: Pure cross-thread dispatch latency (no UCX, no GPU)
    # ================================================================
    print("\n[TEST 1] Pure run_coroutine_threadsafe dispatch latency...")
    dispatch_times = []
    for _ in range(WARMUP + ITERS):
        t_submit = time.perf_counter()
        async def _noop():
            t_exec = time.perf_counter()
            return t_exec
        fut = asyncio.run_coroutine_threadsafe(_noop(), bridge._loop)
        t_exec = fut.result()
        dispatch_times.append((t_exec - t_submit) * 1e6)
    dispatch_times = dispatch_times[WARMUP:]

    # ================================================================
    # TEST 2: Cross-thread dispatch + UCX send (the real path)
    # ================================================================
    print("[TEST 2] run_coroutine_threadsafe + UCX send...")
    ctrl = np.array([WARMUP + ITERS], dtype=np.int64)
    bridge.run(ep.send(ctrl))

    cross_thread_send_times = []
    dispatch_to_send = []
    for _ in range(WARMUP + ITERS):
        torch.cuda.synchronize()
        t_submit = time.perf_counter()

        async def _send_with_timing():
            t_start = time.perf_counter()
            async with send_lock:
                await ep.send(x)
            t_end = time.perf_counter()
            return t_start, t_end

        fut = asyncio.run_coroutine_threadsafe(_send_with_timing(), bridge._loop)
        t_start, t_end = fut.result()

        # recv ack
        bridge.run(ep.recv(ack_buf))

        dispatch_to_send.append((t_start - t_submit) * 1e6)
        cross_thread_send_times.append((t_end - t_submit) * 1e6)

    dispatch_to_send = dispatch_to_send[WARMUP:]
    cross_thread_send_times = cross_thread_send_times[WARMUP:]

    # ================================================================
    # TEST 3: Cross-thread dispatch with GIL contention
    # (main thread holds GIL doing CPU work)
    # ================================================================
    print("[TEST 3] Cross-thread dispatch + GIL contention...")
    ctrl = np.array([WARMUP + ITERS], dtype=np.int64)
    bridge.run(ep.send(ctrl))

    gil_contention_times = []
    gil_dispatch_delays = []

    for _ in range(WARMUP + ITERS):
        torch.cuda.synchronize()
        t_submit = time.perf_counter()

        async def _send_gil():
            t_start = time.perf_counter()
            await ep.send(x)
            t_end = time.perf_counter()
            return t_start, t_end

        fut = asyncio.run_coroutine_threadsafe(_send_gil(), bridge._loop)

        # Simulate GIL-holding work on this thread (like the daemon thread does
        # compute_event.synchronize() which holds GIL)
        # Do some CPU-bound work for ~50us
        _dummy = 0
        for _i in range(500):
            _dummy += _i * _i

        t_start, t_end = fut.result()
        bridge.run(ep.recv(ack_buf))

        gil_dispatch_delays.append((t_start - t_submit) * 1e6)
        gil_contention_times.append((t_end - t_submit) * 1e6)

    gil_dispatch_delays = gil_dispatch_delays[WARMUP:]
    gil_contention_times = gil_contention_times[WARMUP:]

    # ================================================================
    # TEST 4: Simulating the REAL daemon thread path
    # daemon thread: event.synchronize() -> run_coroutine_threadsafe(send)
    # ================================================================
    print("[TEST 4] Full daemon thread simulation (event.sync + cross-thread send)...")
    ctrl = np.array([WARMUP + ITERS], dtype=np.int64)
    bridge.run(ep.send(ctrl))

    daemon_times = []
    daemon_breakdown = []

    compute_stream = torch.cuda.current_stream()

    for _ in range(WARMUP + ITERS):
        # Simulate: main thread launches kernel, records event
        # (In reality this is a tiny kernel, we use a no-op)
        torch.cuda.synchronize()
        compute_event = compute_stream.record_event()

        # Now simulate daemon thread behavior
        t0 = time.perf_counter()
        compute_event.synchronize()  # wait for GPU (should be instant since we synced)
        t1 = time.perf_counter()

        async def _daemon_send():
            t_exec = time.perf_counter()
            await ep.send(x)
            t_done = time.perf_counter()
            return t_exec, t_done

        fut = asyncio.run_coroutine_threadsafe(_daemon_send(), bridge._loop)
        t_exec, t_done = fut.result()
        t2 = time.perf_counter()

        bridge.run(ep.recv(ack_buf))

        daemon_breakdown.append({
            'event_sync': (t1 - t0) * 1e6,
            'dispatch_delay': (t_exec - t1) * 1e6,
            'ucx_send': (t_done - t_exec) * 1e6,
            'fut_result_wait': (t2 - t_done) * 1e6,
            'total': (t2 - t0) * 1e6,
        })

    daemon_breakdown = daemon_breakdown[WARMUP:]

    # ================================================================
    # TEST 5: Same as TEST 4 but with a REAL GPU kernel running
    # ================================================================
    print("[TEST 5] Daemon thread with actual GPU kernel in flight...")
    ctrl = np.array([WARMUP + ITERS], dtype=np.int64)
    bridge.run(ep.send(ctrl))

    # Create a matrix to do matmul (simulates attention/FFN compute)
    A = torch.randn(512, 5120, dtype=torch.bfloat16, device="cuda:0")
    B = torch.randn(5120, 5120, dtype=torch.bfloat16, device="cuda:0")

    daemon_gpu_breakdown = []

    for _ in range(WARMUP + ITERS):
        # Launch a real GPU kernel (matmul)
        C = torch.matmul(A, B)
        compute_event = compute_stream.record_event()

        t0 = time.perf_counter()
        compute_event.synchronize()
        t1 = time.perf_counter()

        async def _daemon_send_gpu():
            t_exec = time.perf_counter()
            await ep.send(x)
            t_done = time.perf_counter()
            return t_exec, t_done

        fut = asyncio.run_coroutine_threadsafe(_daemon_send_gpu(), bridge._loop)
        t_exec, t_done = fut.result()
        t2 = time.perf_counter()

        bridge.run(ep.recv(ack_buf))

        daemon_gpu_breakdown.append({
            'event_sync': (t1 - t0) * 1e6,
            'dispatch_delay': (t_exec - t1) * 1e6,
            'ucx_send': (t_done - t_exec) * 1e6,
            'total': (t2 - t0) * 1e6,
        })

    daemon_gpu_breakdown = daemon_gpu_breakdown[WARMUP:]

    # Signal done
    ctrl_stop = np.array([-1], dtype=np.int64)
    bridge.run(ep.send(ctrl_stop))
    bridge.run(ep.close())
    bridge.stop()

    # ================================================================
    # RESULTS
    # ================================================================
    print("\n" + "=" * 90)
    print("  CROSS-THREAD DISPATCH ANALYSIS: Where is the ~193us?")
    print("=" * 90)

    print(f"\n  TEST 1: Pure dispatch (run_coroutine_threadsafe -> coroutine starts)")
    print(f"    p50={p50(dispatch_times):.1f}us  p95={p95(dispatch_times):.1f}us  mean={mean(dispatch_times):.1f}us")

    print(f"\n  TEST 2: Cross-thread dispatch + UCX send")
    print(f"    dispatch delay:  p50={p50(dispatch_to_send):.1f}us  p95={p95(dispatch_to_send):.1f}us")
    print(f"    total (submit->send_done): p50={p50(cross_thread_send_times):.1f}us  p95={p95(cross_thread_send_times):.1f}us")

    print(f"\n  TEST 3: + GIL contention (CPU work on submitting thread)")
    print(f"    dispatch delay:  p50={p50(gil_dispatch_delays):.1f}us  p95={p95(gil_dispatch_delays):.1f}us")
    print(f"    total:           p50={p50(gil_contention_times):.1f}us  p95={p95(gil_contention_times):.1f}us")

    es = [d['event_sync'] for d in daemon_breakdown]
    dd = [d['dispatch_delay'] for d in daemon_breakdown]
    us = [d['ucx_send'] for d in daemon_breakdown]
    fw = [d['fut_result_wait'] for d in daemon_breakdown]
    tt = [d['total'] for d in daemon_breakdown]
    print(f"\n  TEST 4: Full daemon simulation (event.sync + dispatch + send)")
    print(f"    event.synchronize(): p50={p50(es):.1f}us")
    print(f"    dispatch delay:      p50={p50(dd):.1f}us  p95={p95(dd):.1f}us")
    print(f"    UCX send:            p50={p50(us):.1f}us")
    print(f"    fut.result() wait:   p50={p50(fw):.1f}us")
    print(f"    TOTAL:               p50={p50(tt):.1f}us")

    es2 = [d['event_sync'] for d in daemon_gpu_breakdown]
    dd2 = [d['dispatch_delay'] for d in daemon_gpu_breakdown]
    us2 = [d['ucx_send'] for d in daemon_gpu_breakdown]
    tt2 = [d['total'] for d in daemon_gpu_breakdown]
    print(f"\n  TEST 5: Daemon with REAL GPU kernel (512x5120 matmul)")
    print(f"    event.synchronize(): p50={p50(es2):.1f}us  p95={p95(es2):.1f}us")
    print(f"    dispatch delay:      p50={p50(dd2):.1f}us  p95={p95(dd2):.1f}us")
    print(f"    UCX send:            p50={p50(us2):.1f}us  p95={p95(us2):.1f}us")
    print(f"    TOTAL:               p50={p50(tt2):.1f}us  p95={p95(tt2):.1f}us")

    print(f"\n  {'='*90}")
    print(f"  CONCLUSION:")
    print(f"  {'='*90}")
    print(f"    Pure UCX send (in-loop):          {p50([108.1]):.1f} us")
    print(f"    + cross-thread dispatch:         +{p50(dd):.1f} us")
    print(f"    + event.sync (no real kernel):   +{p50(es):.1f} us")
    print(f"    = Daemon path (no GPU load):      {p50(tt):.1f} us")
    print(f"    + event.sync (real kernel):      +{p50(es2):.1f} us")
    print(f"    = Daemon path (with GPU load):    {p50(tt2):.1f} us")
    print(f"    In-inference measured:             ~307 us")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BASE_PORT)
    parser.add_argument("--gpu-server", type=int, default=0)
    parser.add_argument("--gpu-client", type=int, default=1)
    args = parser.parse_args()

    server_proc = multiprocessing.Process(
        target=run_server, args=(args.host, args.port, args.gpu_server)
    )
    client_proc = multiprocessing.Process(
        target=run_client, args=(args.host, args.port, args.gpu_client)
    )

    server_proc.start()
    time.sleep(3)
    client_proc.start()
    client_proc.join(timeout=120)
    server_proc.join(timeout=10)


if __name__ == "__main__":
    main()
