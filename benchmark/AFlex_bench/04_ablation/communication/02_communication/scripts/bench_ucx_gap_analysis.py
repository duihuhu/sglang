#!/usr/bin/env python3
"""Diagnose the ~200us gap between pure UCX transfer (~107us) and in-inference send (~307us).

Measures each potential contributor:
1. asyncio event loop scheduling delay (run_coroutine_threadsafe -> actual execution)
2. _send_lock acquisition time
3. x.contiguous() cost
4. Event loop contention (recv coroutine competing on same loop)

Usage:
    python3 bench_ucx_gap_analysis.py --gpu-server 0 --gpu-client 1
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


UCX_TLS = os.environ.get("AFD_UCX_TLS", "rc,tcp,cuda_copy,cuda_ipc")
BASE_PORT = 19877
WARMUP_ITERS = 20
BENCH_ITERS = 200


def _setup_ucx():
    os.environ.setdefault("UCX_TLS", UCX_TLS)
    os.environ.setdefault("UCX_LOG_LEVEL", "error")
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


def run_server(host, port, device, results_dict):
    """Server: receives tensors and sends acks."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.cuda.set_device(0)
    ucp = _setup_ucx()
    bridge = AsyncBridge()

    async def _main():
        connected = asyncio.get_event_loop().create_future()
        async def _on_connect(ep):
            connected.set_result(ep)
        listener = ucp.create_listener(_on_connect, port)
        print(f"[Server] Listening on port {port}, GPU={device}")
        ep = await connected
        print(f"[Server] Connected")

        size_bytes = 10240  # 10KB = 1 token of Qwen3-32B
        num_elements = size_bytes // 2
        buf = torch.empty(num_elements, dtype=torch.bfloat16, device="cuda:0")
        ack = np.array([1], dtype=np.uint8)
        ctrl = np.empty(1, dtype=np.int64)

        # Keep receiving until client signals done
        while True:
            await ep.recv(ctrl)
            if ctrl[0] == -1:
                break
            for _ in range(int(ctrl[0])):
                await ep.recv(buf)
                await ep.send(ack)

        await ep.close()
        listener.close()

    bridge.run(_main())
    bridge.stop()


def run_client(host, port, device, results_dict):
    """Client: runs various experiments to isolate the 200us gap."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)
    torch.cuda.set_device(0)
    ucp = _setup_ucx()
    bridge = AsyncBridge()

    size_bytes = 10240  # 10KB
    num_elements = size_bytes // 2
    x_contig = torch.randn(num_elements, dtype=torch.bfloat16, device="cuda:0")
    # Non-contiguous tensor (transposed view)
    x_base = torch.randn(64, 5120, dtype=torch.bfloat16, device="cuda:0")
    x_noncontig = x_base.t()[:num_elements // 64, :]  # non-contiguous slice
    torch.cuda.synchronize()

    async def _main():
        print(f"[Client] Connecting to {host}:{port}, GPU={device}")
        ep = await ucp.create_endpoint(host, port)
        print(f"[Client] Connected")

        send_lock = asyncio.Lock()
        ack_buf = np.empty(1, dtype=np.uint8)
        ctrl = np.array([WARMUP_ITERS + BENCH_ITERS], dtype=np.int64)

        # ============================================================
        # TEST 1: Baseline - direct await send (no lock, no scheduling)
        # ============================================================
        await ep.send(ctrl)
        for _ in range(WARMUP_ITERS):
            await ep.send(x_contig)
            await ep.recv(ack_buf)

        times_baseline = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            await ep.send(x_contig)
            t1 = time.perf_counter()
            await ep.recv(ack_buf)
            times_baseline.append((t1 - t0) * 1e6)

        # ============================================================
        # TEST 2: With _send_lock (async with lock)
        # ============================================================
        await ep.send(ctrl)
        for _ in range(WARMUP_ITERS):
            async with send_lock:
                await ep.send(x_contig)
            await ep.recv(ack_buf)

        times_with_lock = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            async with send_lock:
                await ep.send(x_contig)
            t1 = time.perf_counter()
            await ep.recv(ack_buf)
            times_with_lock.append((t1 - t0) * 1e6)

        # ============================================================
        # TEST 3: With x.contiguous() call (already contiguous)
        # ============================================================
        await ep.send(ctrl)
        for _ in range(WARMUP_ITERS):
            y = x_contig.contiguous()
            await ep.send(y)
            await ep.recv(ack_buf)

        times_contig_noop = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = x_contig.contiguous()
            await ep.send(y)
            t1 = time.perf_counter()
            await ep.recv(ack_buf)
            times_contig_noop.append((t1 - t0) * 1e6)

        # ============================================================
        # TEST 4: With x.contiguous() on NON-contiguous tensor
        # ============================================================
        await ep.send(ctrl)
        x_nc = x_base[:, :num_elements // 64].t().reshape(-1)[:num_elements]
        # Make sure it's truly non-contiguous
        x_nc_view = x_base.t().contiguous().view(-1)[:num_elements]
        # Actually create a real non-contiguous case
        x_strided = torch.as_strided(x_base, (num_elements,), (2,))

        for _ in range(WARMUP_ITERS):
            y = x_strided.contiguous()
            await ep.send(y)
            await ep.recv(ack_buf)

        times_contig_real = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            y = x_strided.contiguous()
            t1 = time.perf_counter()
            await ep.send(y)
            t2 = time.perf_counter()
            await ep.recv(ack_buf)
            times_contig_real.append(((t1 - t0) * 1e6, (t2 - t1) * 1e6))

        # ============================================================
        # TEST 5: run_coroutine_threadsafe scheduling delay
        # (simulate what happens in the real code path)
        # ============================================================
        await ep.send(ctrl)
        for _ in range(WARMUP_ITERS):
            await ep.send(x_contig)
            await ep.recv(ack_buf)

        # Now measure from external thread perspective
        loop = asyncio.get_event_loop()

        times_scheduling = []
        scheduling_delays = []

        def _measure_from_thread():
            for _ in range(BENCH_ITERS):
                torch.cuda.synchronize()
                t_submit = time.perf_counter()

                async def _timed_send():
                    t_start = time.perf_counter()
                    sched_delay = (t_start - t_submit) * 1e6
                    await ep.send(x_contig)
                    t_end = time.perf_counter()
                    return sched_delay, (t_end - t_start) * 1e6

                fut = asyncio.run_coroutine_threadsafe(_timed_send(), loop)
                sched_delay, send_time = fut.result()
                t_done = time.perf_counter()

                # recv ack synchronously via bridge
                fut2 = asyncio.run_coroutine_threadsafe(ep.recv(ack_buf), loop)
                fut2.result()

                total = (t_done - t_submit) * 1e6
                scheduling_delays.append(sched_delay)
                times_scheduling.append(total)

        # Can't run from external thread while inside async - use a trick
        # Instead, simulate the scheduling delay within the loop
        times_sched_sim = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            # Yield to event loop (simulates scheduling)
            await asyncio.sleep(0)
            t1 = time.perf_counter()
            await ep.send(x_contig)
            t2 = time.perf_counter()
            await ep.recv(ack_buf)
            sched = (t1 - t0) * 1e6
            send = (t2 - t1) * 1e6
            times_sched_sim.append((sched, send))

        # ============================================================
        # TEST 6: Event loop contention (recv running concurrently)
        # ============================================================
        await ep.send(ctrl)

        # Simulate a background recv task competing for the event loop
        contention_active = True
        contention_count = [0]
        dummy_buf = torch.empty(100, dtype=torch.bfloat16, device="cuda:0")

        async def _background_work():
            while contention_active:
                # Simulate recv-like work on the event loop
                await asyncio.sleep(0)
                contention_count[0] += 1

        bg_task = asyncio.ensure_future(_background_work())

        for _ in range(WARMUP_ITERS):
            await ep.send(x_contig)
            await ep.recv(ack_buf)

        times_contention = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            await ep.send(x_contig)
            t1 = time.perf_counter()
            await ep.recv(ack_buf)
            times_contention.append((t1 - t0) * 1e6)

        contention_active = False
        await bg_task

        # ============================================================
        # TEST 7: Full simulation (lock + contiguous + scheduling)
        # ============================================================
        await ep.send(ctrl)
        for _ in range(WARMUP_ITERS):
            async with send_lock:
                y = x_contig.contiguous()
                await ep.send(y)
            await ep.recv(ack_buf)

        times_full_sim = []
        for _ in range(BENCH_ITERS):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            await asyncio.sleep(0)  # scheduling yield
            t1 = time.perf_counter()
            async with send_lock:
                t2 = time.perf_counter()
                y = x_contig.contiguous()
                t3 = time.perf_counter()
                await ep.send(y)
                t4 = time.perf_counter()
            await ep.recv(ack_buf)
            times_full_sim.append({
                'sched': (t1 - t0) * 1e6,
                'lock': (t2 - t1) * 1e6,
                'contig': (t3 - t2) * 1e6,
                'send': (t4 - t3) * 1e6,
                'total': (t4 - t0) * 1e6,
            })

        # Signal server to stop
        ctrl_stop = np.array([-1], dtype=np.int64)
        await ep.send(ctrl_stop)
        await ep.close()

        # ============================================================
        # RESULTS
        # ============================================================
        def p50(data):
            s = sorted(data)
            return s[len(s)//2]
        def p95(data):
            s = sorted(data)
            return s[int(len(s)*0.95)]

        print("\n" + "=" * 90)
        print("  200us GAP ANALYSIS: Pure UCX (~107us) vs In-Inference (~307us)")
        print("=" * 90)

        print(f"\n  TEST 1: Baseline (direct await send, no overhead)")
        print(f"    send p50={p50(times_baseline):.1f}us  p95={p95(times_baseline):.1f}us")

        print(f"\n  TEST 2: + async lock (async with send_lock)")
        print(f"    send p50={p50(times_with_lock):.1f}us  p95={p95(times_with_lock):.1f}us")
        print(f"    lock overhead: {p50(times_with_lock) - p50(times_baseline):.1f}us")

        print(f"\n  TEST 3: + contiguous() on already-contiguous tensor")
        print(f"    send p50={p50(times_contig_noop):.1f}us  p95={p95(times_contig_noop):.1f}us")
        print(f"    contiguous(noop) overhead: {p50(times_contig_noop) - p50(times_baseline):.1f}us")

        contig_costs = [t[0] for t in times_contig_real]
        send_after_contig = [t[1] for t in times_contig_real]
        print(f"\n  TEST 4: + contiguous() on NON-contiguous tensor (strided)")
        print(f"    contiguous() p50={p50(contig_costs):.1f}us  p95={p95(contig_costs):.1f}us")
        print(f"    send after   p50={p50(send_after_contig):.1f}us  p95={p95(send_after_contig):.1f}us")

        sched_costs = [t[0] for t in times_sched_sim]
        send_after_sched = [t[1] for t in times_sched_sim]
        print(f"\n  TEST 5: + event loop scheduling (asyncio.sleep(0) yield)")
        print(f"    sched delay  p50={p50(sched_costs):.1f}us  p95={p95(sched_costs):.1f}us")
        print(f"    send after   p50={p50(send_after_sched):.1f}us  p95={p95(send_after_sched):.1f}us")

        print(f"\n  TEST 6: + event loop contention (background task)")
        print(f"    send p50={p50(times_contention):.1f}us  p95={p95(times_contention):.1f}us")
        print(f"    contention overhead: {p50(times_contention) - p50(times_baseline):.1f}us")
        print(f"    (bg task ran {contention_count[0]} iterations during test)")

        sched_all = [t['sched'] for t in times_full_sim]
        lock_all = [t['lock'] for t in times_full_sim]
        contig_all = [t['contig'] for t in times_full_sim]
        send_all = [t['send'] for t in times_full_sim]
        total_all = [t['total'] for t in times_full_sim]

        print(f"\n  TEST 7: FULL SIMULATION (sched + lock + contiguous + send)")
        print(f"    sched delay  p50={p50(sched_all):.1f}us")
        print(f"    lock acquire p50={p50(lock_all):.1f}us")
        print(f"    contiguous() p50={p50(contig_all):.1f}us")
        print(f"    UCX send     p50={p50(send_all):.1f}us")
        print(f"    TOTAL        p50={p50(total_all):.1f}us")
        print(f"    overhead vs baseline: {p50(total_all) - p50(times_baseline):.1f}us")

        print(f"\n  {'='*90}")
        print(f"  SUMMARY: Where does the ~200us gap come from?")
        print(f"  {'='*90}")
        print(f"    Pure UCX send (baseline):     {p50(times_baseline):>8.1f} us")
        print(f"    + lock overhead:              {p50(times_with_lock) - p50(times_baseline):>8.1f} us")
        print(f"    + scheduling yield:           {p50(sched_costs):>8.1f} us")
        print(f"    + contention:                 {p50(times_contention) - p50(times_baseline):>8.1f} us")
        print(f"    Full sim total:               {p50(total_all):>8.1f} us")
        print(f"    In-inference measured:            ~307 us")
        print(f"    Remaining unexplained:        {307 - p50(total_all):>8.1f} us")
        print(f"      (likely: run_coroutine_threadsafe cross-thread dispatch,")
        print(f"       GIL contention, daemon thread scheduling jitter)")
        print()

    bridge.run(_main())
    bridge.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=BASE_PORT)
    parser.add_argument("--gpu-server", type=int, default=0)
    parser.add_argument("--gpu-client", type=int, default=1)
    args = parser.parse_args()

    manager = multiprocessing.Manager()
    results = manager.dict()

    server_proc = multiprocessing.Process(
        target=run_server, args=(args.host, args.port, args.gpu_server, results)
    )
    client_proc = multiprocessing.Process(
        target=run_client, args=(args.host, args.port, args.gpu_client, results)
    )

    server_proc.start()
    import time as _t
    _t.sleep(3)
    client_proc.start()

    client_proc.join(timeout=120)
    server_proc.join(timeout=10)


if __name__ == "__main__":
    main()
