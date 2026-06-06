"""Benchmark: C++ IPC vs Python IPC communication latency.

Measures per-iteration send+recv latency for AF disaggregation communication.
Runs sender and receiver in separate processes (simulating real AF deployment).

Usage:
    # Run both processes (needs 2 GPUs):
    python bench_cpp_ipc.py --role sender --device 0 --peer 1 --sync-mode ipc_event
    python bench_cpp_ipc.py --role receiver --device 1 --peer 0 --sync-mode ipc_event

    # Or use the launcher:
    python bench_cpp_ipc.py --launch --sync-mode ipc_event
"""

import argparse
import os
import sys
import time
import subprocess
import signal

import torch
import numpy as np


def run_sender(args):
    """Sender process: sends tensors in a loop, measures send latency."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["AFD_IPC_PEER_DEVICE"] = str(args.peer)
    os.environ["AFD_IPC_SYNC_MODE"] = args.sync_mode
    os.environ["AFD_IPC_CPP"] = "1" if args.backend == "cpp" else "0"

    torch.cuda.set_device(0)  # visible device 0

    if args.backend == "cpp":
        from sglang.srt.layers.afd_ipc_cpp.communicator import CppIpcTensorCommunicator
        from sglang.srt.layers.afd_type import AFDPerspective
        comm = CppIpcTensorCommunicator(
            AFDPerspective.AFD_PERSPECTIVE_ATTN, mb_id=None
        )
    else:
        from sglang.srt.layers.ipc_comm import IpcTensorCommunicator
        from sglang.srt.layers.afd_type import AFDPerspective
        comm = IpcTensorCommunicator(
            AFDPerspective.AFD_PERSPECTIVE_ATTN, mb_id=None
        )

    # Wait for handshake
    comm._wait_ready()
    print(f"[Sender] Handshake complete, sync_mode={args.sync_mode}")

    # Create test tensor (typical decode: bs=1, hidden=5120, bf16 = 10KB)
    bs = args.batch_size
    hidden = args.hidden_size
    x = torch.randn(bs, hidden, dtype=torch.bfloat16, device="cuda:0")

    # Warmup
    for _ in range(args.warmup):
        comm.send_tensor(x)
        # Wait for receiver to process (simple barrier via recv)
        _ = comm.recv_tensor()

    # Benchmark
    torch.cuda.synchronize()
    latencies = []
    for i in range(args.iters):
        t0 = time.perf_counter()
        comm.send_tensor(x)
        # Round-trip: wait for receiver's ack
        _ = comm.recv_tensor()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1e6)  # us

    latencies = np.array(latencies)
    print(f"\n[Sender] Backend={args.backend}, SyncMode={args.sync_mode}")
    print(f"  Tensor: [{bs}, {hidden}] bf16 = {bs * hidden * 2 / 1024:.1f} KB")
    print(f"  Round-trip latency (send + recv_ack):")
    print(f"    Mean:   {latencies.mean():.1f} us")
    print(f"    Median: {np.median(latencies):.1f} us")
    print(f"    P95:    {np.percentile(latencies, 95):.1f} us")
    print(f"    P99:    {np.percentile(latencies, 99):.1f} us")
    print(f"    Min:    {latencies.min():.1f} us")
    print(f"    Max:    {latencies.max():.1f} us")


def run_receiver(args):
    """Receiver process: receives tensors and sends ack back."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    os.environ["AFD_IPC_PEER_DEVICE"] = str(args.peer)
    os.environ["AFD_IPC_SYNC_MODE"] = args.sync_mode
    os.environ["AFD_IPC_CPP"] = "1" if args.backend == "cpp" else "0"

    torch.cuda.set_device(0)

    if args.backend == "cpp":
        from sglang.srt.layers.afd_ipc_cpp.communicator import CppIpcTensorCommunicator
        from sglang.srt.layers.afd_type import AFDPerspective
        comm = CppIpcTensorCommunicator(
            AFDPerspective.AFD_PERSPECTIVE_FFN, mb_id=None
        )
    else:
        from sglang.srt.layers.ipc_comm import IpcTensorCommunicator
        from sglang.srt.layers.afd_type import AFDPerspective
        comm = IpcTensorCommunicator(
            AFDPerspective.AFD_PERSPECTIVE_FFN, mb_id=None
        )

    comm._wait_ready()
    print(f"[Receiver] Handshake complete")

    # Ack tensor
    ack = torch.ones(1, dtype=torch.bfloat16, device="cuda:0")

    total_iters = args.warmup + args.iters
    recv_latencies = []

    for i in range(total_iters):
        t0 = time.perf_counter()
        data = comm.recv_tensor()
        t1 = time.perf_counter()
        # Send ack back
        comm.send_tensor(ack)

        if i >= args.warmup:
            recv_latencies.append((t1 - t0) * 1e6)

    recv_latencies = np.array(recv_latencies)
    print(f"\n[Receiver] Backend={args.backend}, SyncMode={args.sync_mode}")
    print(f"  Recv latency (flag_poll + P2P_copy + decode):")
    print(f"    Mean:   {recv_latencies.mean():.1f} us")
    print(f"    Median: {np.median(recv_latencies):.1f} us")
    print(f"    P95:    {np.percentile(recv_latencies, 95):.1f} us")
    print(f"    Min:    {recv_latencies.min():.1f} us")


def launch_both(args):
    """Launch sender and receiver as separate processes."""
    base_cmd = [
        sys.executable, __file__,
        "--batch-size", str(args.batch_size),
        "--hidden-size", str(args.hidden_size),
        "--warmup", str(args.warmup),
        "--iters", str(args.iters),
        "--backend", args.backend,
        "--sync-mode", args.sync_mode,
    ]

    # Start receiver first (it listens)
    recv_cmd = base_cmd + ["--role", "receiver", "--device", "1", "--peer", "0"]
    send_cmd = base_cmd + ["--role", "sender", "--device", "0", "--peer", "1"]

    print(f"Launching benchmark: backend={args.backend}, sync_mode={args.sync_mode}")
    print(f"  Tensor: [{args.batch_size}, {args.hidden_size}] bf16")
    print(f"  Iters: {args.iters} (warmup: {args.warmup})")
    print()

    recv_proc = subprocess.Popen(recv_cmd)
    time.sleep(1)  # Give receiver time to start listening
    send_proc = subprocess.Popen(send_cmd)

    try:
        send_proc.wait()
        recv_proc.wait()
    except KeyboardInterrupt:
        send_proc.send_signal(signal.SIGTERM)
        recv_proc.send_signal(signal.SIGTERM)


def main():
    parser = argparse.ArgumentParser(description="Benchmark C++ vs Python IPC")
    parser.add_argument("--role", choices=["sender", "receiver"], default=None)
    parser.add_argument("--launch", action="store_true",
                        help="Launch both sender and receiver")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--peer", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (num tokens)")
    parser.add_argument("--hidden-size", type=int, default=5120,
                        help="Hidden dimension")
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--backend", choices=["cpp", "python"], default="cpp")
    parser.add_argument("--sync-mode", choices=["cpu_flag", "ipc_event", "gpu_signal"],
                        default="ipc_event")

    args = parser.parse_args()

    if args.launch:
        launch_both(args)
    elif args.role == "sender":
        run_sender(args)
    elif args.role == "receiver":
        run_receiver(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
