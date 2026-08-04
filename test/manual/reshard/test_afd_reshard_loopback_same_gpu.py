#!/usr/bin/env python3
"""Two-process, one-GPU smoke test for afd_reshard_loopback.

Run on node1 (or any CUDA node):
  python test/manual/reshard/test_afd_reshard_loopback_same_gpu.py --device 0
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import socket
import time

import torch


def _free_base_port() -> int:
    for _ in range(100):
        first = socket.socket()
        first.bind(("127.0.0.1", 0))
        port = first.getsockname()[1]
        first.close()
        if port >= 65535:
            continue
        second = socket.socket()
        try:
            second.bind(("127.0.0.1", port + 1))
        except OSError:
            second.close()
            continue
        second.close()
        return port
    raise RuntimeError("could not reserve adjacent TCP ports")


def _worker(
    role: str, device: int, port: int, delay_s: float, result: mp.Queue
) -> None:
    try:
        time.sleep(delay_s)
        torch.cuda.set_device(device)
        from sglang.srt.layers.afd_reshard_loopback import (
            AFDReshardLoopbackTensorCommunicator,
        )
        from sglang.srt.layers.afd_type import AFDPerspective

        perspective = (
            AFDPerspective.AFD_PERSPECTIVE_ATTN
            if role == "attn"
            else AFDPerspective.AFD_PERSPECTIVE_FFN
        )
        with AFDReshardLoopbackTensorCommunicator(
            perspective,
            local_device=torch.device("cuda", device),
            base_port=port,
            timeout_s=30,
            capacity_bytes=4 * 1024 * 1024,
        ) as comm:
            if role == "attn":
                sent = torch.arange(
                    30, device="cuda", dtype=torch.float32
                ).reshape(5, 6)
                comm.send_tensor(sent)
                received = comm.recv_tensor()
                assert received.device.index == device
                assert received.dtype == torch.int64
                assert received.tolist() == [9, 7, 5, 3]
            else:
                received = comm.recv_tensor()
                assert received.device.index == device
                assert received.dtype == torch.float32
                assert torch.equal(
                    received,
                    torch.arange(30, device="cuda", dtype=torch.float32).reshape(5, 6),
                )
                comm.send_tensor(
                    torch.tensor([9, 7, 5, 3], device="cuda", dtype=torch.int64)
                )
            comm.fence(epoch=7)
        result.put((role, "ok", device))
    except BaseException as exc:
        result.put((role, "error", repr(exc)))
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--delay-role", choices=("attn", "ffn"), default="attn")
    parser.add_argument("--delay-seconds", type=float, default=2.0)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error(
            f"invalid --device {args.device}; visible count={torch.cuda.device_count()}"
        )

    ctx = mp.get_context("spawn")
    result = ctx.Queue()
    port = _free_base_port()
    processes = [
        ctx.Process(
            target=_worker,
            args=(
                role,
                args.device,
                port,
                args.delay_seconds if role == args.delay_role else 0.0,
                result,
            ),
            name=f"afd-{role}",
        )
        for role in ("ffn", "attn")
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=60)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)
            raise TimeoutError(f"{process.name} timed out")
        if process.exitcode != 0:
            raise RuntimeError(f"{process.name} exited with {process.exitcode}")
    values = sorted(result.get(timeout=2) for _ in processes)
    if any(item[1] != "ok" for item in values):
        raise RuntimeError(values)
    print(
        {
            "status": "ok",
            "physical_device": args.device,
            "processes": values,
            "port": port,
            "delayed_role": args.delay_role,
            "delay_seconds": args.delay_seconds,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
