"""Three-GPU shared-PA IPC smoke test.

Topology:
  GPU2 PA(TP1) -> GPU0 PF group 0
               -> GPU1 PF group 1

This validates independent channel handshake, concurrent in-flight receives,
and wait-any continuation ordering without launching a model server.
"""

import argparse
import multiprocessing as mp
import os
import queue
import time
import traceback

import torch

from sglang.srt.layers.afd_multi_peer import (
    AFDPeerChannelPool,
    AFDPeerSpec,
)
from sglang.srt.layers.afd_type import AFDPerspective
from sglang.srt.layers.afd_ipc_cpp.communicator import (
    CppIpcTensorCommunicator,
)

CHANNEL_BASE = 700
PA_DEVICE = 2
PF_DEVICES = (0, 1)


def _pf_worker(group_id: int, rounds: int, result_queue):
    try:
        torch.cuda.set_device(PF_DEVICES[group_id])
        os.environ["AFD_IPC_SYNC_MODE"] = "ipc_event"
        comm = CppIpcTensorCommunicator(
            AFDPerspective.AFD_PERSPECTIVE_FFN,
            peer_device=PA_DEVICE,
            channel_id=CHANNEL_BASE + group_id * 16,
        )
        for _ in range(rounds):
            tensor = comm.recv_tensor()
            output = tensor + (group_id + 1) * 10
            comm.send_tensor(output)
            # Keep the temporary output storage alive until ipc_cpp has copied it
            # into its persistent send pool.
            torch.cuda.current_stream().synchronize()
        torch.cuda.synchronize()
        result_queue.put(("ok", f"pf{group_id}"))
    except BaseException:
        result_queue.put(("error", f"pf{group_id}", traceback.format_exc()))


def _run_pa(rounds: int, tokens: int):
    torch.cuda.set_device(PA_DEVICE)
    os.environ["AFD_IPC_SYNC_MODE"] = "ipc_event"
    pool = AFDPeerChannelPool(
        AFDPerspective.AFD_PERSPECTIVE_ATTN,
        [
            AFDPeerSpec(0, CHANNEL_BASE, PF_DEVICES[0]),
            AFDPeerSpec(1, CHANNEL_BASE + 16, PF_DEVICES[1]),
        ],
    )

    completion_queue = queue.Queue()
    completion_order = []
    for round_id in range(rounds):
        expected = {}
        send_tensors = []
        for group_id in pool.group_ids:
            value = float(round_id * 100 + group_id)
            tensor = torch.full(
                (tokens, 4096),
                value,
                dtype=torch.bfloat16,
                device=f"cuda:{PA_DEVICE}",
            )
            send_tensors.append(tensor)
            expected[group_id] = float(
                (
                    torch.tensor(value, dtype=torch.bfloat16)
                    + (group_id + 1) * 10
                )
                .float()
                .item()
            )
            pool[group_id].async_comm.send_async(tensor)
            pool[group_id].recv_start(completion_queue.put)

        seen = set()
        while len(seen) < len(pool):
            group_id = completion_queue.get(timeout=30)
            if group_id in seen:
                raise RuntimeError(
                    f"duplicate completion for group {group_id}"
                )
            result = pool[group_id].async_comm.recv_wait()
            actual = float(result[0, 0].float().item())
            if actual != expected[group_id]:
                raise AssertionError(
                    f"round={round_id} group={group_id}: "
                    f"expected={expected[group_id]}, actual={actual}"
                )
            seen.add(group_id)
            completion_order.append(group_id)
        del send_tensors

    pool.drain()
    torch.cuda.synchronize()
    return completion_order


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=16)
    parser.add_argument("--tokens", type=int, default=256)
    args = parser.parse_args()

    if torch.cuda.device_count() < 3:
        raise RuntimeError("This smoke test requires at least three GPUs")

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    workers = [
        ctx.Process(
            target=_pf_worker,
            args=(group_id, args.rounds, result_queue),
        )
        for group_id in range(2)
    ]
    for worker in workers:
        worker.start()
    time.sleep(1)

    try:
        order = _run_pa(args.rounds, args.tokens)
    except BaseException:
        for worker in workers:
            if worker.is_alive():
                worker.kill()
            worker.join(timeout=5)
        raise

    child_results = [result_queue.get(timeout=30) for _ in workers]
    for worker in workers:
        worker.join(timeout=30)
        if worker.is_alive():
            worker.kill()
            raise RuntimeError(f"PF worker {worker.pid} did not exit")
        if worker.exitcode != 0:
            raise RuntimeError(
                f"PF worker {worker.pid} exited with {worker.exitcode}"
            )
    errors = [item for item in child_results if item[0] != "ok"]
    if errors:
        raise RuntimeError(f"PF worker failures: {errors}")

    counts = {group_id: order.count(group_id) for group_id in range(2)}
    print(
        f"PASS shared-PA IPC: rounds={args.rounds} tokens={args.tokens} "
        f"completion_counts={counts} first_completions={order[:12]}"
    )


if __name__ == "__main__":
    main()
