"""Manual 4-GPU integration entry for component staging.

Launch with torchrun --nproc-per-node=4 after supplying a real Qwen3 ModelRunner
factory through SGLANG_AFD_COMPONENT_RUNNER_FACTORY=module:function. The factory
must return the local max-world runner with serving TP=2.
"""
import importlib
import os

import torch.distributed as dist


def main():
    spec = os.environ["SGLANG_AFD_COMPONENT_RUNNER_FACTORY"]
    module, function = spec.split(":", 1)
    runner = getattr(importlib.import_module(module), function)()
    rank = dist.get_rank()
    ranks = list(range(4))
    # Explicit groups independent from the serving TP subgroup.
    data = dist.new_group(ranks=ranks, backend="gloo")
    control = dist.new_group(ranks=ranks, backend="gloo")
    stager = runner.attach_afd_component_stager(
        data, None, control, collective_commands_enabled=True
    )
    request = {
        "operation_id": "manual-a2-to-a4",
        "epoch": 0,
        "expected_attn_tp": 2,
        "expected_ffn_tp": 2,
        "target_attn_tp": 4,
        "target_ffn_tp": 4,
    }
    state = stager.prepare(request)
    assert state.state.value == "READY"
    dist.barrier(group=control)
    stager.activate(request)
    stager.retire(request)
    if rank == 0:
        print("AFD component TP2->TP4 staging/commit completed")


if __name__ == "__main__":
    main()
