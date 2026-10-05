"""Diagnostic launcher: snapshot memory only on idle get_server_info requests.

The top-level hook is also installed in Python spawn children. No scheduling,
allocator, or kernel implementation is replaced.
"""

import gc
import json
import os
from pathlib import Path
import sys
import time


def memory_snapshot(scheduler):
    import torch

    torch.cuda.synchronize()
    device = torch.cuda.current_device()
    free, total = torch.cuda.mem_get_info(device)
    stats = {
        "allocated": torch.cuda.memory_allocated(device),
        "reserved": torch.cuda.memory_reserved(device),
        "peak_allocated_since_previous_audit": torch.cuda.max_memory_allocated(device),
        "peak_reserved_since_previous_audit": torch.cuda.max_memory_reserved(device),
        "cuda_free": free, "cuda_total": total, "cuda_used": total - free,
    }
    tensors, seen = {}, set()

    def walk(obj, path, depth=0):
        if id(obj) in seen:
            return
        seen.add(id(obj))
        if isinstance(obj, torch.Tensor):
            if obj.is_cuda and obj.device.index == device:
                tensors[path] = obj
            return
        if depth > 8:
            return
        if isinstance(obj, dict):
            entries = obj.items()
        elif isinstance(obj, (list, tuple)):
            entries = enumerate(obj)
        elif type(obj).__module__.startswith("sglang.") and hasattr(obj, "__dict__"):
            entries = vars(obj).items()
        else:
            return
        for name, value in entries:
            walk(value, f"{path}.{name}", depth + 1)

    model = scheduler.tp_worker.model_runner.model
    for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
        walk(tensor, f"weights.{name}")
    walk(scheduler.req_to_token_pool.req_to_token, "request_table")
    walk(scheduler.req_to_token_pool, "request_pool")
    walk(scheduler.token_to_kv_pool_allocator, "cache_allocator")
    # Also account for Python-visible workspaces not reachable from these roots.
    for obj in gc.get_objects():
        if type(obj) in (torch.Tensor, torch.nn.Parameter):
            walk(obj, f"other_tensor.{id(obj)}")
    storages = {}
    for name, tensor in tensors.items():
        storage = tensor.untyped_storage()
        key = storage._cdata
        if key not in storages:
            storages[key] = {"bytes": storage.nbytes(), "names": []}
        storages[key]["names"].append(name)
    unique = sum(s["bytes"] for s in storages.values())
    allocator = scheduler.token_to_kv_pool_allocator
    req_pool = scheduler.req_to_token_pool
    pool_attributes = {}
    for label, obj in (("request_pool", req_pool), ("allocator", allocator),
                       ("kv_pool", allocator.get_kvcache()),
                       ("mamba_pool", getattr(req_pool, "mamba_pool", None))):
        if obj is None:
            continue
        values = {k: v for k, v in vars(obj).items()
                  if isinstance(v, (str, bool, int, float, type(None))) and not k.startswith("__")}
        pool_attributes[label] = {"class": type(obj).__name__, "scalars": values}
    snapshot = {
        "pid": os.getpid(), "tp_rank": scheduler.ps.tp_rank, "cuda_device": device,
        "created_unix_s": time.time(), "memory": stats,
        "request_table_shape": list(req_pool.req_to_token.shape),
        "effective_request_cap": scheduler.max_running_requests,
        "token_capacity": scheduler.max_total_num_tokens,
        "storages": list(storages.values()), "unique_python_visible_storage_bytes": unique,
        "allocated_minus_visible_storages": stats["allocated"] - unique,
        "device_used_minus_torch_reserved": stats["cuda_used"] - stats["reserved"],
        "pool_attributes": pool_attributes,
        "scope": "Idle census; Python-visible tensors deduplicated by storage. Non-Torch and invisible allocations remain residuals.",
    }
    torch.cuda.reset_peak_memory_stats(device)
    return snapshot


def install():
    from sglang.srt.managers.scheduler import Scheduler

    original = Scheduler.get_internal_state

    def audited(self, request):
        result = original(self, request)
        if not self.is_fully_idle():
            result.internal_state["m2_pool_audit"] = {"skipped": "scheduler not idle"}
            return result
        snapshot = memory_snapshot(self)
        index = getattr(self, "_m2_memory_audit_index", 0) + 1
        self._m2_memory_audit_index = index
        target = Path(os.environ["M2_POOL_AUDIT_DIR"]) / f"rank{self.ps.tp_rank}-{index:04d}.json"
        with target.open("x") as handle:
            json.dump(snapshot, handle, indent=2, allow_nan=False)
        result.internal_state["m2_pool_audit"] = {
            "path": str(target), "memory": snapshot["memory"], "audit_index": index,
            "unique_storage_bytes": snapshot["unique_python_visible_storage_bytes"],
        }
        return result

    Scheduler.get_internal_state = audited


if os.environ.get("M2_POOL_AUDIT_DIR"):
    install()


if __name__ == "__main__":
    from sglang.launch_server import run_server
    from sglang.srt.plugins import load_plugins
    from sglang.srt.server_args import prepare_server_args
    from sglang.srt.utils import kill_process_tree

    load_plugins()
    args = prepare_server_args(sys.argv[1:])
    try:
        run_server(args)
    finally:
        kill_process_tree(os.getpid(), include_parent=False)
