#!/usr/bin/env python3
"""Read-only source witness: count selected work separately from new slots.

Exploratory diagnostic added after observing E2 state-low traces. Executes
the repository's actual scalar gate and allocator methods without importing
the serving stack or changing a running server.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
SCHEDULER = ROOT / "python/sglang/srt/managers/scheduler.py"
POOL = ROOT / "python/sglang/srt/mem_cache/memory_pool.py"


def method(path, class_name, method_name, namespace):
    tree = ast.parse(path.read_text())
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    function = next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == method_name)
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), function], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace[method_name]


def witness():
    parallel = SimpleNamespace(pp_max_micro_batch_size=8)
    get_limit = method(SCHEDULER, "Scheduler", "get_num_allocatable_reqs",
                       {"get_parallel": lambda: parallel})
    allocate = method(POOL, "ReqToTokenPool", "alloc", {})
    gate = method(SCHEDULER, "Scheduler", "_prefill_req_limit_reached",
                  {"get_parallel": lambda: parallel})
    pool = SimpleNamespace(free_slots=list(range(4, 9)), req_generation={i: 0 for i in range(1, 9)})
    pool.available_size = lambda: len(pool.free_slots)
    scheduler = SimpleNamespace(req_to_token_pool=pool)
    scheduler.get_num_allocatable_reqs = lambda running: get_limit(scheduler, running)
    continuation = SimpleNamespace(req_pool_idx=3, inflight_middle_chunks=1, kv_committed_len=4096)
    fresh = [SimpleNamespace(req_pool_idx=None, inflight_middle_chunks=0, kv_committed_len=0)
             for _ in range(5)]
    selected = [continuation, *fresh[:4]]
    scheduler._tenant_legacy_slot_gate = True
    blocked = gate(scheduler, 2, len(selected), 1, False)
    scheduler._tenant_legacy_slot_gate = False
    corrected_blocks = gate(scheduler, 2, len(selected), 1, False)
    free_before = pool.available_size()
    allocated = allocate(pool, [continuation, *fresh])
    result = {
        "classification": "actual legacy and corrected gate source witness; GPU effects measured separately",
        "running_requests": 2, "existing_continuations": 1, "selected_new_requests": 4,
        "free_request_slots_before": free_before, "runtime_cap": 8,
        "actual_gate_blocks_next_candidate": blocked,
        "corrected_gate_blocks_next_candidate": corrected_blocks,
        "actual_allocator_accepts_continuation_plus_five_new": allocated is not None,
        "new_slots_consumed": free_before - pool.available_size(),
        "active_requests_if_candidate_added": 8,
        "scope": "Request-slot gate only. Other budgets and serving semantics require separate checks. Not hybrid-exclusive.",
        "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (SCHEDULER, POOL)},
    }
    assert blocked and not corrected_blocks and allocated is not None and result["new_slots_consumed"] == 5
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(witness(), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result)
    print(result)


if __name__ == "__main__":
    main()
