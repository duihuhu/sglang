#!/usr/bin/env python3
"""CPU/meta-only M2 pool audit. This is not a serving benchmark or HBM census."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


BASE = Path(__file__).resolve().parent
ROOT = BASE.parents[2]
REFERENCE = BASE / "results/m2_state_mediation_20260916/inspect03/m2mediate-natural-mixed-4096-cap485-rep1-inspect.json"
COUNT = 607
INPUT = 512
OUTPUT = 128


def bootstrap_plan(state_bytes, token_bytes, context_len, natural, target=COUNT):
    """Hold the old main-cache budget, not full HBM, for bootstrap estimates."""
    slots, tokens, requests = natural
    if min(state_bytes, token_bytes, context_len, slots, tokens, requests, target) <= 0:
        raise ValueError("All capacities and byte sizes must be positive")
    if requests != slots:
        raise ValueError("This protocol expects radix-off one-state-per-request")
    budget = (slots + 1) * state_bytes + tokens * token_bytes
    rebalanced_tokens = (budget - (target + 1) * state_bytes) // token_bytes
    if rebalanced_tokens < 2 * target * (INPUT + OUTPUT):
        raise ValueError("Rebalanced bootstrap has insufficient KV headroom")
    rows = []
    for name, state_cap, kv_cap, unified in (
        ("static-natural", slots, tokens, False),
        ("static-rebalanced", target, rebalanced_tokens, False),
        ("unified-raised-cap", target, rebalanced_tokens, True),
        ("unified-old-cap-control", slots, tokens, True),
    ):
        main_bytes = kv_cap * token_bytes + (state_cap + (not unified)) * state_bytes
        tail_bytes = token_bytes if unified else 0  # MLA page_size=1
        static_padding = 0 if unified else token_bytes  # MLATokenToKVPool: size + page_size
        rows.append({
            "arm": name,
            "enable_unified_memory": unified,
            "bootstrap_state_cap": state_cap,
            "bootstrap_kv_tokens": kv_cap,
            "main_cache_bytes": main_bytes,
            "unified_tail_pad_bytes": tail_bytes,
            "static_kv_padding_bytes": static_padding,
            "main_allocation_bytes": main_bytes + tail_bytes + static_padding,
            "main_budget_delta_bytes": main_bytes - budget,
            "main_allocation_delta_bytes": main_bytes + tail_bytes + static_padding - (budget + token_bytes),
        })
    return rows


def unique_storage_bytes(named_tensors):
    """Count supplied live storages once; not a recursive process-wide census."""
    groups = {}
    for name, tensor in named_tensors.items():
        storage = tensor.untyped_storage()
        # _cdata identifies shared StorageImpl even on meta (all data_ptrs are 0).
        key = (str(tensor.device), storage._cdata)
        if key not in groups:
            groups[key] = {"device": str(tensor.device), "bytes": storage.nbytes(), "names": []}
        groups[key]["names"].append(name)
    return {"unique_bytes": sum(g["bytes"] for g in groups.values()),
            "storage_count": len(groups), "groups": list(groups.values())}


def native_request_cap(state_cap, token_capacity, context_len, unified, requested=None):
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.runtime_context import get_context

    # Exercise the real resolver with a minimal receiver, without profiling a GPU.
    resolver = object.__new__(KVCacheConfigurator)
    resolver.model_config = SimpleNamespace(context_len=context_len)
    resolver.ps = SimpleNamespace(attn_dp_size=1)
    resolver.mambaish_config = object()
    override = get_context().override_server_args(
        disable_radix_cache=True, max_mamba_cache_size=state_cap,
        max_running_requests=requested, enable_unified_memory=unified,
    )
    override.install()
    try:
        return resolver.resolve_max_num_reqs(token_capacity)
    finally:
        override.restore()


def backend_guard_check(state):
    from sglang.srt.server_args import ServerArgs

    expected = {
        "attention_backend": "triton", "linear_attn_backend": "triton",
        "mamba_backend": "triton", "disable_radix_cache": True,
        "enable_unified_memory": False, "page_size": 1, "tp_size": 2,
        "dcp_size": 1, "pp_size": 1, "speculative_algorithm": None,
        "cuda_graph_backend_decode": "disabled", "cuda_graph_backend_prefill": "disabled",
        "dtype": "auto", "kv_cache_dtype": "auto", "mamba_ssm_dtype": None,
    }
    for key, value in expected.items():
        if key not in state or state[key] != value:
            raise ValueError(f"Reference mismatch: {key}={state.get(key)!r}, expected {value!r}")
    args = SimpleNamespace(**state)
    args.enable_unified_memory = True
    args.cuda_graph_config = None  # Both graph backends checked disabled above.
    args.use_mla_backend = lambda: True
    args._resolved_attention_backends = lambda: [
        args.prefill_attention_backend or args.attention_backend,
        args.decode_attention_backend or args.attention_backend,
    ]
    ServerArgs._handle_page_major_kv_layout(args)
    ServerArgs._handle_unified_memory_pool(args)
    return {"guard_checks_passed": True, "page_major_implied": args.enable_page_major_kv_layout,
            "full_model_startup_or_kernel_parity_tested": False}


def audit(model_path, reference_path):
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise RuntimeError("Run with CUDA_VISIBLE_DEVICES='' before importing torch")
    import torch
    from sglang.srt.configs.kimi_linear import KimiLinearConfig
    from sglang.srt.configs.mamba_utils import KimiLinearCacheParams, KimiLinearStateShape
    from sglang.srt.mem_cache.unified_memory_pool import MLASubPoolSpec, MambaSubPoolSpec, UnifiedKVPool
    from sglang.srt.utils.hf_transformers.common import get_context_length

    if torch.cuda.is_available():
        raise RuntimeError("GPU must be hidden for this audit")
    model_json = json.loads(model_path.read_text())
    cfg = KimiLinearConfig(**model_json)
    reference = json.loads(reference_path.read_text())
    state = reference["server_initial"]["internal_states"][0]
    guards = backend_guard_check(state)
    context_len = get_context_length(cfg)
    if context_len != 1048576 or state["context_length"] is not None:
        raise ValueError("Unexpected context length; review request-table assumptions")
    if str(model_json.get("dtype")) != "bfloat16" or state["kv_cache_dtype"] != "auto":
        raise ValueError("This audit is frozen to the observed BF16 model/KV profile")
    linear = cfg.linear_attn_config
    shape = KimiLinearStateShape.create(
        tp_world_size=2, num_heads=linear["num_heads"], head_dim=linear["head_dim"],
        conv_kernel_size=linear["short_conv_kernel_size"],
    )
    cp = KimiLinearCacheParams(shape=shape, layers=cfg.linear_layer_ids)
    if (cp.dtype.conv, cp.dtype.temporal) != (torch.bfloat16, torch.float32):
        raise ValueError("State dtype differs from the previous serving profile")
    full = MLASubPoolSpec(name="full", layer_num=len(cfg.full_attention_layer_ids),
                         kv_lora_rank=cfg.kv_lora_rank, qk_rope_head_dim=cfg.qk_rope_head_dim,
                         store_dtype=torch.bfloat16, grow_direction="down")
    mamba = MambaSubPoolSpec(name="mamba", layer_num=len(cp.layers),
                           conv_state_shapes=tuple(shape.conv), conv_dtype=cp.dtype.conv,
                           temporal_state_shape=shape.temporal, temporal_dtype=cp.dtype.temporal,
                           conv_slice_axis=shape.conv_slice_axis, grow_direction="up")
    if mamba.entry_bytes() != cp.mamba_cache_per_req:
        raise ValueError("Unified and static state geometry disagree")
    natural = reference["resolved_capacity"][:3]
    if natural != [485, 1458643, 485]:
        raise ValueError("Unexpected reference capacity; do not silently change the protocol")
    rows = bootstrap_plan(mamba.entry_bytes(), full.entry_bytes(), context_len, natural)
    for row in rows:
        unified = row["enable_unified_memory"]
        requests = native_request_cap(row["bootstrap_state_cap"], row["bootstrap_kv_tokens"], context_len, unified)
        row["resolved_request_cap_offline"] = requests
        row["request_table_bytes"] = (requests + 1) * (context_len + 4) * 4
        row["known_cache_plus_request_table_bytes"] = row["main_allocation_bytes"] + row["request_table_bytes"]
        row["cache_budget_is_full_hbm_budget"] = False
        row["fresh_server_cap_and_memory_readback_required"] = True
        if unified:
            pool = UnifiedKVPool(total_bytes=row["main_cache_bytes"], sub_pool_specs=[full, mamba],
                                 device="meta", enable_memory_saver=False, page_size=1,
                                 view_tail_pad_bytes=row["unified_tail_pad_bytes"])
            conv, temporal = pool.mamba_views_for("mamba")
            tensors = {"raw": pool._raw, "temporal": temporal}
            tensors.update({f"conv_{i}": t for i, t in enumerate(conv)})
            tensors.update({f"mla_layer_{i}": t for i, t in enumerate(pool.mla_views_for("full"))})
            storage = unique_storage_bytes(tensors)
            if storage["storage_count"] != 1 or storage["unique_bytes"] != row["main_allocation_bytes"]:
                raise ValueError("Shared view storage accounting mismatch")
            row["meta_layout"] = {
                "unique_storage": storage,
                "virtual_slots": {name: pool.max_slots(name) for name in ("mamba", "full")},
                "reserved_slot_prefix": {name: pool.min_slot_index(name) for name in ("mamba", "full")},
                "virtual_capacities_are_not_simultaneously_allocatable": True,
                "views": {name: {"shape": list(t.shape), "stride": list(t.stride()),
                                  "storage_offset": t.storage_offset()} for name, t in tensors.items()},
            }
    if [row["resolved_request_cap_offline"] for row in rows] != [485, COUNT, COUNT, 485]:
        raise ValueError("Request-cap semantics changed; review the experiment")
    paths = [Path(__file__), BASE / "m2_pool_comparison_protocol_20260916.md", model_path, reference_path]
    paths += [ROOT / "python/sglang/srt" / name for name in (
        "configs/kimi_linear.py", "configs/mamba_utils.py", "mem_cache/unified_memory_pool.py",
        "mem_cache/multi_ended_allocator.py", "mem_cache/kv_cache_configurator.py",
        "mem_cache/memory_pool.py", "model_executor/pool_configurator.py", "server_args.py",
    )]
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": "offline_preflight_passed_live_validation_pending",
        "torch_version": torch.__version__, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "gpu_memory_allocated_by_audit": False, "kernel_parity_tested": False,
        "geometry_per_rank": {"state_bytes": mamba.entry_bytes(), "kv_bytes_per_token": full.entry_bytes(),
                              "context_len": context_len, "kda_layers": len(cp.layers),
                              "mla_layers": len(cfg.full_attention_layer_ids), "tp": 2,
                              "conv_dtype": str(cp.dtype.conv), "temporal_dtype": str(cp.dtype.temporal)},
        "backend_guards": guards, "rows": rows,
        "workload_cache_lower_accounting_bytes": COUNT * mamba.entry_bytes() + COUNT * (INPUT + OUTPUT) * full.entry_bytes(),
        "limitations": ["No live allocator collision/compaction or kernel tests", "No full-HBM census",
                        "Bootstrap KV estimates require fresh server resolution", "No serving performance claim"],
        "sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
    }


def render_report(result):
    g = result["geometry_per_rank"]
    lines = ["# M2 Pool Comparison: Offline Preflight", "", f"Status: `{result['status']}`.", "",
             f"Per rank: state={g['state_bytes']} B, KV={g['kv_bytes_per_token']} B/token, context={g['context_len']}.", "",
             "CPU/meta audit only. No model startup, kernel parity, full-HBM match, or performance result.", "",
             "| Arm | Bootstrap state / KV | Resolved requests (offline) | Main allocation B | Request table B |",
             "|---|---:|---:|---:|---:|"]
    for row in result["rows"]:
        lines.append(f"| {row['arm']} | {row['bootstrap_state_cap']} / {row['bootstrap_kv_tokens']} | "
                     f"{row['resolved_request_cap_offline']} | {row['main_allocation_bytes']} | {row['request_table_bytes']} |")
    lines += ["", "The main allocation column includes static KV padding or unified MLA tail padding.",
              "Other auxiliary allocations still require a live storage census; this table is not full HBM.", "",
              "Unified virtual state/KV maxima cannot be filled simultaneously. The unchanged-cap control",
              "still resolves to 485 requests even though the byte pool is shared.", "",
              "Next gate: live per-rank storage/memory audit and output correctness, then freeze budget-matched runs.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", type=Path, default=Path("/models/Kimi-Linear-48B-A3B/config.json"))
    parser.add_argument("--reference", type=Path, default=REFERENCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(f"Refusing to overwrite {args.out}")
    result = audit(args.model_config, args.reference)
    args.out.mkdir(parents=True, exist_ok=False)
    for name, content in (("preflight.json", json.dumps(result, indent=2) + "\n"), ("report.md", render_report(result))):
        with (args.out / name).open("x") as handle:
            handle.write(content)
    print(render_report(result))


if __name__ == "__main__":
    main()
