#!/usr/bin/env python3
"""Format skewed breakdown JSON from bench_ep_decode_breakdown.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

COMPONENTS = ("gate", "topk", "dispatch", "moe_core", "combine", "ep_allreduce", "total")
LABELS = {
    "gate": "gate",
    "topk": "topk",
    "dispatch": "dispatch",
    "moe_core": "moe_core",
    "combine": "combine",
    "ep_allreduce": "EP allreduce",
    "total": "CUDA分段合计",
}


def load(path: Path) -> dict:
    rows = json.loads(path.read_text())
    if not rows:
        raise SystemExit(f"empty: {path}")
    return rows[0]


def main() -> None:
    nccl = load(Path(sys.argv[1]))
    custom = load(Path(sys.argv[2]))
    m_nccl = nccl["max_rank_us"]
    m_custom = custom["max_rank_us"]
    wall_nccl = nccl.get("per_rank_us", [{}])[0].get("wall_max_us") or max(
        r.get("wall_max_us", 0) for r in nccl["per_rank_us"]
    )
    # wall stored per-rank in local dict - check structure
    if "wall_max_us" in nccl:
        wall_nccl = nccl["wall_max_us"]
    else:
        walls = [r.get("wall_max_us", r.get("wall_us", 0)) for r in nccl["per_rank_us"]]
        wall_nccl = max(walls) if walls else 0
    if "wall_max_us" in custom:
        wall_custom = custom["wall_max_us"]
    else:
        walls = [r.get("wall_max_us", r.get("wall_us", 0)) for r in custom["per_rank_us"]]
        wall_custom = max(walls) if walls else 0

    # wall from print line - actually per_rank has wall_max_us on each? check structure
    # From bench script: local has wall_max_us only on rank0 gather - each rank has wall in profile_moe_wall_clock on rank0 only
    # Actually local dict per rank: gate, topk, ... wall_max_us, wall_per_rank_us only added to local on each rank?
    # local["wall_max_us"] = wall["wall_max_us"] - same max on all ranks from profile_world
    # local doesn't have per-rank wall_us in gathered - only wall_per_rank_us on rank0's local?
    # rank0 local has wall_max_us and wall_per_rank_us list

    rank0_nccl = nccl["per_rank_us"][0]
    rank0_custom = custom["per_rank_us"][0]
    wall_nccl = nccl.get("wall_max_us") or max(r.get("wall_us", 0) for r in nccl["per_rank_us"])
    wall_custom = custom.get("wall_max_us") or max(r.get("wall_us", 0) for r in custom["per_rank_us"])
    wall_per_nccl = [r.get("wall_us", 0) for r in nccl["per_rank_us"]]
    wall_per_custom = [r.get("wall_us", 0) for r in custom["per_rank_us"]]

    print("=" * 78)
    print("skewed_rank0 breakdown | batch=32 | EP=4 | length=512 | 930 MHz | 单位：µs")
    print("=" * 78)
    print()
    print("disable_custom_all_reduce=True  → NCCL（与 EP 端到端 profiling 一致）")
    print("disable_custom_all_reduce=False → CustomAllReduce")
    print()
    print("--- 总时间（端到端，与 data/EP 同口径：max rank wall clock）---")
    print(f"  NCCL wall_total     : {wall_nccl:,.1f} µs")
    print(f"  CustomAR wall_total : {wall_custom:,.1f} µs")
    print(f"  Δ (Custom − NCCL)   : {wall_custom - wall_nccl:+,.1f} µs")
    print()
    print("--- CUDA Event 分段（max across ranks，不含 layernorm）---")
    print(f"{'分段':<14} {'NCCL (µs)':>12} {'CustomAR (µs)':>14} {'Δ (µs)':>10}")
    for c in COMPONENTS:
        a, b = m_nccl[c], m_custom[c]
        print(f"{LABELS[c]:<14} {a:>12,.1f} {b:>14,.1f} {b - a:>+10,.1f}")
    sum_nccl = sum(m_nccl[c] for c in COMPONENTS if c != "total")
    print(f"{'各段之和*':<14} {sum_nccl:>12,.1f}  (*total 含段间空隙)")
    print()
    print("--- 各 rank wall_total (µs) ---")
    print(f"  NCCL     : {[round(x, 1) for x in wall_per_nccl]}")
    print(f"  CustomAR : {[round(x, 1) for x in wall_per_custom]}")
    print()
    print("--- 各 rank EP allreduce CUDA 段 (µs) ---")
    for i in range(4):
        print(
            f"  rank{i}: NCCL {nccl['per_rank_us'][i]['ep_allreduce']:,.1f}  |  "
            f"CustomAR {custom['per_rank_us'][i]['ep_allreduce']:,.1f}"
        )


if __name__ == "__main__":
    main()
