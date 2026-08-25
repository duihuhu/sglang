#!/usr/bin/env python3
"""Format balanced vs skewed breakdown JSON from bench_ep_decode_breakdown.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

MOE_PARTS = ("gate", "topk", "dispatch", "moe_core", "combine", "ep_allreduce")

WATERFALL = (
    ("wall_total", "wall_us", "① wall_total（CPU 墙钟，layernorm+layer.mlp，与 EP 同口径）"),
    ("cuda_full_mlp", "cuda_full_mlp", "② cuda_full_mlp（CUDA Event：layernorm+layer.mlp 整段）"),
    ("gap_wall_vs_cuda_mlp", "gap_wall_vs_cuda_mlp", "③ gap_wall−cuda_mlp（同步/CPU−GPU 差）"),
    ("layernorm", "layernorm", "④ layernorm（CUDA Event）"),
    ("gate", "gate", "⑤ gate"),
    ("topk", "topk", "⑥ topk"),
    ("dispatch", "dispatch", "⑦ dispatch"),
    ("moe_core", "moe_core", "⑧ moe_core"),
    ("combine", "combine", "⑨ combine"),
    ("ep_allreduce", "ep_allreduce", "⑩ EP allreduce"),
    ("moe_cuda_total", "moe_cuda_total", "⑪ moe_cuda_total（CUDA：gate→AR 连续段）"),
    ("moe_parts_sum", "moe_parts_sum", "⑫ moe_parts_sum（⑤–⑩ 各段之和）"),
    ("gap_cuda_mlp_vs_moe_total", "gap_cuda_mlp_vs_moe_total", "⑬ gap_cuda_mlp−moe_cuda（整段 mlp vs 拆段路径差）"),
)


def load_rows(path: Path) -> dict[str, dict]:
    rows = json.loads(path.read_text())
    return {r["routing"]: r for r in rows}


def wall_max(row: dict) -> float:
    return max(r.get("wall_us", 0) for r in row["per_rank_us"])


def val(row: dict, key: str) -> float:
    if key == "wall_us":
        return wall_max(row)
    return row["max_rank_us"][key]


def main() -> None:
    rows = load_rows(Path(sys.argv[1]))
    bal = rows["balanced"]
    skew = rows["skewed_rank0"]

    print("=" * 88)
    print("balanced vs skewed | decode | EP=4 batch=32 len=512 930MHz | CustomAllReduce")
    print("max across ranks | 单位 µs")
    print("=" * 88)
    print(f"{'分段':<42} {'balanced':>12} {'skewed':>12} {'Δ(skew-bal)':>12}")
    for _, key, label in WATERFALL:
        a, b = val(bal, key), val(skew, key)
        print(f"{label:<42} {a:>12,.1f} {b:>12,.1f} {b - a:>+12,.1f}")

    w_bal, w_skew = wall_max(bal), wall_max(skew)
    print()
    print(f"skew/bal wall ratio: {w_skew / w_bal:.3f}")
    print()
    print("--- 各 rank wall_total (µs) ---")
    print(f"  balanced: {[round(r['wall_us'], 1) for r in bal['per_rank_us']]}")
    print(f"  skewed  : {[round(r['wall_us'], 1) for r in skew['per_rank_us']]}")
    print()
    print("--- 各 rank topk / moe_core / ep_allreduce (µs) ---")
    for i in range(len(bal["per_rank_us"])):
        br, sr = bal["per_rank_us"][i], skew["per_rank_us"][i]
        print(
            f"  rank{i} topk: {br['topk']:,.1f} vs {sr['topk']:,.1f} | "
            f"core: {br['moe_core']:,.1f} vs {sr['moe_core']:,.1f} | "
            f"ar: {br['ep_allreduce']:,.1f} vs {sr['ep_allreduce']:,.1f}"
        )


if __name__ == "__main__":
    main()
