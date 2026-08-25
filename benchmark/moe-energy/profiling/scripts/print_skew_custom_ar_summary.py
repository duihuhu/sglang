#!/usr/bin/env python3
"""Print skewed custom AR A/B tables from skewed_custom_ar_fast JSON outputs."""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text())


def main() -> None:
    nccl_path = Path(sys.argv[1])
    car_path = Path(sys.argv[2])
    nccl, car = load(nccl_path), load(car_path)

    print("=" * 72)
    print("skewed_rank0 | batch=32 | EP=4 | 单位：微秒 (µs)")
    print("=" * 72)
    print()
    print("【参数含义】")
    print("  disable_custom_all_reduce=True  → NCCL（EP 端到端 profiling 用的）")
    print("  disable_custom_all_reduce=False → CustomAllReduce（旧 breakdown 默认）")
    print()
    print("【测的是什么】")
    print("  wall_total     : CPU 墙钟，layernorm + 整段 layer.mlp()（与 data/EP 一致）")
    print("  cuda_full_mlp  : CUDA Event，layernorm + 整段 layer.mlp()")
    print("  cuda_moe_core  : CUDA Event，仅 experts.run_moe_core（不含 AR）")
    print("  cuda_ep_ar     : CUDA Event，仅 moe_expert_parallel_all_reduce")
    print("  EP 延迟取 max(rank)，即最慢 rank 的时间")
    print()

    headers = ("指标", "NCCL max", "CustomAR max", "Δ (Custom−NCCL)")
    print("--- 端到端总时间（看排名用这个）---")
    print(f"{headers[0]:<16} {headers[1]:>10} {headers[2]:>12} {headers[3]:>14}")
    nccl_wall = max(r["wall_us"] for r in nccl)
    car_wall = max(r["wall_us"] for r in car)
    print(f"{'wall_total':<16} {nccl_wall:>10.0f} {car_wall:>12.0f} {car_wall-nccl_wall:>+14.0f}")

    nccl_full = max(r["cuda_full_mlp_us"] for r in nccl)
    car_full = max(r["cuda_full_mlp_us"] for r in car)
    print(f"{'cuda_full_mlp':<16} {nccl_full:>10.0f} {car_full:>12.0f} {car_full-nccl_full:>+14.0f}")
    print()

    print("--- 分段（CUDA Event，max rank）---")
    for key, label in [
        ("cuda_moe_core_us", "moe_core"),
        ("cuda_ep_ar_us", "ep_allreduce"),
    ]:
        a = max(r[key] for r in nccl)
        b = max(r[key] for r in car)
        print(f"{label:<16} {a:>10.0f} {b:>12.0f} {b-a:>+14.0f}")

    print()
    print("--- 各 rank 明细：ep_allreduce（µs）← 参数主要改这里 ---")
    print("rank | NCCL wall | NCCL ep_ar | CustomAR wall | CustomAR ep_ar")
    for i in range(4):
        n, c = nccl[i], car[i]
        print(
            f"  {i}  | {n['wall_us']:9.0f} | {n['cuda_ep_ar_us']:10.0f} | "
            f"{c['wall_us']:11.0f} | {c['cuda_ep_ar_us']:12.0f}"
        )

    print()
    print("--- 各 rank 明细：moe_core（µs）---")
    print("rank | NCCL core | CustomAR core")
    for i in range(4):
        print(f"  {i}  | {nccl[i]['cuda_moe_core_us']:9.0f} | {car[i]['cuda_moe_core_us']:12.0f}")

    print()
    print("【结论】")
    ar_nccl = [nccl[i]["cuda_ep_ar_us"] for i in range(4)]
    ar_car = [car[i]["cuda_ep_ar_us"] for i in range(4)]
    print(
        f"  ep_allreduce 隔离 CUDA 段：rank0/1/3  NCCL {ar_nccl[0]:.0f}/{ar_nccl[1]:.0f}/{ar_nccl[3]:.0f} µs"
        f"  vs CustomAR {ar_car[0]:.0f}/{ar_car[1]:.0f}/{ar_car[3]:.0f} µs"
    )
    print(f"  wall_total max：NCCL {nccl_wall:.0f} µs vs CustomAR {car_wall:.0f} µs（Δ {car_wall-nccl_wall:+.0f} µs）")


if __name__ == "__main__":
    main()
