#!/usr/bin/env python3
"""Summarize Attn(DP|TP) × MoE(TP|EP) matrix benchmark."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CONFIGS = [
    ("attn_tp_moe_tp", "Attn TP + MoE TP"),
    ("attn_tp_moe_ep", "Attn TP + MoE EP"),
    ("attn_dp_moe_tp", "Attn DP + MoE TP"),
    ("attn_dp_moe_ep", "Attn DP + MoE EP"),
]


def load(path: Path) -> dict:
    return json.loads(path.read_text().splitlines()[-1])


def main() -> None:
    base = Path(sys.argv[1])
    rows = []
    for key, label in CONFIGS:
        d = base / key
        for p in sorted(d.glob("c*.json"), key=lambda x: int(x.stem[1:])):
            c = int(p.stem[1:])
            j = load(p)
            rows.append({
                "config": key,
                "label": label,
                "concurrency": c,
                "completed": j.get("completed"),
                "output_throughput": j.get("output_throughput"),
                "mean_ttft_ms": j.get("mean_ttft_ms"),
                "median_ttft_ms": j.get("median_ttft_ms"),
                "p90_ttft_ms": j.get("p90_ttft_ms"),
                "p99_ttft_ms": j.get("p99_ttft_ms"),
                "mean_tpot_ms": j.get("mean_tpot_ms"),
                "mean_e2e_latency_ms": j.get("mean_e2e_latency_ms"),
                "duration_s": j.get("duration"),
            })

    # throughput pivot
    print("\n=== 输出吞吐 (tok/s) ===")
    hdr = f"{'C':>6}" + "".join(f"{label:>18}" for _, label in CONFIGS)
    print(hdr)
    print("-" * len(hdr))
    all_c = sorted({r["concurrency"] for r in rows})
    by = {(r["config"], r["concurrency"]): r for r in rows}
    for c in all_c:
        line = f"{c:>6}"
        for key, _ in CONFIGS:
            r = by.get((key, c))
            line += f"{r['output_throughput']:>18.0f}" if r else f"{'-':>18}"
        print(line)

    print("\n=== TTFT mean (ms) ===")
    print(hdr)
    print("-" * len(hdr))
    for c in all_c:
        line = f"{c:>6}"
        for key, _ in CONFIGS:
            r = by.get((key, c))
            line += f"{r['mean_ttft_ms']:>18.0f}" if r else f"{'-':>18}"
        print(line)

    print("\n=== TTFT p99 (ms) ===")
    print(hdr)
    print("-" * len(hdr))
    for c in all_c:
        line = f"{c:>6}"
        for key, _ in CONFIGS:
            r = by.get((key, c))
            line += f"{r['p99_ttft_ms']:>18.0f}" if r else f"{'-':>18}"
        print(line)

    print("\n=== TPOT mean (ms) ===")
    print(hdr)
    print("-" * len(hdr))
    for c in all_c:
        line = f"{c:>6}"
        for key, _ in CONFIGS:
            r = by.get((key, c))
            line += f"{r['mean_tpot_ms']:>18.1f}" if r else f"{'-':>18}"
        print(line)

    print("\n=== 明细 ===")
    print(f"{'Config':<20} {'C':>6} {'out/s':>9} {'TTFT':>8} {'p99TTFT':>9} {'TPOT':>8} {'dur':>7}")
    print("-" * 72)
    for r in sorted(rows, key=lambda x: (x["concurrency"], x["config"])):
        print(
            f"{r['label']:<20} {r['concurrency']:>6} "
            f"{r['output_throughput']:>9.0f} {r['mean_ttft_ms']:>8.0f} "
            f"{r['p99_ttft_ms']:>9.0f} {r['mean_tpot_ms']:>8.1f} {r['duration_s']:>7.1f}"
        )

    out = base / "summary.json"
    out.write_text(json.dumps({"rows": rows}, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
