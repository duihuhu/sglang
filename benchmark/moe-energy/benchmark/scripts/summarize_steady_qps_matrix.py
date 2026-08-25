#!/usr/bin/env python3
"""Summarize steady QPS matrix results with batch-size stats."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

CONFIGS = [
    ("attn_tp_moe_tp", "Attn TP + MoE TP"),
    ("attn_tp_moe_ep", "Attn TP + MoE EP"),
    ("attn_dp_moe_tp", "Attn DP + MoE TP"),
    ("attn_dp_moe_ep", "Attn DP + MoE EP"),
]


def load_jsonl_last(path: Path) -> dict:
    text = path.read_text().strip()
    return json.loads(text.splitlines()[-1])


def batch_stats(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    vals = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if "sum_running_reqs" in row:
            vals.append(int(row["sum_running_reqs"]))
    if not vals:
        return {}
    n = len(vals)
    lo = n // 5
    hi = n - n // 5
    steady = sorted(vals[lo:hi] if hi > lo else vals)
    if not steady:
        steady = sorted(vals)

    def pct(p: float) -> float:
        idx = min(int(len(steady) * p), len(steady) - 1)
        return float(steady[idx])

    return {
        "batch_samples": n,
        "batch_running_mean": sum(steady) / len(steady),
        "batch_running_p50": pct(0.50),
        "batch_running_p90": pct(0.90),
        "batch_running_max": float(max(vals)),
    }


def main() -> None:
    base = Path(sys.argv[1])
    rows = []
    for key, label in CONFIGS:
        cfg_dir = base / key
        if not cfg_dir.is_dir():
            continue
        for qdir in sorted(cfg_dir.glob("qps*"), key=lambda p: int(p.name[3:])):
            qps = int(qdir.name[3:])
            result = qdir / "result.jsonl"
            if not result.is_file():
                continue
            j = load_jsonl_last(result)
            row = {
                "config": key,
                "label": label,
                "request_rate": qps,
                "completed": j.get("completed"),
                "output_throughput": j.get("output_throughput"),
                "max_output_tokens_per_s": j.get("max_output_tokens_per_s", 0),
                "concurrency": j.get("concurrency"),
                "max_concurrent_requests": j.get("max_concurrent_requests", 0),
                "mean_ttft_ms": j.get("mean_ttft_ms"),
                "p99_ttft_ms": j.get("p99_ttft_ms"),
                "mean_tpot_ms": j.get("mean_tpot_ms"),
                "duration_s": j.get("duration"),
            }
            row.update(batch_stats(qdir / "batch_samples.jsonl"))
            rows.append(row)

    if not rows:
        print(f"No results under {base}")
        return

    print("\n=== 四宫格稳态 QPS 汇总（含 running batch）===")
    hdr = (
        f"{'Config':<20} {'QPS':>4} {'avg_out':>8} {'peak_out':>8} "
        f"{'conc':>6} {'batch_p50':>9} {'batch_p90':>9} {'batch_max':>9} "
        f"{'ttft99':>8} {'tpot':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(
            f"{r['label']:<20} {r['request_rate']:>4} "
            f"{r['output_throughput']:>8.0f} {r['max_output_tokens_per_s']:>8.0f} "
            f"{r.get('concurrency', 0):>6.0f} "
            f"{r.get('batch_running_p50', 0):>9.0f} "
            f"{r.get('batch_running_p90', 0):>9.0f} "
            f"{r.get('batch_running_max', 0):>9.0f} "
            f"{r.get('p99_ttft_ms', 0):>8.0f} {r.get('mean_tpot_ms', 0):>7.1f}"
        )

    out = base / "summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
