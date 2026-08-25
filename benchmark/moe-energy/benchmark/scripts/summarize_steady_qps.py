#!/usr/bin/env python3
"""Summarize steady-state QPS benchmark results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

def load(path: Path) -> dict:
    text = path.read_text().strip()
    if not text:
        raise ValueError(f"empty file: {path}")
    return json.loads(text.splitlines()[-1])


def iter_result_files(base: Path):
  """Support layout: base/qps30/result.jsonl or base/attn_dp_moe_ep/qps30.jsonl."""
  # parallel same-config: base/qps{N}/result.jsonl
  for d in sorted(base.glob("qps*"), key=lambda p: int(p.name[3:])):
    p = d / "result.jsonl"
    if p.is_file():
      yield int(d.name[3:]), p, "Attn DP + MoE EP"
  # legacy: base/<config>/qps{N}.jsonl
  for key, label in [
      ("attn_dp_moe_ep", "Attn DP + MoE EP"),
      ("attn_tp_moe_tp", "Attn TP + MoE TP"),
      ("attn_tp_moe_ep", "Attn TP + MoE EP"),
      ("attn_dp_moe_tp", "Attn DP + MoE TP"),
  ]:
    d = base / key
    if not d.is_dir():
      continue
    for p in sorted(d.glob("qps*.jsonl"), key=lambda x: int(x.stem[3:])):
      yield int(p.stem[3:]), p, label


def main() -> None:
    base = Path(sys.argv[1])
    rows = []
    for qps, p, label in iter_result_files(base):
        j = load(p)
        rows.append({
            "config": label,
            "label": label,
            "request_rate": qps,
            "completed": j.get("completed"),
            "num_prompts": j.get("max_concurrency"),
            "output_throughput": j.get("output_throughput"),
            "max_output_tokens_per_s": j.get("max_output_tokens_per_s", 0),
            "concurrency": j.get("concurrency"),
            "max_concurrent_requests": j.get("max_concurrent_requests", 0),
            "mean_ttft_ms": j.get("mean_ttft_ms"),
            "p99_ttft_ms": j.get("p99_ttft_ms"),
            "mean_tpot_ms": j.get("mean_tpot_ms"),
            "mean_e2e_latency_ms": j.get("mean_e2e_latency_ms"),
            "duration_s": j.get("duration"),
        })

    if not rows:
        print(f"No results under {base}")
        return

    print("\n=== 稳态 QPS 压测汇总 ===")
    print(f"{'Config':<22} {'QPS':>4} {'done':>5} {'avg_out':>8} {'peak_out':>8} "
          f"{'conc':>6} {'max_conc':>8} {'ttft99':>8} {'tpot':>7} {'dur_s':>7}")
    print("-" * 95)
    for r in rows:
        print(
            f"{r['label']:<22} {r['request_rate']:>4} {r['completed']:>5} "
            f"{r['output_throughput']:>8.0f} {r['max_output_tokens_per_s']:>8.0f} "
            f"{r['concurrency']:>6.0f} {r['max_concurrent_requests']:>8} "
            f"{r['p99_ttft_ms']:>8.0f} {r['mean_tpot_ms']:>7.1f} {r['duration_s']:>7.1f}"
        )

    out = base / "summary.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
