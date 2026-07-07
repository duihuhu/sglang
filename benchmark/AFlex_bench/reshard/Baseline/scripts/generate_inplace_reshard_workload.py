#!/usr/bin/env python3
"""Build a long, multi-QPS workload for in-place reshard SLO testing.

QPS rises with expected TP (more parallelism -> higher offered load).
Outputs JSONL plus a suggested sequential reshard plan.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# (phase_duration_s, target_qps, label)
DEFAULT_PHASES = [
    (45.0, 1.5, "tp1"),
    (40.0, 3.0, "tp2"),
    (40.0, 5.5, "tp4"),
    (45.0, 8.0, "tp8"),
]


def load_seed_rows(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["arrival_time_s"])
    return rows


def generate(
    seed_rows: list[dict],
    phases: list[tuple[float, float, str]],
    jitter_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    out: list[dict] = []
    reshard_plan: list[dict] = []
    t = 0.0
    new_tps = [2, 4, 8]

    for phase_idx, (duration, qps, label) in enumerate(phases):
        if phase_idx > 0:
            pre_drain = max(4.0, min(12.0, 8.0 * qps / 5.5))
            reshard_plan.append(
                {
                    "at_s": round(t, 2),
                    "new_tp": new_tps[phase_idx - 1],
                    "pre_drain_sec": round(pre_drain, 1),
                }
            )
        interval = 1.0 / qps
        phase_end = t + duration
        i = 0
        while t < phase_end:
            src = seed_rows[i % len(seed_rows)]
            i += 1
            jitter = 1.0 + rng.uniform(-jitter_frac, jitter_frac)
            row = {
                "input_len": int(src["input_len"]),
                "output_len": int(src["output_len"]),
                "arrival_time_s": round(t, 4),
                "phase": label,
                "target_qps": qps,
            }
            out.append(row)
            t += interval * jitter

    return out, reshard_plan


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-workload", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plan-out", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    seed_rows = load_seed_rows(args.seed_workload)
    rows, plan = generate(seed_rows, DEFAULT_PHASES, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    span = rows[-1]["arrival_time_s"] - rows[0]["arrival_time_s"]
    print(f"wrote {len(rows)} requests, span={span:.1f}s -> {args.output}")
    for label in ["tp1", "tp2", "tp4", "tp8"]:
        n = sum(1 for r in rows if r["phase"] == label)
        qps = next(p[1] for p in DEFAULT_PHASES if p[2] == label)
        print(f"  {label}: {n} reqs target_qps={qps}")

    plan_path = args.plan_out or args.output.with_suffix(".reshard_plan.json")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"reshard plan -> {plan_path}")
    print(json.dumps(plan))


if __name__ == "__main__":
    main()
