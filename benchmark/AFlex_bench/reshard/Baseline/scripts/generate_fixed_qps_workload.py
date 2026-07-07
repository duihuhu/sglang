#!/usr/bin/env python3
"""Generate fixed-QPS workload + TP1->8 reshard plan for timeline benchmarks."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

# B-round style (low QPS)
PLAN_QPS1 = [
    {"at_s": 30.0, "new_tp": 2, "pre_drain_sec": 4.0},
    {"at_s": 60.0, "new_tp": 4, "pre_drain_sec": 6.0},
    {"at_s": 75.0, "new_tp": 8, "pre_drain_sec": 8.0},
]

# ~3 QPS (C-round style): compress timeline
PLAN_QPS3 = [
    {"at_s": 10.0, "new_tp": 2, "pre_drain_sec": 6.0},
    {"at_s": 32.0, "new_tp": 4, "pre_drain_sec": 8.0},
    {"at_s": 48.0, "new_tp": 8, "pre_drain_sec": 10.0},
]


def load_seed(path: Path) -> list[dict]:
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: r["arrival_time_s"])
    return rows


def generate(
    seed_rows: list[dict],
    qps: float,
    duration_s: float,
    label: str,
    *,
    seed: int = 42,
    jitter_frac: float = 0.12,
) -> list[dict]:
    rng = random.Random(seed)
    out: list[dict] = []
    t = 0.0
    i = 0
    interval = 1.0 / qps
    while t < duration_s:
        src = seed_rows[i % len(seed_rows)]
        i += 1
        jitter = 1.0 + rng.uniform(-jitter_frac, jitter_frac)
        out.append(
            {
                "input_len": int(src["input_len"]),
                "output_len": int(src["output_len"]),
                "arrival_time_s": round(t, 4),
                "phase": label,
                "target_qps": qps,
            }
        )
        t += interval * jitter
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-workload", type=Path, required=True)
    ap.add_argument("--qps", type=float, required=True)
    ap.add_argument("--duration-s", type=float, default=90.0)
    ap.add_argument("--label", default="mixed")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--plan-out", type=Path, default=None)
    ap.add_argument(
        "--plan-style",
        choices=("qps1", "qps3"),
        default="qps1",
        help="qps1=@30/60/75; qps3=@10/32/48 (for ~3 QPS workloads)",
    )
    args = ap.parse_args()

    seed_rows = load_seed(args.seed_workload)
    rows = generate(seed_rows, args.qps, args.duration_s, args.label)
    plan = PLAN_QPS3 if args.plan_style == "qps3" else PLAN_QPS1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    plan_path = args.plan_out or args.output.with_suffix(".reshard_plan.json")
    plan_path.write_text(json.dumps(plan, indent=2) + "\n")

    span = rows[-1]["arrival_time_s"] - rows[0]["arrival_time_s"]
    print(f"wrote {len(rows)} reqs span={span:.1f}s qps={args.qps} -> {args.output}")
    print(f"plan -> {plan_path}")


if __name__ == "__main__":
    main()
