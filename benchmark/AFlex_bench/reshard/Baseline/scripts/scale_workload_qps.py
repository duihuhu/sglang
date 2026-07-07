#!/usr/bin/env python3
"""Scale workload arrival times to increase effective QPS."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument(
        "--scale",
        type=float,
        default=0.25,
        help="Multiply arrival_time_s by this factor (<1 speeds up = higher QPS)",
    )
    args = ap.parse_args()
    rows = [json.loads(l) for l in args.input.read_text().splitlines() if l.strip()]
    for r in rows:
        r["arrival_time_s"] = round(float(r["arrival_time_s"]) * args.scale, 4)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    span = rows[-1]["arrival_time_s"] - rows[0]["arrival_time_s"]
    qps = len(rows) / span if span > 0 else 0
    print(f"wrote {len(rows)} requests, span={span:.2f}s, ~{qps:.1f} req/s -> {args.output}")


if __name__ == "__main__":
    main()
