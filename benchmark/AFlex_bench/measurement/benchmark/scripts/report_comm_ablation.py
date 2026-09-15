#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from aflex_benchmark.comm_ablation_report import build_report, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate communication-ablation results")
    parser.add_argument("--qps2", type=Path,
                        default=ROOT / "results" / "comm_ablation_qps2")
    parser.add_argument("--smoke", type=Path,
                        default=ROOT / "results" / "comm_ablation_smoke")
    parser.add_argument("--output-dir", type=Path,
                        default=ROOT / "results" / "comm_ablation_report")
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="write partial output instead of failing on missing repeats")
    args = parser.parse_args()
    report = build_report(args.qps2, args.smoke,
                          strict_repeats=not args.allow_incomplete)
    json_path, markdown_path = write_report(report, args.output_dir)
    print(json_path)
    print(markdown_path)


if __name__ == "__main__":
    main()
