#!/usr/bin/env python3
"""Run no-DVFS benchmarks, merge results, and replot tier_perf_energy.pdf."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-aflex", action="store_true")
    parser.add_argument("--skip-vanilla", action="store_true")
    parser.add_argument("--skip-merge", action="store_true")
    parser.add_argument("--skip-plot", action="store_true")
    args = parser.parse_args()

    resume = ["--resume"] if args.resume else []
    steps: list[list[str]] = []
    if not args.skip_aflex:
        steps.append([sys.executable, str(HERE / "run_aflex_e2e_no_dvfs.py"), *resume])
    if not args.skip_vanilla:
        steps.append([sys.executable, str(HERE / "run_vanilla_1p1d_tp4_no_dvfs.py"), *resume])
    if not args.skip_merge:
        steps.append([sys.executable, str(HERE / "merge_tier_perf_data.py")])
    if not args.skip_plot:
        steps.append([sys.executable, str(ROOT / "charts/plot_tier_perf.py")])

    for cmd in steps:
        print("RUN:", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
