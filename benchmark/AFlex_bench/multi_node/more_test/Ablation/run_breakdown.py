#!/usr/bin/env python3
"""Unified entry for Ablation breakdown benchmarks and figures.

Usage:
  python3 run_breakdown.py plot [--moe] [--node] [--tier1] [--all] [--paper]
  python3 run_breakdown.py run tier1-megascale
  python3 run_breakdown.py run tier1-aflex-no-dvfs
  python3 run_breakdown.py prepare tier1-existing
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ABLATION_ROOT = Path(__file__).resolve().parent
MOE_PLOT_SCRIPT = ABLATION_ROOT / "model_scalibility/charts/plot_moe_energy.py"
MOE_RUN_SCRIPT = ABLATION_ROOT / "model_scalibility/scripts/run_moe_retest.py"
NODE_SCALABILITY_SCRIPT = (
    ABLATION_ROOT / "node_scalibility/charts/plot_node_scalability_energy.py"
)
TIER1_PLOT_SCRIPT = ABLATION_ROOT / "Energy_breakdown/charts/plot_tier_perf.py"
TIER1_MEGASCALE_SCRIPT = (
    ABLATION_ROOT / "Energy_breakdown/scripts/run_megascale_fixed_1p1d_tp4.py"
)
TIER1_NO_DVFS_SCRIPT = (
    ABLATION_ROOT / "Energy_breakdown/scripts/run_aflex_tier1_no_dvfs.py"
)
TIER1_PREPARE_SCRIPT = (
    ABLATION_ROOT / "Energy_breakdown/scripts/prepare_existing_data.py"
)


def _run_script(script: Path, *args: str) -> None:
    cmd = [sys.executable, str(script), *args]
    subprocess.run(cmd, check=True, cwd=str(script.parent))


def cmd_plot(args: argparse.Namespace) -> None:
    targets = []
    if args.all or args.moe:
        targets.append(("moe", MOE_PLOT_SCRIPT))
    if args.all or args.node:
        targets.append(("node", NODE_SCALABILITY_SCRIPT))
    if args.all or args.tier1:
        targets.append(("tier1", TIER1_PLOT_SCRIPT))
    if not targets:
        raise SystemExit("specify at least one of --moe, --node, --tier1, or --all")

    extra = ["--paper"] if args.paper else []
    for name, script in targets:
        print(f"==> plot {name}")
        _run_script(script, *extra)


def cmd_run(args: argparse.Namespace) -> None:
    mapping = {
        "tier1-megascale": TIER1_MEGASCALE_SCRIPT,
        "tier1-aflex-no-dvfs": TIER1_NO_DVFS_SCRIPT,
        "moe-retest": MOE_RUN_SCRIPT,
    }
    script = mapping.get(args.target)
    if script is None:
        raise SystemExit(f"unknown run target: {args.target}")
    _run_script(script)


def cmd_prepare(args: argparse.Namespace) -> None:
    if args.target != "tier1-existing":
        raise SystemExit(f"unknown prepare target: {args.target}")
    _run_script(TIER1_PREPARE_SCRIPT)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plot = sub.add_parser("plot", help="regenerate breakdown charts")
    plot.add_argument("--moe", action="store_true")
    plot.add_argument("--node", action="store_true")
    plot.add_argument("--tier1", action="store_true")
    plot.add_argument("--all", action="store_true")
    plot.add_argument(
        "--paper",
        action="store_true",
        help="write paper PDFs (exclude MegaScale where applicable)",
    )
    plot.set_defaults(func=cmd_plot)

    run = sub.add_parser("run", help="run a benchmark")
    run.add_argument(
        "target",
        choices=["tier1-megascale", "tier1-aflex-no-dvfs", "moe-retest"],
    )
    run.set_defaults(func=cmd_run)

    prep = sub.add_parser("prepare", help="prepare frozen data subsets")
    prep.add_argument("target", choices=["tier1-existing"])
    prep.set_defaults(func=cmd_prepare)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
