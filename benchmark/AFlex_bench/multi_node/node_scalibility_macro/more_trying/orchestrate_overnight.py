#!/usr/bin/env python3
"""Wait for PDAF deploy sweep, pick best config, run 6-scheme benchmark, plot."""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("orchestrate")

HERE = Path(__file__).resolve().parent
LOG_DIR = HERE.parent.parent / "logs"
RESULTS_DIR = HERE / "results"
POLL_S = 120


def _pgrep(pattern: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True)
    return r.returncode == 0


def wait_sweep():
    log.info("Waiting for pdaf_deploy_sweep to finish...")
    while _pgrep("run_pdaf_deploy_sweep_code.py"):
        log.info("  sweep still running, sleep %ds", POLL_S)
        time.sleep(POLL_S)
    final = sorted(RESULTS_DIR.glob("pdaf_deploy_sweep_code_final_*.json"))
    if not final:
        partial = sorted(RESULTS_DIR.glob("pdaf_deploy_sweep_code_partial_*.json"))
        if partial:
            log.warning("No final sweep JSON; using latest partial: %s", partial[-1].name)
        else:
            raise RuntimeError("No sweep results found")
    log.info("Sweep finished.")


def pick_best():
    sys.path.insert(0, str(HERE))
    import pdaf_deploy_utils as PDU

    key, e = PDU.pick_best_tier_key()
    out = PDU.save_best_config(key, e)
    log.info("Best deploy: %s (%.3f J/tok) -> %s", key, e, out)
    return out


def run_benchmark(best_cfg: Path):
    cmd = (
        f"nohup env MN_NODE1_IP={os.environ.get('MN_NODE1_IP', '10.252.129.34')} "
        f"MN_NODE2_IP={os.environ.get('MN_NODE2_IP', '10.252.129.33')} "
        f"{sys.executable} -u {HERE / 'run_best_pdaf_6scheme_benchmark.py'} "
        f"--best-config {best_cfg} --resume "
        f"> {LOG_DIR / 'best_pdaf_6scheme.log'} 2>&1 < /dev/null &"
    )
    log.info("Starting 6-scheme benchmark (background)")
    subprocess.Popen(cmd, shell=True, cwd=str(HERE))
    # wait for completion
    while _pgrep("run_best_pdaf_6scheme_benchmark.py"):
        time.sleep(120)
    finals = sorted(RESULTS_DIR.glob("best_pdaf_6scheme_final_*.json"))
    if not finals:
        raise RuntimeError("6-scheme benchmark did not produce final json")
    log.info("Benchmark done: %s", finals[-1].name)


def run_plots():
    for script in ("plot_pdaf_deploy_sweep_code.py", "plot_best_pdaf_6scheme.py"):
        p = HERE / script
        if p.exists():
            log.info("Plot: %s", script)
            subprocess.run([sys.executable, str(p)], check=False)


def main():
    wait_sweep()
    best_cfg = pick_best()
    run_benchmark(best_cfg)
    run_plots()
    log.info("Overnight pipeline complete.")


if __name__ == "__main__":
    main()
