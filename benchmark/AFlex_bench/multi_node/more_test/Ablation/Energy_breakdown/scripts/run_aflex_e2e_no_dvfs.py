#!/usr/bin/env python3
"""Run macro-e2e AFlex topologies at 1410 MHz with runtime DVFS disabled (Vanilla+G)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_aflex_e2e_no_dvfs")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_no_dvfs_common as BNC
import macro_e2e_layouts as MEL

OUTPUT = HERE.parent / "data" / "aflex_e2e_no_dvfs.json"
BENCHMARK_NAME = "aflex_e2e_no_dvfs"
TOPOLOGY_SOURCE = "macro_e2e_all.json.aflex_tier1"


def initial_payload() -> dict:
    return {
        "meta": {
            "benchmark": BENCHMARK_NAME,
            "description": (
                "AFlex macro end-to-end topologies at fixed 1410 MHz with runtime DVFS disabled."
            ),
            "source_file": str(MEL.E2E_JSON),
            "node3": BNC.NODE3,
            "node4": BNC.NODE4,
            "datasets": list(BNC.DATASETS),
            "qps": list(BNC.QPS_LIST),
            "ttft_slo_ms": BNC.RMB.TTFT_SLO_MS,
            "tpot_slo_ms": BNC.RMB.TPOT_SLO_MS,
            "topology_source": TOPOLOGY_SOURCE,
            "dvfs": False,
            "locked_freq_mhz": BNC.LOCKED_FREQ_MHZ,
        },
        "results": {},
    }


def load_config(dataset: str, qps: int):
    return MEL.to_tier1_test_config(
        dataset,
        qps,
        locked_freq_mhz=BNC.LOCKED_FREQ_MHZ,
        tier=False,
        name_prefix="aflex_e2e_no_dvfs_",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default="code,conv")
    parser.add_argument("--qps-list", default="2,4,8,16")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    datasets = BNC.parse_csv(args.datasets, BNC.DATASETS)
    qps_values = BNC.parse_csv(args.qps_list, BNC.QPS_LIST, int)
    BNC.run_benchmark_loop(
        output=args.output,
        benchmark_name=BENCHMARK_NAME,
        topology_source=TOPOLOGY_SOURCE,
        load_config=load_config,
        initial_payload=initial_payload,
        datasets=datasets,
        qps_values=qps_values,
        resume=args.resume,
    )
    log.info("Finished: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
