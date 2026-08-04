#!/usr/bin/env python3
"""Run fixed 1P+1D TP4 at 1410 MHz with runtime DVFS disabled (Vanilla)."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("run_vanilla_1p1d_tp4_no_dvfs")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench_no_dvfs_common as BNC
import bench_tier1_v2 as BT2

OUTPUT = HERE.parent / "data" / "vanilla_1p1d_tp4_no_dvfs.json"
BENCHMARK_NAME = "vanilla_1p1d_tp4_no_dvfs"
TOPOLOGY_SOURCE = "megascale_fixed_1p1d_tp4"
FIXED_TOPOLOGY = {
    "k_p": 1,
    "k_d": 1,
    "tp_pa": 4,
    "tp_pf": 4,
    "tp_da": 4,
    "tp_df": 4,
}


def initial_payload() -> dict:
    return {
        "meta": {
            "benchmark": BENCHMARK_NAME,
            "description": (
                "MegaScale fixed 1P+1D TP4 topology at fixed 1410 MHz with runtime DVFS disabled."
            ),
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


def load_config(dataset: str, qps: int) -> BT2.Tier1TestConfig:
    return BT2.Tier1TestConfig(
        name=f"vanilla_1p1d_tp4_no_dvfs_{dataset}_qps{qps}",
        **FIXED_TOPOLOGY,
        f_pa=BNC.LOCKED_FREQ_MHZ,
        f_pf=BNC.LOCKED_FREQ_MHZ,
        f_da=BNC.LOCKED_FREQ_MHZ,
        f_df=BNC.LOCKED_FREQ_MHZ,
        tier=False,
    )


def validate_fixed_allocation(allocation: dict, gpu_map: dict[str, list[int]]) -> None:
    p_hosts = [pair[0] for pair in allocation["p_pairs"]]
    d_hosts = [instance["host"] for instance in allocation["decode_instances"]]
    expected_gpu_map = {BNC.NODE3: list(range(8)), BNC.NODE4: list(range(8))}
    if p_hosts != [BNC.NODE4] or d_hosts != [BNC.NODE3] or gpu_map != expected_gpu_map:
        raise RuntimeError(
            "Vanilla fixed_1p1d_tp4 must place the complete D pair on node3 and the "
            f"complete P pair on node4; got p_hosts={p_hosts}, "
            f"d_hosts={d_hosts}, allocated_gpus={gpu_map}"
        )
    if allocation.get("total_gpu") != 16:
        raise RuntimeError(
            f"Vanilla fixed_1p1d_tp4 must allocate 16 GPUs; got {allocation.get('total_gpu')}"
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
        validate_allocation=validate_fixed_allocation,
    )
    log.info("Finished: %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
