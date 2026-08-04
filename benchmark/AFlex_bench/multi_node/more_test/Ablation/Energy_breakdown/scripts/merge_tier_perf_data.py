#!/usr/bin/env python3
"""Merge Tier1 energy breakdown results into a single JSON file."""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MACRO_E2E = ROOT.parents[1] / "macro/data/macro_e2e_all.json"
DEFAULT_VANILLA = ROOT / "data/vanilla_1p1d_tp4_no_dvfs.json"
DEFAULT_NO_DVFS = ROOT / "data/aflex_e2e_no_dvfs.json"
DEFAULT_OUTPUT = ROOT / "data/tier_perf_energy_all.json"

DATASETS = ("code", "conv")
QPS_LIST = (2, 4, 8, 16)
SERIES = (
    ("megascale", "Vanilla", DEFAULT_VANILLA),
    ("aflex_no_dvfs", "+Scheduler", DEFAULT_NO_DVFS),
    ("aflex_dvfs", "AFlex", None),
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _macro_aflex_points(macro: dict) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for dataset in DATASETS:
        for qps in QPS_LIST:
            key = f"{dataset}_qps{qps}"
            out[key] = deepcopy(macro["results"][dataset][f"qps_{qps}"]["aflex_tier1"])
    return out


def merge(
    vanilla_path: Path,
    no_dvfs_path: Path,
    macro_path: Path,
) -> dict:
    vanilla = _load(vanilla_path)
    no_dvfs = _load(no_dvfs_path)
    macro = _load(macro_path)
    aflex_dvfs = _macro_aflex_points(macro)

    return {
        "meta": {
            "description": (
                "Tier1 energy breakdown: Vanilla (1P+1D TP4, no DVFS), "
                "+Scheduler (macro e2e AFlex topology, no DVFS), "
                "and AFlex (macro e2e AFlex topology, with DVFS)."
            ),
            "datasets": list(DATASETS),
            "qps": list(QPS_LIST),
            "series": {key: label for key, label, _ in SERIES},
            "series_labels": {key: label for key, label, _ in SERIES},
            "plot_energy_metric": {
                "megascale": "energy_per_token_j_all_tokens",
                "aflex_no_dvfs": "energy_per_token_j_all_tokens",
                "aflex_dvfs": "energy_per_token_mj_over_1000",
            },
            "sources": {
                "megascale": str(vanilla_path),
                "aflex_no_dvfs": str(no_dvfs_path),
                "aflex_dvfs": str(macro_path),
            },
            "ttft_slo_ms": 2000.0,
            "tpot_slo_ms": 100.0,
            "nodes": {
                "node3": "10.252.129.34",
                "node4": "10.252.129.33",
            },
        },
        "results": {
            "megascale": vanilla["results"],
            "aflex_no_dvfs": no_dvfs["results"],
            "aflex_dvfs": aflex_dvfs,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vanilla", type=Path, default=DEFAULT_VANILLA)
    parser.add_argument("--no-dvfs", type=Path, default=DEFAULT_NO_DVFS)
    parser.add_argument("--macro", type=Path, default=MACRO_E2E)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    payload = merge(args.vanilla, args.no_dvfs, args.macro)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {args.output}")
    for scheme in payload["results"]:
        n = len(payload["results"][scheme])
        rr = sum(
            len(pt.get("request_results", []))
            for pt in payload["results"][scheme].values()
            if isinstance(pt, dict)
        )
        print(f"  {scheme}: {n} points, {rr} request_results rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
