#!/usr/bin/env python3
"""Plot 6-scheme clean benchmark results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from plot_7scheme_6dataset import (
    CHARTS_DIR,
    DATASET_TITLES,
    _load,
    plot_dataset_dashboard_4panel,
    plot_energy_by_dataset,
)

PLOT_SERIES = [
    ("native_tp1_baseline", "SGLang", "#9E9E9E", "-", "o"),
    ("native_tp1_tier", "DynamoLLM", "#607D8B", "--", "s"),
    ("pd_hetero_baseline", "DistServe", "#FF9800", "-", "^"),
    ("pd_hetero_tier_biscale", "BiScale", "#FFC107", "--", "v"),
    ("megascale_tier1", "MegaScale", "#2196F3", "-", "D"),
    ("aflex_tier1", "AFlex", "#4CAF50", "--", "*"),
]

SUBTITLE = (
    "SGLang/DynamoLLM: TP1×16 | DistServe/BiScale: PD hetero xnode | "
    "MegaScale/AFlex: Tier1 ILP layout per (dataset,QPS); MS=1410MHz, AFlex=DVFS"
)

SAVING_BASELINES = [
    ("DistServe", "pd_hetero_baseline"),
    ("DynamoLLM", "native_tp1_tier"),
    ("BiScale", "pd_hetero_tier_biscale"),
    ("MegaScale", "megascale_tier1"),
]


def _load_6scheme(path: Path | None) -> tuple[dict, dict]:
    if path is None:
        d = Path(__file__).resolve().parent / "results"
        for pattern in (
            "6scheme_6dataset_final_*.json",
            "6scheme_6dataset_dataset_*.json",
            "6scheme_6dataset_partial_*.json",
        ):
            files = sorted(d.glob(pattern))
            if files:
                path = files[-1]
                break
        if path is None:
            raise FileNotFoundError("No 6scheme_6dataset results")
    print(f"Loading {path.name}")
    payload = json.loads(path.read_text())
    return payload.get("results", payload), payload.get("meta", {})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--dataset", default=None, help="Single dataset dashboard")
    parser.add_argument("--all", action="store_true", help="Plot all datasets with data")
    parser.add_argument("--overview", action="store_true")
    args = parser.parse_args()

    data, meta = _load_6scheme(args.input)
    datasets = meta.get("datasets", list(DATASET_TITLES))

    targets = [args.dataset] if args.dataset else datasets
    if not args.dataset and not args.all:
        # default: plot any dataset that has AFlex passes
        targets = [
            ds for ds in datasets
            if any(
                data.get("aflex_tier1", {}).get(f"{ds}_qps{q}", {}).get("status") == "PASS"
                for q in meta.get("qps", [2, 8, 16])
            )
        ] or ["code"]

    for ds in targets:
        out = CHARTS_DIR / f"{ds}_6scheme_dashboard_4panel.png"
        import plot_7scheme_6dataset as p7
        old_sb, old_ak = p7.SAVING_BASELINES, p7.AFLEX_SCHEME_KEYS
        p7.SAVING_BASELINES = SAVING_BASELINES
        p7.AFLEX_SCHEME_KEYS = ("aflex_tier1",)
        plot_dataset_dashboard_4panel(
            data, meta, ds, out,
            plot_series=PLOT_SERIES,
            scheme_count=6,
            subtitle=SUBTITLE,
            annotate_savings=True,
        )
        p7.SAVING_BASELINES, p7.AFLEX_SCHEME_KEYS = old_sb, old_ak

    if args.overview:
        # patch SAVING_BASELINES in plot - use aflex_tier1
        import plot_7scheme_6dataset as p7
        old = p7.SAVING_BASELINES, p7.AFLEX_SCHEME_KEYS
        p7.SAVING_BASELINES = SAVING_BASELINES
        p7.AFLEX_SCHEME_KEYS = ("aflex_tier1",)
        plot_energy_by_dataset(data, meta, CHARTS_DIR / "6scheme_6dataset_energy.png")
        p7.SAVING_BASELINES, p7.AFLEX_SCHEME_KEYS = old


if __name__ == "__main__":
    main()
