#!/usr/bin/env python3
"""Regenerate the paper dashboards while excluding the MegaScale series."""
from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Iterator

ABLATION_ROOT = Path(__file__).resolve().parent
AFLEX_ROOT = ABLATION_ROOT.parents[2]
MORE_TEST = AFLEX_ROOT / "multi_node/more_test"

MACRO_SCRIPT = MORE_TEST / "macro/plot_e2e_dashboard.py"
MACRO_INPUT = MORE_TEST / "macro/data/plan_dense_e2e.json"
MACRO_OUTPUT = MORE_TEST / "macro/charts"
MICRO_SCRIPT = MORE_TEST / "micro/plot_micro_dashboard.py"
MICRO_INPUT = MORE_TEST / "micro/data/micro_4ds_e2e.json"
MICRO_OUTPUT = MORE_TEST / "micro/charts"
NODE_ENERGY_SCRIPT = (
    ABLATION_ROOT / "node_scalibility/charts/plot_node_scalability_energy.py"
)
MOE_ENERGY_SCRIPT = ABLATION_ROOT / "model_scalibility/charts/plot_moe_energy.py"
TIER_ENERGY_SCRIPT = ABLATION_ROOT / "Energy_breakdown/charts/plot_tier_perf.py"

MACRO_PLOT_SERIES = [
    ("native_tp1_baseline", "SGLang", "#1f77b4", "-", "o", 1.8, 5),
    ("native_tp1_tier", "DynamoLLM", "#aec7e8", "--", "s", 1.8, 5),
    ("pd_hetero_baseline", "DistServe", "#ff7f0e", "-", "^", 1.8, 5),
    ("pd_hetero_tier_biscale", "BiScale", "#ffbb78", "--", "v", 1.8, 5),
    ("aflex_tier1", "AFlex", "#d62728", "-", "*", 2.6, 8),
]


def _load_module(name: str, path: Path) -> ModuleType:
    """Load an original plotting script without changing its source."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import plotting module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_results(path: Path) -> dict:
    print(f"Loading {path}")
    with path.open(encoding="utf-8") as file:
        return json.load(file)["results"]


@contextmanager
def _without_series(module: ModuleType, excluded_key: str) -> Iterator[None]:
    """Temporarily filter one series and restore module state afterwards."""
    original = module.PLOT_SERIES
    module.PLOT_SERIES = [series for series in original if series[0] != excluded_key]
    try:
        yield
    finally:
        module.PLOT_SERIES = original


@contextmanager
def _temporary_module_attribute(
    module: ModuleType, attribute: str, value: object
) -> Iterator[None]:
    """Temporarily override one module attribute and restore it afterwards."""
    original = getattr(module, attribute)
    setattr(module, attribute, value)
    try:
        yield
    finally:
        setattr(module, attribute, original)


def plot_macro(input_path: Path, output_dir: Path) -> None:
    module = _load_module("aflex_original_macro_plot", MACRO_SCRIPT)
    data = _load_results(input_path)
    with _temporary_module_attribute(module, "PLOT_SERIES", MACRO_PLOT_SERIES):
        with _temporary_module_attribute(module, "WSPACE_BC", 0.21):
            for dataset, _title, trace_suffix, output_stem in module.DATASETS:
                module.plot_dataset_dashboard_3panel(
                    data,
                    dataset,
                    trace_suffix,
                    output_dir / output_stem,
                    ttft_percentile="p90",
                    tpot_percentile="p90",
                )


def plot_micro(input_path: Path, output_dir: Path) -> None:
    module = _load_module("aflex_original_micro_plot", MICRO_SCRIPT)
    data = _load_results(input_path)
    with _without_series(module, "megascale"):
        for dataset, title, subtitle in module.DATASETS:
            wspace_bc = 0.21 if dataset == "qa_lpld" else 0.18
            with _temporary_module_attribute(module, "WSPACE_BC", wspace_bc):
                module.plot_dataset_dashboard(
                    data,
                    dataset,
                    title,
                    subtitle,
                    output_dir / f"micro_{dataset}",
                    variant="main",
                    ttft_percentile="p90",
                    tpot_percentile="p90",
                )


def plot_node_scalability_energy() -> None:
    subprocess.run(
        [sys.executable, str(NODE_ENERGY_SCRIPT), "--paper"],
        check=True,
        cwd=str(NODE_ENERGY_SCRIPT.parent),
    )


def plot_moe_energy() -> None:
    subprocess.run(
        [sys.executable, str(MOE_ENERGY_SCRIPT), "--paper"],
        check=True,
        cwd=str(MOE_ENERGY_SCRIPT.parent),
    )


def plot_tier1_energy() -> None:
    subprocess.run(
        [sys.executable, str(TIER_ENERGY_SCRIPT), "--paper"],
        check=True,
        cwd=str(TIER_ENERGY_SCRIPT.parent),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--macro-input", type=Path, default=MACRO_INPUT)
    parser.add_argument("--micro-input", type=Path, default=MICRO_INPUT)
    parser.add_argument("--macro-output", type=Path, default=MACRO_OUTPUT)
    parser.add_argument("--micro-output", type=Path, default=MICRO_OUTPUT)
    args = parser.parse_args()

    plot_macro(args.macro_input.resolve(), args.macro_output.resolve())
    plot_micro(args.micro_input.resolve(), args.micro_output.resolve())
    plot_node_scalability_energy()
    plot_moe_energy()
    plot_tier1_energy()


if __name__ == "__main__":
    main()
