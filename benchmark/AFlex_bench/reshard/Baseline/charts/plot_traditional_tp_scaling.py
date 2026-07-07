#!/usr/bin/env python3
"""Plot traditional whole-instance SGLang TP scaling results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_INPUT = REPO_ROOT / "benchmark/AFlex_bench/reshard/Baseline/results/traditional_tp_scaling.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "benchmark/AFlex_bench/reshard/Baseline/charts"

PHASE_LABELS = {
    "start_source_tp": "Old TP ready",
    "probe_source_before_scale": "Old TP probe",
    "start_target_tp_and_load_model": "New TP startup + disk load",
    "export_source_ipc_handles_online": "Export IPC handles",
    "start_target_tp_from_ipc_nvlink": "New TP startup + IPC/NVLink",
    "drain_old_instance": "Drain old TP",
    "visible_drain_and_cutover": "Visible drain/cutover",
    "probe_target_after_cutover": "New TP probe",
    "stop_source_tp": "Stop old TP",
}

SAMPLE_RECORD = {
    "scenario": "TP2->TP4 (example)",
    "phases": [
        {"name": "start_source_tp", "duration_s": 0.0},
        {"name": "probe_source_before_scale", "duration_s": 2.0},
        {"name": "start_target_tp_and_load_model", "duration_s": 28.0},
        {"name": "drain_old_instance", "duration_s": 5.0},
        {"name": "probe_target_after_cutover", "duration_s": 2.0},
        {"name": "stop_source_tp", "duration_s": 1.0},
    ],
    "before": {"ttft_avg_ms": 1450.0, "e2e_avg_ms": 1840.0, "success": 4, "requests": 4},
    "after": {"ttft_avg_ms": 1240.0, "e2e_avg_ms": 1610.0, "success": 4, "requests": 4},
}


def load_records(path: Path, allow_sample: bool) -> list[dict[str, Any]]:
    if not path.exists():
        if allow_sample:
            print(f"input {path} not found; using embedded sample data")
            return [SAMPLE_RECORD]
        raise FileNotFoundError(path)
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    if not records:
        if allow_sample:
            return [SAMPLE_RECORD]
        raise ValueError(f"no records in {path}")
    return records


def latest_per_scenario(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_scenario: dict[str, dict[str, Any]] = {}
    for record in records:
        by_scenario[record.get("scenario", "unknown")] = record
    return list(by_scenario.values())


def save(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        path = out_dir / f"{stem}.{suffix}"
        fig.savefig(path, bbox_inches="tight", dpi=200)
        print(f"wrote {path}")


def plot_breakdown(records: list[dict[str, Any]], out_dir: Path) -> None:
    phase_names = [
        "start_target_tp_and_load_model",
        "export_source_ipc_handles_online",
        "start_target_tp_from_ipc_nvlink",
        "drain_old_instance",
        "visible_drain_and_cutover",
        "probe_target_after_cutover",
        "stop_source_tp",
    ]
    scenarios = [
        f"{r.get('scenario', f'run{i}')}\n{r.get('scheme', 'unknown').replace('traditional_sglang_tp_', '').replace('optimized_sglang_tp_', '')}"
        for i, r in enumerate(records)
    ]
    fig, ax = plt.subplots(figsize=(11.5, 3.6))
    lefts = [0.0] * len(records)
    y_positions = list(range(len(records)))
    colors = ["#4C78A8", "#72B7B2", "#4C78A8", "#F58518", "#FF9DA6", "#54A24B", "#B279A2"]

    for idx, phase_name in enumerate(phase_names):
        values = []
        for record in records:
            phase_map = {p["name"]: p.get("duration_s", 0.0) for p in record.get("phases", [])}
            values.append(float(phase_map.get(phase_name, 0.0)))
        ax.barh(
            y_positions,
            values,
            left=lefts,
            height=0.5,
            label=PHASE_LABELS.get(phase_name, phase_name),
            color=colors[idx],
        )
        for y, left, value in zip(y_positions, lefts, values):
            if value >= 1.0:
                ax.text(
                    left + value / 2,
                    y,
                    f"{value:.1f}s",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if value >= 3.0 else "black",
                )
        lefts = [left + value for left, value in zip(lefts, values)]

    ax.set_yticks(y_positions, scenarios)
    ax.invert_yaxis()
    ax.set_xlabel("Time (s)")
    ax.set_title("SGLang TP Scaling Overhead Timeline")
    ax.legend(frameon=False, ncols=3, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.35))
    ax.grid(axis="x", alpha=0.25)
    save(fig, out_dir, "traditional_tp_breakdown")


def plot_latency(records: list[dict[str, Any]], out_dir: Path) -> None:
    scenarios = [f"{r.get('scenario', f'run{i}')}\n{r.get('scheme', 'unknown').replace('traditional_sglang_tp_', '').replace('optimized_sglang_tp_', '')}" for i, r in enumerate(records)]
    before = [r.get("before", {}).get("ttft_avg_ms") or 0.0 for r in records]
    after = [r.get("after", {}).get("ttft_avg_ms") or 0.0 for r in records]
    x = range(len(records))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - width / 2 for i in x], before, width=width, label="Before scale", color="#72B7B2")
    ax.bar([i + width / 2 for i in x], after, width=width, label="After cutover", color="#E45756")
    ax.set_xticks(list(x), scenarios)
    ax.set_ylabel("Average TTFT (ms)")
    ax.set_title("Request Latency Before vs. After Whole-Instance TP Cutover")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir, "traditional_tp_latency")




def plot_downtime(records: list[dict[str, Any]], out_dir: Path) -> None:
    scenarios = [f"{r.get('scenario', f'run{i}')}\n{r.get('scheme', 'unknown').replace('traditional_sglang_tp_', '').replace('optimized_sglang_tp_', '')}" for i, r in enumerate(records)]
    async_prepare = []
    visible = []
    for record in records:
        phase_map = {p["name"]: p.get("duration_s", 0.0) for p in record.get("phases", [])}
        async_prepare.append(float(record.get("async_prepare_s") or phase_map.get("start_target_tp_and_load_model", 0.0)))
        visible.append(float(record.get("visible_downtime_s") or phase_map.get("drain_old_instance", 0.0)))
    x = range(len(records))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar([i - width / 2 for i in x], async_prepare, width=width, label="Async prepare", color="#4C78A8")
    ax.bar([i + width / 2 for i in x], visible, width=width, label="Visible interruption", color="#E45756")
    ax.set_xticks(list(x), scenarios)
    ax.set_ylabel("Time (s)")
    ax.set_title("Async Preparation vs. Visible Interruption")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    save(fig, out_dir, "traditional_tp_downtime")


def main() -> int:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-sample", action="store_true", help="Fail if no input records are available")
    args = parser.parse_args()
    records = latest_per_scenario(load_records(args.input, allow_sample=not args.no_sample))
    plot_breakdown(records, args.out_dir)
    plot_latency(records, args.out_dir)
    plot_downtime(records, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
