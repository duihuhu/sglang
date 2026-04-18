#!/usr/bin/env python3
"""
Convert per-run bench CSV artifacts into pd_latency_big_table long format.

Input files (under each tp*_in*_clk*_bs*_ol*/processed directory):
  - *_af_ttft_af.csv
  - *_ttft_ttft_af.csv

Output schema matches `pivot_pd_ops_wide.py` expectation:
  tp,input_len,output_len,gpu_clock,batch_size,stage,op_name,count,latency_us,energy_uj
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
from typing import Dict, List, Optional, Tuple

_ALLOWED_OPS = ("TTFT", "TPOT", "A", "F")


def _parse_run_dir_name(run_dir_name: str) -> Tuple[int, int, int, int, int]:
    # Example: tp1_in1023_clk1170_bs16_ol512
    parts = run_dir_name.split("_")
    if len(parts) < 3:
        raise RuntimeError(f"Cannot parse run metadata from dir: {run_dir_name}")
    tp = int(parts[0].replace("tp", ""))
    input_len = int(parts[1].replace("in", ""))
    gpu_clock = int(parts[2].replace("clk", ""))
    batch_size = 1
    target_output_len = 1
    for part in parts[3:]:
        if part.startswith("bs"):
            batch_size = int(part.replace("bs", ""))
        elif part.startswith("ol"):
            target_output_len = int(part.replace("ol", ""))
    return tp, input_len, gpu_clock, batch_size, target_output_len


def _to_float(v: object) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _to_int(v: object) -> Optional[int]:
    f = _to_float(v)
    if f is None:
        return None
    return int(f)


def _collect_rows_from_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _accumulate_metric(
    *,
    agg: Dict[str, Dict[str, float]],
    op_name: str,
    latency_avg: Optional[float],
    count: Optional[int],
    energy_avg: Optional[float],
    energy_count: Optional[int],
) -> None:
    if op_name not in _ALLOWED_OPS:
        return
    if latency_avg is None or count is None or count <= 0 or latency_avg < 0:
        return
    slot = agg.setdefault(
        op_name,
        {
            "latency_weighted_sum": 0.0,
            "count_sum": 0.0,
            "energy_weighted_sum": 0.0,
            "energy_count_sum": 0.0,
        },
    )
    slot["latency_weighted_sum"] += latency_avg * count
    slot["count_sum"] += count

    if (
        energy_avg is not None
        and energy_count is not None
        and energy_count > 0
        and energy_avg >= 0
    ):
        slot["energy_weighted_sum"] += energy_avg * energy_count
        slot["energy_count_sum"] += energy_count


def _aggregate_one_run(
    *,
    processed_dir: str,
    output_lens: List[int],
    bench_stage: str,
) -> List[Dict[str, object]]:
    run_dir = os.path.dirname(processed_dir)
    tp, input_len, gpu_clock, batch_size, target_output_len = _parse_run_dir_name(
        os.path.basename(run_dir)
    )

    input_len_display = input_len
    if bench_stage == "D":
        # Align with existing sync-op dump conversion behavior.
        input_len_display = input_len - target_output_len + 1

    csv_paths = sorted(glob.glob(os.path.join(processed_dir, "*_af_ttft_af.csv")))
    csv_paths += sorted(glob.glob(os.path.join(processed_dir, "*_ttft_ttft_af.csv")))
    if not csv_paths:
        return []

    agg: Dict[str, Dict[str, float]] = {}
    for p in csv_paths:
        for row in _collect_rows_from_csv(p):
            _accumulate_metric(
                agg=agg,
                op_name="TTFT",
                latency_avg=_to_float(row.get("ttft_avg_us")),
                count=_to_int(row.get("ttft_count")),
                energy_avg=None,
                energy_count=None,
            )
            _accumulate_metric(
                agg=agg,
                op_name="TPOT",
                latency_avg=_to_float(row.get("tpot_avg_us")),
                count=_to_int(row.get("tpot_count")),
                energy_avg=None,
                energy_count=None,
            )
            _accumulate_metric(
                agg=agg,
                op_name="A",
                latency_avg=_to_float(row.get("a_avg_us")),
                count=_to_int(row.get("a_count")),
                energy_avg=_to_float(row.get("a_avg_energy_uj")),
                energy_count=_to_int(row.get("a_energy_count")),
            )
            _accumulate_metric(
                agg=agg,
                op_name="F",
                latency_avg=_to_float(row.get("f_avg_us")),
                count=_to_int(row.get("f_count")),
                energy_avg=_to_float(row.get("f_avg_energy_uj")),
                energy_count=_to_int(row.get("f_energy_count")),
            )

    if not agg:
        return []

    out_rows: List[Dict[str, object]] = []
    output_len_values = [target_output_len] if target_output_len > 1 else output_lens

    for op_name in _ALLOWED_OPS:
        slot = agg.get(op_name)
        if not slot or slot["count_sum"] <= 0:
            continue
        latency_us = slot["latency_weighted_sum"] / slot["count_sum"]
        energy_uj = None
        if slot["energy_count_sum"] > 0:
            energy_uj = slot["energy_weighted_sum"] / slot["energy_count_sum"]
        for out_len_val in output_len_values:
            out_rows.append(
                {
                    "tp": tp,
                    "input_len": input_len_display,
                    "output_len": out_len_val,
                    "gpu_clock": gpu_clock,
                    "batch_size": batch_size,
                    "stage": bench_stage,
                    "op_name": op_name,
                    "count": int(slot["count_sum"]),
                    "latency_us": latency_us,
                    "energy_uj": energy_uj,
                }
            )
    return out_rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert per-run AF/TTFT CSV artifacts into pd_latency_big_table format."
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="bash-test/pd_batch_work_d",
        help="Directory containing tp*_in*_clk*_bs*_ol*/processed/*.csv artifacts.",
    )
    parser.add_argument(
        "--output-lens",
        type=str,
        default="64,256,512",
        help="Comma-separated output lengths (used when run dir has no ol>1 suffix).",
    )
    parser.add_argument(
        "--bench-stage",
        type=str,
        default="D",
        choices=["P", "D", "p", "d"],
        help="Stage label to write into final big table.",
    )
    parser.add_argument(
        "--final-csv",
        type=str,
        default="bash-test/pd_latency_big_table_from_csv.csv",
        help="Output long-format CSV path.",
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    output_lens = [int(x.strip()) for x in args.output_lens.split(",") if x.strip()]
    bench_stage = str(args.bench_stage).strip().upper()
    if not output_lens:
        raise ValueError("--output-lens must not be empty")

    processed_dirs = sorted(glob.glob(os.path.join(work_dir, "tp*", "processed")))
    if not processed_dirs:
        raise RuntimeError(f"No processed dirs found under: {work_dir}")

    all_rows: List[Dict[str, object]] = []
    skipped_no_csv = 0
    skipped_no_metrics = 0
    for processed_dir in processed_dirs:
        has_source_csv = bool(glob.glob(os.path.join(processed_dir, "*_af_ttft_af.csv"))) or bool(
            glob.glob(os.path.join(processed_dir, "*_ttft_ttft_af.csv"))
        )
        if not has_source_csv:
            skipped_no_csv += 1
            continue
        run_rows = _aggregate_one_run(
            processed_dir=processed_dir,
            output_lens=output_lens,
            bench_stage=bench_stage,
        )
        if not run_rows:
            skipped_no_metrics += 1
            continue
        all_rows.extend(run_rows)

    if not all_rows:
        raise RuntimeError(
            "No rows exported. Source CSV files may be missing metrics "
            "or all counts are zero."
        )

    out_csv_path = os.path.abspath(args.final_csv)
    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    with open(out_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "tp",
            "input_len",
            "output_len",
            "gpu_clock",
            "batch_size",
            "stage",
            "op_name",
            "count",
            "latency_us",
            "energy_uj",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)

    print(f"[saved] {out_csv_path}")
    print(f"[info] rows: {len(all_rows)}")
    print(f"[info] skipped dirs without source csv: {skipped_no_csv}")
    print(f"[info] skipped dirs without valid metrics: {skipped_no_metrics}")


if __name__ == "__main__":
    main()
