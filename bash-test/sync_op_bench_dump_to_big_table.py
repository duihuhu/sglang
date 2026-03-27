#!/workspace/env/sglang-main/bin/python
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


_D_POS_NAME_RE = re.compile(r"^(D_[A-Za-z0-9_]+)_pos(-?\d+)$")


def _parse_run_dir_name(run_dir_name: str) -> Tuple[int, int, int, int]:
    # base like "tp1_in128_clk210_bs4"
    parts = run_dir_name.split("_")
    if len(parts) < 3:
        raise RuntimeError(f"Cannot parse run metadata from dir: {run_dir_name}")
    tp = int(parts[0].replace("tp", ""))
    input_len = int(parts[1].replace("in", ""))
    gpu_clock = int(parts[2].replace("clk", ""))
    batch_size = 1
    for part in parts[3:]:
        if part.startswith("bs"):
            batch_size = int(part.replace("bs", ""))
            break
    return tp, input_len, gpu_clock, batch_size


def _load_and_merge_dump_values(processed_dir: str) -> Dict[str, List[float]]:
    # Expect multiple per-rank json dumps written by llama.py signal/atexit handler.
    dump_paths = sorted(glob.glob(os.path.join(processed_dir, "*.json")))
    if not dump_paths:
        raise FileNotFoundError(f"No *.json dumps found under: {processed_dir}")

    merged: Dict[str, List[float]] = defaultdict(list)
    for p in dump_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            # Likely a partially-written dump (process terminated while writing).
            print(f"[warn] skip broken json dump: {p}. err={e}")
            continue
        values = payload.get("values") or {}
        for op_key, dur_list in values.items():
            if not isinstance(dur_list, list):
                continue
            merged[op_key].extend([float(x) for x in dur_list])
    return merged


def _aggregate_one_run(
    *,
    processed_dir: str,
    output_lens: List[int],
) -> List[Dict[str, object]]:
    run_dir = os.path.dirname(processed_dir)
    tp, input_len, gpu_clock, batch_size = _parse_run_dir_name(os.path.basename(run_dir))

    values = _load_and_merge_dump_values(processed_dir)

    rows_out: List[Dict[str, object]] = []

    # P stage: for each P_* op, use mean across all samples and replicate across output_len.
    for op_key, arr in values.items():
        if not op_key.startswith("P_"):
            continue
        if not arr:
            continue
        op_name = op_key[len("P_") :]
        mean_us = sum(arr) / len(arr)
        cnt = len(arr)
        for out_len in output_lens:
            rows_out.append(
                {
                    "tp": tp,
                    "input_len": input_len,
                    "output_len": out_len,
                    "gpu_clock": gpu_clock,
                    "batch_size": batch_size,
                    "stage": "P",
                    "op_name": op_name,
                    "count": cnt,
                    "latency_us": mean_us,
                }
            )

    # D stage: group by base name (D_*) and decode position suffix (_pos{...}).
    d_group: Dict[str, Dict[int, List[float]]] = defaultdict(lambda: defaultdict(list))
    # Fallback: for D_* ops without _pos suffix (e.g. stage-level TPOT),
    # replicate the mean across all output_lens.
    d_fallback: Dict[str, List[float]] = defaultdict(list)  # op_key -> durations
    for op_key, arr in values.items():
        if not op_key.startswith("D_"):
            continue
        m = _D_POS_NAME_RE.match(op_key)
        if m is None:
            d_fallback[op_key].extend(arr)
            continue
        base_name = m.group(1)  # includes D_ prefix
        pos_val = int(m.group(2))
        d_group[base_name][pos_val].extend(arr)

    for base_name, pos_map in d_group.items():
        if not pos_map:
            continue
        op_name = base_name[len("D_") :]
        sorted_positions = sorted(pos_map.keys())
        total_cnt = sum(len(v) for v in pos_map.values())
        for out_len in output_lens:
            idx = out_len - 1  # old convention: pick_pos is 1-based
            if idx < 0 or idx >= len(sorted_positions):
                continue
            pos_val = sorted_positions[idx]
            vals = pos_map.get(pos_val) or []
            if not vals:
                continue
            mean_us = sum(vals) / len(vals)
            rows_out.append(
                {
                    "tp": tp,
                    "input_len": input_len,
                    "output_len": out_len,
                    "gpu_clock": gpu_clock,
                    "batch_size": batch_size,
                    "stage": "D",
                    "op_name": op_name,
                    "count": total_cnt,
                    "latency_us": mean_us,
                }
            )

    # Fallback D_* ops without explicit decode position suffix.
    for op_key, arr in d_fallback.items():
        if not arr:
            continue
        # op_key like "D_TPOT" => op_name "TPOT"
        op_name = op_key[len("D_") :]
        mean_us = sum(arr) / len(arr)
        cnt = len(arr)
        for out_len in output_lens:
            rows_out.append(
                {
                    "tp": tp,
                    "input_len": input_len,
                    "output_len": out_len,
                    "gpu_clock": gpu_clock,
                    "batch_size": batch_size,
                    "stage": "D",
                    "op_name": op_name,
                    "count": cnt,
                    "latency_us": mean_us,
                }
            )

    return rows_out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert sync-op-bench llama.py dumps into pd_latency_big_table.csv for pivot script."
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default="bash-test/pd_batch_work",
        help="Directory containing tp*_in*_clk*_bs*/processed/*_rank*.json dumps.",
    )
    parser.add_argument(
        "--output-lens",
        type=str,
        default="64,256,512",
        help="Comma-separated output/token positions to pick for D stage.",
    )
    parser.add_argument(
        "--final-csv",
        type=str,
        default="bash-test/pd_latency_big_table.csv",
        help="Output long-format CSV path (same schema expected by pivot_pd_ops_wide.py).",
    )
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    output_lens = [int(x.strip()) for x in args.output_lens.split(",") if x.strip()]
    if not output_lens:
        raise ValueError("--output-lens must not be empty")

    processed_dirs = sorted(glob.glob(os.path.join(work_dir, "tp*", "processed")))
    if not processed_dirs:
        raise RuntimeError(f"No processed dirs found under: {work_dir}")

    out_rows: List[Dict[str, object]] = []
    for processed_dir in processed_dirs:
        out_rows.extend(
            _aggregate_one_run(
                processed_dir=processed_dir,
                output_lens=output_lens,
            )
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
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)

    print(f"[saved] {out_csv_path}")
    print(f"[info] rows: {len(out_rows)}")


if __name__ == "__main__":
    main()

