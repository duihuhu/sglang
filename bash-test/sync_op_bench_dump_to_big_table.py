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
_ALLOWED_OPS = {"A", "F", "TTFT", "TPOT"}


def _parse_run_dir_name(run_dir_name: str) -> Tuple[int, int, int, int, int]:
    # base like "tp1_in128_clk210_bs4_ol128"
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


def _load_and_merge_dump_values(
    processed_dir: str,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    # Expect multiple per-rank json dumps written by llama.py signal/atexit handler.
    dump_paths = sorted(glob.glob(os.path.join(processed_dir, "*.json")))
    if not dump_paths:
        raise FileNotFoundError(f"No *.json dumps found under: {processed_dir}")

    merged: Dict[str, List[float]] = defaultdict(list)
    merged_energy_uj: Dict[str, List[float]] = defaultdict(list)
    for p in dump_paths:
        try:
            with open(p, "r", encoding="utf-8") as f:
                payload = json.load(f)
        except json.JSONDecodeError as e:
            # Likely a partially-written dump (process terminated while writing).
            print(f"[warn] skip broken json dump: {p}. err={e}")
            continue
        values = payload.get("values") or {}
        energy_uj = payload.get("energy_uj") or {}
        for op_key, dur_list in values.items():
            if not isinstance(dur_list, list):
                continue
            merged[op_key].extend([float(x) for x in dur_list])
        for op_key, e_list in energy_uj.items():
            if not isinstance(e_list, list):
                continue
            merged_energy_uj[op_key].extend([float(x) for x in e_list])
    return merged, merged_energy_uj


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

    values, energy_uj_map = _load_and_merge_dump_values(processed_dir)

    rows_out: List[Dict[str, object]] = []

    # In D proxy mode, recover original input length from:
    # proxy_input_len = original_input_len + target_output_len - 1
    input_len_display = input_len
    if bench_stage == "D":
        input_len_display = input_len - target_output_len + 1

    # P stage: for each P_* op, use mean across all samples and replicate across output_len.
    if bench_stage == "P":
        for op_key, arr in values.items():
            if not op_key.startswith("P_"):
                continue
            if not arr:
                continue
            op_name = op_key[len("P_") :]
            if op_name not in _ALLOWED_OPS:
                continue
            mean_us = sum(arr) / len(arr)
            cnt = len(arr)
            energy_mean = None
            e_vals = energy_uj_map.get(op_key) or []
            if e_vals:
                energy_mean = sum(e_vals) / len(e_vals)
            for out_len in output_lens:
                out_len_val = target_output_len if target_output_len > 1 else out_len
                rows_out.append(
                    {
                        "tp": tp,
                        "input_len": input_len_display,
                        "output_len": out_len_val,
                        "gpu_clock": gpu_clock,
                        "batch_size": batch_size,
                        "stage": "P",
                        "op_name": op_name,
                        "count": cnt,
                        "latency_us": mean_us,
                        "energy_uj": energy_mean,
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

    if bench_stage == "D":
        for base_name, pos_map in d_group.items():
            if not pos_map:
                continue
            op_name = base_name[len("D_") :]
            if op_name not in _ALLOWED_OPS:
                continue
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
                e_vals = energy_uj_map.get(f"{base_name}_pos{pos_val}") or []
                energy_mean = (sum(e_vals) / len(e_vals)) if e_vals else None
                out_len_val = target_output_len if target_output_len > 1 else out_len
                rows_out.append(
                    {
                        "tp": tp,
                        "input_len": input_len_display,
                        "output_len": out_len_val,
                        "gpu_clock": gpu_clock,
                        "batch_size": batch_size,
                        "stage": "D",
                        "op_name": op_name,
                        "count": total_cnt,
                        "latency_us": mean_us,
                        "energy_uj": energy_mean,
                    }
                )

    # Fallback D_* ops without explicit decode position suffix.
    if bench_stage == "D":
        for op_key, arr in d_fallback.items():
            if not arr:
                continue
            # op_key like "D_TPOT" => op_name "TPOT"
            op_name = op_key[len("D_") :]
            if op_name not in _ALLOWED_OPS:
                continue
            mean_us = sum(arr) / len(arr)
            cnt = len(arr)
            e_vals = energy_uj_map.get(op_key) or []
            energy_mean = (sum(e_vals) / len(e_vals)) if e_vals else None
            for out_len in output_lens:
                out_len_val = target_output_len if target_output_len > 1 else out_len
                rows_out.append(
                    {
                        "tp": tp,
                        "input_len": input_len_display,
                        "output_len": out_len_val,
                        "gpu_clock": gpu_clock,
                        "batch_size": batch_size,
                        "stage": "D",
                        "op_name": op_name,
                        "count": cnt,
                        "latency_us": mean_us,
                        "energy_uj": energy_mean,
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
        "--bench-stage",
        type=str,
        default="D",
        choices=["P", "D", "p", "d"],
        help="Export only this stage from dumps.",
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
    bench_stage = str(args.bench_stage).strip().upper()
    if not output_lens:
        raise ValueError("--output-lens must not be empty")

    processed_dirs = sorted(glob.glob(os.path.join(work_dir, "tp*", "processed")))
    if not processed_dirs:
        raise RuntimeError(f"No processed dirs found under: {work_dir}")

    out_rows: List[Dict[str, object]] = []
    missing_dump_dirs: List[str] = []
    for processed_dir in processed_dirs:
        if not glob.glob(os.path.join(processed_dir, "*.json")):
            missing_dump_dirs.append(processed_dir)
            continue
        out_rows.extend(
            _aggregate_one_run(
                processed_dir=processed_dir,
                output_lens=output_lens,
                bench_stage=bench_stage,
            )
        )

    if missing_dump_dirs:
        preview = ", ".join(missing_dump_dirs[:3])
        suffix = "" if len(missing_dump_dirs) <= 3 else f", ... (+{len(missing_dump_dirs) - 3} more)"
        print(
            "[warn] skip processed dirs without sync-op dumps: "
            f"{len(missing_dump_dirs)} dirs. examples: {preview}{suffix}"
        )

    if not out_rows:
        raise RuntimeError(
            "No rows exported from sync-op dumps. "
            "No valid *.json dump files were found under any tp*/processed directory. "
            "Current work-dir may contain only bench CSV artifacts (e.g. *_af_ttft_af.csv), "
            "which are not sync-op dump inputs."
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
        for r in out_rows:
            w.writerow(r)

    print(f"[saved] {out_csv_path}")
    print(f"[info] rows: {len(out_rows)}")


if __name__ == "__main__":
    main()

