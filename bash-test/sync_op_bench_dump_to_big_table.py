#!/workspace/env/sglang-main/bin/python
import argparse
import csv
import glob
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Optional, Tuple


_D_POS_NAME_RE = re.compile(r"^(D_[A-Za-z0-9_]+)_pos(-?\d+)$")


def _is_allowed_op_name(op_name: str) -> bool:
    if op_name in {"TTFT", "TPOT"}:
        return True
    if op_name.startswith("A") or op_name.startswith("F"):
        return True
    return False


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


def _load_and_merge_dump_values(processed_dir: str) -> Dict[str, List[float]]:
    # Expect multiple per-rank json dumps written by llama.py signal/atexit handler.
    dump_paths = sorted(glob.glob(os.path.join(processed_dir, "*.json")))
    if not dump_paths:
        return {}

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


def _load_ttft_af_summary_rows(processed_dir: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in sorted(glob.glob(os.path.join(processed_dir, "*_ttft_af.csv"))):
        try:
            with open(p, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(dict(r))
        except Exception as e:
            print(f"[warn] skip broken ttft/af csv: {p}. err={e}")
            continue
    return rows


def _load_ttft_af_sample_rows(processed_dir: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for p in sorted(glob.glob(os.path.join(processed_dir, "*_ttft_af_samples.csv"))):
        try:
            with open(p, "r", encoding="utf-8", newline="") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    rows.append(dict(r))
        except Exception as e:
            print(f"[warn] skip broken ttft/af samples csv: {p}. err={e}")
            continue
    return rows


def _to_float(v) -> float:
    if v is None:
        return 0.0
    s = str(v).strip()
    if not s:
        return 0.0
    return float(s)


def _to_int(v) -> int:
    if v is None:
        return 0
    s = str(v).strip()
    if not s:
        return 0
    return int(float(s))


def _to_optional_float(v) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in ("none", "nan"):
        return None
    try:
        return float(s)
    except ValueError:
        return None


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

    values = _load_and_merge_dump_values(processed_dir)
    summary_rows = _load_ttft_af_summary_rows(processed_dir)

    rows_out: List[Dict[str, object]] = []

    # In D proxy mode, recover original input length from:
    # proxy_input_len = original_input_len + target_output_len - 1
    input_len_display = input_len
    if bench_stage == "D":
        input_len_display = input_len - target_output_len + 1

    # P stage: first consume summary CSV produced by qwen3.py (TTFT/A/F),
    # then merge legacy JSON keys for backward compatibility.
    if bench_stage == "P":
        for srow in summary_rows:
            # (op_name, lat_key, cnt_key, energy_key, energy_cnt_key)
            op_specs = [
                ("TTFT", "ttft_avg_us", "ttft_count", None, None),
                ("A", "a_avg_us", "a_count", "a_avg_energy_uj", "a_energy_count"),
                ("F", "f_avg_us", "f_count", "f_avg_energy_uj", "f_energy_count"),
            ]
            for spec in op_specs:
                op_name, lat_key, cnt_key = spec[0], spec[1], spec[2]
                energy_key = spec[3]
                energy_cnt_key = spec[4]
                cnt = _to_int(srow.get(cnt_key, 0))
                if cnt <= 0:
                    continue
                mean_us = _to_float(srow.get(lat_key, 0.0))
                energy_uj: Optional[float] = None
                if energy_key and energy_cnt_key:
                    ec = _to_int(srow.get(energy_cnt_key, 0))
                    if ec > 0:
                        energy_uj = _to_optional_float(srow.get(energy_key))
                for out_len in output_lens:
                    out_len_val = target_output_len if target_output_len > 1 else out_len
                    row_d: Dict[str, object] = {
                        "tp": tp,
                        "input_len": input_len_display,
                        "output_len": out_len_val,
                        "gpu_clock": gpu_clock,
                        "batch_size": batch_size,
                        "stage": "P",
                        "op_name": op_name,
                        "count": cnt,
                        "latency_us": mean_us,
                        "energy_uj": energy_uj if energy_uj is not None else "",
                    }
                    rows_out.append(row_d)

    # P stage: for each P_* op in JSON, use mean across all samples and replicate across output_len.
    # If model-side stage tagging emits D_A*/D_F* in prefill-only runs, accept them here
    # and still export as stage=P to avoid dropping valid prefill measurements.
    if bench_stage == "P":
        for op_key, arr in values.items():
            if op_key.startswith("P_"):
                op_name = op_key[len("P_") :]
            elif op_key.startswith("D_"):
                m = _D_POS_NAME_RE.match(op_key)
                if m is not None:
                    # Accept D_*_posX fallback in P runs; strip position suffix.
                    op_name = m.group(1)[len("D_") :]
                else:
                    op_name = op_key[len("D_") :]
            else:
                continue
            if not arr:
                continue
            if not _is_allowed_op_name(op_name):
                continue
            mean_us = sum(arr) / len(arr)
            cnt = len(arr)
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
                        "energy_uj": "",
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
            if not _is_allowed_op_name(op_name):
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
                        "energy_uj": "",
                    }
                )

    # Fallback D_* ops without explicit decode position suffix.
    if bench_stage == "D":
        for op_key, arr in d_fallback.items():
            if not arr:
                continue
            # op_key like "D_TPOT" => op_name "TPOT"
            op_name = op_key[len("D_") :]
            if not _is_allowed_op_name(op_name):
                continue
            mean_us = sum(arr) / len(arr)
            cnt = len(arr)
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
                        "energy_uj": "",
                    }
                )

    return rows_out


def _derive_samples_csv_path(final_csv: str) -> str:
    root, ext = os.path.splitext(os.path.abspath(final_csv))
    return f"{root}_samples{ext or '.csv'}"


def _aggregate_af_per_sample_rows(
    *,
    processed_dir: str,
    output_lens: List[int],
    bench_stage: str,
) -> List[Dict[str, object]]:
    """
    One row per (run config, op_name in {A,F}, sample index) from sync-op JSON dumps.
    Latency list comes from qwen3_loop_bench_decoder_block (one append per measurement).
    """
    run_dir = os.path.dirname(processed_dir)
    tp, input_len, gpu_clock, batch_size, target_output_len = _parse_run_dir_name(
        os.path.basename(run_dir)
    )

    values = _load_and_merge_dump_values(processed_dir)
    sample_csv_rows = _load_ttft_af_sample_rows(processed_dir)

    input_len_display = input_len
    if bench_stage == "D":
        input_len_display = input_len - target_output_len + 1

    rows_out: List[Dict[str, object]] = []
    if bench_stage != "P":
        return rows_out

    out_len_val = (
        target_output_len
        if target_output_len > 1
        else (output_lens[0] if output_lens else 1)
    )

    # 1) New per-sample CSV rows from qwen3.py.
    for srow in sample_csv_rows:
        op_name = str(srow.get("op", "")).strip().upper()
        if op_name not in {"TTFT", "A", "F"}:
            continue
        eu = _to_optional_float(srow.get("energy_uj"))
        rows_out.append(
            {
                "tp": tp,
                "input_len": input_len_display,
                "output_len": out_len_val,
                "gpu_clock": gpu_clock,
                "batch_size": batch_size,
                "stage": "P",
                "op_name": op_name,
                "sample_idx": _to_int(srow.get("sample_idx", 0)),
                "latency_us": _to_float(srow.get("latency_us", 0.0)),
                "energy_uj": eu if eu is not None else "",
            }
        )

    # 2) Backward-compatible legacy JSON rows (A/F only).
    for op_key, arr in values.items():
        if not op_key.startswith("P_"):
            continue
        if not arr:
            continue
        op_name = op_key[len("P_") :]
        if not (op_name.startswith("A") or op_name.startswith("F")):
            continue
        for i, lat in enumerate(arr):
            rows_out.append(
                {
                    "tp": tp,
                    "input_len": input_len_display,
                    "output_len": out_len_val,
                    "gpu_clock": gpu_clock,
                    "batch_size": batch_size,
                    "stage": "P",
                    "op_name": op_name,
                    "sample_idx": i,
                    "latency_us": float(lat),
                    "energy_uj": "",
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
    parser.add_argument(
        "--final-csv-samples",
        type=str,
        default=None,
        help="Per-sample A/F CSV path (default: <final-csv stem>_samples.csv).",
    )
    parser.add_argument(
        "--skip-af-samples",
        action="store_true",
        help="Do not write per-sample A/F CSV.",
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
    sample_rows: List[Dict[str, object]] = []
    for processed_dir in processed_dirs:
        out_rows.extend(
            _aggregate_one_run(
                processed_dir=processed_dir,
                output_lens=output_lens,
                bench_stage=bench_stage,
            )
        )
        sample_rows.extend(
            _aggregate_af_per_sample_rows(
                processed_dir=processed_dir,
                output_lens=output_lens,
                bench_stage=bench_stage,
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
            "energy_uj",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            row = dict(r)
            if "energy_uj" not in row:
                row["energy_uj"] = ""
            w.writerow(row)

    print(f"[saved] {out_csv_path}")
    print(f"[info] rows: {len(out_rows)}")

    if not args.skip_af_samples and sample_rows:
        samples_path = os.path.abspath(
            args.final_csv_samples or _derive_samples_csv_path(args.final_csv)
        )
        os.makedirs(os.path.dirname(samples_path) or ".", exist_ok=True)
        sample_fields = [
            "tp",
            "input_len",
            "output_len",
            "gpu_clock",
            "batch_size",
            "stage",
            "op_name",
            "sample_idx",
            "latency_us",
            "energy_uj",
        ]
        with open(samples_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=sample_fields)
            w.writeheader()
            for r in sample_rows:
                row = dict(r)
                if "energy_uj" not in row:
                    row["energy_uj"] = ""
                w.writerow(row)
        print(f"[saved] per-sample A/F: {samples_path}")
        print(f"[info] sample rows: {len(sample_rows)}")
    elif not args.skip_af_samples and bench_stage == "P":
        print("[info] no per-sample A/F rows (empty dumps or no P_A/P_F keys)")


if __name__ == "__main__":
    main()

