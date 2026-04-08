#!/workspace/env/sglang-main/bin/python
import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

A_OPS = {"A", "PA", "DA", "input_layernorm", "qkv_proj", "rotary_emb", "attn", "o_proj"}


def _calc_a_f(op_map: Dict[str, float]) -> Tuple[float, float]:
    a_sum = 0.0
    f_sum = 0.0
    # TTFT/TPOT are stage-level timings (not per-op). Keep them as standalone columns,
    # but do not include them in A/F breakdown.
    exclude_from_a_f = {"TTFT", "TPOT"}
    for op_name, latency_us in op_map.items():
        if op_name in exclude_from_a_f:
            continue
        if op_name in A_OPS:
            a_sum += latency_us
        else:
            f_sum += latency_us
    return a_sum, f_sum


def _write_wide_block(
    w: csv.writer,
    stage: str,
    label: str,
    data_by_stage: Dict[str, Dict[Tuple[int, int, int, int, int], Dict[str, float]]],
    op_names_by_stage: Dict[str, List[str]],
    *,
    drop_p_output_len: bool = False,
) -> None:
    op_names = op_names_by_stage[stage]
    if stage == "P" and drop_p_output_len:
        rows = sorted(data_by_stage[stage].keys(), key=lambda k: (k[0], k[1], k[2], k[3]))  # type: ignore[index]
    else:
        rows = sorted(  # type: ignore[index]
            data_by_stage[stage].keys(), key=lambda k: (k[0], k[1], k[2], k[3], k[4])
        )
    # Section title row: label in column A.
    # For P-with-drop: columns=tp,input_len,gpu_clock,batch_size + ops + P_A,P_F,D_A,D_F
    # => 8 + len(op_names) total columns.
    if stage == "P" and drop_p_output_len:
        w.writerow([label] + [""] * (7 + len(op_names)))
    else:
        # P(without-drop) and D: columns=tp,input_len,output_len,gpu_clock,batch_size + ops + 4 summary cols
        # => 9 + len(op_names) total columns.
        w.writerow([label] + [""] * (8 + len(op_names)))
    if stage == "P" and drop_p_output_len:
        w.writerow(
            [
                "tp",
                "input_len",
                "gpu_clock",
                "batch_size",
                *op_names,
                "P_A",
                "P_F",
                "D_A",
                "D_F",
            ]
        )
        for (tp, input_len, gpu_clock, batch_size) in rows:  # type: ignore[misc]
            row_vals = ["" for _ in op_names]
            op_map = data_by_stage[stage][(tp, input_len, gpu_clock, batch_size)]  # type: ignore[index]
            for i, op in enumerate(op_names):
                if op in op_map:
                    row_vals[i] = f"{op_map[op]:.2f}"
            a_sum, f_sum = _calc_a_f(op_map)
            merged_cols = [f"{a_sum:.2f}", f"{f_sum:.2f}", "", ""]
            w.writerow([tp, input_len, gpu_clock, batch_size, *row_vals, *merged_cols])
        return

    w.writerow(
        [
            "tp",
            "input_len",
            "output_len",
            "gpu_clock",
            "batch_size",
            *op_names,
            "P_A",
            "P_F",
            "D_A",
            "D_F",
        ]
    )
    for (tp, input_len, output_len, gpu_clock, batch_size) in rows:
        row_vals = ["" for _ in op_names]
        op_map = data_by_stage[stage][
            (tp, input_len, output_len, gpu_clock, batch_size)
        ]
        for i, op in enumerate(op_names):
            if op in op_map:
                row_vals[i] = f"{op_map[op]:.2f}"
        a_sum, f_sum = _calc_a_f(op_map)
        if stage == "P":
            merged_cols = [f"{a_sum:.2f}", f"{f_sum:.2f}", "", ""]
        else:
            merged_cols = ["", "", f"{a_sum:.2f}", f"{f_sum:.2f}"]
        w.writerow(
            [tp, input_len, output_len, gpu_clock, batch_size, *row_vals, *merged_cols]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pivot pd_latency_big_table.csv into wide tables where op_name expands along horizontal axis."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="/mnt/nvme1/lt/cache/pd_latency_big_table.csv",
        help="Path to pd_latency_big_table.csv",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="/workspace/benchmark/sglang-main/bash-test/pivot_pd_ops_out",
        help="Output directory",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="pd_ops_wide.csv",
        help="Single combined wide CSV filename (under --out-dir). Contains P and D as two sections.",
    )
    parser.add_argument(
        "--split-files",
        action="store_true",
        help="Also write legacy pd_ops_P_wide.csv and pd_ops_D_wide.csv (two separate files).",
    )
    parser.add_argument(
        "--drop-p-output-len",
        action="store_true",
        help="For P stage, drop output_len dimension in the wide output (merge rows across output_len).",
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="both",
        choices=["P", "D", "both"],
        help="Which stage to pivot",
    )
    parser.add_argument(
        "--only-af",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only keep A/F coarse ops (plus TTFT/TPOT).",
    )
    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    data_by_stage: Dict[str, Dict[object, Dict[str, float]]] = {
        "P": defaultdict(dict),
        "D": defaultdict(dict),
    }
    op_names_by_stage: Dict[str, List[str]] = {"P": [], "D": []}
    seen_op: Dict[str, set] = {"P": set(), "D": set()}

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = ["tp", "input_len", "output_len", "gpu_clock", "stage", "op_name", "latency_us"]
        for r in required:
            if r not in reader.fieldnames:
                raise RuntimeError(f"Missing column {r} in {csv_path}. found={reader.fieldnames}")

        for row in reader:
            stage = (row["stage"] or "").strip()
            if stage not in ("P", "D"):
                continue
            if args.stage != "both" and stage != args.stage:
                continue
            op_name = (row["op_name"] or "").strip()
            if not op_name:
                continue
            if args.only_af and op_name not in {"A", "F", "TTFT", "TPOT"}:
                continue
            tp = int(row["tp"])
            input_len = int(row["input_len"])
            output_len = int(row["output_len"])
            gpu_clock = int(row["gpu_clock"])
            # Backward compatibility: old big tables may not contain batch_size.
            batch_size = int((row.get("batch_size") or "1").strip())
            latency_us = float(row["latency_us"])

            if stage == "P" and args.drop_p_output_len:
                cfg_key = (tp, input_len, gpu_clock, batch_size)
            else:
                cfg_key = (tp, input_len, output_len, gpu_clock, batch_size)
            data_by_stage[stage][cfg_key][op_name] = latency_us

            if op_name not in seen_op[stage]:
                seen_op[stage].add(op_name)
                op_names_by_stage[stage].append(op_name)

    def write_wide_file(stage: str, path: str) -> None:
        op_names = op_names_by_stage[stage]
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if stage == "P" and args.drop_p_output_len:
                w.writerow(
                    [
                        "tp",
                        "input_len",
                        "gpu_clock",
                        "batch_size",
                        *op_names,
                        "P_A",
                        "P_F",
                        "D_A",
                        "D_F",
                    ]
                )
                rows = sorted(data_by_stage[stage].keys(), key=lambda k: (k[0], k[1], k[2], k[3]))  # type: ignore[index]
                for (tp, input_len, gpu_clock, batch_size) in rows:  # type: ignore[misc]
                    row_vals = ["" for _ in op_names]
                    op_map = data_by_stage[stage][(tp, input_len, gpu_clock, batch_size)]  # type: ignore[index]
                    for i, op in enumerate(op_names):
                        if op in op_map:
                            row_vals[i] = f"{op_map[op]:.2f}"
                    a_sum, f_sum = _calc_a_f(op_map)
                    merged_cols = [f"{a_sum:.2f}", f"{f_sum:.2f}", "", ""]
                    w.writerow([tp, input_len, gpu_clock, batch_size, *row_vals, *merged_cols])
                return

            rows = sorted(data_by_stage[stage].keys(), key=lambda k: (k[0], k[1], k[2], k[3], k[4]))  # type: ignore[index]
            w.writerow(
                [
                    "tp",
                    "input_len",
                    "output_len",
                    "gpu_clock",
                    "batch_size",
                    *op_names,
                    "P_A",
                    "P_F",
                    "D_A",
                    "D_F",
                ]
            )
            for (tp, input_len, output_len, gpu_clock, batch_size) in rows:  # type: ignore[misc]
                row_vals = ["" for _ in op_names]
                op_map = data_by_stage[stage][
                    (tp, input_len, output_len, gpu_clock, batch_size)
                ]
                for i, op in enumerate(op_names):
                    if op in op_map:
                        row_vals[i] = f"{op_map[op]:.2f}"
                a_sum, f_sum = _calc_a_f(op_map)
                if stage == "P":
                    merged_cols = [f"{a_sum:.2f}", f"{f_sum:.2f}", "", ""]
                else:
                    merged_cols = ["", "", f"{a_sum:.2f}", f"{f_sum:.2f}"]
                w.writerow(
                    [
                        tp,
                        input_len,
                        output_len,
                        gpu_clock,
                        batch_size,
                        *row_vals,
                        *merged_cols,
                    ]
                )

    outputs: List[str] = []

    # Default: one CSV with two sections (P block, blank row, D block)
    combined_path = os.path.join(out_dir, args.out)
    with open(combined_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        first = True
        if args.stage in ("P", "both") and op_names_by_stage["P"]:
            # Do not use leading "=" (e.g. "=== ...") — Excel treats that as a formula.
            _write_wide_block(
                w,
                "P",
                "[P] Prefill",
                data_by_stage,  # type: ignore[arg-type]
                op_names_by_stage,
                drop_p_output_len=bool(args.drop_p_output_len),
            )
            first = False
        if args.stage in ("D", "both") and op_names_by_stage["D"]:
            if not first:
                w.writerow([])  # blank row between the two tables
            _write_wide_block(w, "D", "[D] Decode", data_by_stage, op_names_by_stage)
    outputs.append(combined_path)

    if args.split_files:
        if args.stage in ("P", "both") and op_names_by_stage["P"]:
            p_path = os.path.join(out_dir, "pd_ops_P_wide.csv")
            write_wide_file("P", p_path)
            outputs.append(p_path)
        if args.stage in ("D", "both") and op_names_by_stage["D"]:
            d_path = os.path.join(out_dir, "pd_ops_D_wide.csv")
            write_wide_file("D", d_path)
            outputs.append(d_path)

    print("[saved]")
    for p in outputs:
        print(p)


if __name__ == "__main__":
    main()
