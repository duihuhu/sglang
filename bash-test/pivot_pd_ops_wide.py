#!/workspace/env/sglang-main/bin/python
import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Tuple

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
_DEFAULT_BIG_TABLE_CSV = os.path.join(_REPO_ROOT, "bash-test", "pd_latency_big_table.csv")

A_OPS = {"A", "PA", "DA", "input_layernorm", "qkv_proj", "rotary_emb", "attn", "o_proj"}


def _calc_a_f(op_map: Dict[str, float]) -> Tuple[float, float]:
    a_sum = 0.0
    f_sum = 0.0
    has_err = False
    exclude_from_a_f = {"TTFT", "TPOT"}
    for op_name, latency_us in op_map.items():
        if op_name in exclude_from_a_f:
            continue
        if latency_us < 0:
            has_err = True
            continue
        if op_name in A_OPS or op_name.startswith("A"):
            a_sum += latency_us
        else:
            f_sum += latency_us
    if has_err and a_sum == 0.0 and f_sum == 0.0:
        return -1.0, -1.0
    return a_sum, f_sum


def _calc_a_f_energy(metric_map: Dict[str, float], op_map: Dict[str, float]) -> Tuple[float, float]:
    a_sum = 0.0
    f_sum = 0.0
    has_err = False
    exclude = {"TTFT", "TPOT"}
    for op_name in op_map.keys():
        if op_name in exclude:
            continue
        val = metric_map.get(op_name)
        if val is None:
            continue
        if val < 0:
            has_err = True
            continue
        if op_name in A_OPS or op_name.startswith("A"):
            a_sum += val
        else:
            f_sum += val
    if has_err and a_sum == 0.0 and f_sum == 0.0:
        return -1.0, -1.0
    return a_sum, f_sum


def _display_op_value(op_name: str, latency_us: float) -> object:
    if latency_us < 0:
        return "ERR"
    if op_name in ("TTFT", "TPOT"):
        return latency_us / 1000.0
    return latency_us


def _fmt(val: float, divisor: float = 1.0) -> str:
    """Format a numeric value; return 'ERR' for negative sentinel values."""
    if val < 0:
        return "ERR"
    return f"{val / divisor:.2f}"


def _build_row_metrics(
    op_names: List[str],
    op_map: Dict[str, float],
    energy_map: Dict[str, float],
) -> Tuple[List[str], str, str, str, bool]:
    """
    Build display values for one output row.
    Returns (op_columns, af_ms, a_energy_mj, f_energy_mj, has_err).
    """
    row_vals = ["" for _ in op_names]
    has_err = False
    for i, op in enumerate(op_names):
        op_lookup = "TTFT" if op == "TTFT_ms" else ("TPOT" if op == "TPOT_ms" else op)
        if op_lookup in op_map:
            dv = _display_op_value(op_lookup, op_map[op_lookup])
            if dv == "ERR":
                row_vals[i] = "ERR"
                has_err = True
            else:
                row_vals[i] = f"{dv:.2f}"

    a_val, f_val = _calc_a_f(op_map)
    af_64_us = (a_val + f_val) * 64
    a_eu, f_eu = _calc_a_f_energy(energy_map, op_map)
    af_ms = _fmt(af_64_us, 1000.0)
    a_energy_mj = _fmt(a_eu, 1000.0)
    f_energy_mj = _fmt(f_eu, 1000.0)
    if af_ms == "ERR" or a_energy_mj == "ERR" or f_energy_mj == "ERR":
        has_err = True

    return row_vals, af_ms, a_energy_mj, f_energy_mj, has_err


def _write_wide_block(
    w: csv.writer,
    stage: str,
    label: str,
    data_by_stage: Dict[str, Dict[Tuple[int, int, int, int, int], Dict[str, float]]],
    energy_by_stage: Dict[str, Dict[Tuple[int, int, int, int, int], Dict[str, float]]],
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
    if stage == "P" and drop_p_output_len:
        w.writerow([label] + [""] * (4 + len(op_names)))
    else:
        w.writerow([label] + [""] * (5 + len(op_names)))
    if stage == "P" and drop_p_output_len:
        w.writerow(
            [
                "tp",
                "input_len",
                "gpu_clock",
                "batch_size",
                *op_names,
                "(A+F)*64_ms",
                "A_energy_mj",
                "F_energy_mj",
            ]
        )
        for (tp, input_len, gpu_clock, batch_size) in rows:  # type: ignore[misc]
            op_map = data_by_stage[stage][(tp, input_len, gpu_clock, batch_size)]  # type: ignore[index]
            energy_map = energy_by_stage[stage][(tp, input_len, gpu_clock, batch_size)]  # type: ignore[index]
            row_vals, af_ms, a_energy_mj, f_energy_mj, has_err = _build_row_metrics(
                op_names, op_map, energy_map
            )
            if has_err:
                continue
            w.writerow(
                [
                    tp,
                    input_len,
                    gpu_clock,
                    batch_size,
                    *row_vals,
                    af_ms,
                    a_energy_mj,
                    f_energy_mj,
                ]
            )
        return

    w.writerow(
        [
            "tp",
            "input_len",
            "output_len",
            "gpu_clock",
            "batch_size",
            *op_names,
            "(A+F)*64_ms",
            "A_energy_mj",
            "F_energy_mj",
        ]
    )
    for (tp, input_len, output_len, gpu_clock, batch_size) in rows:
        op_map = data_by_stage[stage][
            (tp, input_len, output_len, gpu_clock, batch_size)
        ]
        energy_map = energy_by_stage[stage][
            (tp, input_len, output_len, gpu_clock, batch_size)
        ]
        row_vals, af_ms, a_energy_mj, f_energy_mj, has_err = _build_row_metrics(
            op_names, op_map, energy_map
        )
        if has_err:
            continue
        w.writerow(
            [
                tp,
                input_len,
                output_len,
                gpu_clock,
                batch_size,
                *row_vals,
                af_ms,
                a_energy_mj,
                f_energy_mj,
            ]
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pivot pd_latency_big_table.csv into wide tables where op_name expands along horizontal axis."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=_DEFAULT_BIG_TABLE_CSV,
        help="Path to pd_latency_big_table.csv (default: bash-test/pd_latency_big_table.csv under repo root).",
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
    if not os.path.isfile(csv_path):
        print(f"[error] input CSV not found: {csv_path}", file=sys.stderr)
        sys.exit(2)

    data_by_stage: Dict[str, Dict[object, Dict[str, float]]] = {
        "P": defaultdict(dict),
        "D": defaultdict(dict),
    }
    energy_by_stage: Dict[str, Dict[object, Dict[str, float]]] = {
        "P": defaultdict(dict),
        "D": defaultdict(dict),
    }
    # Accumulate duplicate (cfg, op) rows using weighted mean by `count`.
    # key -> [lat_sum, weight_sum]
    latency_acc: Dict[str, Dict[object, Dict[str, List[float]]]] = {
        "P": defaultdict(dict),
        "D": defaultdict(dict),
    }
    # key -> [energy_sum, weight_sum]
    energy_acc: Dict[str, Dict[object, Dict[str, List[float]]]] = {
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
            if args.only_af and not (
                op_name in {"TTFT", "TPOT"}
                or op_name.startswith("A")
                or op_name.startswith("F")
            ):
                continue
            tp = int(row["tp"])
            input_len = int(row["input_len"])
            output_len = int(row["output_len"])
            gpu_clock = int(row["gpu_clock"])
            # Backward compatibility: old big tables may not contain batch_size.
            batch_size = int((row.get("batch_size") or "1").strip())
            latency_us = float(row["latency_us"])
            energy_uj = (
                float(row.get("energy_uj"))
                if row.get("energy_uj") not in (None, "", "None")
                else None
            )

            if stage == "P" and args.drop_p_output_len:
                cfg_key = (tp, input_len, gpu_clock, batch_size)
            else:
                cfg_key = (tp, input_len, output_len, gpu_clock, batch_size)
            weight = float((row.get("count") or "1").strip() or "1")
            if weight <= 0:
                weight = 1.0

            if op_name not in latency_acc[stage][cfg_key]:
                latency_acc[stage][cfg_key][op_name] = [0.0, 0.0]
            latency_acc[stage][cfg_key][op_name][0] += latency_us * weight
            latency_acc[stage][cfg_key][op_name][1] += weight

            if energy_uj is not None:
                if op_name not in energy_acc[stage][cfg_key]:
                    energy_acc[stage][cfg_key][op_name] = [0.0, 0.0]
                energy_acc[stage][cfg_key][op_name][0] += energy_uj * weight
                energy_acc[stage][cfg_key][op_name][1] += weight

            if op_name not in seen_op[stage]:
                seen_op[stage].add(op_name)
                op_names_by_stage[stage].append(op_name)

    # Finalize weighted means into wide-table maps.
    for stage in ("P", "D"):
        for cfg_key, op_map in latency_acc[stage].items():
            for op_name, (lat_sum, w_sum) in op_map.items():
                if w_sum > 0:
                    data_by_stage[stage][cfg_key][op_name] = lat_sum / w_sum
        for cfg_key, op_map in energy_acc[stage].items():
            for op_name, (e_sum, w_sum) in op_map.items():
                if w_sum > 0:
                    energy_by_stage[stage][cfg_key][op_name] = e_sum / w_sum

    # Rename stage-level timing columns to indicate display unit in output tables.
    for stage in ("P", "D"):
        op_names_by_stage[stage] = [
            "TTFT_ms" if name == "TTFT"
            else ("TPOT_ms" if name == "TPOT" else name)
            for name in op_names_by_stage[stage]
        ]

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
                        "(A+F)*64_ms",
                        "A_energy_mj",
                        "F_energy_mj",
                    ]
                )
                rows = sorted(data_by_stage[stage].keys(), key=lambda k: (k[0], k[1], k[2], k[3]))  # type: ignore[index]
                for (tp, input_len, gpu_clock, batch_size) in rows:  # type: ignore[misc]
                    op_map = data_by_stage[stage][(tp, input_len, gpu_clock, batch_size)]  # type: ignore[index]
                    energy_map = energy_by_stage[stage][(tp, input_len, gpu_clock, batch_size)]  # type: ignore[index]
                    row_vals, af_ms, a_energy_mj, f_energy_mj, has_err = _build_row_metrics(
                        op_names, op_map, energy_map
                    )
                    if has_err:
                        continue
                    w.writerow(
                        [
                            tp,
                            input_len,
                            gpu_clock,
                            batch_size,
                            *row_vals,
                            af_ms,
                            a_energy_mj,
                            f_energy_mj,
                        ]
                    )
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
                    "(A+F)*64_ms",
                    "A_energy_mj",
                    "F_energy_mj",
                ]
            )
            for (tp, input_len, output_len, gpu_clock, batch_size) in rows:  # type: ignore[misc]
                op_map = data_by_stage[stage][
                    (tp, input_len, output_len, gpu_clock, batch_size)
                ]
                energy_map = energy_by_stage[stage][
                    (tp, input_len, output_len, gpu_clock, batch_size)
                ]
                row_vals, af_ms, a_energy_mj, f_energy_mj, has_err = _build_row_metrics(
                    op_names, op_map, energy_map
                )
                if has_err:
                    continue
                w.writerow(
                    [
                        tp,
                        input_len,
                        output_len,
                        gpu_clock,
                        batch_size,
                        *row_vals,
                        af_ms,
                        a_energy_mj,
                        f_energy_mj,
                    ]
                )

    outputs: List[str] = []

    # Default: one CSV with two sections (P block, blank row, D block)
    combined_path = os.path.join(out_dir, args.out)
    wrote_any = False
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
                energy_by_stage,  # type: ignore[arg-type]
                op_names_by_stage,
                drop_p_output_len=bool(args.drop_p_output_len),
            )
            wrote_any = True
            first = False
        if args.stage in ("D", "both") and op_names_by_stage["D"]:
            if not first:
                w.writerow([])  # blank row between the two tables
            _write_wide_block(
                w,
                "D",
                "[D] Decode",
                data_by_stage,  # type: ignore[arg-type]
                energy_by_stage,  # type: ignore[arg-type]
                op_names_by_stage,
            )
            wrote_any = True
        if not wrote_any:
            print(
                f"[warn] pivot: no rows matched stage={args.stage!r} from {csv_path}. "
                "Big table may be header-only, wrong path, or use --stage P for prefill-only data.",
                file=sys.stderr,
            )
            w.writerow(
                [
                    "# pivot_pd_ops_wide: no data",
                    f"input={csv_path}",
                    f"stage={args.stage}",
                ]
            )
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
