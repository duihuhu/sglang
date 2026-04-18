#!/usr/bin/env python3
"""
将 prefill_data.txt（制表符分隔）转为 plot_latency_energy_pareto.py 可读的 CSV。

源列（默认表头）：
  tp, input_len, gpu_clock, batch_size, P_A_lat, P_F_lat, P_A_energy, P_F_energy

目标列与 P_data.csv 一致：
  tp, input_len, gpu_clock, batch_size, A, F, TTFT_ms, (A+F)*64_ms, A_energy_mj, F_energy_mj

映射：P_A_lat→A，P_F_lat→F，P_A_energy→A_energy_mj，P_F_energy→F_energy_mj。
TTFT_ms、(A+F)*64_ms 在源文件中无对应项时留空（绘图脚本不依赖这两列）。

默认仅导出 batch_size=1 的行，与现有 P_data.csv 习惯一致；可用 --batch-size 指定或 --all-batch-sizes。
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

# 与 plot_latency_energy_pareto / P_data 表头一致
CSV_TITLE = "[P] Prefill,,,,,,,"
CSV_HEADER = [
    "tp",
    "input_len",
    "gpu_clock",
    "batch_size",
    "A",
    "F",
    "TTFT_ms",
    "(A+F)*64_ms",
    "A_energy_mj",
    "F_energy_mj",
]

def _norm_key(s: str) -> str:
    return s.strip().lower().replace(" ", "_")


def sniff_delimiter(first_line: str) -> str:
    if "\t" in first_line:
        return "\t"
    return ","


def read_txt_rows(path: Path, delimiter: str) -> tuple[list[str], list[dict[str, str]]]:
    text = path.read_text(encoding="utf-8-sig")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"空文件: {path}")
    if delimiter == "auto":
        delimiter = sniff_delimiter(lines[0])
    reader = csv.DictReader(lines, delimiter=delimiter)
    if not reader.fieldnames:
        raise ValueError("无法解析表头")
    fieldnames = [f.strip() for f in reader.fieldnames]
    rows = []
    for raw in reader:
        row = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in raw.items() if k}
        rows.append(row)
    return fieldnames, rows


def resolve_txt_columns(fieldnames: list[str]) -> dict[str, str]:
    """返回 {标准键: 源表头原文字符串}。"""
    lower_map = {_norm_key(f): f for f in fieldnames}
    need = {
        "tp": ("tp",),
        "input_len": ("input_len",),
        "gpu_clock": ("gpu_clock",),
        "batch_size": ("batch_size",),
        "P_A_lat": ("p_a_lat", "a", "p_a", "a_lat"),
        "P_F_lat": ("p_f_lat", "f", "p_f", "f_lat"),
        "P_A_energy": ("p_a_energy", "a_energy", "p_a_energy_mj"),
        "P_F_energy": ("p_f_energy", "f_energy", "p_f_energy_mj"),
    }
    out: dict[str, str] = {}
    for canon, aliases in need.items():
        found = None
        for a in aliases:
            if a in lower_map:
                found = lower_map[a]
                break
        if not found:
            raise KeyError(f"表头缺少与「{canon}」匹配的列（尝试过 {aliases}），实际表头: {fieldnames}")
        out[canon] = found
    return out


def convert_row(src: dict[str, str], colmap: dict[str, str]) -> list[str]:
    def g(canon: str) -> str:
        return src[colmap[canon]].strip()

    tp = g("tp")
    input_len = g("input_len")
    gpu_clock = g("gpu_clock")
    batch_size = g("batch_size")
    a = g("P_A_lat")
    f = g("P_F_lat")
    ae = g("P_A_energy")
    fe = g("P_F_energy")
    return [
        tp,
        input_len,
        gpu_clock,
        batch_size,
        a,
        f,
        "",  # TTFT_ms
        "",  # (A+F)*64_ms
        ae,
        fe,
    ]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path(__file__).resolve().parent / "prefill_data.txt",
        help="输入 txt 路径（默认：同目录 prefill_data.txt）",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="输出 CSV（默认：与输入同目录，文件名 <stem>_p_data.csv）",
    )
    p.add_argument(
        "--delimiter",
        choices=("auto", "\t", ","),
        default="auto",
        help="分隔符（默认自动：有制表符则用 TAB）",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        metavar="N",
        help="只保留 batch_size=N 的行（默认 1）。与 --all-batch-sizes 互斥",
    )
    p.add_argument(
        "--all-batch-sizes",
        action="store_true",
        help="不过滤 batch_size，导出全部行",
    )
    args = p.parse_args()

    delim = "\t" if args.delimiter == "\t" else ("," if args.delimiter == "," else "auto")
    fieldnames, rows = read_txt_rows(args.input, delim)
    colmap = resolve_txt_columns(fieldnames)

    out_path = args.output
    if out_path is None:
        out_path = args.input.parent / f"{args.input.stem}_p_data.csv"

    out_rows: list[list[str]] = []
    for src in rows:
        try:
            bs = int(src[colmap["batch_size"]].strip())
        except (ValueError, KeyError):
            continue
        if not args.all_batch_sizes and bs != args.batch_size:
            continue
        out_rows.append(convert_row(src, colmap))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        f.write(CSV_TITLE + "\n")
        w = csv.writer(f)
        w.writerow(CSV_HEADER)
        w.writerows(out_rows)

    print(f"读取 {args.input}，表头: {fieldnames}")
    print(f"写出 {len(out_rows)} 行 -> {out_path}")


if __name__ == "__main__":
    main()
