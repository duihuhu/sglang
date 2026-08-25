#!/usr/bin/env python3
"""Export kernel-only EP JSONL into PF/DF-style TSV under data/kernel-EP/."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "qwen3-af-v2"
PHASE_FILES = {
    "prefill": ("PF-{routing}.txt", "input_len"),
    "decode": ("DF-{routing}.txt", "context_len"),
}
ROUTING_ALIASES = {
    "skewed_rank0": "skewed",
    "middle_rank0": "middle",
}
KERNEL_ROUTING_MODES = ("balanced", "middle_rank0", "skewed_rank0")


class ExportError(RuntimeError):
    pass


@dataclass
class Summary:
    files_scanned: int = 0
    rows_read: int = 0
    rows_exported: int = 0
    skipped: int = 0
    duplicates: int = 0

    def render(self) -> str:
        return (
            f"export-kernel-ep files_scanned={self.files_scanned} "
            f"rows_read={self.rows_read} rows_exported={self.rows_exported} "
            f"skipped={self.skipped} duplicates={self.duplicates}"
        )


def _routing_mode(row: dict[str, Any]) -> str | None:
    routing = row.get("routing")
    if not isinstance(routing, dict):
        return None
    mode = routing.get("forced_mode")
    if mode in KERNEL_ROUTING_MODES:
        return mode
    return None


def _row_key(row: dict[str, Any], routing_mode: str) -> tuple:
    return (
        routing_mode,
        str(row["phase"]),
        int(row["world_size"]),
        int(row["length"]),
        int(row["batch"]),
        int(row["freq_mhz"]),
    )


def _format_number(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def load_kernel_rows(raw_root: Path, summary: Summary) -> dict[tuple, dict[str, Any]]:
    rows: dict[tuple, dict[str, Any]] = {}
    for path in sorted(raw_root.glob("node*/*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        summary.files_scanned += 1
        with path.open() as handle:
            for line_number, line in enumerate(handle, 1):
                line = line.strip()
                if not line:
                    continue
                summary.rows_read += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ExportError(
                        f"{path}:{line_number}: invalid JSON: {exc}"
                    ) from exc
                if row.get("schema_version") != SCHEMA_VERSION or row.get("status") != "ok":
                    summary.skipped += 1
                    continue
                if row.get("component") != "K" or row.get("parallel_mode") != "moe_ep":
                    summary.skipped += 1
                    continue
                routing_mode = _routing_mode(row)
                if routing_mode is None:
                    summary.skipped += 1
                    continue
                key = _row_key(row, routing_mode)
                if key in rows:
                    summary.duplicates += 1
                    continue
                rows[key] = row
    return rows


def export_matrices(raw_root: Path, output_dir: Path) -> Summary:
    summary = Summary()
    rows = load_kernel_rows(raw_root, summary)
    grouped: dict[tuple[str, str], list[tuple]] = {}
    for key, row in rows.items():
        routing_mode, phase, *_rest = key
        grouped.setdefault((phase, routing_mode), []).append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    for phase, routing_mode in sorted(grouped):
        filename_template, length_col = PHASE_FILES[phase]
        routing_label = ROUTING_ALIASES.get(routing_mode, routing_mode)
        outfile = output_dir / filename_template.format(routing=routing_label)
        header = (
            f"parallel_mode\tsize\t{length_col}\tgpu_clock\tbatch_size\t"
            f"latency_us\tenergy_mj\n"
        )
        lines = [header]
        subset = grouped[(phase, routing_mode)]
        for row in sorted(
            subset,
            key=lambda r: (
                int(r["world_size"]),
                int(r["length"]),
                int(r["batch"]),
                int(r["freq_mhz"]),
            ),
        ):
            lines.append(
                "\t".join(
                    [
                        "ep",
                        str(int(row["world_size"])),
                        str(int(row["length"])),
                        str(int(row["freq_mhz"])),
                        str(int(row["batch"])),
                        _format_number(float(row["latency_us"])),
                        _format_number(float(row["energy_total_mj"])),
                    ]
                )
                + "\n"
            )
        _atomic_write(outfile, lines)
        summary.rows_exported += len(subset)
        print(f"Wrote {outfile} ({len(subset)} rows)")
    print(summary.render())
    return summary


def _atomic_write(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w") as handle:
            for line in lines:
                handle.write(line)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def parse_args() -> argparse.Namespace:
    profile_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=profile_root / "data" / "kernel-EP" / "raw",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=profile_root / "data" / "kernel-EP",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        export_matrices(args.raw_root, args.output_dir)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
