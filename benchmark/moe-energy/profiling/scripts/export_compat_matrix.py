#!/usr/bin/env python3
"""Export qwen3-af-v2 component JSONL into AFlex-compatible TSV matrices."""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

SCHEMA_VERSION = "qwen3-af-v2"
PHASE_PREFIX = {"prefill": "P", "decode": "D"}
LENGTH_COLUMN = {"prefill": "input_len", "decode": "context_len"}


class ExportError(RuntimeError):
    pass


@dataclass
class Summary:
    files: int = 0
    raw_rows: int = 0
    v2_rows: int = 0
    complete_rows: int = 0
    incomplete_pairs: int = 0
    rejected_pairs: int = 0
    duplicate_rows: int = 0
    invalid_rows: int = 0

    def render(self) -> str:
        values = " ".join(f"{key}={value}" for key, value in vars(self).items())
        return f"export-summary {values}"


def _shape_key(row: dict[str, Any]) -> tuple[str, int, int, int, int]:
    return (
        str(row["phase"]),
        int(row["world_size"]),
        int(row["length"]),
        int(row["batch"]),
        int(row["freq_mhz"]),
    )


def _source_key(row: dict[str, Any]) -> tuple[str, str, str, int, int, int, int]:
    return (str(row["component"]), str(row["parallel_mode"]), *_shape_key(row))


def _validate_row(row: dict[str, Any], source: Path, line_number: int) -> None:
    required = {
        "schema_version", "status", "phase", "component", "parallel_mode",
        "world_size", "length", "batch", "freq_mhz", "latency_us",
        "energy_total_mj", "request_seed", "hidden_seed", "attn_tp",
        "moe_tp", "moe_ep",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ExportError(f"{source}:{line_number}: missing fields: {', '.join(missing)}")
    if row["phase"] not in PHASE_PREFIX:
        raise ExportError(f"{source}:{line_number}: invalid phase {row['phase']!r}")
    source_pair = (row["component"], row["parallel_mode"])
    if source_pair not in {("A", "attn_tp"), ("F", "moe_tp"), ("F", "moe_ep")}:
        raise ExportError(f"{source}:{line_number}: invalid component/mode {source_pair!r}")
    world = int(row["world_size"])
    if source_pair == ("A", "attn_tp") and int(row["attn_tp"]) != world:
        raise ExportError(f"{source}:{line_number}: Attention attn_tp != world_size")
    if source_pair == ("F", "moe_tp") and (int(row["moe_tp"]) != world or int(row["moe_ep"]) != 1):
        raise ExportError(f"{source}:{line_number}: F-TP topology is not moe_tp=world, moe_ep=1")
    if source_pair == ("F", "moe_ep") and (int(row["moe_ep"]) != world or int(row["moe_tp"]) != 1):
        raise ExportError(f"{source}:{line_number}: F-EP topology is not moe_ep=world, moe_tp=1")


def load_rows(raw_root: Path, summary: Summary, issues: list[str]) -> dict[tuple, dict[str, Any]]:
    rows: dict[tuple, dict[str, Any]] = {}
    for path in sorted(raw_root.rglob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        summary.files += 1
        with path.open() as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                summary.raw_rows += 1
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    summary.invalid_rows += 1
                    issues.append(f"{path}:{line_number}: invalid JSON: {exc}")
                    continue
                if row.get("schema_version") != SCHEMA_VERSION or row.get("status") != "ok":
                    continue
                try:
                    _validate_row(row, path, line_number)
                    key = _source_key(row)
                except (ExportError, KeyError, TypeError, ValueError) as exc:
                    summary.invalid_rows += 1
                    issues.append(str(exc))
                    continue
                summary.v2_rows += 1
                previous = rows.get(key)
                if previous is not None:
                    summary.duplicate_rows += 1
                    if previous != row:
                        summary.invalid_rows += 1
                        issues.append(f"conflicting duplicate raw rows for {key}")
                    continue
                rows[key] = row
    return rows


def _format_number(value: Any) -> str:
    return str(float(value))


def _atomic_write(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w") as f:
            for line in lines:
                f.write(line)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def export_matrices(raw_root: Path, output_dir: Path, *, strict: bool = False) -> Summary:
    summary = Summary()
    issues: list[str] = []
    rows = load_rows(raw_root, summary, issues)
    groups: dict[tuple[str, str, int], list[tuple[tuple, dict, dict]]] = {}
    shape_phases = {_shape_key(row) for row in rows.values()}
    for phase, size, length, batch, freq in sorted(shape_phases, key=lambda x: (x[0], x[1], x[2], x[3], x[4])):
        a = rows.get(("A", "attn_tp", phase, size, length, batch, freq))
        for family, mode in (("tp", "moe_tp"), ("ep", "moe_ep")):
            f_row = rows.get(("F", mode, phase, size, length, batch, freq))
            if a is None or f_row is None:
                summary.incomplete_pairs += 1
                continue
            seed_a = (a["request_seed"], a["hidden_seed"])
            seed_f = (f_row["request_seed"], f_row["hidden_seed"])
            if seed_a != seed_f:
                summary.rejected_pairs += 1
                issues.append(
                    f"seed mismatch for {family} phase={phase} size={size} length={length} "
                    f"batch={batch} freq={freq}: A={seed_a}, F={seed_f}"
                )
                continue
            groups.setdefault((family, phase, size), []).append(((length, batch, freq), a, f_row))
            summary.complete_rows += 1

    if issues:
        for issue in issues:
            print(f"WARNING: {issue}", file=sys.stderr)
        if strict or summary.rejected_pairs:
            raise ExportError(f"export rejected: {len(issues)} issue(s); {summary.render()}")

    for family in ("tp", "ep"):
        for phase in PHASE_PREFIX:
            sizes = sorted({size for f, p, size in groups if f == family and p == phase})
            for size in sizes:
                prefix = PHASE_PREFIX[phase]
                first = family
                header = (
                    f"{first}\t{LENGTH_COLUMN[phase]}\tgpu_clock\tbatch_size\t"
                    f"{prefix}_A_lat\t{prefix}_F_lat\t{prefix}_A_energy\t{prefix}_F_energy\n"
                )
                lines = [header]
                for (length, batch, freq), a, f_row in sorted(groups[(family, phase, size)], key=lambda x: x[0]):
                    lines.append(
                        "\t".join(
                            [str(size), str(length), str(freq), str(batch),
                             _format_number(a["latency_us"]), _format_number(f_row["latency_us"]),
                             _format_number(a["energy_total_mj"]), _format_number(f_row["energy_total_mj"])]
                        ) + "\n"
                    )
                _atomic_write(output_dir / f"{phase}_test_matrix_{family}{size}.txt", lines)

    print(summary.render())
    return summary


def parse_args() -> argparse.Namespace:
    profile_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", type=Path, default=profile_root / "data" / "raw_v2")
    parser.add_argument("--output-dir", type=Path, default=profile_root / "data" / "compat")
    parser.add_argument("--strict", action="store_true", help="fail on malformed/conflicting raw rows in addition to seed mismatches")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        export_matrices(args.raw_root, args.output_dir, strict=args.strict)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
