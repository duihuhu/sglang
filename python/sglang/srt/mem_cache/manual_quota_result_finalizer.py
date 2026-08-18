"""Finalize deferred manual quota outcomes from pressure logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists() or p.stat().st_size == 0:
        return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def _write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def _audit_by_intent(audit_rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    grouped: dict[int, dict[str, Any]] = {}
    for row in audit_rows:
        intent_id = int(row["intent_id"])
        slot = grouped.setdefault(intent_id, {})
        kind = str(row["kind"])
        if kind == "intent_received":
            slot["intent_received"] = row
        elif kind == "allocator_ack":
            slot["allocator_ack"] = row
        elif kind == "effective_quota_changed":
            slot["effective_quota_changed"] = row
    return grouped


def _first_backup_after_effective(
    pressure_rows: list[dict[str, Any]], *, effective_ns: int, effective_pages: int
) -> dict[str, Any] | None:
    for row in pressure_rows:
        monotonic_time_s = row.get("monotonic_time_s")
        if monotonic_time_s is None:
            continue
        row_ns = int(float(monotonic_time_s) * 1_000_000_000)
        if row_ns < effective_ns:
            continue
        if int(row.get("active_quota_pages", 0)) < effective_pages:
            continue
        backup_slots = int(row.get("backup_slots", 0))
        if backup_slots <= 0:
            continue
        page_size = int(row.get("page_size_tokens", 1))
        return {
            "monotonic_time_s": float(monotonic_time_s),
            "time_ns": row_ns,
            "backup_slots": backup_slots,
            "backup_pages": (backup_slots + page_size - 1) // page_size,
            "active_quota_pages": int(row.get("active_quota_pages", 0)),
            "alloc_pages": int(row.get("alloc_pages", 0)),
            "live_pages": int(row.get("live_pages", 0)),
        }
    return None


def finalize_manual_results(
    *,
    result_jsonl: str | Path,
    audit_jsonl: str | Path,
    pressure_logs: dict[str, str | Path],
    output_jsonl: str | Path,
) -> list[dict[str, Any]]:
    results = _read_jsonl(result_jsonl)
    audits = _audit_by_intent(_read_jsonl(audit_jsonl))
    pressure_by_model = {
        model_id: _read_jsonl(path) for model_id, path in pressure_logs.items()
    }
    finalized: list[dict[str, Any]] = []
    for record in results:
        row = dict(record)
        manager_intent_id = int(row["manager_intent_id"])
        audit = audits.get(manager_intent_id, {})
        intent = audit.get("intent_received")
        ack = audit.get("allocator_ack")
        effective = audit.get("effective_quota_changed")
        if effective is not None:
            model_id = str(effective["model_id"])
            effective_ns = int(effective["time_ns"])
            effective_pages = int(effective["pages"])
            row["effective_quota"] = {
                "model_id": model_id,
                "pages": effective_pages,
                "time_ns": effective_ns,
            }
            if intent is not None:
                row["intent_received_ns"] = int(intent["time_ns"])
            if ack is not None:
                row["allocator_ack_ns"] = int(ack["time_ns"])
                if intent is not None:
                    row["intent_to_ack_s"] = (
                        int(ack["time_ns"]) - int(intent["time_ns"])
                    ) / 1_000_000_000
            backup = _first_backup_after_effective(
                pressure_by_model.get(model_id, []),
                effective_ns=effective_ns,
                effective_pages=effective_pages,
            )
            if backup is not None:
                row["first_actual_backup"] = {
                    "observer": backup,
                    "effective_to_first_backup_s": (
                        int(backup["time_ns"]) - effective_ns
                    )
                    / 1_000_000_000,
                }
                if intent is not None:
                    row["t_usable_s"] = (
                        int(backup["time_ns"]) - int(intent["time_ns"])
                    ) / 1_000_000_000
        finalized.append(row)
    _write_jsonl(output_jsonl, finalized)
    return finalized


def _parse_pressure_log(values: list[str]) -> dict[str, str]:
    logs: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError("--pressure-log must be MODEL_ID=PATH")
        model_id, path = value.split("=", 1)
        if not model_id or not path:
            raise ValueError("--pressure-log must be MODEL_ID=PATH")
        logs[model_id] = path
    return logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Finalize deferred manual quota result JSONL from pressure logs."
    )
    parser.add_argument("--result-jsonl", required=True)
    parser.add_argument("--audit-jsonl", required=True)
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--pressure-log", action="append", default=[])
    args = parser.parse_args()
    finalize_manual_results(
        result_jsonl=args.result_jsonl,
        audit_jsonl=args.audit_jsonl,
        pressure_logs=_parse_pressure_log(args.pressure_log),
        output_jsonl=args.output_jsonl,
    )


if __name__ == "__main__":
    main()
