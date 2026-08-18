"""Build D-window observations from completed client rows and pressure logs.

This module is an observation adapter only.  It does not decide quota and it
does not talk to Central I/O.
"""

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


def _read_jsonl_many(paths: str | Path) -> list[dict[str, Any]]:
    if isinstance(paths, Path):
        return _read_jsonl(paths)
    rows: list[dict[str, Any]] = []
    for item in str(paths).split(","):
        item = item.strip()
        if item:
            rows.extend(_read_jsonl(item))
    return rows


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def _run_wall_zero_s(client_rows: list[dict[str, Any]]) -> float | None:
    anchors = [
        float(row["wall_dispatch_time_s"]) - float(row["scheduled_time_s"])
        for row in client_rows
        if row.get("wall_dispatch_time_s") is not None
        and row.get("scheduled_time_s") is not None
    ]
    if not anchors:
        return None
    return min(anchors)


def _pressure_rows_in_window(
    pressure_rows: list[dict[str, Any]],
    *,
    run_wall_zero_s: float | None,
    window_start_s: float,
    window_end_s: float,
) -> tuple[list[dict[str, Any]], str]:
    if run_wall_zero_s is None:
        return pressure_rows, "unfiltered_missing_client_wall_anchor"
    timed_rows = [
        row
        for row in pressure_rows
        if row.get("wall_time_s") is not None
        and window_start_s
        <= float(row["wall_time_s"]) - run_wall_zero_s
        < window_end_s
    ]
    if timed_rows or all(row.get("wall_time_s") is not None for row in pressure_rows):
        return timed_rows, "wall_time_window"
    return pressure_rows, "unfiltered_missing_pressure_wall_time"


def _previous_prompt_tokens_by_session(
    client_rows: list[dict[str, Any]],
) -> dict[tuple[int, int], int]:
    previous_by_session: dict[int, int] = {}
    result: dict[tuple[int, int], int] = {}
    ordered_rows = sorted(
        client_rows,
        key=lambda row: (
            int(row.get("main_request_id", -1)),
            int(row.get("sub_request_id", -1)),
            float(row.get("scheduled_time_s", 0.0)),
        ),
    )
    for row in ordered_rows:
        session_id = int(row.get("main_request_id", -1))
        turn_id = int(row.get("sub_request_id", -1))
        prompt_tokens = int(row.get("prompt_tokens") or 0)
        if session_id in previous_by_session:
            result[(session_id, turn_id)] = previous_by_session[session_id]
        previous_by_session[session_id] = prompt_tokens
    return result


def _report_for_model(
    *,
    client_jsonl: str | Path,
    pressure_jsonl: str | Path,
    window_start_s: float,
    window_end_s: float,
) -> dict[str, Any]:
    all_client_rows = [row for row in _read_jsonl_many(client_jsonl) if row.get("success")]
    previous_prompt_tokens = _previous_prompt_tokens_by_session(all_client_rows)
    client_rows = [
        row
        for row in all_client_rows
        if window_start_s <= float(row.get("scheduled_time_s", -1.0)) < window_end_s
    ]
    revisit_rows = [
        row for row in client_rows if int(row.get("sub_request_id", 0)) > 0
    ]
    pressure_rows_all = _read_jsonl(pressure_jsonl)
    pressure_rows, pressure_filter = _pressure_rows_in_window(
        pressure_rows_all,
        run_wall_zero_s=_run_wall_zero_s(all_client_rows),
        window_start_s=window_start_s,
        window_end_s=window_end_s,
    )
    raw_retention_loss = sum(
        int(row.get("retention_loss_tokens", 0)) for row in pressure_rows
    )
    opportunity = sum(int(row.get("prompt_tokens") or 0) for row in revisit_rows)
    cache_match = sum(int(row.get("cached_tokens") or 0) for row in revisit_rows)
    cache_miss = sum(
        max(0, int(row.get("prompt_tokens") or 0) - int(row.get("cached_tokens") or 0))
        for row in revisit_rows
    )
    incremental = 0
    seen_cache_miss = 0
    for row in revisit_rows:
        session_id = int(row.get("main_request_id", -1))
        turn_id = int(row.get("sub_request_id", -1))
        prompt_tokens = int(row.get("prompt_tokens") or 0)
        cached_tokens = int(row.get("cached_tokens") or 0)
        previous_tokens = previous_prompt_tokens.get((session_id, turn_id), 0)
        new_tokens = max(0, prompt_tokens - previous_tokens)
        incremental += new_tokens
        seen_cache_miss += max(0, min(previous_tokens, prompt_tokens) - cached_tokens)
    d_raw = min(raw_retention_loss, seen_cache_miss)
    reprefill = d_raw
    ttft_ms = [
        float(row["ttft_s"]) * 1000.0
        for row in revisit_rows
        if row.get("ttft_s") is not None
    ]
    return {
        "window_start_s": window_start_s,
        "window_end_s": window_end_s,
        "raw_retention_loss_tokens": raw_retention_loss,
        "confirmed_reuse_loss_tokens": d_raw,
        "revisit_opportunity_tokens": opportunity,
        "reprefill_tokens": reprefill,
        "revisit_cache_match_tokens": cache_match,
        "revisit_cache_miss_tokens": cache_miss,
        "revisit_incremental_tokens": incremental,
        "revisit_seen_cache_miss_tokens": seen_cache_miss,
        "d_coverage_of_seen_miss": (
            None if seen_cache_miss == 0 else d_raw / seen_cache_miss
        ),
        "conditional_ttft_ms_p50": _percentile(ttft_ms, 0.5),
        "host_evicted_tokens": sum(
            int(row.get("host_evict_slots", 0)) for row in pressure_rows
        ),
        "admission_shortfall_pages": sum(
            int(row.get("admission_shortfall_pages", 0)) for row in pressure_rows
        ),
        "completed_requests": len(client_rows),
        "completed_revisit_requests": len(revisit_rows),
        "pressure_rows": len(pressure_rows),
        "pressure_rows_total": len(pressure_rows_all),
        "pressure_filter": pressure_filter,
    }


def build_window(
    *,
    decision_window_id: str,
    window_start_s: float,
    window_end_s: float,
    models: dict[str, dict[str, str | Path]],
) -> dict[str, Any]:
    if window_end_s <= window_start_s:
        raise ValueError("window_end_s must be greater than window_start_s")
    return {
        "decision_window_id": decision_window_id,
        "reports": {
            model_id: _report_for_model(
                client_jsonl=paths["client_jsonl"],
                pressure_jsonl=paths["pressure_jsonl"],
                window_start_s=window_start_s,
                window_end_s=window_end_s,
            )
            for model_id, paths in models.items()
        },
    }


def _parse_model_arg(values: list[str]) -> dict[str, dict[str, str]]:
    models = {}
    for value in values:
        parts = value.split("=")
        if len(parts) != 3:
            raise ValueError("--model must be MODEL_ID=CLIENT_JSONL=PRESSURE_JSONL")
        model_id, client_jsonl, pressure_jsonl = parts
        models[model_id] = {
            "client_jsonl": client_jsonl,
            "pressure_jsonl": pressure_jsonl,
        }
    return models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decision-window-id", required=True)
    parser.add_argument("--window-start-s", type=float, required=True)
    parser.add_argument("--window-end-s", type=float, required=True)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    window = build_window(
        decision_window_id=args.decision_window_id,
        window_start_s=args.window_start_s,
        window_end_s=args.window_end_s,
        models=_parse_model_arg(args.model),
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps([window], indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
