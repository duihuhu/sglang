#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))

from rq1lib_v2 import atomic_json, load, utcnow


MIGRATION_ID = "qps0.25_request_count_64_v1"
RESET_FIELDS = (
    "finished_at",
    "sla_pass",
    "sla",
    "metrics",
    "artifact",
    "failure_class",
)


def migrate(progress_path: Path) -> dict:
    progress = load(progress_path)
    if progress.get("schema_version") != 2 or progress.get("rq") != "RQ1-v2":
        raise ValueError("progress is not an RQ1-v2 progress file")

    history = progress.setdefault("migrations", [])
    if any(item.get("id") == MIGRATION_ID for item in history):
        return {"status": "already_applied", "reset": 0}

    reset = 0
    reset_by_status = {}
    for row in progress["runs"].values():
        if row.get("phase") != "formal_v2" or float(row.get("qps", 0)) != 0.25:
            continue
        if row.get("status") == "pending":
            continue

        previous = {
            key: row[key]
            for key in (
                "status",
                "artifact",
                "finished_at",
                "sla_pass",
                "sla",
                "metrics",
                "failure_class",
            )
            if key in row
        }
        row.setdefault("superseded_results", []).append(
            {
                "reason": "QPS 0.25 request count changed from 256 to 64",
                "superseded_at": utcnow(),
                **previous,
            }
        )
        old_status = str(row.get("status"))
        reset_by_status[old_status] = reset_by_status.get(old_status, 0) + 1
        row["status"] = "pending"
        for field in RESET_FIELDS:
            row.pop(field, None)
        reset += 1

    migrated_at = utcnow()
    history.append(
        {
            "id": MIGRATION_ID,
            "at": migrated_at,
            "qps": 0.25,
            "old_request_count": 256,
            "new_request_count": 64,
            "reset": reset,
            "reset_by_status": reset_by_status,
        }
    )
    progress["updated_at"] = migrated_at
    progress.pop("pause_reason", None)
    progress.pop("paused_at", None)
    atomic_json(progress_path, progress)
    return {
        "status": "applied",
        "reset": reset,
        "reset_by_status": reset_by_status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset completed RQ1-v2 QPS 0.25 points for the 64-request trace"
    )
    parser.add_argument(
        "--results", type=Path, default=ROOT / "results/v2"
    )
    args = parser.parse_args()
    print(json.dumps(migrate(args.results / "progress.json"), indent=2))


if __name__ == "__main__":
    main()
