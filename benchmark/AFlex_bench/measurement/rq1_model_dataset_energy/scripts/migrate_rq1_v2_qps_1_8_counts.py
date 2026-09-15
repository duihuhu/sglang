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


MIGRATION_ID = "qps_1_2_4_8_request_counts_32_32_64_64_v1"
ACTIVE_REQUEST_COUNTS = {1.0: 32, 2.0: 32, 4.0: 64, 8.0: 64}
RESET_FIELDS = (
    "finished_at",
    "sla_pass",
    "sla",
    "metrics",
    "artifact",
    "failure_class",
)


def snapshot(row: dict, reason: str) -> dict:
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
    return {"reason": reason, "superseded_at": utcnow(), **previous}


def migrate(progress_path: Path) -> dict:
    progress = load(progress_path)
    if progress.get("schema_version") != 2 or progress.get("rq") != "RQ1-v2":
        raise ValueError("progress is not an RQ1-v2 progress file")

    history = progress.setdefault("migrations", [])
    if any(item.get("id") == MIGRATION_ID for item in history):
        return {"status": "already_applied", "reset": 0, "excluded": 0}

    reset = 0
    excluded = 0
    canaries_superseded = 0
    reset_by_status = {}
    for row in progress["runs"].values():
        phase = row.get("phase")
        qps = float(row.get("qps", 0))
        if phase == "formal_v2" and qps not in ACTIVE_REQUEST_COUNTS:
            row.setdefault("superseded_results", []).append(
                snapshot(row, "QPS removed from the active RQ1-v2 matrix")
            )
            row["previous_status"] = row.get("status")
            row["status"] = "superseded"
            row["superseded_reason"] = "active QPS set changed to [1, 2, 4, 8]"
            excluded += 1
            continue

        if phase == "formal_v2" and qps in ACTIVE_REQUEST_COUNTS:
            if row.get("status") == "pending":
                continue
            row.setdefault("superseded_results", []).append(
                snapshot(
                    row,
                    "Request count changed for the active QPS point",
                )
            )
            old_status = str(row.get("status"))
            reset_by_status[old_status] = reset_by_status.get(old_status, 0) + 1
            row["status"] = "pending"
            for field in RESET_FIELDS:
                row.pop(field, None)
            reset += 1
            continue

        if phase == "canary_v2" and (
            qps != 1.0 or row.get("workload") != "balanced_mpmd"
        ):
            row.setdefault("superseded_results", []).append(
                snapshot(row, "Canary QPS changed to 1")
            )
            row["previous_status"] = row.get("status")
            row["status"] = "superseded"
            row["superseded_reason"] = "active canary changed to QPS 1"
            canaries_superseded += 1

    migrated_at = utcnow()
    history.append(
        {
            "id": MIGRATION_ID,
            "at": migrated_at,
            "active_request_counts": {
                str(int(qps)): count
                for qps, count in ACTIVE_REQUEST_COUNTS.items()
            },
            "reset": reset,
            "reset_by_status": reset_by_status,
            "excluded": excluded,
            "canaries_superseded": canaries_superseded,
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
        "excluded": excluded,
        "canaries_superseded": canaries_superseded,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate RQ1-v2 to QPS 1/2/4/8 with 32/32/64/64 requests"
    )
    parser.add_argument("--results", type=Path, default=ROOT / "results/v2")
    args = parser.parse_args()
    print(json.dumps(migrate(args.results / "progress.json"), indent=2))


if __name__ == "__main__":
    main()
