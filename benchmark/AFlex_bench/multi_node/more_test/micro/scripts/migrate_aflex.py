#!/usr/bin/env python3
"""Migrate AFlex results from micro_4ds_e2e.json into micro/data/micro_e2e_*.json."""
from __future__ import annotations

import json
import time
from pathlib import Path

import bench_common as BC

SRC = Path(__file__).resolve().parent.parent / "data" / "micro_4ds_e2e.json"


def main() -> None:
    src = json.loads(SRC.read_text())
    aflex = src["results"]["aflex"]
    flat = {"aflex": {}}
    for key, entry in aflex.items():
        flat["aflex"][key] = BC.recompute_percentiles_from_requests(
            BC.recompute_energy_per_total_token(dict(entry), key)
        )

    meta = {
        "aflex_source": str(SRC),
        "aflex_migrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "nodes": {
            "group_a": ["10.252.129.36", "10.252.129.35"],
            "group_b": ["10.252.129.34", "10.252.129.33"],
        },
    }
    out = BC.save_all(flat, meta)
    print(f"Migrated {len(flat['aflex'])} AFlex points -> {', '.join(p.name for p in out)}")


if __name__ == "__main__":
    main()
