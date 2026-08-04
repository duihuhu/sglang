#!/usr/bin/env python3
"""Summarize AFD reshard results using paper-aligned, non-overlapping labels.

Rank/component work is concurrent. Reported critical values are maxima, never sums.
Legacy files remain usable but explicitly mark new instrumentation unavailable.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

METRICS = (
    "local_repartition_s", "local_repartition_bytes",
    "peer_transfer_s", "peer_transfer_bytes",
    "host_stage_d2h_s", "host_stage_d2h_bytes",
    "host_load_remaining_s", "host_load_remaining_bytes",
    "group_prepare_s", "activate_rebuild_groups_s",
    "activate_refresh_scheduler_groups_s", "activate_ensure_shell_s",
    "activate_materialize_s", "materialize_h2d_s", "materialize_h2d_bytes",
    "activate_refresh_runtime_s", "activate_refresh_bootstrap_topology_s",
    "activate_consensus_s", "activate_final_barrier_s",
)

def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)

def _metric(component: Mapping[str, Any], name: str) -> dict[str, Any]:
    ranks = component.get("ranks") or []
    samples = [rank[name] for rank in ranks if isinstance(rank, Mapping) and _number(rank.get(name))]
    critical = component.get("critical") or {}
    if not samples and _number(critical.get(name)):
        samples = [critical[name]]
    if not samples:
        return {"status": "unavailable", "value": None}
    return {"status": "measured", "value": max(samples), "aggregation": "max_across_ranks"}

def summarize(data: Mapping[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "schema_version": 1,
        "semantics": {
            "timings": "rank-local wall-clock; stage/component critical path is max across ranks",
            "bytes": "rank-local stage throughput; value is max across ranks, not cluster traffic; the same payload may be counted once per stage it traverses",
            "composition": "A/F and prefill/decode may overlap; do not sum fields to reconstruct total latency",
            "host_load_remaining": "zero means implemented instrumentation observed no checkpoint/host fallback; unavailable means legacy input",
        },
        "transitions": [],
    }
    for key, transition in data.items():
        if not isinstance(transition, Mapping) or "transition" not in transition:
            continue
        item = {"id": key, "transition": transition.get("transition"), "stages": {}}
        for stage in ("prefill", "decode"):
            status = transition.get(stage) or {}
            breakdown = status.get("breakdown") or {}
            stage_out: dict[str, Any] = {
                "operation_id": status.get("operation_id") or transition.get(f"{stage}_operation_id"),
                "phase": status.get("phase"),
                "wall_clock": {
                    name: ({"status": "measured", "value": breakdown[name]} if _number(breakdown.get(name)) else {"status": "unavailable", "value": None})
                    for name in ("prepare_s", "quiesce_drain_s", "drain_s", "activate_s", "redirect_readiness_s", "retire_s")
                },
                "components": {},
            }
            for component_name in ("attn", "ffn"):
                component = breakdown.get(component_name) or {}
                metrics = {name: _metric(component, name) for name in METRICS}
                host_load = metrics["host_load_remaining_s"]
                if host_load["status"] == "measured" and host_load["value"] == 0:
                    host_load["implementation"] = "not_implemented"
                    metrics["host_load_remaining_bytes"]["implementation"] = "not_implemented"
                stage_out["components"][component_name] = metrics
            item["stages"][stage] = stage_out
        output["transitions"].append(item)
    return output

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="full_sequence_results.json")
    parser.add_argument("--output", "-o", type=Path)
    args = parser.parse_args()
    result = summarize(json.loads(args.input.read_text()))
    text = json.dumps(result, indent=2, sort_keys=False) + "\n"
    if args.output:
        args.output.write_text(text)
    else:
        print(text, end="")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
