#!/usr/bin/env python3
"""Normalize Energy/Token to input+output total tokens (macro trace workloads)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

AFLEX_ROOT = Path(__file__).resolve().parent.parent
WORKLOAD_DIR = AFLEX_ROOT / "multi_node/more_test/macro/data/workloads"


def parse_workload_key(key: str) -> tuple[str, int]:
    dataset, qps_s = key.rsplit("_qps", 1)
    return dataset, int(qps_s)


@lru_cache(maxsize=None)
def workload_token_totals(workload_key: str) -> tuple[int, int, int]:
    """Return (input_tokens, output_tokens, all_tokens) for a macro workload key."""
    dataset, qps = parse_workload_key(workload_key)
    path = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"missing workload file: {path}")
    reqs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    total_input = sum(r["input_len"] for r in reqs)
    total_output = sum(r["output_len"] for r in reqs)
    return total_input, total_output, total_input + total_output


def energy_per_token_mj_all_tokens(entry: dict, workload_key: str) -> float:
    """Convert stored Energy/Token to mJ/token with input+output denominator."""
    if not isinstance(entry, dict):
        return 0.0
    _, total_output, total_all = workload_token_totals(workload_key)
    if total_all <= 0:
        return 0.0
    total_energy_j = entry.get("total_energy_j")
    if total_energy_j is not None:
        return total_energy_j * 1000.0 / total_all
    old_mj = entry.get("energy_per_token_mj")
    if old_mj is None:
        return 0.0
    output_tokens = entry.get("total_tokens", total_output)
    return old_mj * output_tokens / total_all


def energy_per_token_j_all_tokens(entry: dict, workload_key: str) -> float:
    return energy_per_token_mj_all_tokens(entry, workload_key) / 1000.0
