#!/usr/bin/env python3
"""Build standalone node_scalability_all.json for energy chart only."""
from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent / "data"
OUT = DATA_DIR / "node_scalability_all.json"

SCHEMES = ["sglang", "dynamollm", "distserve", "biscale", "aflex"]
NODE_QPS = {1: 8, 2: 16, 4: 32}

# Deploy topology labels per (scheme, nodes). QPS follows NODE_QPS.
DEPLOY: dict[tuple[str, int], dict] = {
    ("sglang", 1): {
        "scheme": "SGLang",
        "topology": "8×TP1",
        "nodes": 1,
        "gpus": 8,
        "freq": "max lock 1410MHz",
    },
    ("sglang", 2): {
        "scheme": "SGLang",
        "topology": "16×TP1",
        "nodes": 2,
        "gpus": 16,
        "freq": "max lock 1410MHz",
    },
    ("sglang", 4): {
        "scheme": "SGLang",
        "topology": "16×TP2",
        "nodes": 4,
        "gpus": 32,
        "freq": "max lock 1410MHz",
    },
    ("dynamollm", 1): {
        "scheme": "DynamoLLM",
        "topology": "8×TP1",
        "nodes": 1,
        "gpus": 8,
        "freq": "unified DVFS",
    },
    ("dynamollm", 2): {
        "scheme": "DynamoLLM",
        "topology": "16×TP1",
        "nodes": 2,
        "gpus": 16,
        "freq": "unified DVFS",
    },
    ("dynamollm", 4): {
        "scheme": "DynamoLLM",
        "topology": "8×TP4",  # code; conv overridden to 16×TP2 in _slim()
        "nodes": 4,
        "gpus": 32,
        "freq": "unified DVFS",
    },
    ("distserve", 1): {
        "scheme": "DistServe",
        "topology": "4P(TP1)+4D(TP1)",
        "nodes": 1,
        "gpus": 8,
        "freq": "max lock 1410MHz",
    },
    ("distserve", 2): {
        "scheme": "DistServe",
        "topology": "2P(TP4)+4D(TP2)",
        "nodes": 2,
        "gpus": 16,
        "freq": "max lock 1410MHz",
    },
    ("distserve", 4): {
        "scheme": "DistServe",
        "topology": "2×[2P(TP4)+4D(TP2)]",
        "nodes": 4,
        "gpus": 32,
        "freq": "max lock 1410MHz",
    },
    ("biscale", 1): {
        "scheme": "BiScale",
        "topology": "4P(TP1)+4D(TP1)",
        "nodes": 1,
        "gpus": 8,
        "freq": "BiScale DVFS",
    },
    ("biscale", 2): {
        "scheme": "BiScale",
        "topology": "2P(TP4)+4D(TP2)",
        "nodes": 2,
        "gpus": 16,
        "freq": "BiScale DVFS",
    },
    ("biscale", 4): {
        "scheme": "BiScale",
        "topology": "2×[2P(TP4)+4D(TP2)]",
        "nodes": 4,
        "gpus": 32,
        "freq": "BiScale DVFS",
    },
    ("aflex", 1): {
        "scheme": "AFlex",
        "topology": "3P+1D ipc_cpp",
        "nodes": 1,
        "gpus": 8,
        "freq": "tier AFD DVFS",
    },
    ("aflex", 2): {
        "scheme": "AFlex",
        "topology": "Tier1 AFD (plan_dense)",
        "nodes": 2,
        "gpus": 16,
        "freq": "tier AFD DVFS",
    },
    ("aflex", 4): {
        "scheme": "AFlex",
        "topology": "4×[3P+1D ipc_cpp]",
        "nodes": 4,
        "gpus": 32,
        "freq": "tier AFD DVFS",
    },
}

# Source files (read once, then discarded)
SOURCES = {
    ("code", 1): ("8gpu_six_schemes_code_qps8.json", "code_qps8"),
    ("code", 2): ("16gpu_six_schemes_code_qps16.json", "code_qps16"),
    ("code", 4): ("32gpu_six_schemes.json", "code_qps32"),
    ("conv", 1): ("8gpu_six_schemes_conv_qps8.json", "conv_qps8"),
    ("conv", 2): ("16gpu_six_schemes_conv_qps16.json", "conv_qps16"),
    ("conv", 4): ("32gpu_six_schemes.json", "conv_qps32"),
}

# DynamoLLM @ 4 nodes: code=8×TP4, conv=16×TP2


def _slim(entry: dict, dataset: str, nodes: int, scheme: str) -> dict:
    qps = NODE_QPS[nodes]
    wl_key = f"{dataset}_qps{qps}"
    deploy = deepcopy(DEPLOY[(scheme, nodes)])
    if scheme == "dynamollm" and nodes == 4:
        deploy["topology"] = "8×TP4" if dataset == "code" else "16×TP2"
    out = {
        "status": entry.get("status", "PASS"),
        "workload": wl_key,
        "dataset": dataset,
        "qps": qps,
        "nodes": nodes,
        "gpus": deploy["gpus"],
        "deploy": deploy,
        "throughput_tok_s": entry.get("throughput_tok_s"),
        "energy_per_token_mj": entry.get("energy_per_token_mj"),
        "total_energy_j": entry.get("total_energy_j"),
        "total_input_tokens": entry.get("total_input_tokens"),
        "total_tokens_all": entry.get("total_tokens_all"),
        "energy_denominator": entry.get("energy_denominator", "input_plus_output"),
        "ttft_proc_p50_ms": entry.get("ttft_proc_p50_ms"),
        "tpot_p50_ms": entry.get("tpot_p50_ms"),
        "total_requests": entry.get("total_requests"),
        "successful": entry.get("successful"),
        "slo_violations": entry.get("slo_violations", 0),
    }
    return {k: v for k, v in out.items() if v is not None}


def build_from_existing(data_dir: Path) -> dict:
    results: dict = {"code": {}, "conv": {}}
    for (dataset, nodes), (fname, wl_key) in SOURCES.items():
        src = json.loads((data_dir / fname).read_text())["results"]
        node_bucket: dict = {}
        for scheme in SCHEMES:
            if scheme not in src or wl_key not in src[scheme]:
                continue
            node_bucket[scheme] = _slim(src[scheme][wl_key], dataset, nodes, scheme)
        results[dataset][str(nodes)] = node_bucket
    return {
        "meta": {
            "benchmark": "node_scalability_energy",
            "version": 1,
            "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "description": "Energy per token vs node count (5 schemes × code/conv)",
            "datasets": ["code", "conv"],
            "nodes": [1, 2, 4],
            "node_qps": {"1": 8, "2": 16, "4": 32},
            "schemes": SCHEMES,
            "energy_denominator": "input_plus_output",
        },
        "results": results,
    }


def main() -> None:
    payload = build_from_existing(DATA_DIR)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
