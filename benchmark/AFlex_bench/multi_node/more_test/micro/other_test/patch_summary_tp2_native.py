#!/usr/bin/env python3
"""Patch SGLang/DynamoLLM TP2 native results into micro_e2e_summary_hphd.json."""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TARGET = HERE.parent / "data" / "micro_e2e_summary_hphd.json"
SRC = HERE / "data" / "summary_tp2_native.json"

DATASET = "summary_hphd"
QPS_LIST = [2, 4, 8, 16]

# Keep micro chart keys; replace with TP2 data.
SCHEME_MAP = {
    "native_tp2_baseline": "native_tp1_baseline",
    "native_tp2_tier": "native_tp1_tier",
}

DEPLOY_TEMPLATES = {
    "native_tp1_baseline": {
        "label": "SGLang",
        "topology": "native_tp2",
        "tier": False,
        "gpus": 16,
        "nodes": 2,
        "tp": 2,
        "instances": 8,
        "freq_policy": "locked_1410mhz",
        "deploy_policy": "restart_per_point",
    },
    "native_tp1_tier": {
        "label": "DynamoLLM",
        "topology": "native_tp2",
        "tier": True,
        "gpus": 16,
        "nodes": 2,
        "tp": 2,
        "instances": 8,
        "freq_policy": "unified_dvfs",
        "deploy_policy": "restart_per_point",
    },
}


def build_entry(src_entry: dict, dst_scheme: str, qps: int) -> dict:
    entry = copy.deepcopy(src_entry)
    src_deploy = src_entry.get("deploy", {})
    deploy = copy.deepcopy(DEPLOY_TEMPLATES[dst_scheme])
    deploy["node1"] = src_deploy.get("node1")
    deploy["node2"] = src_deploy.get("node2")
    deploy["router_url"] = src_deploy.get("router_url")
    deploy["qps"] = qps
    entry["deploy"] = deploy
    entry["tp2_native_patch"] = {
        "source": "other_test/data/summary_tp2_native.json",
        "patched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "src_scheme": "native_tp2_baseline" if dst_scheme == "native_tp1_baseline" else "native_tp2_tier",
        "node_pair": f"{src_deploy.get('node1')}+{src_deploy.get('node2')}",
    }
    return entry


def main() -> None:
    src = json.loads(SRC.read_text())
    data = json.loads(TARGET.read_text())
    src_results = src["results"]
    patched_qps: list[int] = []

    for src_scheme, dst_scheme in SCHEME_MAP.items():
        for qps in QPS_LIST:
            src_key = f"{DATASET}_qps{qps}"
            dst_key = f"qps_{qps}"
            if src_key not in src_results[src_scheme]:
                raise KeyError(f"missing {src_scheme}/{src_key}")
            entry = build_entry(src_results[src_scheme][src_key], dst_scheme, qps)
            data["results"][DATASET][dst_key][dst_scheme] = entry
            patched_qps.append(qps)
            print(
                f"  {dst_scheme} qps={qps}: E/tok={entry['energy_per_token_mj']:.1f} mJ, "
                f"TTFT_p90={entry['ttft_proc_p90_ms']:.1f} ms, "
                f"TPOT_p90={entry['tpot_p90_ms']:.1f} ms, "
                f"reqs={len(entry.get('request_results', []))}"
            )

    for dst_scheme, tpl in DEPLOY_TEMPLATES.items():
        deploy = data["deploy"].setdefault(dst_scheme, {})
        deploy.update(copy.deepcopy(tpl))
        deploy["node1"] = src["meta"]["node1"]
        deploy["node2"] = src["meta"]["node2"]

    data["meta"]["tp2_native_patch"] = {
        "source": "other_test/data/summary_tp2_native.json",
        "node_pair": f"{src['meta']['node1']}+{src['meta']['node2']}",
        "patched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qps": QPS_LIST,
        "schemes": list(SCHEME_MAP.values()),
        "note": "SGLang/DynamoLLM replaced with native TP2 results (scheme keys unchanged for charts)",
    }
    data["meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    TARGET.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {TARGET}")


if __name__ == "__main__":
    main()
