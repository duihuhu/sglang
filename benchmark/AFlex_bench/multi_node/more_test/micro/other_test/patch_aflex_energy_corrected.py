#!/usr/bin/env python3
"""Patch AFlex results from energy-corrected rerun into micro_e2e_*.json."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MICRO_DATA = HERE.parent / "data"
SRC = HERE / "data" / "aflex_all_energy_corrected.json"

DEFAULT_PATCHES = {
    "qa_lpld": MICRO_DATA / "micro_e2e_qa_lpld.json",
    "chatbot_lphd": MICRO_DATA / "micro_e2e_chatbot_lphd.json",
    "rag_hpld": MICRO_DATA / "micro_e2e_rag_hpld.json",
    "summary_hphd": MICRO_DATA / "micro_e2e_summary_hphd.json",
}

NODE1 = "10.252.129.36"
NODE2 = "10.252.129.35"
QPS_LIST = [2, 4, 8, 16]


def percentile(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    return round(float(np.percentile(vals, p)), 1)


def router_urls_from_routes(route_distribution: dict) -> list[str]:
    urls = sorted({url.replace("/generate", "") for url in route_distribution})
    return urls


def build_config(old_config: dict, new_config: dict, qps: int) -> dict:
    cfg = copy.deepcopy(old_config)
    cfg.update(new_config)
    cfg["qps"] = qps
    cfg.setdefault("tier", True)
    cfg.setdefault("topology_source", "optimal")
    return cfg


def build_deploy(old_deploy: dict, full_config: dict, router_urls: list[str]) -> dict:
    deploy = copy.deepcopy(old_deploy)
    deploy["config"] = copy.deepcopy(full_config)
    deploy["router_urls"] = router_urls
    deploy["node1"] = NODE1
    deploy["node2"] = NODE2
    deploy["energy_scope"] = "active_gpus_only"
    return deploy


def build_aflex_entry(old_aflex: dict, new_entry: dict, qps: int) -> dict:
    reqs = new_entry.get("request_results", [])
    ttft_vals = [r["ttft_proc_ms"] for r in reqs if r.get("success")]
    tpot_vals = [r["tpot_ms"] for r in reqs if r.get("success")]
    router_urls = router_urls_from_routes(new_entry.get("route_distribution", {}))

    old_config = old_aflex.get("config", {})
    full_config = build_config(old_config, new_entry.get("config", {}), qps)
    old_deploy = old_aflex.get("deploy", {})
    deploy = build_deploy(old_deploy, full_config, router_urls)

    copy_keys = [
        "status",
        "duration_s",
        "total_requests",
        "successful",
        "failed",
        "timed_out",
        "missing",
        "total_tokens",
        "total_input_tokens",
        "total_tokens_all",
        "throughput_tok_s",
        "ttft_proc_avg_ms",
        "ttft_proc_p50_ms",
        "ttft_proc_p99_ms",
        "tpot_avg_ms",
        "tpot_p50_ms",
        "tpot_p99_ms",
        "energy_node1_j",
        "energy_node2_j",
        "energy_extra_j",
        "total_energy_j",
        "energy_per_token_mj",
        "energy_denominator",
        "slo_violation_rate",
        "slo_violating_requests",
        "unsuccessful_requests",
        "route_distribution",
        "assigned_route_distribution",
        "request_results",
        "freq_timeline",
    ]

    result: dict = {k: copy.deepcopy(new_entry[k]) for k in copy_keys if k in new_entry}
    result["energy_scope"] = new_entry.get("energy_scope", "active_gpus_only")
    result["ttft_proc_p90_ms"] = percentile(ttft_vals, 90)
    result["ttft_proc_p95_ms"] = percentile(ttft_vals, 95)
    result["tpot_p90_ms"] = percentile(tpot_vals, 90)
    result["tpot_p95_ms"] = percentile(tpot_vals, 95)

    for key in ("run_window", "provenance", "source", "solver_version"):
        if key in old_aflex:
            result[key] = copy.deepcopy(old_aflex[key])

    result["router_urls"] = router_urls
    result["config"] = full_config
    result["deploy"] = deploy
    result["energy_corrected_patch"] = {
        "source": "other_test/data/aflex_all_energy_corrected.json",
        "patched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "node_pair": f"{NODE1}+{NODE2}",
        "energy_scope": "active_gpus_only",
    }
    return result


def patch_dataset(dataset: str, target: Path, src_results: dict) -> None:
    data = json.loads(target.read_text())
    patched_qps: list[int] = []

    for qps in QPS_LIST:
        src_key = f"{dataset}_qps{qps}"
        dst_key = f"qps_{qps}"
        if src_key not in src_results:
            raise KeyError(f"missing source result: {src_key}")
        old_aflex = data["results"][dataset][dst_key]["aflex"]
        data["results"][dataset][dst_key]["aflex"] = build_aflex_entry(
            old_aflex, src_results[src_key], qps
        )
        patched_qps.append(qps)
        entry = data["results"][dataset][dst_key]["aflex"]
        print(
            f"  {dataset} qps={qps}: E/tok={entry['energy_per_token_mj']:.1f} mJ, "
            f"TTFT_p90={entry['ttft_proc_p90_ms']:.1f} ms, "
            f"TPOT_p90={entry['tpot_p90_ms']:.1f} ms, "
            f"reqs={len(entry['request_results'])}"
        )

    data.setdefault("deploy", {}).setdefault("aflex", {})
    data["deploy"]["aflex"].update(
        {
            "label": "AFlex",
            "topology": "tier1_afd",
            "tier": True,
            "deploy_policy": "solver_per_qps",
            "layout_source": "tier1_selected_solutions",
            "node1": NODE1,
            "node2": NODE2,
            "energy_scope": "active_gpus_only",
        }
    )

    data["meta"]["aflex_energy_corrected_patch"] = {
        "source": "other_test/data/aflex_all_energy_corrected.json",
        "node_pair": f"{NODE1}+{NODE2}",
        "patched_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "qps": patched_qps,
        "energy_scope": "active_gpus_only",
    }
    data["meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")

    target.write_text(json.dumps(data, indent=2) + "\n")
    print(f"Wrote {target}")


def main() -> None:
    global NODE2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default="qa_lpld,chatbot_lphd",
        help="Comma-separated dataset keys to patch",
    )
    parser.add_argument("--node2", default=NODE2, help="Node2 IP used in the benchmark run")
    args = parser.parse_args()
    NODE2 = args.node2

    src = json.loads(SRC.read_text())
    src_results = src["results"]

    for dataset in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        target = DEFAULT_PATCHES.get(dataset)
        if target is None:
            raise SystemExit(f"Unknown dataset: {dataset}")
        print(f"\nPatching {dataset} -> {target.name} (node_pair={NODE1}+{NODE2})")
        patch_dataset(dataset, target, src_results)


if __name__ == "__main__":
    main()
