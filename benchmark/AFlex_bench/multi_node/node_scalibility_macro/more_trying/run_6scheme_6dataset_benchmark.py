#!/usr/bin/env python3
"""Clean 6-scheme x 6-dataset benchmark (QPS 2/8/16, dataset-first).

Schemes:
  SGLang, DynamoLLM, DistServe, BiScale  — fixed topology
  MegaScale, AFlex                     — Tier1 ILP optimal layout per (dataset, qps);
                                         MegaScale = lock 1410 MHz; AFlex = compositional DVFS

SLO: TTFT=5s, TPOT=300ms
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("6scheme_6ds")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(HERE))

import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB
from run_biscale_pd_hetero_code import start_pd_hetero_biscale
from deploy_tier1_layout import deploy_tier1_layout
from tier1_layout_utils import layout_for, layout_meta
import run_fixed_6scheme_7dataset as F67
from run_fixed_6scheme_7dataset import (
    NGPU,
    run_one_dataset_qps,
    _is_pass,
    _wl_key,
    _workload_file,
)

_orig_afd_common = RMB._afd_common


def _patched_afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    result = _orig_afd_common(tp, ib_dev, gpu_step, tier, ngpu)
    result = result.replace("--afd-micro-batch 2", "--afd-micro-batch 1")
    result = result.replace("--afd-dynamic-micro-batch", "")
    if tier:
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
    return result


RMB._afd_common = _patched_afd_common
RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

DEFAULT_DATASETS = [
    "code", "conv",
    "qa_lpld", "chatbot_lphd", "balanced_mpmd", "summary_hphd",
]
DEFAULT_QPS = [2, 8, 16]
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# scheme_key -> display label
LABELS = {
    "native_tp1_baseline": "SGLang",
    "native_tp1_tier": "DynamoLLM",
    "pd_hetero_baseline": "DistServe",
    "pd_hetero_tier_biscale": "BiScale",
    "megascale_tier1": "MegaScale",
    "aflex_tier1": "AFlex",
}

# Deploy once per dataset batch (fixed topology)
FIXED_SCHEMES = frozenset({
    "native_tp1_baseline",
    "native_tp1_tier",
    "pd_hetero_baseline",
    "pd_hetero_tier_biscale",
})

# Per (dataset, qps) deploy — Tier1 layout changes with workload
TIER1_SCHEMES = {
    "megascale_tier1": True,   # megascale
    "aflex_tier1": False,      # aflex
}


def _save(all_results: dict, meta: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"6scheme_6dataset_{tag}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2)
    log.info("Saved %s", out)
    return out


def _missing_qps(scheme_data: dict, dataset: str, qps_list: list[int]) -> list[int]:
    missing = []
    for q in qps_list:
        key = _wl_key(dataset, q)
        if _is_pass(scheme_data.get(key)):
            continue
        if _workload_file(dataset, q) is None:
            log.warning("No workload file for %s", key)
            continue
        missing.append(q)
    return missing


def _run_fixed_scheme(
    name: str,
    deploy_fn,
    existing: dict,
    dataset: str,
    qps_list: list[int],
    gpus: list[int],
) -> dict:
    out = dict(existing)
    out.pop("__status__", None)
    missing = _missing_qps(out, dataset, qps_list)
    if not missing:
        return out

    log.info("DEPLOY %s (%s) | %s | QPS=%s", name, LABELS[name], dataset, missing)
    RMB.cleanup_all()
    time.sleep(3)
    url = deploy_fn()
    if url is None:
        out["__status__"] = "DEPLOY_FAILED"
        return out

    RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
    if not RMB.test_generate(url):
        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        out["__status__"] = "WARMUP_FAILED"
        return out
    time.sleep(3)

    for q in missing:
        res = run_one_dataset_qps(url, dataset, q, gpus)
        if res is not None:
            out[res[0]] = res[1]
        time.sleep(5)

    RMB.unlock_freq_both(gpus)
    RMB.cleanup_all()
    out.pop("__status__", None)
    return out


def _run_tier1_scheme(
    name: str,
    megascale: bool,
    existing: dict,
    dataset: str,
    qps_list: list[int],
    gpus: list[int],
) -> dict:
    out = dict(existing)
    out.pop("__status__", None)

    for q in qps_list:
        key = _wl_key(dataset, q)
        if _is_pass(out.get(key)):
            continue
        if _workload_file(dataset, q) is None:
            continue

        meta = layout_meta(dataset, q)
        cfg = layout_for(dataset, q, name, megascale=megascale)
        if cfg is None:
            out[key] = {"status": "NO_TIER1_LAYOUT", "tier1_layout": meta}
            continue

        log.info(
            "DEPLOY %s (%s) | %s QPS%d | 1P+%dD tier=%s",
            name, LABELS[name], dataset, q, cfg.k_d, cfg.tier,
        )
        RMB.cleanup_all()
        time.sleep(3)
        url = deploy_tier1_layout(cfg)
        if url is None:
            out[key] = {"status": "DEPLOY_FAILED", "tier1_layout": meta}
            continue
        if not RMB.test_generate(url):
            RMB.cleanup_all()
            out[key] = {"status": "WARMUP_FAILED", "tier1_layout": meta}
            continue
        time.sleep(3)

        res = run_one_dataset_qps(url, dataset, q, gpus)
        if res is None:
            out[key] = {"status": "NO_WORKLOAD", "tier1_layout": meta}
        else:
            _, summary = res
            summary = dict(summary)
            summary["tier1_layout"] = meta
            out[key] = summary

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        time.sleep(5)

    return out


def build_fixed_deployers() -> dict[str, callable]:
    return {
        "native_tp1_baseline": lambda: MTS.start_native_tp_variant(1, False),
        "native_tp1_tier": lambda: MTS.start_native_tp_variant(1, True),
        "pd_hetero_baseline": lambda: RMB.start_pd_hetero(NGPU, False),
        "pd_hetero_tier_biscale": lambda: start_pd_hetero_biscale(NGPU, True),
    }


def main():
    import resource

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument("--qps-list", default=",".join(str(q) for q in DEFAULT_QPS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    qps_list = [int(x.strip()) for x in args.qps_list.split(",") if x.strip()]
    F67.QPS_LIST = qps_list

    meta = {
        "benchmark": "6scheme_6dataset_clean",
        "node1": RMB.NODE1_IP,
        "node2": RMB.NODE2_IP,
        "datasets": datasets,
        "qps": qps_list,
        "scheme_labels": LABELS,
        "megascale": "Tier1 per-(dataset,qps) topology + lock 1410 MHz",
        "aflex": "Tier1 per-(dataset,qps) topology + compositional DVFS",
        "ttft_slo_ms": RMB.TTFT_SLO_MS,
        "tpot_slo_ms": RMB.TPOT_SLO_MS,
        "run_order": "dataset-first",
    }

    all_results: dict = {}
    if args.resume:
        files = sorted(RESULTS_DIR.glob("6scheme_6dataset_partial_*.json"))
        if files:
            all_results = json.loads(files[-1].read_text()).get("results", {})
            log.info("Resume from %s", files[-1].name)

    deployers = build_fixed_deployers()
    gpus = RMB.card_gpus(NGPU)
    scheme_order = list(LABELS.keys())

    log.info("=" * 72)
    log.info(
        "6-SCHEME x 6-DATASET (clean) | QPS=%s | datasets=%s | %d pts",
        qps_list, datasets, len(scheme_order) * len(datasets) * len(qps_list),
    )
    log.info("=" * 72)

    for dataset in datasets:
        log.info("\n" + "#" * 72)
        log.info("DATASET %s", dataset)
        log.info("#" * 72)

        for name in scheme_order:
            existing = dict(all_results.get(name, {}))
            if name in FIXED_SCHEMES:
                all_results[name] = _run_fixed_scheme(
                    name, deployers[name], existing, dataset, qps_list, gpus,
                )
            else:
                all_results[name] = _run_tier1_scheme(
                    name, TIER1_SCHEMES[name], existing, dataset, qps_list, gpus,
                )
            _save(all_results, meta)

        _save(all_results, meta, tag=f"dataset_{dataset}")
        log.info("DATASET %s done", dataset)

    out = _save(all_results, meta, tag="final")
    log.info("All done -> %s", out)


if __name__ == "__main__":
    main()
