#!/usr/bin/env python3
"""Fixed 6-scheme x 7-dataset benchmark on node3+node4 (16-card).

Architecture (same as fixed_6scheme_complete):
  SGLang      = native_tp1_baseline   (TP1 x16)
  DynamoLLM   = native_tp1_tier
  DistServe   = pd_xnode_tp1_baseline (8P+8D xnode TP1)
  BiScale     = pd_xnode_tp1_tier
  MegaScale   = pdaf_tp1_baseline      (PDAF TP1 x4)
  AFlex       = pdaf_tp1_tier         (M=1, V1 compositional DVFS)

Datasets (7):
  Macro: conv, code
  Micro: qa_lpld, chatbot_lphd, balanced_mpmd, rag_hpld, summary_hphd

QPS: 2,4,6,8,12,16 | SLO: TTFT=5s, TPOT=300ms

Skips conv workloads already present in fixed_6scheme_complete_final_*.json.
"""
from __future__ import annotations

import argparse
import asyncio
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
log = logging.getLogger("fixed_6scheme_7ds")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB

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

QPS_LIST = [2, 4, 6, 8, 12, 16]
MACRO_DATASETS = ["conv", "code"]
MICRO_DATASETS = [
    "qa_lpld",
    "chatbot_lphd",
    "balanced_mpmd",
    "rag_hpld",
    "summary_hphd",
]
ALL_DATASETS = MACRO_DATASETS + MICRO_DATASETS
MAX_RUN_S = 400
NGPU = 16
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MACRO_WL_DIR = MACRO_DIR / "workloads"
MICRO_WL_DIR = MACRO_DIR.parent / "node_scalibility" / "exp-all" / "workloads"

SCHEMES = [
    ("native_tp1_baseline", lambda: MTS.start_native_tp_variant(1, False)),
    ("native_tp1_tier", lambda: MTS.start_native_tp_variant(1, True)),
    ("pd_xnode_tp1_baseline", lambda: MTS.start_pd_xnode(1, False)),
    ("pd_xnode_tp1_tier", lambda: MTS.start_pd_xnode(1, True)),
    ("pdaf_tp1_baseline", lambda: MTS.start_pdaf_multi(1, False)),
    ("pdaf_tp1_tier", lambda: MTS.start_pdaf_multi(1, True)),
]

LABELS = {
    "native_tp1_baseline": "SGLang",
    "native_tp1_tier": "DynamoLLM",
    "pd_xnode_tp1_baseline": "DistServe",
    "pd_xnode_tp1_tier": "BiScale",
    "pdaf_tp1_baseline": "MegaScale",
    "pdaf_tp1_tier": "AFlex",
}


def _wl_key(dataset: str, qps: int) -> str:
    return f"{dataset}_qps{qps}"


def _is_pass(entry: dict | None) -> bool:
    return isinstance(entry, dict) and entry.get("status") == "PASS"


def _workload_file(dataset: str, qps: int) -> Path | None:
    if dataset in MACRO_DATASETS:
        path = MACRO_WL_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    else:
        path = MICRO_WL_DIR / f"micro_{dataset}_qps{qps}.jsonl"
    return path if path.exists() else None


def _load_conv_skip() -> dict[str, dict]:
    """Load conv results from fixed_6scheme_complete_final for reuse."""
    merged: dict[str, dict] = {}
    for path in sorted(RESULTS_DIR.glob("fixed_6scheme_complete_final_*.json")):
        data = json.loads(path.read_text())
        for scheme, wl in data.get("results", {}).items():
            if not isinstance(wl, dict) or "__status__" in wl:
                continue
            bucket = merged.setdefault(scheme, {})
            for key, metric in wl.items():
                if key.startswith("conv_qps") and _is_pass(metric):
                    bucket[key] = metric
    return merged


def _missing_points(scheme_data: dict, datasets: list[str]) -> list[tuple[str, int]]:
    missing = []
    for ds in datasets:
        for q in QPS_LIST:
            key = _wl_key(ds, q)
            if not _is_pass(scheme_data.get(key)):
                if _workload_file(ds, q) is None:
                    log.warning("Workload missing: %s", key)
                    continue
                missing.append((ds, q))
    return missing


def _save(all_results: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"fixed_6scheme_7dataset_{tag}_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": ALL_DATASETS,
            "qps": QPS_LIST,
            "dvfs_policy": "min_energy_under_slo",
            "config": {
                "native": "TP1 x16",
                "pd": "xnode TP1 (8P+8D)",
                "pdaf": "TP1 x4 instances",
            },
            "scheme_labels": LABELS,
        },
        "results": all_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)
    return out


def run_one_dataset_qps(url: str, dataset: str, qps: int, gpus: list[int]):
    wl_file = _workload_file(dataset, qps)
    if wl_file is None:
        return None
    with open(wl_file) as f:
        reqs = [json.loads(line) for line in f]
    wl_key = _wl_key(dataset, qps)
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(MAX_RUN_S, last_arrival + 150), 900))
    log.info("  %s (%d reqs, run_window=%ds)", wl_key, len(reqs), run_s)
    summary = asyncio.run(
        RMB.run_workload(reqs, url + "/generate", gpus, gpus, run_s)
    )
    if summary.get("status") == "PASS":
        log.info(
            "  Thpt=%.1f tok/s | TTFT=%.1fms | TPOT=%.1fms | "
            "E=%.0fJ (%.1f mJ/tok) | SLO=%.1f%%",
            summary["throughput_tok_s"],
            summary["ttft_proc_avg_ms"],
            summary["tpot_avg_ms"],
            summary["total_energy_j"],
            summary["energy_per_token_mj"],
            summary["slo_violation_rate"],
        )
    else:
        log.error("  FAIL: %s", summary)
    return wl_key, summary


def main():
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(ALL_DATASETS),
        help="Comma-separated dataset list",
    )
    parser.add_argument(
        "--schemes",
        nargs="*",
        choices=[s[0] for s in SCHEMES],
        help="Limit to specific schemes",
    )
    parser.add_argument(
        "--no-skip-conv",
        action="store_true",
        help="Re-run conv even if fixed_6scheme_complete_final exists",
    )
    args = parser.parse_args()

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    schemes = SCHEMES
    if args.schemes:
        wanted = set(args.schemes)
        schemes = [(n, fn) for n, fn in SCHEMES if n in wanted]

    all_results: dict = {}
    if not args.no_skip_conv and "conv" in datasets:
        conv_skip = _load_conv_skip()
        for scheme, wl in conv_skip.items():
            all_results.setdefault(scheme, {}).update(wl)
        if conv_skip:
            log.info(
                "Reused conv from fixed_6scheme_complete_final (%d schemes)",
                len(conv_skip),
            )

    log.info("=" * 72)
    log.info(
        "FIXED 6-SCHEME x 7-DATASET | %d schemes | datasets=%s | QPS=%s",
        len(schemes),
        datasets,
        QPS_LIST,
    )
    log.info("=" * 72)

    gpus = RMB.card_gpus(NGPU)

    for name, deploy_fn in schemes:
        existing = dict(all_results.get(name, {}))
        existing.pop("__status__", None)
        missing = _missing_points(existing, datasets)
        if not missing:
            log.info("SKIP %s (%s): all workloads present", name, LABELS[name])
            all_results[name] = existing
            continue

        log.info("\n" + "=" * 72)
        log.info(
            "DEPLOY: %s (%s) | %d missing workloads",
            name,
            LABELS[name],
            len(missing),
        )
        log.info("=" * 72)

        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            existing["__status__"] = "DEPLOY_FAILED"
            all_results[name] = existing
            _save(all_results)
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            existing["__status__"] = "WARMUP_FAILED"
            all_results[name] = existing
            _save(all_results)
            continue
        time.sleep(3)

        deploy_results = dict(existing)
        for dataset, qps in missing:
            log.info("-" * 50)
            res = run_one_dataset_qps(url, dataset, qps, gpus)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        deploy_results.pop("__status__", None)
        all_results[name] = deploy_results
        _save(all_results)

    out = _save(all_results, tag="final")
    log.info("All done -> %s", out)


if __name__ == "__main__":
    main()
