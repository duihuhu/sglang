#!/usr/bin/env python3
"""7-scheme benchmark: 6 baselines + Tier1 QPS4 1P+3D on 6 macro/micro datasets.

Schemes:
  SGLang, DynamoLLM, DistServe, BiScale, MegaScale, AFlex  (same as best_pdaf_6scheme)
  AFlex-Tier1  = tier1_ilp_tier2  (QPS4 ILP 1P+3D + compositional DVFS)

Datasets (6):
  Macro: code, conv
  Micro: qa_lpld, chatbot_lphd, balanced_mpmd, summary_hphd

QPS: default 2,4,6,8,12,16 (override via --qps-list) | SLO: TTFT=5s, TPOT=300ms

Default run order: dataset-first (all schemes per dataset, then next dataset).
Use --order scheme for the legacy scheme-first loop.
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
log = logging.getLogger("7scheme_6ds")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(HERE))

import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB
import pdaf_deploy_utils as PDU
from run_biscale_pd_hetero_code import start_pd_hetero_biscale
from deploy_tier1_layout import Tier1Layout, deploy_tier1_layout
import run_fixed_6scheme_7dataset as F67
from run_fixed_6scheme_7dataset import (
    MAX_RUN_S,
    NGPU,
    MACRO_WL_DIR,
    MICRO_WL_DIR,
    run_one_dataset_qps,
    _is_pass,
    _missing_points,
    _wl_key,
    _workload_file,
)

DEFAULT_QPS_LIST = [2, 4, 6, 8, 12, 16]

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

RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

LABELS = {
    "native_tp1_baseline": "SGLang",
    "native_tp1_tier": "DynamoLLM",
    "pd_hetero_baseline": "DistServe",
    "pd_hetero_tier_biscale": "BiScale",
    "pdaf_best_baseline": "MegaScale",
    "pdaf_best_tier": "AFlex",
    "tier1_ilp_tier2": "AFlex-Tier1",
}

TIER1_NO_LOCK = frozenset({"tier1_ilp_tier2"})


def _tier1_layout() -> Tier1Layout:
    sol_path = HERE / "results" / "tier1_code_solutions.json"
    row = next(
        r for r in json.loads(sol_path.read_text())
        if r["qps"] == 4 and r.get("feasible_ilp")
    )
    s = row["solution"]
    return Tier1Layout(
        name="tier1_ilp_tier2",
        k_d=s["k_d"],
        tp_pa=s["pa"][0], tp_pf=s["pf"][0],
        tp_da=s["da"][0], tp_df=s["df"][0],
        f_pa=s["pa"][1], f_pf=s["pf"][1],
        f_da=s["da"][1], f_df=s["df"][1],
        tier=True,
    )


def _load_best_config(path: Path) -> dict:
    if not path.exists():
        key, e = PDU.pick_best_tier_key()
        path = PDU.save_best_config(key, e)
    return json.loads(path.read_text())


def _load_resume_results() -> dict:
    """Merge dataset snapshots and latest partial/final checkpoints."""
    merged: dict = {}
    for path in sorted(RESULTS_DIR.glob("7scheme_6dataset_dataset_*.json")):
        for scheme, wl in json.loads(path.read_text()).get("results", {}).items():
            merged.setdefault(scheme, {}).update(wl)
    for pattern in ("7scheme_6dataset_partial_*.json", "7scheme_6dataset_final_*.json"):
        files = sorted(RESULTS_DIR.glob(pattern))
        if files:
            for scheme, wl in json.loads(files[-1].read_text()).get("results", {}).items():
                merged.setdefault(scheme, {}).update(wl)
            log.info("Resume overlay from %s", files[-1].name)
    return merged


def _save(all_results: dict, meta: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"7scheme_6dataset_{tag}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2)
    log.info("Saved %s", out)
    return out


def _missing_qps_for_dataset(
    scheme_data: dict, dataset: str, qps_list: list[int]
) -> list[int]:
    missing = []
    for q in qps_list:
        key = _wl_key(dataset, q)
        if _is_pass(scheme_data.get(key)):
            continue
        if _workload_file(dataset, q) is None:
            log.warning("Workload missing: %s", key)
            continue
        missing.append(q)
    return missing


def _dataset_complete(
    all_results: dict, schemes: list[tuple[str, callable]], dataset: str, qps_list: list[int]
) -> bool:
    for name, _ in schemes:
        if _missing_qps_for_dataset(all_results.get(name, {}), dataset, qps_list):
            return False
    return True


def _run_scheme_points(
    name: str,
    deploy_fn,
    existing: dict,
    points: list[tuple[str, int]],
    gpus: list[int],
) -> dict:
    """Deploy once and run the given (dataset, qps) points."""
    deploy_results = dict(existing)
    deploy_results.pop("__status__", None)

    log.info("\n" + "=" * 72)
    log.info(
        "DEPLOY %s (%s) | %d workloads",
        name, LABELS[name], len(points),
    )
    log.info("=" * 72)

    RMB.cleanup_all()
    time.sleep(3)
    url = deploy_fn()
    if url is None:
        deploy_results["__status__"] = "DEPLOY_FAILED"
        return deploy_results

    if name not in TIER1_NO_LOCK:
        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)

    if not RMB.test_generate(url):
        if name not in TIER1_NO_LOCK:
            RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        deploy_results["__status__"] = "WARMUP_FAILED"
        return deploy_results
    time.sleep(3)

    for dataset, qps in points:
        log.info("-" * 50)
        res = run_one_dataset_qps(url, dataset, qps, gpus)
        if res is not None:
            deploy_results[res[0]] = res[1]
        time.sleep(5)

    if name not in TIER1_NO_LOCK:
        RMB.unlock_freq_both(gpus)
    RMB.cleanup_all()
    deploy_results.pop("__status__", None)
    return deploy_results


def _run_scheme_first(
    schemes: list[tuple[str, callable]],
    datasets: list[str],
    qps_list: list[int],
    all_results: dict,
    meta: dict,
    gpus: list[int],
) -> None:
    for name, deploy_fn in schemes:
        existing = dict(all_results.get(name, {}))
        existing.pop("__status__", None)
        missing = _missing_points(existing, datasets)
        if not missing:
            log.info("SKIP %s (%s): complete", name, LABELS[name])
            all_results[name] = existing
            continue

        all_results[name] = _run_scheme_points(name, deploy_fn, existing, missing, gpus)
        _save(all_results, meta)


def _run_dataset_first(
    schemes: list[tuple[str, callable]],
    datasets: list[str],
    qps_list: list[int],
    all_results: dict,
    meta: dict,
    gpus: list[int],
) -> None:
    for dataset in datasets:
        if _dataset_complete(all_results, schemes, dataset, qps_list):
            log.info("SKIP dataset %s: all schemes complete", dataset)
            continue

        log.info("\n" + "#" * 72)
        log.info("DATASET %s — run all schemes (QPS=%s)", dataset, qps_list)
        log.info("#" * 72)

        for name, deploy_fn in schemes:
            existing = dict(all_results.get(name, {}))
            missing_qps = _missing_qps_for_dataset(existing, dataset, qps_list)
            if not missing_qps:
                log.info("  SKIP %s (%s) on %s: complete", name, LABELS[name], dataset)
                all_results[name] = existing
                continue

            points = [(dataset, q) for q in missing_qps]
            all_results[name] = _run_scheme_points(
                name, deploy_fn, existing, points, gpus
            )
            _save(all_results, meta)

        log.info("DATASET %s done — ready for horizontal compare", dataset)
        _save(all_results, meta, tag=f"dataset_{dataset}")


def build_schemes(best: dict) -> list[tuple[str, callable]]:
    tier_key = best["scheme_key"]
    base_key = best.get("baseline_key", tier_key.replace("_tier", "_baseline"))
    tier1 = _tier1_layout()

    return [
        ("native_tp1_baseline", lambda: MTS.start_native_tp_variant(1, False)),
        ("native_tp1_tier", lambda: MTS.start_native_tp_variant(1, True)),
        ("pd_hetero_baseline", lambda: RMB.start_pd_hetero(NGPU, False)),
        ("pd_hetero_tier_biscale", lambda: start_pd_hetero_biscale(NGPU, True)),
        ("pdaf_best_baseline", lambda: PDU.deploy_from_key(base_key)),
        ("pdaf_best_tier", lambda: PDU.deploy_from_key(tier_key)),
        ("tier1_ilp_tier2", lambda: deploy_tier1_layout(tier1)),
    ]


def main():
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DEFAULT_DATASETS))
    parser.add_argument(
        "--best-config",
        type=Path,
        default=RESULTS_DIR / "pdaf_best_deploy.json",
    )
    parser.add_argument("--schemes", nargs="*", help="Limit scheme keys")
    parser.add_argument(
        "--qps-list",
        default=",".join(str(q) for q in DEFAULT_QPS_LIST),
        help="Comma-separated QPS points (default: 2,4,6,8,12,16)",
    )
    parser.add_argument(
        "--order",
        choices=("dataset", "scheme"),
        default="dataset",
        help="Run order: dataset=all schemes per dataset (default); scheme=legacy",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    qps_list = [int(x.strip()) for x in args.qps_list.split(",") if x.strip()]
    F67.QPS_LIST = qps_list

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    best = _load_best_config(args.best_config)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    schemes = build_schemes(best)
    if args.schemes:
        wanted = set(args.schemes)
        schemes = [(n, fn) for n, fn in schemes if n in wanted]

    tier1_row = next(
        r for r in json.loads(
            (HERE / "results" / "tier1_code_solutions.json").read_text()
        )
        if r["qps"] == 4
    )

    meta = {
        "node1": RMB.NODE1_IP,
        "node2": RMB.NODE2_IP,
        "datasets": datasets,
        "macro_datasets": [d for d in datasets if d in ("code", "conv")],
        "micro_datasets": [d for d in datasets if d not in ("code", "conv")],
        "qps": qps_list,
        "disable_radix_cache": True,
        "best_pdaf_deploy": best,
        "tier1_qps4_layout": tier1_row,
        "scheme_labels": LABELS,
        "ttft_slo_ms": RMB.TTFT_SLO_MS,
        "tpot_slo_ms": RMB.TPOT_SLO_MS,
        "macro_wl_dir": str(MACRO_WL_DIR),
        "micro_wl_dir": str(MICRO_WL_DIR),
        "run_order": args.order,
    }

    all_results: dict = {}
    if args.resume:
        all_results = _load_resume_results()
        if all_results:
            log.info("Resume merged %d scheme keys", len(all_results))

    gpus = RMB.card_gpus(NGPU)
    total_pts = len(schemes) * len(datasets) * len(qps_list)

    log.info("=" * 72)
    log.info(
        "7-SCHEME x 6-DATASET | %d schemes | datasets=%s | QPS=%s | order=%s | ~%d pts",
        len(schemes), datasets, qps_list, args.order, total_pts,
    )
    log.info("=" * 72)

    if args.order == "dataset":
        _run_dataset_first(schemes, datasets, qps_list, all_results, meta, gpus)
    else:
        _run_scheme_first(schemes, datasets, qps_list, all_results, meta, gpus)

    out = _save(all_results, meta, tag="final")
    log.info("All done -> %s", out)


if __name__ == "__main__":
    main()
