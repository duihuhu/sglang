#!/usr/bin/env python3
"""6-scheme benchmark: best PDAF deploy + 5 baselines on macro/micro datasets.

Schemes:
  SGLang      = native_tp1_baseline
  DynamoLLM   = native_tp1_tier
  DistServe   = pd_hetero_baseline (P4×TP2 + D2×TP4)
  BiScale     = pd_hetero_tier_biscale
  MegaScale   = best PDAF deploy (baseline)
  AFlex       = best PDAF deploy (tier)

Default datasets (4): code, conv, balanced_mpmd, summary_hphd
  (micro workloads: node_scalibility/exp-all/workloads, charts in 16card_datasets)

QPS: 2,4,6,8,12,16 | SLO: TTFT=5s, TPOT=300ms | --disable-radix-cache on non-PDAF.
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
log = logging.getLogger("best_pdaf_6scheme")

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
from run_fixed_6scheme_7dataset import (
    QPS_LIST,
    MAX_RUN_S,
    NGPU,
    RESULTS_DIR,
    MICRO_DATASETS,
    run_one_dataset_qps,
    _missing_points,
    _is_pass,
    _wl_key,
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

DEFAULT_DATASETS = ["code", "conv", "balanced_mpmd", "summary_hphd"]

LABELS = {
    "native_tp1_baseline": "SGLang",
    "native_tp1_tier": "DynamoLLM",
    "pd_hetero_baseline": "DistServe",
    "pd_hetero_tier_biscale": "BiScale",
    "pdaf_best_baseline": "MegaScale",
    "pdaf_best_tier": "AFlex",
}


def _load_best_config(path: Path) -> dict:
    if not path.exists():
        key, e = PDU.pick_best_tier_key()
        path = PDU.save_best_config(key, e)
    return json.loads(path.read_text())


def _save(all_results: dict, meta: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"best_pdaf_6scheme_{tag}_{ts}.json"
    payload = {"meta": meta, "results": all_results}
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)
    return out


def build_schemes(best: dict) -> list[tuple[str, callable]]:
    tier_key = best["scheme_key"]
    base_key = best.get("baseline_key", tier_key.replace("_tier", "_baseline"))

    return [
        ("native_tp1_baseline", lambda: MTS.start_native_tp_variant(1, False)),
        ("native_tp1_tier", lambda: MTS.start_native_tp_variant(1, True)),
        ("pd_hetero_baseline", lambda: RMB.start_pd_hetero(NGPU, False)),
        ("pd_hetero_tier_biscale", lambda: start_pd_hetero_biscale(NGPU, True)),
        ("pdaf_best_baseline", lambda: PDU.deploy_from_key(base_key)),
        ("pdaf_best_tier", lambda: PDU.deploy_from_key(tier_key)),
    ]


def main():
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets",
    )
    parser.add_argument(
        "--best-config",
        type=Path,
        default=RESULTS_DIR / "pdaf_best_deploy.json",
    )
    parser.add_argument("--schemes", nargs="*", help="Limit scheme keys")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest partial results")
    args = parser.parse_args()

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    best = _load_best_config(args.best_config)
    log.info("Best PDAF deploy: %s (%.3f J/tok @ code QPS16)",
             best["scheme_key"], best.get("code_qps16_energy_j_per_tok", 0))

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    schemes = build_schemes(best)
    if args.schemes:
        wanted = set(args.schemes)
        schemes = [(n, fn) for n, fn in schemes if n in wanted]

    all_results: dict = {}
    if args.resume:
        files = sorted(RESULTS_DIR.glob("best_pdaf_6scheme_partial_*.json"))
        if files:
            all_results = json.loads(files[-1].read_text()).get("results", {})
            log.info("Resume from %s", files[-1].name)

    meta = {
        "node1": RMB.NODE1_IP,
        "node2": RMB.NODE2_IP,
        "datasets": datasets,
        "qps": QPS_LIST,
        "disable_radix_cache": True,
        "best_pdaf_deploy": best,
        "scheme_labels": LABELS,
        "ttft_slo_ms": RMB.TTFT_SLO_MS,
        "tpot_slo_ms": RMB.TPOT_SLO_MS,
    }

    all_results: dict = {}
    gpus = RMB.card_gpus(NGPU)

    log.info("=" * 72)
    log.info("BEST-PDAF 6-SCHEME | datasets=%s | QPS=%s", datasets, QPS_LIST)
    log.info("=" * 72)

    for name, deploy_fn in schemes:
        existing = dict(all_results.get(name, {}))
        existing.pop("__status__", None)
        missing = _missing_points(existing, datasets)
        if not missing:
            log.info("SKIP %s (%s): complete", name, LABELS[name])
            all_results[name] = existing
            continue

        log.info("\n" + "=" * 72)
        log.info("DEPLOY %s (%s) | %d workloads", name, LABELS[name], len(missing))
        log.info("=" * 72)

        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            existing["__status__"] = "DEPLOY_FAILED"
            all_results[name] = existing
            _save(all_results, meta)
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            existing["__status__"] = "WARMUP_FAILED"
            all_results[name] = existing
            _save(all_results, meta)
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
        _save(all_results, meta)

    out = _save(all_results, meta, tag="final")
    log.info("All done -> %s", out)


if __name__ == "__main__":
    main()
