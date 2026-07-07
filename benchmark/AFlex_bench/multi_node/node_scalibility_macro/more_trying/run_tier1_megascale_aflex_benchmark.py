#!/usr/bin/env python3
"""Tier1-layout MegaScale vs AFlex on 6 datasets x 6 QPS.

Both schemes use Tier1 ILP optimal topology (k_P, k_D, TP grid) per (dataset, qps).
Only difference:
  MegaScale = Tier1 topology + lock all GPUs at max freq (1410 MHz), no DVFS
  AFlex     = Tier1 topology + compositional dynamic DVFS (initial ILP freqs)

QPS default: 2,4,6,8,12,16 | SLO: TTFT=5s, TPOT=300ms

Usage:
  python3 run_tier1_megascale_aflex_benchmark.py --resume
  python3 run_tier1_megascale_aflex_benchmark.py --datasets code --qps-list 4,8
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
log = logging.getLogger("tier1_ms_aflex")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(HERE))

import run_macro_benchmark as RMB
from deploy_tier1_layout import Tier1Layout, deploy_tier1_layout
from run_fixed_6scheme_7dataset import (
    MAX_RUN_S,
    NGPU,
    run_one_dataset_qps,
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

DEFAULT_DATASETS = [
    "code",
    "conv",
    "qa_lpld",
    "chatbot_lphd",
    "balanced_mpmd",
    "summary_hphd",
]
DEFAULT_QPS = [2, 4, 6, 8, 12, 16]
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SCHEMES = (
    ("megascale_tier1", "MegaScale"),
    ("aflex_tier1", "AFlex"),
)


def _solutions_path(dataset: str) -> Path:
    return RESULTS_DIR / f"tier1_{dataset}_solutions.json"


def _load_tier1_row(dataset: str, qps: int) -> dict | None:
    path = _solutions_path(dataset)
    if not path.exists():
        log.error("Missing Tier1 solutions: %s", path)
        return None
    for row in json.loads(path.read_text()):
        if row["qps"] == qps:
            return row
    log.error("No Tier1 row for %s QPS%d in %s", dataset, qps, path.name)
    return None


def _layout_from_row(row: dict, scheme_key: str, tier: bool, maxfreq: bool) -> Tier1Layout:
    s = row["solution"]
    if maxfreq:
        f_pa = f_pf = f_da = f_df = RMB.MAX_GPU_FREQ
    else:
        f_pa, f_pf = s["pa"][1], s["pf"][1]
        f_da, f_df = s["da"][1], s["df"][1]
    return Tier1Layout(
        name=scheme_key,
        k_d=s["k_d"],
        tp_pa=s["pa"][0],
        tp_pf=s["pf"][0],
        tp_da=s["da"][0],
        tp_df=s["df"][0],
        f_pa=f_pa,
        f_pf=f_pf,
        f_da=f_da,
        f_df=f_df,
        tier=tier,
    )


def _layout_for_point(dataset: str, qps: int, scheme_key: str) -> Tier1Layout | None:
    row = _load_tier1_row(dataset, qps)
    if row is None:
        return None
    if scheme_key == "megascale_tier1":
        return _layout_from_row(row, scheme_key, tier=False, maxfreq=True)
    if scheme_key == "aflex_tier1":
        return _layout_from_row(row, scheme_key, tier=True, maxfreq=False)
    raise ValueError(scheme_key)


def _tier1_meta(dataset: str, qps: int) -> dict:
    row = _load_tier1_row(dataset, qps)
    if row is None:
        return {}
    s = row["solution"]
    return {
        "feasible_ilp": row.get("feasible_ilp"),
        "k_p": s["k_p"],
        "k_d": s["k_d"],
        "pa": s["pa"],
        "pf": s["pf"],
        "da": s["da"],
        "df": s["df"],
        "gpus": s["gpus"],
    }


def _save(results: dict, meta: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"tier1_megascale_aflex_{tag}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    log.info("Saved %s", out)
    return out


def _is_complete(results: dict, dataset: str, qps: int, scheme_key: str) -> bool:
    wl = _wl_key(dataset, qps)
    entry = results.get(scheme_key, {}).get(wl)
    return entry is not None and _is_pass(entry)


def main():
    import resource

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        default=",".join(DEFAULT_DATASETS),
        help="Comma-separated datasets",
    )
    parser.add_argument(
        "--qps-list",
        default=",".join(str(q) for q in DEFAULT_QPS),
        help="Comma-separated QPS values",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--schemes",
        nargs="*",
        choices=[s[0] for s in SCHEMES],
        help="Limit schemes",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    qps_list = [int(x.strip()) for x in args.qps_list.split(",") if x.strip()]
    schemes = SCHEMES
    if args.schemes:
        wanted = set(args.schemes)
        schemes = [s for s in SCHEMES if s[0] in wanted]

    gpus = RMB.card_gpus(NGPU)
    meta = {
        "benchmark": "tier1_megascale_aflex",
        "datasets": datasets,
        "qps_list": qps_list,
        "schemes": {k: v for k, v in schemes},
        "ttft_slo_ms": RMB.TTFT_SLO_MS,
        "tpot_slo_ms": RMB.TPOT_SLO_MS,
        "megascale": "Tier1 topology + lock 1410 MHz, no DVFS",
        "aflex": "Tier1 topology + compositional AFD DVFS",
        "tier1_solutions": {ds: str(_solutions_path(ds)) for ds in datasets},
    }

    results: dict = {}
    if args.resume:
        partials = sorted(RESULTS_DIR.glob("tier1_megascale_aflex_partial_*.json"))
        finals = sorted(RESULTS_DIR.glob("tier1_megascale_aflex_final_*.json"))
        for path in (finals + partials)[-3:]:
            try:
                payload = json.loads(path.read_text())
                results = payload.get("results", results)
                log.info("Resume from %s (%d scheme keys)", path.name, len(results))
                break
            except Exception:
                continue

    total = len(datasets) * len(qps_list) * len(schemes)
    done = sum(
        1
        for ds in datasets
        for q in qps_list
        for sk, _ in schemes
        if _is_complete(results, ds, q, sk)
    )
    log.info("=" * 72)
    log.info(
        "TIER1 MegaScale vs AFlex | %d datasets | QPS=%s | progress %d/%d",
        len(datasets),
        qps_list,
        done,
        total,
    )
    log.info("=" * 72)

    for dataset in datasets:
        log.info("\n" + "#" * 72)
        log.info("DATASET %s", dataset)
        log.info("#" * 72)

        for qps in qps_list:
            tier_meta = _tier1_meta(dataset, qps)
            if not tier_meta:
                log.warning("SKIP %s QPS%d: no Tier1 layout", dataset, qps)
                continue

            log.info(
                "QPS%d layout: %dP+%dD PA TP%d@%d PF TP%d@%d | feasible=%s",
                qps,
                tier_meta["k_p"],
                tier_meta["k_d"],
                tier_meta["pa"][0],
                tier_meta["pa"][1],
                tier_meta["pf"][0],
                tier_meta["pf"][1],
                tier_meta.get("feasible_ilp"),
            )

            for scheme_key, label in schemes:
                if _is_complete(results, dataset, qps, scheme_key):
                    log.info(
                        "SKIP %s %s QPS%d (complete)",
                        label,
                        dataset,
                        qps,
                    )
                    continue

                cfg = _layout_for_point(dataset, qps, scheme_key)
                if cfg is None:
                    continue

                log.info("-" * 60)
                log.info(
                    "RUN %s (%s) | %s QPS%d | 1P+%dD tier=%s",
                    label,
                    scheme_key,
                    dataset,
                    qps,
                    cfg.k_d,
                    cfg.tier,
                )

                RMB.cleanup_all()
                time.sleep(3)
                url = deploy_tier1_layout(cfg)
                if url is None:
                    results.setdefault(scheme_key, {})[_wl_key(dataset, qps)] = {
                        "status": "DEPLOY_FAILED",
                        "tier1_layout": tier_meta,
                    }
                    _save(results, meta)
                    continue

                if not RMB.test_generate(url):
                    results.setdefault(scheme_key, {})[_wl_key(dataset, qps)] = {
                        "status": "WARMUP_FAILED",
                        "tier1_layout": tier_meta,
                    }
                    RMB.cleanup_all()
                    _save(results, meta)
                    continue

                time.sleep(3)
                wl_key = _wl_key(dataset, qps)
                res = run_one_dataset_qps(url, dataset, qps, gpus)
                if res is None:
                    summary = {"status": "NO_WORKLOAD", "tier1_layout": tier_meta}
                else:
                    wl_key, summary = res
                    summary = dict(summary)
                    summary["tier1_layout"] = tier_meta

                results.setdefault(scheme_key, {})[wl_key] = summary
                RMB.unlock_freq_both(gpus)
                RMB.cleanup_all()
                _save(results, meta)
                time.sleep(5)

        _save(results, meta, tag=f"dataset_{dataset}")

    out = _save(results, meta, tag="final")
    log.info("=" * 72)
    log.info("DONE -> %s", out.name)
    log.info("=" * 72)


if __name__ == "__main__":
    main()
