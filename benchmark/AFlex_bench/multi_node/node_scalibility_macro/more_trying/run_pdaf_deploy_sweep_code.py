#!/usr/bin/env python3
"""Large PDAF deployment sweep on code (QPS 2/8/16). See README.md."""
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
log = logging.getLogger("pdaf_deploy_sweep")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(HERE))

import run_macro_benchmark as RMB
import pdaf_deploy as PD
from run_fixed_6scheme_7dataset import MAX_RUN_S, NGPU, RESULTS_DIR, run_one_dataset_qps

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

DATASET = "code"
QPS_LIST = [2, 8, 16]
LAYOUTS = ("tp4x1", "tp2x2", "tp1x4")


def _scheme_key(cat: str, p: str, d: str, tier: bool) -> str:
    t = "tier" if tier else "baseline"
    if cat == "c3":
        return f"pdaf_c3_xnode_tp1x4_{t}"
    if cat == "c2":
        return f"pdaf_c2_intra_{p}_{t}"
    return f"pdaf_c1_xnode_p{p}_d{d}_{t}"


def build_configs(skip_c3: bool = False):
    cfgs = []
    if not skip_c3:
        for tier in (False, True):
            key = _scheme_key("c3", "tp1x4", "tp1x4", tier)
            cfgs.append((key, "c3", "tp1x4", "tp1x4", tier,
                         lambda t=tier: PD.deploy_c3_xnode_tp1x4(t)))
    for layout in ("tp2x2", "tp1x4"):
        for tier in (False, True):
            key = _scheme_key("c2", layout, layout, tier)
            cfgs.append((key, "c2", layout, layout, tier,
                         lambda ly=layout, t=tier: PD.deploy_c2_intra(ly, t)))
    for p in LAYOUTS:
        for d in LAYOUTS:
            if len(PD.side_groups(p)) != len(PD.side_groups(d)):
                continue  # C1 requires matched P/D instance counts
            for tier in (False, True):
                key = _scheme_key("c1", p, d, tier)
                cfgs.append((key, "c1", p, d, tier,
                             lambda pp=p, dd=d, t=tier: PD.deploy_c1_xnode(pp, dd, t)))
    return cfgs


def _is_skipped_layout(key: str) -> bool:
    """Permanently skip C1 configs with mismatched P/D instance counts."""
    import re
    m = re.match(r"pdaf_c1_xnode_p(.+)_d(.+)_(baseline|tier)$", key)
    if not m:
        return False
    return len(PD.side_groups(m.group(1))) != len(PD.side_groups(m.group(2)))


def _is_complete(wl: dict) -> bool:
    if not isinstance(wl, dict):
        return False
    if wl.get("__status__"):
        return False
    return all(
        wl.get(f"code_qps{q}", {}).get("status") == "PASS" for q in QPS_LIST
    )


def _load_resume() -> dict:
    files = sorted(RESULTS_DIR.glob("pdaf_deploy_sweep_code_partial_*.json"))
    if not files:
        return {}
    data = json.loads(files[-1].read_text()).get("results", {})
    log.info("Resume from %s (%d schemes)", files[-1].name, len(data))
    return data


def _save(all_results: dict, meta_extra: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"pdaf_deploy_sweep_code_{tag}_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "dataset": DATASET,
            "qps": QPS_LIST,
            "disable_radix_cache": True,
            **meta_extra,
        },
        "results": all_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Saved %s", out)
    return out


def main():
    import resource

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-c3", action="store_true",
                        help="Skip cat3 (already in pdaf_code_final.json)")
    parser.add_argument("--only", nargs="*", help="Limit scheme keys")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest partial results")
    args = parser.parse_args()

    resource.setrlimit(
        resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY)
    )

    cfgs = build_configs(skip_c3=args.skip_c3)
    if args.only:
        wanted = set(args.only)
        cfgs = [c for c in cfgs if c[0] in wanted]

    all_results: dict = _load_resume() if args.resume else {}
    gpus = RMB.card_gpus(NGPU)

    pending = [(k, c, p, d, t, fn) for k, c, p, d, t, fn in cfgs
               if not _is_complete(all_results.get(k, {}))
               and not _is_skipped_layout(k)]
    log.info("=" * 72)
    log.info("PDAF DEPLOY SWEEP | code | QPS=%s | %d pending / %d total",
             QPS_LIST, len(pending), len(cfgs))
    log.info("=" * 72)

    for key, cat, p, d, tier, deploy_fn in pending:
        log.info("\n" + "=" * 72)
        log.info("DEPLOY %s | cat=%s P=%s D=%s tier=%s", key, cat, p, d, tier)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            all_results[key] = {"__status__": "DEPLOY_FAILED"}
            _save(all_results, {"classes": ["c1", "c2", "c3"]})
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            all_results[key] = {"__status__": "WARMUP_FAILED"}
            _save(all_results, {"classes": ["c1", "c2", "c3"]})
            continue
        time.sleep(3)

        deploy_results = {k: v for k, v in all_results.get(key, {}).items()
                          if k.startswith("code_qps")}
        for qps in QPS_LIST:
            log.info("-" * 50)
            res = run_one_dataset_qps(url, DATASET, qps, gpus)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        all_results[key] = deploy_results
        _save(all_results, {"classes": ["c1", "c2", "c3"]})

    out = _save(all_results, {"classes": ["c1", "c2", "c3"]}, tag="final")
    log.info("All done -> %s", out)


if __name__ == "__main__":
    main()
