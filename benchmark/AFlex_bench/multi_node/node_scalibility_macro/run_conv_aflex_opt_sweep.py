#!/usr/bin/env python3
"""AFlex conv optimization sweep.

Runs AFlex-only variants on node3+node4 with the same conv workload/QPS/SLO
as conv_6scheme_slo5000_300. The script is resumable: existing PASS results
in the output JSON are skipped, and known reference results are imported from
previous benchmark JSONs when available.
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import resource
import sys
import time
from pathlib import Path
from typing import Callable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aflex_opt_sweep")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_macro_benchmark as RMB  # noqa: E402
import run_conv_pdaf_2decode as P2D  # noqa: E402

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0
P2D.RMB.TTFT_SLO_MS = RMB.TTFT_SLO_MS
P2D.RMB.TPOT_SLO_MS = RMB.TPOT_SLO_MS

QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ORIG_AFD_COMMON = RMB._afd_common


def set_afd_common(micro_batch: int = 1, dynamic: bool = False,
                   async_pipeline: bool = False,
                   overlap_schedule: bool = False,
                   decode_compositional: bool = False,
                   energy_model_dir: str | None = None,
                   comm_us: float | None = None):
    """Patch RMB._afd_common for one single-decode PDAF variant."""

    def _patched(tp, ib_dev, gpu_step, tier, ngpu=8):
        result = ORIG_AFD_COMMON(tp, ib_dev, gpu_step, tier, ngpu)
        result = result.replace("--afd-micro-batch 2", f"--afd-micro-batch {micro_batch}")
        result = result.replace("--afd-dynamic-micro-batch", "")
        extras = []
        if dynamic:
            extras.append("--afd-dynamic-micro-batch")
        if async_pipeline:
            extras.append("--afd-async-pipeline")
        if overlap_schedule:
            extras.append("--afd-enable-overlap-schedule")
        if decode_compositional:
            extras.append("--afd-dvfs-decode-compositional")
        if comm_us is not None:
            extras.extend(["--afd-dvfs-comm-us", str(comm_us)])
        if energy_model_dir is not None:
            result = result.replace(RMB.ENERGY_MODEL_DIR_V2, energy_model_dir)
        if extras:
            result += " " + " ".join(extras)
        return result

    RMB._afd_common = _patched


def reset_afd_common():
    RMB._afd_common = ORIG_AFD_COMMON


def patch_p2d_common(micro_batch: int = 1, dynamic: bool = False,
                     async_pipeline: bool = False,
                     overlap_schedule: bool = False,
                     decode_compositional: bool = False,
                     energy_model_dir: str | None = None,
                     comm_us: float | None = None):
    """Patch run_conv_pdaf_2decode._afd_common for 2-decode variants."""
    orig = P2D._afd_common

    def _patched(tp, ib_dev, gpu_step, tier, bs_port):
        result = orig(tp, ib_dev, gpu_step, tier, bs_port)
        result = result.replace("--afd-micro-batch 1", f"--afd-micro-batch {micro_batch}")
        result = result.replace("--afd-dynamic-micro-batch", "")
        extras = []
        if dynamic:
            extras.append("--afd-dynamic-micro-batch")
        if async_pipeline:
            extras.append("--afd-async-pipeline")
        if overlap_schedule:
            extras.append("--afd-enable-overlap-schedule")
        if decode_compositional:
            extras.append("--afd-dvfs-decode-compositional")
        if comm_us is not None:
            extras.extend(["--afd-dvfs-comm-us", str(comm_us)])
        if energy_model_dir is not None:
            result = result.replace(RMB.ENERGY_MODEL_DIR_V2, energy_model_dir)
        if extras:
            result += " " + " ".join(extras)
        return result

    P2D._afd_common = _patched
    return orig


def latest_json(prefix: str) -> Path | None:
    files = sorted(RESULTS_DIR.glob(prefix))
    return files[-1] if files else None


def load_json(path: Path | None) -> dict:
    if path and path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def load_payload(out_file: Path) -> dict:
    if out_file.exists():
        return load_json(out_file)
    return {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "model": RMB.MODEL,
            "ngpu_total": NGPU,
            "gpus_per_node": RMB.card_gpus(NGPU),
            "dataset": DATASET,
            "qps": QPS_LIST,
            "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS,
            "status": "partial",
        },
        "results": {},
    }


def save_payload(payload: dict, out_file: Path, status: str = "partial"):
    payload["meta"]["status"] = status
    payload["meta"]["updated_at"] = time.strftime("%Y%m%d_%H%M%S")
    tmp = out_file.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(out_file)
    log.info("Results saved: %s", out_file)


def import_reference_results(payload: dict):
    """Seed the sweep with already-collected reference data."""
    six = load_json(latest_json("conv_6scheme_slo5000_300_*.json"))
    six_results = six.get("results", {})
    mapping = {
        "distserve_ref": "pd_hetero_baseline",
        "biscale_ref": "pd_hetero_tier",
        "aflex_m1_v2_ref": "pdaf_tier",
        "megascale_m1_ref": "pdaf_baseline",
    }
    for dst, src in mapping.items():
        if dst not in payload["results"] and src in six_results:
            payload["results"][dst] = copy.deepcopy(six_results[src])

    v1v2 = load_json(RESULTS_DIR / "conv_aflex_v1_v2_20260704_192807.json")
    v1 = v1v2.get("results", {}).get("aflex_v1_model")
    if v1 and "aflex_m1_v1_ref" not in payload["results"]:
        payload["results"]["aflex_m1_v1_ref"] = copy.deepcopy(v1)


def workload_done(results: dict, qps: int) -> bool:
    return results.get(f"{DATASET}_qps{qps}", {}).get("status") == "PASS"


def run_deploy(name: str, start_fn: Callable[[], str | None], payload: dict,
               out_file: Path, qps_list: list[int]):
    deploy_results = payload["results"].setdefault(name, {})
    missing = [q for q in qps_list if not workload_done(deploy_results, q)]
    if not missing:
        log.info("%s complete, skip", name)
        return

    gpus = RMB.card_gpus(NGPU)
    log.info("=" * 80)
    log.info("DEPLOY %s | missing qps=%s", name, missing)
    log.info("=" * 80)
    RMB.cleanup_all()
    url = start_fn()
    if url is None:
        deploy_results["__status__"] = "DEPLOY_FAILED"
        save_payload(payload, out_file)
        RMB.cleanup_all()
        return

    RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
    log.info("Warmup...")
    if not RMB.test_generate(url):
        deploy_results["__status__"] = "WARMUP_FAILED"
        save_payload(payload, out_file)
        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        return
    time.sleep(3)

    try:
        for qps in missing:
            log.info("-" * 60)
            res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
            if res is not None:
                deploy_results[res[0]] = res[1]
                save_payload(payload, out_file)
            time.sleep(5)
    finally:
        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        save_payload(payload, out_file)


def main():
    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None, help="Output JSON path")
    parser.add_argument("--qps", default=",".join(map(str, QPS_LIST)))
    parser.add_argument("--variants", default="m2_v2,dynamic_m2_v2,m2_v1,2decode_m1_v2,2decode_m1_v1")
    parser.add_argument("--no-import", action="store_true", help="Do not import prior reference results")
    args = parser.parse_args()

    qps_list = [int(x) for x in args.qps.split(",") if x]
    variants = [x.strip() for x in args.variants.split(",") if x.strip()]
    out_file = Path(args.out) if args.out else RESULTS_DIR / f"conv_aflex_opt_sweep_{time.strftime('%Y%m%d_%H%M%S')}.json"
    payload = load_payload(out_file)
    if not args.no_import:
        import_reference_results(payload)
        save_payload(payload, out_file)

    variant_fns: dict[str, Callable[[], None]] = {}

    def m2_v2_start():
        reset_afd_common()
        set_afd_common(micro_batch=2, energy_model_dir=RMB.ENERGY_MODEL_DIR_V2)
        return RMB.SCHEMES["pdaf"](NGPU, True)
    variant_fns["m2_v2"] = m2_v2_start

    def dynamic_m2_v2_start():
        reset_afd_common()
        set_afd_common(micro_batch=2, dynamic=True, async_pipeline=True,
                       energy_model_dir=RMB.ENERGY_MODEL_DIR_V2)
        return RMB.SCHEMES["pdaf"](NGPU, True)
    variant_fns["dynamic_m2_v2"] = dynamic_m2_v2_start

    def m2_v1_start():
        reset_afd_common()
        set_afd_common(micro_batch=2, decode_compositional=True,
                       energy_model_dir=RMB.ENERGY_MODEL_DIR_V1)
        return RMB.SCHEMES["pdaf"](NGPU, True)
    variant_fns["m2_v1"] = m2_v1_start

    def overlap_m2_v1_start():
        reset_afd_common()
        set_afd_common(micro_batch=2, async_pipeline=True, overlap_schedule=True,
                       decode_compositional=True, energy_model_dir=RMB.ENERGY_MODEL_DIR_V1)
        return RMB.SCHEMES["pdaf"](NGPU, True)
    variant_fns["overlap_m2_v1"] = overlap_m2_v1_start

    def p2d_m1_v2_start():
        orig = patch_p2d_common(micro_batch=1, energy_model_dir=RMB.ENERGY_MODEL_DIR_V2)
        try:
            return P2D.start_pdaf_2decode(True)
        finally:
            P2D._afd_common = orig
    variant_fns["2decode_m1_v2"] = p2d_m1_v2_start

    def p2d_m1_v1_start():
        orig = patch_p2d_common(micro_batch=1, decode_compositional=True,
                                energy_model_dir=RMB.ENERGY_MODEL_DIR_V1)
        try:
            return P2D.start_pdaf_2decode(True)
        finally:
            P2D._afd_common = orig
    variant_fns["2decode_m1_v1"] = p2d_m1_v1_start

    for variant in variants:
        if variant not in variant_fns:
            raise ValueError(f"Unknown variant: {variant}; available={sorted(variant_fns)}")
        run_deploy(f"aflex_{variant}", variant_fns[variant], payload, out_file, qps_list)

    save_payload(payload, out_file, status="completed")
    log.info("SWEEP COMPLETED: %s", out_file)


if __name__ == "__main__":
    main()
