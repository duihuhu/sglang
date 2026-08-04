#!/usr/bin/env python3
"""Deploy helpers for macro end-to-end benchmark schemes."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger("macro_deploy")

_MACRO_ROOT = Path(__file__).resolve().parent
_MULTI_NODE = _MACRO_ROOT.parents[2]
MORE_TRYING = _MACRO_ROOT
MACRO_DIR = _MACRO_ROOT

sys.path.insert(0, str(_MACRO_ROOT))
import run_macro_benchmark as RMB

sys.path.insert(0, str(_MACRO_ROOT / "other_tier1"))
import run_more_trying_sweep as MTS

os.environ.setdefault("MN_NODE1_IP", "10.252.129.36")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.35")

import plan_dense_e2e_layouts as E2E_LAYOUT

sys.path.insert(0, str(_MACRO_ROOT))
import run_code_4schemes_sweep as SWEEP

from bench_common import SCHEME_LABELS

_orig_afd_common = RMB._afd_common
_orig_wait_health = RMB.wait_health


def _patched_wait_health(host, port, timeout=60, **kwargs):
    return _orig_wait_health(host, port, max(timeout, 600), **kwargs)


def _patched_afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    result = _orig_afd_common(tp, ib_dev, gpu_step, tier, ngpu)
    result = result.replace("--afd-micro-batch 2", "--afd-micro-batch 1")
    result = result.replace("--afd-dynamic-micro-batch", "")
    if tier:
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
    return result


RMB._afd_common = _patched_afd_common
RMB.wait_health = _patched_wait_health
RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

ALL_GPUS = list(range(8))
TIER_DVFS_SCHEMES = frozenset({"native_tp1_tier", "pd_hetero_tier_biscale", "aflex_tier1"})
BASELINE_SCHEMES = frozenset({"native_tp1_baseline", "pd_hetero_baseline"})
AFLEX_WARMUP_TIMEOUT_S = 240
_aflex_afd_patch_orig = None


def _sync_sweep_nodes() -> None:
    n1 = os.environ["MN_NODE1_IP"]
    n2 = os.environ["MN_NODE2_IP"]
    RMB.NODE1_IP = n1
    RMB.NODE2_IP = n2
    SWEEP.RMB.NODE1_IP = n1
    SWEEP.RMB.NODE2_IP = n2
    SWEEP.NODE1 = n1
    SWEEP.NODE2 = n2


_sync_sweep_nodes()


def cleanup() -> None:
    RMB.cleanup_all()
    time.sleep(8)


def aflex_result_extra(dataset: str, qps: int) -> dict:
    return {"tier1_layout": E2E_LAYOUT.layout_meta(dataset, qps)}


def warmup_url(url: str, scheme: str = "") -> bool:
    timeout = AFLEX_WARMUP_TIMEOUT_S if scheme == "aflex_tier1" else 120
    for attempt in range(3):
        try:
            r = requests.post(
                url + "/generate",
                json={
                    "text": "Hello, explain quantum computing:",
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
                },
                timeout=timeout,
            )
            if r.status_code == 200 and "text" in r.json():
                return True
            log.warning("warmup attempt %d: status=%d", attempt + 1, r.status_code)
        except Exception as e:
            log.warning("warmup attempt %d failed: %s", attempt + 1, e)
        time.sleep(10)
    return False


def deploy_scheme(scheme: str, dataset: str | None = None, qps: int | None = None):
    if scheme == "aflex_tier1":
        raise ValueError("aflex_tier1 uses deploy_aflex_e2e_point(dataset, qps)")

    _sync_sweep_nodes()
    cleanup()

    if scheme == "native_tp1_baseline":
        RMB.lock_freq_both(ALL_GPUS, RMB.MAX_GPU_FREQ)
        url = MTS.start_native_tp_variant(1, False)
    elif scheme == "native_tp1_tier":
        url = MTS.start_native_tp_variant(1, True)
        if url:
            RMB.lock_freq_both(ALL_GPUS, RMB.MAX_GPU_FREQ)
    elif scheme == "pd_hetero_baseline":
        url = SWEEP.deploy_pd("distserve")
    elif scheme == "pd_hetero_tier_biscale":
        url = SWEEP.deploy_pd("biscale")
    else:
        raise ValueError(f"unknown scheme: {scheme}")

    return url, None


def deploy_aflex_e2e_point(dataset: str, qps: int) -> tuple[list[str] | None, dict | None]:
    import bench_tier1_v2 as BT2
    from freq_timeline_utils import ensure_container_log_dir, patch_tier1_afd_dvfs_log

    _sync_sweep_nodes()
    cleanup()

    cfg = E2E_LAYOUT.to_tier1_test_config(dataset, qps)
    meta = E2E_LAYOUT.layout_meta(dataset, qps)

    log.info(
        "DEPLOY AFlex | %s | QPS=%d | %s | k_p=%d k_d=%d | %d GPU",
        dataset, qps, cfg.name, cfg.k_p, cfg.k_d, meta["gpus"],
    )

    BT2.RMB.NODE1_IP = RMB.NODE1_IP
    BT2.RMB.NODE2_IP = RMB.NODE2_IP
    BT2.RMB.TTFT_SLO_MS = RMB.TTFT_SLO_MS
    BT2.RMB.TPOT_SLO_MS = RMB.TPOT_SLO_MS
    BT2.DATASET = dataset
    BT2.QPS = qps

    ensure_container_log_dir("aflex_tier1")
    global _aflex_afd_patch_orig
    _aflex_afd_patch_orig = patch_tier1_afd_dvfs_log("aflex_tier1")

    urls = BT2.deploy(cfg)
    if urls is None:
        teardown_aflex_e2e_point()
        return None, None

    if not RMB.test_generate(urls[0]):
        teardown_aflex_e2e_point()
        return None, None

    time.sleep(3)
    return urls, {
        "tier1_layout": meta,
        "config": {k: getattr(cfg, k) for k in cfg.__dataclass_fields__},
    }


def run_aflex_e2e_benchmark(urls: list[str], dataset: str, qps: int) -> dict:
    import bench_tier1_v2 as BT2

    BT2.DATASET = dataset
    BT2.QPS = qps
    cfg = E2E_LAYOUT.to_tier1_test_config(dataset, qps)
    return BT2.run_benchmark(urls, cfg)


def teardown_aflex_e2e_point() -> None:
    global _aflex_afd_patch_orig
    from freq_timeline_utils import restore_tier1_afd_env

    unlock_after_run("aflex_tier1")
    cleanup()
    if _aflex_afd_patch_orig is not None:
        restore_tier1_afd_env(_aflex_afd_patch_orig)
        _aflex_afd_patch_orig = None
    time.sleep(5)


def post_deploy_lock(scheme: str) -> None:
    if scheme in TIER_DVFS_SCHEMES or scheme in ("pd_hetero_baseline", "pd_hetero_tier_biscale"):
        return
    if scheme in BASELINE_SCHEMES:
        RMB.lock_freq_both(ALL_GPUS, RMB.MAX_GPU_FREQ)


def unlock_after_run(scheme: str) -> None:
    if scheme != "native_tp1_baseline":
        RMB.unlock_freq_both(ALL_GPUS)
