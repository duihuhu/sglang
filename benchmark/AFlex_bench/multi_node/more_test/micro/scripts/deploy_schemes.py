#!/usr/bin/env python3
"""Deploy helpers for micro 4-baseline schemes (node1+node2)."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path

import requests

log = logging.getLogger("micro_deploy")

MICRO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(MICRO_ROOT))

from run_micro_6scheme_sweep import (  # noqa: E402
    deploy_native_tp1,
    deploy_pd_hetero,
)
import run_micro_6scheme_sweep as MS  # noqa: E402

sys.path.insert(0, str(MICRO_ROOT.parents[0] / "macro" / "scripts"))
import run_macro_benchmark as RMB  # noqa: E402

os.environ.setdefault("MN_NODE1_IP", "10.252.129.36")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.35")

from bench_common import SCHEME_LABELS  # noqa: E402

SCHEME_DEPLOY_FN = {
    "native_tp1_baseline": lambda: deploy_native_tp1("sglang"),
    "native_tp1_tier": lambda: deploy_native_tp1("dynamollm"),
    "pd_hetero_baseline": lambda: deploy_pd_hetero("distserve"),
    "pd_hetero_tier_biscale": lambda: deploy_pd_hetero("biscale"),
}


def _sync_nodes() -> None:
    n1 = os.environ["MN_NODE1_IP"]
    n2 = os.environ["MN_NODE2_IP"]
    RMB.NODE1_IP = n1
    RMB.NODE2_IP = n2
    MS.RMB.NODE1_IP = n1
    MS.RMB.NODE2_IP = n2
    MS.NODE1 = n1
    MS.NODE2 = n2


def cleanup() -> None:
    RMB.cleanup_all()
    time.sleep(8)


def deploy_scheme(scheme: str):
    _sync_nodes()
    fn = SCHEME_DEPLOY_FN.get(scheme)
    if fn is None:
        raise ValueError(f"unknown scheme: {scheme}")
    return fn(), None


def warmup_url(url: str, scheme: str = "") -> bool:
    for attempt in range(3):
        try:
            r = requests.post(
                url + "/generate",
                json={
                    "text": "Hello:",
                    "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
                },
                timeout=120,
            )
            if r.status_code == 200 and "text" in r.json():
                return True
            log.warning("warmup attempt %d: status=%d", attempt + 1, r.status_code)
        except Exception as e:
            log.warning("warmup attempt %d failed: %s", attempt + 1, e)
        time.sleep(10)
    return False


def post_deploy_lock(scheme: str) -> None:
    return


def unlock_after_run(scheme: str) -> None:
    if scheme == "native_tp1_baseline":
        return
    RMB.unlock_freq_both(list(range(8)))
