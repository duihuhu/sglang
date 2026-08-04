"""Load Tier1 ILP layout per (dataset, qps) for MegaScale / AFlex deploy."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from deploy_tier1_layout import Tier1Layout

log = logging.getLogger("tier1_layout")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"


def solutions_path(dataset: str) -> Path:
    return RESULTS_DIR / f"tier1_{dataset}_solutions.json"


def load_row(dataset: str, qps: int) -> dict | None:
    path = solutions_path(dataset)
    if not path.exists():
        log.error("Missing Tier1 solutions: %s", path)
        return None
    for row in json.loads(path.read_text()):
        if row["qps"] == qps:
            return row
    log.error("No Tier1 row for %s QPS%d", dataset, qps)
    return None


def layout_from_row(
    row: dict,
    scheme_key: str,
    *,
    tier: bool,
    maxfreq: bool,
    max_gpu_freq: int = 1410,
) -> Tier1Layout:
    s = row["solution"]
    if maxfreq:
        f_pa = f_pf = f_da = f_df = max_gpu_freq
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


def layout_for(dataset: str, qps: int, scheme_key: str, megascale: bool) -> Tier1Layout | None:
    row = load_row(dataset, qps)
    if row is None:
        return None
    if not row.get("feasible_ilp", True):
        return ilp_fallback_layout(scheme_key, megascale=megascale)
    return layout_from_row(
        row,
        scheme_key,
        tier=not megascale,
        maxfreq=megascale,
    )


def layout_meta(dataset: str, qps: int) -> dict:
    row = load_row(dataset, qps)
    if row is None:
        return {}
    if not row.get("feasible_ilp", True):
        meta = ilp_fallback_layout_meta()
        meta["target_qps"] = qps
        return meta
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


def ilp_fallback_layout_meta() -> dict:
    """1P(8G PA/PF TP4) + 3D(6G DA/DF TP1) — used when Tier1 ILP is infeasible."""
    return {
        "feasible_ilp": False,
        "ilp_fallback": "1P8G_TP4+3D6G_TP1",
        "k_p": 1,
        "k_d": 3,
        "pa": [4, 1410],
        "pf": [4, 1410],
        "da": [1, 1410],
        "df": [1, 1410],
        "gpus": 14,
    }


def ilp_fallback_layout(scheme_key: str, megascale: bool) -> Tier1Layout:
    """14-GPU capacity fallback: 1P@TP4/TP4 (8G) + 3×D@TP1/TP1 (6G)."""
    return _tier1_fixed_layout(scheme_key, megascale, ilp_fallback_layout_meta())


def ilp_fallback_16g_layout_meta() -> dict:
    """1P(8G PA/PF TP4) + 4D(8G DA/DF TP1) — full 16-GPU Tier1 layout."""
    return {
        "feasible_ilp": False,
        "ilp_fallback": "1P8G_TP4+4D8G_TP1",
        "k_p": 1,
        "k_d": 4,
        "pa": [4, 1410],
        "pf": [4, 1410],
        "da": [1, 1410],
        "df": [1, 1410],
        "gpus": 16,
    }


def ilp_fallback_16g_layout(scheme_key: str, megascale: bool) -> Tier1Layout:
    """16-GPU layout: 1P@TP4/TP4 (8G) + 4×D@TP1/TP1 (8G)."""
    return _tier1_fixed_layout(scheme_key, megascale, ilp_fallback_16g_layout_meta())


def _tier1_fixed_layout(
    scheme_key: str, megascale: bool, meta: dict
) -> Tier1Layout:
    f = 1410
    return Tier1Layout(
        name=scheme_key,
        k_d=meta["k_d"],
        tp_pa=meta["pa"][0],
        tp_pf=meta["pf"][0],
        tp_da=meta["da"][0],
        tp_df=meta["df"][0],
        f_pa=f,
        f_pf=f,
        f_da=f,
        f_df=f,
        tier=not megascale,
    )
