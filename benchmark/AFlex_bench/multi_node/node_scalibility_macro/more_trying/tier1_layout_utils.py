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
