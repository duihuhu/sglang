"""Load Tier1 ILP layout for a (dataset, qps) point."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import run_macro_benchmark as RMB
from deploy_tier1_layout import Tier1Layout

log = logging.getLogger("tier1_layout")

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"

# scheme_key -> (tier DVFS, lock max freq)
TIER1_SCHEME_MODES: dict[str, tuple[bool, bool]] = {
    "pdaf_best_baseline": (False, True),   # MegaScale
    "megascale_tier1": (False, True),
    "pdaf_best_tier": (True, False),       # AFlex
    "aflex_tier1": (True, False),
    "tier1_ilp_tier2": (True, False),      # AFlex-Tier1
}

TIER1_PER_QPS_SCHEMES = frozenset(TIER1_SCHEME_MODES)
TIER1_DVFS_SCHEMES = frozenset(
    k for k, (tier, _) in TIER1_SCHEME_MODES.items() if tier
)


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
    log.error("No Tier1 row for %s QPS%d in %s", dataset, qps, path.name)
    return None


def layout_meta(row: dict) -> dict:
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


def layout_from_row(
    row: dict, scheme_key: str, tier: bool | None = None, maxfreq: bool | None = None,
) -> Tier1Layout:
    if tier is None or maxfreq is None:
        tier_d, maxfreq_d = TIER1_SCHEME_MODES[scheme_key]
        tier = tier if tier is not None else tier_d
        maxfreq = maxfreq if maxfreq is not None else maxfreq_d
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


def layout_for_point(dataset: str, qps: int, scheme_key: str) -> Tier1Layout | None:
    row = load_row(dataset, qps)
    if row is None:
        return None
    return layout_from_row(row, scheme_key)


def result_has_per_qps_layout(entry: dict | None) -> bool:
    """PASS results from fixed code-QPS4 deploy lack tier1_layout metadata."""
    return isinstance(entry, dict) and entry.get("status") == "PASS" and "tier1_layout" in entry
