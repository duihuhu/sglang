"""AFlex layouts from plan_dense_e2e.json (per dataset × QPS, includes k_p)."""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

_MACRO_ROOT = Path(__file__).resolve().parent
_SCRIPTS_DIR = _MACRO_ROOT
_MULTI_NODE = _MACRO_ROOT.parents[2]
sys.path.insert(0, str(_SCRIPTS_DIR))
sys.path.insert(0, str(_SCRIPTS_DIR / "other_tier1"))

from bench_tier1_v2 import Tier1TestConfig

E2E_JSON = _MACRO_ROOT.parent / "data" / "plan_dense_e2e.json"

_CACHE: dict | None = None


@dataclass
class PlanDenseAflexConfig:
    name: str
    qps: int
    k_p: int
    k_d: int
    tp_pa: int
    tp_pf: int
    tp_da: int
    tp_df: int
    f_pa: int
    f_pf: int
    f_da: int
    f_df: int
    tier: bool = True

    @property
    def gpus(self) -> int:
        return self.k_p * (self.tp_pa + self.tp_pf) + self.k_d * (
            self.tp_da + self.tp_df
        )


def _load_e2e() -> dict:
    global _CACHE
    if _CACHE is None:
        _CACHE = json.loads(E2E_JSON.read_text())
    return _CACHE


def config_for(dataset: str, qps: int) -> PlanDenseAflexConfig:
    key = f"{dataset}_qps{qps}"
    aflex = _load_e2e()["results"]["aflex_tier1"]
    if key not in aflex or "config" not in aflex[key]:
        raise KeyError(f"no AFlex config in plan_dense_e2e.json for {key}")
    c = aflex[key]["config"]
    return PlanDenseAflexConfig(
        name=c["name"],
        qps=c["qps"],
        k_p=c["k_p"],
        k_d=c["k_d"],
        tp_pa=c["tp_pa"],
        tp_pf=c["tp_pf"],
        tp_da=c["tp_da"],
        tp_df=c["tp_df"],
        f_pa=c["f_pa"],
        f_pf=c["f_pf"],
        f_da=c["f_da"],
        f_df=c["f_df"],
        tier=c.get("tier", True),
    )


def to_tier1_test_config(dataset: str, qps: int) -> Tier1TestConfig:
    c = config_for(dataset, qps)
    return Tier1TestConfig(
        name=c.name,
        k_p=c.k_p,
        k_d=c.k_d,
        tp_pa=c.tp_pa,
        tp_pf=c.tp_pf,
        tp_da=c.tp_da,
        tp_df=c.tp_df,
        f_pa=c.f_pa,
        f_pf=c.f_pf,
        f_da=c.f_da,
        f_df=c.f_df,
        tier=c.tier,
    )


def layout_meta(dataset: str, qps: int) -> dict:
    c = config_for(dataset, qps)
    return {
        "source": "plan_dense_e2e.json",
        "name": c.name,
        "dataset": dataset,
        "target_qps": qps,
        "k_p": c.k_p,
        "k_d": c.k_d,
        "pa": [c.tp_pa, c.f_pa],
        "pf": [c.tp_pf, c.f_pf],
        "da": [c.tp_da, c.f_da],
        "df": [c.tp_df, c.f_df],
        "gpus": c.gpus,
        "tier": c.tier,
    }
