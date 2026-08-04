"""AFlex layouts from macro_e2e_all.json (per dataset x QPS)."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

E2E_JSON = Path(__file__).resolve().parents[2] / "macro" / "data" / "macro_e2e_all.json"

_CACHE: dict | None = None

if TYPE_CHECKING:
    from bench_tier1_v2 import Tier1TestConfig


@dataclass
class MacroE2EAflexConfig:
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


def config_for(dataset: str, qps: int) -> MacroE2EAflexConfig:
    data = _load_e2e()["results"][dataset][f"qps_{qps}"]["aflex_tier1"]["config"]
    return MacroE2EAflexConfig(
        name=data["name"],
        qps=qps,
        k_p=data["k_p"],
        k_d=data["k_d"],
        tp_pa=data["tp_pa"],
        tp_pf=data["tp_pf"],
        tp_da=data["tp_da"],
        tp_df=data["tp_df"],
        f_pa=data["f_pa"],
        f_pf=data["f_pf"],
        f_da=data["f_da"],
        f_df=data["f_df"],
        tier=data.get("tier", True),
    )


def to_tier1_test_config(
    dataset: str,
    qps: int,
    *,
    locked_freq_mhz: int | None = None,
    tier: bool | None = None,
    name_prefix: str = "",
) -> "Tier1TestConfig":
    import sys
    import types

    macro_dir = Path(__file__).resolve().parents[2] / "macro"
    macro_scripts = macro_dir / "scripts"
    wl_dir = macro_dir / "data" / "workloads"
    if "run_fixed_6scheme_7dataset" not in sys.modules:
        stub = types.ModuleType("run_fixed_6scheme_7dataset")
        stub.MAX_RUN_S = 400
        stub._wl_key = lambda ds, qps: f"{ds}_qps{qps}"
        stub._workload_file = lambda ds, qps: wl_dir / f"macro_{ds}_qps{qps}.jsonl"
        sys.modules["run_fixed_6scheme_7dataset"] = stub
    sys.path.insert(0, str(macro_scripts / "other_tier1"))
    sys.path.insert(0, str(macro_scripts))
    from bench_tier1_v2 import Tier1TestConfig

    c = config_for(dataset, qps)
    freq = locked_freq_mhz if locked_freq_mhz is not None else c.f_pa
    use_tier = c.tier if tier is None else tier
    return Tier1TestConfig(
        name=f"{name_prefix}{c.name}" if name_prefix else c.name,
        k_p=c.k_p,
        k_d=c.k_d,
        tp_pa=c.tp_pa,
        tp_pf=c.tp_pf,
        tp_da=c.tp_da,
        tp_df=c.tp_df,
        f_pa=freq,
        f_pf=freq,
        f_da=freq,
        f_df=freq,
        tier=use_tier,
    )


def layout_meta(dataset: str, qps: int) -> dict:
    c = config_for(dataset, qps)
    return {
        "source": "macro_e2e_all.json",
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


def energy_per_token_j(dataset: str, qps: int) -> float:
    pt = _load_e2e()["results"][dataset][f"qps_{qps}"]["aflex_tier1"]
    return pt["energy_per_token_mj"] / 1000.0
