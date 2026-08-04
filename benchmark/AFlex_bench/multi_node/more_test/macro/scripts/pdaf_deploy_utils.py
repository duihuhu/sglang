"""Helpers: pick best PDAF deploy + launch from scheme key."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pdaf_deploy as PD

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
QPS_BEST = 16


def parse_scheme_key(key: str) -> tuple[str, str, str, bool]:
    """Return (category, p_layout, d_layout, tier)."""
    tier = key.endswith("_tier")
    if key.startswith("pdaf_c3_"):
        return "c3", "tp1x4", "tp1x4", tier
    m = re.match(r"pdaf_c2_intra_(.+)_(baseline|tier)$", key)
    if m:
        ly = m.group(1)
        return "c2", ly, ly, tier
    m = re.match(r"pdaf_c1_xnode_p(.+)_d(.+)_(baseline|tier)$", key)
    if m:
        return "c1", m.group(1), m.group(2), tier
    raise ValueError(f"unknown PDAF scheme key: {key}")


def deploy_from_key(key: str):
    cat, p, d, tier = parse_scheme_key(key)
    if cat == "c3":
        return PD.deploy_c3_xnode_tp1x4(tier)
    if cat == "c2":
        return PD.deploy_c2_intra(p, tier)
    return PD.deploy_c1_xnode(p, d, tier)


def _metric(entry: dict | None, field: str = "energy_per_token_mj") -> float | None:
    if not isinstance(entry, dict) or entry.get("status") != "PASS":
        return None
    return entry.get(field)


def load_sweep_results(sweep_path: Path | None = None) -> dict:
    if sweep_path is not None:
        return json.loads(sweep_path.read_text()).get("results", {})
    for pattern in ("pdaf_deploy_sweep_code_final_*.json",
                    "pdaf_deploy_sweep_code_partial_*.json"):
        files = sorted(RESULTS_DIR.glob(pattern))
        if files:
            return json.loads(files[-1].read_text()).get("results", {})
    return {}


def merge_c3_tier(results: dict) -> dict:
    """Add C3 tier/baseline from pdaf_code_final if missing."""
    c3_path = RESULTS_DIR / "pdaf_code_final.json"
    if not c3_path.exists():
        return results
    c3 = json.loads(c3_path.read_text()).get("results", {})
    remap = {
        "pdaf_tp1_baseline": "pdaf_c3_xnode_tp1x4_baseline",
        "pdaf_tp1_tier": "pdaf_c3_xnode_tp1x4_tier",
    }
    out = dict(results)
    for old, new in remap.items():
        if new in out or old not in c3:
            continue
        wl = {k: v for k, v in c3[old].items()
              if k.startswith("code_qps") and isinstance(v, dict)}
        if wl:
            out[new] = wl
    return out


def pick_best_tier_key(
    results: dict | None = None,
    qps: int = QPS_BEST,
) -> tuple[str, float]:
    """Return (scheme_key, energy_j_per_tok) for best AFlex-tier deploy on code."""
    if results is None:
        results = merge_c3_tier(load_sweep_results())
    else:
        results = merge_c3_tier(results)

    best_key, best_e = None, float("inf")
    for key, wl in results.items():
        if not key.endswith("_tier") or not key.startswith("pdaf_"):
            continue
        if isinstance(wl, dict) and wl.get("__status__"):
            continue
        e = _metric(wl.get(f"code_qps{qps}"))
        if e is not None and e < best_e:
            best_key, best_e = key, e
    if best_key is None:
        raise RuntimeError("No valid tier PDAF result for best-config pick")
    return best_key, best_e / 1000.0


def save_best_config(key: str, energy_j: float, out: Path | None = None) -> Path:
    cat, p, d, _ = parse_scheme_key(key)
    out = out or RESULTS_DIR / "pdaf_best_deploy.json"
    payload = {
        "scheme_key": key,
        "baseline_key": key.replace("_tier", "_baseline"),
        "category": cat,
        "p_layout": p,
        "d_layout": d,
        "code_qps16_energy_j_per_tok": energy_j,
    }
    out.write_text(json.dumps(payload, indent=2))
    return out
