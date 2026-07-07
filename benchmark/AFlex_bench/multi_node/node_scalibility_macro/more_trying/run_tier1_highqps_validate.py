#!/usr/bin/env python3
"""Test fixed Tier1 QPS4配比 (1P+3D) at high QPS vs C2.

Uses the QPS4 ILP solution topology/freqs regardless of target QPS,
to see if a low-QPS-planned layout holds at QPS 8/12/16.

Configs per QPS:
  - tier1_ilp_locked   (1P+3D, PA4@930 PF4@1170, 3×DA1@930)
  - tier1_ilp_tier2    (same + compositional DVFS)
  - c2_intra_tp1x4_tier (sweep winner baseline)

Usage:
  python3 run_tier1_highqps_validate.py --qps-list 8,12,16
"""
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
log = logging.getLogger("tier1_highqps")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
RESULTS_DIR = HERE / "results"
SOLUTIONS = HERE / "results" / "tier1_code_solutions.json"
MAX_RUN_S = 400
NGPU = 16

sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(HERE))

import run_macro_benchmark as RMB
import pdaf_deploy as PD
from deploy_tier1_layout import Tier1Layout, deploy_tier1_layout

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
RMB.WORKLOAD_DIR = MACRO_DIR / "workloads"


def _fixed_tier1_qps4_layouts() -> list[Tier1Layout]:
    """Tier1 ILP solution at QPS4 — reused at all target QPS."""
    row = next(
        r for r in json.loads(SOLUTIONS.read_text())
        if r["qps"] == 4 and r.get("feasible_ilp")
    )
    s = row["solution"]
    base = dict(
        k_d=s["k_d"],
        tp_pa=s["pa"][0], tp_pf=s["pf"][0],
        tp_da=s["da"][0], tp_df=s["df"][0],
        f_pa=s["pa"][1], f_pf=s["pf"][1],
        f_da=s["da"][1], f_df=s["df"][1],
    )
    return [
        Tier1Layout(name="tier1_ilp_locked", tier=False, **base),
        Tier1Layout(name="tier1_ilp_tier2", tier=True, **base),
        Tier1Layout(
            name="c2_intra_tp1x4_tier",
            k_d=0, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1,
            f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
            tier=True,
        ),
    ]


def _deploy(cfg: Tier1Layout):
    if cfg.name.startswith("c2_intra"):
        return PD.deploy_c2_intra("tp1x4", tier=cfg.tier)
    return deploy_tier1_layout(cfg)


def _save(all_results: dict, meta: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"tier1_highqps_validate_{tag}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": all_results}, f, indent=2)
    log.info("Saved %s", out)
    return out


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps-list", default="8,12,16",
                        help="Comma-separated target QPS values")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    qps_list = [int(x) for x in args.qps_list.split(",") if x.strip()]
    configs = _fixed_tier1_qps4_layouts()
    gpus = RMB.card_gpus(NGPU)

    row4 = next(
        r for r in json.loads(SOLUTIONS.read_text())
        if r["qps"] == 4 and r.get("feasible_ilp")
    )
    meta = {
        "dataset": "code",
        "qps_list": qps_list,
        "fixed_layout": "tier1_qps4_ilp_1p3d",
        "tier1_qps4": row4,
        "configs": [c.name for c in configs],
        "ttft_slo_ms": RMB.TTFT_SLO_MS,
        "tpot_slo_ms": RMB.TPOT_SLO_MS,
    }

    all_results: dict = {}
    if args.resume:
        partials = sorted(RESULTS_DIR.glob("tier1_highqps_validate_partial_*.json"))
        if partials:
            all_results = json.loads(partials[-1].read_text()).get("results", {})
            log.info("Resume from %s", partials[-1].name)

    log.info("=" * 72)
    log.info("TIER1 HIGH-QPS | fixed QPS4 1P+3D | QPS=%s", qps_list)
    log.info("=" * 72)

    for qps in qps_list:
        qk = f"code_qps{qps}"
        if qk not in all_results:
            all_results[qk] = {}

        for cfg in configs:
            ck = cfg.name
            if all_results[qk].get(ck, {}).get("status") == "PASS":
                log.info("SKIP %s @ QPS%d", ck, qps)
                continue

            log.info("\n" + "-" * 72)
            log.info("QPS%d | CONFIG %s", qps, ck)
            log.info("-" * 72)

            RMB.cleanup_all()
            time.sleep(3)
            url = _deploy(cfg)
            if url is None:
                all_results[qk][ck] = {"status": "DEPLOY_FAILED"}
                _save(all_results, meta)
                continue

            if cfg.name.startswith("c2_intra"):
                RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)

            if not RMB.test_generate(url):
                all_results[qk][ck] = {"status": "WARMUP_FAILED"}
                RMB.cleanup_all()
                _save(all_results, meta)
                continue

            time.sleep(3)
            res = RMB.run_one_workload(url, "code", qps, gpus, gpus, MAX_RUN_S)
            if res is None:
                all_results[qk][ck] = {"status": "NO_WORKLOAD"}
            else:
                _, summary = res
                all_results[qk][ck] = summary

            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            _save(all_results, meta)
            time.sleep(5)

    # Summary table
    summary_rows = []
    for qps in qps_list:
        qk = f"code_qps{qps}"
        row = {"qps": qps}
        for ck, m in all_results.get(qk, {}).items():
            if m.get("status") == "PASS":
                row[ck] = m.get("energy_per_token_mj")
        ranked = sorted(
            [(k, v) for k, v in row.items() if k != "qps" and isinstance(v, (int, float))],
            key=lambda x: x[1],
        )
        row["best"] = ranked[0][0] if ranked else None
        row["tier1_wins"] = bool(ranked and ranked[0][0].startswith("tier1"))
        summary_rows.append(row)

    meta["summary"] = summary_rows
    out = _save(all_results, meta, tag="final")
    log.info("=" * 72)
    log.info("DONE -> %s", out.name)
    for row in summary_rows:
        log.info("  QPS%d: best=%s tier1_wins=%s", row["qps"], row.get("best"), row.get("tier1_wins"))
        for ck in configs:
            v = all_results.get(f"code_qps{row['qps']}", {}).get(ck.name, {})
            if v.get("status") == "PASS":
                log.info("    %s: %.1f mJ/tok | %.1f tok/s | SLO=%.1f%%",
                         ck.name, v["energy_per_token_mj"], v["throughput_tok_s"],
                         v.get("slo_violation_rate", 0))
    log.info("=" * 72)


if __name__ == "__main__":
    main()
