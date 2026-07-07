#!/usr/bin/env python3
"""Validate Tier1 配比 energy optimality at a specified QPS on code dataset.

Compares Tier1 ILP solution vs alternative feasible layouts at the same QPS.
Stops/resumes are handled via PAUSE_BENCH.flag (watchdog respects it).

Usage (on node34):
  python3 run_tier1_energy_validate.py --qps 4
  python3 run_tier1_energy_validate.py --qps 4 --resume
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
log = logging.getLogger("tier1_validate")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parent
RESULTS_DIR = HERE / "results"
LOG_DIR = MACRO_DIR.parent / "logs"
PAUSE_FLAG = HERE / "PAUSE_BENCH.flag"
SOLUTIONS = HERE / "results" / "tier1_code_solutions.json"

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
MAX_RUN_S = 400
NGPU = 16


def _configs_for_qps(qps: int) -> list[Tier1Layout]:
    """Build Tier1 optimal + alternative layouts for comparison."""
    sol_path = SOLUTIONS
    tier1 = None
    if sol_path.exists():
        for row in json.loads(sol_path.read_text()):
            if row["qps"] == qps and row.get("feasible_ilp"):
                s = row["solution"]
                tier1 = Tier1Layout(
                    name="tier1_ilp_locked",
                    k_d=s["k_d"],
                    tp_pa=s["pa"][0], tp_pf=s["pf"][0],
                    tp_da=s["da"][0], tp_df=s["df"][0],
                    f_pa=s["pa"][1], f_pf=s["pf"][1],
                    f_da=s["da"][1], f_df=s["df"][1],
                    tier=False,
                )
                break

    if tier1 is None:
        raise RuntimeError(f"No feasible Tier1 solution for QPS{qps} in {sol_path}")

    alts: list[Tier1Layout] = [
        tier1,
        Tier1Layout(
            name="tier1_ilp_tier2",
            k_d=tier1.k_d,
            tp_pa=tier1.tp_pa, tp_pf=tier1.tp_pf,
            tp_da=tier1.tp_da, tp_df=tier1.tp_df,
            f_pa=tier1.f_pa, f_pf=tier1.f_pf,
            f_da=tier1.f_da, f_df=tier1.f_df,
            tier=True,
        ),
        Tier1Layout(
            name="tier1_topo_maxfreq",
            k_d=tier1.k_d,
            tp_pa=tier1.tp_pa, tp_pf=tier1.tp_pf,
            tp_da=tier1.tp_da, tp_df=tier1.tp_df,
            f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
            tier=False,
        ),
    ]

    # Other feasible ILP points from solutions file (different k_P/k_D topology)
    if sol_path.exists():
        for row in json.loads(sol_path.read_text()):
            if row["qps"] == qps or not row.get("feasible_ilp"):
                continue
            s = row["solution"]
            # only nearby QPS alternatives with same λ target — skip
            pass

    # Empirical sweep winner: C2 intra tp1x4 (baseline + tier)
    for suffix, tier in (("baseline", False), ("tier", True)):
        alts.append(Tier1Layout(
            name=f"c2_intra_tp1x4_{suffix}",
            k_d=0, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1,
            f_pa=1410, f_pf=1410, f_da=1410, f_df=1410,
            tier=tier,
        ))

    # Dedupe by name
    seen = set()
    out = []
    for c in alts:
        if c.name in seen:
            continue
        seen.add(c.name)
        out.append(c)
    return out


def _deploy(cfg: Tier1Layout):
    if cfg.name.startswith("c2_intra_tp1x4"):
        return PD.deploy_c2_intra("tp1x4", tier=cfg.tier)
    return deploy_tier1_layout(cfg)


def _save(results: dict, meta: dict, tag: str = "partial") -> Path:
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"tier1_energy_validate_{tag}_{ts}.json"
    with open(out, "w") as f:
        json.dump({"meta": meta, "results": results}, f, indent=2)
    log.info("Saved %s", out)
    return out


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", type=int, default=4,
                        help="Target QPS (default 4 = Tier1 1P+3D)")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    configs = _configs_for_qps(args.qps)
    gpus = RMB.card_gpus(NGPU)
    wl_key = f"code_qps{args.qps}"

    meta = {
        "qps": args.qps,
        "dataset": "code",
        "wl_key": wl_key,
        "ttft_slo_ms": RMB.TTFT_SLO_MS,
        "tpot_slo_ms": RMB.TPOT_SLO_MS,
        "configs": [c.name for c in configs],
        "tier1_solutions": str(SOLUTIONS),
    }

    results: dict = {}
    if args.resume:
        partials = sorted(RESULTS_DIR.glob("tier1_energy_validate_partial_*.json"))
        if partials:
            results = json.loads(partials[-1].read_text()).get("results", {})
            log.info("Resume from %s", partials[-1].name)

    log.info("=" * 72)
    log.info("TIER1 ENERGY VALIDATE | code QPS%d | %d configs", args.qps, len(configs))
    log.info("=" * 72)

    for cfg in configs:
        if cfg.name in results and results[cfg.name].get("status") == "PASS":
            log.info("SKIP %s (already PASS)", cfg.name)
            continue

        log.info("\n" + "-" * 72)
        log.info("CONFIG %s", cfg.name)
        log.info("-" * 72)

        RMB.cleanup_all()
        time.sleep(3)
        url = _deploy(cfg)
        if url is None:
            results[cfg.name] = {"status": "DEPLOY_FAILED"}
            _save(results, meta)
            continue

        # Tier1 locked freqs are set in deploy_tier1_layout; only boost C2 to 1410 baseline
        if cfg.name.startswith("c2_intra"):
            RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)

        if not RMB.test_generate(url):
            results[cfg.name] = {"status": "WARMUP_FAILED"}
            RMB.cleanup_all()
            _save(results, meta)
            continue

        time.sleep(3)
        res = RMB.run_one_workload(url, "code", args.qps, gpus, gpus, MAX_RUN_S)
        if res is None:
            results[cfg.name] = {"status": "NO_WORKLOAD"}
        else:
            _, summary = res
            results[cfg.name] = summary

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        _save(results, meta)
        time.sleep(5)

    # Rank by energy
    ranked = []
    for name, m in results.items():
        if m.get("status") == "PASS":
            ranked.append((name, m.get("energy_per_token_mj", 1e9)))
    ranked.sort(key=lambda x: x[1])

    meta["ranking"] = [{"config": n, "mJ_per_tok": e} for n, e in ranked]
    if ranked:
        meta["tier1_is_best"] = ranked[0][0].startswith("tier1_ilp")

    out = _save(results, meta, tag="final")
    log.info("=" * 72)
    log.info("DONE -> %s", out.name)
    for i, (n, e) in enumerate(ranked):
        log.info("  #%d %s: %.1f mJ/tok", i + 1, n, e)
    log.info("=" * 72)

    # Clear pause and resume 6-scheme
    if PAUSE_FLAG.exists():
        import subprocess
        resume_sh = HERE / "resume_6scheme.sh"
        if resume_sh.exists():
            log.info("Running resume_6scheme.sh ...")
            subprocess.run(["bash", str(resume_sh)], check=False)
        else:
            PAUSE_FLAG.unlink()


if __name__ == "__main__":
    main()
