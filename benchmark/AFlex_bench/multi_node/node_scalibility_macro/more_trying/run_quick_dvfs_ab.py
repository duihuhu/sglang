#!/usr/bin/env python3
"""Quick A/B: one architecture baseline vs tier after min-energy DVFS fix."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("quick_dvfs_ab")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_more_trying_sweep as MTS
import run_macro_benchmark as RMB

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [2, 8, 16]

ARCH_DEPLOY = {
    "pd_xnode_tp1": ("pd_xnode", 1, "DistServe(b)", "BiScale(新)", "BiScale(旧)"),
    "native_tp1": ("native", 1, "SGLang(b)", "DynamoLLM(新)", "DynamoLLM(旧)"),
}
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

OLD_RESULTS = HERE / "results" / "more_trying_sweep_20260706_041646.json"


def _deploy(arch: str, tier: bool):
    kind, tp, *_ = ARCH_DEPLOY[arch]
    if kind == "pd_xnode":
        return MTS.start_pd_xnode(tp, tier)
    return MTS.start_native_tp_variant(tp, tier)


def run_mode(arch: str, tier: bool):
    name = f"{arch}_{'tier' if tier else 'baseline'}"
    log.info("=" * 60)
    log.info("DEPLOY %s", name)
    gpus = RMB.card_gpus(NGPU)
    RMB.cleanup_all()
    url = _deploy(arch, tier)
    if url is None:
        return name, {"__status__": "DEPLOY_FAILED"}
    RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
    if not RMB.test_generate(url):
        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        return name, {"__status__": "WARMUP_FAILED"}
    time.sleep(3)
    out = {}
    for qps in QPS_LIST:
        res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
        if res:
            out[res[0]] = res[1]
        time.sleep(3)
    RMB.unlock_freq_both(gpus)
    RMB.cleanup_all()
    return name, out


def main():
    import argparse
    import resource

    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", default="pd_xnode_tp1", choices=list(ARCH_DEPLOY))
    args = parser.parse_args()
    arch = args.arch
    _, _, lbl_base, lbl_new, lbl_old = ARCH_DEPLOY[arch]

    resource.setrlimit(resource.RLIMIT_MEMLOCK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    results = {}
    for tier in (False, True):
        name, data = run_mode(arch, tier)
        results[name] = data

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"quick_dvfs_ab_{arch}_{ts}.json"
    payload = {"meta": {"arch": arch, "qps": QPS_LIST, "dvfs": "min_energy_under_slo"}, "results": results}
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)

    old = {}
    if OLD_RESULTS.exists():
        old = json.load(open(OLD_RESULTS)).get("results", {})

    print("\n" + "=" * 90)
    print(f"  QUICK A/B: {arch} | QPS {QPS_LIST} | min-energy DVFS fix")
    print("=" * 90)
    print(f"{'QPS':>4} {lbl_base:>12} {lbl_new:>14} {'节省%':>8} | {lbl_old:>14} {'旧节省%':>8}")
    print("-" * 90)
    base_key = f"{arch}_baseline"
    new_key = f"{arch}_tier"
    old_key = f"{arch}_tier"
    for q in QPS_LIST:
        wl = f"conv_qps{q}"
        eb = results.get(base_key, {}).get(wl, {})
        en = results.get(new_key, {}).get(wl, {})
        eo = old.get(old_key, {}).get(wl, {})
        if eb.get("status") != "PASS" or en.get("status") != "PASS":
            print(f"{q:>4} FAIL")
            continue
        b, n = eb["energy_per_token_mj"], en["energy_per_token_mj"]
        pct = (b - n) / b * 100
        if eo.get("status") == "PASS":
            o = eo["energy_per_token_mj"]
            opct = (b - o) / b * 100
            print(f"{q:>4} {b:>12.1f} {n:>12.1f} {pct:>7.1f}% | {o:>12.1f} {opct:>7.1f}%")
        else:
            print(f"{q:>4} {b:>12.1f} {n:>12.1f} {pct:>7.1f}% | {'N/A':>12} {'N/A':>8}")
    print("=" * 90)
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
