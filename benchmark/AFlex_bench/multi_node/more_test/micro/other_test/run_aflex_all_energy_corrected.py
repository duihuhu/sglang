#!/usr/bin/env python3
"""Re-run AFlex for all 4 micro datasets × 4 QPS with corrected energy scope.

Uses node1 + node2 (default 36+35; set MN_NODE2_IP=10.252.129.34 for node3).
Energy measured only on active GPUs (from freq_map).
Configs loaded from micro/data/micro_e2e_*.json (fallback: AFLEX_CONFIGS).

Usage (on node1 HOST):
  python3 run_aflex_all_energy_corrected.py
  MN_NODE2_IP=10.252.129.34 python3 run_aflex_all_energy_corrected.py --datasets rag_hpld,summary_hphd --force
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger("aflex_all")

root = Path("/workspace/sglang")
macro = root / "benchmark/AFlex_bench/multi_node/more_test/macro"
tier1_dir = root / "benchmark/AFlex_bench/multi_node/more_test/macro/scripts/other_tier1"
more_trying = root / "benchmark/AFlex_bench/multi_node/more_test/macro/scripts"

sys.path.insert(0, str(macro))
sys.path.insert(0, str(tier1_dir))
sys.path.insert(0, str(more_trying))
sys.path.insert(0, str(root / "benchmark/AFlex_bench/multi_node/more_test/macro/scripts"))

os.environ.setdefault("MN_NODE1_IP", "10.252.129.36")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.34")

import bench_tier1_v2 as BT2
import deploy_schemes as DS
import run_macro_benchmark as RMB

DS._sync_sweep_nodes()
BT2.RMB.NODE1_IP = os.environ["MN_NODE1_IP"]
BT2.RMB.NODE2_IP = os.environ["MN_NODE2_IP"]
BT2.RMB.TTFT_SLO_MS = 5000.0
BT2.RMB.TPOT_SLO_MS = 300.0

HERE = Path(__file__).resolve().parent
MICRO_DATA = HERE.parent / "data"
OUT_DIR = HERE / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULT_FILE = OUT_DIR / "aflex_all_energy_corrected.json"

# Monkey-patch plan_allocation to inject energy_hosts (only active GPUs)
_original_plan = BT2.plan_allocation


def corrected_plan(cfg):
    alloc = _original_plan(cfg)
    alloc["energy_hosts"] = {
        host: sorted(freq_map.keys()) for host, freq_map in alloc["freq_map"].items()
    }
    log.info("ENERGY_SCOPE %s: %s (total_gpu=%d)", cfg.name, alloc["energy_hosts"], alloc["total_gpu"])
    return alloc


BT2.plan_allocation = corrected_plan

# All AFlex configs per (dataset, qps)
AFLEX_CONFIGS = {
    # qa_lpld
    ("qa_lpld", 2): dict(name="aflex_qa_lpld_q2", k_p=1, k_d=2, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=690, f_df=930),
    ("qa_lpld", 4): dict(name="aflex_qa_lpld_q4", k_p=1, k_d=3, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=930),
    ("qa_lpld", 8): dict(name="aflex_qa_lpld_q8", k_p=1, k_d=3, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=930),
    ("qa_lpld", 16): dict(name="aflex_qa_lpld_q16", k_p=1, k_d=4, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=930),
    # chatbot_lphd
    ("chatbot_lphd", 2): dict(name="aflex_chatbot_lphd_q2", k_p=1, k_d=2, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=1410),
    ("chatbot_lphd", 4): dict(name="aflex_chatbot_lphd_q4", k_p=1, k_d=2, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=1410),
    ("chatbot_lphd", 8): dict(name="aflex_chatbot_lphd_q8", k_p=1, k_d=2, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=1410),
    ("chatbot_lphd", 16): dict(name="aflex_chatbot_lphd_q16", k_p=1, k_d=2, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=1410),
    # rag_hpld
    ("rag_hpld", 2): dict(name="aflex_2p2d_rag_hpld_q2", k_p=2, k_d=2, tp_pa=2, tp_pf=2, tp_da=1, tp_df=1, f_pa=1410, f_pf=1410, f_da=930, f_df=930),
    ("rag_hpld", 4): dict(name="aflex_3p4d_rag_hpld_q4", k_p=3, k_d=4, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=930, f_pf=930, f_da=930, f_df=1410),
    ("rag_hpld", 8): dict(name="aflex_rag_hpld_q8", k_p=4, k_d=4, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=1410, f_pf=1410, f_da=930, f_df=930),
    ("rag_hpld", 16): dict(name="aflex_rag_hpld_q16", k_p=5, k_d=3, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=1410, f_pf=1410, f_da=930, f_df=930),
    # summary_hphd
    ("summary_hphd", 2): dict(name="aflex_summary_hphd_q2", k_p=3, k_d=3, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=1170, f_pf=1410, f_da=1410, f_df=1410),
    ("summary_hphd", 4): dict(name="aflex_summary_hphd_q4", k_p=3, k_d=3, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=1170, f_pf=1410, f_da=1410, f_df=1410),
    ("summary_hphd", 8): dict(name="aflex_3p2d_tp2_summary_hphd_q8", k_p=3, k_d=2, tp_pa=2, tp_pf=2, tp_da=1, tp_df=1, f_pa=1410, f_pf=1410, f_da=930, f_df=930),
    ("summary_hphd", 16): dict(name="aflex_summary_hphd_q16", k_p=3, k_d=3, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1, f_pa=1170, f_pf=1410, f_da=1410, f_df=1410),
}

DATASETS = ["qa_lpld", "chatbot_lphd", "rag_hpld", "summary_hphd"]
QPS_LIST = [2, 4, 8, 16]
CONFIG_KEYS = (
    "name", "k_p", "k_d", "tp_pa", "tp_pf", "tp_da", "tp_df",
    "f_pa", "f_pf", "f_da", "f_df",
)


def load_micro_aflex_config(dataset: str, qps: int) -> dict | None:
    path = MICRO_DATA / f"micro_e2e_{dataset}.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    entry = payload["results"][dataset][f"qps_{qps}"]["aflex"]
    cfg = entry.get("config") or entry.get("deploy", {}).get("config", {})
    out = {k: cfg[k] for k in CONFIG_KEYS if k in cfg}
    return out or None


def resolve_aflex_config(dataset: str, qps: int) -> dict | None:
    cfg = load_micro_aflex_config(dataset, qps)
    if cfg is not None:
        return cfg
    return AFLEX_CONFIGS.get((dataset, qps))


def is_pass(entry: dict | None) -> bool:
    return isinstance(entry, dict) and entry.get("status") in ("PASS", "PARTIAL_TIMEOUT")


def run_one_point(dataset: str, qps: int, results: dict, force: bool = False) -> bool:
    key = f"{dataset}_qps{qps}"
    if not force and is_pass(results.get(key)):
        log.info("Skip PASS: %s", key)
        return True

    cfg_dict = resolve_aflex_config(dataset, qps)
    if cfg_dict is None:
        log.error("No config for %s qps=%d", dataset, qps)
        results[key] = {"status": "NO_CONFIG"}
        return False

    cfg = BT2.Tier1TestConfig(**{k: v for k, v in cfg_dict.items() if k != "name"}, name=cfg_dict["name"])
    log.info("\n" + "=" * 72)
    log.info("RUN %s | GPU=%d | k_p=%d k_d=%d | tp_PA=%d tp_PF=%d tp_DA=%d tp_DF=%d",
             cfg.name, cfg.total_gpu(), cfg.k_p, cfg.k_d, cfg.tp_pa, cfg.tp_pf, cfg.tp_da, cfg.tp_df)
    log.info("    f_PA=%d f_PF=%d f_DA=%d f_DF=%d", cfg.f_pa, cfg.f_pf, cfg.f_da, cfg.f_df)
    log.info("=" * 72)

    BT2.DATASET = dataset
    BT2.QPS = qps

    RMB.cleanup_all()
    time.sleep(8)

    urls = BT2.deploy(cfg)
    if urls is None:
        log.error("DEPLOY_FAILED: %s", key)
        results[key] = {"status": "DEPLOY_FAILED"}
        RMB.cleanup_all()
        return False

    if not RMB.test_generate(urls[0]):
        log.error("WARMUP_FAILED: %s", key)
        results[key] = {"status": "WARMUP_FAILED"}
        RMB.cleanup_all()
        return False

    time.sleep(3)
    summary = BT2.run_benchmark(urls, cfg)

    RMB.unlock_freq_both(list(range(8)))
    RMB.cleanup_all()

    if isinstance(summary, dict) and summary.get("status") in ("PASS", "PARTIAL_TIMEOUT"):
        log.info(
            "RESULT %s: status=%s thpt=%.1f TTFT_p50/p90=%.1f/%.1f TPOT_p50/p90=%.1f/%.1f E/tok=%.2f mJ",
            key,
            summary["status"],
            summary["throughput_tok_s"],
            summary.get("ttft_proc_p50_ms", 0), summary.get("ttft_proc_p90_ms", 0),
            summary.get("tpot_p50_ms", 0), summary.get("tpot_p90_ms", 0),
            summary.get("energy_per_token_mj", 0),
        )
    else:
        log.error("RESULT_FAIL %s: %s", key, summary.get("status") if isinstance(summary, dict) else summary)

    summary["config"] = cfg_dict
    summary["energy_scope"] = "active_gpus_only"
    results[key] = summary
    return is_pass(summary)


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--qps", default=",".join(str(q) for q in QPS_LIST))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    qps_list = [int(x) for x in args.qps.split(",") if x.strip()]

    if RESULT_FILE.exists():
        data = json.loads(RESULT_FILE.read_text())
    else:
        data = {"meta": {}, "results": {}}

    data["meta"].update({
        "benchmark": "aflex_all_micro_energy_corrected",
        "node1": os.environ["MN_NODE1_IP"],
        "node2": os.environ["MN_NODE2_IP"],
        "energy_scope": "active_gpus_only",
        "datasets": datasets,
        "qps": qps_list,
        "updated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    results = data.setdefault("results", {})

    for dataset in datasets:
        for qps in qps_list:
            run_one_point(dataset, qps, results, force=args.force)
            data["meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            RESULT_FILE.write_text(json.dumps(data, indent=2))
            log.info("  Saved -> %s", RESULT_FILE.name)
            time.sleep(5)

    log.info("\n" + "=" * 72)
    log.info("ALL DONE. Results: %s", RESULT_FILE)
    n_pass = sum(1 for v in results.values() if is_pass(v))
    log.info("  %d/%d PASS", n_pass, len(datasets) * len(qps_list))
    log.info("=" * 72)


if __name__ == "__main__":
    main()
