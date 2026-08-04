#!/usr/bin/env python3
"""Re-run AFlex chatbot_lphd QPS16 with corrected energy scope (6 active GPUs only).

Config: k_p=1, k_d=2, TP_PA=1, TP_PF=1, TP_DA=1, TP_DF=1
        f_PA=930, f_PF=930, f_DA=930, f_DF=1410
Active GPUs: node1 [0,1,2,3,4,5], node2 [] (none)
Energy measured: only 6 active GPUs on node1.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)

root = Path("/workspace/sglang")
macro = root / "benchmark/AFlex_bench/multi_node/more_test/macro"
tier1_dir = root / "benchmark/AFlex_bench/multi_node/more_test/macro/scripts/other_tier1"

sys.path.insert(0, str(macro))
sys.path.insert(0, str(tier1_dir))
sys.path.insert(0, str(root / "benchmark/AFlex_bench/multi_node/more_test/macro/scripts"))

os.environ["MN_NODE1_IP"] = "10.252.129.36"
os.environ["MN_NODE2_IP"] = "10.252.129.35"

import bench_common as BC
import bench_tier1_v2 as BT2
import deploy_schemes as DS
import run_macro_benchmark as RMB

DS._sync_sweep_nodes()
BT2.RMB.NODE1_IP = "10.252.129.36"
BT2.RMB.NODE2_IP = "10.252.129.35"

OUT_DIR = Path(__file__).resolve().parent / "data"
OUT_DIR.mkdir(parents=True, exist_ok=True)

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

# Config
cfg = BT2.Tier1TestConfig(
    name="aflex_chatbot_lphd_q16",
    k_p=1,
    k_d=2,
    tp_pa=1,
    tp_pf=1,
    tp_da=1,
    tp_df=1,
    f_pa=930,
    f_pf=930,
    f_da=930,
    f_df=1410,
    tier=True,
)
log.info("CONFIG: %s GPU=%d k_p=%d k_d=%d", cfg.name, cfg.total_gpu(), cfg.k_p, cfg.k_d)

BT2.DATASET = "chatbot_lphd"
BT2.QPS = 16
BT2.RMB.TTFT_SLO_MS = 5000.0
BT2.RMB.TPOT_SLO_MS = 300.0

from freq_timeline_utils import (
    ensure_container_log_dir,
    patch_tier1_afd_dvfs_log,
    restore_tier1_afd_env,
)

DS.RMB.cleanup_all()
time.sleep(8)

ensure_container_log_dir("chatbot_q16_energy_cor")
patch_orig = patch_tier1_afd_dvfs_log("chatbot_q16_energy_cor")

urls = BT2.deploy(cfg)
if urls is None:
    log.error("DEPLOY_FAILED")
    restore_tier1_afd_env(patch_orig)
    raise SystemExit(1)

if not DS.RMB.test_generate(urls[0]):
    log.error("WARMUP_FAILED")
    DS.RMB.cleanup_all()
    restore_tier1_afd_env(patch_orig)
    raise SystemExit(1)

time.sleep(3)
summary = BT2.run_benchmark(urls, cfg)

DS.RMB.unlock_freq_both(list(range(8)))
DS.RMB.cleanup_all()
restore_tier1_afd_env(patch_orig)

if isinstance(summary, dict) and summary.get("status") in ("PASS", "PARTIAL_TIMEOUT"):
    log.info(
        "RESULT: status=%s thpt=%.1f TTFT_p50=%.1f TPOT_p50=%.1f "
        "E_n1=%.1fJ E_n2=%.1fJ E_tot=%.1fJ E/tok=%.2f mJ",
        summary["status"],
        summary["throughput_tok_s"],
        summary.get("ttft_proc_p50_ms", 0),
        summary.get("tpot_p50_ms", 0),
        summary.get("energy_node1_j", 0),
        summary.get("energy_node2_j", 0),
        summary.get("total_energy_j", 0),
        summary.get("energy_per_token_mj", 0),
    )
else:
    log.error("RESULT_FAIL: %s", summary)

out_file = OUT_DIR / "chatbot_q16_energy_corrected.json"
out_file.write_text(
    json.dumps(
        {
            "config": {
                "name": cfg.name,
                "k_p": cfg.k_p,
                "k_d": cfg.k_d,
                "tp_pa": cfg.tp_pa,
                "tp_pf": cfg.tp_pf,
                "tp_da": cfg.tp_da,
                "tp_df": cfg.tp_df,
                "f_pa": cfg.f_pa,
                "f_pf": cfg.f_pf,
                "f_da": cfg.f_da,
                "f_df": cfg.f_df,
                "gpus": cfg.total_gpu(),
                "energy_scope": "active_gpus_only",
                "energy_hosts": {"10.252.129.36": [0, 1, 2, 3, 4, 5], "10.252.129.35": []},
            },
            "result": summary,
        },
        indent=2,
    )
)
log.info("SAVED %s", out_file)
