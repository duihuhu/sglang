#!/usr/bin/env python3
"""Run AFlex 14P(TP1)+2D(TP1) = k_p=7, k_d=1 on 16 GPU, code QPS16."""
from __future__ import annotations

import json
import logging
import os
import sys
import time
import traceback
import types
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
MACRO_DIR = HERE.parents[2] / "macro/scripts"
BT2_DIR = MACRO_DIR / "other_tier1"
WL_DIR = MACRO_DIR.parent / "data" / "workloads"
OUT = HERE.parent / "results" / "aflex_14p2d_code_qps16.json"
LOG_DIR = HERE.parent / "logs"

SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/mnt/workspace/lt/sglang"))

NODE1 = os.environ.get("BENCH_NODE1", "10.252.129.36")
NODE2 = os.environ.get("BENCH_NODE2", "10.252.129.35")

stub = types.ModuleType("run_fixed_6scheme_7dataset")
stub.MAX_RUN_S = 400
stub._wl_key = lambda ds, qps: f"{ds}_qps{qps}"
stub._workload_file = lambda ds, qps: WL_DIR / f"macro_{ds}_qps{qps}.jsonl"
sys.modules["run_fixed_6scheme_7dataset"] = stub
sys.path[0:0] = [str(MACRO_DIR), str(BT2_DIR)]

import run_macro_benchmark as RMB  # noqa: E402
import bench_tier1_v2 as BT2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aflex_14p2d")

RMB.NODE1_IP = NODE1
RMB.NODE2_IP = NODE2
RMB.TTFT_SLO_MS = 2000.0
RMB.TPOT_SLO_MS = 100.0
RMB.DVFS_PY_SRC = SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "dvfs.py"
if not RMB.DVFS_PY_SRC.exists():
    raise FileNotFoundError(f"dvfs.py not found: {RMB.DVFS_PY_SRC}")

CFG = BT2.Tier1TestConfig(
    name="aflex_14p2d_code_qps16",
    k_p=7,
    k_d=1,
    tp_pa=1,
    tp_pf=1,
    tp_da=1,
    tp_df=1,
    f_pa=1410,
    f_pf=1410,
    f_da=930,
    f_df=930,
    tier=True,
)


def ensure_remote_log_dirs() -> None:
    for host in (NODE1, NODE2):
        RMB.dexec_on_host(host, f"mkdir -p {RMB.LOG_C}")


def main() -> None:
    BT2.DATASET = "code"
    BT2.QPS = 16
    alloc = BT2.plan_allocation(CFG)
    log.info("Layout: 14P(TP1)+2D(TP1) => k_p=%d k_d=%d, %d GPU", CFG.k_p, CFG.k_d, alloc["total_gpu"])
    for i, (host, a, f, _spec) in enumerate(alloc["p_pairs"]):
        log.info("  P%d: %s attn=%s ffn=%s", i, host.split(".")[-1], a, f)
    for i, d in enumerate(alloc["decode_instances"]):
        log.info("  D%d: %s attn=%s ffn=%s", i, d["host"].split(".")[-1], d["attn"], d["ffn"])

    payload = {
        "meta": {
            "benchmark": "aflex_14p2d_code_qps16",
            "topology": "14P(TP1)+2D(TP1)",
            "nodes": 2,
            "gpus": 16,
            "dataset": "code",
            "qps": 16,
            "node1": NODE1,
            "node2": NODE2,
        },
        "config": asdict(CFG),
        "allocation": {
            "p_pairs": [(h, a, f) for h, a, f, _ in alloc["p_pairs"]],
            "decode_instances": alloc["decode_instances"],
        },
    }

    try:
        ensure_remote_log_dirs()
        RMB.cleanup_all()
        time.sleep(8)
        urls = BT2.deploy(CFG)
        if urls is None:
            payload["result"] = {"status": "DEPLOY_FAILED"}
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(payload, indent=2) + "\n")
            return
        if not RMB.test_generate(urls[0]):
            payload["result"] = {"status": "WARMUP_FAILED"}
            OUT.parent.mkdir(parents=True, exist_ok=True)
            OUT.write_text(json.dumps(payload, indent=2) + "\n")
            return
        time.sleep(3)
        summary = BT2.run_benchmark(urls, CFG)
        payload["result"] = summary
        log.info(
            "DONE: status=%s thpt=%.1f TTFT_p50=%.1f TPOT_p50=%.1f E/tok=%.1f mJ",
            summary.get("status"),
            summary.get("throughput_tok_s", 0),
            summary.get("ttft_proc_p50_ms", 0),
            summary.get("tpot_p50_ms", 0),
            summary.get("energy_per_token_mj", 0),
        )
    except Exception:
        payload["result"] = {
            "status": "ERROR",
            "error": traceback.format_exc(),
        }
        raise
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        try:
            RMB.cleanup_all()
        except Exception:
            log.exception("cleanup failed")


if __name__ == "__main__":
    main()
