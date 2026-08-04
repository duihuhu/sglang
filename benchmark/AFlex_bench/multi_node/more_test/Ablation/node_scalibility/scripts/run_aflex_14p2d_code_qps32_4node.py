#!/usr/bin/env python3
"""AFlex 14P(TP1)+2D(TP1) on 4 nodes (32 GPU), code QPS32.

Each P/D instance = PA+PF or DA+DF = 2 GPUs at TP1.
  - n1,n2,n3: 4 P each  -> 12 P  (24 GPU)
  - n4:       2 P + 2 D  -> 14 P + 2 D (8 GPU)
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MACRO_DIR = HERE.parents[2] / "macro/scripts"
BT2_DIR = MACRO_DIR / "other_tier1"
WL_DIR = MACRO_DIR.parent / "data" / "workloads"
OUT = ROOT / "results" / "aflex_14p2d_code_qps32_4node.json"

NODE1 = os.environ.get("BENCH_NODE1", "10.252.129.36")
NODE2 = os.environ.get("BENCH_NODE2", "10.252.129.35")
NODE3 = os.environ.get("BENCH_NODE3", "10.252.129.34")
NODE4 = os.environ.get("BENCH_NODE4", "10.252.129.33")
NODES = [NODE1, NODE2, NODE3, NODE4]

SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/mnt/workspace/lt/sglang"))

stub = types.ModuleType("run_fixed_6scheme_7dataset")
stub.MAX_RUN_S = 400
stub._wl_key = lambda ds, qps: f"{ds}_qps{qps}"
stub._workload_file = lambda ds, qps: WL_DIR / f"macro_{ds}_qps{qps}.jsonl"
sys.modules["run_fixed_6scheme_7dataset"] = stub
sys.path[0:0] = [str(MACRO_DIR), str(BT2_DIR), str(HERE)]

import run_macro_benchmark as RMB  # noqa: E402
import bench_tier1_v2 as BT2  # noqa: E402

_four_node_spec = importlib.util.spec_from_file_location(
    "run_4node_scalability_benchmark",
    str(HERE / "run_4node_scalability_benchmark.py"),
)
four_node = importlib.util.module_from_spec(_four_node_spec)
assert _four_node_spec.loader is not None
_four_node_spec.loader.exec_module(four_node)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("aflex_14p2d_4n")

RMB.NODE1_IP = NODE1
RMB.NODE2_IP = NODE2
RMB.TTFT_SLO_MS = 2000.0
RMB.TPOT_SLO_MS = 100.0
RMB.DVFS_PY_SRC = SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "dvfs.py"
if not RMB.DVFS_PY_SRC.exists():
    raise FileNotFoundError(f"dvfs.py not found: {RMB.DVFS_PY_SRC}")

DATASET = "code"
QPS = 32
NODE_GPUS = {h: list(range(8)) for h in NODES}

# Dedicated port range (avoid stale qps16 deploy on 43200/45000)
PORT_OFFSET = 3000
BT2.PREFILL_PORT_BASE = 43200 + PORT_OFFSET   # 46200, 14P -> up to 46330
BT2.DECODE_PORT_BASE = 43020 + PORT_OFFSET    # 46020, 2D  -> 46020/46040
BT2.SUB_ROUTER_BASE = 45000 + PORT_OFFSET     # 48000, 14 sub-routers -> 48013
BT2.NCCL_PORT_BASE = 37300 + 2000             # 39300

CFG = BT2.Tier1TestConfig(
    name="aflex_14p2d_code_qps32",
    k_p=14,
    k_d=2,
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


def _pair_gpus(slot: int) -> tuple[list[int], list[int]]:
    """Return (attn_gpus, ffn_gpus) for TP1 pair at slot 0..3."""
    base = slot * 2
    return [base + 1], [base]


def plan_allocation_14p2d_4node(cfg: BT2.Tier1TestConfig) -> dict:
    """n1-n3: 4 P each; n4: 2 D + 2 P (decode on GPU 0-3, prefill on 4-7)."""
    spec = cfg.resolved_prefill_specs()[0]
    p_pairs: list[tuple[str, list[int], list[int], BT2.PrefillSpec]] = []

    for host in (NODE1, NODE2, NODE3):
        for slot in range(4):
            attn, ffn = _pair_gpus(slot)
            p_pairs.append((host, attn, ffn, spec))

    for slot in range(2):
        attn, ffn = _pair_gpus(2 + slot)  # GPU pairs (4,5) and (6,7)
        p_pairs.append((NODE4, attn, ffn, spec))

    decode_instances = []
    for slot in range(2):
        attn, ffn = _pair_gpus(slot)  # GPU pairs (0,1) and (2,3)
        decode_instances.append({"host": NODE4, "attn": attn, "ffn": ffn})

    fmap = {h: {} for h in NODES}
    for host, attn, ffn, pspec in p_pairs:
        for g in attn:
            fmap[host][g] = pspec.f_pa
        for g in ffn:
            fmap[host][g] = pspec.f_pf
    for d in decode_instances:
        for g in d["attn"]:
            fmap[d["host"]][g] = cfg.f_da
        for g in d["ffn"]:
            fmap[d["host"]][g] = cfg.f_df

    return {
        "p_pairs": p_pairs,
        "decode": decode_instances[0],
        "decode_instances": decode_instances,
        "freq_map": fmap,
        "total_gpu": cfg.total_gpu(),
        "energy_hosts": NODE_GPUS,
    }


def ensure_remote_log_dirs() -> None:
    for host in NODES:
        RMB.dexec_on_host(host, f"mkdir -p {RMB.LOG_C}")


def kill_stale_ports() -> None:
    """Kill leftover AFD/router ports from prior failed runs."""
    legacy_ports = list(range(43000, 43450)) + list(range(44990, 45120))
    new_ports = (
        list(range(BT2.DECODE_PORT_BASE, BT2.DECODE_PORT_BASE + 60, 2))
        + list(range(BT2.PREFILL_PORT_BASE, BT2.PREFILL_PORT_BASE + 150, 2))
        + list(range(BT2.SUB_ROUTER_BASE, BT2.SUB_ROUTER_BASE + 20))
    )
    ports = sorted(set(legacy_ports + new_ports))
    for host in NODES:
        four_node.kill_ports(host, ports)
    time.sleep(2)


def sync_dvfs_all_nodes() -> None:
    for host in NODES:
        RMB.sync_dvfs_py_to_host(host)


def log_allocation(alloc: dict) -> None:
    log.info("Layout 14P(TP1)+2D(TP1): %d GPU", alloc["total_gpu"])
    for i, (host, attn, ffn, _) in enumerate(alloc["p_pairs"]):
        log.info("  P%d: %s attn=%s ffn=%s", i, host.split(".")[-1], attn, ffn)
    for i, d in enumerate(alloc["decode_instances"]):
        log.info("  D%d: %s attn=%s ffn=%s", i, d["host"].split(".")[-1], d["attn"], d["ffn"])


async def run_workload(urls: list[str], reqs: list[dict], run_s: int) -> dict:
    import aiohttp
    import numpy as np
    from collections import Counter

    before = four_node.read_energy_map(NODE_GPUS)
    generate_urls = [u if u.endswith("/generate") else u + "/generate" for u in urls]
    assigned = BT2.assign_prefill_urls(urls, CFG, len(reqs))
    for req, target in zip(reqs, assigned):
        req["_target_url"] = target

    results: list[dict] = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=run_s + 120)) as session:
        base_time = time.monotonic()

        async def send_indexed(index: int, req: dict) -> None:
            await RMB.send_one(session, req["_target_url"], req, base_time, results)

        tasks = [asyncio.create_task(send_indexed(i, r)) for i, r in enumerate(reqs)]
        done, pending = await asyncio.wait(tasks, timeout=run_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)
        duration_s = time.monotonic() - base_time

    after = four_node.read_energy_map(NODE_GPUS)
    total_energy_j = four_node.energy_delta_j(before, after, NODE_GPUS)
    ok = [r for r in results if r.get("success")]
    unsuccessful = len(reqs) - len(ok)
    src = [r.get("ttft_proc_ms") or r.get("ttft_ms", 0) for r in ok]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    total_output_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    total_input_tokens = sum(
        reqs[r.get("request_index", i)]["input_len"]
        for i, r in enumerate(ok)
        if r.get("request_index") is not None
    )
    if not total_input_tokens:
        total_input_tokens = sum(r.get("input_len", 0) for r in ok)
    total_tokens_all = total_input_tokens + total_output_tokens
    violated = sum(1 for v in src if v > RMB.TTFT_SLO_MS) + sum(
        1 for v in tpots if v > RMB.TPOT_SLO_MS
    )

    return {
        "status": "PASS" if ok and not unsuccessful else ("PARTIAL" if ok else "FAIL"),
        "dataset": DATASET,
        "qps": QPS,
        "nodes": 4,
        "gpus": 32,
        "topology": "n1-n3:12P(4/node), n4:2P+2D",
        "k_p": CFG.k_p,
        "k_d": CFG.k_d,
        "sub_routers": len(urls),
        "total_requests": len(reqs),
        "successful": len(ok),
        "unsuccessful_requests": unsuccessful,
        "throughput_tok_s": round(total_output_tokens / duration_s, 1) if duration_s else 0,
        "ttft_proc_avg_ms": round(float(np.mean(src)), 1) if src else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(src, 50)), 1) if src else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(src, 99)), 1) if src else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens_all, 2)
        if total_tokens_all
        else 0,
        "energy_denominator": "input_plus_output",
        "slo_violation_rate": round((violated + unsuccessful) / len(reqs) * 100, 1) if reqs else 0,
        "slo_violating_requests": violated,
        "route_distribution": dict(Counter(r.get("target_url", "unknown") for r in results)),
        "assigned_route_distribution": dict(Counter(r["_target_url"] for r in reqs)),
        "duration_s": round(duration_s, 1),
    }


def main() -> None:
    BT2.DATASET = DATASET
    BT2.QPS = QPS
    BT2.plan_allocation = plan_allocation_14p2d_4node  # type: ignore[method-assign]

    wl = WL_DIR / f"macro_{DATASET}_qps{QPS}.jsonl"
    if not wl.exists():
        raise FileNotFoundError(wl)

    alloc = plan_allocation_14p2d_4node(CFG)
    log_allocation(alloc)

    payload = {
        "meta": {
            "benchmark": "aflex_14p2d_code_qps32_4node",
            "topology": "n1-n3:12P(4/node), n4:2P+2D",
            "nodes": 4,
            "gpus": 32,
            "dataset": DATASET,
            "qps": QPS,
            "nodes_list": NODES,
        },
        "config": {
            "k_p": CFG.k_p,
            "k_d": CFG.k_d,
            "tp_pa": CFG.tp_pa,
            "tp_pf": CFG.tp_pf,
            "tp_da": CFG.tp_da,
            "tp_df": CFG.tp_df,
            "f_pa": CFG.f_pa,
            "f_pf": CFG.f_pf,
            "f_da": CFG.f_da,
            "f_df": CFG.f_df,
        },
        "allocation": {
            "p_pairs": [(h, a, f) for h, a, f, _ in alloc["p_pairs"]],
            "decode_instances": alloc["decode_instances"],
        },
    }

    try:
        ensure_remote_log_dirs()
        sync_dvfs_all_nodes()
        four_node.cleanup_all_nodes()
        kill_stale_ports()
        four_node.cleanup_all_nodes()
        time.sleep(8)

        urls = BT2.deploy(CFG)
        if not urls:
            payload["result"] = {"status": "DEPLOY_FAILED"}
            return
        log.info("Deployed %d prefill sub-routers on %s", len(urls), NODE1.split(".")[-1])

        if not four_node.test_generate(urls[0], timeout=240):
            payload["result"] = {"status": "WARMUP_FAILED"}
            return
        time.sleep(3)

        with open(wl) as f:
            reqs = [json.loads(line) for line in f if line.strip()]
        last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
        run_s = int(min(max(stub.MAX_RUN_S, last_arrival + 150), 900))
        log.info("Workload: %d reqs, run_window=%ds", len(reqs), run_s)

        summary = asyncio.run(run_workload(urls, reqs, run_s))
        payload["result"] = summary
        log.info(
            "DONE: %s thpt=%.1f TTFT_p50=%.1f TPOT_p50=%.1f E/tok=%.1f mJ",
            summary.get("status"),
            summary.get("throughput_tok_s", 0),
            summary.get("ttft_proc_p50_ms", 0),
            summary.get("tpot_p50_ms", 0),
            summary.get("energy_per_token_mj", 0),
        )
    except Exception:
        payload["result"] = {"status": "ERROR", "error": traceback.format_exc()}
        raise
    finally:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(payload, indent=2) + "\n")
        try:
            four_node.cleanup_all_nodes()
        except Exception:
            log.exception("cleanup failed")


if __name__ == "__main__":
    main()
