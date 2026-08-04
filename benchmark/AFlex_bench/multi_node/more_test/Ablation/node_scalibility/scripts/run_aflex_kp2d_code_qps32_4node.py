#!/usr/bin/env python3
"""AFlex kP(TP1)+2D(TP1) on 4 nodes, code QPS32.

Placement: n1-n3 up to 4P each, remainder P on n4 (GPU 4-7); 2D always on n4 GPU 0-3.
Energy counted only on active GPUs (k_p*2 + k_d*2).
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
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
log = logging.getLogger("aflex_kp2d_4n")

RMB.NODE1_IP = NODE1
RMB.NODE2_IP = NODE2
RMB.TTFT_SLO_MS = 2000.0
RMB.TPOT_SLO_MS = 100.0
RMB.DVFS_PY_SRC = SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "dvfs.py"

DATASET = "code"
QPS = 32
K_D = 2

PORT_OFFSET = 3000
BT2.PREFILL_PORT_BASE = 43200 + PORT_OFFSET
BT2.DECODE_PORT_BASE = 43020 + PORT_OFFSET
BT2.SUB_ROUTER_BASE = 45000 + PORT_OFFSET
BT2.NCCL_PORT_BASE = 41000  # avoid stale 393xx from prior runs

# Set per run
CFG: BT2.Tier1TestConfig
ENERGY_GPUS: dict[str, list[int]] = {}
K_P = 0


def _pair_gpus(slot: int) -> tuple[list[int], list[int]]:
    base = slot * 2
    return [base + 1], [base]


def _topology_desc(k_p: int) -> str:
    n4_p = max(0, k_p - 12)
    if n4_p:
        return f"n1-n3:12P(4/node), n4:{n4_p}P+2D"
    n3_extra = k_p - 8
    if n3_extra < 4:
        return f"n1-n2:8P(4/node), n3:{n3_extra}P, n4:2D"
    return f"n1-n3:12P(4/node), n4:2D"


def plan_allocation_kp2d(cfg: BT2.Tier1TestConfig) -> dict:
    """n1-n3: up to 4P each; extra P on n4 GPU 4-7; 2D on n4 GPU 0-3."""
    spec = cfg.resolved_prefill_specs()[0]
    p_pairs: list[tuple[str, list[int], list[int], BT2.PrefillSpec]] = []

    remaining = cfg.k_p
    for host in (NODE1, NODE2, NODE3):
        for slot in range(4):
            if remaining <= 0:
                break
            attn, ffn = _pair_gpus(slot)
            p_pairs.append((host, attn, ffn, spec))
            remaining -= 1
        if remaining <= 0:
            break

    for slot in range(2, 4):
        if remaining <= 0:
            break
        attn, ffn = _pair_gpus(slot)
        p_pairs.append((NODE4, attn, ffn, spec))
        remaining -= 1

    if remaining > 0:
        raise RuntimeError(f"cannot place {cfg.k_p} P pairs")

    decode_instances = []
    for slot in range(K_D):
        attn, ffn = _pair_gpus(slot)
        decode_instances.append({"host": NODE4, "attn": attn, "ffn": ffn})

    fmap: dict[str, dict[int, int]] = {h: {} for h in NODES}
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

    energy: dict[str, set[int]] = {h: set() for h in NODES}
    for host, attn, ffn, _ in p_pairs:
        energy[host].update(attn + ffn)
    for d in decode_instances:
        energy[d["host"]].update(d["attn"] + d["ffn"])

    global ENERGY_GPUS
    ENERGY_GPUS = {h: sorted(g) for h, g in energy.items() if g}

    idle = {h: sorted(set(range(8)) - energy.get(h, set())) for h in NODES}
    idle = {h: g for h, g in idle.items() if g}

    return {
        "p_pairs": p_pairs,
        "decode": decode_instances[0],
        "decode_instances": decode_instances,
        "freq_map": fmap,
        "total_gpu": cfg.total_gpu(),
        "energy_hosts": ENERGY_GPUS,
        "idle_gpus": idle,
    }


def ensure_remote_log_dirs() -> None:
    for host in NODES:
        RMB.dexec_on_host(host, f"mkdir -p {RMB.LOG_C}")


def kill_stale_ports(max_sub_routers: int) -> None:
    legacy_ports = list(range(43000, 43450)) + list(range(44990, 45120))
    new_ports = (
        list(range(BT2.DECODE_PORT_BASE, BT2.DECODE_PORT_BASE + 60, 2))
        + list(range(BT2.PREFILL_PORT_BASE, BT2.PREFILL_PORT_BASE + max_sub_routers * 10, 2))
        + list(range(BT2.SUB_ROUTER_BASE, BT2.SUB_ROUTER_BASE + max_sub_routers + 5))
    )
    # NCCL/TCPStore ports used by torch distributed (step=10 per instance)
    nccl_ports = list(range(39200, 40200, 10)) + list(range(41000, 41400, 10))
    ports = sorted(set(legacy_ports + new_ports + nccl_ports))
    for host in NODES:
        four_node.kill_ports(host, ports)
    time.sleep(2)


def sync_dvfs_all_nodes() -> None:
    for host in NODES:
        RMB.sync_dvfs_py_to_host(host)


def log_allocation(alloc: dict, k_p: int) -> None:
    active = k_p * 2 + K_D * 2
    log.info("Layout %dP(TP1)+2D(TP1): %d active GPU", k_p, active)
    for i, (host, attn, ffn, _) in enumerate(alloc["p_pairs"]):
        log.info("  P%d: %s attn=%s ffn=%s", i, host.split(".")[-1], attn, ffn)
    for i, d in enumerate(alloc["decode_instances"]):
        log.info("  D%d: %s attn=%s ffn=%s", i, d["host"].split(".")[-1], d["attn"], d["ffn"])
    for host, gpus in alloc.get("idle_gpus", {}).items():
        log.info("  idle %s: GPU %s", host.split(".")[-1], gpus)


async def run_workload(urls: list[str], reqs: list[dict], run_s: int, k_p: int) -> dict:
    import aiohttp
    import numpy as np
    from collections import Counter

    before = four_node.read_energy_map(ENERGY_GPUS)
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

    after = four_node.read_energy_map(ENERGY_GPUS)
    total_energy_j = four_node.energy_delta_j(before, after, ENERGY_GPUS)

    ok = [r for r in results if r.get("success")]
    unsuccessful = len(reqs) - len(ok)
    src = [r.get("ttft_proc_ms") or r.get("ttft_ms", 0) for r in ok]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    total_output_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    total_input_tokens = sum(r["input_len"] for r in reqs)
    total_tokens_all = total_input_tokens + total_output_tokens
    violated = sum(1 for v in src if v > RMB.TTFT_SLO_MS) + sum(
        1 for v in tpots if v > RMB.TPOT_SLO_MS
    )
    active_gpus = k_p * 2 + K_D * 2

    return {
        "status": "PASS" if ok and not unsuccessful else ("PARTIAL" if ok else "FAIL"),
        "dataset": DATASET,
        "qps": QPS,
        "nodes": 4,
        "gpus": active_gpus,
        "topology": _topology_desc(k_p),
        "k_p": k_p,
        "k_d": K_D,
        "sub_routers": len(urls),
        "energy_gpus": ENERGY_GPUS,
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
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "total_tokens_all": total_tokens_all,
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


def run_one(k_p: int) -> dict:
    global CFG, K_P
    K_P = k_p
    out = ROOT / "results" / f"aflex_{k_p}p2d_code_qps32_4node.json"

    CFG = BT2.Tier1TestConfig(
        name=f"aflex_{k_p}p2d_code_qps32",
        k_p=k_p,
        k_d=K_D,
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

    BT2.DATASET = DATASET
    BT2.QPS = QPS
    BT2.plan_allocation = plan_allocation_kp2d  # type: ignore[method-assign]

    wl = WL_DIR / f"macro_{DATASET}_qps{QPS}.jsonl"
    if not wl.exists():
        raise FileNotFoundError(wl)

    alloc = plan_allocation_kp2d(CFG)
    log_allocation(alloc, k_p)

    payload = {
        "meta": {
            "benchmark": f"aflex_{k_p}p2d_code_qps32_4node",
            "topology": _topology_desc(k_p),
            "nodes": 4,
            "gpus": k_p * 2 + K_D * 2,
            "dataset": DATASET,
            "qps": QPS,
            "nodes_list": NODES,
        },
        "config": {
            "k_p": k_p,
            "k_d": K_D,
            "tp_pa": 1,
            "tp_pf": 1,
            "tp_da": 1,
            "tp_df": 1,
            "f_pa": 1410,
            "f_pf": 1410,
            "f_da": 930,
            "f_df": 930,
        },
        "allocation": {
            "p_pairs": [(h, a, f) for h, a, f, _ in alloc["p_pairs"]],
            "decode_instances": alloc["decode_instances"],
            "idle_gpus": alloc.get("idle_gpus", {}),
        },
    }

    try:
        ensure_remote_log_dirs()
        sync_dvfs_all_nodes()
        four_node.cleanup_all_nodes()
        kill_stale_ports(k_p)
        four_node.cleanup_all_nodes()
        time.sleep(8)

        urls = BT2.deploy(CFG)
        if not urls:
            payload["result"] = {"status": "DEPLOY_FAILED"}
            return payload

        log.info("Deployed %d prefill sub-routers", len(urls))
        if not four_node.test_generate(urls[0], timeout=240):
            payload["result"] = {"status": "WARMUP_FAILED"}
            return payload
        time.sleep(3)

        with open(wl) as f:
            reqs = [json.loads(line) for line in f if line.strip()]
        last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
        run_s = int(min(max(stub.MAX_RUN_S, last_arrival + 150), 900))
        log.info("Workload: %d reqs, run_window=%ds", len(reqs), run_s)

        summary = asyncio.run(run_workload(urls, reqs, run_s, k_p))
        payload["result"] = summary
        log.info(
            "DONE %dP+2D: %s thpt=%.1f TTFT_p50=%.1f TPOT_p50=%.1f E/tok=%.1f mJ",
            k_p,
            summary.get("status"),
            summary.get("throughput_tok_s", 0),
            summary.get("ttft_proc_p50_ms", 0),
            summary.get("tpot_p50_ms", 0),
            summary.get("energy_per_token_mj", 0),
        )
        return payload
    except Exception:
        payload["result"] = {"status": "ERROR", "error": traceback.format_exc()}
        raise
    finally:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        try:
            four_node.cleanup_all_nodes()
        except Exception:
            log.exception("cleanup failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k-p", type=int, nargs="+", default=[12, 11],
                        help="prefill instance counts to test (default: 12 11)")
    args = parser.parse_args()

    if not RMB.DVFS_PY_SRC.exists():
        raise FileNotFoundError(f"dvfs.py not found: {RMB.DVFS_PY_SRC}")

    for k_p in args.k_p:
        log.info("=" * 60)
        log.info("Running %dP+2D code QPS%d", k_p, QPS)
        log.info("=" * 60)
        run_one(k_p)
        time.sleep(10)


if __name__ == "__main__":
    main()
