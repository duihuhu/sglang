#!/usr/bin/env python3
"""DistServe / BiScale: 6P(TP4) on n1-n3 + 4D(TP2) on n4, macro QPS32."""
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
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
MACRO_DIR = HERE.parents[2] / "macro/scripts"
WL_DIR = MACRO_DIR.parent / "data" / "workloads"
SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/mnt/workspace/lt/sglang"))

sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(MACRO_DIR.parent))
sys.path.insert(0, str(HERE))

import bench_common as BC  # noqa: E402
import run_macro_benchmark as RMB  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "run_4node_scalability_benchmark", str(HERE / "run_4node_scalability_benchmark.py")
)
fn = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(fn)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("pd_6p4d_4n")

RMB.DVFS_PY_SRC = SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "dvfs.py"

NODE1, NODE2, NODE3, NODE4 = fn.NODE1, fn.NODE2, fn.NODE3, fn.NODE4
NODES = fn.NODES
DATASET = "code"
QPS = 32
# 2×P(TP4) per prefill node, 4×D(TP2) on decode node
P_PLACEMENT = [
    (NODE1, 0, fn.P_GROUPS[0]),
    (NODE1, 1, fn.P_GROUPS[1]),
    (NODE2, 2, fn.P_GROUPS[0]),
    (NODE2, 3, fn.P_GROUPS[1]),
    (NODE3, 4, fn.P_GROUPS[0]),
    (NODE3, 5, fn.P_GROUPS[1]),
]
D_PLACEMENT = [(NODE4, i, gpus) for i, gpus in enumerate(fn.D_GROUPS)]

SUB_ROUTER_PORT = 48001
TOPOLOGY = "6P(TP4)(n1,n2,n3)+4D(TP2)(n4)"

# Per-scheme port bases avoid stale ports when running back-to-back.
PORT_BASES = {
    "distserve": {"p": 53100, "d": 53150, "bs": 49100, "nccl_p": 34000, "nccl_d": 34050},
    "biscale": {"p": 54100, "d": 54150, "bs": 50100, "nccl_p": 35000, "nccl_d": 35050},
}


def _kill_pd_ports() -> None:
    ports = list(range(48000, 48010))
    ports += list(range(53100, 53250)) + list(range(54100, 54250))
    ports += list(range(49100, 49200)) + list(range(50100, 50200))
    ports += list(range(34000, 34100)) + list(range(35000, 35100))
    ports += list(range(48500, 51100)) + list(range(54000, 56000, 10))
    for host in NODES:
        fn.kill_ports(host, ports)
    time.sleep(3)


def deploy_pd_6p4d(scheme: str) -> str | None:
    is_biscale = scheme == "biscale"
    tag = f"{scheme}_6p4d"
    bases = PORT_BASES[scheme]
    log.info("Deploy %s: %s (ports P=%d D=%d)", scheme, TOPOLOGY, bases["p"], bases["d"])

    fn.cleanup_all_nodes()
    _kill_pd_ports()
    fn.cleanup_all_nodes()
    time.sleep(10)

    if is_biscale:
        fn.lock_freq_nodes(NODES, fn.ALL_GPUS, None)
    else:
        fn.lock_freq_nodes(NODES, fn.ALL_GPUS, RMB.MAX_GPU_FREQ)

    p_insts: list[dict] = []
    last_host = None
    for host, p_idx, gpus in P_PLACEMENT:
        p_port = bases["p"] + p_idx * 10
        bs_port = bases["bs"] + p_idx * 10
        nccl_port = bases["nccl_p"] + p_idx * 10
        fn.kill_ports(host, [p_port, nccl_port, bs_port, p_port + 1])
        nic = fn.GPU_NIC[gpus[0]]
        csv = ",".join(map(str, gpus))
        dvfs = fn.BISCALE_DVFS if is_biscale else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={host}; "
        )
        extra = (
            f"--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {bs_port} --disaggregation-ib-device {nic} "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {fn.PYTHON} -m sglang.launch_server "
            f"--model-path {fn.MODEL} --tp 4 --host {host} --port {p_port} "
            f"--nccl-port {nccl_port} {fn.FLAGS}{extra}{dvfs}"
        )
        fn.launch(host, cmd, f"{tag}_p{p_idx}")
        p_insts.append({"host": host, "port": p_port, "bs_port": bs_port, "idx": p_idx})
        # Same-node back-to-back P launches need extra gap for model load.
        time.sleep(8 if host == last_host else 4)
        last_host = host

    bootstrap = p_insts[0]["bs_port"]
    d_insts: list[dict] = []
    for host, d_idx, gpus in D_PLACEMENT:
        d_port = bases["d"] + d_idx * 10
        nccl_port = bases["nccl_d"] + d_idx * 10
        fn.kill_ports(host, [d_port, nccl_port, d_port + 1])
        nic = fn.GPU_NIC[gpus[0]]
        csv = ",".join(map(str, gpus))
        dvfs = fn.BISCALE_DVFS if is_biscale else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={host}; "
        )
        extra = (
            f"--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {bootstrap} --disaggregation-ib-device {nic} "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {fn.PYTHON} -m sglang.launch_server "
            f"--model-path {fn.MODEL} --tp 2 --host {host} --port {d_port} "
            f"--nccl-port {nccl_port} {fn.FLAGS}{extra}{dvfs}"
        )
        fn.launch(host, cmd, f"{tag}_d{d_idx}")
        d_insts.append({"host": host, "port": d_port, "idx": d_idx})
        time.sleep(3)

    for inst in p_insts:
        name = f"{tag}_p{inst['idx']}"
        if not fn.wait_health_with_log(inst["host"], inst["port"], name, 600):
            log.error("Prefill %s on %s:%d failed", name, inst["host"], inst["port"])
            return None
    for inst in d_insts:
        name = f"{tag}_d{inst['idx']}"
        if not fn.wait_health_with_log(inst["host"], inst["port"], name, 600):
            log.error("Decode %s on %s:%d failed", name, inst["host"], inst["port"])
            return None

    fn.kill_ports(NODE1, [SUB_ROUTER_PORT, fn.PROM_PORT_BASE])
    rc_parts = [
        f"setsid {fn.PYTHON} -m sglang_router.launch_router --pd-disaggregation",
        f"--host {NODE1} --port {SUB_ROUTER_PORT}{fn._prom_port_arg(0)}",
    ]
    for inst in p_insts:
        rc_parts.append(f"--prefill http://{inst['host']}:{inst['port']} {inst['bs_port']}")
    for inst in d_insts:
        rc_parts.append(f"--decode http://{inst['host']}:{inst['port']}")
    fn.launch(NODE1, " ".join(rc_parts), f"{tag}_subrouter")
    if not fn.wait_health_with_log(NODE1, SUB_ROUTER_PORT, f"{tag}_subrouter", 120):
        log.error("Sub-router failed on %s:%d", NODE1, SUB_ROUTER_PORT)
        return None
    return f"http://{NODE1}:{SUB_ROUTER_PORT}"


def run_scheme(scheme: str, dataset: str) -> dict:
    out = ROOT / "results" / f"{scheme}_6p4d_{dataset}_qps32_4node.json"
    wl = WL_DIR / f"macro_{dataset}_qps{QPS}.jsonl"
    node_gpus = {h: fn.ALL_GPUS[:] for h in NODES}

    payload = {
        "meta": {
            "benchmark": f"{scheme}_6p4d_{dataset}_qps32_4node",
            "topology": TOPOLOGY,
            "nodes": 4,
            "gpus": 32,
            "dataset": dataset,
            "qps": QPS,
        },
        "deploy": {
            "scheme": scheme.capitalize() if scheme == "distserve" else "BiScale",
            "topology": TOPOLOGY,
            "prefill": "6×P(TP4) on n1,n2,n3 (2 per node)",
            "decode": "4×D(TP2) on n4",
            "nodes": 4,
            "gpus": 32,
            "freq": "max lock 1410MHz" if scheme == "distserve" else "BiScale DVFS",
        },
    }

    try:
        url = deploy_pd_6p4d(scheme)
        if not url:
            payload["result"] = {"status": "DEPLOY_FAILED"}
            return payload
        if not fn.test_generate(url, timeout=180):
            payload["result"] = {"status": "WARMUP_FAILED"}
            return payload
        time.sleep(3)

        reqs = [json.loads(line) for line in wl.read_text().splitlines() if line.strip()]
        last = max(r["arrival_time_s"] for r in reqs)
        run_s = int(min(max(last + 150, 300), 900))
        log.info("Workload: %d reqs, run_window=%ds", len(reqs), run_s)

        result = asyncio.run(fn._run_workload(url, reqs, run_s, node_gpus))
        result = BC.recompute_energy_per_total_token(result, BC.wl_key(dataset, QPS))
        result["topology"] = TOPOLOGY
        payload["result"] = result
        log.info(
            "DONE %s: %s thpt=%.1f TTFT_p50=%.1f TPOT_p50=%.1f E/tok=%.1f mJ",
            scheme,
            result.get("status"),
            result.get("throughput_tok_s", 0),
            result.get("ttft_proc_p50_ms", 0),
            result.get("tpot_p50_ms", 0),
            result.get("energy_per_token_mj", 0) or 0,
        )
        return payload
    except Exception:
        payload["result"] = {"status": "ERROR", "error": traceback.format_exc()}
        raise
    finally:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n")
        try:
            fn.cleanup_all_nodes()
            fn.lock_freq_nodes(NODES, fn.ALL_GPUS, None)
        except Exception:
            log.exception("cleanup failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="code", choices=["code", "conv"])
    parser.add_argument(
        "--schemes",
        default="distserve,biscale",
        help="comma-separated schemes (default: distserve,biscale)",
    )
    args = parser.parse_args()
    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]

    if not RMB.DVFS_PY_SRC.exists():
        raise FileNotFoundError(f"dvfs.py not found: {RMB.DVFS_PY_SRC}")

    for scheme in schemes:
        log.info("=" * 60)
        log.info("Running %s %s %s QPS%d", scheme, TOPOLOGY, args.dataset, QPS)
        log.info("=" * 60)
        run_scheme(scheme, args.dataset)
        time.sleep(30)


if __name__ == "__main__":
    main()
