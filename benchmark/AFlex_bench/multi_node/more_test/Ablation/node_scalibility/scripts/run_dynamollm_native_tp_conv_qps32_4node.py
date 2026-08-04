#!/usr/bin/env python3
"""DynamoLLM native TP sweep on 4 nodes (32 GPU): TP1 / TP2 / TP4 / TP8, conv QPS32."""
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
log = logging.getLogger("dynamollm_tp_sweep")

RMB.DVFS_PY_SRC = SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "dvfs.py"

DATASET = "conv"
QPS = 32
SCHEME = "dynamollm"
NODE_GPUS = {h: fn.ALL_GPUS[:] for h in fn.NODES}

TP_GROUPS: dict[int, list[list[int]]] = {
    1: [[g] for g in range(8)],
    2: fn.TP2_GROUPS,
    4: fn.TP4_GROUPS,
    8: [fn.ALL_GPUS],
}

TP_PORTS: dict[int, dict[str, int]] = {
    1: {"port": 52000, "nccl": 32000, "router": 48002, "prom": 22},
    2: {"port": 53200, "nccl": 33300, "router": 48005, "prom": 20},
    4: {"port": 53400, "nccl": 33500, "router": 48003, "prom": 30},
    8: {"port": 53600, "nccl": 33700, "router": 48004, "prom": 32},
}


def _kill_tp_ports(tp: int) -> None:
    cfg = TP_PORTS[tp]
    ports = list(range(cfg["router"], cfg["router"] + 5))
    n_inst = len(fn.NODES) * len(TP_GROUPS[tp])
    ports += list(range(cfg["port"], cfg["port"] + n_inst * 10 + 20))
    ports += list(range(cfg["nccl"], cfg["nccl"] + n_inst * 10 + 50, 10))
    ports += list(range(48000, 48010)) + list(range(53200, 53700))
    for host in fn.NODES:
        fn.kill_ports(host, ports)
    time.sleep(3)


def deploy_dynamollm_tp(tp: int) -> str | None:
    groups = TP_GROUPS[tp]
    cfg = TP_PORTS[tp]
    n_inst = len(fn.NODES) * len(groups)
    tag = f"tp{tp}"
    log.info("Deploy DynamoLLM %dxTP%d (%d/node, %d instances)", n_inst, tp, len(groups), n_inst)

    fn.cleanup_all_nodes()
    _kill_tp_ports(tp)
    fn.cleanup_all_nodes()
    time.sleep(8)

    insts: list[tuple[str, list[int], int, int]] = []
    idx = 0
    for host in fn.NODES:
        for g in groups:
            insts.append((host, g, cfg["port"] + idx * 10, idx))
            idx += 1

    def _launch_one(host: str, gpus: list[int], port: int, i: int) -> None:
        nccl_port = cfg["nccl"] + i * 10
        fn.kill_ports(host, [port, nccl_port, port + 1])
        csv = ",".join(map(str, gpus))
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={host}; "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {fn.PYTHON} -m sglang.launch_server "
            f"--model-path {fn.MODEL} --tp {tp} --host {host} --port {port} "
            f"--nccl-port {nccl_port} {fn.FLAGS}{fn.NATIVE_DVFS}"
        )
        fn.launch(host, cmd, f"dynamollm_{tag}_{i}")

    last_host = None
    gap = 2 if tp == 1 else 3
    for host, gpus, port, i in insts:
        _launch_one(host, gpus, port, i)
        time.sleep(gap + (4 if host == last_host and tp == 1 else 0))
        last_host = host

    time.sleep(10)
    for host, gpus, port, i in insts:
        log_name = f"dynamollm_{tag}_{i}"
        if fn.wait_health_with_log(host, port, log_name, 900):
            continue
        log.warning("TP%d instance %d on %s:%d failed, retrying once", tp, i, host, port)
        fn.kill_ports(host, [port, cfg["nccl"] + i * 10, port + 1])
        _launch_one(host, gpus, port, i)
        time.sleep(15)
        if not fn.wait_health_with_log(host, port, log_name, 900):
            log.error("TP%d instance %d on %s:%d failed after retry", tp, i, host, port)
            return None

    fn.lock_freq_nodes(fn.NODES, fn.ALL_GPUS, RMB.MAX_GPU_FREQ)
    log.info("DynamoLLM DVFS baseline locked to %dMHz", RMB.MAX_GPU_FREQ)

    router_port = cfg["router"]
    workers = " ".join(f"http://{h}:{p}" for h, _, p, _ in insts)
    fn.kill_ports(fn.NODE1, [router_port, fn.PROM_PORT_BASE + cfg["prom"]])
    rc = (
        f"setsid {fn.PYTHON} -m sglang_router.launch_router --host {fn.NODE1} "
        f"--port {router_port} --policy round_robin --worker-urls {workers}"
        f"{fn._prom_port_arg(cfg['prom'])}"
    )
    fn.launch(fn.NODE1, rc, f"dynamollm_{tag}_router")
    if not fn.wait_health(fn.NODE1, router_port, 180):
        return None
    return f"http://{fn.NODE1}:{router_port}"


def apply_tpot_slo(tpot_slo_ms: float) -> None:
    fn.TPOT_SLO_MS = tpot_slo_ms
    RMB.TPOT_SLO_MS = tpot_slo_ms
    BC.TPOT_SLO_MS = tpot_slo_ms
    fn.NATIVE_DVFS = (
        f" --dvfs-enabled --dvfs-energy-model-dir {fn.ENERGY_MODEL_V1} "
        f"--dvfs-ttft-slo-ms {int(fn.TTFT_SLO_MS)} "
        f"--dvfs-tpot-slo-us {int(tpot_slo_ms * 1000)}"
    )


def run_tp(tp: int, tpot_slo_ms: float) -> dict:
    slo_tag = f"_tpot{int(tpot_slo_ms)}" if tpot_slo_ms != 100.0 else ""
    out = ROOT / "results" / f"dynamollm_tp{tp}_conv_qps32{slo_tag}_4node.json"
    wl = WL_DIR / f"macro_{DATASET}_qps{QPS}.jsonl"
    n_inst = len(fn.NODES) * len(TP_GROUPS[tp])
    topology = f"{n_inst}xTP{tp}"

    payload = {
        "meta": {
            "benchmark": f"dynamollm_tp{tp}_conv_qps32{slo_tag}_4node",
            "scheme": "DynamoLLM",
            "topology": topology,
            "nodes": 4,
            "gpus": 32,
            "dataset": DATASET,
            "qps": QPS,
            "tp": tp,
            "instances": n_inst,
            "tpot_slo_ms": tpot_slo_ms,
        },
        "deploy": {
            "scheme": "DynamoLLM",
            "topology": topology,
            "instances_per_node": len(TP_GROUPS[tp]),
            "nodes": 4,
            "gpus": 32,
            "freq": "unified DVFS",
            "tpot_slo_ms": tpot_slo_ms,
        },
    }

    try:
        url = deploy_dynamollm_tp(tp)
        if not url:
            payload["result"] = {"status": "DEPLOY_FAILED"}
            return payload
        if not fn.test_generate(url, timeout=240):
            payload["result"] = {"status": "WARMUP_FAILED"}
            return payload
        time.sleep(3)

        reqs = [json.loads(line) for line in wl.read_text().splitlines() if line.strip()]
        last = max(r["arrival_time_s"] for r in reqs)
        run_s = int(min(max(last + 150, 300), 900))
        log.info("Workload: %d reqs, run_window=%ds", len(reqs), run_s)

        result = asyncio.run(fn._run_workload(url, reqs, run_s, NODE_GPUS))
        result = BC.recompute_energy_per_total_token(result, BC.wl_key(DATASET, QPS))
        result["topology"] = topology
        result["tp"] = tp
        result["instances"] = n_inst
        payload["result"] = result
        log.info(
            "DONE TP%d: %s thpt=%.1f TTFT_p50=%.1f TPOT_p50=%.1f E/tok=%.1f mJ",
            tp,
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
            fn.lock_freq_nodes(fn.NODES, fn.ALL_GPUS, None)
        except Exception:
            log.exception("cleanup failed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tp", default="1,2,4,8", help="comma-separated TP sizes (default: 1,2,4,8)")
    parser.add_argument(
        "--tpot-slo-ms",
        type=float,
        default=100.0,
        help="TPOT SLO for DVFS and violation counting (default: 100)",
    )
    args = parser.parse_args()
    tps = [int(x.strip()) for x in args.tp.split(",") if x.strip()]

    if not RMB.DVFS_PY_SRC.exists():
        raise FileNotFoundError(f"dvfs.py not found: {RMB.DVFS_PY_SRC}")

    apply_tpot_slo(args.tpot_slo_ms)
    log.info("TPOT SLO set to %.0f ms (DVFS emergency brake at %.0f ms)",
             args.tpot_slo_ms, args.tpot_slo_ms * 0.65)

    for tp in tps:
        if tp not in TP_GROUPS:
            raise ValueError(f"unsupported TP={tp}, choose from {sorted(TP_GROUPS)}")
        log.info("=" * 60)
        log.info(
            "Running DynamoLLM %dxTP%d conv QPS%d TPOT_SLO=%.0fms",
            len(fn.NODES) * len(TP_GROUPS[tp]), tp, QPS, args.tpot_slo_ms,
        )
        log.info("=" * 60)
        run_tp(tp, args.tpot_slo_ms)
        time.sleep(30)


if __name__ == "__main__":
    main()
