#!/usr/bin/env python3
"""Shared helpers for macro-e2e-topology no-DVFS benchmarks."""
from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
import traceback
import types
from dataclasses import asdict
from pathlib import Path

NODE3 = "10.252.129.34"
NODE4 = "10.252.129.33"
os.environ["MN_NODE3_IP"] = NODE3
os.environ["MN_NODE4_IP"] = NODE4

HERE = Path(__file__).resolve().parent
ENERGY_ROOT = HERE.parent
MACRO_DIR = HERE.parents[2] / "macro"
MACRO_SCRIPTS = MACRO_DIR / "scripts"
BT2_DIR = MACRO_SCRIPTS / "other_tier1"
WL_DIR = MACRO_DIR / "data" / "workloads"

stub = types.ModuleType("run_fixed_6scheme_7dataset")
stub.MAX_RUN_S = 400
stub._wl_key = lambda ds, qps: f"{ds}_qps{qps}"
stub._workload_file = lambda ds, qps: WL_DIR / f"macro_{ds}_qps{qps}.jsonl"
sys.modules["run_fixed_6scheme_7dataset"] = stub
sys.path[0:0] = [str(HERE), str(MACRO_DIR), str(BT2_DIR), str(MACRO_SCRIPTS)]

import run_macro_benchmark as RMB
import bench_tier1_v2 as BT2

SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/workspace/sglang"))
RMB.DVFS_PY_SRC = SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "dvfs.py"
if not RMB.DVFS_PY_SRC.exists():
    alt = Path("/mnt/workspace/lt/sglang/python/sglang/srt/layers/dvfs.py")
    if alt.exists():
        RMB.DVFS_PY_SRC = alt
    else:
        raise FileNotFoundError(f"dvfs.py not found: {RMB.DVFS_PY_SRC}")

log = logging.getLogger("bench_no_dvfs_common")

DATASETS = ("code", "conv")
QPS_LIST = (2, 4, 8, 16)
TOPOLOGY_FIELDS = ("k_p", "k_d", "tp_pa", "tp_pf", "tp_da", "tp_df")
LOCKED_FREQ_MHZ = 1410

RMB.NODE1_IP = NODE3
RMB.NODE2_IP = NODE4
RMB.TTFT_SLO_MS = 2000.0
RMB.TPOT_SLO_MS = 100.0
CLEANUP_SCRIPT = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"
BENCHMARK_HOSTS = (NODE3, NODE4)
PORT_CHECK_CMD = (
    "ss -tlnp 2>/dev/null | grep -v '127.0.0.1' "
    "| grep -oP ':\\K(43[0-9]{3}|45[0-9]{3})' | sort -u || true"
)


def _occupied_benchmark_ports(host: str) -> list[str]:
    inner = f"docker exec {RMB.CONTAINER} bash -lc {shlex.quote(PORT_CHECK_CMD)}"
    result = subprocess.run(RMB._ssh(host, inner), capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def cleanup_host(host: str) -> None:
    inner = f"docker exec {RMB.CONTAINER} bash -lc {shlex.quote(f'bash {CLEANUP_SCRIPT}')}"
    subprocess.run(RMB._ssh(host, inner), check=False)


def ensure_cluster_clean(*, retries: int = 4, sleep_s: float = 8.0) -> None:
    """Force-clean node3/node4 and verify benchmark ports are released."""
    RMB._LOCAL_NODE1 = None
    for attempt in range(1, retries + 1):
        for host in BENCHMARK_HOSTS:
            cleanup_host(host)
        RMB.cleanup_all()
        time.sleep(sleep_s)
        occupied = {host: _occupied_benchmark_ports(host) for host in BENCHMARK_HOSTS}
        busy = {host: ports for host, ports in occupied.items() if ports}
        if not busy:
            log.info("Cluster cleanup complete; benchmark ports are free")
            return
        log.warning("Benchmark ports still occupied after cleanup attempt %d: %s", attempt, busy)
        for host, ports in busy.items():
            RMB._kill_ports_on_host(host, [f":{port}" for port in ports])
        time.sleep(sleep_s)
    raise RuntimeError(f"failed to free benchmark ports on cluster: {busy}")


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def used_gpu_map(allocation: dict) -> dict[str, list[int]]:
    used = {NODE3: set(), NODE4: set()}
    for pair in allocation["p_pairs"]:
        host, attn, ffn = pair[0], pair[1], pair[2]
        used[host].update(attn)
        used[host].update(ffn)
    for instance in allocation["decode_instances"]:
        used[instance["host"]].update(instance["attn"])
        used[instance["host"]].update(instance["ffn"])
    return {host: sorted(gpus) for host, gpus in used.items() if gpus}


def lock_used_gpus(gpu_map: dict[str, list[int]], freq_mhz: int = LOCKED_FREQ_MHZ) -> None:
    for host, gpus in gpu_map.items():
        RMB.lock_freq_map_on_host(host, {gpu: freq_mhz for gpu in gpus})
        log.info("Locked %s GPUs %s to %d MHz", host, gpus, freq_mhz)


def unlock_on_host(host: str, gpus: list[int]) -> None:
    if not gpus:
        return
    RMB.sync_dvfs_py_to_host(host)
    command = f"{RMB.PYTHON} -c {shlex.quote(RMB._dvfs_py_snippet_unlock(gpus))}"
    RMB._ensure_local_node1_flag()
    if host == RMB.NODE1_IP and RMB._LOCAL_NODE1:
        result = subprocess.run(
            ["docker", "exec", RMB.CONTAINER, "bash", "-lc", command],
            capture_output=True,
            text=True,
        )
    else:
        inner = f"docker exec {RMB.CONTAINER} bash -lc {shlex.quote(command)}"
        result = subprocess.run(RMB._ssh(host, inner), capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(f"unlock failed on {host}: {(result.stderr or result.stdout).strip()}")
    log.info("Unlocked %s GPUs %s", host, gpus)


def annotate(
    summary: dict,
    cfg: BT2.Tier1TestConfig,
    gpu_map: dict[str, list[int]],
    *,
    topology_source: str,
    benchmark_name: str,
) -> dict:
    result = dict(summary)
    result.update(
        {
            "config": asdict(cfg),
            "topology_source": topology_source,
            "benchmark": benchmark_name,
            "dvfs": False,
            "locked_freq_mhz": LOCKED_FREQ_MHZ,
            "allocated_gpus": gpu_map,
        }
    )
    return result


def run_tier1_point(
    dataset: str,
    qps: int,
    cfg: BT2.Tier1TestConfig,
    payload: dict,
    output: Path,
    *,
    topology_source: str,
    benchmark_name: str,
    validate_allocation=None,
) -> None:
    key = f"{dataset}_qps{qps}"
    allocation = BT2.plan_allocation(cfg)
    gpu_map = used_gpu_map(allocation)
    if validate_allocation is not None:
        validate_allocation(allocation, gpu_map)
    locked = False
    log.info(
        "Testing %s with topology %s",
        key,
        {field: getattr(cfg, field) for field in TOPOLOGY_FIELDS},
    )
    try:
        ensure_cluster_clean()
        BT2.DATASET = dataset
        BT2.QPS = qps
        urls = BT2.deploy(cfg)
        if urls is None:
            payload["results"][key] = annotate(
                {"status": "DEPLOY_FAILED"},
                cfg,
                gpu_map,
                topology_source=topology_source,
                benchmark_name=benchmark_name,
            )
            return
        locked = True
        lock_used_gpus(gpu_map)
        if not RMB.test_generate(urls[0]):
            payload["results"][key] = annotate(
                {"status": "WARMUP_FAILED"},
                cfg,
                gpu_map,
                topology_source=topology_source,
                benchmark_name=benchmark_name,
            )
            return
        time.sleep(3)
        payload["results"][key] = annotate(
            BT2.run_benchmark(urls, cfg),
            cfg,
            gpu_map,
            topology_source=topology_source,
            benchmark_name=benchmark_name,
        )
    except BaseException as exc:
        payload["results"][key] = annotate(
            {
                "status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
            cfg,
            gpu_map,
            topology_source=topology_source,
            benchmark_name=benchmark_name,
        )
        raise
    finally:
        atomic_write(output, payload)
        if locked:
            for host, gpus in gpu_map.items():
                try:
                    unlock_on_host(host, gpus)
                except Exception:
                    log.exception("Failed to unlock %s GPUs %s", host, gpus)
        try:
            ensure_cluster_clean(retries=3, sleep_s=5)
        except Exception:
            log.exception("Cleanup failed")
        time.sleep(5)


def parse_csv(value: str, allowed, cast=str):
    values = [cast(item.strip()) for item in value.split(",") if item.strip()]
    invalid = [item for item in values if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"unsupported values {invalid}; allowed: {list(allowed)}"
        )
    return values


def run_benchmark_loop(
    *,
    output: Path,
    benchmark_name: str,
    topology_source: str,
    load_config,
    initial_payload,
    datasets,
    qps_values,
    resume: bool,
    validate_allocation=None,
) -> None:
    if resume and output.is_file():
        payload = json.loads(output.read_text())
        meta = payload.get("meta", {})
        if meta.get("benchmark") != benchmark_name:
            raise ValueError(f"refusing to resume incompatible output: {output}")
        log.info("Resuming from %s", output)
    else:
        payload = initial_payload()
    atomic_write(output, payload)
    try:
        for dataset in datasets:
            for qps in qps_values:
                key = f"{dataset}_qps{qps}"
                if resume and payload["results"].get(key, {}).get("status") == "PASS":
                    log.info("Resume: skipping completed point %s", key)
                    continue
                cfg = load_config(dataset, qps)
                run_tier1_point(
                    dataset,
                    qps,
                    cfg,
                    payload,
                    output,
                    topology_source=topology_source,
                    benchmark_name=benchmark_name,
                    validate_allocation=validate_allocation,
                )
    finally:
        atomic_write(output, payload)
