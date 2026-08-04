#!/usr/bin/env python3
"""4-node (32 GPU) node-scalability benchmark.

Schemes (MegaScale skipped):
  - sglang / dynamollm : 16x TP2 across 4 nodes (4 instances/node)
  - distserve / biscale: 2x [2P(TP4)+4D(TP2)] on (n1,n2) and (n3,n4)
  - aflex              : 4x per-node 3P+1D (ipc_cpp), top round-robin router

Energy is summed only over GPUs used by each scheme's deployment.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import logging
import os
import shlex
import statistics
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import requests

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AFLEX_ROOT = ROOT.parents[3]
MACRO_DIR = AFLEX_ROOT / "multi_node/more_test/macro/scripts"
sys.path.insert(0, str(MACRO_DIR))
sys.path.insert(0, str(MACRO_DIR.parent))

import bench_common as BC
import run_macro_benchmark as RMB

# run_macro_benchmark resolves DVFS_PY_SRC relative to AFlex_bench; fix to sglang repo root.
SGLANG_ROOT = Path(os.environ.get("SGLANG_ROOT", "/workspace/sglang"))
RMB.DVFS_PY_SRC = SGLANG_ROOT / "python" / "sglang" / "srt" / "layers" / "dvfs.py"
if not RMB.DVFS_PY_SRC.exists():
    raise FileNotFoundError(f"dvfs.py not found: {RMB.DVFS_PY_SRC}")

COMMON_PATH = HERE / "bench_common_cross.py"
_spec = importlib.util.spec_from_file_location("bench_common_cross", str(COMMON_PATH))
cross = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(cross)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("node_scalability_4n")

NODE1 = os.environ.get("BENCH_NODE1", "10.252.129.36")
NODE2 = os.environ.get("BENCH_NODE2", "10.252.129.35")
NODE3 = os.environ.get("BENCH_NODE3", "10.252.129.34")
NODE4 = os.environ.get("BENCH_NODE4", "10.252.129.33")
NODES = [NODE1, NODE2, NODE3, NODE4]

RMB.NODE1_IP = NODE1
RMB.NODE2_IP = NODE2

MODEL = RMB.MODEL
PYTHON = RMB.PYTHON
LOG_C = RMB.LOG_C
FLAGS = RMB.COMMON_BENCH_SERVER_FLAGS
GPU_NIC = RMB.GPU_NIC
CLEANUP = RMB.CLEANUP
WORKLOAD_DIR = MACRO_DIR.parent / "data" / "workloads"
DATA_DIR = ROOT / "data"
RESULTS_DIR = HERE / "results"
REMOTE_LOG_BASE = (
    "/workspace/sglang/benchmark/AFlex_bench/multi_node/more_test/"
    "Ablation/node_scalibility/logs/32g_four_node"
)
HOST_LOG_BASE = (
    "/mnt/workspace/lt/sglang/benchmark/AFlex_bench/multi_node/more_test/"
    "Ablation/node_scalibility/logs/32g_four_node"
)
LOG_FATAL_PATTERNS = (
    "address already in use",
    "error while attempting to bind",
    "OSError: [Errno 98]",
)

ROUTER_PORT = 48000
PROM_PORT_BASE = 29100
ALL_GPUS = list(range(8))
TP2_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]
TP4_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]
P_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]
D_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]

TTFT_SLO_MS = 2000.0
TPOT_SLO_MS = 100.0
RMB.TTFT_SLO_MS = TTFT_SLO_MS
RMB.TPOT_SLO_MS = TPOT_SLO_MS
BC.TTFT_SLO_MS = TTFT_SLO_MS
BC.TPOT_SLO_MS = TPOT_SLO_MS

ENERGY_MODEL_V1 = RMB.ENERGY_MODEL_DIR_V1
NATIVE_DVFS = (
    f" --dvfs-enabled --dvfs-energy-model-dir {ENERGY_MODEL_V1} "
    f"--dvfs-ttft-slo-ms {int(TTFT_SLO_MS)} --dvfs-tpot-slo-us {int(TPOT_SLO_MS * 1000)}"
)
BISCALE_DVFS = NATIVE_DVFS + " --dvfs-policy biscale"
AFLEX_DVFS = (
    f" --afd-dvfs-enabled --afd-energy-model-dir {ENERGY_MODEL_V1} "
    f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)} --afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)} "
    "--afd-dvfs-decode-compositional --afd-dvfs-idle-lock"
)

SCHEME_GPU_MAP: dict[str, dict[str, list[int]]] = {
    "sglang": {h: ALL_GPUS[:] for h in NODES},
    "dynamollm": {h: ALL_GPUS[:] for h in NODES},
    "distserve": {NODE1: ALL_GPUS[:], NODE2: ALL_GPUS[:], NODE3: ALL_GPUS[:], NODE4: ALL_GPUS[:]},
    "biscale": {NODE1: ALL_GPUS[:], NODE2: ALL_GPUS[:], NODE3: ALL_GPUS[:], NODE4: ALL_GPUS[:]},
    "aflex": {h: ALL_GPUS[:] for h in NODES},
}


def dexec(host: str, cmd: str) -> None:
    RMB.dexec_on_host(host, cmd)


def launch(host: str, cmd: str, name: str) -> None:
    full = f"mkdir -p {REMOTE_LOG_BASE}; {cmd} > {REMOTE_LOG_BASE}/{name}.log 2>&1 < /dev/null &"
    dexec(host, full)


def wait_health(host: str, port: int, timeout: int = 600) -> bool:
    return RMB.wait_health(host, port, timeout=timeout)


def kill_ports(host: str, ports: list[int]) -> None:
    if not ports:
        return
    port_csv = " ".join(str(p) for p in ports)
    dexec(
        host,
        (
            f"for p in {port_csv}; do "
            f"for pid in $(ss -tlnp sport = :$p 2>/dev/null | grep -oP 'pid=\\K[0-9]+'); do "
            f"kill -9 $pid 2>/dev/null || true; done; "
            f"fuser -k ${{p}}/tcp 2>/dev/null || true; done"
        ),
    )


def _check_remote_log_fatal(host: str, log_name: str) -> str | None:
    log_path = f"{HOST_LOG_BASE}/{log_name}.log"
    ret = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", host,
            f"test -f {log_path} && tail -40 {log_path} || true",
        ],
        capture_output=True, text=True, timeout=20,
    )
    tail = ret.stdout or ""
    for pat in LOG_FATAL_PATTERNS:
        if pat in tail:
            return pat
    return None


def wait_health_with_log(host: str, port: int, log_name: str, timeout: int = 600) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{host}:{port}/health", timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        fatal = _check_remote_log_fatal(host, log_name)
        if fatal:
            log.error("FATAL in %s on %s:%d: %s", log_name, host, port, fatal)
            return False
        time.sleep(3)
    log.error("Health timeout %s:%d (log=%s)", host, port, log_name)
    return False


def _prom_port_arg(offset: int) -> str:
    """Unique Prometheus port per router (default 29000 collides on same host)."""
    return f" --prometheus-port {PROM_PORT_BASE + offset}"


def _host_cleanup_script(container: str = RMB.CONTAINER) -> str:
    """Cleanup body executed via `ssh host bash -s` (stdin script)."""
    return f"""
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
pgrep -f 'python.*-m sglang.launch_server' | xargs -r kill -9 2>/dev/null || true
pgrep -f 'sglang::scheduler' | xargs -r kill -9 2>/dev/null || true
pgrep -f 'sglang::detokenizer' | xargs -r kill -9 2>/dev/null || true
pgrep -f 'launch_router' | xargs -r kill -9 2>/dev/null || true
pgrep -f 'sglang::router' | xargs -r kill -9 2>/dev/null || true
pgrep -f 'sglang_router' | xargs -r kill -9 2>/dev/null || true
sleep 5
for pid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
  kill -9 "$pid" 2>/dev/null || true
done
echo -n remaining_compute_apps=
nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l
echo max_gpu_mem_mib=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
# cleanup_node.sh uses host network namespace; run after status echo (may drop SSH).
docker exec {container} bash -lc 'bash {CLEANUP}' 2>/dev/null || true
""".strip()


def _gpu_compute_app_count(host: str) -> int:
    ret = subprocess.run(
        [
            "ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", host,
            "nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader",
        ],
        capture_output=True, text=True, timeout=30,
    )
    if ret.returncode != 0:
        return -1
    lines = [ln.strip() for ln in (ret.stdout or "").splitlines() if ln.strip()]
    return len(lines)


def _ssh_host_cleanup(host: str, label: str = "") -> int:
    ret = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=no", host, "bash", "-s"],
        input=_host_cleanup_script(),
        capture_output=True, text=True, timeout=180,
    )
    out = (ret.stdout or "").strip()
    if ret.returncode != 0:
        log.warning("%scleanup %s failed rc=%s stderr=%s", label, host, ret.returncode, ret.stderr.strip())
    log.info("%scleanup %s: %s", f"{label} " if label else "", host, out or ret.stderr.strip())
    remaining = -1
    for line in out.splitlines():
        if line.startswith("remaining_compute_apps="):
            remaining = int(line.split("=", 1)[1].strip())
    if remaining < 0:
        remaining = _gpu_compute_app_count(host)
        if remaining >= 0:
            log.info("%sverify %s: remaining_compute_apps=%d (ssh output lost)", f"{label} " if label else "", host, remaining)
    return remaining


def kill_node1_gpu_processes() -> None:
    log.info("Killing ALL GPU processes on node1 (host PID namespace + all containers)")
    remaining = _ssh_host_cleanup(NODE1, label="node1")
    if remaining != 0:
        log.warning("node1 still has %d GPU compute process(es), retrying", remaining)
        remaining = _ssh_host_cleanup(NODE1, label="node1-retry")
    if remaining != 0:
        raise RuntimeError(f"node1 cleanup failed: {remaining} GPU compute process(es) remain")


def cleanup_all_nodes() -> None:
    log.info("Cleaning all 4 nodes (host-level)")
    for host in NODES:
        _ssh_host_cleanup(host)
    time.sleep(3)


def lock_freq_nodes(hosts: list[str], gpus: list[int], freq: int | None) -> None:
    """Lock/unlock GPU clocks via dvfs.py inside operator_test (same as run_macro_benchmark)."""
    for host in hosts:
        RMB.sync_dvfs_py_to_host(host)
    py = RMB._dvfs_py_snippet_unlock(gpus) if freq is None else RMB._dvfs_py_snippet_lock(gpus, freq)
    RMB._ensure_local_node1_flag()
    for host in hosts:
        if host == NODE1 and RMB._LOCAL_NODE1:
            RMB._dvfs_exec_local(py)
        else:
            RMB._dvfs_exec_remote(host, py)
    log.info(
        "GPU clocks %s via dvfs.py on %s (gpus=%s)",
        f"locked {freq}MHz" if freq else "reset",
        hosts,
        gpus,
    )


def read_energy_map(node_gpus: dict[str, list[int]]) -> dict[str, dict[int, int]]:
    out: dict[str, dict[int, int]] = {}
    for host, gpus in node_gpus.items():
        if gpus:
            out[host] = RMB.get_energy_on_host(host, gpus)
    return out


def energy_delta_j(before: dict[str, dict[int, int]], after: dict[str, dict[int, int]],
                   node_gpus: dict[str, list[int]]) -> float:
    total = 0.0
    for host, gpus in node_gpus.items():
        for g in gpus:
            total += (after.get(host, {}).get(g, 0) - before.get(host, {}).get(g, 0)) / 1000.0
    return total


def test_generate(url: str, timeout: int = 120) -> bool:
    for attempt in range(3):
        try:
            r = requests.post(
                url + "/generate",
                json={"text": "Hello:", "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}},
                timeout=timeout,
            )
            if r.status_code == 200 and "text" in r.json():
                return True
        except Exception as exc:
            log.warning("warmup attempt %d failed: %s", attempt + 1, exc)
        time.sleep(10)
    return False


def deploy_native_tp2(scheme: str) -> str | None:
    return _deploy_native_tp(scheme, tp=2)


def deploy_native_tp4(scheme: str) -> str | None:
    return _deploy_native_tp(scheme, tp=4)


def _deploy_native_tp(scheme: str, *, tp: int) -> str | None:
    is_dynamo = scheme == "dynamollm"
    groups = TP2_GROUPS if tp == 2 else TP4_GROUPS
    n_inst = len(NODES) * len(groups)
    port_base = 53200 if tp == 2 else 53400
    nccl_base = 33300 if tp == 2 else 33500
    prom_off = 20 if tp == 2 else 30
    tag = f"tp{tp}"
    log.info("Deploy %s: %dxTP%d (%d/node) round-robin", scheme, n_inst, tp, len(groups))
    cleanup_all_nodes()
    if is_dynamo:
        # DynamoLLM: instances own per-GPU DVFS; baseline lock comes after deploy.
        pass
    else:
        lock_freq_nodes(NODES, ALL_GPUS, RMB.MAX_GPU_FREQ)

    insts: list[tuple[str, list[int], int, int]] = []
    idx = 0
    for host in NODES:
        for g in groups:
            insts.append((host, g, port_base + idx * 10, idx))
            idx += 1

    def _launch_one(host: str, gpus: list[int], port: int, i: int) -> None:
        nccl_port = nccl_base + i * 10
        kill_ports(host, [port, nccl_port, port + 1])
        csv = ",".join(map(str, gpus))
        dvfs = NATIVE_DVFS if is_dynamo else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={host}; "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp {tp} --host {host} --port {port} "
            f"--nccl-port {nccl_port} {FLAGS}{dvfs}"
        )
        launch(host, cmd, f"{scheme}_{tag}_{i}")

    for host, gpus, port, i in insts:
        _launch_one(host, gpus, port, i)
        time.sleep(3)

    time.sleep(8)
    for host, gpus, port, i in insts:
        log_name = f"{scheme}_{tag}_{i}"
        if wait_health_with_log(host, port, log_name, 600):
            continue
        log.warning("TP%d instance %d on %s:%d failed, retrying once", tp, i, host, port)
        kill_ports(host, [port, nccl_base + i * 10, port + 1])
        _launch_one(host, gpus, port, i)
        time.sleep(12)
        if not wait_health_with_log(host, port, log_name, 600):
            log.error("TP%d instance %d on %s:%d failed after retry", tp, i, host, port)
            return None

    if is_dynamo:
        lock_freq_nodes(NODES, ALL_GPUS, RMB.MAX_GPU_FREQ)
        log.info("DynamoLLM DVFS baseline locked to %dMHz on all nodes", RMB.MAX_GPU_FREQ)

    workers = " ".join(f"http://{h}:{p}" for h, _, p, _ in insts)
    kill_ports(NODE1, [ROUTER_PORT, PROM_PORT_BASE + prom_off])
    rc = (
        f"setsid {PYTHON} -m sglang_router.launch_router --host {NODE1} "
        f"--port {ROUTER_PORT} --policy round_robin --worker-urls {workers}"
        f"{_prom_port_arg(prom_off)}"
    )
    launch(NODE1, rc, f"{scheme}_router")
    if not wait_health(NODE1, ROUTER_PORT, 120):
        return None
    return f"http://{NODE1}:{ROUTER_PORT}"


def _deploy_pd_instance(p_host: str, d_host: str, scheme: str, inst_id: int,
                        sub_router_port: int) -> str | None:
    is_biscale = scheme == "biscale"
    tag = f"{scheme}_i{inst_id}"
    p_insts, d_insts = [], []

    for i, gpus in enumerate(P_GROUPS):
        p_port = 53100 + inst_id * 100 + i * 10
        bs_port = 49100 + inst_id * 10 + i
        nccl_port = 34000 + inst_id * 100 + i * 10
        kill_ports(p_host, [p_port, nccl_port, bs_port, p_port + 1])
        nic = GPU_NIC[gpus[0]]
        csv = ",".join(map(str, gpus))
        dvfs = BISCALE_DVFS if is_biscale else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={p_host}; "
        )
        extra = (
            f"--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {bs_port} --disaggregation-ib-device {nic} "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp 4 --host {p_host} --port {p_port} "
            f"--nccl-port {34000 + inst_id * 100 + i * 10} {FLAGS}{extra}{dvfs}"
        )
        launch(p_host, cmd, f"{tag}_p{i}")
        p_insts.append({"port": p_port, "bs_port": bs_port})
        time.sleep(3)

    for i, gpus in enumerate(D_GROUPS):
        d_port = 53150 + inst_id * 100 + i * 10
        nccl_port = 34050 + inst_id * 100 + i * 10
        kill_ports(d_host, [d_port, nccl_port, d_port + 1])
        nic = GPU_NIC[gpus[0]]
        csv = ",".join(map(str, gpus))
        dvfs = BISCALE_DVFS if is_biscale else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={d_host}; "
        )
        extra = (
            f"--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {p_insts[0]['bs_port']} "
            f"--disaggregation-ib-device {nic} "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp 2 --host {d_host} --port {d_port} "
            f"--nccl-port {34050 + inst_id * 100 + i * 10} {FLAGS}{extra}{dvfs}"
        )
        launch(d_host, cmd, f"{tag}_d{i}")
        d_insts.append({"port": d_port})
        time.sleep(3)

    for i, inst in enumerate(p_insts):
        log_name = f"{tag}_p{i}"
        if not wait_health_with_log(p_host, inst["port"], log_name, 600):
            log.error("PD prefill %s on %s:%d failed", log_name, p_host, inst["port"])
            return None
    for i, inst in enumerate(d_insts):
        log_name = f"{tag}_d{i}"
        if not wait_health_with_log(d_host, inst["port"], log_name, 600):
            log.error("PD decode %s on %s:%d failed", log_name, d_host, inst["port"])
            return None

    kill_ports(p_host, [sub_router_port, PROM_PORT_BASE + inst_id])
    rc_parts = [
        f"setsid {PYTHON} -m sglang_router.launch_router --pd-disaggregation",
        f"--host {p_host} --port {sub_router_port}{_prom_port_arg(inst_id)}",
    ]
    for inst in p_insts:
        rc_parts.append(f"--prefill http://{p_host}:{inst['port']} {inst['bs_port']}")
    for inst in d_insts:
        rc_parts.append(f"--decode http://{d_host}:{inst['port']}")
    launch(p_host, " ".join(rc_parts), f"{tag}_subrouter")
    if not wait_health_with_log(p_host, sub_router_port, f"{tag}_subrouter", 120):
        log.error("PD sub-router %s on %s:%d failed", tag, p_host, sub_router_port)
        return None
    return f"http://{p_host}:{sub_router_port}"


def deploy_pd_dual(scheme: str) -> str | None:
    log.info("Deploy %s: 2x[2P(TP4)+4D(TP2)]", scheme)
    cleanup_all_nodes()
    if scheme == "biscale":
        for host in NODES:
            lock_freq_nodes([host], ALL_GPUS, None)
    else:
        lock_freq_nodes(NODES, ALL_GPUS, RMB.MAX_GPU_FREQ)

    sub_urls = []
    pairs = [(NODE1, NODE2, 0, 48001), (NODE3, NODE4, 1, 48002)]
    for p_host, d_host, inst_id, sub_port in pairs:
        url = _deploy_pd_instance(p_host, d_host, scheme, inst_id, sub_port)
        if not url:
            return None
        sub_urls.append(url)

    kill_ports(NODE1, [ROUTER_PORT, PROM_PORT_BASE + 10])
    workers = " ".join(sub_urls)
    rc = (
        f"setsid {PYTHON} -m sglang_router.launch_router --host {NODE1} "
        f"--port {ROUTER_PORT} --policy round_robin --worker-urls {workers}"
        f"{_prom_port_arg(10)}"
    )
    launch(NODE1, rc, f"{scheme}_top_router")
    if not wait_health(NODE1, ROUTER_PORT, 120):
        return None
    return f"http://{NODE1}:{ROUTER_PORT}"


def deploy_aflex_3p1d_on_node(host: str, node_idx: int) -> str | None:
    server_base = 48100 + node_idx * 200
    router_port = ROUTER_PORT  # same port, different hosts
    dvfs = AFLEX_DVFS
    tag = f"aflex_n{node_idx}"

    dexec(host, (
        f"CUDA_VISIBLE_DEVICES=0 {PYTHON} -c \"import sys;sys.path.insert(0,'/workspace/sglang/python');"
        "from sglang.srt.layers.afd_ipc_cpp import get_module;get_module()\""
    ))
    pairs = [("prefill", i, i * 2, i * 2 + 1) for i in range(3)] + [("decode", 0, 6, 7)]
    endpoints: dict = {}
    slot = 0
    for role, idx, fg, ag in pairs:
        attn_port = server_base + slot * 3
        ffn_port = server_base + slot * 3 + 2
        bootstrap = server_base + slot * 3 + 1
        ucx = 49200 + node_idx * 1000 + slot * 20
        sched = 49300 + node_idx * 1000 + slot * 20
        for perspective, port, gpu, peer in [("ffn", ffn_port, fg, 1), ("attn", attn_port, ag, 0)]:
            env = (
                f"export CUDA_VISIBLE_DEVICES={fg},{ag} SGLANG_HOST_IP={host} "
                f"AFD_IPC_PEER_DEVICE={peer} AFD_IPC_SYNC_MODE=ipc_event AFD_ASYNC_PIPELINE=1 "
                f"AFD_UCX_FFN_HOST=127.0.0.1 AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_NVML_DEVICE_INDEX={gpu} AFD_NVML_DEVICE_INDICES={gpu} "
                f"SGLANG_DISABLE_REQUEST_LOGGING=true; "
            )
            base_gpu = 0 if perspective == "ffn" else 1
            cmd = (
                f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
                f"--model-path {MODEL} --tp 1 --base-gpu-id {base_gpu} --host {host} --port {port} "
                f"--nccl-port {49400 + node_idx * 1000 + slot * 20 + peer} "
                f"--afd-perspective {perspective} --afd-comm-backend ipc_cpp "
                f"--afd-micro-batch 1 --afd-attn-tp 1 --afd-ffn-tp 1 "
                f"--disaggregation-mode {role} --disaggregation-transfer-backend mooncake "
                f"--disaggregation-ib-device mlx5_bond_0 {FLAGS} "
                f"--watchdog-timeout 900 --afd-disagg-interleave-poll --num-reserved-decode-tokens 512{dvfs}"
            )
            launch(host, cmd, f"{tag}_{role}{idx}_{perspective}")
            time.sleep(2)
        endpoints[(role, idx)] = (attn_port, bootstrap)
        slot += 1

    for role, idx, _, _ in pairs:
        ap, _ = endpoints[(role, idx)]
        # FFN /health needs detokenizer heartbeat (503 in PD+AFD); macro uses get_model_info.
        if not RMB.wait_health(host, ap + 2, 600, check_model_info=True):
            log.error("aflex %s ffn health failed on %s:%d", tag, host, ap + 2)
            return None
        if not RMB.wait_health(host, ap, 600, check_model_info=True):
            log.error("aflex %s attn health failed on %s:%d", tag, host, ap)
            return None

    sub_ports = []
    d_ep = endpoints[("decode", 0)]
    for i in range(3):
        p_ap, p_bs = endpoints[("prefill", i)]
        sp = router_port + 1 + i
        prom_off = node_idx * 10 + i
        kill_ports(host, [sp, PROM_PORT_BASE + prom_off])
        rc = (
            f"setsid {PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
            f"--prefill http://{host}:{p_ap} {p_bs} --decode http://{host}:{d_ep[0]} "
            f"--host {host} --port {sp}{_prom_port_arg(prom_off)}"
        )
        launch(host, rc, f"{tag}_sub_p{i}")
        if not wait_health(host, sp, 120):
            return None
        sub_ports.append(sp)

    workers = " ".join(f"http://{host}:{sp}" for sp in sub_ports)
    prom_node = node_idx * 10 + 5
    kill_ports(host, [router_port, PROM_PORT_BASE + prom_node])
    rc = (
        f"setsid {PYTHON} -m sglang_router.launch_router --host {host} "
        f"--port {router_port} --policy round_robin --worker-urls {workers}"
        f"{_prom_port_arg(prom_node)}"
    )
    launch(host, rc, f"{tag}_node_router")
    if not wait_health(host, router_port, 120):
        return None
    return f"http://{host}:{router_port}"


def deploy_aflex_4node() -> str | None:
    log.info("Deploy aflex: 4x per-node 3P+1D")
    cleanup_all_nodes()
    for host in NODES:
        lock_freq_nodes([host], ALL_GPUS, None)

    node_urls = []
    for i, host in enumerate(NODES):
        url = deploy_aflex_3p1d_on_node(host, i)
        if not url:
            return None
        node_urls.append(url)

    kill_ports(NODE1, [ROUTER_PORT, PROM_PORT_BASE + 50])
    workers = " ".join(node_urls)
    rc = (
        f"setsid {PYTHON} -m sglang_router.launch_router --host {NODE1} "
        f"--port {ROUTER_PORT} --policy round_robin --worker-urls {workers}"
        f"{_prom_port_arg(50)}"
    )
    launch(NODE1, rc, "aflex_top_router")
    if not wait_health(NODE1, ROUTER_PORT, 120):
        return None
    return f"http://{NODE1}:{ROUTER_PORT}"


DEPLOY_FNS = {
    "sglang": lambda s: deploy_native_tp2("sglang"),
    "dynamollm": lambda s: deploy_native_tp2("dynamollm"),
    "distserve": deploy_pd_dual,
    "biscale": deploy_pd_dual,
    "aflex": lambda s: deploy_aflex_4node(),
}

NATIVE_TP = 2


def _native_deploy_fn(scheme: str):
    if scheme in ("sglang", "dynamollm"):
        return deploy_native_tp4 if NATIVE_TP == 4 else deploy_native_tp2
    return DEPLOY_FNS[scheme]


async def _run_workload(url: str, reqs: list[dict], run_s: int,
                        node_gpus: dict[str, list[int]]) -> dict:
    before = read_energy_map(node_gpus)
    rows: list[dict] = []

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=run_s + 120)) as session:
        base = time.monotonic()
        tasks = [
            asyncio.create_task(cross.send_one(session, url + "/generate", req, base, rows))
            for req in reqs
        ]
        done, pending = await asyncio.wait(tasks, timeout=run_s)
        for t in pending:
            t.cancel()
        await asyncio.gather(*done, *pending, return_exceptions=True)
        duration = time.monotonic() - base

    after = read_energy_map(node_gpus)
    energy_j = energy_delta_j(before, after, node_gpus)
    ok = [r for r in rows if r.get("success")]
    fail = len(reqs) - len(ok)
    ttft = [r.get("ttft_proc_ms") or r.get("ttft_ms", 0) for r in ok]
    tpot = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    out_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    inp_tokens = sum(reqs[r.get("request_index", i)]["input_len"]
                     for i, r in enumerate(ok) if r.get("request_index") is not None)
    if not inp_tokens:
        inp_tokens = sum(r.get("input_len", 0) for r in ok)
    total_all = inp_tokens + out_tokens
    n_ttft_viol = sum(1 for v in ttft if v > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for v in tpot if v > TPOT_SLO_MS)

    status = "PASS"
    if not ok:
        status = "FAIL"
    elif fail:
        status = "PARTIAL"

    return {
        "status": status,
        "total_requests": len(reqs),
        "successful": len(ok),
        "failed": fail,
        "slo_violations": n_ttft_viol + n_tpot_viol,
        "throughput_tok_s": round(out_tokens / duration, 2) if duration else 0,
        "ttft_proc_avg_ms": round(statistics.fmean(ttft), 2) if ttft else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(ttft, 50)), 2) if ttft else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(ttft, 99)), 2) if ttft else 0,
        "tpot_avg_ms": round(statistics.fmean(tpot), 2) if tpot else 0,
        "tpot_p50_ms": round(float(np.percentile(tpot, 50)), 2) if tpot else 0,
        "tpot_p99_ms": round(float(np.percentile(tpot, 99)), 2) if tpot else 0,
        "total_energy_j": round(energy_j, 2),
        "energy_per_token_mj": round(energy_j * 1000 / total_all, 3) if total_all else None,
        "energy_denominator": "input_plus_output",
        "duration_s": round(duration, 2),
    }


def run_benchmark_point(scheme: str, dataset: str, qps: int) -> dict:
    wl = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    if not wl.exists():
        return {"status": "NO_WORKLOAD"}
    reqs = [json.loads(line) for line in wl.read_text().splitlines() if line.strip()]
    last = max(r["arrival_time_s"] for r in reqs)
    run_s = int(min(max(last + 150, 300), 900))
    node_gpus = SCHEME_GPU_MAP[scheme]

    deploy_fn = _native_deploy_fn(scheme)
    url = deploy_fn(scheme)
    if not url:
        return {"status": "DEPLOY_FAILED"}
    if not test_generate(url, timeout=240 if scheme == "aflex" else 120):
        cleanup_all_nodes()
        return {"status": "WARMUP_FAILED"}
    time.sleep(3)
    result = asyncio.run(_run_workload(url, reqs, run_s, node_gpus))
    cleanup_all_nodes()
    lock_freq_nodes(NODES, ALL_GPUS, None)
    result = BC.recompute_energy_per_total_token(result, BC.wl_key(dataset, qps))
    if scheme in ("sglang", "dynamollm"):
        n_inst = len(NODES) * (len(TP4_GROUPS) if NATIVE_TP == 4 else len(TP2_GROUPS))
        result["config"] = {"tp": NATIVE_TP, "instances": n_inst, "topology": f"{n_inst}xTP{NATIVE_TP}"}
    return result


def load_results(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "meta": {
            "benchmark": "32gpu_4node_six_schemes",
            "datasets": ["code", "conv"],
            "qps": [32],
            "gpu_count": 32,
            "nodes": 4,
        },
        "results": {},
    }


def save_results(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    log.info("Saved %s", path)


def main() -> None:
    global NATIVE_TP
    parser = argparse.ArgumentParser()
    parser.add_argument("--schemes", default="sglang,dynamollm,distserve,biscale,aflex")
    parser.add_argument("--datasets", default="code,conv")
    parser.add_argument("--qps", default="32")
    parser.add_argument("--native-tp", type=int, default=2, choices=[2, 4],
                        help="Native TP size for sglang/dynamollm (2=16xTP2, 4=8xTP4)")
    parser.add_argument("--output", default=str(DATA_DIR / "32gpu_six_schemes.json"))
    parser.add_argument("--skip-kill-node1", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-run even if previous PASS exists")
    args = parser.parse_args()
    NATIVE_TP = args.native_tp

    schemes = [s.strip() for s in args.schemes.split(",") if s.strip()]
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    qps_list = [int(q) for q in args.qps.split(",") if q.strip()]
    out_path = Path(args.output)

    if not args.skip_kill_node1:
        kill_node1_gpu_processes()
    cleanup_all_nodes()

    payload = load_results(out_path)
    payload["meta"]["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    payload["meta"]["qps"] = sorted(set(payload["meta"].get("qps", []) + qps_list))
    if NATIVE_TP == 4:
        payload["meta"]["native_tp"] = 4
        payload["meta"]["native_topology"] = "8xTP4"

    for scheme in schemes:
        payload["results"].setdefault(scheme, {})
        for dataset in datasets:
            for qps in qps_list:
                key = BC.wl_key(dataset, qps)
                prev = payload["results"][scheme].get(key)
                if not args.force and isinstance(prev, dict) and prev.get("status") == "PASS":
                    log.info("Skip PASS %s %s", scheme, key)
                    continue
                log.info("\n%s\n===== %s | %s | QPS=%d =====\n%s", "#" * 72, scheme, dataset, qps, "#" * 72)
                try:
                    result = run_benchmark_point(scheme, dataset, qps)
                except Exception as exc:
                    log.exception("%s %s failed", scheme, key)
                    result = {"status": "FAILED", "error": str(exc)}
                payload["results"][scheme][key] = result
                save_results(payload, out_path)
                time.sleep(10)

    log.info("All done -> %s", out_path)


if __name__ == "__main__":
    main()
