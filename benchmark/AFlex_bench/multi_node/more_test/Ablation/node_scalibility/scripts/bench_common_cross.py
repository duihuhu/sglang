#!/usr/bin/env python3
"""AFlex 4-node cross-node A/F validation and benchmark.

Each PA/PF/DA/DF role uses one node. The default is TP8 on physical GPUs
0-7 (32 GPUs total); TP4 supports a selected four-GPU subset (16 total).

Unlike the existing same-node A/F launchers, this supports ZMQ (the default
reliable baseline) and UCX backends with cross-node scheduler sockets.  ``--smoke-only`` stops after health + one
``/generate`` request; the default run restarts all four services per QPS.
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import fcntl
import json
import logging
import os
import shlex
import subprocess
import sys
import textwrap
import time
import uuid
from pathlib import Path

import aiohttp
import numpy as np
import requests
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
MACRO = REPO / "benchmark/AFlex_bench/multi_node/more_test/macro/scripts"
sys.path.insert(0, str(MACRO))
import run_macro_benchmark as RMB

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("aflex32_cross_af")

NODES = {
    "PA": os.environ.get("AFLEX32_NODE_PA", "10.252.129.36"),
    "PF": os.environ.get("AFLEX32_NODE_PF", "10.252.129.35"),
    "DA": os.environ.get("AFLEX32_NODE_DA", "10.252.129.34"),
    "DF": os.environ.get("AFLEX32_NODE_DF", "10.252.129.33"),
}
CONTAINER = os.environ.get("MN_CONTAINER", "operator_test")
PYTHON = "/usr/bin/python3"
MODEL = "/models/Qwen3-32B/"
TP = 8
GPU_IDS = tuple(range(8))
RESULTS_DIR = HERE / "results"
LOGS_DIR = HERE / "logs"
REMOTE_LOG_DIR = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scalibility/logs"
WORKLOAD_DIR = MACRO / "workloads"
TELEMETRY_DIR = RESULTS_DIR / "telemetry"
TELEMETRY_ENABLED = False
TELEMETRY_INTERVAL_S = 1.0
ZMQ_SHARDING = False
ZMQ_DOUBLE_BUFFER = False
ORCHESTRATOR_LOCK_PATH = Path("/tmp/aflex_cross_af_orchestrator.lock")

ROUTER_PORT = 46000
PA_PORT, PF_PORT = 46010, 46020
DA_PORT, DF_PORT = 46030, 46040
BS_PORT = 46999
P_UCX_PORT, D_UCX_PORT = 46200, 46300
P_ZMQ_FFN_PORT, P_ZMQ_ATTN_PORT = 47000, 47100
D_ZMQ_FFN_PORT, D_ZMQ_ATTN_PORT = 47200, 47300
AFD_BACKEND = os.environ.get("AFLEX32_AFD_BACKEND", "zmq").lower()
ZMQ_TIMEOUT_MS = os.environ.get("AFLEX32_ZMQ_TIMEOUT_MS", "300000")
if AFD_BACKEND not in {"zmq", "ucx"}:
    raise ValueError(f"AFLEX32_AFD_BACKEND must be zmq or ucx, got {AFD_BACKEND!r}")

# ZMQ is bidirectional: each side connects to its A/F counterpart. This is
# intentionally independent of the UCX FFN host used by Attn -> FFN setup.
ZMQ_PEER_ROLE = {"PA": "PF", "PF": "PA", "DA": "DF", "DF": "DA"}
ROLE_CONFIG = {
    "PA": ("attn", "prefill"),
    "PF": ("ffn", "prefill"),
    "DA": ("attn", "decode"),
    "DF": ("ffn", "decode"),
}
P_SCHED_PORT, D_SCHED_PORT = 46400, 46500
P_NCCL_PORT, D_NCCL_PORT = 46600, 46700
P_HANDSHAKE_PORTS = tuple(range(60001, 60009))
D_HANDSHAKE_PORTS = tuple(range(61001, 61009))
SELF_PORTS = (
    ROUTER_PORT, PA_PORT, PF_PORT, DA_PORT, DF_PORT,
    *range(46200, 46204), *range(46300, 46304),
    *range(P_ZMQ_FFN_PORT, P_ZMQ_FFN_PORT + 9),
    *range(P_ZMQ_ATTN_PORT, P_ZMQ_ATTN_PORT + 9),
    *range(D_ZMQ_FFN_PORT, D_ZMQ_FFN_PORT + 9),
    *range(D_ZMQ_ATTN_PORT, D_ZMQ_ATTN_PORT + 9),
    P_SCHED_PORT, D_SCHED_PORT,
    P_NCCL_PORT, P_NCCL_PORT + 10, D_NCCL_PORT, D_NCCL_PORT + 10,
    BS_PORT,
    *P_HANDSHAKE_PORTS, *D_HANDSHAKE_PORTS,
)
MOONCAKE_IB_DEV = os.environ.get("AFLEX32_MOONCAKE_IB_DEV", "mlx5_bond_0")
UCX_RNDV_THRESH = os.environ.get("AFLEX32_UCX_RNDV_THRESH", "64")
UCX_ZCOPY_THRESH = os.environ.get("AFLEX32_UCX_ZCOPY_THRESH", "64")
TTFT_SLO_MS = 2000.0
TPOT_SLO_MS = 100.0


class OrchestratorLock:
    """Non-blocking process lock protecting one four-node orchestration run."""

    def __init__(self, path: Path = ORCHESTRATOR_LOCK_PATH):
        self.path = Path(path)
        self.file = None
        self.owner = {
            "pid": os.getpid(),
            "start": time.strftime("%Y-%m-%d %H:%M:%S %z"),
            "cmd": shlex.join(sys.argv),
        }

    def acquire(self) -> dict:
        self.file = self.path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self.file.seek(0)
            owner = self.file.read().strip() or "<owner metadata unavailable>"
            self.file.close()
            self.file = None
            raise RuntimeError(
                f"Another four-node AFlex orchestrator holds {self.path}: {owner}. "
                "Refusing to clean up or modify the running deployment."
            ) from None
        self.file.seek(0)
        self.file.truncate()
        json.dump(self.owner, self.file, sort_keys=True)
        self.file.write("\n")
        self.file.flush()
        os.fsync(self.file.fileno())
        return dict(self.owner)

    def release(self) -> None:
        if self.file is None:
            return
        try:
            fcntl.flock(self.file.fileno(), fcntl.LOCK_UN)
        finally:
            self.file.close()
            self.file = None


def ssh(host: str, cmd: str, timeout: int = 30, check: bool = False):
    return subprocess.run(
        ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", host, cmd],
        text=True, capture_output=True, timeout=timeout, check=check,
    )


def dexec(host: str, command: str, timeout: int = 30, check: bool = False):
    inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(command)}"
    return ssh(host, inner, timeout=timeout, check=check)


def ensure_dirs():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    for host in set(NODES.values()):
        dexec(host, f"mkdir -p {REMOTE_LOG_DIR}")


def gpu_csv() -> str:
    return ",".join(map(str, GPU_IDS))


def _remote_port_script(action: str, ports=SELF_PORTS) -> str:
    """Build cleanup/check code scoped strictly to SELF_PORTS owners."""
    common = r"""
import os, re, shutil, signal, subprocess, sys, time
ports = [int(port) for port in sys.argv[1:]]
def run(args): return subprocess.run(args, text=True, capture_output=True, check=False)
def existing(pids): return {pid for pid in pids if pid > 1 and os.path.exists(f"/proc/{pid}")}
def owners(port):
    pids = set(map(int, re.findall(r"pid=(\d+)", run(["ss", "-H", "-ltnp", f"sport = :{port}"]).stdout)))
    if shutil.which("fuser"):
        pids.update(map(int, re.findall(r"\b\d+\b", run(["fuser", "-n", "tcp", str(port)]).stdout)))
    if shutil.which("lsof"):
        pids.update(map(int, re.findall(r"^\d+$", run(["lsof", "-nP", "-t", f"-iTCP:{port}", "-sTCP:LISTEN"]).stdout, re.M)))
    return existing(pids)
"""
    cleanup = r"""
def descendants(roots):
    found = existing(roots)
    while True:
        added = set()
        for pid in tuple(found):
            try:
                with open(f"/proc/{pid}/task/{pid}/children") as f: added.update(map(int, f.read().split()))
            except (FileNotFoundError, PermissionError, ProcessLookupError): pass
        new = existing(added) - found
        if not new: return found
        found.update(new)
def signal_tree(pids, sig):
    pids = existing(pids); groups = set(); grouped = set()
    for pid in pids:
        try:
            if os.getpgid(pid) == pid: groups.add(pid)
        except ProcessLookupError: pass
    for pgid in groups:
        try: os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError): continue
        for pid in pids:
            try:
                if os.getpgid(pid) == pgid: grouped.add(pid)
            except ProcessLookupError: pass
    for pid in pids - grouped:
        try: os.kill(pid, sig)
        except (ProcessLookupError, PermissionError): pass
roots = set()
for port in ports: roots.update(owners(port))
targets = descendants(roots)
if targets:
    signal_tree(targets, signal.SIGTERM); deadline = time.monotonic() + 5
    while existing(targets) and time.monotonic() < deadline: time.sleep(0.2)
    signal_tree(existing(targets), signal.SIGKILL)
"""
    check = r"""
for port in ports:
    result = run(["ss", "-H", "-ltnp", f"sport = :{port}"])
    if not result.stdout.strip(): continue
    pids = owners(port)
    if not pids: print(f"{port}\t?\t<owner unavailable>\t{result.stdout.strip()}")
    for pid in sorted(pids):
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f: cmd = f.read().replace(b"\0", b" ").decode(errors="replace").strip()
        except (FileNotFoundError, PermissionError): cmd = "<exited or inaccessible>"
        print(f"{port}\t{pid}\t{cmd}")
"""
    body = common + (cleanup if action == "cleanup" else check)
    return f"{PYTHON} -c {shlex.quote(textwrap.dedent(body))} " + " ".join(map(str, ports))


def _port_check(host: str, ports) -> list[str]:
    result = dexec(host, _remote_port_script("check", ports), timeout=20)
    if result.returncode != 0:
        raise RuntimeError(
            f"Cannot inspect ports on {host}: {result.stderr or result.stdout}"
        )
    return [line for line in result.stdout.splitlines() if line.strip()]


def _role_zmq_listener_ports(role: str) -> tuple[int, ...]:
    perspective, mode = ROLE_CONFIG[role]
    if AFD_BACKEND != "zmq":
        return ()
    base = (P_ZMQ_FFN_PORT if mode == "prefill" else D_ZMQ_FFN_PORT)
    if perspective == "ffn":
        base = P_ZMQ_ATTN_PORT if mode == "prefill" else D_ZMQ_ATTN_PORT
    ranks = TP if ZMQ_SHARDING else 1
    return tuple(base + 1 + rank for rank in range(ranks))


def assert_role_ports_free(role: str) -> None:
    ports = _role_zmq_listener_ports(role)
    if not ports:
        return
    busy = _port_check(NODES[role], ports)
    if busy:
        raise RuntimeError(
            f"{role} expected ZMQ listener ports are busy before launch:\n"
            + "\n".join(busy)
        )


def wait_role_ports_bound(role: str, timeout: int = 120) -> bool:
    ports = _role_zmq_listener_ports(role)
    if not ports:
        return True
    deadline = time.time() + timeout
    while time.time() < deadline:
        occupied = {int(line.split("\t", 1)[0])
                    for line in _port_check(NODES[role], ports)}
        if occupied >= set(ports):
            log.info("  %s owns expected ZMQ listeners %s", role, ports)
            return True
        time.sleep(1)
    log.error("  %s did not bind all expected ZMQ listeners %s", role, ports)
    return False


def cleanup_all():
    log.info("Cleaning all four nodes")
    for host in set(NODES.values()):
        try:
            result = dexec(host, _remote_port_script("cleanup"), timeout=30)
            if result.returncode != 0:
                log.warning("cleanup command failed on %s: %s", host, result.stderr or result.stdout)
        except Exception as exc:
            log.warning("cleanup failed on %s: %s", host, exc)
        try:
            ssh(host, f"nvidia-smi -i {gpu_csv()} -rgc >/dev/null 2>&1 || true", timeout=15)
        except Exception:
            pass
    deadline = time.time() + 60
    while time.time() < deadline:
        busy = []
        for host in set(NODES.values()):
            r = dexec(host, _remote_port_script("check"), timeout=20)
            if r.returncode != 0:
                raise RuntimeError(
                    f"Cannot verify SELF_PORTS on {host}: {r.stderr or r.stdout}"
                )
            busy.extend(f"{host}\t{line}" for line in r.stdout.splitlines() if line.strip())
        if not busy: break
        log.warning("Waiting for stale AFD ports to close:\n%s", "\n".join(busy))
        time.sleep(3)
    else:
        raise RuntimeError("AFD ports still busy after cleanup (host, port, pid, cmd):\n" + "\n".join(busy))
    time.sleep(5)


def set_clocks(freq: int | None) -> None:
    for host in set(NODES.values()):
        if freq is None:
            ssh(host, f"nvidia-smi -i {gpu_csv()} -rgc >/dev/null 2>&1 || true", timeout=20)
        else:
            r = ssh(host, f"nvidia-smi -i {gpu_csv()} -lgc {freq},{freq}", timeout=20)
            if r.returncode != 0:
                raise RuntimeError(f"Failed to lock {host} at {freq}MHz: {r.stderr}")
    log.info("GPU clocks %s on all four nodes", "reset" if freq is None else f"locked at {freq}MHz")


def preflight() -> None:
    log.info("Preflight four nodes")
    if len(set(NODES.values())) != len(NODES):
        duplicates = {host:[role for role, value in NODES.items() if value == host]
                      for host in set(NODES.values()) if list(NODES.values()).count(host) > 1}
        raise ValueError(f"PA/PF/DA/DF must use unique hosts, duplicates: {duplicates}")
    code = (
        "import json,pynvml; pynvml.nvmlInit(); "
        "n=pynvml.nvmlDeviceGetCount(); busy={}; "
        f"selected={list(GPU_IDS)!r}; "
        "[(busy.__setitem__(i,[p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses("
        "pynvml.nvmlDeviceGetHandleByIndex(i))])) for i in selected]; "
        "print(json.dumps({'count':n,'busy':busy})); pynvml.nvmlShutdown()"
    )
    for role, host in NODES.items():
        imports = "import torch" if AFD_BACKEND == "zmq" else "import ucp,torch"
        r = dexec(host, f"{PYTHON} -c \"{imports}\" && {PYTHON} -c {shlex.quote(code)}", timeout=60)
        lines = [line for line in r.stdout.splitlines() if line.startswith("{")]
        if r.returncode != 0 or not lines:
            raise RuntimeError(f"{role}@{host} preflight failed: {r.stderr or r.stdout}")
        status = json.loads(lines[-1])
        if status["count"] < 8:
            raise RuntimeError(f"{role}@{host} has only {status['count']} physical GPUs; need at least 8")
        busy = {gpu: pids for gpu, pids in status["busy"].items() if pids}
        if busy:
            raise RuntimeError(f"{role}@{host} selected GPUs have compute processes: {busy}")
        log.info("  %s@%s: backend=%s, selected physical GPUs %s are idle", role, host, AFD_BACKEND, gpu_csv())


def validate_role_mapping() -> None:
    expected_peers = {"PA": "PF", "PF": "PA", "DA": "DF", "DF": "DA"}
    if ZMQ_PEER_ROLE != expected_peers:
        raise ValueError(f"Invalid ZMQ role mapping: {ZMQ_PEER_ROLE!r}")
    for role, peer_role in ZMQ_PEER_ROLE.items():
        if ZMQ_PEER_ROLE.get(peer_role) != role:
            raise ValueError(f"ZMQ peer mapping is not symmetric: {role} -> {peer_role}")
        if ROLE_CONFIG[role][1] != ROLE_CONFIG[peer_role][1]:
            raise ValueError(f"ZMQ peers cross disaggregation modes: {role} -> {peer_role}")
        if ROLE_CONFIG[role][0] == ROLE_CONFIG[peer_role][0]:
            raise ValueError(f"ZMQ peers have the same perspective: {role} -> {peer_role}")
    log.info(
        "Validated ZMQ role peers: %s",
        ", ".join(f"{role}->{peer}" for role, peer in ZMQ_PEER_ROLE.items()),
    )


def launch(role: str, perspective: str, mode: str, port: int, ucx_ffn_host: str,
           ucx_port: int, sched_port: int, nccl_port: int, tier: bool) -> None:
    host = NODES[role]
    expected_config = ROLE_CONFIG.get(role)
    if expected_config != (perspective, mode):
        raise ValueError(
            f"Invalid launch config for {role}: {(perspective, mode)!r}; "
            f"expected {expected_config!r}"
        )
    env = [
        f"CUDA_VISIBLE_DEVICES={gpu_csv()}",
        "LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/libucx/lib:$LD_LIBRARY_PATH",
        f"SGLANG_HOST_IP={host}",
        "SGLANG_DISABLE_REQUEST_LOGGING=true",
        f"AFD_LOCAL_TP={TP}",
        "AFD_CROSS_NODE_EXPERIMENTAL=1",
        f"AFD_SCHED_HOST={NODES['PF'] if perspective == 'attn' and mode == 'prefill' else NODES['DF'] if perspective == 'attn' else '0.0.0.0'}",
        f"AFD_SCHED_PORT={sched_port}",
        f"AFD_NVML_DEVICE_INDICES={gpu_csv()}",
        # This variable is an NVML (physical) index, unlike CUDA local ordinals.
        f"AFD_NVML_DEVICE_INDEX={GPU_IDS[0]}",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600",
    ]
    if AFD_BACKEND == "ucx":
        env += [
            "UCX_LOG_LEVEL=fatal", "UCX_WARN_UNUSED_ENV_VARS=n",
            f"AFD_UCX_TLS={os.environ.get('AFLEX32_UCX_TLS', 'tcp,cuda_copy')}",
            f"AFD_UCX_HOST_STAGING={os.environ.get('AFLEX32_UCX_HOST_STAGING', '1')}",
            "AFD_UCX_PINNED_STAGING=0",
            f"UCX_RNDV_SCHEME={os.environ.get('AFLEX32_UCX_RNDV_SCHEME', 'get_zcopy')}",
            f"UCX_RNDV_THRESH={UCX_RNDV_THRESH}", f"UCX_ZCOPY_THRESH={UCX_ZCOPY_THRESH}",
            f"UCX_NET_DEVICES={os.environ.get('AFLEX32_UCX_NET_DEVICES', 'bond0')}",
            f"AFD_UCX_NUM_NICS={os.environ.get('AFLEX32_UCX_NUM_NICS', str(min(4, TP)))}",
            "AFD_UCX_TIMEOUT=600", f"AFD_UCX_BASE_PORT={ucx_port}",
            "UCX_KEEPALIVE_INTERVAL=60s",
            "UCX_PEER_FAILURE_TIMEOUT=300s",
        ]
        if perspective == "attn":
            env.append(f"AFD_UCX_FFN_HOST={ucx_ffn_host}")
            log.info("  %s UCX Attn->FFN peer: %s", role, ucx_ffn_host)
    else:
        is_prefill = mode == "prefill"
        ffn_base = P_ZMQ_FFN_PORT if is_prefill else D_ZMQ_FFN_PORT
        attn_base = P_ZMQ_ATTN_PORT if is_prefill else D_ZMQ_ATTN_PORT
        peer_role = ZMQ_PEER_ROLE[role]
        zmq_peer_host = NODES[peer_role]
        env += [
            f"AFD_ZMQ_PEER_HOST={zmq_peer_host}",
            f"AFD_FFN_BASE_PORT={ffn_base}",
            f"AFD_ATTN_BASE_PORT={attn_base}",
            f"AFD_ZMQ_TIMEOUT_MS={ZMQ_TIMEOUT_MS}",
            f"AFD_ZMQ_SHARDING={int(ZMQ_SHARDING)}",
            f"AFD_ZMQ_DOUBLE_BUFFER={int(ZMQ_DOUBLE_BUFFER)}",
        ]
        log.info(
            "  %s ZMQ peer: %s@%s (ffn_base=%d, attn_base=%d, timeout_ms=%s)",
            role, peer_role, zmq_peer_host, ffn_base, attn_base, ZMQ_TIMEOUT_MS,
        )
    flags = [
        f"{PYTHON} -m sglang.launch_server",
        f"--model-path {MODEL}", f"--tp {TP}", f"--nccl-port {nccl_port}",
        f"--host {host}", f"--port {port}", f"--afd-perspective {perspective}",
        f"--afd-comm-backend {AFD_BACKEND}", "--afd-micro-batch 1",
        f"--disaggregation-mode {mode}",
        "--disaggregation-transfer-backend mooncake",
        f"--disaggregation-bootstrap-port {BS_PORT}",
        f"--disaggregation-ib-device {MOONCAKE_IB_DEV}",
        "--mem-fraction-static 0.85", "--max-running-requests 512",
        "--skip-server-warmup", "--watchdog-timeout 900",
        "--disable-cuda-graph", "--disable-piecewise-cuda-graph",
        "--afd-disagg-interleave-poll", "--disable-radix-cache",
        "--num-reserved-decode-tokens 512", "--enable-metrics",
    ]
    if tier:
        flags += [
            "--afd-dvfs-enabled",
            f"--afd-energy-model-dir {RMB.ENERGY_MODEL_DIR_V1}",
            f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)}",
            f"--afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)}",
            "--afd-dvfs-decode-compositional", "--afd-dvfs-idle-lock",
        ]
    assert_role_ports_free(role)
    command = "export " + " ".join(env) + "; setsid prlimit --memlock=unlimited:unlimited " + " ".join(flags)
    command += f" > {REMOTE_LOG_DIR}/{role.lower()}.log 2>&1 < /dev/null &"
    r = dexec(host, command)
    if r.returncode != 0:
        raise RuntimeError(f"launch {role} failed: {r.stderr or r.stdout}")
    log.info("  launched %s on %s:%d", role, host, port)


def wait_health(host: str, port: int, name: str, timeout: int = 900) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{host}:{port}/get_model_info", timeout=8)
            if r.status_code == 200:
                log.info("  %s ready", name)
                return True
        except Exception:
            pass
        time.sleep(3)
    log.error("  %s health timeout", name)
    return False



def wait_afd_ready(role: str, timeout: int = 600) -> bool:
    host = NODES[role]
    log_path = f"{REMOTE_LOG_DIR}/{role.lower()}.log"
    deadline = time.time() + timeout
    while time.time() < deadline:
        code = (
            "from pathlib import Path; "
            f"p=Path({log_path!r}); "
            "print(p.read_text(errors='ignore')[-20000:] if p.exists() else '')"
        )
        r = dexec(host, f"{PYTHON} -c {shlex.quote(code)}", timeout=20)
        text = r.stdout
        ready_markers = (("AFD ZMQ handshake ready",) if AFD_BACKEND == "zmq" else ("AF communicator ready", "UCX communicator ready"))
        if AFD_BACKEND == "zmq" and ZMQ_SHARDING:
            ready_ranks = {int(rank) for rank in re.findall(
                r"AFD ZMQ handshake ready:.*?rank=(\d+)", text
            )}
            if ready_ranks >= set(range(TP)):
                log.info("  %s AFD communicator ready on all TP ranks %s", role, sorted(ready_ranks))
                return True
        elif AFD_BACKEND == "ucx":
            # For UCX, TP0 is the representative that does actual RDMA.
            # FFN side logs "AF communicator ready (ffn)", Attn side logs
            # "UCX communicator ready". Match either pattern for TP0.
            tp0_ready = bool(re.search(
                r"TP0\].*(UCX communicator ready|AF communicator ready)", text
            ))
            if tp0_ready:
                log.info("  %s AFD communicator ready (TP0 confirmed)", role)
                return True
        elif any(marker in text for marker in ready_markers):
            log.info("  %s AFD communicator ready", role)
            return True
        if any(marker in text for marker in (
            "AF communicator init failed", "Scheduler hit an exception",
            "Received sigquit", "Device is busy",
        )):
            log.error("  %s AFD communicator failed", role)
            return False
        time.sleep(2)
    log.error("  %s AFD communicator timeout", role)
    return False


def deploy(tier: bool = True) -> str | None:
    # FFN listeners must start before their remote Attn peers connect.
    launch("PF", "ffn", "prefill", PF_PORT, NODES["PA"], P_UCX_PORT, P_SCHED_PORT, P_NCCL_PORT, tier)
    launch("DF", "ffn", "decode", DF_PORT, NODES["DA"], D_UCX_PORT, D_SCHED_PORT, D_NCCL_PORT, tier)
    time.sleep(12)
    launch("PA", "attn", "prefill", PA_PORT, NODES["PF"], P_UCX_PORT, P_SCHED_PORT, P_NCCL_PORT + 10, tier)
    launch("DA", "attn", "decode", DA_PORT, NODES["DF"], D_UCX_PORT, D_SCHED_PORT, D_NCCL_PORT + 10, tier)

    # Before HTTP health, every persistent data listener must be owned by the
    # service. Occupancy is expected after launch and is not treated as stale.
    for role in ("PF", "DF", "PA", "DA"):
        if not wait_role_ports_bound(role):
            return None
    for role, port in (("PF", PF_PORT), ("DF", DF_PORT), ("PA", PA_PORT), ("DA", DA_PORT)):
        if not wait_health(NODES[role], port, role, timeout=300):
            return None
    # HTTP readiness only proves tokenizer/server startup.  The AFD scheduler
    # initializes UCX lazily and may lag by tens of seconds, especially TP8.
    # Do not expose the deployment to Router traffic until both A/F pairs have
    # completed their listener/connect handshake.
    for role in ("PF", "DF", "PA", "DA"):
        if not wait_afd_ready(role):
            return None

    cmd = (
        f"setsid {PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
        f"--prefill http://{NODES['PA']}:{PA_PORT} {BS_PORT} "
        f"--decode http://{NODES['DA']}:{DA_PORT} "
        f"--host {NODES['PA']} --port {ROUTER_PORT} "
        f"> {REMOTE_LOG_DIR}/router.log 2>&1 < /dev/null &"
    )
    r = dexec(NODES["PA"], cmd)
    if r.returncode != 0 or not wait_health(NODES["PA"], ROUTER_PORT, "router", 120):
        return None
    return f"http://{NODES['PA']}:{ROUTER_PORT}"


def test_generate(url: str) -> bool:
    try:
        r = requests.post(url + "/generate", json={
            "text": "Hello, explain quantum computing:",
            "sampling_params": {"max_new_tokens": 16, "temperature": 0.0},
        }, timeout=480)
        if r.status_code == 200 and "text" in r.json():
            log.info("Cross-node A/F smoke generation PASS")
            return True
        log.error("Smoke generation failed: %s %s", r.status_code, r.text[:300])
    except Exception as exc:
        log.error("Smoke generation exception: %s", exc)
    return False



# This program runs on each host, outside the inference container.  It probes
# query-gpu fields independently because NVIDIA driver versions differ in PCIe
# throughput support, then falls back to a one-shot dmon sample for PCIe.
_TELEMETRY_COLLECTOR = r"""
import csv, json, os, re, signal, subprocess, sys, time
out, pidfile, interval, gpu_csv = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
gpus = [int(x) for x in gpu_csv.split(',')]
running = True
def stop(*_):
    global running
    running = False
signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
with open(pidfile, 'w') as f: f.write(str(os.getpid()))
def run(args): return subprocess.run(args, text=True, capture_output=True, timeout=20, check=False)
def nvlink_snapshot():
    errors = []
    commands = [
        ['nvidia-smi', 'nvlink', '-i', gpu_csv, '--getthroughput', 'd'],
        ['nvidia-smi', 'nvlink', '--getthroughput', 'd', '-i', gpu_csv],
        ['nvidia-smi', 'nvlink', '-i', gpu_csv, '-gt', 'd'],
        ['nvidia-smi', 'nvlink', '-gt', 'd', '-i', gpu_csv],
    ]
    for command in commands:
        r = run(command)
        if r.returncode == 0:
            return {'available': True, 'command': ' '.join(command[2:]), 'raw': r.stdout,
                    'timestamp': time.time()}
        errors.append((r.stderr or r.stdout).strip())
    return {'available': False, 'timestamp': time.time(), 'errors': errors}
def query_field(field):
    r = run(['nvidia-smi', '-i', gpu_csv, '--query-gpu=index,' + field,
             '--format=csv,noheader,nounits'])
    return r.returncode == 0 and len(r.stdout.splitlines()) == len(gpus)
base = {'utilization.gpu':'sm_util_pct', 'utilization.memory':'memory_util_pct',
        'power.draw':'power_w'}
pcie = {'pcie.tx_util':'pcie_tx_kib_s', 'pcie.rx_util':'pcie_rx_kib_s'}
supported = {f: query_field(f) for f in [*base, *pcie]}
def dmon_pcie():
    r = run(['nvidia-smi', 'dmon', '-i', gpu_csv, '-s', 't', '-c', '1'])
    values = {}
    if r.returncode:
        return values, (r.stderr or r.stdout).strip()
    header = None
    for line in r.stdout.splitlines():
        stripped = line.strip().lstrip('#').strip()
        if not stripped: continue
        cols = stripped.split()
        if 'gpu' in cols and ('rxpci' in cols or 'txpci' in cols): header = cols; continue
        if header and cols[0].isdigit() and len(cols) >= len(header):
            row = dict(zip(header, cols)); idx = int(row['gpu'])
            for src, dst in (('txpci','pcie_tx_kib_s'), ('rxpci','pcie_rx_kib_s')):
                try: values.setdefault(idx, {})[dst] = float(row[src]) * 1024.0
                except (KeyError, ValueError): pass
    return values, None if values else 'could not parse nvidia-smi dmon PCIe output'
def emit(record):
    with open(out, 'a') as f: f.write(json.dumps(record, separators=(',', ':')) + '\n')
emit({'kind':'metadata','timestamp':time.time(),'gpu_ids':gpus,'query_support':supported,
      'nvlink_start':nvlink_snapshot()})
try:
    while running:
        started = time.monotonic(); fields = [f for f, ok in supported.items() if ok]
        records = {i:{'kind':'sample','timestamp':time.time(),'gpu':i} for i in gpus}
        errors = []
        if fields:
            r = run(['nvidia-smi','-i',gpu_csv,'--query-gpu=index,'+','.join(fields),
                     '--format=csv,noheader,nounits'])
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    vals = [x.strip() for x in next(csv.reader([line]))]
                    try: idx = int(vals[0])
                    except (ValueError, IndexError): continue
                    for field, value in zip(fields, vals[1:]):
                        try: records[idx][base.get(field, pcie.get(field))] = float(value)
                        except ValueError: pass
            else: errors.append((r.stderr or r.stdout).strip())
        if not all(supported[f] for f in pcie):
            fallback, error = dmon_pcie()
            for idx, values in fallback.items(): records[idx].update(values)
            if error: errors.append(error)
        for record in records.values():
            if errors: record['errors'] = errors
            emit(record)
        time.sleep(max(0, interval - (time.monotonic() - started)))
finally:
    emit({'kind':'metadata','timestamp':time.time(),'nvlink_end':nvlink_snapshot()})
    try: os.unlink(pidfile)
    except OSError: pass
"""


def _percentile(values, q):
    return round(float(np.percentile(values, q)), 3) if values else None


def _parse_nvlink_by_gpu(raw: str) -> dict[int, dict[str, float]]:
    """Best-effort parser preserving physical GPU totals across driver formats."""
    result = {}; current_gpu = None
    for line in raw.splitlines():
        gpu_match = re.search(r'\bGPU\s*[#:]?\s*(\d+)', line, re.I)
        if gpu_match: current_gpu = int(gpu_match.group(1))
        lower = line.lower()
        marker = re.search(r'\b(tx|rx)\b|transmit|receive', lower)
        if current_gpu is None or not marker: continue
        token = marker.group(0); direction = 'tx' if token in ('tx','transmit') else 'rx'
        value_text = lower[marker.end():]
        match = re.search(r'(-?\d+(?:\.\d+)?)\s*(kb|kib|mb|mib|gb|gib)?(?:/s)?', value_text)
        if not match: continue
        scale = {'kb':1e3, 'kib':1024, 'mb':1e6, 'mib':1024**2,
                 'gb':1e9, 'gib':1024**3}.get(match.group(2), 1.0)
        totals = result.setdefault(current_gpu, {'tx':0.0, 'rx':0.0})
        totals[direction] += float(match.group(1)) * scale
    return result


class TelemetrySession:
    def __init__(self, run_id: str, interval_s: float = 1.0):
        self.run_id = run_id; self.interval_s = interval_s
        self.local_dir = TELEMETRY_DIR / run_id
        self.local_dir.mkdir(parents=True, exist_ok=True)
        self.state = {}; self.errors = []

    def start(self):
        encoded = base64.b64encode(_TELEMETRY_COLLECTOR.encode()).decode()
        for role, host in NODES.items():
            token = f'aflex_telemetry_{self.run_id}_{role.lower()}'
            script, raw, pid = f'/tmp/{token}.py', f'/tmp/{token}.jsonl', f'/tmp/{token}.pid'
            command = (f'echo {shlex.quote(encoded)} | base64 -d > {script}; rm -f {raw} {pid}; '
                       f'nohup python3 {script} {raw} {pid} {self.interval_s} {gpu_csv()} '
                       f'>/tmp/{token}.log 2>&1 </dev/null &')
            try:
                r = ssh(host, command, timeout=20)
                if r.returncode: raise RuntimeError(r.stderr or r.stdout)
                self.state[role] = {'host':host, 'raw':raw, 'pid':pid, 'script':script,
                                    'local':self.local_dir / f'{role.lower()}.jsonl'}
            except Exception as exc:
                self.errors.append(f'{role}@{host} start: {exc}')
        time.sleep(min(1.0, self.interval_s))

    def stop_and_collect(self):
        for role, state in self.state.items():
            host = state['host']
            try:
                cmd = (f'if test -s {state["pid"]}; then kill -TERM $(cat {state["pid"]}) 2>/dev/null || true; '
                       f'for i in $(seq 1 50); do test ! -e {state["pid"]} && break; sleep .1; done; fi; '
                       f'base64 -w0 {state["raw"]} 2>/dev/null || true')
                r = ssh(host, cmd, timeout=30)
                if r.returncode: raise RuntimeError(r.stderr or r.stdout)
                if r.stdout.strip(): state['local'].write_bytes(base64.b64decode(r.stdout.strip()))
                else: self.errors.append(f'{role}@{host}: telemetry output missing')
            except Exception as exc:
                self.errors.append(f'{role}@{host} stop/collect: {exc}')
            finally:
                try: ssh(host, f'rm -f {state["raw"]} {state["pid"]} {state["script"]}', timeout=10)
                except Exception: pass
        return self.summary()

    def summary(self):
        result = {'run_id':self.run_id, 'interval_s':self.interval_s, 'roles':{}, 'errors':list(self.errors)}
        for role, state in self.state.items():
            role_errors = []; samples = {}; metadata = []
            try:
                for line in state['local'].read_text().splitlines():
                    record = json.loads(line)
                    if record.get('kind') == 'sample': samples.setdefault(int(record['gpu']), []).append(record)
                    else: metadata.append(record)
                    role_errors.extend(record.get('errors', []))
            except Exception as exc: role_errors.append(str(exc))
            gpu_summary = {}
            for gpu in GPU_IDS:
                rows = samples.get(gpu, [])
                def vals(name): return [float(r[name]) for r in rows if name in r]
                gpu_summary[str(gpu)] = {
                    'samples':len(rows), 'sm_avg':round(float(np.mean(vals('sm_util_pct'))),3) if vals('sm_util_pct') else None,
                    'sm_p50':_percentile(vals('sm_util_pct'), 50), 'sm_p95':_percentile(vals('sm_util_pct'), 95),
                    'pcie_tx_avg_kib_s':round(float(np.mean(vals('pcie_tx_kib_s'))),3) if vals('pcie_tx_kib_s') else None,
                    'pcie_tx_p95_kib_s':_percentile(vals('pcie_tx_kib_s'),95),
                    'pcie_rx_avg_kib_s':round(float(np.mean(vals('pcie_rx_kib_s'))),3) if vals('pcie_rx_kib_s') else None,
                    'pcie_rx_p95_kib_s':_percentile(vals('pcie_rx_kib_s'),95),
                    'power_avg_w':round(float(np.mean(vals('power_w'))),3) if vals('power_w') else None,
                }
            start = next((m.get('nvlink_start') for m in metadata if m.get('nvlink_start')), {})
            end = next((m.get('nvlink_end') for m in reversed(metadata) if m.get('nvlink_end')), {})
            nv = {'available':bool(start.get('available') and end.get('available')),
                  'start_command':start.get('command'), 'end_command':end.get('command'),
                  'errors':start.get('errors', []) + end.get('errors', [])}
            if nv['available']:
                a, b = _parse_nvlink_by_gpu(start.get('raw','')), _parse_nvlink_by_gpu(end.get('raw',''))
                elapsed = max(float(end.get('timestamp',0))-float(start.get('timestamp',0)), 0)
                nv['gpus'] = {}
                for gpu in GPU_IDS:
                    gpu_nv = {}
                    for direction in ('tx','rx'):
                        if direction in a.get(gpu,{}) and direction in b.get(gpu,{}):
                            delta = b[gpu][direction]-a[gpu][direction]
                            gpu_nv[f'{direction}_delta_bytes'] = delta
                            gpu_nv[f'{direction}_rate_bytes_s'] = delta/elapsed if elapsed else None
                    nv['gpus'][str(gpu)] = gpu_nv
                    gpu_summary[str(gpu)]['nvlink'] = gpu_nv
            result['roles'][role] = {'host':state['host'], 'raw_path':str(state['local']),
                                     'gpus':gpu_summary, 'nvlink':nv,
                                     'errors':sorted(set(x for x in role_errors if x))}
        return result


def read_energy(role: str, host: str) -> dict:
    code = f"""
import json, pynvml
selected = {list(GPU_IDS)!r}
out = {{'values_mj': {{}}, 'errors': []}}
try:
    pynvml.nvmlInit()
    for i in selected:
        try:
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            out['values_mj'][i] = int(pynvml.nvmlDeviceGetTotalEnergyConsumption(h))
        except Exception as exc:
            out['errors'].append(f'gpu{{i}}: {{type(exc).__name__}}: {{exc}}')
finally:
    try: pynvml.nvmlShutdown()
    except Exception: pass
print(json.dumps(out))
"""
    try:
        r = dexec(host, f"{PYTHON} -c {shlex.quote(code)}", timeout=30)
        lines = [x for x in r.stdout.splitlines() if x.startswith("{")]
        if r.returncode != 0 or not lines:
            return {'values_mj':{}, 'errors':[f'{role}@{host}: '+(r.stderr or r.stdout or 'no JSON output').strip()]}
        data = json.loads(lines[-1]); data['values_mj'] = {int(k):int(v) for k,v in data.get('values_mj',{}).items()}
        data['errors'] = [f'{role}@{host}: {e}' for e in data.get('errors',[])]
        return data
    except Exception as exc:
        return {'values_mj':{}, 'errors':[f'{role}@{host}: {type(exc).__name__}: {exc}']}


async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"text": "x" * req["input_len"], "sampling_params": {
        "max_new_tokens": req["output_len"], "temperature": 0.0, "ignore_eos": True}, "stream": True}
    t0 = time.monotonic(); first = None; count = 0; meta = {}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                results.append({"success": False}); return
            async for line in resp.content:
                now = time.monotonic(); text = line.decode().strip()
                if not text or text.startswith(":"): continue
                if text.startswith("data:"): text = text[5:].strip()
                if text == "[DONE]": break
                try:
                    chunk = json.loads(text)
                    if first is None: first = now
                    count += 1
                    if isinstance(chunk, dict) and "meta_info" in chunk: meta = chunk["meta_info"]
                except json.JSONDecodeError: pass
    except Exception:
        results.append({"success": False}); return
    end = time.monotonic()
    proc = (meta.get("ttft_pure_processing") or meta.get("time_to_first_token_processing") or 0) * 1000
    ttft_ms = (first-t0)*1000 if first else 0
    tpot_ms = (end-first)*1000/(count-1) if first and count > 1 else 0
    effective_ttft = proc or ttft_ms
    results.append({"success": True, "completion_tokens": count,
                    "ttft_ms": ttft_ms, "ttft_proc_ms": proc, "tpot_ms": tpot_ms,
                    "ttft_violation": effective_ttft > TTFT_SLO_MS,
                    "tpot_violation": tpot_ms > TPOT_SLO_MS if tpot_ms > 0 else False})


async def run_workload(reqs, url, max_run_s=900, telemetry_run_id=None):
    start_energy = {role: read_energy(role, host) for role, host in NODES.items()}
    telemetry = TelemetrySession(telemetry_run_id, TELEMETRY_INTERVAL_S) if telemetry_run_id else None
    if telemetry:
        telemetry.start()
    results = []; base = time.monotonic()
    try:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=max_run_s+60)) as session:
            base = time.monotonic()
            tasks = [asyncio.create_task(send_one(session, url, r, base, results)) for r in reqs]
            try:
                await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), max_run_s)
            except asyncio.TimeoutError:
                log.warning("Workload timeout after %ds", max_run_s)
                for task in tasks: task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        duration = time.monotonic() - base
        telemetry_summary = telemetry.stop_and_collect() if telemetry else None
    end_energy = {role: read_energy(role, host) for role, host in NODES.items()}
    energy_by_role = {}; energy_errors = []
    for role in NODES:
        before, after = start_energy[role], end_energy[role]
        energy_errors.extend(before['errors']); energy_errors.extend(after['errors'])
        deltas = []
        for i in GPU_IDS:
            if i in before['values_mj'] and i in after['values_mj']:
                delta = after['values_mj'][i] - before['values_mj'][i]
                if delta < 0: energy_errors.append(f'{role}: gpu{i} energy counter decreased')
                else: deltas.append(delta / 1000.0)
            else: energy_errors.append(f'{role}: gpu{i} energy counter unavailable')
        energy_by_role[role] = sum(deltas) if deltas else None
    ok = [r for r in results if r.get("success")]; fail = len(reqs)-len(ok)
    if not ok:
        return {"status":"FAIL", "total_requests":len(reqs), "successful":0, "failed":fail,
                "energy_by_role_j":{k:(round(v,1) if v is not None else None) for k,v in energy_by_role.items()},
                "energy_errors":energy_errors,
                "slo_violation_rate":100.0 if reqs else 0, "slo_violating_requests":fail,
                **({"telemetry_summary":telemetry_summary} if telemetry_summary is not None else {})}
    ttft = [r["ttft_proc_ms"] or r["ttft_ms"] for r in ok]
    tpot = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    tokens = sum(r["completion_tokens"] for r in ok)
    available_energy = [v for v in energy_by_role.values() if v is not None]
    total_e = sum(available_energy) if len(available_energy) == len(NODES) else None
    viol_ttft = sum(v > TTFT_SLO_MS for v in ttft); viol_tpot = sum(v > TPOT_SLO_MS for v in tpot)
    request_violations = sum(not r.get('success') or r.get('ttft_violation') or r.get('tpot_violation') for r in results)
    request_violations += max(0, len(reqs) - len(results))
    return {
        "status":"PASS" if len(ok)==len(reqs) else "PARTIAL", "duration_s":round(duration,1),
        "total_requests":len(reqs), "successful":len(ok), "failed":fail, "total_tokens":tokens,
        "throughput_tok_s":round(tokens/duration,1),
        "ttft_proc_avg_ms":round(float(np.mean(ttft)),1), "ttft_proc_p50_ms":round(float(np.percentile(ttft,50)),1),
        "ttft_proc_p99_ms":round(float(np.percentile(ttft,99)),1),
        "tpot_avg_ms":round(float(np.mean(tpot)),1), "tpot_p50_ms":round(float(np.percentile(tpot,50)),1),
        "tpot_p99_ms":round(float(np.percentile(tpot,99)),1),
        "energy_by_role_j":{k:(round(v,1) if v is not None else None) for k,v in energy_by_role.items()},
        "total_energy_j":round(total_e,1) if total_e is not None else None,
        "energy_per_token_mj":round(total_e*1000/tokens,2) if total_e is not None and tokens else None,
        "energy_errors":energy_errors,
        "slo_violation_rate":round(request_violations/len(reqs)*100,1) if reqs else 0,
        "slo_violating_requests":request_violations,
        "ttft_violations":viol_ttft, "tpot_violations":viol_tpot,
        **({"telemetry_summary":telemetry_summary} if telemetry_summary is not None else {}),
    }


def run_point(dataset: str, qps: int, tier: bool):
    cleanup_all()
    preflight()
    if not tier:
        set_clocks(1410)
    url = deploy(tier=tier)
    if url is None: return {"status":"DEPLOY_FAILED"}
    if not test_generate(url): return {"status":"WARMUP_FAILED"}
    wl = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    reqs = [json.loads(x) for x in wl.read_text().splitlines() if x.strip()]
    last = max((x["arrival_time_s"] for x in reqs), default=0)
    run_id = f'{time.strftime("%Y%m%d_%H%M%S")}_{dataset}_qps{qps}_{uuid.uuid4().hex[:8]}'
    result = asyncio.run(run_workload(reqs, url+"/generate", int(min(max(400,last+150),900)),
                                      run_id if TELEMETRY_ENABLED else None))
    result["config"] = {"gpu_count":4 * TP,"gpu_ids":list(GPU_IDS),"pa":{"host":NODES["PA"],"tp":TP},"pf":{"host":NODES["PF"],"tp":TP},
                        "da":{"host":NODES["DA"],"tp":TP},"df":{"host":NODES["DF"],"tp":TP},
                        "comm_backend":AFD_BACKEND,"zmq_timeout_ms":ZMQ_TIMEOUT_MS,"zmq_sharding":ZMQ_SHARDING,"zmq_double_buffer":ZMQ_DOUBLE_BUFFER,
                        "ucx_rndv_thresh":UCX_RNDV_THRESH,
                        "ucx_zcopy_thresh":UCX_ZCOPY_THRESH,"tier":tier}
    return result


def save(payload):
    out = RESULTS_DIR / f"aflex_32g_cross_af_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(payload, indent=2)); log.info("Saved %s", out); return out


def _main_locked(lock: OrchestratorLock, lock_owner: dict):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--dataset", choices=["code","conv","both"], default="both")
    parser.add_argument("--qps", default="2,4,8,16")
    parser.add_argument("--no-tier", action="store_true")
    parser.add_argument("--tp", type=int, choices=(4, 8), default=8)
    parser.add_argument("--gpu-ids", default="", help="Comma-separated physical GPU IDs (0-7)")
    parser.add_argument("--telemetry", action=argparse.BooleanOptionalAction, default=False,
                        help="Collect per-GPU host telemetry during workload (default: off)")
    parser.add_argument("--zmq-sharding", action=argparse.BooleanOptionalAction, default=False,
                        help="Use one token-sharded ZMQ connection per TP rank (default: off)")
    parser.add_argument("--zmq-double-buffer", action="store_true",
                        help="Double-buffer pinned staging for sharded ZMQ (requires --zmq-sharding)")
    parser.add_argument("--telemetry-interval", type=float, default=1.0, metavar="SECONDS")
    args=parser.parse_args()
    global TP, GPU_IDS, TELEMETRY_ENABLED, TELEMETRY_INTERVAL_S, ZMQ_SHARDING, ZMQ_DOUBLE_BUFFER
    TP = args.tp; TELEMETRY_ENABLED = args.telemetry; TELEMETRY_INTERVAL_S = args.telemetry_interval; ZMQ_SHARDING = args.zmq_sharding; ZMQ_DOUBLE_BUFFER = args.zmq_double_buffer
    if ZMQ_DOUBLE_BUFFER and not ZMQ_SHARDING: parser.error("--zmq-double-buffer requires --zmq-sharding")
    if TELEMETRY_INTERVAL_S not in (0.5, 1.0): parser.error("--telemetry-interval must be 0.5 or 1.0")
    try:
        GPU_IDS = tuple(int(x.strip()) for x in args.gpu_ids.split(",") if x.strip()) if args.gpu_ids else tuple(range(TP))
    except ValueError as exc:
        parser.error(f"--gpu-ids must be comma-separated integers: {exc}")
    if len(GPU_IDS) != TP or len(set(GPU_IDS)) != TP or any(i < 0 or i > 7 for i in GPU_IDS):
        parser.error(f"--gpu-ids must contain exactly {TP} unique IDs in 0-7")
    validate_role_mapping(); ensure_dirs(); cleanup_all(); preflight()
    payload={"meta":{"orchestrator":{"pid":lock_owner["pid"],"lock_path":str(lock.path),"start":lock_owner["start"],"cmd":lock_owner["cmd"]},"topology":f"PA/PF/DA/DF each TP{TP} on separate node","nodes":NODES,"gpu_count":4 * TP,"gpu_ids":list(GPU_IDS),
                     "comm_backend":AFD_BACKEND,"zmq_timeout_ms":ZMQ_TIMEOUT_MS,"zmq_sharding":ZMQ_SHARDING,"zmq_double_buffer":ZMQ_DOUBLE_BUFFER,
                     "ucx_rndv_thresh":UCX_RNDV_THRESH,
                     "ucx_zcopy_thresh":UCX_ZCOPY_THRESH,"restart_per_qps":True,
                     "timestamp":time.strftime('%Y-%m-%d %H:%M:%S')},"results":{}}
    try:
        if args.smoke_only:
            if args.no_tier:
                set_clocks(1410)
            url=deploy(tier=not args.no_tier)
            payload["results"]["smoke"]={"status":"PASS" if url and test_generate(url) else "FAIL"}
        else:
            datasets=["code","conv"] if args.dataset=="both" else [args.dataset]
            for ds in datasets:
                for q in map(int,args.qps.split(",")):
                    log.info("===== %s QPS=%d =====",ds,q)
                    payload["results"][f"{ds}_qps{q}"]=run_point(ds,q,tier=not args.no_tier)
                    save(payload)
    finally:
        cleanup_all()
    save(payload)


def main():
    lock = OrchestratorLock()
    try:
        lock_owner = lock.acquire()
    except RuntimeError as exc:
        log.error("%s", exc)
        return 2
    try:
        _main_locked(lock, lock_owner)
        return 0
    finally:
        lock.release()



if __name__ == "__main__": sys.exit(main())
