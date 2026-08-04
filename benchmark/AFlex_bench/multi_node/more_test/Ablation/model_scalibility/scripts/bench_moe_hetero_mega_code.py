#!/usr/bin/env python3
"""Run heterogeneous Mixtral MegaScale code benchmarks.

This host-side driver is intended to run on node3.  It starts SGLang inside
``operator_test`` locally and over SSH on node4.  Every QPS is an independent
cleanup -> unlock -> deploy -> lock(1410 MHz) -> warmup -> benchmark cycle.
The default ``1p3d`` topology preserves the original 14-GPU result semantics.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import aiohttp

HERE = Path(__file__).resolve().parent
WORKLOAD_DIR = HERE.parent / "workloads"
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

NODE3_IP = os.environ.get("MN_NODE3_IP", "10.252.129.34")
NODE4_IP = os.environ.get("MN_NODE4_IP", "10.252.129.33")
CONTAINER = os.environ.get("SGLANG_BENCH_CONTAINER", "operator_test")
PYTHON = os.environ.get("SGLANG_CONTAINER_PYTHON", "/usr/bin/python3")
MODEL = os.environ.get("MIXTRAL_MODEL_PATH", "/models/Mixtral-8x7B/")
CONTAINER_ROOT = "/workspace/sglang"
LOG_DIR = f"{CONTAINER_ROOT}/benchmark/AFlex_bench/multi_node/more_test/Ablation/model_scalibility/scripts/logs"
CLEANUP = f"{CONTAINER_ROOT}/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"
IB_JSON_FILE = "/tmp/ib_scal_map.json"
AFLEX_ENERGY_MODEL_DIR = "/workspace/sglang/benchmark/AFlex_bench/energy_model/Mixtral-8x7B/models_v1"

MAX_FREQ_MHZ = 1410
BENCH_TOKEN_ID = 1000  # Valid ordinary token in Mixtral's 32k vocabulary.
GPUS_PER_NODE = 8
FULL_NODE_GPUS = list(range(GPUS_PER_NODE))
GPU_NIC = {
    0: "mlx5_0", 1: "mlx5_0", 2: "mlx5_1", 3: "mlx5_1",
    4: "mlx5_4", 5: "mlx5_4", 6: "mlx5_5", 7: "mlx5_5",
}
ROUTER_PORT_1P3D = 44000
ROUTER_PORT_3P1D = 44100
SUB_ROUTER_PORTS_3P1D = (45100, 45101, 45102)
ROUTER_PORT_3P1D_TP2 = 44300
SUB_ROUTER_PORTS_3P1D_TP2 = (45300, 45301, 45302)
TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("bench_moe_hetero_mega_code")


@dataclass(frozen=True)
class AFPair:
    name: str
    host: str
    mode: str
    ffn_gpus: tuple[int, ...]
    attn_gpus: tuple[int, ...]
    ffn_port: int
    attn_port: int
    ucx_port: int
    sched_port: int
    ffn_nccl_port: int
    attn_nccl_port: int
    ffn_bootstrap_port: int
    attn_bootstrap_port: int

    @property
    def cvd(self) -> tuple[int, ...]:
        return self.ffn_gpus + self.attn_gpus


@dataclass(frozen=True)
class Topology:
    name: str
    pairs: tuple[AFPair, ...]
    router_port: int
    sub_router_ports: tuple[int, ...]
    active_gpus: dict[str, list[int]]
    placement: str

    @property
    def prefills(self) -> tuple[AFPair, ...]:
        return tuple(pair for pair in self.pairs if pair.mode == "prefill")

    @property
    def decodes(self) -> tuple[AFPair, ...]:
        return tuple(pair for pair in self.pairs if pair.mode == "decode")


# Physical placement (FFN first, then attention) gives each process a private CVD.
PAIRS_1P3D = (
    AFPair("p", NODE3_IP, "prefill", (0, 1, 2, 3), (4,), 43202, 43200,
           28200, 68400, 37300, 37310, 50000, 49999),
    AFPair("d0", NODE3_IP, "decode", (5, 6), (7,), 43022, 43020,
           28300, 68500, 37320, 37330, 50010, 50011),
    AFPair("d1", NODE4_IP, "decode", (0, 1), (2,), 43042, 43040,
           28400, 68600, 37340, 37350, 50020, 50021),
    AFPair("d2", NODE4_IP, "decode", (3, 4), (5,), 43062, 43060,
           28500, 68700, 37360, 37370, 50030, 50031),
)

# 3P1D uses a disjoint port range so both topology layouts are unambiguous.
PAIRS_3P1D = (
    AFPair("p0", NODE3_IP, "prefill", (0, 1), (2,), 44202, 44200,
           29200, 69400, 38300, 38310, 51000, 50999),
    AFPair("p1", NODE3_IP, "prefill", (3, 4), (5,), 44222, 44220,
           29300, 69500, 38320, 38330, 51010, 51011),
    AFPair("p2", NODE4_IP, "prefill", (0, 1), (2,), 44242, 44240,
           29400, 69600, 38340, 38350, 51020, 51021),
    AFPair("d0", NODE4_IP, "decode", (3, 4), (5,), 44262, 44260,
           29500, 69700, 38360, 38370, 51030, 51031),
)

# 3P1D TP2 dedicates four GPUs to every instance: FFN TP2 + attention TP2.
PAIRS_3P1D_TP2 = (
    AFPair("p0", NODE3_IP, "prefill", (0, 1), (2, 3), 45202, 45200,
           30200, 60400, 39300, 39310, 52000, 51999),
    AFPair("p1", NODE3_IP, "prefill", (4, 5), (6, 7), 45222, 45220,
           30300, 60500, 39320, 39330, 52010, 52011),
    AFPair("p2", NODE4_IP, "prefill", (0, 1), (2, 3), 45242, 45240,
           30400, 60600, 39340, 39350, 52020, 52021),
    AFPair("d0", NODE4_IP, "decode", (4, 5), (6, 7), 45262, 45260,
           30500, 60700, 39360, 39370, 52030, 52031),
)

TOPOLOGIES = {
    "1p3d": Topology(
        "1p3d", PAIRS_1P3D, ROUTER_PORT_1P3D, (),
        {NODE3_IP: list(range(8)), NODE4_IP: list(range(6))},
        "node3 P(FFN0-3/A4)+D0(FFN5-6/A7), node4 D1(FFN0-1/A2)+D2(FFN3-4/A5)",
    ),
    "3p1d": Topology(
        "3p1d", PAIRS_3P1D, ROUTER_PORT_3P1D, SUB_ROUTER_PORTS_3P1D,
        {NODE3_IP: list(range(6)), NODE4_IP: list(range(6))},
        "node3 P0(FFN0-1/A2)+P1(FFN3-4/A5), node4 P2(FFN0-1/A2)+D0(FFN3-4/A5)",
    ),
    "3p1d_tp2": Topology(
        "3p1d_tp2", PAIRS_3P1D_TP2,
        ROUTER_PORT_3P1D_TP2, SUB_ROUTER_PORTS_3P1D_TP2,
        {NODE3_IP: list(range(8)), NODE4_IP: list(range(8))},
        "node3 P0(FFN0-1/A2-3)+P1(FFN4-5/A6-7), "
        "node4 P2(FFN0-1/A2-3)+D0(FFN4-5/A6-7)",
    ),
}

# Cleanup is topology-independent and covers every HTTP, NCCL, bootstrap, UCX,
# scheduler, sub-router, and top-router port used by any layout.
CLEANUP_PORTS = tuple(sorted({
    *(topology.router_port for topology in TOPOLOGIES.values()),
    *(port for topology in TOPOLOGIES.values() for port in topology.sub_router_ports),
    *(port for topology in TOPOLOGIES.values() for pair in topology.pairs for port in (
        pair.ffn_port, pair.attn_port, pair.ucx_port, pair.sched_port,
        pair.ffn_nccl_port, pair.attn_nccl_port,
        pair.ffn_bootstrap_port, pair.attn_bootstrap_port,
    )),
}))
CLEANUP_VERIFY_PORTS = CLEANUP_PORTS


def _ssh(host: str, command: str) -> list[str]:
    return [
        "ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes",
        host, command,
    ]


def _run(argv: list[str], *, timeout: int | None = None, capture: bool = False) -> subprocess.CompletedProcess[str]:
    log.debug("exec: %s", shlex.join(argv))
    return subprocess.run(
        argv, check=False, text=True, timeout=timeout,
        capture_output=capture,
    )


def dexec(host: str, shell_command: str, *, timeout: int | None = None,
          capture: bool = False) -> subprocess.CompletedProcess[str]:
    inner = ["docker", "exec", CONTAINER, "bash", "-lc", shell_command]
    if host == NODE3_IP:
        return _run(inner, timeout=timeout, capture=capture)
    return _run(_ssh(host, shlex.join(inner)), timeout=timeout, capture=capture)


def _force_cleanup_command() -> str:
    ports = " ".join(map(str, CLEANUP_PORTS))
    return f"""
if ! command -v ss >/dev/null 2>&1; then
    echo "ss is required for benchmark cleanup" >&2
    exit 1
fi
kill_port_users() {{
    signal=$1
    port=$2
    listeners="$(ss -H -ltnp "sport = :$port" 2>/dev/null || true)"
    while [[ $listeners =~ pid=([0-9]+) ]]; do
        pid="${{BASH_REMATCH[1]}}"
        kill -"$signal" "$pid" 2>/dev/null || true
        listeners="${{listeners#*pid=$pid}}"
    done
}}
kill_sglang_trees() {{
    signal=$1
    roots="$(pgrep -f '[s]glang\.launch_server|[s]glang_router|^sglang::' || true)"
    descendants=""
    frontier="$roots"
    while [ -n "$frontier" ]; do
        children=""
        for pid in $frontier; do
            children="$children $(pgrep -P "$pid" 2>/dev/null || true)"
        done
        descendants="$children $descendants"
        frontier="$children"
    done
    # Children first, then parents. The bracketed patterns above cannot match this bash.
    for pid in $descendants $roots; do
        kill -"$signal" "$pid" 2>/dev/null || true
    done
}}
for port in {ports}; do
    kill_port_users TERM "$port"
done
kill_sglang_trees TERM
sleep 2
for port in {ports}; do
    kill_port_users KILL "$port"
done
kill_sglang_trees KILL
"""


def _listening_ports_command() -> str:
    ports = " ".join(map(str, CLEANUP_VERIFY_PORTS))
    return f"""
if ! command -v ss >/dev/null 2>&1; then
    echo "ss is required for benchmark cleanup verification" >&2
    exit 1
fi
for port in {ports}; do
    if [ -n "$(ss -H -ltn "sport = :$port")" ]; then
        echo "$port"
    fi
done
"""


def cleanup_all() -> None:
    log.info("Cleaning services on node3 and node4")
    errors: list[str] = []
    for host in (NODE3_IP, NODE4_IP):
        result = dexec(host, f"bash {shlex.quote(CLEANUP)}", timeout=120)
        if result.returncode:
            log.warning("cleanup returned %d on %s", result.returncode, host)

        result = dexec(host, _force_cleanup_command(), timeout=120, capture=True)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            errors.append(f"forced cleanup failed on {host}: {detail or result.returncode}")

    time.sleep(2)
    for host in (NODE3_IP, NODE4_IP):
        result = dexec(host, _listening_ports_command(), timeout=30, capture=True)
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            errors.append(f"cleanup verification failed on {host}: {detail or result.returncode}")
        elif result.stdout.strip():
            errors.append(
                f"ports still listening on {host}: "
                + ", ".join(result.stdout.split())
            )

    if errors:
        raise RuntimeError("Cleanup did not leave a deployable topology: " + "; ".join(errors))


def set_frequency(host: str, gpus: list[int], *, lock: bool) -> None:
    if lock:
        commands = [
            f"nvidia-smi -i {gpu} --lock-gpu-clocks={MAX_FREQ_MHZ},{MAX_FREQ_MHZ}"
            for gpu in gpus
        ]
        action = f"lock {MAX_FREQ_MHZ}MHz"
    else:
        commands = [f"nvidia-smi -i {gpu} --reset-gpu-clocks" for gpu in gpus]
        action = "unlock"
    result = dexec(host, " && ".join(commands), timeout=120, capture=True)
    if result.returncode:
        raise RuntimeError(
            f"Failed to {action} GPUs {gpus} on {host}: "
            f"{result.stderr.strip() or result.stdout.strip()}"
        )
    log.info("%s GPUs %s on %s", action, gpus, host)


def unlock_all_gpus() -> None:
    for host in (NODE3_IP, NODE4_IP):
        set_frequency(host, FULL_NODE_GPUS, lock=False)


def lock_active_gpus(topology: Topology) -> None:
    for host, gpus in topology.active_gpus.items():
        set_frequency(host, gpus, lock=True)


def write_ib_json() -> None:
    mapping = {str(gpu): nic for gpu, nic in GPU_NIC.items()}
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(mapping, handle)
        temp_name = handle.name
    try:
        local_copy = _run(["docker", "cp", temp_name, f"{CONTAINER}:{IB_JSON_FILE}"])
        if local_copy.returncode:
            raise RuntimeError("docker cp of IB mapping failed on node3")
        remote_temp = f"/tmp/ib_scal_map_{os.getpid()}.json"
        scp = _run([
            "scp", "-o", "StrictHostKeyChecking=no", "-q",
            temp_name, f"{NODE4_IP}:{remote_temp}",
        ])
        if scp.returncode:
            raise RuntimeError("scp of IB mapping failed for node4")
        remote_copy = _run(_ssh(
            NODE4_IP,
            shlex.join(["docker", "cp", remote_temp, f"{CONTAINER}:{IB_JSON_FILE}"])
            + f"; rm -f {shlex.quote(remote_temp)}",
        ))
        if remote_copy.returncode:
            raise RuntimeError("docker cp of IB mapping failed on node4")
    finally:
        Path(temp_name).unlink(missing_ok=True)


def ensure_log_dirs() -> None:
    for host in (NODE3_IP, NODE4_IP):
        result = dexec(host, f"mkdir -p {shlex.quote(LOG_DIR)}", timeout=30)
        if result.returncode:
            raise RuntimeError(f"Cannot create log directory on {host}")


def prewarm_ipc_cpp() -> None:
    command = (
        "CUDA_VISIBLE_DEVICES=0 " + PYTHON + " -c "
        + shlex.quote(
            "import sys; sys.path.insert(0, '/workspace/sglang/python'); "
            "from sglang.srt.layers.afd_ipc_cpp import get_module; "
            "get_module(); print('JIT_WARM_OK')"
        )
    )
    for host in (NODE3_IP, NODE4_IP):
        result = dexec(host, command, timeout=600, capture=True)
        if result.returncode:
            raise RuntimeError(
                f"ipc_cpp JIT warmup failed on {host}: "
                f"{result.stderr.strip() or result.stdout.strip()}"
            )


def _afd_env(pair: AFPair, role: str, ipc_per_rank: bool = False) -> str:
    is_attn = role == "attn"
    role_gpus = pair.attn_gpus if is_attn else pair.ffn_gpus
    peer_device = 0 if is_attn else len(pair.ffn_gpus)
    values = {
        "CUDA_VISIBLE_DEVICES": ",".join(map(str, pair.cvd)),
        "SGLANG_HOST_IP": pair.host,
        "SGLANG_DISABLE_REQUEST_LOGGING": "true",
        "UCX_LOG_LEVEL": "fatal",
        "AFD_UCX_TLS": "rc,tcp,cuda_copy,cuda_ipc",
        "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE": "128",
        "AFD_ASYNC_PIPELINE": "1",
        "AFD_IPC_SYNC_MODE": "ipc_event",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "600",
        "AFD_UCX_BASE_PORT": str(pair.ucx_port),
        "AFD_SCHED_PORT": str(pair.sched_port),
        "AFD_NVML_DEVICE_INDICES": ",".join(map(str, role_gpus)),
        "AFD_NVML_DEVICE_INDEX": str(role_gpus[0]),
    }
    if ipc_per_rank:
        values["AFD_IPC_PEER_OFFSET"] = "-2" if is_attn else "+2"
        values["AFD_IPC_CHANNEL_BASE"] = str(pair.sched_port % 1000)
    else:
        values["AFD_IPC_PEER_DEVICE"] = str(peer_device)
    if is_attn:
        values["AFD_UCX_FFN_HOST"] = "127.0.0.1"
    # No AFD_DVFS_* variable and no DVFS server flag: MegaScale is fixed-frequency.
    return "export " + " ".join(f"{key}={shlex.quote(value)}" for key, value in values.items()) + ";"


def _afd_flags(pair: AFPair, role: str, ipc_per_rank: bool = False, aflex: bool = False) -> str:
    tp = len(pair.attn_gpus) if role == "attn" else len(pair.ffn_gpus)
    flags = [
        "--model-path", shlex.quote(MODEL),
        "--tp", str(tp),
        "--afd-comm-backend", "ipc_cpp",
        "--afd-micro-batch", "1",
        "--afd-attn-tp", str(len(pair.attn_gpus)),
        "--afd-ffn-tp", str(len(pair.ffn_gpus)),
        "--mem-fraction-static", "0.85",
        "--max-running-requests", "512",
        "--skip-server-warmup",
        "--watchdog-timeout", "600",
        "--disable-cuda-graph",
        "--disable-piecewise-cuda-graph",
        "--afd-disagg-interleave-poll",
        "--disable-radix-cache",
        "--num-reserved-decode-tokens", "512",
        "--disaggregation-transfer-backend", "mooncake",
        "--disaggregation-ib-device", IB_JSON_FILE,
        "--enable-metrics",
    ]
    if ipc_per_rank:
        flags.append("--afd-ipc-per-rank")
    if aflex:
        flags.extend([
            "--afd-dvfs-enabled",
            "--afd-energy-model-dir", AFLEX_ENERGY_MODEL_DIR,
            "--afd-ttft-slo-ms", str(int(TTFT_SLO_MS)),
            "--afd-tpot-slo-us", str(int(TPOT_SLO_MS * 1000)),
            "--afd-dvfs-decode-compositional",
            "--afd-dvfs-idle-lock",
        ])
    return " ".join(flags)


def launch_process(
    pair: AFPair, role: str, run_tag: str, ipc_per_rank: bool = False,
    aflex: bool = False,
) -> None:
    is_attn = role == "attn"
    port = pair.attn_port if is_attn else pair.ffn_port
    nccl_port = pair.attn_nccl_port if is_attn else pair.ffn_nccl_port
    bootstrap = pair.attn_bootstrap_port if is_attn else pair.ffn_bootstrap_port
    base_gpu_id = len(pair.ffn_gpus) if is_attn else 0
    command = (
        f"{_afd_env(pair, role, ipc_per_rank)} "
        "setsid prlimit --memlock=unlimited:unlimited "
        f"{PYTHON} -m sglang.launch_server "
        f"--host {pair.host} --port {port} "
        f"--afd-perspective {role} --disaggregation-mode {pair.mode} "
        f"--base-gpu-id {base_gpu_id} --nccl-port {nccl_port} "
        f"--disaggregation-bootstrap-port {bootstrap} "
        f"{_afd_flags(pair, role, ipc_per_rank, aflex)} "
        f"> {LOG_DIR}/{run_tag}_{pair.name}_{role}.log 2>&1 < /dev/null &"
    )
    result = dexec(pair.host, command, timeout=30)
    if result.returncode:
        raise RuntimeError(f"Failed to launch {pair.name}/{role} on {pair.host}")


def wait_health(host: str, port: int, timeout: int, *, model_info: bool) -> bool:
    import requests

    endpoint = "get_model_info" if model_info else "health"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"http://{host}:{port}/{endpoint}", timeout=10)
            if response.status_code == 200:
                return True
        except requests.RequestException:
            pass
        time.sleep(3)
    return False


def _launch_router(command: str, port: int, log_name: str) -> None:
    result = dexec(NODE3_IP, command, timeout=30)
    if result.returncode or not wait_health(NODE3_IP, port, 90, model_info=False):
        raise RuntimeError(f"Router on port {port} failed health check; see {log_name}")


def deploy(topology: Topology, run_tag: str, ipc_per_rank: bool = False, aflex: bool = False) -> str:
    ensure_log_dirs()
    write_ib_json()
    prewarm_ipc_cpp()
    log.info("Deploying %s topology: %s", topology.name, topology.placement)
    for pair in topology.pairs:
        launch_process(pair, "ffn", run_tag, ipc_per_rank, aflex)
        # The FFN HTTP endpoint is ready only after TP distributed initialization.
        # Waiting here prevents its NCCL/bootstrap setup from racing the paired A.
        if not wait_health(pair.host, pair.ffn_port, 600, model_info=True):
            raise RuntimeError(
                f"{pair.name}/ffn failed health check; "
                f"see {run_tag}_{pair.name}_ffn.log"
            )
        log.info("FFN ready: %s on %s", pair.name, pair.host)
        launch_process(pair, "attn", run_tag, ipc_per_rank, aflex)
        time.sleep(2)

    for pair in topology.pairs:
        if not wait_health(pair.host, pair.attn_port, 600, model_info=False):
            raise RuntimeError(f"{pair.name}/attn failed health check; see {run_tag}_{pair.name}_attn.log")
        log.info("Ready: %s on %s", pair.name, pair.host)

    decode_args = " ".join(
        f"--decode http://{pair.host}:{pair.attn_port}" for pair in topology.decodes
    )
    if len(topology.prefills) == 1:
        prefill = topology.prefills[0]
        router_log = f"{run_tag}_router.log"
        router_command = (
            f"setsid {PYTHON} -m sglang_router.launch_router "
            "--pd-disaggregation --mini-lb "
            f"--prefill http://{prefill.host}:{prefill.attn_port} {prefill.attn_bootstrap_port} "
            f"{decode_args} --host {NODE3_IP} --port {topology.router_port} "
            f"> {LOG_DIR}/{router_log} 2>&1 < /dev/null &"
        )
        _launch_router(router_command, topology.router_port, router_log)
    else:
        if len(topology.sub_router_ports) != len(topology.prefills):
            raise RuntimeError("Each prefill instance must have one sub-router port")
        sub_urls: list[str] = []
        for index, (prefill, port) in enumerate(
            zip(topology.prefills, topology.sub_router_ports, strict=True)
        ):
            router_log = f"{run_tag}_sub_router_{index}.log"
            router_command = (
                f"setsid {PYTHON} -m sglang_router.launch_router "
                "--pd-disaggregation --mini-lb "
                f"--prefill http://{prefill.host}:{prefill.attn_port} {prefill.attn_bootstrap_port} "
                f"{decode_args} --host {NODE3_IP} --port {port} "
                f"> {LOG_DIR}/{router_log} 2>&1 < /dev/null &"
            )
            _launch_router(router_command, port, router_log)
            sub_urls.append(f"http://{NODE3_IP}:{port}")

        router_log = f"{run_tag}_top_router.log"
        top_router_command = (
            f"setsid {PYTHON} -m sglang_router.launch_router "
            f"--host {NODE3_IP} --port {topology.router_port} "
            f"--policy round_robin --worker-urls {' '.join(sub_urls)} "
            f"> {LOG_DIR}/{router_log} 2>&1 < /dev/null &"
        )
        _launch_router(top_router_command, topology.router_port, router_log)
        log.info("Top-level router balances across: %s", sub_urls)

    return f"http://{NODE3_IP}:{topology.router_port}"


def read_energy(host: str) -> dict[int, int]:
    code = (
        "import json,pynvml; pynvml.nvmlInit(); "
        "print(json.dumps({i:pynvml.nvmlDeviceGetTotalEnergyConsumption("
        "pynvml.nvmlDeviceGetHandleByIndex(i)) for i in range(8)})); "
        "pynvml.nvmlShutdown()"
    )
    result = dexec(host, f"{PYTHON} -c {shlex.quote(code)}", timeout=30, capture=True)
    lines = [line for line in result.stdout.splitlines() if line.strip().startswith("{")]
    if result.returncode or not lines:
        raise RuntimeError(
            f"Energy read failed on {host}: {result.stderr.strip() or result.stdout.strip()}"
        )
    return {int(key): int(value) for key, value in json.loads(lines[-1]).items()}


def energy_delta(start: dict[str, dict[int, int]], end: dict[str, dict[int, int]],
                 selection: dict[str, list[int]]) -> dict[str, Any]:
    per_node: dict[str, float] = {}
    for host, gpus in selection.items():
        per_node[host] = sum(
            max(0, end[host][gpu] - start[host][gpu]) for gpu in gpus
        ) / 1000.0
    return {
        "per_node_j": {host: round(value, 3) for host, value in per_node.items()},
        "total_j": round(sum(per_node.values()), 3),
        "gpu_selection": selection,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * p / 100.0
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def meta_number(meta: dict[str, Any], *keys: str) -> float:
    for key in keys:
        value = meta.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    return 0.0


async def send_one(session: aiohttp.ClientSession, url: str, request: dict[str, Any],
                   base_time: float, index: int, token_id: int) -> dict[str, Any]:
    delay = float(request["arrival_time_s"]) - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {
        "input_ids": [token_id] * int(request["input_len"]),
        "sampling_params": {
            "max_new_tokens": int(request["output_len"]),
            "temperature": 0.0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    started = time.monotonic()
    first_chunk_at: float | None = None
    chunks = 0
    last_meta: dict[str, Any] = {}
    try:
        async with session.post(url, json=payload) as response:
            if response.status != 200:
                body = (await response.text())[:500]
                return {"request_index": index, "success": False, "error": f"HTTP {response.status}: {body}"}
            async for raw_line in response.content:
                text = raw_line.decode(errors="replace").strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    chunk = json.loads(text)
                except json.JSONDecodeError:
                    continue
                if first_chunk_at is None:
                    first_chunk_at = time.monotonic()
                chunks += 1
                if isinstance(chunk, dict) and isinstance(chunk.get("meta_info"), dict):
                    last_meta = chunk["meta_info"]
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        return {"request_index": index, "success": False, "error": repr(exc)}

    ended = time.monotonic()
    if first_chunk_at is None:
        return {"request_index": index, "success": False, "error": "stream ended without a token"}
    completion_tokens = int(last_meta.get("completion_tokens") or chunks)
    client_ttft_ms = (first_chunk_at - started) * 1000.0
    pure_ttft_s = meta_number(
        last_meta,
        "ttft_pure_processing",       # includes cached PA pure-processing value in PD+AF
        "cached_ttft_processing",
        "time_to_first_token_processing",
    )
    pure_ttft_ms = pure_ttft_s * 1000.0
    # Inter-token period excludes TTFT and divides by intervals, not token count.
    tpot_ms = (
        (ended - first_chunk_at) * 1000.0 / (completion_tokens - 1)
        if completion_tokens > 1 else 0.0
    )
    return {
        "request_index": index,
        "success": True,
        "input_tokens": int(request["input_len"]),
        "requested_output_tokens": int(request["output_len"]),
        "completion_tokens": completion_tokens,
        "client_ttft_ms": client_ttft_ms,
        "ttft_pure_processing_ms": pure_ttft_ms,
        "ttft_metric_source": (
            "server_meta_info" if pure_ttft_ms > 0 else "client_fallback"
        ),
        "tpot_ms": tpot_ms,
        "e2e_ms": (ended - started) * 1000.0,
    }


async def run_workload(requests: list[dict[str, Any]], url: str, max_run_s: int,
                       request_timeout_s: int, token_id: int,
                       active_gpus: dict[str, list[int]]) -> dict[str, Any]:
    energy_start = {host: read_energy(host) for host in (NODE3_IP, NODE4_IP)}
    timeout = aiohttp.ClientTimeout(total=request_timeout_s)
    tasks: dict[asyncio.Task[dict[str, Any]], int] = {}
    started = time.monotonic()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for index, request in enumerate(requests):
            task = asyncio.create_task(send_one(
                session, url + "/generate", request, started, index, token_id
            ))
            tasks[task] = index
        done, pending = await asyncio.wait(tasks, timeout=max_run_s)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
    duration_s = time.monotonic() - started
    energy_end = {host: read_energy(host) for host in (NODE3_IP, NODE4_IP)}

    by_index: dict[int, dict[str, Any]] = {}
    for task in done:
        index = tasks[task]
        try:
            result = task.result()
        except Exception as exc:
            result = {"request_index": index, "success": False, "error": repr(exc)}
        by_index[index] = result
    for task in pending:
        index = tasks[task]
        by_index[index] = {
            "request_index": index,
            "success": False,
            "timed_out": True,
            "error": f"global benchmark timeout after {max_run_s}s",
        }
    # This invariant prevents completed, failed, or timed-out requests disappearing.
    for index in range(len(requests)):
        by_index.setdefault(index, {
            "request_index": index,
            "success": False,
            "error": "request task produced no result",
        })
    results = [by_index[index] for index in range(len(requests))]
    successful = [result for result in results if result.get("success")]
    failed = [result for result in results if not result.get("success")]
    timed_out = [result for result in failed if result.get("timed_out") or "TimeoutError" in result.get("error", "")]

    server_ttft = [
        float(result["ttft_pure_processing_ms"])
        for result in successful if result.get("ttft_pure_processing_ms", 0) > 0
    ]
    client_ttft = [float(result["client_ttft_ms"]) for result in successful]
    preferred_ttft = [
        float(result.get("ttft_pure_processing_ms") or result["client_ttft_ms"])
        for result in successful
    ]
    tpots = [float(result["tpot_ms"]) for result in successful if result.get("completion_tokens", 0) > 1]
    completion_tokens = sum(int(result.get("completion_tokens", 0)) for result in successful)
    violations = sum(
        1 for result in successful
        if (float(result.get("ttft_pure_processing_ms") or result["client_ttft_ms"]) > TTFT_SLO_MS
            or (result.get("completion_tokens", 0) > 1 and float(result["tpot_ms"]) > TPOT_SLO_MS))
    )
    active_energy = energy_delta(energy_start, energy_end, active_gpus)
    full_energy = energy_delta(
        energy_start, energy_end,
        {NODE3_IP: FULL_NODE_GPUS, NODE4_IP: FULL_NODE_GPUS},
    )
    summary: dict[str, Any] = {
        "status": "PASS" if not failed else ("TIMEOUT" if len(timed_out) == len(failed) else "FAIL"),
        "duration_s": round(duration_s, 3),
        "total_requests": len(requests),
        "successful": len(successful),
        "failed": len(failed),
        "timed_out": len(timed_out),
        "completion_tokens": completion_tokens,
        "throughput_tok_s": round(completion_tokens / duration_s, 3) if duration_s else 0.0,
        "ttft_preferred_metric": "per-request server pure-processing with client fallback",
        "ttft_preferred_avg_ms": round(sum(preferred_ttft) / len(preferred_ttft), 3) if preferred_ttft else 0.0,
        "ttft_preferred_p50_ms": round(percentile(preferred_ttft, 50), 3),
        "ttft_preferred_p99_ms": round(percentile(preferred_ttft, 99), 3),
        "ttft_server_pure_count": len(server_ttft),
        "ttft_server_pure_avg_ms": round(sum(server_ttft) / len(server_ttft), 3) if server_ttft else 0.0,
        "ttft_client_avg_ms": round(sum(client_ttft) / len(client_ttft), 3) if client_ttft else 0.0,
        "ttft_client_p50_ms": round(percentile(client_ttft, 50), 3),
        "ttft_client_p99_ms": round(percentile(client_ttft, 99), 3),
        "tpot_avg_ms": round(sum(tpots) / len(tpots), 3) if tpots else 0.0,
        "tpot_p50_ms": round(percentile(tpots, 50), 3),
        "tpot_p99_ms": round(percentile(tpots, 99), 3),
        "active_gpu_energy": active_energy,
        "full_node_energy": full_energy,
        "active_energy_per_output_token_mj": (
            round(active_energy["total_j"] * 1000.0 / completion_tokens, 3)
            if completion_tokens else 0.0
        ),
        "full_node_energy_per_output_token_mj": (
            round(full_energy["total_j"] * 1000.0 / completion_tokens, 3)
            if completion_tokens else 0.0
        ),
        "slo_violating_successes": violations,
        "unsuccessful_requests": len(failed),
        "slo_violation_rate_pct": round(
            (violations + len(failed)) * 100.0 / len(requests), 3
        ) if requests else 0.0,
        "request_results": results,
    }
    return summary


def warmup(router_url: str, token_id: int) -> None:
    import requests

    payload = {
        "input_ids": [token_id] * 32,
        "sampling_params": {"max_new_tokens": 8, "temperature": 0.0, "ignore_eos": True},
        "stream": False,
    }
    response = requests.post(router_url + "/generate", json=payload, timeout=300)
    if response.status_code != 200:
        raise RuntimeError(f"Warmup failed: HTTP {response.status_code}: {response.text[:500]}")
    log.info("Warmup completed")


def load_workload(dataset: str, qps: int) -> list[dict[str, Any]]:
    path = WORKLOAD_DIR / f"macro_{dataset}_qps{qps}.jsonl"
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as handle:
        requests = [json.loads(line) for line in handle if line.strip()]
    for index, request in enumerate(requests):
        missing = {"input_len", "output_len", "arrival_time_s"} - request.keys()
        if missing:
            raise ValueError(f"{path}:{index + 1} missing fields {sorted(missing)}")
    return requests


def atomic_write_json(path: Path, document: dict[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    os.replace(temp, path)


def topology_metadata(topology: Topology) -> list[dict[str, Any]]:
    return [
        {
            **asdict(pair),
            "ffn_tp": len(pair.ffn_gpus),
            "attn_tp": len(pair.attn_gpus),
            "cuda_visible_devices": list(pair.cvd),
        }
        for pair in topology.pairs
    ]


def parse_qps_list(value: str) -> list[int]:
    qps_values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not qps_values or any(qps <= 0 for qps in qps_values):
        raise argparse.ArgumentTypeError("qps-list must contain positive comma-separated integers")
    return qps_values


def run_one(args: argparse.Namespace, topology: Topology, qps: int) -> dict[str, Any]:
    aflex_tag = "_aflex" if args.aflex else ""
    run_tag = f"hetero_mega_code_{topology.name}{aflex_tag}_q{qps}_{time.strftime('%Y%m%d_%H%M%S')}"
    requests = load_workload(args.dataset, qps)
    last_arrival = max((float(req["arrival_time_s"]) for req in requests), default=0.0)
    max_run_s = int(min(max(args.min_run_seconds, last_arrival + args.timeout_slack), args.max_run_seconds))
    log.info("QPS=%d: %d requests, last arrival %.1fs, timeout %ds", qps, len(requests), last_arrival, max_run_s)

    cleanup_all()
    unlock_all_gpus()
    time.sleep(args.settle_seconds)
    try:
        router_url = deploy(topology, run_tag, args.ipc_per_rank, args.aflex)
        if args.aflex:
            lock_active_gpus(topology)
            unlock_all_gpus()
            log.info("AFlex: locked and immediately unlocked to seed DVFS baseline")
        else:
            lock_active_gpus(topology)
        warmup(router_url, args.token_id)
        time.sleep(args.warmup_settle_seconds)
        summary = asyncio.run(run_workload(
            requests, router_url, max_run_s, args.request_timeout_seconds, args.token_id,
            topology.active_gpus,
        ))
        summary.update({"qps": qps, "run_tag": run_tag, "workload_requests": len(requests)})
        log.info(
            "QPS=%d %s: ok=%d/%d TTFT=%.1fms TPOT=%.1fms active=%.1fJ full-node=%.1fJ",
            qps, summary["status"], summary["successful"], summary["total_requests"],
            summary["ttft_preferred_avg_ms"], summary["tpot_avg_ms"],
            summary["active_gpu_energy"]["total_j"], summary["full_node_energy"]["total_j"],
        )
        return summary
    except Exception as exc:
        log.exception("QPS=%d failed", qps)
        return {"qps": qps, "run_tag": run_tag, "status": "ERROR", "error": repr(exc)}
    finally:
        try:
            cleanup_all()
        finally:
            try:
                unlock_all_gpus()
            except Exception:
                log.exception("Final GPU unlock failed")
        time.sleep(args.settle_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["code"], default="code")
    parser.add_argument(
        "--topology", choices=sorted(TOPOLOGIES), default="1p3d",
        help="Deployment topology (default: 1p3d, preserving legacy behavior)",
    )
    parser.add_argument("--qps-list", type=parse_qps_list, default=parse_qps_list("2,4,8,16"))
    parser.add_argument("--token-id", type=int, default=BENCH_TOKEN_ID)
    parser.add_argument(
        "--ipc-per-rank",
        action="store_true",
        help="Enable homogeneous TP2 per-rank ipc_cpp channels (3p1d_tp2 only).",
    )
    parser.add_argument(
        "--aflex",
        action="store_true",
        help="Enable AFlex DVFS mode (requires --topology 3p1d_tp2 --ipc-per-rank).",
    )
    parser.add_argument("--request-timeout-seconds", type=int, default=900)
    parser.add_argument("--min-run-seconds", type=int, default=400)
    parser.add_argument("--max-run-seconds", type=int, default=1200)
    parser.add_argument("--timeout-slack", type=int, default=180)
    parser.add_argument("--settle-seconds", type=int, default=8)
    parser.add_argument("--warmup-settle-seconds", type=int, default=3)
    parser.add_argument("--output", type=Path, help="Result JSON path (default: results/<timestamp>.json)")
    args = parser.parse_args()
    if not 0 <= args.token_id < 32000:
        parser.error("--token-id must be in Mixtral vocabulary range [0, 32000)")
    if args.ipc_per_rank and args.topology != "3p1d_tp2":
        parser.error("--ipc-per-rank is only supported with --topology 3p1d_tp2")
    if args.aflex:
        if args.topology != "3p1d_tp2":
            parser.error("--aflex requires --topology 3p1d_tp2")
        if not args.ipc_per_rank:
            parser.error("--aflex requires --ipc-per-rank")

    topology = TOPOLOGIES[args.topology]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    aflex_suffix = "_aflex" if args.aflex else ""
    output = args.output or RESULTS_DIR / f"moe_hetero_mega_code_{topology.name}{aflex_suffix}_{timestamp}.json"
    if topology.name not in output.stem:
        output = output.with_name(f"{output.stem}_{topology.name}{aflex_suffix}{output.suffix}")
    output.parent.mkdir(parents=True, exist_ok=True)
    document: dict[str, Any] = {
        "meta": {
            "benchmark": "Mixtral heterogeneous MegaScale code",
            "topology_name": topology.name,
            "created_at": timestamp,
            "node3": NODE3_IP,
            "node4": NODE4_IP,
            "container": CONTAINER,
            "model": MODEL,
            "dataset": args.dataset,
            "qps_list": args.qps_list,
            "frequency_mhz": MAX_FREQ_MHZ,
            "afd_dvfs_enabled": args.aflex,
            "afd_micro_batch": 1,
            "afd_dynamic_micro_batch": False,
            "afd_comm_backend": "ipc_cpp",
            "afd_ipc_per_rank": args.ipc_per_rank,
            "aflex_mode": args.aflex,
            "pd_transfer_backend": "mooncake",
            "bench_token_id": args.token_id,
            "active_gpu_count": sum(len(gpus) for gpus in topology.active_gpus.values()),
            "active_gpus": topology.active_gpus,
            "full_node_gpu_count": 16,
            "topology": topology_metadata(topology),
        },
        "results": {},
    }
    atomic_write_json(output, document)
    log.info("Incremental result file: %s", output)

    try:
        for qps in args.qps_list:
            document["results"][str(qps)] = run_one(args, topology, qps)
            atomic_write_json(output, document)
            log.info("Persisted QPS=%d result to %s", qps, output)
    except KeyboardInterrupt:
        log.warning("Interrupted; completed QPS results remain in %s", output)
        return 130

    failures = [result for result in document["results"].values() if result.get("status") != "PASS"]
    log.info("Finished %d QPS points; failures=%d; output=%s", len(args.qps_list), len(failures), output)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
