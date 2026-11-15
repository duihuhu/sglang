"""
Run all 4 schemes for Code dataset, QPS = 2, 4, 8, 16. Restart per QPS.
  - SGLang: 16xTP1 (8/node), round-robin, locked 1410MHz
  - DynamoLLM: 16xTP1 (8/node), round-robin, DVFS unified
  - DistServe: 2P(TP4) + 4D(TP2) = 16GPU, locked 1410MHz
  - BiScale: 2P(TP4) + 4D(TP2) = 16GPU, DVFS biscale
"""

import json
import os
import time

from . import fabric_affinity as FABRIC
from . import runtime as RMB

RMB.NODE1_IP = os.environ.get("MN_NODE1_IP", RMB.NODE1_IP)
RMB.NODE2_IP = os.environ.get("MN_NODE2_IP", RMB.NODE2_IP)

NODE1 = RMB.NODE1_IP
NODE2 = RMB.NODE2_IP
MODEL = RMB.MODEL
PYTHON = RMB.PYTHON
LOG_C = RMB.LOG_C
FLAGS = RMB.COMMON_BENCH_SERVER_FLAGS
GPU_NIC = RMB.GPU_NIC
MOONCAKE_IP_BASE = {
    NODE1: os.environ.get("MN_NODE1_MOONCAKE_IP_BASE", "10.0.0.70"),
    NODE2: os.environ.get("MN_NODE2_MOONCAKE_IP_BASE", "10.0.0.170"),
}


def _mooncake_rank_ips(host: str, physical_gpus: list[int]) -> str:
    base = MOONCAKE_IP_BASE[host]
    octets = base.split(".")
    start = int(octets[-1])
    prefix = ".".join(octets[:-1])
    return ",".join(f"{prefix}.{start + gpu}" for gpu in physical_gpus)


def _rdma_device_for_gpu(gpu: int, host=None) -> str:
    if os.environ.get("MN_USE_HIGH_SPEED_IB") == "1":
        return "mlx5_2"
    node = next(n for n in RMB.CONFIG["nodes"] if n["host"] == host)
    return node["gpu_nic"][str(gpu)]


def _mooncake_bandwidth_env() -> str:
    bandwidth = os.environ.get("SGLANG_MOONCAKE_BANDWIDTH_GBPS")
    if not bandwidth:
        return ""
    log_path = os.environ.get(
        "SGLANG_MOONCAKE_THROTTLE_LOG", "/tmp/mooncake_throttle.jsonl"
    )
    lock_path = os.environ.get(
        "SGLANG_MOONCAKE_THROTTLE_LOCK", "/tmp/mooncake_bandwidth.lock"
    )
    return (
        f"SGLANG_MOONCAKE_BANDWIDTH_GBPS={bandwidth} "
        f"SGLANG_MOONCAKE_THROTTLE_LOG={log_path} "
        f"SGLANG_MOONCAKE_THROTTLE_LOCK={lock_path} "
    )


ROUTER_PORT = RMB.ROUTER_PORT

ENERGY_MODEL_DIR_V1 = "UNCONFIGURED"
BISCALE_TTFT_SLO_MS = 2000
BISCALE_TPOT_SLO_MS = 100


def _native_dvfs_flags() -> str:
    """DynamoLLM unified DVFS; honors RMB.TTFT_SLO_MS set by run_benchmark."""
    ttft = getattr(RMB, "TTFT_SLO_MS", 5000)
    tpot = getattr(RMB, "TPOT_SLO_MS", 300)
    return (
        f" --dvfs-enabled "
        f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR_V1} "
        f"--dvfs-ttft-slo-ms {int(ttft)} "
        f"--dvfs-tpot-slo-us {int(tpot * 1000)}"
    )


def _biscale_dvfs_flags() -> str:
    ttft = getattr(RMB, "TTFT_SLO_MS", BISCALE_TTFT_SLO_MS)
    tpot = getattr(RMB, "TPOT_SLO_MS", BISCALE_TPOT_SLO_MS)
    return (
        f" --dvfs-enabled "
        f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR_V1} "
        f"--dvfs-ttft-slo-ms {int(ttft)} "
        f"--dvfs-tpot-slo-us {int(tpot * 1000)} "
        f"--dvfs-policy biscale"
    )


def _lock_dvfs_baseline():
    """1410 MHz baseline for DVFS (avoid hardware-default ~1140 MHz)."""
    RMB.lock_freq_both(ALL_GPUS, RMB.MAX_GPU_FREQ)


DVFS_LOG_DIR = "UNCONFIGURED"

import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("code_4scheme_sweep")

ALL_GPUS = list(range(8))
P_GROUPS = [[0, 1, 2, 3], [4, 5, 6, 7]]
D_GROUPS = [[0, 1], [2, 3], [4, 5], [6, 7]]
# Keep service ports below Linux's default ephemeral range (32768-60999).
# The old 531xx ports could collide with short-lived Mooncake/RPC connections.
PD_PREFILL_PORT_BASE = int(os.environ.get("MN_PD_PREFILL_PORT_BASE", "23100"))
PD_DECODE_PORT_BASE = int(os.environ.get("MN_PD_DECODE_PORT_BASE", "23150"))
PD_BOOTSTRAP_PORT_BASE = int(os.environ.get("MN_PD_BOOTSTRAP_PORT_BASE", "19100"))


def deploy_native_tp(scheme: str, tp: int = 1):
    """Deploy 16xTP1, round-robin router. scheme='sglang' or 'dynamollm'."""
    is_dynamo = scheme == "dynamollm"
    log.info("=" * 60)
    if tp not in (1, 2, 4, 8):
        raise ValueError(f"unsupported native TP: {tp}")
    groups = [ALL_GPUS[i : i + tp] for i in range(0, len(ALL_GPUS), tp)]
    log.info(
        "Deploying %s: %dxTP%d (%d/node), round-robin",
        scheme.upper(),
        16 // tp,
        tp,
        len(groups),
    )
    log.info("=" * 60)

    RMB.cleanup_all()
    time.sleep(8)

    if is_dynamo:
        pass  # DVFS baseline applied after servers are healthy.
    else:
        RMB.lock_freq_both(ALL_GPUS, RMB.MAX_GPU_FREQ)
        log.info("  Locked all GPUs to 1410MHz")

    insts = []
    idx = 0
    for host in (NODE1, NODE2):
        for gpus in groups:
            port = NATIVE_PORT_BASE + idx * 10
            insts.append({"host": host, "gpus": gpus, "port": port, "idx": idx})
            idx += 1

    for inst in insts:
        gpus = inst["gpus"]
        gpu = gpus[0]
        host = inst["host"]
        port = inst["port"]
        i = inst["idx"]
        csv = ",".join(str(g) for g in gpus)
        node_label = "n1" if host == NODE1 else "n2"
        dvfs_flags = _native_dvfs_flags() if is_dynamo else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpu} "
            f"AFD_NVML_DEVICE_INDICES={csv}; "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp {tp} --host {host} --port {port} "
            f"--nccl-port {NCCL_PORT_BASE + i * 10} {FLAGS}{dvfs_flags} "
            f"> {LOG_C}/{scheme}_{node_label}_g{gpu}.log 2>&1 < /dev/null &"
        )
        if host == NODE1:
            RMB.dexec_local(cmd)
        else:
            RMB.dexec_remote(cmd)
        if i % 4 == 3:
            time.sleep(2)

    log.info("  Waiting for all 16 instances...")
    time.sleep(10)
    for inst in insts:
        if not RMB.wait_health(inst["host"], inst["port"], 400):
            log.error("  Instance %s:%d FAILED!", inst["host"], inst["port"])
            return None
    log.info("  All 16 instances healthy!")

    worker_urls = " ".join(f"http://{inst['host']}:{inst['port']}" for inst in insts)
    rc = (
        f"setsid {PYTHON} -m sglang_router.launch_router "
        f"--host {NODE1} --port {ROUTER_PORT} --policy round_robin "
        f"--worker-urls {worker_urls} > {LOG_C}/router.log 2>&1 < /dev/null &"
    )
    RMB.dexec_local(rc)

    if not RMB.wait_health(NODE1, ROUTER_PORT, 60):
        log.error("  Router FAILED!")
        return None

    if is_dynamo:
        _lock_dvfs_baseline()
        log.info("  Locked GPUs to 1410MHz DVFS baseline")

    url = f"http://{NODE1}:{ROUTER_PORT}"
    if not RMB.test_generate(url):
        log.error("  Warmup FAILED!")
        return None
    log.info("  Deploy OK: %s", url)
    return url


def deploy_native_tp1(scheme: str):
    return deploy_native_tp(scheme, tp=1)


def deploy_pd(scheme: str, p_groups=None, d_groups=None):
    """Deploy a 16-GPU PD layout. Defaults to 2P(TP4)+4D(TP2)."""
    p_groups = (
        [(NODE1, g) for g in P_GROUPS]
        if p_groups is None
        else [
            (
                (NODE1 if not isinstance(x, tuple) else x[0]),
                (x if not isinstance(x, tuple) else x[1]),
            )
            for x in p_groups
        ]
    )
    d_groups = (
        [(NODE2, g) for g in D_GROUPS]
        if d_groups is None
        else [
            (
                (NODE2 if not isinstance(x, tuple) else x[0]),
                (x if not isinstance(x, tuple) else x[1]),
            )
            for x in d_groups
        ]
    )
    is_biscale = scheme == "biscale"
    log.info("=" * 60)
    log.info(
        "Deploying %s: %dP(TP%d)+%dD(TP%d)=16GPU",
        scheme.upper(),
        len(p_groups),
        len(p_groups[0][1]),
        len(d_groups),
        len(d_groups[0][1]),
    )
    log.info("=" * 60)

    RMB.cleanup_all()
    time.sleep(8)

    if is_biscale:
        RMB.unlock_freq_both(ALL_GPUS)
        log.info("  Unlocked all GPUs (DVFS biscale)")
    else:
        RMB.lock_freq_both(ALL_GPUS, RMB.MAX_GPU_FREQ)
        log.info("  Locked all GPUs to 1410MHz")

    p_insts = []
    d_insts = []
    fabric_mode = FABRIC.enabled()
    if fabric_mode:
        p_nic_maps, d_nic_maps, fabric_groups = FABRIC.plan_pd_groups(
            len(p_groups), len(p_groups[0][1]), len(d_groups), len(d_groups[0][1])
        )
        for mapping in p_nic_maps + d_nic_maps:
            FABRIC.validate_single_hca_mapping(mapping)
        log.info("  Fabric-affinity groups: %s", fabric_groups)
    else:
        p_nic_maps = d_nic_maps = None
        fabric_groups = []

    for i, (host, gpus) in enumerate(p_groups):
        p_port = PD_PREFILL_PORT_BASE + i * 10
        bs_port = PD_BOOTSTRAP_PORT_BASE + i
        # bond0 is active-backup. Use the HCA backing the host's currently
        # active slave; inactive-slave HCAs cannot carry RoCE reliably.
        nic_map = (
            p_nic_maps[i]
            if fabric_mode
            else {
                str(local_rank): _rdma_device_for_gpu(gpu, host)
                for local_rank, gpu in enumerate(gpus)
            }
        )
        nic = json.dumps(nic_map, separators=(",", ":"))
        csv = ",".join(str(g) for g in gpus)
        dvfs_log = f"{DVFS_LOG_DIR}/{scheme}_p{i}_gpu{gpus[0]}.jsonl"
        dvfs_flags = _biscale_dvfs_flags() if is_biscale else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={host} "
            f"SGLANG_MOONCAKE_HOST_IPS={_mooncake_rank_ips(host, gpus)} "
            f"AFD_DVFS_DECISION_LOG='{dvfs_log}' "
            f"SGLANG_MOONCAKE_REQUIRE_SINGLE_HCA={'1' if fabric_mode else '0'} "
            f"SGLANG_MOONCAKE_FABRIC_ID=pd_p{i} "
            f"{_mooncake_bandwidth_env()}; "
        )
        extra = (
            f"--disaggregation-mode prefill "
            f"--disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {bs_port} "
            f"--disaggregation-ib-device '{nic}' "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp {len(gpus)} --host {host} --port {p_port} "
            f"--nccl-port {NCCL_PORT_BASE + i * 10} {FLAGS}{extra}{dvfs_flags} "
            f"> {LOG_C}/{scheme}_p{i}.log 2>&1 < /dev/null &"
        )
        (RMB.dexec_local(cmd) if host == NODE1 else RMB.dexec_remote(cmd))
        p_insts.append(
            {"host": host, "port": p_port, "bs_port": bs_port, "gpus": gpus, "idx": i}
        )
        log.info("  Started P%d: TP%d GPUs=%s port=%d", i, len(gpus), gpus, p_port)
        time.sleep(3)

    for i, (host, gpus) in enumerate(d_groups):
        d_port = PD_DECODE_PORT_BASE + i * 10
        # node2 bond0 currently uses ens1, which is backed by mlx5_4.
        nic_map = (
            d_nic_maps[i]
            if fabric_mode
            else {
                str(local_rank): _rdma_device_for_gpu(gpu, host)
                for local_rank, gpu in enumerate(gpus)
            }
        )
        nic = json.dumps(nic_map, separators=(",", ":"))
        csv = ",".join(str(g) for g in gpus)
        dvfs_log = f"{DVFS_LOG_DIR}/{scheme}_d{i}_gpu{gpus[0]}.jsonl"
        dvfs_flags = _biscale_dvfs_flags() if is_biscale else ""
        env = (
            f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
            f"AFD_NVML_DEVICE_INDICES={csv} SGLANG_HOST_IP={host} "
            f"SGLANG_MOONCAKE_HOST_IPS={_mooncake_rank_ips(host, gpus)} "
            f"AFD_DVFS_DECISION_LOG='{dvfs_log}' "
            f"SGLANG_MOONCAKE_REQUIRE_SINGLE_HCA={'1' if fabric_mode else '0'} "
            f"SGLANG_MOONCAKE_FABRIC_ID=pd_d{i} "
            f"{_mooncake_bandwidth_env()}; "
        )
        extra = (
            f"--disaggregation-mode decode "
            f"--disaggregation-transfer-backend mooncake "
            f"--disaggregation-bootstrap-port {p_insts[0]['bs_port']} "
            f"--disaggregation-ib-device '{nic}' "
        )
        cmd = (
            f"{env}setsid prlimit --memlock=unlimited:unlimited {PYTHON} -m sglang.launch_server "
            f"--model-path {MODEL} --tp {len(gpus)} --host {host} --port {d_port} "
            f"--nccl-port {NCCL_PORT_BASE + 50 + i * 10} {FLAGS}{extra}{dvfs_flags} "
            f"> {LOG_C}/{scheme}_d{i}.log 2>&1 < /dev/null &"
        )
        (RMB.dexec_local(cmd) if host == NODE1 else RMB.dexec_remote(cmd))
        d_insts.append({"host": host, "port": d_port, "gpus": gpus, "idx": i})
        log.info("  Started D%d: TP%d GPUs=%s port=%d", i, len(gpus), gpus, d_port)
        time.sleep(3)

    log.info("  Waiting for servers...")
    for inst in p_insts:
        if not RMB.wait_health(inst["host"], inst["port"], 400):
            log.error("  P%d FAILED!", inst["idx"])
            return None
    for inst in d_insts:
        if not RMB.wait_health(inst["host"], inst["port"], 400):
            log.error("  D%d FAILED!", inst["idx"])
            return None
    log.info("  All P/D servers healthy!")

    if fabric_mode:
        sub_urls = []
        for group_index, group in enumerate(fabric_groups):
            sub_port = ROUTER_PORT + 100 + group_index
            p_inst = p_insts[group.prefill_indices[0]]
            rc_parts = [
                f"setsid {PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation --mini-lb",
                f"--host {NODE1} --port {sub_port}",
                f"--prefill http://{p_inst['host']}:{p_inst['port']} {p_inst['bs_port']}",
            ]
            for d_index in group.decode_indices:
                d_inst = d_insts[d_index]
                rc_parts.append(f"--decode http://{d_inst['host']}:{d_inst['port']}")
            rc = (
                " ".join(rc_parts)
                + f" > {LOG_C}/router_fabric_{group_index}.log 2>&1 < /dev/null &"
            )
            RMB.dexec_local(rc)
            if not RMB.wait_health(NODE1, sub_port, 60):
                log.error("  Fabric router %d FAILED!", group_index)
                return None
            sub_urls.append(f"http://{NODE1}:{sub_port}")
        worker_urls = " ".join(sub_urls)
        rc = (
            f"setsid {PYTHON} -m sglang_router.launch_router --host {NODE1} "
            f"--port {ROUTER_PORT} --policy round_robin --worker-urls {worker_urls} "
            f"> {LOG_C}/router.log 2>&1 < /dev/null &"
        )
        RMB.dexec_local(rc)
    else:
        rc_parts = [
            f"setsid {PYTHON} -m sglang_router.launch_router",
            "--pd-disaggregation",
            f"--host {NODE1} --port {ROUTER_PORT}",
        ]
        for inst in p_insts:
            rc_parts.append(
                f"--prefill http://{inst['host']}:{inst['port']} {inst['bs_port']}"
            )
        for inst in d_insts:
            rc_parts.append(f"--decode http://{inst['host']}:{inst['port']}")
        rc = " ".join(rc_parts) + f" > {LOG_C}/router.log 2>&1 < /dev/null &"
        RMB.dexec_local(rc)

    if not RMB.wait_health(NODE1, ROUTER_PORT, 60):
        log.error("  Router FAILED!")
        return None

    url = f"http://{NODE1}:{ROUTER_PORT}"
    if not RMB.test_generate(url):
        log.error("  Warmup FAILED!")
        return None
    log.info("  Deploy OK: %s", url)
    return url
