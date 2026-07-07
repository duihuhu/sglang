"""Deploy Tier1-style 1P+kD PDAF layout (hetero TP + per-pool frequencies).

Prefill on node1; decode replicas split across node1 (after P) and node2.
Router: k_D sub MiniLB routers + top-level round-robin.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import run_macro_benchmark as RMB

log = logging.getLogger("deploy_tier1")

BS_PORT = 49999
ROUTER_PORT = 43000
PA_PORT, PF_PORT = 43010, 43011
SUB_ROUTER_BASE = 43050
DECODE_PORT_BASE = 43020  # D{i}A = base + i*10, D{i}F = base + i*10 + 1
GPUS_PER_NODE = 8


@dataclass
class Tier1Layout:
    name: str
    k_d: int
    tp_pa: int
    tp_pf: int
    tp_da: int
    tp_df: int
    f_pa: int
    f_pf: int
    f_da: int
    f_df: int
    tier: bool = False  # compositional DVFS on top of initial locks

    @property
    def k_p(self) -> int:
        return 1


@dataclass
class _DecodePlacement:
    host: str
    attn: list[int]
    ffn: list[int]
    idx: int


def _tp_groups(tp: int) -> tuple[list[int], list[int]]:
    if tp == 1:
        return [0], [1]
    if tp == 2:
        return [0, 2], [1, 3]
    if tp == 4:
        return [0, 2, 4, 6], [1, 3, 5, 7]
    raise ValueError(f"unsupported tp={tp}")


def _afd_env(role, attn_gpus, ffn_gpus, ucx_port, sched_port):
    cvd = "0,1,2,3,4,5,6,7"
    ipc_sync = "ipc_event"
    base = (
        "export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
        "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
        "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
        f"AFD_IPC_SYNC_MODE={ipc_sync} "
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
        f"CUDA_VISIBLE_DEVICES={cvd} "
    )
    if role == "ffn":
        nvml = ",".join(str(g) for g in ffn_gpus)
        return (
            base
            + f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
            f"AFD_IPC_PEER_OFFSET=-1 "
            f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={ffn_gpus[0]};"
        )
    nvml = ",".join(str(g) for g in attn_gpus)
    return (
        base
        + f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
        f"AFD_IPC_PEER_OFFSET=1 "
        f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={attn_gpus[0]} "
        f"AFD_UCX_FFN_HOST=127.0.0.1;"
    )


def _afd_common(tp, tier: bool):
    flags = (
        f"--model-path {RMB.MODEL} --tp {tp} --gpu-id-step 2 "
        "--afd-comm-backend ipc_cpp --afd-micro-batch 1 "
        "--mem-fraction-static 0.85 --max-running-requests 512 "
        "--skip-server-warmup --watchdog-timeout 600 "
        "--disable-cuda-graph --disable-piecewise-cuda-graph "
        "--afd-disagg-interleave-poll --disable-radix-cache "
        "--num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-bootstrap-port {BS_PORT} "
        f"--disaggregation-ib-device {RMB.IB_JSON_FILE} --enable-metrics"
    )
    if tier:
        flags += (
            " --afd-dvfs-enabled "
            f"--afd-energy-model-dir {RMB.ENERGY_MODEL_DIR_V1} "
            f"--afd-ttft-slo-ms {int(RMB.TTFT_SLO_MS)} "
            f"--afd-tpot-slo-us {int(RMB.TPOT_SLO_MS * 1000)} "
            "--afd-dvfs-idle-lock --afd-dvfs-decode-compositional"
        )
    return flags


def _lock_node_gpus(host: str, gpu_freq: dict[int, int]):
    import shlex
    import subprocess
    for gpu, freq in sorted(gpu_freq.items()):
        cmd = f"nvidia-smi -i {gpu} --lock-gpu-clocks={freq},{freq}"
        if host == RMB.NODE1_IP:
            RMB.dexec_local(cmd)
        else:
            subprocess.run(
                RMB._ssh(host, f"docker exec {RMB.CONTAINER} bash -lc {shlex.quote(cmd)}"),
                check=False,
            )


def _launch(host, role, mode, attn, ffn, tp, port, base_gpu, cf, tag, ucx, sched):
    persp = "ffn" if role == "f" else "attn"
    env = _afd_env(persp, attn, ffn, ucx, sched)
    cmd = (
        f"{env} {RMB.PYTHON} -m sglang.launch_server --host {host} "
        f"--port {port} --afd-perspective {persp} "
        f"--disaggregation-mode {mode} --base-gpu-id {base_gpu} {cf}"
    )
    RMB._launch(host, cmd, tag)


def _find_decode_slot(occupied: set[int], tp_d: int) -> tuple[list[int], list[int]] | None:
    """Find the lowest GPU slot on a node for one DA/DF chain."""
    if tp_d == 1:
        for off in range(0, GPUS_PER_NODE, 2):
            gpus = {off, off + 1}
            if not gpus & occupied:
                return [off], [off + 1]
        return None
    if tp_d == 2:
        for off in range(0, GPUS_PER_NODE, 4):
            gpus = {off, off + 1, off + 2, off + 3}
            if not gpus & occupied:
                return [off, off + 2], [off + 1, off + 3]
        return None
    if tp_d == 4:
        gpus = set(range(GPUS_PER_NODE))
        if not gpus & occupied:
            return [0, 2, 4, 6], [1, 3, 5, 7]
        return None
    raise ValueError(f"unsupported decode tp={tp_d}")


def _plan_decode_placements(
    k_d: int, tp_d: int, p_attn: list[int], p_ffn: list[int],
) -> list[_DecodePlacement]:
    """Pack k_D decode chains: fill node1 after P, then node2."""
    node_occ: dict[str, set[int]] = {
        RMB.NODE1_IP: set(p_attn) | set(p_ffn),
        RMB.NODE2_IP: set(),
    }
    placements: list[_DecodePlacement] = []
    for idx in range(k_d):
        placed = False
        for host in (RMB.NODE1_IP, RMB.NODE2_IP):
            slot = _find_decode_slot(node_occ[host], tp_d)
            if slot is None:
                continue
            attn, ffn = slot
            node_occ[host].update(attn)
            node_occ[host].update(ffn)
            placements.append(_DecodePlacement(host, attn, ffn, idx))
            placed = True
            break
        if not placed:
            raise ValueError(
                f"cannot place decode replica {idx + 1}/{k_d} "
                f"(tp_d={tp_d}); need {k_d * 2 * tp_d} decode GPUs cluster-wide"
            )
    n1 = sum(1 for p in placements if p.host == RMB.NODE1_IP)
    n2 = k_d - n1
    log.info("Decode placement: %d on node1 (co-located), %d on node2", n1, n2)
    return placements


def deploy_tier1_layout(cfg: Tier1Layout) -> Optional[str]:
    """Start 1P + k_D layout. Returns router URL or None."""
    log.info(
        "Deploy Tier1 %s: 1P(tp %d/%d @%d/%d) + %dD(tp %d/%d @%d/%d) tier=%s",
        cfg.name, cfg.tp_pa, cfg.tp_pf, cfg.f_pa, cfg.f_pf,
        cfg.k_d, cfg.tp_da, cfg.tp_df, cfg.f_da, cfg.f_df, cfg.tier,
    )

    ib_map = {str(g): RMB.GPU_NIC[g] for g in range(GPUS_PER_NODE)}
    RMB.write_ib_json(ib_map)

    p_attn, p_ffn = _tp_groups(cfg.tp_pa)
    if _tp_groups(cfg.tp_pf) != (p_attn, p_ffn) and cfg.tp_pf != cfg.tp_pa:
        log.warning("hetero TP: using PA gpu layout for PF tp=%d", cfg.tp_pf)

    p_cf = _afd_common(cfg.tp_pa, cfg.tier)
    d_cf = _afd_common(cfg.tp_da, cfg.tier)

    dec_placements = _plan_decode_placements(cfg.k_d, cfg.tp_da, p_attn, p_ffn)

    # Lock per-GPU frequencies before launch
    freq_by_host: dict[str, dict[int, int]] = {RMB.NODE1_IP: {}, RMB.NODE2_IP: {}}
    for g in p_attn:
        freq_by_host[RMB.NODE1_IP][g] = cfg.f_pa
    for g in p_ffn:
        freq_by_host[RMB.NODE1_IP][g] = cfg.f_pf
    for pl in dec_placements:
        for g in pl.attn:
            freq_by_host[pl.host][g] = cfg.f_da
        for g in pl.ffn:
            freq_by_host[pl.host][g] = cfg.f_df
    for host, fmap in freq_by_host.items():
        if fmap:
            _lock_node_gpus(host, fmap)

    # Prefill on node1
    _launch(RMB.NODE1_IP, "f", "prefill", p_attn, p_ffn, cfg.tp_pf, PF_PORT, p_ffn[0],
            p_cf, "tier1_pf", 28200, 68400)
    time.sleep(5)
    _launch(RMB.NODE1_IP, "a", "prefill", p_attn, p_ffn, cfg.tp_pa, PA_PORT, p_attn[0],
            p_cf, "tier1_pa", 28200, 68400)

    decode_endpoints: list[tuple[str, int, str]] = []
    for pl in dec_placements:
        da_port = DECODE_PORT_BASE + pl.idx * 10
        df_port = DECODE_PORT_BASE + pl.idx * 10 + 1
        ucx = 28300 + pl.idx * 100
        sched = 68500 + pl.idx * 100
        time.sleep(5)
        _launch(pl.host, "f", "decode", pl.attn, pl.ffn, cfg.tp_df, df_port, pl.ffn[0],
                d_cf, f"tier1_d{pl.idx}f", ucx, sched)
        time.sleep(5)
        _launch(pl.host, "a", "decode", pl.attn, pl.ffn, cfg.tp_da, da_port, pl.attn[0],
                d_cf, f"tier1_d{pl.idx}a", ucx, sched)
        decode_endpoints.append((pl.host, da_port, f"tier1_d{pl.idx}a"))

    checks = [
        (RMB.NODE1_IP, PF_PORT, True, "PF"),
        (RMB.NODE1_IP, PA_PORT, False, "PA"),
    ]
    for host, da_port, name in decode_endpoints:
        checks.append((host, da_port, False, name))
    for host, port, mi, name in checks:
        if not RMB.wait_health(host, port, 300, check_model_info=mi):
            log.error("%s health failed", name)
            return None

    sub_ports = []
    for host, da_port, _ in decode_endpoints:
        sub_port = SUB_ROUTER_BASE + len(sub_ports)
        rc = (
            f"setsid {RMB.PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
            f"--prefill http://{RMB.NODE1_IP}:{PA_PORT} "
            f"--decode http://{host}:{da_port} "
            f"--host {RMB.NODE1_IP} --port {sub_port} "
            f"> {RMB.LOG_C}/tier1_sub_{len(sub_ports)}.log 2>&1 < /dev/null &"
        )
        RMB.dexec_local(rc)
        if not RMB.wait_health(RMB.NODE1_IP, sub_port, 60):
            log.error("sub-router %d failed", len(sub_ports))
            return None
        sub_ports.append(sub_port)

    workers = " ".join(f"http://{RMB.NODE1_IP}:{p}" for p in sub_ports)
    rc = (
        f"setsid {RMB.PYTHON} -m sglang_router.launch_router "
        f"--host {RMB.NODE1_IP} --port {ROUTER_PORT} --policy round_robin "
        f"--worker-urls {workers} > {RMB.LOG_C}/tier1_router.log 2>&1 < /dev/null &"
    )
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, ROUTER_PORT, 60):
        return None
    return f"http://{RMB.NODE1_IP}:{ROUTER_PORT}"
