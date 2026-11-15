#!/usr/bin/env python3
"""Benchmark Tier1-optimized AFD configs on QPS=16 code, node3+node4.

Canonical heterogeneous AF-disaggregation deployment and benchmark runtime.
Adds multi-pair (k_P > 1) support with custom (tp, freq) per pool.
"""
from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("aflex")

from . import fabric_affinity as FABRIC
from . import runtime as RMB

GPUS_PER_NODE = 8
PORT_OFFSET = int(os.environ.get("AFLEX_PORT_OFFSET", "0"))
ROUTER_PORT = 44000 + PORT_OFFSET
SUB_ROUTER_BASE = 45000 + PORT_OFFSET
DECODE_PORT_BASE = 43020 + PORT_OFFSET
PREFILL_PORT_BASE = 43200 + PORT_OFFSET
NCCL_PORT_BASE = 37300 + PORT_OFFSET
QPS = 16
DATASET = "code"
# Optional per-deployment flags; callers must reset after use.
SERVER_EXTRA_FLAGS = ""


# ── Tier1 configs ──
@dataclass(frozen=True)
class PrefillSpec:
    tp_pa: int
    tp_pf: int
    f_pa: int
    f_pf: int


@dataclass
class Tier1TestConfig:
    name: str
    k_p: int
    k_d: int
    tp_pa: int
    tp_pf: int
    tp_da: int
    tp_df: int
    f_pa: int
    f_pf: int
    f_da: int
    f_df: int
    tier: bool = True
    prefill_specs: Optional[tuple[PrefillSpec, ...]] = None

    def resolved_prefill_specs(self) -> list[PrefillSpec]:
        if self.prefill_specs:
            return list(self.prefill_specs)
        base = PrefillSpec(self.tp_pa, self.tp_pf, self.f_pa, self.f_pf)
        return [base] * self.k_p

    def effective_k_p(self) -> int:
        return len(self.resolved_prefill_specs())

    def total_gpu(self) -> int:
        p_gpus = sum(s.tp_pa + s.tp_pf for s in self.resolved_prefill_specs())
        return p_gpus + self.k_d * (self.tp_da + self.tp_df)


def alloc_af_gpus_contiguous(tp_f, tp_a, base_gpus):
    """Contiguous allocation: FFN first, then Attn (like run_pdaf_4pa2pf_test.py)."""
    ffn = base_gpus[:tp_f]
    attn = base_gpus[tp_f : tp_f + tp_a]
    return attn, ffn


def plan_allocation(cfg):
    node1_pool = os.environ.get("AFLEX_NODE1_GPU_POOL")
    node2_pool = os.environ.get("AFLEX_NODE2_GPU_POOL")
    pool_spec = os.environ.get("AFLEX_GPU_POOL")
    target_host = os.environ.get("AFLEX_SINGLE_HOST")
    if node1_pool is not None or node2_pool is not None:
        n1 = [int(value) for value in (node1_pool or "").split(",") if value.strip()]
        n2 = [int(value) for value in (node2_pool or "").split(",") if value.strip()]
        if len(n1) + len(n2) < cfg.total_gpu():
            raise RuntimeError(
                f"per-node GPU pools have {len(n1) + len(n2)} GPUs, need {cfg.total_gpu()}"
            )
    elif pool_spec:
        pool = [int(value) for value in pool_spec.split(",") if value.strip()]
        if len(pool) < cfg.total_gpu():
            raise RuntimeError(
                f"AFLEX_GPU_POOL has {len(pool)} GPUs, need {cfg.total_gpu()}"
            )
        if target_host == RMB.NODE2_IP:
            n1, n2 = [], pool
        elif target_host == RMB.NODE1_IP:
            n1, n2 = pool, []
        else:
            raise RuntimeError(
                f"AFLEX_SINGLE_HOST must be node1 or node2: {target_host}"
            )
    else:
        n1 = list(range(8))
        n2 = list(range(8))
    # For single-node 8-GPU sweeps, allow reserving node1 for unrelated work.
    prefer_node2 = os.environ.get("AFLEX_PREFER_NODE2", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    placement_order = (
        (("n4", n2), ("n3", n1)) if prefer_node2 else (("n3", n1), ("n4", n2))
    )
    # Decode instances (k_d replicas, starting on node3)
    d_instances = []
    for _ in range(cfg.k_d):
        need = cfg.tp_da + cfg.tp_df
        placed = False
        placement_order = (
            (("n4", n2), ("n3", n1)) if prefer_node2 else (("n3", n1), ("n4", n2))
        )
        for label, free in placement_order:
            if need > len(free):
                continue
            d_a, d_f = alloc_af_gpus_contiguous(cfg.tp_df, cfg.tp_da, free[:need])
            d_instances.append((label, d_a, d_f))
            if label == "n3":
                n1 = [g for g in n1 if g not in set(d_a) | set(d_f)]
            else:
                n2 = [g for g in n2 if g not in set(d_a) | set(d_f)]
            placed = True
            break
        if not placed:
            raise RuntimeError(f"Cannot place D instance #{len(d_instances)+1}")
    # Prefill pairs (optionally heterogeneous TP/freq per instance)
    # Benchmark-only split placement keeps Decode on node1 and Prefill on node2
    # so one request exercises both intra-node A/F IPC and cross-node PD RDMA.
    split_pd_nodes = os.environ.get("AFLEX_SPLIT_PD_NODES", "0").lower() in (
        "1",
        "true",
        "yes",
    )
    pspecs = cfg.resolved_prefill_specs()
    pairs = []
    for spec in pspecs:
        need = spec.tp_pa + spec.tp_pf
        placed = False
        placement_order = (
            (("n4", n2),)
            if split_pd_nodes
            else (("n4", n2), ("n3", n1)) if prefer_node2 else (("n3", n1), ("n4", n2))
        )
        for label, free in placement_order:
            if need > len(free):
                continue
            a, f = alloc_af_gpus_contiguous(spec.tp_pf, spec.tp_pa, free[:need])
            pairs.append((label, a, f, spec))
            if label == "n3":
                n1 = [g for g in n1 if g not in set(a) | set(f)]
            else:
                n2 = [g for g in n2 if g not in set(a) | set(f)]
            placed = True
            break
        if not placed:
            raise RuntimeError(
                f"Cannot place P pair #{len(pairs)+1} (need {need} GPUs)"
            )
    if len(pairs) != len(pspecs):
        raise RuntimeError(
            f"Cannot place all P pairs: planned {len(pairs)}, expected {len(pspecs)}"
        )
    # Freq map
    fmap = {RMB.NODE1_IP: {}, RMB.NODE2_IP: {}}
    for label, a, f, spec in pairs:
        host = RMB.NODE1_IP if label == "n3" else RMB.NODE2_IP
        for g in a:
            fmap[host][g] = spec.f_pa
        for g in f:
            fmap[host][g] = spec.f_pf
    for label, d_a, d_f in d_instances:
        host = RMB.NODE1_IP if label == "n3" else RMB.NODE2_IP
        for g in d_a:
            fmap[host][g] = cfg.f_da
        for g in d_f:
            fmap[host][g] = cfg.f_df
    # Use first D instance as primary (for backward compat)
    primary_d_label, primary_d_a, primary_d_f = d_instances[0]
    primary_d_host = RMB.NODE1_IP if primary_d_label == "n3" else RMB.NODE2_IP
    return {
        "p_pairs": [
            (RMB.NODE1_IP if l == "n3" else RMB.NODE2_IP, a, f, spec)
            for l, a, f, spec in pairs
        ],
        "decode": {"host": primary_d_host, "attn": primary_d_a, "ffn": primary_d_f},
        "decode_instances": [
            {"host": RMB.NODE1_IP if l == "n3" else RMB.NODE2_IP, "attn": a, "ffn": f}
            for l, a, f in d_instances
        ],
        "freq_map": fmap,
        "total_gpu": cfg.total_gpu(),
        "energy_hosts": {
            host: sorted(freq_map.keys()) for host, freq_map in fmap.items()
        },
    }


# ── Deploy with heterogeneous TP support (no gpu-id-step, use AFD_IPC_PEER_DEVICE) ──
def _mooncake_rank_ips(host: str, physical_gpus: list[int]) -> str:
    base = (
        os.environ.get("MN_NODE1_MOONCAKE_IP_BASE", "10.0.0.70")
        if host == RMB.NODE1_IP
        else os.environ.get("MN_NODE2_MOONCAKE_IP_BASE", "10.0.0.170")
    )
    parts = base.split(".")
    prefix, start = ".".join(parts[:-1]), int(parts[-1])
    return ",".join(f"{prefix}.{start + gpu}" for gpu in physical_gpus)


def _build_env_str(
    host,
    cvd_gpus,
    ucx_port,
    sched_port,
    peer_device,
    nvml_gpus,
    is_attn=False,
    active_freq=None,
):
    """Build env prefix, pinning the host IP to avoid incorrect multi-NIC auto-detection."""
    cvd = ",".join(str(g) for g in cvd_gpus)
    nvml = ",".join(str(g) for g in nvml_gpus)
    bandwidth = os.environ.get("SGLANG_MOONCAKE_BANDWIDTH_GBPS")
    bandwidth_env = ""
    if bandwidth:
        log_path = os.environ.get(
            "SGLANG_MOONCAKE_THROTTLE_LOG", "/tmp/mooncake_throttle.jsonl"
        )
        lock_path = os.environ.get(
            "SGLANG_MOONCAKE_THROTTLE_LOCK", "/tmp/mooncake_bandwidth.lock"
        )
        bandwidth_env = (
            f"SGLANG_MOONCAKE_BANDWIDTH_GBPS={bandwidth} "
            f"SGLANG_MOONCAKE_THROTTLE_LOG={log_path} "
            f"SGLANG_MOONCAKE_THROTTLE_LOCK={lock_path} "
        )
    fabric_env = ""
    if FABRIC.enabled():
        fabric_env = "SGLANG_MOONCAKE_REQUIRE_SINGLE_HCA=1 "
    env = (
        f"export CUDA_VISIBLE_DEVICES={cvd} SGLANG_HOST_IP={host} "
        f"{fabric_env}"
        f"SGLANG_MOONCAKE_HOST_IPS={_mooncake_rank_ips(host, nvml_gpus)} "
        f"{bandwidth_env}"
        "SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
        "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
        "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
        "AFD_IPC_SYNC_MODE=ipc_event "
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
        f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
        f"AFD_IPC_PEER_DEVICE={peer_device} "
        f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={nvml_gpus[0]} "
    )
    for key in (
        "AFD_TIMING",
        "AFD_DETAILED_TIMING",
        "AFD_DVFS_SWITCH_LOG",
        "SGLANG_MOONCAKE_AUDIT_LOG",
        "SGLANG_MOONCAKE_AUDIT_LOCK",
    ):
        value = os.environ.get(key)
        if value:
            env += f"{key}={value} "
    nvlink_links = os.environ.get("AFD_NVLINK_EQUIVALENT_LINKS")
    if nvlink_links:
        env += (
            f"AFD_NVLINK_EQUIVALENT_LINKS={nvlink_links} "
            f"AFD_NVLINK_PER_LINK_GBS={os.environ.get('AFD_NVLINK_PER_LINK_GBS', '25')} "
        )
        bandwidth_override = os.environ.get("AFD_NVLINK_BANDWIDTH_GBS")
        if bandwidth_override:
            env += f"AFD_NVLINK_BANDWIDTH_GBS={bandwidth_override} "
        audit_path = os.environ.get("AFD_NVLINK_AUDIT_LOG")
        if audit_path:
            env += f"AFD_NVLINK_AUDIT_LOG={audit_path} "
    sample_dir = os.environ.get("AFD_DECODE_BATCH_SAMPLE_DIR")
    if sample_dir:
        sample_role = "attn" if is_attn else "ffn"
        sample_gpu = "_".join(str(gpu) for gpu in nvml_gpus)
        sample_path = f"{sample_dir}/decode_{sample_role}_gpus_{sample_gpu}.csv"
        env += f"AFD_DECODE_BATCH_SAMPLE_PATH={sample_path} "
    dvfs_log_tmpl = os.environ.get("AFD_DVFS_DECISION_LOG")
    if dvfs_log_tmpl:
        env += f"AFD_DVFS_DECISION_LOG={dvfs_log_tmpl} "
    if active_freq is not None:
        env += f"AFD_FIXED_ACTIVE_FREQ={int(active_freq)} "
    if is_attn:
        env += "AFD_UCX_FFN_HOST=127.0.0.1 "
    env += ";"
    return env


def _build_afd_flags(tp, tp_pa, tp_pf, tier, is_decode=False, ib_device=None):
    """Build server flags for heterogeneous TP (no --gpu-id-step)."""
    mem_frac = "0.88" if is_decode else "0.75"
    ib_device = ib_device or RMB.IB_JSON_FILE
    flags = (
        f"--model-path {RMB.MODEL} --tp {tp} "
        "--afd-comm-backend ipc_cpp --afd-micro-batch 1 "
        f"--mem-fraction-static {mem_frac} --max-running-requests 64 "
        "--skip-server-warmup --watchdog-timeout 600 "
        "--disable-cuda-graph --disable-piecewise-cuda-graph "
        "--afd-disagg-interleave-poll --disable-radix-cache "
        "--num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-ib-device '{ib_device}' --enable-metrics "
        f"--afd-attn-tp {tp_pa} --afd-ffn-tp {tp_pf} "
        f"{SERVER_EXTRA_FLAGS} "
    )
    runtime_dvfs_enabled = tier and os.environ.get(
        "AFLEX_DISABLE_RUNTIME_DVFS", "0"
    ) not in ("1", "true", "yes")
    if os.environ.get("AFLEX_FREEZE_ACTIVE_FREQ", "0").lower() in ("1", "true", "yes"):
        flags += "--afd-dvfs-freeze-active-freq "
    if runtime_dvfs_enabled:
        flags += (
            "--afd-dvfs-enabled "
            f"--afd-energy-model-dir {RMB.ENERGY_MODEL_DIR_V1} "
            f"--afd-ttft-slo-ms {int(RMB.TTFT_SLO_MS)} "
            f"--afd-tpot-slo-us {int(RMB.TPOT_SLO_MS * 1000)} "
            "--afd-dvfs-decode-compositional --afd-dvfs-idle-lock "
        )
    return flags


def deploy(cfg):
    alloc = plan_allocation(cfg)
    p_pairs = alloc["p_pairs"]
    d_info = alloc["decode"]

    log.info("=" * 60)
    log.info(
        "DEPLOY %s: %d GPU (hetero TP: PA=%d PF=%d DA=%d DF=%d)",
        cfg.name,
        alloc["total_gpu"],
        cfg.tp_pa,
        cfg.tp_pf,
        cfg.tp_da,
        cfg.tp_df,
    )
    for i, (host, a, f, spec) in enumerate(p_pairs):
        log.info(
            "  P%d: %s attn=%s ffn=%s (TP PA=%d PF=%d)",
            i,
            host.split(".")[-1],
            a,
            f,
            spec.tp_pa,
            spec.tp_pf,
        )
    for i, d_inst in enumerate(alloc.get("decode_instances", [alloc["decode"]])):
        log.info(
            "  D%d: %s attn=%s ffn=%s",
            i,
            d_inst["host"].split(".")[-1],
            d_inst["attn"],
            d_inst["ffn"],
        )

    # Lock frequencies
    for host, fmap in alloc["freq_map"].items():
        if fmap and cfg.tier:
            log.info(
                "  Locking %s: %s",
                host,
                {g: f"{fm}MHz" for g, fm in sorted(fmap.items())},
            )
            RMB.lock_freq_map_on_host(host, fmap)

    fabric_mode = FABRIC.enabled()
    d_instances = alloc.get("decode_instances", [alloc["decode"]])
    if fabric_mode:
        p_fabrics, d_fabrics, p_to_d = FABRIC.plan_instance_affinity(
            len(p_pairs), len(d_instances)
        )
        default_hca = FABRIC.hcas()[0]
        maps = {
            RMB.NODE1_IP: {str(g): default_hca for g in range(GPUS_PER_NODE)},
            RMB.NODE2_IP: {str(g): default_hca for g in range(GPUS_PER_NODE)},
        }
        for (host, attn, ffn, _spec), fabric in zip(p_pairs, p_fabrics):
            for gpu in attn + ffn:
                maps[host][str(gpu)] = fabric
        for d_inst, fabric in zip(d_instances, d_fabrics):
            for gpu in d_inst["attn"] + d_inst["ffn"]:
                maps[d_inst["host"]][str(gpu)] = fabric
        for mapping in maps.values():
            FABRIC.validate_single_hca_mapping(mapping)
        ib_map_n1, ib_map_n2 = maps[RMB.NODE1_IP], maps[RMB.NODE2_IP]
        log.info("  Fabric affinity: P=%s D=%s P->D=%s", p_fabrics, d_fabrics, p_to_d)
    elif os.environ.get("MN_USE_HIGH_SPEED_IB") == "1":
        ib_map_n1 = {str(g): "mlx5_2" for g in range(GPUS_PER_NODE)}
        ib_map_n2 = dict(ib_map_n1)
        p_to_d = []
    else:
        ib_map_n1 = {str(g): "mlx5_0" for g in range(GPUS_PER_NODE)}
        ib_map_n2 = {str(g): "mlx5_4" for g in range(GPUS_PER_NODE)}
        p_to_d = []
    RMB.write_ib_json(ib_map_n1, node2_mapping=ib_map_n2)

    # Pre-warm JIT compilation on both nodes (avoid file lock contention)
    import shlex

    jit_code = "from sglang.srt.layers.afd_ipc_cpp import get_module; get_module()"
    jit_warmup_cmd = f"CUDA_VISIBLE_DEVICES=0 {RMB.PYTHON} -c {shlex.quote(jit_code)}"
    hosts_used = set(h for h, _, _, _ in p_pairs)
    hosts_used.update(d["host"] for d in alloc["decode_instances"])
    for host in hosts_used:
        log.info("  Pre-warming JIT on %s ...", host.split(".")[-1])
        if host == RMB.NODE1_IP:
            RMB.dexec_local(jit_warmup_cmd)
        else:
            RMB.dexec_on_host(host, jit_warmup_cmd)
    log.info("  JIT pre-warm done")

    # Launch prefill pairs
    prefill_eps = []
    nccl_idx = 0
    for i, (host, attn_gpus, ffn_gpus, spec) in enumerate(p_pairs):
        bootstrap_port = BOOTSTRAP_PORT_BASE + i
        RMB.ensure_ports_free(host, [bootstrap_port])
        pa_port = PREFILL_PORT_BASE + i * 10
        pf_port = PREFILL_PORT_BASE + i * 10 + 2
        ucx = UCX_PORT_BASE + i * 100
        sched = SCHED_PORT_BASE + i * 100
        # CVD = all GPUs in this pair (ffn + attn, contiguous)
        cvd_gpus = ffn_gpus + attn_gpus
        pair_fabric = p_fabrics[i] if fabric_mode else None
        pair_ib = (
            json.dumps(
                {
                    str(rank): pair_fabric
                    for rank in range(max(spec.tp_pa, spec.tp_pf) + 1)
                },
                separators=(",", ":"),
            )
            if pair_fabric
            else None
        )
        # PF: base_gpu_id=0 (first in CVD), peer_device = len(ffn_gpus) (attn start)
        pf_peer = len(ffn_gpus)  # index within CVD where attn starts
        pf_env = _build_env_str(
            host,
            cvd_gpus,
            ucx,
            sched,
            pf_peer,
            nvml_gpus=ffn_gpus,
            is_attn=False,
            active_freq=spec.f_pf,
        )
        pf_flags = _build_afd_flags(
            spec.tp_pf, spec.tp_pa, spec.tp_pf, cfg.tier, ib_device=pair_ib
        )
        pf_nccl_port = NCCL_PORT_BASE + nccl_idx * 10
        pf_cmd = (
            f"{pf_env} {RMB.PYTHON} -m sglang.launch_server "
            f"--host {host} --port {pf_port} --afd-perspective ffn "
            f"--disaggregation-mode prefill --base-gpu-id 0 "
            f"--nccl-port {pf_nccl_port} {pf_flags}"
        )
        nccl_idx += 1
        RMB.ensure_ports_free(host, [pf_port, pf_nccl_port])
        RMB._launch(host, pf_cmd, f"{cfg.name}_p{i}f")
        time.sleep(5)
        # PA: base_gpu_id=len(ffn_gpus) (attn start in CVD), peer_device=0 (ffn start)
        pa_peer = 0
        pa_env = _build_env_str(
            host,
            cvd_gpus,
            ucx,
            sched,
            pa_peer,
            nvml_gpus=attn_gpus,
            is_attn=True,
            active_freq=spec.f_pa,
        )
        pa_flags = _build_afd_flags(
            spec.tp_pa, spec.tp_pa, spec.tp_pf, cfg.tier, ib_device=pair_ib
        )
        pa_nccl_port = NCCL_PORT_BASE + nccl_idx * 10
        pa_cmd = (
            f"{pa_env} {RMB.PYTHON} -m sglang.launch_server "
            f"--host {host} --port {pa_port} --afd-perspective attn "
            f"--disaggregation-mode prefill --disaggregation-bootstrap-port {bootstrap_port} --base-gpu-id {len(ffn_gpus)} "
            f"--nccl-port {pa_nccl_port} {pa_flags}"
        )
        nccl_idx += 1
        RMB.ensure_ports_free(host, [pa_port, pa_nccl_port])
        RMB._launch(host, pa_cmd, f"{cfg.name}_p{i}a")
        time.sleep(2)
        prefill_eps.append((host, pa_port))

    # Launch decode instance(s)
    decode_eps = []
    for di, d_inst in enumerate(d_instances):
        d_ucx = UCX_PORT_BASE + (cfg.effective_k_p() + di) * 100
        d_sched = SCHED_PORT_BASE + (cfg.effective_k_p() + di) * 100
        da_port = DECODE_PORT_BASE + di * 20
        df_port = DECODE_PORT_BASE + di * 20 + 2
        d_cvd = d_inst["ffn"] + d_inst["attn"]
        decode_fabric = d_fabrics[di] if fabric_mode else None
        decode_ib = (
            json.dumps(
                {
                    str(rank): decode_fabric
                    for rank in range(max(cfg.tp_da, cfg.tp_df) + 1)
                },
                separators=(",", ":"),
            )
            if decode_fabric
            else None
        )
        # DF
        df_peer = len(d_inst["ffn"])
        df_env = _build_env_str(
            d_inst["host"],
            d_cvd,
            d_ucx,
            d_sched,
            df_peer,
            nvml_gpus=d_inst["ffn"],
            is_attn=False,
            active_freq=cfg.f_df,
        )
        df_flags = _build_afd_flags(
            cfg.tp_df,
            cfg.tp_da,
            cfg.tp_df,
            cfg.tier,
            is_decode=True,
            ib_device=decode_ib,
        )
        df_nccl_port = NCCL_PORT_BASE + nccl_idx * 10
        df_cmd = (
            f"{df_env} {RMB.PYTHON} -m sglang.launch_server "
            f"--host {d_inst['host']} --port {df_port} --afd-perspective ffn "
            f"--disaggregation-mode decode --base-gpu-id 0 "
            f"--nccl-port {df_nccl_port} {df_flags}"
        )
        nccl_idx += 1
        RMB.ensure_ports_free(d_inst["host"], [df_port, df_nccl_port])
        RMB._launch(d_inst["host"], df_cmd, f"{cfg.name}_d{di}f")
        time.sleep(5)
        # DA
        da_peer = 0
        da_env = _build_env_str(
            d_inst["host"],
            d_cvd,
            d_ucx,
            d_sched,
            da_peer,
            nvml_gpus=d_inst["attn"],
            is_attn=True,
            active_freq=cfg.f_da,
        )
        da_flags = _build_afd_flags(
            cfg.tp_da,
            cfg.tp_da,
            cfg.tp_df,
            cfg.tier,
            is_decode=True,
            ib_device=decode_ib,
        )
        da_nccl_port = NCCL_PORT_BASE + nccl_idx * 10
        da_cmd = (
            f"{da_env} {RMB.PYTHON} -m sglang.launch_server "
            f"--host {d_inst['host']} --port {da_port} --afd-perspective attn "
            f"--disaggregation-mode decode --base-gpu-id {len(d_inst['ffn'])} "
            f"--nccl-port {da_nccl_port} {da_flags}"
        )
        nccl_idx += 1
        RMB.ensure_ports_free(d_inst["host"], [da_port, da_nccl_port])
        RMB._launch(d_inst["host"], da_cmd, f"{cfg.name}_d{di}a")
        time.sleep(2)
        decode_eps.append((d_inst["host"], da_port))

    # Health checks
    for i, (host, _) in enumerate(prefill_eps):
        pf_port = PREFILL_PORT_BASE + i * 10 + 2
        pa_port = PREFILL_PORT_BASE + i * 10
        if not RMB.wait_health(host, pf_port, 600, check_model_info=True):
            log.error("PF%d health failed", i)
            return None
        if not RMB.wait_health(host, pa_port, 600, check_model_info=True):
            log.error("PA%d health failed", i)
            return None
    for di, (d_host, da_port) in enumerate(decode_eps):
        if not RMB.wait_health(d_host, da_port + 2, 600, check_model_info=True):
            log.error("DF%d health failed", di)
            return None
        if not RMB.wait_health(d_host, da_port, 600, check_model_info=True):
            log.error("DA%d health failed", di)
            return None

    # Router CLI supports repeated --decode; expose the complete D pool to every P.
    sub_ports = []
    for i, (p_host, pa_port) in enumerate(prefill_eps):
        sp = SUB_ROUTER_BASE + i
        bs_port = BOOTSTRAP_PORT_BASE + i
        selected_decode_eps = (
            [decode_eps[index] for index in p_to_d[i]] if fabric_mode else decode_eps
        )
        decode_args = " ".join(
            f"--decode http://{h}:{p}" for h, p in selected_decode_eps
        )
        rc = (
            f"setsid {RMB.PYTHON} -m sglang_router.launch_router "
            f"--pd-disaggregation --mini-lb "
            f"--prefill http://{p_host}:{pa_port} {bs_port} "
            f"{decode_args} "
            f"--host {RMB.NODE1_IP} --port {sp} "
            f"> {RMB.LOG_C}/{cfg.name}_sub_{i}.log 2>&1 < /dev/null &"
        )
        RMB.dexec_local(rc)
        if not RMB.wait_health(RMB.NODE1_IP, sp, 60):
            log.error("sub-router %d failed", i)
            return None
        sub_ports.append(sp)

    sub_urls = [f"http://{RMB.NODE1_IP}:{p}" for p in sub_ports]
    log.info("Sub-router URLs (client-side RR): %s", sub_urls)
    return sub_urls


def _prefill_route_weights(cfg: Tier1TestConfig) -> list[int]:
    """Per-P client routing weight; default uses attn TP as capacity proxy."""
    return [max(s.tp_pa, s.tp_pf) for s in cfg.resolved_prefill_specs()]


def assign_prefill_urls(
    urls: list[str], cfg: Tier1TestConfig, n_reqs: int
) -> list[str]:
    """Assign /generate URLs across sub-routers (one per P).

    Uniform TP: simple round-robin (1 req per P per cycle).
    Mixed TP: weighted round-robin, e.g. 2xTP2 + 1xTP1 -> [2,2,1] per 5-req cycle.
    """
    weights = _prefill_route_weights(cfg)
    generate = [u if u.endswith("/generate") else u + "/generate" for u in urls]
    if not generate:
        return []
    if len(set(weights)) == 1:
        w = weights[0]
        return [generate[i % len(generate)] for i in range(n_reqs)]
    pool: list[str] = []
    for url, weight in zip(generate, weights):
        pool.extend([url] * weight)
    return [pool[i % len(pool)] for i in range(n_reqs)]


def derive_active_energy_hosts(alloc: dict) -> dict[str, list[int]]:
    """Derive and validate the exact GPU energy scope from an allocation.

    Energy measurement must fail closed: it is never safe to replace missing
    allocation metadata with every GPU on every node.
    """
    freq_map = alloc.get("freq_map")
    total_gpu = alloc.get("total_gpu")
    if not isinstance(freq_map, dict) or not freq_map:
        raise ValueError("allocation must contain a non-empty freq_map")
    if not isinstance(total_gpu, int) or total_gpu <= 0:
        raise ValueError(f"invalid allocation total_gpu: {total_gpu!r}")

    energy_hosts: dict[str, list[int]] = {}
    for host, gpu_freq in freq_map.items():
        if not isinstance(host, str) or not host:
            raise ValueError(f"invalid allocation host: {host!r}")
        if not isinstance(gpu_freq, dict):
            raise ValueError(f"freq_map[{host!r}] must be a dict")
        gpus: list[int] = []
        for gpu, freq in gpu_freq.items():
            if not isinstance(gpu, int) or not 0 <= gpu < GPUS_PER_NODE:
                raise ValueError(f"invalid GPU id for {host}: {gpu!r}")
            if not isinstance(freq, (int, float)) or freq <= 0:
                raise ValueError(f"invalid frequency for {host} GPU{gpu}: {freq!r}")
            gpus.append(gpu)
        if len(gpus) != len(set(gpus)):
            raise ValueError(f"duplicate GPU ids for {host}: {gpus}")
        energy_hosts[host] = sorted(gpus)

    measured_total = sum(len(gpus) for gpus in energy_hosts.values())
    if measured_total != total_gpu:
        raise ValueError(
            f"active energy GPU count mismatch: allocation={total_gpu}, "
            f"derived={measured_total}, hosts={energy_hosts}"
        )

    declared = alloc.get("energy_hosts")
    if declared is not None:
        normalized = {host: sorted(gpus) for host, gpus in declared.items()}
        if normalized != energy_hosts:
            raise ValueError(
                f"declared energy_hosts disagrees with freq_map: "
                f"declared={normalized}, derived={energy_hosts}"
            )
    return energy_hosts


def validate_energy_counters(
    label: str, start: dict[int, int], end: dict[int, int], gpus: list[int]
) -> None:
    """Require complete, positive, monotonic NVML counters for active GPUs."""
    for gpu in gpus:
        if gpu not in start or gpu not in end:
            raise RuntimeError(
                f"{label} energy counter missing GPU{gpu}: start={start}, end={end}"
            )
        if start[gpu] <= 0 or end[gpu] <= 0:
            raise RuntimeError(
                f"{label} energy counter is non-positive on GPU{gpu}: "
                f"start={start[gpu]}, end={end[gpu]}"
            )
        if end[gpu] < start[gpu]:
            raise RuntimeError(
                f"{label} energy counter decreased on GPU{gpu}: "
                f"start={start[gpu]}, end={end[gpu]}"
            )
