#!/usr/bin/env python3
"""Benchmark Tier1-optimized AFD configs on QPS=16 code, node3+node4.

Reuses proven deploy_tier1_layout + run_macro_benchmark infra.
Adds multi-pair (k_P > 1) support with custom (tp, freq) per pool.
"""
from __future__ import annotations

import argparse, asyncio, json, logging, os, sys, time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("bench_tier1_v2")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))

import run_macro_benchmark as RMB
from freq_timeline_utils import FreqTimelineSession, ensure_container_log_dir
from run_fixed_6scheme_7dataset import _wl_key, _workload_file, MAX_RUN_S as F67_MAX_RUN_S

# ── Override to node3+node4 ──
RMB.NODE1_IP = os.environ.get("MN_NODE3_IP", "10.252.129.34")   # node3
RMB.NODE2_IP = os.environ.get("MN_NODE4_IP", "10.252.129.33")   # node4
RMB.TTFT_SLO_MS = 2000.0
RMB.TPOT_SLO_MS = 100.0

RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

GPUS_PER_NODE = 8
ROUTER_PORT = 44000
SUB_ROUTER_BASE = 45000
DECODE_PORT_BASE = 43020
PREFILL_PORT_BASE = 43200
NCCL_PORT_BASE = 37300
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
    k_p: int; k_d: int
    tp_pa: int; tp_pf: int
    tp_da: int; tp_df: int
    f_pa: int; f_pf: int
    f_da: int; f_df: int
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

CONFIGS = [
    Tier1TestConfig("tier1_14g", k_p=4, k_d=1, tp_pa=2, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=1170, f_da=930, f_df=930, tier=True),
    Tier1TestConfig("tier1_10g", k_p=4, k_d=1, tp_pa=1, tp_pf=1, tp_da=1, tp_df=1,
                    f_pa=930, f_pf=930, f_da=930, f_df=930, tier=True),
]


# ── GPU allocation ──
def alloc_af_gpus_contiguous(tp_f, tp_a, base_gpus):
    """Contiguous allocation: FFN first, then Attn (like run_pdaf_4pa2pf_test.py)."""
    ffn = base_gpus[:tp_f]
    attn = base_gpus[tp_f:tp_f + tp_a]
    return attn, ffn


def _uses_three_node_18gpu(cfg: Tier1TestConfig) -> bool:
    return (
        bool(getattr(RMB, "NODE_EXTRA_IP", ""))
        and cfg.k_p == 3
        and cfg.k_d == 3
        and cfg.tp_pa == cfg.tp_pf == 2
        and cfg.tp_da == cfg.tp_df == 1
        and cfg.total_gpu() == 18
    )


def plan_allocation_three_node_18gpu(cfg: Tier1TestConfig):
    """3P(TP2)+3D(TP1) on node1(8) + node2(NODE2_IP, 8) + extra(NODE_EXTRA_IP, GPU0-1).

    Layout (decode first, then prefill):
      D0: extra GPU [0,1]
      D1,D2: node1 GPU [0,1], [2,3]
      P0: node1 GPU [4,5,6,7]
      P1,P2: node2 GPU [0..3], [4..7]
    """
    extra_ip = RMB.NODE_EXTRA_IP
    n1_ip = RMB.NODE1_IP
    n2_ip = RMB.NODE2_IP
    pspec = cfg.resolved_prefill_specs()[0]

    decode_instances = [
        {"host": extra_ip, "attn": [1], "ffn": [0]},
        {"host": n1_ip, "attn": [1], "ffn": [0]},
        {"host": n1_ip, "attn": [3], "ffn": [2]},
    ]
    p_pairs = [
        (n1_ip, [6, 7], [4, 5], pspec),
        (n2_ip, [2, 3], [0, 1], pspec),
        (n2_ip, [6, 7], [4, 5], pspec),
    ]

    fmap = {n1_ip: {}, n2_ip: {}, extra_ip: {}}
    for host, a, f, spec in p_pairs:
        for g in a:
            fmap[host][g] = spec.f_pa
        for g in f:
            fmap[host][g] = spec.f_pf
    for d_inst in decode_instances:
        for g in d_inst["attn"]:
            fmap[d_inst["host"]][g] = cfg.f_da
        for g in d_inst["ffn"]:
            fmap[d_inst["host"]][g] = cfg.f_df

    primary = decode_instances[0]
    return {
        "p_pairs": p_pairs,
        "decode": {"host": primary["host"], "attn": primary["attn"], "ffn": primary["ffn"]},
        "decode_instances": decode_instances,
        "freq_map": fmap,
        "total_gpu": cfg.total_gpu(),
        "energy_hosts": {
            n1_ip: sorted(fmap[n1_ip].keys()),
            n2_ip: sorted(fmap[n2_ip].keys()),
            extra_ip: sorted(fmap[extra_ip].keys()),
        },
    }


def plan_allocation(cfg):
    if _uses_three_node_18gpu(cfg):
        return plan_allocation_three_node_18gpu(cfg)
    n1 = list(range(8)); n2 = list(range(8))
    # Decode instances (k_d replicas, starting on node3)
    d_instances = []
    for _ in range(cfg.k_d):
        need = cfg.tp_da + cfg.tp_df
        placed = False
        for label, free in [("n3", n1), ("n4", n2)]:
            if need > len(free): continue
            d_a, d_f = alloc_af_gpus_contiguous(cfg.tp_df, cfg.tp_da, free[:need])
            d_instances.append((label, d_a, d_f))
            if label == "n3": n1 = [g for g in n1 if g not in set(d_a)|set(d_f)]
            else: n2 = [g for g in n2 if g not in set(d_a)|set(d_f)]
            placed = True
            break
        if not placed:
            raise RuntimeError(f"Cannot place D instance #{len(d_instances)+1}")
    # Prefill pairs (optionally heterogeneous TP/freq per instance)
    pspecs = cfg.resolved_prefill_specs()
    pairs = []
    for spec in pspecs:
        need = spec.tp_pa + spec.tp_pf
        placed = False
        for label, free in [("n3", n1), ("n4", n2)]:
            if need > len(free): continue
            a, f = alloc_af_gpus_contiguous(spec.tp_pf, spec.tp_pa, free[:need])
            pairs.append((label, a, f, spec))
            if label == "n3": n1 = [g for g in n1 if g not in set(a)|set(f)]
            else: n2 = [g for g in n2 if g not in set(a)|set(f)]
            placed = True
            break
        if not placed:
            raise RuntimeError(f"Cannot place P pair #{len(pairs)+1} (need {need} GPUs)")
    if len(pairs) != len(pspecs):
        raise RuntimeError(f"Cannot place all P pairs: planned {len(pairs)}, expected {len(pspecs)}")
    # Freq map
    fmap = {RMB.NODE1_IP: {}, RMB.NODE2_IP: {}}
    for label, a, f, spec in pairs:
        host = RMB.NODE1_IP if label == "n3" else RMB.NODE2_IP
        for g in a: fmap[host][g] = spec.f_pa
        for g in f: fmap[host][g] = spec.f_pf
    for label, d_a, d_f in d_instances:
        host = RMB.NODE1_IP if label == "n3" else RMB.NODE2_IP
        for g in d_a: fmap[host][g] = cfg.f_da
        for g in d_f: fmap[host][g] = cfg.f_df
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
    }


# ── Deploy with heterogeneous TP support (no gpu-id-step, use AFD_IPC_PEER_DEVICE) ──
def _build_env_str(host, cvd_gpus, ucx_port, sched_port, peer_device, nvml_gpus,
                   is_attn=False):
    """Build env prefix, pinning the host IP to avoid incorrect multi-NIC auto-detection."""
    cvd = ",".join(str(g) for g in cvd_gpus)
    nvml = ",".join(str(g) for g in nvml_gpus)
    env = (
        f"export CUDA_VISIBLE_DEVICES={cvd} SGLANG_HOST_IP={host} "
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
    sample_dir = os.environ.get("AFD_DECODE_BATCH_SAMPLE_DIR")
    if sample_dir:
        sample_role = "attn" if is_attn else "ffn"
        sample_gpu = "_".join(str(gpu) for gpu in nvml_gpus)
        sample_path = f"{sample_dir}/decode_{sample_role}_gpus_{sample_gpu}.csv"
        env += f"AFD_DECODE_BATCH_SAMPLE_PATH={sample_path} "
    dvfs_log_tmpl = os.environ.get("AFD_DVFS_DECISION_LOG")
    if dvfs_log_tmpl:
        env += f"AFD_DVFS_DECISION_LOG={dvfs_log_tmpl} "
    if is_attn:
        env += "AFD_UCX_FFN_HOST=127.0.0.1 "
    env += ";"
    return env


def _build_afd_flags(tp, tp_pa, tp_pf, tier, is_decode=False):
    """Build server flags for heterogeneous TP (no --gpu-id-step)."""
    mem_frac = "0.88" if is_decode else "0.75"
    flags = (
        f"--model-path {RMB.MODEL} --tp {tp} "
        "--afd-comm-backend ipc_cpp --afd-micro-batch 1 "
        f"--mem-fraction-static {mem_frac} --max-running-requests 64 "
        "--skip-server-warmup --watchdog-timeout 600 "
        "--disable-cuda-graph --disable-piecewise-cuda-graph "
        "--afd-disagg-interleave-poll --disable-radix-cache "
        "--num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-ib-device {RMB.IB_JSON_FILE} --enable-metrics "
        f"--afd-attn-tp {tp_pa} --afd-ffn-tp {tp_pf} "
        f"{SERVER_EXTRA_FLAGS} "
    )
    if tier:
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
    log.info("DEPLOY %s: %d GPU (hetero TP: PA=%d PF=%d DA=%d DF=%d)",
             cfg.name, alloc["total_gpu"], cfg.tp_pa, cfg.tp_pf, cfg.tp_da, cfg.tp_df)
    for i, (host, a, f, spec) in enumerate(p_pairs):
        log.info("  P%d: %s attn=%s ffn=%s (TP PA=%d PF=%d)",
                 i, host.split(".")[-1], a, f, spec.tp_pa, spec.tp_pf)
    for i, d_inst in enumerate(alloc.get("decode_instances", [alloc["decode"]])):
        log.info("  D%d: %s attn=%s ffn=%s", i,
                 d_inst["host"].split(".")[-1], d_inst["attn"], d_inst["ffn"])

    # Lock frequencies
    for host, fmap in alloc["freq_map"].items():
        if fmap and cfg.tier:
            log.info("  Locking %s: %s", host,
                     {g: f"{fm}MHz" for g, fm in sorted(fmap.items())})
            RMB.lock_freq_map_on_host(host, fmap)

    # Write IB JSON
    ib_map = {str(g): RMB.GPU_NIC[g] for g in range(GPUS_PER_NODE)}
    RMB.write_ib_json(ib_map)

    # Pre-warm JIT compilation on both nodes (avoid file lock contention)
    jit_warmup_cmd = (
        f"CUDA_VISIBLE_DEVICES=0 {RMB.PYTHON} -c "
        "\"import sys; sys.path.insert(0,'/workspace/sglang/python'); "
        "from sglang.srt.layers.afd_ipc_cpp import get_module; get_module(); "
        "print('JIT_WARM_OK')\""
    )
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
        pa_port = PREFILL_PORT_BASE + i * 10
        pf_port = PREFILL_PORT_BASE + i * 10 + 2
        ucx = 28200 + i * 100
        sched = 68400 + i * 100
        # CVD = all GPUs in this pair (ffn + attn, contiguous)
        cvd_gpus = ffn_gpus + attn_gpus
        # PF: base_gpu_id=0 (first in CVD), peer_device = len(ffn_gpus) (attn start)
        pf_peer = len(ffn_gpus)  # index within CVD where attn starts
        pf_env = _build_env_str(host, cvd_gpus, ucx, sched, pf_peer,
                                nvml_gpus=ffn_gpus, is_attn=False)
        pf_flags = _build_afd_flags(spec.tp_pf, spec.tp_pa, spec.tp_pf, cfg.tier)
        pf_cmd = (f"{pf_env} {RMB.PYTHON} -m sglang.launch_server "
                  f"--host {host} --port {pf_port} --afd-perspective ffn "
                  f"--disaggregation-mode prefill --base-gpu-id 0 "
                  f"--nccl-port {NCCL_PORT_BASE + nccl_idx * 10} {pf_flags}")
        nccl_idx += 1
        RMB._launch(host, pf_cmd, f"{cfg.name}_p{i}f")
        time.sleep(5)
        # PA: base_gpu_id=len(ffn_gpus) (attn start in CVD), peer_device=0 (ffn start)
        pa_peer = 0
        pa_env = _build_env_str(host, cvd_gpus, ucx, sched, pa_peer,
                                nvml_gpus=attn_gpus, is_attn=True)
        pa_flags = _build_afd_flags(spec.tp_pa, spec.tp_pa, spec.tp_pf, cfg.tier)
        pa_cmd = (f"{pa_env} {RMB.PYTHON} -m sglang.launch_server "
                  f"--host {host} --port {pa_port} --afd-perspective attn "
                  f"--disaggregation-mode prefill --base-gpu-id {len(ffn_gpus)} "
                  f"--nccl-port {NCCL_PORT_BASE + nccl_idx * 10} {pa_flags}")
        nccl_idx += 1
        RMB._launch(host, pa_cmd, f"{cfg.name}_p{i}a")
        time.sleep(2)
        prefill_eps.append((host, pa_port))

    # Launch decode instance(s)
    decode_eps = []
    d_instances = alloc.get("decode_instances", [alloc["decode"]])
    for di, d_inst in enumerate(d_instances):
        d_ucx = 28200 + (cfg.effective_k_p() + di) * 100
        d_sched = 68400 + (cfg.effective_k_p() + di) * 100
        da_port = DECODE_PORT_BASE + di * 20
        df_port = DECODE_PORT_BASE + di * 20 + 2
        d_cvd = d_inst["ffn"] + d_inst["attn"]
        # DF
        df_peer = len(d_inst["ffn"])
        df_env = _build_env_str(d_inst["host"], d_cvd, d_ucx, d_sched, df_peer,
                                nvml_gpus=d_inst["ffn"], is_attn=False)
        df_flags = _build_afd_flags(cfg.tp_df, cfg.tp_da, cfg.tp_df, cfg.tier, is_decode=True)
        df_cmd = (f"{df_env} {RMB.PYTHON} -m sglang.launch_server "
                  f"--host {d_inst['host']} --port {df_port} --afd-perspective ffn "
                  f"--disaggregation-mode decode --base-gpu-id 0 "
                  f"--nccl-port {NCCL_PORT_BASE + nccl_idx * 10} {df_flags}")
        nccl_idx += 1
        RMB._launch(d_inst["host"], df_cmd, f"{cfg.name}_d{di}f")
        time.sleep(5)
        # DA
        da_peer = 0
        da_env = _build_env_str(d_inst["host"], d_cvd, d_ucx, d_sched, da_peer,
                                nvml_gpus=d_inst["attn"], is_attn=True)
        da_flags = _build_afd_flags(cfg.tp_da, cfg.tp_da, cfg.tp_df, cfg.tier, is_decode=True)
        da_cmd = (f"{da_env} {RMB.PYTHON} -m sglang.launch_server "
                  f"--host {d_inst['host']} --port {da_port} --afd-perspective attn "
                  f"--disaggregation-mode decode --base-gpu-id {len(d_inst['ffn'])} "
                  f"--nccl-port {NCCL_PORT_BASE + nccl_idx * 10} {da_flags}")
        nccl_idx += 1
        RMB._launch(d_inst["host"], da_cmd, f"{cfg.name}_d{di}a")
        time.sleep(2)
        decode_eps.append((d_inst["host"], da_port))

    # Health checks
    for i, (host, _) in enumerate(prefill_eps):
        pf_port = PREFILL_PORT_BASE + i * 10 + 2
        pa_port = PREFILL_PORT_BASE + i * 10
        if not RMB.wait_health(host, pf_port, 600, check_model_info=True):
            log.error("PF%d health failed", i); return None
        if not RMB.wait_health(host, pa_port, 600, check_model_info=True):
            log.error("PA%d health failed", i); return None
    for di, (d_host, da_port) in enumerate(decode_eps):
        if not RMB.wait_health(d_host, da_port + 2, 600, check_model_info=True):
            log.error("DF%d health failed", di); return None
        if not RMB.wait_health(d_host, da_port, 600, check_model_info=True):
            log.error("DA%d health failed", di); return None

    # Router CLI supports repeated --decode; expose the complete D pool to every P.
    sub_ports = []
    for i, (p_host, pa_port) in enumerate(prefill_eps):
        sp = SUB_ROUTER_BASE + i
        bs_port = pa_port + 1
        decode_args = " ".join(f"--decode http://{h}:{p}" for h, p in decode_eps)
        rc = (f"setsid {RMB.PYTHON} -m sglang_router.launch_router "
              f"--pd-disaggregation --mini-lb "
              f"--prefill http://{p_host}:{pa_port} {bs_port} "
              f"{decode_args} "
              f"--host {RMB.NODE1_IP} --port {sp} "
              f"> {RMB.LOG_C}/{cfg.name}_sub_{i}.log 2>&1 < /dev/null &")
        RMB.dexec_local(rc)
        if not RMB.wait_health(RMB.NODE1_IP, sp, 60):
            log.error("sub-router %d failed", i); return None
        sub_ports.append(sp)

    sub_urls = [f"http://{RMB.NODE1_IP}:{p}" for p in sub_ports]
    log.info("Sub-router URLs (client-side RR): %s", sub_urls)
    return sub_urls


def _prefill_route_weights(cfg: Tier1TestConfig) -> list[int]:
    """Per-P client routing weight; default uses attn TP as capacity proxy."""
    return [max(s.tp_pa, s.tp_pf) for s in cfg.resolved_prefill_specs()]


def assign_prefill_urls(urls: list[str], cfg: Tier1TestConfig, n_reqs: int) -> list[str]:
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


async def _run_workload_rr(reqs, n1_gpus, n2_gpus, max_run_s=400, extra_energy=None):
    """Run RR workload with exact completion and timeout accounting."""
    import aiohttp as _aio
    import numpy as _np
    from collections import Counter

    e1s = RMB.get_energy_local(n1_gpus); e2s = RMB.get_energy_remote(n2_gpus)
    extra_energy = extra_energy or {}
    extra_s = {
        host: RMB.get_energy_on_host(host, gpus)
        for host, gpus in extra_energy.items()
    }
    results = []
    async with _aio.ClientSession(timeout=_aio.ClientTimeout(total=max_run_s + 60)) as session:
        base_time = time.monotonic()
        async def send_indexed(index, req):
            before = len(results)
            await RMB.send_one(session, req["_target_url"], req, base_time, results)
            if len(results) > before:
                results[-1].setdefault("request_index", index)
                results[-1].setdefault("target_url", req["_target_url"])
        tasks = [asyncio.create_task(send_indexed(i, r)) for i, r in enumerate(reqs)]
        done, pending = await asyncio.wait(tasks, timeout=max_run_s)
        timed_out = len(pending)
        for task in pending: task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
            log.warning("Timed out after %ds; cancelled %d pending requests", max_run_s, timed_out)
        if done: await asyncio.gather(*done, return_exceptions=True)
    duration_s = time.monotonic() - base_time
    e1e = RMB.get_energy_local(n1_gpus); e2e = RMB.get_energy_remote(n2_gpus)
    extra_e = {
        host: RMB.get_energy_on_host(host, gpus)
        for host, gpus in extra_energy.items()
    }
    energy_n1_j = sum((e1e.get(i, 0)-e1s.get(i, 0))/1000 for i in n1_gpus)
    energy_n2_j = sum((e2e.get(i, 0)-e2s.get(i, 0))/1000 for i in n2_gpus)
    energy_extra_j = sum(
        (extra_e[host].get(i, 0) - extra_s[host].get(i, 0)) / 1000
        for host, gpus in extra_energy.items()
        for i in gpus
    )
    total_energy_j = energy_n1_j + energy_n2_j + energy_extra_j
    ok = [r for r in results if r.get("success")]; fail = [r for r in results if not r.get("success")]
    missing = max(0, len(reqs) - len(results) - timed_out); completed = len(ok)
    status = ("PASS" if completed == len(reqs) else "TIMEOUT" if timed_out and not completed
              else "PARTIAL_TIMEOUT" if timed_out else "FAIL")
    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in ok if r.get("ttft_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r.get("tpot_ms", 0) > 0]
    src = ttfts_proc if ttfts_proc else ttfts
    req_by_idx = {i: req for i, req in enumerate(reqs)}
    total_output_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    total_input_tokens = 0
    for r in ok:
        idx = r.get("request_index")
        inp = r.get("input_len")
        if inp is None and idx is not None:
            inp = req_by_idx.get(idx, {}).get("input_len", 0)
        total_input_tokens += inp or 0
    total_tokens_all = total_input_tokens + total_output_tokens
    violated = sum(1 for r in ok if ((r.get("ttft_proc_ms") or r.get("ttft_ms", 0)) > RMB.TTFT_SLO_MS
                                     or r.get("tpot_ms", 0) > RMB.TPOT_SLO_MS))
    unsuccessful = len(reqs) - completed
    return {
        "status": status, "duration_s": round(duration_s, 1), "total_requests": len(reqs),
        "successful": completed, "failed": len(fail), "timed_out": timed_out, "missing": missing,
        "total_tokens": total_output_tokens,
        "total_input_tokens": total_input_tokens,
        "total_tokens_all": total_tokens_all,
        "throughput_tok_s": round(total_output_tokens/duration_s, 1) if duration_s else 0,
        "ttft_proc_avg_ms": round(float(_np.mean(src)), 1) if src else 0,
        "ttft_proc_p50_ms": round(float(_np.percentile(src, 50)), 1) if src else 0,
        "ttft_proc_p99_ms": round(float(_np.percentile(src, 99)), 1) if src else 0,
        "tpot_avg_ms": round(float(_np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(_np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(_np.percentile(tpots, 99)), 1) if tpots else 0,
        "energy_node1_j": round(energy_n1_j, 1), "energy_node2_j": round(energy_n2_j, 1),
        "energy_extra_j": round(energy_extra_j, 1) if extra_energy else 0,
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": (
            round(total_energy_j * 1000 / total_tokens_all, 2) if total_tokens_all else 0
        ),
        "energy_denominator": "input_plus_output",
        "slo_violation_rate": round((violated+unsuccessful)/len(reqs)*100, 1) if reqs else 0,
        "slo_violating_requests": violated, "unsuccessful_requests": unsuccessful,
        "route_distribution": dict(Counter(r.get("target_url", "unknown") for r in results)),
        "assigned_route_distribution": dict(Counter(r["_target_url"] for r in reqs)),
        "request_results": results,
    }


# ── Workload runner ──
def run_benchmark(urls, cfg):
    """urls is a list of sub-router URLs; assign requests across P sub-routers."""
    wl_file = _workload_file(DATASET, QPS)
    if wl_file is None:
        return {"status": "NO_WORKLOAD"}
    with open(wl_file) as f:
        reqs = [json.loads(line) for line in f]
    last_arrival = max((r["arrival_time_s"] for r in reqs), default=0)
    run_s = int(min(max(F67_MAX_RUN_S, last_arrival + 150), 900))
    weights = _prefill_route_weights(cfg)
    if len(set(weights)) == 1:
        log.info("Workload: %d reqs, run_window=%ds, %d sub-routers (RR)",
                 len(reqs), run_s, len(urls))
    else:
        log.info(
            "Workload: %d reqs, run_window=%ds, %d sub-routers (weighted RR weights=%s)",
            len(reqs), run_s, len(urls), weights,
        )

    assigned = assign_prefill_urls(urls, cfg, len(reqs))
    for r, target in zip(reqs, assigned):
        r["_target_url"] = target

    gpus = list(range(GPUS_PER_NODE))
    alloc = plan_allocation(cfg)
    energy_hosts = alloc.get("energy_hosts")
    if energy_hosts:
        n1_gpus = energy_hosts.get(RMB.NODE1_IP, gpus)
        n2_gpus = energy_hosts.get(RMB.NODE2_IP, gpus)
        extra_energy = {
            host: gpulist
            for host, gpulist in energy_hosts.items()
            if host not in (RMB.NODE1_IP, RMB.NODE2_IP)
        }
    else:
        n1_gpus = gpus
        n2_gpus = gpus
        extra_energy = {}
    ensure_container_log_dir(cfg.name)
    sess = FreqTimelineSession(cfg.name, DATASET, QPS)
    sess.start()
    try:
        summary = asyncio.run(_run_workload_rr(reqs, n1_gpus, n2_gpus, run_s, extra_energy))
    finally:
        freq_meta = sess.stop()

    if isinstance(summary, dict) and summary.get("status") == "PASS":
        summary = dict(summary)
        summary["freq_timeline"] = freq_meta
        summary["config"] = {k: getattr(cfg, k) for k in cfg.__dataclass_fields__}
        log.info("PASS: thpt=%.1f TTFT=%.1fms TPOT=%.1fms E/tok=%.1fmJ SLO=%.1f%%",
                 summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                 summary["tpot_avg_ms"], summary.get("energy_per_token_mj", 0),
                 summary["slo_violation_rate"])
    else:
        log.error("FAIL: %s", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", nargs="*", choices=[c.name for c in CONFIGS])
    args = parser.parse_args()
    targets = [c for c in CONFIGS if args.configs is None or c.name in args.configs]
    results = {}

    for cfg in targets:
        log.info("\n" + "#" * 72)
        log.info("TESTING: %s", cfg.name)
        log.info("#" * 72)
        RMB.cleanup_all()
        time.sleep(8)
        url = deploy(cfg)
        if url is None:
            results[cfg.name] = {"status": "DEPLOY_FAILED"}; continue
        if not RMB.test_generate(url[0]):
            results[cfg.name] = {"status": "WARMUP_FAILED"}; RMB.cleanup_all(); continue
        time.sleep(3)
        summary = run_benchmark(url, cfg)
        results[cfg.name] = summary
        # Unlock + cleanup
        for host in [RMB.NODE1_IP, RMB.NODE2_IP]:
            try: RMB.unlock_freq_both(gpus)
            except: pass
        RMB.cleanup_all()
        time.sleep(5)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"tier1_v2_{ts}.json"
    meta = {"qps": QPS, "dataset": DATASET, "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS, "n3": RMB.NODE1_IP, "n4": RMB.NODE2_IP}
    out.write_text(json.dumps({"meta": meta, "results": results}, indent=2))
    log.info("Saved %s", out)

    print(f"\n{'='*60}")
    for name, r in results.items():
        if r.get("status") == "PASS":
            print(f"  {name}: thpt={r['throughput_tok_s']:.1f} TTFT={r['ttft_proc_avg_ms']:.1f}ms "
                  f"TPOT={r['tpot_avg_ms']:.1f}ms E/tok={r['energy_per_token_mj']:.1f}mJ")
        else:
            print(f"  {name}: {r.get('status')}")


if __name__ == "__main__":
    main()
