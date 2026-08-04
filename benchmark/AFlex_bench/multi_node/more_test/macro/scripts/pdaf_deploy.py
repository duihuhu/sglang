"""PDAF deployment launchers for 3-class sweep (see README.md)."""
from __future__ import annotations

import logging
import time

import run_macro_benchmark as RMB

log = logging.getLogger("pdaf_deploy")

GPUS = list(range(8))
LAYOUTS = ("tp4x1", "tp2x2", "tp1x4")
HEALTH_TIMEOUT = 180  # fail fast if deploy stuck


def side_groups(layout: str) -> list[tuple[list[int], list[int], int]]:
    """Return [(attn_gpus, ffn_gpus, tp), ...] for one 8-GPU node side."""
    if layout == "tp4x1":
        return [([0, 2, 4, 6], [1, 3, 5, 7], 4)]
    if layout == "tp2x2":
        return [([0, 2], [1, 3], 2), ([4, 6], [5, 7], 2)]
    if layout == "tp1x4":
        return [([g], [g + 1], 1) for g in (0, 2, 4, 6)]
    raise ValueError(f"unknown layout: {layout}")


def _afd_env_local(role, attn_gpus, ffn_gpus, is_prefill, ucx_off=0, sched_off=0):
    import run_more_trying_sweep as MTS
    return MTS._afd_env_local(role, attn_gpus, ffn_gpus, is_prefill, ucx_off, sched_off)


def _afd_common(tp, bs_port, tier):
    cf = RMB._afd_common(tp, RMB.IB_JSON_FILE, 2 if tp > 1 else 1, tier, ngpu=8)
    return cf.replace(f"--disaggregation-bootstrap-port {RMB.BS_PORT}",
                      f"--disaggregation-bootstrap-port {bs_port}")


def _write_ib():
    ib_map = {str(g): RMB.GPU_NIC[g] for g in GPUS}
    RMB.write_ib_json(ib_map)


def _launch_afd(host, role, mode, attn_gpus, ffn_gpus, tp, port, base_gpu, cf, tag,
                ucx_off: int = 0, sched_off: int = 0):
    """role: pa/pf/da/df, mode: prefill/decode."""
    is_p = mode == "prefill"
    persp = "ffn" if role in ("pf", "df") else "attn"
    env = _afd_env_local(persp, attn_gpus, ffn_gpus, is_p, ucx_off, sched_off)
    cmd = (f"{env} {RMB.PYTHON} -m sglang.launch_server --host {host} "
           f"--port {port} --afd-perspective {persp} "
           f"--disaggregation-mode {mode} --base-gpu-id {base_gpu} {cf}")
    RMB._launch(host, cmd, tag)


def _launch_xnode_pair(p_grp, d_grp, idx, tier, bs_base):
    """One matched PA/PF on node1 + DA/DF on node2."""
    attn_p, ffn_p, tp_p = p_grp
    attn_d, ffn_d, tp_d = d_grp
    if tp_p != tp_d:
        log.warning("pair %d: P tp=%d != D tp=%d (still launching)", idx, tp_p, tp_d)
    bs = bs_base + idx
    cf_p = _afd_common(tp_p, bs, tier)
    cf_d = _afd_common(tp_d, bs, tier)
    pa_port = 42010 + idx * 20
    pf_port = 42011 + idx * 20
    da_port = 42020 + idx * 20
    df_port = 42021 + idx * 20

    off = idx * 4
    _launch_afd(RMB.NODE1_IP, "pf", "prefill", attn_p, ffn_p, tp_p, pf_port, ffn_p[0],
                cf_p, f"pdaf_pf_{idx}", off, off)
    time.sleep(3)
    _launch_afd(RMB.NODE1_IP, "pa", "prefill", attn_p, ffn_p, tp_p, pa_port, attn_p[0],
                cf_p, f"pdaf_pa_{idx}", off, off)
    _launch_afd(RMB.NODE2_IP, "df", "decode", attn_d, ffn_d, tp_d, df_port, ffn_d[0],
                cf_d, f"pdaf_df_{idx}", off, off)
    time.sleep(3)
    _launch_afd(RMB.NODE2_IP, "da", "decode", attn_d, ffn_d, tp_d, da_port, attn_d[0],
                cf_d, f"pdaf_da_{idx}", off, off)
    time.sleep(3)

    checks = [
        (RMB.NODE1_IP, pf_port, True), (RMB.NODE1_IP, pa_port, False),
        (RMB.NODE2_IP, df_port, True), (RMB.NODE2_IP, da_port, False),
    ]
    for host, port, mi in checks:
        if not RMB.wait_health(host, port, HEALTH_TIMEOUT, check_model_info=mi):
            return None
    return {"pa": pa_port, "da": da_port, "bs": bs}


def _launch_intra_chain(host, base: int, tp: int, idx: int, tier: bool,
                        bs_base: int, host_off: int = 0):
    """Full PDAF chain on one host (P+D local). TP2 uses all 8 GPUs; TP1 uses 4."""
    bs = bs_base + idx + host_off * 10
    cf = _afd_common(tp, bs, tier)
    pa_port = 42210 + idx * 20 + host_off * 100
    pf_port = 42211 + idx * 20 + host_off * 100
    da_port = 42220 + idx * 20 + host_off * 100
    df_port = 42221 + idx * 20 + host_off * 100
    if tp == 1:
        p_attn, p_ffn = [base], [base + 1]
        d_attn, d_ffn = [base + 2], [base + 3]
    else:
        # One 8-GPU chain: P attn/ffn on 0-3, D attn/ffn on 4-7 (interleaved TP2)
        p_attn, p_ffn = [0, 2], [1, 3]
        d_attn, d_ffn = [4, 6], [5, 7]

    off = (host_off * 8 + idx) * 4

    _launch_afd(host, "pf", "prefill", p_attn, p_ffn, tp, pf_port, p_ffn[0], cf,
                f"intra_pf_{host_off}_{idx}", off, off)
    time.sleep(3)
    _launch_afd(host, "pa", "prefill", p_attn, p_ffn, tp, pa_port, p_attn[0], cf,
                f"intra_pa_{host_off}_{idx}", off, off)
    _launch_afd(host, "df", "decode", d_attn, d_ffn, tp, df_port, d_ffn[0], cf,
                f"intra_df_{host_off}_{idx}", off, off)
    time.sleep(3)
    _launch_afd(host, "da", "decode", d_attn, d_ffn, tp, da_port, d_attn[0], cf,
                f"intra_da_{host_off}_{idx}", off, off)
    time.sleep(3)
    for port, mi in [(pf_port, True), (pa_port, False), (df_port, True), (da_port, False)]:
        if not RMB.wait_health(host, port, HEALTH_TIMEOUT, check_model_info=mi):
            return None
    return {"pa": pa_port, "da": da_port, "bs": bs}


def _router_pd(pairs):
  rc_parts = [f"setsid {RMB.PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT}"]
  for p in pairs:
      rc_parts.append(f"--prefill http://{RMB.NODE1_IP}:{p['pa']} {p['bs']}")
  for p in pairs:
      rc_parts.append(f"--decode http://{RMB.NODE2_IP}:{p['da']}")
  rc = " ".join(rc_parts) + f" > {RMB.LOG_C}/router.log 2>&1 < /dev/null &"
  RMB.dexec_local(rc)
  if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
      return None
  return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


def _router_intra(pairs_n1, pairs_n2):
    """Round-robin across node-local PDAF stacks."""
    rc_parts = [f"setsid {RMB.PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT}"]
    for host, pairs in ((RMB.NODE1_IP, pairs_n1), (RMB.NODE2_IP, pairs_n2)):
        for p in pairs:
            rc_parts.append(f"--prefill http://{host}:{p['pa']} {p['bs']}")
    for host, pairs in ((RMB.NODE1_IP, pairs_n1), (RMB.NODE2_IP, pairs_n2)):
        for p in pairs:
            rc_parts.append(f"--decode http://{host}:{p['da']}")
    rc = " ".join(rc_parts) + f" > {RMB.LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        return None
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


def deploy_c3_xnode_tp1x4(tier: bool = False):
    """Category 3: 4 cross-node instances, TP1."""
    log.info("Deploy C3: xnode 4×TP1 tier=%s", tier)
    import run_more_trying_sweep as MTS
    return MTS.start_pdaf_multi(1, tier)


def deploy_c1_xnode(p_layout: str, d_layout: str, tier: bool = False):
    """Category 1: 1 cross-node deployment, configurable P/D AF layouts."""
    log.info("Deploy C1: xnode P=%s D=%s tier=%s", p_layout, d_layout, tier)
    _write_ib()
    p_grps = side_groups(p_layout)
    d_grps = side_groups(d_layout)
    if len(p_grps) != len(d_grps):
        log.error(
            "C1 unsupported: P=%s (%d inst) != D=%s (%d inst); need matched counts",
            p_layout, len(p_grps), d_layout, len(d_grps),
        )
        return None
    pairs = []
    for i, (pg, dg) in enumerate(zip(p_grps, d_grps)):
        info = _launch_xnode_pair(pg, dg, i, tier, 49900)
        if info is None:
            return None
        pairs.append(info)
    return _router_pd(pairs)


def deploy_c2_intra(layout: str, tier: bool = False):
    """Category 2: 2 node-local instances (no cross-node KV)."""
    log.info("Deploy C2: intra layout=%s tier=%s", layout, tier)
    _write_ib()
    if layout == "tp2x2":
        chains = [(0, 2)]  # one 8-GPU TP2 chain per node
    elif layout == "tp1x4":
        chains = [(0, 1), (4, 1)]  # two 4-GPU TP1 chains per node
    else:
        raise ValueError(layout)

    pairs_n1, pairs_n2 = [], []
    for host_off, host in enumerate((RMB.NODE1_IP, RMB.NODE2_IP)):
        pairs = pairs_n1 if host_off == 0 else pairs_n2
        for i, (base, tp) in enumerate(chains):
            info = _launch_intra_chain(host, base, tp, i, tier, 49800, host_off)
            if info is None:
                return None
            pairs.append(info)
    return _router_intra(pairs_n1, pairs_n2)
