#!/usr/bin/env python3
"""Large-scale TP sweep on 16-card two-node cluster (macro conv dataset).

See README.md for the full matrix. Summary:
  1. Native / DynamoLLM: TP1, TP2, TP4, TP8
  2. DistServe / BiScale (PD xnode): TP1, TP2, TP4, TP8
  3. DistServe / BiScale (PD intra): TP1, TP2, TP4
  4. MegaScale / AFlex (PDAF): TP1 x4 inst, TP2 x2 inst, TP4 x1 inst

SLO: TTFT=5000ms, TPOT=300ms. QPS: 2,4,6,8,12,16.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("more_trying")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import run_macro_benchmark as RMB

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

_orig_afd_common = RMB._afd_common


def _patched_afd_common(tp, ib_dev, gpu_step, tier, ngpu=8):
    result = _orig_afd_common(tp, ib_dev, gpu_step, tier, ngpu)
    result = result.replace("--afd-micro-batch 2", "--afd-micro-batch 1")
    result = result.replace("--afd-dynamic-micro-batch", "")
    if tier:
        result = result.replace(RMB.ENERGY_MODEL_DIR_V2, RMB.ENERGY_MODEL_DIR_V1)
        result += " --afd-dvfs-decode-compositional"
    return result


RMB._afd_common = _patched_afd_common

QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "conv"
NGPU = 16
MAX_RUN_S = 400
RESULTS_DIR = HERE / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

GPUS = [0, 1, 2, 3, 4, 5, 6, 7]


def _gpu_groups(tp: int) -> list[list[int]]:
    if 8 % tp != 0:
        raise ValueError(f"tp={tp} does not divide 8 GPUs per node")
    return [GPUS[i : i + tp] for i in range(0, 8, tp)]


# ---------------------------------------------------------------------------
# 1. Native TP sweep
# ---------------------------------------------------------------------------

def start_native_tp_variant(tp: int, tier: bool = False):
    groups = _gpu_groups(tp)
    log.info("Native TP%d-DP 16-card: groups %s x2 nodes, tier=%s", tp, groups, tier)
    insts = []
    idx = 0
    for host in (RMB.NODE1_IP, RMB.NODE2_IP):
        for g in groups:
            insts.append((host, g, 53200 + idx * 10, idx))
            idx += 1
    for host, g, port, i in insts:
        RMB._launch_tp(host, g, port, 33300 + i * 10,
                       f"ntp_{'n1' if host == RMB.NODE1_IP else 'n2'}_{g[0]}", tier)
        time.sleep(2)
    for host, g, port, i in insts:
        if not RMB.wait_health(host, port, 500):
            log.error("Native-TP inst %s:%d failed", host, port)
            return None
    worker_urls = " ".join(f"http://{h}:{p}" for h, _, p, _ in insts)
    rc = (f"setsid {RMB.PYTHON} -m sglang_router.launch_router "
          f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT} --policy round_robin "
          f"--worker-urls {worker_urls} > {RMB.LOG_C}/router.log 2>&1 < /dev/null &")
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        return None
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


# ---------------------------------------------------------------------------
# 2. PD xnode (DistServe / BiScale separated P/D)
# ---------------------------------------------------------------------------

def start_pd_xnode(tp: int, tier: bool = False):
    """P-node N x TP{tp} prefill + D-node N x TP{tp} decode."""
    groups = _gpu_groups(tp)
    log.info("PD xnode TP%d: P=%dxTP%d @node1, D=%dxTP%d @node2, tier=%s",
             tp, len(groups), tp, len(groups), tp, tier)

    p_insts, d_insts = [], []
    for i, g in enumerate(groups):
        p_port = 53100 + i * 10
        bs_port = 49100 + i
        p_insts.append({"gpus": g, "port": p_port, "bs_port": bs_port, "idx": i})
        d_port = 53150 + i * 10
        d_insts.append({"gpus": g, "port": d_port, "idx": i})

    for inst in p_insts:
        nic = RMB.GPU_NIC[inst["gpus"][0]]
        extra = ("--disaggregation-mode prefill "
                 "--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {inst['bs_port']} "
                 f"--disaggregation-ib-device {nic}")
        dvfs_log = None
        if tier:
            dvfs_log = (f"{RMB.DVFS_LOG_DIR}/biscale_p{inst['idx']}_"
                        f"gpu{inst['gpus'][0]}.jsonl")
        cmd = (f"{RMB._plain_env_multi(inst['gpus'], dvfs_log)} setsid prlimit "
               f"--memlock=unlimited:unlimited {RMB.PYTHON} -m sglang.launch_server "
               f"--model-path {RMB.MODEL} --tp {tp} --host {RMB.NODE1_IP} "
               f"--port {inst['port']} --nccl-port {34000 + inst['idx'] * 10} "
               f"{RMB.COMMON_BENCH_SERVER_FLAGS}"
               f"{extra}{RMB._pd_dvfs_flags(tier)} "
               f"> {RMB.LOG_C}/pd_xn_p{inst['idx']}.log 2>&1 < /dev/null &")
        RMB.dexec_local(cmd)
        time.sleep(3)

    for inst in d_insts:
        nic = RMB.GPU_NIC[inst["gpus"][0]]
        extra = ("--disaggregation-mode decode "
                 "--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-bootstrap-port {p_insts[0]['bs_port']} "
                 f"--disaggregation-ib-device {nic}")
        dvfs_log = None
        if tier:
            dvfs_log = (f"{RMB.DVFS_LOG_DIR}/biscale_d{inst['idx']}_"
                        f"gpu{inst['gpus'][0]}.jsonl")
        cmd = (f"{RMB._plain_env_multi(inst['gpus'], dvfs_log)} setsid prlimit "
               f"--memlock=unlimited:unlimited {RMB.PYTHON} -m sglang.launch_server "
               f"--model-path {RMB.MODEL} --tp {tp} --host {RMB.NODE2_IP} "
               f"--port {inst['port']} --nccl-port {34050 + inst['idx'] * 10} "
               f"{RMB.COMMON_BENCH_SERVER_FLAGS}"
               f"{extra}{RMB._pd_dvfs_flags(tier)} "
               f"> {RMB.LOG_C}/pd_xn_d{inst['idx']}.log 2>&1 < /dev/null &")
        RMB.dexec_remote(cmd)
        time.sleep(3)

    for inst in p_insts:
        if not RMB.wait_health(RMB.NODE1_IP, inst["port"], 500):
            return None
    for inst in d_insts:
        if not RMB.wait_health(RMB.NODE2_IP, inst["port"], 500):
            return None

    rc_parts = [f"setsid {RMB.PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT}"]
    for inst in p_insts:
        rc_parts.append(f"--prefill http://{RMB.NODE1_IP}:{inst['port']} {inst['bs_port']}")
    for inst in d_insts:
        rc_parts.append(f"--decode http://{RMB.NODE2_IP}:{inst['port']}")
    rc = " ".join(rc_parts) + f" > {RMB.LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        return None
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


# ---------------------------------------------------------------------------
# 3. PD intra (same-node multi-instance)
# ---------------------------------------------------------------------------

def start_pd_intra(tp: int, tier: bool = False):
    """Intra-node PD: (16/tp)/2 instances, each 2*tp GPUs (tp P + tp D)."""
    cards_per_inst = 2 * tp
    if 16 % cards_per_inst != 0:
        log.error("pd_intra: 16 cards not divisible by 2*tp=%d", cards_per_inst)
        return None
    insts_per_node = (8 // cards_per_inst)
    log.info("PD intra TP%d: %d inst/node (%d total), tier=%s",
             tp, insts_per_node, insts_per_node * 2, tier)

    instances = []
    idx = 0
    for host in (RMB.NODE1_IP, RMB.NODE2_IP):
        for j in range(insts_per_node):
            base = j * cards_per_inst
            p_gpus = GPUS[base : base + tp]
            d_gpus = GPUS[base + tp : base + cards_per_inst]
            instances.append({
                "idx": idx, "host": host,
                "p_gpus": p_gpus, "d_gpus": d_gpus,
                "p_port": 53100 + idx * 10,
                "d_port": 53101 + idx * 10,
                "bs_port": 49100 + idx,
            })
            idx += 1

    for inst in instances:
        host = inst["host"]
        p_nic = RMB.GPU_NIC[inst["p_gpus"][0]]
        d_nic = RMB.GPU_NIC[inst["d_gpus"][0]]
        p_extra = ("--disaggregation-mode prefill "
                   "--disaggregation-transfer-backend mooncake "
                   f"--disaggregation-bootstrap-port {inst['bs_port']} "
                   f"--disaggregation-ib-device {p_nic}")
        d_extra = ("--disaggregation-mode decode "
                   "--disaggregation-transfer-backend mooncake "
                   f"--disaggregation-bootstrap-port {inst['bs_port']} "
                   f"--disaggregation-ib-device {d_nic}")

        if tp == 1:
            RMB._launch_plain(host, inst["p_gpus"][0], inst["p_port"],
                              34000 + inst["idx"] * 10,
                              f"pdin_{inst['idx']}_p", p_extra, tier, pd=True)
            time.sleep(2)
            RMB._launch_plain(host, inst["d_gpus"][0], inst["d_port"],
                              34001 + inst["idx"] * 10,
                              f"pdin_{inst['idx']}_d", d_extra, tier, pd=True)
        else:
            for role, gpus, port, nccl_off, suffix, extra in [
                ("p", inst["p_gpus"], inst["p_port"], 0, "p", p_extra),
                ("d", inst["d_gpus"], inst["d_port"], 1, "d", d_extra),
            ]:
                cmd = (f"{RMB._plain_env_multi(gpus)} setsid prlimit "
                       f"--memlock=unlimited:unlimited {RMB.PYTHON} -m sglang.launch_server "
                       f"--model-path {RMB.MODEL} --tp {tp} --host {host} "
                       f"--port {port} --nccl-port {34000 + inst['idx'] * 10 + nccl_off} "
                       f"{RMB.COMMON_BENCH_SERVER_FLAGS}"
                       f"{extra}{RMB._pd_dvfs_flags(tier)} "
                       f"> {RMB.LOG_C}/pdin_{inst['idx']}_{suffix}.log 2>&1 < /dev/null &")
                if host == RMB.NODE1_IP:
                    RMB.dexec_local(cmd)
                else:
                    RMB.dexec_remote(cmd)
        time.sleep(2)

    for inst in instances:
        for port in (inst["p_port"], inst["d_port"]):
            if not RMB.wait_health(inst["host"], port, 500):
                return None

    rc_parts = [f"setsid {RMB.PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT}"]
    for inst in instances:
        rc_parts.append(f"--prefill http://{inst['host']}:{inst['p_port']} {inst['bs_port']}")
    for inst in instances:
        rc_parts.append(f"--decode http://{inst['host']}:{inst['d_port']}")
    rc = " ".join(rc_parts) + f" > {RMB.LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        return None
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


# ---------------------------------------------------------------------------
# 4. PDAF multi-instance
# ---------------------------------------------------------------------------

def _afd_env_local(role, attn_gpus, ffn_gpus, is_prefill, ucx_off=0, sched_off=0):
    cvd = "0,1,2,3,4,5,6,7"
    base = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            "AFD_IPC_SYNC_MODE=ipc_event "
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            f"CUDA_VISIBLE_DEVICES={cvd} ")
    ucx = str(28200 + ucx_off) if is_prefill else str(28300 + ucx_off)
    sched = str(68400 + sched_off) if is_prefill else str(68500 + sched_off)
    if role == "ffn":
        nvml = ",".join(str(g) for g in ffn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET=-1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={ffn_gpus[0]};")
    nvml = ",".join(str(g) for g in attn_gpus)
    return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
            f"AFD_IPC_PEER_OFFSET=1 "
            f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={attn_gpus[0]} "
            f"AFD_UCX_FFN_HOST=127.0.0.1;")


def start_pdaf_multi(tp: int, tier: bool = False):
    """PDAF with 16/(4*tp) instances (TP1->4, TP2->2, TP4->1)."""
    gpus_per_node = 2 * tp
    n_inst = 8 // gpus_per_node
    log.info("PDAF multi TP%d: %d instances, tier=%s", tp, n_inst, tier)

    ib_map = {str(g): RMB.GPU_NIC[g] for g in GPUS}
    RMB.write_ib_json(ib_map)
    ib_dev = RMB.IB_JSON_FILE
    step = 2 if tp > 1 else 1

    for i in range(n_inst):
        base = i * gpus_per_node
        if tp == 1:
            attn_gpus = [base]
            ffn_gpus = [base + 1]
        else:
            attn_gpus = list(range(base, base + 2 * tp, 2))
            ffn_gpus = [g + 1 for g in attn_gpus]
        attn_base = attn_gpus[0]
        ffn_base = ffn_gpus[0]

        pa_port = 42010 + i * 20
        pf_port = 42011 + i * 20
        da_port = 42020 + i * 20
        df_port = 42021 + i * 20
        bs_port = 49990 + i

        cf = RMB._afd_common(tp, ib_dev, step, tier, ngpu=8)
        cf = cf.replace(f"--disaggregation-bootstrap-port {RMB.BS_PORT}",
                        f"--disaggregation-bootstrap-port {bs_port}")

        # node1 PF + PA
        env = _afd_env_local("ffn", attn_gpus, ffn_gpus, True, i * 4, i * 4)
        RMB._launch(RMB.NODE1_IP,
                    f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                    f"--port {pf_port} --afd-perspective ffn --disaggregation-mode prefill "
                    f"--base-gpu-id {ffn_base} {cf}", f"pf_{i}")
        time.sleep(4)
        env = _afd_env_local("attn", attn_gpus, ffn_gpus, True, i * 4, i * 4)
        RMB._launch(RMB.NODE1_IP,
                    f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                    f"--port {pa_port} --afd-perspective attn --disaggregation-mode prefill "
                    f"--base-gpu-id {attn_base} {cf}", f"pa_{i}")
        # node2 DF + DA
        env = _afd_env_local("ffn", attn_gpus, ffn_gpus, False, i * 4, i * 4)
        RMB._launch(RMB.NODE2_IP,
                    f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                    f"--port {df_port} --afd-perspective ffn --disaggregation-mode decode "
                    f"--base-gpu-id {ffn_base} {cf}", f"df_{i}")
        time.sleep(4)
        env = _afd_env_local("attn", attn_gpus, ffn_gpus, False, i * 4, i * 4)
        RMB._launch(RMB.NODE2_IP,
                    f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                    f"--port {da_port} --afd-perspective attn --disaggregation-mode decode "
                    f"--base-gpu-id {attn_base} {cf}", f"da_{i}")
        time.sleep(4)

        log.info("  instance %d: PA/PF gpu attn=%s ffn=%s", i, attn_gpus, ffn_gpus)

    # Health check all 4*n_inst servers
    for i in range(n_inst):
        pa_port = 42010 + i * 20
        pf_port = 42011 + i * 20
        da_port = 42020 + i * 20
        df_port = 42021 + i * 20
        for host, port, name, mi in [
            (RMB.NODE1_IP, pf_port, f"PF{i}", True),
            (RMB.NODE1_IP, pa_port, f"PA{i}", False),
            (RMB.NODE2_IP, df_port, f"DF{i}", True),
            (RMB.NODE2_IP, da_port, f"DA{i}", False),
        ]:
            if not RMB.wait_health(host, port, 600, check_model_info=mi):
                log.error("  %s failed", name)
                return None

    rc_parts = [f"setsid {RMB.PYTHON} -m sglang_router.launch_router",
                "--pd-disaggregation", f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT}"]
    for i in range(n_inst):
        rc_parts.append(
            f"--prefill http://{RMB.NODE1_IP}:{42010 + i * 20} {49990 + i}")
    for i in range(n_inst):
        rc_parts.append(f"--decode http://{RMB.NODE2_IP}:{42020 + i * 20}")
    rc = " ".join(rc_parts) + f" > {RMB.LOG_C}/router.log 2>&1 < /dev/null &"
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        return None
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


# ---------------------------------------------------------------------------
# Test matrix
# ---------------------------------------------------------------------------

def _configs():
    cfgs = []
    for tp in (1, 2, 4, 8):
        cfgs.append((f"native_tp{tp}_baseline", lambda t=tp: start_native_tp_variant(t, False)))
        cfgs.append((f"native_tp{tp}_tier", lambda t=tp: start_native_tp_variant(t, True)))
    for tp in (1, 2, 4, 8):
        cfgs.append((f"pd_xnode_tp{tp}_baseline", lambda t=tp: start_pd_xnode(t, False)))
        cfgs.append((f"pd_xnode_tp{tp}_tier", lambda t=tp: start_pd_xnode(t, True)))
    for tp in (1, 2, 4):
        cfgs.append((f"pd_intra_tp{tp}_baseline", lambda t=tp: start_pd_intra(t, False)))
        cfgs.append((f"pd_intra_tp{tp}_tier", lambda t=tp: start_pd_intra(t, True)))
    for tp in (1, 2, 4):
        cfgs.append((f"pdaf_tp{tp}_baseline", lambda t=tp: start_pdaf_multi(t, False)))
        cfgs.append((f"pdaf_tp{tp}_tier", lambda t=tp: start_pdaf_multi(t, True)))
    return cfgs


def _save(all_results):
    ts = time.strftime("%Y%m%d_%H%M%S")
    out = RESULTS_DIR / f"more_trying_sweep_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP,
            "node2": RMB.NODE2_IP,
            "model": RMB.MODEL,
            "ngpu_total": NGPU,
            "dataset": DATASET,
            "qps": QPS_LIST,
            "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS,
        },
        "results": all_results,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Results saved: %s", out)
    return out


def main():
    import resource

    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    gpus = RMB.card_gpus(NGPU)
    all_results = {}
    configs = _configs()

    log.info("=" * 72)
    log.info("MORE_TRYING SWEEP | %d configs | conv QPS=%s", len(configs), QPS_LIST)
    log.info("Nodes: %s (prefill) + %s (decode)", RMB.NODE1_IP, RMB.NODE2_IP)
    log.info("=" * 72)

    for name, deploy_fn in configs:
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s", name)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = deploy_fn()
        if url is None:
            log.error("%s DEPLOY_FAILED", name)
            RMB.cleanup_all()
            all_results[name] = {"__status__": "DEPLOY_FAILED"}
            _save(all_results)
            continue

        RMB.lock_freq_both(gpus, RMB.MAX_GPU_FREQ)
        if not RMB.test_generate(url):
            log.error("%s WARMUP_FAILED", name)
            RMB.unlock_freq_both(gpus)
            RMB.cleanup_all()
            all_results[name] = {"__status__": "WARMUP_FAILED"}
            _save(all_results)
            continue
        time.sleep(3)

        deploy_results = {}
        for qps in QPS_LIST:
            log.info("-" * 50)
            res = RMB.run_one_workload(url, DATASET, qps, gpus, gpus, MAX_RUN_S)
            if res is not None:
                deploy_results[res[0]] = res[1]
            time.sleep(5)

        RMB.unlock_freq_both(gpus)
        RMB.cleanup_all()
        all_results[name] = deploy_results
        _save(all_results)

    print("\n" + "=" * 104)
    print("  MORE_TRYING SWEEP COMPLETE")
    print("=" * 104)
    for dep, wl in all_results.items():
        if "__status__" in wl:
            print(f"{dep:<30} {wl['__status__']}")
            continue
        passed = sum(1 for m in wl.values() if m.get("status") == "PASS")
        print(f"{dep:<30} {passed}/{len(QPS_LIST)} QPS passed")


if __name__ == "__main__":
    main()
