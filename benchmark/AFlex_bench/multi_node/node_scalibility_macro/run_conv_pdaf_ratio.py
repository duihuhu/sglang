#!/usr/bin/env python3
"""PDAF heterogeneous with different PA:PF ratios on Prefill side.

Tests two configurations (both Decode=TP4 on node2):
  1. PA:PF = 4:2 -> PA=TP4(GPU[0,2,4,6]), PF=TP2(GPU[1,3]) on node1 (6 cards P)
  2. PA:PF = 2:4 -> PA=TP2(GPU[0,2]), PF=TP4(GPU[1,3,5,7]) on node1 (6 cards P)

Decode side: DA=TP4(GPU[0,2,4,6]) + DF=TP4(GPU[1,3,5,7]) on node2 (8 cards)

Total: 14 cards (config1) or 14 cards (config2).

SLO: TTFT=5000ms, TPOT=300ms, M=1, conv dataset.
"""
from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pdaf_ratio")

os.environ.setdefault("MN_NODE1_IP", "10.252.129.34")
os.environ.setdefault("MN_NODE2_IP", "10.252.129.33")

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_macro_benchmark as RMB

RMB.TTFT_SLO_MS = 5000.0
RMB.TPOT_SLO_MS = 300.0

QPS_LIST = [2, 4, 6, 8, 12, 16]
DATASET = "conv"
MAX_RUN_S = 400
RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _afd_env(role, attn_gpus, ffn_gpus, is_prefill):
    """Build env for PDAF server."""
    cvd = "0,1,2,3,4,5,6,7"
    base = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            "AFD_IPC_SYNC_MODE=ipc_event "
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            f"CUDA_VISIBLE_DEVICES={cvd} ")
    ucx = "28200" if is_prefill else "28300"
    sched = "68400" if is_prefill else "68500"
    if role == "ffn":
        nvml = ",".join(str(g) for g in ffn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET=-1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={ffn_gpus[0]};")
    else:
        nvml = ",".join(str(g) for g in attn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx} AFD_SCHED_PORT={sched} "
                f"AFD_IPC_PEER_OFFSET=1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={attn_gpus[0]} "
                f"AFD_UCX_FFN_HOST=127.0.0.1;")


def _afd_common(tp, ib_dev, gpu_step, tier):
    flags = (f"--model-path {RMB.MODEL} --tp {tp} --gpu-id-step {gpu_step} "
             "--afd-comm-backend ipc_cpp "
             "--afd-micro-batch 1 --mem-fraction-static 0.85 "
             "--max-running-requests 512 --skip-server-warmup "
             "--watchdog-timeout 600 "
             "--disable-cuda-graph --disable-piecewise-cuda-graph "
             "--afd-disagg-interleave-poll --disable-radix-cache "
             "--num-reserved-decode-tokens 512 "
             "--disaggregation-transfer-backend mooncake "
             f"--disaggregation-bootstrap-port {RMB.BS_PORT} "
             f"--disaggregation-ib-device {ib_dev} --enable-metrics")
    if tier:
        flags += (" --afd-dvfs-enabled "
                  f"--afd-energy-model-dir {RMB.ENERGY_MODEL_DIR_V2} "
                  f"--afd-ttft-slo-ms {int(RMB.TTFT_SLO_MS)} "
                  f"--afd-tpot-slo-us {int(RMB.TPOT_SLO_MS * 1000)} "
                  "--afd-dvfs-idle-lock")
    return flags


def start_pdaf_ratio(pa_tp, pf_tp, pa_gpus, pf_gpus, tier=False):
    """Start PDAF with specified PA/PF TP and GPU assignments.

    Decode side always: DA=TP4(GPU[0,2,4,6]), DF=TP4(GPU[1,3,5,7]) on node2.
    """
    d_tp = 4
    d_step = 2
    d_attn_gpus = [0, 2, 4, 6]
    d_ffn_gpus = [1, 3, 5, 7]

    pa_step = pa_gpus[1] - pa_gpus[0] if len(pa_gpus) > 1 else 1
    pf_step = pf_gpus[1] - pf_gpus[0] if len(pf_gpus) > 1 else 1

    log.info("PDAF Ratio: PA(TP=%d, GPU%s, step=%d) + PF(TP=%d, GPU%s, step=%d) | "
             "D: TP=%d | tier=%s",
             pa_tp, pa_gpus, pa_step, pf_tp, pf_gpus, pf_step, d_tp, tier)

    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    ib_map = {str(g): RMB.GPU_NIC[g] for g in all_gpus}
    RMB.write_ib_json(ib_map)
    ib_dev = RMB.IB_JSON_FILE

    p_pa_cf = _afd_common(pa_tp, ib_dev, pa_step, tier)
    p_pf_cf = _afd_common(pf_tp, ib_dev, pf_step, tier)
    d_cf = _afd_common(d_tp, ib_dev, d_step, tier)

    # node1: PF
    env = _afd_env("ffn", pa_gpus, pf_gpus, is_prefill=True)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {RMB.PF_PORT} --afd-perspective ffn --disaggregation-mode prefill "
                f"--base-gpu-id {pf_gpus[0]} {p_pf_cf}", "pf")
    time.sleep(6)

    # node1: PA
    env = _afd_env("attn", pa_gpus, pf_gpus, is_prefill=True)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {RMB.PA_PORT} --afd-perspective attn --disaggregation-mode prefill "
                f"--base-gpu-id {pa_gpus[0]} {p_pa_cf}", "pa")

    # node2: DF
    env = _afd_env("ffn", d_attn_gpus, d_ffn_gpus, is_prefill=False)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {RMB.DF_PORT} --afd-perspective ffn --disaggregation-mode decode "
                f"--base-gpu-id {d_ffn_gpus[0]} {d_cf}", "df")
    time.sleep(8)

    # node2: DA
    env = _afd_env("attn", d_attn_gpus, d_ffn_gpus, is_prefill=False)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {RMB.DA_PORT} --afd-perspective attn --disaggregation-mode decode "
                f"--base-gpu-id {d_attn_gpus[0]} {d_cf}", "da")

    log.info("Waiting for servers...")
    for host, port, name, mi in [(RMB.NODE1_IP, RMB.PF_PORT, "PF", True),
                                  (RMB.NODE1_IP, RMB.PA_PORT, "PA", False),
                                  (RMB.NODE2_IP, RMB.DF_PORT, "DF", True),
                                  (RMB.NODE2_IP, RMB.DA_PORT, "DA", False)]:
        if not RMB.wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed", name, host, port)
            return None
        log.info("  %s ready", name)

    rc = (f"setsid {RMB.PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
          f"--prefill http://{RMB.NODE1_IP}:{RMB.PA_PORT} --decode http://{RMB.NODE2_IP}:{RMB.DA_PORT} "
          f"--host {RMB.NODE1_IP} --port {RMB.ROUTER_PORT} > {RMB.LOG_C}/router.log 2>&1 < /dev/null &")
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, RMB.ROUTER_PORT, 60):
        log.error("  router failed")
        return None
    log.info("  router ready")
    return f"http://{RMB.NODE1_IP}:{RMB.ROUTER_PORT}"


# Configuration definitions:
# PA:PF = 4:2 -> PA uses 4 GPUs (TP=4), PF uses 2 GPUs (TP=2)
# PA:PF = 2:4 -> PA uses 2 GPUs (TP=2), PF uses 4 GPUs (TP=4)
CONFIGS = [
    {
        "name": "PA4_PF2",
        "label": "PA:PF=4:2",
        "pa_tp": 4, "pa_gpus": [0, 2, 4, 6],
        "pf_tp": 2, "pf_gpus": [1, 3],
        "p_cards": 6,
    },
    {
        "name": "PA2_PF4",
        "label": "PA:PF=2:4",
        "pa_tp": 2, "pa_gpus": [0, 2],
        "pf_tp": 4, "pf_gpus": [1, 3, 5, 7],
        "p_cards": 6,
    },
]


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    all_results = {}

    log.info("=" * 72)
    log.info("PDAF PA:PF RATIO TEST | D=TP4(8cards) | conv | M=1")
    log.info("SLO TTFT=5000ms TPOT=300ms")
    log.info("=" * 72)

    for cfg in CONFIGS:
        for mode in ["baseline", "tier"]:
            tier = mode == "tier"
            full_name = f"{cfg['name']}_{mode}"
            total_cards = cfg["p_cards"] + 8
            log.info("\n" + "=" * 72)
            log.info("DEPLOY: %s (%s, %d cards, tier=%s)",
                     cfg["label"], full_name, total_cards, tier)
            log.info("=" * 72)

            RMB.cleanup_all()
            url = start_pdaf_ratio(
                cfg["pa_tp"], cfg["pf_tp"],
                cfg["pa_gpus"], cfg["pf_gpus"],
                tier=tier)
            if url is None:
                log.error("%s deployment FAILED", full_name)
                RMB.cleanup_all()
                all_results[full_name] = {"__status__": "DEPLOY_FAILED"}
                continue

            RMB.lock_freq_both(all_gpus, RMB.MAX_GPU_FREQ)
            log.info("Warmup...")
            if not RMB.test_generate(url):
                log.error("%s warmup FAILED", full_name)
                RMB.unlock_freq_both(all_gpus)
                RMB.cleanup_all()
                all_results[full_name] = {"__status__": "WARMUP_FAILED"}
                continue
            time.sleep(3)

            deploy_results = {}
            for qps in QPS_LIST:
                log.info("-" * 50)
                res = RMB.run_one_workload(url, DATASET, qps, all_gpus, all_gpus, MAX_RUN_S)
                if res is not None:
                    deploy_results[res[0]] = res[1]
                time.sleep(5)

            RMB.unlock_freq_both(all_gpus)
            RMB.cleanup_all()
            all_results[full_name] = deploy_results

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"conv_pdaf_ratio_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP,
            "model": RMB.MODEL,
            "configs": [c["label"] for c in CONFIGS],
            "decode": "DA(TP4)+DF(TP4)=8cards@node2",
            "dataset": DATASET, "qps": QPS_LIST,
            "ttft_slo_ms": RMB.TTFT_SLO_MS,
            "tpot_slo_ms": RMB.TPOT_SLO_MS,
            "pdaf_micro_batch": 1,
        },
        "results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Results saved: %s", out_file)

    print("\n" + "=" * 108)
    print("  PDAF PA:PF RATIO | D=TP4 | SLO TTFT=5s TPOT=300ms")
    print("=" * 108)
    print(f"{'Config':<20} {'Mode':<10} {'QPS':<6} {'Thpt':>8} {'TTFT':>8} {'TPOT':>8} {'E_tot':>9} {'mJ/tok':>8} {'SLO%':>6}")
    print("-" * 108)
    for cfg in CONFIGS:
        for mode in ["baseline", "tier"]:
            full_name = f"{cfg['name']}_{mode}"
            wl = all_results.get(full_name, {})
            if "__status__" in wl:
                print(f"{cfg['label']:<20} {mode:<10} {wl['__status__']}")
                continue
            for q in QPS_LIST:
                key = f"conv_qps{q}"
                m = wl.get(key, {})
                if m.get("status") != "PASS":
                    continue
                print(f"{cfg['label']:<20} {mode:<10} {q:<6} {m['throughput_tok_s']:>8.1f} "
                      f"{m['ttft_proc_avg_ms']:>8.1f} {m['tpot_avg_ms']:>8.1f} "
                      f"{m['total_energy_j']:>9.0f} {m['energy_per_token_mj']:>8.1f} "
                      f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 108)


if __name__ == "__main__":
    main()
