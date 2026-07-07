#!/usr/bin/env python3
"""PDAF heterogeneous: P(TP=2) + D(TP=4), conv benchmark.

Layout (12 cards total):
  node1 (Prefill): PA=GPU[0,2] (TP=2,step=2) + PF=GPU[1,3] (TP=2,step=2) = 4 cards
  node2 (Decode):  DA=GPU[0,2,4,6] (TP=4,step=2) + DF=GPU[1,3,5,7] (TP=4,step=2) = 8 cards

SLO: TTFT=5000ms, TPOT=300ms, M=1
"""
from __future__ import annotations

import json
import logging
import os
import shlex
import sys
import tempfile
import time
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("pdaf_hetero")

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


def _afd_env_hetero(role, tp, attn_gpus, ffn_gpus):
    """Build env for heterogeneous PDAF (different TP for P vs D)."""
    cvd = "0,1,2,3,4,5,6,7"
    base = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            "AFD_IPC_SYNC_MODE=ipc_event "
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            f"CUDA_VISIBLE_DEVICES={cvd} ")
    ucx = "28200" if role in ("PF", "PA") else "28300"
    sched = "68400" if role in ("PF", "PA") else "68500"
    if role in ("PF", "DF"):
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


def _afd_common_hetero(tp, ib_dev, gpu_step, tier):
    """Common flags for PDAF server (per-side TP)."""
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


def start_pdaf_hetero(tier=False):
    """PDAF heterogeneous: Prefill TP=2 (4 cards node1), Decode TP=4 (8 cards node2)."""
    p_tp, p_step = 2, 2
    d_tp, d_step = 4, 2
    p_attn_gpus = [0, 2]
    p_ffn_gpus = [1, 3]
    d_attn_gpus = [0, 2, 4, 6]
    d_ffn_gpus = [1, 3, 5, 7]

    log.info("PDAF Hetero: P(TP=%d)=[A:%s F:%s]@node1, D(TP=%d)=[A:%s F:%s]@node2, tier=%s",
             p_tp, p_attn_gpus, p_ffn_gpus, d_tp, d_attn_gpus, d_ffn_gpus, tier)

    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    ib_map = {str(g): RMB.GPU_NIC[g] for g in all_gpus}
    RMB.write_ib_json(ib_map)
    ib_dev = RMB.IB_JSON_FILE

    p_cf = _afd_common_hetero(p_tp, ib_dev, p_step, tier)
    d_cf = _afd_common_hetero(d_tp, ib_dev, d_step, tier)

    # node1: PF (FFN prefill, TP=2)
    env = _afd_env_hetero("PF", p_tp, p_attn_gpus, p_ffn_gpus)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {RMB.PF_PORT} --afd-perspective ffn --disaggregation-mode prefill "
                f"--base-gpu-id {p_ffn_gpus[0]} {p_cf}", "pf")
    time.sleep(6)

    # node1: PA (ATTN prefill, TP=2)
    env = _afd_env_hetero("PA", p_tp, p_attn_gpus, p_ffn_gpus)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {RMB.PA_PORT} --afd-perspective attn --disaggregation-mode prefill "
                f"--base-gpu-id {p_attn_gpus[0]} {p_cf}", "pa")

    # node2: DF (FFN decode, TP=4)
    env = _afd_env_hetero("DF", d_tp, d_attn_gpus, d_ffn_gpus)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {RMB.DF_PORT} --afd-perspective ffn --disaggregation-mode decode "
                f"--base-gpu-id {d_ffn_gpus[0]} {d_cf}", "df")
    time.sleep(8)

    # node2: DA (ATTN decode, TP=4)
    env = _afd_env_hetero("DA", d_tp, d_attn_gpus, d_ffn_gpus)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {RMB.DA_PORT} --afd-perspective attn --disaggregation-mode decode "
                f"--base-gpu-id {d_attn_gpus[0]} {d_cf}", "da")

    log.info("Waiting for PDAF-Hetero servers...")
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


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    all_results = {}

    log.info("=" * 72)
    log.info("PDAF HETERO: P(TP=2, 4 cards) + D(TP=4, 8 cards) = 12 cards")
    log.info("SLO TTFT=5000ms TPOT=300ms | M=1 | conv dataset")
    log.info("=" * 72)

    for mode in ["baseline", "tier"]:
        tier = mode == "tier"
        label = "MegaScale_hetero" if not tier else "AFlex_hetero"
        full_name = f"pdaf_hetero_{mode}"
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s (%s)", label, full_name)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = start_pdaf_hetero(tier)
        if url is None:
            log.error("%s deployment FAILED", label)
            RMB.cleanup_all()
            all_results[full_name] = {"__status__": "DEPLOY_FAILED"}
            continue

        RMB.lock_freq_both(all_gpus, RMB.MAX_GPU_FREQ)
        log.info("Warmup...")
        if not RMB.test_generate(url):
            log.error("%s warmup FAILED", label)
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
        out_file = RESULTS_DIR / f"conv_pdaf_hetero_{ts}.json"
        payload = {
            "meta": {
                "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP,
                "model": RMB.MODEL,
                "layout": "P: PA(TP2)+PF(TP2)=4cards@node1, D: DA(TP4)+DF(TP4)=8cards@node2",
                "total_cards": 12,
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

    print("\n" + "=" * 104)
    print("  PDAF HETERO: P(TP=2) + D(TP=4) | 12 cards | SLO TTFT=5s TPOT=300ms")
    print("=" * 104)
    print(f"{'Deploy':<24} {'Workload':<14} {'Thpt':>8} {'TTFT':>8} {'TPOT':>8} {'E_tot':>9} {'mJ/tok':>8} {'SLO%':>6}")
    print("-" * 104)
    for dep, wl in all_results.items():
        if "__status__" in wl:
            print(f"{dep:<24} {wl['__status__']}")
            continue
        for w, m in wl.items():
            if m.get("status") != "PASS":
                print(f"{dep:<24} {w:<14} FAIL")
                continue
            print(f"{dep:<24} {w:<14} {m['throughput_tok_s']:>8.1f} "
                  f"{m['ttft_proc_avg_ms']:>8.1f} {m['tpot_avg_ms']:>8.1f} "
                  f"{m['total_energy_j']:>9.0f} {m['energy_per_token_mj']:>8.1f} "
                  f"{m['slo_violation_rate']:>6.1f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
