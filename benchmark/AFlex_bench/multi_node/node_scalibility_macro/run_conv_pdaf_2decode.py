#!/usr/bin/env python3
"""PDAF with P(TP=4) + 2x decode instances D(TP=2), conv benchmark.

Layout (16 cards):
  node1 (Prefill): PA=GPU[0,2,4,6](TP=4) + PF=GPU[1,3,5,7](TP=4) = 8 cards
  node2 (Decode):
    Instance0: DA0=GPU[0,2](TP=2) + DF0=GPU[1,3](TP=2) = 4 cards
    Instance1: DA1=GPU[4,6](TP=2) + DF1=GPU[5,7](TP=2) = 4 cards

Router: two PD sub-routers (PA+DA0, PA+DA1) + top-level round_robin

SLO: TTFT=5000ms, TPOT=300ms, M=1
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
log = logging.getLogger("pdaf_2d")

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

# Ports for each instance
PA_PORT = 42010
PF_PORT = 42011
DA0_PORT = 42020
DF0_PORT = 42021
DA1_PORT = 42030
DF1_PORT = 42031
ROUTER_PORT = 42000
SUB_ROUTER0_PORT = 42040
SUB_ROUTER1_PORT = 42041
BS_PORT = 49999


def _afd_comm_opt_env():
    """AF-comm optimization toggles, forwarded from host env.

    - AFD_FUSED_PIPELINE=1: use the C++ fused send_recv (one Python->C++
      boundary per layer instead of two) for the M=1 hot path.
    - AFD_FUSED_COMM_STREAM=1: run the fused send on a dedicated high-priority
      comm stream so it overlaps with the next layer's compute.
    - AFD_IPC_SYNC_MODE: override the IPC sync primitive (ipc_event default).
    """
    parts = []
    for key, default in (
        ("AFD_FUSED_PIPELINE", None),
        ("AFD_FUSED_COMM_STREAM", None),
        ("AFD_GPU_ONLY_IPC", None),
    ):
        val = os.environ.get(key, default)
        if val is not None:
            parts.append(f"{key}={val}")
    return (" " + " ".join(parts) + " ") if parts else " "


def _afd_env(role, attn_gpus, ffn_gpus, ucx_port, sched_port):
    """Build env for one PDAF server."""
    cvd = "0,1,2,3,4,5,6,7"
    ipc_sync = os.environ.get("AFD_IPC_SYNC_MODE", "ipc_event")
    # Optional per-instance DVFS decision log. The template uses {persp}/{disagg}/
    # {gpu} placeholders; {gpu}=AFD_NVML_DEVICE_INDEX differs per sub-instance
    # (DA0=0, DA1=4, DF0=1, DF1=5), so each of the four decode processes writes
    # its own file — this is how we verify the two decode instances tune
    # frequency independently rather than in lockstep.
    dvfs_log = os.environ.get("AFD_DVFS_DECISION_LOG")
    dvfs_log_env = f"AFD_DVFS_DECISION_LOG={dvfs_log} " if dvfs_log else ""
    base = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
            "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
            f"AFD_IPC_SYNC_MODE={ipc_sync} "
            f"{_afd_comm_opt_env().strip()} "
            f"{dvfs_log_env}"
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
            f"CUDA_VISIBLE_DEVICES={cvd} ")
    if role == "ffn":
        nvml = ",".join(str(g) for g in ffn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
                f"AFD_IPC_PEER_OFFSET=-1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={ffn_gpus[0]};")
    else:
        nvml = ",".join(str(g) for g in attn_gpus)
        return (base + f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
                f"AFD_IPC_PEER_OFFSET=1 "
                f"AFD_NVML_DEVICE_INDICES={nvml} AFD_NVML_DEVICE_INDEX={attn_gpus[0]} "
                f"AFD_UCX_FFN_HOST=127.0.0.1;")


def _afd_common(tp, ib_dev, gpu_step, tier, bs_port):
    flags = (f"--model-path {RMB.MODEL} --tp {tp} --gpu-id-step {gpu_step} "
             "--afd-comm-backend ipc_cpp "
             "--afd-micro-batch 1 --mem-fraction-static 0.85 "
             "--max-running-requests 512 --skip-server-warmup "
             "--watchdog-timeout 600 "
             "--disable-cuda-graph --disable-piecewise-cuda-graph "
             "--afd-disagg-interleave-poll --disable-radix-cache "
             "--num-reserved-decode-tokens 512 "
             "--disaggregation-transfer-backend mooncake "
             f"--disaggregation-bootstrap-port {bs_port} "
             f"--disaggregation-ib-device {ib_dev} --enable-metrics")
    if tier:
        flags += (" --afd-dvfs-enabled "
                  f"--afd-energy-model-dir {RMB.ENERGY_MODEL_DIR_V2} "
                  f"--afd-ttft-slo-ms {int(RMB.TTFT_SLO_MS)} "
                  f"--afd-tpot-slo-us {int(RMB.TPOT_SLO_MS * 1000)} "
                  "--afd-dvfs-idle-lock")
        # Optional: enable online calibration to correct predictor bias. The
        # decode latency predictor over-estimates iteration time (~175% here),
        # which pins frequency high; calibration feeds observed TPOT back to
        # rescale predictions so DVFS can exploit the real SLO headroom.
        if os.environ.get("AFD_DVFS_ONLINE_CALIBRATION") == "1":
            flags += " --afd-dvfs-online-calibration"
        ema = os.environ.get("AFD_DVFS_CALIBRATION_EMA")
        if ema:
            flags += f" --afd-dvfs-calibration-ema {ema}"
    return flags


def start_pdaf_2decode(tier=False):
    """Start PDAF: 1 prefill instance (TP=4) + 2 decode instances (TP=2 each)."""
    p_tp = 4
    d_tp = 2
    step = 2

    # Prefill: PA=[0,2,4,6], PF=[1,3,5,7] (TP=4)
    p_attn = [0, 2, 4, 6]
    p_ffn = [1, 3, 5, 7]
    # Decode0: DA=[0,2], DF=[1,3]
    d0_attn = [0, 2]
    d0_ffn = [1, 3]
    # Decode1: DA=[4,6], DF=[5,7]
    d1_attn = [4, 6]
    d1_ffn = [5, 7]

    log.info("PDAF 2×Decode: P=[A:%s F:%s] TP=%d | D0=[A:%s F:%s] D1=[A:%s F:%s] TP=%d | tier=%s",
             p_attn, p_ffn, p_tp, d0_attn, d0_ffn, d1_attn, d1_ffn, d_tp, tier)

    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    ib_map = {str(g): RMB.GPU_NIC[g] for g in all_gpus}
    RMB.write_ib_json(ib_map)
    ib_dev = RMB.IB_JSON_FILE

    p_cf = _afd_common(p_tp, ib_dev, step, tier, BS_PORT)
    d0_cf = _afd_common(d_tp, ib_dev, step, tier, BS_PORT)
    # Both decode instances must connect to the same prefill bootstrap server.
    # Using BS_PORT + 1 makes DA1 query a non-existent bootstrap endpoint.
    d1_cf = _afd_common(d_tp, ib_dev, step, tier, BS_PORT)

    # --- Prefill (node1) ---
    # PF (TP=4)
    env = _afd_env("ffn", p_attn, p_ffn, ucx_port=28200, sched_port=68400)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {PF_PORT} --afd-perspective ffn --disaggregation-mode prefill "
                f"--base-gpu-id {p_ffn[0]} {p_cf}", "pf")
    time.sleep(6)
    # PA (TP=4)
    env = _afd_env("attn", p_attn, p_ffn, ucx_port=28200, sched_port=68400)
    RMB._launch(RMB.NODE1_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE1_IP} "
                f"--port {PA_PORT} --afd-perspective attn --disaggregation-mode prefill "
                f"--base-gpu-id {p_attn[0]} {p_cf}", "pa")

    # --- Decode instance 0 (node2, GPU 0-3, TP=2) ---
    env = _afd_env("ffn", d0_attn, d0_ffn, ucx_port=28300, sched_port=68500)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {DF0_PORT} --afd-perspective ffn --disaggregation-mode decode "
                f"--base-gpu-id {d0_ffn[0]} {d0_cf}", "df0")
    time.sleep(8)
    env = _afd_env("attn", d0_attn, d0_ffn, ucx_port=28300, sched_port=68500)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {DA0_PORT} --afd-perspective attn --disaggregation-mode decode "
                f"--base-gpu-id {d0_attn[0]} {d0_cf}", "da0")

    # --- Decode instance 1 (node2, GPU 4-7, TP=2) ---
    time.sleep(5)
    env = _afd_env("ffn", d1_attn, d1_ffn, ucx_port=28400, sched_port=68600)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {DF1_PORT} --afd-perspective ffn --disaggregation-mode decode "
                f"--base-gpu-id {d1_ffn[0]} {d1_cf}", "df1")
    time.sleep(8)
    env = _afd_env("attn", d1_attn, d1_ffn, ucx_port=28400, sched_port=68600)
    RMB._launch(RMB.NODE2_IP,
                f"{env} {RMB.PYTHON} -m sglang.launch_server --host {RMB.NODE2_IP} "
                f"--port {DA1_PORT} --afd-perspective attn --disaggregation-mode decode "
                f"--base-gpu-id {d1_attn[0]} {d1_cf}", "da1")

    log.info("Waiting for all PDAF servers...")
    for host, port, name, mi in [
        (RMB.NODE1_IP, PF_PORT, "PF", True),
        (RMB.NODE1_IP, PA_PORT, "PA", False),
        (RMB.NODE2_IP, DF0_PORT, "DF0", True),
        (RMB.NODE2_IP, DA0_PORT, "DA0", False),
        (RMB.NODE2_IP, DF1_PORT, "DF1", True),
        (RMB.NODE2_IP, DA1_PORT, "DA1", False),
    ]:
        if not RMB.wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed", name, host, port)
            return None
        log.info("  %s ready", name)

    # Router: use two independent PD MiniLB sub-routers and a top-level round-robin
    # router. A single MiniLB with two --decode URLs can choose randomly and makes
    # it hard to verify/use both decode instances deterministically.
    sub_router_specs = [
        (SUB_ROUTER0_PORT, DA0_PORT, "router_d0"),
        (SUB_ROUTER1_PORT, DA1_PORT, "router_d1"),
    ]
    for sub_port, da_port, log_name in sub_router_specs:
        rc = (f"setsid {RMB.PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
              f"--prefill http://{RMB.NODE1_IP}:{PA_PORT} "
              f"--decode http://{RMB.NODE2_IP}:{da_port} "
              f"--host {RMB.NODE1_IP} --port {sub_port} "
              f"> {RMB.LOG_C}/{log_name}.log 2>&1 < /dev/null &")
        RMB.dexec_local(rc)
        if not RMB.wait_health(RMB.NODE1_IP, sub_port, 60):
            log.error("  sub-router %s failed", log_name)
            return None
        log.info("  sub-router %s ready", log_name)

    worker_urls = " ".join(
        f"http://{RMB.NODE1_IP}:{port}" for port, _, _ in sub_router_specs
    )
    rc = (f"setsid {RMB.PYTHON} -m sglang_router.launch_router "
          f"--host {RMB.NODE1_IP} --port {ROUTER_PORT} --policy round_robin "
          f"--worker-urls {worker_urls} > {RMB.LOG_C}/router.log 2>&1 < /dev/null &")
    RMB.dexec_local(rc)
    if not RMB.wait_health(RMB.NODE1_IP, ROUTER_PORT, 60):
        log.error("  top-level router failed")
        return None
    log.info("  top-level round-robin router ready")
    return f"http://{RMB.NODE1_IP}:{ROUTER_PORT}"


def main():
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    all_gpus = [0, 1, 2, 3, 4, 5, 6, 7]
    all_results = {}

    log.info("=" * 72)
    log.info("PDAF 2×Decode(TP=2): P(TP=4,8cards) + 2×D(TP=2,4cards each)")
    log.info("Total 16 cards | SLO TTFT=5000ms TPOT=300ms | M=1 | conv")
    log.info("=" * 72)

    for mode in ["baseline", "tier"]:
        tier = mode == "tier"
        full_name = f"pdaf_2d_tp2_{mode}"
        log.info("\n" + "=" * 72)
        log.info("DEPLOY: %s (tier=%s)", full_name, tier)
        log.info("=" * 72)

        RMB.cleanup_all()
        url = start_pdaf_2decode(tier)
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
    out_file = RESULTS_DIR / f"conv_pdaf_2decode_{ts}.json"
    payload = {
        "meta": {
            "node1": RMB.NODE1_IP, "node2": RMB.NODE2_IP,
            "model": RMB.MODEL,
            "layout": "P: PA(TP4)+PF(TP4)=8cards@node1 | D: 2×[DA(TP2)+DF(TP2)]=8cards@node2",
            "total_cards": 16,
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
    print("  PDAF P(TP=4) + 2×Decode(TP=2) | 16 cards | SLO TTFT=5s TPOT=300ms")
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
