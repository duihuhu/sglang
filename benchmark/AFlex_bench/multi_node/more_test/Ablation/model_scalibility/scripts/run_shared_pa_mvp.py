#!/usr/bin/env python3
"""Deploy and benchmark shared-PA multi-PF (1PA+3PF) with elastic Decode.

Topology (per P-node):
  GPU0-1: PF0 (TP2)
  GPU2-3: PF1 (TP2)
  GPU4-5: PF2 (TP2)
  GPU6:   PA  (TP1)

Decode (elastic, per D-pair):
  DF (TP2, 2 GPU) + DA (TP1, 1 GPU) = 3 GPU per pair
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import shlex
import time
from pathlib import Path

import run_moe_retest as base

log = logging.getLogger(__name__)

P_NODES = [base.NODE1_IP, base.NODE3_IP]
D_NODES = [base.NODE4_IP, base.NODE2_IP]
P_CVD = "0,1,2,3,4"
ROUTER_PORT = 42000

NUM_PF_GROUPS = 2
PA_GPU = 4
PA_PORT = 45360

ENERGY_MODEL_DIR_V1 = ("/workspace/sglang/benchmark/AFlex_bench/energy_model/"
                       "Mixtral-8x7B/models_v1")
TTFT_SLO_MS = 1000.0
TPOT_SLO_MS = 150.0

PF_GROUPS = [
    {"group_id": 0, "base_gpu": 0, "sched_port": 60400, "http_port": 45300,
     "nccl_port": 39600, "bootstrap_port": 52300},
    {"group_id": 1, "base_gpu": 2, "sched_port": 60500, "http_port": 45310,
     "nccl_port": 39610, "bootstrap_port": 52310},
]

DECODE_CONFIG = {
    4:  {"n_pairs": 2, "p_nodes": 2, "d_nodes": [D_NODES[0]]},
    8:  {"n_pairs": 2, "p_nodes": 2, "d_nodes": [D_NODES[0]]},
}


def _env(values):
    return "export " + " ".join(
        f"{key}={shlex.quote(str(value))}" for key, value in values.items()
    ) + ";"


def _afd_dvfs_flags():
    return (" --afd-dvfs-enabled "
            f"--afd-energy-model-dir {ENERGY_MODEL_DIR_V1} "
            f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)} "
            f"--afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)} "
            "--afd-dvfs-idle-lock "
            "--afd-dvfs-prefill-fixed-max "
            "--afd-dvfs-decode-compositional")


def _common_flags(tp, dvfs=False):
    flags = (
        f"--model-path {base.MODEL} --tp {tp} "
        f"{base.COMMON_FLAGS} "
        "--afd-comm-backend ipc_cpp "
        "--afd-attn-tp 1 --afd-ffn-tp 2 "
        "--max-running-requests 512 --watchdog-timeout 600 "
        "--num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-ib-device {base.IB_JSON_FILE} "
        "--enable-metrics"
    )
    if dvfs:
        flags += _afd_dvfs_flags()
    return flags


def _launch(host, name, env, flags, port, nccl_port, bootstrap_port):
    command = (
        f"{_env(env)} setsid prlimit --memlock=unlimited:unlimited "
        f"{base.PYTHON} -m sglang.launch_server "
        f"--host {host} --port {port} --nccl-port {nccl_port} "
        f"--disaggregation-bootstrap-port {bootstrap_port} "
        f"{flags} > {base.LOG_C}/shared_pa_{name}.log 2>&1 < /dev/null &"
    )
    base.dexec(host, command)


# --------------- Prefill deployment ---------------

def deploy_shared_prefill(host, dvfs=False):
    """Deploy 2 PF groups + 1 PA on a single P-node (5 GPU)."""
    host_tag = host.rsplit(".", 1)[-1]
    shared_env = {
        "SGLANG_HOST_IP": host,
        "SGLANG_DISABLE_REQUEST_LOGGING": "true",
        "CUDA_VISIBLE_DEVICES": P_CVD,
        "AFD_IPC_SYNC_MODE": "ipc_event",
        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
        "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
        "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "600",
    }

    peer_devices = ",".join(str(g["base_gpu"]) for g in PF_GROUPS)
    sched_endpoints = ",".join(
        f"127.0.0.1:{g['sched_port']}" for g in PF_GROUPS
    )

    for g in PF_GROUPS:
        env = {
            **shared_env,
            "AFD_SCHED_HOST": "127.0.0.1",
            "AFD_SCHED_PORT": g["sched_port"],
            "AFD_NVML_DEVICE_INDICES": f"{g['base_gpu']},{g['base_gpu'] + 1}",
        }
        flags = (
            f"--afd-perspective ffn --base-gpu-id {g['base_gpu']} "
            "--disaggregation-mode prefill --afd-micro-batch 1 "
            "--chunked-prefill-size -1 --max-prefill-tokens 16384 "
            f"--afd-multi-pf-continuation --afd-pf-group-count {NUM_PF_GROUPS} "
            f"--afd-pf-group-id {g['group_id']} --afd-pf-peer-devices {PA_GPU} "
            "--afd-pf-channel-base 700 "
            + _common_flags(2, dvfs=dvfs)
        )
        _launch(
            host, f"{host_tag}_pf{g['group_id']}",
            env, flags,
            g["http_port"], g["nccl_port"], g["bootstrap_port"],
        )

    for g in PF_GROUPS:
        if not base.wait_health(host, g["http_port"], 600):
            raise RuntimeError(f"{host} PF{g['group_id']} failed health")
        log.info("%s PF%d ready", host, g["group_id"])

    pa_env = {
        **shared_env,
        "AFD_NVML_DEVICE_INDICES": str(PA_GPU),
    }
    pa_flags = (
        f"--afd-perspective attn --base-gpu-id {PA_GPU} "
        f"--disaggregation-mode prefill --afd-micro-batch {NUM_PF_GROUPS} "
        "--chunked-prefill-size -1 --max-prefill-tokens 16384 "
        f"--afd-multi-pf-continuation --afd-pf-group-count {NUM_PF_GROUPS} "
        f"--afd-pf-group-id 0 --afd-pf-peer-devices {peer_devices} "
        "--afd-pf-channel-base 700 "
        f"--afd-pf-scheduler-endpoints {sched_endpoints} "
        + _common_flags(1, dvfs=dvfs)
    )
    _launch(host, f"{host_tag}_pa", pa_env, pa_flags,
            PA_PORT, 39700, 52450)
    if not base.wait_health(host, PA_PORT, 600):
        raise RuntimeError(f"{host} PA failed health")
    log.info("%s PA ready", host)
    return f"http://{host}:{PA_PORT}"


# --------------- Elastic Decode deployment ---------------

D_PAIR_BASE = [
    {"base_gpu_df": 0, "base_gpu_da": 2, "df_port": 45400,
     "da_port": 45401, "nccl_df": 39800, "nccl_da": 39810,
     "bs_df": 52500, "bs_da": 52510, "sched_port": 60800},
    {"base_gpu_df": 3, "base_gpu_da": 5, "df_port": 45410,
     "da_port": 45411, "nccl_df": 39820, "nccl_da": 39830,
     "bs_df": 52520, "bs_da": 52530, "sched_port": 60900},
    {"base_gpu_df": 6, "base_gpu_da": 8, "df_port": 45420,
     "da_port": 45421, "nccl_df": 39840, "nccl_da": 39850,
     "bs_df": 52540, "bs_da": 52550, "sched_port": 61000},
]


def deploy_decode(n_pairs, dvfs=False, target_d_nodes=None):
    """Deploy n_pairs D-pairs across available D-nodes. Returns DA URLs."""
    da_urls = []
    nodes = target_d_nodes if target_d_nodes else D_NODES

    pairs_deployed = 0
    for d_node in nodes:
        if pairs_deployed >= n_pairs:
            break
        pairs_on_node = min(n_pairs - pairs_deployed, 2)
        # Skip GPU 0 (zombie process on node4), use GPU 1-7
        cvd = ",".join(str(g) for g in range(1, 8))

        for local_idx in range(pairs_on_node):
            pair_template = D_PAIR_BASE[local_idx]
            d_env_base = {
                "SGLANG_HOST_IP": d_node,
                "SGLANG_DISABLE_REQUEST_LOGGING": "true",
                "CUDA_VISIBLE_DEVICES": cvd,
                "AFD_IPC_SYNC_MODE": "ipc_event",
                "AFD_SCHED_PORT": str(pair_template["sched_port"]),
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
                "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
            }
            df_env = {
                **d_env_base,
                "AFD_IPC_PEER_OFFSET": "+2",
                "AFD_NVML_DEVICE_INDICES": (
                    f"{pair_template['base_gpu_df']},"
                    f"{pair_template['base_gpu_df']+1}"
                ),
            }
            df_flags = (
                f"--afd-perspective ffn --base-gpu-id {pair_template['base_gpu_df']} "
                "--disaggregation-mode decode --afd-micro-batch 1 "
                + _common_flags(2, dvfs=dvfs)
            )
            tag = f"d{pairs_deployed}_df"
            _launch(d_node, tag, df_env, df_flags,
                    pair_template["df_port"], pair_template["nccl_df"],
                    pair_template["bs_df"])

            da_env = {
                **d_env_base,
                "AFD_IPC_PEER_OFFSET": "-2",
                "AFD_NVML_DEVICE_INDICES": str(pair_template["base_gpu_da"]),
                "AFD_UCX_FFN_HOST": "127.0.0.1",
            }
            da_flags = (
                f"--afd-perspective attn --base-gpu-id {pair_template['base_gpu_da']} "
                "--disaggregation-mode decode --afd-micro-batch 1 "
                + _common_flags(1, dvfs=dvfs)
            )
            tag = f"d{pairs_deployed}_da"
            _launch(d_node, tag, da_env, da_flags,
                    pair_template["da_port"], pair_template["nccl_da"],
                    pair_template["bs_da"])

            pairs_deployed += 1

        for local_idx in range(pairs_on_node):
            pair_template = D_PAIR_BASE[local_idx]
            if not base.wait_health(d_node, pair_template["df_port"], 600):
                raise RuntimeError(
                    f"DF{local_idx}@{d_node} failed health")
            log.info("DF%d@%s ready", local_idx, d_node)
            if not base.wait_health(d_node, pair_template["da_port"], 600):
                raise RuntimeError(
                    f"DA{local_idx}@{d_node} failed health")
            log.info("DA%d@%s ready", local_idx, d_node)
            da_urls.append(f"http://{d_node}:{pair_template['da_port']}")

    return da_urls


# --------------- Full deploy ---------------

def deploy_full(n_decode_pairs, n_p_nodes=2, dvfs=False, d_nodes=None):
    """Deploy Prefill + Decode (elastic) + Router."""
    p_nodes_used = P_NODES[:n_p_nodes]
    d_nodes_used = d_nodes if d_nodes else D_NODES[:2]
    all_nodes = list(set(p_nodes_used + d_nodes_used))
    base.write_ib_json(all_nodes)

    prefill_urls = [deploy_shared_prefill(host, dvfs=dvfs) for host in p_nodes_used]
    da_urls = deploy_decode(n_decode_pairs, dvfs=dvfs, target_d_nodes=d_nodes_used)

    prefill_args = " ".join(
        f"--prefill {url} 52450" for url in prefill_urls
    )
    decode_args = " ".join(
        f"--decode {url}" for url in da_urls
    )
    router_host = P_NODES[0]
    router_command = (
        f"setsid {base.PYTHON} -m sglang_router.launch_router "
        "--pd-disaggregation "
        "--prefill-policy round_robin "
        f"--host {router_host} --port {ROUTER_PORT} "
        f"{prefill_args} {decode_args} "
        f"> {base.LOG_C}/shared_pa_router.log 2>&1 < /dev/null &"
    )
    base.dexec(router_host, router_command)
    if not base.wait_health(router_host, ROUTER_PORT, 60):
        raise RuntimeError("shared-PA router failed health")
    log.info("Router ready with %d prefill, %d decode",
             len(prefill_urls), len(da_urls))
    return f"http://{router_host}:{ROUTER_PORT}"


# --------------- Main ---------------

def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workload",
        help="Run one trace (e.g. code_qps2) or 'all' for batch run",
    )
    parser.add_argument(
        "--monitor-sm-dir", type=Path,
        default=Path(__file__).resolve().parent / "results" / "shared_pa_3pf_sm",
    )
    parser.add_argument("--max-requests", type=int)
    parser.add_argument(
        "--dvfs", action="store_true",
        help="Enable AFD DVFS (prefill locked max, decode compositional)",
    )
    args = parser.parse_args()

    if args.workload == "all":
        workloads = [
            f"{ds}_qps{qps}"
            for ds in ["code", "conv"]
            for qps in [2, 4, 8, 16]
        ]
    elif args.workload:
        workloads = [args.workload]
    else:
        workloads = []

    tag = "shared_pa_3pf_dvfs" if args.dvfs else "shared_pa_3pf"
    all_results = {}
    all_hosts = list(set(P_NODES + [D_NODES[0]]))  # exclude NODE2 (other user)

    if not workloads:
        base.cleanup_hosts(*all_hosts)
        try:
            url = deploy_full(1, dvfs=args.dvfs)
            time.sleep(2)
            if not base.test_generate(url, timeout=180):
                raise RuntimeError("generate smoke failed")
            print(f"PASS {tag} Mixtral smoke: {url}")
        finally:
            base.cleanup_hosts(*all_hosts)
        return

    for wl_name in workloads:
        qps = int(wl_name.split("qps")[-1])
        cfg = DECODE_CONFIG.get(qps, {"n_pairs": 1, "p_nodes": 1, "d_nodes": [D_NODES[0]]})
        n_pairs = cfg["n_pairs"]
        n_p_nodes = cfg.get("p_nodes", 2)
        d_nodes = cfg.get("d_nodes", [D_NODES[1]])

        log.info("=== %s: %s (p_nodes=%d, decode_pairs=%d) ===",
                 tag, wl_name, n_p_nodes, n_pairs)
        base.cleanup_hosts(*all_hosts)
        time.sleep(3)

        try:
            url = deploy_full(n_pairs, n_p_nodes=n_p_nodes, dvfs=args.dvfs, d_nodes=d_nodes)
            time.sleep(2)
            if not base.test_generate(url, timeout=180):
                log.error("  WARMUP_FAILED: %s", wl_name)
                all_results[wl_name] = {"status": "WARMUP_FAILED"}
                continue

            workload_path = base.WORKLOAD_DIR / f"macro_{wl_name}.jsonl"
            if not workload_path.exists():
                log.warning("  workload not found: %s", workload_path)
                all_results[wl_name] = {"status": "WORKLOAD_MISSING"}
                continue
            with workload_path.open() as f:
                reqs = [json.loads(line) for line in f]
            if args.max_requests:
                reqs = reqs[: args.max_requests]

            d_nodes_used = cfg["d_nodes"]
            p_nodes_used = P_NODES[:n_p_nodes]
            energy_hosts_gpus = [
                (p, list(range(5))) for p in p_nodes_used
            ]
            for d_node in d_nodes_used:
                pairs_on_node = min(n_pairs, 2)
                energy_hosts_gpus.append(
                    (d_node, list(range(pairs_on_node * 3)))
                )

            summary = asyncio.run(
                base.run_workload(
                    reqs,
                    url + "/generate",
                    energy_hosts_gpus,
                    max_run_s=400,
                    sm_monitor_dir=args.monitor_sm_dir,
                    sm_monitor_tag=f"{tag}_{wl_name}",
                )
            )
            all_results[wl_name] = summary
            if summary.get("status") == "PASS":
                log.info("  PASS: thpt=%.1f ttft=%.1f tpot=%.1f E=%.0fmJ/tok",
                         summary["throughput_tok_s"],
                         summary["ttft_proc_p50_ms"],
                         summary["tpot_p50_ms"],
                         summary["energy_per_token_mj"])
            else:
                log.error("  FAIL: %s", summary)
        except Exception as exc:
            log.error("  EXCEPTION: %s %s", wl_name, exc)
            all_results[wl_name] = {"status": "EXCEPTION", "error": str(exc)}
        finally:
            base.cleanup_hosts(*all_hosts)
            time.sleep(5)

    output = (
        base.RESULTS_DIR
        / f"{tag}_batch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(all_results, indent=2))
    print(f"\n{'='*60}")
    print(f"ALL RESULTS ({tag}):")
    for k, v in all_results.items():
        if v.get("status") == "PASS":
            print(f"  {k}: E={v['energy_per_token_mj']:.0f}mJ "
                  f"TTFT={v['ttft_proc_p50_ms']:.0f}ms "
                  f"thpt={v['throughput_tok_s']:.1f}tok/s")
        else:
            print(f"  {k}: {v.get('status', 'UNKNOWN')}")
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
