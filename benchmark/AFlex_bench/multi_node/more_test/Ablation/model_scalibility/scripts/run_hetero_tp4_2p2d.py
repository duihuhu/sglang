#!/usr/bin/env python3
"""Deploy and benchmark 2P(PA1+PF4) + 2D(DA1+DF2) heterogeneous AFD.

Topology (16 GPU, 2 nodes):
  node1:
    GPU0-3: PF0 (FFN TP4, prefill)
    GPU4:   PA0 (Attn TP1, prefill)
    GPU5-6: DF0 (FFN TP2, decode)
    GPU7:   DA0 (Attn TP1, decode)
  node3:
    GPU0-3: PF1 (FFN TP4, prefill)
    GPU4:   PA1 (Attn TP1, prefill)
    GPU5-6: DF1 (FFN TP2, decode)
    GPU7:   DA1 (Attn TP1, decode)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

NODES = [base.NODE1_IP, base.NODE3_IP]
ROUTER_PORT = 42000


def _env(values):
    return "export " + " ".join(
        f"{k}={shlex.quote(str(v))}" for k, v in values.items()
    ) + ";"


def _launch(host, name, env_dict, flags, port, nccl_port, bootstrap_port):
    cmd = (
        f"{_env(env_dict)} setsid prlimit --memlock=unlimited:unlimited "
        f"{base.PYTHON} -m sglang.launch_server "
        f"--host {host} --port {port} --nccl-port {nccl_port} "
        f"--disaggregation-bootstrap-port {bootstrap_port} "
        f"{flags} > {base.LOG_C}/hetero_{name}.log 2>&1 < /dev/null &"
    )
    base.dexec(host, cmd)


def deploy():
    base.write_ib_json(NODES)

    common = (
        f"--model-path {base.MODEL} "
        f"{base.COMMON_FLAGS} "
        "--afd-comm-backend ipc_cpp "
        "--max-running-requests 512 --watchdog-timeout 600 "
        "--num-reserved-decode-tokens 512 "
        "--disaggregation-transfer-backend mooncake "
        f"--disaggregation-ib-device {base.IB_JSON_FILE} "
        "--enable-metrics"
    )

    p_ports = []
    d_ports = []

    for idx, host in enumerate(NODES):
        tag = host.split(".")[-1]
        shared_env = {
            "SGLANG_HOST_IP": host,
            "SGLANG_DISABLE_REQUEST_LOGGING": "true",
            "AFD_IPC_SYNC_MODE": "ipc_event",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "600",
        }

        # --- Prefill: PF(TP4, GPU0-3) + PA(TP1, GPU4) ---
        pf_port = 45400 + idx * 20
        pa_port = 45401 + idx * 20
        pf_env = {
            **shared_env,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4",
            "AFD_IPC_PEER_OFFSET": "+4",
            "AFD_SCHED_PORT": str(60400 + idx * 100),
            "AFD_NVML_DEVICE_INDICES": "0,1,2,3",
        }
        pf_flags = (
            f"--afd-perspective ffn --tp 4 --base-gpu-id 0 "
            f"--afd-attn-tp 1 --afd-ffn-tp 4 "
            f"--disaggregation-mode prefill --afd-micro-batch 1 "
            f"{common}"
        )
        _launch(host, f"{tag}_pf", pf_env, pf_flags, pf_port,
                39400 + idx * 20, 52400 + idx * 20)

        pa_env = {
            **shared_env,
            "CUDA_VISIBLE_DEVICES": "0,1,2,3,4",
            "AFD_IPC_PEER_OFFSET": "-4",
            "AFD_SCHED_PORT": str(60400 + idx * 100),
            "AFD_NVML_DEVICE_INDICES": "4",
            "AFD_UCX_FFN_HOST": "127.0.0.1",
        }
        pa_flags = (
            f"--afd-perspective attn --tp 1 --base-gpu-id 4 "
            f"--afd-attn-tp 1 --afd-ffn-tp 4 "
            f"--disaggregation-mode prefill --afd-micro-batch 1 "
            f"{common}"
        )
        _launch(host, f"{tag}_pa", pa_env, pa_flags, pa_port,
                39401 + idx * 20, 52401 + idx * 20)

        # --- Decode: DF(TP2, GPU5-6) + DA(TP1, GPU7) ---
        df_port = 45410 + idx * 20
        da_port = 45411 + idx * 20
        df_env = {
            **shared_env,
            "CUDA_VISIBLE_DEVICES": "5,6,7",
            "AFD_IPC_PEER_OFFSET": "+2",
            "AFD_SCHED_PORT": str(60600 + idx * 100),
            "AFD_NVML_DEVICE_INDICES": "5,6",
        }
        df_flags = (
            f"--afd-perspective ffn --tp 2 --base-gpu-id 0 "
            f"--afd-attn-tp 1 --afd-ffn-tp 2 "
            f"--disaggregation-mode decode --afd-micro-batch 1 "
            f"{common}"
        )
        _launch(host, f"{tag}_df", df_env, df_flags, df_port,
                39410 + idx * 20, 52410 + idx * 20)

        da_env = {
            **shared_env,
            "CUDA_VISIBLE_DEVICES": "5,6,7",
            "AFD_IPC_PEER_OFFSET": "-2",
            "AFD_SCHED_PORT": str(60600 + idx * 100),
            "AFD_NVML_DEVICE_INDICES": "7",
            "AFD_UCX_FFN_HOST": "127.0.0.1",
        }
        da_flags = (
            f"--afd-perspective attn --tp 1 --base-gpu-id 2 "
            f"--afd-attn-tp 1 --afd-ffn-tp 2 "
            f"--disaggregation-mode decode --afd-micro-batch 1 "
            f"{common}"
        )
        _launch(host, f"{tag}_da", da_env, da_flags, da_port,
                39411 + idx * 20, 52411 + idx * 20)

        p_ports.append((host, pa_port, 52401 + idx * 20))
        d_ports.append((host, da_port))

    # Wait for health
    for host, port, _ in p_ports:
        if not base.wait_health(host, port, 600):
            raise RuntimeError(f"PA {host}:{port} failed health")
        log.info("PA %s:%d ready", host, port)
    for host, port in d_ports:
        if not base.wait_health(host, port, 600):
            raise RuntimeError(f"DA {host}:{port} failed health")
        log.info("DA %s:%d ready", host, port)

    # Router
    prefill_args = " ".join(
        f"--prefill http://{h}:{p} {bs}" for h, p, bs in p_ports
    )
    decode_args = " ".join(
        f"--decode http://{h}:{p}" for h, p in d_ports
    )
    router_host = NODES[0]
    router_cmd = (
        f"setsid {base.PYTHON} -m sglang_router.launch_router "
        f"--pd-disaggregation --host {router_host} --port {ROUTER_PORT} "
        f"{prefill_args} {decode_args} "
        f"> {base.LOG_C}/hetero_router.log 2>&1 < /dev/null &"
    )
    base.dexec(router_host, router_cmd)
    if not base.wait_health(router_host, ROUTER_PORT, 60):
        raise RuntimeError("Router failed health")
    log.info("Router ready")
    return f"http://{router_host}:{ROUTER_PORT}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", help="e.g. code_qps16, conv_qps4")
    parser.add_argument(
        "--monitor-sm-dir", type=Path,
        default=Path(__file__).resolve().parent / "results" / "hetero_tp4_sm",
    )
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()

    base.cleanup_hosts(*NODES)
    try:
        url = deploy()
        time.sleep(2)
        if not base.test_generate(url, timeout=180):
            raise RuntimeError("generate smoke failed")
        print(f"PASS hetero TP4 smoke: {url}")

        if args.workload:
            wl_path = base.WORKLOAD_DIR / f"macro_{args.workload}.jsonl"
            with wl_path.open() as f:
                reqs = [json.loads(line) for line in f]
            if args.max_requests:
                reqs = reqs[:args.max_requests]

            energy_hosts_gpus = [
                (NODES[0], list(range(8))),
                (NODES[1], list(range(8))),
            ]
            summary = asyncio.run(
                base.run_workload(
                    reqs, url + "/generate", energy_hosts_gpus,
                    max_run_s=400,
                    sm_monitor_dir=args.monitor_sm_dir,
                    sm_monitor_tag=f"hetero_tp4_{args.workload}",
                )
            )
            out = (
                base.RESULTS_DIR
                / f"hetero_tp4_{args.workload}_{time.strftime('%Y%m%d_%H%M%S')}.json"
            )
            out.write_text(json.dumps(summary, indent=2))
            print(f"RESULT {json.dumps(summary, sort_keys=True)}")
            print(f"Saved: {out}")
    finally:
        base.cleanup_hosts(*NODES)


if __name__ == "__main__":
    main()
