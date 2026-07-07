#!/usr/bin/env python3
"""MoE Heterogeneous PDAF benchmark for Mixtral-8x7B.

Profile data shows:
  - Decode:  F/A latency ≈ 1.50x
  - Prefill: F/A latency ≈ 2.33x

Test heterogeneous TP configurations:
  - homo:  tp_a=4, tp_f=4 (16 GPUs, baseline same as before)
  - het12: tp_a=2, tp_f=4 (12 GPUs, A:F = 1:2)

Usage:
    python3 run_moe_hetero_pdaf.py --configs homo,het12 --mode all --scenario all --qps 1,3,5,7,9
"""
import argparse
import asyncio
import json
import logging
import os
import shlex
import subprocess
import time
from pathlib import Path

import aiohttp
import numpy as np
import requests

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("moe_hetero")

NODE1_IP = os.environ.get("MN_NODE1_IP", "10.252.129.36")
NODE2_IP = os.environ.get("MN_NODE2_IP", "10.252.129.35")
CONTAINER = os.environ.get("MN_CONTAINER", "operator_test")
PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"
CLEANUP = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"

MAX_GPU_FREQ = 1410
ENERGY_MODEL_DIR_V2 = ("/workspace/sglang/benchmark/AFlex_bench/06_others/"
                       "Mixtral_test/energy_model/models_v2")

ROUTER_PORT = 42000
PA_PORT, PF_PORT = 42010, 42011
DA_PORT, DF_PORT = 42020, 42021
BS_PORT = 49999
IB_JSON_FILE = "/tmp/ib_hetero_map.json"

TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0

GPU_NIC = {0: "mlx5_0", 1: "mlx5_0", 2: "mlx5_1", 3: "mlx5_1",
           4: "mlx5_4", 5: "mlx5_4", 6: "mlx5_5", 7: "mlx5_5"}

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results_hetero"
WORKLOAD_DIR = Path("/mnt/workspace/lt/sglang/benchmark/AFlex_bench/retesting/workloads")
LOG_C = "/workspace/sglang/benchmark/AFlex_bench/multi_node/logs_hetero"


def _ssh(host, cmd):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", host, cmd]


def dexec_local(cmd):
    subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", cmd], check=False)


def dexec_remote(cmd):
    inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(cmd)}"
    subprocess.run(_ssh(NODE2_IP, inner), check=False)


def write_ib_json(mapping):
    content = json.dumps(mapping)
    for host in [NODE1_IP, NODE2_IP]:
        cmd = f"echo '{content}' > {IB_JSON_FILE}"
        if host == NODE1_IP:
            dexec_local(cmd)
        else:
            dexec_remote(cmd)
    time.sleep(1)


def cleanup_all():
    log.info("Cleaning up all nodes...")
    for host in [NODE1_IP, NODE2_IP]:
        cmd = f"bash {CLEANUP} 2>/dev/null; pkill -f sglang 2>/dev/null; pkill -f sglang_router 2>/dev/null; sleep 2"
        if host == NODE1_IP:
            dexec_local(cmd)
        else:
            dexec_remote(cmd)
    time.sleep(8)


def wait_health(host, port, timeout=600, check_model_info=False):
    ep = "get_model_info" if check_model_info else "health"
    url = f"http://{host}:{port}/{ep}"
    end = time.time() + timeout
    while time.time() < end:
        try:
            r = requests.get(url, timeout=10)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(3)
    return False


def get_energy_local(gpus):
    try:
        import pynvml
        pynvml.nvmlInit()
        res = {i: pynvml.nvmlDeviceGetTotalEnergyConsumption(
            pynvml.nvmlDeviceGetHandleByIndex(i)) for i in gpus}
        pynvml.nvmlShutdown()
        return res
    except Exception as e:
        log.warning("local NVML energy read failed: %s", e)
        return {i: 0 for i in gpus}


def get_energy_remote(gpus):
    idx_csv = ",".join(str(i) for i in gpus)
    pycode = (
        "import pynvml,json;pynvml.nvmlInit();"
        f"idxs=[int(x) for x in '{idx_csv}'.split(',')];"
        "print(json.dumps({i:pynvml.nvmlDeviceGetTotalEnergyConsumption("
        "pynvml.nvmlDeviceGetHandleByIndex(i)) for i in idxs}));"
        "pynvml.nvmlShutdown()"
    )
    inner = f"{PYTHON} -c {shlex.quote(pycode)}"
    try:
        out = subprocess.run(_ssh(NODE2_IP, inner), capture_output=True,
                            text=True, timeout=30)
        line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
        return {int(k): v for k, v in json.loads(line[-1]).items()} if line else \
               {i: 0 for i in gpus}
    except Exception as e:
        log.warning("remote NVML energy read failed: %s", e)
        return {i: 0 for i in gpus}


def lock_freq_both(gpus, freq=MAX_GPU_FREQ):
    cmd = ";".join(f"nvidia-smi -i {i} --lock-gpu-clocks={freq},{freq}"
                   for i in gpus) + ";true"
    dexec_local(cmd)
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc {shlex.quote(cmd)}"),
                   check=False, capture_output=True)
    log.info("  Locked GPUs %s to %d MHz on both nodes", gpus, freq)


def unlock_freq_both(gpus):
    cmd = ";".join(f"nvidia-smi -i {i} --reset-gpu-clocks" for i in gpus) + ";true"
    dexec_local(cmd)
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc {shlex.quote(cmd)}"),
                   check=False, capture_output=True)


def start_hetero_pdaf(tp_a, tp_f, tier=False):
    """Start heterogeneous PDAF deployment."""
    if tp_a == 2 and tp_f == 4:
        attn_gpus = [0, 2]
        ffn_gpus = [1, 3, 5, 7]
        attn_base, ffn_base = 0, 1
        gpu_step = 2
    elif tp_a == 4 and tp_f == 4:
        attn_gpus = [0, 2, 4, 6]
        ffn_gpus = [1, 3, 5, 7]
        attn_base, ffn_base = 0, 1
        gpu_step = 2
    elif tp_a == 2 and tp_f == 2:
        attn_gpus = [0, 2]
        ffn_gpus = [4, 6]
        attn_base, ffn_base = 0, 4
        gpu_step = 2
    else:
        log.error("Unsupported tp_a=%d tp_f=%d", tp_a, tp_f)
        return None, []

    all_gpus = sorted(set(attn_gpus + ffn_gpus))
    ib_map = {str(g): GPU_NIC[g] for g in range(8)}
    write_ib_json(ib_map)

    log.info("Hetero PDAF: tp_a=%d, tp_f=%d, A_GPUs=%s, F_GPUs=%s, tier=%s",
             tp_a, tp_f, attn_gpus, ffn_gpus, tier)

    dvfs_flags = ""
    if tier:
        dvfs_flags = (f" --afd-dvfs-enabled "
                      f"--afd-energy-model-dir {ENERGY_MODEL_DIR_V2} "
                      f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)} "
                      f"--afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)} "
                      "--afd-dvfs-idle-lock")

    grouped_flag = "--afd-grouped-stepmesh " if tp_a != tp_f else ""
    common = (f"--model-path {MODEL} "
              f"--afd-attn-tp {tp_a} --afd-ffn-tp {tp_f} {grouped_flag}"
              "--afd-comm-backend ipc_cpp "
              f"--afd-micro-batch 2 --afd-dynamic-micro-batch --mem-fraction-static 0.85 "
              f"--max-running-requests 512 --skip-server-warmup "
              "--watchdog-timeout 600 "
              "--disable-cuda-graph --disable-piecewise-cuda-graph "
              "--afd-disagg-interleave-poll --disable-radix-cache "
              "--num-reserved-decode-tokens 512 "
              "--disaggregation-transfer-backend mooncake "
              f"--disaggregation-bootstrap-port {BS_PORT} "
              f"--disaggregation-ib-device {IB_JSON_FILE} --enable-metrics"
              f"{dvfs_flags}")

    cvd = "0,1,2,3,4,5,6,7"
    base_env = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
                "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
                "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
                "AFD_IPC_SYNC_MODE=ipc_event "
                "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
                "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
                "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
                f"CUDA_VISIBLE_DEVICES={cvd} ")

    # Create log dir inside containers on both nodes
    dexec_local(f"mkdir -p {LOG_C}")
    dexec_remote(f"mkdir -p {LOG_C}")
    time.sleep(2)

    # Node1: PF
    env_pf = base_env + f"AFD_NVML_DEVICE_INDICES={','.join(map(str,ffn_gpus))}; "
    cmd_pf = (f"{env_pf}setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
              f"-m sglang.launch_server --host {NODE1_IP} --port {PF_PORT} "
              f"--tp {tp_f} --gpu-id-step {gpu_step} --base-gpu-id {ffn_base} "
              f"--afd-perspective ffn --disaggregation-mode prefill "
              f"{common} > {LOG_C}/pf.log 2>&1 < /dev/null &")
    dexec_local(cmd_pf)
    time.sleep(6)

    # Node1: PA
    env_pa = base_env + f"AFD_NVML_DEVICE_INDICES={','.join(map(str,attn_gpus))}; "
    cmd_pa = (f"{env_pa}setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
              f"-m sglang.launch_server --host {NODE1_IP} --port {PA_PORT} "
              f"--tp {tp_a} --gpu-id-step {gpu_step} --base-gpu-id {attn_base} "
              f"--afd-perspective attn --disaggregation-mode prefill "
              f"{common} > {LOG_C}/pa.log 2>&1 < /dev/null &")
    dexec_local(cmd_pa)
    time.sleep(6)

    # Node2: DF
    env_df = base_env + f"AFD_NVML_DEVICE_INDICES={','.join(map(str,ffn_gpus))}; "
    cmd_df = (f"{env_df}setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
              f"-m sglang.launch_server --host {NODE2_IP} --port {DF_PORT} "
              f"--tp {tp_f} --gpu-id-step {gpu_step} --base-gpu-id {ffn_base} "
              f"--afd-perspective ffn --disaggregation-mode decode "
              f"{common} > {LOG_C}/df.log 2>&1 < /dev/null &")
    dexec_remote(cmd_df)
    time.sleep(8)

    # Node2: DA
    env_da = base_env + f"AFD_NVML_DEVICE_INDICES={','.join(map(str,attn_gpus))}; "
    cmd_da = (f"{env_da}setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
              f"-m sglang.launch_server --host {NODE2_IP} --port {DA_PORT} "
              f"--tp {tp_a} --gpu-id-step {gpu_step} --base-gpu-id {attn_base} "
              f"--afd-perspective attn --disaggregation-mode decode "
              f"{common} > {LOG_C}/da.log 2>&1 < /dev/null &")
    dexec_remote(cmd_da)

    log.info("Waiting for hetero PDAF servers...")
    for host, port, name, mi in [(NODE1_IP, PF_PORT, "PF", True),
                                 (NODE1_IP, PA_PORT, "PA", False),
                                 (NODE2_IP, DF_PORT, "DF", True),
                                 (NODE2_IP, DA_PORT, "DA", False)]:
        if not wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed", name, host, port)
            return None, []
        log.info("  %s ready", name)

    # Router
    rc = (f"setsid {PYTHON} -m sglang_router.launch_router --pd-disaggregation --mini-lb "
          f"--prefill http://{NODE1_IP}:{PA_PORT} --decode http://{NODE2_IP}:{DA_PORT} "
          f"--host {NODE1_IP} --port {ROUTER_PORT} > {LOG_C}/router.log 2>&1 < /dev/null &")
    dexec_local(rc)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  router failed")
        return None, []
    log.info("  router ready")
    return f"http://{NODE1_IP}:{ROUTER_PORT}", all_gpus




async def send_one(session, url, req, base_time, collector):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"text": "x" * req["input_len"],
               "sampling_params": {"max_new_tokens": req["output_len"],
                                   "temperature": 0.0, "ignore_eos": True},
               "stream": True}
    t0 = time.monotonic()
    first_token_time = None
    token_count = 0
    last_meta = {}
    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                collector.append({"success": False, "error": f"status={resp.status}"})
                return
            async for line in resp.content:
                now = time.monotonic()
                text = line.decode().strip()
                if not text or text.startswith(":"):
                    continue
                if text.startswith("data:"):
                    text = text[5:].strip()
                if text == "[DONE]":
                    break
                try:
                    chunk = json.loads(text)
                    if first_token_time is None:
                        first_token_time = now
                    token_count += 1
                    if isinstance(chunk, dict) and "meta_info" in chunk:
                        last_meta = chunk["meta_info"]
                except json.JSONDecodeError:
                    pass
    except Exception as e:
        collector.append({"success": False, "error": str(e)})
        return

    t_end = time.monotonic()
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0
    ttft_proc_ms = 0.0
    if last_meta.get("time_to_first_token_processing"):
        ttft_proc_ms = last_meta["time_to_first_token_processing"] * 1000
    elif first_token_time:
        ttft_proc_ms = ttft_ms
    tpot_ms = 0.0
    if token_count > 1 and first_token_time:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)
    collector.append({
        "success": True, "ttft_ms": ttft_ms, "ttft_proc_ms": ttft_proc_ms,
        "tpot_ms": tpot_ms, "completion_tokens": token_count,
    })


async def run_workload(reqs, url, n1_gpus, n2_gpus, max_run_s=400):
    e1s = get_energy_local(n1_gpus)
    e2s = get_energy_remote(n2_gpus)
    results = []
    timeout = aiohttp.ClientTimeout(total=max_run_s + 60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        base_time = time.monotonic()
        tasks = [asyncio.create_task(send_one(session, url, r, base_time, results))
                 for r in reqs]
        if tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout=max_run_s)
            except asyncio.TimeoutError:
                log.warning("Timed out after %ds", max_run_s)
    duration_s = time.monotonic() - base_time
    e1e = get_energy_local(n1_gpus)
    e2e = get_energy_remote(n2_gpus)
    energy_n1_j = sum((e1e.get(i, 0) - e1s.get(i, 0)) / 1000.0 for i in n1_gpus)
    energy_n2_j = sum((e2e.get(i, 0) - e2s.get(i, 0)) / 1000.0 for i in n2_gpus)
    total_energy_j = energy_n1_j + energy_n2_j

    ok = [r for r in results if r.get("success")]
    if not ok:
        return {"status": "FAIL", "failed": len(results)}

    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0
    src = ttfts_proc if ttfts_proc else []
    n_ttft_viol = sum(1 for v in src if v > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for r in ok if r["tpot_ms"] > TPOT_SLO_MS)
    slo_rate = (n_ttft_viol + n_tpot_viol + (len(results) - len(ok))) / len(results) * 100

    return {
        "status": "PASS", "duration_s": round(duration_s, 1),
        "total_requests": len(reqs), "successful": len(ok), "failed": len(results) - len(ok),
        "total_tokens": total_tokens, "throughput_tok_s": round(throughput, 1),
        "ttft_proc_avg_ms": round(float(np.mean(src)), 1) if src else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(src, 50)), 1) if src else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(src, 99)), 1) if src else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "energy_node1_j": round(energy_n1_j, 1),
        "energy_node2_j": round(energy_n2_j, 1),
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens else 0,
        "slo_violation_rate": round(slo_rate, 1),
    }


def run_one_workload(url, scenario, qps, n1_gpus, n2_gpus, max_run_s):
    wl_file = WORKLOAD_DIR / f"micro_{scenario}_qps{qps}.jsonl"
    if not wl_file.exists():
        log.warning("  workload not found: %s", wl_file)
        return None
    with open(wl_file) as f:
        reqs = [json.loads(l) for l in f]
    n_reqs = len(reqs)
    run_s = min(max_run_s, max(120, n_reqs / max(qps, 1) * 1.5))
    log.info("-" * 50)
    log.info("  %s_qps%d (%d reqs, run_window=%.0fs)", scenario, qps, n_reqs, run_s)
    summary = asyncio.run(run_workload(reqs, url + "/generate", n1_gpus, n2_gpus, run_s))
    if summary["status"] == "PASS":
        log.info("  Thpt=%.1f tok/s | TTFT=%.1fms | TPOT=%.1fms | E=%.0fJ (%.1f mJ/tok) | SLO=%.1f%%",
                 summary["throughput_tok_s"], summary["ttft_proc_avg_ms"],
                 summary["tpot_avg_ms"], summary["total_energy_j"],
                 summary["energy_per_token_mj"], summary["slo_violation_rate"])
    else:
        log.info("  FAIL")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", default="homo,het12",
                        help="Comma-sep: homo (tp4:tp4), het12 (tp2:tp4)")
    parser.add_argument("--mode", default="all", help="baseline,tier or all")
    parser.add_argument("--scenario", default="all",
                        help="chatbot,qa,rag,summary or all")
    parser.add_argument("--qps", default="1,3,5,7,9,11,13",
                        help="Comma-sep QPS values")
    parser.add_argument("--max-run-s", type=int, default=400)
    args = parser.parse_args()

    configs = args.configs.split(",") if args.configs != "all" else ["homo", "het12"]
    modes = args.mode.split(",") if args.mode != "all" else ["baseline", "tier"]
    scenarios = args.scenario.split(",") if args.scenario != "all" else ["chatbot", "qa", "rag", "summary"]
    qps_list = [int(x) for x in args.qps.split(",")]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    CONFIG_MAP = {
        "homo": (4, 4),
        "het12": (2, 4),
    }

    all_results = {}

    for config_name in configs:
        tp_a, tp_f = CONFIG_MAP[config_name]
        for mode in modes:
            tier = (mode == "tier")
            scheme_key = f"pdaf_{config_name}_{mode}"

            log.info("=" * 80)
            log.info("DEPLOY: %s [tp_a=%d, tp_f=%d, tier=%s]", scheme_key, tp_a, tp_f, tier)

            cleanup_all()
            url, used_gpus = start_hetero_pdaf(tp_a, tp_f, tier=tier)
            if url is None:
                log.error("  Deployment FAILED for %s", scheme_key)
                all_results[scheme_key] = {"_deploy_error": True}
                continue

            n1_gpus = list(range(8))
            n2_gpus = list(range(8))

            if not tier:
                lock_freq_both(n1_gpus, MAX_GPU_FREQ)

            # Warmup
            log.info("Warmup...")
            try:
                requests.post(f"{url}/generate", json={
                    "text": "Hello world",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 10}
                }, timeout=60)
            except:
                pass
            time.sleep(3)

            results_this = {}
            for scenario in scenarios:
                for qps in qps_list:
                    key = f"{scenario}_qps{qps}"
                    result = run_one_workload(url, scenario, qps, n1_gpus, n2_gpus, args.max_run_s)
                    if result:
                        results_this[key] = result

            all_results[scheme_key] = results_this
            unlock_freq_both(n1_gpus)

    # Save
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"hetero_{ts}.json"
    with open(out_file, "w") as f:
        json.dump({"results": all_results, "timestamp": ts}, f, indent=2)
    log.info("Results saved to %s", out_file)

    # Summary table
    log.info("=" * 100)
    log.info("%-28s %-12s %-6s %-8s %-8s %-8s %-10s", 
             "Scheme", "Scenario", "QPS", "Thpt", "TTFT", "TPOT", "E(mJ/tok)")
    log.info("-" * 100)
    for scheme_key in sorted(all_results.keys()):
        entries = all_results[scheme_key]
        if not isinstance(entries, dict) or "_deploy_error" in entries:
            continue
        for key in sorted(entries.keys()):
            val = entries[key]
            if isinstance(val, dict) and val.get("status") == "PASS":
                parts = key.rsplit("_qps", 1)
                log.info("%-28s %-12s %-6s %-8.1f %-8.1f %-8.1f %-10.1f",
                         scheme_key, parts[0], parts[1],
                         val["throughput_tok_s"], val["ttft_proc_avg_ms"],
                         val["tpot_avg_ms"], val["energy_per_token_mj"])


if __name__ == "__main__":
    main()
