#!/usr/bin/env python3
"""Extreme prefill workload: in=16384, out=16 to stress RDMA KV transfer.

KV cache per request: 16384 tokens × 64 layers × 2(k+v) × 1024 bytes ≈ 2 GB
At QPS=2, that's 4 GB/s sustained cross-node transfer demand.

Usage:
    python3 bench_extreme_prefill.py --deploy affine --qps 0.5,1,1.5,2
    python3 bench_extreme_prefill.py --deploy all --qps 0.5,1,1.5,2
"""
import argparse
import asyncio
import json
import logging
import os
import shlex
import subprocess
import tempfile
import time
from pathlib import Path

import aiohttp
import numpy as np
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("extreme_bench")

NODE1_IP = os.environ.get("MN_NODE1_IP", "10.252.129.36")
NODE2_IP = os.environ.get("MN_NODE2_IP", "10.252.129.35")
CONTAINER = os.environ.get("MN_CONTAINER", "operator_test")
PYTHON = "/usr/bin/python3"
MODEL = "/models/Qwen3-32B/"
TP = 4

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
CHARTS_DIR = HERE / "charts"
LOG_C = "/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"
CLEANUP = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"

MAX_GPU_FREQ = 1410
PA_PORT, PF_PORT = 42010, 42011
DA_PORT, DF_PORT = 42020, 42021
ROUTER_PORT = 42000
BS_PORT = 49999

IB_JSON_AFFINE = '{"0":"mlx5_0","1":"mlx5_0","2":"mlx5_1","3":"mlx5_1","4":"mlx5_4","5":"mlx5_4","6":"mlx5_5","7":"mlx5_5"}'
IB_JSON_FILE = "/tmp/ib_affine_map.json"
IB_DEV_NONAFFINE = "mlx5_bond_0"

# Extreme workload params
IN_LEN = 1024
OUT_LEN = 8


def _ssh(host, cmd):
    return ["ssh", "-o", "StrictHostKeyChecking=no", "-o", "BatchMode=yes", host, cmd]


def dexec_local(shell_cmd):
    subprocess.run(["docker", "exec", CONTAINER, "bash", "-lc", shell_cmd], check=False)


def dexec_remote(shell_cmd):
    inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(shell_cmd)}"
    subprocess.run(_ssh(NODE2_IP, inner), check=False)


def cleanup_all():
    dexec_local(f"bash {CLEANUP}")
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc 'bash {CLEANUP}'"),
                   check=False)
    time.sleep(5)


def wait_health(host, port, timeout=600, check_model_info=False):
    deadline = time.time() + timeout
    ep = "get_model_info" if check_model_info else "health"
    while time.time() < deadline:
        try:
            r = requests.get(f"http://{host}:{port}/{ep}", timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(3)
    return False


def lock_freq(freq=MAX_GPU_FREQ):
    cmd = ";".join(f"nvidia-smi -i {i} --lock-gpu-clocks={freq},{freq}"
                   for i in range(8)) + ";true"
    dexec_local(cmd)
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc {shlex.quote(cmd)}"),
                   check=False)


def unlock_freq():
    cmd = ";".join(f"nvidia-smi -i {i} --reset-gpu-clocks" for i in range(8)) + ";true"
    dexec_local(cmd)
    subprocess.run(_ssh(NODE2_IP, f"docker exec {CONTAINER} bash -lc {shlex.quote(cmd)}"),
                   check=False)


def _base_env(ucx_port, sched_port, ipc_offset, nvml_indices, nvml_index,
              ffn_host=None):
    env = ("export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
           "AFD_UCX_TLS=rc,tcp,cuda_copy,cuda_ipc "
           "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE=128 AFD_ASYNC_PIPELINE=1 "
           "AFD_IPC_SYNC_MODE=ipc_event "
           "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=600 "
           "SGLANG_DISAGGREGATION_WAITING_TIMEOUT=600 "
           "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE=0 "
           "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "
           f"AFD_UCX_BASE_PORT={ucx_port} AFD_SCHED_PORT={sched_port} "
           f"AFD_IPC_PEER_OFFSET={ipc_offset} "
           f"AFD_NVML_DEVICE_INDICES={nvml_indices} AFD_NVML_DEVICE_INDEX={nvml_index}")
    if ffn_host:
        env += f" AFD_UCX_FFN_HOST={ffn_host}"
    return env + ";"


def _common_flags(ib_dev, gpu_step=None):
    flags = (f"--model-path {MODEL} --tp {TP} --afd-comm-backend ipc_cpp "
             "--afd-micro-batch 2 --afd-dynamic-micro-batch --mem-fraction-static 0.85 "
             "--max-running-requests 512 --skip-server-warmup "
             "--disable-cuda-graph --disable-piecewise-cuda-graph "
             "--afd-disagg-interleave-poll --disable-radix-cache "
             "--num-reserved-decode-tokens 512 "
             "--disaggregation-transfer-backend mooncake "
             f"--disaggregation-bootstrap-port {BS_PORT} "
             f"--disaggregation-ib-device {ib_dev} --enable-metrics")
    if gpu_step:
        flags += f" --gpu-id-step {gpu_step}"
    return flags


def _launch_server(host, port, perspective, disagg_mode, base_gpu, logname,
                   env_str, common_flags):
    cmd = (f"{env_str} setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
           f"-m sglang.launch_server --host {host} --port {port} "
           f"--afd-perspective {perspective} --disaggregation-mode {disagg_mode} "
           f"--base-gpu-id {base_gpu} {common_flags} "
           f"> {LOG_C}/{logname}.log 2>&1 < /dev/null &")
    if host == NODE1_IP:
        dexec_local(cmd)
    else:
        dexec_remote(cmd)


def _write_ib_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(IB_JSON_AFFINE)
        tmp = f.name
    subprocess.run(["docker", "cp", tmp, f"{CONTAINER}:{IB_JSON_FILE}"], check=False)
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-q",
                    tmp, f"{NODE2_IP}:/tmp/ib_affine_map.json"], check=False)
    subprocess.run(_ssh(NODE2_IP, f"docker cp /tmp/ib_affine_map.json {CONTAINER}:{IB_JSON_FILE}"),
                   check=False)
    os.unlink(tmp)


def start_pdaf_nonaffine():
    log.info("Deploying PDAF non-affine (continuous GPU layout)")
    cf = _common_flags(IB_DEV_NONAFFINE)
    env = _base_env(28200, 68400, TP, "0,1,2,3", 0)
    _launch_server(NODE1_IP, PF_PORT, "ffn", "prefill", 0, "pf_nonaffine", env, cf)
    time.sleep(6)
    env = _base_env(28200, 68400, -TP, "4,5,6,7", 4, ffn_host="127.0.0.1")
    _launch_server(NODE1_IP, PA_PORT, "attn", "prefill", TP, "pa_nonaffine", env, cf)
    env = _base_env(28300, 68500, TP, "0,1,2,3", 0)
    _launch_server(NODE2_IP, DF_PORT, "ffn", "decode", 0, "df_nonaffine", env, cf)
    time.sleep(8)
    env = _base_env(28300, 68500, -TP, "4,5,6,7", 4, ffn_host="127.0.0.1")
    _launch_server(NODE2_IP, DA_PORT, "attn", "decode", TP, "da_nonaffine", env, cf)
    return _wait_and_start_router()


def start_pdaf_affine():
    log.info("Deploying PDAF affine (interleaved GPU layout)")
    _write_ib_json()
    cf = _common_flags(IB_JSON_FILE, gpu_step=2)
    env = _base_env(28200, 68400, -1, "1,3,5,7", 1)
    _launch_server(NODE1_IP, PF_PORT, "ffn", "prefill", 1, "pf_affine", env, cf)
    time.sleep(6)
    env = _base_env(28200, 68400, 1, "0,2,4,6", 0, ffn_host="127.0.0.1")
    _launch_server(NODE1_IP, PA_PORT, "attn", "prefill", 0, "pa_affine", env, cf)
    env = _base_env(28300, 68500, -1, "1,3,5,7", 1)
    _launch_server(NODE2_IP, DF_PORT, "ffn", "decode", 1, "df_affine", env, cf)
    time.sleep(8)
    env = _base_env(28300, 68500, 1, "0,2,4,6", 0, ffn_host="127.0.0.1")
    _launch_server(NODE2_IP, DA_PORT, "attn", "decode", 0, "da_affine", env, cf)
    return _wait_and_start_router()


def _wait_and_start_router():
    log.info("Waiting for PDAF servers...")
    checks = [(NODE1_IP, PF_PORT, "PF", True), (NODE1_IP, PA_PORT, "PA", False),
              (NODE2_IP, DF_PORT, "DF", True), (NODE2_IP, DA_PORT, "DA", False)]
    for host, port, name, mi in checks:
        if not wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed", name, host, port)
            return None
        log.info("  %s ready", name)
    router_cmd = (f"setsid {PYTHON} -m sglang_router.launch_router "
                  "--pd-disaggregation --mini-lb "
                  f"--prefill http://{NODE1_IP}:{PA_PORT} "
                  f"--decode http://{NODE2_IP}:{DA_PORT} "
                  f"--host {NODE1_IP} --port {ROUTER_PORT} "
                  f"> {LOG_C}/router.log 2>&1 < /dev/null &")
    dexec_local(router_cmd)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  router failed")
        return None
    log.info("  router ready")
    return f"http://{NODE1_IP}:{ROUTER_PORT}"


def generate_workload(qps, duration_s=90):
    rng = np.random.default_rng(42)
    n_requests = max(int(qps * duration_s), 1)
    inter_arrivals = rng.exponential(1.0 / qps, n_requests) if qps > 0 else [0]
    arrivals = np.cumsum(inter_arrivals)
    return [{"input_len": IN_LEN, "output_len": OUT_LEN,
             "arrival_time_s": float(t)} for t in arrivals]


async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    prompt = "Analyze the following long document comprehensively:\n" + "x " * (req["input_len"] // 2)
    payload = {"text": prompt,
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
                results.append({"success": False})
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
    except Exception:
        results.append({"success": False})
        return

    t_end = time.monotonic()
    ttft_ms = (first_token_time - t0) * 1000 if first_token_time else 0
    ttft_proc_ms = 0.0
    if last_meta.get("ttft_pure_processing"):
        ttft_proc_ms = last_meta["ttft_pure_processing"] * 1000
    elif last_meta.get("time_to_first_token_processing"):
        ttft_proc_ms = last_meta["time_to_first_token_processing"] * 1000
    tpot_ms = 0.0
    if token_count > 1 and first_token_time:
        tpot_ms = (t_end - first_token_time) * 1000 / (token_count - 1)
    results.append({
        "success": True, "completion_tokens": token_count,
        "ttft_ms": ttft_ms, "ttft_proc_ms": ttft_proc_ms,
        "tpot_ms": tpot_ms, "e2e_s": t_end - t0,
    })


async def run_workload(reqs, url, max_run_s=600):
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

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    if not ok:
        return {"status": "FAIL", "failed": len(fail), "successful": 0}

    ttfts = [r["ttft_ms"] for r in ok]
    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0
    ttft_src = ttfts_proc if ttfts_proc else ttfts

    return {
        "status": "PASS", "duration_s": round(duration_s, 1),
        "total_requests": len(reqs), "successful": len(ok), "failed": len(fail),
        "total_tokens": total_tokens, "throughput_tok_s": round(throughput, 1),
        "ttft_avg_ms": round(float(np.mean(ttft_src)), 1),
        "ttft_p50_ms": round(float(np.percentile(ttft_src, 50)), 1),
        "ttft_p99_ms": round(float(np.percentile(ttft_src, 99)), 1),
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
    }


DEPLOY_FNS = {"nonaffine": start_pdaf_nonaffine, "affine": start_pdaf_affine}


def test_generate(url):
    try:
        r = requests.post(url + "/generate", json={
            "text": "Hello",
            "sampling_params": {"max_new_tokens": 8, "temperature": 0.0}}, timeout=120)
        return "text" in r.json()
    except Exception as e:
        log.error("warmup failed: %s", e)
        return False


def run_deploy(deploy_name, qps_list, duration_s):
    log.info("=" * 70)
    log.info("DEPLOY: pdaf_%s | extreme prefill (in=%d, out=%d)", deploy_name, IN_LEN, OUT_LEN)
    log.info("=" * 70)
    cleanup_all()
    url = DEPLOY_FNS[deploy_name]()
    if url is None:
        cleanup_all()
        return {"__status__": "DEPLOY_FAILED"}
    lock_freq(MAX_GPU_FREQ)
    if not test_generate(url):
        unlock_freq(); cleanup_all()
        return {"__status__": "WARMUP_FAILED"}
    time.sleep(3)

    results = {}
    for qps in qps_list:
        log.info("-" * 50)
        log.info("  QPS=%.1f (duration=%ds, in=%d, out=%d)", qps, duration_s, IN_LEN, OUT_LEN)
        reqs = generate_workload(qps, duration_s)
        last_arr = max(r["arrival_time_s"] for r in reqs)
        run_s = int(min(max(600, last_arr + 200), 900))
        summary = asyncio.run(run_workload(reqs, url + "/generate", run_s))
        results[f"qps_{qps}"] = summary
        if summary.get("status") == "PASS":
            log.info("  Thpt=%.1f tok/s | TTFT_avg=%.1fms p99=%.1fms | "
                     "TPOT_avg=%.1fms | ok=%d fail=%d",
                     summary["throughput_tok_s"], summary["ttft_avg_ms"],
                     summary["ttft_p99_ms"], summary["tpot_avg_ms"],
                     summary["successful"], summary["failed"])
        else:
            log.error("  FAIL: %s", summary)
        time.sleep(5)

    unlock_freq(); cleanup_all()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--deploy", default="all")
    parser.add_argument("--qps", default="0.5,1,1.5,2,3")
    parser.add_argument("--duration", type=int, default=90)
    args = parser.parse_args()

    deploys = list(DEPLOY_FNS.keys()) if args.deploy == "all" else args.deploy.split(",")
    qps_list = [float(q) for q in args.qps.split(",")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for d in deploys:
        all_results[f"pdaf_{d}"] = run_deploy(d, qps_list, args.duration)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"extreme_prefill_{ts}.json"
    payload = {"meta": {"workload": f"extreme prefill (in={IN_LEN}, out={OUT_LEN})",
                        "model": MODEL, "tp": TP, "deploys": deploys, "qps": qps_list},
               "results": all_results}
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Results saved: %s", out_file)

    print(f"\n{'='*90}")
    print(f"  EXTREME PREFILL BENCHMARK (in={IN_LEN}, out={OUT_LEN})")
    print(f"{'='*90}")
    print(f"{'Deploy':<20} {'QPS':>5} {'Thpt':>8} {'TTFT_avg':>9} {'TTFT_p99':>9} {'TPOT':>8} {'ok/fail':>10}")
    print("-" * 90)
    for dep, res in all_results.items():
        if "__status__" in res:
            print(f"{dep:<20} {res['__status__']}")
            continue
        for k, m in res.items():
            qv = k.replace("qps_", "")
            if m.get("status") != "PASS":
                print(f"{dep:<20} {qv:>5} FAIL ({m.get('failed',0)} failed)")
                continue
            print(f"{dep:<20} {qv:>5} {m['throughput_tok_s']:>8.1f} "
                  f"{m['ttft_avg_ms']:>9.1f} {m['ttft_p99_ms']:>9.1f} "
                  f"{m['tpot_avg_ms']:>8.1f} {m['successful']:>4}/{m['failed']}")
    print("=" * 90)


if __name__ == "__main__":
    main()
