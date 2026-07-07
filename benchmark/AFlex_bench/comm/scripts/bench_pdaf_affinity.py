#!/usr/bin/env python3
"""PDAF NIC-affinity A/B benchmark: interleaved (affine) vs continuous (non-affine) GPU layout.

Deploys PDAF in two modes on 2 nodes (each 8× A800), runs prefill-heavy workloads
(summary scenario: in_len=4096, out_len=64) at various QPS, and collects TTFT / TPOT /
throughput / energy metrics.

Usage (on node1 host):
    python3 bench_pdaf_affinity.py --qps 1,2,3,4,6,8
    python3 bench_pdaf_affinity.py --deploy affine --qps 2,4
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("affinity_bench")

# -- Topology --
NODE1_IP = os.environ.get("MN_NODE1_IP", "10.252.129.36")
NODE2_IP = os.environ.get("MN_NODE2_IP", "10.252.129.35")
CONTAINER = os.environ.get("MN_CONTAINER", "operator_test")
PYTHON = "/usr/bin/python3"
MODEL = "/models/Qwen3-32B/"
TP = 4

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
LOG_C = "/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"
CLEANUP = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"

MAX_GPU_FREQ = 1410
PA_PORT, PF_PORT = 42010, 42011
DA_PORT, DF_PORT = 42020, 42021
ROUTER_PORT = 42000
BS_PORT = 49999

TTFT_SLO_MS = 5000.0
TPOT_SLO_MS = 300.0

# NIC affinity JSON (GPU -> IB device, PCIe PXB relationship)
IB_JSON_AFFINE = '{"0":"mlx5_0","1":"mlx5_0","2":"mlx5_1","3":"mlx5_1","4":"mlx5_4","5":"mlx5_4","6":"mlx5_5","7":"mlx5_5"}'
IB_JSON_FILE = "/tmp/ib_affine_map.json"
IB_DEV_NONAFFINE = "mlx5_bond_0"


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


def get_energy_local(gpus):
    try:
        import pynvml
        pynvml.nvmlInit()
        res = {}
        for idx in gpus:
            h = pynvml.nvmlDeviceGetHandleByIndex(idx)
            res[idx] = pynvml.nvmlDeviceGetTotalEnergyConsumption(h)
        pynvml.nvmlShutdown()
        return res
    except Exception:
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
        out = subprocess.run(_ssh(NODE2_IP, inner), capture_output=True, text=True, timeout=30)
        line = [l for l in out.stdout.strip().splitlines() if l.startswith("{")]
        return {int(k): v for k, v in json.loads(line[-1]).items()} if line else {i: 0 for i in gpus}
    except Exception:
        return {i: 0 for i in gpus}


# ============================================================
# PDAF deployment: two modes
# ============================================================

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


def start_pdaf_nonaffine():
    """Standard continuous layout: PF GPU 0-3, PA GPU 4-7. IPC offset ±4."""
    log.info("Deploying PDAF non-affine (continuous GPU layout)")
    cf = _common_flags(IB_DEV_NONAFFINE)

    # node1 PF (ffn, prefill, GPU 0-3)
    env = _base_env(28200, 68400, TP, "0,1,2,3", 0)
    _launch_server(NODE1_IP, PF_PORT, "ffn", "prefill", 0, "pf_nonaffine", env, cf)
    time.sleep(6)

    # node1 PA (attn, prefill, GPU 4-7)
    env = _base_env(28200, 68400, -TP, "4,5,6,7", 4, ffn_host="127.0.0.1")
    _launch_server(NODE1_IP, PA_PORT, "attn", "prefill", TP, "pa_nonaffine", env, cf)

    # node2 DF (ffn, decode, GPU 0-3)
    env = _base_env(28300, 68500, TP, "0,1,2,3", 0)
    _launch_server(NODE2_IP, DF_PORT, "ffn", "decode", 0, "df_nonaffine", env, cf)
    time.sleep(8)

    # node2 DA (attn, decode, GPU 4-7)
    env = _base_env(28300, 68500, -TP, "4,5,6,7", 4, ffn_host="127.0.0.1")
    _launch_server(NODE2_IP, DA_PORT, "attn", "decode", TP, "da_nonaffine", env, cf)

    return _wait_and_start_router()


def start_pdaf_affine():
    """Interleaved layout: PA/DA GPU 0,2,4,6 (step=2); PF/DF GPU 1,3,5,7. IPC offset ±1."""
    log.info("Deploying PDAF affine (interleaved GPU layout)")
    # Write JSON IB mapping file to both containers via docker cp
    import tempfile
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write(IB_JSON_AFFINE)
        tmp_path = f.name
    subprocess.run(["docker", "cp", tmp_path, f"{CONTAINER}:{IB_JSON_FILE}"], check=False)
    subprocess.run(["scp", "-o", "StrictHostKeyChecking=no", "-q",
                    tmp_path, f"{NODE2_IP}:/tmp/ib_affine_map.json"], check=False)
    subprocess.run(_ssh(NODE2_IP, f"docker cp /tmp/ib_affine_map.json {CONTAINER}:{IB_JSON_FILE}"),
                   check=False)
    os.unlink(tmp_path)
    cf = _common_flags(IB_JSON_FILE, gpu_step=2)

    # node1 PF (ffn, prefill, GPU 1,3,5,7)
    env = _base_env(28200, 68400, -1, "1,3,5,7", 1)
    _launch_server(NODE1_IP, PF_PORT, "ffn", "prefill", 1, "pf_affine", env, cf)
    time.sleep(6)

    # node1 PA (attn, prefill, GPU 0,2,4,6)
    env = _base_env(28200, 68400, 1, "0,2,4,6", 0, ffn_host="127.0.0.1")
    _launch_server(NODE1_IP, PA_PORT, "attn", "prefill", 0, "pa_affine", env, cf)

    # node2 DF (ffn, decode, GPU 1,3,5,7)
    env = _base_env(28300, 68500, -1, "1,3,5,7", 1)
    _launch_server(NODE2_IP, DF_PORT, "ffn", "decode", 1, "df_affine", env, cf)
    time.sleep(8)

    # node2 DA (attn, decode, GPU 0,2,4,6)
    env = _base_env(28300, 68500, 1, "0,2,4,6", 0, ffn_host="127.0.0.1")
    _launch_server(NODE2_IP, DA_PORT, "attn", "decode", 0, "da_affine", env, cf)

    return _wait_and_start_router()


def _wait_and_start_router():
    log.info("Waiting for PDAF servers...")
    checks = [(NODE1_IP, PF_PORT, "PF", True), (NODE1_IP, PA_PORT, "PA", False),
              (NODE2_IP, DF_PORT, "DF", True), (NODE2_IP, DA_PORT, "DA", False)]
    for host, port, name, mi in checks:
        if not wait_health(host, port, 600, check_model_info=mi):
            log.error("  %s (%s:%d) failed to start", name, host, port)
            return None
        log.info("  %s ready (%s:%d)", name, host, port)

    router_cmd = (f"setsid {PYTHON} -m sglang_router.launch_router "
                  "--pd-disaggregation --mini-lb "
                  f"--prefill http://{NODE1_IP}:{PA_PORT} "
                  f"--decode http://{NODE2_IP}:{DA_PORT} "
                  f"--host {NODE1_IP} --port {ROUTER_PORT} "
                  f"> {LOG_C}/router_aff.log 2>&1 < /dev/null &")
    dexec_local(router_cmd)
    if not wait_health(NODE1_IP, ROUTER_PORT, 60):
        log.error("  router failed")
        return None
    log.info("  router ready")
    return f"http://{NODE1_IP}:{ROUTER_PORT}"


# ============================================================
# Workload generation (summary scenario: in=4096, out=64)
# ============================================================

def generate_workload(qps, duration_s=120):
    """Generate Poisson-arrival summary workload (in_len=4096, out_len=64)."""
    rng = np.random.default_rng(42)
    n_requests = int(qps * duration_s)
    inter_arrivals = rng.exponential(1.0 / qps, n_requests)
    arrivals = np.cumsum(inter_arrivals)
    reqs = []
    for t in arrivals:
        reqs.append({"input_len": 4096, "output_len": 64, "arrival_time_s": float(t)})
    return reqs


# ============================================================
# Async workload runner
# ============================================================

async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    prompt = "Summarize the following document in detail:\n" + "x " * (req["input_len"] // 2)
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


async def run_workload(reqs, url, max_run_s=400):
    n1_gpus = list(range(8))
    n2_gpus = list(range(8))
    e1_start = get_energy_local(n1_gpus)
    e2_start = get_energy_remote(n2_gpus)
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
    e1_end = get_energy_local(n1_gpus)
    e2_end = get_energy_remote(n2_gpus)
    energy_n1_j = sum((e1_end.get(i, 0) - e1_start.get(i, 0)) / 1000.0 for i in n1_gpus)
    energy_n2_j = sum((e2_end.get(i, 0) - e2_start.get(i, 0)) / 1000.0 for i in n2_gpus)
    total_energy_j = energy_n1_j + energy_n2_j

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
    n_ttft_viol = sum(1 for v in ttft_src if v > TTFT_SLO_MS)
    n_tpot_viol = sum(1 for r in ok if r["tpot_ms"] > TPOT_SLO_MS)
    slo_rate = (n_ttft_viol + n_tpot_viol + len(fail)) / len(results) * 100 if results else 0

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
        "energy_node1_j": round(energy_n1_j, 1),
        "energy_node2_j": round(energy_n2_j, 1),
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(total_energy_j * 1000 / total_tokens, 2) if total_tokens else 0,
        "slo_violation_rate": round(slo_rate, 1),
    }


# ============================================================
# Main orchestration
# ============================================================

DEPLOY_FNS = {
    "nonaffine": start_pdaf_nonaffine,
    "affine": start_pdaf_affine,
}


def test_generate(url):
    try:
        r = requests.post(url + "/generate", json={
            "text": "Hello, the capital of France is",
            "sampling_params": {"max_new_tokens": 16, "temperature": 0.0}}, timeout=120)
        return "text" in r.json()
    except Exception as e:
        log.error("warmup generate failed: %s", e)
        return False


def run_deploy(deploy_name, qps_list, duration_s, max_run_s):
    log.info("=" * 70)
    log.info("DEPLOY: pdaf_%s (tp=%d, Qwen3-32B, summary workload)", deploy_name, TP)
    log.info("=" * 70)

    cleanup_all()
    start_fn = DEPLOY_FNS[deploy_name]
    url = start_fn()
    if url is None:
        log.error("  Deployment FAILED")
        cleanup_all()
        return {"__status__": "DEPLOY_FAILED"}

    lock_freq(MAX_GPU_FREQ)

    log.info("Warmup inference test...")
    if not test_generate(url):
        log.error("  warmup failed")
        unlock_freq()
        cleanup_all()
        return {"__status__": "WARMUP_FAILED"}
    time.sleep(5)

    results = {}
    for qps in qps_list:
        log.info("-" * 50)
        log.info("  QPS=%d (duration=%ds)", qps, duration_s)
        reqs = generate_workload(qps, duration_s)
        last_arrival = max(r["arrival_time_s"] for r in reqs)
        run_s = int(min(max(max_run_s, last_arrival + 150), 900))
        summary = asyncio.run(run_workload(reqs, url + "/generate", run_s))
        results[f"qps_{qps}"] = summary
        if summary.get("status") == "PASS":
            log.info("  Thpt=%.1f tok/s | TTFT_avg=%.1fms p99=%.1fms | "
                     "TPOT_avg=%.1fms p99=%.1fms | E=%.0fJ | SLO=%.1f%%",
                     summary["throughput_tok_s"], summary["ttft_avg_ms"],
                     summary["ttft_p99_ms"], summary["tpot_avg_ms"],
                     summary["tpot_p99_ms"], summary["total_energy_j"],
                     summary["slo_violation_rate"])
        else:
            log.error("  FAIL: %s", summary)
        time.sleep(5)

    unlock_freq()
    cleanup_all()
    return results


def main():
    parser = argparse.ArgumentParser(description="PDAF NIC-affinity A/B benchmark")
    parser.add_argument("--deploy", default="all",
                        help="Comma-sep: affine,nonaffine or 'all'")
    parser.add_argument("--qps", default="1,2,3,4,6,8",
                        help="Comma-sep QPS values")
    parser.add_argument("--duration", type=int, default=120,
                        help="Workload duration per QPS point (seconds)")
    parser.add_argument("--max-run-s", type=int, default=400,
                        help="Max seconds to wait for workload completion")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (default: auto-generated)")
    args = parser.parse_args()

    deploys = list(DEPLOY_FNS.keys()) if args.deploy == "all" else args.deploy.split(",")
    qps_list = [int(q) for q in args.qps.split(",")]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for deploy_name in deploys:
        if deploy_name not in DEPLOY_FNS:
            log.error("Unknown deploy: %s (valid: affine, nonaffine)", deploy_name)
            continue
        all_results[f"pdaf_{deploy_name}"] = run_deploy(
            deploy_name, qps_list, args.duration, args.max_run_s)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = Path(args.output) if args.output else RESULTS_DIR / f"affinity_bench_{ts}.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "node1": NODE1_IP, "node2": NODE2_IP, "model": MODEL, "tp": TP,
            "workload": "summary (in=4096, out=64)",
            "deploys": deploys, "qps_list": qps_list, "duration_s": args.duration,
            "ib_nonaffine": IB_DEV_NONAFFINE, "ib_affine": IB_JSON_AFFINE,
        },
        "results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Results saved: %s", out_file)

    # Print summary table
    print("\n" + "=" * 100)
    print("  PDAF AFFINITY BENCHMARK (Qwen3-32B, summary in=4096 out=64)")
    print("=" * 100)
    hdr = (f"{'Deploy':<20} {'QPS':>4} {'Thpt':>8} {'TTFT_avg':>9} {'TTFT_p99':>9} "
           f"{'TPOT_avg':>9} {'TPOT_p99':>9} {'Energy':>8} {'SLO%':>6}")
    print(hdr)
    print("-" * 100)
    for dep, dep_res in all_results.items():
        if "__status__" in dep_res:
            print(f"{dep:<20} {dep_res['__status__']}")
            continue
        for qps_key, m in dep_res.items():
            qps_val = qps_key.replace("qps_", "")
            if m.get("status") != "PASS":
                print(f"{dep:<20} {qps_val:>4} FAIL")
                continue
            print(f"{dep:<20} {qps_val:>4} {m['throughput_tok_s']:>8.1f} "
                  f"{m['ttft_avg_ms']:>9.1f} {m['ttft_p99_ms']:>9.1f} "
                  f"{m['tpot_avg_ms']:>9.1f} {m['tpot_p99_ms']:>9.1f} "
                  f"{m['total_energy_j']:>8.0f} {m['slo_violation_rate']:>6.1f}")
    print("=" * 100)


if __name__ == "__main__":
    main()
