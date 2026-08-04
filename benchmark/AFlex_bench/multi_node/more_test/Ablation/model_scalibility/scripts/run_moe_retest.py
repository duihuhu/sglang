#!/usr/bin/env python3
"""MoE补测脚本: DistServe/BiScale + SGLang_TP4/DynamoLLM_TP4 + AFlex_new

测试计划:
1. DistServe: 2P(TP4)@Pnode + 4D(TP2)@Dnode (并行两组: A=node1+2, B=node3+4)
   BiScale: 同上 + DVFS
2. SGLang_TP4: 4×TP4 (并行两组: A=node1+2, B=node3+4)
   DynamoLLM_TP4: 同上 + DVFS
3. AFlex_new: 4P(PA_TP1+PF_TP2)+1D(DA_TP1+DF_TP2) 需3节点

每项: 2 datasets(conv,code) × 4 QPS(2,4,8,16)
"""
import asyncio
import csv
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ============================================================
# Configuration
# ============================================================
NODE1_IP = "10.252.129.36"
NODE2_IP = "10.252.129.35"
NODE3_IP = "10.252.129.34"
NODE4_IP = "10.252.129.33"

CONTAINER = os.environ.get("MN_CONTAINER", "operator_test")
PYTHON = "/usr/bin/python3"
MODEL = "/models/Mixtral-8x7B/"
MAX_GPU_FREQ = 1410
TTFT_SLO_MS = 1000.0
TPOT_SLO_MS = 150.0
ROUTER_PORT = 42000

HERE = Path(__file__).resolve().parent
AFLEX_ROOT = HERE.parents[4]

RESULTS_DIR = HERE.parent / "data"
WORKLOAD_DIR = AFLEX_ROOT / "multi_node/more_test/macro/data/workloads"
CLEANUP = "/workspace/sglang/benchmark/AFlex_bench/multi_node/scripts/cleanup_node.sh"
LOG_C = "/workspace/sglang/benchmark/AFlex_bench/multi_node/logs"
IB_JSON_FILE = "/tmp/ib_scal_map.json"

ENERGY_MODEL_DIR_V1 = ("/workspace/sglang/benchmark/AFlex_bench/energy_model/"
                       "Mixtral-8x7B/models_v1")

GPU_NIC = {0: "mlx5_0", 1: "mlx5_0", 2: "mlx5_1", 3: "mlx5_1",
            4: "mlx5_4", 5: "mlx5_4", 6: "mlx5_5", 7: "mlx5_5"}

DATASETS = ["conv", "code"]
QPS_LIST = [2, 4, 8, 16]

COMMON_FLAGS = ("--mem-fraction-static 0.85 --disable-cuda-graph "
                "--disable-piecewise-cuda-graph --skip-server-warmup "
                "--disable-radix-cache")

# ============================================================
# Utility functions
# ============================================================
def _ssh(host, cmd):
    return ["ssh", "-o", "StrictHostKeyChecking=no", f"root@{host}", cmd]


def dexec(host, shell_cmd):
    inner = f"docker exec {CONTAINER} bash -lc {shlex.quote(shell_cmd)}"
    subprocess.run(_ssh(host, inner), check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cleanup_hosts(*hosts):
    for h in hosts:
        try:
            subprocess.run(
                _ssh(h, f"docker exec {CONTAINER} bash -lc 'bash {CLEANUP}'"),
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=90,
            )
        except subprocess.TimeoutExpired:
            log.warning("cleanup timed out on %s; continuing", h)
    time.sleep(3)


def wait_health(host, port, timeout=600):
    import urllib.request
    url = f"http://{host}:{port}/health"
    url_fallback = f"http://{host}:{port}/get_model_info"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = urllib.request.urlopen(url, timeout=5)
            if r.status == 200:
                return True
        except Exception:
            try:
                r2 = urllib.request.urlopen(url_fallback, timeout=5)
                if r2.status == 200:
                    return True
            except Exception:
                pass
        time.sleep(3)
    return False


def get_energy(host, gpus):
    pycode = (
        "import pynvml; pynvml.nvmlInit(); "
        f"gpus={list(gpus)}; "
        "res={}; "
        "[res.__setitem__(g, pynvml.nvmlDeviceGetTotalEnergyConsumption("
        "pynvml.nvmlDeviceGetHandleByIndex(g))) for g in gpus]; "
        "print(res)"
    )
    inner = f"{PYTHON} -c {shlex.quote(pycode)}"
    full = f"docker exec {CONTAINER} bash -lc {shlex.quote(inner)}"
    out = subprocess.run(_ssh(host, full), capture_output=True, text=True, timeout=15)
    if out.returncode == 0:
        return eval(out.stdout.strip())
    return {}


def start_sm_monitors(energy_hosts_gpus, output_dir, tag, interval_ms=200):
    """Start per-host nvidia-smi sampling for the workload window."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    monitors = []
    query = (
        "timestamp,index,utilization.gpu,clocks.current.sm,"
        "power.draw,memory.used"
    )
    for host, gpus in energy_hosts_gpus:
        path = output_dir / f"{tag}_{host.replace('.', '_')}.csv"
        fp = path.open("w")
        fp.write("timestamp,index,sm_util_pct,sm_clock_mhz,power_w,memory_mib\n")
        fp.flush()
        gpu_ids = ",".join(str(g) for g in gpus)
        command = (
            f"nvidia-smi --id={gpu_ids} --query-gpu={query} "
            f"--format=csv,noheader,nounits --loop-ms={interval_ms}"
        )
        proc = subprocess.Popen(
            _ssh(host, command),
            stdout=fp,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        monitors.append((host, gpus, path, fp, proc))
    return monitors


def summarize_sm_trace(path, interval_ms=200):
    """Summarize SM samples, including the longest observed zero-util gap."""
    by_gpu = {}
    try:
        raw = path.read_bytes().replace(b"\x00", b"")
        lines = raw.decode("utf-8", errors="replace").splitlines()
    except Exception:
        return {}
    reader = csv.DictReader(lines)
    for row in reader:
            try:
                idx_val = row.get("index") or row.get(" index") or ""
                gpu = int(idx_val.strip())
                sample = {
                    "util": float((row.get("sm_util_pct") or row.get(" sm_util_pct") or "0").strip()),
                    "clock": float((row.get("sm_clock_mhz") or row.get(" sm_clock_mhz") or "0").strip()),
                    "power": float((row.get("power_w") or row.get(" power_w") or "0").strip()),
                }
            except (KeyError, TypeError, ValueError, AttributeError):
                continue
            by_gpu.setdefault(gpu, []).append(sample)

    result = {}
    for gpu, samples in sorted(by_gpu.items()):
        utils = np.asarray([s["util"] for s in samples])
        clocks = np.asarray([s["clock"] for s in samples])
        powers = np.asarray([s["power"] for s in samples])
        longest_zero = current_zero = 0
        for util in utils:
            if util <= 0:
                current_zero += 1
                longest_zero = max(longest_zero, current_zero)
            else:
                current_zero = 0
        result[str(gpu)] = {
            "samples": len(samples),
            "sm_util_avg_pct": round(float(utils.mean()), 2),
            "sm_util_p95_pct": round(float(np.percentile(utils, 95)), 2),
            "sm_util_max_pct": round(float(utils.max()), 2),
            "active_sample_ratio": round(float(np.mean(utils > 0)), 4),
            "saturated_sample_ratio": round(float(np.mean(utils >= 90)), 4),
            "longest_zero_util_s": round(longest_zero * interval_ms / 1000, 2),
            "sm_clock_avg_mhz": round(float(clocks.mean()), 1),
            "sm_clock_min_mhz": round(float(clocks.min()), 1),
            "sm_clock_max_mhz": round(float(clocks.max()), 1),
            "power_avg_w": round(float(powers.mean()), 2),
        }
    return result


def stop_sm_monitors(monitors, output_dir, tag, interval_ms=200):
    """Stop samplers and write a compact JSON summary."""
    for _, _, _, _, proc in monitors:
        proc.terminate()
    for _, _, _, fp, proc in monitors:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        fp.close()

    summary = {}
    traces = {}
    for host, gpus, path, _, _ in monitors:
        summary[host] = summarize_sm_trace(path, interval_ms)
        traces[host] = {"gpus": list(gpus), "csv": str(path)}
    payload = {
        "sample_interval_ms": interval_ms,
        "traces": traces,
        "summary": summary,
    }
    summary_path = Path(output_dir) / f"{tag}_summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    log.info("  SM utilization summary saved: %s", summary_path)
    return summary_path


def lock_freq(hosts, gpus, freq=MAX_GPU_FREQ):
    for h in hosts:
        pycode = (
            "import pynvml; pynvml.nvmlInit(); "
            f"[pynvml.nvmlDeviceSetGpuLockedClocks("
            f"pynvml.nvmlDeviceGetHandleByIndex(g),{freq},{freq}) "
            f"for g in {list(gpus)}]"
        )
        inner = f"{PYTHON} -c {shlex.quote(pycode)}"
        full = f"docker exec {CONTAINER} bash -lc {shlex.quote(inner)}"
        subprocess.run(_ssh(h, full), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def unlock_freq(hosts, gpus):
    for h in hosts:
        pycode = (
            "import pynvml; pynvml.nvmlInit(); "
            f"[pynvml.nvmlDeviceResetGpuLockedClocks("
            f"pynvml.nvmlDeviceGetHandleByIndex(g)) "
            f"for g in {list(gpus)}]"
        )
        inner = f"{PYTHON} -c {shlex.quote(pycode)}"
        full = f"docker exec {CONTAINER} bash -lc {shlex.quote(inner)}"
        subprocess.run(_ssh(h, full), check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def write_ib_json(hosts):
    ib_map = {str(g): GPU_NIC[g] for g in range(8)}
    json_str = json.dumps(ib_map)
    for h in hosts:
        cmd = f"echo {shlex.quote(json_str)} > {IB_JSON_FILE}"
        dexec(h, cmd)


def test_generate(url, timeout=120):
    import urllib.request
    payload = json.dumps({
        "input_ids": [1000] * 10,
        "sampling_params": {"max_new_tokens": 5, "temperature": 0.0},
    }).encode()
    try:
        req = urllib.request.Request(url + "/generate",
                                     data=payload,
                                     headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(req, timeout=timeout)
        return r.status == 200
    except Exception as e:
        log.warning("test_generate failed: %s", e)
        return False


def _dvfs_flags(tier):
    if not tier:
        return ""
    return (" --dvfs-enabled "
            f"--dvfs-energy-model-dir {ENERGY_MODEL_DIR_V1} "
            f"--dvfs-ttft-slo-ms {int(TTFT_SLO_MS)} "
            f"--dvfs-tpot-slo-us {int(TPOT_SLO_MS * 1000)}")


def _afd_dvfs_flags():
    return (" --afd-dvfs-enabled "
            f"--afd-energy-model-dir {ENERGY_MODEL_DIR_V1} "
            f"--afd-ttft-slo-ms {int(TTFT_SLO_MS)} "
            f"--afd-tpot-slo-us {int(TPOT_SLO_MS * 1000)} "
            "--afd-dvfs-idle-lock "
            "--afd-dvfs-prefill-slack-factor 0.5 "
            "--afd-dvfs-decode-compositional")


# ============================================================
# Workload runner (from run_moe_macro_node34.py)
# ============================================================
async def send_one(session, url, req, base_time, results):
    delay = req["arrival_time_s"] - (time.monotonic() - base_time)
    if delay > 0:
        await asyncio.sleep(delay)
    payload = {"input_ids": [1000] * req["input_len"],
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
                except json.JSONDecodeError:
                    continue
                meta = chunk.get("meta_info", chunk.get("usage", {}))
                if meta:
                    last_meta = meta
                if first_token_time is None:
                    first_token_time = now
                token_count += 1
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


async def run_workload(
    reqs,
    url,
    energy_hosts_gpus,
    max_run_s=400,
    sm_monitor_dir=None,
    sm_monitor_tag=None,
):
    """Run workload and measure energy across specified (host, gpus) pairs."""
    monitors = []
    if sm_monitor_dir:
        monitors = start_sm_monitors(
            energy_hosts_gpus,
            sm_monitor_dir,
            sm_monitor_tag or f"sm_{int(time.time())}",
        )
    e_start = {h: get_energy(h, gpus) for h, gpus in energy_hosts_gpus}
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
    e_end = {h: get_energy(h, gpus) for h, gpus in energy_hosts_gpus}
    if monitors:
        stop_sm_monitors(
            monitors,
            sm_monitor_dir,
            sm_monitor_tag or f"sm_{int(time.time())}",
        )
    total_energy_j = 0.0
    for h, gpus in energy_hosts_gpus:
        for g in gpus:
            total_energy_j += (e_end[h].get(g, 0) - e_start[h].get(g, 0)) / 1000.0

    ok = [r for r in results if r.get("success")]
    fail = [r for r in results if not r.get("success")]
    missing = max(0, len(reqs) - len(results))
    failed_count = len(fail) + missing
    if not ok:
        return {"status": "FAIL", "failed": failed_count}

    ttfts_proc = [r["ttft_proc_ms"] for r in ok if r.get("ttft_proc_ms", 0) > 0]
    ttfts = [r["ttft_ms"] for r in ok if r["ttft_ms"] > 0]
    tpots = [r["tpot_ms"] for r in ok if r["tpot_ms"] > 0]
    total_tokens = sum(r.get("completion_tokens", 0) for r in ok)
    throughput = total_tokens / duration_s if duration_s > 0 else 0
    src = ttfts_proc if ttfts_proc else ttfts
    energy_per_tok = (total_energy_j * 1000 / total_tokens) if total_tokens > 0 else 0

    return {
        "status": "PASS" if failed_count == 0 else "FAIL",
        "duration_s": round(duration_s, 1),
        "total_requests": len(reqs), "successful": len(ok),
        "failed": failed_count,
        "total_tokens": total_tokens, "throughput_tok_s": round(throughput, 1),
        "ttft_proc_avg_ms": round(float(np.mean(src)), 1) if src else 0,
        "ttft_proc_p50_ms": round(float(np.percentile(src, 50)), 1) if src else 0,
        "ttft_proc_p99_ms": round(float(np.percentile(src, 99)), 1) if src else 0,
        "tpot_avg_ms": round(float(np.mean(tpots)), 1) if tpots else 0,
        "tpot_p50_ms": round(float(np.percentile(tpots, 50)), 1) if tpots else 0,
        "tpot_p99_ms": round(float(np.percentile(tpots, 99)), 1) if tpots else 0,
        "total_energy_j": round(total_energy_j, 1),
        "energy_per_token_mj": round(energy_per_tok, 2),
    }


# ============================================================
# Deploy: DistServe / BiScale — 2P(TP4)@Pnode + 4D(TP2)@Dnode
# ============================================================
def deploy_distserve(p_node, d_node, tier=False):
    """Deploy DistServe: 2P(TP4)@p_node + 4D(TP2)@d_node = 16 GPU."""
    write_ib_json([p_node, d_node])
    p_groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    d_groups = [[0, 1], [2, 3], [4, 5], [6, 7]]
    p_ports = [53100, 53110]
    d_ports = [53150, 53160, 53170, 53180]
    bs_ports = [49100, 49110]  # each P needs unique bootstrap port

    for i, gpus in enumerate(p_groups):
        csv = ",".join(str(g) for g in gpus)
        nic = GPU_NIC[gpus[0]]
        cmd = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
               f"SGLANG_HOST_IP={p_node} "
               f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDICES={csv}; "
               f"setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
               f"-m sglang.launch_server --model-path {MODEL} --tp 4 "
               f"--host {p_node} --port {p_ports[i]} --nccl-port {34000+i*10} "
               f"{COMMON_FLAGS} "
               "--disaggregation-mode prefill --disaggregation-transfer-backend mooncake "
               f"--disaggregation-bootstrap-port {bs_ports[i]} "
               f"--disaggregation-ib-device {nic} "
               f"{_dvfs_flags(tier)} "
               f"> {LOG_C}/dist_p{i}.log 2>&1 < /dev/null &")
        dexec(p_node, cmd)
        time.sleep(3)

    for i, gpus in enumerate(d_groups):
        csv = ",".join(str(g) for g in gpus)
        nic = GPU_NIC[gpus[0]]
        cmd = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
               f"SGLANG_HOST_IP={d_node} "
               f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDICES={csv}; "
               f"setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
               f"-m sglang.launch_server --model-path {MODEL} --tp 2 "
               f"--host {d_node} --port {d_ports[i]} --nccl-port {34050+i*10} "
               f"{COMMON_FLAGS} "
               "--disaggregation-mode decode --disaggregation-transfer-backend mooncake "
               f"--disaggregation-ib-device {nic} "
               f"{_dvfs_flags(tier)} "
               f"> {LOG_C}/dist_d{i}.log 2>&1 < /dev/null &")
        dexec(d_node, cmd)
        time.sleep(3)

    for i in range(2):
        if not wait_health(p_node, p_ports[i], 500):
            log.error("DistServe P%d failed health", i)
            return None
    log.info("  Prefill instances ready")
    for i in range(4):
        if not wait_health(d_node, d_ports[i], 500):
            log.error("DistServe D%d failed health", i)
            return None
    log.info("  Decode instances ready")

    rc = (f"setsid {PYTHON} -m sglang_router.launch_router "
          "--pd-disaggregation "
          f"--host {p_node} --port {ROUTER_PORT} "
          f"--prefill http://{p_node}:{p_ports[0]} {bs_ports[0]} "
          f"--prefill http://{p_node}:{p_ports[1]} {bs_ports[1]} "
          f"--decode http://{d_node}:{d_ports[0]} "
          f"--decode http://{d_node}:{d_ports[1]} "
          f"--decode http://{d_node}:{d_ports[2]} "
          f"--decode http://{d_node}:{d_ports[3]} "
          f"> {LOG_C}/dist_router.log 2>&1 < /dev/null &")
    dexec(p_node, rc)
    if not wait_health(p_node, ROUTER_PORT, 60):
        log.error("DistServe router failed")
        return None
    log.info("  Router ready")
    return f"http://{p_node}:{ROUTER_PORT}"


# ============================================================
# Deploy: SGLang_TP4 / DynamoLLM_TP4 — 4×TP4 across 2 nodes
# ============================================================
def deploy_sglang_tp4(node_a, node_b, tier=False):
    """Deploy 4×TP4 instances: 2 per node."""
    groups = [[0, 1, 2, 3], [4, 5, 6, 7]]
    ports_a = [53200, 53210]
    ports_b = [53200, 53210]
    insts = []
    for i, gpus in enumerate(groups):
        insts.append((node_a, gpus, ports_a[i]))
        insts.append((node_b, gpus, ports_b[i]))

    for host, gpus, port in insts:
        csv = ",".join(str(g) for g in gpus)
        cmd = (f"export SGLANG_DISABLE_REQUEST_LOGGING=true UCX_LOG_LEVEL=fatal "
               f"SGLANG_HOST_IP={host} "
               f"CUDA_VISIBLE_DEVICES={csv} AFD_NVML_DEVICE_INDEX={gpus[0]} "
               f"AFD_NVML_DEVICE_INDICES={csv}; "
               f"setsid prlimit --memlock=unlimited:unlimited {PYTHON} "
               f"-m sglang.launch_server --model-path {MODEL} --tp 4 "
               f"--host {host} --port {port} --nccl-port {33300+gpus[0]*10} "
               f"{COMMON_FLAGS} "
               f"{_dvfs_flags(tier)} "
               f"> {LOG_C}/sglang_tp4_{host.split('.')[-1]}_{gpus[0]}.log 2>&1 < /dev/null &")
        dexec(host, cmd)
        time.sleep(3)

    workers = []
    for host, gpus, port in insts:
        if not wait_health(host, port, 500):
            log.error("SGLang_TP4 %s:%d failed health", host, port)
            return None
        workers.append(f"http://{host}:{port}")
        log.info("  %s:%d ready", host, port)

    worker_urls = " ".join(workers)
    rc = (f"setsid {PYTHON} -m sglang_router.launch_router "
          f"--host {node_a} --port {ROUTER_PORT} --policy round_robin "
          f"--worker-urls {worker_urls} "
          f"> {LOG_C}/sglang_tp4_router.log 2>&1 < /dev/null &")
    dexec(node_a, rc)
    if not wait_health(node_a, ROUTER_PORT, 60):
        log.error("SGLang_TP4 router failed")
        return None
    log.info("  Router ready")
    return f"http://{node_a}:{ROUTER_PORT}"


# ============================================================
# Deploy: AFlex_new — 4P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2)
# 3 nodes needed: 2 nodes for P, 1 node for D (partial)
# Layout:
#   node_p1: P0(PF:GPU0,1 PA:GPU2) + P1(PF:GPU3,4 PA:GPU5) + partial P2(PF:GPU6,7 PA:???)
#   Actually: each P pair = PA(TP1,1GPU) + PF(TP2,2GPU) = 3 GPU
#   4P = 12 GPU → node_p1(6GPU: P0+P1) + node_p2(6GPU: P2+P3)
#   1D = DA(TP1,1GPU) + DF(TP2,2GPU) = 3 GPU → node_d(3GPU)
#   Total = 15 GPU across 3 nodes
#
# Simplified layout:
#   node_p1: P0(PF:0,1 PA:2) P1(PF:3,4 PA:5) — 6 GPU
#   node_p2: P2(PF:0,1 PA:2) P3(PF:3,4 PA:5) — 6 GPU
#   node_d:  D0(DF:0,1 DA:2) — 3 GPU
# ============================================================
def deploy_aflex_new(node_p1, node_p2, node_d):
    """Deploy AFlex: 4P(PA_TP1+PF_TP2) + 1D(DA_TP1+DF_TP2) across 3 nodes."""
    write_ib_json([node_p1, node_p2, node_d])

    pairs = [
        {"name": "p0", "host": node_p1, "mode": "prefill",
         "ffn_gpus": (0, 1), "attn_gpus": (2,),
         "ffn_port": 45200, "attn_port": 45201,
         "ucx_port": 30200, "sched_port": 60400,
         "ffn_nccl": 39300, "attn_nccl": 39310,
         "ffn_bootstrap": 52000, "attn_bootstrap": 51999},
        {"name": "p1", "host": node_p1, "mode": "prefill",
         "ffn_gpus": (3, 4), "attn_gpus": (5,),
         "ffn_port": 45220, "attn_port": 45221,
         "ucx_port": 30300, "sched_port": 60500,
         "ffn_nccl": 39320, "attn_nccl": 39330,
         "ffn_bootstrap": 52010, "attn_bootstrap": 52011},
        {"name": "p2", "host": node_p2, "mode": "prefill",
         "ffn_gpus": (0, 1), "attn_gpus": (2,),
         "ffn_port": 45240, "attn_port": 45241,
         "ucx_port": 30400, "sched_port": 60600,
         "ffn_nccl": 39340, "attn_nccl": 39350,
         "ffn_bootstrap": 52020, "attn_bootstrap": 52021},
        {"name": "p3", "host": node_p2, "mode": "prefill",
         "ffn_gpus": (3, 4), "attn_gpus": (5,),
         "ffn_port": 45260, "attn_port": 45261,
         "ucx_port": 30500, "sched_port": 60700,
         "ffn_nccl": 39360, "attn_nccl": 39370,
         "ffn_bootstrap": 52030, "attn_bootstrap": 52031},
        {"name": "d0", "host": node_d, "mode": "decode",
         "ffn_gpus": (0, 1), "attn_gpus": (2,),
         "ffn_port": 45280, "attn_port": 45281,
         "ucx_port": 30600, "sched_port": 60800,
         "ffn_nccl": 39380, "attn_nccl": 39390,
         "ffn_bootstrap": 52040, "attn_bootstrap": 52041},
    ]

    log.info("AFlex 4P1D (PA_TP1+PF_TP2): p1=%s, p2=%s, d=%s", node_p1, node_p2, node_d)

    def _build_env(pair, role):
        is_attn = (role == "attn")
        role_gpus = pair["attn_gpus"] if is_attn else pair["ffn_gpus"]
        all_gpus = pair["ffn_gpus"] + pair["attn_gpus"]
        cvd = ",".join(str(g) for g in all_gpus)
        nvml = ",".join(str(g) for g in role_gpus)
        tp_f = len(pair["ffn_gpus"])
        tp_a = len(pair["attn_gpus"])
        ipc_offset = f"-{tp_f}" if is_attn else f"+{tp_f}"
        env = {
            "SGLANG_HOST_IP": pair["host"],
            "SGLANG_DISABLE_REQUEST_LOGGING": "true",
            "UCX_LOG_LEVEL": "fatal",
            "AFD_UCX_TLS": "rc,tcp,cuda_copy,cuda_ipc",
            "SGLANG_DISAGGREGATION_THREAD_POOL_SIZE": "128",
            "AFD_ASYNC_PIPELINE": "1",
            "AFD_IPC_SYNC_MODE": "ipc_event",
            "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "0",
            "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
            "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "600",
            "CUDA_VISIBLE_DEVICES": cvd,
            "AFD_UCX_BASE_PORT": str(pair["ucx_port"]),
            "AFD_SCHED_PORT": str(pair["sched_port"]),
            "AFD_NVML_DEVICE_INDICES": nvml,
            "AFD_NVML_DEVICE_INDEX": str(role_gpus[0]),
            "AFD_IPC_PEER_OFFSET": ipc_offset,
            "AFD_IPC_CHANNEL_BASE": str(pair["sched_port"] % 1000),
        }
        if is_attn:
            env["AFD_UCX_FFN_HOST"] = "127.0.0.1"
        return "export " + " ".join(f"{k}={shlex.quote(str(v))}" for k, v in env.items()) + ";"

    def _build_flags(pair, role):
        tp_a = len(pair["attn_gpus"])
        tp_f = len(pair["ffn_gpus"])
        tp = tp_a if role == "attn" else tp_f
        flags = (f"--model-path {MODEL} --tp {tp} "
                 "--afd-comm-backend ipc_cpp --afd-micro-batch 1 "
                 f"--afd-attn-tp {tp_a} --afd-ffn-tp {tp_f} "
                 f"{COMMON_FLAGS} "
                 "--max-running-requests 512 --watchdog-timeout 600 "
                 "--afd-disagg-interleave-poll "
                 "--num-reserved-decode-tokens 512 "
                 "--disaggregation-transfer-backend mooncake "
                 f"--disaggregation-ib-device {IB_JSON_FILE} "
                 "--enable-metrics"
                 f"{_afd_dvfs_flags()}")
        return flags

    def _launch_pair(pair, run_tag):
        tp_f = len(pair["ffn_gpus"])
        base_gpu_id_attn = tp_f

        ffn_env = _build_env(pair, "ffn")
        ffn_cmd = (f"{ffn_env} setsid prlimit --memlock=unlimited:unlimited "
                   f"{PYTHON} -m sglang.launch_server "
                   f"--host {pair['host']} --port {pair['ffn_port']} "
                   f"--afd-perspective ffn --disaggregation-mode {pair['mode']} "
                   f"--base-gpu-id 0 --nccl-port {pair['ffn_nccl']} "
                   f"--disaggregation-bootstrap-port {pair['ffn_bootstrap']} "
                   f"{_build_flags(pair, 'ffn')} "
                   f"> {LOG_C}/{run_tag}_{pair['name']}_ffn.log 2>&1 < /dev/null &")
        dexec(pair["host"], ffn_cmd)
        if not wait_health(pair["host"], pair["ffn_port"], 600):
            log.error("  %s/ffn failed health", pair["name"])
            return False
        log.info("  %s/ffn ready", pair["name"])

        attn_env = _build_env(pair, "attn")
        attn_cmd = (f"{attn_env} setsid prlimit --memlock=unlimited:unlimited "
                    f"{PYTHON} -m sglang.launch_server "
                    f"--host {pair['host']} --port {pair['attn_port']} "
                    f"--afd-perspective attn --disaggregation-mode {pair['mode']} "
                    f"--base-gpu-id {base_gpu_id_attn} --nccl-port {pair['attn_nccl']} "
                    f"--disaggregation-bootstrap-port {pair['attn_bootstrap']} "
                    f"{_build_flags(pair, 'attn')} "
                    f"> {LOG_C}/{run_tag}_{pair['name']}_attn.log 2>&1 < /dev/null &")
        dexec(pair["host"], attn_cmd)
        if not wait_health(pair["host"], pair["attn_port"], 600):
            log.error("  %s/attn failed health", pair["name"])
            return False
        log.info("  %s/attn ready", pair["name"])
        return True

    run_tag = "aflex4p1d"
    for pair in pairs:
        if not _launch_pair(pair, run_tag):
            return None

    p_pairs = [p for p in pairs if p["mode"] == "prefill"]
    d_pairs = [p for p in pairs if p["mode"] == "decode"]

    top_cmd = (f"setsid {PYTHON} -m sglang_router.launch_router "
               "--pd-disaggregation "
               f"--host {node_p1} --port {ROUTER_PORT} "
               + " ".join(
                   f"--prefill http://{p['host']}:{p['attn_port']} "
                   f"{p['attn_bootstrap']}"
                   for p in p_pairs
               )
               + " "
               + " ".join(
                   f"--decode http://{d['host']}:{d['attn_port']}"
                   for d in d_pairs
               )
               + " "
               f"> {LOG_C}/{run_tag}_router.log 2>&1 < /dev/null &")
    dexec(node_p1, top_cmd)
    if not wait_health(node_p1, ROUTER_PORT, 60):
        log.error("AFlex router failed")
        return None
    log.info("  AFlex router ready")
    return f"http://{node_p1}:{ROUTER_PORT}"


# ============================================================
# Test runner
# ============================================================
def run_scheme_tests(scheme_name, deploy_fn, deploy_args, hosts_for_cleanup,
                     energy_hosts_gpus, lock_hosts, lock_gpus,
                     datasets=DATASETS, qps_list=QPS_LIST, max_run_s=400,
                     monitor_sm_dir=None):
    """Run a scheme across all datasets and QPS."""
    results = {}
    for ds in datasets:
        for qps in qps_list:
            key = f"{ds}_qps{qps}"
            log.info("--- [%s] %s: deploying ---", scheme_name, key)
            cleanup_hosts(*hosts_for_cleanup)
            url = deploy_fn(*deploy_args)
            if url is None:
                log.error("  DEPLOY_FAILED: %s %s", scheme_name, key)
                results[key] = {"status": "DEPLOY_FAILED"}
                cleanup_hosts(*hosts_for_cleanup)
                continue

            lock_freq(lock_hosts, lock_gpus, MAX_GPU_FREQ)
            log.info("  Warmup...")
            if not test_generate(url):
                log.error("  WARMUP_FAILED: %s %s", scheme_name, key)
                results[key] = {"status": "WARMUP_FAILED"}
                unlock_freq(lock_hosts, lock_gpus)
                cleanup_hosts(*hosts_for_cleanup)
                continue
            time.sleep(3)

            wl_file = WORKLOAD_DIR / f"macro_{ds}_qps{qps}.jsonl"
            if not wl_file.exists():
                log.warning("  workload not found: %s", wl_file)
                results[key] = {"status": "WORKLOAD_MISSING"}
                unlock_freq(lock_hosts, lock_gpus)
                cleanup_hosts(*hosts_for_cleanup)
                continue
            with open(wl_file) as f:
                reqs = [json.loads(l) for l in f]
            last_arr = max((r["arrival_time_s"] for r in reqs), default=0)
            run_s = int(min(max(max_run_s, last_arr + 150), 900))

            log.info("  Running %s (%d reqs, %ds)...", key, len(reqs), run_s)
            monitor_tag = (
                f"{scheme_name}_{key}_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            summary = asyncio.run(
                run_workload(
                    reqs,
                    url + "/generate",
                    energy_hosts_gpus,
                    run_s,
                    sm_monitor_dir=monitor_sm_dir,
                    sm_monitor_tag=monitor_tag,
                )
            )
            if summary.get("status") == "PASS":
                log.info("  PASS: thpt=%.1f ttft=%.1f tpot=%.1f E=%.0fmJ/tok",
                         summary["throughput_tok_s"], summary["ttft_proc_p50_ms"],
                         summary["tpot_p50_ms"], summary["energy_per_token_mj"])
            else:
                log.error("  FAIL: %s", summary)
            results[key] = summary

            unlock_freq(lock_hosts, lock_gpus)
            cleanup_hosts(*hosts_for_cleanup)
            time.sleep(5)
    return results


# ============================================================
# Main orchestrator
# ============================================================
def main():
    import argparse
    import resource
    resource.setrlimit(resource.RLIMIT_MEMLOCK,
                       (resource.RLIM_INFINITY, resource.RLIM_INFINITY))

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all",
                        help="all | distserve | sglang_tp4 | aflex | distserve_sglang")
    parser.add_argument("--max-run-s", type=int, default=400)
    parser.add_argument("--dataset", choices=DATASETS,
                        help="Run one AFlex dataset instead of the full sweep")
    parser.add_argument("--qps", type=int, choices=QPS_LIST,
                        help="Run one AFlex QPS point instead of the full sweep")
    parser.add_argument("--monitor-sm-dir", type=Path,
                        help="Record per-GPU SM utilization during each workload")
    args = parser.parse_args()

    group = os.environ.get("MN_GROUP", "A")
    if group == "B":
        NODE1_IP_L, NODE2_IP_L = NODE3_IP, NODE4_IP
    else:
        NODE1_IP_L, NODE2_IP_L = NODE1_IP, NODE2_IP

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_results = {}
    gpus_all = list(range(8))
    selected_datasets = [args.dataset] if args.dataset else DATASETS
    selected_qps = [args.qps] if args.qps else QPS_LIST

    # ─── Phase 1: DistServe + BiScale (parallel on A=node1+2, B=node3+4) ───
    if args.phase in ("all", "distserve", "distserve_sglang"):
        log.info("=" * 70)
        log.info("PHASE 1: DistServe + BiScale [2P(TP4)+4D(TP2)] Group=%s", group)
        log.info("=" * 70)

        log.info(">> DistServe baseline (P=%s, D=%s)", NODE1_IP_L, NODE2_IP_L)
        r = run_scheme_tests(
            f"pd_dp_baseline_{group}",
            deploy_distserve, (NODE1_IP_L, NODE2_IP_L, False),
            hosts_for_cleanup=[NODE1_IP_L, NODE2_IP_L],
            energy_hosts_gpus=[(NODE1_IP_L, gpus_all), (NODE2_IP_L, gpus_all)],
            lock_hosts=[NODE1_IP_L, NODE2_IP_L], lock_gpus=gpus_all,
            max_run_s=args.max_run_s)
        all_results["pd_dp_baseline"] = r

        log.info(">> BiScale (P=%s, D=%s)", NODE1_IP_L, NODE2_IP_L)
        r = run_scheme_tests(
            f"pd_dp_tier_{group}",
            deploy_distserve, (NODE1_IP_L, NODE2_IP_L, True),
            hosts_for_cleanup=[NODE1_IP_L, NODE2_IP_L],
            energy_hosts_gpus=[(NODE1_IP_L, gpus_all), (NODE2_IP_L, gpus_all)],
            lock_hosts=[NODE1_IP_L, NODE2_IP_L], lock_gpus=gpus_all,
            max_run_s=args.max_run_s)
        all_results["pd_dp_tier"] = r

    # ─── Phase 2: SGLang_TP4 + DynamoLLM_TP4 ───
    if args.phase in ("all", "sglang_tp4", "distserve_sglang"):
        log.info("=" * 70)
        log.info("PHASE 2: SGLang_TP4 + DynamoLLM_TP4 [4×TP4] Group=%s", group)
        log.info("=" * 70)

        log.info(">> SGLang_TP4 baseline (%s + %s)", NODE1_IP_L, NODE2_IP_L)
        r = run_scheme_tests(
            f"native_tp4_baseline_{group}",
            deploy_sglang_tp4, (NODE1_IP_L, NODE2_IP_L, False),
            hosts_for_cleanup=[NODE1_IP_L, NODE2_IP_L],
            energy_hosts_gpus=[(NODE1_IP_L, gpus_all), (NODE2_IP_L, gpus_all)],
            lock_hosts=[NODE1_IP_L, NODE2_IP_L], lock_gpus=gpus_all,
            max_run_s=args.max_run_s)
        all_results["native_tp4_baseline"] = r

        log.info(">> DynamoLLM_TP4 (%s + %s)", NODE1_IP_L, NODE2_IP_L)
        r = run_scheme_tests(
            f"native_tp4_tier_{group}",
            deploy_sglang_tp4, (NODE1_IP_L, NODE2_IP_L, True),
            hosts_for_cleanup=[NODE1_IP_L, NODE2_IP_L],
            energy_hosts_gpus=[(NODE1_IP_L, gpus_all), (NODE2_IP_L, gpus_all)],
            lock_hosts=[NODE1_IP_L, NODE2_IP_L], lock_gpus=gpus_all,
            max_run_s=args.max_run_s)
        all_results["native_tp4_tier"] = r

    # ─── Phase 3: AFlex new config ───
    if args.phase in ("all", "aflex"):
        log.info("=" * 70)
        log.info("PHASE 3: AFlex [4P(PA1+PF2)+1D(DA1+DF2)] — 3 nodes")
        log.info("=" * 70)

        # Keep node2 free for the concurrently running cold-cache experiments.
        # Use node1+node3 for P and node4 for D.
        p_gpus_n1 = [0, 1, 2, 3, 4, 5]
        p_gpus_n3 = [0, 1, 2, 3, 4, 5]
        d_gpus_n4 = [0, 1, 2]

        log.info(">> AFlex_new (P: node1+node3, D: node4)")
        r = run_scheme_tests(
            "pdaf_tier_new",
            deploy_aflex_new, (NODE1_IP, NODE3_IP, NODE4_IP),
            hosts_for_cleanup=[NODE1_IP, NODE3_IP, NODE4_IP],
            energy_hosts_gpus=[(NODE1_IP, p_gpus_n1), (NODE3_IP, p_gpus_n3),
                               (NODE4_IP, d_gpus_n4)],
            lock_hosts=[NODE1_IP, NODE3_IP, NODE4_IP],
            lock_gpus=list(range(8)),
            datasets=selected_datasets,
            qps_list=selected_qps,
            max_run_s=args.max_run_s,
            monitor_sm_dir=args.monitor_sm_dir)
        all_results["pdaf_tier_new"] = r

    # ─── Save results ───
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_file = RESULTS_DIR / f"moe_retest_{ts}.json"
    payload = {
        "meta": {
            "benchmark": "moe_retest_补测",
            "model": "Mixtral-8x7B",
            "nodes": {"node1": NODE1_IP, "node2": NODE2_IP,
                      "node3": NODE3_IP, "node4": NODE4_IP},
            "datasets": DATASETS, "qps": QPS_LIST,
            "ttft_slo_ms": TTFT_SLO_MS, "tpot_slo_ms": TPOT_SLO_MS,
            "schemes": {
                "pd_dp_baseline": "DistServe 2P(TP4)+4D(TP2)",
                "pd_dp_tier": "BiScale 2P(TP4)+4D(TP2)+DVFS",
                "native_tp4_baseline": "SGLang 4×TP4",
                "native_tp4_tier": "DynamoLLM 4×TP4+DVFS",
                "pdaf_tier_new": "AFlex 4P(PA1+PF2)+1D(DA1+DF2)",
            },
            "timestamp": ts,
        },
        "results": all_results,
    }
    with open(out_file, "w") as f:
        json.dump(payload, f, indent=2)
    log.info("Results saved: %s", out_file)

    # Print summary table
    print("\n" + "=" * 100)
    print("  MoE RETEST RESULTS")
    print("=" * 100)
    hdr = f"{'Scheme':<25} {'Dataset':<12} {'Status':<8} {'Thpt':>8} {'TTFT_p50':>10} {'TPOT_p50':>10} {'mJ/tok':>10}"
    print(hdr)
    print("-" * 100)
    for scheme, data in all_results.items():
        if not isinstance(data, dict):
            continue
        for key in sorted(data.keys()):
            row = data[key]
            if not isinstance(row, dict):
                continue
            st = row.get("status", "?")
            if st == "PASS":
                print(f"{scheme:<25} {key:<12} {st:<8} "
                      f"{row['throughput_tok_s']:>8.1f} "
                      f"{row['ttft_proc_p50_ms']:>10.1f} "
                      f"{row['tpot_p50_ms']:>10.1f} "
                      f"{row['energy_per_token_mj']:>10.1f}")
            else:
                print(f"{scheme:<25} {key:<12} {st:<8}")
    print("=" * 100)


if __name__ == "__main__":
    main()
